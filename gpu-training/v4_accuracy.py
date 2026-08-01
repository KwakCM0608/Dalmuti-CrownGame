from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from v4_export import (
    canonical_json_bytes,
    load_v4_actor_checkpoint,
    sha256_file,
    verify_v4_actor_bundle,
)
from v4_model import V4_ACTION_COUNT, V4ActorConfig, V4PublicActor


REPORT_FORMAT = "dalmuti-v4-normal-action-agreement"
REPORT_VERSION = 1
DATASET_FORMAT = "dalmuti-v4-trajectory-npz"
DATASET_VERSION = 1
PREPARATION_FORMAT = "dalmuti-v4-prepared-dataset-metadata"
NORMAL_INPUT_FORMAT = "dalmuti-v4-normal-warmstart-ndjson"
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20260801
ROLE_NAMES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)
PUBLIC_ARRAY_NAMES = (
    "global_features",
    "rank_features",
    "player_features",
    "player_mask",
    "memory_trace_features",
    "history_features",
    "history_mask",
    "legal_masks",
    "actions",
    "valid_masks",
    "trajectory_ids",
)
DEVICE_PUBLIC_TENSORS = (
    "global_features",
    "rank_features",
    "player_features",
    "player_mask",
    "memory_trace_features",
    "history_features",
    "history_mask",
    "legal_masks",
)
ACT_PATTERN = re.compile(r":act-([1-9][0-9]*):")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PublicAccuracyDataset:
    """Prepared actor inputs only; privileged critic state is never loaded."""

    path: Path
    sha256: str
    sidecar_sha256: str
    fingerprint: str
    actor_config: V4ActorConfig
    metadata: Mapping[str, object]
    global_features: np.ndarray
    rank_features: np.ndarray
    player_features: np.ndarray
    player_mask: np.ndarray
    memory_trace_features: np.ndarray
    history_features: np.ndarray
    history_mask: np.ndarray
    legal_masks: np.ndarray
    normal_actions: np.ndarray
    valid_masks: np.ndarray
    trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class AccuracyRows:
    cluster_ids: np.ndarray
    player_counts: np.ndarray
    role_indices: np.ndarray
    acts: np.ndarray
    legal_counts: np.ndarray
    top1: np.ndarray
    top3: np.ndarray
    top5: np.ndarray
    nll: np.ndarray
    illegal_probability_mass: np.ndarray


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_dataset_sidecar(source: Path, actual_sha256: str) -> str:
    sidecar = Path(f"{source}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"prepared dataset checksum is missing: {sidecar}")
    raw = sidecar.read_bytes()
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("prepared dataset checksum must be ASCII") from error
    parts = text.split()
    if (
        len(parts) not in (1, 2)
        or not SHA256_PATTERN.fullmatch(parts[0])
        or (len(parts) == 2 and parts[1] != source.name)
    ):
        raise ValueError("prepared dataset checksum sidecar is malformed")
    if parts[0] != actual_sha256:
        raise ValueError("prepared dataset checksum does not match")
    return _sha256_bytes(raw)


def _metadata_from_archive(archive: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in archive.files:
        raise ValueError("prepared dataset metadata_json is missing")
    value = archive["metadata_json"]
    if value.shape != ():
        raise ValueError("prepared dataset metadata_json must be scalar")
    try:
        metadata = json.loads(str(value.item()))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("prepared dataset metadata_json is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("prepared dataset metadata must be an object")
    if (
        metadata.get("format") != DATASET_FORMAT
        or metadata.get("version") != DATASET_VERSION
    ):
        raise ValueError("unsupported V4 prepared dataset contract")
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or not SHA256_PATTERN.fullmatch(fingerprint):
        raise ValueError("prepared dataset fingerprint is invalid")
    actor_config = metadata.get("actorConfig")
    if not isinstance(actor_config, dict):
        raise ValueError("prepared dataset actorConfig is missing")
    # Accuracy against Normal is meaningful only for the strict rollout
    # preparation contract whose actionIndex is the exact Normal decision.
    if metadata.get("preparationFormat") != PREPARATION_FORMAT:
        raise ValueError("accuracy requires a strictly prepared V4 rollout dataset")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("prepared dataset input bindings are missing")
    for record in inputs:
        if not isinstance(record, dict) or record.get("format") != NORMAL_INPUT_FORMAT:
            raise ValueError("prepared dataset is not bound to exact Normal rollouts")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError("prepared dataset input SHA-256 is invalid")
    return metadata


def _array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    dtype: np.dtype[object] | None = None,
    dtype_kind: str | None = None,
) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"prepared dataset array is missing: {name}")
    value = np.asarray(archive[name])
    if (dtype is not None and value.dtype != dtype) or (
        dtype_kind is not None and value.dtype.kind != dtype_kind
    ):
        raise ValueError(f"prepared dataset array has an invalid dtype: {name}")
    return value


def _prefix_masks_are_valid(mask: np.ndarray, *, allow_empty: bool) -> bool:
    if mask.ndim != 2:
        return False
    if not allow_empty and np.any(mask.sum(axis=-1) < 1):
        return False
    invalid_seen = np.maximum.accumulate(~mask, axis=-1)
    return not bool(np.any(invalid_seen & mask))


def load_public_accuracy_dataset(path: str | Path) -> PublicAccuracyDataset:
    source = Path(path).resolve()
    if not source.is_file() or source.suffix.lower() != ".npz":
        raise FileNotFoundError(f"prepared V4 NPZ is missing: {source}")
    dataset_sha256 = sha256_file(source)
    sidecar_sha256 = _read_dataset_sidecar(source, dataset_sha256)
    with np.load(source, allow_pickle=False) as archive:
        metadata = _metadata_from_archive(archive)
        actor_config = V4ActorConfig(**metadata["actorConfig"])
        global_features = _array(archive, "global_features", dtype=np.dtype(np.float32))
        rank_features = _array(archive, "rank_features", dtype=np.dtype(np.float32))
        player_features = _array(archive, "player_features", dtype=np.dtype(np.float32))
        player_mask = _array(archive, "player_mask", dtype=np.dtype(np.bool_))
        memory_trace_features = _array(
            archive, "memory_trace_features", dtype=np.dtype(np.float32)
        )
        history_features = _array(archive, "history_features", dtype=np.dtype(np.float32))
        history_mask = _array(archive, "history_mask", dtype=np.dtype(np.bool_))
        legal_masks = _array(archive, "legal_masks", dtype=np.dtype(np.bool_))
        normal_actions = _array(archive, "actions", dtype=np.dtype(np.int64))
        valid_masks = _array(archive, "valid_masks", dtype=np.dtype(np.bool_))
        trajectory_ids_array = _array(
            archive, "trajectory_ids", dtype_kind="U"
        )

    if valid_masks.ndim != 2 or not valid_masks.any():
        raise ValueError("prepared dataset requires valid [trajectory, time] samples")
    trajectory_count, time_steps = valid_masks.shape
    prefix = (trajectory_count, time_steps)
    expected_shapes = {
        "global_features": (*prefix, actor_config.global_features),
        "rank_features": (
            *prefix,
            actor_config.rank_tokens,
            actor_config.rank_features,
        ),
        "player_features": (
            *prefix,
            actor_config.max_players,
            actor_config.player_features,
        ),
        "player_mask": (*prefix, actor_config.max_players),
        "memory_trace_features": (
            *prefix,
            actor_config.memory_tokens,
            actor_config.memory_features,
        ),
        "history_features": (
            *prefix,
            actor_config.max_history,
            actor_config.history_features,
        ),
        "history_mask": (*prefix, actor_config.max_history),
        "legal_masks": (*prefix, V4_ACTION_COUNT),
        "normal_actions": prefix,
    }
    values = {
        "global_features": global_features,
        "rank_features": rank_features,
        "player_features": player_features,
        "player_mask": player_mask,
        "memory_trace_features": memory_trace_features,
        "history_features": history_features,
        "history_mask": history_mask,
        "legal_masks": legal_masks,
        "normal_actions": normal_actions,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(f"prepared dataset shape does not match: {name}")
    if trajectory_ids_array.shape != (trajectory_count,):
        raise ValueError("prepared dataset trajectory_ids shape does not match")
    trajectory_ids = tuple(str(value) for value in trajectory_ids_array.tolist())
    if len(set(trajectory_ids)) != trajectory_count or any(not value for value in trajectory_ids):
        raise ValueError("prepared dataset trajectory IDs must be unique and non-empty")

    # Only valid public samples are audited. Padding is still required to be a
    # contiguous suffix, preventing hidden data from being selected by mistake.
    if np.any(valid_masks[:, 1:] & ~valid_masks[:, :-1]) or not valid_masks[:, 0].all():
        raise ValueError("prepared dataset valid samples must be contiguous prefixes")
    flat_players = player_mask[valid_masks]
    flat_history = history_mask[valid_masks]
    if not _prefix_masks_are_valid(flat_players, allow_empty=False):
        raise ValueError("public player masks must be non-empty contiguous prefixes")
    if not _prefix_masks_are_valid(flat_history, allow_empty=True):
        raise ValueError("public history masks must be contiguous prefixes")
    flat_legal = legal_masks[valid_masks]
    flat_actions = normal_actions[valid_masks]
    if not flat_legal.any(axis=-1).all():
        raise ValueError("every valid sample requires a legal action")
    if np.any((flat_actions < 0) | (flat_actions >= V4_ACTION_COUNT)):
        raise ValueError("Normal action is outside the action catalogue")
    if not np.take_along_axis(flat_legal, flat_actions[:, None], axis=-1).all():
        raise ValueError("Normal action is not legal in its bound observation")
    for name in (
        "global_features",
        "rank_features",
        "player_features",
        "memory_trace_features",
        "history_features",
    ):
        if not np.isfinite(values[name]).all():
            raise ValueError(f"prepared public array contains non-finite values: {name}")

    return PublicAccuracyDataset(
        path=source,
        sha256=dataset_sha256,
        sidecar_sha256=sidecar_sha256,
        fingerprint=str(metadata["fingerprint"]),
        actor_config=actor_config,
        metadata=metadata,
        global_features=global_features,
        rank_features=rank_features,
        player_features=player_features,
        player_mask=player_mask,
        memory_trace_features=memory_trace_features,
        history_features=history_features,
        history_mask=history_mask,
        legal_masks=legal_masks,
        normal_actions=normal_actions,
        valid_masks=valid_masks,
        trajectory_ids=trajectory_ids,
    )


def _resolved_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
    return device


def _load_verified_actors(
    bundle_paths: Sequence[str | Path],
    *,
    actor_config: V4ActorConfig,
    device: torch.device,
) -> tuple[list[torch.nn.Module], list[dict[str, object]], str, str]:
    if len(bundle_paths) not in (1, 3):
        raise ValueError("--bundle must be supplied once or exactly three times")
    loaded: list[tuple[str, torch.nn.Module, dict[str, object]]] = []
    for value in bundle_paths:
        bundle = Path(value).resolve()
        manifest = verify_v4_actor_bundle(bundle)
        model, payload = load_v4_actor_checkpoint(bundle / "actor.pt")
        if model.config != actor_config:
            raise ValueError("actor bundle config does not match prepared dataset")
        actor_record = manifest.get("files", {}).get("actor.pt")
        actor_sha256 = actor_record.get("sha256") if isinstance(actor_record, dict) else None
        if not isinstance(actor_sha256, str) or not SHA256_PATTERN.fullmatch(actor_sha256):
            raise ValueError("actor bundle manifest has an invalid actor SHA-256")
        record = {
            "bundleName": bundle.name,
            "manifestSha256": sha256_file(bundle / "manifest.json"),
            "actorSha256": actor_sha256,
            "kind": payload.get("kind"),
            "checkpointMetadata": payload.get("metadata", {}),
        }
        loaded.append((actor_sha256, model, record))
    loaded.sort(key=lambda item: item[0])
    actor_hashes = [item[0] for item in loaded]
    if len(actor_hashes) == 3:
        if len(set(actor_hashes)) != 3:
            raise ValueError("three-bundle ensemble requires three distinct actors")
        if any(record[2]["kind"] != "actor" for record in loaded):
            raise ValueError("three-bundle ensemble requires single-actor bundles")
        if any(not isinstance(record[1], V4PublicActor) for record in loaded):
            raise ValueError("three-bundle ensemble may contain public actors only")
        ensemble_rule = "mean-of-per-actor-logits-centered-over-legal-actions"
        actor_binding = _sha256_bytes(canonical_json_bytes({
            "format": "dalmuti-v4-centered-logit-ensemble-binding",
            "version": 1,
            "actorSha256s": actor_hashes,
            "rule": ensemble_rule,
        }))
    else:
        actor_binding = actor_hashes[0]
        ensemble_rule = (
            "exported-centered-logit-ensemble"
            if loaded[0][2]["kind"] == "centered-logit-ensemble"
            else "single-actor"
        )
    models = [item[1].to(device).eval() for item in loaded]
    return models, [item[2] for item in loaded], actor_binding, ensemble_rule


def _derive_sample_labels(
    dataset: PublicAccuracyDataset,
    trajectory_indices: np.ndarray,
    time_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    players = dataset.player_mask[trajectory_indices, time_indices]
    player_counts = players.sum(axis=-1).astype(np.int16)
    encoded_player_counts = (player_counts.astype(np.float64) - 4.0) / 6.0
    globals_ = dataset.global_features[trajectory_indices, time_indices]
    if not np.allclose(globals_[:, 0], encoded_player_counts, rtol=0.0, atol=1e-6):
        raise ValueError("public player-count feature does not match player masks")
    role_features = globals_[:, 2:7]
    role_indices = role_features.argmax(axis=-1).astype(np.int8)
    expected_roles = np.eye(5, dtype=np.float32)[role_indices]
    if not np.allclose(role_features, expected_roles, rtol=0.0, atol=1e-6):
        raise ValueError("public actor-role feature is not canonical one-hot")

    trajectory_acts = np.empty(len(dataset.trajectory_ids), dtype=np.int64)
    for index, trajectory_id in enumerate(dataset.trajectory_ids):
        match = ACT_PATTERN.search(trajectory_id)
        if match is None:
            raise ValueError("trajectory ID does not bind an act")
        trajectory_acts[index] = int(match.group(1))
    acts = trajectory_acts[trajectory_indices]
    encoded_acts = np.tanh((acts.astype(np.float64) - 1.0) / 10.0)
    if not np.allclose(globals_[:, 1], encoded_acts, rtol=0.0, atol=2e-6):
        raise ValueError("public act feature does not match trajectory binding")
    legal_counts = dataset.legal_masks[
        trajectory_indices, time_indices
    ].sum(axis=-1).astype(np.int16)
    return player_counts, role_indices, acts, legal_counts


def _torch_batch(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(device)


def infer_accuracy_rows(
    dataset: PublicAccuracyDataset,
    models: Sequence[torch.nn.Module],
    *,
    device: torch.device,
    batch_size: int,
) -> AccuracyRows:
    _positive_integer(batch_size, "batch size")
    if len(models) not in (1, 3):
        raise ValueError("inference requires one actor bundle or three actor bundles")
    trajectory_indices, time_indices = np.nonzero(dataset.valid_masks)
    sample_count = len(trajectory_indices)
    player_counts, role_indices, acts, legal_counts = _derive_sample_labels(
        dataset, trajectory_indices, time_indices
    )
    top1 = np.empty(sample_count, dtype=np.bool_)
    top3 = np.empty(sample_count, dtype=np.bool_)
    top5 = np.empty(sample_count, dtype=np.bool_)
    nll = np.empty(sample_count, dtype=np.float64)
    illegal_mass = np.empty(sample_count, dtype=np.float64)

    for start in range(0, sample_count, batch_size):
        end = min(sample_count, start + batch_size)
        trajectories = trajectory_indices[start:end]
        times = time_indices[start:end]
        player_mask_numpy = dataset.player_mask[trajectories, times]
        history_mask_numpy = dataset.history_mask[trajectories, times]
        player_width = int(np.flatnonzero(player_mask_numpy.any(axis=0))[-1]) + 1
        history_used = np.flatnonzero(history_mask_numpy.any(axis=0))
        history_width = 0 if history_used.size == 0 else int(history_used[-1]) + 1
        legal = _torch_batch(dataset.legal_masks[trajectories, times], device).bool()
        inputs = (
            _torch_batch(dataset.global_features[trajectories, times], device),
            _torch_batch(dataset.rank_features[trajectories, times], device),
            _torch_batch(
                dataset.player_features[trajectories, times, :player_width], device
            ),
            _torch_batch(player_mask_numpy[:, :player_width], device).bool(),
            _torch_batch(dataset.memory_trace_features[trajectories, times], device),
            _torch_batch(
                dataset.history_features[trajectories, times, :history_width], device
            ),
            _torch_batch(history_mask_numpy[:, :history_width], device).bool(),
            legal,
        )
        normal = _torch_batch(dataset.normal_actions[trajectories, times], device).long()
        with torch.inference_mode():
            member_logits = [model(*inputs) for model in models]
            for logits in member_logits:
                if logits.shape != (end - start, V4_ACTION_COUNT):
                    raise ValueError("actor returned an invalid logit shape")
                if not bool(torch.isfinite(logits[legal]).all().item()):
                    raise ValueError("actor returned non-finite legal logits")
            if len(member_logits) == 1:
                logits = member_logits[0]
            else:
                legal_float = legal.to(dtype=member_logits[0].dtype)
                denominator = legal_float.sum(dim=-1, keepdim=True)
                centered = [
                    member.masked_fill(~legal, 0.0).sub(
                        member.masked_fill(~legal, 0.0).sum(dim=-1, keepdim=True)
                        / denominator
                    )
                    for member in member_logits
                ]
                logits = torch.stack(centered, dim=0).mean(dim=0)
            masked_logits = logits.float().masked_fill(~legal, float("-inf"))
            probabilities = torch.softmax(masked_logits, dim=-1)
            batch_illegal_mass = probabilities.masked_fill(legal, 0.0).sum(dim=-1)
            # This is a hard privacy/action-validity invariant, not a tolerance.
            if bool(torch.count_nonzero(batch_illegal_mass).item()):
                raise RuntimeError("illegal action probability mass must be exactly zero")
            log_probabilities = torch.log_softmax(masked_logits, dim=-1)
            target_logits = masked_logits.gather(-1, normal[:, None])
            action_indices = torch.arange(V4_ACTION_COUNT, device=device)[None, :]
            better = legal & (
                (masked_logits > target_logits)
                | (
                    (masked_logits == target_logits)
                    & (action_indices < normal[:, None])
                )
            )
            target_rank = better.sum(dim=-1) + 1
            legal_count = legal.sum(dim=-1)
            batch_top1 = target_rank <= torch.minimum(
                torch.ones_like(legal_count), legal_count
            )
            batch_top3 = target_rank <= torch.minimum(
                torch.full_like(legal_count, 3), legal_count
            )
            batch_top5 = target_rank <= torch.minimum(
                torch.full_like(legal_count, 5), legal_count
            )
            batch_nll = -log_probabilities.gather(-1, normal[:, None]).squeeze(-1)
        top1[start:end] = batch_top1.cpu().numpy()
        top3[start:end] = batch_top3.cpu().numpy()
        top5[start:end] = batch_top5.cpu().numpy()
        nll[start:end] = batch_nll.double().cpu().numpy()
        illegal_mass[start:end] = batch_illegal_mass.double().cpu().numpy()

    if not np.isfinite(nll).all():
        raise ValueError("actor produced a non-finite Normal cross-entropy")
    if np.any(illegal_mass != 0.0):
        raise RuntimeError("illegal action probability mass must be exactly zero")
    return AccuracyRows(
        cluster_ids=trajectory_indices.astype(np.int64, copy=False),
        player_counts=player_counts,
        role_indices=role_indices,
        acts=acts,
        legal_counts=legal_counts,
        top1=top1,
        top3=top3,
        top5=top5,
        nll=nll,
        illegal_probability_mass=illegal_mass,
    )


def _scope_seed(base_seed: int, label: str) -> int:
    material = f"dalmuti-v4-accuracy-bootstrap-v1:{base_seed}:{label}".encode("utf-8")
    result = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return result or 1


def _cluster_bootstrap(
    successes: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    _positive_integer(resamples, "bootstrap resamples")
    unique, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_successes = np.bincount(
        inverse, weights=successes.astype(np.float64), minlength=len(unique)
    )
    cluster_counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    point = float(cluster_successes.sum() / cluster_counts.sum())
    if len(unique) == 1:
        low = high = point
    else:
        rng = np.random.default_rng(seed)
        values = np.empty(resamples, dtype=np.float64)
        offset = 0
        while offset < resamples:
            width = min(256, resamples - offset)
            sampled = rng.integers(0, len(unique), size=(width, len(unique)))
            values[offset : offset + width] = (
                cluster_successes[sampled].sum(axis=1)
                / cluster_counts[sampled].sum(axis=1)
            )
            offset += width
        low, high = np.quantile(values, (0.025, 0.975)).tolist()
    return {
        "statistic": "legal-greedy-agreement-with-exact-Normal",
        "method": "deterministic-trajectory-cluster-percentile-bootstrap",
        "unit": "actor-trajectory",
        "confidence": 0.95,
        "clusters": int(len(unique)),
        "resamples": int(resamples),
        "seed": int(seed),
        "pointEstimate": point,
        "lower": float(low),
        "upper": float(high),
    }


def _metric_block(
    rows: AccuracyRows,
    selection: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    scope_label: str,
) -> dict[str, object] | None:
    count = int(selection.sum())
    if count == 0:
        return None
    cluster_ids = rows.cluster_ids[selection]
    top1 = rows.top1[selection]
    top3 = rows.top3[selection]
    top5 = rows.top5[selection]
    illegal = rows.illegal_probability_mass[selection]
    correct1 = int(top1.sum())
    correct3 = int(top3.sum())
    correct5 = int(top5.sum())
    seed = _scope_seed(bootstrap_seed, scope_label)
    return {
        "samples": count,
        "trajectories": int(np.unique(cluster_ids).size),
        "greedyAgreement": {
            "correct": correct1,
            "total": count,
            "rate": correct1 / count,
            "tieBreak": "lowest-action-index",
        },
        "crossEntropyNll": float(rows.nll[selection].mean()),
        "topKAccuracy": {
            "1": {"correct": correct1, "total": count, "rate": correct1 / count},
            "3": {"correct": correct3, "total": count, "rate": correct3 / count},
            "5": {"correct": correct5, "total": count, "rate": correct5 / count},
        },
        "illegalProbabilityMass": {
            "sum": float(illegal.sum()),
            "mean": float(illegal.mean()),
            "maximum": float(illegal.max()),
            "requiredMaximum": 0.0,
            "passed": bool(np.all(illegal == 0.0)),
        },
        "trajectoryClusterBootstrap95": _cluster_bootstrap(
            top1,
            cluster_ids,
            seed=seed,
            resamples=bootstrap_resamples,
        ),
    }


def _scope(
    rows: AccuracyRows,
    selection: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    label: str,
) -> dict[str, object]:
    non_forced = selection & (rows.legal_counts > 1)
    return {
        "allValid": _metric_block(
            rows,
            selection,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            scope_label=f"{label}:all-valid",
        ),
        "nonForced": _metric_block(
            rows,
            non_forced,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            scope_label=f"{label}:non-forced",
        ),
    }


def _legal_bucket(count: int) -> str:
    if count == 1:
        return "1"
    for lower, upper in (
        (2, 2),
        (3, 4),
        (5, 8),
        (9, 16),
        (17, 32),
        (33, 64),
        (65, 128),
        (129, V4_ACTION_COUNT),
    ):
        if lower <= count <= upper:
            return str(lower) if lower == upper else f"{lower}-{upper}"
    raise ValueError("legal action count is outside the action catalogue")


def build_accuracy_report(
    dataset: PublicAccuracyDataset,
    rows: AccuracyRows,
    *,
    model_records: Sequence[Mapping[str, object]],
    actor_binding_sha256: str,
    ensemble_rule: str,
    device: torch.device,
    batch_size: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    _positive_integer(bootstrap_seed, "bootstrap seed")
    _positive_integer(bootstrap_resamples, "bootstrap resamples")
    sample_count = len(rows.top1)
    all_valid = np.ones(sample_count, dtype=np.bool_)
    non_forced = rows.legal_counts > 1
    per_player_count = {
        f"p{int(value)}": _scope(
            rows,
            rows.player_counts == value,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            label=f"player-count:{int(value)}",
        )
        for value in sorted(np.unique(rows.player_counts).tolist())
    }
    per_role = {
        ROLE_NAMES[int(value)]: _scope(
            rows,
            rows.role_indices == value,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            label=f"role:{ROLE_NAMES[int(value)]}",
        )
        for value in sorted(np.unique(rows.role_indices).tolist())
    }
    per_act = {
        str(int(value)): _scope(
            rows,
            rows.acts == value,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            label=f"act:{int(value)}",
        )
        for value in sorted(np.unique(rows.acts).tolist())
    }
    bucket_labels = np.asarray(
        [_legal_bucket(int(value)) for value in rows.legal_counts], dtype=np.str_
    )
    bucket_order = (
        "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-236"
    )
    per_legal_bucket = {
        label: _scope(
            rows,
            bucket_labels == label,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            label=f"legal-action-count:{label}",
        )
        for label in bucket_order
        if bool(np.any(bucket_labels == label))
    }
    all_valid_metrics = _metric_block(
        rows,
        all_valid,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        scope_label="overall:all-valid",
    )
    non_forced_metrics = _metric_block(
        rows,
        non_forced,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        scope_label="overall:non-forced",
    )
    assert all_valid_metrics is not None
    if not all_valid_metrics["illegalProbabilityMass"]["passed"]:  # type: ignore[index]
        raise RuntimeError("illegal action probability mass gate failed")
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "bindings": {
            "dataset": {
                "fileName": dataset.path.name,
                "sha256": dataset.sha256,
                "checksumSidecarSha256": dataset.sidecar_sha256,
                "fingerprint": dataset.fingerprint,
                "format": dataset.metadata["format"],
                "version": dataset.metadata["version"],
                "normalReferenceTensor": "actions",
                "normalReferenceContract": "actionIndex selected by exact Normal",
                "inputSha256s": sorted(
                    str(record["sha256"])
                    for record in dataset.metadata["inputs"]  # type: ignore[index]
                ),
            },
            "model": {
                "actorBindingSha256": actor_binding_sha256,
                "bundleCount": len(model_records),
                "bundles": [dict(record) for record in model_records],
                "ensembleRule": ensemble_rule,
                "actorConfigSha256": _sha256_bytes(
                    canonical_json_bytes(dataset.actor_config.to_dict())
                ),
            },
        },
        "inference": {
            "device": str(device),
            "batchSize": batch_size,
            "dynamicPlayerTrim": True,
            "dynamicHistoryTrim": True,
            "selection": "legal-greedy-lowest-index-tie-break",
            "probabilityNormalization": "softmax-over-legal-actions-only",
        },
        "privacyAudit": {
            "actorPublicOnly": True,
            "loadedDatasetArrays": list(PUBLIC_ARRAY_NAMES),
            "ignoredDatasetArrays": ["privileged_states"],
            "actorInputsTransferredToDevice": list(DEVICE_PUBLIC_TENSORS),
            "metricReferenceTransferredToDevice": ["actions"],
            "privilegedStateLoaded": False,
            "privilegedStateTransferred": False,
            "passed": True,
        },
        "samples": {
            "trajectories": int(np.unique(rows.cluster_ids).size),
            "allValid": sample_count,
            "forced": int((~non_forced).sum()),
            "nonForced": int(non_forced.sum()),
        },
        "bootstrap": {
            "unit": "actor-trajectory",
            "method": "deterministic-trajectory-cluster-percentile-bootstrap",
            "confidence": 0.95,
            "requestedSeed": bootstrap_seed,
            "resamplesPerScope": bootstrap_resamples,
        },
        "metrics": {
            "allValid": all_valid_metrics,
            "nonForced": non_forced_metrics,
            "perPlayerCount": per_player_count,
            "perRole": per_role,
            "perAct": per_act,
            "legalActionCountBuckets": {
                "definitions": list(bucket_order),
                "metrics": per_legal_bucket,
            },
        },
        "gates": {
            "illegalProbabilityMassMustEqualZero": True,
            "illegalProbabilityMassPassed": True,
        },
    }


def evaluate_v4_accuracy(
    dataset_path: str | Path,
    bundle_paths: Sequence[str | Path],
    *,
    device: str = "auto",
    batch_size: int = 256,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    _positive_integer(batch_size, "batch size")
    _positive_integer(bootstrap_seed, "bootstrap seed")
    _positive_integer(bootstrap_resamples, "bootstrap resamples")
    public_dataset = load_public_accuracy_dataset(dataset_path)
    resolved_device = _resolved_device(device)
    models, model_records, actor_binding, ensemble_rule = _load_verified_actors(
        bundle_paths,
        actor_config=public_dataset.actor_config,
        device=resolved_device,
    )
    rows = infer_accuracy_rows(
        public_dataset,
        models,
        device=resolved_device,
        batch_size=batch_size,
    )
    return build_accuracy_report(
        public_dataset,
        rows,
        model_records=model_records,
        actor_binding_sha256=actor_binding,
        ensemble_rule=ensemble_rule,
        device=resolved_device,
        batch_size=batch_size,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )


def _publish_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_accuracy_report_exclusive(
    output_path: str | Path, report: Mapping[str, object]
) -> dict[str, object]:
    output = Path(output_path).resolve()
    checksum = output.with_name(output.name + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("accuracy report and checksum are immutable")
    payload = canonical_json_bytes(report)
    digest = _sha256_bytes(payload)
    _publish_exclusive(output, payload)
    try:
        _publish_exclusive(
            checksum, f"{digest}  {output.name}\n".encode("ascii")
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {
        "path": str(output),
        "sha256Path": str(checksum),
        "sha256": digest,
        "bytes": len(payload),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure a verified V4 public actor's legal agreement with exact Normal."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--bundle", "--actor-bundle", dest="bundles", type=Path,
        action="append", required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = evaluate_v4_accuracy(
        arguments.dataset,
        arguments.bundles,
        device=arguments.device,
        batch_size=arguments.batch_size,
        bootstrap_seed=arguments.bootstrap_seed,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    artifact = write_accuracy_report_exclusive(arguments.output, report)
    print(json.dumps({
        "output": artifact["path"],
        "sha256": artifact["sha256"],
        "allValidAgreement": report["metrics"]["allValid"]["greedyAgreement"]["rate"],
        "nonForcedAgreement": (
            None
            if report["metrics"]["nonForced"] is None
            else report["metrics"]["nonForced"]["greedyAgreement"]["rate"]
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AccuracyRows",
    "PublicAccuracyDataset",
    "build_accuracy_report",
    "evaluate_v4_accuracy",
    "infer_accuracy_rows",
    "load_public_accuracy_dataset",
    "main",
    "write_accuracy_report_exclusive",
]
