from __future__ import annotations

"""Offline confidence calibration for the V4 exact-Normal fallback router.

This module intentionally answers a narrow question: on states sampled from a
checksum-verified exact-Normal rollout, how often does a public actor agree
with Normal at different inclusive confidence thresholds?  It is a behavioral
safety diagnostic.  It is not an outcome evaluation and cannot establish that
the actor, or a routed actor/Normal policy, beats Normal.

Only public actor arrays are materialized from the prepared NPZ.  In
particular, ``privileged_states`` is never indexed or loaded.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
from typing import Mapping, Sequence

import numpy as np


FORMAT = "dalmuti-v4-fallback-calibration"
VERSION = 1
DATASET_FORMAT = "dalmuti-v4-trajectory-npz"
DATASET_VERSION = 1
PREPARATION_FORMAT = "dalmuti-v4-prepared-dataset-metadata"
PREPARATION_VERSION = 1
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
    "expert_actions",
    "valid_masks",
    "trajectory_ids",
)
PRIVILEGED_ARRAY_NAMES = frozenset(
    {
        "privileged_states",
        "finish_places",
        "environment_terminals",
        "rewards",
        "advantages",
        "old_action_log_probs",
        "dones",
        "source_steps",
        "trajectory_input_sha256s",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def parse_threshold_grid(value: str, *, probability: bool) -> tuple[float, ...]:
    label = "probability" if probability else "margin"
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{label} grid must be a comma-separated numeric list")
    result = sorted({_finite(float(part), f"{label} threshold") for part in parts})
    if any(item < 0.0 for item in result):
        raise ValueError(f"{label} thresholds must be non-negative")
    if probability and any(item > 1.0 for item in result):
        raise ValueError("probability thresholds must be at most one")
    return tuple(result)


def route_actor_inclusive(
    *,
    legal_action_count: int,
    legal_logit_margin: float | None,
    top_probability: float,
    minimum_legal_logit_margin: float | None,
    minimum_top_probability: float | None,
) -> bool:
    """Mirror ``CandidatePolicyRouting`` confidence-fallback semantics."""

    if legal_action_count < 1:
        raise ValueError("legal action count must be positive")
    if legal_action_count == 1:
        return False
    if legal_logit_margin is None:
        raise ValueError("non-forced decisions require a legal-logit margin")
    margin = _finite(legal_logit_margin, "legal-logit margin")
    probability = _finite(top_probability, "top probability")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("top probability must be from zero through one")
    if minimum_legal_logit_margin is not None:
        threshold = _finite(
            minimum_legal_logit_margin, "minimum legal-logit margin"
        )
        if threshold < 0.0 or margin < threshold:
            return False
    if minimum_top_probability is not None:
        threshold = _finite(minimum_top_probability, "minimum top probability")
        if threshold < 0.0 or threshold > 1.0 or probability < threshold:
            return False
    return True


def legal_diagnostics_from_logits(
    logits: np.ndarray, legal_masks: np.ndarray
) -> dict[str, np.ndarray]:
    """Small NumPy oracle used to audit legality and illegal softmax mass."""

    values = np.asarray(logits, dtype=np.float64)
    legal = np.asarray(legal_masks, dtype=np.bool_)
    if values.ndim != 2 or values.shape != legal.shape or values.shape[1] != 236:
        raise ValueError("logits and legal masks must have shape [batch, 236]")
    if not np.isfinite(values[legal]).all() or not legal.any(axis=1).all():
        raise ValueError("legal logits must be finite and every row must be legal")
    masked = np.where(legal, values, -np.inf)
    actions = np.argmax(masked, axis=1).astype(np.int64)
    maximum = np.max(masked, axis=1, keepdims=True)
    exponentials = np.where(legal, np.exp(masked - maximum), 0.0)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    legal_counts = legal.sum(axis=1).astype(np.int64)
    top_probability = probabilities[np.arange(len(actions)), actions]
    without_top = masked.copy()
    without_top[np.arange(len(actions)), actions] = -np.inf
    second = np.max(without_top, axis=1)
    margin = masked[np.arange(len(actions)), actions] - second
    margin = np.where(legal_counts == 1, np.nan, margin)
    return {
        "actions": actions,
        "legalCounts": legal_counts,
        "margins": margin,
        "topProbabilities": top_probability,
        "illegalProbabilityMass": np.where(legal, 0.0, probabilities).sum(axis=1),
    }


@dataclass(frozen=True)
class PublicPreparedDataset:
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]
    dataset_sha256: str
    metadata_sha256: str


def _read_dataset_sidecar(dataset: Path) -> str:
    sidecar = Path(f"{dataset}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError("prepared dataset checksum sidecar is missing")
    parts = sidecar.read_text(encoding="ascii").strip().split()
    if len(parts) not in (1, 2) or not _SHA256_RE.fullmatch(parts[0]):
        raise ValueError("prepared dataset checksum sidecar is malformed")
    if len(parts) == 2 and parts[1] != dataset.name:
        raise ValueError("prepared dataset checksum sidecar names another file")
    actual = sha256_file(dataset)
    if parts[0] != actual:
        raise ValueError("prepared dataset checksum does not match")
    return actual


def load_public_prepared_normal_dataset(path: str | Path) -> PublicPreparedDataset:
    """Load a strict prepared Normal NPZ without materializing private arrays."""

    dataset = Path(path).resolve()
    dataset_sha = _read_dataset_sidecar(dataset)
    external_path = Path(f"{dataset}.metadata.json")
    if not external_path.is_file():
        raise FileNotFoundError("prepared dataset external metadata is missing")
    external_bytes = external_path.read_bytes()
    if not external_bytes.endswith(b"\n"):
        raise ValueError("prepared dataset metadata must be LF-terminated")
    external = json.loads(external_bytes)
    if not isinstance(external, dict) or external.get("npzSha256") != dataset_sha:
        raise ValueError("prepared dataset external metadata does not bind the NPZ")

    # NpzFile is lazy.  Only the explicit public allowlist below is indexed;
    # merely checking archive.files does not deserialize privileged payloads.
    with np.load(dataset, allow_pickle=False) as archive:
        names = set(archive.files)
        required = set(PUBLIC_ARRAY_NAMES) | {"metadata_json"}
        if not required.issubset(names):
            raise ValueError("prepared dataset is missing required public arrays")
        embedded = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name] for name in PUBLIC_ARRAY_NAMES}

    if not isinstance(embedded, dict):
        raise ValueError("prepared dataset embedded metadata must be an object")
    comparison = dict(external)
    comparison.pop("npzSha256", None)
    if comparison != embedded:
        raise ValueError("prepared dataset embedded and external metadata differ")
    if (
        embedded.get("format") != DATASET_FORMAT
        or embedded.get("version") != DATASET_VERSION
        or embedded.get("preparationFormat") != PREPARATION_FORMAT
        or embedded.get("preparationVersion") != PREPARATION_VERSION
        or embedded.get("privilegedCriticExportAllowed") is not False
    ):
        raise ValueError("unsupported strict prepared V4 dataset contract")
    synthetic = embedded.get("syntheticDefaults")
    fields_present = embedded.get("sampleFieldsPresent")
    if (
        not isinstance(synthetic, dict)
        or synthetic.get("expertActionIndex") is not True
        or not isinstance(fields_present, list)
        or "expertActionIndex" in fields_present
    ):
        raise ValueError("calibration requires an exact-Normal warm-start dataset")
    _validate_public_arrays(arrays, embedded)
    return PublicPreparedDataset(
        arrays=arrays,
        metadata=embedded,
        dataset_sha256=dataset_sha,
        metadata_sha256=hashlib.sha256(external_bytes).hexdigest(),
    )


def _validate_public_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> None:
    actions = arrays["actions"]
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
        raise ValueError("actions must be non-empty [trajectory, time]")
    prefix = actions.shape
    actor = metadata.get("actorConfig")
    if not isinstance(actor, dict):
        raise ValueError("prepared dataset actor config is missing")
    expected = {
        "global_features": (*prefix, int(actor["global_features"])),
        "rank_features": (
            *prefix,
            int(actor["rank_tokens"]),
            int(actor["rank_features"]),
        ),
        "player_features": (
            *prefix,
            int(actor["max_players"]),
            int(actor["player_features"]),
        ),
        "player_mask": (*prefix, int(actor["max_players"])),
        "memory_trace_features": (
            *prefix,
            int(actor["memory_tokens"]),
            int(actor["memory_features"]),
        ),
        "history_features": (
            *prefix,
            int(actor["max_history"]),
            int(actor["history_features"]),
        ),
        "history_mask": (*prefix, int(actor["max_history"])),
        "legal_masks": (*prefix, 236),
        "actions": prefix,
        "expert_actions": prefix,
        "valid_masks": prefix,
        "trajectory_ids": (prefix[0],),
    }
    boolean = {"player_mask", "history_mask", "legal_masks", "valid_masks"}
    integer = {"actions", "expert_actions"}
    for name, shape in expected.items():
        array = arrays[name]
        if array.shape != shape:
            raise ValueError(f"{name} shape does not match actor metadata")
        if name in boolean and array.dtype != np.bool_:
            raise ValueError(f"{name} must use bool")
        if name in integer and not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must use integers")
        if name not in boolean | integer | {"trajectory_ids"} and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite public values")
    valid = arrays["valid_masks"]
    if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("valid masks must be non-empty contiguous prefixes")
    legal = arrays["legal_masks"]
    if not (legal.any(axis=-1) | ~valid).all():
        raise ValueError("every valid decision requires a legal action")
    for name in ("actions", "expert_actions"):
        selected = arrays[name]
        safe = np.clip(selected, 0, 235)
        selected_legal = np.take_along_axis(legal, safe[..., None], axis=-1)[..., 0]
        if (((selected < 0) | (selected >= 236) | ~selected_legal) & valid).any():
            raise ValueError(f"{name} contains an illegal valid action")
    if not np.array_equal(
        arrays["actions"][valid], arrays["expert_actions"][valid]
    ):
        raise ValueError("exact-Normal actions and expert actions differ")
    identifiers = arrays["trajectory_ids"].astype(str).tolist()
    if len(set(identifiers)) != len(identifiers) or any(not item for item in identifiers):
        raise ValueError("trajectory IDs must be unique non-empty strings")


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    result = {}
    for name in (
        "v4_fallback_calibration.py",
        "v4_evaluate.py",
        "v4_export.py",
        "v4_model.py",
    ):
        result[name] = sha256_file(directory / name)
    return result


def _observation(arrays: Mapping[str, np.ndarray], trajectory: int, time: int):
    import torch

    public = SimpleNamespace(
        global_features=torch.from_numpy(arrays["global_features"][trajectory, time]),
        rank_features=torch.from_numpy(arrays["rank_features"][trajectory, time]),
        player_features=torch.from_numpy(arrays["player_features"][trajectory, time]),
        player_mask=torch.from_numpy(arrays["player_mask"][trajectory, time]),
        memory_trace_features=torch.from_numpy(
            arrays["memory_trace_features"][trajectory, time]
        ),
        history_features=torch.from_numpy(arrays["history_features"][trajectory, time]),
        history_mask=torch.from_numpy(arrays["history_mask"][trajectory, time]),
        legal_mask=torch.from_numpy(arrays["legal_masks"][trajectory, time]),
    )
    return SimpleNamespace(public=public)


def _infer_public_diagnostics(
    prepared: PublicPreparedDataset,
    policy: object,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch size must be positive")
    arrays = prepared.arrays
    valid_positions = np.argwhere(arrays["valid_masks"])
    count = len(valid_positions)
    actions = np.empty(count, dtype=np.int64)
    margins = np.full(count, np.nan, dtype=np.float64)
    probabilities = np.empty(count, dtype=np.float64)
    legal_counts = np.empty(count, dtype=np.int64)
    normal = np.empty(count, dtype=np.int64)
    trajectory = valid_positions[:, 0].astype(np.int64)
    times = valid_positions[:, 1].astype(np.int64)
    player_counts = np.empty(count, dtype=np.int64)
    roles = np.empty(count, dtype=np.int64)
    acts = np.empty(count, dtype=np.int64)

    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        observations = [
            _observation(arrays, int(t), int(s))
            for t, s in valid_positions[start:stop]
        ]
        diagnostics = getattr(policy, "action_diagnostics")(observations)
        if len(diagnostics) != stop - start:
            raise RuntimeError("actor diagnostics changed batch cardinality")
        for offset, diagnostic in enumerate(diagnostics, start=start):
            actions[offset] = int(diagnostic.action)
            legal_counts[offset] = int(diagnostic.legal_action_count)
            probabilities[offset] = float(diagnostic.top_probability)
            if diagnostic.legal_logit_margin is not None:
                margins[offset] = float(diagnostic.legal_logit_margin)

    for index, (trajectory_index, time_index) in enumerate(valid_positions):
        legal = arrays["legal_masks"][trajectory_index, time_index]
        if not bool(legal[actions[index]]):
            raise ValueError("actor selected an illegal action")
        actual_count = int(legal.sum())
        if actual_count != legal_counts[index]:
            raise ValueError("actor legal count differs from prepared public mask")
        normal[index] = int(arrays["actions"][trajectory_index, time_index])
        mask = arrays["player_mask"][trajectory_index, time_index]
        p = int(mask.sum())
        if p < 4 or p > 10 or not mask[:p].all() or mask[p:].any():
            raise ValueError("public player mask is not a p4-p10 valid prefix")
        player_counts[index] = p
        global_row = arrays["global_features"][trajectory_index, time_index]
        encoded_p = int(round(float(global_row[0]) * 6.0 + 4.0))
        if encoded_p != p:
            raise ValueError("public global player count disagrees with player mask")
        role_values = np.asarray(global_row[2:7], dtype=np.float64)
        role = int(np.argmax(role_values))
        if not np.allclose(role_values, np.eye(5)[role], atol=1e-6, rtol=0.0):
            raise ValueError("public actor role is not exact one-hot")
        roles[index] = role
        act = int(round(math.atanh(float(global_row[1])) * 10.0 + 1.0))
        if act < 1 or not math.isclose(
            float(global_row[1]), math.tanh((act - 1) / 10.0), abs_tol=1e-6
        ):
            raise ValueError("public act encoding is invalid")
        acts[index] = act

    forced = legal_counts == 1
    if not np.isnan(margins[forced]).all() or not np.isfinite(margins[~forced]).all():
        raise ValueError("actor margin forced/non-forced contract is invalid")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("actor top probabilities are invalid")
    return {
        "actions": actions,
        "margins": margins,
        "probabilities": probabilities,
        "legalCounts": legal_counts,
        "normalActions": normal,
        "trajectoryIndexes": trajectory,
        "timeIndexes": times,
        "playerCounts": player_counts,
        "roles": roles,
        "acts": acts,
    }


def _summary_counts(selected: np.ndarray, agreement: np.ndarray, total: int) -> dict[str, object]:
    actor = int(selected.sum())
    agreed = int((selected & agreement).sum())
    deviations = actor - agreed
    return {
        "decisions": int(total),
        "actorDecisions": actor,
        "fallbackDecisions": int(total - actor),
        "actorCoverage": actor / total if total else 0.0,
        "fallbackRate": (total - actor) / total if total else 0.0,
        "actorNormalAgreements": agreed,
        "actorNormalDeviations": deviations,
        "actorNormalAgreement": None if actor == 0 else agreed / actor,
        "routedPolicyNormalAgreement": (total - deviations) / total if total else 1.0,
    }


def _bootstrap_seed(binding: str, margin: float, probability: float) -> int:
    material = f"dalmuti-v4-fallback-bootstrap-v1:{binding}:{margin!r}:{probability!r}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "little")


def trajectory_cluster_bootstrap(
    selected: np.ndarray,
    agreement: np.ndarray,
    trajectory_indexes: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    if isinstance(resamples, bool) or resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    actor_trajectories = np.unique(trajectory_indexes[selected])
    if len(actor_trajectories) == 0:
        return {
            "unit": "actor-containing-trajectory",
            "method": "deterministic-pcg64-percentile-cluster-bootstrap",
            "clusters": 0,
            "resamples": resamples,
            "seed": seed,
            "mean": None,
            "low": None,
            "high": None,
        }
    numerators = np.asarray(
        [int((selected & agreement & (trajectory_indexes == item)).sum()) for item in actor_trajectories],
        dtype=np.float64,
    )
    denominators = np.asarray(
        [int((selected & (trajectory_indexes == item)).sum()) for item in actor_trajectories],
        dtype=np.float64,
    )
    mean = float(numerators.sum() / denominators.sum())
    if len(actor_trajectories) == 1:
        low = high = mean
    else:
        generator = np.random.Generator(np.random.PCG64(seed))
        values = np.empty(resamples, dtype=np.float64)
        offset = 0
        while offset < resamples:
            size = min(256, resamples - offset)
            indexes = generator.integers(
                0,
                len(actor_trajectories),
                size=(size, len(actor_trajectories)),
                dtype=np.int64,
            )
            values[offset : offset + size] = (
                numerators[indexes].sum(axis=1) / denominators[indexes].sum(axis=1)
            )
            offset += size
        low, high = np.quantile(values, [0.025, 0.975], method="linear")
        low, high = float(low), float(high)
    return {
        "unit": "actor-containing-trajectory",
        "method": "deterministic-pcg64-percentile-cluster-bootstrap",
        "clusters": int(len(actor_trajectories)),
        "resamples": resamples,
        "seed": seed,
        "mean": mean,
        "low": low,
        "high": high,
    }


def _worst_stratum(
    labels: np.ndarray,
    selected: np.ndarray,
    agreement: np.ndarray,
    *,
    render,
) -> dict[str, object] | None:
    rows = []
    for value in sorted(np.unique(labels).tolist()):
        mask = labels == value
        summary = _summary_counts(selected & mask, agreement, int(mask.sum()))
        summary["stratum"] = render(value)
        rows.append(summary)
    candidates = [row for row in rows if row["actorDecisions"] > 0]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row["actorNormalAgreement"]),
            float(row["actorCoverage"]),
            str(row["stratum"]),
        ),
    )


def _pair_report(
    diagnostics: Mapping[str, np.ndarray],
    *,
    margin: float,
    probability: float,
    binding: str,
    bootstrap_resamples: int,
) -> dict[str, object]:
    legal_counts = diagnostics["legalCounts"]
    forced = legal_counts == 1
    selected = (
        ~forced
        & (diagnostics["margins"] >= margin)
        & (diagnostics["probabilities"] >= probability)
    )
    agreement = diagnostics["actions"] == diagnostics["normalActions"]
    summary = _summary_counts(selected, agreement, len(selected))
    summary.update(
        {
            "minimumLegalLogitMargin": margin,
            "minimumTopProbability": probability,
            "thresholdComparison": "all-configured-thresholds-met-inclusive",
            "forcedDecisions": int(forced.sum()),
            "forcedFallbackDecisions": int(forced.sum()),
            "nonForcedDecisions": int((~forced).sum()),
            "nonForcedActorCoverage": (
                float(selected.sum() / (~forced).sum()) if (~forced).any() else 0.0
            ),
        }
    )
    seed = _bootstrap_seed(binding, margin, probability)
    summary["trajectoryClusterBootstrap95"] = trajectory_cluster_bootstrap(
        selected,
        agreement,
        diagnostics["trajectoryIndexes"],
        seed=seed,
        resamples=bootstrap_resamples,
    )
    summary["worstStrata"] = {
        "playerCount": _worst_stratum(
            diagnostics["playerCounts"], selected, agreement, render=lambda value: f"p{value}"
        ),
        "role": _worst_stratum(
            diagnostics["roles"], selected, agreement, render=lambda value: ROLE_NAMES[int(value)]
        ),
        "act": _worst_stratum(
            diagnostics["acts"], selected, agreement, render=lambda value: f"act-{value}"
        ),
    }
    return summary


def _unique_sweep(
    values: np.ndarray,
    forced: np.ndarray,
    agreement: np.ndarray,
    *,
    label: str,
) -> list[dict[str, object]]:
    thresholds = sorted({float(value) for value in values[~forced]})
    rows = []
    total = len(values)
    for threshold in thresholds:
        selected = ~forced & (values >= threshold)
        counts = _summary_counts(selected, agreement, total)
        rows.append({label: threshold, **counts})
    return rows


def build_calibration_report(
    *,
    prepared: PublicPreparedDataset,
    policy: object,
    actor_binding: str,
    margin_grid: Sequence[float],
    probability_grid: Sequence[float],
    target_agreement_lcb: float,
    minimum_coverage: float,
    bootstrap_resamples: int,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    target = _finite(target_agreement_lcb, "target agreement LCB")
    coverage = _finite(minimum_coverage, "minimum coverage")
    if not 0.0 <= target <= 1.0 or not 0.0 <= coverage <= 1.0:
        raise ValueError("agreement LCB and coverage targets must be in [0, 1]")
    margins = tuple(sorted({_finite(value, "margin grid") for value in margin_grid}))
    probabilities = tuple(
        sorted({_finite(value, "probability grid") for value in probability_grid})
    )
    if not margins or not probabilities:
        raise ValueError("margin and probability grids must be non-empty")
    if min(margins) < 0.0 or min(probabilities) < 0.0 or max(probabilities) > 1.0:
        raise ValueError("threshold grids are outside routing ranges")
    diagnostics = _infer_public_diagnostics(
        prepared, policy, batch_size=batch_size
    )
    forced = diagnostics["legalCounts"] == 1
    agreement = diagnostics["actions"] == diagnostics["normalActions"]
    binding = hashlib.sha256(
        canonical_json_bytes(
            {
                "datasetSha256": prepared.dataset_sha256,
                "actorBindingSha256": actor_binding,
            }
        )
    ).hexdigest()
    pair_reports = [
        _pair_report(
            diagnostics,
            margin=margin,
            probability=probability,
            binding=binding,
            bootstrap_resamples=bootstrap_resamples,
        )
        for margin in margins
        for probability in probabilities
    ]
    eligible = [
        row
        for row in pair_reports
        if row["actorCoverage"] >= coverage
        and row["trajectoryClusterBootstrap95"]["low"] is not None
        and row["trajectoryClusterBootstrap95"]["low"] >= target
    ]
    recommended = sorted(
        eligible,
        key=lambda row: (
            -float(row["actorCoverage"]),
            -float(row["trajectoryClusterBootstrap95"]["low"]),
            float(row["minimumLegalLogitMargin"]),
            float(row["minimumTopProbability"]),
        ),
    )[:1]
    metadata = prepared.metadata
    inputs = metadata.get("inputs")
    source_contracts = []
    if isinstance(inputs, list):
        source_contracts = sorted(
            {
                json.dumps(item.get("sourceHashes"), sort_keys=True, separators=(",", ":"))
                for item in inputs
                if isinstance(item, dict) and isinstance(item.get("sourceHashes"), dict)
            }
        )
    report: dict[str, object] = {
        "format": FORMAT,
        "version": VERSION,
        "purpose": "offline-behavioral-safety-calibration",
        "outcomeSuperiorityEvidence": False,
        "warning": (
            "This report calibrates actor agreement with exact Normal on Normal-sampled "
            "public states; it is not evidence that the actor or routed policy beats Normal."
        ),
        "bindings": {
            "datasetSha256": prepared.dataset_sha256,
            "datasetMetadataSha256": prepared.metadata_sha256,
            "datasetFingerprint": metadata.get("fingerprint"),
            "datasetInputSha256s": sorted(
                str(item.get("sha256"))
                for item in (inputs if isinstance(inputs, list) else [])
                if isinstance(item, dict)
            ),
            "datasetSourceContractJson": source_contracts,
            "actorBindingSha256": actor_binding,
            "bundleActorSha256s": list(getattr(policy, "audit_metadata")["bundleActorSha256s"]),
            "bundleManifestSha256s": list(getattr(policy, "audit_metadata")["bundleManifestSha256s"]),
            "sourceFilesSha256": _source_hashes(),
        },
        "datasetAudit": {
            "trajectories": int(prepared.arrays["actions"].shape[0]),
            "validDecisions": int(len(diagnostics["actions"])),
            "forcedDecisions": int(forced.sum()),
            "nonForcedDecisions": int((~forced).sum()),
            "playerCounts": sorted(np.unique(diagnostics["playerCounts"]).astype(int).tolist()),
            "acts": sorted(np.unique(diagnostics["acts"]).astype(int).tolist()),
            "publicArraysLoaded": list(PUBLIC_ARRAY_NAMES),
            "privilegedArraysLoaded": [],
            "privilegedArraysExplicitlyExcluded": sorted(PRIVILEGED_ARRAY_NAMES),
        },
        "actorAudit": {
            **dict(getattr(policy, "audit_metadata")),
            "allTop1ActionsLegal": True,
            "illegalProbabilityMassMaximum": 0.0,
            "illegalProbabilityMassRule": "illegal logits replaced by negative infinity before softmax",
        },
        "routingContract": {
            "mode": "confidence-fallback",
            "thresholdComparison": "all-configured-thresholds-met-inclusive",
            "forcedDecisionRule": "exact-normal",
            "runtimeErrorFallback": False,
        },
        "targets": {
            "minimumActorNormalAgreementBootstrapLcb95": target,
            "minimumActorCoverage": coverage,
        },
        "exactUniqueThresholdSweeps": {
            "scope": "one-dimensional exact observed thresholds; descriptive counts without bootstrap",
            "marginOnly": _unique_sweep(
                diagnostics["margins"], forced, agreement, label="minimumLegalLogitMargin"
            ),
            "probabilityOnly": _unique_sweep(
                diagnostics["probabilities"], forced, agreement, label="minimumTopProbability"
            ),
        },
        "requestedGrid": {
            "margins": list(margins),
            "probabilities": list(probabilities),
            "thresholdPairs": pair_reports,
        },
        "recommendation": {
            "eligibleThresholdPairCount": len(eligible),
            "recommendedThresholdPairs": [
                {
                    "minimumLegalLogitMargin": row["minimumLegalLogitMargin"],
                    "minimumTopProbability": row["minimumTopProbability"],
                    "actorCoverage": row["actorCoverage"],
                    "actorNormalAgreementBootstrapLcb95": row[
                        "trajectoryClusterBootstrap95"
                    ]["low"],
                }
                for row in recommended
            ],
            "selectionRule": "highest coverage, then highest LCB, then lowest thresholds",
            "outcomeClaim": None,
        },
        "deploymentTriggered": False,
    }
    return report


def validate_calibration_report(report: Mapping[str, object]) -> None:
    if (
        report.get("format") != FORMAT
        or report.get("version") != VERSION
        or report.get("purpose") != "offline-behavioral-safety-calibration"
        or report.get("outcomeSuperiorityEvidence") is not False
        or report.get("deploymentTriggered") is not False
    ):
        raise ValueError("invalid V4 fallback calibration report")
    audit = report.get("datasetAudit")
    routing = report.get("routingContract")
    if (
        not isinstance(audit, Mapping)
        or audit.get("privilegedArraysLoaded") != []
        or not isinstance(routing, Mapping)
        or routing.get("thresholdComparison")
        != "all-configured-thresholds-met-inclusive"
        or routing.get("forcedDecisionRule") != "exact-normal"
    ):
        raise ValueError("fallback calibration safety contract is invalid")


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


def write_calibration_report_exclusive(
    output_path: str | Path, report: Mapping[str, object]
) -> dict[str, object]:
    validate_calibration_report(report)
    output = Path(output_path)
    checksum = output.with_name(output.name + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("calibration report and checksum are immutable")
    payload = canonical_json_bytes(report)
    digest = hashlib.sha256(payload).hexdigest()
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


def _bundle_seed(bundle: str | Path) -> int:
    manifest = json.loads((Path(bundle) / "manifest.json").read_text("utf-8"))
    seed = manifest.get("metadata", {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError("three actor bundles require explicit or metadata actor seeds")
    return seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate V4 actor confidence thresholds against exact Normal labels."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--actor-bundle", action="append", required=True)
    parser.add_argument("--actor-seed", action="append", type=int, default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--margin-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--probability-grid", default="0,0.25,0.5,0.75,0.9")
    parser.add_argument("--target-lcb", type=float, default=0.95)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from v4_evaluate import _load_cli_actor_policy

    bundles = [str(Path(value).resolve()) for value in arguments.actor_bundle]
    seeds = list(arguments.actor_seed)
    if len(bundles) == 3 and not seeds:
        seeds = [_bundle_seed(bundle) for bundle in bundles]
    policy, actor_binding = _load_cli_actor_policy(
        bundles,
        actor_seeds=seeds,
        device=arguments.device,
        compile_actor=False,
    )
    prepared = load_public_prepared_normal_dataset(arguments.dataset)
    if dict(getattr(policy, "config").to_dict()) != prepared.metadata.get("actorConfig"):
        raise ValueError("actor and prepared public tensor configurations differ")
    report = build_calibration_report(
        prepared=prepared,
        policy=policy,
        actor_binding=actor_binding,
        margin_grid=parse_threshold_grid(arguments.margin_grid, probability=False),
        probability_grid=parse_threshold_grid(
            arguments.probability_grid, probability=True
        ),
        target_agreement_lcb=arguments.target_lcb,
        minimum_coverage=arguments.min_coverage,
        bootstrap_resamples=arguments.bootstrap,
        batch_size=arguments.batch,
        device=arguments.device,
    )
    published = write_calibration_report_exclusive(arguments.output, report)
    print(json.dumps(published, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PublicPreparedDataset",
    "build_calibration_report",
    "canonical_json_bytes",
    "legal_diagnostics_from_logits",
    "load_public_prepared_normal_dataset",
    "main",
    "parse_threshold_grid",
    "route_actor_inclusive",
    "trajectory_cluster_bootstrap",
    "validate_calibration_report",
    "write_calibration_report_exclusive",
]
