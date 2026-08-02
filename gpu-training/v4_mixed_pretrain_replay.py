#!/usr/bin/env python3
"""Independent CUDA replay and post-training hard gates for the mixed V4 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from v4_dataset import load_v4_dataset_npz
from v4_export import load_v4_actor_checkpoint, sha256_file, verify_v4_actor_bundle
from v4_mixed_workflow import (
    BACKEND_MAP,
    BEHAVIOR_ACTOR_SHA256,
    BEHAVIOR_MANIFEST_SHA256,
    canonical_json_bytes,
)
from v4_model import (
    canonical_v4_policy_numerics_contract,
    configure_v4_policy_numerics,
)
from v4_objectives import masked_log_probabilities
from v4_train import (
    V4_CUDA_POLICY_AUDIT_BATCH_SIZE,
    _audit_initial_policy_reproduction,
    _batch_to_device,
    _flatten_time,
    _trim_public_padding,
)


# Both independent CUDA replay passes must use the exact training-audit kernel
# shape.  Importing the sealed training constant prevents these trust
# boundaries from silently drifting apart.
V4_MIXED_REPLAY_AUDIT_BATCH_SIZE = V4_CUDA_POLICY_AUDIT_BATCH_SIZE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class _FrozenFile:
    source: Path
    frozen: Path
    sha256: str
    identity: tuple[int, int, int, int, int]


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(stat.S_IMODE(value.st_mode)),
    )


def _same_file(path_stat: os.stat_result, descriptor_stat: os.stat_result) -> bool:
    if int(path_stat.st_size) != int(descriptor_stat.st_size):
        return False
    if int(path_stat.st_ino) and int(descriptor_stat.st_ino):
        return (
            int(path_stat.st_dev) == int(descriptor_stat.st_dev)
            and int(path_stat.st_ino) == int(descriptor_stat.st_ino)
        )
    return True


def _freeze_file(source: Path, destination: Path, label: str) -> _FrozenFile:
    before_path = source.lstat()
    _require(
        stat.S_ISREG(before_path.st_mode) and not source.is_symlink(),
        f"{label} is not a regular file",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, flags)
    output_descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o400,
    )
    digest = hashlib.sha256()
    try:
        before_fd = os.fstat(source_descriptor)
        _require(_same_file(before_path, before_fd), f"{label} changed while opening")
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_descriptor, view)
                view = view[written:]
        os.fsync(output_descriptor)
        after_fd = os.fstat(source_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(output_descriptor)
    after_path = source.lstat()
    _require(_identity(before_fd) == _identity(after_fd), f"{label} changed while reading")
    _require(_identity(before_path) == _identity(after_path), f"{label} path changed while reading")
    _require(_same_file(after_path, after_fd), f"{label} was replaced while reading")
    os.chmod(destination, 0o400)
    return _FrozenFile(
        source=source,
        frozen=destination,
        sha256=digest.hexdigest(),
        identity=_identity(after_path),
    )


def _rehash_and_recheck(snapshot: _FrozenFile, label: str) -> None:
    current = snapshot.source.lstat()
    _require(
        _identity(current) == snapshot.identity
        and stat.S_ISREG(current.st_mode)
        and not snapshot.source.is_symlink(),
        f"{label} changed after freezing",
    )
    _require(sha256_file(snapshot.source) == snapshot.sha256, f"{label} digest changed after freezing")


def _freeze_dataset_and_actor(
    dataset: Path, actor_bundle: Path, temporary_root: Path
) -> tuple[Path, Path, list[_FrozenFile]]:
    frozen_dataset = temporary_root / "dataset" / dataset.name
    dataset_sources = (
        dataset,
        Path(f"{dataset}.sha256"),
        Path(f"{dataset}.metadata.json"),
        Path(f"{dataset}.metadata.json.sha256"),
    )
    snapshots: list[_FrozenFile] = []
    for source in dataset_sources:
        snapshots.append(
            _freeze_file(
                source,
                frozen_dataset.parent / source.name,
                f"dataset artifact {source.name}",
            )
        )
    frozen_actor = temporary_root / "actor"
    for name in ("actor.pt", "manifest.json", "manifest.json.sha256"):
        source = actor_bundle / name
        snapshots.append(
            _freeze_file(source, frozen_actor / name, f"Actor artifact {name}")
        )
    npz_digest = snapshots[0].sha256
    _require(
        snapshots[1].frozen.read_bytes()
        == f"{npz_digest}  {dataset.name}\n".encode("ascii"),
        "dataset checksum sidecar is stale",
    )
    metadata_snapshot = snapshots[2]
    _require(
        snapshots[3].frozen.read_bytes()
        == f"{metadata_snapshot.sha256}  {metadata_snapshot.frozen.name}\n".encode(
            "ascii"
        ),
        "dataset metadata checksum sidecar is stale",
    )
    return frozen_dataset, frozen_actor, snapshots


def _canonical_json(path: Path, label: str) -> Mapping[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(payload == canonical_json_bytes(value), f"{label} is not canonical JSON")
    return value


def _publish(path: Path, value: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{path}.sha256")
    _require(not path.exists() and not sidecar.exists(), "immutable audit output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for target, data in (
        (path, payload),
        (sidecar, f"{digest}  {path.name}\n".encode("ascii")),
    ):
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
    return digest


def _fixed_plan(metadata: Mapping[str, object]) -> Mapping[str, object]:
    loss = metadata.get("lossEligibility")
    _require(isinstance(loss, Mapping), "merged metadata lacks loss eligibility")
    plans = loss.get("fixedCollectionPlans")
    _require(isinstance(plans, list) and len(plans) == 1, "merged dataset must bind one fixed collection plan")
    plan = plans[0]
    _require(isinstance(plan, Mapping), "fixed collection plan is invalid")
    fields = plan.get("canonicalFields")
    _require(isinstance(fields, Mapping), "fixed collection plan fields are missing")
    _require(fields.get("version") == 2, "mixed replay requires fixed collection plan version 2")
    _require(fields.get("matchShardCount") == 14, "mixed replay requires fourteen shards")
    _require(
        fields.get("shardBackendMap")
        == {str(index): backend for index, backend in enumerate(BACKEND_MAP)},
        "mixed replay backend map drifted",
    )
    digest = plan.get("canonicalSha256")
    _require(isinstance(digest, str) and len(digest) == 64, "fixed collection plan SHA-256 is invalid")
    opaque = plan.get("opaqueId")
    _require(opaque == f"fixed-complete-mixed-backend-shard-plan-v2:sha256={digest}", "fixed collection plan opaque ID drifted")
    return plan


def _grouped_replay(
    actor: torch.nn.Module,
    dataset: object,
    dataset_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    with np.load(dataset_path, allow_pickle=False) as archive:
        match_indices = np.asarray(archive["trajectory_match_indices"], dtype=np.int64)
        player_counts = np.asarray(archive["trajectory_player_counts"], dtype=np.int64)
    _require(len(match_indices) == len(dataset), "trajectory shard provenance length drifted")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    groups: dict[tuple[int, int, str], dict[str, float | int]] = {}
    trajectory_offset = 0
    was_training = actor.training
    actor.eval()
    try:
        with torch.no_grad():
            for cpu_batch in loader:
                trajectory_count = int(cpu_batch["actions"].shape[0])
                batch_match_indices = match_indices[
                    trajectory_offset : trajectory_offset + trajectory_count
                ]
                batch_player_counts = player_counts[
                    trajectory_offset : trajectory_offset + trajectory_count
                ]
                trajectory_offset += trajectory_count
                batch = _batch_to_device(_trim_public_padding(cpu_batch), device)
                valid = batch["valid_masks"].reshape(-1)
                eligible = batch["ppo_eligible_masks"].reshape(-1) & valid
                if not bool(eligible.any()):
                    continue
                legal = _flatten_time(batch["legal_masks"]).clone()
                legal[~valid, 0] = True
                with torch.cuda.amp.autocast(enabled=False):
                    logits = actor(
                        _flatten_time(batch["global_features"]).float(),
                        _flatten_time(batch["rank_features"]).float(),
                        _flatten_time(batch["player_features"]).float(),
                        _flatten_time(batch["player_mask"]),
                        _flatten_time(batch["memory_trace_features"]).float(),
                        _flatten_time(batch["history_features"]).float(),
                        _flatten_time(batch["history_mask"]),
                        legal,
                    ).float()
                log_probabilities = masked_log_probabilities(
                    logits[eligible], legal[eligible]
                ).float()
                actions = batch["actions"].reshape(-1)[eligible]
                current = log_probabilities.gather(1, actions[:, None]).squeeze(1)
                old = batch["old_action_log_probs"].reshape(-1)[eligible].float()
                absolute_errors = (current.to(torch.float64) - old.to(torch.float64)).abs()
                _require(bool(torch.isfinite(absolute_errors).all()), "stratified replay produced non-finite error")
                time_steps = int(batch["valid_masks"].shape[1])
                repeated_matches = np.repeat(batch_match_indices, time_steps)
                repeated_players = np.repeat(batch_player_counts, time_steps)
                selected = eligible.detach().cpu().numpy().astype(bool, copy=False)
                selected_matches = repeated_matches[selected]
                selected_players = repeated_players[selected]
                selected_errors = absolute_errors.detach().cpu().numpy()
                for player_count, match_index, error in zip(
                    selected_players.tolist(),
                    selected_matches.tolist(),
                    selected_errors.tolist(),
                ):
                    shard_index = int(match_index) % 14
                    backend = BACKEND_MAP[shard_index]
                    key = (int(player_count), shard_index, backend)
                    record = groups.setdefault(
                        key, {"count": 0, "errorSum": 0.0, "maximum": 0.0}
                    )
                    record["count"] = int(record["count"]) + 1
                    record["errorSum"] = float(record["errorSum"]) + float(error)
                    record["maximum"] = max(float(record["maximum"]), float(error))
    finally:
        actor.train(was_training)
    _require(trajectory_offset == len(dataset), "stratified replay did not traverse the full dataset")
    rows = []
    for (player_count, shard_index, backend), record in sorted(groups.items()):
        count = int(record["count"])
        rows.append(
            {
                "backend": backend,
                "count": count,
                "maximumAbsoluteLogProbabilityError": float(record["maximum"]),
                "meanAbsoluteLogProbabilityError": float(record["errorSum"]) / count,
                "playerCount": player_count,
                "shardIndex": shard_index,
            }
        )
    _require({row["shardIndex"] for row in rows} == set(range(14)), "stratified replay is missing a shard")
    _require({row["playerCount"] for row in rows} == set(range(4, 11)), "stratified replay is missing a player count")
    return {"byPlayerCountShardAndBackend": rows}


def replay(
    dataset_path: Path,
    actor_bundle: Path,
    output: Path,
    *,
    device_name: str,
    tolerance: float,
) -> Mapping[str, object]:
    _require(tolerance == 2.0e-5, "mixed replay tolerance is immutable at 2e-5")
    _require(device_name == "cuda", "mixed pretraining replay must use CUDA")
    _require(torch.cuda.is_available(), "CUDA replay requested but CUDA is unavailable")
    with tempfile.TemporaryDirectory(prefix="dalmuti-v4-mixed-replay-") as temporary:
        frozen_dataset_path, frozen_actor_bundle, snapshots = (
            _freeze_dataset_and_actor(
                dataset_path.resolve(strict=True),
                actor_bundle.resolve(strict=True),
                Path(temporary),
            )
        )
        manifest = verify_v4_actor_bundle(frozen_actor_bundle)
        files = manifest.get("files")
        _require(isinstance(files, Mapping) and isinstance(files.get("actor.pt"), Mapping), "actor bundle manifest is incomplete")
        actor_sha = files["actor.pt"].get("sha256")
        manifest_sha = sha256_file(frozen_actor_bundle / "manifest.json")
        _require(actor_sha == BEHAVIOR_ACTOR_SHA256, "behavior Actor SHA-256 drifted")
        _require(manifest_sha == BEHAVIOR_MANIFEST_SHA256, "behavior manifest SHA-256 drifted")
        dataset = load_v4_dataset_npz(frozen_dataset_path)
        plan = _fixed_plan(dataset.metadata)
        actor, _ = load_v4_actor_checkpoint(frozen_actor_bundle / "actor.pt")
        device = torch.device("cuda")
        policy_numerics = configure_v4_policy_numerics(device)
        actor = actor.to(device)
        audit = _audit_initial_policy_reproduction(
            actor,
            dataset,
            device=device,
            batch_size=V4_MIXED_REPLAY_AUDIT_BATCH_SIZE,
            num_workers=0,
            clip_ratio=0.12,
        )
        _require(
            float(audit["maximumAbsoluteLogProbabilityError"]) <= tolerance,
            "independent CUDA replay exceeded 2e-5",
        )
        strata = _grouped_replay(
            actor,
            dataset,
            frozen_dataset_path,
            device=device,
            batch_size=V4_MIXED_REPLAY_AUDIT_BATCH_SIZE,
        )
        rows = strata["byPlayerCountShardAndBackend"]
        assert isinstance(rows, list)
        _require(
            max(float(row["maximumAbsoluteLogProbabilityError"]) for row in rows)
            <= tolerance,
            "a replay stratum exceeded 2e-5",
        )
        for snapshot in snapshots:
            _rehash_and_recheck(snapshot, f"input artifact {snapshot.source.name}")
        value: dict[str, object] = {
            "actorSha256": actor_sha,
            "audit": audit,
            "datasetFingerprint": dataset.fingerprint,
            "datasetSha256": snapshots[0].sha256,
            "device": "cuda",
            "fixedCollectionPlanSha256": plan["canonicalSha256"],
            "format": "dalmuti-v4-mixed-pretraining-replay",
            "manifestSha256": manifest_sha,
            "passed": True,
            "policyNumerics": policy_numerics,
            "strata": strata,
            "version": 1,
        }
        digest = _publish(output, value)
        for snapshot in snapshots:
            _rehash_and_recheck(snapshot, f"input artifact {snapshot.source.name}")
    return {**value, "reportSha256": digest}


def _verify_training_gates_frozen(
    training_result: Path,
    run_manifest: Path,
    candidate: Path,
    output: Path,
    *,
    maximum_approx_kl: float,
    maximum_clip_fraction: float,
    minimum_entropy_retention: float,
) -> Mapping[str, object]:
    _require(maximum_approx_kl == 0.020, "maximum KL gate drifted")
    _require(maximum_clip_fraction == 0.25, "maximum clip-fraction gate drifted")
    _require(minimum_entropy_retention == 0.70, "minimum entropy-retention gate drifted")
    result = _canonical_json(training_result, "training result")
    manifest = _canonical_json(run_manifest, "training run manifest")
    candidate_manifest_path = candidate / "manifest.json"
    candidate_actor_path = candidate / "actor.pt"
    candidate_manifest = _canonical_json(
        candidate_manifest_path, "candidate manifest"
    )
    candidate_actor_sha = sha256_file(candidate_actor_path)
    candidate_manifest_sha = sha256_file(candidate_manifest_path)
    candidate_files = candidate_manifest.get("files")
    candidate_actor_record = (
        candidate_files.get("actor.pt")
        if isinstance(candidate_files, Mapping)
        else None
    )
    _require(
        candidate_manifest.get("format") == "dalmuti-v4-candidate-manifest"
        and candidate_manifest.get("version") == 2
        and isinstance(candidate_actor_record, Mapping)
        and candidate_actor_record.get("sha256") == candidate_actor_sha
        and candidate_actor_record.get("bytes") == candidate_actor_path.stat().st_size,
        "candidate Actor manifest binding drifted",
    )
    _require(
        (candidate / "manifest.json.sha256").read_bytes()
        == f"{candidate_manifest_sha}  manifest.json\n".encode("ascii")
        and (candidate / "actor.pt.sha256").read_bytes()
        == f"{candidate_actor_sha}  actor.pt\n".encode("ascii"),
        "candidate checksum sidecar is stale or malformed",
    )
    _require(
        result.get("candidate") == candidate_manifest,
        "training result does not bind the exact candidate manifest",
    )
    _require(result.get("format") == "dalmuti-v4-training-result", "unsupported training result")
    _require(
        manifest.get("format") == "dalmuti-v4-training-run"
        and manifest.get("device") == "cuda"
        and manifest.get("ampEnabled") is True
        and manifest.get("privilegedCriticExported") is False,
        "training run device or AMP contract drifted",
    )
    _require(result.get("completedEpochs") == 1, "only exactly one training epoch is eligible")
    initial_actor = {
        "actorSha256": BEHAVIOR_ACTOR_SHA256,
        "manifestSha256": BEHAVIOR_MANIFEST_SHA256,
    }
    _require(manifest.get("initialActor") == initial_actor, "training initial Actor drifted")
    training_config = manifest.get("trainingConfig")
    _require(isinstance(training_config, Mapping), "training configuration is missing")
    expected_plan_sha = training_config.get("expected_fixed_collection_plan_sha256")
    _require(
        isinstance(expected_plan_sha, str) and len(expected_plan_sha) == 64,
        "training fixed collection plan binding is invalid",
    )
    _require(
        training_config
        == {
            "actor_learning_rate": 2.0e-5,
            "amp": True,
            "batch_size": 2,
            "bc_weight": 0.05,
            "checkpoint_every": 1,
            "clip_ratio": 0.12,
            "critic_learning_rate": 2.0e-4,
            "critic_weight": 0.2,
            "entropy_coefficient": 0.0005,
            "epochs": 1,
            "expected_fixed_collection_plan_sha256": expected_plan_sha,
            "gamma": 1.0,
            "gradient_accumulation": 1,
            "lambda_": 0.95,
            "max_gradient_norm": 1.0,
            "num_workers": 0,
            "ppo_weight": 1.0,
            "q_boost_coefficient": 0.0,
            "seed": 670000001,
            "weight_decay": 0.0001,
        },
        "training hyperparameters drifted",
    )
    contract = result.get("trainingContract")
    _require(
        isinstance(contract, Mapping)
        and manifest.get("trainingContract") == contract,
        "result and run-manifest training contracts differ",
    )
    fixed_ppo_execution = contract.get("fixedPpoExecutionContract")
    player_count_balance = contract.get("playerCountBalancedLoss")
    canonical_policy_numerics = canonical_v4_policy_numerics_contract()
    _require(
        isinstance(fixed_ppo_execution, Mapping)
        and type(fixed_ppo_execution.get("version")) is int
        and fixed_ppo_execution.get("version") == 2
        and fixed_ppo_execution.get("policyNumerics")
        == canonical_policy_numerics,
        "trainer fixed PPO execution policy numerics contract is missing or non-canonical",
    )
    _require(
        isinstance(player_count_balance, Mapping)
        and player_count_balance.get("fixedPpoPolicyNumerics")
        == canonical_policy_numerics,
        "trainer player-count balance policy numerics contract is missing or non-canonical",
    )
    _require(
        contract.get("requestedWeights")
        == {"behaviorCloning": 0.05, "critic": 0.2, "ppo": 1.0}
        and contract.get("ppoBehaviorActorSha256s") == [BEHAVIOR_ACTOR_SHA256]
        and contract.get("fixedCollectionPlanSha256") == expected_plan_sha,
        "training weights, behavior Actor, or plan binding drifted",
    )
    plan_ids = contract.get("fixedCollectionPlanIds")
    _require(
        plan_ids
        == [
            "fixed-complete-mixed-backend-shard-plan-v2:sha256="
            + expected_plan_sha
        ],
        "training did not use the mixed version-2 collection plan",
    )
    initial_replay = contract.get("initialPolicyReproductionAudit")
    _require(
        isinstance(initial_replay, Mapping)
        and initial_replay.get("passed") is True
        and float(initial_replay.get("maximumAbsoluteLogProbabilityError", math.inf))
        <= 2.0e-5
        and contract.get("initialPolicyReproductionAuditFingerprint")
        == hashlib.sha256(canonical_json_bytes(initial_replay)).hexdigest(),
        "trainer mandatory initial replay is missing or failed",
    )
    audit = result.get("finalPostEpochPolicyDriftAudit")
    _require(isinstance(audit, Mapping), "training result lacks final policy audit")
    expected_fingerprint = hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
    _require(
        result.get("finalPostEpochPolicyDriftAuditFingerprint") == expected_fingerprint,
        "final policy audit fingerprint drifted",
    )
    approx_kl = audit.get("approxKl")
    clip_fraction = audit.get("clipFraction")
    entropy_retention = audit.get("entropyRetentionRatio")
    for value, label in (
        (approx_kl, "approximate KL"),
        (clip_fraction, "clip fraction"),
        (entropy_retention, "entropy retention"),
    ):
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"{label} is invalid",
        )
    _require(abs(float(approx_kl)) <= maximum_approx_kl, "absolute KL hard gate failed")
    _require(float(clip_fraction) <= maximum_clip_fraction, "clip-fraction hard gate failed")
    _require(float(entropy_retention) >= minimum_entropy_retention, "entropy-retention hard gate failed")
    _require(audit.get("entropyCollapseExceeds30Percent") is False, "entropy-collapse hard gate failed")
    _require(
        audit.get("actorForwardDtype") == "torch.float32"
        and audit.get("actorAutocastEnabled") is False
        and audit.get("actorMode") == "eval",
        "post-epoch Actor precision contract drifted",
    )
    dataset_fingerprint = result.get("datasetFingerprint")
    candidate = result.get("candidate")
    candidate_metadata = (
        candidate.get("metadata") if isinstance(candidate, Mapping) else None
    )
    _require(
        isinstance(dataset_fingerprint, str)
        and manifest.get("datasetFingerprint") == dataset_fingerprint
        and audit.get("datasetFingerprint") == dataset_fingerprint
        and audit.get("fixedCollectionPlanSha256") == expected_plan_sha
        and isinstance(candidate_metadata, Mapping)
        and candidate_metadata.get("datasetFingerprint") == dataset_fingerprint
        and candidate_metadata.get("initialActor") == initial_actor
        and candidate_metadata.get("seed") == 670000001,
        "dataset, plan, initial Actor, or training seed binding drifted",
    )
    per_player = audit.get("perPlayerCount")
    _require(isinstance(per_player, Mapping) and set(per_player) == {str(p) for p in range(4, 11)}, "post-epoch audit lacks p4-p10 strata")
    value: dict[str, object] = {
        "approxKl": float(approx_kl),
        "clipFraction": float(clip_fraction),
        "entropyRetentionRatio": float(entropy_retention),
        "format": "dalmuti-v4-mixed-training-hard-gates",
        "passed": True,
        "perPlayerCount": per_player,
        "candidateActorSha256": candidate_actor_sha,
        "candidateManifestSha256": candidate_manifest_sha,
        "fixedCollectionPlanSha256": expected_plan_sha,
        "runManifestSha256": hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest(),
        "trainingResultSha256": hashlib.sha256(
            canonical_json_bytes(result)
        ).hexdigest(),
        "version": 1,
    }
    digest = _publish(output, value)
    return {**value, "reportSha256": digest}


def verify_training_gates(
    training_result: Path,
    run_manifest: Path,
    candidate: Path,
    output: Path,
    *,
    maximum_approx_kl: float,
    maximum_clip_fraction: float,
    minimum_entropy_retention: float,
) -> Mapping[str, object]:
    """Verify hard gates against one immutable result/manifest/Actor snapshot."""

    with tempfile.TemporaryDirectory(prefix="dalmuti-v4-training-gates-") as temporary:
        frozen_root = Path(temporary)
        frozen_candidate = frozen_root / "candidate"
        snapshots = [
            _freeze_file(
                training_result,
                frozen_root / "result.json",
                "training result",
            ),
            _freeze_file(
                run_manifest,
                frozen_root / "run-manifest.json",
                "training run manifest",
            ),
        ]
        for name in (
            "actor.pt",
            "actor.pt.sha256",
            "manifest.json",
            "manifest.json.sha256",
        ):
            snapshots.append(
                _freeze_file(
                    candidate / name,
                    frozen_candidate / name,
                    f"candidate artifact {name}",
                )
            )
        for snapshot in snapshots:
            _rehash_and_recheck(snapshot, f"hard-gate input {snapshot.source.name}")
        value = _verify_training_gates_frozen(
            frozen_root / "result.json",
            frozen_root / "run-manifest.json",
            frozen_candidate,
            output,
            maximum_approx_kl=maximum_approx_kl,
            maximum_clip_fraction=maximum_clip_fraction,
            minimum_entropy_retention=minimum_entropy_retention,
        )
        for snapshot in snapshots:
            _rehash_and_recheck(snapshot, f"hard-gate input {snapshot.source.name}")
        return value


def verify_promotion_gates(
    screening_report: Path,
    output: Path,
    *,
    minimum_mean: float,
    minimum_lower: float,
    minimum_pairwise: float,
) -> Mapping[str, object]:
    _require(minimum_mean == 0.25, "mean chip-difference gate drifted")
    _require(minimum_lower == 0.15, "clustered lower-bound gate drifted")
    _require(minimum_pairwise == 0.55, "pairwise gate drifted")
    sidecar = Path(f"{screening_report}.sha256")
    with tempfile.TemporaryDirectory(prefix="dalmuti-v4-promotion-") as temporary:
        frozen_root = Path(temporary)
        report_snapshot = _freeze_file(
            screening_report,
            frozen_root / screening_report.name,
            "screening report",
        )
        sidecar_snapshot = _freeze_file(
            sidecar,
            frozen_root / sidecar.name,
            "screening report sidecar",
        )
        report_sha = report_snapshot.sha256
        _require(
            sidecar_snapshot.frozen.read_bytes()
            == f"{report_sha}  {screening_report.name}\n".encode("ascii"),
            "screening sidecar is stale or malformed",
        )
        report = _canonical_json(report_snapshot.frozen, "screening report")
        _require(
            report.get("format") == "dalmuti-model-benchmark"
            and report.get("evaluationMode") == "screening"
            and report.get("playerCounts") == list(range(4, 11)),
            "screening report contract drifted",
        )
        _require(
            report.get("seed") == 450000001
            and isinstance(report.get("seedFamily"), Mapping)
            and report["seedFamily"].get("id")
            == "attempt004-screening-seed450000001",
            "screening seed family drifted",
        )
        _require(
            report.get("matchCountsByPlayerCount")
            == {str(player_count): 60 for player_count in range(4, 11)},
            "screening match counts drifted",
        )
        policy = report.get("candidatePolicy")
        routing = policy.get("routing") if isinstance(policy, Mapping) else None
        _require(
            isinstance(policy, Mapping)
            and policy.get("actorCount") == 1
            and isinstance(routing, Mapping)
            and routing.get("mode") == "pure-actor"
            and routing.get("runtimeErrorFallback") is False,
            "screening is not a single pure Actor",
        )
        results = report.get("results")
        _require(isinstance(results, list) and len(results) == 7, "screening lacks p4-p10 results")
        by_player: dict[str, object] = {}
        for expected_player, raw in zip(range(4, 11), results):
            _require(isinstance(raw, Mapping), "screening result is invalid")
            inference = raw.get("meanChipDifferenceInference")
            pairwise = raw.get("pairwiseCandidateBeforeNormal")
            clusters = raw.get("matchClusters")
            mean = raw.get("meanChipDifference")
            lower = inference.get("low") if isinstance(inference, Mapping) else None
            rate = pairwise.get("rate") if isinstance(pairwise, Mapping) else None
            for value, label in ((mean, "mean"), (lower, "lower bound"), (rate, "pairwise rate")):
                _require(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value)),
                    f"screening {label} is invalid",
                )
            _require(
                raw.get("playerCount") == expected_player
                and raw.get("matches") == 60
                and raw.get("actsPerMatch") == 5
                and isinstance(clusters, Mapping)
                and clusters.get("count") == 60
                and isinstance(inference, Mapping)
                and inference.get("clusters") == 60
                and inference.get("resamples") == 10000
                and inference.get("unit") == "seed-matched-match",
                f"screening p{expected_player} design drifted",
            )
            passed = (
                float(mean) >= minimum_mean
                and float(lower) >= minimum_lower
                and float(rate) >= minimum_pairwise
            )
            _require(passed, f"screening p{expected_player} did not pass all promotion gates")
            by_player[str(expected_player)] = {
                "clustered95LowerBound": float(lower),
                "meanChipDifferencePerAct": float(mean),
                "pairwiseBeforeNormal": float(rate),
                "passed": True,
            }
        value: dict[str, object] = {
            "allPlayerCountsPassed": True,
            "format": "dalmuti-v4-mixed-promotion-gates",
            "gates": {
                "minimumClustered95LowerBound": minimum_lower,
                "minimumMeanChipDifferencePerAct": minimum_mean,
                "minimumPairwiseBeforeNormal": minimum_pairwise,
            },
            "passed": True,
            "perPlayerCount": by_player,
            "screeningReportSha256": report_sha,
            "version": 1,
        }
        _rehash_and_recheck(report_snapshot, "screening report")
        _rehash_and_recheck(sidecar_snapshot, "screening report sidecar")
        digest = _publish(output, value)
        _rehash_and_recheck(report_snapshot, "screening report")
        _rehash_and_recheck(sidecar_snapshot, "screening report sidecar")
        return {**value, "reportSha256": digest}


def publish_candidate_sidecar(candidate: Path) -> Mapping[str, object]:
    manifest = verify_v4_actor_bundle(candidate)
    files = manifest.get("files")
    actor_record = files.get("actor.pt") if isinstance(files, Mapping) else None
    _require(isinstance(actor_record, Mapping), "candidate manifest lacks actor.pt")
    expected = actor_record.get("sha256")
    actor = candidate / "actor.pt"
    _require(
        isinstance(expected, str) and sha256_file(actor) == expected,
        "candidate Actor digest drifted",
    )
    sidecar = candidate / "actor.pt.sha256"
    _require(not sidecar.exists() and not sidecar.is_symlink(), "candidate Actor sidecar is immutable")
    payload = f"{expected}  actor.pt\n".encode("ascii")
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    _require(
        sidecar.read_bytes() == payload and sha256_file(actor) == expected,
        "candidate Actor changed while publishing its sidecar",
    )
    return {
        "actorSha256": expected,
        "format": "dalmuti-v4-candidate-actor-sidecar",
        "passed": True,
        "sidecar": str(sidecar),
        "version": 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--dataset", type=Path, required=True)
    replay_parser.add_argument("--actor-bundle", type=Path, required=True)
    replay_parser.add_argument("--device", choices=("cuda",), required=True)
    replay_parser.add_argument(
        "--maximum-absolute-log-probability-error", type=float, required=True
    )
    replay_parser.add_argument("--output", type=Path, required=True)
    gates = commands.add_parser("verify-training-gates")
    gates.add_argument("--training-result", type=Path, required=True)
    gates.add_argument("--run-manifest", type=Path, required=True)
    gates.add_argument("--candidate", type=Path, required=True)
    gates.add_argument("--maximum-approx-kl", type=float, required=True)
    gates.add_argument("--maximum-clip-fraction", type=float, required=True)
    gates.add_argument("--minimum-entropy-retention", type=float, required=True)
    gates.add_argument("--output", type=Path, required=True)
    promotion = commands.add_parser("verify-promotion-gates")
    promotion.add_argument("--screening-report", type=Path, required=True)
    promotion.add_argument(
        "--minimum-mean-chip-difference-per-act", type=float, required=True
    )
    promotion.add_argument(
        "--minimum-clustered-95-lower-bound", type=float, required=True
    )
    promotion.add_argument(
        "--minimum-pairwise-before-normal", type=float, required=True
    )
    promotion.add_argument("--output", type=Path, required=True)
    candidate = commands.add_parser("publish-candidate-sidecar")
    candidate.add_argument("--candidate", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "replay":
        result = replay(
            arguments.dataset,
            arguments.actor_bundle,
            arguments.output,
            device_name=arguments.device,
            tolerance=arguments.maximum_absolute_log_probability_error,
        )
    elif arguments.command == "verify-training-gates":
        result = verify_training_gates(
            arguments.training_result,
            arguments.run_manifest,
            arguments.candidate,
            arguments.output,
            maximum_approx_kl=arguments.maximum_approx_kl,
            maximum_clip_fraction=arguments.maximum_clip_fraction,
            minimum_entropy_retention=arguments.minimum_entropy_retention,
        )
    elif arguments.command == "verify-promotion-gates":
        result = verify_promotion_gates(
            arguments.screening_report,
            arguments.output,
            minimum_mean=arguments.minimum_mean_chip_difference_per_act,
            minimum_lower=arguments.minimum_clustered_95_lower_bound,
            minimum_pairwise=arguments.minimum_pairwise_before_normal,
        )
    else:
        result = publish_candidate_sidecar(arguments.candidate)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
