from __future__ import annotations

"""Evaluation-aligned, fixed-identity five-act PPO rollout collection.

This collector intentionally has a preparation format distinct from the legacy
per-act rotating league collector.  The candidate identity set is chosen once
from the *initial* seating by the exact evaluation rotation and remains fixed
for the complete five-act match.  Exactly one candidate identity is the
stochastic learner; the other candidate identities use the frozen Actor
greedily and every remaining identity uses exact Normal.

The public Actor snapshot and the privileged critic vector are serialized in
separate arrays.  Only the learner's public decisions are trajectories.  An
act-t trajectory receives the gamma=1 suffix return through act five, while
raw per-act candidate-group-vs-Normal-group components remain separately
bound for reward ablations.
"""

import argparse
from dataclasses import dataclass, fields
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from v4_collect_dagger import (
    PUBLIC_MODEL_INPUT_FIELDS,
    _configure_determinism,
    _deterministic_npz_bytes,
    _exclusive_publish,
    _snapshot_public,
    _validate_actor_contract,
    audit_hidden_state_privacy,
)
from v4_collect_ppo import (
    CANONICAL_PRIVILEGED_LAYOUT,
    CANONICAL_PRIVILEGED_LAYOUT_ID,
    CANONICAL_PRIVILEGED_LAYOUT_SHA256,
    NAMESPACE_CHARACTERS,
    _allocate_arrays,
    _batch_candidate_logits,
    _canonical_text,
    _derive_uint32,
    _finite_stats,
    _keyed_uniform,
    _sha256_bytes,
    assert_canonical_privileged_layout,
    masked_categorical_probabilities,
    sample_masked_categorical,
)
from v4_dataset import (
    V4_DATASET_FORMAT,
    V4_DATASET_VERSION,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
    fixed_match_shard_identity_sha256,
)
from v4_compare_fixed_match_backends import (
    FixedMatchBackendCalibrationVerification,
    load_verified_fixed_match_backend_calibration,
)
from v4_env import (
    ACTION_COUNT,
    PRIVILEGED_STATE_SIZE,
    ROLES,
    DalmutiScalarEnv,
    V4ActorObservation,
    role_for_index,
    round_chip_award,
)
from v4_export import canonical_json_bytes, load_v4_actor_checkpoint, sha256_file, verify_v4_actor_bundle
from v4_model import V4ActorConfig, V4CriticConfig
from v4_ppo_advantages import BASELINE_FALLBACK_HIERARCHY, BaselineRecord, BaselineResult, leave_one_match_out_baselines


FIXED_MATCH_PPO_PREPARATION_FORMAT = "dalmuti-v4-ppo-fixed-match-suffix-direct-npz"
FIXED_MATCH_PPO_PREPARATION_VERSION = 1
ACTS_PER_MATCH = 5
DEFAULT_PAIRWISE_COEFFICIENT = 0.25
FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT = "raw-masked-softmax-v1"
FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT_VERSION = 1
FIXED_MATCH_BEHAVIOR_TEMPERATURE = 1.0
FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR = 0.0
FIXED_MATCH_INITIAL_LOG_PROBABILITY_TOLERANCE = 2.0e-5
DEFAULT_MATCH_COUNTS = (
    (4, 320),
    (5, 256),
    (6, 192),
    (7, 160),
    (8, 128),
    (9, 112),
    (10, 96),
)
SOURCE_FILES = (
    "gpu-training/v4_collect_fixed_match_ppo.py",
    "gpu-training/v4_collect_ppo.py",
    "gpu-training/v4_collect_dagger.py",
    "gpu-training/v4_env.py",
    "gpu-training/v4_evaluate.py",
    "gpu-training/v4_model.py",
    "gpu-training/v4_export.py",
    "gpu-training/v4_dataset.py",
    "gpu-training/v4_ppo_advantages.py",
    "gpu-training/v4_compare_fixed_match_backends.py",
    "gpu-training/v3_action_conditioned.py",
    "lib/bot-strategy.ts",
)


@dataclass(frozen=True)
class FixedMatchPPOCollectionConfig:
    run_namespace: str
    seed_base: int
    match_counts: tuple[tuple[int, int], ...] = DEFAULT_MATCH_COUNTS
    match_start: int = 0
    match_shard_count: int = 1
    match_shard_index: int = 0
    temperature: float = FIXED_MATCH_BEHAVIOR_TEMPERATURE
    epsilon_floor: float = FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR
    pairwise_coefficient: float = DEFAULT_PAIRWISE_COEFFICIENT
    standardize_advantages: bool = True
    lane_count: int = 16
    device: str = "cuda"
    resume_existing: bool = False
    shard_backend_map: tuple[str, ...] | None = None
    cross_backend_calibration_report: str | Path | None = None
    cross_backend_calibration_cpu_npz: str | Path | None = None
    cross_backend_calibration_cuda_npz: str | Path | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_namespace, str)
            or not 1 <= len(self.run_namespace) <= 128
            or self.run_namespace[0] not in NAMESPACE_CHARACTERS - {".", "_", "-"}
            or any(character not in NAMESPACE_CHARACTERS for character in self.run_namespace)
        ):
            raise ValueError("run_namespace must use 1..128 safe ASCII characters")
        if (
            isinstance(self.seed_base, bool)
            or not isinstance(self.seed_base, int)
            or not 0 <= self.seed_base <= 0xFFFF_FFFF
        ):
            raise ValueError("seed_base must be a uint32 integer")
        if not isinstance(self.match_counts, tuple) or not self.match_counts:
            raise ValueError("match_counts must be a non-empty tuple")
        player_counts: list[int] = []
        for item in self.match_counts:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("each match_counts entry must be (player_count, matches)")
            player_count, count = item
            if (
                isinstance(player_count, bool)
                or not isinstance(player_count, int)
                or not 4 <= player_count <= 10
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise ValueError("match_counts must bind p4..p10 to positive counts")
            player_counts.append(player_count)
        if player_counts != sorted(set(player_counts)):
            raise ValueError("match_counts player counts must be sorted and unique")
        for name in ("match_start", "match_shard_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("match_shard_count", "lane_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.match_shard_index >= self.match_shard_count:
            raise ValueError("match_shard_index must be in [0, match_shard_count)")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) != FIXED_MATCH_BEHAVIOR_TEMPERATURE
        ):
            raise ValueError(
                "fixed-match PPO requires canonical on-policy temperature=1.0"
            )
        floor = float(self.epsilon_floor)
        if (
            isinstance(self.epsilon_floor, bool)
            or not isinstance(self.epsilon_floor, (int, float))
            or not math.isfinite(floor)
            or floor != FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR
        ):
            raise ValueError(
                "fixed-match PPO requires canonical on-policy epsilon_floor=0.0"
            )
        coefficient = float(self.pairwise_coefficient)
        if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
            raise ValueError("pairwise_coefficient must be finite and in [0, 1]")
        if not isinstance(self.standardize_advantages, bool):
            raise ValueError("standardize_advantages must be boolean")
        if not isinstance(self.resume_existing, bool):
            raise ValueError("resume_existing must be boolean")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string")
        _validate_shard_backend_binding(self)
        calibration_paths = (
            self.cross_backend_calibration_report,
            self.cross_backend_calibration_cpu_npz,
            self.cross_backend_calibration_cuda_npz,
        )
        calibration_is_complete = all(
            path is not None for path in calibration_paths
        )
        calibration_is_absent = all(path is None for path in calibration_paths)
        if (
            self.shard_backend_map is None
            and not calibration_is_absent
        ) or (
            self.shard_backend_map is not None
            and not calibration_is_complete
        ):
            raise ValueError(
                "cross_backend_calibration_report, "
                "cross_backend_calibration_cpu_npz, and "
                "cross_backend_calibration_cuda_npz are required together "
                "iff shard_backend_map is supplied"
            )
        for name, path in zip(
            (
                "cross_backend_calibration_report",
                "cross_backend_calibration_cpu_npz",
                "cross_backend_calibration_cuda_npz",
            ),
            calibration_paths,
            strict=True,
        ):
            if path is not None and (
                not isinstance(path, (str, Path)) or not str(path)
            ):
                raise ValueError(f"{name} must be a path")


def _validate_shard_backend_binding(
    config: FixedMatchPPOCollectionConfig,
) -> str | None:
    """Fail before collection when a mixed-plan shard uses the wrong backend."""

    backend_map = config.shard_backend_map
    if backend_map is None:
        return None
    if (
        not isinstance(backend_map, tuple)
        or len(backend_map) != config.match_shard_count
        or any(backend not in {"cpu", "cuda"} for backend in backend_map)
        or set(backend_map) != {"cpu", "cuda"}
    ):
        raise ValueError(
            "shard_backend_map must be a complete mixed cpu/cuda tuple with "
            "one entry per match shard"
        )
    try:
        actual_backend = torch.device(config.device).type
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("device must be a valid torch device string") from error
    expected_backend = backend_map[config.match_shard_index]
    if actual_backend != expected_backend:
        raise ValueError(
            f"match shard {config.match_shard_index} is precommitted to "
            f"backend {expected_backend}, not {actual_backend}"
        )
    return expected_backend


def _shard_backend_map_record(
    config: FixedMatchPPOCollectionConfig,
) -> dict[str, str] | None:
    if config.shard_backend_map is None:
        return None
    return {
        str(index): backend
        for index, backend in enumerate(config.shard_backend_map)
    }


@dataclass(frozen=True)
class FixedMatchPPOCollectionResult:
    output_path: Path
    metadata_path: Path
    checksum_path: Path
    metadata_checksum_path: Path
    npz_sha256: str
    metadata_sha256: str
    fingerprint: str
    trajectories: int
    samples: int
    complete_matches: int


@dataclass(frozen=True)
class _MatchSpec:
    player_count: int
    match_index: int
    seed: int
    learner_initial_seat: int
    learner_physical_id: int


@dataclass
class _Lane:
    spec: _MatchSpec
    env: DalmutiScalarEnv
    initial_order: tuple[int, ...]
    candidate_initial_seats: tuple[int, ...]
    candidate_ids: frozenset[int]
    learner_id: int
    decision_index: int = 0


def evaluation_candidate_initial_seats(player_count: int, match_index: int) -> tuple[int, ...]:
    """Independent byte-for-byte semantic copy of evaluator seat rotation."""

    if isinstance(player_count, bool) or not isinstance(player_count, int) or not 4 <= player_count <= 10:
        raise ValueError("player_count must be from 4 through 10")
    if isinstance(match_index, bool) or not isinstance(match_index, int) or match_index < 0:
        raise ValueError("match_index must be non-negative")
    lower = player_count // 2
    candidate_count = lower if player_count % 2 == 0 or match_index % 2 == 1 else lower + 1
    extras_before = (match_index + 1) // 2 if player_count % 2 else 0
    assigned_before = match_index * lower + extras_before
    start = assigned_before % player_count
    return tuple((start + offset) % player_count for offset in range(candidate_count))


def _option_capacity_matching(
    keys: Sequence[int],
    options_by_key: Mapping[int, Sequence[int]],
    capacities: Sequence[int],
) -> dict[int, int] | None:
    """Deterministic bipartite b-matching from keys to bounded options."""

    slots = tuple(
        (option, slot)
        for option, capacity in enumerate(capacities)
        for slot in range(int(capacity))
    )
    slot_index = {slot: index for index, slot in enumerate(slots)}
    owners = [-1] * len(slots)
    candidate_slots = {
        position: tuple(
            slot_index[(option, slot)]
            for option in options_by_key[int(key)]
            for slot in range(int(capacities[option]))
        )
        for position, key in enumerate(keys)
    }

    def assign(position: int, visited: set[int]) -> bool:
        for candidate_slot in candidate_slots[position]:
            if candidate_slot in visited:
                continue
            visited.add(candidate_slot)
            owner = owners[candidate_slot]
            if owner < 0 or assign(owner, visited):
                owners[candidate_slot] = position
                return True
        return False

    for position in range(len(keys)):
        if not assign(position, set()):
            return None
    result: dict[int, int] = {}
    for slot_index_value, owner in enumerate(owners):
        if owner >= 0:
            result[int(keys[owner])] = int(slots[slot_index_value][0])
    return result if len(result) == len(keys) else None


def balanced_learner_initial_seats(
    player_count: int, match_indexes: Sequence[int]
) -> dict[int, int]:
    """Choose one evaluator-candidate seat per match with max imbalance <= 1."""

    indexes = tuple(int(value) for value in match_indexes)
    if not indexes or len(set(indexes)) != len(indexes) or any(value < 0 for value in indexes):
        raise ValueError("match_indexes must be unique non-negative integers")
    low, high_count = divmod(len(indexes), player_count)
    options = {
        match_index: evaluation_candidate_initial_seats(player_count, match_index)
        for match_index in indexes
    }
    for high_seats in combinations(range(player_count), high_count):
        high_set = set(high_seats)
        capacities = tuple(low + int(seat in high_set) for seat in range(player_count))
        result = _option_capacity_matching(indexes, options, capacities)
        if result is not None:
            counts = [sum(int(value == seat) for value in result.values()) for seat in range(player_count)]
            if max(counts) - min(counts) > 1:
                raise RuntimeError("balanced learner assignment invariant failed")
            return result
    raise RuntimeError("could not balance learner seats within evaluator candidate sets")


def balanced_learner_physical_ids(
    player_count: int,
    initial_orders: Mapping[int, Sequence[int]],
) -> dict[int, int]:
    """Balance the fixed learner physical ID over the complete unsharded plan."""

    indexes = tuple(sorted(int(value) for value in initial_orders))
    if not indexes:
        raise ValueError("initial_orders must be non-empty")
    options: dict[int, tuple[int, ...]] = {}
    for match_index in indexes:
        order = tuple(int(value) for value in initial_orders[match_index])
        if len(order) != player_count or set(order) != set(range(player_count)):
            raise ValueError("each initial order must be an exact physical-ID permutation")
        options[match_index] = tuple(
            order[seat]
            for seat in evaluation_candidate_initial_seats(player_count, match_index)
        )
    low, high_count = divmod(len(indexes), player_count)
    for high_ids in combinations(range(player_count), high_count):
        high_set = set(high_ids)
        capacities = tuple(low + int(actor_id in high_set) for actor_id in range(player_count))
        result = _option_capacity_matching(indexes, options, capacities)
        if result is not None:
            counts = [sum(int(value == actor_id) for value in result.values()) for actor_id in range(player_count)]
            if max(counts) - min(counts) > 1:
                raise RuntimeError("physical learner balance invariant failed")
            return result
    raise RuntimeError("could not balance physical learners within evaluator candidate sets")


def _complete_match_id(spec: _MatchSpec, namespace: str) -> str:
    return (
        f"v4-fixed-match-{namespace}-p{spec.player_count}-m{spec.match_index}"
        f"-seed{spec.seed:08x}"
    )


def _trajectory_id(spec: _MatchSpec, namespace: str, learner_id: int, act: int) -> str:
    return f"{_complete_match_id(spec, namespace)}-learner{learner_id}-act{act}"


def _new_lane(spec: _MatchSpec) -> _Lane:
    env = DalmutiScalarEnv(spec.player_count, acts=ACTS_PER_MATCH, seed=spec.seed, device="cpu")
    initial_order = tuple(int(value) for value in env._order)
    candidate_seats = evaluation_candidate_initial_seats(spec.player_count, spec.match_index)
    candidate_ids = frozenset(initial_order[seat] for seat in candidate_seats)
    learner_id = int(spec.learner_physical_id)
    if initial_order[spec.learner_initial_seat] != learner_id:
        raise RuntimeError("learner physical ID and initial seat binding drifted")
    if learner_id not in candidate_ids:
        raise RuntimeError("learner identity is not in the evaluator candidate set")
    return _Lane(spec, env, initial_order, candidate_seats, candidate_ids, learner_id)


def _source_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required fixed-match PPO source is missing: {relative}")
        result[relative] = sha256_file(source)
    return result


def _sidecar_digest(path: Path, expected_name: str) -> str:
    fields_value = path.read_text(encoding="ascii").split()
    if len(fields_value) != 2 or fields_value[1] != expected_name or len(fields_value[0]) != 64:
        raise ValueError(f"invalid checksum sidecar: {path}")
    return fields_value[0]


def _resume_existing_result(
    output: Path,
    metadata_path: Path,
    checksum_path: Path,
    metadata_checksum_path: Path,
    config: FixedMatchPPOCollectionConfig,
    actor_checkpoint_sha256: str,
    cross_backend_calibration_report_sha256: str | None,
    source_hashes: Mapping[str, str],
) -> FixedMatchPPOCollectionResult | None:
    targets = (output, metadata_path, checksum_path, metadata_checksum_path)
    exists = tuple(target.exists() for target in targets)
    if not any(exists):
        return None
    if not config.resume_existing:
        raise FileExistsError(f"output already exists: {targets[exists.index(True)]}")
    if not all(exists):
        raise ValueError("resume-existing refuses a partial fixed-match shard artifact")
    npz_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if _sidecar_digest(checksum_path, output.name) != npz_sha:
        raise ValueError("resume-existing NPZ checksum mismatch")
    if _sidecar_digest(metadata_checksum_path, metadata_path.name) != metadata_sha:
        raise ValueError("resume-existing metadata checksum mismatch")
    external = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(output, allow_pickle=False) as archive:
        embedded = json.loads(str(archive["metadata_json"].item()))
    expected_external = dict(embedded)
    expected_external["npzSha256"] = npz_sha
    if canonical_json_bytes(external) != canonical_json_bytes(expected_external):
        raise ValueError("resume-existing external and embedded manifests disagree")
    shard = embedded.get("shard")
    collection = embedded.get("collection")
    reward = embedded.get("rewardContract")
    returns = embedded.get("returnsAndAdvantages")
    model = embedded.get("modelBinding")
    execution = embedded.get("execution")
    expected_counts = {str(player_count): count for player_count, count in config.match_counts}
    selected_match_count = sum(
        int(match_index % config.match_shard_count == config.match_shard_index)
        for _, match_count in config.match_counts
        for match_index in range(config.match_start, config.match_start + match_count)
    )
    expected_device = str(torch.device(config.device))
    expected_backend_map = _shard_backend_map_record(config)
    expected_plan_version = 2 if expected_backend_map is not None else 1
    expected_backend = (
        expected_backend_map[str(config.match_shard_index)]
        if expected_backend_map is not None
        else None
    )
    if (
        embedded.get("preparationFormat") != FIXED_MATCH_PPO_PREPARATION_FORMAT
        or embedded.get("preparationVersion") != FIXED_MATCH_PPO_PREPARATION_VERSION
        or not isinstance(shard, Mapping)
        or shard.get("runNamespace") != config.run_namespace
        or shard.get("seedBase") != config.seed_base
        or shard.get("matchCounts") != expected_counts
        or shard.get("matchStart") != config.match_start
        or shard.get("matchShardCount") != config.match_shard_count
        or shard.get("matchShardIndex") != config.match_shard_index
        or (
            expected_plan_version == 1
            and (
                "collectionPlanVersion" in shard
                or "shardBackendMap" in shard
                or "crossBackendCalibrationReportSha256" in shard
            )
        )
        or (
            expected_plan_version == 2
            and (
                shard.get("collectionPlanVersion") != 2
                or shard.get("shardBackendMap") != expected_backend_map
                or shard.get("crossBackendCalibrationReportSha256")
                != cross_backend_calibration_report_sha256
            )
        )
        or not isinstance(collection, Mapping)
        or collection.get("requestedLaneCount") != config.lane_count
        or collection.get("rollingCpuEnvironmentLanes")
        != min(config.lane_count, selected_match_count)
        or collection.get("batchedGpuMaskedLogitInference")
        is not (torch.device(config.device).type == "cuda")
        or float(collection.get("temperature", math.nan)) != float(config.temperature)
        or float(collection.get("epsilonFloorPerLegalAction", math.nan)) != float(config.epsilon_floor)
        or not isinstance(reward, Mapping)
        or float(reward.get("pairwiseCoefficient", math.nan)) != float(config.pairwise_coefficient)
        or not isinstance(returns, Mapping)
        or returns.get("standardized") is not config.standardize_advantages
        or not isinstance(model, Mapping)
        or model.get("actorCheckpointSha256") != actor_checkpoint_sha256
        or not isinstance(execution, Mapping)
        or execution.get("device") != expected_device
        or (
            expected_plan_version == 2
            and (
                execution.get("fixedCollectionPlanVersion") != 2
                or execution.get("plannedShardBackend") != expected_backend
            )
        )
        or embedded.get("sourceHashes") != dict(source_hashes)
    ):
        raise ValueError("resume-existing manifest does not match the exact requested shard")
    from v4_dataset import load_v4_dataset_npz

    dataset = load_v4_dataset_npz(output)
    trajectories = int(embedded.get("trajectoryCount", -1))
    samples = int(embedded.get("sampleCount", -1))
    complete_matches = int(embedded.get("completeMatchCount", -1))
    if trajectories != len(dataset) or samples != int(dataset.tensors.valid_masks.sum()) or trajectories != complete_matches * ACTS_PER_MATCH:
        raise ValueError("resume-existing manifest counts do not match validated tensors")
    return FixedMatchPPOCollectionResult(
        output,
        metadata_path,
        checksum_path,
        metadata_checksum_path,
        npz_sha,
        metadata_sha,
        dataset.fingerprint,
        trajectories,
        samples,
        complete_matches,
    )


def assert_evaluator_candidate_seat_parity(
    player_counts_and_indexes: Mapping[int, Sequence[int]],
) -> dict[str, object]:
    """Fail closed if the independently implemented rotation differs at all."""

    from v4_evaluate import rotating_candidate_seats

    checked: list[tuple[int, int, tuple[int, ...]]] = []
    for player_count in sorted(player_counts_and_indexes):
        for match_index in player_counts_and_indexes[player_count]:
            local = evaluation_candidate_initial_seats(player_count, int(match_index))
            evaluator = tuple(rotating_candidate_seats(player_count, int(match_index)))
            if local != evaluator:
                raise RuntimeError(
                    f"candidate identity schedule drifted from evaluator at p{player_count} match {match_index}"
                )
            checked.append((player_count, int(match_index), local))
    return {
        "checkedScheduleEntries": len(checked),
        "allEntriesMatched": True,
        "scheduleBindingSha256": _sha256_bytes(canonical_json_bytes(checked)),
    }


def evaluator_group_reward_components(
    finish_order: Sequence[int],
    chip_awards: Mapping[int, int] | Mapping[str, int],
    candidate_ids: Sequence[int],
) -> tuple[float, float, float, int, int, float, float]:
    """Return the evaluator-aligned group chip and pairwise components."""

    order = tuple(int(value) for value in finish_order)
    if len(order) < 4 or len(set(order)) != len(order):
        raise ValueError("finish_order is invalid")
    candidate = {int(value) for value in candidate_ids}
    if not candidate or not candidate < set(order):
        raise ValueError("candidate identities must be a non-empty proper finish-order subset")
    normal = set(order) - candidate
    try:
        award_keyset = {int(value) for value in chip_awards}
    except (TypeError, ValueError) as error:
        raise ValueError("chip award keys must be exact physical IDs") from error
    if award_keyset != set(order):
        raise ValueError("chip award keyset must exactly equal finish-order physical IDs")
    awards: dict[int, float] = {}
    for finish_index, actor in enumerate(order, start=1):
        value = chip_awards.get(actor)
        if value is None:
            value = chip_awards.get(str(actor))  # type: ignore[arg-type]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("chip awards must bind every physical identity to a finite number")
        expected = round_chip_award(finish_index, len(order))
        if float(value) != expected:
            raise ValueError("chip award disagrees with exact finish-place score")
        awards[actor] = float(value)
    if sum(awards.values()) != 2 * len(order):
        raise ValueError("chip award total drifted from exact round scoring")
    candidate_mean = sum(awards[actor] for actor in candidate) / len(candidate)
    normal_mean = sum(awards[actor] for actor in normal) / len(normal)
    chip_difference = candidate_mean - normal_mean
    positions = {actor: index for index, actor in enumerate(order)}
    comparisons = len(candidate) * len(normal)
    before = sum(
        int(positions[candidate_id] < positions[normal_id])
        for candidate_id in candidate
        for normal_id in normal
    )
    if not 0 <= before <= comparisons or comparisons != len(candidate) * len(normal):
        raise RuntimeError("candidate-Normal pairwise comparison invariant failed")
    rate = before / comparisons
    return candidate_mean, normal_mean, chip_difference, before, comparisons, rate, rate - 0.5


def suffix_reward_components(
    chip_rewards: Sequence[float],
    pairwise_centered_rewards: Sequence[float],
    pairwise_coefficient: float = DEFAULT_PAIRWISE_COEFFICIENT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chip = np.asarray(chip_rewards, dtype=np.float64)
    pair = np.asarray(pairwise_centered_rewards, dtype=np.float64)
    coefficient = float(pairwise_coefficient)
    if chip.shape != (ACTS_PER_MATCH,) or pair.shape != (ACTS_PER_MATCH,):
        raise ValueError("suffix rewards require exactly five chip and pairwise components")
    if not np.all(np.isfinite(chip)) or not np.all(np.isfinite(pair)):
        raise ValueError("suffix reward components must be finite")
    if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("pairwise coefficient must be finite and in [0,1]")
    # Evaluation reports mean chip difference per act over a five-act match.
    # Dividing every act contribution by five aligns G_1 exactly with that
    # match-level metric while later G_t remains an undiscounted suffix.
    total = (chip + coefficient * pair) / ACTS_PER_MATCH
    suffix_chip = np.cumsum(chip[::-1])[::-1].copy()
    suffix_pair = np.cumsum(pair[::-1])[::-1].copy()
    suffix_total = (suffix_chip + coefficient * suffix_pair) / ACTS_PER_MATCH
    return total, suffix_chip, suffix_pair, suffix_total


def greedy_masked_candidate_action(
    logits: torch.Tensor | np.ndarray,
    legal_mask: torch.Tensor | np.ndarray,
) -> int:
    """Exact frozen teammate mode: deterministic lowest-index masked argmax."""

    values = np.asarray(
        logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits,
        dtype=np.float64,
    )
    legal = np.asarray(
        legal_mask.detach().cpu().numpy() if isinstance(legal_mask, torch.Tensor) else legal_mask,
        dtype=np.bool_,
    )
    if values.shape != (ACTION_COUNT,) or legal.shape != (ACTION_COUNT,) or not legal.any():
        raise ValueError("greedy teammate requires one 236-action logit/mask row")
    if not np.all(np.isfinite(values[legal])):
        raise ValueError("greedy teammate legal logits must be finite")
    legal_indexes = np.flatnonzero(legal)
    return int(legal_indexes[int(np.argmax(values[legal]))])


def _finalize_complete_match_rewards(
    trajectories: Sequence[dict[str, object]], pairwise_coefficient: float
) -> None:
    if len(trajectories) != ACTS_PER_MATCH:
        raise RuntimeError("a fixed-match learner must have exactly five act trajectories")
    ordered = sorted(trajectories, key=lambda item: int(item["act"]))
    if [int(item["act"]) for item in ordered] != list(range(1, ACTS_PER_MATCH + 1)):
        raise RuntimeError("complete match trajectories must contain acts one through five")
    actor_ids = {int(item["actor_id"]) for item in ordered}
    match_ids = {str(item["complete_match_id"]) for item in ordered}
    if len(actor_ids) != 1 or len(match_ids) != 1:
        raise RuntimeError("learner physical identity and complete match ID must remain fixed")
    chip = [float(item["act_group_chip_difference"]) for item in ordered]
    pair = [float(item["act_pairwise_centered_reward"]) for item in ordered]
    totals, suffix_chip, suffix_pair, suffix_total = suffix_reward_components(
        chip, pair, pairwise_coefficient
    )
    for index, trajectory in enumerate(ordered):
        rows = trajectory["rows"]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("every act segment requires at least one learner decision")
        if any(bool(row["dones"]) for row in rows):
            raise RuntimeError("act segment was terminalized before complete-match reward binding")
        rows[-1]["dones"] = True
        rows[-1]["rewards"] = float(suffix_total[index])
        trajectory["act_total_reward"] = float(totals[index])
        trajectory["suffix_group_chip_sum"] = float(suffix_chip[index])
        trajectory["suffix_pairwise_centered_return"] = float(suffix_pair[index])
        trajectory["suffix_total_return"] = float(suffix_total[index])
        trajectory["terminal_reward"] = float(suffix_total[index])


def _build_arrays(
    trajectories: Sequence[Mapping[str, object]],
    actor: V4ActorConfig,
    critic: V4CriticConfig,
    *,
    standardize: bool,
) -> tuple[dict[str, np.ndarray], V4TrajectoryDataset, tuple[BaselineResult, ...]]:
    records = tuple(
        BaselineRecord(
            int(item["player_count"]),
            str(item["role"]),
            int(item["act"]),
            str(item["complete_match_id"]),
            float(item["terminal_reward"]),
        )
        for item in trajectories
    )
    baselines = leave_one_match_out_baselines(records)
    maximum = max(len(item["rows"]) for item in trajectories)
    arrays = _allocate_arrays(len(trajectories), maximum, actor, critic)
    prefix = (len(trajectories), maximum)
    arrays.update({
        "raw_act_candidate_mean_chips": np.zeros(prefix, np.float32),
        "raw_act_normal_mean_chips": np.zeros(prefix, np.float32),
        "raw_act_group_chip_differences": np.zeros(prefix, np.float32),
        "raw_act_pairwise_rates": np.zeros(prefix, np.float32),
        "raw_act_pairwise_centered_rewards": np.zeros(prefix, np.float32),
        "raw_act_total_rewards": np.zeros(prefix, np.float32),
        "suffix_group_chip_sums": np.zeros(prefix, np.float32),
        "suffix_pairwise_centered_returns": np.zeros(prefix, np.float32),
        "suffix_total_returns": np.zeros(prefix, np.float32),
        "pairwise_candidate_before_normal_counts": np.zeros(prefix, np.int16),
        "pairwise_candidate_normal_comparison_counts": np.zeros(prefix, np.int16),
    })
    public_names = (
        "global_features", "rank_features", "player_features", "player_mask",
        "memory_trace_features", "history_features", "history_mask", "legal_masks",
    )
    row_names = (
        "actions", "expert_actions", "old_action_log_probs", "rewards", "dones",
        "privileged_states", "selected_action_probabilities", "policy_entropies",
        "terminal_chip_awards", "forced_masks", "source_decision_indices",
    )
    for trajectory_index, (trajectory, baseline) in enumerate(zip(trajectories, baselines, strict=True)):
        rows = trajectory["rows"]
        if not rows or not rows[-1]["dones"] or sum(int(row["dones"]) for row in rows) != 1:
            raise RuntimeError("each act segment requires exactly one final terminal marker")
        suffix_total = float(trajectory["suffix_total_return"])
        for time_index, row in enumerate(rows):
            for name in public_names:
                arrays[name][trajectory_index, time_index] = row[name]
            for name in row_names:
                arrays[name][trajectory_index, time_index] = row[name]
            raw_advantage = suffix_total - baseline.baseline
            arrays["raw_returns"][trajectory_index, time_index] = suffix_total
            arrays["baseline_values"][trajectory_index, time_index] = baseline.baseline
            arrays["raw_advantages"][trajectory_index, time_index] = raw_advantage
            arrays["advantage_scales"][trajectory_index, time_index] = baseline.scale
            arrays["baseline_tiers"][trajectory_index, time_index] = baseline.tier
            arrays["baseline_reference_counts"][trajectory_index, time_index] = baseline.reference_count
            arrays["advantages"][trajectory_index, time_index] = (
                raw_advantage / baseline.scale if standardize else raw_advantage
            )
            arrays["suffix_group_chip_sums"][trajectory_index, time_index] = trajectory["suffix_group_chip_sum"]
            arrays["suffix_pairwise_centered_returns"][trajectory_index, time_index] = trajectory["suffix_pairwise_centered_return"]
            arrays["suffix_total_returns"][trajectory_index, time_index] = suffix_total
            arrays["valid_masks"][trajectory_index, time_index] = True
        terminal = len(rows) - 1
        arrays["raw_act_candidate_mean_chips"][trajectory_index, terminal] = trajectory["act_candidate_mean_chip"]
        arrays["raw_act_normal_mean_chips"][trajectory_index, terminal] = trajectory["act_normal_mean_chip"]
        arrays["raw_act_group_chip_differences"][trajectory_index, terminal] = trajectory["act_group_chip_difference"]
        arrays["raw_act_pairwise_rates"][trajectory_index, terminal] = trajectory["act_pairwise_rate"]
        arrays["raw_act_pairwise_centered_rewards"][trajectory_index, terminal] = trajectory["act_pairwise_centered_reward"]
        arrays["raw_act_total_rewards"][trajectory_index, terminal] = trajectory["act_total_reward"]
        arrays["pairwise_candidate_before_normal_counts"][trajectory_index, terminal] = trajectory["pairwise_candidate_before_normal"]
        arrays["pairwise_candidate_normal_comparison_counts"][trajectory_index, terminal] = trajectory["pairwise_candidate_normal_comparisons"]

    standard_names = {field.name for field in fields(V4TrajectoryTensors)}
    bool_names = {"player_mask", "history_mask", "legal_masks", "dones", "valid_masks"}
    int_names = {"actions", "expert_actions"}
    tensors: dict[str, torch.Tensor] = {}
    for name in standard_names:
        tensor = torch.from_numpy(arrays[name])
        if name in bool_names:
            tensor = tensor.bool()
        elif name in int_names:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        tensors[name] = tensor
    dataset = V4TrajectoryDataset(V4TrajectoryTensors(**tensors), actor, critic)
    return arrays, dataset, baselines


def _assignment_report(specs: Sequence[_MatchSpec]) -> dict[str, object]:
    learner: dict[str, dict[str, int]] = {}
    physical: dict[str, dict[str, int]] = {}
    candidate: dict[str, dict[str, int]] = {}
    for spec in specs:
        learner_counts = learner.setdefault(str(spec.player_count), {str(seat): 0 for seat in range(spec.player_count)})
        physical_counts = physical.setdefault(str(spec.player_count), {str(actor): 0 for actor in range(spec.player_count)})
        candidate_counts = candidate.setdefault(str(spec.player_count), {str(seat): 0 for seat in range(spec.player_count)})
        learner_counts[str(spec.learner_initial_seat)] += 1
        physical_counts[str(spec.learner_physical_id)] += 1
        for seat in evaluation_candidate_initial_seats(spec.player_count, spec.match_index):
            candidate_counts[str(seat)] += 1
    imbalance = {key: max(value.values()) - min(value.values()) for key, value in learner.items()}
    physical_imbalance = {key: max(value.values()) - min(value.values()) for key, value in physical.items()}
    return {
        "learnerAssignmentsByInitialSeat": learner,
        "learnerAssignmentsByPhysicalIdentity": physical,
        "evaluatorCandidateAssignmentsByInitialSeat": candidate,
        "learnerInitialSeatMaxMinusMin": imbalance,
        "learnerPhysicalIdentityMaxMinusMin": physical_imbalance,
        "learnerPhysicalIdentityBalancedWithinOne": all(value <= 1 for value in physical_imbalance.values()),
        "candidateSetRule": "exact v4_evaluate.rotating_candidate_seats over initial seating",
        "learnerRule": "deterministic physical-ID-balanced bipartite assignment inside candidate set",
    }


def _build_complete_match_specs(
    config: FixedMatchPPOCollectionConfig,
) -> tuple[list[_MatchSpec], dict[int, tuple[int, ...]]]:
    complete_specs: list[_MatchSpec] = []
    schedule_indexes: dict[int, tuple[int, ...]] = {}
    for player_count, match_count in config.match_counts:
        indexes = tuple(range(config.match_start, config.match_start + match_count))
        schedule_indexes[player_count] = indexes
        seeds = {
            match_index: _derive_uint32(
                config.run_namespace,
                config.seed_base,
                "fixed-match-environment",
                player_count,
                match_index,
            )
            for match_index in indexes
        }
        initial_orders = {
            match_index: tuple(
                int(value)
                for value in DalmutiScalarEnv(
                    player_count,
                    acts=ACTS_PER_MATCH,
                    seed=seeds[match_index],
                    device="cpu",
                )._order
            )
            for match_index in indexes
        }
        learner_ids = balanced_learner_physical_ids(player_count, initial_orders)
        for match_index in indexes:
            learner_id = learner_ids[match_index]
            complete_specs.append(_MatchSpec(
                player_count,
                match_index,
                seeds[match_index],
                initial_orders[match_index].index(learner_id),
                learner_id,
            ))
    if len({spec.seed for spec in complete_specs}) != len(complete_specs):
        raise RuntimeError("derived complete-match environment seed schedule contains a collision")
    return complete_specs, schedule_indexes


def collect_v4_fixed_match_ppo(
    bundle_directory: str | Path,
    output_path: str | Path,
    config: FixedMatchPPOCollectionConfig,
    *,
    repository_root: str | Path | None = None,
) -> FixedMatchPPOCollectionResult:
    planned_backend = _validate_shard_backend_binding(config)
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("fixed-match PPO output must end in .npz")
    metadata_path = Path(f"{output}.metadata.json")
    checksum_path = Path(f"{output}.sha256")
    metadata_checksum_path = Path(f"{metadata_path}.sha256")
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parent.parent
    bundle = Path(bundle_directory).resolve()
    manifest = verify_v4_actor_bundle(bundle)
    model, payload = load_v4_actor_checkpoint(bundle / "actor.pt")
    actor_config = getattr(model, "config", None)
    if not isinstance(actor_config, V4ActorConfig):
        raise ValueError("verified bundle did not load a V4 actor configuration")
    player_counts = tuple(player_count for player_count, _ in config.match_counts)
    _validate_actor_contract(actor_config, player_counts)
    if float(actor_config.dropout) != 0.0:
        raise ValueError(
            "fixed-match PPO requires actor dropout=0.0 for exact "
            "train/rollout policy parity"
        )
    if payload.get("criticExcluded") is not True:
        raise ValueError("candidate actor checkpoint did not exclude the critic")
    source_hashes = _source_hashes(root)
    actor_checkpoint_sha256 = sha256_file(bundle / "actor.pt")
    bundle_manifest_sha256 = sha256_file(bundle / "manifest.json")
    cross_backend_calibration_report_sha256: str | None = None
    calibration_verification: FixedMatchBackendCalibrationVerification | None = None
    if config.cross_backend_calibration_report is not None:
        assert config.cross_backend_calibration_cpu_npz is not None
        assert config.cross_backend_calibration_cuda_npz is not None
        calibration_verification = load_verified_fixed_match_backend_calibration(
            config.cross_backend_calibration_report,
            config.cross_backend_calibration_cpu_npz,
            config.cross_backend_calibration_cuda_npz,
            expected_actor_checkpoint_sha256=actor_checkpoint_sha256,
            expected_bundle_manifest_sha256=bundle_manifest_sha256,
            expected_source_hashes=source_hashes,
        )
        cross_backend_calibration_report_sha256 = (
            calibration_verification.report_sha256
        )
    resumed = _resume_existing_result(
        output,
        metadata_path,
        checksum_path,
        metadata_checksum_path,
        config,
        actor_checkpoint_sha256,
        cross_backend_calibration_report_sha256,
        source_hashes,
    )
    if resumed is not None:
        if calibration_verification is not None:
            calibration_verification.recheck_unchanged()
        return resumed
    device = torch.device(config.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    execution = _configure_determinism(device)
    execution["cublasWorkspaceConfig"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device.type == "cuda" else None
    if planned_backend is not None:
        execution["fixedCollectionPlanVersion"] = 2
        execution["plannedShardBackend"] = planned_backend
    model = model.to(device).eval()

    complete_specs, schedule_indexes = _build_complete_match_specs(config)
    specs = [
        spec for spec in complete_specs
        if spec.match_index % config.match_shard_count == config.match_shard_index
    ]
    if not specs:
        raise ValueError("the requested match shard is empty")
    evaluator_parity_audit = assert_evaluator_candidate_seat_parity(schedule_indexes)

    lanes = [_new_lane(spec) for spec in specs[: config.lane_count]]
    next_spec = len(lanes)
    trajectories: dict[str, dict[str, object]] = {}
    match_trajectory_ids: dict[str, list[str]] = {}
    privacy_audits: dict[str, object] = {}
    privileged_layout_audits: dict[str, object] = {}
    action_counts = {
        "stochasticLearner": {"decisions": 0, "forced": 0, "differentFromNormal": 0},
        "greedyCandidateTeammate": {"decisions": 0, "forced": 0, "differentFromNormal": 0},
        "exactNormalOpponent": {"decisions": 0, "forced": 0},
    }
    entropy_values: list[float] = []
    total_environment_decisions = 0
    while lanes:
        publics = [lane.env.public_observation() for lane in lanes]
        normal_actions = [lane.env.normal_action() for lane in lanes]
        candidate_lane_indexes = [
            index for index, lane in enumerate(lanes)
            if lane.env.current_player_id in lane.candidate_ids
        ]
        logits = _batch_candidate_logits(model, [publics[index] for index in candidate_lane_indexes], device)
        candidate_results: dict[int, tuple[int, float, float, float]] = {}
        for lane_index, row_logits in zip(candidate_lane_indexes, logits, strict=True):
            lane = lanes[lane_index]
            public = publics[lane_index]
            if lane.env.current_player_id == lane.learner_id:
                probabilities = masked_categorical_probabilities(
                    row_logits,
                    public.legal_mask,
                    temperature=config.temperature,
                    epsilon_floor=config.epsilon_floor,
                )
                uniform = _keyed_uniform(
                    config.run_namespace,
                    config.seed_base,
                    "fixed-match-learner-action",
                    lane.spec.player_count,
                    lane.spec.match_index,
                    lane.spec.seed,
                    int(lane.env._act),
                    lane.learner_id,
                    lane.decision_index,
                )
                action, log_probability, entropy = sample_masked_categorical(probabilities, uniform)
                candidate_results[lane_index] = (action, log_probability, entropy, float(probabilities[action]))
            else:
                action = greedy_masked_candidate_action(row_logits, public.legal_mask)
                candidate_results[lane_index] = (action, 0.0, 0.0, 1.0)

        replacements: list[_Lane] = []
        for lane_index, (lane, public, normal_action) in enumerate(zip(lanes, publics, normal_actions, strict=True)):
            env = lane.env
            actor_id = env.current_player_id
            act = int(env._act)
            if actor_id == lane.learner_id:
                kind = "stochasticLearner"
                behavior_action, old_log_probability, entropy, selected_probability = candidate_results[lane_index]
                entropy_values.append(entropy)
            elif actor_id in lane.candidate_ids:
                kind = "greedyCandidateTeammate"
                behavior_action, old_log_probability, entropy, selected_probability = candidate_results[lane_index]
            else:
                kind = "exactNormalOpponent"
                behavior_action, old_log_probability, entropy, selected_probability = normal_action, 0.0, 0.0, 1.0
            legal = public.legal_mask
            if not bool(legal[normal_action]) or not bool(legal[behavior_action]):
                raise RuntimeError("Normal or fixed-match behavior selected an illegal action")
            forced = int(legal.sum()) == 1
            action_counts[kind]["decisions"] += 1
            action_counts[kind]["forced"] += int(forced)
            if kind != "exactNormalOpponent":
                action_counts[kind]["differentFromNormal"] += int(behavior_action != normal_action)
            key = str(lane.spec.player_count)
            if key not in privacy_audits:
                privileged_layout_audits[key] = assert_canonical_privileged_layout(env)
                privacy_audits[key] = audit_hidden_state_privacy(
                    env, _derive_uint32(config.run_namespace, config.seed_base, "fixed-match-privacy", lane.spec.player_count)
                )
            if actor_id == lane.learner_id:
                role = role_for_index(env._order.index(actor_id), lane.spec.player_count)
                identifier = _trajectory_id(lane.spec, config.run_namespace, lane.learner_id, act)
                complete_id = _complete_match_id(lane.spec, config.run_namespace)
                trajectory = trajectories.setdefault(identifier, {
                    "id": identifier,
                    "complete_match_id": complete_id,
                    "player_count": lane.spec.player_count,
                    "match_index": lane.spec.match_index,
                    "match_seed": lane.spec.seed,
                    "act": act,
                    "actor_id": lane.learner_id,
                    "role": role,
                    "learner_initial_seat": lane.spec.learner_initial_seat,
                    "initial_order": ",".join(str(value) for value in lane.initial_order),
                    "candidate_initial_seats": ",".join(str(value) for value in lane.candidate_initial_seats),
                    "candidate_ids": ",".join(str(value) for value in sorted(lane.candidate_ids)),
                    "act_player_order": ",".join(str(value) for value in env._order),
                    "act_finish_order": None,
                    "act_chip_awards_by_physical_id": None,
                    "rows": [],
                    "finish_place": None,
                    "learner_act_chip_award": None,
                    "act_candidate_mean_chip": None,
                    "act_normal_mean_chip": None,
                    "act_group_chip_difference": None,
                    "act_pairwise_rate": None,
                    "act_pairwise_centered_reward": None,
                    "pairwise_candidate_before_normal": None,
                    "pairwise_candidate_normal_comparisons": None,
                    "terminal_reward": None,
                })
                if trajectory["role"] != role:
                    raise RuntimeError("learner role changed within one act segment")
                row = {
                    **_snapshot_public(public, actor_config),
                    "actions": behavior_action,
                    "expert_actions": normal_action,
                    "old_action_log_probs": old_log_probability,
                    "rewards": 0.0,
                    "dones": False,
                    "privileged_states": env.privileged_state().detach().cpu().numpy().astype(np.float32, copy=True),
                    "selected_action_probabilities": selected_probability,
                    "policy_entropies": entropy,
                    "terminal_chip_awards": 0,
                    "forced_masks": forced,
                    "source_decision_indices": lane.decision_index,
                }
                trajectory["rows"].append(row)
                complete_ids = match_trajectory_ids.setdefault(complete_id, [])
                if identifier not in complete_ids:
                    complete_ids.append(identifier)
            lane.decision_index += 1
            total_environment_decisions += 1
            result = env.step(int(behavior_action))
            if result.act_ended:
                act_result = result.info.get("act_result")
                if not isinstance(act_result, Mapping):
                    raise RuntimeError("act terminal is missing its exact result")
                finish_order = tuple(int(value) for value in act_result["finish_order"])
                chip_awards = act_result.get("chip_awards")
                if not isinstance(chip_awards, Mapping):
                    raise RuntimeError("act terminal chip awards are missing")
                terminal_act_order = tuple(int(value) for value in act_result.get("player_order", ()))
                identifier = _trajectory_id(lane.spec, config.run_namespace, lane.learner_id, act)
                trajectory = trajectories.get(identifier)
                if trajectory is None or not trajectory["rows"]:
                    raise RuntimeError("learner act trajectory is missing at terminal")
                stored_act_order = tuple(int(value) for value in str(trajectory["act_player_order"]).split(","))
                if terminal_act_order != stored_act_order:
                    raise RuntimeError("actual act player order drifted before terminal binding")
                award = int(chip_awards[lane.learner_id])
                chip_reward = float(result.rewards[lane.learner_id])
                if chip_reward != (award - 2) / 2.0:
                    raise RuntimeError("environment centered chip reward drifted")
                (
                    candidate_mean,
                    normal_mean,
                    chip_difference,
                    before,
                    comparisons,
                    rate,
                    centered,
                ) = evaluator_group_reward_components(
                    finish_order, chip_awards, lane.candidate_ids
                )
                trajectory["finish_place"] = finish_order.index(lane.learner_id) + 1
                trajectory["act_finish_order"] = ",".join(str(value) for value in finish_order)
                trajectory["act_chip_awards_by_physical_id"] = ",".join(
                    str(int(chip_awards[actor_id]))
                    for actor_id in range(lane.spec.player_count)
                )
                trajectory["learner_act_chip_award"] = award
                trajectory["act_candidate_mean_chip"] = candidate_mean
                trajectory["act_normal_mean_chip"] = normal_mean
                trajectory["act_group_chip_difference"] = chip_difference
                trajectory["act_pairwise_rate"] = rate
                trajectory["act_pairwise_centered_reward"] = centered
                trajectory["pairwise_candidate_before_normal"] = before
                trajectory["pairwise_candidate_normal_comparisons"] = comparisons
                trajectory["rows"][-1]["terminal_chip_awards"] = award
                if result.terminated:
                    complete_id = _complete_match_id(lane.spec, config.run_namespace)
                    match_items = [trajectories[value] for value in match_trajectory_ids.get(complete_id, [])]
                    _finalize_complete_match_rewards(match_items, config.pairwise_coefficient)
            if result.terminated:
                if next_spec < len(specs):
                    replacements.append(_new_lane(specs[next_spec]))
                    next_spec += 1
            else:
                replacements.append(lane)
        lanes = replacements

    ordered = [trajectories[key] for key in sorted(trajectories)]
    if len(ordered) != len(specs) * ACTS_PER_MATCH or any(item["terminal_reward"] is None for item in ordered):
        raise RuntimeError("collection must contain exactly five finalized trajectories per complete match")
    critic_config = V4CriticConfig(privileged_features=PRIVILEGED_STATE_SIZE)
    arrays, dataset, baselines = _build_arrays(
        ordered, actor_config, critic_config, standardize=config.standardize_advantages
    )
    arrays.update({
        "trajectory_ids": np.asarray([item["id"] for item in ordered], dtype=np.str_),
        "trajectory_complete_match_ids": np.asarray([item["complete_match_id"] for item in ordered], dtype=np.str_),
        "trajectory_player_counts": np.asarray([item["player_count"] for item in ordered], np.int16),
        "trajectory_roles": np.asarray([ROLES.index(str(item["role"])) for item in ordered], np.int8),
        "trajectory_acts": np.asarray([item["act"] for item in ordered], np.int16),
        "trajectory_actor_ids": np.asarray([item["actor_id"] for item in ordered], np.int16),
        "trajectory_match_indices": np.asarray([item["match_index"] for item in ordered], np.int32),
        "trajectory_match_seeds": np.asarray([item["match_seed"] for item in ordered], np.uint32),
        "trajectory_match_clusters": np.asarray([item["complete_match_id"] for item in ordered], dtype=np.str_),
        "trajectory_finish_places": np.asarray([item["finish_place"] for item in ordered], np.int16),
        "trajectory_learner_initial_seats": np.asarray([item["learner_initial_seat"] for item in ordered], np.int16),
        "trajectory_initial_player_orders": np.asarray([item["initial_order"] for item in ordered], dtype=np.str_),
        "trajectory_candidate_initial_seats": np.asarray([item["candidate_initial_seats"] for item in ordered], dtype=np.str_),
        "trajectory_candidate_ids": np.asarray([item["candidate_ids"] for item in ordered], dtype=np.str_),
        "trajectory_act_player_orders": np.asarray([item["act_player_order"] for item in ordered], dtype=np.str_),
        "trajectory_act_finish_orders": np.asarray([item["act_finish_order"] for item in ordered], dtype=np.str_),
        "trajectory_act_chip_awards_by_physical_id": np.asarray([item["act_chip_awards_by_physical_id"] for item in ordered], dtype=np.str_),
        "trajectory_act_candidate_mean_chips": np.asarray([item["act_candidate_mean_chip"] for item in ordered], np.float32),
        "trajectory_act_normal_mean_chips": np.asarray([item["act_normal_mean_chip"] for item in ordered], np.float32),
        "trajectory_act_group_chip_differences": np.asarray([item["act_group_chip_difference"] for item in ordered], np.float32),
        "trajectory_act_pairwise_rates": np.asarray([item["act_pairwise_rate"] for item in ordered], np.float32),
        "trajectory_act_pairwise_centered_rewards": np.asarray([item["act_pairwise_centered_reward"] for item in ordered], np.float32),
        "trajectory_act_total_rewards": np.asarray([item["act_total_reward"] for item in ordered], np.float32),
        "trajectory_suffix_group_chip_sums": np.asarray([item["suffix_group_chip_sum"] for item in ordered], np.float32),
        "trajectory_suffix_pairwise_centered_returns": np.asarray([item["suffix_pairwise_centered_return"] for item in ordered], np.float32),
        "trajectory_suffix_total_returns": np.asarray([item["suffix_total_return"] for item in ordered], np.float32),
    })
    valid = arrays["valid_masks"]
    tier_counts = {name: 0 for name in BASELINE_FALLBACK_HIERARCHY}
    tier_counts_by_player_count: dict[str, dict[str, int]] = {}
    reference_counts_by_player_count: dict[str, list[float]] = {}
    for baseline in baselines:
        tier_counts[BASELINE_FALLBACK_HIERARCHY[baseline.tier]] += 1
    for item, baseline in zip(ordered, baselines, strict=True):
        player_key = str(item["player_count"])
        by_tier = tier_counts_by_player_count.setdefault(
            player_key, {name: 0 for name in BASELINE_FALLBACK_HIERARCHY}
        )
        by_tier[BASELINE_FALLBACK_HIERARCHY[baseline.tier]] += 1
        reference_counts_by_player_count.setdefault(player_key, []).append(
            float(baseline.reference_count)
        )
    player_count_distribution: dict[str, dict[str, int]] = {}
    for player_count, _ in config.match_counts:
        indexes = [
            index for index, item in enumerate(ordered)
            if int(item["player_count"]) == player_count
        ]
        valid_player = arrays["valid_masks"][indexes]
        forced_player = arrays["forced_masks"][indexes] & valid_player
        valid_count = int(valid_player.sum())
        forced_count = int(forced_player.sum())
        player_count_distribution[str(player_count)] = {
            "completeMatches": len(indexes) // ACTS_PER_MATCH,
            "learnerActTrajectories": len(indexes),
            "learnerDecisionSamples": valid_count,
            "ppoEligibleSamples": valid_count,
            "criticEligibleSamples": valid_count,
            "forcedSamples": forced_count,
            "nonforcedPolicySamples": valid_count - forced_count,
            "behaviorCloningEligibleForcedSamples": forced_count,
            "behaviorCloningEligibleNonforcedSamples": valid_count - forced_count,
            "ppoEligibleForcedSamples": forced_count,
            "ppoEligibleNonforcedSamples": valid_count - forced_count,
        }
    selected_report = _assignment_report(specs)
    selected_report["completeMatchRangeAcrossAllShards"] = _assignment_report(complete_specs)
    action_report: dict[str, object] = {}
    for name, counts in action_counts.items():
        row: dict[str, object] = dict(counts)
        if name != "exactNormalOpponent":
            row["differentFromNormalRate"] = counts["differentFromNormal"] / max(1, counts["decisions"])
        action_report[name] = row
    shard_backend_map = _shard_backend_map_record(config)
    shard_record: dict[str, object] = {
        "runNamespace": config.run_namespace,
        "seedBase": config.seed_base,
        "matchCounts": {str(player_count): count for player_count, count in config.match_counts},
        "matchStart": config.match_start,
        "matchShardCount": config.match_shard_count,
        "matchShardIndex": config.match_shard_index,
        "environmentSeeds": [spec.seed for spec in specs],
        "completeUnshardedLearnerAssignmentSha256": _sha256_bytes(
            canonical_json_bytes([
                [
                    spec.player_count,
                    spec.match_index,
                    spec.seed,
                    spec.learner_initial_seat,
                    spec.learner_physical_id,
                ]
                for spec in complete_specs
            ])
        ),
        "rolloutKeysIndependentOfLaneScheduling": True,
        "trajectoryIdsIndependentOfShardPartition": True,
        "completeMatchTrajectoryIdsIncludeNamespacePlayerMatchSeedLearnerAndAct": True,
    }
    if shard_backend_map is not None:
        shard_record["collectionPlanVersion"] = 2
        shard_record["shardBackendMap"] = shard_backend_map
        assert cross_backend_calibration_report_sha256 is not None
        shard_record["crossBackendCalibrationReportSha256"] = (
            cross_backend_calibration_report_sha256
        )
    shard_record["identitySha256"] = fixed_match_shard_identity_sha256(
        shard_record,
        FIXED_MATCH_PPO_PREPARATION_FORMAT,
    )

    metadata: dict[str, object] = {
        "format": V4_DATASET_FORMAT,
        "version": V4_DATASET_VERSION,
        "preparationFormat": FIXED_MATCH_PPO_PREPARATION_FORMAT,
        "preparationVersion": FIXED_MATCH_PPO_PREPARATION_VERSION,
        "fingerprint": dataset.fingerprint,
        "actorConfig": actor_config.to_dict(),
        "criticConfig": critic_config.to_dict(),
        "collection": {
            "algorithm": "evaluation-aligned fixed-physical-ID five-act suffix PPO rollout",
            "actsPerCompleteMatch": ACTS_PER_MATCH,
            "completeMatchesOnly": True,
            "evaluationCandidateIdentityParity": True,
            "candidateIdentitySetFixedForCompleteMatch": True,
            "learnerPhysicalIdentityFixedForCompleteMatch": True,
            "stochasticLearnersPerMatch": 1,
            "learnerBehavior": "frozen candidate stochastic masked categorical",
            "candidateTeammateBehavior": "frozen candidate greedy masked argmax",
            "normalOpponentBehavior": "exact DalmutiScalarEnv.normal_action",
            "behaviorPolicyContract": FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT,
            "behaviorPolicyContractVersion": FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT_VERSION,
            "rawMaskedSoftmaxExactBinding": True,
            "initialOldCurrentRatioMathematicallyOneForFrozenActor": True,
            "initialOldCurrentLogProbabilityAbsoluteTolerance": (
                FIXED_MATCH_INITIAL_LOG_PROBABILITY_TOLERANCE
            ),
            "fixedPpoActorAutocastDisabled": True,
            "requiresFullDatasetInitialPolicyReproductionAudit": True,
            "dropoutDisabled": True,
            "temperature": config.temperature,
            "epsilonFloorPerLegalAction": config.epsilon_floor,
            "exactOldLogProbabilityForEveryLearnerDecision": True,
            "exactNormalExpertLabelForEveryLearnerDecision": True,
            "requestedLaneCount": config.lane_count,
            "rollingCpuEnvironmentLanes": min(config.lane_count, len(specs)),
            "batchedGpuMaskedLogitInference": device.type == "cuda",
        },
        "rewardContract": {
            "version": 1,
            "chipComponent": "mean exact chip award(fixed candidate IDs) - mean exact chip award(fixed Normal IDs)",
            "pairwiseRate": "candidate-before-Normal finish pairs / (candidate identity count * Normal identity count)",
            "pairwiseCenteredComponent": "pairwiseRate - 0.5",
            "pairwiseCoefficient": config.pairwise_coefficient,
            "actTotal": "(chipComponent + pairwiseCoefficient * pairwiseCenteredComponent) / 5",
            "trajectoryReturn": "sum of actTotal from trajectory act through act five; never divide by remaining horizon",
            "rawComponentsSeparatelyBoundForAblation": True,
        },
        "returnsAndAdvantages": {
            "monteCarloGamma": 1.0,
            "baseline": "deterministic leave-one-entire-complete-match-cluster-out mean",
            "fallbackHierarchy": list(BASELINE_FALLBACK_HIERARCHY),
            "fallbackCounts": tier_counts,
            "fallbackCountsByPlayerCount": tier_counts_by_player_count,
            "referenceCountStatsByPlayerCount": {
                key: _finite_stats(values)
                for key, values in reference_counts_by_player_count.items()
            },
            "standardized": config.standardize_advantages,
            "standardizationScale": "population std from same cluster-excluded references; 1 if degenerate",
            "ownCompleteMatchClusterExcludedAtEveryTier": True,
            "futureRewardsUsedOnlyAsTrainingTargets": True,
            "futureRewardsExcludedFromActorAndCriticInputs": True,
            "opponentHiddenHandsUsed": False,
            "rawReturnStats": _finite_stats(arrays["raw_returns"][valid].tolist()),
            "trainingAdvantageStats": _finite_stats(arrays["advantages"][valid].tolist()),
        },
        "trainingRequirements": {
            "ppoSourceContract": "fixed-physical-id-five-act-suffix-v1",
            "qBoostCoefficient": 0.0,
            "qBoostMustRemainOff": True,
            "requiresPlayerCountBalancedLoss": True,
        },
        "shard": shard_record,
        "modelBinding": {
            "bundleManifestSha256": bundle_manifest_sha256,
            "actorCheckpointSha256": actor_checkpoint_sha256,
            "manifestFormat": manifest.get("format"),
            "manifestVersion": manifest.get("version"),
            "modelKind": manifest.get("model", {}).get("kind"),
            "criticExcluded": True,
            "sameFrozenActorForLearnerAndCandidateTeammates": True,
        },
        "environmentBinding": {
            "implementation": "DalmutiScalarEnv",
            "normalExpertCallback": "DalmutiScalarEnv.normal_action",
            "v4EnvSha256": source_hashes["gpu-training/v4_env.py"],
            "normalSourceSha256": source_hashes["lib/bot-strategy.ts"],
            "evaluatorSourceSha256": source_hashes["gpu-training/v4_evaluate.py"],
            "candidateIdentityFunction": "v4_evaluate.rotating_candidate_seats + initial-order physical-ID mapping",
            "candidateSeatParityAudit": evaluator_parity_audit,
            "cpuStepping": True,
        },
        "privilegedCriticBinding": {
            "layoutId": CANONICAL_PRIVILEGED_LAYOUT_ID,
            "layoutSha256": CANONICAL_PRIVILEGED_LAYOUT_SHA256,
            "layout": CANONICAL_PRIVILEGED_LAYOUT,
            "featureCount": PRIVILEGED_STATE_SIZE,
            "environmentSourceSha256": source_hashes["gpu-training/v4_env.py"],
            "perPlayerCountLiveLayoutAudits": privileged_layout_audits,
            "actorExportAllowed": False,
        },
        "sourceHashes": source_hashes,
        "execution": execution,
        "privacy": {
            "actorPublicOnly": True,
            "actorInputFields": list(PUBLIC_MODEL_INPUT_FIELDS),
            "opponentPhysicalHandsExcluded": True,
            "taxCardIdentitiesExcluded": True,
            "privilegedCriticStateSeparate": True,
            "privilegedCriticExportAllowed": False,
            "futureActRewardsExcludedFromActorInputs": True,
            "perPlayerCountAudits": privacy_audits,
        },
        "actionRates": action_report,
        "policyEntropy": _finite_stats(entropy_values),
        "opponentAndSeatBalance": selected_report,
        "playerCountDistribution": player_count_distribution,
        "completeMatchCount": len(specs),
        "trajectoryCount": len(ordered),
        "sampleCount": int(valid.sum()),
        "environmentDecisionCount": total_environment_decisions,
        "maxTimeSteps": int(arrays["actions"].shape[1]),
        "auxiliaryArrays": sorted(set(arrays) - {field.name for field in fields(V4TrajectoryTensors)}),
        "padding": "zero-valued invalid suffix; documented integer sentinels remain canonical",
    }
    arrays["metadata_json"] = np.asarray(_canonical_text(metadata))
    npz_bytes = _deterministic_npz_bytes(arrays)
    npz_sha = _sha256_bytes(npz_bytes)
    external = dict(metadata)
    external["npzSha256"] = npz_sha
    metadata_bytes = (_canonical_text(external) + "\n").encode("utf-8")
    metadata_sha = _sha256_bytes(metadata_bytes)
    if calibration_verification is not None:
        calibration_verification.recheck_unchanged()
    _exclusive_publish({
        output: npz_bytes,
        metadata_path: metadata_bytes,
        checksum_path: f"{npz_sha}  {output.name}\n".encode("ascii"),
        metadata_checksum_path: f"{metadata_sha}  {metadata_path.name}\n".encode("ascii"),
    })
    try:
        if calibration_verification is not None:
            calibration_verification.recheck_unchanged()
    except BaseException:
        for published in (
            output,
            metadata_path,
            checksum_path,
            metadata_checksum_path,
        ):
            published.unlink(missing_ok=True)
        raise
    return FixedMatchPPOCollectionResult(
        output,
        metadata_path,
        checksum_path,
        metadata_checksum_path,
        npz_sha,
        metadata_sha,
        dataset.fingerprint,
        len(ordered),
        int(valid.sum()),
        len(specs),
    )


def _parse_match_counts(value: str) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    try:
        for item in value.split(","):
            player_count, matches = item.strip().split(":", 1)
            result.append((int(player_count), int(matches)))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("match counts must look like 4:320,5:256,...") from error
    normalized = tuple(result)
    try:
        FixedMatchPPOCollectionConfig("parse-check", 1, match_counts=normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return normalized


def _parse_shard_backends(value: str) -> tuple[str, ...]:
    backends = tuple(item.strip() for item in value.split(","))
    if (
        not backends
        or any(backend not in {"cpu", "cuda"} for backend in backends)
        or set(backends) != {"cpu", "cuda"}
    ):
        raise argparse.ArgumentTypeError(
            "shard backends must be a complete mixed list such as "
            "cpu,cpu,cuda,cuda"
        )
    return backends


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect evaluation-aligned fixed-match V4 PPO suffix trajectories.")
    parser.add_argument("--actor-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-namespace", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--match-counts", type=_parse_match_counts, default=DEFAULT_MATCH_COUNTS)
    parser.add_argument("--match-start", type=int, default=0)
    parser.add_argument("--match-shard-count", type=int, default=1)
    parser.add_argument("--match-shard-index", type=int, default=0)
    parser.add_argument(
        "--temperature", type=float, default=FIXED_MATCH_BEHAVIOR_TEMPERATURE
    )
    parser.add_argument(
        "--epsilon-floor", type=float, default=FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR
    )
    parser.add_argument("--pairwise-coefficient", type=float, default=DEFAULT_PAIRWISE_COEFFICIENT)
    parser.add_argument("--no-standardize-advantages", action="store_true")
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shard-backend-map",
        "--shard-backends",
        dest="shard_backend_map",
        type=_parse_shard_backends,
        help=(
            "precommitted backend for every match shard in index order; "
            "supplying this enables the mixed-host fixed collection plan v2"
        ),
    )
    parser.add_argument(
        "--cross-backend-calibration-report",
        type=Path,
        help=(
            "canonical CPU/CUDA calibration JSON (with adjacent .sha256); "
            "required exactly when --shard-backend-map is supplied"
        ),
    )
    parser.add_argument(
        "--cross-backend-calibration-cpu-npz",
        type=Path,
        help=(
            "authoritative CPU NPZ used to build the calibration report; "
            "required with the report and CUDA NPZ for mixed-host plans"
        ),
    )
    parser.add_argument(
        "--cross-backend-calibration-cuda-npz",
        type=Path,
        help=(
            "authoritative CUDA NPZ used to build the calibration report; "
            "required with the report and CPU NPZ for mixed-host plans"
        ),
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = FixedMatchPPOCollectionConfig(
        run_namespace=arguments.run_namespace,
        seed_base=arguments.seed_base,
        match_counts=arguments.match_counts,
        match_start=arguments.match_start,
        match_shard_count=arguments.match_shard_count,
        match_shard_index=arguments.match_shard_index,
        temperature=arguments.temperature,
        epsilon_floor=arguments.epsilon_floor,
        pairwise_coefficient=arguments.pairwise_coefficient,
        standardize_advantages=not arguments.no_standardize_advantages,
        lane_count=arguments.lanes,
        device=arguments.device,
        shard_backend_map=arguments.shard_backend_map,
        cross_backend_calibration_report=(
            arguments.cross_backend_calibration_report
        ),
        cross_backend_calibration_cpu_npz=(
            arguments.cross_backend_calibration_cpu_npz
        ),
        cross_backend_calibration_cuda_npz=(
            arguments.cross_backend_calibration_cuda_npz
        ),
        resume_existing=arguments.resume_existing,
    )
    result = collect_v4_fixed_match_ppo(
        arguments.actor_bundle,
        arguments.output,
        config,
        repository_root=arguments.repository_root,
    )
    print(_canonical_text({
        "output": str(result.output_path),
        "npzSha256": result.npz_sha256,
        "metadataSha256": result.metadata_sha256,
        "fingerprint": result.fingerprint,
        "completeMatches": result.complete_matches,
        "trajectories": result.trajectories,
        "samples": result.samples,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTS_PER_MATCH",
    "DEFAULT_MATCH_COUNTS",
    "DEFAULT_PAIRWISE_COEFFICIENT",
    "FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR",
    "FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT",
    "FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT_VERSION",
    "FIXED_MATCH_BEHAVIOR_TEMPERATURE",
    "FIXED_MATCH_INITIAL_LOG_PROBABILITY_TOLERANCE",
    "FIXED_MATCH_PPO_PREPARATION_FORMAT",
    "FIXED_MATCH_PPO_PREPARATION_VERSION",
    "FixedMatchPPOCollectionConfig",
    "FixedMatchPPOCollectionResult",
    "balanced_learner_initial_seats",
    "balanced_learner_physical_ids",
    "collect_v4_fixed_match_ppo",
    "evaluation_candidate_initial_seats",
    "main",
    "evaluator_group_reward_components",
    "greedy_masked_candidate_action",
    "suffix_reward_components",
]
