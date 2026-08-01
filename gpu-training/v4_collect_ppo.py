from __future__ import annotations

"""On-policy/league rollout collection for the public-only V4 actor.

The learner occupies deterministically rotating physical player identities.  Its
remaining opponents are a configurable deterministic mix of exact Normal and
the same frozen candidate.  Candidate decisions are sampled from a masked
categorical distribution; Normal is only ever queried through the exact
``DalmutiScalarEnv.normal_action`` callback.

Only complete learner actor-per-act trajectories are serialized.  Actor inputs
and the privileged 512-value critic state are kept in separate arrays.  The
return baseline excludes the target trajectory's entire match cluster at every
fallback tier, so neither another seat from the same deal nor a later act from
that match can leak into its advantage.
"""

import argparse
from dataclasses import dataclass, fields
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
    _history_bucket,
    _pad_mask,
    _pad_rows,
    _snapshot_public,
    _trim_public_for_model,
    _validate_actor_contract,
    audit_hidden_state_privacy,
)
from v4_dataset import (
    V4_DATASET_FORMAT,
    V4_DATASET_VERSION,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
)
from v4_env import (
    ACTION_COUNT,
    PRIVILEGED_STATE_SIZE,
    PRIVILEGED_STATE_LAYOUT,
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
    ROLES,
    DalmutiScalarEnv,
    V4ActorObservation,
    role_for_index,
)
from v4_export import (
    canonical_json_bytes,
    load_v4_actor_checkpoint,
    sha256_file,
    verify_v4_actor_bundle,
)
from v4_model import V4ActorConfig, V4CriticConfig


PPO_PREPARATION_FORMAT = "dalmuti-v4-ppo-league-direct-npz"
PPO_PREPARATION_VERSION = 1
NAMESPACE_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
SOURCE_FILES = (
    "gpu-training/v4_collect_ppo.py",
    "gpu-training/v4_collect_dagger.py",
    "gpu-training/v4_env.py",
    "gpu-training/v4_model.py",
    "gpu-training/v4_export.py",
    "gpu-training/v4_dataset.py",
    "gpu-training/v3_action_conditioned.py",
    "lib/bot-strategy.ts",
)
BASELINE_FALLBACK_HIERARCHY = (
    "same-player-count-role-act",
    "same-player-count-role",
    "same-player-count-act",
    "same-player-count",
    "all-player-counts",
    "zero-no-other-match",
)
CANONICAL_PRIVILEGED_LAYOUT_ID = PRIVILEGED_STATE_LAYOUT_ID
CANONICAL_PRIVILEGED_LAYOUT: Mapping[str, object] = PRIVILEGED_STATE_LAYOUT
CANONICAL_PRIVILEGED_LAYOUT_SHA256 = PRIVILEGED_STATE_LAYOUT_SHA256
EXPECTED_CANONICAL_PRIVILEGED_LAYOUT_SHA256 = (
    "be332c07e1753b6e87082917bbf5528faef8fed3cda794c853f655d3ade0110f"
)


@dataclass(frozen=True)
class PPOCollectionConfig:
    run_namespace: str
    seed_base: int
    player_counts: tuple[int, ...] = tuple(range(4, 11))
    matches_per_player_count: int = 8
    match_start: int = 0
    match_shard_count: int = 1
    match_shard_index: int = 0
    acts: int = 5
    act_filter: tuple[int, ...] | None = None
    role_filter: tuple[str, ...] = ROLES
    candidate_seats_per_act: int = 1
    opponent_candidate_fraction: float = 0.5
    temperature: float = 1.0
    epsilon_floor: float = 1.0e-6
    gamma: float = 1.0
    standardize_advantages: bool = True
    lane_count: int = 32
    device: str = "cuda"

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
        if (
            not isinstance(self.player_counts, tuple)
            or not self.player_counts
            or tuple(sorted(set(self.player_counts))) != self.player_counts
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 4 <= value <= 10
                for value in self.player_counts
            )
        ):
            raise ValueError("player_counts must be a sorted unique tuple from 4 through 10")
        for name in (
            "matches_per_player_count",
            "match_shard_count",
            "acts",
            "candidate_seats_per_act",
            "lane_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.match_start, bool) or not isinstance(self.match_start, int) or self.match_start < 0:
            raise ValueError("match_start must be a non-negative integer")
        if (
            isinstance(self.match_shard_index, bool)
            or not isinstance(self.match_shard_index, int)
            or not 0 <= self.match_shard_index < self.match_shard_count
        ):
            raise ValueError("match_shard_index must be in [0, match_shard_count)")
        if self.candidate_seats_per_act > min(self.player_counts):
            raise ValueError("candidate_seats_per_act cannot exceed a requested table size")
        if (
            self.matches_per_player_count * self.acts * self.candidate_seats_per_act
            < max(self.player_counts)
        ):
            raise ValueError(
                "the complete match range must give every player identity learner experience"
            )
        roles = tuple(self.role_filter)
        if not roles or len(set(roles)) != len(roles) or any(role not in ROLES for role in roles):
            raise ValueError("role_filter must be a unique non-empty subset of exact roles")
        acts = self.act_filter if self.act_filter is not None else tuple(range(1, self.acts + 1))
        if (
            not isinstance(acts, tuple)
            or not acts
            or tuple(sorted(set(acts))) != acts
            or any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= self.acts for value in acts)
        ):
            raise ValueError("act_filter must be a sorted unique tuple inside the match act range")
        for name in ("opponent_candidate_fraction", "gamma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive and finite")
        floor = float(self.epsilon_floor)
        if not math.isfinite(floor) or not 0.0 <= floor < 1.0 / ACTION_COUNT:
            raise ValueError("epsilon_floor must be finite and in [0, 1 / action_count)")
        if not isinstance(self.standardize_advantages, bool):
            raise ValueError("standardize_advantages must be boolean")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string")

    @property
    def collected_acts(self) -> tuple[int, ...]:
        return self.act_filter or tuple(range(1, self.acts + 1))


@dataclass(frozen=True)
class PPOCollectionResult:
    output_path: Path
    metadata_path: Path
    checksum_path: Path
    metadata_checksum_path: Path
    npz_sha256: str
    metadata_sha256: str
    fingerprint: str
    trajectories: int
    samples: int


@dataclass(frozen=True)
class BaselineRecord:
    player_count: int
    role: str
    act: int
    match_cluster: str
    value: float


@dataclass(frozen=True)
class BaselineResult:
    baseline: float
    scale: float
    tier: int
    reference_count: int


@dataclass(frozen=True)
class _MatchSpec:
    player_count: int
    match_index: int
    seed: int


@dataclass
class _Lane:
    spec: _MatchSpec
    env: DalmutiScalarEnv
    decision_index: int = 0


def _canonical_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _derive_uint32(namespace: str, seed_base: int, *parts: object) -> int:
    value = int.from_bytes(
        hashlib.sha256(canonical_json_bytes([namespace, seed_base, *parts])).digest()[:4],
        "little",
    )
    return value or 1


def _keyed_uniform(namespace: str, seed_base: int, *parts: object) -> float:
    digest = hashlib.sha256(
        canonical_json_bytes(["v4-ppo-categorical-v1", namespace, seed_base, *parts])
    ).digest()
    integer = int.from_bytes(digest[:8], "big") >> 11
    return integer / float(1 << 53)


def masked_categorical_probabilities(
    logits: torch.Tensor | np.ndarray,
    legal_mask: torch.Tensor | np.ndarray,
    *,
    temperature: float,
    epsilon_floor: float,
) -> np.ndarray:
    """Return float64 probabilities with an exact per-legal-action floor."""

    values = np.asarray(
        logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits,
        dtype=np.float64,
    )
    legal = np.asarray(
        legal_mask.detach().cpu().numpy() if isinstance(legal_mask, torch.Tensor) else legal_mask,
        dtype=np.bool_,
    )
    if values.shape != (ACTION_COUNT,) or legal.shape != (ACTION_COUNT,):
        raise ValueError("categorical logits and mask must each have 236 entries")
    if not bool(legal.any()) or not np.all(np.isfinite(values[legal])):
        raise ValueError("categorical distribution requires finite legal logits")
    temperature_value = float(temperature)
    floor = float(epsilon_floor)
    legal_count = int(legal.sum())
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(floor) or not 0.0 <= floor < 1.0 / ACTION_COUNT:
        raise ValueError("epsilon_floor is outside the supported range")
    scaled = values[legal] / temperature_value
    scaled -= float(np.max(scaled))
    weights = np.exp(scaled)
    softmax = weights / float(weights.sum())
    probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
    probabilities[legal] = softmax * (1.0 - floor * legal_count) + floor
    # Absorb only floating-point summation residue into the largest legal mass.
    legal_indexes = np.flatnonzero(legal)
    residue = 1.0 - float(probabilities.sum())
    probabilities[legal_indexes[int(np.argmax(probabilities[legal]))]] += residue
    if np.any(probabilities[legal] < floor - 1.0e-15) or np.any(probabilities[~legal] != 0.0):
        raise RuntimeError("masked categorical floor or legality invariant failed")
    return probabilities


def sample_masked_categorical(probabilities: np.ndarray, uniform: float) -> tuple[int, float, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (ACTION_COUNT,) or np.any(values < 0.0) or not np.isclose(values.sum(), 1.0, atol=1.0e-12):
        raise ValueError("probabilities must be a normalized 236-action vector")
    if not math.isfinite(float(uniform)) or not 0.0 <= float(uniform) < 1.0:
        raise ValueError("uniform must be finite and in [0, 1)")
    positive = np.flatnonzero(values > 0.0)
    if not len(positive):
        raise ValueError("probabilities must include one positive action")
    cumulative = np.cumsum(values)
    action = min(int(np.searchsorted(cumulative, uniform, side="right")), ACTION_COUNT - 1)
    if values[action] <= 0.0:
        action = int(positive[-1])
    probability = float(values[action])
    entropy = float(-np.sum(values[positive] * np.log(values[positive])))
    return action, math.log(probability), entropy


def _batch_candidate_logits(
    model: object,
    observations: Sequence[V4ActorObservation],
    device: torch.device,
) -> list[torch.Tensor]:
    """Batched masked-logit inference grouped by public player/history shape."""

    if not observations:
        return []
    config = getattr(model, "config", None)
    if not isinstance(config, V4ActorConfig):
        raise ValueError("candidate model is missing a V4ActorConfig")
    rows = [_trim_public_for_model(public, config) for public in observations]
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        key = (int(row[2].shape[0]), _history_bucket(int(row[5].shape[0]), config.max_history))
        groups.setdefault(key, []).append(index)
    output: list[torch.Tensor | None] = [None] * len(rows)
    with torch.inference_mode():
        for (player_width, history_width), indexes in sorted(groups.items()):
            selected = [rows[index] for index in indexes]
            tensors = (
                torch.stack([row[0] for row in selected]).to(device),
                torch.stack([row[1] for row in selected]).to(device),
                torch.stack([_pad_rows(row[2], player_width) for row in selected]).to(device),
                torch.stack([_pad_mask(row[3], player_width) for row in selected]).to(device),
                torch.stack([row[4] for row in selected]).to(device),
                torch.stack([_pad_rows(row[5], history_width) for row in selected]).to(device),
                torch.stack([_pad_mask(row[6], history_width) for row in selected]).to(device),
                torch.stack([row[7].to(dtype=torch.bool) for row in selected]).to(device),
            )
            legal = tensors[-1]
            logits = model(*tensors)
            if tuple(logits.shape) != (len(indexes), ACTION_COUNT):
                raise ValueError("candidate returned an invalid masked-logit shape")
            if not bool(torch.isfinite(logits[legal]).all().item()):
                raise ValueError("candidate returned non-finite legal logits")
            logits_cpu = logits.detach().to(device="cpu", dtype=torch.float64)
            for local_index, source_index in enumerate(indexes):
                row_logits = logits_cpu[local_index].clone()
                row_logits[~rows[source_index][7].to(dtype=torch.bool)] = float("-inf")
                output[source_index] = row_logits
    if any(value is None for value in output):
        raise RuntimeError("candidate batching lost an observation")
    return [value for value in output if value is not None]


def learner_actor_ids(player_count: int, rotation_index: int, act: int, acts: int, count: int) -> tuple[int, ...]:
    """Rotate learner identities with a discrepancy of at most one assignment."""

    slot = (rotation_index * acts + (act - 1)) * count
    return tuple((slot + offset) % player_count for offset in range(count))


def candidate_opponent_ids(
    player_count: int,
    rotation_index: int,
    act: int,
    acts: int,
    learner_ids: Sequence[int],
    fraction: float,
) -> tuple[int, ...]:
    remaining = tuple(actor for actor in range(player_count) if actor not in set(learner_ids))
    if not remaining or fraction <= 0.0:
        return ()
    if fraction >= 1.0:
        return remaining
    slot = rotation_index * acts + (act - 1)
    quota_before = math.floor(slot * len(remaining) * fraction + 1.0e-12)
    quota_after = math.floor((slot + 1) * len(remaining) * fraction + 1.0e-12)
    count = quota_after - quota_before
    start = slot % len(remaining)
    rotated = remaining[start:] + remaining[:start]
    return tuple(rotated[:count])


def leave_one_match_out_baselines(records: Sequence[BaselineRecord]) -> tuple[BaselineResult, ...]:
    """Compute leakage-safe baselines, excluding the whole target match.

    Fallback order is p/role/act, p/role, p/act, p, global, then zero.
    Standard-deviation scales are computed from the same excluded-reference
    population.  Degenerate reference populations use scale 1.
    """

    output: list[BaselineResult] = []
    for target in records:
        other = [record for record in records if record.match_cluster != target.match_cluster]
        filters = (
            lambda value: value.player_count == target.player_count and value.role == target.role and value.act == target.act,
            lambda value: value.player_count == target.player_count and value.role == target.role,
            lambda value: value.player_count == target.player_count and value.act == target.act,
            lambda value: value.player_count == target.player_count,
            lambda value: True,
        )
        references: list[BaselineRecord] = []
        tier = len(BASELINE_FALLBACK_HIERARCHY) - 1
        for index, predicate in enumerate(filters):
            references = [record for record in other if predicate(record)]
            if references:
                tier = index
                break
        if references:
            values = np.asarray([record.value for record in references], dtype=np.float64)
            baseline = float(values.mean())
            deviation = float(values.std(ddof=0))
            scale = deviation if deviation >= 1.0e-8 else 1.0
        else:
            baseline = 0.0
            scale = 1.0
        output.append(BaselineResult(baseline, scale, tier, len(references)))
    return tuple(output)


def _source_hashes(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required PPO collection source is missing: {relative}")
        output[relative] = sha256_file(source)
    return output


def _canonical_privileged_vector(env: DalmutiScalarEnv) -> torch.Tensor:
    """Rebuild the raw TS ``V4_PRIVILEGED_CRITIC_LAYOUT`` independently."""

    if env.terminated:
        raise ValueError("privileged layout audit requires an active environment")
    actor_id = env.current_player_id
    relative = env._relative_order(actor_id)
    table = env._table
    table_offset = -1 if table is None else relative.index(table.player_id)
    vector = torch.zeros(PRIVILEGED_STATE_SIZE, dtype=torch.float32)
    actor_position = env._order.index(actor_id)
    vector[:16] = torch.tensor(
        [
            env.player_count,
            env._act,
            env._revolution,
            int(table is not None),
            0 if table is None else table.rank,
            0 if table is None else table.natural_count,
            0 if table is None else table.joker_count,
            0 if table is None else table.count,
            table_offset,
            sum(env._public_played),
            sum(int(bool(env._hands[player_id])) for player_id in env._order),
            len(env._finish_order),
            ROLES.index(role_for_index(actor_position, env.player_count)),
            env._scores[actor_id],
            len(env._hands[actor_id]),
            len(env._history),
        ],
        dtype=torch.float32,
    )
    vector[16:29] = torch.tensor(env._public_played, dtype=torch.float32)
    for relative_offset, player_id in enumerate(relative):
        counts = [0] * 13
        for card in env._hands[player_id]:
            counts[card.rank - 1] += 1
        position = env._order.index(player_id)
        role_index = ROLES.index(role_for_index(position, env.player_count))
        finish_place = (
            env._finish_order.index(player_id) + 1
            if player_id in env._finish_order
            else 0
        )
        row = [
            1,
            relative_offset,
            *[int(index == role_index) for index in range(5)],
            env._scores[player_id],
            len(env._hands[player_id]),
            int(player_id in env._passed),
            int(not env._hands[player_id]),
            finish_place,
            *counts,
        ]
        start = 29 + relative_offset * 25
        vector[start : start + 25] = torch.tensor(row, dtype=torch.float32)
    return vector


def assert_canonical_privileged_layout(env: DalmutiScalarEnv) -> dict[str, object]:
    recomputed_layout_sha = _sha256_bytes(
        _canonical_text(CANONICAL_PRIVILEGED_LAYOUT).encode("utf-8")
    )
    if (
        CANONICAL_PRIVILEGED_LAYOUT_ID
        != "dalmuti-v4-ts-privileged-critic-raw-v1"
        or CANONICAL_PRIVILEGED_LAYOUT_SHA256 != recomputed_layout_sha
        or recomputed_layout_sha != EXPECTED_CANONICAL_PRIVILEGED_LAYOUT_SHA256
    ):
        raise RuntimeError("declared privileged critic layout identifier or hash drifted")
    actual = env.privileged_state().detach().cpu().to(dtype=torch.float32)
    expected = _canonical_privileged_vector(env)
    if actual.shape != (PRIVILEGED_STATE_SIZE,):
        raise RuntimeError("privileged critic vector does not have 512 features")
    if not torch.equal(actual, expected):
        mismatches = torch.nonzero(actual != expected, as_tuple=False).flatten()
        first = int(mismatches[0].item()) if mismatches.numel() else -1
        raise RuntimeError(
            "privileged critic layout drifted from training/simulator.ts "
            f"(first mismatch at feature {first})"
        )
    if bool(actual[279:].ne(0).any().item()):
        raise RuntimeError("privileged critic reserved tail must remain exactly zero")
    return {
        "layoutId": CANONICAL_PRIVILEGED_LAYOUT_ID,
        "layoutSha256": CANONICAL_PRIVILEGED_LAYOUT_SHA256,
        "layout": CANONICAL_PRIVILEGED_LAYOUT,
        "liveVectorMatchedCanonicalLayout": True,
        "reservedZeroTailVerified": True,
    }


def _new_lane(spec: _MatchSpec, acts: int) -> _Lane:
    return _Lane(spec, DalmutiScalarEnv(spec.player_count, acts=acts, seed=spec.seed, device="cpu"))


def _match_cluster(spec: _MatchSpec) -> str:
    return f"p{spec.player_count}-m{spec.match_index}-seed{spec.seed:08x}"


def _trajectory_id(config: PPOCollectionConfig, spec: _MatchSpec, act: int, actor_id: int) -> str:
    return (
        f"v4-ppo-{config.run_namespace}-shard{config.match_shard_index}of{config.match_shard_count}"
        f"-p{spec.player_count}-m{spec.match_index}-seed{spec.seed:08x}-act{act}-actor{actor_id}"
    )


def _finite_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("statistics received non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _allocate_arrays(count: int, steps: int, actor: V4ActorConfig, critic: V4CriticConfig) -> dict[str, np.ndarray]:
    prefix = (count, steps)
    return {
        "global_features": np.zeros((*prefix, actor.global_features), np.float32),
        "rank_features": np.zeros((*prefix, actor.rank_tokens, actor.rank_features), np.float32),
        "player_features": np.zeros((*prefix, actor.max_players, actor.player_features), np.float32),
        "player_mask": np.zeros((*prefix, actor.max_players), np.bool_),
        "memory_trace_features": np.zeros((*prefix, actor.memory_tokens, actor.memory_features), np.float32),
        "history_features": np.zeros((*prefix, actor.max_history, actor.history_features), np.float32),
        "history_mask": np.zeros((*prefix, actor.max_history), np.bool_),
        "legal_masks": np.zeros((*prefix, ACTION_COUNT), np.bool_),
        "actions": np.zeros(prefix, np.int64),
        "expert_actions": np.zeros(prefix, np.int64),
        "old_action_log_probs": np.zeros(prefix, np.float32),
        "advantages": np.zeros(prefix, np.float32),
        "rewards": np.zeros(prefix, np.float32),
        "dones": np.zeros(prefix, np.bool_),
        "valid_masks": np.zeros(prefix, np.bool_),
        "privileged_states": np.zeros((*prefix, critic.privileged_features), np.float32),
        "raw_returns": np.zeros(prefix, np.float32),
        "baseline_values": np.zeros(prefix, np.float32),
        "raw_advantages": np.zeros(prefix, np.float32),
        "advantage_scales": np.ones(prefix, np.float32),
        "baseline_tiers": np.full(prefix, -1, np.int8),
        "baseline_reference_counts": np.zeros(prefix, np.int32),
        "selected_action_probabilities": np.zeros(prefix, np.float64),
        "policy_entropies": np.zeros(prefix, np.float32),
        "terminal_chip_awards": np.zeros(prefix, np.int8),
        "forced_masks": np.zeros(prefix, np.bool_),
        "source_decision_indices": np.full(prefix, -1, np.int64),
    }


def _build_arrays(
    trajectories: Sequence[Mapping[str, object]],
    actor: V4ActorConfig,
    critic: V4CriticConfig,
    *,
    gamma: float,
    standardize: bool,
) -> tuple[dict[str, np.ndarray], V4TrajectoryDataset, tuple[BaselineResult, ...]]:
    if not trajectories:
        raise RuntimeError("PPO collection produced no learner trajectories")
    records = tuple(
        BaselineRecord(
            int(trajectory["player_count"]),
            str(trajectory["role"]),
            int(trajectory["act"]),
            str(trajectory["match_cluster"]),
            float(trajectory["terminal_reward"]),
        )
        for trajectory in trajectories
    )
    baselines = leave_one_match_out_baselines(records)
    maximum = max(len(trajectory["rows"]) for trajectory in trajectories)
    arrays = _allocate_arrays(len(trajectories), maximum, actor, critic)
    public_names = (
        "global_features", "rank_features", "player_features", "player_mask",
        "memory_trace_features", "history_features", "history_mask", "legal_masks",
    )
    standard_row_names = (
        "actions", "expert_actions", "old_action_log_probs", "rewards", "dones",
        "privileged_states", "selected_action_probabilities", "policy_entropies",
        "terminal_chip_awards", "forced_masks", "source_decision_indices",
    )
    for trajectory_index, (trajectory, baseline) in enumerate(zip(trajectories, baselines, strict=True)):
        rows = trajectory["rows"]
        if not rows or sum(int(bool(row["dones"])) for row in rows) != 1 or not rows[-1]["dones"]:
            raise RuntimeError("each learner trajectory requires exactly one terminal final row")
        running_return = 0.0
        returns = [0.0] * len(rows)
        for index in range(len(rows) - 1, -1, -1):
            running_return = float(rows[index]["rewards"]) + float(gamma) * running_return
            returns[index] = running_return
        for time_index, row in enumerate(rows):
            for name in public_names:
                arrays[name][trajectory_index, time_index] = row[name]
            for name in standard_row_names:
                arrays[name][trajectory_index, time_index] = row[name]
            raw_advantage = returns[time_index] - baseline.baseline
            advantage = raw_advantage / baseline.scale if standardize else raw_advantage
            arrays["raw_returns"][trajectory_index, time_index] = returns[time_index]
            arrays["baseline_values"][trajectory_index, time_index] = baseline.baseline
            arrays["raw_advantages"][trajectory_index, time_index] = raw_advantage
            arrays["advantage_scales"][trajectory_index, time_index] = baseline.scale
            arrays["baseline_tiers"][trajectory_index, time_index] = baseline.tier
            arrays["baseline_reference_counts"][trajectory_index, time_index] = baseline.reference_count
            arrays["advantages"][trajectory_index, time_index] = advantage
            arrays["valid_masks"][trajectory_index, time_index] = True
    standard_names = {field.name for field in fields(V4TrajectoryTensors)}
    boolean_names = {"player_mask", "history_mask", "legal_masks", "dones", "valid_masks"}
    integer_names = {"actions", "expert_actions"}
    tensors: dict[str, torch.Tensor] = {}
    for name in standard_names:
        tensor = torch.from_numpy(arrays[name])
        if name in boolean_names:
            tensor = tensor.to(dtype=torch.bool)
        elif name in integer_names:
            tensor = tensor.to(dtype=torch.long)
        else:
            tensor = tensor.to(dtype=torch.float32)
        tensors[name] = tensor
    dataset = V4TrajectoryDataset(V4TrajectoryTensors(**tensors), actor, critic)
    return arrays, dataset, baselines


def _assignment_report(specs: Sequence[_MatchSpec], config: PPOCollectionConfig) -> dict[str, object]:
    learner: dict[str, dict[str, int]] = {}
    opponents: dict[str, dict[str, int]] = {}
    for spec in specs:
        learner_counts = learner.setdefault(str(spec.player_count), {str(actor): 0 for actor in range(spec.player_count)})
        opponent_counts = opponents.setdefault(str(spec.player_count), {str(actor): 0 for actor in range(spec.player_count)})
        for act in range(1, config.acts + 1):
            learner_ids = learner_actor_ids(spec.player_count, spec.match_index, act, config.acts, config.candidate_seats_per_act)
            candidate_opponents = candidate_opponent_ids(
                spec.player_count, spec.match_index, act, config.acts,
                learner_ids, config.opponent_candidate_fraction,
            )
            for actor in learner_ids:
                learner_counts[str(actor)] += 1
            for actor in candidate_opponents:
                opponent_counts[str(actor)] += 1
    imbalance = {
        player_count: max(counts.values()) - min(counts.values())
        for player_count, counts in learner.items()
    }
    return {
        "learnerAssignmentsByPlayerIdentity": learner,
        "candidateOpponentAssignmentsByPlayerIdentity": opponents,
        "learnerIdentityMaxMinusMin": imbalance,
        "rotationRule": "cyclic physical identity over match-index, act, learner slot",
        "allIdentitiesBalancedWhenScheduleContainsAFullCycle": all(value <= 1 for value in imbalance.values()),
        "everyIdentityReceivesLearnerExperience": all(
            min(counts.values()) > 0 for counts in learner.values()
        ),
    }


def collect_v4_ppo(
    bundle_directory: str | Path,
    output_path: str | Path,
    config: PPOCollectionConfig,
    *,
    repository_root: str | Path | None = None,
) -> PPOCollectionResult:
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("PPO output must end in .npz")
    metadata_path = Path(f"{output}.metadata.json")
    checksum_path = Path(f"{output}.sha256")
    metadata_checksum_path = Path(f"{metadata_path}.sha256")
    for target in (output, metadata_path, checksum_path, metadata_checksum_path):
        if target.exists():
            raise FileExistsError(f"output already exists: {target}")
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parent.parent
    bundle = Path(bundle_directory).resolve()
    manifest = verify_v4_actor_bundle(bundle)
    model, payload = load_v4_actor_checkpoint(bundle / "actor.pt")
    actor_config = getattr(model, "config", None)
    if not isinstance(actor_config, V4ActorConfig):
        raise ValueError("verified bundle did not load a V4 actor configuration")
    _validate_actor_contract(actor_config, config.player_counts)
    if payload.get("criticExcluded") is not True:
        raise ValueError("candidate actor checkpoint did not exclude the critic")
    device = torch.device(config.device)
    if device.type == "cuda":
        # PyTorch deterministic CuBLAS kernels require this before the first
        # CUDA matrix multiplication.  setdefault preserves an operator's
        # stricter preconfigured choice.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    execution = _configure_determinism(device)
    execution["cublasWorkspaceConfig"] = (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device.type == "cuda" else None
    )
    model = model.to(device).eval()

    specs: list[_MatchSpec] = []
    complete_specs: list[_MatchSpec] = []
    for player_count in config.player_counts:
        complete_indexes = list(
            range(
                config.match_start,
                config.match_start + config.matches_per_player_count,
            )
        )
        selected_indexes = [
            match_index
            for match_index in complete_indexes
            if match_index % config.match_shard_count == config.match_shard_index
        ]
        complete_specs.extend(
            _MatchSpec(
                player_count,
                match_index,
                _derive_uint32(
                    config.run_namespace,
                    config.seed_base,
                    "environment",
                    player_count,
                    match_index,
                ),
            )
            for match_index in complete_indexes
        )
        specs.extend(
            _MatchSpec(
                player_count,
                match_index,
                _derive_uint32(
                    config.run_namespace,
                    config.seed_base,
                    "environment",
                    player_count,
                    match_index,
                ),
            )
            for match_index in selected_indexes
        )
    if not specs:
        raise ValueError("the requested match shard is empty")
    if len({spec.seed for spec in specs}) != len(specs):
        raise RuntimeError("derived environment seed schedule contains a collision")
    lanes = [_new_lane(spec, config.acts) for spec in specs[: config.lane_count]]
    next_spec = len(lanes)
    trajectories: dict[str, dict[str, object]] = {}
    privacy_audits: dict[str, object] = {}
    privileged_layout_audits: dict[str, object] = {}
    action_counts = {
        "learner": {"decisions": 0, "differentFromNormal": 0, "forced": 0},
        "candidateOpponent": {"decisions": 0, "differentFromNormal": 0, "forced": 0},
        "normalOpponent": {"decisions": 0, "forced": 0},
    }
    entropy_values: list[float] = []
    total_environment_decisions = 0
    while lanes:
        publics = [lane.env.public_observation() for lane in lanes]
        normal_actions = [lane.env.normal_action() for lane in lanes]
        candidate_lane_indexes: list[int] = []
        candidate_kinds: dict[int, str] = {}
        assignments: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for index, lane in enumerate(lanes):
            env = lane.env
            learner_ids = learner_actor_ids(
                lane.spec.player_count, lane.spec.match_index, int(env._act), config.acts,
                config.candidate_seats_per_act,
            )
            opponent_ids = candidate_opponent_ids(
                lane.spec.player_count, lane.spec.match_index, int(env._act), config.acts,
                learner_ids, config.opponent_candidate_fraction,
            )
            assignments[index] = (learner_ids, opponent_ids)
            actor_id = env.current_player_id
            if actor_id in learner_ids:
                candidate_lane_indexes.append(index)
                candidate_kinds[index] = "learner"
            elif actor_id in opponent_ids:
                candidate_lane_indexes.append(index)
                candidate_kinds[index] = "candidateOpponent"
        logits = _batch_candidate_logits(
            model, [publics[index] for index in candidate_lane_indexes], device
        )
        candidate_results: dict[int, tuple[int, float, float, float]] = {}
        for lane_index, row_logits in zip(candidate_lane_indexes, logits, strict=True):
            lane = lanes[lane_index]
            public = publics[lane_index]
            probabilities = masked_categorical_probabilities(
                row_logits, public.legal_mask,
                temperature=config.temperature, epsilon_floor=config.epsilon_floor,
            )
            uniform = _keyed_uniform(
                config.run_namespace, config.seed_base, "action", lane.spec.player_count,
                lane.spec.match_index, lane.spec.seed, int(lane.env._act),
                lane.env.current_player_id, lane.decision_index,
            )
            action, log_probability, entropy = sample_masked_categorical(probabilities, uniform)
            candidate_results[lane_index] = (action, log_probability, entropy, float(probabilities[action]))

        replacements: list[_Lane] = []
        for lane_index, (lane, public, normal_action) in enumerate(zip(lanes, publics, normal_actions, strict=True)):
            env = lane.env
            spec = lane.spec
            act = int(env._act)
            actor_id = env.current_player_id
            actor_position = env._order.index(actor_id)
            role = role_for_index(actor_position, spec.player_count)
            learner_ids, _ = assignments[lane_index]
            kind = candidate_kinds.get(lane_index, "normalOpponent")
            if kind == "normalOpponent":
                behavior_action = normal_action
                old_log_probability = 0.0
                entropy = 0.0
                selected_probability = 1.0
            else:
                behavior_action, old_log_probability, entropy, selected_probability = candidate_results[lane_index]
                entropy_values.append(entropy)
            legal = public.legal_mask
            if not bool(legal[normal_action].item()) or not bool(legal[behavior_action].item()):
                raise RuntimeError("Normal or behavior policy selected an illegal action")
            forced = int(legal.sum().item()) == 1
            counter = action_counts[kind]
            counter["decisions"] += 1
            counter["forced"] += int(forced)
            if kind != "normalOpponent":
                counter["differentFromNormal"] += int(behavior_action != normal_action)
            if str(spec.player_count) not in privacy_audits:
                privileged_layout_audits[str(spec.player_count)] = (
                    assert_canonical_privileged_layout(env)
                )
                privacy_audits[str(spec.player_count)] = audit_hidden_state_privacy(
                    env,
                    _derive_uint32(config.run_namespace, config.seed_base, "privacy", spec.player_count),
                )
            collect = actor_id in learner_ids and act in config.collected_acts and role in config.role_filter
            if collect:
                identifier = _trajectory_id(config, spec, act, actor_id)
                trajectory = trajectories.setdefault(
                    identifier,
                    {
                        "id": identifier,
                        "player_count": spec.player_count,
                        "match_index": spec.match_index,
                        "match_seed": spec.seed,
                        "match_cluster": _match_cluster(spec),
                        "act": act,
                        "actor_id": actor_id,
                        "role": role,
                        "rows": [],
                        "terminal_reward": None,
                        "finish_place": None,
                    },
                )
                row: dict[str, object] = {
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
            lane.decision_index += 1
            total_environment_decisions += 1
            result = env.step(int(behavior_action))
            if result.act_ended:
                act_result = result.info.get("act_result")
                if not isinstance(act_result, Mapping):
                    raise RuntimeError("act terminal is missing its exact result")
                finish_order = tuple(int(value) for value in act_result["finish_order"])
                chip_awards = act_result["chip_awards"]
                if not isinstance(chip_awards, Mapping):
                    raise RuntimeError("act terminal chip awards are missing")
                for finish_place, finished_actor in enumerate(finish_order, start=1):
                    identifier = _trajectory_id(config, spec, act, finished_actor)
                    trajectory = trajectories.get(identifier)
                    if trajectory is None:
                        continue
                    rows = trajectory["rows"]
                    if not rows or rows[-1]["dones"]:
                        raise RuntimeError("learner actor trajectory terminal is missing or duplicated")
                    reward = float(result.rewards[finished_actor].item())
                    rows[-1]["rewards"] = reward
                    rows[-1]["dones"] = True
                    rows[-1]["terminal_chip_awards"] = int(chip_awards[finished_actor])
                    trajectory["terminal_reward"] = reward
                    trajectory["finish_place"] = finish_place
            if result.terminated:
                if next_spec < len(specs):
                    replacements.append(_new_lane(specs[next_spec], config.acts))
                    next_spec += 1
            else:
                replacements.append(lane)
        lanes = replacements

    ordered = [trajectories[key] for key in sorted(trajectories)]
    if not ordered or any(item["terminal_reward"] is None for item in ordered):
        raise RuntimeError("collection contains no trajectories or an unterminated learner trajectory")
    critic_config = V4CriticConfig(privileged_features=PRIVILEGED_STATE_SIZE)
    arrays, dataset, baselines = _build_arrays(
        ordered, actor_config, critic_config,
        gamma=config.gamma, standardize=config.standardize_advantages,
    )
    arrays.update(
        {
            "trajectory_ids": np.asarray([item["id"] for item in ordered], dtype=np.str_),
            "trajectory_player_counts": np.asarray([item["player_count"] for item in ordered], np.int16),
            "trajectory_roles": np.asarray([ROLES.index(str(item["role"])) for item in ordered], np.int8),
            "trajectory_acts": np.asarray([item["act"] for item in ordered], np.int16),
            "trajectory_actor_ids": np.asarray([item["actor_id"] for item in ordered], np.int16),
            "trajectory_match_indices": np.asarray([item["match_index"] for item in ordered], np.int32),
            "trajectory_match_seeds": np.asarray([item["match_seed"] for item in ordered], np.uint32),
            "trajectory_match_clusters": np.asarray([item["match_cluster"] for item in ordered], dtype=np.str_),
            "trajectory_finish_places": np.asarray([item["finish_place"] for item in ordered], np.int16),
        }
    )
    source_hashes = _source_hashes(root)
    valid = arrays["valid_masks"]
    action_count_report: dict[str, object] = {}
    for name, counts in action_counts.items():
        decisions = int(counts["decisions"])
        record: dict[str, object] = dict(counts)
        if name != "normalOpponent":
            record["differentFromNormalRate"] = counts["differentFromNormal"] / max(1, decisions)
        action_count_report[name] = record
    tier_counts = {name: 0 for name in BASELINE_FALLBACK_HIERARCHY}
    for baseline in baselines:
        tier_counts[BASELINE_FALLBACK_HIERARCHY[baseline.tier]] += 1
    seat_balance = _assignment_report(specs, config)
    seat_balance["completeMatchRangeAcrossAllShards"] = _assignment_report(
        complete_specs, config
    )
    metadata: dict[str, object] = {
        "format": V4_DATASET_FORMAT,
        "version": V4_DATASET_VERSION,
        "preparationFormat": PPO_PREPARATION_FORMAT,
        "preparationVersion": PPO_PREPARATION_VERSION,
        "fingerprint": dataset.fingerprint,
        "actorConfig": actor_config.to_dict(),
        "criticConfig": critic_config.to_dict(),
        "collection": {
            "algorithm": "on-policy PPO league rollout",
            "learnerBehavior": "frozen candidate stochastic masked categorical",
            "opponents": "deterministic rotating mix of exact Normal and the same frozen candidate",
            "temperature": config.temperature,
            "epsilonFloorPerLegalAction": config.epsilon_floor,
            "opponentCandidateFraction": config.opponent_candidate_fraction,
            "candidateSeatsPerAct": config.candidate_seats_per_act,
            "exactOldLogProbabilityForEveryLearnerDecision": True,
            "exactNormalExpertLabelForEveryLearnerDecision": True,
            "actsPerMatch": config.acts,
            "collectedActs": list(config.collected_acts),
            "collectedRoles": list(config.role_filter),
            "rollingCpuEnvironmentLanes": min(config.lane_count, len(specs)),
            "batchedGpuMaskedLogitInference": device.type == "cuda",
            "terminalReward": "(exact round chip award - 2) / 2 on the final learner decision",
        },
        "returnsAndAdvantages": {
            "monteCarloGamma": config.gamma,
            "baseline": "deterministic leave-one-entire-match-cluster-out mean",
            "fallbackHierarchy": list(BASELINE_FALLBACK_HIERARCHY),
            "fallbackCounts": tier_counts,
            "standardized": config.standardize_advantages,
            "standardizationScale": "population std from the same leave-one-match-out references; 1 if degenerate",
            "futureHoldoutUsed": False,
            "opponentHiddenHandsUsed": False,
            "rawReturnStats": _finite_stats(arrays["raw_returns"][valid].tolist()),
            "baselineStats": _finite_stats(arrays["baseline_values"][valid].tolist()),
            "rawAdvantageStats": _finite_stats(arrays["raw_advantages"][valid].tolist()),
            "trainingAdvantageStats": _finite_stats(arrays["advantages"][valid].tolist()),
        },
        "shard": {
            "runNamespace": config.run_namespace,
            "seedBase": config.seed_base,
            "playerCounts": list(config.player_counts),
            "roleFilter": list(config.role_filter),
            "actFilter": list(config.collected_acts),
            "matchStart": config.match_start,
            "matchesPerPlayerCount": config.matches_per_player_count,
            "matchShardCount": config.match_shard_count,
            "matchShardIndex": config.match_shard_index,
            "environmentSeeds": [spec.seed for spec in specs],
            "identitySha256": _sha256_bytes(canonical_json_bytes([
                PPO_PREPARATION_FORMAT, config.run_namespace, config.seed_base,
                list(config.player_counts), list(config.role_filter), list(config.collected_acts),
                config.match_start, config.matches_per_player_count,
                config.match_shard_count, config.match_shard_index,
            ])),
            "trajectoryIdsBindPlayerActMatchActorAndNamespace": True,
            "trajectoryRolesAreBoundByTheHashedNpzArray": True,
        },
        "modelBinding": {
            "bundleManifestSha256": sha256_file(bundle / "manifest.json"),
            "actorCheckpointSha256": sha256_file(bundle / "actor.pt"),
            "manifestFormat": manifest.get("format"),
            "manifestVersion": manifest.get("version"),
            "modelKind": manifest.get("model", {}).get("kind"),
            "criticExcluded": True,
        },
        "environmentBinding": {
            "implementation": "DalmutiScalarEnv",
            "normalExpertCallback": "DalmutiScalarEnv.normal_action",
            "v4EnvSha256": source_hashes["gpu-training/v4_env.py"],
            "normalSourceSha256": source_hashes["lib/bot-strategy.ts"],
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
            "perPlayerCountAudits": privacy_audits,
        },
        "actionRates": action_count_report,
        "policyEntropy": _finite_stats(entropy_values),
        "opponentAndSeatBalance": seat_balance,
        "trajectoryCount": len(ordered),
        "sampleCount": int(valid.sum()),
        "environmentDecisionCount": total_environment_decisions,
        "maxTimeSteps": int(arrays["actions"].shape[1]),
        "auxiliaryArrays": sorted(set(arrays) - {field.name for field in fields(V4TrajectoryTensors)}),
        "padding": "zero-valued invalid suffix; auxiliary integer sentinels are -1 where documented",
    }
    arrays["metadata_json"] = np.asarray(_canonical_text(metadata))
    npz_bytes = _deterministic_npz_bytes(arrays)
    npz_sha = _sha256_bytes(npz_bytes)
    external = dict(metadata)
    external["npzSha256"] = npz_sha
    metadata_bytes = (_canonical_text(external) + "\n").encode("utf-8")
    metadata_sha = _sha256_bytes(metadata_bytes)
    _exclusive_publish(
        {
            output: npz_bytes,
            metadata_path: metadata_bytes,
            checksum_path: f"{npz_sha}  {output.name}\n".encode("ascii"),
            metadata_checksum_path: f"{metadata_sha}  {metadata_path.name}\n".encode("ascii"),
        }
    )
    return PPOCollectionResult(
        output, metadata_path, checksum_path, metadata_checksum_path,
        npz_sha, metadata_sha, dataset.fingerprint, len(ordered), int(valid.sum()),
    )


def _parse_int_tuple(value: str, label: str, minimum: int, maximum: int) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from error
    if not result or any(not minimum <= item <= maximum for item in result):
        raise argparse.ArgumentTypeError(f"{label} must be from {minimum} through {maximum}")
    return result


def _parse_roles(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result) or any(item not in ROLES for item in result):
        raise argparse.ArgumentTypeError(f"roles must be a unique subset of {','.join(ROLES)}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect checksum-bound V4 on-policy PPO league trajectories.")
    parser.add_argument("--actor-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-namespace", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--player-counts", type=lambda value: _parse_int_tuple(value, "player counts", 4, 10), default=tuple(range(4, 11)))
    parser.add_argument("--matches-per-player-count", type=int, default=8)
    parser.add_argument("--match-start", type=int, default=0)
    parser.add_argument("--match-shard-count", type=int, default=1)
    parser.add_argument("--match-shard-index", type=int, default=0)
    parser.add_argument("--acts", type=int, default=5)
    parser.add_argument("--act-filter", type=lambda value: _parse_int_tuple(value, "acts", 1, 1_000_000))
    parser.add_argument("--role-filter", type=_parse_roles, default=ROLES)
    parser.add_argument("--candidate-seats-per-act", type=int, default=1)
    parser.add_argument("--opponent-candidate-fraction", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon-floor", type=float, default=1.0e-6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--no-standardize-advantages", action="store_true")
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = PPOCollectionConfig(
        run_namespace=arguments.run_namespace,
        seed_base=arguments.seed_base,
        player_counts=arguments.player_counts,
        matches_per_player_count=arguments.matches_per_player_count,
        match_start=arguments.match_start,
        match_shard_count=arguments.match_shard_count,
        match_shard_index=arguments.match_shard_index,
        acts=arguments.acts,
        act_filter=arguments.act_filter,
        role_filter=arguments.role_filter,
        candidate_seats_per_act=arguments.candidate_seats_per_act,
        opponent_candidate_fraction=arguments.opponent_candidate_fraction,
        temperature=arguments.temperature,
        epsilon_floor=arguments.epsilon_floor,
        gamma=arguments.gamma,
        standardize_advantages=not arguments.no_standardize_advantages,
        lane_count=arguments.lanes,
        device=arguments.device,
    )
    result = collect_v4_ppo(
        arguments.actor_bundle, arguments.output, config,
        repository_root=arguments.repository_root,
    )
    print(_canonical_text({
        "output": str(result.output_path), "npzSha256": result.npz_sha256,
        "metadataSha256": result.metadata_sha256, "fingerprint": result.fingerprint,
        "trajectories": result.trajectories, "samples": result.samples,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_FALLBACK_HIERARCHY",
    "CANONICAL_PRIVILEGED_LAYOUT",
    "CANONICAL_PRIVILEGED_LAYOUT_ID",
    "CANONICAL_PRIVILEGED_LAYOUT_SHA256",
    "PPO_PREPARATION_FORMAT",
    "PPO_PREPARATION_VERSION",
    "BaselineRecord",
    "BaselineResult",
    "PPOCollectionConfig",
    "PPOCollectionResult",
    "candidate_opponent_ids",
    "collect_v4_ppo",
    "assert_canonical_privileged_layout",
    "learner_actor_ids",
    "leave_one_match_out_baselines",
    "main",
    "masked_categorical_probabilities",
    "sample_masked_categorical",
]
