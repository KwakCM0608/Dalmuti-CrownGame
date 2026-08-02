from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from v4_model import (
    V4_ACTION_COUNT,
    V4ActorConfig,
    V4CriticConfig,
    validate_v4_policy_numerics_contract,
)
from v4_env import (
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
    PRIVILEGED_STATE_SIZE,
    ROLES,
)
from v4_ppo_advantages import BaselineRecord, leave_one_match_out_baselines, validate_merged_ppo_advantages


V4_DATASET_FORMAT = "dalmuti-v4-trajectory-npz"
V4_DATASET_VERSION = 1

# Preparation formats are part of the loss contract, not merely descriptive
# provenance.  In particular, the legacy Normal and DAgger tensors contain
# finite placeholders in the PPO columns; those values must never silently
# become PPO/critic training targets.
V4_NORMAL_PREPARATION_FORMAT = "dalmuti-v4-prepared-dataset-metadata"
V4_DAGGER_PREPARATION_FORMAT = "dalmuti-v4-dagger-direct-npz"
V4_PPO_PREPARATION_FORMAT = "dalmuti-v4-ppo-league-direct-npz"
V4_FIXED_MATCH_PPO_PREPARATION_FORMAT = "dalmuti-v4-ppo-fixed-match-suffix-direct-npz"
V4_MERGED_PREPARATION_FORMAT = "dalmuti-v4-merged-prepared-dataset-metadata"
V4_SMOKE_PREPARATION_FORMAT = "dalmuti-v4-smoke-generated"
V4_LOSS_ELIGIBILITY_VERSION = 1
V4_LEGACY_PPO_SOURCE_CONTRACT = "legacy-per-act-v1"
V4_FIXED_PPO_SOURCE_CONTRACT = "fixed-physical-id-five-act-suffix-v1"
V4_FIXED_PPO_REWARD_CONTRACT_ID = "fixed-group-chip-pairwise-five-act-suffix-v1"
V4_FIXED_PPO_BEHAVIOR_POLICY_CONTRACT_ID = "raw-masked-softmax-v1"
V4_FIXED_COLLECTION_PLAN_ID = "fixed-complete-shard-plan-v1"
V4_FIXED_MIXED_BACKEND_COLLECTION_PLAN_ID = (
    "fixed-complete-mixed-backend-shard-plan-v2"
)
_FIXED_COLLECTION_NAMESPACE_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
V4_LOSS_MASK_NAMES = {
    "behaviorCloning": "bc_eligible_masks",
    "ppo": "ppo_eligible_masks",
    "critic": "critic_eligible_masks",
}


@dataclass(frozen=True)
class V4LossEligibility:
    """Per-sample, provenance-bound eligibility for each training loss."""

    behavior_cloning: torch.Tensor
    ppo: torch.Tensor
    critic: torch.Tensor
    preparation_format: str
    preparation_version: int
    behavior_actor_sha256s: tuple[str, ...] = ()
    ppo_source_contracts: tuple[str, ...] = ()
    requires_player_count_balanced_loss: bool = False
    requires_qboost_coefficient_zero: bool = False
    ppo_reward_contracts: tuple[str, ...] = ()
    ppo_behavior_policy_contracts: tuple[str, ...] = ()
    fixed_collection_plan_ids: tuple[str, ...] = ()

    def masks(self) -> dict[str, torch.Tensor]:
        return {
            V4_LOSS_MASK_NAMES["behaviorCloning"]: self.behavior_cloning,
            V4_LOSS_MASK_NAMES["ppo"]: self.ppo,
            V4_LOSS_MASK_NAMES["critic"]: self.critic,
        }


def _integer(
    value: object, minimum: int, maximum: int, label: str
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be from {minimum} to {maximum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sequence(value: object, size: int, label: str) -> Sequence[object]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} entries")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = set(value.keys())
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        detail = unknown[0] if unknown else missing[0]
        raise ValueError(f"{label} has an unknown or missing field: {detail}")


def _one_hot(index: int, size: int) -> list[float]:
    result = [0.0] * size
    result[index] = 1.0
    return result


def _deck_copies(rank_index: int) -> int:
    return 2 if rank_index == 12 else rank_index + 1


@dataclass(frozen=True)
class V4PublicTensors:
    global_features: torch.Tensor
    rank_features: torch.Tensor
    player_features: torch.Tensor
    player_mask: torch.Tensor
    memory_trace_features: torch.Tensor
    history_features: torch.Tensor
    history_mask: torch.Tensor


def tensorize_v4_public_observation(
    value: Mapping[str, object],
    config: V4ActorConfig | None = None,
) -> V4PublicTensors:
    """Convert the canonical public-history JSON contract to actor tensors."""

    cfg = config or V4ActorConfig()
    _exact_keys(
        value,
        {
            "schemaVersion",
            "playerCount",
            "act",
            "actorRole",
            "revolution",
            "ownHandCounts",
            "publicPlayedCounts",
            "table",
            "playerTokens",
            "historyTokens",
            "memoryTraceVectors",
            "truncatedHistoryCount",
        },
        "public observation",
    )
    if cfg.global_features != 12 or cfg.rank_features != 6:
        raise ValueError("the V4 JSON tensorizer requires global=12 and rank=6")
    if cfg.player_features != 12 or cfg.history_features != 20:
        raise ValueError("the V4 JSON tensorizer requires player=12 and history=20")
    if cfg.memory_tokens != 4 or cfg.memory_features != 20:
        raise ValueError("the V4 JSON tensorizer requires four 20-value memory traces")
    schema_version = _integer(
        value.get("schemaVersion"), 1, cfg.observation_schema_version, "schemaVersion"
    )
    if schema_version != cfg.observation_schema_version:
        raise ValueError("public observation schema version mismatch")
    player_count = _integer(value.get("playerCount"), 4, cfg.max_players, "playerCount")
    act = _integer(value.get("act"), 1, 1_000_000, "act")
    actor_role = _integer(value.get("actorRole"), 0, 4, "actorRole")
    revolution = _integer(value.get("revolution"), 0, 2, "revolution")
    truncated = _integer(
        value.get("truncatedHistoryCount", 0),
        0,
        1_000_000_000,
        "truncatedHistoryCount",
    )
    table_value = value.get("table")
    if table_value is not None and not isinstance(table_value, Mapping):
        raise ValueError("table must be null or an object")

    global_vector = [
        (player_count - 4) / 6.0,
        math.tanh((act - 1) / 10.0),
        *_one_hot(actor_role, 5),
        *_one_hot(revolution, 3),
        math.tanh(truncated / max(1, cfg.max_history)),
        1.0 if table_value is not None else 0.0,
    ]
    own_counts = _sequence(value.get("ownHandCounts"), 13, "ownHandCounts")
    public_counts = _sequence(
        value.get("publicPlayedCounts"), 13, "publicPlayedCounts"
    )
    table_rank = -1
    table_natural = 0
    table_jokers = 0
    if table_value is not None:
        _exact_keys(
            table_value,
            {"actorOffset", "rank", "naturalCount", "jokerCount", "totalCount"},
            "table",
        )
        _integer(
            table_value.get("actorOffset"),
            0,
            player_count - 1,
            "table.actorOffset",
        )
        table_rank = _integer(table_value.get("rank"), 1, 13, "table.rank") - 1
        table_natural = _integer(
            table_value.get("naturalCount"), 0, 14, "table.naturalCount"
        )
        table_jokers = _integer(
            table_value.get("jokerCount"), 0, 2, "table.jokerCount"
        )
        total_count = _integer(
            table_value.get("totalCount"), 1, 14, "table.totalCount"
        )
        if table_natural + table_jokers != total_count:
            raise ValueError("table natural and joker counts must equal totalCount")
    rank_rows: list[list[float]] = []
    for rank_index in range(13):
        copies = _deck_copies(rank_index)
        own = _integer(own_counts[rank_index], 0, copies, f"ownHandCounts[{rank_index}]")
        played = _integer(
            public_counts[rank_index], 0, copies, f"publicPlayedCounts[{rank_index}]"
        )
        if own + played > copies:
            raise ValueError("own and publicly played counts exceed the deck")
        is_table_rank = rank_index == table_rank
        rank_rows.append(
            [
                own / copies,
                played / copies,
                1.0 if is_table_rank else 0.0,
                table_natural / 14.0 if is_table_rank else 0.0,
                table_jokers / 2.0 if is_table_rank else 0.0,
                (copies - own - played) / copies,
            ]
        )

    player_values = value.get("playerTokens")
    if not isinstance(player_values, list) or len(player_values) != player_count:
        raise ValueError("playerTokens must match playerCount")
    player_rows = torch.zeros(cfg.max_players, cfg.player_features)
    player_mask = torch.zeros(cfg.max_players, dtype=torch.bool)
    seen_offsets: set[int] = set()
    for index, player in enumerate(player_values):
        if not isinstance(player, Mapping):
            raise ValueError("each player token must be an object")
        _exact_keys(
            player,
            {
                "relativeOffset",
                "handCount",
                "finished",
                "passed",
                "self",
                "tableLeader",
                "role",
                "score",
            },
            "player token",
        )
        relative_offset = _integer(
            player.get("relativeOffset"), 0, player_count - 1, "player.relativeOffset"
        )
        if relative_offset in seen_offsets:
            raise ValueError("player relative offsets must be unique")
        seen_offsets.add(relative_offset)
        hand_count = _integer(player.get("handCount"), 0, 20, "player.handCount")
        role = _integer(player.get("role"), 0, 4, "player.role")
        flags = []
        for name in ("finished", "passed", "self", "tableLeader"):
            flag = player.get(name)
            flag_value = _integer(flag, 0, 1, f"player.{name}")
            flags.append(float(flag_value))
        score = _number(player.get("score"), "player.score")
        row = [
            relative_offset / max(1, player_count - 1),
            hand_count / 20.0,
            *flags,
            *_one_hot(role, 5),
            math.tanh(score / 10.0),
        ]
        player_rows[index] = torch.tensor(row)
        player_mask[index] = True
    if seen_offsets != set(range(player_count)):
        raise ValueError("player relative offsets must cover all players")

    memory_values = value.get("memoryTraceVectors")
    if not isinstance(memory_values, list) or len(memory_values) != cfg.memory_tokens:
        raise ValueError("memoryTraceVectors must contain four EMA vectors")
    memory_rows = torch.zeros(cfg.memory_tokens, cfg.memory_features)
    for trace_index, trace in enumerate(memory_values):
        trace_values = _sequence(
            trace, cfg.memory_features, f"memoryTraceVectors[{trace_index}]"
        )
        memory_rows[trace_index] = torch.tensor(
            [
                _number(item, f"memoryTraceVectors[{trace_index}][{item_index}]")
                for item_index, item in enumerate(trace_values)
            ],
            dtype=torch.float32,
        )

    history_values = value.get("historyTokens")
    if not isinstance(history_values, list):
        raise ValueError("historyTokens must be a list")
    history_values = history_values[-cfg.max_history :]
    history_rows = torch.zeros(cfg.max_history, cfg.history_features)
    history_mask = torch.zeros(cfg.max_history, dtype=torch.bool)
    previous_sequence: int | None = None
    for index, event in enumerate(history_values):
        if not isinstance(event, Mapping):
            raise ValueError("each history token must be an object")
        _exact_keys(
            event,
            {
                "sequence",
                "type",
                "actorOffset",
                "handCountBefore",
                "handCountAfter",
                "rank",
                "naturalCount",
                "jokerCount",
                "totalCount",
                "passReason",
                "clearReason",
                "nextLeaderOffset",
                "finishPlace",
            },
            "history token",
        )
        sequence = _integer(event.get("sequence"), 0, 1_000_000_000, "event.sequence")
        if previous_sequence is not None and sequence <= previous_sequence:
            raise ValueError("history token sequence must be strictly increasing")
        previous_sequence = sequence
        event_type = _integer(event.get("type"), 0, 3, "event.type")
        actor_offset = _integer(
            event.get("actorOffset"), 0, player_count - 1, "event.actorOffset"
        )
        before = _integer(
            event.get("handCountBefore"), 0, 20, "event.handCountBefore"
        )
        after = _integer(
            event.get("handCountAfter"), 0, 20, "event.handCountAfter"
        )
        rank = _integer(event.get("rank", 0), 0, 13, "event.rank")
        natural = _integer(
            event.get("naturalCount", 0), 0, 14, "event.naturalCount"
        )
        jokers = _integer(event.get("jokerCount", 0), 0, 2, "event.jokerCount")
        total = _integer(event.get("totalCount", 0), 0, 14, "event.totalCount")
        pass_reason = _integer(
            event.get("passReason", 0), 0, 4, "event.passReason"
        )
        clear_reason = _integer(
            event.get("clearReason", 0), 0, 3, "event.clearReason"
        )
        next_leader = _integer(
            event.get("nextLeaderOffset", -1),
            -1,
            player_count - 1,
            "event.nextLeaderOffset",
        )
        finish_place = _integer(
            event.get("finishPlace", 0), 0, player_count, "event.finishPlace"
        )
        pass_one_hot = (
            [0.0] * 4 if pass_reason == 0 else _one_hot(pass_reason - 1, 4)
        )
        clear_one_hot = (
            [0.0] * 3 if clear_reason == 0 else _one_hot(clear_reason - 1, 3)
        )
        row = [
            *_one_hot(event_type, 4),
            actor_offset / max(1, player_count - 1),
            before / 20.0,
            after / 20.0,
            rank / 13.0,
            natural / 14.0,
            jokers / 2.0,
            total / 14.0,
            *pass_one_hot,
            *clear_one_hot,
            0.0 if next_leader < 0 else next_leader / max(1, player_count - 1),
            finish_place / player_count,
        ]
        history_rows[index] = torch.tensor(row)
        history_mask[index] = True

    return V4PublicTensors(
        global_features=torch.tensor(global_vector, dtype=torch.float32),
        rank_features=torch.tensor(rank_rows, dtype=torch.float32),
        player_features=player_rows,
        player_mask=player_mask,
        memory_trace_features=memory_rows,
        history_features=history_rows,
        history_mask=history_mask,
    )


@dataclass(frozen=True)
class V4TrajectoryTensors:
    global_features: torch.Tensor
    rank_features: torch.Tensor
    player_features: torch.Tensor
    player_mask: torch.Tensor
    memory_trace_features: torch.Tensor
    history_features: torch.Tensor
    history_mask: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    expert_actions: torch.Tensor
    old_action_log_probs: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    valid_masks: torch.Tensor
    privileged_states: torch.Tensor


def _expected_shapes(
    trajectory_count: int,
    time_steps: int,
    actor: V4ActorConfig,
    critic: V4CriticConfig,
) -> dict[str, tuple[int, ...]]:
    prefix = (trajectory_count, time_steps)
    return {
        "global_features": (*prefix, actor.global_features),
        "rank_features": (*prefix, actor.rank_tokens, actor.rank_features),
        "player_features": (*prefix, actor.max_players, actor.player_features),
        "player_mask": (*prefix, actor.max_players),
        "memory_trace_features": (
            *prefix,
            actor.memory_tokens,
            actor.memory_features,
        ),
        "history_features": (*prefix, actor.max_history, actor.history_features),
        "history_mask": (*prefix, actor.max_history),
        "legal_masks": (*prefix, V4_ACTION_COUNT),
        "actions": prefix,
        "expert_actions": prefix,
        "old_action_log_probs": prefix,
        "advantages": prefix,
        "rewards": prefix,
        "dones": prefix,
        "valid_masks": prefix,
        "privileged_states": (*prefix, critic.privileged_features),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_FIXED_REWARD_TEXT_FIELDS = {
    "chipComponent": (
        "mean exact chip award(fixed candidate IDs) - "
        "mean exact chip award(fixed Normal IDs)"
    ),
    "pairwiseRate": (
        "candidate-before-Normal finish pairs / "
        "(candidate identity count * Normal identity count)"
    ),
    "pairwiseCenteredComponent": "pairwiseRate - 0.5",
    "actTotal": (
        "(chipComponent + pairwiseCoefficient * pairwiseCenteredComponent) / 5"
    ),
    "trajectoryReturn": (
        "sum of actTotal from trajectory act through act five; "
        "never divide by remaining horizon"
    ),
}
_FIXED_REWARD_FIELDS = {
    "version",
    "chipComponent",
    "pairwiseRate",
    "pairwiseCenteredComponent",
    "pairwiseCoefficient",
    "actTotal",
    "trajectoryReturn",
    "rawComponentsSeparatelyBoundForAblation",
}
_FIXED_BEHAVIOR_FIELDS = {
    "behaviorPolicyContract",
    "behaviorPolicyContractVersion",
    "rawMaskedSoftmaxExactBinding",
    "initialOldCurrentRatioMathematicallyOneForFrozenActor",
    "initialOldCurrentLogProbabilityAbsoluteTolerance",
    "fixedPpoActorAutocastDisabled",
    "requiresFullDatasetInitialPolicyReproductionAudit",
    "dropoutDisabled",
    "temperature",
    "epsilonFloorPerLegalAction",
}


def _canonical_contract_sha256(fields_value: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(fields_value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_canonical_fixed_collection_namespace(value: object) -> bool:
    """Mirror the fixed collector's safe, filename-stable namespace domain."""

    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value[0]
        in _FIXED_COLLECTION_NAMESPACE_CHARACTERS - {".", "_", "-"}
        and all(
            character in _FIXED_COLLECTION_NAMESPACE_CHARACTERS
            for character in value
        )
    )


def _fixed_environment_seed(
    run_namespace: str,
    seed_base: int,
    player_count: int,
    match_index: int,
) -> int:
    payload = json.dumps(
        [
            run_namespace,
            seed_base,
            "fixed-match-environment",
            player_count,
            match_index,
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return value or 1


def canonical_fixed_ppo_reward_contract(
    value: Mapping[str, object],
    *,
    require_exact_fields: bool = True,
) -> tuple[str, dict[str, object]]:
    """Return the opaque ID and self-verifying canonical reward record."""

    if not isinstance(value, Mapping):
        raise ValueError("fixed-match reward contract must be an object")
    if require_exact_fields and set(value) != _FIXED_REWARD_FIELDS:
        raise ValueError("fixed-match reward contract fields are non-canonical")
    coefficient_value = value.get("pairwiseCoefficient")
    if (
        isinstance(coefficient_value, bool)
        or not isinstance(coefficient_value, (int, float))
        or not math.isfinite(float(coefficient_value))
        or not 0.0 <= float(coefficient_value) <= 1.0
    ):
        raise ValueError("fixed-match pairwise coefficient is non-canonical")
    coefficient = float(coefficient_value)
    if (
        isinstance(value.get("version"), bool)
        or not isinstance(value.get("version"), int)
        or value.get("rawComponentsSeparatelyBoundForAblation") is not True
    ):
        raise ValueError("fixed-match reward version/ablation binding is non-canonical")
    canonical_fields: dict[str, object] = {
        "version": 1,
        **_FIXED_REWARD_TEXT_FIELDS,
        "pairwiseCoefficient": coefficient,
        "rawComponentsSeparatelyBoundForAblation": True,
    }
    # Preserve the collector field order only for readability; the digest is
    # sort-key canonical and therefore independent of mapping order.
    canonical_fields = {
        "version": canonical_fields["version"],
        "chipComponent": canonical_fields["chipComponent"],
        "pairwiseRate": canonical_fields["pairwiseRate"],
        "pairwiseCenteredComponent": canonical_fields["pairwiseCenteredComponent"],
        "pairwiseCoefficient": canonical_fields["pairwiseCoefficient"],
        "actTotal": canonical_fields["actTotal"],
        "trajectoryReturn": canonical_fields["trajectoryReturn"],
        "rawComponentsSeparatelyBoundForAblation": canonical_fields[
            "rawComponentsSeparatelyBoundForAblation"
        ],
    }
    if any(value.get(name) != expected for name, expected in canonical_fields.items()):
        raise ValueError("fixed-match reward formula fields are non-canonical")
    digest = _canonical_contract_sha256(canonical_fields)
    coefficient_text = json.dumps(coefficient, allow_nan=False, separators=(",", ":"))
    opaque_id = (
        f"{V4_FIXED_PPO_REWARD_CONTRACT_ID}:lambda={coefficient_text}:"
        f"sha256={digest}"
    )
    return opaque_id, {
        "opaqueId": opaque_id,
        "canonicalSha256": digest,
        "canonicalFields": canonical_fields,
    }


def canonical_fixed_ppo_behavior_policy_contract(
    value: Mapping[str, object],
    *,
    require_exact_fields: bool = False,
) -> tuple[str, dict[str, object]]:
    """Return the canonical raw-Actor behavior policy binding."""

    if not isinstance(value, Mapping):
        raise ValueError("fixed-match behavior policy contract must be an object")
    if require_exact_fields and set(value) != _FIXED_BEHAVIOR_FIELDS:
        raise ValueError("fixed-match behavior policy fields are non-canonical")
    if (
        isinstance(value.get("behaviorPolicyContractVersion"), bool)
        or not isinstance(value.get("behaviorPolicyContractVersion"), int)
        or isinstance(value.get("temperature"), bool)
        or not isinstance(value.get("temperature"), (int, float))
        or isinstance(value.get("epsilonFloorPerLegalAction"), bool)
        or not isinstance(value.get("epsilonFloorPerLegalAction"), (int, float))
        or value.get("rawMaskedSoftmaxExactBinding") is not True
        or value.get("initialOldCurrentRatioMathematicallyOneForFrozenActor")
        is not True
        or isinstance(
            value.get("initialOldCurrentLogProbabilityAbsoluteTolerance"), bool
        )
        or not isinstance(
            value.get("initialOldCurrentLogProbabilityAbsoluteTolerance"),
            (int, float),
        )
        or float(value.get("initialOldCurrentLogProbabilityAbsoluteTolerance"))
        != 2.0e-5
        or value.get("fixedPpoActorAutocastDisabled") is not True
        or value.get("requiresFullDatasetInitialPolicyReproductionAudit")
        is not True
        or value.get("dropoutDisabled") is not True
    ):
        raise ValueError("fixed-match behavior policy field types are non-canonical")
    canonical_fields: dict[str, object] = {
        "behaviorPolicyContract": V4_FIXED_PPO_BEHAVIOR_POLICY_CONTRACT_ID,
        "behaviorPolicyContractVersion": 1,
        "rawMaskedSoftmaxExactBinding": True,
        "initialOldCurrentRatioMathematicallyOneForFrozenActor": True,
        "initialOldCurrentLogProbabilityAbsoluteTolerance": 2.0e-5,
        "fixedPpoActorAutocastDisabled": True,
        "requiresFullDatasetInitialPolicyReproductionAudit": True,
        "dropoutDisabled": True,
        "temperature": 1.0,
        "epsilonFloorPerLegalAction": 0.0,
    }
    if any(value.get(name) != expected for name, expected in canonical_fields.items()):
        raise ValueError(
            "fixed-match behavior policy must be raw masked Actor softmax "
            "at temperature=1.0 and epsilon=0.0"
        )
    digest = _canonical_contract_sha256(canonical_fields)
    opaque_id = (
        f"{V4_FIXED_PPO_BEHAVIOR_POLICY_CONTRACT_ID}:sha256={digest}"
    )
    return opaque_id, {
        "opaqueId": opaque_id,
        "canonicalSha256": digest,
        "canonicalFields": canonical_fields,
    }


_FIXED_COLLECTION_PLAN_V1_FIELDS = {
    "version",
    "runNamespace",
    "seedBase",
    "matchCounts",
    "matchStart",
    "matchShardCount",
    "completeUnshardedLearnerAssignmentSha256",
    "actorCheckpointSha256",
    "bundleManifestSha256",
    "rewardContract",
    "behaviorPolicyContract",
    "sourceHashesSha256",
}
_FIXED_COLLECTION_PLAN_V2_FIELDS = (
    _FIXED_COLLECTION_PLAN_V1_FIELDS
    | {
        "shardBackendMap",
        "crossBackendCalibrationReportSha256",
    }
)


def _canonical_shard_backend_map(
    value: object,
    shard_count: int,
) -> dict[str, str]:
    """Canonicalize a complete numeric shard-index -> cpu/cuda map."""

    if not isinstance(value, Mapping) or len(value) != shard_count:
        raise ValueError("fixed collection plan shard backend map is invalid")
    parsed: dict[int, str] = {}
    for key, backend in value.items():
        try:
            shard_index = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "fixed collection plan shard backend index is invalid"
            ) from error
        if (
            not isinstance(key, str)
            or key != str(shard_index)
            or not 0 <= shard_index < shard_count
            or shard_index in parsed
            or backend not in {"cpu", "cuda"}
        ):
            raise ValueError(
                "fixed collection plan shard backend map is non-canonical"
            )
        parsed[shard_index] = str(backend)
    if set(parsed) != set(range(shard_count)) or set(parsed.values()) != {
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "fixed collection plan v2 requires every shard and both cpu/cuda backends"
        )
    return {str(index): parsed[index] for index in range(shard_count)}


def fixed_match_shard_identity_sha256(
    shard: Mapping[str, object],
    preparation_format: str = V4_FIXED_MATCH_PPO_PREPARATION_FORMAT,
) -> str:
    """Recompute one direct shard identity from its declared canonical fields.

    The v1 payload is byte-for-byte compatible with the original collector.
    V2 adds an explicit generation marker, the complete backend plan, and the
    immutable CPU/CUDA calibration report hash.  Keeping this derivation in the
    loader prevents metadata-only removal of the v2 fields from being accepted
    while the original v2 identity remains attached.
    """

    if not isinstance(shard, Mapping):
        raise ValueError("fixed-match shard identity metadata is missing")
    if preparation_format != V4_FIXED_MATCH_PPO_PREPARATION_FORMAT:
        raise ValueError("fixed-match shard identity preparation format is invalid")
    namespace = shard.get("runNamespace")
    seed_base = shard.get("seedBase")
    match_start = shard.get("matchStart")
    shard_count = shard.get("matchShardCount")
    shard_index = shard.get("matchShardIndex")
    match_counts_value = shard.get("matchCounts")
    if (
        not _is_canonical_fixed_collection_namespace(namespace)
        or isinstance(seed_base, bool)
        or not isinstance(seed_base, int)
        or not 0 <= seed_base <= 0xFFFF_FFFF
        or isinstance(match_start, bool)
        or not isinstance(match_start, int)
        or match_start < 0
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
        or not isinstance(match_counts_value, Mapping)
        or not match_counts_value
    ):
        raise ValueError("fixed-match shard identity fields are invalid")
    parsed_match_counts: dict[int, int] = {}
    for key, raw_count in match_counts_value.items():
        try:
            player_count = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("fixed-match shard identity match counts are invalid") from error
        if (
            not isinstance(key, str)
            or key != str(player_count)
            or not 4 <= player_count <= 10
            or player_count in parsed_match_counts
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 1
        ):
            raise ValueError("fixed-match shard identity match counts are invalid")
        parsed_match_counts[player_count] = raw_count
    ordered_match_counts = [
        [player_count, parsed_match_counts[player_count]]
        for player_count in sorted(parsed_match_counts)
    ]

    version = shard.get("collectionPlanVersion", 1)
    if version == 1:
        if any(
            name in shard
            for name in (
                "collectionPlanVersion",
                "shardBackendMap",
                "crossBackendCalibrationReportSha256",
            )
        ):
            raise ValueError("fixed-match shard v1 identity carries v2 fields")
        payload: list[object] = [
            preparation_format,
            namespace,
            seed_base,
            ordered_match_counts,
            match_start,
            shard_count,
            shard_index,
        ]
    elif version == 2:
        backend_map = _canonical_shard_backend_map(
            shard.get("shardBackendMap"), shard_count
        )
        calibration_sha256 = shard.get("crossBackendCalibrationReportSha256")
        if not _is_sha256(calibration_sha256):
            raise ValueError(
                "fixed-match shard v2 calibration report hash is invalid"
            )
        payload = [
            preparation_format,
            namespace,
            seed_base,
            ordered_match_counts,
            match_start,
            shard_count,
            shard_index,
            2,
            backend_map,
            calibration_sha256,
        ]
    else:
        raise ValueError("fixed-match shard identity version is unsupported")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fixed_collection_plan_sha256(value: object) -> str:
    """Extract the SHA from either canonical fixed collection plan generation."""

    if not isinstance(value, str):
        raise ValueError("fixed collection plan ID is non-canonical")
    for contract_id in (
        V4_FIXED_COLLECTION_PLAN_ID,
        V4_FIXED_MIXED_BACKEND_COLLECTION_PLAN_ID,
    ):
        prefix = f"{contract_id}:sha256="
        if value.startswith(prefix):
            digest = value[len(prefix) :]
            if _is_sha256(digest):
                return digest
            break
    raise ValueError("fixed collection plan ID is non-canonical")


def _canonical_fixed_collection_plan_fields(
    value: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("fixed collection plan fields are non-canonical")
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("fixed collection plan integer field is invalid")
    expected_fields = (
        _FIXED_COLLECTION_PLAN_V1_FIELDS
        if version == 1
        else _FIXED_COLLECTION_PLAN_V2_FIELDS
        if version == 2
        else None
    )
    if expected_fields is None or set(value) != expected_fields:
        raise ValueError("fixed collection plan fields are non-canonical")
    match_counts_value = value.get("matchCounts")
    if not isinstance(match_counts_value, Mapping) or not match_counts_value:
        raise ValueError("fixed collection plan match counts are invalid")
    parsed_match_counts: dict[int, int] = {}
    for key, raw_count in match_counts_value.items():
        try:
            player_count = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("fixed collection plan player count is invalid") from error
        if (
            not isinstance(key, str)
            or key != str(player_count)
            or not 4 <= player_count <= 10
            or player_count in parsed_match_counts
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 1
        ):
            raise ValueError("fixed collection plan match counts are non-canonical")
        parsed_match_counts[player_count] = raw_count
    # JSON ``sort_keys=True`` orders numeric-looking object keys
    # lexicographically ("10", "4", ...).  Input mapping order therefore
    # cannot be part of this provenance contract.  Rebuild the canonical
    # representation in numeric player-count order after validating every
    # key/value and rejecting numeric aliases or duplicate player counts.
    match_counts = {
        str(player_count): parsed_match_counts[player_count]
        for player_count in sorted(parsed_match_counts)
    }
    integer_fields = ("version", "seedBase", "matchStart", "matchShardCount")
    if any(
        isinstance(value.get(name), bool) or not isinstance(value.get(name), int)
        for name in integer_fields
    ):
        raise ValueError("fixed collection plan integer field is invalid")
    if (
        int(value["seedBase"]) < 0
        or int(value["seedBase"]) > 0xFFFF_FFFF
        or int(value["matchStart"]) < 0
        or int(value["matchShardCount"]) < 1
        or not _is_canonical_fixed_collection_namespace(
            value.get("runNamespace")
        )
        or not _is_sha256(value.get("completeUnshardedLearnerAssignmentSha256"))
        or not _is_sha256(value.get("actorCheckpointSha256"))
        or not _is_sha256(value.get("bundleManifestSha256"))
        or not _is_sha256(value.get("sourceHashesSha256"))
        or not isinstance(value.get("rewardContract"), str)
        or not isinstance(value.get("behaviorPolicyContract"), str)
        or (
            version == 2
            and not _is_sha256(
                value.get("crossBackendCalibrationReportSha256")
            )
        )
    ):
        raise ValueError("fixed collection plan provenance is invalid")
    canonical_fields: dict[str, object] = {
        "version": version,
        "runNamespace": str(value["runNamespace"]),
        "seedBase": int(value["seedBase"]),
        "matchCounts": match_counts,
        "matchStart": int(value["matchStart"]),
        "matchShardCount": int(value["matchShardCount"]),
        "completeUnshardedLearnerAssignmentSha256": str(
            value["completeUnshardedLearnerAssignmentSha256"]
        ),
        "actorCheckpointSha256": str(value["actorCheckpointSha256"]),
        "bundleManifestSha256": str(value["bundleManifestSha256"]),
        "rewardContract": str(value["rewardContract"]),
        "behaviorPolicyContract": str(value["behaviorPolicyContract"]),
        "sourceHashesSha256": str(value["sourceHashesSha256"]),
    }
    if version == 2:
        canonical_fields["shardBackendMap"] = _canonical_shard_backend_map(
            value.get("shardBackendMap"),
            int(value["matchShardCount"]),
        )
        canonical_fields["crossBackendCalibrationReportSha256"] = str(
            value["crossBackendCalibrationReportSha256"]
        )
    if dict(value) != canonical_fields:
        raise ValueError("fixed collection plan canonicalization drifted")
    digest = _canonical_contract_sha256(canonical_fields)
    contract_id = (
        V4_FIXED_COLLECTION_PLAN_ID
        if version == 1
        else V4_FIXED_MIXED_BACKEND_COLLECTION_PLAN_ID
    )
    return f"{contract_id}:sha256={digest}", canonical_fields


def canonical_fixed_collection_plan(
    metadata: Mapping[str, object],
    reward_contract: str,
    behavior_policy_contract: str,
) -> tuple[str, dict[str, object]]:
    """Build one direct shard's immutable fixed collection plan record."""

    shard = metadata.get("shard")
    model = metadata.get("modelBinding")
    sources = metadata.get("sourceHashes")
    if (
        not isinstance(shard, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(sources, Mapping)
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in sources.items()
        )
    ):
        raise ValueError("fixed collection plan source metadata is invalid")
    source_hashes = {
        name: str(sources[name]) for name in sorted(sources)
    }
    match_counts_value = shard.get("matchCounts")
    if not isinstance(match_counts_value, Mapping):
        raise ValueError("fixed collection plan match counts are missing")
    match_counts = {
        str(key): value
        for key, value in sorted(
            match_counts_value.items(), key=lambda item: int(item[0])
        )
    }
    collection_plan_version = shard.get("collectionPlanVersion", 1)
    if (
        isinstance(collection_plan_version, bool)
        or not isinstance(collection_plan_version, int)
        or collection_plan_version not in {1, 2}
        or (
            collection_plan_version == 1
            and (
                "collectionPlanVersion" in shard
                or "shardBackendMap" in shard
            )
        )
        or (
            collection_plan_version == 2
            and "shardBackendMap" not in shard
        )
    ):
        raise ValueError("fixed collection plan version/backend metadata is invalid")
    fields_value: dict[str, object] = {
        "version": collection_plan_version,
        "runNamespace": shard.get("runNamespace"),
        "seedBase": shard.get("seedBase"),
        "matchCounts": match_counts,
        "matchStart": shard.get("matchStart"),
        "matchShardCount": shard.get("matchShardCount"),
        "completeUnshardedLearnerAssignmentSha256": shard.get(
            "completeUnshardedLearnerAssignmentSha256"
        ),
        "actorCheckpointSha256": model.get("actorCheckpointSha256"),
        "bundleManifestSha256": model.get("bundleManifestSha256"),
        "rewardContract": reward_contract,
        "behaviorPolicyContract": behavior_policy_contract,
        "sourceHashesSha256": _canonical_contract_sha256(source_hashes),
    }
    if collection_plan_version == 2:
        fields_value["shardBackendMap"] = shard.get("shardBackendMap")
        fields_value["crossBackendCalibrationReportSha256"] = shard.get(
            "crossBackendCalibrationReportSha256"
        )
    opaque_id, canonical_fields = _canonical_fixed_collection_plan_fields(
        fields_value
    )
    shard_index = shard.get("matchShardIndex")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < int(canonical_fields["matchShardCount"])
    ):
        raise ValueError("fixed collection plan shard index is invalid")
    return opaque_id, {
        "opaqueId": opaque_id,
        "canonicalSha256": opaque_id.rsplit("=", 1)[1],
        "canonicalFields": canonical_fields,
        "coveredShardIndices": [shard_index],
    }


def complete_fixed_collection_plan_record(
    base_record: Mapping[str, object],
    covered_shard_indices: Sequence[int],
) -> dict[str, object]:
    fields_value = base_record.get("canonicalFields")
    if not isinstance(fields_value, Mapping):
        raise ValueError("fixed collection plan canonical fields are missing")
    opaque_id, canonical_fields = _canonical_fixed_collection_plan_fields(fields_value)
    canonical_sha = opaque_id.rsplit("=", 1)[1]
    shard_count = int(canonical_fields["matchShardCount"])
    covered = list(covered_shard_indices)
    if covered != list(range(shard_count)):
        raise ValueError(
            "fixed collection plan must cover every shard index exactly once"
        )
    coverage_fields = {
        "opaqueId": opaque_id,
        "coveredShardIndices": covered,
    }
    expected_base = {
        "opaqueId": opaque_id,
        "canonicalSha256": canonical_sha,
        "canonicalFields": canonical_fields,
    }
    if any(base_record.get(key) != value for key, value in expected_base.items()):
        raise ValueError("fixed collection plan hash binding is invalid")
    return {
        **expected_base,
        "coveredShardIndices": covered,
        "coverageSha256": _canonical_contract_sha256(coverage_fields),
    }


def validate_merged_fixed_collection_plans(
    contract: Mapping[str, object],
    *,
    has_fixed_source: bool,
) -> tuple[dict[str, object], ...]:
    raw_plans = contract.get("fixedCollectionPlans")
    raw_coverage_sha = contract.get("fixedCollectionPlanCoverageSha256")
    if not has_fixed_source and raw_plans is None and raw_coverage_sha is None:
        return ()
    if not isinstance(raw_plans, list):
        raise ValueError("merged fixed collection plans are invalid")
    plans: list[dict[str, object]] = []
    previous_id = ""
    for raw_plan in raw_plans:
        if not isinstance(raw_plan, Mapping):
            raise ValueError("merged fixed collection plan record is invalid")
        covered = raw_plan.get("coveredShardIndices")
        if not isinstance(covered, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in covered
        ):
            raise ValueError("merged fixed collection plan coverage is invalid")
        expected = complete_fixed_collection_plan_record(raw_plan, covered)
        if dict(raw_plan) != expected or expected["opaqueId"] <= previous_id:
            raise ValueError("merged fixed collection plan ordering/hash is invalid")
        previous_id = str(expected["opaqueId"])
        plans.append(expected)
    if has_fixed_source != bool(plans):
        raise ValueError("fixed source and fixed collection plan presence disagree")
    expected_coverage_sha = _canonical_contract_sha256(
        {"fixedCollectionPlans": plans}
    )
    if raw_coverage_sha != expected_coverage_sha:
        raise ValueError("merged fixed collection plan coverage SHA is invalid")
    return tuple(plans)


def _validate_merged_contract_records(
    contract: Mapping[str, object],
    *,
    has_fixed_source: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate transitive fixed reward/behavior records in merged metadata."""

    specifications = (
        (
            "ppoRewardContracts",
            "ppoRewardContractRecords",
            canonical_fixed_ppo_reward_contract,
        ),
        (
            "ppoBehaviorPolicyContracts",
            "ppoBehaviorPolicyContractRecords",
            canonical_fixed_ppo_behavior_policy_contract,
        ),
    )
    parsed: list[tuple[str, ...]] = []
    for ids_name, records_name, canonicalizer in specifications:
        raw_ids = contract.get(ids_name)
        raw_records = contract.get(records_name)
        if not has_fixed_source and raw_ids is None and raw_records is None:
            # Pre-fixed legacy merged artifacts remain byte/API compatible.
            parsed.append(())
            continue
        if (
            not isinstance(raw_ids, list)
            or raw_ids != sorted(set(raw_ids))
            or any(not isinstance(value, str) or not value for value in raw_ids)
            or not isinstance(raw_records, list)
        ):
            raise ValueError(f"merged {ids_name} provenance is invalid")
        expected_records: dict[str, dict[str, object]] = {}
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"merged {records_name} entry is invalid")
            fields_value = raw_record.get("canonicalFields")
            if not isinstance(fields_value, Mapping):
                raise ValueError(f"merged {records_name} canonical fields are invalid")
            opaque_id, expected = canonicalizer(
                fields_value,
                require_exact_fields=True,
            )
            if dict(raw_record) != expected or opaque_id in expected_records:
                raise ValueError(f"merged {records_name} hash binding is invalid")
            expected_records[opaque_id] = expected
        if sorted(expected_records) != raw_ids:
            raise ValueError(f"merged {ids_name} record set is inconsistent")
        if has_fixed_source and len(raw_ids) != 1:
            raise ValueError(
                "all fixed-match PPO inputs must share exactly one canonical "
                f"{ids_name} binding"
            )
        if not has_fixed_source and raw_ids:
            raise ValueError(f"legacy-only merged PPO cannot carry {ids_name}")
        parsed.append(tuple(raw_ids))
    return parsed[0], parsed[1]


def validate_merged_ppo_provenance(
    contract: Mapping[str, object],
    actor_hashes: Sequence[str],
) -> tuple[
    tuple[str, ...],
    bool,
    bool,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Normalize and fail-closed validate merged transitive PPO provenance."""

    raw_source_contracts = contract.get("ppoSourceContracts")
    if raw_source_contracts is None:
        source_contracts = (V4_LEGACY_PPO_SOURCE_CONTRACT,) if actor_hashes else ()
    elif (
        not isinstance(raw_source_contracts, list)
        or raw_source_contracts != sorted(set(raw_source_contracts))
        or any(
            value not in {
                V4_LEGACY_PPO_SOURCE_CONTRACT,
                V4_FIXED_PPO_SOURCE_CONTRACT,
            }
            for value in raw_source_contracts
        )
    ):
        raise ValueError("merged V4 PPO source contracts are invalid")
    else:
        source_contracts = tuple(str(value) for value in raw_source_contracts)
    has_fixed = V4_FIXED_PPO_SOURCE_CONTRACT in source_contracts
    expected_balanced = source_contracts == (V4_FIXED_PPO_SOURCE_CONTRACT,)
    raw_balanced = contract.get("requiresPlayerCountBalancedLoss")
    if raw_balanced is None and not has_fixed:
        requires_balanced = False
    elif not isinstance(raw_balanced, bool) or raw_balanced != expected_balanced:
        raise ValueError("merged V4 player-count-balanced loss requirement is invalid")
    else:
        requires_balanced = raw_balanced
    raw_qboost = contract.get("requiresQBoostCoefficientZero")
    if raw_qboost is None and not has_fixed:
        requires_qboost_zero = False
    elif not isinstance(raw_qboost, bool) or raw_qboost != has_fixed:
        raise ValueError("merged V4 q-boost requirement binding is invalid")
    else:
        requires_qboost_zero = raw_qboost
    reward_contracts, behavior_contracts = _validate_merged_contract_records(
        contract,
        has_fixed_source=has_fixed,
    )
    collection_plans = validate_merged_fixed_collection_plans(
        contract,
        has_fixed_source=has_fixed,
    )
    for plan in collection_plans:
        fields_value = plan["canonicalFields"]
        assert isinstance(fields_value, Mapping)
        if (
            fields_value.get("rewardContract") not in reward_contracts
            or fields_value.get("behaviorPolicyContract") not in behavior_contracts
            or fields_value.get("actorCheckpointSha256") not in actor_hashes
        ):
            raise ValueError(
                "merged fixed collection plan disagrees with PPO contract bindings"
            )
    return (
        source_contracts,
        requires_balanced,
        requires_qboost_zero,
        reward_contracts,
        behavior_contracts,
    )


def _validate_loss_masks(
    eligibility: V4LossEligibility,
    valid_masks: torch.Tensor,
) -> V4LossEligibility:
    expected_shape = valid_masks.shape
    for name, mask in eligibility.masks().items():
        if mask.dtype != torch.bool or mask.shape != expected_shape:
            raise ValueError(f"{name} must be bool [trajectory, time]")
        if (mask & ~valid_masks).any():
            raise ValueError(f"{name} marks an invalid suffix as loss-eligible")
    if not torch.equal(eligibility.behavior_cloning, valid_masks):
        raise ValueError(
            "every valid V4 sample must retain its exact Normal BC label"
        )
    if not torch.equal(eligibility.ppo, eligibility.critic):
        raise ValueError(
            "V4 PPO and critic eligibility must select the same collector samples"
        )
    if (eligibility.ppo & ~eligibility.behavior_cloning).any():
        raise ValueError("V4 PPO samples must also have an exact Normal BC label")
    ppo_trajectory = eligibility.ppo.any(dim=1)
    if not torch.equal(
        eligibility.ppo[ppo_trajectory], valid_masks[ppo_trajectory]
    ):
        raise ValueError(
            "PPO/critic eligibility must cover complete actor trajectories"
        )
    if eligibility.preparation_version != 1:
        raise ValueError("unsupported V4 loss preparation version")
    if any(not _is_sha256(value) for value in eligibility.behavior_actor_sha256s):
        raise ValueError("V4 PPO behavior Actor binding contains an invalid SHA-256")
    if tuple(sorted(set(eligibility.behavior_actor_sha256s))) != (
        eligibility.behavior_actor_sha256s
    ):
        raise ValueError("V4 PPO behavior Actor hashes must be sorted and unique")
    if bool(eligibility.ppo.any()) != bool(eligibility.behavior_actor_sha256s):
        raise ValueError(
            "V4 PPO eligibility and behavior Actor bindings must be present together"
        )
    allowed_sources = {
        V4_LEGACY_PPO_SOURCE_CONTRACT,
        V4_FIXED_PPO_SOURCE_CONTRACT,
    }
    has_fixed_source = V4_FIXED_PPO_SOURCE_CONTRACT in eligibility.ppo_source_contracts
    if (
        tuple(sorted(set(eligibility.ppo_source_contracts)))
        != eligibility.ppo_source_contracts
        or any(value not in allowed_sources for value in eligibility.ppo_source_contracts)
        or bool(eligibility.ppo.any()) != bool(eligibility.ppo_source_contracts)
        or not isinstance(eligibility.requires_player_count_balanced_loss, bool)
        or eligibility.requires_player_count_balanced_loss
        != (eligibility.ppo_source_contracts == (V4_FIXED_PPO_SOURCE_CONTRACT,))
        or not isinstance(eligibility.requires_qboost_coefficient_zero, bool)
        or eligibility.requires_qboost_coefficient_zero != has_fixed_source
        or (
            has_fixed_source
            and (
                len(eligibility.ppo_reward_contracts) != 1
                or len(eligibility.ppo_behavior_policy_contracts) != 1
                or not eligibility.fixed_collection_plan_ids
            )
        )
        or (
            not has_fixed_source
            and (
                eligibility.ppo_reward_contracts
                or eligibility.ppo_behavior_policy_contracts
                or eligibility.fixed_collection_plan_ids
            )
        )
        or tuple(sorted(set(eligibility.ppo_reward_contracts)))
        != eligibility.ppo_reward_contracts
        or tuple(sorted(set(eligibility.ppo_behavior_policy_contracts)))
        != eligibility.ppo_behavior_policy_contracts
        or tuple(sorted(set(eligibility.fixed_collection_plan_ids)))
        != eligibility.fixed_collection_plan_ids
    ):
        raise ValueError("V4 PPO source/loss provenance is invalid")
    if has_fixed_source:
        reward_id = eligibility.ppo_reward_contracts[0]
        behavior_id = eligibility.ppo_behavior_policy_contracts[0]
        reward_prefix = f"{V4_FIXED_PPO_REWARD_CONTRACT_ID}:lambda="
        if not reward_id.startswith(reward_prefix) or ":sha256=" not in reward_id:
            raise ValueError("V4 PPO canonical reward contract ID is invalid")
        coefficient_text, reward_sha = reward_id[len(reward_prefix):].split(
            ":sha256=", 1
        )
        try:
            coefficient = float(coefficient_text)
        except ValueError as error:
            raise ValueError("V4 PPO canonical reward coefficient is invalid") from error
        if not math.isfinite(coefficient):
            raise ValueError("V4 PPO canonical reward coefficient is invalid")
        canonical_value = json.dumps(
            coefficient, allow_nan=False, separators=(",", ":")
        )
        expected_reward_id, _ = canonical_fixed_ppo_reward_contract(
            {
                "version": 1,
                **_FIXED_REWARD_TEXT_FIELDS,
                "pairwiseCoefficient": coefficient,
                "rawComponentsSeparatelyBoundForAblation": True,
            }
        )
        if (
            coefficient_text != canonical_value
            or not _is_sha256(reward_sha)
            or reward_id != expected_reward_id
        ):
            raise ValueError("V4 PPO canonical reward contract hash is invalid")
        expected_behavior_id, _ = canonical_fixed_ppo_behavior_policy_contract(
            {
                "behaviorPolicyContract": V4_FIXED_PPO_BEHAVIOR_POLICY_CONTRACT_ID,
                "behaviorPolicyContractVersion": 1,
                "rawMaskedSoftmaxExactBinding": True,
                "initialOldCurrentRatioMathematicallyOneForFrozenActor": True,
                "initialOldCurrentLogProbabilityAbsoluteTolerance": 2.0e-5,
                "fixedPpoActorAutocastDisabled": True,
                "requiresFullDatasetInitialPolicyReproductionAudit": True,
                "dropoutDisabled": True,
                "temperature": 1.0,
                "epsilonFloorPerLegalAction": 0.0,
            },
            require_exact_fields=True,
        )
        if behavior_id != expected_behavior_id:
            raise ValueError("V4 PPO canonical behavior policy contract is invalid")
        try:
            for value in eligibility.fixed_collection_plan_ids:
                fixed_collection_plan_sha256(value)
        except ValueError as error:
            raise ValueError(
                "V4 PPO fixed collection plan ID is invalid"
            ) from error
    return eligibility


def _loss_contract_fingerprint(eligibility: V4LossEligibility) -> str:
    digest = hashlib.sha256()
    digest.update(eligibility.preparation_format.encode("utf-8"))
    digest.update(str(eligibility.preparation_version).encode("ascii"))
    for name, mask in eligibility.masks().items():
        tensor = mask.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    for actor_sha256 in eligibility.behavior_actor_sha256s:
        digest.update(actor_sha256.encode("ascii"))
    # Preserve every pre-fixed-match legacy fingerprint byte-for-byte.  Only
    # the new/mixed provenance domains extend the digest.
    if (
        eligibility.ppo_source_contracts
        not in {(), (V4_LEGACY_PPO_SOURCE_CONTRACT,)}
        or eligibility.requires_player_count_balanced_loss
        or eligibility.requires_qboost_coefficient_zero
        or eligibility.ppo_reward_contracts
        or eligibility.ppo_behavior_policy_contracts
        or eligibility.fixed_collection_plan_ids
    ):
        digest.update(b"\x00dalmuti-v4-ppo-source-contracts-v1\x00")
        for source_contract in eligibility.ppo_source_contracts:
            digest.update(source_contract.encode("ascii"))
            digest.update(b"\x00")
        digest.update(str(int(eligibility.requires_player_count_balanced_loss)).encode("ascii"))
        digest.update(str(int(eligibility.requires_qboost_coefficient_zero)).encode("ascii"))
        digest.update(b"\x00dalmuti-v4-ppo-reward-contracts-v1\x00")
        for reward_contract in eligibility.ppo_reward_contracts:
            digest.update(reward_contract.encode("ascii"))
            digest.update(b"\x00")
        digest.update(b"\x00dalmuti-v4-ppo-behavior-policy-contracts-v1\x00")
        for behavior_contract in eligibility.ppo_behavior_policy_contracts:
            digest.update(behavior_contract.encode("ascii"))
            digest.update(b"\x00")
        digest.update(b"\x00dalmuti-v4-fixed-collection-plans-v1\x00")
        for plan_id in eligibility.fixed_collection_plan_ids:
            digest.update(plan_id.encode("ascii"))
            digest.update(b"\x00")
    return digest.hexdigest()


def _csv_ints(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a canonical comma-separated integer list")
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{label} contains a non-integer") from error
    if ",".join(str(item) for item in parsed) != value:
        raise ValueError(f"{label} is not canonically encoded")
    return parsed


def _evaluation_candidate_seats(player_count: int, match_index: int) -> tuple[int, ...]:
    lower = player_count // 2
    count = lower if player_count % 2 == 0 or match_index % 2 == 1 else lower + 1
    extras_before = (match_index + 1) // 2 if player_count % 2 else 0
    start = (match_index * lower + extras_before) % player_count
    return tuple((start + offset) % player_count for offset in range(count))


def _role_index(position: int, player_count: int) -> int:
    if position == 0:
        return 0
    if position == 1:
        return 1
    if position == player_count - 2:
        return 3
    if position == player_count - 1:
        return 4
    return 2


def _fixed_group_reward_from_outcome(
    finish_order: Sequence[int],
    candidate_ids: Sequence[int],
) -> tuple[list[int], float, float, float, int, int, float, float]:
    """Canonical evaluator group D/Q components from one raw act outcome."""

    player_count = len(finish_order)
    if (
        not 4 <= player_count <= 10
        or len(set(finish_order)) != player_count
        or set(finish_order) != set(range(player_count))
    ):
        raise ValueError("fixed-match finish order is not a physical-ID permutation")
    candidate_set = set(candidate_ids)
    if (
        len(candidate_set) != len(candidate_ids)
        or not candidate_set
        or not candidate_set < set(range(player_count))
    ):
        raise ValueError("fixed-match candidate identity set is invalid")
    normal_ids = set(range(player_count)) - candidate_set
    awards = [0] * player_count
    for finish_place, actor_id in enumerate(finish_order, start=1):
        awards[actor_id] = (
            4
            if finish_place == 1
            else 3
            if finish_place == 2
            else 1
            if finish_place == player_count - 1
            else 0
            if finish_place == player_count
            else 2
        )
    candidate_mean = sum(awards[value] for value in candidate_set) / len(candidate_set)
    normal_mean = sum(awards[value] for value in normal_ids) / len(normal_ids)
    chip_difference = candidate_mean - normal_mean
    finish_positions = {value: index for index, value in enumerate(finish_order)}
    before = sum(
        int(finish_positions[candidate_id] < finish_positions[normal_id])
        for candidate_id in candidate_set
        for normal_id in normal_ids
    )
    comparisons = len(candidate_set) * len(normal_ids)
    pair_rate = before / comparisons
    return (
        awards,
        candidate_mean,
        normal_mean,
        chip_difference,
        before,
        comparisons,
        pair_rate,
        pair_rate - 0.5,
    )


def _fixed_suffix_reward_components(
    chip_differences: Sequence[float],
    pairwise_centered: Sequence[float],
    coefficient: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chip = np.asarray(chip_differences, dtype=np.float64)
    pair = np.asarray(pairwise_centered, dtype=np.float64)
    if chip.shape != (5,) or pair.shape != (5,):
        raise ValueError("fixed-match suffix reward requires exactly five acts")
    suffix_chip = np.cumsum(chip[::-1])[::-1]
    suffix_pair = np.cumsum(pair[::-1])[::-1]
    return suffix_chip, suffix_pair, (suffix_chip + coefficient * suffix_pair) / 5.0


def _validate_fixed_shard_backend_execution(
    shard: Mapping[str, object],
    collection: Mapping[str, object],
    execution: object,
) -> None:
    """Bind a v2 direct shard's declared execution to its precommitted slot."""

    version = shard.get("collectionPlanVersion", 1)
    if version == 1:
        if (
            "collectionPlanVersion" in shard
            or "shardBackendMap" in shard
            or "crossBackendCalibrationReportSha256" in shard
            or (
                isinstance(execution, Mapping)
                and (
                    "fixedCollectionPlanVersion" in execution
                    or "plannedShardBackend" in execution
                )
            )
        ):
            raise ValueError(
                "fixed collection plan v1 cannot carry mixed backend metadata"
            )
        return
    if version != 2:
        raise ValueError("fixed collection plan version is unsupported")
    shard_count = shard.get("matchShardCount")
    shard_index = shard.get("matchShardIndex")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("fixed collection plan shard index is invalid")
    backend_map = _canonical_shard_backend_map(
        shard.get("shardBackendMap"), shard_count
    )
    planned_backend = backend_map[str(shard_index)]
    if not isinstance(execution, Mapping):
        raise ValueError("fixed collection plan execution binding is missing")
    try:
        validate_v4_policy_numerics_contract(execution.get("policyNumerics"))
    except ValueError as error:
        raise ValueError(
            "fixed collection plan policy numerics binding is invalid"
        ) from error
    device = execution.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("fixed collection plan execution device is invalid")
    try:
        actual_backend = torch.device(device).type
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("fixed collection plan execution device is invalid") from error
    if (
        actual_backend != planned_backend
        or execution.get("fixedCollectionPlanVersion") != 2
        or execution.get("plannedShardBackend") != planned_backend
        or execution.get("deterministicAlgorithms") is not True
        or collection.get("batchedGpuMaskedLogitInference")
        is not (planned_backend == "cuda")
        or (
            planned_backend == "cuda"
            and (
                execution.get("cudaAvailable") is not True
                or execution.get("tf32Allowed") is not False
                or execution.get("cublasWorkspaceConfig") != ":4096:8"
            )
        )
        or (
            planned_backend == "cpu"
            and (
                execution.get("cudaAvailable") is not False
                or execution.get("tf32Allowed") is not None
                or execution.get("cublasWorkspaceConfig") is not None
            )
        )
    ):
        raise ValueError(
            "fixed collection shard execution does not match its precommitted backend"
        )


def _validate_fixed_match_ppo_contract(
    metadata: Mapping[str, object],
    archive: Mapping[str, np.ndarray],
    tensors: V4TrajectoryTensors,
) -> tuple[str, str, str, str]:
    """Validate the composite fixed-match return without legacy chip assumptions."""

    collection = metadata.get("collection")
    reward = metadata.get("rewardContract")
    returns = metadata.get("returnsAndAdvantages")
    training = metadata.get("trainingRequirements")
    model = metadata.get("modelBinding")
    actor_config_metadata = metadata.get("actorConfig")
    privacy = metadata.get("privacy")
    critic_binding = metadata.get("privilegedCriticBinding")
    environment = metadata.get("environmentBinding")
    sources = metadata.get("sourceHashes")
    shard = metadata.get("shard")
    execution = metadata.get("execution")
    if not isinstance(collection, Mapping):
        raise ValueError("fixed-match PPO collection contract is missing")
    if not isinstance(reward, Mapping):
        raise ValueError("fixed-match PPO reward contract is missing")
    reward_contract_id, _ = canonical_fixed_ppo_reward_contract(reward)
    behavior_contract_id, _ = canonical_fixed_ppo_behavior_policy_contract(
        collection
    )
    if (
        collection.get("algorithm")
        != "evaluation-aligned fixed-physical-ID five-act suffix PPO rollout"
        or collection.get("actsPerCompleteMatch") != 5
        or collection.get("completeMatchesOnly") is not True
        or collection.get("evaluationCandidateIdentityParity") is not True
        or collection.get("candidateIdentitySetFixedForCompleteMatch") is not True
        or collection.get("learnerPhysicalIdentityFixedForCompleteMatch") is not True
        or collection.get("stochasticLearnersPerMatch") != 1
        or collection.get("candidateTeammateBehavior")
        != "frozen candidate greedy masked argmax"
        or collection.get("normalOpponentBehavior")
        != "exact DalmutiScalarEnv.normal_action"
        or collection.get("exactOldLogProbabilityForEveryLearnerDecision") is not True
        or collection.get("exactNormalExpertLabelForEveryLearnerDecision") is not True
        or not isinstance(returns, Mapping)
        or isinstance(returns.get("monteCarloGamma"), bool)
        or not isinstance(returns.get("monteCarloGamma"), (int, float))
        or float(returns.get("monteCarloGamma", math.nan)) != 1.0
        or not isinstance(returns.get("standardized"), bool)
        or returns.get("ownCompleteMatchClusterExcludedAtEveryTier") is not True
        or returns.get("futureRewardsExcludedFromActorAndCriticInputs") is not True
        or not isinstance(training, Mapping)
        or training.get("ppoSourceContract") != "fixed-physical-id-five-act-suffix-v1"
        or training.get("qBoostMustRemainOff") is not True
        or isinstance(training.get("qBoostCoefficient"), bool)
        or not isinstance(training.get("qBoostCoefficient"), (int, float))
        or float(training.get("qBoostCoefficient", math.nan)) != 0.0
        or training.get("requiresPlayerCountBalancedLoss") is not True
        or not isinstance(model, Mapping)
        or model.get("criticExcluded") is not True
        or model.get("sameFrozenActorForLearnerAndCandidateTeammates") is not True
        or not _is_sha256(model.get("actorCheckpointSha256"))
        or not isinstance(actor_config_metadata, Mapping)
        or isinstance(actor_config_metadata.get("dropout"), bool)
        or not isinstance(actor_config_metadata.get("dropout"), (int, float))
        or float(actor_config_metadata.get("dropout", math.nan)) != 0.0
        or not isinstance(privacy, Mapping)
        or privacy.get("actorPublicOnly") is not True
        or privacy.get("opponentPhysicalHandsExcluded") is not True
        or privacy.get("taxCardIdentitiesExcluded") is not True
        or privacy.get("privilegedCriticStateSeparate") is not True
        or privacy.get("privilegedCriticExportAllowed") is not False
        or privacy.get("futureActRewardsExcludedFromActorInputs") is not True
        or not isinstance(critic_binding, Mapping)
        or critic_binding.get("layoutId") != PRIVILEGED_STATE_LAYOUT_ID
        or critic_binding.get("layoutSha256") != PRIVILEGED_STATE_LAYOUT_SHA256
        or critic_binding.get("featureCount") != PRIVILEGED_STATE_SIZE
        or critic_binding.get("actorExportAllowed") is not False
        or not isinstance(environment, Mapping)
        or environment.get("normalExpertCallback") != "DalmutiScalarEnv.normal_action"
        or not _is_sha256(environment.get("evaluatorSourceSha256"))
        or not isinstance(environment.get("candidateSeatParityAudit"), Mapping)
        or environment["candidateSeatParityAudit"].get("allEntriesMatched") is not True
        or not _is_sha256(
            environment["candidateSeatParityAudit"].get("scheduleBindingSha256")
        )
        or not isinstance(sources, Mapping)
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in sources.items()
        )
        or "gpu-training/v4_collect_fixed_match_ppo.py" not in sources
        or environment.get("v4EnvSha256") != sources.get("gpu-training/v4_env.py")
        or environment.get("evaluatorSourceSha256") != sources.get("gpu-training/v4_evaluate.py")
        or environment.get("normalSourceSha256") != sources.get("lib/bot-strategy.ts")
        or not isinstance(shard, Mapping)
        or not _is_canonical_fixed_collection_namespace(
            shard.get("runNamespace")
        )
        or isinstance(shard.get("seedBase"), bool)
        or not isinstance(shard.get("seedBase"), int)
        or not 0 <= int(shard.get("seedBase", -1)) <= 0xFFFF_FFFF
        or not isinstance(shard.get("matchCounts"), Mapping)
        or isinstance(shard.get("matchShardCount"), bool)
        or not isinstance(shard.get("matchShardCount"), int)
        or int(shard.get("matchShardCount", 0)) < 1
        or isinstance(shard.get("matchShardIndex"), bool)
        or not isinstance(shard.get("matchShardIndex"), int)
        or not 0 <= int(shard.get("matchShardIndex", -1)) < int(shard.get("matchShardCount", 0))
        or shard.get("trajectoryIdsIndependentOfShardPartition") is not True
        or shard.get("completeMatchTrajectoryIdsIncludeNamespacePlayerMatchSeedLearnerAndAct") is not True
        or not _is_sha256(shard.get("completeUnshardedLearnerAssignmentSha256"))
    ):
        raise ValueError("fixed-match PPO metadata semantics are missing or incompatible")
    assert isinstance(shard, Mapping)
    expected_shard_identity = fixed_match_shard_identity_sha256(
        shard,
        str(metadata.get("preparationFormat")),
    )
    if shard.get("identitySha256") != expected_shard_identity:
        raise ValueError("fixed-match shard identitySha256 is invalid")
    _validate_fixed_shard_backend_execution(shard, collection, execution)

    sample_float32 = {
        "raw_returns", "baseline_values", "raw_advantages", "advantage_scales",
        "policy_entropies", "raw_act_candidate_mean_chips", "raw_act_normal_mean_chips",
        "raw_act_group_chip_differences", "raw_act_pairwise_rates",
        "raw_act_pairwise_centered_rewards", "raw_act_total_rewards",
        "suffix_group_chip_sums", "suffix_pairwise_centered_returns",
        "suffix_total_returns",
    }
    sample_exact = {
        "selected_action_probabilities": np.dtype(np.float64),
        "terminal_chip_awards": np.dtype(np.int8),
        "forced_masks": np.dtype(np.bool_),
        "source_decision_indices": np.dtype(np.int64),
        "baseline_tiers": np.dtype(np.int8),
        "baseline_reference_counts": np.dtype(np.int32),
        "pairwise_candidate_before_normal_counts": np.dtype(np.int16),
        "pairwise_candidate_normal_comparison_counts": np.dtype(np.int16),
    }
    trajectory_float32 = {
        "trajectory_act_candidate_mean_chips", "trajectory_act_normal_mean_chips",
        "trajectory_act_group_chip_differences", "trajectory_act_pairwise_rates",
        "trajectory_act_pairwise_centered_rewards", "trajectory_act_total_rewards",
        "trajectory_suffix_group_chip_sums", "trajectory_suffix_pairwise_centered_returns",
        "trajectory_suffix_total_returns",
    }
    trajectory_exact: dict[str, np.dtype | None] = {
        "trajectory_ids": None,
        "trajectory_complete_match_ids": None,
        "trajectory_match_clusters": None,
        "trajectory_initial_player_orders": None,
        "trajectory_candidate_initial_seats": None,
        "trajectory_candidate_ids": None,
        "trajectory_act_player_orders": None,
        "trajectory_act_finish_orders": None,
        "trajectory_act_chip_awards_by_physical_id": None,
        "trajectory_player_counts": np.dtype(np.int16),
        "trajectory_roles": np.dtype(np.int8),
        "trajectory_acts": np.dtype(np.int16),
        "trajectory_actor_ids": np.dtype(np.int16),
        "trajectory_match_indices": np.dtype(np.int32),
        "trajectory_match_seeds": np.dtype(np.uint32),
        "trajectory_finish_places": np.dtype(np.int16),
        "trajectory_learner_initial_seats": np.dtype(np.int16),
    }
    required = sample_float32 | set(sample_exact) | trajectory_float32 | set(trajectory_exact)
    missing = sorted(required - set(archive.keys()))
    if missing:
        raise ValueError(f"fixed-match PPO data lacks required array {missing[0]}")
    valid = tensors.valid_masks.numpy()
    shape = valid.shape
    count = shape[0]
    for name in sample_float32:
        array = np.asarray(archive[name])
        if array.shape != shape or array.dtype != np.dtype(np.float32) or not np.isfinite(array).all():
            raise ValueError(f"fixed-match PPO sample array {name} has invalid shape/dtype/value")
    for name, dtype in sample_exact.items():
        array = np.asarray(archive[name])
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"fixed-match PPO sample array {name} has invalid shape/dtype")
    invalid = ~valid
    fixed_zero_sample_names = {
        "raw_act_candidate_mean_chips", "raw_act_normal_mean_chips",
        "raw_act_group_chip_differences", "raw_act_pairwise_rates",
        "raw_act_pairwise_centered_rewards", "raw_act_total_rewards",
        "suffix_group_chip_sums", "suffix_pairwise_centered_returns",
        "suffix_total_returns", "pairwise_candidate_before_normal_counts",
        "pairwise_candidate_normal_comparison_counts",
    }
    if any(np.any(np.asarray(archive[name])[invalid] != 0) for name in fixed_zero_sample_names):
        raise ValueError("fixed-match auxiliary invalid suffix must be exactly zero")
    for name in trajectory_float32:
        array = np.asarray(archive[name])
        if array.shape != (count,) or array.dtype != np.dtype(np.float32) or not np.isfinite(array).all():
            raise ValueError(f"fixed-match PPO trajectory array {name} is invalid")
    for name, dtype in trajectory_exact.items():
        array = np.asarray(archive[name])
        if array.shape != (count,) or (dtype is None and array.dtype.kind != "U") or (dtype is not None and array.dtype != dtype):
            raise ValueError(f"fixed-match PPO trajectory array {name} is invalid")

    selected = np.asarray(archive["selected_action_probabilities"])
    if np.any(valid & ((selected <= 0.0) | (selected > 1.0) | ~np.isfinite(selected))):
        raise ValueError("fixed-match behavior probabilities are invalid")
    if not np.allclose(tensors.old_action_log_probs.numpy()[valid], np.log(selected[valid]), rtol=0.0, atol=2.0e-6):
        raise ValueError("fixed-match old log probabilities lack exact behavior binding")
    scales = np.asarray(archive["advantage_scales"])
    if np.any(valid & ((scales <= 0.0) | ~np.isfinite(scales))):
        raise ValueError("fixed-match advantage scales are invalid")
    raw_expected = np.asarray(archive["raw_returns"]) - np.asarray(archive["baseline_values"])
    if not np.allclose(np.asarray(archive["raw_advantages"])[valid], raw_expected[valid], rtol=0.0, atol=2.0e-6):
        raise ValueError("fixed-match raw advantages lack return-baseline binding")
    expected_advantage = raw_expected / scales if bool(returns["standardized"]) else raw_expected
    if not np.allclose(tensors.advantages.numpy()[valid], expected_advantage[valid], rtol=0.0, atol=2.0e-6):
        raise ValueError("fixed-match training advantages lack exact derivation")

    coefficient = float(reward["pairwiseCoefficient"])
    terminal = tensors.dones.numpy() & valid
    nonterminal = valid & ~tensors.dones.numpy()
    if np.any(tensors.rewards.numpy()[nonterminal] != 0.0):
        raise ValueError("fixed-match nonterminal rewards must be zero")
    lengths = valid.sum(axis=1, dtype=np.int64)
    if np.any(terminal.sum(axis=1) != 1):
        raise ValueError("each fixed-match act segment requires one terminal")

    by_match: dict[str, list[int]] = {}
    for trajectory in range(count):
        length = int(lengths[trajectory])
        if length < 1 or not tensors.dones[trajectory, length - 1]:
            raise ValueError("fixed-match terminal must be the final valid decision")
        player_count = int(archive["trajectory_player_counts"][trajectory])
        match_index = int(archive["trajectory_match_indices"][trajectory])
        actor_id = int(archive["trajectory_actor_ids"][trajectory])
        act = int(archive["trajectory_acts"][trajectory])
        role = int(archive["trajectory_roles"][trajectory])
        declared_match_count = shard["matchCounts"].get(str(player_count))
        match_start = shard.get("matchStart")
        if (
            not 4 <= player_count <= 10
            or not 1 <= act <= 5
            or not 0 <= actor_id < player_count
            or isinstance(declared_match_count, bool)
            or not isinstance(declared_match_count, int)
            or declared_match_count < 1
            or isinstance(match_start, bool)
            or not isinstance(match_start, int)
            or not match_start <= match_index < match_start + declared_match_count
            or match_index % int(shard["matchShardCount"]) != int(shard["matchShardIndex"])
            or int(archive["trajectory_match_seeds"][trajectory])
            != _fixed_environment_seed(
                str(shard["runNamespace"]),
                int(shard["seedBase"]),
                player_count,
                match_index,
            )
        ):
            raise ValueError("fixed-match trajectory identity values are invalid")
        initial_order = _csv_ints(archive["trajectory_initial_player_orders"][trajectory], "initial order")
        candidate_seats = _csv_ints(archive["trajectory_candidate_initial_seats"][trajectory], "candidate seats")
        candidate_ids = _csv_ints(archive["trajectory_candidate_ids"][trajectory], "candidate IDs")
        act_order = _csv_ints(archive["trajectory_act_player_orders"][trajectory], "act player order")
        finish_order = _csv_ints(archive["trajectory_act_finish_orders"][trajectory], "act finish order")
        chip_awards = _csv_ints(
            archive["trajectory_act_chip_awards_by_physical_id"][trajectory],
            "act chip awards by physical ID",
        )
        if (
            len(initial_order) != player_count
            or set(initial_order) != set(range(player_count))
            or candidate_seats != _evaluation_candidate_seats(player_count, match_index)
            or tuple(sorted(initial_order[seat] for seat in candidate_seats)) != candidate_ids
            or actor_id not in candidate_ids
            or int(archive["trajectory_learner_initial_seats"][trajectory]) != initial_order.index(actor_id)
            or len(act_order) != player_count
            or set(act_order) != set(range(player_count))
            or role != _role_index(act_order.index(actor_id), player_count)
            or len(finish_order) != player_count
            or set(finish_order) != set(range(player_count))
            or len(chip_awards) != player_count
        ):
            raise ValueError("fixed-match evaluator identity or actual-role binding drifted")
        role_one_hot = tensors.global_features[trajectory, :length, 2:7].numpy()
        expected_one_hot = np.eye(5, dtype=np.float32)[role]
        if not np.allclose(role_one_hot, expected_one_hot[None, :], rtol=0.0, atol=1.0e-7):
            raise ValueError("fixed-match public actor role one-hot disagrees with actual act order")
        complete_id = str(archive["trajectory_complete_match_ids"][trajectory])
        expected_complete_id = (
            f"v4-fixed-match-{shard['runNamespace']}-p{player_count}-m{match_index}"
            f"-seed{int(archive['trajectory_match_seeds'][trajectory]):08x}"
        )
        expected_trajectory_id = f"{expected_complete_id}-learner{actor_id}-act{act}"
        if (
            complete_id != expected_complete_id
            or complete_id != str(archive["trajectory_match_clusters"][trajectory])
            or str(archive["trajectory_ids"][trajectory]) != expected_trajectory_id
        ):
            raise ValueError("fixed-match cluster and complete-match IDs must be exact and equal")
        by_match.setdefault(complete_id, []).append(trajectory)

        last = length - 1
        terminal_names = {
            "raw_act_candidate_mean_chips": "trajectory_act_candidate_mean_chips",
            "raw_act_normal_mean_chips": "trajectory_act_normal_mean_chips",
            "raw_act_group_chip_differences": "trajectory_act_group_chip_differences",
            "raw_act_pairwise_rates": "trajectory_act_pairwise_rates",
            "raw_act_pairwise_centered_rewards": "trajectory_act_pairwise_centered_rewards",
            "raw_act_total_rewards": "trajectory_act_total_rewards",
        }
        for sample_name, trajectory_name in terminal_names.items():
            sample = np.asarray(archive[sample_name])[trajectory]
            if np.any(sample[:last] != 0.0) or not np.isclose(sample[last], archive[trajectory_name][trajectory], rtol=0.0, atol=1.0e-6):
                raise ValueError(f"fixed-match raw component {sample_name} lacks terminal binding")
        for sample_name, trajectory_name in {
            "suffix_group_chip_sums": "trajectory_suffix_group_chip_sums",
            "suffix_pairwise_centered_returns": "trajectory_suffix_pairwise_centered_returns",
            "suffix_total_returns": "trajectory_suffix_total_returns",
        }.items():
            if not np.allclose(archive[sample_name][trajectory, :length], archive[trajectory_name][trajectory], rtol=0.0, atol=1.0e-6):
                raise ValueError(f"fixed-match suffix component {sample_name} is inconsistent")
        candidate_mean = float(archive["trajectory_act_candidate_mean_chips"][trajectory])
        normal_mean = float(archive["trajectory_act_normal_mean_chips"][trajectory])
        chip_difference = float(archive["trajectory_act_group_chip_differences"][trajectory])
        pair_rate = float(archive["trajectory_act_pairwise_rates"][trajectory])
        pair_centered = float(archive["trajectory_act_pairwise_centered_rewards"][trajectory])
        act_total = float(archive["trajectory_act_total_rewards"][trajectory])
        before = int(archive["pairwise_candidate_before_normal_counts"][trajectory, last])
        comparisons = int(archive["pairwise_candidate_normal_comparison_counts"][trajectory, last])
        (
            expected_awards,
            expected_candidate_mean,
            expected_normal_mean,
            expected_chip_difference,
            expected_before,
            expected_comparisons,
            expected_pair_rate,
            expected_pair_centered,
        ) = _fixed_group_reward_from_outcome(
            finish_order,
            candidate_ids,
        )
        if (
            list(chip_awards) != expected_awards
            or int(archive["trajectory_finish_places"][trajectory]) != finish_order.index(actor_id) + 1
            or int(archive["terminal_chip_awards"][trajectory, last]) != expected_awards[actor_id]
            or np.any(archive["terminal_chip_awards"][trajectory, :last] != 0)
            or not np.isclose(candidate_mean, expected_candidate_mean, atol=1.0e-6)
            or not np.isclose(normal_mean, expected_normal_mean, atol=1.0e-6)
            or not np.isclose(chip_difference, expected_chip_difference, atol=1.0e-6)
            or comparisons != expected_comparisons
            or before != expected_before
            or not 0 <= before <= comparisons
            or not np.isclose(pair_rate, expected_pair_rate, atol=1.0e-6)
            or not np.isclose(pair_centered, expected_pair_centered, atol=1.0e-6)
            or not np.isclose(act_total, (chip_difference + coefficient * pair_centered) / 5.0, atol=1.0e-6)
        ):
            raise ValueError("fixed-match raw group reward math is invalid")
        suffix_total = float(archive["trajectory_suffix_total_returns"][trajectory])
        if (
            not np.isclose(float(tensors.rewards[trajectory, last]), suffix_total, atol=1.0e-6)
            or not np.allclose(archive["raw_returns"][trajectory, :length], suffix_total, atol=1.0e-6)
        ):
            raise ValueError("fixed-match composite suffix reward binding is invalid")

    for complete_id, members in by_match.items():
        ordered = sorted(members, key=lambda value: int(archive["trajectory_acts"][value]))
        if len(ordered) != 5 or [int(archive["trajectory_acts"][value]) for value in ordered] != [1, 2, 3, 4, 5]:
            raise ValueError("each complete match must contain exactly act segments 1..5")
        invariant_names = (
            "trajectory_player_counts", "trajectory_actor_ids", "trajectory_match_indices",
            "trajectory_match_seeds", "trajectory_initial_player_orders",
            "trajectory_candidate_initial_seats", "trajectory_candidate_ids",
        )
        if any(len({archive[name][value].item() for value in ordered}) != 1 for name in invariant_names):
            raise ValueError("fixed-match physical identities changed across acts")
        suffix_chip, suffix_pair, suffix_total = _fixed_suffix_reward_components(
            [archive["trajectory_act_group_chip_differences"][value] for value in ordered],
            [archive["trajectory_act_pairwise_centered_rewards"][value] for value in ordered],
            coefficient,
        )
        if (
            not np.allclose([archive["trajectory_suffix_group_chip_sums"][value] for value in ordered], suffix_chip, atol=1.0e-6)
            or not np.allclose([archive["trajectory_suffix_pairwise_centered_returns"][value] for value in ordered], suffix_pair, atol=1.0e-6)
            or not np.allclose([archive["trajectory_suffix_total_returns"][value] for value in ordered], suffix_total, atol=1.0e-6)
        ):
            raise ValueError("fixed-match suffix return math is invalid")
    if (
        metadata.get("completeMatchCount") != len(by_match)
        or metadata.get("trajectoryCount") != count
        or metadata.get("sampleCount") != int(valid.sum())
        or metadata.get("maxTimeSteps") != valid.shape[1]
    ):
        raise ValueError("fixed-match manifest counts do not match complete trajectories")

    baseline_records = tuple(
        BaselineRecord(
            int(archive["trajectory_player_counts"][index]),
            ROLES[int(archive["trajectory_roles"][index])],
            int(archive["trajectory_acts"][index]),
            str(archive["trajectory_complete_match_ids"][index]),
            float(archive["trajectory_suffix_total_returns"][index]),
        )
        for index in range(count)
    )
    expected_baselines = leave_one_match_out_baselines(baseline_records)
    for index, expected in enumerate(expected_baselines):
        length = int(lengths[index])
        if (
            not np.allclose(archive["baseline_values"][index, :length], expected.baseline, atol=2.0e-6)
            or not np.allclose(archive["advantage_scales"][index, :length], expected.scale, atol=2.0e-6)
            or np.any(archive["baseline_tiers"][index, :length] != expected.tier)
            or np.any(archive["baseline_reference_counts"][index, :length] != expected.reference_count)
        ):
            raise ValueError("fixed-match leave-one-complete-match baseline binding is invalid")
    expected_distribution: dict[str, dict[str, int]] = {}
    forced_masks = np.asarray(archive["forced_masks"])
    for player_count in sorted(set(int(value) for value in archive["trajectory_player_counts"])):
        rows = np.flatnonzero(np.asarray(archive["trajectory_player_counts"]) == player_count)
        selected_valid = valid[rows]
        selected_forced = forced_masks[rows] & selected_valid
        valid_count = int(selected_valid.sum())
        forced_count = int(selected_forced.sum())
        expected_distribution[str(player_count)] = {
            "completeMatches": len(rows) // 5,
            "learnerActTrajectories": len(rows),
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
    if metadata.get("playerCountDistribution") != expected_distribution:
        raise ValueError("fixed-match per-player-count eligibility distribution is invalid")
    balance = metadata.get("opponentAndSeatBalance")
    complete_balance = (
        balance.get("completeMatchRangeAcrossAllShards")
        if isinstance(balance, Mapping)
        else None
    )
    if not isinstance(balance, Mapping) or not isinstance(complete_balance, Mapping):
        raise ValueError("fixed-match learner physical-ID balance metadata is missing")
    selected_physical: dict[str, dict[str, int]] = {}
    for members in by_match.values():
        index = members[0]
        player_count = int(archive["trajectory_player_counts"][index])
        actor_id = int(archive["trajectory_actor_ids"][index])
        counts = selected_physical.setdefault(
            str(player_count), {str(value): 0 for value in range(player_count)}
        )
        counts[str(actor_id)] += 1
    if balance.get("learnerAssignmentsByPhysicalIdentity") != selected_physical:
        raise ValueError("fixed-match selected learner physical-ID counts are invalid")
    complete_physical = complete_balance.get("learnerAssignmentsByPhysicalIdentity")
    if (
        not isinstance(complete_physical, Mapping)
        or complete_balance.get("learnerPhysicalIdentityBalancedWithinOne") is not True
    ):
        raise ValueError("fixed-match complete physical-ID balance contract is invalid")
    for player_key, declared_count in shard["matchCounts"].items():
        counts = complete_physical.get(player_key)
        if (
            not isinstance(counts, Mapping)
            or sorted(counts) != [str(value) for value in range(int(player_key))]
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values())
            or sum(counts.values()) != declared_count
            or max(counts.values()) - min(counts.values()) > 1
        ):
            raise ValueError("fixed-match complete learner physical-ID balance is invalid")
    plan_id, _ = canonical_fixed_collection_plan(
        metadata,
        reward_contract_id,
        behavior_contract_id,
    )
    return (
        str(model["actorCheckpointSha256"]),
        reward_contract_id,
        behavior_contract_id,
        plan_id,
    )


def _direct_loss_eligibility(
    metadata: Mapping[str, object],
    archive: Mapping[str, np.ndarray],
    tensors: V4TrajectoryTensors,
) -> V4LossEligibility:
    valid_masks = tensors.valid_masks
    preparation = metadata.get("preparationFormat")
    version = metadata.get("preparationVersion")
    if version != 1:
        raise ValueError("V4 dataset lacks a supported loss preparation version")
    all_valid = valid_masks.clone()
    none_valid = torch.zeros_like(valid_masks)
    behavior_actor_sha256s: tuple[str, ...] = ()
    ppo_source_contracts: tuple[str, ...] = ()
    requires_player_count_balanced_loss = False
    requires_qboost_coefficient_zero = False
    ppo_reward_contracts: tuple[str, ...] = ()
    ppo_behavior_policy_contracts: tuple[str, ...] = ()
    fixed_collection_plan_ids: tuple[str, ...] = ()
    if preparation == V4_NORMAL_PREPARATION_FORMAT:
        if metadata.get("privilegedCriticExportAllowed") is not False:
            raise ValueError("Normal V4 data lacks its training-only critic boundary")
        ppo = critic = none_valid
    elif preparation == V4_DAGGER_PREPARATION_FORMAT:
        collection = metadata.get("collection")
        if (
            not isinstance(collection, Mapping)
            or collection.get("algorithm") != "DAgger"
            or collection.get("expert") != "exact-v4-env-Normal"
            or collection.get("expertLabelForEveryDecision") is not True
        ):
            raise ValueError("DAgger V4 data lacks exact Normal expert semantics")
        ppo = critic = none_valid
    elif preparation == V4_PPO_PREPARATION_FORMAT:
        collection = metadata.get("collection")
        returns = metadata.get("returnsAndAdvantages")
        model = metadata.get("modelBinding")
        privacy = metadata.get("privacy")
        critic_binding = metadata.get("privilegedCriticBinding")
        if (
            not isinstance(collection, Mapping)
            or collection.get("algorithm") != "on-policy PPO league rollout"
            or collection.get("exactOldLogProbabilityForEveryLearnerDecision")
            is not True
            or collection.get("exactNormalExpertLabelForEveryLearnerDecision")
            is not True
            or not isinstance(returns, Mapping)
            or not isinstance(returns.get("standardized"), bool)
            or isinstance(returns.get("monteCarloGamma"), bool)
            or not isinstance(returns.get("monteCarloGamma"), (int, float))
            or not math.isfinite(float(returns.get("monteCarloGamma", math.nan)))
            or not 0.0 <= float(returns.get("monteCarloGamma", math.nan)) <= 1.0
            or not isinstance(model, Mapping)
            or model.get("criticExcluded") is not True
            or not _is_sha256(model.get("actorCheckpointSha256"))
            or not isinstance(privacy, Mapping)
            or privacy.get("actorPublicOnly") is not True
            or privacy.get("opponentPhysicalHandsExcluded") is not True
            or privacy.get("taxCardIdentitiesExcluded") is not True
            or privacy.get("privilegedCriticStateSeparate") is not True
            or privacy.get("privilegedCriticExportAllowed") is not False
            or not isinstance(critic_binding, Mapping)
            or critic_binding.get("layoutId") != PRIVILEGED_STATE_LAYOUT_ID
            or critic_binding.get("layoutSha256")
            != PRIVILEGED_STATE_LAYOUT_SHA256
            or critic_binding.get("featureCount") != PRIVILEGED_STATE_SIZE
            or critic_binding.get("actorExportAllowed") is not False
        ):
            raise ValueError("PPO V4 data lacks on-policy loss provenance")
        required_arrays = {
            "selected_action_probabilities",
            "raw_returns",
            "baseline_values",
            "raw_advantages",
            "advantage_scales",
            "terminal_chip_awards",
        }
        missing = sorted(required_arrays - set(archive.keys()))
        if missing:
            raise ValueError(f"PPO V4 data lacks loss array {missing[0]}")
        shape = valid_masks.shape
        arrays = {name: np.asarray(archive[name]) for name in required_arrays}
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("PPO V4 loss arrays must match [trajectory, time]")
        valid = valid_masks.numpy()
        selected = arrays["selected_action_probabilities"]
        if np.any(valid & ((selected <= 0.0) | (selected > 1.0) | ~np.isfinite(selected))):
            raise ValueError("PPO V4 behavior probabilities are invalid")
        if not np.allclose(
            tensors.old_action_log_probs.numpy()[valid],
            np.log(selected[valid]),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError("PPO V4 old log probabilities lack exact behavior binding")
        scales = arrays["advantage_scales"]
        if np.any(valid & ((scales <= 0.0) | ~np.isfinite(scales))):
            raise ValueError("PPO V4 advantage scales are invalid")
        raw_expected = arrays["raw_returns"] - arrays["baseline_values"]
        if not np.allclose(
            arrays["raw_advantages"][valid], raw_expected[valid],
            rtol=0.0, atol=2.0e-6,
        ):
            raise ValueError("PPO V4 raw advantages lack their return binding")
        expected_advantages = (
            arrays["raw_advantages"] / scales
            if bool(returns["standardized"])
            else arrays["raw_advantages"]
        )
        if not np.allclose(
            tensors.advantages.numpy()[valid], expected_advantages[valid],
            rtol=0.0, atol=2.0e-6,
        ):
            raise ValueError("PPO V4 training advantages lack their derivation binding")
        gamma = float(returns["monteCarloGamma"])
        lengths = valid.sum(axis=1, dtype=np.int64)
        for trajectory, length in enumerate(lengths):
            running = 0.0
            expected_returns = np.zeros(int(length), dtype=np.float64)
            for time_index in range(int(length) - 1, -1, -1):
                running = (
                    float(tensors.rewards[trajectory, time_index].item())
                    + gamma * running
                )
                expected_returns[time_index] = running
            if not np.allclose(
                arrays["raw_returns"][trajectory, : int(length)],
                expected_returns,
                rtol=0.0,
                atol=2.0e-6,
            ):
                raise ValueError("PPO V4 raw returns violate their Monte Carlo binding")
        terminal = tensors.dones.numpy() & valid
        nonterminal = valid & ~tensors.dones.numpy()
        chip_awards = arrays["terminal_chip_awards"]
        if not np.issubdtype(chip_awards.dtype, np.integer) or np.any(
            valid & ((chip_awards < 0) | (chip_awards > 4))
        ):
            raise ValueError("PPO V4 chip awards are invalid")
        if np.any(nonterminal & (chip_awards != 0)) or not np.allclose(
            tensors.rewards.numpy()[nonterminal], 0.0,
            rtol=0.0, atol=1.0e-7,
        ):
            raise ValueError("PPO V4 non-terminal rewards must remain zero")
        expected_rewards = (
            chip_awards.astype(np.float32) - 2.0
        ) / 2.0
        if not np.allclose(
            tensors.rewards.numpy()[terminal], expected_rewards[terminal],
            rtol=0.0, atol=1.0e-7,
        ):
            raise ValueError("PPO V4 terminal rewards lack their chip-award binding")
        behavior_actor_sha256s = (str(model["actorCheckpointSha256"]),)
        ppo_source_contracts = (V4_LEGACY_PPO_SOURCE_CONTRACT,)
        ppo = critic = all_valid
    elif preparation == V4_FIXED_MATCH_PPO_PREPARATION_FORMAT:
        actor_sha256, reward_contract, behavior_contract, plan_id = (
            _validate_fixed_match_ppo_contract(metadata, archive, tensors)
        )
        behavior_actor_sha256s = (actor_sha256,)
        ppo_source_contracts = (V4_FIXED_PPO_SOURCE_CONTRACT,)
        requires_player_count_balanced_loss = True
        requires_qboost_coefficient_zero = True
        ppo_reward_contracts = (reward_contract,)
        ppo_behavior_policy_contracts = (behavior_contract,)
        fixed_collection_plan_ids = (plan_id,)
        ppo = critic = all_valid
    elif preparation == V4_SMOKE_PREPARATION_FORMAT:
        # Synthetic smoke data exercises serialization and BC training only.
        # It must not be accepted as evidence for the PPO/critic objectives.
        ppo = critic = none_valid
    else:
        raise ValueError(
            "unsupported or missing V4 dataset loss preparation format"
        )
    return _validate_loss_masks(
        V4LossEligibility(
            behavior_cloning=all_valid,
            ppo=ppo,
            critic=critic,
            preparation_format=str(preparation),
            preparation_version=int(version),
            behavior_actor_sha256s=behavior_actor_sha256s,
            ppo_source_contracts=ppo_source_contracts,
            requires_player_count_balanced_loss=requires_player_count_balanced_loss,
            requires_qboost_coefficient_zero=requires_qboost_coefficient_zero,
            ppo_reward_contracts=ppo_reward_contracts,
            ppo_behavior_policy_contracts=ppo_behavior_policy_contracts,
            fixed_collection_plan_ids=fixed_collection_plan_ids,
        ),
        valid_masks,
    )


def _validate_merged_fixed_match_raw_contract(
    archive: Mapping[str, np.ndarray],
    tensors: V4TrajectoryTensors,
    fixed_rows: np.ndarray,
    coefficient: float,
    collection_plans: Sequence[Mapping[str, object]],
) -> None:
    """Revalidate fixed raw outcomes after one or more merge/repackaging hops."""

    count, time_steps = tensors.valid_masks.shape
    if fixed_rows.dtype != np.dtype(np.bool_) or fixed_rows.shape != (count,):
        raise ValueError("merged fixed-match row selector is invalid")
    indices = np.flatnonzero(fixed_rows)
    if not len(indices):
        raise ValueError("merged fixed-match provenance has no fixed trajectories")
    sample_names = {
        "raw_returns",
        "raw_act_candidate_mean_chips",
        "raw_act_normal_mean_chips",
        "raw_act_group_chip_differences",
        "raw_act_pairwise_rates",
        "raw_act_pairwise_centered_rewards",
        "raw_act_total_rewards",
        "suffix_group_chip_sums",
        "suffix_pairwise_centered_returns",
        "suffix_total_returns",
        "pairwise_candidate_before_normal_counts",
        "pairwise_candidate_normal_comparison_counts",
        "terminal_chip_awards",
        "selected_action_probabilities",
    }
    trajectory_names = {
        "trajectory_ids",
        "trajectory_complete_match_ids",
        "trajectory_match_clusters",
        "trajectory_player_counts",
        "trajectory_roles",
        "trajectory_acts",
        "trajectory_actor_ids",
        "trajectory_match_indices",
        "trajectory_match_seeds",
        "trajectory_monte_carlo_gammas",
        "trajectory_learner_initial_seats",
        "trajectory_initial_player_orders",
        "trajectory_candidate_initial_seats",
        "trajectory_candidate_ids",
        "trajectory_act_player_orders",
        "trajectory_act_finish_orders",
        "trajectory_act_chip_awards_by_physical_id",
        "trajectory_finish_places",
        "trajectory_act_candidate_mean_chips",
        "trajectory_act_normal_mean_chips",
        "trajectory_act_group_chip_differences",
        "trajectory_act_pairwise_rates",
        "trajectory_act_pairwise_centered_rewards",
        "trajectory_act_total_rewards",
        "trajectory_suffix_group_chip_sums",
        "trajectory_suffix_pairwise_centered_returns",
        "trajectory_suffix_total_returns",
    }
    missing = sorted((sample_names | trajectory_names) - set(archive.keys()))
    if missing:
        raise ValueError(
            f"merged fixed-match PPO provenance lacks raw array {missing[0]}"
        )
    for name in sample_names:
        if np.asarray(archive[name]).shape != (count, time_steps):
            raise ValueError(f"merged fixed-match sample array {name} has invalid shape")
    for name in trajectory_names:
        if np.asarray(archive[name]).shape != (count,):
            raise ValueError(
                f"merged fixed-match trajectory array {name} has invalid shape"
            )

    valid = tensors.valid_masks.numpy()
    dones = tensors.dones.numpy()
    rewards = tensors.rewards.numpy()
    lengths = valid.sum(axis=1, dtype=np.int64)
    by_match: dict[str, list[int]] = {}
    matches_by_plan: dict[str, dict[str, dict[int, str]]] = {
        str(plan["opaqueId"]): {} for plan in collection_plans
    }
    for trajectory in indices.tolist():
        length = int(lengths[trajectory])
        last = length - 1
        if (
            length < 1
            or not dones[trajectory, last]
            or int((dones[trajectory] & valid[trajectory]).sum()) != 1
            or np.any(rewards[trajectory, :last] != 0.0)
        ):
            raise ValueError("merged fixed-match terminal/reward boundary is invalid")
        player_count = int(archive["trajectory_player_counts"][trajectory])
        match_index = int(archive["trajectory_match_indices"][trajectory])
        seed = int(archive["trajectory_match_seeds"][trajectory])
        actor_id = int(archive["trajectory_actor_ids"][trajectory])
        act = int(archive["trajectory_acts"][trajectory])
        role = int(archive["trajectory_roles"][trajectory])
        initial_order = _csv_ints(
            archive["trajectory_initial_player_orders"][trajectory],
            "merged fixed initial order",
        )
        candidate_seats = _csv_ints(
            archive["trajectory_candidate_initial_seats"][trajectory],
            "merged fixed candidate seats",
        )
        candidate_ids = _csv_ints(
            archive["trajectory_candidate_ids"][trajectory],
            "merged fixed candidate IDs",
        )
        act_order = _csv_ints(
            archive["trajectory_act_player_orders"][trajectory],
            "merged fixed act order",
        )
        finish_order = _csv_ints(
            archive["trajectory_act_finish_orders"][trajectory],
            "merged fixed finish order",
        )
        chip_awards = _csv_ints(
            archive["trajectory_act_chip_awards_by_physical_id"][trajectory],
            "merged fixed chip awards",
        )
        if (
            not 4 <= player_count <= 10
            or not 1 <= act <= 5
            or match_index < 0
            or not 0 <= actor_id < player_count
            or len(initial_order) != player_count
            or set(initial_order) != set(range(player_count))
            or candidate_seats != _evaluation_candidate_seats(
                player_count, match_index
            )
            or tuple(sorted(initial_order[seat] for seat in candidate_seats))
            != candidate_ids
            or actor_id not in candidate_ids
            or int(archive["trajectory_learner_initial_seats"][trajectory])
            != initial_order.index(actor_id)
            or len(act_order) != player_count
            or set(act_order) != set(range(player_count))
            or role != _role_index(act_order.index(actor_id), player_count)
            or len(chip_awards) != player_count
        ):
            raise ValueError(
                "merged fixed-match evaluator identity or actual-role binding drifted"
            )
        selected = np.asarray(archive["selected_action_probabilities"])[trajectory]
        if (
            np.any((selected[:length] <= 0.0) | (selected[:length] > 1.0))
            or not np.isfinite(selected[:length]).all()
            or not np.allclose(
                tensors.old_action_log_probs[trajectory, :length].numpy(),
                np.log(selected[:length]),
                rtol=0.0,
                atol=2.0e-6,
            )
        ):
            raise ValueError(
                "merged fixed-match old log probabilities lack raw behavior binding"
            )
        gamma = archive["trajectory_monte_carlo_gammas"][trajectory]
        if isinstance(gamma, (bool, np.bool_)) or not np.isclose(
            float(gamma), 1.0, rtol=0.0, atol=0.0
        ):
            raise ValueError("merged fixed-match Monte Carlo gamma must equal 1.0")
        expected_one_hot = np.eye(5, dtype=np.float32)[role]
        if not np.allclose(
            tensors.global_features[trajectory, :length, 2:7].numpy(),
            expected_one_hot[None, :],
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError("merged fixed-match public role binding drifted")
        complete_id = str(archive["trajectory_complete_match_ids"][trajectory])
        expected_suffix = f"-p{player_count}-m{match_index}-seed{seed:08x}"
        expected_trajectory_id = f"{complete_id}-learner{actor_id}-act{act}"
        if (
            not complete_id.startswith("v4-fixed-match-")
            or not complete_id.endswith(expected_suffix)
            or complete_id != str(archive["trajectory_match_clusters"][trajectory])
            or str(archive["trajectory_ids"][trajectory]) != expected_trajectory_id
        ):
            raise ValueError("merged fixed-match cluster identity binding is invalid")
        matching_plans: list[Mapping[str, object]] = []
        for plan in collection_plans:
            fields_value = plan.get("canonicalFields")
            covered_value = plan.get("coveredShardIndices")
            assert isinstance(fields_value, Mapping)
            assert isinstance(covered_value, list)
            plan_start = int(fields_value["matchStart"])
            match_counts = fields_value["matchCounts"]
            assert isinstance(match_counts, Mapping)
            declared_count = match_counts.get(str(player_count))
            if (
                complete_id
                == f"v4-fixed-match-{fields_value['runNamespace']}{expected_suffix}"
                and isinstance(declared_count, int)
                and plan_start <= match_index < plan_start + declared_count
                and seed
                == _fixed_environment_seed(
                    str(fields_value["runNamespace"]),
                    int(fields_value["seedBase"]),
                    player_count,
                    match_index,
                )
                and match_index % int(fields_value["matchShardCount"])
                in covered_value
            ):
                matching_plans.append(plan)
        if len(matching_plans) != 1:
            raise ValueError(
                "merged fixed-match trajectory does not bind to exactly one "
                "complete collection plan"
            )
        plan_id = str(matching_plans[0]["opaqueId"])
        player_matches = matches_by_plan[plan_id].setdefault(
            str(player_count), {}
        )
        previous_complete_id = player_matches.setdefault(match_index, complete_id)
        if previous_complete_id != complete_id:
            raise ValueError(
                "merged fixed collection plan has multiple complete IDs for one "
                "match index"
            )
        by_match.setdefault(complete_id, []).append(trajectory)

        (
            expected_awards,
            expected_candidate_mean,
            expected_normal_mean,
            expected_difference,
            expected_before,
            expected_comparisons,
            expected_rate,
            expected_centered,
        ) = _fixed_group_reward_from_outcome(finish_order, candidate_ids)
        expected_total = (expected_difference + coefficient * expected_centered) / 5.0
        declared_values = (
            float(archive["trajectory_act_candidate_mean_chips"][trajectory]),
            float(archive["trajectory_act_normal_mean_chips"][trajectory]),
            float(archive["trajectory_act_group_chip_differences"][trajectory]),
            float(archive["trajectory_act_pairwise_rates"][trajectory]),
            float(archive["trajectory_act_pairwise_centered_rewards"][trajectory]),
            float(archive["trajectory_act_total_rewards"][trajectory]),
        )
        expected_values = (
            expected_candidate_mean,
            expected_normal_mean,
            expected_difference,
            expected_rate,
            expected_centered,
            expected_total,
        )
        before_sample = np.asarray(
            archive["pairwise_candidate_before_normal_counts"]
        )[trajectory]
        comparison_sample = np.asarray(
            archive["pairwise_candidate_normal_comparison_counts"]
        )[trajectory]
        terminal_awards = np.asarray(archive["terminal_chip_awards"])[trajectory]
        if (
            list(chip_awards) != expected_awards
            or int(archive["trajectory_finish_places"][trajectory])
            != finish_order.index(actor_id) + 1
            or not np.allclose(declared_values, expected_values, atol=1.0e-6)
            or np.any(before_sample[:last] != 0)
            or int(before_sample[last]) != expected_before
            or np.any(comparison_sample[:last] != 0)
            or int(comparison_sample[last]) != expected_comparisons
            or np.any(terminal_awards[:last] != 0)
            or int(terminal_awards[last]) != expected_awards[actor_id]
        ):
            raise ValueError("merged fixed-match raw finish/chips D/Q math is invalid")
        for sample_name, expected_value in (
            ("raw_act_candidate_mean_chips", expected_candidate_mean),
            ("raw_act_normal_mean_chips", expected_normal_mean),
            ("raw_act_group_chip_differences", expected_difference),
            ("raw_act_pairwise_rates", expected_rate),
            ("raw_act_pairwise_centered_rewards", expected_centered),
            ("raw_act_total_rewards", expected_total),
        ):
            sample = np.asarray(archive[sample_name])[trajectory]
            if np.any(sample[:last] != 0.0) or not np.isclose(
                sample[last], expected_value, atol=1.0e-6
            ):
                raise ValueError(
                    f"merged fixed-match raw sample {sample_name} is invalid"
                )
        suffix_total = float(
            archive["trajectory_suffix_total_returns"][trajectory]
        )
        for sample_name, trajectory_name in (
            ("suffix_group_chip_sums", "trajectory_suffix_group_chip_sums"),
            (
                "suffix_pairwise_centered_returns",
                "trajectory_suffix_pairwise_centered_returns",
            ),
            ("suffix_total_returns", "trajectory_suffix_total_returns"),
        ):
            if not np.allclose(
                archive[sample_name][trajectory, :length],
                archive[trajectory_name][trajectory],
                atol=1.0e-6,
            ):
                raise ValueError(
                    f"merged fixed-match suffix sample {sample_name} is invalid"
                )
        if (
            not np.isclose(rewards[trajectory, last], suffix_total, atol=1.0e-6)
            or not np.allclose(
                archive["raw_returns"][trajectory, :length],
                suffix_total,
                atol=2.0e-6,
            )
        ):
            raise ValueError("merged fixed-match suffix target binding is invalid")

    for complete_id, members in by_match.items():
        ordered = sorted(
            members, key=lambda value: int(archive["trajectory_acts"][value])
        )
        if len(ordered) != 5 or [
            int(archive["trajectory_acts"][value]) for value in ordered
        ] != [1, 2, 3, 4, 5]:
            raise ValueError(
                "merged fixed-match cluster must contain all five act segments"
            )
        invariant_names = (
            "trajectory_player_counts",
            "trajectory_actor_ids",
            "trajectory_match_indices",
            "trajectory_match_seeds",
            "trajectory_initial_player_orders",
            "trajectory_candidate_initial_seats",
            "trajectory_candidate_ids",
        )
        if any(
            len({archive[name][value].item() for value in ordered}) != 1
            for name in invariant_names
        ):
            raise ValueError(
                "merged fixed-match physical identities changed across acts"
            )
        suffix_chip, suffix_pair, suffix_total = _fixed_suffix_reward_components(
            [
                archive["trajectory_act_group_chip_differences"][value]
                for value in ordered
            ],
            [
                archive["trajectory_act_pairwise_centered_rewards"][value]
                for value in ordered
            ],
            coefficient,
        )
        if (
            not np.allclose(
                [
                    archive["trajectory_suffix_group_chip_sums"][value]
                    for value in ordered
                ],
                suffix_chip,
                atol=1.0e-6,
            )
            or not np.allclose(
                [
                    archive["trajectory_suffix_pairwise_centered_returns"][value]
                    for value in ordered
                ],
                suffix_pair,
                atol=1.0e-6,
            )
            or not np.allclose(
                [
                    archive["trajectory_suffix_total_returns"][value]
                    for value in ordered
                ],
                suffix_total,
                atol=1.0e-6,
            )
        ):
            raise ValueError("merged fixed-match five-act suffix math is invalid")
    for plan in collection_plans:
        plan_id = str(plan["opaqueId"])
        fields_value = plan["canonicalFields"]
        assert isinstance(fields_value, Mapping)
        match_counts = fields_value["matchCounts"]
        assert isinstance(match_counts, Mapping)
        actual_counts = matches_by_plan[plan_id]
        start = int(fields_value["matchStart"])
        if set(actual_counts) != set(match_counts) or any(
            set(actual_counts[player_key])
            != set(range(start, start + declared_count))
            for player_key, declared_count in match_counts.items()
        ):
            raise ValueError(
                "merged fixed collection plan trajectory coverage is incomplete"
            )


def _merged_loss_eligibility(
    metadata: Mapping[str, object],
    archive: Mapping[str, np.ndarray],
    tensors: V4TrajectoryTensors,
) -> V4LossEligibility:
    valid_masks = tensors.valid_masks
    contract = metadata.get("lossEligibility")
    expected_semantics = {
        "behaviorCloning": "exact Normal expert label",
        "ppo": "on-policy PPO collector samples only",
        "critic": "on-policy PPO collector samples only",
    }
    if (
        not isinstance(contract, Mapping)
        or contract.get("version") != V4_LOSS_ELIGIBILITY_VERSION
        or contract.get("masks") != V4_LOSS_MASK_NAMES
        or contract.get("semantics") != expected_semantics
        or not isinstance(contract.get("eligibleSampleCounts"), Mapping)
    ):
        raise ValueError("merged V4 data lacks an exact loss eligibility contract")
    missing = sorted(set(V4_LOSS_MASK_NAMES.values()) - set(archive.keys()))
    if missing:
        raise ValueError(f"merged V4 data lacks eligibility mask {missing[0]}")
    masks: dict[str, torch.Tensor] = {}
    for semantic, name in V4_LOSS_MASK_NAMES.items():
        array = archive[name]
        if array.dtype != np.dtype(np.bool_):
            raise ValueError(f"merged eligibility array {name} must use bool")
        masks[semantic] = torch.from_numpy(np.array(array, copy=True))
    counts = contract["eligibleSampleCounts"]
    for semantic, mask in masks.items():
        count = counts.get(semantic)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("merged V4 eligibility counts must be integers")
        if count != int(mask.sum()):
            raise ValueError(
                f"merged V4 {semantic} eligibility count does not match its mask"
            )
    actor_hashes = contract.get("ppoBehaviorActorSha256s", [])
    if not isinstance(actor_hashes, list) or any(
        not _is_sha256(value) for value in actor_hashes
    ):
        raise ValueError("merged V4 PPO behavior Actor bindings are invalid")
    (
        source_contracts,
        requires_balanced,
        requires_qboost_zero,
        reward_contracts,
        behavior_contracts,
    ) = validate_merged_ppo_provenance(contract, actor_hashes)
    actor_config_metadata = metadata.get("actorConfig")
    if (
        V4_FIXED_PPO_SOURCE_CONTRACT in source_contracts
        and (
            not isinstance(actor_config_metadata, Mapping)
            or isinstance(actor_config_metadata.get("dropout"), bool)
            or not isinstance(actor_config_metadata.get("dropout"), (int, float))
            or float(actor_config_metadata.get("dropout", math.nan)) != 0.0
        )
    ):
        raise ValueError("merged fixed-match PPO requires actor dropout=0.0")
    collection_plans = validate_merged_fixed_collection_plans(
        contract,
        has_fixed_source=V4_FIXED_PPO_SOURCE_CONTRACT in source_contracts,
    )
    ppo_trajectories = masks["ppo"].any(dim=1).numpy()
    if V4_FIXED_PPO_SOURCE_CONTRACT in source_contracts:
        required_fixed = {
            "trajectory_complete_match_ids", "trajectory_candidate_ids",
            "trajectory_act_finish_orders", "trajectory_act_chip_awards_by_physical_id",
            "trajectory_suffix_total_returns",
        }
        if not required_fixed <= set(archive.keys()):
            raise ValueError("merged fixed-match PPO provenance lacks fixed auxiliary arrays")
        declared_fixed_rows = np.asarray([
            bool(str(value)) for value in archive["trajectory_complete_match_ids"]
        ])
        if np.any(declared_fixed_rows & ~ppo_trajectories):
            raise ValueError(
                "merged fixed-match identity is attached to a non-PPO trajectory"
            )
        fixed_rows = declared_fixed_rows & ppo_trajectories
        if source_contracts == (V4_FIXED_PPO_SOURCE_CONTRACT,) and not np.array_equal(fixed_rows, ppo_trajectories):
            raise ValueError("fixed-only merged PPO contains a trajectory without fixed-match identity")
        if V4_LEGACY_PPO_SOURCE_CONTRACT in source_contracts and (
            not fixed_rows.any() or not (ppo_trajectories & ~fixed_rows).any()
        ):
            raise ValueError("mixed merged PPO source contracts lack both trajectory kinds")
        reward_records = contract.get("ppoRewardContractRecords")
        assert isinstance(reward_records, list) and len(reward_records) == 1
        canonical_fields = reward_records[0].get("canonicalFields")
        assert isinstance(canonical_fields, Mapping)
        coefficient = float(canonical_fields["pairwiseCoefficient"])
        _validate_merged_fixed_match_raw_contract(
            archive,
            tensors,
            fixed_rows,
            coefficient,
            collection_plans,
        )
    elif "trajectory_complete_match_ids" in archive and np.any(
        np.asarray([bool(str(value)) for value in archive["trajectory_complete_match_ids"]])
        & ppo_trajectories
    ):
        raise ValueError("legacy-only merged PPO unexpectedly carries fixed-match identities")
    returns_contract = metadata.get("returnsAndAdvantages")
    if not isinstance(returns_contract, Mapping):
        raise ValueError("merged V4 data lacks its v2 global advantage contract")
    validate_merged_ppo_advantages(archive, returns_contract)
    eligibility = V4LossEligibility(
        behavior_cloning=masks["behaviorCloning"],
        ppo=masks["ppo"],
        critic=masks["critic"],
        preparation_format=V4_MERGED_PREPARATION_FORMAT,
        preparation_version=1,
        behavior_actor_sha256s=tuple(actor_hashes),
        ppo_source_contracts=source_contracts,
        requires_player_count_balanced_loss=requires_balanced,
        requires_qboost_coefficient_zero=requires_qboost_zero,
        ppo_reward_contracts=reward_contracts,
        ppo_behavior_policy_contracts=behavior_contracts,
        fixed_collection_plan_ids=tuple(
            str(plan["opaqueId"]) for plan in collection_plans
        ),
    )
    return _validate_loss_masks(eligibility, valid_masks)


class V4TrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        tensors: V4TrajectoryTensors,
        actor_config: V4ActorConfig,
        critic_config: V4CriticConfig,
        *,
        loss_eligibility: V4LossEligibility | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.tensors = tensors
        self.actor_config = actor_config
        self.critic_config = critic_config
        if tensors.actions.ndim != 2 or tensors.actions.shape[0] < 1 or tensors.actions.shape[1] < 1:
            raise ValueError("V4 dataset requires non-empty [trajectory, time] actions")
        trajectory_count, time_steps = tensors.actions.shape
        expected = _expected_shapes(
            trajectory_count, time_steps, actor_config, critic_config
        )
        boolean_names = {
            "player_mask", "history_mask", "legal_masks", "dones", "valid_masks"
        }
        integer_names = {"actions", "expert_actions"}
        for field in fields(tensors):
            name = field.name
            tensor = getattr(tensors, name)
            if tensor.shape != expected[name]:
                raise ValueError(f"{name} shape does not match the V4 dataset contract")
            if name in boolean_names and tensor.dtype != torch.bool:
                raise ValueError(f"{name} must use torch.bool")
            if name in integer_names and tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")
            if name not in boolean_names | integer_names and not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
        if (tensors.valid_masks[:, 1:] & ~tensors.valid_masks[:, :-1]).any():
            raise ValueError("valid trajectory samples must be contiguous prefixes")
        if not tensors.valid_masks[:, 0].all():
            raise ValueError("every trajectory requires one valid sample")
        valid_lengths = tensors.valid_masks.sum(dim=1)
        last_indices = valid_lengths - 1
        last_is_terminal = tensors.dones.gather(1, last_indices[:, None]).squeeze(1)
        terminal_counts = (tensors.dones & tensors.valid_masks).sum(dim=1)
        if not last_is_terminal.all() or not (terminal_counts == 1).all():
            raise ValueError(
                "each V4 trajectory must have exactly one terminal final sample"
            )
        for action_name in ("actions", "expert_actions"):
            actions = getattr(tensors, action_name)
            safe_actions = actions.clamp(0, V4_ACTION_COUNT - 1)
            selected_legal = tensors.legal_masks.gather(
                -1, safe_actions.unsqueeze(-1)
            ).squeeze(-1)
            if (
                ((actions < 0) | (actions >= V4_ACTION_COUNT) | ~selected_legal)
                & tensors.valid_masks
            ).any():
                raise ValueError(f"{action_name} contains an invalid legal action")
        if not (tensors.legal_masks.any(dim=-1) | ~tensors.valid_masks).all():
            raise ValueError("each valid sample requires a legal action")
        self.fingerprint = fingerprint_v4_tensors(tensors)
        self.metadata = dict(metadata or {})
        self.loss_eligibility = (
            _validate_loss_masks(loss_eligibility, tensors.valid_masks)
            if loss_eligibility is not None
            else None
        )
        self.loss_contract_fingerprint = (
            _loss_contract_fingerprint(self.loss_eligibility)
            if self.loss_eligibility is not None
            else None
        )

    def __len__(self) -> int:
        return self.tensors.actions.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        result = {
            field.name: getattr(self.tensors, field.name)[index]
            for field in fields(self.tensors)
        }
        if self.loss_eligibility is not None:
            for name, mask in self.loss_eligibility.masks().items():
                result[name] = mask[index]
        return result


def fingerprint_v4_tensors(tensors: V4TrajectoryTensors) -> str:
    digest = hashlib.sha256()
    for field in fields(tensors):
        tensor = getattr(tensors, field.name).detach().cpu().contiguous()
        digest.update(field.name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def create_v4_smoke_dataset(
    actor_config: V4ActorConfig,
    critic_config: V4CriticConfig,
    *,
    trajectories: int = 4,
    time_steps: int = 4,
    seed: int = 20260801,
) -> V4TrajectoryDataset:
    if trajectories < 1 or time_steps < 1:
        raise ValueError("smoke dataset dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    prefix = (trajectories, time_steps)
    global_features = torch.randn(
        *prefix, actor_config.global_features, generator=generator
    )
    rank_features = torch.randn(
        *prefix,
        actor_config.rank_tokens,
        actor_config.rank_features,
        generator=generator,
    )
    player_features = torch.randn(
        *prefix,
        actor_config.max_players,
        actor_config.player_features,
        generator=generator,
    )
    player_mask = torch.ones(*prefix, actor_config.max_players, dtype=torch.bool)
    memory_trace_features = torch.randn(
        *prefix,
        actor_config.memory_tokens,
        actor_config.memory_features,
        generator=generator,
    )
    history_features = torch.randn(
        *prefix,
        actor_config.max_history,
        actor_config.history_features,
        generator=generator,
    )
    history_mask = torch.ones(*prefix, actor_config.max_history, dtype=torch.bool)
    legal_masks = torch.zeros(*prefix, V4_ACTION_COUNT, dtype=torch.bool)
    random_legal = torch.rand(*prefix, V4_ACTION_COUNT, generator=generator) < 0.08
    legal_masks |= random_legal
    legal_masks[..., 0] = True
    legal_scores = torch.rand(*prefix, V4_ACTION_COUNT, generator=generator)
    legal_scores = legal_scores.masked_fill(~legal_masks, -1.0)
    actions = legal_scores.argmax(dim=-1)
    expert_scores = torch.rand(*prefix, V4_ACTION_COUNT, generator=generator)
    expert_scores = expert_scores.masked_fill(~legal_masks, -1.0)
    expert_actions = expert_scores.argmax(dim=-1)
    behavior_logits = torch.randn(*prefix, V4_ACTION_COUNT, generator=generator)
    behavior_logits = behavior_logits.masked_fill(~legal_masks, -1.0e9)
    old_log_probs = torch.log_softmax(behavior_logits, dim=-1).gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    advantages = torch.randn(*prefix, generator=generator)
    rewards = torch.randn(*prefix, generator=generator) * 0.1
    dones = torch.zeros(*prefix, dtype=torch.bool)
    dones[:, -1] = True
    valid_masks = torch.ones(*prefix, dtype=torch.bool)
    privileged_states = torch.randn(
        *prefix, critic_config.privileged_features, generator=generator
    )
    valid_masks = torch.ones(*prefix, dtype=torch.bool)
    loss_eligibility = V4LossEligibility(
        behavior_cloning=valid_masks.clone(),
        ppo=torch.zeros_like(valid_masks),
        critic=torch.zeros_like(valid_masks),
        preparation_format=V4_SMOKE_PREPARATION_FORMAT,
        preparation_version=1,
    )
    return V4TrajectoryDataset(
        V4TrajectoryTensors(
            global_features,
            rank_features,
            player_features,
            player_mask,
            memory_trace_features,
            history_features,
            history_mask,
            legal_masks,
            actions,
            expert_actions,
            old_log_probs,
            advantages,
            rewards,
            dones,
            valid_masks,
            privileged_states,
        ),
        actor_config,
        critic_config,
        loss_eligibility=loss_eligibility,
        metadata={
            "preparationFormat": V4_SMOKE_PREPARATION_FORMAT,
            "preparationVersion": 1,
        },
    )


def save_v4_dataset_npz(
    dataset: V4TrajectoryDataset, output_path: str | Path
) -> None:
    if dataset.loss_eligibility is None:
        raise ValueError("cannot save a V4 dataset without a bound loss contract")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        field.name: getattr(dataset.tensors, field.name).cpu().numpy()
        for field in fields(dataset.tensors)
    }
    metadata = {
        "format": V4_DATASET_FORMAT,
        "version": V4_DATASET_VERSION,
        "actorConfig": dataset.actor_config.to_dict(),
        "criticConfig": dataset.critic_config.to_dict(),
        "fingerprint": dataset.fingerprint,
        "preparationFormat": dataset.loss_eligibility.preparation_format,
        "preparationVersion": dataset.loss_eligibility.preparation_version,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
    }
    arrays["metadata_json"] = np.array(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(output, **arrays)


def load_v4_dataset_npz(path: str | Path) -> V4TrajectoryDataset:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if (
            metadata.get("format") != V4_DATASET_FORMAT
            or metadata.get("version") != V4_DATASET_VERSION
        ):
            raise ValueError("unsupported V4 dataset contract")
        actor_config = V4ActorConfig(**metadata["actorConfig"])
        critic_config = V4CriticConfig(**metadata["criticConfig"])
        boolean_names = {
            "player_mask", "history_mask", "legal_masks", "dones", "valid_masks"
        }
        integer_names = {"actions", "expert_actions"}
        tensors: dict[str, torch.Tensor] = {}
        for field in fields(V4TrajectoryTensors):
            array = archive[field.name]
            if field.name in boolean_names:
                tensor = torch.from_numpy(array.astype(np.bool_, copy=False))
            elif field.name in integer_names:
                tensor = torch.from_numpy(array.astype(np.int64, copy=False))
            else:
                tensor = torch.from_numpy(array.astype(np.float32, copy=False))
            tensors[field.name] = tensor
        tensor_values = V4TrajectoryTensors(**tensors)
        if metadata.get("preparationFormat") == V4_MERGED_PREPARATION_FORMAT:
            eligibility = _merged_loss_eligibility(
                metadata, archive, tensor_values
            )
        else:
            eligibility = _direct_loss_eligibility(
                metadata, archive, tensor_values
            )
    dataset = V4TrajectoryDataset(
        tensor_values,
        actor_config,
        critic_config,
        loss_eligibility=eligibility,
        metadata=metadata,
    )
    if metadata.get("fingerprint") != dataset.fingerprint:
        raise ValueError("V4 dataset fingerprint does not match")
    declared_loss_fingerprint = metadata.get("lossContractFingerprint")
    if (
        declared_loss_fingerprint is not None
        and declared_loss_fingerprint != dataset.loss_contract_fingerprint
    ):
        raise ValueError("V4 dataset loss contract fingerprint does not match")
    return dataset


__all__ = [
    "V4_DATASET_FORMAT",
    "V4_DATASET_VERSION",
    "V4_DAGGER_PREPARATION_FORMAT",
    "V4_FIXED_COLLECTION_PLAN_ID",
    "V4_FIXED_MIXED_BACKEND_COLLECTION_PLAN_ID",
    "V4_FIXED_MATCH_PPO_PREPARATION_FORMAT",
    "V4_FIXED_PPO_BEHAVIOR_POLICY_CONTRACT_ID",
    "V4_FIXED_PPO_REWARD_CONTRACT_ID",
    "V4_FIXED_PPO_SOURCE_CONTRACT",
    "V4_LEGACY_PPO_SOURCE_CONTRACT",
    "V4_LOSS_ELIGIBILITY_VERSION",
    "V4_LOSS_MASK_NAMES",
    "V4_MERGED_PREPARATION_FORMAT",
    "V4_NORMAL_PREPARATION_FORMAT",
    "V4_PPO_PREPARATION_FORMAT",
    "V4_SMOKE_PREPARATION_FORMAT",
    "V4LossEligibility",
    "V4PublicTensors",
    "V4TrajectoryDataset",
    "V4TrajectoryTensors",
    "canonical_fixed_ppo_behavior_policy_contract",
    "canonical_fixed_collection_plan",
    "canonical_fixed_ppo_reward_contract",
    "complete_fixed_collection_plan_record",
    "create_v4_smoke_dataset",
    "fingerprint_v4_tensors",
    "fixed_collection_plan_sha256",
    "fixed_match_shard_identity_sha256",
    "load_v4_dataset_npz",
    "save_v4_dataset_npz",
    "tensorize_v4_public_observation",
    "validate_merged_ppo_provenance",
    "validate_merged_fixed_collection_plans",
]
