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
)
from v4_env import (
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
    PRIVILEGED_STATE_SIZE,
)
from v4_ppo_advantages import validate_merged_ppo_advantages


V4_DATASET_FORMAT = "dalmuti-v4-trajectory-npz"
V4_DATASET_VERSION = 1

# Preparation formats are part of the loss contract, not merely descriptive
# provenance.  In particular, the legacy Normal and DAgger tensors contain
# finite placeholders in the PPO columns; those values must never silently
# become PPO/critic training targets.
V4_NORMAL_PREPARATION_FORMAT = "dalmuti-v4-prepared-dataset-metadata"
V4_DAGGER_PREPARATION_FORMAT = "dalmuti-v4-dagger-direct-npz"
V4_PPO_PREPARATION_FORMAT = "dalmuti-v4-ppo-league-direct-npz"
V4_MERGED_PREPARATION_FORMAT = "dalmuti-v4-merged-prepared-dataset-metadata"
V4_SMOKE_PREPARATION_FORMAT = "dalmuti-v4-smoke-generated"
V4_LOSS_ELIGIBILITY_VERSION = 1
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
    return digest.hexdigest()


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
        ),
        valid_masks,
    )


def _merged_loss_eligibility(
    metadata: Mapping[str, object],
    archive: Mapping[str, np.ndarray],
    valid_masks: torch.Tensor,
) -> V4LossEligibility:
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
                metadata, archive, tensor_values.valid_masks
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
    "create_v4_smoke_dataset",
    "fingerprint_v4_tensors",
    "load_v4_dataset_npz",
    "save_v4_dataset_npz",
    "tensorize_v4_public_observation",
]
