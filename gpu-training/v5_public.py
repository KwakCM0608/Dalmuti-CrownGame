from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import math
import struct
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from v5_contract import (
    V5_ACTION_COUNT,
    V5_CLEAR_REASONS,
    V5_DECK_COUNTS,
    V5_DECK_SIZE,
    V5_EVENT_TYPES,
    V5_GLOBAL_FIELDS,
    V5_HISTORY_FIELDS,
    V5_MAX_HISTORY,
    V5_MAX_OPPONENTS,
    V5_MAX_PLAYERS,
    V5_MIN_PLAYERS,
    V5_PASS_REASONS,
    V5_PLAYER_FIELDS,
    V5_PUBLIC_CONTRACT_SHA256,
    V5_PUBLIC_SCHEMA_VERSION,
    V5_RANK_COUNT,
    V5_TABLE_FIELDS,
)


_UINT8 = np.dtype(np.uint8)
_INT32 = np.dtype(np.int32)
_INT64 = np.dtype(np.int64)
_BOOL = np.dtype(np.bool_)
_FLOAT32 = np.dtype(np.float32)

_OBSERVATION_FIELDS = {
    "global_codes",
    "own_rank_counts",
    "public_played_counts",
    "player_codes",
    "player_mask",
    "table_codes",
    "history_codes",
    "history_mask",
    "legal_mask",
}

_PACKED_PUBLIC_KEYS = {
    "global_codes",
    "own_rank_counts",
    "public_played_counts",
    "player_codes",
    "player_masks",
    "table_codes",
    "legal_action_bits",
    "belief_response_feasibility",
    "history_events",
    "history_end",
}

# These are public training labels/indices published by v5_dataset alongside
# the public features.  The batch decoder may receive the complete actor mmap
# mapping, but no arbitrary key and no critic/private partition.
_KNOWN_PACKED_ACTOR_AUXILIARY_KEYS = {
    "match_offsets",
    "candidate_bitsets",
    "player_counts",
    "decision_actor_ids",
    "decision_acts",
    "normal_actions",
    "actions",
    "old_log_probs",
    "old_values",
    "reward_to_next",
    "done",
    "forced",
    "next_decision",
    "selected_action_probabilities",
    "policy_entropies",
    "advantages",
    "returns",
    "deltas",
    "policy_mask",
    "value_mask",
    "policy_loss_weights",
    "value_loss_weights",
}

_PACKED_DECISION_AUXILIARY_DTYPES = {
    "decision_actor_ids": np.dtype(np.uint8),
    "decision_acts": np.dtype(np.uint8),
    "normal_actions": np.dtype(np.uint16),
    "actions": np.dtype(np.uint16),
    "old_log_probs": _FLOAT32,
    "old_values": _FLOAT32,
    "reward_to_next": _FLOAT32,
    "done": _BOOL,
    "forced": _BOOL,
    "next_decision": np.dtype(np.int32),
    "selected_action_probabilities": _FLOAT32,
    "policy_entropies": _FLOAT32,
    "advantages": _FLOAT32,
    "returns": _FLOAT32,
    "deltas": _FLOAT32,
    "policy_mask": _BOOL,
    "value_mask": _BOOL,
    "policy_loss_weights": _FLOAT32,
    "value_loss_weights": _FLOAT32,
}


@dataclass(frozen=True)
class V5PublicObservation:
    """The complete V5 actor boundary for one decision.

    The class deliberately has no field capable of carrying opponent card
    identities or rank counts.  ``from_mapping`` also rejects unknown keys, so
    adding a private payload to an otherwise valid record fails closed.
    """

    global_codes: np.ndarray
    own_rank_counts: np.ndarray
    public_played_counts: np.ndarray
    player_codes: np.ndarray
    player_mask: np.ndarray
    table_codes: np.ndarray
    history_codes: np.ndarray
    history_mask: np.ndarray
    legal_mask: np.ndarray

    @property
    def player_count(self) -> int:
        return int(self.global_codes[1])

    @property
    def opponent_remaining_counts(self) -> np.ndarray:
        return self.player_codes[1 : self.player_count, 1]


@dataclass(frozen=True)
class V5BeliefFeatures:
    unknown_rank_counts: np.ndarray
    expected_counts: np.ndarray
    probability_at_least_one: np.ndarray
    probability_at_least_required: np.ndarray
    response_feasibility: np.ndarray
    opponent_mask: np.ndarray


@dataclass(frozen=True)
class V5ActorPublicTensors:
    global_codes: torch.Tensor
    own_rank_counts: torch.Tensor
    public_played_counts: torch.Tensor
    player_codes: torch.Tensor
    player_mask: torch.Tensor
    table_codes: torch.Tensor
    history_codes: torch.Tensor
    history_mask: torch.Tensor
    legal_mask: torch.Tensor
    belief_unknown_rank_counts: torch.Tensor
    belief_expected_counts: torch.Tensor
    belief_probability_at_least_one: torch.Tensor
    belief_probability_at_least_required: torch.Tensor
    belief_response_feasibility: torch.Tensor
    opponent_mask: torch.Tensor


@dataclass(frozen=True)
class V5ActorPublicBatch:
    global_codes: torch.Tensor
    own_rank_counts: torch.Tensor
    public_played_counts: torch.Tensor
    player_codes: torch.Tensor
    player_mask: torch.Tensor
    table_codes: torch.Tensor
    history_codes: torch.Tensor
    history_mask: torch.Tensor
    legal_mask: torch.Tensor
    belief_unknown_rank_counts: torch.Tensor
    belief_expected_counts: torch.Tensor
    belief_probability_at_least_one: torch.Tensor
    belief_probability_at_least_required: torch.Tensor
    belief_response_feasibility: torch.Tensor
    opponent_mask: torch.Tensor


def _require_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[object],
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _prefix_mask(mask: np.ndarray, expected: int, name: str) -> None:
    canonical = np.arange(mask.size) < expected
    if not np.array_equal(mask, canonical):
        raise ValueError(f"{name} must be a right-padded contiguous prefix")


def _role_for_social_index(index: int, player_count: int) -> int:
    if index == 0:
        return 0
    if index == 1:
        return 1
    if index == player_count - 2:
        return 3
    if index == player_count - 1:
        return 4
    return 2


def _validate_bundle(
    rank: int,
    natural: int,
    jokers: int,
    total: int,
    label: str,
) -> None:
    if not 1 <= rank <= 13:
        raise ValueError(f"{label} rank must be from 1 through 13")
    if not 0 <= jokers <= 2 or total != natural + jokers:
        raise ValueError(f"{label} has an invalid joker or total count")
    if rank == 13:
        if (natural, jokers, total) != (0, 1, 1):
            raise ValueError(f"{label} rank 13 must be one solo joker")
    elif not (1 <= natural <= rank and 1 <= total <= 14):
        raise ValueError(f"{label} has an invalid natural-card count")


def _expected_legal_mask(
    own_rank_counts: np.ndarray, table_codes: np.ndarray
) -> np.ndarray:
    result = np.zeros(V5_ACTION_COUNT, dtype=np.bool_)
    present, table_rank, required, _, _, _ = (
        int(value) for value in table_codes
    )
    if present:
        result[0] = True
    elif int(own_rank_counts[12]) > 0:
        result[1] = True
    action_index = 2
    available_jokers = int(own_rank_counts[12])
    for rank in range(1, 13):
        for natural in range(1, rank + 1):
            for jokers in range(3):
                available = (
                    int(own_rank_counts[rank - 1]) >= natural
                    and available_jokers >= jokers
                )
                if present:
                    legal_position = rank < table_rank and natural + jokers == required
                else:
                    legal_position = True
                result[action_index] = available and legal_position
                action_index += 1
    if action_index != V5_ACTION_COUNT:
        raise AssertionError("the fixed 236-action catalogue is incomplete")
    return result


def v5_legal_mask_from_public_cards(
    own_rank_counts: np.ndarray,
    table_codes: np.ndarray,
) -> np.ndarray:
    """Build the fixed action mask without consulting any opponent cards."""

    own = _require_array(
        own_rank_counts,
        name="own_rank_counts",
        dtype=_UINT8,
        shape=(V5_RANK_COUNT,),
    )
    table = _require_array(
        table_codes,
        name="table_codes",
        dtype=_UINT8,
        shape=(len(V5_TABLE_FIELDS),),
    )
    deck = np.asarray(V5_DECK_COUNTS, dtype=np.uint8)
    if np.any(own > deck):
        raise ValueError("own rank counts exceed the physical deck")
    present, rank, required, natural, jokers, _ = (
        int(value) for value in table
    )
    if present not in (0, 1):
        raise ValueError("table present flag must be binary")
    if not present:
        if np.any(table != 0):
            raise ValueError("an empty table must use all-zero table codes")
    else:
        _validate_bundle(rank, natural, jokers, required, "table_codes")
    return _expected_legal_mask(own, table)


def _validate_history(
    codes: np.ndarray, mask: np.ndarray, player_count: int
) -> None:
    history_count = int(mask.sum())
    _prefix_mask(mask, history_count, "history_mask")
    if np.any(codes[history_count:] != 0):
        raise ValueError("masked history rows must be all-zero padding")
    for index, raw in enumerate(codes[:history_count]):
        (
            event_type,
            actor_offset,
            before,
            after,
            rank,
            natural,
            jokers,
            total,
            pass_reason,
            clear_reason,
            next_leader_plus_one,
            finish_place,
        ) = (int(value) for value in raw)
        label = f"history_codes[{index}]"
        if not 1 <= event_type <= 4:
            raise ValueError(f"{label} has an invalid event type")
        if not 0 <= actor_offset < player_count:
            raise ValueError(f"{label} has an invalid actor offset")
        if before > 20 or after > 20:
            raise ValueError(f"{label} hand counts cannot exceed 20")
        if next_leader_plus_one > player_count:
            raise ValueError(f"{label} has an invalid next-leader offset")
        if finish_place > player_count:
            raise ValueError(f"{label} has an invalid finish place")

        if event_type == V5_EVENT_TYPES["play"]:
            _validate_bundle(rank, natural, jokers, total, label)
            if before - after != total:
                raise ValueError(f"{label} play transition must equal total")
            if pass_reason or clear_reason or next_leader_plus_one or finish_place:
                raise ValueError(f"{label} play has non-play categorical fields")
        elif event_type == V5_EVENT_TYPES["pass"]:
            if before != after or not 1 <= pass_reason <= 4:
                raise ValueError(f"{label} has an invalid pass transition")
            if any((rank, natural, jokers, total, clear_reason, next_leader_plus_one, finish_place)):
                raise ValueError(f"{label} pass has non-pass categorical fields")
        elif event_type == V5_EVENT_TYPES["clear"]:
            _validate_bundle(rank, natural, jokers, total, label)
            if before != after or not 1 <= clear_reason <= 3:
                raise ValueError(f"{label} has an invalid clear transition")
            if pass_reason or finish_place:
                raise ValueError(f"{label} clear has non-clear categorical fields")
        else:
            if before or after or not 1 <= finish_place <= player_count:
                raise ValueError(f"{label} has an invalid finish transition")
            if any((rank, natural, jokers, total, pass_reason, clear_reason, next_leader_plus_one)):
                raise ValueError(f"{label} finish has non-finish categorical fields")


def _validate_batched_history_codes(
    codes: np.ndarray,
    mask: np.ndarray,
    player_counts: np.ndarray,
) -> None:
    """Vectorized hot-path equivalent of ``_validate_history``."""

    if np.any(codes[~mask] != 0):
        raise ValueError("packed masked history rows must be zero")
    if not mask.any():
        return
    active = codes[mask].astype(np.int16, copy=False)
    repeated_players = np.broadcast_to(
        player_counts[:, None], mask.shape
    )[mask].astype(np.int16, copy=False)
    event = active[:, 0]
    if (
        np.any(event < 1)
        or np.any(event > 4)
        or np.any(active[:, 1] >= repeated_players)
        or np.any(active[:, 2:4] > 20)
        or np.any(active[:, 4] > 13)
        or np.any(active[:, 5] > 12)
        or np.any(active[:, 6] > 2)
        or np.any(active[:, 7] > 14)
        or np.any(active[:, 8] > 4)
        or np.any(active[:, 9] > 3)
        or np.any(active[:, 10] > repeated_players)
        or np.any(active[:, 11] > repeated_players)
    ):
        raise ValueError("packed history categorical value escaped its range")
    rank = active[:, 4]
    natural = active[:, 5]
    jokers = active[:, 6]
    total = active[:, 7]
    bundle_valid = (
        (total == natural + jokers)
        & (
            (
                (rank == 13)
                & (natural == 0)
                & (jokers == 1)
                & (total == 1)
            )
            | (
                (rank >= 1)
                & (rank <= 12)
                & (natural >= 1)
                & (natural <= rank)
            )
        )
    )
    play = event == V5_EVENT_TYPES["play"]
    passed = event == V5_EVENT_TYPES["pass"]
    clear = event == V5_EVENT_TYPES["clear"]
    finish = event == V5_EVENT_TYPES["finish"]
    if np.any(
        play
        & (
            ~bundle_valid
            | (active[:, 2] - active[:, 3] != total)
            | np.any(active[:, 8:12] != 0, axis=1)
        )
    ):
        raise ValueError("packed play history row violates its event contract")
    if np.any(
        passed
        & (
            (active[:, 2] != active[:, 3])
            | (active[:, 8] < 1)
            | np.any(active[:, 4:8] != 0, axis=1)
            | np.any(active[:, 9:12] != 0, axis=1)
        )
    ):
        raise ValueError("packed pass history row violates its event contract")
    if np.any(
        clear
        & (
            ~bundle_valid
            | (active[:, 2] != active[:, 3])
            | (active[:, 9] < 1)
            | (active[:, 8] != 0)
            | (active[:, 11] != 0)
        )
    ):
        raise ValueError("packed clear history row violates its event contract")
    if np.any(
        finish
        & (
            (active[:, 2] != 0)
            | (active[:, 3] != 0)
            | (active[:, 11] < 1)
            | np.any(active[:, 4:11] != 0, axis=1)
        )
    ):
        raise ValueError("packed finish history row violates its event contract")


def validate_v5_public_observation(
    value: V5PublicObservation,
) -> V5PublicObservation:
    if type(value) is not V5PublicObservation:
        raise TypeError("value must be exactly V5PublicObservation")
    global_codes = _require_array(
        value.global_codes,
        name="global_codes",
        dtype=_INT32,
        shape=(len(V5_GLOBAL_FIELDS),),
    )
    own = _require_array(
        value.own_rank_counts,
        name="own_rank_counts",
        dtype=_UINT8,
        shape=(V5_RANK_COUNT,),
    )
    played = _require_array(
        value.public_played_counts,
        name="public_played_counts",
        dtype=_UINT8,
        shape=(V5_RANK_COUNT,),
    )
    players = _require_array(
        value.player_codes,
        name="player_codes",
        dtype=_UINT8,
        shape=(V5_MAX_PLAYERS, len(V5_PLAYER_FIELDS)),
    )
    player_mask = _require_array(
        value.player_mask,
        name="player_mask",
        dtype=_BOOL,
        shape=(V5_MAX_PLAYERS,),
    )
    table = _require_array(
        value.table_codes,
        name="table_codes",
        dtype=_UINT8,
        shape=(len(V5_TABLE_FIELDS),),
    )
    history = _require_array(
        value.history_codes,
        name="history_codes",
        dtype=_UINT8,
        shape=(V5_MAX_HISTORY, len(V5_HISTORY_FIELDS)),
    )
    history_mask = _require_array(
        value.history_mask,
        name="history_mask",
        dtype=_BOOL,
        shape=(V5_MAX_HISTORY,),
    )
    legal = _require_array(
        value.legal_mask,
        name="legal_mask",
        dtype=_BOOL,
        shape=(V5_ACTION_COUNT,),
    )

    schema, player_count, act, actor_role, revolution, truncated = (
        int(item) for item in global_codes
    )
    if schema != V5_PUBLIC_SCHEMA_VERSION:
        raise ValueError("global_codes has the wrong schema version")
    if not V5_MIN_PLAYERS <= player_count <= V5_MAX_PLAYERS:
        raise ValueError("player count must be from 4 through 10")
    if not 1 <= act <= 1_000_000:
        raise ValueError("act must be from 1 through 1000000")
    if not 0 <= actor_role <= 4 or not 0 <= revolution <= 2:
        raise ValueError("actor role or revolution category is invalid")
    if not 0 <= truncated <= 1_000_000_000:
        raise ValueError("truncated history count is invalid")
    _prefix_mask(player_mask, player_count, "player_mask")
    if np.any(players[player_count:] != 0):
        raise ValueError("masked player rows must be all-zero padding")

    deck = np.asarray(V5_DECK_COUNTS, dtype=np.int16)
    if np.any(own.astype(np.int16) > deck):
        raise ValueError("own rank counts exceed the physical deck")
    if np.any(played.astype(np.int16) > deck):
        raise ValueError("public played counts exceed the physical deck")
    if np.any(own.astype(np.int16) + played.astype(np.int16) > deck):
        raise ValueError("own and public cards exceed the physical deck")

    active_players = players[:player_count]
    if not np.array_equal(active_players[:, 0], np.arange(player_count, dtype=np.uint8)):
        raise ValueError("player rows must use actor-relative offsets 0..p-1")
    if np.any(active_players[:, 1] > 20):
        raise ValueError("remaining player cards cannot exceed 20")
    if np.any(active_players[:, 2] > 4):
        raise ValueError("player role category is invalid")
    if np.any(active_players[:, 3:] > 1):
        raise ValueError("player flags must be binary")
    if not np.array_equal(active_players[:, 3], active_players[:, 1] == 0):
        raise ValueError("finished flags must match zero remaining cards")
    if int(active_players[0, 2]) != actor_role:
        raise ValueError("actor role must match actor-relative player row zero")
    expected_roles = sorted(
        _role_for_social_index(index, player_count)
        for index in range(player_count)
    )
    if sorted(int(role) for role in active_players[:, 2]) != expected_roles:
        raise ValueError("player roles do not form a valid social-rank table")
    if int(own.sum(dtype=np.int64)) != int(active_players[0, 1]):
        raise ValueError("own rank counts must equal the actor remaining count")
    if int(played.sum(dtype=np.int64)) + int(
        active_players[:, 1].sum(dtype=np.int64)
    ) != V5_DECK_SIZE:
        raise ValueError("public played and remaining card counts must total 80")

    present, table_rank, required, natural, jokers, table_actor = (
        int(item) for item in table
    )
    if present not in (0, 1):
        raise ValueError("table present flag must be binary")
    if not present:
        if np.any(table != 0) or np.any(active_players[:, 5] != 0):
            raise ValueError("an empty table must use all-zero table codes")
    else:
        _validate_bundle(table_rank, natural, jokers, required, "table_codes")
        if not 0 <= table_actor < player_count:
            raise ValueError("table actor offset is invalid")
        leaders = np.flatnonzero(active_players[:, 5])
        if not np.array_equal(leaders, np.asarray([table_actor])):
            raise ValueError("table leader flag must identify the table actor")
        if natural > int(played[table_rank - 1]) or jokers > int(played[12]):
            raise ValueError("table cards must already be publicly played")

    _validate_history(history, history_mask, player_count)
    if not legal.any():
        raise ValueError("legal_mask must contain at least one legal action")
    if not np.array_equal(legal, _expected_legal_mask(own, table)):
        raise ValueError("legal_mask disagrees with own cards and public table")
    return value


def v5_public_observation_from_mapping(
    value: Mapping[str, object],
) -> V5PublicObservation:
    if not isinstance(value, Mapping):
        raise TypeError("V5 public observation must be a mapping")
    actual = set(value.keys())
    if actual != _OBSERVATION_FIELDS:
        unknown = sorted(actual - _OBSERVATION_FIELDS)
        missing = sorted(_OBSERVATION_FIELDS - actual)
        detail = unknown[0] if unknown else missing[0]
        raise ValueError(
            f"V5 public observation has an unknown or missing field: {detail}"
        )
    observation = V5PublicObservation(
        **{field.name: value[field.name] for field in fields(V5PublicObservation)}
    )
    return validate_v5_public_observation(observation)


def _hypergeometric_tail(
    population: int, successes: int, draws: int, threshold: int
) -> float:
    if not 0 <= successes <= population or not 0 <= draws <= population:
        raise ValueError("invalid hypergeometric population")
    if threshold <= 0:
        return 1.0
    lower = max(0, draws - (population - successes))
    upper = min(draws, successes)
    if threshold <= lower:
        return 1.0
    if threshold > upper:
        return 0.0
    denominator = math.comb(population, draws)
    numerator = sum(
        math.comb(successes, count)
        * math.comb(population - successes, draws - count)
        for count in range(max(lower, threshold), upper + 1)
    )
    return numerator / denominator


@lru_cache(maxsize=1)
def _hypergeometric_tail_table() -> np.ndarray:
    """Return the exact, immutable lookup used by packed-shard decoding.

    The large rank-wise belief tensors are deterministic functions of compact
    public categorical arrays.  Keeping this roughly 1.3 MiB process-local
    table is substantially cheaper than persisting three 9x13 float32 arrays
    for every training decision.  Entries are rounded to float32 only after
    the exact integer-combination calculation, matching the public contract.
    """

    maximum_successes = max(V5_DECK_COUNTS)
    maximum_draws = V5_DECK_SIZE // V5_MIN_PLAYERS
    maximum_threshold = 14
    table = np.zeros(
        (
            V5_DECK_SIZE + 1,
            maximum_successes + 1,
            maximum_draws + 1,
            maximum_threshold + 1,
        ),
        dtype=np.float32,
    )
    for population in range(V5_DECK_SIZE + 1):
        for successes in range(min(maximum_successes, population) + 1):
            for draws in range(min(maximum_draws, population) + 1):
                for threshold in range(maximum_threshold + 1):
                    table[population, successes, draws, threshold] = np.float32(
                        _hypergeometric_tail(
                            population, successes, draws, threshold
                        )
                    )
    table.setflags(write=False)
    return table


def _bounded_category_draw_ways(
    category_counts: Sequence[int],
    other_count: int,
    draws: int,
    maximum_per_category: int,
) -> int:
    ways = [0] * (draws + 1)
    ways[0] = 1
    for count in category_counts:
        updated = [0] * (draws + 1)
        maximum = min(count, maximum_per_category, draws)
        for used, prefix_ways in enumerate(ways):
            if not prefix_ways:
                continue
            for chosen in range(min(maximum, draws - used) + 1):
                updated[used + chosen] += prefix_ways * math.comb(count, chosen)
        ways = updated
    return sum(
        prefix_ways * math.comb(other_count, draws - used)
        for used, prefix_ways in enumerate(ways)
        if prefix_ways and 0 <= draws - used <= other_count
    )


def _response_probability(
    pool_counts: Sequence[int],
    hand_count: int,
    table_rank: int,
    required_count: int,
) -> float:
    population = sum(pool_counts)
    if hand_count == 0:
        return 0.0
    if table_rank == 0:
        return 1.0
    if required_count > hand_count or table_rank <= 1:
        return 0.0
    eligible = list(pool_counts[: table_rank - 1])
    if not eligible or not any(eligible):
        return 0.0
    joker_count = int(pool_counts[12])
    non_joker_population = population - joker_count
    eligible_total = sum(eligible)
    other_count = non_joker_population - eligible_total
    denominator = math.comb(population, hand_count)
    failure_ways = 0
    joker_lower = max(0, hand_count - non_joker_population)
    joker_upper = min(joker_count, hand_count)
    for drawn_jokers in range(joker_lower, joker_upper + 1):
        remaining_draws = hand_count - drawn_jokers
        # A legal bundle always needs at least one natural card.  Jokers may
        # cover the remainder of the public table requirement.
        natural_threshold = max(1, required_count - drawn_jokers)
        failure_without_jokers = _bounded_category_draw_ways(
            eligible,
            other_count,
            remaining_draws,
            natural_threshold - 1,
        )
        failure_ways += math.comb(joker_count, drawn_jokers) * failure_without_jokers
    success_ways = denominator - failure_ways
    if not 0 <= success_ways <= denominator:
        raise ArithmeticError("response probability combination count escaped range")
    return success_ways / denominator


def _expected_response_feasibility_rows(
    unknown_rank_counts: np.ndarray,
    player_codes: np.ndarray,
    table_codes: np.ndarray,
    player_counts: np.ndarray,
) -> np.ndarray:
    """Recompute the stored compact response feature from public integers.

    This deliberately uses the same exact-combination primitive as dense
    tensorization.  It is used once at shard publication and, for unverified
    ad-hoc mappings, on the selected decoder rows.  Published mmap shards carry
    a source/array-bound semantic receipt, so training microbatches do not pay
    this combinatorial check repeatedly.
    """

    row_count = int(unknown_rank_counts.shape[0])
    result = np.zeros((row_count, V5_MAX_OPPONENTS), dtype=np.float32)
    for row in range(row_count):
        player_count = int(player_counts[row])
        pool = tuple(int(value) for value in unknown_rank_counts[row])
        table = table_codes[row]
        table_present = int(table[0])
        table_rank = int(table[1]) if table_present else 0
        required = int(table[2]) if table_present else 1
        # Opponents commonly share a remaining-card count.  Reuse the exact
        # combinatorial result within a row without retaining corpus-sized
        # process caches.
        by_hand_count: dict[int, np.float32] = {}
        for opponent_index in range(player_count - 1):
            hand_count = int(player_codes[row, opponent_index + 1, 1])
            if hand_count not in by_hand_count:
                by_hand_count[hand_count] = np.float32(
                    _response_probability(
                        pool, hand_count, table_rank, required
                    )
                )
            result[row, opponent_index] = by_hand_count[hand_count]
    return result


def compute_v5_public_beliefs(
    value: V5PublicObservation,
) -> V5BeliefFeatures:
    observation = validate_v5_public_observation(value)
    unknown = (
        np.asarray(V5_DECK_COUNTS, dtype=np.int16)
        - observation.own_rank_counts.astype(np.int16)
        - observation.public_played_counts.astype(np.int16)
    )
    if np.any(unknown < 0):
        raise ValueError("public unknown rank count cannot be negative")
    population = int(unknown.sum(dtype=np.int64))
    opponent_counts = observation.opponent_remaining_counts.astype(np.int64)
    if int(opponent_counts.sum(dtype=np.int64)) != population:
        raise ValueError("opponent remaining counts must equal the unseen population")

    expected = np.zeros((V5_MAX_OPPONENTS, V5_RANK_COUNT), dtype=np.float32)
    at_least_one = np.zeros_like(expected)
    at_least_required = np.zeros_like(expected)
    response = np.zeros(V5_MAX_OPPONENTS, dtype=np.float32)
    opponent_mask = np.zeros(V5_MAX_OPPONENTS, dtype=np.bool_)
    table_present = int(observation.table_codes[0])
    table_rank = int(observation.table_codes[1]) if table_present else 0
    required = int(observation.table_codes[2]) if table_present else 1
    pool = [int(count) for count in unknown]
    response_by_hand_count: dict[int, np.float32] = {}

    for opponent_index, raw_hand_count in enumerate(opponent_counts):
        hand_count = int(raw_hand_count)
        opponent_mask[opponent_index] = True
        if population:
            expected[opponent_index] = np.asarray(
                [hand_count * count / population for count in pool],
                dtype=np.float32,
            )
            at_least_one[opponent_index] = np.asarray(
                [
                    _hypergeometric_tail(population, count, hand_count, 1)
                    for count in pool
                ],
                dtype=np.float32,
            )
            at_least_required[opponent_index] = np.asarray(
                [
                    _hypergeometric_tail(
                        population, count, hand_count, required
                    )
                    for count in pool
                ],
                dtype=np.float32,
            )
            if hand_count not in response_by_hand_count:
                response_by_hand_count[hand_count] = np.float32(
                    _response_probability(
                        pool, hand_count, table_rank, required
                    )
                )
            response[opponent_index] = response_by_hand_count[hand_count]
        elif hand_count:
            raise ValueError("a non-empty opponent cannot draw from an empty pool")

    for name, feature in (
        ("expected_counts", expected),
        ("probability_at_least_one", at_least_one),
        ("probability_at_least_required", at_least_required),
        ("response_feasibility", response),
    ):
        if not np.isfinite(feature).all():
            raise ArithmeticError(f"{name} contains a non-finite value")
    if np.any(at_least_one < 0) or np.any(at_least_one > 1):
        raise ArithmeticError("P(rank>=1) escaped [0, 1]")
    if np.any(at_least_required < 0) or np.any(at_least_required > 1):
        raise ArithmeticError("P(rank>=required) escaped [0, 1]")
    if np.any(response < 0) or np.any(response > 1):
        raise ArithmeticError("response feasibility escaped [0, 1]")
    return V5BeliefFeatures(
        unknown_rank_counts=unknown.astype(np.uint8),
        expected_counts=expected,
        probability_at_least_one=at_least_one,
        probability_at_least_required=at_least_required,
        response_feasibility=response,
        opponent_mask=opponent_mask,
    )


def tensorize_v5_public_observation(
    value: V5PublicObservation,
    *,
    device: str | torch.device | None = None,
) -> V5ActorPublicTensors:
    observation = validate_v5_public_observation(value)
    belief = compute_v5_public_beliefs(observation)
    target = torch.device("cpu" if device is None else device)

    def categorical(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.copy()).to(device=target, dtype=torch.long)

    def boolean(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.copy()).to(device=target, dtype=torch.bool)

    def continuous(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.copy()).to(device=target, dtype=torch.float32)

    return V5ActorPublicTensors(
        global_codes=categorical(observation.global_codes),
        own_rank_counts=categorical(observation.own_rank_counts),
        public_played_counts=categorical(observation.public_played_counts),
        player_codes=categorical(observation.player_codes),
        player_mask=boolean(observation.player_mask),
        table_codes=categorical(observation.table_codes),
        history_codes=categorical(observation.history_codes),
        history_mask=boolean(observation.history_mask),
        legal_mask=boolean(observation.legal_mask),
        belief_unknown_rank_counts=categorical(belief.unknown_rank_counts),
        belief_expected_counts=continuous(belief.expected_counts),
        belief_probability_at_least_one=continuous(
            belief.probability_at_least_one
        ),
        belief_probability_at_least_required=continuous(
            belief.probability_at_least_required
        ),
        belief_response_feasibility=continuous(belief.response_feasibility),
        opponent_mask=boolean(belief.opponent_mask),
    )


def stack_v5_actor_public_features(
    values: Sequence[V5ActorPublicTensors],
    *,
    device: str | torch.device | None = None,
) -> V5ActorPublicBatch:
    if not values:
        raise ValueError("cannot stack an empty V5 actor batch")
    target = None if device is None else torch.device(device)
    stacked: dict[str, torch.Tensor] = {}
    for field in fields(V5ActorPublicTensors):
        tensors = [getattr(value, field.name) for value in values]
        first = tensors[0]
        if not all(
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == first.dtype
            and tensor.shape == first.shape
            for tensor in tensors
        ):
            raise ValueError(f"incompatible V5 actor tensor field {field.name}")
        result = torch.stack(tensors)
        stacked[field.name] = result if target is None else result.to(target)
    return V5ActorPublicBatch(**stacked)


def encode_v5_public_observation_bytes(value: V5PublicObservation) -> bytes:
    observation = validate_v5_public_observation(value)
    chunks = [
        b"DALMUTI-V5-PUBLIC\x00",
        bytes.fromhex(V5_PUBLIC_CONTRACT_SHA256),
    ]
    for field in fields(V5PublicObservation):
        array = getattr(observation, field.name)
        name = field.name.encode("ascii")
        chunks.extend(
            (
                struct.pack("<H", len(name)),
                name,
                struct.pack("<Q", array.nbytes),
                array.tobytes(order="C"),
            )
        )
    return b"".join(chunks)


def _cpu_numpy_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"V4 {name} must be a torch.Tensor")
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(
            f"V4 {name} must have dtype {dtype} and shape {shape}"
        )
    return value.detach().cpu().contiguous().numpy()


def _decode_scaled(value: float, scale: int, label: str) -> int:
    decoded = int(round(float(value) * scale))
    if abs(float(value) - decoded / scale) > 2.0e-5:
        raise ValueError(f"cannot losslessly decode V4 public field {label}")
    return decoded


def _decode_one_hot(values: np.ndarray, label: str) -> int:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"V4 {label} is not a finite one-hot vector")
    index = int(np.argmax(values))
    # This decoder runs for every categorical field in every public history
    # row.  Calling np.allclose here creates several temporary arrays per
    # field and dominates large rollout collection.  The short direct scan is
    # equivalent to rtol=0/atol=2e-5 and allocation-free.
    for position, raw in enumerate(values):
        expected = 1.0 if position == index else 0.0
        if abs(float(raw) - expected) > 2.0e-5:
            raise ValueError(f"V4 {label} is not one-hot")
    return index


def v5_public_from_v4_actor_observation(value: object) -> V5PublicObservation:
    """Losslessly project the current V4 *public* actor view into V5.

    This adapter intentionally accepts ``V4ActorObservation`` rather than the
    environment/privileged wrapper.  It reads only its documented public
    tensors, reconstructs their categorical values, and validates the complete
    V5 privacy and deck invariants before returning.
    """

    # Import lazily so the V5 contract remains independently importable.  An
    # exact type check prevents a richer environment/privileged wrapper (or a
    # look-alike object with hidden-hand attributes) from crossing the actor
    # boundary.
    from v4_env import V4ActorObservation

    if type(value) is not V4ActorObservation:
        raise TypeError("value must be exactly the public V4ActorObservation")
    valid = _cpu_numpy_tensor(
        getattr(value, "valid", None), name="valid", dtype=torch.bool, shape=()
    )
    if not bool(valid):
        raise ValueError("a terminal V4 observation cannot become a V5 decision")
    global_features = _cpu_numpy_tensor(
        getattr(value, "global_features", None),
        name="global_features",
        dtype=torch.float32,
        shape=(12,),
    )
    rank_features = _cpu_numpy_tensor(
        getattr(value, "rank_features", None),
        name="rank_features",
        dtype=torch.float32,
        shape=(13, 6),
    )
    player_features = _cpu_numpy_tensor(
        getattr(value, "player_features", None),
        name="player_features",
        dtype=torch.float32,
        shape=(10, 12),
    )
    player_mask = _cpu_numpy_tensor(
        getattr(value, "player_mask", None),
        name="player_mask",
        dtype=torch.bool,
        shape=(10,),
    ).copy()
    history_features = _cpu_numpy_tensor(
        getattr(value, "history_features", None),
        name="history_features",
        dtype=torch.float32,
        shape=(192, 20),
    )
    history_mask = _cpu_numpy_tensor(
        getattr(value, "history_mask", None),
        name="history_mask",
        dtype=torch.bool,
        shape=(192,),
    ).copy()
    memory_trace_features = _cpu_numpy_tensor(
        getattr(value, "memory_trace_features", None),
        name="memory_trace_features",
        dtype=torch.float32,
        shape=(4, 20),
    )
    legal_mask = _cpu_numpy_tensor(
        getattr(value, "legal_mask", None),
        name="legal_mask",
        dtype=torch.bool,
        shape=(236,),
    ).copy()
    if not all(
        np.isfinite(array).all()
        for array in (
            global_features,
            rank_features,
            player_features,
            memory_trace_features,
            history_features,
        )
    ):
        raise ValueError("V4 public observation contains non-finite values")

    player_count = _decode_scaled(float(global_features[0] - 0.0), 6, "player-count") + 4
    # Recover the pre-tanh categorical counters.  Float32 is exact enough for
    # normal game lengths; saturated values fail instead of being guessed.
    if abs(float(global_features[1])) >= 1.0:
        raise ValueError("V4 act encoding is saturated")
    act_float = math.atanh(float(global_features[1])) * 10.0 + 1.0
    act = int(round(act_float))
    if abs(math.tanh((act - 1) / 10.0) - float(global_features[1])) > 2.0e-5:
        raise ValueError("cannot losslessly decode V4 act")
    actor_role = _decode_one_hot(global_features[2:7], "actor role")
    revolution = _decode_one_hot(global_features[7:10], "revolution")
    if abs(float(global_features[10])) >= 1.0:
        raise ValueError("V4 truncated-history encoding is saturated")
    truncated_float = math.atanh(float(global_features[10])) * 192.0
    truncated = int(round(truncated_float))
    if abs(math.tanh(truncated / 192.0) - float(global_features[10])) > 2.0e-5:
        raise ValueError("cannot losslessly decode V4 truncated history count")

    own = np.zeros(13, dtype=np.uint8)
    played = np.zeros(13, dtype=np.uint8)
    for rank_index, copies in enumerate(V5_DECK_COUNTS):
        own[rank_index] = _decode_scaled(
            float(rank_features[rank_index, 0]), copies, f"own rank {rank_index + 1}"
        )
        played[rank_index] = _decode_scaled(
            float(rank_features[rank_index, 1]), copies, f"played rank {rank_index + 1}"
        )

    player_codes = np.zeros((10, 6), dtype=np.uint8)
    for offset in range(player_count):
        row = player_features[offset]
        decoded_offset = _decode_scaled(float(row[0]), player_count - 1, "player offset")
        player_codes[offset] = (
            decoded_offset,
            _decode_scaled(float(row[1]), 20, "player remaining count"),
            _decode_one_hot(row[6:11], "player role"),
            _decode_scaled(float(row[2]), 1, "finished flag"),
            _decode_scaled(float(row[3]), 1, "passed flag"),
            _decode_scaled(float(row[5]), 1, "table leader flag"),
        )

    table_flags = [
        rank_index
        for rank_index in range(13)
        if _decode_scaled(float(rank_features[rank_index, 2]), 1, "table rank flag")
    ]
    table_codes = np.zeros(6, dtype=np.uint8)
    if table_flags:
        if len(table_flags) != 1:
            raise ValueError("V4 public observation has multiple table ranks")
        rank_index = table_flags[0]
        natural = _decode_scaled(float(rank_features[rank_index, 3]), 14, "table natural count")
        jokers = _decode_scaled(float(rank_features[rank_index, 4]), 2, "table joker count")
        leaders = np.flatnonzero(player_codes[:player_count, 5])
        if len(leaders) != 1:
            raise ValueError("V4 public observation has no unique table leader")
        table_codes[:] = (1, rank_index + 1, natural + jokers, natural, jokers, int(leaders[0]))

    history_codes = np.zeros((192, 12), dtype=np.uint8)
    history_count = int(history_mask.sum())
    _prefix_mask(history_mask, history_count, "V4 history_mask")
    for index, row in enumerate(history_features[:history_count]):
        event_type = _decode_one_hot(row[:4], "history event type") + 1
        actor_offset = _decode_scaled(float(row[4]), player_count - 1, "history actor offset")
        before = _decode_scaled(float(row[5]), 20, "history hand before")
        after = _decode_scaled(float(row[6]), 20, "history hand after")
        rank = _decode_scaled(float(row[7]), 13, "history rank")
        natural = _decode_scaled(float(row[8]), 14, "history natural count")
        jokers = _decode_scaled(float(row[9]), 2, "history joker count")
        total = _decode_scaled(float(row[10]), 14, "history total count")
        pass_reason = 0
        clear_reason = 0
        next_leader_plus_one = 0
        finish_place = 0
        if event_type == V5_EVENT_TYPES["pass"]:
            pass_reason = _decode_one_hot(row[11:15], "history pass reason") + 1
        elif event_type == V5_EVENT_TYPES["clear"]:
            clear_reason = _decode_one_hot(row[15:18], "history clear reason") + 1
            # V4 used zero both for relative offset zero and for no leader.
            # Public game semantics disambiguate it: all-passed/dalmuti clears
            # always nominate a next leader; only act-ended has none.
            if clear_reason != V5_CLEAR_REASONS["act-ended"]:
                next_leader_plus_one = (
                    _decode_scaled(float(row[18]), player_count - 1, "history next leader")
                    + 1
                )
        elif event_type == V5_EVENT_TYPES["finish"]:
            finish_place = _decode_scaled(float(row[19]), player_count, "history finish place")
        history_codes[index] = (
            event_type,
            actor_offset,
            before,
            after,
            rank,
            natural,
            jokers,
            total,
            pass_reason,
            clear_reason,
            next_leader_plus_one,
            finish_place,
        )

    observation = V5PublicObservation(
        global_codes=np.asarray(
            [V5_PUBLIC_SCHEMA_VERSION, player_count, act, actor_role, revolution, truncated],
            dtype=np.int32,
        ),
        own_rank_counts=own,
        public_played_counts=played,
        player_codes=player_codes,
        player_mask=player_mask,
        table_codes=table_codes,
        history_codes=history_codes,
        history_mask=history_mask,
        legal_mask=legal_mask,
    )
    return validate_v5_public_observation(observation)


def pack_v5_public_observations(
    values: Iterable[V5PublicObservation],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Pack decisions into fixed actor arrays plus a compact ragged history.

    ``history_end[i]`` is the exclusive cumulative end for decision ``i``.
    Legal masks use NumPy's explicitly-versioned little-bit ordering; the four
    unused high bits of the final byte are guaranteed zero.
    """

    observations = [validate_v5_public_observation(value) for value in values]
    if not observations:
        raise ValueError("cannot pack an empty V5 public observation sequence")
    history_parts: list[np.ndarray] = []
    history_end = np.zeros(len(observations), dtype=np.uint32)
    running = 0
    for index, observation in enumerate(observations):
        count = int(observation.history_mask.sum())
        history_parts.append(observation.history_codes[:count])
        running += count
        if running > np.iinfo(np.uint32).max:
            raise OverflowError("packed V5 history exceeds uint32 offsets")
        history_end[index] = running
    history_events = (
        np.concatenate(history_parts, axis=0)
        if running
        else np.zeros((0, len(V5_HISTORY_FIELDS)), dtype=np.uint8)
    )
    legal_mask_packed = np.packbits(
        np.stack([value.legal_mask for value in observations]),
        axis=-1,
        bitorder="little",
    )
    if legal_mask_packed.shape != (len(observations), 30):
        raise AssertionError("236 legal bits must occupy exactly 30 bytes")
    if np.any(legal_mask_packed[:, -1] & np.uint8(0xF0)):
        raise AssertionError("unused packed legal-mask bits must remain zero")
    actor_arrays = {
        "global_codes": np.stack([value.global_codes for value in observations]),
        "own_rank_counts": np.stack([value.own_rank_counts for value in observations]),
        "public_played_counts": np.stack(
            [value.public_played_counts for value in observations]
        ),
        "player_codes": np.stack([value.player_codes for value in observations]),
        "player_masks": np.stack([value.player_mask for value in observations]),
        "table_codes": np.stack([value.table_codes for value in observations]),
        "legal_action_bits": legal_mask_packed,
        "belief_response_feasibility": np.stack(
            [
                compute_v5_public_beliefs(value).response_feasibility
                for value in observations
            ]
        ),
    }
    return actor_arrays, history_events, history_end


def pack_v5_public_from_v4(
    values: Iterable[object],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    return pack_v5_public_observations(
        [v5_public_from_v4_actor_observation(value) for value in values]
    )


def _packed_array(
    arrays: Mapping[str, object],
    name: str,
    dtype: np.dtype[object],
    shape: tuple[int, ...],
) -> np.ndarray:
    value = arrays[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"packed actor array {name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(
            f"packed actor array {name} must have dtype {dtype.name}"
        )
    if value.shape != shape:
        raise ValueError(
            f"packed actor array {name} must have shape {shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"packed actor array {name} must be C-contiguous")
    return value


@lru_cache(maxsize=1)
def _v5_action_catalogue_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks: list[int] = []
    naturals: list[int] = []
    jokers: list[int] = []
    for rank in range(1, 13):
        for natural in range(1, rank + 1):
            for joker_count in range(3):
                ranks.append(rank)
                naturals.append(natural)
                jokers.append(joker_count)
    if len(ranks) != V5_ACTION_COUNT - 2:
        raise AssertionError("the fixed 236-action catalogue is incomplete")
    values = tuple(
        np.asarray(items, dtype=np.uint8)
        for items in (ranks, naturals, jokers)
    )
    for value in values:
        value.setflags(write=False)
    return values  # type: ignore[return-value]


def _has_verified_packed_semantics(value: Mapping[str, object]) -> bool:
    # A tuple-shaped attribute on an arbitrary Mapping is not authority: a
    # caller could otherwise spoof the loader marker and bypass exact response
    # recomputation.  The delayed import avoids a module cycle during startup.
    from v5_dataset import V5VerifiedActorArrays

    marker = getattr(value, "__v5_exact_public_semantics__", None)
    return (
        type(value) is V5VerifiedActorArrays
        and isinstance(marker, tuple)
        and len(marker) == 2
        and marker[0] == V5_PUBLIC_CONTRACT_SHA256
        and isinstance(marker[1], str)
        and len(marker[1]) == 64
        and all(character in "0123456789abcdef" for character in marker[1])
    )


def validate_packed_v5_public_semantics(
    arrays: Mapping[str, object],
    *,
    verify_response_feasibility: bool,
    chunk_size: int = 4096,
) -> None:
    """Validate packed rows against the canonical dense public contract.

    The inexpensive categorical, history, table, role and legal-action checks
    always run.  Exact response-feasibility recomputation is intentionally
    selectable: publication enables it once, while a loader may rely on the
    immutable array checksums plus its semantic-validation receipt.
    """

    if not isinstance(arrays, Mapping):
        raise TypeError("packed actor arrays must be a mapping")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("packed semantic validation chunk_size must be positive")
    global_value = arrays.get("global_codes")
    if not isinstance(global_value, np.ndarray) or global_value.ndim != 2:
        raise ValueError("packed global_codes must be a rank-two NumPy array")
    decision_count = int(global_value.shape[0])
    if decision_count < 1:
        raise ValueError("packed semantic validation requires decisions")
    global_codes = _packed_array(
        arrays,
        "global_codes",
        _INT32,
        (decision_count, len(V5_GLOBAL_FIELDS)),
    )
    own = _packed_array(
        arrays, "own_rank_counts", _UINT8, (decision_count, V5_RANK_COUNT)
    )
    played = _packed_array(
        arrays,
        "public_played_counts",
        _UINT8,
        (decision_count, V5_RANK_COUNT),
    )
    players = _packed_array(
        arrays,
        "player_codes",
        _UINT8,
        (decision_count, V5_MAX_PLAYERS, len(V5_PLAYER_FIELDS)),
    )
    player_masks = _packed_array(
        arrays,
        "player_masks",
        _BOOL,
        (decision_count, V5_MAX_PLAYERS),
    )
    tables = _packed_array(
        arrays,
        "table_codes",
        _UINT8,
        (decision_count, len(V5_TABLE_FIELDS)),
    )
    legal_bits = _packed_array(
        arrays,
        "legal_action_bits",
        _UINT8,
        (decision_count, (V5_ACTION_COUNT + 7) // 8),
    )
    responses = _packed_array(
        arrays,
        "belief_response_feasibility",
        _FLOAT32,
        (decision_count, V5_MAX_OPPONENTS),
    )
    history_value = arrays.get("history_events")
    if not isinstance(history_value, np.ndarray) or history_value.ndim != 2:
        raise ValueError("packed history_events must be a rank-two NumPy array")
    history_events = _packed_array(
        arrays,
        "history_events",
        _UINT8,
        (int(history_value.shape[0]), len(V5_HISTORY_FIELDS)),
    )
    history_end = _packed_array(
        arrays, "history_end", np.dtype(np.uint32), (decision_count,)
    )

    ends = history_end.astype(np.int64, copy=False)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    lengths = ends - starts
    if (
        np.any(lengths < 0)
        or np.any(lengths > V5_MAX_HISTORY)
        or int(ends[-1]) != len(history_events)
    ):
        raise ValueError("packed history offsets violate the V5 ragged contract")

    action_ranks, action_naturals, action_jokers = (
        _v5_action_catalogue_arrays()
    )
    deck = np.asarray(V5_DECK_COUNTS, dtype=np.int16)
    expected_role_counts = {
        player_count: np.bincount(
            np.asarray(
                [
                    _role_for_social_index(index, player_count)
                    for index in range(player_count)
                ],
                dtype=np.int64,
            ),
            minlength=5,
        )
        for player_count in range(V5_MIN_PLAYERS, V5_MAX_PLAYERS + 1)
    }

    for lower in range(0, decision_count, chunk_size):
        upper = min(decision_count, lower + chunk_size)
        selected_global = global_codes[lower:upper]
        selected_own = own[lower:upper]
        selected_played = played[lower:upper]
        selected_players = players[lower:upper]
        selected_masks = player_masks[lower:upper]
        selected_tables = tables[lower:upper]
        selected_bits = legal_bits[lower:upper]
        selected_responses = responses[lower:upper]
        player_counts = selected_global[:, 1]

        if (
            np.any(selected_global[:, 0] != V5_PUBLIC_SCHEMA_VERSION)
            or np.any(player_counts < V5_MIN_PLAYERS)
            or np.any(player_counts > V5_MAX_PLAYERS)
            or np.any(selected_global[:, 2] < 1)
            or np.any(selected_global[:, 2] > 1_000_000)
            or np.any(selected_global[:, 3] < 0)
            or np.any(selected_global[:, 3] > 4)
            or np.any(selected_global[:, 4] < 0)
            or np.any(selected_global[:, 4] > 2)
            or np.any(selected_global[:, 5] < 0)
            or np.any(selected_global[:, 5] > 1_000_000_000)
        ):
            raise ValueError("packed global categorical value escaped its range")
        expected_masks = (
            np.arange(V5_MAX_PLAYERS)[None, :] < player_counts[:, None]
        )
        if not np.array_equal(selected_masks, expected_masks):
            raise ValueError("packed player masks disagree with player counts")
        if (
            np.any(selected_players[:, :, 0] > 9)
            or np.any(selected_players[:, :, 1] > 20)
            or np.any(selected_players[:, :, 2] > 4)
            or np.any(selected_players[:, :, 3:] > 1)
            or np.any(selected_players[~expected_masks] != 0)
        ):
            raise ValueError("packed player categorical value escaped its contract")
        expected_offsets = np.broadcast_to(
            np.arange(V5_MAX_PLAYERS, dtype=np.uint8),
            expected_masks.shape,
        )
        if np.any(
            (selected_players[:, :, 0] != expected_offsets) & expected_masks
        ):
            raise ValueError("packed player offsets are not actor-relative")
        if np.any(
            (
                selected_players[:, :, 3]
                != (selected_players[:, :, 1] == 0)
            )
            & expected_masks
        ):
            raise ValueError("packed finished flags disagree with remaining cards")
        if not np.array_equal(
            selected_players[:, 0, 2], selected_global[:, 3]
        ):
            raise ValueError("packed actor role disagrees with player row zero")
        for player_count, expected_counts in expected_role_counts.items():
            rows = np.flatnonzero(player_counts == player_count)
            if not rows.size:
                continue
            active_roles = selected_players[rows, :player_count, 2]
            actual_counts = np.stack(
                [
                    np.count_nonzero(active_roles == role, axis=1)
                    for role in range(5)
                ],
                axis=1,
            )
            if np.any(actual_counts != expected_counts[None, :]):
                raise ValueError("packed player roles do not form a social table")

        selected_own16 = selected_own.astype(np.int16)
        selected_played16 = selected_played.astype(np.int16)
        if np.any(selected_own16 + selected_played16 > deck[None, :]):
            raise ValueError("packed own/public counts exceed the physical deck")
        if not np.array_equal(
            selected_own.sum(axis=1, dtype=np.int64),
            selected_players[:, 0, 1].astype(np.int64),
        ):
            raise ValueError("packed own cards disagree with actor remaining count")
        if np.any(
            selected_played.sum(axis=1, dtype=np.int64)
            + selected_players[:, :, 1].sum(axis=1, dtype=np.int64)
            != V5_DECK_SIZE
        ):
            raise ValueError("packed played and remaining cards must total the deck")
        unknown = deck[None, :] - selected_own16 - selected_played16
        opponent_counts = selected_players[:, 1:, 1].astype(np.int64)
        populations = unknown.sum(axis=1, dtype=np.int64)
        if not np.array_equal(
            opponent_counts.sum(axis=1, dtype=np.int64), populations
        ):
            raise ValueError("packed opponent counts disagree with unseen cards")

        if (
            np.any(selected_tables[:, 0] > 1)
            or np.any(selected_tables[:, 1] > 13)
            or np.any(selected_tables[:, 2] > 14)
            or np.any(selected_tables[:, 3] > 12)
            or np.any(selected_tables[:, 4] > 2)
            or np.any(selected_tables[:, 5] > 9)
        ):
            raise ValueError("packed table categorical value escaped its range")
        present = selected_tables[:, 0] != 0
        if np.any(selected_tables[~present] != 0):
            raise ValueError("packed empty table rows must be zero")
        if present.any():
            active_tables = selected_tables[present].astype(np.int16)
            ranks = active_tables[:, 1]
            totals = active_tables[:, 2]
            naturals = active_tables[:, 3]
            jokers = active_tables[:, 4]
            valid_bundle = (totals == naturals + jokers) & (
                (
                    (ranks == 13)
                    & (naturals == 0)
                    & (jokers == 1)
                    & (totals == 1)
                )
                | (
                    (ranks >= 1)
                    & (ranks <= 12)
                    & (naturals >= 1)
                    & (naturals <= ranks)
                )
            )
            if not valid_bundle.all():
                raise ValueError("packed table bundle violates the action contract")
            active_rows = np.flatnonzero(present)
            if np.any(
                selected_tables[active_rows, 5]
                >= player_counts[active_rows]
            ):
                raise ValueError("packed table actor offset is outside the table")
            if np.any(
                naturals
                > selected_played[
                    active_rows, ranks.astype(np.int64) - 1
                ]
            ) or np.any(jokers > selected_played[active_rows, 12]):
                raise ValueError("packed table cards are not publicly played")
        expected_leaders = np.zeros(
            selected_masks.shape, dtype=np.uint8
        )
        active_rows = np.flatnonzero(present)
        if active_rows.size:
            expected_leaders[
                active_rows,
                selected_tables[active_rows, 5].astype(np.int64),
            ] = 1
        if not np.array_equal(selected_players[:, :, 5], expected_leaders):
            raise ValueError("packed table leader flags disagree with table codes")

        expected_legal = np.zeros(
            (upper - lower, V5_ACTION_COUNT), dtype=np.bool_
        )
        expected_legal[:, 0] = present
        expected_legal[:, 1] = (~present) & (selected_own[:, 12] > 0)
        available = (
            selected_own[:, action_ranks.astype(np.int64) - 1]
            >= action_naturals[None, :]
        ) & (selected_own[:, 12, None] >= action_jokers[None, :])
        response_position = (
            action_ranks[None, :] < selected_tables[:, 1, None]
        ) & (
            action_naturals[None, :] + action_jokers[None, :]
            == selected_tables[:, 2, None]
        )
        expected_legal[:, 2:] = available & (
            (~present)[:, None] | response_position
        )
        expected_bits = np.packbits(
            expected_legal, axis=1, bitorder="little"
        )
        if not np.array_equal(selected_bits, expected_bits):
            raise ValueError("packed legal action bits disagree with public cards")

        if (
            not np.isfinite(selected_responses).all()
            or np.any(selected_responses < 0.0)
            or np.any(selected_responses > 1.0)
        ):
            raise ValueError("packed response feasibility escaped [0,1]")
        opponent_masks = (
            np.arange(V5_MAX_OPPONENTS)[None, :]
            < (player_counts - 1)[:, None]
        )
        if np.any(selected_responses[~opponent_masks] != 0.0):
            raise ValueError("packed padded response feasibility must be zero")
        if verify_response_feasibility:
            expected_responses = _expected_response_feasibility_rows(
                unknown.astype(np.uint8),
                selected_players,
                selected_tables,
                player_counts,
            )
            if not np.array_equal(selected_responses, expected_responses):
                raise ValueError(
                    "packed response feasibility is not the exact public value"
                )

        selected_lengths = lengths[lower:upper]
        event_start = int(starts[lower])
        event_stop = int(ends[upper - 1])
        selected_events = history_events[event_start:event_stop]
        event_player_counts = np.repeat(player_counts, selected_lengths)
        if len(event_player_counts) != len(selected_events):
            raise ValueError("packed history ownership is inconsistent")
        if len(selected_events):
            _validate_batched_history_codes(
                selected_events[:, None, :],
                np.ones((len(selected_events), 1), dtype=np.bool_),
                event_player_counts,
            )


def actor_batch_from_packed_arrays(
    arrays: Mapping[str, object],
    indices: Sequence[int] | np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> V5ActorPublicBatch:
    """Decode selected mmap rows into a GPU-ready public Actor batch.

    Only the canonical public feature arrays and known actor-side learning
    labels are accepted.  In particular a caller cannot accidentally pass a
    merged actor/critic dictionary: ``privileged_states`` (and every unknown
    key) is rejected before any tensor is created.
    """

    if not isinstance(arrays, Mapping):
        raise TypeError("packed actor arrays must be a mapping")
    actual = set(arrays)
    missing = _PACKED_PUBLIC_KEYS - actual
    unknown = actual - _PACKED_PUBLIC_KEYS - _KNOWN_PACKED_ACTOR_AUXILIARY_KEYS
    if missing or unknown:
        detail = sorted(unknown)[0] if unknown else sorted(missing)[0]
        raise ValueError(
            f"packed actor arrays have an unknown/private or missing key: {detail}"
        )
    if isinstance(indices, torch.Tensor):
        if indices.dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("batch indices must use an integer dtype")
        index_array = indices.detach().cpu().numpy()
    else:
        index_array = np.asarray(indices)
    if index_array.ndim != 1 or index_array.dtype.kind not in "iu":
        raise TypeError("batch indices must be a one-dimensional integer array")
    if index_array.size == 0:
        raise ValueError("cannot decode an empty V5 actor batch")
    index_array = index_array.astype(np.int64, copy=False)

    global_value = arrays["global_codes"]
    if not isinstance(global_value, np.ndarray) or global_value.ndim != 2:
        raise ValueError("packed global_codes must be a rank-two NumPy array")
    decision_count = int(global_value.shape[0])
    if np.any(index_array < 0) or np.any(index_array >= decision_count):
        raise IndexError("V5 actor batch index is outside the decision arrays")

    # Validate every accepted auxiliary array even though the Actor does not
    # consume it.  This prevents a private tensor from being smuggled through
    # a familiar actor-label key with an unexpected shape.
    for name, dtype in _PACKED_DECISION_AUXILIARY_DTYPES.items():
        if name not in arrays:
            continue
        auxiliary = arrays[name]
        if (
            not isinstance(auxiliary, np.ndarray)
            or auxiliary.dtype != dtype
            or auxiliary.shape != (decision_count,)
            or not auxiliary.flags.c_contiguous
        ):
            raise ValueError(
                f"packed actor auxiliary {name} has a non-canonical layout"
            )
    match_keys = {"match_offsets", "candidate_bitsets", "player_counts"}
    present_match_keys = actual & match_keys
    if present_match_keys and present_match_keys != match_keys:
        raise ValueError("packed match auxiliary arrays must be supplied together")
    if present_match_keys:
        match_offsets = arrays["match_offsets"]
        candidate_bitsets = arrays["candidate_bitsets"]
        match_player_counts = arrays["player_counts"]
        if (
            not isinstance(match_offsets, np.ndarray)
            or match_offsets.dtype != np.dtype(np.uint32)
            or match_offsets.ndim != 1
            or not match_offsets.flags.c_contiguous
            or len(match_offsets) < 2
            or int(match_offsets[0]) != 0
            or int(match_offsets[-1]) != decision_count
        ):
            raise ValueError("packed match_offsets has a non-canonical layout")
        match_count = len(match_offsets) - 1
        if (
            not isinstance(candidate_bitsets, np.ndarray)
            or candidate_bitsets.dtype != np.dtype(np.uint16)
            or candidate_bitsets.shape != (match_count,)
            or not candidate_bitsets.flags.c_contiguous
            or not isinstance(match_player_counts, np.ndarray)
            or match_player_counts.dtype != np.dtype(np.uint8)
            or match_player_counts.shape != (match_count,)
            or not match_player_counts.flags.c_contiguous
        ):
            raise ValueError("packed match metadata has a non-canonical layout")

    global_codes = _packed_array(
        arrays,
        "global_codes",
        _INT32,
        (decision_count, len(V5_GLOBAL_FIELDS)),
    )
    own = _packed_array(
        arrays,
        "own_rank_counts",
        _UINT8,
        (decision_count, V5_RANK_COUNT),
    )
    played = _packed_array(
        arrays,
        "public_played_counts",
        _UINT8,
        (decision_count, V5_RANK_COUNT),
    )
    players = _packed_array(
        arrays,
        "player_codes",
        _UINT8,
        (decision_count, V5_MAX_PLAYERS, len(V5_PLAYER_FIELDS)),
    )
    player_masks = _packed_array(
        arrays,
        "player_masks",
        _BOOL,
        (decision_count, V5_MAX_PLAYERS),
    )
    tables = _packed_array(
        arrays,
        "table_codes",
        _UINT8,
        (decision_count, len(V5_TABLE_FIELDS)),
    )
    legal_bits = _packed_array(
        arrays,
        "legal_action_bits",
        _UINT8,
        (decision_count, (V5_ACTION_COUNT + 7) // 8),
    )
    belief_response = _packed_array(
        arrays,
        "belief_response_feasibility",
        _FLOAT32,
        (decision_count, V5_MAX_OPPONENTS),
    )
    history_events_value = arrays["history_events"]
    if not isinstance(history_events_value, np.ndarray) or history_events_value.ndim != 2:
        raise ValueError("packed history_events must be a rank-two NumPy array")
    history_events = _packed_array(
        arrays,
        "history_events",
        _UINT8,
        (int(history_events_value.shape[0]), len(V5_HISTORY_FIELDS)),
    )
    history_end = _packed_array(
        arrays,
        "history_end",
        np.dtype(np.uint32),
        (decision_count,),
    )
    if decision_count and int(history_end[-1]) != len(history_events):
        raise ValueError("packed history_end must consume every history event")
    if not decision_count and len(history_events):
        raise ValueError("an empty decision array cannot own history events")

    selected_global = global_codes[index_array]
    selected_own = own[index_array]
    selected_played = played[index_array]
    selected_players = players[index_array]
    selected_player_masks = player_masks[index_array]
    selected_tables = tables[index_array]
    selected_belief_response = belief_response[index_array]

    player_counts = selected_global[:, 1]
    if np.any(selected_global[:, 0] != V5_PUBLIC_SCHEMA_VERSION):
        raise ValueError("packed global_codes has a wrong schema version")
    if np.any(player_counts < V5_MIN_PLAYERS) or np.any(player_counts > V5_MAX_PLAYERS):
        raise ValueError("packed player count escaped p4..p10")
    if (
        np.any(selected_global[:, 2] < 1)
        or np.any(selected_global[:, 2] > 1_000_000)
        or np.any(selected_global[:, 3] < 0)
        or np.any(selected_global[:, 3] > 4)
        or np.any(selected_global[:, 4] < 0)
        or np.any(selected_global[:, 4] > 2)
        or np.any(selected_global[:, 5] < 0)
        or np.any(selected_global[:, 5] > 1_000_000_000)
    ):
        raise ValueError("packed global categorical value escaped its range")
    expected_player_masks = (
        np.arange(V5_MAX_PLAYERS)[None, :] < player_counts[:, None]
    )
    expected_opponent_masks = (
        np.arange(V5_MAX_OPPONENTS)[None, :] < (player_counts - 1)[:, None]
    )
    if not np.array_equal(selected_player_masks, expected_player_masks):
        raise ValueError("packed player masks disagree with player counts")
    if (
        np.any(selected_players[:, :, 0] > 9)
        or np.any(selected_players[:, :, 1] > 20)
        or np.any(selected_players[:, :, 2] > 4)
        or np.any(selected_players[:, :, 3:] > 1)
    ):
        raise ValueError("packed player categorical value escaped its range")
    for row, player_count in enumerate(player_counts):
        count = int(player_count)
        if not np.array_equal(
            selected_players[row, :count, 0],
            np.arange(count, dtype=np.uint8),
        ):
            raise ValueError("packed player offsets are not actor-relative")
        if np.any(selected_players[row, count:] != 0):
            raise ValueError("packed player padding must be zero")
        active = selected_players[row, :count]
        if not np.array_equal(active[:, 3], active[:, 1] == 0):
            raise ValueError("packed finished flags disagree with remaining cards")
        if int(active[0, 2]) != int(selected_global[row, 3]):
            raise ValueError("packed actor role disagrees with player row zero")
        expected_roles = sorted(
            _role_for_social_index(index, count) for index in range(count)
        )
        if sorted(int(role) for role in active[:, 2]) != expected_roles:
            raise ValueError("packed player roles do not form a social table")
    if (
        np.any(selected_tables[:, 0] > 1)
        or np.any(selected_tables[:, 1] > 13)
        or np.any(selected_tables[:, 2] > 14)
        or np.any(selected_tables[:, 3] > 12)
        or np.any(selected_tables[:, 4] > 2)
        or np.any(selected_tables[:, 5] > 9)
    ):
        raise ValueError("packed table categorical value escaped its range")
    for row, player_count in enumerate(player_counts):
        table = selected_tables[row]
        if not int(table[0]):
            if np.any(table != 0) or np.any(selected_players[row, :, 5] != 0):
                raise ValueError("packed empty table row must be all zero")
        else:
            _validate_bundle(
                int(table[1]),
                int(table[3]),
                int(table[4]),
                int(table[2]),
                "packed table_codes",
            )
            if int(table[5]) >= int(player_count):
                raise ValueError("packed table actor offset is outside the table")
            expected_leaders = np.zeros(V5_MAX_PLAYERS, dtype=np.uint8)
            expected_leaders[int(table[5])] = 1
            if not np.array_equal(selected_players[row, :, 5], expected_leaders):
                raise ValueError("packed table leader flags disagree with table codes")
            if (
                int(table[3]) > int(selected_played[row, int(table[1]) - 1])
                or int(table[4]) > int(selected_played[row, 12])
            ):
                raise ValueError("packed table cards are not publicly played")
    deck = np.asarray(V5_DECK_COUNTS, dtype=np.int16)
    if np.any(selected_own.astype(np.int16) + selected_played.astype(np.int16) > deck):
        raise ValueError("packed own/public counts exceed the physical deck")
    expected_unknown = (
        deck[None, :]
        - selected_own.astype(np.int16)
        - selected_played.astype(np.int16)
    )
    if np.any(expected_unknown < 0):
        raise ValueError("packed public counts imply negative unseen cards")
    if not np.array_equal(
        selected_own.sum(axis=1, dtype=np.int64),
        selected_players[:, 0, 1].astype(np.int64),
    ):
        raise ValueError("packed own cards disagree with actor remaining count")
    if np.any(
        selected_played.sum(axis=1, dtype=np.int64)
        + selected_players[:, :, 1].sum(axis=1, dtype=np.int64)
        != V5_DECK_SIZE
    ):
        raise ValueError("packed played and remaining cards must total the deck")
    opponent_hand_counts = selected_players[:, 1:, 1].astype(np.int64)
    populations = expected_unknown.sum(axis=1, dtype=np.int64)
    if not np.array_equal(
        opponent_hand_counts.sum(axis=1, dtype=np.int64), populations
    ):
        raise ValueError("packed opponent remaining counts disagree with unseen cards")
    selected_belief_unknown = expected_unknown.astype(np.uint8)
    expected_numerators = (
        opponent_hand_counts[:, :, None]
        * selected_belief_unknown[:, None, :].astype(np.int64)
    )
    selected_belief_expected = np.zeros(
        expected_numerators.shape, dtype=np.float64
    )
    np.divide(
        expected_numerators,
        populations[:, None, None],
        out=selected_belief_expected,
        where=populations[:, None, None] != 0,
    )
    selected_belief_expected = selected_belief_expected.astype(np.float32)
    required_counts = np.where(
        selected_tables[:, 0] != 0, selected_tables[:, 2], 1
    ).astype(np.int64)
    lookup = _hypergeometric_tail_table()
    selected_belief_one = lookup[
        populations[:, None, None],
        selected_belief_unknown[:, None, :],
        opponent_hand_counts[:, :, None],
        np.int64(1),
    ]
    selected_belief_required = lookup[
        populations[:, None, None],
        selected_belief_unknown[:, None, :],
        opponent_hand_counts[:, :, None],
        required_counts[:, None, None],
    ]
    if (
        not np.isfinite(selected_belief_response).all()
        or np.any(selected_belief_response < 0.0)
        or np.any(selected_belief_response > 1.0)
    ):
        raise ValueError("stored belief_response_feasibility escaped its finite range")
    if np.any(selected_belief_response[~expected_opponent_masks] != 0.0):
        raise ValueError("stored padded response feasibility must be zero")
    if not _has_verified_packed_semantics(arrays):
        exact_response = _expected_response_feasibility_rows(
            selected_belief_unknown,
            selected_players,
            selected_tables,
            player_counts,
        )
        if not np.array_equal(selected_belief_response, exact_response):
            raise ValueError(
                "stored response feasibility is not the exact public value"
            )
    if np.any(legal_bits[index_array, -1] & np.uint8(0xF0)):
        raise ValueError("packed legal action bits have nonzero padding")
    selected_legal = np.unpackbits(
        legal_bits[index_array], axis=-1, bitorder="little"
    )[:, :V5_ACTION_COUNT].astype(np.bool_, copy=False)
    action_ranks, action_naturals, action_jokers = _v5_action_catalogue_arrays()
    expected_legal = np.zeros_like(selected_legal)
    present = selected_tables[:, 0] != 0
    expected_legal[:, 0] = present
    expected_legal[:, 1] = (~present) & (selected_own[:, 12] > 0)
    available = (
        selected_own[:, action_ranks.astype(np.int64) - 1]
        >= action_naturals[None, :]
    ) & (selected_own[:, 12, None] >= action_jokers[None, :])
    response_position = (
        action_ranks[None, :] < selected_tables[:, 1, None]
    ) & (
        action_naturals[None, :] + action_jokers[None, :]
        == selected_tables[:, 2, None]
    )
    expected_legal[:, 2:] = available & (
        (~present)[:, None] | response_position
    )
    if not np.array_equal(selected_legal, expected_legal):
        raise ValueError("packed legal action bits disagree with public cards")

    selected_history = np.zeros(
        (len(index_array), V5_MAX_HISTORY, len(V5_HISTORY_FIELDS)),
        dtype=np.uint8,
    )
    selected_history_mask = np.zeros(
        (len(index_array), V5_MAX_HISTORY), dtype=np.bool_
    )
    for output_index, decision_index in enumerate(index_array):
        start = 0 if decision_index == 0 else int(history_end[decision_index - 1])
        stop = int(history_end[decision_index])
        count = stop - start
        if count < 0 or count > V5_MAX_HISTORY:
            raise ValueError("packed decision history exceeds the V5 limit")
        selected_history[output_index, :count] = history_events[start:stop]
        selected_history_mask[output_index, :count] = True
    _validate_batched_history_codes(
        selected_history, selected_history_mask, player_counts
    )

    target = torch.device(device)

    def categorical(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            device=target, dtype=torch.long
        )

    def boolean(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            device=target, dtype=torch.bool
        )

    def continuous(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            device=target, dtype=torch.float32
        )

    return V5ActorPublicBatch(
        global_codes=categorical(selected_global),
        own_rank_counts=categorical(selected_own),
        public_played_counts=categorical(selected_played),
        player_codes=categorical(selected_players),
        player_mask=boolean(selected_player_masks),
        table_codes=categorical(selected_tables),
        history_codes=categorical(selected_history),
        history_mask=boolean(selected_history_mask),
        legal_mask=boolean(selected_legal),
        belief_unknown_rank_counts=categorical(selected_belief_unknown),
        belief_expected_counts=continuous(selected_belief_expected),
        belief_probability_at_least_one=continuous(selected_belief_one),
        belief_probability_at_least_required=continuous(selected_belief_required),
        belief_response_feasibility=continuous(selected_belief_response),
        opponent_mask=boolean(expected_opponent_masks),
    )
