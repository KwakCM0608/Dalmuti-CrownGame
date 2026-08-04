from __future__ import annotations

"""Leakage-safe supervised targets used by the DALMUTI V6 warm start."""

import math

import numpy as np

from v5_gae import (
    _validate_candidate_chains,
    decision_match_ids,
    expand_match_player_counts,
)


V6_MC_RETURN_CONTRACT = "dalmuti-v6-undiscounted-match-monte-carlo-v1"
V6_NORMAL_CLASS_WEIGHT_CONTRACT = "dalmuti-v6-sqrt-balanced-normal-actions-v1"
V6_BELIEF_TARGET_CONTRACT = "dalmuti-v6-private-label-public-input-belief-v1"
ACTION_COUNT = 236
PRIVILEGED_STATE_SIZE = 512
PLAYER_OFFSET = 29
PLAYER_STRIDE = 25
HAND_COUNT_OFFSET = 8
HAND_RANK_OFFSET = 12
MAX_PLAYERS = 10
MAX_OPPONENTS = 9
RANKS = 13


def compute_v6_monte_carlo_returns(
    *,
    reward_to_next: object,
    next_decision: object,
    done: object,
    match_offsets: object,
    decision_actor_ids: object,
    player_counts: object,
    candidate_bitsets: object | None = None,
) -> np.ndarray:
    """Return exact gamma=1 reward-to-match-end along each candidate chain."""

    rewards_raw = np.asarray(reward_to_next)
    if rewards_raw.ndim != 1 or rewards_raw.size < 1:
        raise ValueError("reward_to_next must be a non-empty [decision] array")
    if not np.issubdtype(rewards_raw.dtype, np.floating):
        raise ValueError("reward_to_next must be floating-point")
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    if not np.isfinite(rewards).all():
        raise ValueError("reward_to_next contains a non-finite value")
    decision_count = int(rewards.size)
    successors = np.asarray(next_decision)
    terminals = np.asarray(done)
    actors = np.asarray(decision_actor_ids)
    if (
        successors.shape != (decision_count,)
        or not np.issubdtype(successors.dtype, np.integer)
        or np.issubdtype(successors.dtype, np.bool_)
    ):
        raise ValueError("next_decision must be integer [decision]")
    if terminals.shape != (decision_count,) or terminals.dtype != np.dtype(np.bool_):
        raise ValueError("done must be canonical bool [decision]")
    if (
        actors.shape != (decision_count,)
        or not np.issubdtype(actors.dtype, np.integer)
        or np.issubdtype(actors.dtype, np.bool_)
    ):
        raise ValueError("decision_actor_ids must be integer [decision]")
    successors64 = successors.astype(np.int64, copy=False)
    actors64 = actors.astype(np.int64, copy=False)
    match_ids = decision_match_ids(match_offsets, decision_count)
    decision_players = expand_match_player_counts(
        match_offsets, player_counts, decision_count
    )
    if np.any((actors64 < 0) | (actors64 >= decision_players.astype(np.int64))):
        raise ValueError("decision_actor_ids escaped their match player count")
    _validate_candidate_chains(
        match_ids=match_ids,
        actor_ids=actors64,
        next_decision=successors64,
        done=terminals,
        candidate_bitsets=candidate_bitsets,
    )

    returns = np.empty(decision_count, dtype=np.float64)
    for index in range(decision_count - 1, -1, -1):
        successor = int(successors64[index])
        if bool(terminals[index]):
            returns[index] = rewards[index]
        else:
            returns[index] = rewards[index] + returns[successor]
    if not np.isfinite(returns).all():
        raise RuntimeError("V6 Monte Carlo return became non-finite")
    return returns.astype(np.float32)


def balanced_normal_action_weights(
    normal_actions: object,
    eligible_mask: object,
    *,
    exponent: float = 0.5,
    maximum_ratio: float = 10.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build bounded sqrt-inverse-frequency weights for observed Normal labels."""

    actions = np.asarray(normal_actions)
    eligible = np.asarray(eligible_mask)
    if actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("normal_actions must be integer [decision]")
    if eligible.shape != actions.shape or eligible.dtype != np.dtype(np.bool_):
        raise ValueError("eligible_mask must be canonical bool [decision]")
    if np.any((actions < 0) | (actions >= ACTION_COUNT)):
        raise ValueError("normal_actions escaped the fixed action catalogue")
    if not bool(eligible.any()):
        raise ValueError("Normal class weighting requires eligible rows")
    if (
        isinstance(exponent, bool)
        or not math.isfinite(float(exponent))
        or not 0.0 <= float(exponent) <= 1.0
    ):
        raise ValueError("class-weight exponent must be finite in [0,1]")
    if (
        isinstance(maximum_ratio, bool)
        or not math.isfinite(float(maximum_ratio))
        or float(maximum_ratio) < 1.0
    ):
        raise ValueError("maximum class-weight ratio must be at least one")

    counts = np.bincount(
        actions[eligible].astype(np.int64, copy=False), minlength=ACTION_COUNT
    ).astype(np.int64, copy=False)
    observed = counts > 0
    maximum = int(counts.max())
    class_weights = np.zeros(ACTION_COUNT, dtype=np.float64)
    class_weights[observed] = np.minimum(
        (maximum / counts[observed]) ** float(exponent), float(maximum_ratio)
    )
    row_weights = np.zeros(actions.shape, dtype=np.float32)
    row_weights[eligible] = class_weights[actions[eligible]].astype(np.float32)
    scale = int(eligible.sum()) / math.fsum(float(value) for value in row_weights[eligible])
    row_weights[eligible] *= np.float32(scale)
    observed_weights = class_weights[observed] * scale
    report: dict[str, object] = {
        "contract": V6_NORMAL_CLASS_WEIGHT_CONTRACT,
        "eligibleRows": int(eligible.sum()),
        "observedActions": int(observed.sum()),
        "minimumClassCount": int(counts[observed].min()),
        "maximumClassCount": maximum,
        "exponent": float(exponent),
        "maximumRatio": float(maximum_ratio),
        "realizedClassWeightRatio": float(
            observed_weights.max() / observed_weights.min()
        ),
        "meanEligibleRowWeight": float(row_weights[eligible].mean(dtype=np.float64)),
    }
    return row_weights, report


def extract_v6_opponent_hand_targets(
    privileged_states: object,
    player_counts: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract private hand labels; these are targets and never Actor inputs."""

    states = np.asarray(privileged_states)
    counts = np.asarray(player_counts)
    if (
        states.ndim != 2
        or states.shape[1] != PRIVILEGED_STATE_SIZE
        or not np.issubdtype(states.dtype, np.floating)
        or not np.isfinite(states).all()
    ):
        raise ValueError("privileged_states must be finite float [decision,512]")
    if (
        counts.shape != (states.shape[0],)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any((counts < 4) | (counts > 10))
    ):
        raise ValueError("player_counts must be integer p4..p10 per decision")

    targets = np.zeros((states.shape[0], MAX_OPPONENTS, RANKS), dtype=np.uint8)
    mask = np.zeros((states.shape[0], MAX_OPPONENTS), dtype=np.bool_)
    states64 = states.astype(np.float64, copy=False)
    for row in range(states.shape[0]):
        player_count = int(counts[row])
        for opponent in range(MAX_OPPONENTS):
            relative_offset = opponent + 1
            start = PLAYER_OFFSET + relative_offset * PLAYER_STRIDE
            block = states64[row, start : start + PLAYER_STRIDE]
            if relative_offset < player_count:
                if (
                    block[0] != 1.0
                    or block[1] != float(relative_offset)
                    or not np.equal(block[HAND_RANK_OFFSET:], np.rint(block[HAND_RANK_OFFSET:])).all()
                    or np.any(block[HAND_RANK_OFFSET:] < 0.0)
                ):
                    raise ValueError("privileged opponent player block is malformed")
                hand_counts = block[HAND_RANK_OFFSET:].astype(np.uint8)
                if int(hand_counts.sum()) != int(round(block[HAND_COUNT_OFFSET])):
                    raise ValueError("privileged opponent hand counts do not sum")
                targets[row, opponent] = hand_counts
                mask[row, opponent] = True
            elif bool(np.any(block != 0.0)):
                raise ValueError("absent privileged player block must be all zero")
    return targets, mask


__all__ = [
    "ACTION_COUNT",
    "V6_BELIEF_TARGET_CONTRACT",
    "V6_MC_RETURN_CONTRACT",
    "V6_NORMAL_CLASS_WEIGHT_CONTRACT",
    "balanced_normal_action_weights",
    "compute_v6_monte_carlo_returns",
    "extract_v6_opponent_hand_targets",
]
