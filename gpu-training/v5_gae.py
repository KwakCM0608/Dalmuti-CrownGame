from __future__ import annotations

"""Decision-time SMDP advantages for the V5 shared candidate policy.

The environment may execute many other players' actions between two decisions
made by the same candidate.  Consequently an ordinary adjacent-row GAE is
incorrect.  This module follows the explicit ``next_decision`` links, whose
``reward_to_next`` value already contains the reward accumulated up to that
decision (or to the end of the complete match).

Forced decisions remain in the value/GAE chain.  They are deliberately absent
from the policy mask so that their arbitrary log probability cannot update the
actor.  Player-count weights are derived independently for the policy and
value masks; every represented p4..p10 stratum therefore contributes the same
total loss mass.
"""

from dataclasses import dataclass
import math
import numpy as np


V5_SMDP_GAE_CONTRACT = "dalmuti-v5-decision-time-smdp-gae-v1"
V5_PLAYER_COUNTS = tuple(range(4, 11))
V5_GAMMA = 1.0
V5_GAE_LAMBDA = 0.95


@dataclass(frozen=True)
class V5GAEResult:
    """Canonical GAE targets and per-loss eligibility/weight arrays."""

    advantages: np.ndarray
    returns: np.ndarray
    deltas: np.ndarray
    policy_mask: np.ndarray
    value_mask: np.ndarray
    policy_loss_weights: np.ndarray
    value_loss_weights: np.ndarray


def _one_dimensional(name: str, value: object, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or (length is not None and array.shape != (length,)):
        expected = "[decision]" if length is None else f"[{length}]"
        raise ValueError(f"{name} must have shape {expected}")
    return array


def _bool_array(name: str, value: object, length: int) -> np.ndarray:
    array = _one_dimensional(name, value, length)
    if array.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must use the canonical bool dtype")
    return array


def _integer_array(name: str, value: object, length: int | None = None) -> np.ndarray:
    array = _one_dimensional(name, value, length)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must use an integer dtype")
    return array


def _finite_float_array(name: str, value: object, length: int) -> np.ndarray:
    array = _one_dimensional(name, value, length)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def decision_match_ids(match_offsets: object, decision_count: int) -> np.ndarray:
    """Expand canonical prefix offsets to one match id per decision."""

    if isinstance(decision_count, bool) or not isinstance(decision_count, int):
        raise ValueError("decision_count must be an integer")
    offsets = _integer_array("match_offsets", match_offsets)
    if offsets.size < 2:
        raise ValueError("match_offsets must contain at least one complete match")
    values = offsets.astype(np.int64, copy=False)
    if int(values[0]) != 0 or int(values[-1]) != decision_count:
        raise ValueError("match_offsets must span every decision exactly once")
    if np.any(values[1:] <= values[:-1]):
        raise ValueError("every complete match must contain at least one decision")
    output = np.empty(decision_count, dtype=np.int32)
    for match_index, (start, stop) in enumerate(zip(values[:-1], values[1:])):
        output[int(start) : int(stop)] = match_index
    return output


def expand_match_player_counts(
    match_offsets: object, player_counts: object, decision_count: int
) -> np.ndarray:
    """Return the compact p4..p10 identity for every decision row."""

    match_ids = decision_match_ids(match_offsets, decision_count)
    counts = _integer_array("player_counts", player_counts, int(match_ids.max()) + 1)
    counts64 = counts.astype(np.int64, copy=False)
    if np.any((counts64 < 4) | (counts64 > 10)):
        raise ValueError("player_counts must be in the canonical p4..p10 range")
    return counts64[match_ids].astype(np.uint8, copy=False)


def equal_player_count_loss_weights(
    decision_player_counts: object,
    eligible_mask: object | None = None,
    *,
    require_all_player_counts: bool = False,
) -> np.ndarray:
    """Give every represented player count equal total eligible loss mass.

    The non-zero weights have mean one.  Ineligible rows are exactly zero.
    ``require_all_player_counts`` is useful for final merged training datasets;
    individual collection shards may intentionally contain only one stratum.
    """

    counts = _integer_array("decision_player_counts", decision_player_counts)
    count64 = counts.astype(np.int64, copy=False)
    if np.any((count64 < 4) | (count64 > 10)):
        raise ValueError("decision_player_counts must be in p4..p10")
    if eligible_mask is None:
        eligible = np.ones(count64.shape, dtype=np.bool_)
    else:
        eligible = _bool_array("eligible_mask", eligible_mask, count64.size)
    represented = tuple(
        player_count
        for player_count in V5_PLAYER_COUNTS
        if bool(np.any(eligible & (count64 == player_count)))
    )
    if require_all_player_counts and represented != V5_PLAYER_COUNTS:
        missing = sorted(set(V5_PLAYER_COUNTS) - set(represented))
        raise ValueError(f"eligible loss population is missing player counts: {missing}")
    output = np.zeros(count64.shape, dtype=np.float32)
    eligible_total = int(eligible.sum())
    if eligible_total == 0:
        if require_all_player_counts:
            raise ValueError("eligible loss population is empty")
        return output
    stratum_total = eligible_total / len(represented)
    for player_count in represented:
        mask = eligible & (count64 == player_count)
        output[mask] = stratum_total / int(mask.sum())
    # A float32 cast can introduce a tiny normalization drift.  Binding the
    # mean exactly enough for deterministic loss accounting is preferable to
    # silently renormalizing each minibatch later.
    scale = eligible_total / math.fsum(float(value) for value in output[eligible])
    output[eligible] *= np.float32(scale)
    return output


def _validate_candidate_chains(
    *,
    match_ids: np.ndarray,
    actor_ids: np.ndarray,
    next_decision: np.ndarray,
    done: np.ndarray,
    candidate_bitsets: object | None,
) -> None:
    decision_count = match_ids.size
    predecessor_count = np.zeros(decision_count, dtype=np.uint8)
    for index in range(decision_count):
        successor = int(next_decision[index])
        terminal = bool(done[index])
        if terminal:
            if successor != -1:
                raise ValueError("terminal decision must use next_decision=-1")
            continue
        if successor <= index or successor >= decision_count:
            raise ValueError("non-terminal next_decision must point strictly forward")
        if int(match_ids[successor]) != int(match_ids[index]):
            raise ValueError("next_decision leaks across a complete-match boundary")
        if int(actor_ids[successor]) != int(actor_ids[index]):
            raise ValueError("next_decision must remain on the same candidate identity")
        predecessor_count[successor] += 1
        if predecessor_count[successor] != 1:
            raise ValueError("a candidate decision has more than one predecessor")

    match_count = int(match_ids[-1]) + 1 if decision_count else 0
    bitsets: np.ndarray | None = None
    if candidate_bitsets is not None:
        bitsets = _integer_array("candidate_bitsets", candidate_bitsets, match_count)
        if np.any(bitsets.astype(np.int64, copy=False) <= 0):
            raise ValueError("every match must contain at least one candidate")

    for match_index in range(match_count):
        rows = np.flatnonzero(match_ids == match_index)
        actors = set(int(value) for value in actor_ids[rows])
        if not actors:
            raise ValueError("complete match has no candidate decisions")
        if bitsets is not None:
            declared = int(bitsets[match_index])
            declared_actors = {bit for bit in range(16) if declared & (1 << bit)}
            if actors != declared_actors:
                raise ValueError(
                    "candidate_bitsets must exactly match recorded decision identities"
                )
        for actor in actors:
            chain = rows[actor_ids[rows].astype(np.int64, copy=False) == actor]
            roots = chain[predecessor_count[chain] == 0]
            terminals = chain[done[chain]]
            if roots.size != 1 or terminals.size != 1:
                raise ValueError(
                    "each complete-match candidate sequence needs one root and terminal"
                )
            visited: set[int] = set()
            cursor = int(roots[0])
            while cursor != -1:
                if cursor in visited:
                    raise ValueError("candidate next_decision chain contains a cycle")
                visited.add(cursor)
                cursor = int(next_decision[cursor])
            if visited != set(int(value) for value in chain):
                raise ValueError("candidate decisions do not form one complete chain")


def compute_smdp_gae(
    *,
    reward_to_next: object,
    next_decision: object,
    done: object,
    old_values: object,
    match_offsets: object,
    decision_actor_ids: object,
    player_counts: object,
    forced: object,
    candidate_bitsets: object | None = None,
    gamma: float = V5_GAMMA,
    gae_lambda: float = V5_GAE_LAMBDA,
    require_all_player_counts: bool = False,
) -> V5GAEResult:
    """Compute leakage-safe GAE over explicit candidate decision links.

    ``reward_to_next[i]`` is the SMDP reward accumulated after decision ``i``
    and before ``next_decision[i]``.  If ``done[i]`` is true it instead reaches
    the terminal outcome of that candidate's complete-match sequence.
    """

    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not math.isfinite(float(gamma))
        or float(gamma) != V5_GAMMA
    ):
        raise ValueError("V5 decision-time SMDP GAE requires gamma=1.0")
    if (
        isinstance(gae_lambda, bool)
        or not isinstance(gae_lambda, (int, float))
        or not math.isfinite(float(gae_lambda))
        or float(gae_lambda) != V5_GAE_LAMBDA
    ):
        raise ValueError("V5 decision-time SMDP GAE requires gae_lambda=0.95")

    reward_array = np.asarray(reward_to_next)
    if reward_array.ndim != 1:
        raise ValueError("reward_to_next must have shape [decision]")
    decision_count = int(reward_array.size)
    if decision_count < 1:
        raise ValueError("a complete-match GAE population cannot be empty")
    rewards = _finite_float_array("reward_to_next", reward_array, decision_count)
    values = _finite_float_array("old_values", old_values, decision_count)
    successors = _integer_array("next_decision", next_decision, decision_count).astype(
        np.int64, copy=False
    )
    terminals = _bool_array("done", done, decision_count)
    forced_array = _bool_array("forced", forced, decision_count)
    actors = _integer_array(
        "decision_actor_ids", decision_actor_ids, decision_count
    ).astype(np.int64, copy=False)
    if np.any((actors < 0) | (actors > 9)):
        raise ValueError("decision_actor_ids must be compact physical ids 0..9")

    match_ids = decision_match_ids(match_offsets, decision_count)
    decision_counts = expand_match_player_counts(
        match_offsets, player_counts, decision_count
    )
    if np.any(actors >= decision_counts.astype(np.int64, copy=False)):
        raise ValueError("decision_actor_ids exceed their match player count")
    if candidate_bitsets is not None:
        bits = _integer_array(
            "candidate_bitsets", candidate_bitsets, len(np.asarray(player_counts))
        ).astype(np.int64, copy=False)
        for match_index, player_count in enumerate(np.asarray(player_counts)):
            if int(bits[match_index]) & ~((1 << int(player_count)) - 1):
                raise ValueError("candidate_bitsets contains an out-of-match player bit")
    _validate_candidate_chains(
        match_ids=match_ids,
        actor_ids=actors,
        next_decision=successors,
        done=terminals,
        candidate_bitsets=candidate_bitsets,
    )

    deltas = np.empty(decision_count, dtype=np.float64)
    advantages = np.empty(decision_count, dtype=np.float64)
    gamma_value = float(gamma)
    lambda_value = float(gae_lambda)
    # Links must point forward, so reverse row order is a valid reverse
    # topological traversal even when different candidates are interleaved.
    for index in range(decision_count - 1, -1, -1):
        successor = int(successors[index])
        if bool(terminals[index]):
            bootstrap = 0.0
            continuation = 0.0
        else:
            bootstrap = values[successor]
            continuation = advantages[successor]
        delta = rewards[index] + gamma_value * bootstrap - values[index]
        deltas[index] = delta
        advantages[index] = (
            delta + gamma_value * lambda_value * continuation
        )

    policy_mask = np.logical_not(forced_array)
    value_mask = np.ones(decision_count, dtype=np.bool_)
    policy_weights = equal_player_count_loss_weights(
        decision_counts,
        policy_mask,
        require_all_player_counts=require_all_player_counts,
    )
    value_weights = equal_player_count_loss_weights(
        decision_counts,
        value_mask,
        require_all_player_counts=require_all_player_counts,
    )
    return V5GAEResult(
        advantages=advantages.astype(np.float32),
        returns=(advantages + values).astype(np.float32),
        deltas=deltas.astype(np.float32),
        policy_mask=policy_mask,
        value_mask=value_mask,
        policy_loss_weights=policy_weights,
        value_loss_weights=value_weights,
    )


# A concise alias for callers that already carry the V5 contract context.
compute_gae = compute_smdp_gae
player_count_loss_weights = equal_player_count_loss_weights


__all__ = [
    "V5GAEResult",
    "V5_GAE_LAMBDA",
    "V5_GAMMA",
    "V5_PLAYER_COUNTS",
    "V5_SMDP_GAE_CONTRACT",
    "compute_gae",
    "compute_smdp_gae",
    "decision_match_ids",
    "equal_player_count_loss_weights",
    "expand_match_player_counts",
    "player_count_loss_weights",
]
