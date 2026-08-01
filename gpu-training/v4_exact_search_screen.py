from __future__ import annotations

"""Research-only exact public-information search screening for DALMUTI V4.

This diagnostic intentionally is not a promotion/certification evaluator.  At
each candidate decision it evaluates every legal root action across common
public-consistent hidden-hand hypotheses, then lets the exact frozen Normal
policy finish the current act.  The search never consumes the live ownership
of an opponent hand or a privileged critic tensor.
"""

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Callable, Mapping, Sequence

from v4_env import Card, DalmutiScalarEnv
from v4_evaluate import (
    canonical_json_bytes,
    deterministic_cluster_bootstrap95,
    rotating_candidate_seats,
)


REPORT_FORMAT = "dalmuti-v4-exact-public-search-diagnostic"
REPORT_VERSION = 1
ACTS_PER_MATCH = 5
MIN_PLAYERS = 4
MAX_PLAYERS = 10
MAX_UINT32 = 0xFFFF_FFFF
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DECK_COUNTS = tuple(range(1, 13)) + (2,)
SELECTION_MODES = ("mean", "lcb")


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derived_uint32(seed: int, *parts: object) -> int:
    material = "\0".join((str(seed), *(str(part) for part in parts))).encode(
        "utf-8"
    )
    value = int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
    return value or 1


def _tensor_json_value(value: object) -> object:
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_list()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError("actor-public observation contains an unsupported value")


def _public_snapshot(env: DalmutiScalarEnv) -> dict[str, object]:
    """Return only the fields exposed at the V4 actor boundary."""

    public = env.public_observation()
    return {
        "actorId": int(public.actor_id),
        "valid": _tensor_json_value(public.valid),
        "globalFeatures": _tensor_json_value(public.global_features),
        "rankFeatures": _tensor_json_value(public.rank_features),
        "playerFeatures": _tensor_json_value(public.player_features),
        "playerMask": _tensor_json_value(public.player_mask),
        "memoryTraceFeatures": _tensor_json_value(public.memory_trace_features),
        "historyFeatures": _tensor_json_value(public.history_features),
        "historyMask": _tensor_json_value(public.history_mask),
        "legalMask": _tensor_json_value(public.legal_mask),
    }


def public_observation_sha256(env: DalmutiScalarEnv) -> str:
    return hashlib.sha256(canonical_json_bytes(_public_snapshot(env))).hexdigest()


def _rounded_count(value: object, scale: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    scaled = float(value) * scale
    count = int(round(scaled))
    if count < 0 or count > scale or abs(scaled - count) > 1e-4:
        raise ValueError(f"{label} does not encode an integral public count")
    return count


def _cards(prefix: str, counts: Sequence[int]) -> list[Card]:
    result: list[Card] = []
    for rank, count in enumerate(counts, start=1):
        result.extend(
            Card(f"search-{prefix}-r{rank:02d}-c{copy_index:02d}", rank)
            for copy_index in range(int(count))
        )
    return result


def _canonical_public_root(env: DalmutiScalarEnv) -> DalmutiScalarEnv:
    """Clone ``env`` and erase live hidden ownership before determinization.

    Opponent cards are reconstructed solely from actor-visible own-card counts,
    public played counts, and public hand sizes.  Replacing the complete hand
    mapping (rather than inspecting any opponent list) makes the subsequent
    ``resample_hidden_hands(seed)`` invariant to the live hidden allocation.
    """

    root = copy.deepcopy(env)
    before = public_observation_sha256(root)
    public = root.public_observation()
    actor_id = int(public.actor_id)
    if actor_id < 0:
        raise RuntimeError("cannot search a terminated environment")

    rank_rows = _tensor_json_value(public.rank_features)
    player_rows = _tensor_json_value(public.player_features)
    player_mask = _tensor_json_value(public.player_mask)
    if not isinstance(rank_rows, list) or len(rank_rows) != len(DECK_COUNTS):
        raise ValueError("V4 actor rank features have an unexpected shape")
    own_counts: list[int] = []
    played_counts: list[int] = []
    for rank_index, copies in enumerate(DECK_COUNTS):
        row = rank_rows[rank_index]
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError("V4 actor rank feature row is malformed")
        own_counts.append(
            _rounded_count(row[0], copies, f"rank {rank_index + 1} own count")
        )
        played_counts.append(
            _rounded_count(row[1], copies, f"rank {rank_index + 1} played count")
        )
    unseen_counts = [
        copies - own - played
        for copies, own, played in zip(DECK_COUNTS, own_counts, played_counts)
    ]
    if any(value < 0 for value in unseen_counts):
        raise ValueError("actor-public card counts exceed the physical deck")

    # Player identities and social order are public game state.  Card ownership
    # is not read here; the old hand mapping is replaced wholesale below.
    order = tuple(int(value) for value in getattr(root, "_order"))
    if len(order) != root.player_count or set(order) != set(range(root.player_count)):
        raise ValueError("environment public player order is invalid")
    actor_position = order.index(actor_id)
    relative_ids = tuple(
        order[(actor_position + offset) % root.player_count]
        for offset in range(root.player_count)
    )
    if not isinstance(player_rows, list) or not isinstance(player_mask, list):
        raise ValueError("V4 actor player features have an unexpected shape")
    public_hand_counts: list[int] = []
    for offset in range(root.player_count):
        if offset >= len(player_mask) or not bool(player_mask[offset]):
            raise ValueError("active V4 player row is missing")
        row = player_rows[offset]
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError("V4 actor player feature row is malformed")
        public_hand_counts.append(
            _rounded_count(row[1], 20, f"player offset {offset} hand count")
        )
    if public_hand_counts[0] != sum(own_counts):
        raise ValueError("public self hand size disagrees with rank counts")
    if sum(public_hand_counts[1:]) != sum(unseen_counts):
        raise ValueError("public opponent hand sizes disagree with unseen deck")

    unseen = _cards("unseen", unseen_counts)
    cursor = 0
    rebuilt: dict[int, list[Card]] = {
        actor_id: _cards("own", own_counts),
    }
    for offset, player_id in enumerate(relative_ids[1:], start=1):
        count = public_hand_counts[offset]
        rebuilt[player_id] = unseen[cursor : cursor + count]
        cursor += count
    if cursor != len(unseen) or set(rebuilt) != set(order):
        raise AssertionError("canonical public hidden pool was not fully assigned")
    root._hands = rebuilt
    after = public_observation_sha256(root)
    if after != before:
        raise AssertionError("canonicalization changed actor-public state")
    return root


def _legal_actions(env: DalmutiScalarEnv) -> tuple[int, ...]:
    mask = env.legal_mask()
    values = getattr(mask, "nonzero")().flatten().detach().cpu().tolist()
    result = tuple(int(value) for value in values)
    if not result:
        raise RuntimeError("active environment has no legal action")
    return result


def _is_legal(env: DalmutiScalarEnv, action: int) -> bool:
    mask = env.legal_mask()
    if action < 0 or action >= int(mask.numel()):
        return False
    return bool(mask[action].item())


@dataclass(frozen=True)
class SearchConfig:
    seed: int
    hypotheses: int = 4
    selection: str = "lcb"
    lcb_z: float = 1.0
    max_rollout_steps: int = 2048

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("search seed must be an integer")
        _positive_integer(self.hypotheses, "hypotheses")
        if self.selection not in SELECTION_MODES:
            raise ValueError(f"selection must be one of {SELECTION_MODES}")
        _finite_nonnegative(self.lcb_z, "lcb-z")
        _positive_integer(self.max_rollout_steps, "maximum rollout steps")


@dataclass(frozen=True)
class ActOutcome:
    chip_award: int
    finish_place: int
    environment_reward: float
    simulated_steps: int
    exact_normal_continuation_steps: int


@dataclass(frozen=True)
class RootActionStats:
    action: int
    samples: int
    mean_chip: float
    standard_error_chip: float
    lcb_chip: float
    mean_place: float
    first_rate: float
    last_rate: float
    minimum_chip: int
    maximum_chip: int


@dataclass(frozen=True)
class RootSearchResult:
    action: int
    normal_action: int
    forced: bool
    deviated_from_normal: bool
    public_observation_sha256: str
    hypothesis_seeds: tuple[int, ...]
    action_stats: tuple[RootActionStats, ...]
    determinizations: int
    root_action_evaluations: int
    simulated_steps: int
    exact_normal_continuation_steps: int
    elapsed_seconds: float

    def report_value(self) -> dict[str, object]:
        return {
            "action": self.action,
            "normalAction": self.normal_action,
            "forced": self.forced,
            "deviatedFromNormal": self.deviated_from_normal,
            "publicObservationSha256": self.public_observation_sha256,
            "hypothesisSeeds": list(self.hypothesis_seeds),
            "actionStats": [asdict(value) for value in self.action_stats],
            "determinizations": self.determinizations,
            "rootActionEvaluations": self.root_action_evaluations,
            "simulatedSteps": self.simulated_steps,
            "exactNormalContinuationSteps": self.exact_normal_continuation_steps,
            "elapsedSeconds": self.elapsed_seconds,
        }


def _extract_act_outcome(
    step_result: object,
    *,
    root_actor_id: int,
    player_count: int,
    simulated_steps: int,
) -> ActOutcome | None:
    info = getattr(step_result, "info", None)
    if not isinstance(info, Mapping):
        raise ValueError("environment step omitted its info mapping")
    raw = info.get("act_result")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("act result is not a mapping")
    finish_order = tuple(int(value) for value in raw.get("finish_order", ()))
    awards = raw.get("chip_awards")
    if len(finish_order) != player_count or not isinstance(awards, Mapping):
        raise ValueError("environment returned an invalid act result")
    if root_actor_id not in finish_order:
        raise ValueError("root actor is missing from act finish order")
    place = finish_order.index(root_actor_id) + 1
    raw_award = awards.get(root_actor_id)
    if raw_award is None:
        raw_award = awards.get(str(root_actor_id))
    if isinstance(raw_award, bool) or not isinstance(raw_award, (int, float)):
        raise ValueError("root actor chip award is invalid")
    chip = int(raw_award)
    if float(raw_award) != chip:
        raise ValueError("root actor chip award is not integral")
    rewards = getattr(step_result, "rewards", None)
    if rewards is None:
        raise ValueError("environment step omitted rewards")
    value = rewards[root_actor_id]
    if hasattr(value, "item"):
        value = value.item()
    reward = float(value)
    expected_reward = (chip - 2) / 2.0
    if abs(reward - expected_reward) > 1e-7:
        raise ValueError("environment reward disagrees with actual act chip award")
    return ActOutcome(
        chip_award=chip,
        finish_place=place,
        environment_reward=reward,
        simulated_steps=simulated_steps,
        exact_normal_continuation_steps=simulated_steps - 1,
    )


def _rollout_root_action(
    env: DalmutiScalarEnv,
    action: int,
    *,
    root_actor_id: int,
    max_rollout_steps: int,
) -> ActOutcome:
    """Apply one root action, then use exact Normal until this act ends."""

    if env.current_player_id != root_actor_id:
        raise ValueError("root actor changed before action evaluation")
    if not _is_legal(env, action):
        raise ValueError(f"root action {action} is illegal")
    root_act = int(getattr(env, "_act"))
    steps = 1
    result = env.step(action)
    outcome = _extract_act_outcome(
        result,
        root_actor_id=root_actor_id,
        player_count=env.player_count,
        simulated_steps=steps,
    )
    while outcome is None:
        if steps >= max_rollout_steps:
            raise RuntimeError(
                f"root continuation exceeded {max_rollout_steps} simulated steps"
            )
        if env.terminated:
            raise RuntimeError("match terminated without an act result")
        if int(getattr(env, "_act")) != root_act:
            raise RuntimeError("search crossed the root act boundary")
        normal_action = int(env.normal_action())
        if not _is_legal(env, normal_action):
            raise ValueError("exact Normal continuation selected an illegal action")
        steps += 1
        result = env.step(normal_action)
        outcome = _extract_act_outcome(
            result,
            root_actor_id=root_actor_id,
            player_count=env.player_count,
            simulated_steps=steps,
        )
    return outcome


def _action_statistics(
    action: int,
    outcomes: Sequence[ActOutcome],
    *,
    player_count: int,
    lcb_z: float,
) -> RootActionStats:
    if not outcomes:
        raise ValueError("a legal root action has no hypothesis outcomes")
    chips = [value.chip_award for value in outcomes]
    places = [value.finish_place for value in outcomes]
    mean_chip = sum(chips) / len(chips)
    if len(chips) == 1:
        standard_error = 0.0
    else:
        variance = sum((value - mean_chip) ** 2 for value in chips) / (
            len(chips) - 1
        )
        standard_error = math.sqrt(variance / len(chips))
    return RootActionStats(
        action=action,
        samples=len(outcomes),
        mean_chip=mean_chip,
        standard_error_chip=standard_error,
        lcb_chip=mean_chip - lcb_z * standard_error,
        mean_place=sum(places) / len(places),
        first_rate=sum(value == 1 for value in places) / len(places),
        last_rate=sum(value == player_count for value in places) / len(places),
        minimum_chip=min(chips),
        maximum_chip=max(chips),
    )


def evaluate_root_actions(
    env: DalmutiScalarEnv,
    config: SearchConfig,
    *,
    action_order: Sequence[int] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> RootSearchResult:
    """Choose an action using common public-consistent determinizations."""

    started = float(clock())
    legal_actions = _legal_actions(env)
    normal_action = int(env.normal_action())
    if normal_action not in legal_actions:
        raise ValueError("exact Normal root action is illegal")
    public_digest = public_observation_sha256(env)
    if len(legal_actions) == 1:
        elapsed = max(0.0, float(clock()) - started)
        return RootSearchResult(
            action=normal_action,
            normal_action=normal_action,
            forced=True,
            deviated_from_normal=False,
            public_observation_sha256=public_digest,
            hypothesis_seeds=(),
            action_stats=(),
            determinizations=0,
            root_action_evaluations=0,
            simulated_steps=0,
            exact_normal_continuation_steps=0,
            elapsed_seconds=elapsed,
        )

    ordered_actions = (
        tuple(int(value) for value in action_order)
        if action_order is not None
        else legal_actions
    )
    if len(ordered_actions) != len(legal_actions) or set(ordered_actions) != set(
        legal_actions
    ):
        raise ValueError("action order must be a permutation of every legal action")
    if len(set(ordered_actions)) != len(ordered_actions):
        raise ValueError("action order contains duplicates")

    root_actor_id = int(env.current_player_id)
    canonical_root = _canonical_public_root(env)
    if public_observation_sha256(env) != public_digest:
        raise AssertionError("search preparation mutated the live environment")
    hypothesis_seeds = tuple(
        _derived_uint32(config.seed, "public-hypothesis", public_digest, index)
        for index in range(config.hypotheses)
    )
    outcomes: dict[int, list[ActOutcome]] = {
        action: [] for action in legal_actions
    }
    total_steps = 0
    normal_steps = 0
    for hypothesis_seed in hypothesis_seeds:
        determinized = copy.deepcopy(canonical_root)
        # This is the only hidden-hand sampling operation.  Its input ownership
        # was reconstructed above from actor-public counts, not copied live.
        determinized.resample_hidden_hands(hypothesis_seed)
        if public_observation_sha256(determinized) != public_digest:
            raise AssertionError("hidden-hand determinization changed public state")
        for action in ordered_actions:
            branch = copy.deepcopy(determinized)
            outcome = _rollout_root_action(
                branch,
                action,
                root_actor_id=root_actor_id,
                max_rollout_steps=config.max_rollout_steps,
            )
            outcomes[action].append(outcome)
            total_steps += outcome.simulated_steps
            normal_steps += outcome.exact_normal_continuation_steps

    statistics = tuple(
        _action_statistics(
            action,
            outcomes[action],
            player_count=env.player_count,
            lcb_z=config.lcb_z,
        )
        for action in sorted(legal_actions)
    )

    def selection_key(value: RootActionStats) -> tuple[float, float, float, int]:
        primary = value.mean_chip if config.selection == "mean" else value.lcb_chip
        # Mean chip is the actual objective; mean finish breaks equal-chip
        # plateaus.  The final action-index tie break is stable and order-free.
        return (primary, value.mean_chip, -value.mean_place, -value.action)

    selected = max(statistics, key=selection_key).action
    elapsed = max(0.0, float(clock()) - started)
    return RootSearchResult(
        action=selected,
        normal_action=normal_action,
        forced=False,
        deviated_from_normal=selected != normal_action,
        public_observation_sha256=public_digest,
        hypothesis_seeds=hypothesis_seeds,
        action_stats=statistics,
        determinizations=len(hypothesis_seeds),
        root_action_evaluations=len(hypothesis_seeds) * len(legal_actions),
        simulated_steps=total_steps,
        exact_normal_continuation_steps=normal_steps,
        elapsed_seconds=elapsed,
    )


def _outcome_totals() -> dict[str, float | int]:
    return {"chips": 0.0, "places": 0, "firsts": 0, "lasts": 0, "seatActs": 0}


def _record_outcome(
    totals: dict[str, float | int],
    *,
    chip: float,
    place: int,
    player_count: int,
) -> None:
    totals["chips"] = float(totals["chips"]) + chip
    totals["places"] = int(totals["places"]) + place
    totals["firsts"] = int(totals["firsts"]) + int(place == 1)
    totals["lasts"] = int(totals["lasts"]) + int(place == player_count)
    totals["seatActs"] = int(totals["seatActs"]) + 1


def _summarize_outcomes(
    totals: Mapping[str, float | int]
) -> dict[str, float | int | None]:
    count = int(totals["seatActs"])
    if count == 0:
        return {
            "meanChip": None,
            "meanPlace": None,
            "firstRate": None,
            "lastRate": None,
            "seatActs": 0,
        }
    return {
        "meanChip": float(totals["chips"]) / count,
        "meanPlace": int(totals["places"]) / count,
        "firstRate": int(totals["firsts"]) / count,
        "lastRate": int(totals["lasts"]) / count,
        "seatActs": count,
    }


def _pairwise_finish(
    finish_order: Sequence[int], candidate_ids: set[int]
) -> tuple[int, int]:
    before = comparisons = 0
    for left_index, left in enumerate(finish_order):
        left_candidate = int(left) in candidate_ids
        for right in finish_order[left_index + 1 :]:
            right_candidate = int(right) in candidate_ids
            if left_candidate == right_candidate:
                continue
            comparisons += 1
            before += int(left_candidate)
    if comparisons < 1:
        raise ValueError("candidate/Normal act has no pairwise comparison")
    return before, comparisons


def _act_result(step_result: object, player_count: int) -> Mapping[str, object] | None:
    info = getattr(step_result, "info", None)
    if not isinstance(info, Mapping):
        raise ValueError("environment step omitted its info mapping")
    value = info.get("act_result")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("act result is not a mapping")
    finish = tuple(int(player_id) for player_id in value.get("finish_order", ()))
    awards = value.get("chip_awards")
    if len(finish) != player_count or set(finish) != set(range(player_count)):
        raise ValueError("act result finish order is invalid")
    if not isinstance(awards, Mapping):
        raise ValueError("act result chip awards are invalid")
    return value


def _bootstrap_seed(base_seed: int, player_count: int) -> int:
    return _derived_uint32(base_seed, "exact-public-search-bootstrap", player_count)


def evaluate_player_count_search(
    *,
    player_count: int,
    matches: int,
    acts: int,
    base_seed: int,
    search_config: SearchConfig,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    env_factory: Callable[[int, int, int], DalmutiScalarEnv] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    """Run a small mixed-seat, seed-matched search-vs-Normal diagnostic."""

    if not MIN_PLAYERS <= player_count <= MAX_PLAYERS:
        raise ValueError("player count must be from 4 through 10")
    _positive_integer(matches, "matches")
    _positive_integer(acts, "acts")
    _positive_integer(base_seed, "base seed")
    _positive_integer(bootstrap_resamples, "bootstrap resamples")
    last_seed = base_seed + player_count * 1_000_000 + matches - 1
    if last_seed > MAX_UINT32:
        raise ValueError("match seed exceeds uint32")
    make_env = env_factory or (
        lambda count, act_count, seed: DalmutiScalarEnv(
            count, acts=act_count, seed=seed, device="cpu"
        )
    )
    started = float(clock())
    groups = {"candidate": _outcome_totals(), "normal": _outcome_totals()}
    cluster_differences: list[float] = []
    cluster_records: list[dict[str, object]] = []
    pairwise_before = pairwise_comparisons = 0
    all_decisions = candidate_decisions = forced_decisions = search_decisions = 0
    normal_deviations = determinizations = root_evaluations = 0
    simulated_steps = normal_continuation_steps = 0
    search_elapsed = 0.0
    completed_acts = 0

    for match_index in range(matches):
        match_seed = base_seed + player_count * 1_000_000 + match_index
        env = make_env(player_count, acts, match_seed)
        initial_order = tuple(int(value) for value in getattr(env, "_order"))
        candidate_seats = rotating_candidate_seats(player_count, match_index)
        candidate_ids = {initial_order[index] for index in candidate_seats}
        match_groups = {
            "candidate": _outcome_totals(),
            "normal": _outcome_totals(),
        }
        match_before = match_comparisons = 0
        match_decision_index = 0
        while not env.terminated:
            actor_id = int(env.current_player_id)
            if actor_id in candidate_ids:
                candidate_decisions += 1
                decision_seed = _derived_uint32(
                    search_config.seed,
                    "match-decision",
                    player_count,
                    match_seed,
                    match_decision_index,
                    actor_id,
                )
                decision = evaluate_root_actions(
                    env,
                    replace(search_config, seed=decision_seed),
                    clock=clock,
                )
                action = decision.action
                forced_decisions += int(decision.forced)
                search_decisions += int(not decision.forced)
                normal_deviations += int(decision.deviated_from_normal)
                determinizations += decision.determinizations
                root_evaluations += decision.root_action_evaluations
                simulated_steps += decision.simulated_steps
                normal_continuation_steps += decision.exact_normal_continuation_steps
                search_elapsed += decision.elapsed_seconds
            else:
                action = int(env.normal_action())
            if not _is_legal(env, action):
                raise ValueError(f"screening policy selected illegal action {action}")
            result = env.step(action)
            all_decisions += 1
            match_decision_index += 1
            act_result = _act_result(result, player_count)
            if act_result is None:
                continue
            finish_order = tuple(
                int(value) for value in act_result["finish_order"]  # type: ignore[index]
            )
            awards = act_result["chip_awards"]  # type: ignore[index]
            assert isinstance(awards, Mapping)
            for place, player_id in enumerate(finish_order, start=1):
                raw_chip = awards.get(player_id)
                if raw_chip is None:
                    raw_chip = awards.get(str(player_id))
                if isinstance(raw_chip, bool) or not isinstance(raw_chip, (int, float)):
                    raise ValueError("act result contains an invalid chip award")
                chip = float(raw_chip)
                group = "candidate" if player_id in candidate_ids else "normal"
                _record_outcome(
                    groups[group], chip=chip, place=place, player_count=player_count
                )
                _record_outcome(
                    match_groups[group],
                    chip=chip,
                    place=place,
                    player_count=player_count,
                )
            before, comparisons = _pairwise_finish(finish_order, candidate_ids)
            pairwise_before += before
            pairwise_comparisons += comparisons
            match_before += before
            match_comparisons += comparisons
            completed_acts += 1

        candidate_summary = _summarize_outcomes(match_groups["candidate"])
        normal_summary = _summarize_outcomes(match_groups["normal"])
        if candidate_summary["meanChip"] is None or normal_summary["meanChip"] is None:
            raise ValueError("each match needs both candidate and Normal outcomes")
        difference = float(candidate_summary["meanChip"]) - float(
            normal_summary["meanChip"]
        )
        cluster_differences.append(difference)
        cluster_records.append(
            {
                "matchIndex": match_index,
                "seed": match_seed,
                "candidateInitialSeats": list(candidate_seats),
                "meanChipDifference": difference,
                "candidateBefore": match_before,
                "comparisons": match_comparisons,
            }
        )

    if completed_acts != matches * acts:
        raise ValueError(
            f"environment completed {completed_acts} acts; expected {matches * acts}"
        )
    inference = deterministic_cluster_bootstrap95(
        cluster_differences,
        seed=_bootstrap_seed(base_seed, player_count),
        resamples=bootstrap_resamples,
    )
    candidate = _summarize_outcomes(groups["candidate"])
    normal = _summarize_outcomes(groups["normal"])
    pairwise_rate = pairwise_before / pairwise_comparisons
    elapsed = max(0.0, float(clock()) - started)
    cluster_payload = canonical_json_bytes(cluster_records)
    return {
        "playerCount": player_count,
        "matches": matches,
        "actsPerMatch": acts,
        "actCount": completed_acts,
        "matchSeedRange": {
            "start": base_seed + player_count * 1_000_000,
            "end": last_seed,
        },
        "candidate": candidate,
        "normal": normal,
        "meanChipDifference": inference["mean"],
        "meanChipDifference95": {
            "low": inference["low"],
            "high": inference["high"],
        },
        "meanChipDifferenceInference": inference,
        "pairwiseCandidateBeforeNormal": {
            "candidateBefore": pairwise_before,
            "comparisons": pairwise_comparisons,
            "rate": pairwise_rate,
        },
        "decisionAudit": {
            "allMatchDecisions": all_decisions,
            "candidateDecisions": candidate_decisions,
            "forcedCandidateDecisions": forced_decisions,
            "searchDecisions": search_decisions,
            "normalDeviations": normal_deviations,
            "normalDeviationRateAmongSearchDecisions": (
                None if search_decisions == 0 else normal_deviations / search_decisions
            ),
        },
        "searchWork": {
            "uniquePublicHypothesesRequested": determinizations,
            "determinizations": determinizations,
            "rootActionEvaluations": root_evaluations,
            "simulatedSteps": simulated_steps,
            "exactNormalContinuationSteps": normal_continuation_steps,
            "summedSearchElapsedSeconds": search_elapsed,
        },
        "matchClusters": {
            "unit": "seed-matched-match",
            "count": matches,
            "sha256": hashlib.sha256(cluster_payload).hexdigest(),
        },
        "statisticallyAboveNormal": float(inference["low"]) > 0.0,
        "elapsedSeconds": elapsed,
    }


def _source_hashes(repository_root: Path) -> dict[str, str]:
    relative_paths = (
        "gpu-training/v4_exact_search_screen.py",
        "gpu-training/v4_env.py",
        "gpu-training/v4_evaluate.py",
        "lib/bot-strategy.ts",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source is missing: {path}")
        result[relative] = _sha256_file(path)
    return result


def run_exact_search_diagnostic(
    *,
    min_player_count: int,
    max_player_count: int,
    matches: int,
    acts: int,
    base_seed: int,
    hypotheses: int,
    selection: str,
    lcb_z: float,
    max_rollout_steps: int,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    repository_root: Path | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    if not MIN_PLAYERS <= min_player_count <= max_player_count <= MAX_PLAYERS:
        raise ValueError("player-count range must be within p4 through p10")
    if acts != ACTS_PER_MATCH:
        raise ValueError("the V4 diagnostic requires exactly five acts")
    root = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parent.parent
    )
    config = SearchConfig(
        seed=base_seed,
        hypotheses=hypotheses,
        selection=selection,
        lcb_z=lcb_z,
        max_rollout_steps=max_rollout_steps,
    )
    started = float(clock())
    results = [
        evaluate_player_count_search(
            player_count=player_count,
            matches=matches,
            acts=acts,
            base_seed=base_seed,
            search_config=config,
            bootstrap_resamples=bootstrap_resamples,
            clock=clock,
        )
        for player_count in range(min_player_count, max_player_count + 1)
    ]
    elapsed = max(0.0, float(clock()) - started)
    total_work_keys = (
        "determinizations",
        "rootActionEvaluations",
        "simulatedSteps",
        "exactNormalContinuationSteps",
    )
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "certification": {
            "status": "research-only-not-certification",
            "promotionEligible": False,
            "claim": "This bounded diagnostic does not certify a deployable model.",
        },
        "configuration": {
            "playerCountRange": [min_player_count, max_player_count],
            "matchesPerPlayerCount": matches,
            "actsPerMatch": acts,
            "baseSeed": base_seed,
            "hypothesesPerSearchDecision": hypotheses,
            "selection": selection,
            "lcbZ": float(lcb_z),
            "maxRolloutStepsPerRootEvaluation": max_rollout_steps,
            "bootstrapResamples": bootstrap_resamples,
            "candidateSeatAssignment": "rotating-fixed-policy-identity-within-match",
            "continuationPolicy": "exact-frozen-normal-only",
            "rootActionCoverage": "every-legal-action",
            "rootActionRandomness": "common-public-hypothesis-seeds",
            "rootTieBreak": "metric-then-mean-chip-then-mean-place-then-low-action-index",
        },
        "privacyAudit": {
            "strictPublicInformationOnly": True,
            "liveOpponentOwnershipReadOrBranchedOn": False,
            "privilegedCriticStateConsumed": False,
            "canonicalOpponentPoolBuiltFromActorPublicCounts": True,
            "determinizationsUseDalmutiScalarEnvCopies": True,
            "resampleHiddenHandsCalledForEveryPublicHypothesis": True,
            "statement": (
                "Before resampling, every copied hand is replaced from actor-visible "
                "own/public rank counts and public hand sizes. Search neither reads nor "
                "branches on the live allocation of opponent hands; it never consumes "
                "the privileged critic tensor. Each continuation uses exact Normal only."
            ),
        },
        "sourceHashes": _source_hashes(root),
        "results": results,
        "aggregate": {
            "playerCounts": len(results),
            "matches": sum(int(value["matches"]) for value in results),
            "acts": sum(int(value["actCount"]) for value in results),
            "searchDecisions": sum(
                int(value["decisionAudit"]["searchDecisions"])  # type: ignore[index]
                for value in results
            ),
            "normalDeviations": sum(
                int(value["decisionAudit"]["normalDeviations"])  # type: ignore[index]
                for value in results
            ),
            **{
                key: sum(
                    int(value["searchWork"][key])  # type: ignore[index]
                    for value in results
                )
                for key in total_work_keys
            },
            "elapsedSeconds": elapsed,
        },
        "reproducibility": {
            "matchSeeds": "baseSeed + playerCount * 1000000 + matchIndex",
            "searchSeeds": "sha256(base seed, match seed, decision index, actor id)",
            "hypothesisSeeds": "sha256(search seed, public observation sha256, index)",
            "rootEvaluationOrderAffectsResult": False,
            "timingFieldsExcludedFromSemanticReproducibility": True,
        },
    }


def validate_diagnostic_report(report: Mapping[str, object]) -> None:
    if report.get("format") != REPORT_FORMAT or report.get("version") != REPORT_VERSION:
        raise ValueError("exact-search diagnostic format/version mismatch")
    certification = report.get("certification")
    if not isinstance(certification, Mapping):
        raise ValueError("diagnostic certification disclaimer is missing")
    if (
        certification.get("status") != "research-only-not-certification"
        or certification.get("promotionEligible") is not False
    ):
        raise ValueError("research diagnostic must not claim certification")
    privacy = report.get("privacyAudit")
    if not isinstance(privacy, Mapping):
        raise ValueError("diagnostic privacy audit is missing")
    required_privacy = {
        "strictPublicInformationOnly": True,
        "liveOpponentOwnershipReadOrBranchedOn": False,
        "privilegedCriticStateConsumed": False,
        "canonicalOpponentPoolBuiltFromActorPublicCounts": True,
        "determinizationsUseDalmutiScalarEnvCopies": True,
        "resampleHiddenHandsCalledForEveryPublicHypothesis": True,
    }
    for key, expected in required_privacy.items():
        if privacy.get(key) is not expected:
            raise ValueError(f"diagnostic privacy audit failed: {key}")
    source_hashes = report.get("sourceHashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("diagnostic source hashes are missing")
    for label, digest in source_hashes.items():
        if (
            not isinstance(label, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("diagnostic source hash is invalid")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("diagnostic results must be a list")
    canonical_json_bytes(report)


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


def write_diagnostic_report_exclusive(
    output_path: str | Path, report: Mapping[str, object]
) -> dict[str, object]:
    validate_diagnostic_report(report)
    output = Path(output_path)
    checksum = output.with_name(output.name + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("diagnostic report and checksum are immutable")
    payload = canonical_json_bytes(report)
    digest = hashlib.sha256(payload).hexdigest()
    checksum_payload = f"{digest}  {output.name}\n".encode("ascii")
    _publish_exclusive(output, payload)
    try:
        _publish_exclusive(checksum, checksum_payload)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {
        "path": str(output),
        "sha256Path": str(checksum),
        "sha256": digest,
        "bytes": len(payload),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded research-only V4 exact public-search screen."
    )
    parser.add_argument("--min-player-count", "--p-min", type=int, default=4)
    parser.add_argument("--max-player-count", "--p-max", type=int, default=10)
    parser.add_argument("--matches", type=int, default=2)
    parser.add_argument("--acts", type=int, choices=(ACTS_PER_MATCH,), default=5)
    parser.add_argument("--base-seed", type=int, default=360_000_001)
    parser.add_argument("--hypotheses", type=int, default=4)
    parser.add_argument("--selection", choices=SELECTION_MODES, default="lcb")
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--max-rollout-steps", type=int, default=2048)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    report = run_exact_search_diagnostic(
        min_player_count=args.min_player_count,
        max_player_count=args.max_player_count,
        matches=args.matches,
        acts=args.acts,
        base_seed=args.base_seed,
        hypotheses=args.hypotheses,
        selection=args.selection,
        lcb_z=args.lcb_z,
        max_rollout_steps=args.max_rollout_steps,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    published = write_diagnostic_report_exclusive(args.output, report)
    print(json.dumps(published, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTS_PER_MATCH",
    "ActOutcome",
    "REPORT_FORMAT",
    "REPORT_VERSION",
    "RootActionStats",
    "RootSearchResult",
    "SearchConfig",
    "evaluate_player_count_search",
    "evaluate_root_actions",
    "public_observation_sha256",
    "run_exact_search_diagnostic",
    "validate_diagnostic_report",
    "write_diagnostic_report_exclusive",
]
