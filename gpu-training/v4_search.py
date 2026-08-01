from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import statistics
import time
from typing import Callable, Mapping, Protocol, Sequence, TypeVar, runtime_checkable


V4_SEARCH_ACTION_COUNT = 236
V4_SEARCH_SCHEMA_VERSION = 4
V4_SEARCH_MIN_PLAYERS = 4
V4_SEARCH_MAX_PLAYERS = 10
V4_SEARCH_RANK_COUNT = 13
V4_SEARCH_DECK_COUNTS = tuple(range(1, 13)) + (2,)

_TOP_LEVEL_KEYS = frozenset({
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
})
_TABLE_KEYS = frozenset({
    "actorOffset", "rank", "naturalCount", "jokerCount", "totalCount"
})
_PLAYER_KEYS = frozenset({
    "relativeOffset",
    "handCount",
    "finished",
    "passed",
    "self",
    "tableLeader",
    "role",
    "score",
})
_HISTORY_KEYS = frozenset({
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
})


def _integer(value: object, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be from {minimum} to {maximum}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value.keys())
    if actual != expected:
        unexpected = sorted(str(key) for key in actual - expected)
        missing = sorted(expected - actual)
        field = unexpected[0] if unexpected else missing[0]
        raise ValueError(
            f"{label} crossed the public-information boundary with an "
            f"unknown or missing field: {field}"
        )


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def validate_v4_search_public_observation(
    observation: Mapping[str, object],
) -> None:
    """Reject anything outside the identity-free V4 actor-visible contract."""

    value = _mapping(observation, "search observation")
    _exact_keys(value, _TOP_LEVEL_KEYS, "search observation")
    if value.get("schemaVersion") != V4_SEARCH_SCHEMA_VERSION:
        raise ValueError("search observation schema version mismatch")
    player_count = _integer(
        value.get("playerCount"),
        V4_SEARCH_MIN_PLAYERS,
        V4_SEARCH_MAX_PLAYERS,
        "playerCount",
    )
    _integer(value.get("act"), 1, 1_000_000, "act")
    _integer(value.get("actorRole"), 0, 4, "actorRole")
    _integer(value.get("revolution"), 0, 2, "revolution")
    _integer(
        value.get("truncatedHistoryCount"),
        0,
        1_000_000_000,
        "truncatedHistoryCount",
    )

    own_counts = _list(value.get("ownHandCounts"), "ownHandCounts")
    public_counts = _list(
        value.get("publicPlayedCounts"), "publicPlayedCounts"
    )
    if len(own_counts) != V4_SEARCH_RANK_COUNT or len(public_counts) != V4_SEARCH_RANK_COUNT:
        raise ValueError("card-count arrays must contain exactly 13 ranks")
    for rank_index, deck_count in enumerate(V4_SEARCH_DECK_COUNTS):
        own = _integer(
            own_counts[rank_index], 0, deck_count, f"ownHandCounts[{rank_index}]"
        )
        played = _integer(
            public_counts[rank_index],
            0,
            deck_count,
            f"publicPlayedCounts[{rank_index}]",
        )
        if own + played > deck_count:
            raise ValueError("own and publicly played cards exceed the deck")

    table = value.get("table")
    if table is not None:
        table_value = _mapping(table, "table")
        _exact_keys(table_value, _TABLE_KEYS, "table")
        _integer(table_value.get("actorOffset"), 0, player_count - 1, "table.actorOffset")
        _integer(table_value.get("rank"), 1, 13, "table.rank")
        natural = _integer(
            table_value.get("naturalCount"), 0, 14, "table.naturalCount"
        )
        jokers = _integer(table_value.get("jokerCount"), 0, 2, "table.jokerCount")
        total = _integer(table_value.get("totalCount"), 1, 14, "table.totalCount")
        if natural + jokers != total:
            raise ValueError("table natural and joker counts must equal totalCount")

    players = _list(value.get("playerTokens"), "playerTokens")
    if len(players) != player_count:
        raise ValueError("playerTokens must match playerCount")
    offsets: set[int] = set()
    self_offsets: list[int] = []
    for index, player in enumerate(players):
        player_value = _mapping(player, f"playerTokens[{index}]")
        _exact_keys(player_value, _PLAYER_KEYS, f"playerTokens[{index}]")
        offset = _integer(
            player_value.get("relativeOffset"),
            0,
            player_count - 1,
            f"playerTokens[{index}].relativeOffset",
        )
        if offset in offsets:
            raise ValueError("player relative offsets must be unique")
        if offset != index:
            raise ValueError("playerTokens must be ordered by relative offset")
        offsets.add(offset)
        hand_count = _integer(
            player_value.get("handCount"), 0, 20, f"playerTokens[{index}].handCount"
        )
        for flag in ("finished", "passed", "self", "tableLeader"):
            flag_value = _integer(
                player_value.get(flag), 0, 1, f"playerTokens[{index}].{flag}"
            )
            if flag == "self" and flag_value == 1:
                self_offsets.append(offset)
        _integer(player_value.get("role"), 0, 4, f"playerTokens[{index}].role")
        _finite_number(player_value.get("score"), f"playerTokens[{index}].score")
        if player_value.get("finished") == 1 and hand_count != 0:
            raise ValueError("finished players must have zero public hand count")
    if offsets != set(range(player_count)) or self_offsets != [0]:
        raise ValueError("player offsets must cover the table with relative offset 0 as self")
    if int(players[0]["handCount"]) != sum(int(count) for count in own_counts):
        # playerTokens are canonical relative-order tokens, hence item 0 is self.
        raise ValueError("self hand count does not match ownHandCounts")

    history = _list(value.get("historyTokens"), "historyTokens")
    if len(history) > 192:
        raise ValueError("historyTokens exceeds the V4 192-event limit")
    last_sequence = -1
    for index, event in enumerate(history):
        event_value = _mapping(event, f"historyTokens[{index}]")
        _exact_keys(event_value, _HISTORY_KEYS, f"historyTokens[{index}]")
        sequence = _integer(
            event_value.get("sequence"), 0, 1_000_000_000, "history.sequence"
        )
        if sequence <= last_sequence:
            raise ValueError("history sequence must be strictly increasing")
        last_sequence = sequence
        _integer(event_value.get("type"), 0, 3, "history.type")
        _integer(event_value.get("actorOffset"), 0, player_count - 1, "history.actorOffset")
        _integer(event_value.get("handCountBefore"), 0, 20, "history.handCountBefore")
        _integer(event_value.get("handCountAfter"), 0, 20, "history.handCountAfter")
        _integer(event_value.get("rank"), 0, 13, "history.rank")
        _integer(event_value.get("naturalCount"), 0, 14, "history.naturalCount")
        _integer(event_value.get("jokerCount"), 0, 2, "history.jokerCount")
        _integer(event_value.get("totalCount"), 0, 14, "history.totalCount")
        _integer(event_value.get("passReason"), 0, 4, "history.passReason")
        _integer(event_value.get("clearReason"), 0, 3, "history.clearReason")
        _integer(
            event_value.get("nextLeaderOffset"),
            -1,
            player_count - 1,
            "history.nextLeaderOffset",
        )
        _integer(event_value.get("finishPlace"), 0, player_count, "history.finishPlace")

    memory = _list(value.get("memoryTraceVectors"), "memoryTraceVectors")
    if len(memory) != 4:
        raise ValueError("memoryTraceVectors must contain four EMA traces")
    for trace_index, trace in enumerate(memory):
        trace_values = _list(trace, f"memoryTraceVectors[{trace_index}]")
        if len(trace_values) != 20:
            raise ValueError("each memory trace must contain exactly 20 features")
        for feature_index, feature in enumerate(trace_values):
            _finite_number(
                feature,
                f"memoryTraceVectors[{trace_index}][{feature_index}]",
            )


def _canonical_public_copy(
    observation: Mapping[str, object],
) -> Mapping[str, object]:
    validate_v4_search_public_observation(observation)
    try:
        encoded = json.dumps(
            observation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("search observation must be canonical JSON data") from error
    copied = json.loads(encoded.decode("utf-8"))
    validate_v4_search_public_observation(copied)
    return copied


def _validate_legal_mask(mask: Sequence[bool], label: str) -> tuple[bool, ...]:
    if len(mask) != V4_SEARCH_ACTION_COUNT:
        raise ValueError(f"{label} must contain exactly 236 actions")
    result: list[bool] = []
    for index, value in enumerate(mask):
        if not isinstance(value, bool):
            raise ValueError(f"{label}[{index}] must be boolean")
        result.append(value)
    if not any(result):
        raise ValueError(f"{label} must contain a legal action")
    return tuple(result)


def _derived_seed(root_seed: int, *components: int) -> int:
    payload = ":".join(str(value) for value in (root_seed, *components)).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass(frozen=True)
class V4Determinization:
    rank_counts_by_relative_offset: tuple[tuple[int, ...], ...]
    hypothesis_seed: int
    sha256: str


def determinize_v4_unseen_hands(
    observation: Mapping[str, object], *, seed: int
) -> V4Determinization:
    """Sample unseen cards using only own-hand and public-card constraints."""

    public = _canonical_public_copy(observation)
    player_count = int(public["playerCount"])
    own_counts = tuple(int(value) for value in public["ownHandCounts"])
    public_counts = tuple(int(value) for value in public["publicPlayedCounts"])
    unseen_cards: list[int] = []
    for rank_index, deck_count in enumerate(V4_SEARCH_DECK_COUNTS):
        unseen_cards.extend(
            [rank_index + 1] * (deck_count - own_counts[rank_index] - public_counts[rank_index])
        )
    players = sorted(
        (dict(player) for player in public["playerTokens"]),
        key=lambda player: int(player["relativeOffset"]),
    )
    opponent_targets = [int(player["handCount"]) for player in players[1:]]
    if sum(opponent_targets) != len(unseen_cards):
        raise ValueError(
            "public hand counts do not match the unseen deck; determinization is impossible"
        )
    rng = random.Random(int(seed))
    rng.shuffle(unseen_cards)
    hands: list[tuple[int, ...]] = [own_counts]
    cursor = 0
    for target in opponent_targets:
        counts = [0] * V4_SEARCH_RANK_COUNT
        for rank in unseen_cards[cursor : cursor + target]:
            counts[rank - 1] += 1
        hands.append(tuple(counts))
        cursor += target
    if cursor != len(unseen_cards) or len(hands) != player_count:
        raise RuntimeError("determinization did not consume the unseen deck")
    canonical = json.dumps(hands, separators=(",", ":")).encode("ascii")
    return V4Determinization(
        rank_counts_by_relative_offset=tuple(hands),
        hypothesis_seed=int(seed),
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


@dataclass(frozen=True)
class V4SearchLeaf:
    terminal_value: float | None
    public_observation: Mapping[str, object] | None
    legal_mask: Sequence[bool] | None
    depth: int

    @classmethod
    def terminal(cls, value: float, depth: int) -> "V4SearchLeaf":
        return cls(float(value), None, None, depth)

    @classmethod
    def evaluate(
        cls,
        public_observation: Mapping[str, object],
        legal_mask: Sequence[bool],
        depth: int,
    ) -> "V4SearchLeaf":
        return cls(None, public_observation, legal_mask, depth)


@dataclass(frozen=True)
class V4LeafRequest:
    public_observation: Mapping[str, object]
    legal_mask: tuple[bool, ...]
    root_action: int
    hypothesis_index: int
    rollout_seed: int
    depth: int


V4RolloutPolicy = Callable[
    [Mapping[str, object], tuple[bool, ...]], Sequence[float]
]
V4BatchedLeafEvaluator = Callable[[Sequence[V4LeafRequest]], Sequence[float]]

StateT = TypeVar("StateT")


@runtime_checkable
class V4SearchAdapter(Protocol[StateT]):
    """Loose simulator boundary; implementations may wrap TS or Python envs."""

    def build_root(
        self,
        public_observation: Mapping[str, object],
        determinization: V4Determinization,
        seed: int,
    ) -> StateT:
        ...

    def legal_action_mask(self, root: StateT) -> Sequence[bool]:
        ...

    def simulate_root_action(
        self,
        root: StateT,
        action_index: int,
        rollout_policy: V4RolloutPolicy | None,
        max_rollout_steps: int,
        seed: int,
    ) -> V4SearchLeaf:
        ...


@dataclass(frozen=True)
class V4SearchConfig:
    seed: int = 20260801
    hypotheses: int = 32
    rollouts_per_action: int = 2
    max_evaluations: int = 4096
    max_seconds: float | None = 30.0
    max_rollout_steps: int = 256
    leaf_batch_size: int = 128
    selection: str = "lcb"
    lcb_z: float = 1.0
    distribution_temperature: float = 0.2

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("search seed must be an integer")
        for name in (
            "hypotheses",
            "rollouts_per_action",
            "max_evaluations",
            "max_rollout_steps",
            "leaf_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_seconds is not None:
            seconds = float(self.max_seconds)
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise ValueError("max_seconds must be positive and finite or None")
        if self.selection not in {"mean", "lcb"}:
            raise ValueError("selection must be 'mean' or 'lcb'")
        if not math.isfinite(float(self.lcb_z)) or self.lcb_z < 0.0:
            raise ValueError("lcb_z must be finite and non-negative")
        temperature = float(self.distribution_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("distribution_temperature must be positive and finite")


@dataclass(frozen=True)
class V4ActionSearchStats:
    action_index: int
    samples: int
    mean: float | None
    standard_error: float | None
    lcb: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class V4SearchDiagnostics:
    seed: int
    player_count: int
    legal_action_count: int
    hypotheses_requested: int
    hypotheses_generated: int
    unique_determinizations: int
    evaluations: int
    terminal_evaluations: int
    batched_leaf_evaluations: int
    max_evaluations: int
    max_seconds: float | None
    elapsed_seconds: float
    selection: str
    stopped_reason: str
    incomplete_legal_actions: tuple[int, ...]


@dataclass(frozen=True)
class V4SearchResult:
    teacher_action: int
    teacher_distribution: tuple[float, ...]
    action_scores: tuple[float | None, ...]
    action_stats: tuple[V4ActionSearchStats, ...]
    diagnostics: V4SearchDiagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "teacherAction": self.teacher_action,
            "teacherDistribution": list(self.teacher_distribution),
            "actionScores": list(self.action_scores),
            "actionStats": [asdict(stats) for stats in self.action_stats],
            "searchDiagnostics": asdict(self.diagnostics),
        }


def _private_safe_policy(
    callback: V4RolloutPolicy | None,
) -> V4RolloutPolicy | None:
    if callback is None:
        return None

    def wrapped(
        observation: Mapping[str, object], legal_mask: tuple[bool, ...]
    ) -> Sequence[float]:
        public = _canonical_public_copy(observation)
        legal = _validate_legal_mask(legal_mask, "rollout policy legal mask")
        result = callback(public, legal)
        if len(result) != V4_SEARCH_ACTION_COUNT:
            raise ValueError("rollout policy must return exactly 236 scores")
        scores = tuple(
            _finite_number(score, f"rollout policy score[{index}]")
            for index, score in enumerate(result)
        )
        return scores

    return wrapped


def _validate_leaf(
    leaf: V4SearchLeaf,
    *,
    action_index: int,
    hypothesis_index: int,
    rollout_seed: int,
) -> tuple[float | None, V4LeafRequest | None]:
    if not isinstance(leaf, V4SearchLeaf):
        raise TypeError("search adapter must return V4SearchLeaf")
    if isinstance(leaf.depth, bool) or not isinstance(leaf.depth, int) or leaf.depth < 0:
        raise ValueError("search leaf depth must be a non-negative integer")
    if leaf.terminal_value is not None:
        if leaf.public_observation is not None or leaf.legal_mask is not None:
            raise ValueError("terminal search leaf must not expose another state")
        return _finite_number(leaf.terminal_value, "terminal search value"), None
    if leaf.public_observation is None or leaf.legal_mask is None:
        raise ValueError("non-terminal search leaf requires public observation and legal mask")
    public = _canonical_public_copy(leaf.public_observation)
    legal = _validate_legal_mask(leaf.legal_mask, "leaf legal mask")
    return None, V4LeafRequest(
        public_observation=public,
        legal_mask=legal,
        root_action=action_index,
        hypothesis_index=hypothesis_index,
        rollout_seed=rollout_seed,
        depth=leaf.depth,
    )


def _distribution_from_scores(
    scores: Mapping[int, float],
    legal_actions: Sequence[int],
    temperature: float,
) -> tuple[float, ...]:
    result = [0.0] * V4_SEARCH_ACTION_COUNT
    sampled = [action for action in legal_actions if action in scores]
    if not sampled:
        probability = 1.0 / len(legal_actions)
        for action in legal_actions:
            result[action] = probability
        return tuple(result)
    maximum = max(scores[action] for action in sampled)
    weights = {
        action: math.exp((scores[action] - maximum) / temperature)
        for action in sampled
    }
    denominator = sum(weights.values())
    for action, weight in weights.items():
        result[action] = weight / denominator
    return tuple(result)


def run_v4_search_teacher(
    public_observation: Mapping[str, object],
    legal_mask: Sequence[bool],
    adapter: V4SearchAdapter[object],
    *,
    config: V4SearchConfig | None = None,
    rollout_policy: V4RolloutPolicy | None = None,
    batched_leaf_evaluator: V4BatchedLeafEvaluator | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> V4SearchResult:
    """Search root actions across public-consistent hidden-card hypotheses."""

    cfg = config or V4SearchConfig()
    public = _canonical_public_copy(public_observation)
    root_legal = _validate_legal_mask(legal_mask, "root legal mask")
    legal_actions = tuple(index for index, legal in enumerate(root_legal) if legal)
    player_count = int(public["playerCount"])
    safe_policy = _private_safe_policy(rollout_policy)
    started_at = clock()
    values_by_action: dict[int, list[float]] = {
        action: [] for action in legal_actions
    }
    pending: list[tuple[int, V4LeafRequest]] = []
    hypotheses_generated = 0
    determinization_hashes: set[str] = set()
    evaluations = 0
    terminal_evaluations = 0
    batched_leaf_evaluations = 0
    stopped_reason = "completed"

    def flush_pending() -> None:
        nonlocal batched_leaf_evaluations
        if not pending:
            return
        if batched_leaf_evaluator is None:
            raise ValueError(
                "non-terminal search leaves require a batched leaf evaluator"
            )
        requests = tuple(request for _, request in pending)
        evaluated = batched_leaf_evaluator(requests)
        if len(evaluated) != len(requests):
            raise ValueError("batched leaf evaluator returned the wrong number of values")
        for (action, _), value in zip(pending, evaluated):
            values_by_action[action].append(
                _finite_number(value, "batched leaf evaluation")
            )
        batched_leaf_evaluations += len(pending)
        pending.clear()

    should_stop = False
    for hypothesis_index in range(cfg.hypotheses):
        if should_stop:
            break
        hypothesis_seed = _derived_seed(cfg.seed, 1, hypothesis_index)
        determinization = determinize_v4_unseen_hands(
            public, seed=hypothesis_seed
        )
        hypotheses_generated += 1
        determinization_hashes.add(determinization.sha256)
        # Rotate action order across hypotheses so a tight budget is unbiased.
        rotation = hypothesis_index % len(legal_actions)
        action_order = legal_actions[rotation:] + legal_actions[:rotation]
        for rollout_index in range(cfg.rollouts_per_action):
            for action_index in action_order:
                if evaluations >= cfg.max_evaluations:
                    stopped_reason = "evaluation-budget"
                    should_stop = True
                    break
                if (
                    cfg.max_seconds is not None
                    and clock() - started_at >= cfg.max_seconds
                ):
                    stopped_reason = "time-budget"
                    should_stop = True
                    break
                rollout_seed = _derived_seed(
                    cfg.seed,
                    2,
                    hypothesis_index,
                    rollout_index,
                    action_index,
                )
                root = adapter.build_root(public, determinization, rollout_seed)
                adapter_legal = _validate_legal_mask(
                    adapter.legal_action_mask(root), "adapter root legal mask"
                )
                if adapter_legal != root_legal:
                    raise ValueError(
                        "adapter root legal actions drifted across hidden hypotheses"
                    )
                leaf = adapter.simulate_root_action(
                    root,
                    action_index,
                    safe_policy,
                    cfg.max_rollout_steps,
                    rollout_seed,
                )
                terminal_value, leaf_request = _validate_leaf(
                    leaf,
                    action_index=action_index,
                    hypothesis_index=hypothesis_index,
                    rollout_seed=rollout_seed,
                )
                if terminal_value is not None:
                    values_by_action[action_index].append(terminal_value)
                    terminal_evaluations += 1
                elif leaf_request is not None:
                    pending.append((action_index, leaf_request))
                    if len(pending) >= cfg.leaf_batch_size:
                        flush_pending()
                evaluations += 1
            if should_stop:
                break
    flush_pending()

    action_stats: list[V4ActionSearchStats] = []
    selection_scores: dict[int, float] = {}
    full_scores: list[float | None] = [None] * V4_SEARCH_ACTION_COUNT
    for action_index in legal_actions:
        samples = values_by_action[action_index]
        if samples:
            mean = statistics.fmean(samples)
            standard_error = (
                statistics.stdev(samples) / math.sqrt(len(samples))
                if len(samples) > 1
                else 0.0
            )
            lcb = mean - cfg.lcb_z * standard_error
            selected_score = mean if cfg.selection == "mean" else lcb
            selection_scores[action_index] = selected_score
            full_scores[action_index] = selected_score
            minimum = min(samples)
            maximum = max(samples)
        else:
            mean = None
            standard_error = None
            lcb = None
            minimum = None
            maximum = None
        action_stats.append(V4ActionSearchStats(
            action_index=action_index,
            samples=len(samples),
            mean=mean,
            standard_error=standard_error,
            lcb=lcb,
            minimum=minimum,
            maximum=maximum,
        ))
    distribution = _distribution_from_scores(
        selection_scores, legal_actions, cfg.distribution_temperature
    )
    teacher_action = max(
        legal_actions, key=lambda action: (distribution[action], -action)
    )
    elapsed = max(0.0, clock() - started_at)
    incomplete = tuple(
        action for action in legal_actions if not values_by_action[action]
    )
    diagnostics = V4SearchDiagnostics(
        seed=cfg.seed,
        player_count=player_count,
        legal_action_count=len(legal_actions),
        hypotheses_requested=cfg.hypotheses,
        hypotheses_generated=hypotheses_generated,
        unique_determinizations=len(determinization_hashes),
        evaluations=evaluations,
        terminal_evaluations=terminal_evaluations,
        batched_leaf_evaluations=batched_leaf_evaluations,
        max_evaluations=cfg.max_evaluations,
        max_seconds=cfg.max_seconds,
        elapsed_seconds=elapsed,
        selection=cfg.selection,
        stopped_reason=stopped_reason,
        incomplete_legal_actions=incomplete,
    )
    return V4SearchResult(
        teacher_action=teacher_action,
        teacher_distribution=distribution,
        action_scores=tuple(full_scores),
        action_stats=tuple(action_stats),
        diagnostics=diagnostics,
    )


__all__ = [
    "V4ActionSearchStats",
    "V4BatchedLeafEvaluator",
    "V4Determinization",
    "V4LeafRequest",
    "V4RolloutPolicy",
    "V4SearchAdapter",
    "V4SearchConfig",
    "V4SearchDiagnostics",
    "V4SearchLeaf",
    "V4SearchResult",
    "determinize_v4_unseen_hands",
    "run_v4_search_teacher",
    "validate_v4_search_public_observation",
]
