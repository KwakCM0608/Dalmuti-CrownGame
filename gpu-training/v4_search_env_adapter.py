from __future__ import annotations

"""Privacy-safe bridge from V4 public search records to the scalar rules env.

The bridge deliberately reconstructs a *new* environment for every search
hypothesis.  Opponent cards come exclusively from :class:`V4Determinization`;
the actor record is never augmented with a critic vector or a real hidden
hand.

There is one important strictness rule.  A V4 record compresses history older
than 192 events into actor-relative EMA values.  Those two relative-offset
features cannot be losslessly rebased when another player becomes actor.  For
that reason a record with ``truncatedHistoryCount > 0`` supports only an exact
Normal rollout that reaches the end of the current act.  It may not be passed
to an injected rollout policy and may not produce a non-terminal public leaf.
This is an explicit error rather than a silently inaccurate approximation.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from v4_env import (
    ACTION_COUNT,
    HISTORY_FEATURES,
    JOKER_RANK,
    MAX_HISTORY,
    MEMORY_TRACE_DECAYS,
    ROLES,
    Card,
    DalmutiScalarEnv,
    Mulberry32,
    TablePlay,
    create_deck,
    role_for_index,
)
from v4_search import (
    V4Determinization,
    V4RolloutPolicy,
    V4SearchLeaf,
    validate_v4_search_public_observation,
)


_DECK_COUNTS = tuple(range(1, 13)) + (2,)
_EVENT_TYPES = ("play", "pass", "clear", "finish")
_PASS_REASONS = ("manual", "timeout", "insufficient-cards", "dalmuti")
_CLEAR_REASONS = ("all-passed", "dalmuti", "act-ended")


class V4SearchAdapterUnsupportedError(RuntimeError):
    """Raised when public compression makes an exact search leaf impossible."""


def _canonical_copy(value: Mapping[str, object]) -> dict[str, object]:
    validate_v4_search_public_observation(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("public observation must be canonical JSON data") from error
    copied = json.loads(encoded)
    validate_v4_search_public_observation(copied)
    return copied


def _rank_counts(cards: Sequence[Card]) -> list[int]:
    counts = [0] * 13
    for card in cards:
        counts[card.rank - 1] += 1
    return counts


def _determinization_digest(hands: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(hands, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_determinization(
    public: Mapping[str, object], determinization: V4Determinization
) -> tuple[tuple[int, ...], ...]:
    if type(determinization) is not V4Determinization:
        raise TypeError("determinization must be an exact V4Determinization")
    player_count = int(public["playerCount"])
    hands = determinization.rank_counts_by_relative_offset
    if len(hands) != player_count:
        raise ValueError("determinization player count does not match public state")
    normalized: list[tuple[int, ...]] = []
    for offset, hand in enumerate(hands):
        if len(hand) != 13:
            raise ValueError("each determinized hand must contain 13 rank counts")
        counts: list[int] = []
        for rank_index, count in enumerate(hand):
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError("determinized rank counts must be integers")
            if count < 0 or count > _DECK_COUNTS[rank_index]:
                raise ValueError("determinized rank count exceeds deck supply")
            counts.append(count)
        expected = int(public["playerTokens"][offset]["handCount"])
        if sum(counts) != expected:
            raise ValueError("determinized hand size disagrees with public hand count")
        normalized.append(tuple(counts))
    result = tuple(normalized)
    own = tuple(int(value) for value in public["ownHandCounts"])
    if result[0] != own:
        raise ValueError("determinization changed the actor's own hand")
    played = tuple(int(value) for value in public["publicPlayedCounts"])
    for rank_index, copies in enumerate(_DECK_COUNTS):
        if played[rank_index] + sum(hand[rank_index] for hand in result) != copies:
            raise ValueError("determinization and public cards do not conserve the deck")
    if isinstance(determinization.hypothesis_seed, bool) or not isinstance(
        determinization.hypothesis_seed, int
    ):
        raise ValueError("determinization seed must be an integer")
    if _determinization_digest(result) != determinization.sha256:
        raise ValueError("determinization SHA-256 does not match its rank hands")
    return result


def _social_order_from_relative_roles(
    public: Mapping[str, object],
) -> tuple[list[int], int]:
    player_count = int(public["playerCount"])
    tokens = public["playerTokens"]
    roles = [int(token["role"]) for token in tokens]
    if roles[0] != int(public["actorRole"]):
        raise ValueError("actorRole disagrees with relative player token zero")
    expected = [ROLES.index(role_for_index(index, player_count)) for index in range(player_count)]
    candidates = [
        actor_position
        for actor_position in range(player_count)
        if all(
            roles[offset] == expected[(actor_position + offset) % player_count]
            for offset in range(player_count)
        )
    ]
    if not candidates:
        raise ValueError("public player roles cannot be placed in social-rank order")
    actor_position = candidates[0]
    order = [-1] * player_count
    for offset in range(player_count):
        order[(actor_position + offset) % player_count] = offset
    if sorted(order) != list(range(player_count)):
        raise AssertionError("relative player offsets did not form an order")
    return order, actor_position


def _token_to_event(token: Mapping[str, object]) -> dict[str, object]:
    event_type = _EVENT_TYPES[int(token["type"])]
    event: dict[str, object] = {
        "sequence": int(token["sequence"]),
        "type": event_type,
        "actor_id": int(token["actorOffset"]),
        "hand_before": int(token["handCountBefore"]),
        "hand_after": int(token["handCountAfter"]),
    }
    if event_type == "play":
        event.update(
            rank=int(token["rank"]),
            natural_count=int(token["naturalCount"]),
            joker_count=int(token["jokerCount"]),
            total_count=int(token["totalCount"]),
        )
    elif event_type == "pass":
        reason = int(token["passReason"])
        if not 1 <= reason <= len(_PASS_REASONS):
            raise ValueError("pass history token is missing its reason")
        event["pass_reason"] = _PASS_REASONS[reason - 1]
    elif event_type == "clear":
        reason = int(token["clearReason"])
        if not 1 <= reason <= len(_CLEAR_REASONS):
            raise ValueError("clear history token is missing its reason")
        next_offset = int(token["nextLeaderOffset"])
        event.update(
            rank=int(token["rank"]),
            natural_count=int(token["naturalCount"]),
            joker_count=int(token["jokerCount"]),
            total_count=int(token["totalCount"]),
            clear_reason=_CLEAR_REASONS[reason - 1],
            next_leader_id=None if next_offset < 0 else next_offset,
        )
    else:
        place = int(token["finishPlace"])
        if place < 1:
            raise ValueError("finish history token is missing its place")
        event["finish_place"] = place
    return event


def _finish_order(
    public: Mapping[str, object], events: Sequence[Mapping[str, object]], order: Sequence[int]
) -> list[int]:
    finished = {
        int(token["relativeOffset"])
        for token in public["playerTokens"]
        if int(token["finished"]) == 1
    }
    slots: list[int | None] = [None] * len(finished)
    for event in events:
        if event["type"] != "finish":
            continue
        player_id = int(event["actor_id"])
        place = int(event["finish_place"])
        if player_id not in finished or not 1 <= place <= len(slots):
            raise ValueError("finish history disagrees with public finished players")
        if slots[place - 1] not in (None, player_id):
            raise ValueError("finish history repeats a finish place")
        if player_id in slots and slots[place - 1] != player_id:
            raise ValueError("finish history repeats a player")
        slots[place - 1] = player_id
    missing = [player_id for player_id in order if player_id in finished and player_id not in slots]
    for index, value in enumerate(slots):
        if value is None:
            slots[index] = missing.pop(0)
    if missing or any(value is None for value in slots):
        raise AssertionError("finished players could not be reconstructed")
    return [int(value) for value in slots]


def _validate_public_state_consistency(public: Mapping[str, object]) -> None:
    player_count = int(public["playerCount"])
    tokens = public["playerTokens"]
    if int(tokens[0]["self"]) != 1 or int(tokens[0]["relativeOffset"]) != 0:
        raise ValueError("relative player zero must be the actor")
    if int(tokens[0]["finished"]) or int(tokens[0]["passed"]):
        raise ValueError("the acting player must be active and not passed")
    if int(tokens[0]["handCount"]) < 1:
        raise ValueError("the acting player must have a card")
    table = public["table"]
    table_offset = -1 if table is None else int(table["actorOffset"])
    leaders = [
        int(token["relativeOffset"])
        for token in tokens
        if int(token["tableLeader"]) == 1
    ]
    if leaders != ([] if table is None else [table_offset]):
        raise ValueError("tableLeader flags disagree with the public table")
    if table is None and any(int(token["passed"]) for token in tokens):
        raise ValueError("an empty table cannot retain passed players")
    for token in tokens:
        finished = int(token["finished"]) == 1
        if finished != (int(token["handCount"]) == 0):
            raise ValueError("finished flag must match a zero public hand count")
        if finished and int(token["passed"]):
            raise ValueError("a finished player cannot remain passed")
    if table is not None:
        rank = int(table["rank"])
        natural = int(table["naturalCount"])
        jokers = int(table["jokerCount"])
        total = int(table["totalCount"])
        if rank == JOKER_RANK:
            if (natural, jokers, total) != (0, 1, 1):
                raise ValueError("rank-13 table must be exactly one solo joker")
        elif not (1 <= natural <= rank and natural + jokers == total):
            raise ValueError("public table is not a legal physical bundle")
        if int(tokens[table_offset]["passed"]):
            raise ValueError("the current table leader cannot be passed")
    finished_count = sum(int(token["finished"]) for token in tokens)
    if finished_count >= player_count - 1:
        raise ValueError("an act with all places decided cannot have an actor")
    if int(public["truncatedHistoryCount"]) == 0:
        for trace in public["memoryTraceVectors"]:
            if any(float(value) != 0.0 for value in trace):
                raise ValueError("untruncated public history must have zero EMA traces")


def _cards_from_determinization(
    public: Mapping[str, object], hands: Sequence[Sequence[int]]
) -> dict[int, list[Card]]:
    by_rank: dict[int, list[Card]] = {rank: [] for rank in range(1, 14)}
    for card in create_deck():
        by_rank[card.rank].append(card)
    played = [int(value) for value in public["publicPlayedCounts"]]
    cursors = played[:]
    result = {offset: [] for offset in range(len(hands))}
    for offset, counts in enumerate(hands):
        for rank_index, count in enumerate(counts):
            rank = rank_index + 1
            start = cursors[rank_index]
            end = start + int(count)
            cards = by_rank[rank][start:end]
            if len(cards) != count:
                raise AssertionError("determinization exhausted a physical rank pool")
            result[offset].extend(cards)
            cursors[rank_index] = end
        result[offset].sort(key=lambda card: (-card.rank, card.id))
    if any(cursors[index] != copies for index, copies in enumerate(_DECK_COUNTS)):
        raise AssertionError("determinization did not consume every unplayed card")
    return result


def _event_token(
    event: Mapping[str, object], relative: Sequence[int], player_count: int
) -> dict[str, int]:
    event_type = str(event["type"])
    actor_offset = relative.index(int(event["actor_id"]))
    token = {
        "sequence": int(event["sequence"]),
        "type": _EVENT_TYPES.index(event_type),
        "actorOffset": actor_offset,
        "handCountBefore": int(event["hand_before"]),
        "handCountAfter": int(event["hand_after"]),
        "rank": 0,
        "naturalCount": 0,
        "jokerCount": 0,
        "totalCount": 0,
        "passReason": 0,
        "clearReason": 0,
        "nextLeaderOffset": -1,
        "finishPlace": 0,
    }
    if event_type in ("play", "clear"):
        token.update(
            rank=int(event["rank"]),
            naturalCount=int(event["natural_count"]),
            jokerCount=int(event["joker_count"]),
            totalCount=int(event["total_count"]),
        )
    if event_type == "pass":
        token["passReason"] = _PASS_REASONS.index(str(event["pass_reason"])) + 1
    elif event_type == "clear":
        token["clearReason"] = _CLEAR_REASONS.index(str(event["clear_reason"])) + 1
        next_id = event.get("next_leader_id")
        token["nextLeaderOffset"] = -1 if next_id is None else relative.index(int(next_id))
    elif event_type == "finish":
        token["finishPlace"] = int(event["finish_place"])
    return token


def _trace_features(token: Mapping[str, int], player_count: int) -> list[float]:
    row = [0.0] * HISTORY_FEATURES
    row[int(token["type"])] = 1.0
    row[4] = int(token["actorOffset"]) / max(1, player_count - 1)
    row[5] = int(token["handCountBefore"]) / 20.0
    row[6] = int(token["handCountAfter"]) / 20.0
    row[7] = int(token["rank"]) / 13.0
    row[8] = int(token["naturalCount"]) / 12.0
    row[9] = int(token["jokerCount"]) / 2.0
    row[10] = int(token["totalCount"]) / 14.0
    if int(token["passReason"]) > 0:
        row[10 + int(token["passReason"])] = 1.0
    if int(token["clearReason"]) > 0:
        row[14 + int(token["clearReason"])] = 1.0
    if int(token["nextLeaderOffset"]) >= 0:
        row[18] = (int(token["nextLeaderOffset"]) + 1) / player_count
    row[19] = int(token["finishPlace"]) / player_count
    return row


@dataclass
class DalmutiV4SearchRoot:
    env: DalmutiScalarEnv
    source_public: dict[str, object]
    root_actor_id: int
    truncated_history_count: int
    build_seed: int


class DalmutiV4SearchEnvAdapter:
    """Concrete :class:`V4SearchAdapter` backed by ``DalmutiScalarEnv``."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def build_root(
        self,
        public_observation: Mapping[str, object],
        determinization: V4Determinization,
        seed: int,
    ) -> DalmutiV4SearchRoot:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("root seed must be an integer")
        public = _canonical_copy(public_observation)
        _validate_public_state_consistency(public)
        hands = _validate_determinization(public, determinization)
        order, actor_position = _social_order_from_relative_roles(public)
        player_count = int(public["playerCount"])
        act = int(public["act"])
        env = DalmutiScalarEnv(
            player_count,
            acts=act,
            seed=seed,
            device=self.device,
        )
        events = [_token_to_event(token) for token in public["historyTokens"]]
        env._seed = int(seed)
        env._rng = Mulberry32(int(seed))
        env._scores = {
            int(token["relativeOffset"]): float(token["score"])
            for token in public["playerTokens"]
        }
        env._order = list(order)
        env._act = act
        env._terminated = False
        env._last_act_result = None
        env._hands = _cards_from_determinization(public, hands)
        env._revolution = int(public["revolution"])
        env._tax_audit = ()
        env._finish_order = _finish_order(public, events, order)
        env._passed = {
            int(token["relativeOffset"])
            for token in public["playerTokens"]
            if int(token["passed"]) == 1
        }
        env._public_played = [int(value) for value in public["publicPlayedCounts"]]
        env._history = events
        env._event_sequence = (
            int(events[-1]["sequence"]) + 1
            if events
            else int(public["truncatedHistoryCount"])
        )
        table = public["table"]
        env._table = (
            None
            if table is None
            else TablePlay(
                rank=int(table["rank"]),
                count=int(table["totalCount"]),
                player_id=int(table["actorOffset"]),
                joker_count=int(table["jokerCount"]),
            )
        )
        env._last_played_id = None if table is None else int(table["actorOffset"])
        env._current_index = actor_position
        env._transitions = 0
        if env.current_player_id != 0:
            raise AssertionError("root actor was not reconstructed as relative player zero")
        if env.physical_card_count != 80:
            raise AssertionError("reconstructed root did not conserve 80 cards")
        root = DalmutiV4SearchRoot(
            env=env,
            source_public=public,
            root_actor_id=0,
            truncated_history_count=int(public["truncatedHistoryCount"]),
            build_seed=int(seed),
        )
        if root.truncated_history_count == 0:
            reconstructed = self.public_observation(root)
            if reconstructed != public:
                raise ValueError("reconstructed root drifted from canonical public state")
        return root

    def legal_action_mask(self, root: DalmutiV4SearchRoot) -> tuple[bool, ...]:
        if not isinstance(root, DalmutiV4SearchRoot):
            raise TypeError("root must be a DalmutiV4SearchRoot")
        return tuple(bool(value) for value in root.env.legal_mask().cpu().tolist())

    def public_observation(self, root: DalmutiV4SearchRoot) -> dict[str, object]:
        """Return an exact future actor record when full history is available."""

        if root.truncated_history_count > 0:
            raise V4SearchAdapterUnsupportedError(
                "compressed actor-relative history cannot be exactly rebased"
            )
        env = root.env
        if env.terminated:
            raise RuntimeError("a terminal act has no public search leaf")
        actor_id = env.current_player_id
        relative = env._relative_order(actor_id)
        player_count = env.player_count
        actor_position = env._order.index(actor_id)
        own_counts = _rank_counts(env._hands[actor_id])
        table = env._table
        tokens = [_event_token(event, relative, player_count) for event in env._history]
        truncated = max(0, len(tokens) - MAX_HISTORY)
        memory: list[list[float]] = []
        for decay in MEMORY_TRACE_DECAYS:
            trace = [0.0] * HISTORY_FEATURES
            for token in tokens[:truncated]:
                features = _trace_features(token, player_count)
                trace = [
                    decay * previous + (1.0 - decay) * value
                    for previous, value in zip(trace, features)
                ]
            memory.append(trace)
        result: dict[str, object] = {
            "schemaVersion": 4,
            "playerCount": player_count,
            "act": env._act,
            "actorRole": ROLES.index(role_for_index(actor_position, player_count)),
            "revolution": env._revolution,
            "ownHandCounts": own_counts,
            "publicPlayedCounts": list(env._public_played),
            "table": (
                None
                if table is None
                else {
                    "actorOffset": relative.index(table.player_id),
                    "rank": table.rank,
                    "naturalCount": table.natural_count,
                    "jokerCount": table.joker_count,
                    "totalCount": table.count,
                }
            ),
            "playerTokens": [
                {
                    "relativeOffset": offset,
                    "handCount": len(env._hands[player_id]),
                    "finished": int(not env._hands[player_id]),
                    "passed": int(player_id in env._passed),
                    "self": int(offset == 0),
                    "tableLeader": int(table is not None and table.player_id == player_id),
                    "role": ROLES.index(
                        role_for_index(env._order.index(player_id), player_count)
                    ),
                    "score": env._scores[player_id],
                }
                for offset, player_id in enumerate(relative)
            ],
            "historyTokens": tokens[truncated:],
            "memoryTraceVectors": memory,
            "truncatedHistoryCount": truncated,
        }
        validate_v4_search_public_observation(result)
        return result

    def _policy_action(
        self,
        root: DalmutiV4SearchRoot,
        rollout_policy: V4RolloutPolicy,
    ) -> int:
        public = self.public_observation(root)
        legal = self.legal_action_mask(root)
        scores = rollout_policy(_canonical_copy(public), legal)
        if len(scores) != ACTION_COUNT:
            raise ValueError("rollout policy must return exactly 236 scores")
        normalized: list[float] = []
        for index, score in enumerate(scores):
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"rollout policy score[{index}] must be numeric")
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"rollout policy score[{index}] must be finite")
            normalized.append(value)
        return max(
            (index for index, allowed in enumerate(legal) if allowed),
            key=lambda index: (normalized[index], -index),
        )

    def simulate_root_action(
        self,
        root: DalmutiV4SearchRoot,
        action_index: int,
        rollout_policy: V4RolloutPolicy | None,
        max_rollout_steps: int,
        seed: int,
    ) -> V4SearchLeaf:
        if not isinstance(root, DalmutiV4SearchRoot):
            raise TypeError("root must be a DalmutiV4SearchRoot")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("rollout seed must be an integer")
        if (
            isinstance(max_rollout_steps, bool)
            or not isinstance(max_rollout_steps, int)
            or max_rollout_steps < 1
        ):
            raise ValueError("max_rollout_steps must be a positive integer")
        if root.truncated_history_count > 0 and rollout_policy is not None:
            raise V4SearchAdapterUnsupportedError(
                "injected public rollout policies require lossless untruncated history"
            )
        legal = self.legal_action_mask(root)
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 0 <= action_index < ACTION_COUNT
            or not legal[action_index]
        ):
            raise ValueError(f"illegal root action {action_index}")

        depth = 0
        action = action_index
        while depth < max_rollout_steps:
            result = root.env.step(action)
            depth += 1
            if result.act_ended:
                value = float(result.rewards[root.root_actor_id].item())
                return V4SearchLeaf.terminal(value, depth)
            if root.env.terminated:
                raise AssertionError("environment terminated without ending the current act")
            if depth >= max_rollout_steps:
                break
            action = (
                root.env.normal_action()
                if rollout_policy is None
                else self._policy_action(root, rollout_policy)
            )

        if root.truncated_history_count > 0:
            raise V4SearchAdapterUnsupportedError(
                "compressed-history Normal rollout did not reach the act terminal "
                "within max_rollout_steps"
            )
        return V4SearchLeaf.evaluate(
            self.public_observation(root),
            self.legal_action_mask(root),
            depth,
        )


__all__ = [
    "DalmutiV4SearchEnvAdapter",
    "DalmutiV4SearchRoot",
    "V4SearchAdapterUnsupportedError",
]
