from __future__ import annotations

"""Training-only, deterministic DALMUTI reference and batched environment.

The scalar core is a direct Python port of the card-play semantics in
``training/simulator.ts`` and the exact Normal policy scoring/tie breaking in
``lib/bot-strategy.ts``.  The batched facade returns fixed-shape torch tensors
and uses the same scalar transition core for every lane.  This is deliberate:
it provides a correctness oracle for a future fused/GPU transition kernel.

Actor tensors contain only the acting player's hand and public information.
The full set of hands is encoded exclusively by ``privileged_state()`` for a
training-only critic.  Never concatenate that vector into actor inputs.
"""

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from typing import Iterable, Mapping, Sequence

import torch


MIN_PLAYERS = 4
MAX_PLAYERS = 10
DECK_SIZE = 80
NORMAL_RANKS = 12
JOKER_RANK = 13
ACTION_COUNT = 236
PASS_ACTION_INDEX = 0
SOLO_JOKER_ACTION_INDEX = 1
MAX_HISTORY = 192
HISTORY_FEATURES = 20
MEMORY_TRACE_DECAYS = (0.5, 0.8, 0.95, 0.99)
PRIVILEGED_STATE_VERSION = 1
PRIVILEGED_STATE_SIZE = 512
PRIVILEGED_GLOBAL_OFFSET = 0
PRIVILEGED_GLOBAL_FEATURES = 16
PRIVILEGED_PUBLIC_RANK_OFFSET = 16
PRIVILEGED_PLAYER_OFFSET = 29
PRIVILEGED_PLAYER_STRIDE = 25
PRIVILEGED_RESERVED_OFFSET = 279
PRIVILEGED_STATE_LAYOUT_ID = "dalmuti-v4-ts-privileged-critic-raw-v1"
PRIVILEGED_STATE_LAYOUT = {
    "id": PRIVILEGED_STATE_LAYOUT_ID,
    "version": PRIVILEGED_STATE_VERSION,
    "featureCount": PRIVILEGED_STATE_SIZE,
    "global": {
        "offset": PRIVILEGED_GLOBAL_OFFSET,
        "fields": [
            "playerCount", "act", "revolution", "table.present",
            "table.rank", "table.naturalCount", "table.jokerCount",
            "table.totalCount", "table.actorOffsetOrMinusOne",
            "publicPlayedTotal", "activePlayerCount", "finishedPlayerCount",
            "actorRole", "actorScore", "actorHandCount",
            "publicHistoryEventCount",
        ],
    },
    "publicPlayedRankCounts": {
        "offset": PRIVILEGED_PUBLIC_RANK_OFFSET,
        "length": 13,
        "ranks": "1..13",
    },
    "players": {
        "offset": PRIVILEGED_PLAYER_OFFSET,
        "seats": MAX_PLAYERS,
        "stride": PRIVILEGED_PLAYER_STRIDE,
        "fields": [
            "present", "relativeOffset", "role.oneHot[5]", "score",
            "handCount", "passed", "finished", "finishPlace",
            "handRankCounts[13]",
        ],
    },
    "reservedZeroTail": {
        "offset": PRIVILEGED_RESERVED_OFFSET,
        "length": PRIVILEGED_STATE_SIZE - PRIVILEGED_RESERVED_OFFSET,
    },
}
PRIVILEGED_STATE_LAYOUT_SHA256 = hashlib.sha256(
    json.dumps(
        PRIVILEGED_STATE_LAYOUT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
MAX_TRANSITIONS_PER_ACT = 20_000

ROLES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)

# Compatibility aliases for consumers that imported the original names.
PRIVILEGED_GLOBAL_FIELDS = tuple(
    PRIVILEGED_STATE_LAYOUT["global"]["fields"]
)
PRIVILEGED_LAYOUT: Mapping[str, tuple[int, int]] = {
    "global": (PRIVILEGED_GLOBAL_OFFSET, PRIVILEGED_PUBLIC_RANK_OFFSET),
    "public_played_rank_counts": (
        PRIVILEGED_PUBLIC_RANK_OFFSET,
        PRIVILEGED_PLAYER_OFFSET,
    ),
    "players": (PRIVILEGED_PLAYER_OFFSET, PRIVILEGED_RESERVED_OFFSET),
    "reserved_zero_tail": (PRIVILEGED_RESERVED_OFFSET, PRIVILEGED_STATE_SIZE),
}


@dataclass(frozen=True)
class Card:
    id: str
    rank: int


@dataclass(frozen=True)
class SemanticAction:
    type: str
    rank: int = 0
    count: int = 0
    joker_count: int = 0

    @property
    def natural_count(self) -> int:
        return self.count - self.joker_count if self.type == "play" else 0


@dataclass(frozen=True)
class TablePlay:
    rank: int
    count: int
    player_id: int
    joker_count: int = 0

    @property
    def natural_count(self) -> int:
        return 0 if self.rank == JOKER_RANK else self.count - self.joker_count


@dataclass(frozen=True)
class NormalPublicPlayer:
    player_id: int
    hand_count: int
    finished: bool = False


@dataclass(frozen=True)
class NormalObservation:
    actor_id: int
    hand: tuple[Card, ...]
    table: TablePlay | None
    players: tuple[NormalPublicPlayer, ...]
    passed_player_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class NormalDecision:
    action_index: int
    score: float


@dataclass(frozen=True)
class V4ActorObservation:
    """Public-only tensors for one acting player."""

    actor_id: int
    valid: torch.Tensor
    global_features: torch.Tensor
    rank_features: torch.Tensor
    player_features: torch.Tensor
    player_mask: torch.Tensor
    memory_trace_features: torch.Tensor
    history_features: torch.Tensor
    history_mask: torch.Tensor
    legal_mask: torch.Tensor


@dataclass(frozen=True)
class V4EnvironmentObservation:
    public: V4ActorObservation
    privileged_state: torch.Tensor


@dataclass(frozen=True)
class ScalarStepResult:
    observation: V4EnvironmentObservation
    rewards: torch.Tensor
    terminated: bool
    act_ended: bool
    info: Mapping[str, object]


@dataclass(frozen=True)
class V4BatchedActorObservation:
    actor_ids: torch.Tensor
    valid: torch.Tensor
    global_features: torch.Tensor
    rank_features: torch.Tensor
    player_features: torch.Tensor
    player_mask: torch.Tensor
    memory_trace_features: torch.Tensor
    history_features: torch.Tensor
    history_mask: torch.Tensor
    legal_masks: torch.Tensor


@dataclass(frozen=True)
class V4BatchedEnvironmentObservation:
    public: V4BatchedActorObservation
    privileged_states: torch.Tensor


@dataclass(frozen=True)
class BatchedStepResult:
    observation: V4BatchedEnvironmentObservation
    rewards: torch.Tensor
    terminated: torch.Tensor
    act_ended: torch.Tensor
    infos: tuple[Mapping[str, object], ...]


def role_for_index(index: int, player_count: int) -> str:
    if index == 0:
        return "great-dalmuti"
    if index == 1:
        return "lesser-dalmuti"
    if index == player_count - 2:
        return "lesser-peon"
    if index == player_count - 1:
        return "great-peon"
    return "merchant"


def ranked_deal_counts(total_cards: int, player_count: int) -> list[int]:
    if total_cards < 0 or player_count <= 0:
        raise ValueError("invalid deal dimensions")
    base, remainder = divmod(total_cards, player_count)
    bonus_start = player_count - remainder
    return [base + int(index >= bonus_start) for index in range(player_count)]


def round_chip_award(place: int, player_count: int) -> int:
    if not MIN_PLAYERS <= player_count <= MAX_PLAYERS:
        raise ValueError("player_count must be from 4 through 10")
    if not 1 <= place <= player_count:
        raise ValueError("place is outside the table")
    if place == 1:
        return 4
    if place == 2:
        return 3
    if place == player_count - 1:
        return 1
    if place == player_count:
        return 0
    return 2


def create_deck() -> list[Card]:
    deck = [
        Card(f"{rank}-{copy}", rank)
        for rank in range(1, NORMAL_RANKS + 1)
        for copy in range(rank)
    ]
    deck.extend((Card("joker-1", JOKER_RANK), Card("joker-2", JOKER_RANK)))
    if len(deck) != DECK_SIZE:
        raise AssertionError("physical DALMUTI deck must contain 80 cards")
    return deck


def _first_action_for_rank(rank: int) -> int:
    return 2 + 3 * (rank - 1) * rank // 2


def encode_action(rank: int, natural_count: int, joker_count: int = 0) -> int:
    if not 1 <= rank <= NORMAL_RANKS:
        raise ValueError("action rank must be from 1 through 12")
    if not 1 <= natural_count <= rank:
        raise ValueError("natural_count exceeds the rank's physical copies")
    if not 0 <= joker_count <= 2:
        raise ValueError("joker_count must be from 0 through 2")
    return _first_action_for_rank(rank) + (natural_count - 1) * 3 + joker_count


def decode_action(action_index: int) -> SemanticAction:
    if isinstance(action_index, bool) or not 0 <= int(action_index) < ACTION_COUNT:
        raise ValueError("action_index must be from 0 through 235")
    action_index = int(action_index)
    if action_index == PASS_ACTION_INDEX:
        return SemanticAction("pass")
    if action_index == SOLO_JOKER_ACTION_INDEX:
        return SemanticAction("solo-joker", JOKER_RANK, 1, 1)
    for rank in range(1, NORMAL_RANKS + 1):
        start = _first_action_for_rank(rank)
        end = start + rank * 3
        if start <= action_index < end:
            offset = action_index - start
            natural_count = offset // 3 + 1
            joker_count = offset % 3
            return SemanticAction(
                "play", rank, natural_count + joker_count, joker_count
            )
    raise AssertionError("the fixed action catalogue is incomplete")


ACTION_CATALOGUE = tuple(decode_action(index) for index in range(ACTION_COUNT))


def legal_action_masks(
    hand_counts: torch.Tensor,
    table_ranks: torch.Tensor | None = None,
    table_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized 236-action masks for ``[..., 13]`` rank counts.

    A table rank of zero means an empty table.  Both table tensors must have
    the same leading shape as ``hand_counts``.  The implementation uses only
    torch comparisons, so this portion can run directly on CUDA.
    """

    if hand_counts.shape[-1] != 13:
        raise ValueError("hand_counts must end in 13 physical ranks")
    if hand_counts.dtype == torch.bool or hand_counts.is_floating_point():
        raise ValueError("hand_counts must use an integer dtype")
    leading = hand_counts.shape[:-1]
    device = hand_counts.device
    if table_ranks is None:
        table_ranks = torch.zeros(leading, dtype=torch.long, device=device)
    if table_counts is None:
        table_counts = torch.zeros(leading, dtype=torch.long, device=device)
    if table_ranks.shape != leading or table_counts.shape != leading:
        raise ValueError("table tensors must match the hand leading shape")
    table_ranks = table_ranks.to(device=device, dtype=torch.long)
    table_counts = table_counts.to(device=device, dtype=torch.long)
    if ((table_ranks < 0) | (table_ranks > 13)).any():
        raise ValueError("table ranks must use zero for empty or ranks 1..13")
    if ((table_counts < 0) | (table_counts > 14)).any():
        raise ValueError("table counts must be from zero through 14")
    if ((table_ranks == 0) != (table_counts == 0)).any():
        raise ValueError("empty-table rank and count sentinels must agree")

    lead = table_ranks == 0
    mask = torch.zeros((*leading, ACTION_COUNT), dtype=torch.bool, device=device)
    mask[..., PASS_ACTION_INDEX] = ~lead
    mask[..., SOLO_JOKER_ACTION_INDEX] = lead & (hand_counts[..., 12] >= 1)
    available_jokers = hand_counts[..., 12]
    for index, action in enumerate(ACTION_CATALOGUE[2:], start=2):
        natural_count = action.natural_count
        available = hand_counts[..., action.rank - 1] >= natural_count
        has_jokers = available_jokers >= action.joker_count
        response = (
            (action.rank < table_ranks)
            & (action.count == table_counts)
            & ~lead
        )
        mask[..., index] = available & has_jokers & (lead | response)
    return mask


def legal_action_indices(
    hand: Sequence[Card], table: TablePlay | None
) -> list[int]:
    counts = _rank_counts(hand)
    joker_count = counts[JOKER_RANK]
    result: list[int] = []
    if table is None:
        if joker_count:
            result.append(SOLO_JOKER_ACTION_INDEX)
    else:
        result.append(PASS_ACTION_INDEX)
    for rank in range(1, NORMAL_RANKS + 1):
        if table is not None and rank >= table.rank:
            continue
        for jokers in range(joker_count + 1):
            if table is None:
                natural_counts = range(1, counts[rank] + 1)
            else:
                natural_counts = (table.count - jokers,)
            for naturals in natural_counts:
                if 1 <= naturals <= counts[rank]:
                    result.append(encode_action(rank, naturals, jokers))
    return sorted(set(result))


class Mulberry32:
    """Bit-for-bit port of ``training/random.ts``."""

    _MASK = 0xFFFF_FFFF

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.state = seed & self._MASK

    def next_uint32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & self._MASK
        value = self.state
        value = ((value ^ (value >> 15)) * (value | 1)) & self._MASK
        mixed = ((value ^ (value >> 7)) * (value | 61)) & self._MASK
        value = (value ^ ((value + mixed) & self._MASK)) & self._MASK
        return (value ^ (value >> 14)) & self._MASK

    def next(self) -> float:
        return self.next_uint32() / 4294967296.0

    def integer(self, maximum_exclusive: int) -> int:
        if maximum_exclusive <= 0:
            raise ValueError("maximum_exclusive must be positive")
        return int(self.next() * maximum_exclusive)

    def shuffle(self, values: Iterable[object]) -> list[object]:
        result = list(values)
        for index in range(len(result) - 1, 0, -1):
            target = self.integer(index + 1)
            result[index], result[target] = result[target], result[index]
        return result


def _sorted_cards(cards: Iterable[Card]) -> list[Card]:
    return sorted(cards, key=lambda card: (card.rank, card.id))


def _sorted_hand(cards: Iterable[Card]) -> list[Card]:
    return sorted(cards, key=lambda card: (-card.rank, card.id))


def _rank_counts(cards: Iterable[Card]) -> list[int]:
    counts = [0] * 14
    for card in cards:
        counts[card.rank] += 1
    return counts


def _resolve_cards(hand: Sequence[Card], action: SemanticAction) -> tuple[Card, ...]:
    jokers = _sorted_cards(card for card in hand if card.rank == JOKER_RANK)
    if action.type == "solo-joker":
        if not jokers:
            raise ValueError("solo joker action has no joker")
        return (jokers[0],)
    if action.type != "play":
        raise ValueError("PASS has no physical cards")
    naturals = _sorted_cards(card for card in hand if card.rank == action.rank)
    chosen = naturals[: action.natural_count] + jokers[: action.joker_count]
    if len(chosen) != action.count:
        raise ValueError("semantic action cannot be resolved from this hand")
    return tuple(chosen)


def _estimated_turns(hand: Sequence[Card]) -> int:
    normal_ranks = {card.rank for card in hand if card.rank != JOKER_RANK}
    joker_count = sum(card.rank == JOKER_RANK for card in hand)
    return len(normal_ranks) if normal_ranks else joker_count


def _structure_damage(hand: Sequence[Card], action: SemanticAction) -> float:
    if action.rank == JOKER_RANK:
        return 0.0
    before = sum(card.rank == action.rank for card in hand)
    after = before - action.natural_count
    if after <= 0 or before <= 1:
        return 0.0
    return float(28 + (18 if after == 1 else 0) + min(12, before * 2))


def _next_active_opponent(
    observation: NormalObservation,
) -> NormalPublicPlayer | None:
    actor_index = next(
        index
        for index, player in enumerate(observation.players)
        if player.player_id == observation.actor_id
    )
    for step in range(1, len(observation.players)):
        player = observation.players[(actor_index + step) % len(observation.players)]
        if (
            player.player_id != observation.actor_id
            and not player.finished
            and player.hand_count > 0
            and player.player_id not in observation.passed_player_ids
        ):
            return player
    return None


def _normal_play_score(
    observation: NormalObservation,
    action: SemanticAction,
    cards: Sequence[Card],
) -> float:
    selected_ids = {card.id for card in cards}
    after = tuple(card for card in observation.hand if card.id not in selected_ids)
    before_counts = _rank_counts(observation.hand)
    after_counts = _rank_counts(after)
    score = action.count * 24.0
    if not after:
        score += 100_000.0
    if action.rank == JOKER_RANK:
        score -= 82.0
    else:
        score -= (JOKER_RANK - action.rank) * action.natural_count * 2.0
    if action.joker_count:
        score -= action.joker_count * 78.0
    score -= _structure_damage(observation.hand, action)
    if (
        action.rank != JOKER_RANK
        and before_counts[action.rank] > 0
        and after_counts[action.rank] == 0
    ):
        score += 18.0
    score += (_estimated_turns(observation.hand) - _estimated_turns(after)) * 26.0

    if observation.table is not None:
        table_leader = next(
            (
                player
                for player in observation.players
                if player.player_id == observation.table.player_id
            ),
            None,
        )
        if table_leader and not table_leader.finished and table_leader.hand_count <= 2:
            score += (3 - table_leader.hand_count) * 58.0

    next_opponent = _next_active_opponent(observation)
    if next_opponent and next_opponent.hand_count <= action.count and after:
        rank_risk = (
            1.0
            if action.rank == JOKER_RANK
            else max(0.15, action.rank / 12.0)
        )
        equality_factor = 1.0 if next_opponent.hand_count == action.count else 0.7
        score -= 72.0 * rank_risk * equality_factor
    return score


def _normal_pass_score(
    observation: NormalObservation,
    play_actions: Sequence[SemanticAction],
) -> float:
    if not play_actions:
        return math.inf
    score = 14.0
    if all(action.joker_count > 0 for action in play_actions):
        score += 34.0
    table_leader = next(
        (
            player
            for player in observation.players
            if observation.table is not None
            and player.player_id == observation.table.player_id
        ),
        None,
    )
    if table_leader and not table_leader.finished and table_leader.hand_count <= 2:
        score -= (3 - table_leader.hand_count) * 68.0
    return score


def _action_tie_key(action: SemanticAction, cards: Sequence[Card]) -> str:
    if action.type == "pass":
        return "\uffff"
    card_ids = sorted(card.id for card in cards)
    return f"{action.rank:02d}:{action.count:02d}:{','.join(card_ids)}"


def choose_normal_action(observation: NormalObservation) -> NormalDecision:
    """Exact Normal card-play score and deterministic production tie break."""

    if not observation.hand:
        raise ValueError("a finished player cannot act")
    legal = legal_action_indices(observation.hand, observation.table)
    candidates: list[tuple[float, str, int]] = []
    plays: list[SemanticAction] = []
    for action_index in legal:
        action = decode_action(action_index)
        if action.type == "pass":
            continue
        cards = _resolve_cards(observation.hand, action)
        plays.append(action)
        score = _normal_play_score(observation, action, cards)
        candidates.append((score, _action_tie_key(action, cards), action_index))
    if observation.table is not None:
        candidates.append(
            (
                _normal_pass_score(observation, plays),
                _action_tie_key(SemanticAction("pass"), ()),
                PASS_ACTION_INDEX,
            )
        )
    if not candidates:
        raise RuntimeError("active player has no legal Normal decision")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    score, _, action_index = candidates[0]
    return NormalDecision(action_index, score)


def normal_tax_return_card_ids(hand: Sequence[Card], count: int) -> tuple[str, ...]:
    """Exact Normal noble-return heuristic, including physical-id tie break."""

    if not 0 <= count <= len(hand):
        raise ValueError("invalid tax return count")
    if count == 0:
        return ()
    hand_counts = _rank_counts(hand)

    def score(cards: Sequence[Card]) -> float:
        selected_counts = _rank_counts(cards)
        value = 0.0
        for card in cards:
            if card.rank == JOKER_RANK:
                value -= 400.0
            else:
                value += card.rank * 10.0
                if hand_counts[card.rank] == 1:
                    value += 40.0
        for rank in range(1, JOKER_RANK):
            selected = selected_counts[rank]
            if selected == 0:
                continue
            original = hand_counts[rank]
            remaining = original - selected
            if original > 1:
                value -= 45.0 * selected
                if remaining == 1:
                    value -= 30.0
                if remaining == 0:
                    value -= 22.0
            if selected > 1:
                value -= 20.0 * (selected - 1)
        return value

    choices = []
    for cards in combinations(_sorted_cards(hand), count):
        key = "\0".join(card.id for card in cards)
        choices.append((score(cards), key, cards))
    choices.sort(key=lambda item: (-item[0], item[1]))
    return tuple(card.id for card in choices[0][2])


def normal_revolution_decision(hand: Sequence[Card], role: str) -> int:
    """Return 0=no declaration, 1=revolution, 2=great revolution."""

    if sum(card.rank == JOKER_RANK for card in hand) != 2:
        raise ValueError("a revolution decision requires exactly two jokers")
    if role == "great-peon":
        return 2
    normal_cards = [card.rank for card in hand if card.rank != JOKER_RANK]
    burden = sum(normal_cards) / len(normal_cards) if normal_cards else 0.0
    base = {
        "great-dalmuti": -95.0,
        "lesser-dalmuti": -62.0,
        "merchant": -4.0,
        "lesser-peon": 72.0,
    }[role]
    if role == "lesser-peon":
        base += max(0.0, burden - 6.0) * 3.0
    elif role in ("great-dalmuti", "lesser-dalmuti"):
        base -= max(0.0, burden - 6.0) * 2.0
    return 1 if base > 0.0 else 0


class DalmutiScalarEnv:
    """Scalar correctness environment with a torch observation boundary."""

    def __init__(
        self,
        player_count: int,
        *,
        acts: int = 5,
        seed: int = 1,
        device: torch.device | str = "cpu",
    ):
        if not MIN_PLAYERS <= player_count <= MAX_PLAYERS:
            raise ValueError("player_count must be from 4 through 10")
        if acts < 1:
            raise ValueError("acts must be positive")
        self.player_count = int(player_count)
        self.acts = int(acts)
        self.device = torch.device(device)
        self._seed = int(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> V4EnvironmentObservation:
        if seed is not None:
            self._seed = int(seed)
        self._rng = Mulberry32(self._seed)
        self._scores = {player_id: 0 for player_id in range(self.player_count)}
        self._order = [int(value) for value in self._rng.shuffle(range(self.player_count))]
        self._act = 1
        self._terminated = False
        self._last_act_result: Mapping[str, object] | None = None
        self._start_act()
        return self.observe()

    @property
    def current_player_id(self) -> int:
        if self._terminated:
            return -1
        return self._order[self._current_index]

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def hand_counts(self) -> torch.Tensor:
        rows = [_rank_counts(self._hands[player_id])[1:] for player_id in self._order]
        return torch.tensor(rows, dtype=torch.int16, device=self.device)

    @property
    def physical_card_count(self) -> int:
        return sum(len(hand) for hand in self._hands.values()) + sum(self._public_played)

    def _start_act(self) -> None:
        deck = self._rng.shuffle(create_deck())
        counts = ranked_deal_counts(DECK_SIZE, self.player_count)
        self._hands: dict[int, list[Card]] = {}
        cursor = 0
        for position, player_id in enumerate(self._order):
            self._hands[player_id] = _sorted_hand(deck[cursor : cursor + counts[position]])
            cursor += counts[position]
        self._revolution = 0
        holder = next(
            (
                player_id
                for player_id in self._order
                if sum(card.rank == JOKER_RANK for card in self._hands[player_id]) == 2
            ),
            None,
        )
        if holder is not None:
            position = self._order.index(holder)
            self._revolution = normal_revolution_decision(
                self._hands[holder], role_for_index(position, self.player_count)
            )
            if self._revolution == 2:
                self._order.reverse()
        self._tax_audit: tuple[Mapping[str, object], ...] = ()
        if self._act > 1 and self._revolution == 0:
            self._apply_normal_taxation()
        self._finish_order: list[int] = []
        self._passed: set[int] = set()
        self._public_played = [0] * 13
        self._history: list[dict[str, object]] = []
        self._event_sequence = 0
        self._table: TablePlay | None = None
        self._last_played_id: int | None = None
        self._current_index = 0
        self._transitions = 0

    def _apply_normal_taxation(self) -> None:
        pairs = ((0, self.player_count - 1, 2), (1, self.player_count - 2, 1))
        exchanges = []
        for noble_index, peon_index, count in pairs:
            noble_id = self._order[noble_index]
            peon_id = self._order[peon_index]
            peon_cards = tuple(
                card.id
                for card in sorted(
                    (card for card in self._hands[peon_id] if card.rank != JOKER_RANK),
                    key=lambda card: (card.rank, card.id),
                )[:count]
            )
            noble_cards = normal_tax_return_card_ids(self._hands[noble_id], count)
            exchanges.append((noble_id, peon_id, peon_cards, noble_cards))
        # Both choices are locked against pre-transfer hands, matching production.
        for noble_id, peon_id, peon_cards, _ in exchanges:
            self._transfer(peon_id, noble_id, peon_cards)
        for noble_id, peon_id, _, noble_cards in exchanges:
            self._transfer(noble_id, peon_id, noble_cards)
        self._tax_audit = tuple(
            {
                "noble_id": noble_id,
                "peon_id": peon_id,
                "tribute_card_ids": peon_cards,
                "return_card_ids": noble_cards,
            }
            for noble_id, peon_id, peon_cards, noble_cards in exchanges
        )

    def _transfer(self, source: int, target: int, card_ids: Sequence[str]) -> None:
        selected = set(card_ids)
        cards = [card for card in self._hands[source] if card.id in selected]
        if len(cards) != len(card_ids):
            raise RuntimeError("tax transfer references a missing physical card")
        self._hands[source] = _sorted_hand(
            card for card in self._hands[source] if card.id not in selected
        )
        self._hands[target] = _sorted_hand((*self._hands[target], *cards))

    def _normal_observation(self) -> NormalObservation:
        actor_id = self.current_player_id
        return NormalObservation(
            actor_id=actor_id,
            hand=tuple(self._hands[actor_id]),
            table=self._table,
            players=tuple(
                NormalPublicPlayer(
                    player_id,
                    len(self._hands[player_id]),
                    player_id in self._finish_order,
                )
                for player_id in self._order
            ),
            passed_player_ids=frozenset(self._passed),
        )

    def normal_action(self) -> int:
        if self._terminated:
            raise RuntimeError("a terminated environment cannot act")
        return choose_normal_action(self._normal_observation()).action_index

    def legal_mask(self) -> torch.Tensor:
        if self._terminated:
            return torch.zeros(ACTION_COUNT, dtype=torch.bool, device=self.device)
        actor_id = self.current_player_id
        mask = torch.zeros(ACTION_COUNT, dtype=torch.bool, device=self.device)
        mask[
            legal_action_indices(self._hands[actor_id], self._table)
        ] = True
        return mask

    def _next_active_index(self, from_index: int) -> int:
        for step in range(1, self.player_count + 1):
            index = (from_index + step + self.player_count) % self.player_count
            if self._hands[self._order[index]]:
                return index
        return from_index

    def _append_event(self, **values: object) -> None:
        event = {"sequence": self._event_sequence, **values}
        self._event_sequence += 1
        self._history.append(event)

    def _append_clear(
        self, table: TablePlay, reason: str, next_leader_id: int | None
    ) -> None:
        leader_count = len(self._hands[table.player_id])
        self._append_event(
            type="clear",
            actor_id=table.player_id,
            hand_before=leader_count,
            hand_after=leader_count,
            rank=table.rank,
            natural_count=table.natural_count,
            joker_count=table.joker_count,
            total_count=table.count,
            clear_reason=reason,
            next_leader_id=next_leader_id,
        )

    def _finish_act(self) -> tuple[torch.Tensor, Mapping[str, object]]:
        chips = {
            player_id: round_chip_award(index + 1, self.player_count)
            for index, player_id in enumerate(self._finish_order)
        }
        rewards = torch.zeros(MAX_PLAYERS, dtype=torch.float32, device=self.device)
        for player_id, award in chips.items():
            self._scores[player_id] += award
            rewards[player_id] = (award - 2) / 2.0
        result: Mapping[str, object] = {
            "act": self._act,
            "revolution": self._revolution,
            "player_order": tuple(self._order),
            "finish_order": tuple(self._finish_order),
            "chip_awards": chips,
            "transitions": self._transitions,
            "taxation": self._tax_audit,
        }
        self._last_act_result = result
        if self._act >= self.acts:
            self._terminated = True
        else:
            self._order = list(self._finish_order)
            self._act += 1
            self._start_act()
        return rewards, result

    def step(self, action_index: int) -> ScalarStepResult:
        if self._terminated:
            raise RuntimeError("step called after match termination")
        if isinstance(action_index, bool) or not isinstance(action_index, int):
            raise TypeError("action_index must be an integer")
        legal = self.legal_mask()
        if not 0 <= action_index < ACTION_COUNT or not bool(legal[action_index].item()):
            raise ValueError(f"illegal action {action_index}")
        acting_id = self.current_player_id
        action = decode_action(action_index)
        self._transitions += 1
        if self._transitions > MAX_TRANSITIONS_PER_ACT:
            raise RuntimeError("act exceeded the transition safety bound")
        rewards = torch.zeros(MAX_PLAYERS, dtype=torch.float32, device=self.device)
        act_ended = False
        act_result: Mapping[str, object] | None = None

        if action.type == "pass":
            if self._table is None:
                raise RuntimeError("a leading player cannot pass")
            before = len(self._hands[acting_id])
            self._passed.add(acting_id)
            self._append_event(
                type="pass",
                actor_id=acting_id,
                hand_before=before,
                hand_after=before,
                pass_reason=(
                    "insufficient-cards"
                    if before < self._table.count
                    else "manual"
                ),
            )
            active = [player_id for player_id in self._order if self._hands[player_id]]
            required = [player_id for player_id in active if player_id != self._last_played_id]
            trick_over = all(player_id in self._passed for player_id in required)
            if trick_over:
                if self._last_played_id is None:
                    raise RuntimeError("table has no last player")
                table = self._table
                leader_index = self._order.index(self._last_played_id)
                leader_active = bool(self._hands[self._last_played_id])
                next_index = leader_index if leader_active else self._next_active_index(leader_index)
                next_id = self._order[next_index]
                self._append_clear(table, "all-passed", next_id)
                self._table = None
                self._passed.clear()
                self._current_index = next_index
            else:
                self._current_index = self._next_active_index(self._current_index)
        else:
            cards = _resolve_cards(self._hands[acting_id], action)
            selected = {card.id for card in cards}
            before = len(self._hands[acting_id])
            self._hands[acting_id] = _sorted_hand(
                card for card in self._hands[acting_id] if card.id not in selected
            )
            for card in cards:
                self._public_played[card.rank - 1] += 1
            self._table = TablePlay(
                action.rank, action.count, acting_id, action.joker_count
            )
            self._last_played_id = acting_id
            self._passed.clear()
            self._append_event(
                type="play",
                actor_id=acting_id,
                hand_before=before,
                hand_after=len(self._hands[acting_id]),
                rank=action.rank,
                natural_count=action.natural_count,
                joker_count=action.joker_count,
                total_count=action.count,
            )
            if not self._hands[acting_id]:
                self._finish_order.append(acting_id)
                self._append_event(
                    type="finish",
                    actor_id=acting_id,
                    hand_before=0,
                    hand_after=0,
                    finish_place=len(self._finish_order),
                )
            if len(self._finish_order) == self.player_count - 1:
                last = next(
                    player_id
                    for player_id in self._order
                    if player_id not in self._finish_order
                )
                self._finish_order.append(last)
                self._append_clear(self._table, "act-ended", None)
                act_ended = True
                rewards, act_result = self._finish_act()
            elif action.rank == 1:
                for offset in range(1, self.player_count):
                    player_id = self._order[
                        (self._current_index + offset) % self.player_count
                    ]
                    if not self._hands[player_id]:
                        continue
                    count = len(self._hands[player_id])
                    self._append_event(
                        type="pass",
                        actor_id=player_id,
                        hand_before=count,
                        hand_after=count,
                        pass_reason="dalmuti",
                    )
                actor_active = bool(self._hands[acting_id])
                next_index = (
                    self._current_index
                    if actor_active
                    else self._next_active_index(self._current_index)
                )
                next_id = self._order[next_index]
                self._append_clear(self._table, "dalmuti", next_id)
                self._table = None
                self._passed.clear()
                self._current_index = next_index
            else:
                self._current_index = self._next_active_index(self._current_index)

        info: dict[str, object] = {
            "acting_player_id": acting_id,
            "action_index": action_index,
            "act_ended": act_ended,
        }
        if act_result is not None:
            info["act_result"] = act_result
        return ScalarStepResult(
            observation=self.observe(),
            rewards=rewards,
            terminated=self._terminated,
            act_ended=act_ended,
            info=info,
        )

    def _relative_order(self, actor_id: int) -> list[int]:
        actor_index = self._order.index(actor_id)
        return [
            self._order[(actor_index + offset) % self.player_count]
            for offset in range(self.player_count)
        ]

    def _event_features(
        self,
        event: Mapping[str, object],
        actor_id: int,
        *,
        memory_trace: bool = False,
    ) -> list[float]:
        event_types = ("play", "pass", "clear", "finish")
        pass_reasons = ("manual", "timeout", "insufficient-cards", "dalmuti")
        clear_reasons = ("all-passed", "dalmuti", "act-ended")
        row = [0.0] * HISTORY_FEATURES
        row[event_types.index(str(event["type"]))] = 1.0
        relative = self._relative_order(actor_id)
        event_actor = int(event["actor_id"])
        row[4] = relative.index(event_actor) / max(1, self.player_count - 1)
        row[5] = int(event["hand_before"]) / 20.0
        row[6] = int(event["hand_after"]) / 20.0
        row[7] = int(event.get("rank", 0)) / 13.0
        # Recent history uses the dataset's /14 token scaling. Canonical
        # compressed memory traces predate that tensorizer and use /12.
        row[8] = int(event.get("natural_count", 0)) / (
            12.0 if memory_trace else 14.0
        )
        row[9] = int(event.get("joker_count", 0)) / 2.0
        row[10] = int(event.get("total_count", 0)) / 14.0
        if event["type"] == "pass":
            row[11 + pass_reasons.index(str(event["pass_reason"]))] = 1.0
        if event["type"] == "clear":
            row[15 + clear_reasons.index(str(event["clear_reason"]))] = 1.0
            next_id = event.get("next_leader_id")
            if next_id is not None:
                next_offset = relative.index(int(next_id))
                row[18] = (
                    (next_offset + 1) / self.player_count
                    if memory_trace
                    else next_offset / max(1, self.player_count - 1)
                )
        row[19] = int(event.get("finish_place", 0)) / self.player_count
        return row

    def _terminal_public_observation(self) -> V4ActorObservation:
        zeros = lambda *shape: torch.zeros(
            *shape, dtype=torch.float32, device=self.device
        )
        return V4ActorObservation(
            actor_id=-1,
            valid=torch.tensor(False, dtype=torch.bool, device=self.device),
            global_features=zeros(12),
            rank_features=zeros(13, 6),
            player_features=zeros(MAX_PLAYERS, 12),
            player_mask=torch.zeros(MAX_PLAYERS, dtype=torch.bool, device=self.device),
            memory_trace_features=zeros(4, HISTORY_FEATURES),
            history_features=zeros(MAX_HISTORY, HISTORY_FEATURES),
            history_mask=torch.zeros(MAX_HISTORY, dtype=torch.bool, device=self.device),
            legal_mask=torch.zeros(ACTION_COUNT, dtype=torch.bool, device=self.device),
        )

    def public_observation(self) -> V4ActorObservation:
        if self._terminated:
            return self._terminal_public_observation()
        actor_id = self.current_player_id
        relative = self._relative_order(actor_id)
        actor_position = self._order.index(actor_id)
        role_id = ROLES.index(role_for_index(actor_position, self.player_count))
        truncated = max(0, len(self._history) - MAX_HISTORY)
        global_values = [
            (self.player_count - 4) / 6.0,
            math.tanh((self._act - 1) / 10.0),
            *[float(index == role_id) for index in range(5)],
            *[float(index == self._revolution) for index in range(3)],
            math.tanh(truncated / MAX_HISTORY),
            float(self._table is not None),
        ]
        own_counts = _rank_counts(self._hands[actor_id])[1:]
        rank_rows = []
        for rank_index in range(13):
            copies = 2 if rank_index == 12 else rank_index + 1
            is_table = self._table is not None and self._table.rank == rank_index + 1
            natural = self._table.natural_count if is_table else 0
            jokers = self._table.joker_count if is_table else 0
            rank_rows.append(
                [
                    own_counts[rank_index] / copies,
                    self._public_played[rank_index] / copies,
                    float(is_table),
                    natural / 14.0,
                    jokers / 2.0,
                    (copies - own_counts[rank_index] - self._public_played[rank_index])
                    / copies,
                ]
            )
        player_rows = torch.zeros(
            MAX_PLAYERS, 12, dtype=torch.float32, device=self.device
        )
        player_mask = torch.zeros(MAX_PLAYERS, dtype=torch.bool, device=self.device)
        for offset, player_id in enumerate(relative):
            absolute_position = self._order.index(player_id)
            player_role = ROLES.index(role_for_index(absolute_position, self.player_count))
            row = [
                offset / max(1, self.player_count - 1),
                len(self._hands[player_id]) / 20.0,
                float(not self._hands[player_id]),
                float(player_id in self._passed),
                float(offset == 0),
                float(self._table is not None and self._table.player_id == player_id),
                *[float(index == player_role) for index in range(5)],
                math.tanh(self._scores[player_id] / 10.0),
            ]
            player_rows[offset] = torch.tensor(
                row, dtype=torch.float32, device=self.device
            )
            player_mask[offset] = True
        old_events = self._history[:truncated]
        recent_events = self._history[truncated:]
        memory_rows = []
        for decay in MEMORY_TRACE_DECAYS:
            trace = [0.0] * HISTORY_FEATURES
            for event in old_events:
                features = self._event_features(
                    event, actor_id, memory_trace=True
                )
                trace = [
                    decay * previous + (1.0 - decay) * value
                    for previous, value in zip(trace, features)
                ]
            memory_rows.append(trace)
        history_rows = torch.zeros(
            MAX_HISTORY,
            HISTORY_FEATURES,
            dtype=torch.float32,
            device=self.device,
        )
        history_mask = torch.zeros(MAX_HISTORY, dtype=torch.bool, device=self.device)
        for index, event in enumerate(recent_events):
            history_rows[index] = torch.tensor(
                self._event_features(event, actor_id),
                dtype=torch.float32,
                device=self.device,
            )
            history_mask[index] = True
        return V4ActorObservation(
            actor_id=actor_id,
            valid=torch.tensor(True, dtype=torch.bool, device=self.device),
            global_features=torch.tensor(
                global_values, dtype=torch.float32, device=self.device
            ),
            rank_features=torch.tensor(
                rank_rows, dtype=torch.float32, device=self.device
            ),
            player_features=player_rows,
            player_mask=player_mask,
            memory_trace_features=torch.tensor(
                memory_rows, dtype=torch.float32, device=self.device
            ),
            history_features=history_rows,
            history_mask=history_mask,
            legal_mask=self.legal_mask(),
        )

    def privileged_state(self) -> torch.Tensor:
        """Return the exact TS schema-v1 512-vector; never pass it to the actor."""

        vector = torch.zeros(
            PRIVILEGED_STATE_SIZE, dtype=torch.float32, device=self.device
        )
        actor_id = self.current_player_id if not self._terminated else self._order[0]
        relative = self._relative_order(actor_id)
        table = self._table
        actor_position = self._order.index(actor_id)
        actor_role = role_for_index(actor_position, self.player_count)
        table_actor_offset = (
            -1 if table is None else relative.index(table.player_id)
        )
        global_values = [
            float(self.player_count),
            float(self._act),
            float(self._revolution),
            float(table is not None),
            0.0 if table is None else float(table.rank),
            0.0 if table is None else float(table.natural_count),
            0.0 if table is None else float(table.joker_count),
            0.0 if table is None else float(table.count),
            float(table_actor_offset),
            float(sum(self._public_played)),
            float(sum(bool(self._hands[player_id]) for player_id in self._order)),
            float(len(self._finish_order)),
            float(ROLES.index(actor_role)),
            float(self._scores[actor_id]),
            float(len(self._hands[actor_id])),
            float(len(self._history)),
        ]
        vector[
            PRIVILEGED_GLOBAL_OFFSET:
            PRIVILEGED_GLOBAL_OFFSET + PRIVILEGED_GLOBAL_FEATURES
        ] = torch.tensor(
            global_values, dtype=torch.float32, device=self.device
        )
        vector[
            PRIVILEGED_PUBLIC_RANK_OFFSET:
            PRIVILEGED_PUBLIC_RANK_OFFSET + NORMAL_RANKS + 1
        ] = torch.tensor(
            self._public_played, dtype=torch.float32, device=self.device
        )
        for offset, player_id in enumerate(relative):
            counts = _rank_counts(self._hands[player_id])[1:]
            position = self._order.index(player_id)
            role_id = ROLES.index(role_for_index(position, self.player_count))
            finish_place = (
                self._finish_order.index(player_id) + 1
                if player_id in self._finish_order
                else 0
            )
            player_values = [
                1.0,
                float(offset),
                *[float(index == role_id) for index in range(len(ROLES))],
                float(self._scores[player_id]),
                float(len(self._hands[player_id])),
                float(player_id in self._passed),
                float(not self._hands[player_id]),
                float(finish_place),
                *[float(count) for count in counts],
            ]
            start = PRIVILEGED_PLAYER_OFFSET + offset * PRIVILEGED_PLAYER_STRIDE
            vector[start:start + PRIVILEGED_PLAYER_STRIDE] = torch.tensor(
                player_values, dtype=torch.float32, device=self.device
            )
        if bool(vector[PRIVILEGED_RESERVED_OFFSET:].any()):
            raise AssertionError("privileged critic reserved tail must remain zero")
        return vector

    def observe(self) -> V4EnvironmentObservation:
        return V4EnvironmentObservation(
            public=self.public_observation(),
            privileged_state=self.privileged_state(),
        )

    def resample_hidden_hands(self, seed: int) -> V4EnvironmentObservation:
        """Shuffle only opponents' hidden ownership, preserving public state."""

        if self._terminated:
            raise RuntimeError("cannot determinize a terminated environment")
        actor_id = self.current_player_id
        opponents = [player_id for player_id in self._order if player_id != actor_id]
        counts = [len(self._hands[player_id]) for player_id in opponents]
        pooled = [card for player_id in opponents for card in self._hands[player_id]]
        shuffled = Mulberry32(seed).shuffle(pooled)
        cursor = 0
        for player_id, count in zip(opponents, counts):
            self._hands[player_id] = _sorted_hand(shuffled[cursor : cursor + count])
            cursor += count
        if cursor != len(shuffled):
            raise AssertionError("hidden hand determinization lost cards")
        return self.observe()

    def state_fingerprint(self) -> tuple[object, ...]:
        return (
            self._seed,
            self._act,
            tuple(self._order),
            tuple(
                (player_id, tuple((card.id, card.rank) for card in self._hands[player_id]))
                for player_id in sorted(self._hands)
            ),
            tuple(sorted(self._scores.items())),
            self._current_index,
            self._table,
            self._last_played_id,
            tuple(sorted(self._passed)),
            tuple(self._finish_order),
            tuple(self._public_played),
            tuple(tuple(sorted(event.items())) for event in self._history),
            self._revolution,
            self._terminated,
            self._rng.state,
        )


def _stack_observations(
    observations: Sequence[V4EnvironmentObservation],
) -> V4BatchedEnvironmentObservation:
    public = [observation.public for observation in observations]
    return V4BatchedEnvironmentObservation(
        public=V4BatchedActorObservation(
            actor_ids=torch.tensor(
                [observation.actor_id for observation in public],
                dtype=torch.long,
                device=public[0].global_features.device,
            ),
            valid=torch.stack([observation.valid for observation in public]),
            global_features=torch.stack(
                [observation.global_features for observation in public]
            ),
            rank_features=torch.stack(
                [observation.rank_features for observation in public]
            ),
            player_features=torch.stack(
                [observation.player_features for observation in public]
            ),
            player_mask=torch.stack([observation.player_mask for observation in public]),
            memory_trace_features=torch.stack(
                [observation.memory_trace_features for observation in public]
            ),
            history_features=torch.stack(
                [observation.history_features for observation in public]
            ),
            history_mask=torch.stack(
                [observation.history_mask for observation in public]
            ),
            legal_masks=torch.stack([observation.legal_mask for observation in public]),
        ),
        privileged_states=torch.stack(
            [observation.privileged_state for observation in observations]
        ),
    )


class DalmutiBatchEnv:
    """Fixed-shape torch batch facade over independent exact scalar lanes."""

    def __init__(
        self,
        player_counts: int | Sequence[int],
        *,
        batch_size: int | None = None,
        acts: int = 5,
        seeds: Sequence[int] | None = None,
        device: torch.device | str = "cpu",
        auto_reset: bool = True,
    ):
        if isinstance(player_counts, int):
            size = 1 if batch_size is None else int(batch_size)
            counts = [player_counts] * size
        else:
            counts = list(player_counts)
            if batch_size is not None and batch_size != len(counts):
                raise ValueError("batch_size disagrees with player_counts")
            size = len(counts)
        if size < 1:
            raise ValueError("batch must contain at least one lane")
        seed_values = list(seeds) if seeds is not None else list(range(1, size + 1))
        if len(seed_values) != size:
            raise ValueError("seeds must match the batch size")
        self.device = torch.device(device)
        self.auto_reset = bool(auto_reset)
        self._base_seeds = [int(seed) for seed in seed_values]
        self._episode_numbers = [0] * size
        self.envs = [
            DalmutiScalarEnv(count, acts=acts, seed=seed, device=self.device)
            for count, seed in zip(counts, seed_values)
        ]

    @property
    def batch_size(self) -> int:
        return len(self.envs)

    def reset(self, seeds: Sequence[int] | None = None) -> V4BatchedEnvironmentObservation:
        if seeds is not None and len(seeds) != self.batch_size:
            raise ValueError("seeds must match the batch size")
        if seeds is not None:
            self._base_seeds = [int(seed) for seed in seeds]
        self._episode_numbers = [0] * self.batch_size
        observations = [
            env.reset(self._base_seeds[index])
            for index, env in enumerate(self.envs)
        ]
        return _stack_observations(observations)

    def normal_actions(self) -> torch.Tensor:
        return torch.tensor(
            [env.normal_action() if not env.terminated else -1 for env in self.envs],
            dtype=torch.long,
            device=self.device,
        )

    def step(self, action_indices: torch.Tensor | Sequence[int]) -> BatchedStepResult:
        if isinstance(action_indices, torch.Tensor):
            if action_indices.shape != (self.batch_size,):
                raise ValueError("batched actions must have shape [batch]")
            actions = [int(value) for value in action_indices.detach().cpu().tolist()]
        else:
            actions = [int(value) for value in action_indices]
            if len(actions) != self.batch_size:
                raise ValueError("batched actions must match the batch size")
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations: list[V4EnvironmentObservation] = []
        infos: list[Mapping[str, object]] = []
        for lane, (env, result) in enumerate(zip(self.envs, results)):
            info = dict(result.info)
            observation = result.observation
            if result.terminated and self.auto_reset:
                self._episode_numbers[lane] += 1
                next_seed = (
                    self._base_seeds[lane]
                    + self._episode_numbers[lane] * 0x9E37_79B9
                ) & 0xFFFF_FFFF
                observation = env.reset(next_seed)
                info["auto_reset_seed"] = next_seed
            observations.append(observation)
            infos.append(info)
        return BatchedStepResult(
            observation=_stack_observations(observations),
            rewards=torch.stack([result.rewards for result in results]),
            terminated=torch.tensor(
                [result.terminated for result in results],
                dtype=torch.bool,
                device=self.device,
            ),
            act_ended=torch.tensor(
                [result.act_ended for result in results],
                dtype=torch.bool,
                device=self.device,
            ),
            infos=tuple(infos),
        )


__all__ = [
    "ACTION_CATALOGUE",
    "ACTION_COUNT",
    "BatchedStepResult",
    "Card",
    "DalmutiBatchEnv",
    "DalmutiScalarEnv",
    "DECK_SIZE",
    "HISTORY_FEATURES",
    "JOKER_RANK",
    "MAX_HISTORY",
    "MAX_PLAYERS",
    "MEMORY_TRACE_DECAYS",
    "MIN_PLAYERS",
    "Mulberry32",
    "NormalDecision",
    "NormalObservation",
    "NormalPublicPlayer",
    "PASS_ACTION_INDEX",
    "PRIVILEGED_LAYOUT",
    "PRIVILEGED_GLOBAL_FIELDS",
    "PRIVILEGED_GLOBAL_FEATURES",
    "PRIVILEGED_GLOBAL_OFFSET",
    "PRIVILEGED_PLAYER_OFFSET",
    "PRIVILEGED_PLAYER_STRIDE",
    "PRIVILEGED_PUBLIC_RANK_OFFSET",
    "PRIVILEGED_RESERVED_OFFSET",
    "PRIVILEGED_STATE_LAYOUT",
    "PRIVILEGED_STATE_LAYOUT_ID",
    "PRIVILEGED_STATE_LAYOUT_SHA256",
    "PRIVILEGED_STATE_SIZE",
    "PRIVILEGED_STATE_VERSION",
    "ROLES",
    "SOLO_JOKER_ACTION_INDEX",
    "ScalarStepResult",
    "SemanticAction",
    "TablePlay",
    "V4ActorObservation",
    "V4BatchedActorObservation",
    "V4BatchedEnvironmentObservation",
    "V4EnvironmentObservation",
    "choose_normal_action",
    "create_deck",
    "decode_action",
    "encode_action",
    "legal_action_indices",
    "legal_action_masks",
    "normal_revolution_decision",
    "normal_tax_return_card_ids",
    "ranked_deal_counts",
    "role_for_index",
    "round_chip_award",
]
