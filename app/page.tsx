"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  selectDalmutiReturnCards,
  selectPeonTaxCards,
} from "@/lib/taxation";
import { rankedDealCounts } from "@/lib/dealing";
import { toggleWholeRankSelection } from "@/lib/selection";

type Role =
  | "great-dalmuti"
  | "lesser-dalmuti"
  | "merchant"
  | "lesser-peon"
  | "great-peon";

type Phase =
  | "ready"
  | "reveal-intro"
  | "hand-reveal"
  | "tax-intro"
  | "playing"
  | "play-intro"
  | "revolution"
  | "taxation"
  | "round-end";

type Card = {
  id: string;
  rank: number;
};

type Player = {
  id: string;
  name: string;
  monogram: string;
  isHuman: boolean;
  role: Role;
};

type PlayedSet = {
  rank: number;
  count: number;
  playerId: string;
  cards: Card[];
};

type PublicTurnAction = {
  id: string;
  kind: "play" | "pass";
  player: Player;
  cards: Card[];
  previousTable: PlayedSet | null;
};

type TaxExchange = {
  nobleId: string;
  peonId: string;
  nobleGift: Card[];
  peonGift: Card[];
};

type TaxStage = "selection" | "tribute" | "return";
type TaxDirection = "source" | "destination";

type Point = {
  x: number;
  y: number;
};

type TaxTransferRoute = {
  id: string;
  from: Player;
  to: Player;
  cards: Card[];
  reveal: boolean;
  routeIndex: number;
};

type TaxAnchorMap = {
  players: Record<string, Point>;
  midpoint: Point | null;
};

type GameState = {
  phase: Phase;
  round: number;
  revision: number;
  players: Player[];
  hands: Record<string, Card[]>;
  scores: Record<string, number>;
  currentIndex: number;
  table: PlayedSet | null;
  lastPlayedId: string | null;
  passed: string[];
  finishOrder: string[];
  log: string[];
  revolutionHolder: string | null;
  taxExchanges: TaxExchange[];
  taxStage: TaxStage | null;
  taxAnimationId: string | null;
  tributeHands: Record<string, Card[]> | null;
  taxedHands: Record<string, Card[]> | null;
  publicAction: PublicTurnAction | null;
};

const HUMAN_ID = "you";
const TAX_STAGE_DURATION_MS = 4000;
const REVEAL_INTRO_DURATION_MS = 1600;
const HAND_REVEAL_DURATION_MS = 900;
const TAX_INTRO_DURATION_MS = 1500;
const PLAY_INTRO_DURATION_MS = 1800;
const PUBLIC_ACTION_DURATION_MS = 1500;
const CARD_ART_VERSION = "2026-07-24-2x";

function createTaxAnimationId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createPublicActionId(kind: PublicTurnAction["kind"]) {
  return `${kind}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

const BASE_PLAYERS: Omit<Player, "role">[] = [
  { id: HUMAN_ID, name: "나", monogram: "나", isHuman: true },
  { id: "marco", name: "마르코", monogram: "마", isHuman: false },
  { id: "luna", name: "루나", monogram: "루", isHuman: false },
  { id: "tobias", name: "토비아스", monogram: "토", isHuman: false },
  { id: "seraphine", name: "세라핀", monogram: "세", isHuman: false },
];

const ROLE_LABELS: Record<Role, string> = {
  "great-dalmuti": "대 달무티",
  "lesser-dalmuti": "소 달무티",
  merchant: "상인",
  "lesser-peon": "소 농노",
  "great-peon": "대 농노",
};

const ROLE_MARKS: Record<Role, string> = {
  "great-dalmuti": "♛",
  "lesser-dalmuti": "♕",
  merchant: "◆",
  "lesser-peon": "♙",
  "great-peon": "♟",
};

const RANK_NAMES: Record<number, string> = {
  1: "달무티",
  2: "대주교",
  3: "시종장",
  4: "남작부인",
  5: "수녀원장",
  6: "기사",
  7: "재봉사",
  8: "석공",
  9: "요리사",
  10: "양치기",
  11: "광부",
  12: "농노",
  13: "어릿광대",
};

function subjectLabel(name: string): string {
  return `${name}이(가)`;
}

function roleForIndex(index: number, total: number): Role {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === total - 2) return "lesser-peon";
  if (index === total - 1) return "great-peon";
  return "merchant";
}

function assignRoles(players: Omit<Player, "role">[] | Player[]): Player[] {
  return players.map((player, index) => ({
    ...player,
    role: roleForIndex(index, players.length),
  }));
}

function createDeck(): Card[] {
  const deck: Card[] = [];
  for (let rank = 1; rank <= 12; rank += 1) {
    for (let copy = 0; copy < rank; copy += 1) {
      deck.push({ id: `${rank}-${copy}`, rank });
    }
  }
  deck.push({ id: "joker-1", rank: 13 });
  deck.push({ id: "joker-2", rank: 13 });
  return deck;
}

function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const target = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[target]] = [copy[target], copy[index]];
  }
  return copy;
}

function sortHand(cards: Card[]): Card[] {
  return [...cards].sort((a, b) => b.rank - a.rank || a.id.localeCompare(b.id));
}

function deal(players: Player[]): Record<string, Card[]> {
  const deck = shuffle(createDeck());
  const counts = rankedDealCounts(deck.length, players.length);
  const hands: Record<string, Card[]> = {};
  let cursor = 0;

  players.forEach((player, index) => {
    const count = counts[index];
    hands[player.id] = sortHand(deck.slice(cursor, cursor + count));
    cursor += count;
  });

  return hands;
}

function removeCards(hand: Card[], ids: string[]): Card[] {
  const selected = new Set(ids);
  return hand.filter((card) => !selected.has(card.id));
}

function normalizedSet(cards: Card[]): { rank: number; count: number } | null {
  if (cards.length === 0) return null;
  const normalCards = cards.filter((card) => card.rank !== 13);
  if (normalCards.length === 0) {
    return cards.length === 1 ? { rank: 13, count: 1 } : null;
  }
  const rank = normalCards[0].rank;
  if (normalCards.some((card) => card.rank !== rank)) return null;
  return { rank, count: cards.length };
}

function validationMessage(cards: Card[], table: PlayedSet | null): string | null {
  const set = normalizedSet(cards);
  if (!set) return "같은 계급의 카드만 함께 낼 수 있어요.";
  if (!table) return null;
  if (set.count !== table.count) return `${table.count}장을 내야 해요.`;
  if (set.rank >= table.rank) return `${table.rank}보다 강한 낮은 숫자가 필요해요.`;
  return null;
}

function findPlayer(players: Player[], role: Role): Player {
  return players.find((player) => player.role === role)!;
}

function settleTaxHands(
  tributeHands: Record<string, Card[]>,
  exchanges: TaxExchange[],
): Record<string, Card[]> {
  const hands = Object.fromEntries(
    Object.entries(tributeHands).map(([id, hand]) => [id, [...hand]]),
  );

  for (const exchange of exchanges) {
    hands[exchange.nobleId] = sortHand(
      removeCards(
        hands[exchange.nobleId],
        exchange.nobleGift.map((card) => card.id),
      ),
    );
    hands[exchange.peonId] = sortHand([
      ...hands[exchange.peonId],
      ...exchange.nobleGift,
    ]);
  }

  return hands;
}

function applyTax(
  players: Player[],
  sourceHands: Record<string, Card[]>,
): {
  hands: Record<string, Card[]> | null;
  tributeHands: Record<string, Card[]>;
  notes: string[];
  exchanges: TaxExchange[];
} {
  const notes: string[] = [];
  const exchanges: TaxExchange[] = [];

  const describeExchange = (nobleRole: Role, peonRole: Role, count: number) => {
    const noble = findPlayer(players, nobleRole);
    const peon = findPlayer(players, peonRole);
    const peonGift = selectPeonTaxCards(sourceHands[peon.id], count);
    const nobleGift = noble.isHuman
      ? []
      : selectDalmutiReturnCards(sourceHands[noble.id], count);

    exchanges.push({
      nobleId: noble.id,
      peonId: peon.id,
      nobleGift,
      peonGift,
    });
    notes.push(`${ROLE_LABELS[peonRole]}가 ${ROLE_LABELS[nobleRole]}에게 세금을 바쳤습니다.`);
  };

  describeExchange("great-dalmuti", "great-peon", 2);
  describeExchange("lesser-dalmuti", "lesser-peon", 1);

  const tributeHands = Object.fromEntries(
    Object.entries(sourceHands).map(([id, hand]) => [id, [...hand]]),
  );

  for (const exchange of exchanges) {
    tributeHands[exchange.peonId] = sortHand(
      removeCards(
        tributeHands[exchange.peonId],
        exchange.peonGift.map((card) => card.id),
      ),
    );
    tributeHands[exchange.nobleId] = sortHand([
      ...tributeHands[exchange.nobleId],
      ...exchange.peonGift,
    ]);
  }

  const needsHumanChoice = exchanges.some(
    (exchange) =>
      exchange.nobleId === HUMAN_ID &&
      exchange.nobleGift.length < exchange.peonGift.length,
  );
  const hands = needsHumanChoice
    ? null
    : settleTaxHands(tributeHands, exchanges);

  return { hands, tributeHands, notes, exchanges };
}

function prepareRound(
  orderedPlayers: Player[],
  round: number,
  scores: Record<string, number>,
  forceTaxPreview = false,
  waitForHost = false,
): GameState {
  let players = assignRoles(orderedPlayers);
  const hands = deal(players);
  const holder = forceTaxPreview
    ? undefined
    : players.find(
        (player) => hands[player.id].filter((card) => card.rank === 13).length === 2,
      );
  let phase: Phase = waitForHost ? "ready" : "play-intro";
  let revolutionHolder: string | null = holder?.id ?? null;
  let taxExchanges: TaxExchange[] = [];
  let taxStage: TaxStage | null = null;
  let taxAnimationId: string | null = null;
  let tributeHands: Record<string, Card[]> | null = null;
  let taxedHands: Record<string, Card[]> | null = null;
  let log = [`제 ${round}막이 시작되었습니다.`];

  if (waitForHost) {
    log = ["방장의 PLAY를 기다리고 있습니다.", ...log];
  } else if (holder?.isHuman) {
    phase = "revolution";
    log = ["두 광대가 당신의 손에 모였습니다. 혁명을 선택하세요.", ...log];
  } else if (holder) {
    if (holder.role === "great-peon") {
      players = assignRoles([...players].reverse());
      log = [`${holder.name}의 대혁명! 모든 계급이 뒤집혔습니다.`, ...log];
    } else {
      log = [
        `${subjectLabel(holder.name)} 혁명을 선포해 세금이 취소되었습니다.`,
        ...log,
      ];
    }
    revolutionHolder = null;
  } else {
    const taxed = applyTax(players, hands);
    phase = "tax-intro";
    taxExchanges = taxed.exchanges;
    taxStage = taxed.hands ? "tribute" : "selection";
    taxAnimationId = createTaxAnimationId();
    tributeHands = taxed.tributeHands;
    taxedHands = taxed.hands;
    log = [...taxed.notes, ...log];
  }

  return {
    phase,
    round,
    revision: 0,
    players,
    hands,
    scores,
    currentIndex: 0,
    table: null,
    lastPlayedId: null,
    passed: [],
    finishOrder: [],
    log,
    revolutionHolder,
    taxExchanges,
    taxStage,
    taxAnimationId,
    tributeHands,
    taxedHands,
    publicAction: null,
  };
}

function advanceAfterHandReveal(state: GameState): GameState {
  if (state.phase !== "hand-reveal") return state;

  const holder = state.players.find(
    (player) => player.id === state.revolutionHolder,
  );

  if (holder?.isHuman) {
    return {
      ...state,
      phase: "revolution",
      revision: state.revision + 1,
      log: [
        "패 공개가 끝났습니다.",
        "두 광대가 당신의 손에 모였습니다. 혁명을 선택하세요.",
        ...state.log,
      ].slice(0, 12),
    };
  }

  if (holder) {
    const isGreatRevolution = holder.role === "great-peon";
    const players = isGreatRevolution
      ? assignRoles([...state.players].reverse())
      : state.players;

    return {
      ...state,
      phase: "play-intro",
      revision: state.revision + 1,
      players,
      currentIndex: 0,
      revolutionHolder: null,
      log: [
        "패 공개가 끝났습니다.",
        isGreatRevolution
          ? `${subjectLabel(holder.name)} 대혁명을 선포해 모든 계급이 뒤집혔습니다.`
          : `${subjectLabel(holder.name)} 혁명을 선포해 이번 막의 세금이 취소되었습니다.`,
        ...state.log,
      ].slice(0, 12),
    };
  }

  const taxed = applyTax(state.players, state.hands);
  return {
    ...state,
    phase: "tax-intro",
    revision: state.revision + 1,
    taxExchanges: taxed.exchanges,
    taxStage: taxed.hands ? "tribute" : "selection",
    taxAnimationId: createTaxAnimationId(),
    tributeHands: taxed.tributeHands,
    taxedHands: taxed.hands,
    log: [
      "패 공개가 끝났습니다.",
      ...taxed.notes,
      ...state.log,
    ].slice(0, 12),
  };
}

function nextActiveIndex(state: GameState, fromIndex: number): number {
  for (let step = 1; step <= state.players.length; step += 1) {
    const index = (fromIndex + step) % state.players.length;
    if (state.hands[state.players[index].id].length > 0) return index;
  }
  return fromIndex;
}

function playCards(state: GameState, playerId: string, cardIds: string[]): GameState {
  if (state.phase !== "playing" || state.publicAction) return state;
  const current = state.players[state.currentIndex];
  if (current.id !== playerId) return state;

  const hand = state.hands[playerId];
  const selected = hand.filter((card) => cardIds.includes(card.id));
  const set = normalizedSet(selected);
  if (!set || validationMessage(selected, state.table)) return state;
  const publicAction: PublicTurnAction = {
    id: createPublicActionId("play"),
    kind: "play",
    player: current,
    cards: selected,
    previousTable: state.table,
  };

  const hands = { ...state.hands, [playerId]: removeCards(hand, cardIds) };
  const finishOrder = [...state.finishOrder];
  const scores = { ...state.scores };
  const log = [
    `${subjectLabel(current.name)} ${set.rank === 13 ? "광대" : `${set.rank}등급`} ${set.count}장을 냈습니다.`,
    ...state.log,
  ].slice(0, 12);

  if (hands[playerId].length === 0) {
    finishOrder.push(playerId);
    scores[playerId] += state.players.length - finishOrder.length;
    log.unshift(
      `${subjectLabel(current.name)} ${finishOrder.length}위로 계급 경쟁을 마쳤습니다.`,
    );
  }

  if (finishOrder.length === state.players.length - 1) {
    const last = state.players.find((player) => !finishOrder.includes(player.id));
    if (last) finishOrder.push(last.id);
    return {
      ...state,
      phase: "round-end",
      revision: state.revision + 1,
      hands,
      scores,
      table: { ...set, playerId, cards: selected },
      lastPlayedId: playerId,
      finishOrder,
      log: ["이번 막의 새로운 계급이 결정되었습니다.", ...log],
      publicAction,
    };
  }

  const nextState: GameState = {
    ...state,
    revision: state.revision + 1,
    hands,
    scores,
    table: { ...set, playerId, cards: selected },
    lastPlayedId: playerId,
    passed: [],
    finishOrder,
    log,
    publicAction,
  };
  nextState.currentIndex = nextActiveIndex(nextState, state.currentIndex);
  return nextState;
}

function passTurn(state: GameState, playerId: string): GameState {
  if (state.phase !== "playing" || !state.table || state.publicAction) return state;
  const current = state.players[state.currentIndex];
  if (current.id !== playerId) return state;
  const publicAction: PublicTurnAction = {
    id: createPublicActionId("pass"),
    kind: "pass",
    player: current,
    cards: [],
    previousTable: state.table,
  };

  const passed = [...new Set([...state.passed, playerId])];
  const active = state.players.filter((player) => state.hands[player.id].length > 0);
  const requiredToPass = active.filter((player) => player.id !== state.lastPlayedId);
  const log = [`${subjectLabel(current.name)} 패스했습니다.`, ...state.log].slice(
    0,
    12,
  );

  if (requiredToPass.every((player) => passed.includes(player.id))) {
    const lastIndex = state.players.findIndex(
      (player) => player.id === state.lastPlayedId,
    );
    const lastStillActive =
      lastIndex >= 0 && state.hands[state.players[lastIndex].id].length > 0;
    const cleared: GameState = {
      ...state,
      revision: state.revision + 1,
      table: null,
      passed: [],
      log: ["판이 비워졌습니다. 새로운 묶음을 시작합니다.", ...log].slice(0, 12),
      publicAction,
    };
    cleared.currentIndex = lastStillActive
      ? lastIndex
      : nextActiveIndex(cleared, lastIndex);
    return cleared;
  }

  const nextState = {
    ...state,
    revision: state.revision + 1,
    passed,
    log,
    publicAction,
  };
  nextState.currentIndex = nextActiveIndex(nextState, state.currentIndex);
  return nextState;
}

function chooseBotCards(state: GameState, playerId: string): string[] | null {
  const hand = state.hands[playerId];
  const jokers = hand.filter((card) => card.rank === 13);
  const groups = new Map<number, Card[]>();
  for (const card of hand) {
    if (card.rank === 13) continue;
    groups.set(card.rank, [...(groups.get(card.rank) ?? []), card]);
  }

  if (!state.table) {
    if (jokers.length > 0) return [jokers[0].id];
    const ranks = [...groups.keys()].sort((a, b) => b - a);
    const rank = ranks[0];
    return rank ? groups.get(rank)!.map((card) => card.id) : null;
  }

  const targetCount = state.table.count;
  const ranks = [...groups.keys()]
    .filter((rank) => rank < state.table!.rank)
    .sort((a, b) => b - a);

  for (const rank of ranks) {
    const cards = groups.get(rank)!;
    if (cards.length + jokers.length < targetCount) continue;
    return [
      ...cards.slice(0, targetCount),
      ...jokers.slice(0, Math.max(0, targetCount - cards.length)),
    ].map((card) => card.id);
  }
  return null;
}

function PlayerSeat({
  player,
  handCount,
  score,
  isCurrent,
  isFinished,
  taxDirection,
  isFocusedTaxParty,
  showHandBacks,
  isHandRevealing,
  seatRef,
}: {
  player: Player;
  handCount: number;
  score: number;
  isCurrent: boolean;
  isFinished: boolean;
  taxDirection: TaxDirection | null;
  isFocusedTaxParty: boolean;
  showHandBacks: boolean;
  isHandRevealing: boolean;
  seatRef?: (node: HTMLElement | null) => void;
}) {
  return (
    <article
      ref={seatRef}
      className={`player-seat role-${player.role} ${isCurrent ? "is-current" : ""} ${
        isFinished ? "is-finished" : ""
      } ${taxDirection ? `is-tax-${taxDirection}` : ""} ${
        isFocusedTaxParty ? "is-focused-tax-party" : ""
      }`}
      aria-label={`${player.name}, ${ROLE_LABELS[player.role]}, 카드 ${handCount}장`}
    >
      <div className="player-avatar">
        <span>{player.monogram}</span>
        <i>{ROLE_MARKS[player.role]}</i>
      </div>
      <div className="player-copy">
        <strong>{player.name}</strong>
        <span>{ROLE_LABELS[player.role]}</span>
      </div>
      <div className="player-count">
        <b>{isFinished ? "완료" : handCount}</b>
        <span>{isFinished ? `${score}점` : "장"}</span>
      </div>
      {showHandBacks && !isFinished && (
        <div
          className={`opponent-card-stack ${
            isHandRevealing ? "is-revealing" : ""
          }`}
          aria-hidden="true"
        >
          {Array.from(
            { length: Math.min(4, Math.max(1, handCount)) },
            (_, index) => (
              <span
                key={`${player.id}-back-${index}`}
                className="opponent-hand-back"
                style={{ animationDelay: `${index * 70}ms` }}
              />
            ),
          )}
        </div>
      )}
      {isCurrent && <em className="turn-flag">차례</em>}
      {taxDirection && (
        <em className={`tax-seat-flag is-${taxDirection}`}>
          {taxDirection === "source" ? "보냄" : "받음"}
        </em>
      )}
    </article>
  );
}

function PlayingCard({
  card,
  selected,
  disabled,
  onClick,
  onDoubleClick,
  concealed = false,
  displayOnly = false,
  taxSourcePlaceholder = false,
}: {
  card: Card;
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  onDoubleClick?: () => void;
  concealed?: boolean;
  displayOnly?: boolean;
  taxSourcePlaceholder?: boolean;
}) {
  const isJoker = card.rank === 13;
  const [artLoaded, setArtLoaded] = useState(false);
  const artFile = isJoker ? "joker" : String(card.rank).padStart(2, "0");
  const content = (
    <>
      <img
        className="card-face-art"
        src={`/cards/${artFile}.webp?v=${CARD_ART_VERSION}`}
        alt=""
        aria-hidden="true"
        onLoad={() => setArtLoaded(true)}
        onError={(event) => {
          event.currentTarget.hidden = true;
          setArtLoaded(false);
        }}
      />
      <span className="generated-card-face">
        <span className="card-corner">{isJoker ? "★" : card.rank}</span>
        <span className="card-emblem">
          {isJoker ? "☾" : ROLE_MARKS[roleForCard(card.rank)]}
        </span>
        <strong>{isJoker ? "JESTER" : String(card.rank).padStart(2, "0")}</strong>
        <small>{RANK_NAMES[card.rank]}</small>
        <span className="card-corner card-corner-bottom">
          {isJoker ? "★" : card.rank}
        </span>
      </span>
    </>
  );

  if (displayOnly) {
    return (
      <div
        className={`playing-card ${isJoker ? "is-joker" : ""} ${
          artLoaded ? "has-art" : ""
        }`}
      >
        {content}
      </div>
    );
  }

  return (
    <button
      type="button"
      className={`playing-card ${isJoker ? "is-joker" : ""} ${
        selected ? "is-selected" : ""
      } ${artLoaded ? "has-art" : ""} ${
        taxSourcePlaceholder ? "is-tax-source-placeholder" : ""
      }`}
      disabled={disabled}
      aria-pressed={selected}
      aria-label={
        concealed
          ? "뒤집힌 카드"
          : `${RANK_NAMES[card.rank]} 카드 ${selected ? "선택됨" : ""}`
      }
      data-concealed={concealed || undefined}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      {content}
    </button>
  );
}

function roleForCard(rank: number): Role {
  if (rank === 1) return "great-dalmuti";
  if (rank === 2) return "lesser-dalmuti";
  if (rank >= 11) return "great-peon";
  if (rank >= 9) return "lesser-peon";
  return "merchant";
}

function PrivateCardBack() {
  return (
    <div className="private-card-back" aria-hidden="true">
      <span>♛</span>
      <i />
    </div>
  );
}

function TaxTransferLayer({
  routes,
  anchors,
  taxStage,
  animationKey,
}: {
  routes: TaxTransferRoute[];
  anchors: TaxAnchorMap;
  taxStage: TaxStage;
  animationKey: string;
}) {
  if (!anchors.midpoint) return null;

  return (
    <div
      key={animationKey}
      className="tax-transfer-layer"
      aria-live="polite"
      aria-label={
        taxStage === "tribute"
          ? "농노가 달무티에게 세금 카드를 전달하는 중"
          : "달무티가 농노에게 반환 카드를 전달하는 중"
      }
    >
      {routes.map((route) => {
        const from = anchors.players[route.from.id];
        const to = anchors.players[route.to.id];
        if (!from || !to || !anchors.midpoint) return null;

        const midpoint = route.reveal
          ? anchors.midpoint
          : {
              x: Math.round((from.x + to.x) / 2),
              y: Math.round(
                Math.min(
                  anchors.midpoint.y - 78,
                  Math.max(from.y, to.y) + 88 + route.routeIndex * 22,
                ),
              ),
            };
        const captionStyle = {
          "--mid-x": `${midpoint.x}px`,
          "--mid-y": `${midpoint.y}px`,
        } as React.CSSProperties;

        return (
          <section
            key={route.id}
            className={`tax-transfer-route ${
              route.reveal ? "is-revealed" : "is-concealed"
            }`}
            aria-label={`${subjectLabel(route.from.name)} ${route.to.name}에게 카드 ${route.cards.length}장을 전달하는 중`}
          >
            <div className="tax-route-caption" style={captionStyle}>
              <small>
                {taxStage === "tribute" ? "세금 납부" : "카드 반환"} ·{" "}
                {route.cards.length}장
              </small>
              <strong>
                <span>{route.from.name}</span>
                <i>→</i>
                <span>{route.to.name}</span>
              </strong>
              <em>
                {ROLE_LABELS[route.from.role]}에서 {ROLE_LABELS[route.to.role]}에게
              </em>
            </div>

            {route.cards.map((card, cardIndex) => {
              const centerOffset = cardIndex - (route.cards.length - 1) / 2;
              const cardStyle = {
                "--from-x": `${from.x}px`,
                "--from-y": `${from.y}px`,
                "--mid-x": `${midpoint.x}px`,
                "--mid-y": `${midpoint.y}px`,
                "--to-x": `${to.x}px`,
                "--to-y": `${to.y}px`,
                "--from-spread": `${centerOffset * 18}px`,
                "--mid-spread": `${centerOffset * (route.reveal ? 132 : 42)}px`,
                "--to-spread": `${centerOffset * 18}px`,
                "--tax-delay": `${cardIndex * 110}ms`,
                "--endpoint-scale": route.reveal ? 0.342 : 0.34,
                "--mid-scale": route.reveal ? 1 : 0.62,
              } as React.CSSProperties;

              return (
                <div
                  key={route.reveal ? card.id : `${route.id}-private-${cardIndex}`}
                  className={`tax-transfer-card ${
                    route.reveal ? "is-face-up" : "is-face-down"
                  }`}
                  style={cardStyle}
                >
                  {route.reveal ? (
                    <>
                      <PlayingCard card={card} displayOnly />
                      <span className="tax-card-identity">
                        {card.rank === 13 ? "광대" : `${card.rank}등급`} ·{" "}
                        {RANK_NAMES[card.rank]}
                      </span>
                    </>
                  ) : (
                    <PrivateCardBack />
                  )}
                </div>
              );
            })}
          </section>
        );
      })}
    </div>
  );
}

function PublicTurnActionLayer({
  action,
  anchors,
}: {
  action: PublicTurnAction;
  anchors: TaxAnchorMap;
}) {
  const from = anchors.players[action.player.id];
  const to = anchors.midpoint;
  if (!from || !to) return null;

  const playedSet = action.kind === "play" ? normalizedSet(action.cards) : null;
  const cardCount = action.cards.length;
  const expandedStep =
    cardCount <= 1 ? 0 : Math.min(112, 430 / Math.max(1, cardCount - 1));
  const mobileExpandedStep =
    cardCount <= 1 ? 0 : Math.min(70, 250 / Math.max(1, cardCount - 1));
  const delayStep =
    cardCount <= 1 ? 0 : Math.min(36, 100 / Math.max(1, cardCount - 1));
  const routeStyle = {
    "--from-x": `${from.x}px`,
    "--from-y": `${from.y}px`,
    "--to-x": `${to.x}px`,
    "--to-y": `${to.y}px`,
  } as React.CSSProperties;

  return (
    <div
      key={action.id}
      className={`public-turn-action-layer is-${action.kind}`}
      role="status"
      aria-live="polite"
      aria-label={
        action.kind === "play" && playedSet
          ? `${subjectLabel(action.player.name)} ${RANK_NAMES[playedSet.rank]} 카드 ${playedSet.count}장을 냈습니다`
          : `${subjectLabel(action.player.name)} 패스했습니다`
      }
    >
      {action.kind === "play" && playedSet ? (
        <>
          {action.cards.map((card, cardIndex) => {
            const centerOffset = cardIndex - (cardCount - 1) / 2;
            const cardStyle = {
              ...routeStyle,
              "--from-spread": `${centerOffset * 9}px`,
              "--expanded-x": `${centerOffset * expandedStep}px`,
              "--expanded-x-mobile": `${centerOffset * mobileExpandedStep}px`,
              "--settled-x": `${centerOffset * 46}px`,
              "--settled-x-mobile": `${centerOffset * 24}px`,
              "--settled-angle": `${(cardIndex - 1) * 3}deg`,
              "--action-delay": `${cardIndex * delayStep}ms`,
            } as React.CSSProperties;

            return (
              <div
                key={card.id}
                className="public-play-card"
                style={cardStyle}
                aria-hidden="true"
              >
                <PlayingCard card={card} displayOnly />
              </div>
            );
          })}
          <div className="public-play-caption" style={routeStyle} aria-hidden="true">
            <small>공개 플레이</small>
            <strong>{action.player.name}</strong>
            <span>
              {RANK_NAMES[playedSet.rank]}({playedSet.rank}) x {playedSet.count}장
            </span>
          </div>
        </>
      ) : (
        <div className="public-pass-badge" style={routeStyle} aria-hidden="true">
          <small>{ROLE_LABELS[action.player.role]}</small>
          <strong>PASS</strong>
          <span>{action.player.name} · 패스</span>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [game, setGame] = useState<GameState | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [showRules, setShowRules] = useState(false);
  const [taxAnchors, setTaxAnchors] = useState<TaxAnchorMap>({
    players: {},
    midpoint: null,
  });
  const tableColumnRef = useRef<HTMLDivElement | null>(null);
  const feltCenterRef = useRef<HTMLDivElement | null>(null);
  const humanAnchorRef = useRef<HTMLDivElement | null>(null);
  const seatRefs = useRef<Record<string, HTMLElement | null>>({});

  const currentPlayer = game?.players[game.currentIndex] ?? null;
  const humanHand = game?.hands[HUMAN_ID] ?? [];
  const humanFinished = Boolean(game?.finishOrder.includes(HUMAN_ID));
  const humanFinishRank = game?.finishOrder.indexOf(HUMAN_ID) ?? -1;
  const isHandConcealed =
    game?.phase === "ready" || game?.phase === "reveal-intro";
  const isHandRevealing = game?.phase === "hand-reveal";
  const pendingHumanTaxExchange =
    game?.phase === "taxation" && game.taxStage === "selection"
      ? game.taxExchanges.find(
          (exchange) =>
            exchange.nobleId === HUMAN_ID &&
            exchange.nobleGift.length < exchange.peonGift.length,
        ) ?? null
      : null;
  const humanTaxSelectionCount = pendingHumanTaxExchange?.peonGift.length ?? 0;
  const isHumanTaxSelecting = Boolean(pendingHumanTaxExchange);
  const isHumanTurn =
    game?.phase === "playing" &&
    currentPlayer?.id === HUMAN_ID &&
    !game.publicAction;
  const selectedCards = humanHand.filter((card) => selectedIds.includes(card.id));
  const selectedSet = normalizedSet(selectedCards);
  const selectedError = isHumanTaxSelecting
    ? selectedIds.length === humanTaxSelectionCount
      ? null
      : `반환할 카드 ${humanTaxSelectionCount}장을 선택하세요.`
    : game
      ? validationMessage(selectedCards, game.table)
      : "카드를 선택하세요.";
  const canPlay = Boolean(
    game &&
      isHumanTurn &&
      selectedIds.length > 0 &&
      selectedSet &&
      !selectedError,
  );
  const canConfirmTaxReturn =
    isHumanTaxSelecting &&
    selectedIds.length === humanTaxSelectionCount &&
    selectedCards.length === humanTaxSelectionCount;

  const orderedOpponents = useMemo(
    () => game?.players.filter((player) => !player.isHuman) ?? [],
    [game?.players],
  );

  const activeTaxRoutes = useMemo<TaxTransferRoute[]>(() => {
    if (
      !game ||
      game.phase !== "taxation" ||
      (game.taxStage !== "tribute" && game.taxStage !== "return")
    ) {
      return [];
    }
    const playersById = new Map(game.players.map((player) => [player.id, player]));
    const isTribute = game.taxStage === "tribute";

    return game.taxExchanges.flatMap((exchange, routeIndex) => {
      const fromId = isTribute ? exchange.peonId : exchange.nobleId;
      const toId = isTribute ? exchange.nobleId : exchange.peonId;
      const from = playersById.get(fromId);
      const to = playersById.get(toId);
      if (!from || !to) return [];

      return [
        {
          id: `${exchange.peonId}-${exchange.nobleId}-${game.taxStage}`,
          from,
          to,
          cards: isTribute ? exchange.peonGift : exchange.nobleGift,
          reveal: fromId === HUMAN_ID || toId === HUMAN_ID,
          routeIndex,
        },
      ];
    });
  }, [game]);

  const focusedTaxRoute =
    activeTaxRoutes.find((route) => route.reveal) ?? null;
  const humanTaxDirection: TaxDirection | null = focusedTaxRoute
    ? focusedTaxRoute.from.id === HUMAN_ID
      ? "source"
      : "destination"
    : null;
  const humanSourceIds = new Set(
    humanTaxDirection === "source"
      ? focusedTaxRoute?.cards.map((card) => card.id) ?? []
      : [],
  );

  useLayoutEffect(() => {
    if (!game) return;
    const root = tableColumnRef.current;
    const felt = feltCenterRef.current;
    if (!root || !felt) return;

    const measure = () => {
      const rootRect = root.getBoundingClientRect();
      const feltRect = felt.getBoundingClientRect();
      const players: Record<string, Point> = {};

      for (const player of game.players) {
        if (player.id === HUMAN_ID) continue;
        const seat = seatRefs.current[player.id];
        if (!seat) continue;
        const rect = seat.getBoundingClientRect();
        players[player.id] = {
          x: Math.round(rect.left + rect.width / 2 - rootRect.left),
          y: Math.round(rect.bottom - rootRect.top + 8),
        };
      }

      const humanAnchor = humanAnchorRef.current;
      if (humanAnchor) {
        const rect = humanAnchor.getBoundingClientRect();
        players[HUMAN_ID] = {
          x: Math.round(rect.left + rect.width / 2 - rootRect.left),
          y: Math.round(rect.top - rootRect.top + 42),
        };
      }

      const nextAnchors = {
        players,
        midpoint: {
          x: Math.round(feltRect.left + feltRect.width / 2 - rootRect.left),
          y: Math.round(feltRect.top + feltRect.height * 0.52 - rootRect.top),
        },
      };

      setTaxAnchors((current) => {
        const currentIds = Object.keys(current.players);
        const nextIds = Object.keys(nextAnchors.players);
        const unchanged =
          current.midpoint &&
          Math.abs(current.midpoint.x - nextAnchors.midpoint.x) < 0.5 &&
          Math.abs(current.midpoint.y - nextAnchors.midpoint.y) < 0.5 &&
          currentIds.length === nextIds.length &&
          nextIds.every((id) => {
            const previous = current.players[id];
            const next = nextAnchors.players[id];
            return (
              previous &&
              Math.abs(previous.x - next.x) < 0.5 &&
              Math.abs(previous.y - next.y) < 0.5
            );
          });

        return unchanged ? current : nextAnchors;
      });
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    window.addEventListener("resize", measure);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [game]);

  useEffect(() => {
    const actionId = game?.publicAction?.id;
    if (!actionId) return;

    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (!latest || latest.publicAction?.id !== actionId) return latest;
        return { ...latest, publicAction: null };
      });
    }, PUBLIC_ACTION_DURATION_MS);

    return () => window.clearTimeout(timer);
  }, [game?.publicAction?.id]);

  useEffect(() => {
    if (!game || game.phase !== "reveal-intro") return;
    const introRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "reveal-intro" ||
          latest.revision !== introRevision
        ) {
          return latest;
        }
        return {
          ...latest,
          phase: "hand-reveal",
          revision: latest.revision + 1,
          log: ["패를 공개합니다.", ...latest.log].slice(0, 12),
        };
      });
    }, REVEAL_INTRO_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (!game || game.phase !== "hand-reveal") return;
    const revealRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "hand-reveal" ||
          latest.revision !== revealRevision
        ) {
          return latest;
        }
        return advanceAfterHandReveal(latest);
      });
    }, HAND_REVEAL_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (!game || game.phase !== "tax-intro") return;
    const introRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "tax-intro" ||
          latest.revision !== introRevision
        ) {
          return latest;
        }
        return {
          ...latest,
          phase: "taxation",
          revision: latest.revision + 1,
          taxAnimationId: createTaxAnimationId(),
          log: ["세금 교환을 시작합니다.", ...latest.log].slice(0, 12),
        };
      });
    }, TAX_INTRO_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (!game || game.phase !== "play-intro") return;
    const introRevision = game.revision;
    const starter = game.players[game.currentIndex];
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "play-intro" ||
          latest.revision !== introRevision
        ) {
          return latest;
        }
        return {
          ...latest,
          phase: "playing",
          revision: latest.revision + 1,
          log: [
            `${subjectLabel(starter.name)} 먼저 시작합니다.`,
            ...latest.log,
          ].slice(0, 12),
        };
      });
    }, PLAY_INTRO_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "playing" ||
      game.publicAction ||
      currentPlayer?.isHuman
    ) {
      return;
    }
    const turnRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "playing" ||
          latest.revision !== turnRevision ||
          latest.publicAction
        ) {
          return latest;
        }
        const bot = latest.players[latest.currentIndex];
        if (bot.isHuman) return latest;
        const cards = chooseBotCards(latest, bot.id);
        return cards ? playCards(latest, bot.id, cards) : passTurn(latest, bot.id);
      });
    }, 760);
    return () => window.clearTimeout(timer);
  }, [currentPlayer?.id, currentPlayer?.isHuman, game]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "taxation" ||
      (game.taxStage !== "tribute" && game.taxStage !== "return")
    ) {
      return;
    }
    const taxRevision = game.revision;
    const taxStage = game.taxStage;
    const taxAnimationId = game.taxAnimationId;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "taxation" ||
          latest.revision !== taxRevision ||
          latest.taxStage !== taxStage ||
          latest.taxAnimationId !== taxAnimationId
        ) {
          return latest;
        }

        if (taxStage === "tribute") {
          return {
            ...latest,
            revision: latest.revision + 1,
            hands: latest.tributeHands ?? latest.hands,
            taxStage: "return",
          };
        }

        return {
          ...latest,
          phase: "play-intro",
          revision: latest.revision + 1,
          hands: latest.taxedHands ?? latest.hands,
          currentIndex: 0,
          taxStage: null,
          taxAnimationId: null,
          tributeHands: null,
          taxedHands: null,
          taxExchanges: [],
          log: [
            "세금 교환이 끝났습니다. 게임 시작을 준비합니다.",
            ...latest.log,
          ],
        };
      });
    }, TAX_STAGE_DURATION_MS);

    return () => window.clearTimeout(timer);
  }, [game]);

  const startGame = () => {
    const players = assignRoles(BASE_PLAYERS);
    const scores = Object.fromEntries(players.map((player) => [player.id, 0]));
    setSelectedIds([]);
    setGame(prepareRound(players, 1, scores, true, true));
  };

  const beginHostedGame = () => {
    setSelectedIds([]);
    setGame((current) => {
      if (!current || current.phase !== "ready") return current;
      return {
        ...current,
        phase: "reveal-intro",
        revision: current.revision + 1,
        log: ["방장이 패 공개를 시작했습니다.", ...current.log].slice(0, 12),
      };
    });
  };

  const resolveRevolution = (declare: boolean) => {
    setSelectedIds([]);
    setGame((current) => {
      if (!current || current.phase !== "revolution") return current;
      let players = current.players;
      const hands = current.hands;
      let log = current.log;
      let phase: Phase = "play-intro";
      let taxExchanges: TaxExchange[] = [];
      let taxStage: TaxStage | null = null;
      let taxAnimationId: string | null = null;
      let tributeHands: Record<string, Card[]> | null = null;
      let taxedHands: Record<string, Card[]> | null = null;
      const holder = current.players.find(
        (player) => player.id === current.revolutionHolder,
      );

      if (declare && holder?.role === "great-peon") {
        players = assignRoles([...current.players].reverse());
        log = ["당신의 대혁명으로 모든 계급이 뒤집혔습니다.", ...log];
      } else if (declare) {
        log = ["당신이 혁명을 선포했습니다. 이번 막의 세금은 없습니다.", ...log];
      } else {
        const taxed = applyTax(players, hands);
        phase = "tax-intro";
        taxExchanges = taxed.exchanges;
        taxStage = taxed.hands ? "tribute" : "selection";
        taxAnimationId = createTaxAnimationId();
        tributeHands = taxed.tributeHands;
        taxedHands = taxed.hands;
        log = ["당신은 혁명을 숨겼습니다.", ...taxed.notes, ...log];
      }

      return {
        ...current,
        phase,
        revision: current.revision + 1,
        players,
        hands,
        log,
        revolutionHolder: null,
        taxExchanges,
        taxStage,
        taxAnimationId,
        tributeHands,
        taxedHands,
        currentIndex: 0,
      };
    });
  };

  const toggleCard = (cardId: string) => {
    if (!isHumanTurn && !isHumanTaxSelecting) return;
    setSelectedIds((current) => {
      if (current.includes(cardId)) {
        return current.filter((id) => id !== cardId);
      }
      if (
        isHumanTaxSelecting &&
        current.length >= humanTaxSelectionCount
      ) {
        return current;
      }
      return [...current, cardId];
    });
  };

  const selectAllOfRank = (card: Card) => {
    if (!isHumanTurn) return;
    const sameRankIds =
      card.rank === 13
        ? [card.id]
        : humanHand
            .filter((candidate) => candidate.rank === card.rank)
            .map((candidate) => candidate.id);
    setSelectedIds((current) =>
      toggleWholeRankSelection(current, sameRankIds),
    );
  };

  const confirmTaxReturn = () => {
    if (!canConfirmTaxReturn) return;
    const chosenIds = [...selectedIds];
    setSelectedIds([]);
    setGame((current) => {
      if (
        !current ||
        current.phase !== "taxation" ||
        current.taxStage !== "selection" ||
        !current.tributeHands
      ) {
        return current;
      }

      const exchangeIndex = current.taxExchanges.findIndex(
        (exchange) =>
          exchange.nobleId === HUMAN_ID &&
          exchange.nobleGift.length < exchange.peonGift.length,
      );
      if (exchangeIndex < 0) return current;

      const exchange = current.taxExchanges[exchangeIndex];
      const chosenCards = current.hands[HUMAN_ID].filter((card) =>
        chosenIds.includes(card.id),
      );
      if (chosenCards.length !== exchange.peonGift.length) return current;

      const taxExchanges = current.taxExchanges.map((candidate, index) =>
        index === exchangeIndex
          ? { ...candidate, nobleGift: chosenCards }
          : candidate,
      );

      return {
        ...current,
        revision: current.revision + 1,
        taxExchanges,
        taxStage: "tribute",
        taxAnimationId: createTaxAnimationId(),
        taxedHands: settleTaxHands(current.tributeHands, taxExchanges),
        log: [
          `반환할 카드 ${chosenCards.length}장을 선택했습니다.`,
          ...current.log,
        ].slice(0, 12),
      };
    });
  };

  const playSelected = () => {
    if (!canPlay) return;
    const cardIds = [...selectedIds];
    setSelectedIds([]);
    setGame((current) =>
      current ? playCards(current, HUMAN_ID, cardIds) : current,
    );
  };

  const pass = () => {
    if (!game || !isHumanTurn || !game.table) return;
    setSelectedIds([]);
    setGame((current) =>
      current ? passTurn(current, HUMAN_ID) : current,
    );
  };

  const nextRound = () => {
    if (!game || game.phase !== "round-end" || game.publicAction) return;
    const ordered = game.finishOrder.map(
      (id) => game.players.find((player) => player.id === id)!,
    );
    setSelectedIds([]);
    setGame(prepareRound(ordered, game.round + 1, game.scores, false, true));
  };

  const publicPlayedSet =
    game?.publicAction?.kind === "play"
      ? normalizedSet(game.publicAction.cards)
      : null;
  const visibleTable = game?.publicAction
    ? game.publicAction.previousTable
    : game?.table ?? null;

  const turnMessage = !game
    ? "왕실의 자리가 비어 있습니다"
    : game.publicAction
      ? game.publicAction.kind === "play" && publicPlayedSet
        ? `${subjectLabel(game.publicAction.player.name)} ${RANK_NAMES[publicPlayedSet.rank]} 카드 ${publicPlayedSet.count}장을 내는 중`
        : `${subjectLabel(game.publicAction.player.name)} 패스했습니다`
      : game.phase === "ready"
        ? "방장이 PLAY를 누르면 패를 공개합니다"
        : game.phase === "reveal-intro"
          ? "모든 플레이어의 패 공개를 준비합니다"
          : game.phase === "hand-reveal"
            ? "각 플레이어가 자신의 패를 확인하는 중"
            : game.phase === "tax-intro"
              ? "세금 교환을 준비합니다"
              : game.phase === "play-intro"
                ? `${subjectLabel(currentPlayer?.name ?? "")} 먼저 시작합니다`
                : game.phase === "round-end"
                  ? "새로운 계급이 결정되었습니다"
                  : game.phase === "revolution"
                    ? "두 광대가 혁명을 기다립니다"
                    : game.phase === "taxation"
                      ? game.taxStage === "selection"
                        ? `농노에게 돌려줄 카드 ${humanTaxSelectionCount}장을 선택하세요`
                        : focusedTaxRoute
                          ? `${focusedTaxRoute.from.name} → ${focusedTaxRoute.to.name} · 카드 ${focusedTaxRoute.cards.length}장 전달 중`
                          : "당사자끼리 비공개 세금 교환 중"
                      : isHumanTurn
                        ? game.table
                          ? `${game.table.rank}보다 낮은 숫자의 카드 ${game.table.count}장을 내세요`
                          : "새로운 묶음을 시작하세요"
                        : `${currentPlayer?.name}의 선택을 기다리는 중`;

  const tablePreview = visibleTable?.cards ?? [];

  return (
    <main className="game-shell">
      <div className="paper-grain" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <span className="brand-seal" aria-hidden="true" />
          <div>
            <strong>DALMUTI</strong>
            <small>DCLab의 계급전</small>
          </div>
        </div>

        <div className="round-chip" aria-label="게임 정보">
          <span>제 {game?.round ?? 1}막</span>
          <i />
          <span>5인</span>
        </div>

        <nav className="top-actions" aria-label="게임 메뉴">
          <button type="button" onClick={() => setShowRules(true)}>
            규칙
          </button>
          <button type="button" onClick={startGame}>
            새 게임
          </button>
        </nav>
      </header>

      <section className="game-stage" aria-label="달무티 게임 테이블">
        <aside className="score-rail">
          <div className="rail-heading">
            <span>랩실 서열</span>
            <small>현재 계급</small>
          </div>
          <ol>
            {(game?.players ?? assignRoles(BASE_PLAYERS)).map((player) => (
              <li key={player.id} className={player.id === HUMAN_ID ? "is-you" : ""}>
                <span>{ROLE_MARKS[player.role]}</span>
                <div>
                  <b>{player.name}</b>
                  <small>{ROLE_LABELS[player.role]}</small>
                </div>
                <em>{game?.scores[player.id] ?? 0}</em>
              </li>
            ))}
          </ol>
          <div className="rail-note">
            <span>계급의 법칙</span>
            <p>숫자가 낮을수록 강합니다. 같은 장수로 더 강하게 맞서세요.</p>
          </div>
        </aside>

        <div className="table-column" ref={tableColumnRef}>
          <div
            className={`opponent-row ${
              isHandRevealing ? "is-revealing" : ""
            }`}
          >
            {(orderedOpponents.length
              ? orderedOpponents
              : assignRoles(BASE_PLAYERS).filter((player) => !player.isHuman)
            ).map((player) => {
              const route = activeTaxRoutes.find(
                (candidate) =>
                  candidate.from.id === player.id || candidate.to.id === player.id,
              );
              const taxDirection: TaxDirection | null = route
                ? route.from.id === player.id
                  ? "source"
                  : "destination"
                : null;

              return (
                <PlayerSeat
                  key={player.id}
                  player={player}
                  handCount={game?.hands[player.id]?.length ?? 16}
                  score={game?.scores[player.id] ?? 0}
                  isCurrent={
                    game?.phase === "playing" &&
                    !game.publicAction &&
                    currentPlayer?.id === player.id
                  }
                  isFinished={Boolean(game?.finishOrder.includes(player.id))}
                  taxDirection={taxDirection}
                  isFocusedTaxParty={Boolean(route?.reveal)}
                  showHandBacks={Boolean(game)}
                  isHandRevealing={isHandRevealing}
                  seatRef={(node) => {
                    seatRefs.current[player.id] = node;
                  }}
                />
              );
            })}
          </div>

          {game?.phase === "taxation" &&
            (game.taxStage === "tribute" || game.taxStage === "return") &&
            game.taxAnimationId &&
            activeTaxRoutes.length > 0 && (
              <TaxTransferLayer
                routes={activeTaxRoutes}
                anchors={taxAnchors}
                taxStage={game.taxStage}
                animationKey={`${game.taxAnimationId}-${game.taxStage}`}
              />
            )}

          {game?.publicAction && (
            <PublicTurnActionLayer
              action={game.publicAction}
              anchors={taxAnchors}
            />
          )}

          <div className="felt-table" ref={feltCenterRef}>
            <div className="table-ring" aria-hidden="true">
              <span>♜</span>
              <i />
              <span>♞</span>
              <i />
              <span>♝</span>
            </div>

            <section
              className={`play-area ${
                game?.publicAction?.kind === "play" ? "is-resolving-play" : ""
              }`}
              aria-live="polite"
            >
              {game?.phase === "ready" ? (
                <div className="phase-intro is-ready">
                  <small>HOST CONTROL</small>
                  <strong>준비가 끝났습니다</strong>
                  <span>방장이 시작 신호를 보내면 모두의 패를 공개합니다</span>
                  <button
                    type="button"
                    className="ready-play-button"
                    onClick={beginHostedGame}
                  >
                    <i>▶</i>
                    PLAY
                  </button>
                </div>
              ) : game?.phase === "reveal-intro" ? (
                <div
                  key={`reveal-intro-${game.revision}`}
                  className="hand-reveal-intro"
                >
                  <small>HAND REVEAL</small>
                  <strong>패를 공개합니다</strong>
                  <span>모든 플레이어가 동시에 자신의 패를 확인합니다</span>
                </div>
              ) : game?.phase === "hand-reveal" ? (
                <div className="private-tax-state">
                  <span className="play-kicker">HAND REVEAL</span>
                  <strong>패를 확인하는 중</strong>
                  <small>패 공개가 끝나면 세금 교환을 시작합니다</small>
                </div>
              ) : game?.phase === "tax-intro" ? (
                <div
                  key={`tax-intro-${game.revision}`}
                  className="phase-intro is-tax"
                >
                  <small>TRIBUTE PHASE</small>
                  <strong>세금 교환</strong>
                  <span>계급에 따른 카드 교환을 시작합니다</span>
                </div>
              ) : game?.phase === "play-intro" ? (
                <div
                  key={`play-intro-${game.revision}`}
                  className="phase-intro is-play"
                >
                  <small>ROUND {game.round}</small>
                  <strong>게임 시작</strong>
                  <span>
                    {subjectLabel(game.players[game.currentIndex].name)} 먼저
                    시작합니다
                  </span>
                </div>
              ) : game?.phase === "taxation" ? (
                game.taxStage === "selection" ? (
                  <div className="tax-selection-state">
                    <span className="play-kicker">RETURN CARD</span>
                    <strong>농노에게 돌려줄 카드를 고르세요</strong>
                    <small>
                      내 원래 손패에서 원하는 카드 {humanTaxSelectionCount}장을
                      선택합니다
                    </small>
                    <span className="tax-selection-progress">
                      {selectedIds.length} / {humanTaxSelectionCount}
                    </span>
                  </div>
                ) : (
                  <div className="private-tax-state">
                    <span className="play-kicker">
                      {game.taxStage === "tribute" ? "TRIBUTE" : "RETURN"}
                    </span>
                    <strong>
                      {focusedTaxRoute
                        ? game.taxStage === "tribute"
                          ? "농노의 세금 카드 전달"
                          : "달무티의 반환 카드 전달"
                        : "비공개 카드 전달 중"}
                    </strong>
                    <small>
                      {focusedTaxRoute
                        ? "카드가 중앙에서 확대된 뒤 상대 좌석으로 이동합니다"
                        : "당사자가 아닌 플레이어에게 카드 정보는 공개되지 않습니다"}
                    </small>
                  </div>
                )
              ) : (
                <>
                  <span className="play-kicker">
                    {visibleTable ? "마지막으로 놓인 패" : "비어 있는 판"}
                  </span>
                  <div className={`table-cards ${tablePreview.length ? "" : "is-empty"}`}>
                    {tablePreview.length ? (
                      tablePreview.map((card, index) => (
                        <div
                          key={card.id}
                          className="table-card-wrap"
                          style={{ "--card-index": index } as React.CSSProperties}
                        >
                          <PlayingCard card={card} displayOnly />
                        </div>
                      ))
                    ) : (
                      <div className="empty-pile">
                        <span>♛</span>
                        <small>선 플레이어가<br />새 묶음을 냅니다</small>
                      </div>
                    )}
                  </div>
                  {visibleTable && (
                    <strong className="table-callout">
                      {RANK_NAMES[visibleTable.rank]}({visibleTable.rank}) x{" "}
                      {visibleTable.count}장
                    </strong>
                  )}
                </>
              )}
              <p>{turnMessage}</p>
            </section>
          </div>

          <section
            className={`human-zone ${isHumanTurn ? "is-active" : ""} ${
              game?.phase === "taxation" ? "is-taxing" : ""
            } ${humanTaxDirection ? `is-tax-${humanTaxDirection}` : ""} ${
              focusedTaxRoute ? "is-focused-tax-party" : ""
            } ${isHumanTaxSelecting ? "is-tax-selecting" : ""} ${
              humanFinished ? "is-finished" : ""
            }`}
          >
            <div className="human-status">
              <div className="human-avatar">나</div>
              <div>
                <span>{game ? ROLE_LABELS[game.players.find((p) => p.id === HUMAN_ID)!.role] : "상인"}</span>
                <strong>
                  {humanFinished
                    ? `이번 막 완료 · ${humanFinishRank + 1}위`
                    : isHumanTaxSelecting
                      ? `반환 카드 ${humanTaxSelectionCount}장을 선택하세요`
                    : isHandConcealed
                      ? "PLAY 전까지 패가 뒤집혀 있습니다"
                    : isHandRevealing
                      ? "패를 공개하는 중"
                    : game?.phase === "taxation" && focusedTaxRoute
                    ? humanTaxDirection === "source"
                      ? `카드 ${focusedTaxRoute.cards.length}장을 보내는 중`
                      : `카드 ${focusedTaxRoute.cards.length}장을 받는 중`
                    : game?.publicAction?.player.id === HUMAN_ID
                      ? game.publicAction.kind === "play"
                        ? "카드를 내는 중"
                        : "패스하는 중"
                    : isHumanTurn
                      ? "당신의 차례"
                      : "나의 손패"}
                </strong>
              </div>
              <em>{humanFinished ? "완료" : `${game ? humanHand.length : 16}장`}</em>
              {humanTaxDirection && (
                <i className={`human-tax-flag is-${humanTaxDirection}`}>
                  {humanTaxDirection === "source" ? "보냄" : "받음"}
                </i>
              )}
            </div>

            <div className="hand-wrap" ref={humanAnchorRef}>
              <div
                className={`hand ${
                  isHandConcealed
                    ? "is-concealed"
                    : isHandRevealing
                      ? "is-revealing"
                      : ""
                }`}
                data-testid="player-hand"
              >
                {humanFinished ? (
                  <div
                    className="finished-hand-state"
                    role="status"
                    aria-live="polite"
                  >
                    <span>✓</span>
                    <strong>모든 카드를 냈습니다</strong>
                    <small>이번 막을 {humanFinishRank + 1}위로 마쳤습니다</small>
                  </div>
                ) : (
                  (game
                    ? humanHand
                    : [
                        { id: "demo-12", rank: 12 },
                        { id: "demo-11", rank: 11 },
                        { id: "demo-10", rank: 10 },
                        { id: "demo-9", rank: 9 },
                        { id: "demo-8", rank: 8 },
                        { id: "demo-7", rank: 7 },
                        { id: "demo-6", rank: 6 },
                        { id: "demo-5", rank: 5 },
                      ]
                  ).map((card) => (
                    <PlayingCard
                      key={card.id}
                      card={card}
                      selected={selectedIds.includes(card.id)}
                      disabled={
                        !game || (!isHumanTurn && !isHumanTaxSelecting)
                      }
                      concealed={isHandConcealed}
                      taxSourcePlaceholder={humanSourceIds.has(card.id)}
                      onClick={() => toggleCard(card.id)}
                      onDoubleClick={() => selectAllOfRank(card)}
                    />
                  ))
                )}
              </div>
            </div>

            <div className="turn-controls">
              {humanFinished ? (
                <div className="selection-hint is-valid finished-control-state">
                  <span>이번 막 완료</span>
                  <small>다른 플레이어가 순위를 결정하는 중입니다</small>
                </div>
              ) : isHumanTaxSelecting ? (
                <>
                  <div
                    className={`selection-hint ${
                      selectedError ? "has-error" : "is-valid"
                    }`}
                  >
                    <span>
                      {selectedIds.length
                        ? `${selectedIds.length}장 선택`
                        : "돌려줄 카드를 선택하세요"}
                    </span>
                    <small>
                      {selectedError ??
                        `선택한 ${humanTaxSelectionCount}장은 농노에게 전달됩니다`}
                    </small>
                  </div>
                  <button
                    type="button"
                    className="play-button tax-confirm-button"
                    disabled={!canConfirmTaxReturn}
                    onClick={confirmTaxReturn}
                  >
                    반환 카드 확정
                    <span>→</span>
                  </button>
                </>
              ) : (
                <>
                  <div className={`selection-hint ${selectedError ? "has-error" : "is-valid"}`}>
                    <span>{selectedIds.length ? `${selectedIds.length}장 선택` : "카드를 선택하세요"}</span>
                    <small>
                      {selectedIds.length
                        ? selectedError ?? `${selectedSet?.rank}등급 묶음 · 낼 수 있습니다`
                        : game?.table
                          ? `현재 ${game.table.count}장 묶음 · 더블클릭하면 같은 숫자 전체 선택`
                          : "한 번 클릭: 개별 선택 · 더블클릭: 같은 숫자 전체 선택"}
                    </small>
                  </div>
                  <button
                    type="button"
                    className="pass-button"
                    disabled={!isHumanTurn || !game?.table}
                    onClick={pass}
                  >
                    패스
                  </button>
                  <button
                    type="button"
                    className="play-button"
                    disabled={!canPlay}
                    onClick={playSelected}
                  >
                    패 내기
                    <span>↗</span>
                  </button>
                </>
              )}
            </div>
          </section>
        </div>

        <aside className="history-rail">
          <div className="rail-heading">
            <span>기록</span>
            <small>최근 행동</small>
          </div>
          <ul>
            {(game?.log ?? [
              "빠른 대전을 시작해 왕관을 차지하세요.",
              "첫 판의 계급은 이미 정해져 있습니다.",
              "방장의 PLAY 이후 세금과 게임이 진행됩니다.",
            ]).map((entry, index) => (
              <li key={`${entry}-${index}`}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <p>{entry}</p>
              </li>
            ))}
          </ul>
          <div className="legend">
            <div><span className="legend-dot strongest" />1은 가장 강함</div>
            <div><span className="legend-dot weakest" />12는 가장 약함</div>
            <div><span className="legend-dot joker" />광대는 만능 카드</div>
          </div>
        </aside>
      </section>

      {!game && (
        <div className="welcome-layer">
          <section className="welcome-card" role="dialog" aria-labelledby="welcome-title">
            <div className="welcome-crown">♛</div>
            <span className="eyebrow">PLAYABLE PROTOTYPE · 5 PLAYERS</span>
            <h1 id="welcome-title">왕관은<br /><em>공평하지 않다</em></h1>
            <p>
              약한 패부터 영리하게 털어내고, 계급을 뒤집으세요.
              네 명의 AI와 바로 한 판을 시작합니다.
            </p>
            <div className="welcome-features">
              <span>80장 정식 덱</span>
              <span>세금과 혁명</span>
              <span>연속 라운드</span>
            </div>
            <button type="button" className="start-button" onClick={startGame}>
              <span>5인 빠른 대전</span>
              <i>게임 시작</i>
              <b>→</b>
            </button>
            <a className="online-start-link" href="/online">
              <span>친구들과 온라인</span>
              <i>초대 코드로 4~8인 방 만들기</i>
              <b>↗</b>
            </a>
            <small className="welcome-note">
              혼자 연습하거나, 온라인 방을 만들어 함께 플레이하세요.
            </small>
          </section>
        </div>
      )}

      {game?.phase === "revolution" && (
        <div className="modal-layer">
          <section className="decision-card" role="dialog" aria-labelledby="revolution-title">
            <span className="decision-icon">☾ ☾</span>
            <small>두 광대가 한 손에 모였습니다</small>
            <h2 id="revolution-title">혁명을 선포하시겠습니까?</h2>
            <p>
              혁명을 선포하면 이번 막의 세금이 사라집니다.
              대 농노라면 모든 계급까지 뒤집힙니다.
            </p>
            <div>
              <button type="button" className="secondary-button" onClick={() => resolveRevolution(false)}>
                조용히 지나간다
              </button>
              <button type="button" className="play-button" onClick={() => resolveRevolution(true)}>
                혁명 선포
              </button>
            </div>
          </section>
        </div>
      )}

      {game?.phase === "round-end" && !game.publicAction && (
        <div className="modal-layer">
          <section className="result-card" role="dialog" aria-labelledby="result-title">
            <span className="eyebrow">THE COURT HAS SPOKEN</span>
            <h2 id="result-title">제 {game.round}막의 새로운 계급</h2>
            <ol>
              {game.finishOrder.map((id, index) => {
                const player = game.players.find((candidate) => candidate.id === id)!;
                const nextRole = roleForIndex(index, game.players.length);
                return (
                  <li key={id} className={id === HUMAN_ID ? "is-you" : ""}>
                    <span>{index + 1}</span>
                    <div>
                      <b>{player.name}</b>
                      <small>{ROLE_LABELS[nextRole]}</small>
                    </div>
                    <em>{game.scores[id]}점</em>
                  </li>
                );
              })}
            </ol>
            <button type="button" className="start-button" onClick={nextRound}>
              <span>다음 막으로</span>
              <i>새 계급으로 카드 배분</i>
              <b>→</b>
            </button>
          </section>
        </div>
      )}

      {showRules && (
        <div className="modal-layer">
          <section className="rules-card" role="dialog" aria-labelledby="rules-title">
            <button
              type="button"
              className="close-button"
              aria-label="규칙 닫기"
              onClick={() => setShowRules(false)}
            >
              ×
            </button>
            <span className="eyebrow">HOW TO PLAY</span>
            <h2 id="rules-title">세 가지만 기억하세요</h2>
            <div className="rules-grid">
              <article>
                <span>01</span>
                <h3>같은 숫자를 묶기</h3>
                <p>한 장 또는 같은 숫자 여러 장을 한 번에 냅니다.</p>
              </article>
              <article>
                <span>02</span>
                <h3>낮은 숫자로 이기기</h3>
                <p>앞사람과 같은 장수이면서 더 낮은 숫자만 낼 수 있습니다.</p>
              </article>
              <article>
                <span>03</span>
                <h3>가장 먼저 털기</h3>
                <p>손패를 먼저 비울수록 다음 막의 계급이 높아집니다.</p>
              </article>
            </div>
            <div className="rule-detail">
              광대는 다른 카드와 함께 내면 그 숫자로 변하고, 단독으로는 가장 약한
              13입니다. 이 게임의 하우스 룰에서는 세금 계산에 한해 광대를 가장 강한
              카드로 취급합니다. 농노는 광대를 먼저 바치고, 달무티는 일반 카드부터
              돌려줍니다.
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
