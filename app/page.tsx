"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  selectDalmutiReturnCards,
  selectPeonTaxCards,
} from "@/lib/taxation";
import { rankedDealCounts } from "@/lib/dealing";
import { resolveQuickDalmutiAutoPass } from "@/lib/quick-dalmuti";
import { scoreChipCount } from "@/lib/score-chips";
import { toggleWholeRankSelection } from "@/lib/selection";

type Role =
  | "great-dalmuti"
  | "lesser-dalmuti"
  | "merchant"
  | "lesser-peon"
  | "great-peon";

type Phase =
  | "ready"
  | "rank-intro"
  | "rank-selection"
  | "rank-reveal"
  | "rank-confirm"
  | "reveal-intro"
  | "hand-reveal"
  | "tax-intro"
  | "playing"
  | "play-intro"
  | "revolution"
  | "revolution-intro"
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
  automatic?: boolean;
  automaticReason?: "timeout" | "insufficient-cards";
  autoPassedPlayerIds?: string[];
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

type OpeningRankSelection = {
  cards: number[];
  selectedBy: Array<string | null>;
  pickOrder: string[];
  countdown: number;
};

type RevolutionAnnouncement = {
  id: string;
  playerId: string;
  playerName: string;
  kind: "revolution" | "great-revolution";
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
  openingRankSelection: OpeningRankSelection | null;
  revolutionAnnouncement: RevolutionAnnouncement | null;
};

const HUMAN_ID = "you";
const RANK_COUNTDOWN_STEP_MS = 1100;
const BOT_RANK_PICK_DELAY_MS = 750;
const RANK_ALL_SELECTED_PAUSE_MS = 1500;
const RANK_REVEAL_DURATION_MS = 3400;
const RANK_CONFIRM_DURATION_MS = 2600;
const TAX_STAGE_DURATION_MS = 6000;
const REVEAL_INTRO_DURATION_MS = 2400;
const HAND_REVEAL_DURATION_MS = 1400;
const TAX_INTRO_DURATION_MS = 2400;
const PLAY_INTRO_DURATION_MS = 2600;
const REVOLUTION_INTRO_DURATION_MS = 3300;
const PUBLIC_ACTION_DURATION_MS = 2250;
const PASS_ACTION_DURATION_MS = 1500;
const DALMUTI_ACTION_DURATION_MS = 3300;
const FAST_PUBLIC_ACTION_DURATION_MS = 420;
const FAST_PASS_ACTION_DURATION_MS = 280;
const FAST_DALMUTI_ACTION_DURATION_MS = 700;
const FAST_BOT_THINK_MS = 120;
const TURN_LIMIT_MS = 30_000;
const RANK_TRANSITION_DURATION_MS = 2300;
const RANK_RESULT_REVEAL_DELAY_MS = 280;
const CARD_ART_VERSION = "2026-07-24-2x";

function createTaxAnimationId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createPublicActionId(kind: PublicTurnAction["kind"]) {
  return `${kind}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createRevolutionAnnouncement(
  player: Player,
  kind: RevolutionAnnouncement["kind"],
): RevolutionAnnouncement {
  return {
    id: `${kind}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    playerId: player.id,
    playerName: player.name,
    kind,
  };
}

const BASE_PLAYERS: Omit<Player, "role">[] = [
  { id: HUMAN_ID, name: "나", monogram: "나", isHuman: true },
  { id: "marco", name: "마르코", monogram: "마", isHuman: false },
  { id: "luna", name: "루나", monogram: "루", isHuman: false },
  { id: "tobias", name: "토비아스", monogram: "토", isHuman: false },
  { id: "seraphine", name: "세라핀", monogram: "세", isHuman: false },
];

const ROLE_LABELS: Record<Role, string> = {
  "great-dalmuti": "달무티",
  "lesser-dalmuti": "총리대신",
  merchant: "상인",
  "lesser-peon": "소작농",
  "great-peon": "농노",
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

function seatPosition(rankIndex: number, total: number): React.CSSProperties {
  const angle =
    total <= 1 ? 270 : 150 + (240 * rankIndex) / Math.max(1, total - 1);
  const radians = (angle * Math.PI) / 180;
  return {
    "--seat-x": `${50 + Math.cos(radians) * 42}%`,
    "--seat-y": `${46 + Math.sin(radians) * 34}%`,
    "--seat-grid-column": rankIndex + 1,
    "--seat-grid-row": 1,
  } as React.CSSProperties;
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

function createOpeningRound(
  basePlayers: Omit<Player, "role">[],
  scores: Record<string, number>,
): GameState {
  const players = assignRoles(basePlayers);
  return {
    phase: "ready",
    round: 1,
    revision: 0,
    players,
    hands: Object.fromEntries(players.map((player) => [player.id, []])),
    scores,
    currentIndex: 0,
    table: null,
    lastPlayedId: null,
    passed: [],
    finishOrder: [],
    log: [
      "제 1막은 계급 카드를 직접 골라 서열을 정합니다.",
      "방장의 PLAY를 기다리고 있습니다.",
    ],
    revolutionHolder: null,
    taxExchanges: [],
    taxStage: null,
    taxAnimationId: null,
    tributeHands: null,
    taxedHands: null,
    publicAction: null,
    openingRankSelection: null,
    revolutionAnnouncement: null,
  };
}

function selectedOpeningRank(
  selection: OpeningRankSelection | null,
  playerId: string,
): number | null {
  if (!selection) return null;
  const cardIndex = selection.selectedBy.findIndex(
    (selectedPlayerId) => selectedPlayerId === playerId,
  );
  return cardIndex >= 0 ? selection.cards[cardIndex] : null;
}

function completeOpeningRankSelection(state: GameState): GameState {
  const selection = state.openingRankSelection;
  if (!selection || selection.selectedBy.some((playerId) => !playerId)) {
    return state;
  }

  const rankByPlayer = new Map(
    selection.selectedBy.map((playerId, cardIndex) => [
      playerId!,
      selection.cards[cardIndex],
    ]),
  );
  const players = assignRoles(
    [...state.players].sort(
      (left, right) =>
        (rankByPlayer.get(left.id) ?? Number.MAX_SAFE_INTEGER) -
        (rankByPlayer.get(right.id) ?? Number.MAX_SAFE_INTEGER),
    ),
  );
  const hands = deal(players);
  const holder = players.find(
    (player) => hands[player.id].filter((card) => card.rank === 13).length === 2,
  );
  const rankLog = players.map(
    (player) =>
      `${player.name} · ${RANK_NAMES[rankByPlayer.get(player.id)!]}(${rankByPlayer.get(player.id)})`,
  );

  return {
    ...state,
    phase: "reveal-intro",
    revision: state.revision + 1,
    players,
    hands,
    currentIndex: 0,
    revolutionHolder: holder?.id ?? null,
    openingRankSelection: null,
    log: [
      "계급 선택이 끝났습니다. 확정된 서열대로 패를 나눴습니다.",
      ...rankLog,
      ...state.log,
    ].slice(0, 12),
  };
}

function autoAssignFinalOpeningRankCard(state: GameState): GameState {
  if (state.phase !== "rank-selection" || !state.openingRankSelection) {
    return state;
  }

  const availableIndexes = state.openingRankSelection.selectedBy
    .map((selectedPlayerId, index) => (selectedPlayerId ? -1 : index))
    .filter((index) => index >= 0);
  const unassignedPlayers = state.players.filter(
    (player) => !state.openingRankSelection!.pickOrder.includes(player.id),
  );
  if (availableIndexes.length !== 1 || unassignedPlayers.length !== 1) {
    return state;
  }

  const cardIndex = availableIndexes[0];
  const player = unassignedPlayers[0];
  const selectedBy = [...state.openingRankSelection.selectedBy];
  selectedBy[cardIndex] = player.id;

  return {
    ...state,
    revision: state.revision + 1,
    openingRankSelection: {
      ...state.openingRankSelection,
      selectedBy,
      pickOrder: [...state.openingRankSelection.pickOrder, player.id],
    },
    log: [
      `${subjectLabel(player.name)} 남은 계급 카드를 자동으로 받았습니다.`,
      ...state.log,
    ].slice(0, 12),
  };
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
  let revolutionAnnouncement: RevolutionAnnouncement | null = null;
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
      revolutionAnnouncement = createRevolutionAnnouncement(
        holder,
        "great-revolution",
      );
    } else {
      log = [
        `${subjectLabel(holder.name)} 혁명을 선포해 세금이 취소되었습니다.`,
        ...log,
      ];
      revolutionAnnouncement = createRevolutionAnnouncement(holder, "revolution");
    }
    phase = "revolution-intro";
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
    openingRankSelection: null,
    revolutionAnnouncement,
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
      phase: "revolution-intro",
      revision: state.revision + 1,
      players,
      currentIndex: 0,
      revolutionHolder: null,
      revolutionAnnouncement: createRevolutionAnnouncement(
        holder,
        isGreatRevolution ? "great-revolution" : "revolution",
      ),
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
  const hands = { ...state.hands, [playerId]: removeCards(hand, cardIds) };
  const dalmutiResolution =
    set.rank === 1
      ? resolveQuickDalmutiAutoPass(
          state.players,
          hands,
          playerId,
          state.currentIndex,
        )
      : null;
  const publicAction: PublicTurnAction = {
    id: createPublicActionId("play"),
    kind: "play",
    player: current,
    cards: selected,
    previousTable: state.table,
    autoPassedPlayerIds: dalmutiResolution?.autoPassedPlayerIds,
  };
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

  if (dalmutiResolution?.autoPassedPlayerIds.length) {
    const autoPassedNames = dalmutiResolution.autoPassedPlayerIds
      .map(
        (id) => state.players.find((player) => player.id === id)?.name ?? id,
      )
      .join(", ");
    log.unshift(
      `달무티로 ${autoPassedNames}이(가) 자동 패스했습니다.`,
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

  if (dalmutiResolution) {
    nextState.table = null;
    nextState.passed = [];
    nextState.currentIndex = dalmutiResolution.nextPlayerIndex;
    nextState.log = [
      hands[playerId].length > 0
        ? `${subjectLabel(current.name)} 달무티를 내고 새로운 묶음을 시작합니다.`
        : "달무티로 판이 비워졌습니다. 다음 활성 플레이어가 새로운 묶음을 시작합니다.",
      ...log,
    ].slice(0, 12);
    return nextState;
  }

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

function timeoutPassTurn(state: GameState, playerId: string): GameState {
  if (state.phase !== "playing" || state.publicAction) return state;
  const current = state.players[state.currentIndex];
  if (current.id !== playerId) return state;
  if (state.table) {
    const passed = passTurn(state, playerId);
    const ordinaryLog = `${subjectLabel(current.name)} 패스했습니다.`;
    const timeoutLog = `${subjectLabel(current.name)} 제한시간이 끝나 자동으로 패스했습니다.`;
    return passed === state
      ? state
      : {
          ...passed,
          publicAction: passed.publicAction
            ? {
                ...passed.publicAction,
                automatic: true,
                automaticReason: "timeout",
              }
            : null,
          log: passed.log
            .map((entry) => (entry === ordinaryLog ? timeoutLog : entry))
            .slice(0, 12),
        };
  }

  const nextState: GameState = {
    ...state,
    revision: state.revision + 1,
    log: [
      `${subjectLabel(current.name)} 제한시간이 끝나 자동으로 패스했습니다.`,
      ...state.log,
    ].slice(0, 12),
    publicAction: {
      id: createPublicActionId("pass"),
      kind: "pass",
      player: current,
      cards: [],
      previousTable: null,
      automatic: true,
      automaticReason: "timeout",
    },
  };
  nextState.currentIndex = nextActiveIndex(nextState, state.currentIndex);
  return nextState;
}

function insufficientCardsPassTurn(
  state: GameState,
  playerId: string,
): GameState {
  if (state.phase !== "playing" || state.publicAction || !state.table) {
    return state;
  }
  const current = state.players[state.currentIndex];
  const handCount = state.hands[playerId]?.length ?? 0;
  if (
    current.id !== playerId ||
    handCount === 0 ||
    handCount >= state.table.count
  ) {
    return state;
  }

  const passed = passTurn(state, playerId);
  if (passed === state) return state;
  const ordinaryLog = `${subjectLabel(current.name)} 패스했습니다.`;
  const automaticLog =
    `${subjectLabel(current.name)} 필요한 장수보다 손패가 적어 자동으로 패스했습니다.`;
  return {
    ...passed,
    publicAction: passed.publicAction
      ? {
          ...passed.publicAction,
          automatic: true,
          automaticReason: "insufficient-cards",
        }
      : null,
    log: passed.log
      .map((entry) => (entry === ordinaryLog ? automaticLog : entry))
      .slice(0, 12),
  };
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

function skipRemainingBotTurns(state: GameState): GameState {
  let next: GameState = state.publicAction
    ? { ...state, publicAction: null }
    : state;
  const maximumSteps = 2_000;
  let completed = false;

  for (let step = 0; step < maximumSteps; step += 1) {
    if (next.phase === "round-end") {
      completed = true;
      break;
    }
    if (next.phase !== "playing") break;

    const bot = next.players[next.currentIndex];
    if (!bot || bot.isHuman) break;
    const previousRevision = next.revision;
    const cardIds = chooseBotCards(next, bot.id);
    const advanced = cardIds
      ? playCards(next, bot.id, cardIds)
      : passTurn(next, bot.id);
    next = advanced.publicAction
      ? { ...advanced, publicAction: null }
      : advanced;
    if (next.revision === previousRevision) break;
  }

  return completed
    ? {
        ...next,
        log: ["남은 플레이어의 순위를 빠르게 확정했습니다.", ...next.log].slice(
          0,
          12,
        ),
      }
    : next;
}

function PlayerSeat({
  player,
  handCount,
  score,
  isCurrent,
  isFinished,
  finishRank,
  taxDirection,
  isFocusedTaxParty,
  showHandBacks,
  isHandRevealing,
  rankSelectionLabel,
  rankSelectionMark,
  rankSeat,
  isDalmutiHighlighted,
  seatRef,
}: {
  player: Player;
  handCount: number;
  score: number;
  isCurrent: boolean;
  isFinished: boolean;
  finishRank: number | null;
  taxDirection: TaxDirection | null;
  isFocusedTaxParty: boolean;
  showHandBacks: boolean;
  isHandRevealing: boolean;
  rankSelectionLabel?: string;
  rankSelectionMark?: string;
  rankSeat: number;
  isDalmutiHighlighted: boolean;
  seatRef?: (node: HTMLElement | null) => void;
}) {
  const visibleRoleLabel = rankSelectionLabel ?? ROLE_LABELS[player.role];
  const visibleRoleMark = rankSelectionMark ?? ROLE_MARKS[player.role];
  return (
    <article
      ref={seatRef}
      className={`player-seat role-${player.role} ${isCurrent ? "is-current" : ""} ${
        isFinished ? "is-finished" : ""
      } ${taxDirection ? `is-tax-${taxDirection}` : ""} ${
        isFocusedTaxParty ? "is-focused-tax-party" : ""
      } ${isDalmutiHighlighted ? "is-dalmuti-highlighted" : ""
      }`}
      style={seatPosition(rankSeat - 1, 5)}
      data-rank-seat={rankSeat}
      aria-label={`${player.name}, ${visibleRoleLabel}, ${
        isFinished && finishRank
          ? `${finishRank}위로 마침`
          : `카드 ${handCount}장`
      }`}
    >
      <div className="player-avatar">
        <span>{player.monogram}</span>
        <i>{visibleRoleMark}</i>
      </div>
      <div className="player-copy">
        <strong>{player.name}</strong>
        <span>{visibleRoleLabel}</span>
      </div>
      <div className="player-count">
        <b>{isFinished && finishRank ? `${finishRank}위` : handCount}</b>
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
          ? "하위 계급이 상위 계급에게 세금 카드를 전달하는 중"
          : "상위 계급이 하위 계급에게 반환 카드를 전달하는 중"
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
  players,
  fastForward,
}: {
  action: PublicTurnAction;
  anchors: TaxAnchorMap;
  players: Player[];
  fastForward: boolean;
}) {
  const from = anchors.players[action.player.id];
  const to = anchors.midpoint;
  if (!from || !to) return null;

  const playedSet = action.kind === "play" ? normalizedSet(action.cards) : null;
  const isDalmuti = playedSet?.rank === 1;
  const autoPassedPlayers = (action.autoPassedPlayerIds ?? [])
    .map((playerId) => players.find((player) => player.id === playerId))
    .filter((player): player is Player => Boolean(player));
  const cardCount = action.cards.length;
  const expandedStep =
    cardCount <= 1 ? 0 : Math.min(112, 430 / Math.max(1, cardCount - 1));
  const mobileExpandedStep =
    cardCount <= 1 ? 0 : Math.min(70, 250 / Math.max(1, cardCount - 1));
  const delayStep =
    cardCount <= 1
      ? 0
      : fastForward
        ? Math.min(7, 20 / Math.max(1, cardCount - 1))
        : Math.min(36, 100 / Math.max(1, cardCount - 1));
  const routeStyle = {
    "--from-x": `${from.x}px`,
    "--from-y": `${from.y}px`,
    "--to-x": `${to.x}px`,
    "--to-y": `${to.y}px`,
  } as React.CSSProperties;

  return (
    <div
      key={action.id}
      className={`public-turn-action-layer is-${action.kind} ${
        isDalmuti ? "is-dalmuti" : ""
      } ${fastForward ? "is-fast-forward" : ""}`}
      style={routeStyle}
      role="status"
      aria-live="polite"
      aria-label={
        action.kind === "play" && playedSet
          ? isDalmuti
            ? `${subjectLabel(action.player.name)} 달무티를 내 나머지 플레이어가 자동 패스했습니다`
            : `${subjectLabel(action.player.name)} ${RANK_NAMES[playedSet.rank]} 카드 ${playedSet.count}장을 냈습니다`
          : action.automatic
            ? action.automaticReason === "insufficient-cards"
              ? `${subjectLabel(action.player.name)} 필요한 장수보다 손패가 적어 자동으로 패스했습니다`
              : `${subjectLabel(action.player.name)} 제한시간이 끝나 자동으로 패스했습니다`
            : `${subjectLabel(action.player.name)} 패스했습니다`
      }
    >
      {isDalmuti && (
        <div className="dalmuti-action-effects" aria-hidden="true">
          <i />
          <i />
          {Array.from({ length: 12 }, (_, index) => (
            <span
              key={`dalmuti-spark-${index}`}
              style={
                {
                  "--spark-index": index,
                  "--spark-angle": `${index * 30}deg`,
                  "--spark-delay": `${(index % 4) * 90}ms`,
                } as React.CSSProperties
              }
            />
          ))}
        </div>
      )}
      {isDalmuti &&
        autoPassedPlayers.map((player, playerIndex) => {
          const passFrom = anchors.players[player.id];
          if (!passFrom) return null;
          const passOffset =
            playerIndex - (autoPassedPlayers.length - 1) / 2;
          const passStyle = {
            "--pass-from-x": `${passFrom.x}px`,
            "--pass-from-y": `${passFrom.y}px`,
            "--pass-to-x": `${to.x}px`,
            "--pass-to-y": `${to.y}px`,
            "--pass-offset-x": `${passOffset * 104}px`,
            "--pass-delay": `${
              fastForward
                ? 55 + playerIndex * 14
                : 360 + playerIndex * 90
            }ms`,
          } as React.CSSProperties;

          return (
            <div
              key={`${action.id}-auto-pass-${player.id}`}
              className="dalmuti-auto-pass-badge"
              style={passStyle}
              aria-hidden="true"
            >
              <span>{player.name}</span>
              <strong>PASS</strong>
            </div>
          );
        })}
      {isDalmuti && autoPassedPlayers.length > 0 && (
        <div
          className="dalmuti-auto-pass-banner"
          style={routeStyle}
          aria-hidden="true"
        >
          나머지 플레이어 자동 PASS
        </div>
      )}
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
            <small>{isDalmuti ? "달무티" : "공개 플레이"}</small>
            <strong>{action.player.name}</strong>
            <span>
              {RANK_NAMES[playedSet.rank]}({playedSet.rank}) x {playedSet.count}장
            </span>
          </div>
        </>
      ) : (
        <div className="public-pass-badge" style={routeStyle} aria-hidden="true">
          <small>
            {action.automatic
              ? action.automaticReason === "insufficient-cards"
                ? "카드 부족 · 자동 PASS"
                : "TIME OUT · 자동 PASS"
              : ROLE_LABELS[action.player.role]}
          </small>
          <strong>PASS</strong>
          <span>
            {action.player.name} ·{" "}
            {action.automatic
              ? action.automaticReason === "insufficient-cards"
                ? "제출 장수 부족"
                : "시간 초과"
              : "패스"}
          </span>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [game, setGame] = useState<GameState | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [showRules, setShowRules] = useState(false);
  const [turnTimer, setTurnTimer] = useState<{
    playerId: string;
    deadline: number;
  } | null>(null);
  const [turnClock, setTurnClock] = useState(() => Date.now());
  const [revealedRoundResultKey, setRevealedRoundResultKey] = useState<
    string | null
  >(null);
  const [taxAnchors, setTaxAnchors] = useState<TaxAnchorMap>({
    players: {},
    midpoint: null,
  });
  const tableColumnRef = useRef<HTMLDivElement | null>(null);
  const feltCenterRef = useRef<HTMLDivElement | null>(null);
  const humanAnchorRef = useRef<HTMLDivElement | null>(null);
  const seatRefs = useRef<Record<string, HTMLElement | null>>({});
  const previousSeatRectsRef = useRef<Record<string, DOMRect>>({});
  const previousRankSeatOrderRef = useRef("");

  const currentPlayer = game?.players[game.currentIndex] ?? null;
  const humanHand = game?.hands[HUMAN_ID] ?? [];
  const humanFinished = Boolean(game?.finishOrder.includes(HUMAN_ID));
  const humanFinishRank = game?.finishOrder.indexOf(HUMAN_ID) ?? -1;
  const canSkipRemainingBots =
    humanFinished && game?.phase === "playing";
  const isFastForwardingBots =
    canSkipRemainingBots && currentPlayer?.isHuman === false;
  const isOpeningRankEvent =
    game?.phase === "rank-intro" ||
    game?.phase === "rank-selection" ||
    game?.phase === "rank-reveal" ||
    game?.phase === "rank-confirm";
  const openingRankRolesHidden =
    !game ||
    (game.round === 1 &&
      (game.phase === "ready" || isOpeningRankEvent));
  const humanOpeningRank = selectedOpeningRank(
    game?.openingRankSelection ?? null,
    HUMAN_ID,
  );
  const humanOpeningRole =
    humanOpeningRank && game
      ? roleForIndex(humanOpeningRank - 1, game.players.length)
      : null;
  const hasDealtHands = Boolean(
    game && Object.values(game.hands).some((hand) => hand.length > 0),
  );
  const isHandConcealed =
    game?.phase === "ready" ||
    isOpeningRankEvent ||
    game?.phase === "reveal-intro";
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
  const pendingHumanTaxRecipient =
    pendingHumanTaxExchange && game
      ? game.players.find(
          (player) => player.id === pendingHumanTaxExchange.peonId,
        ) ?? null
      : null;
  const humanTaxRecipientLabel = pendingHumanTaxRecipient
    ? ROLE_LABELS[pendingHumanTaxRecipient.role]
    : "하위 계급";
  const isHumanTaxSelecting = Boolean(pendingHumanTaxExchange);
  const currentPlayerMustAutoPass = Boolean(
    game?.phase === "playing" &&
      !game.publicAction &&
      game.table &&
      currentPlayer &&
      (game.hands[currentPlayer.id]?.length ?? 0) > 0 &&
      (game.hands[currentPlayer.id]?.length ?? 0) < game.table.count,
  );
  const isHumanTurn =
    game?.phase === "playing" &&
    currentPlayer?.id === HUMAN_ID &&
    !game.publicAction &&
    !currentPlayerMustAutoPass;
  const activePublicSet =
    game?.publicAction?.kind === "play"
      ? normalizedSet(game.publicAction.cards)
      : null;
  const dalmutiHighlightPlayerId =
    activePublicSet?.rank === 1 ? game?.publicAction?.player.id ?? null : null;
  const turnTimerPlayerId =
    game?.phase === "playing" &&
    !game.publicAction &&
    !currentPlayerMustAutoPass
      ? currentPlayer?.id ?? null
      : null;
  const turnRemainingMs =
    turnTimerPlayerId !== null &&
    turnTimer?.playerId === turnTimerPlayerId
      ? Math.max(0, turnTimer.deadline - turnClock)
      : null;
  const turnSecondsRemaining =
    turnRemainingMs === null ? null : Math.ceil(turnRemainingMs / 1000);
  const turnProgress =
    turnRemainingMs === null ? 0 : Math.min(1, turnRemainingMs / TURN_LIMIT_MS);
  const turnUrgency =
    turnRemainingMs !== null
      ? Math.max(0, Math.min(1, (10_000 - turnRemainingMs) / 10_000))
      : 0;
  const turnAccentHue = 43 - turnUrgency * 39;
  const scoreRailPlayers = game?.players ?? assignRoles(BASE_PLAYERS);
  const highestScore = Math.max(
    1,
    ...scoreRailPlayers.map((player) => game?.scores[player.id] ?? 0),
  );
  // The announcement itself is transient, but the revolution changes the
  // atmosphere of the whole act. Keep the base red field active until the
  // next round replaces this announcement state.
  const isRevolutionActive = Boolean(game?.revolutionAnnouncement);
  const isGreatRevolutionActive =
    isRevolutionActive &&
    game?.revolutionAnnouncement?.kind === "great-revolution";
  const roundResultKey =
    game?.phase === "round-end" && !game.publicAction
      ? `${game.round}-${game.revision}-${game.finishOrder.join("|")}`
      : null;
  const roundResultReady =
    roundResultKey !== null && revealedRoundResultKey === roundResultKey;
  const isRankTransitioning =
    game?.phase === "round-end" &&
    !game.publicAction &&
    !roundResultReady;
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

  const visibleRankPlayers = useMemo(() => {
    if (!game) return assignRoles(BASE_PLAYERS);
    if (
      game.phase === "round-end" &&
      !game.publicAction &&
      game.finishOrder.length === game.players.length
    ) {
      return assignRoles(
        game.finishOrder.map(
          (playerId) =>
            game.players.find((player) => player.id === playerId)!,
        ),
      );
    }
    return game.players;
  }, [game]);
  const orderedOpponents = useMemo(
    () => visibleRankPlayers.filter((player) => !player.isHuman),
    [visibleRankPlayers],
  );
  const rankSeatOrder = visibleRankPlayers
    .map((player) => player.id)
    .join("|");

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

  useEffect(() => {
    if (!roundResultKey) return;

    const pendingResultKey = roundResultKey;
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const timer = window.setTimeout(
      () => setRevealedRoundResultKey(pendingResultKey),
      reduceMotion
        ? 80
        : RANK_TRANSITION_DURATION_MS + RANK_RESULT_REVEAL_DELAY_MS,
    );

    return () => window.clearTimeout(timer);
  }, [roundResultKey]);

  useLayoutEffect(() => {
    const previousRects = previousSeatRectsRef.current;
    const previousOrder = previousRankSeatOrderRef.current;
    const nextRects: Record<string, DOMRect> = {};
    for (const player of orderedOpponents) {
      const seat = seatRefs.current[player.id];
      if (seat) nextRects[player.id] = seat.getBoundingClientRect();
    }
    previousSeatRectsRef.current = nextRects;
    previousRankSeatOrderRef.current = rankSeatOrder;

    if (
      !previousOrder ||
      previousOrder === rankSeatOrder ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return;
    }

    for (const player of orderedOpponents) {
      const seat = seatRefs.current[player.id];
      const previousRect = previousRects[player.id];
      const nextRect = nextRects[player.id];
      if (!seat || !previousRect || !nextRect) continue;

      const deltaX = previousRect.left - nextRect.left;
      const deltaY = previousRect.top - nextRect.top;
      if (Math.abs(deltaX) < 2 && Math.abs(deltaY) < 2) continue;

      const distance = Math.max(1, Math.hypot(deltaX, deltaY));
      const direction =
        orderedOpponents.findIndex((candidate) => candidate.id === player.id) %
          2 ===
        0
          ? 1
          : -1;
      const arc = Math.min(78, Math.max(30, distance * 0.2)) * direction;
      const arcX = (-deltaY / distance) * arc;
      const arcY = (deltaX / distance) * arc;

      seat.classList.add("is-changing-rank");
      const animation = seat.animate(
        [
          {
            transform: `translate(${deltaX}px, ${deltaY}px) scale(0.92) rotate(${
              direction * -2.4
            }deg)`,
            opacity: 0.46,
            filter: "brightness(0.88) saturate(0.82)",
          },
          {
            transform: `translate(${deltaX * 0.55 + arcX}px, ${
              deltaY * 0.55 + arcY
            }px) scale(1.075) rotate(${direction * 1.8}deg)`,
            opacity: 1,
            filter: "brightness(1.48) saturate(1.32)",
            offset: 0.46,
          },
          {
            transform: `translate(${deltaX * 0.1 - arcX * 0.28}px, ${
              deltaY * 0.1 - arcY * 0.28
            }px) scale(1.035) rotate(${direction * -0.7}deg)`,
            opacity: 1,
            filter: "brightness(1.22) saturate(1.18)",
            offset: 0.78,
          },
          {
            transform: "translate(0, 0) scale(1)",
            opacity: 1,
            filter: "brightness(1) saturate(1)",
          },
        ],
        {
          duration: RANK_TRANSITION_DURATION_MS,
          easing: "cubic-bezier(0.16, 0.74, 0.2, 1)",
        },
      );
      void animation.finished
        .catch(() => undefined)
        .finally(() => seat.classList.remove("is-changing-rank"));
    }
  }, [rankSeatOrder, orderedOpponents]);

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
    observer.observe(felt);
    window.addEventListener("resize", measure);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [game]);

  useEffect(() => {
    if (!turnTimerPlayerId) return;

    const expiredPlayerId = turnTimerPlayerId;
    const startedAt = Date.now();
    const deadline = startedAt + TURN_LIMIT_MS;
    const beginTimer = window.setTimeout(() => {
      setTurnClock(Date.now());
      setTurnTimer({ playerId: expiredPlayerId, deadline });
    }, 0);
    const clockTimer = window.setInterval(() => {
      setTurnClock(Date.now());
    }, 100);
    const expirationTimer = window.setTimeout(() => {
      setTurnTimer((current) =>
        current?.playerId === expiredPlayerId &&
        current.deadline === deadline
          ? null
          : current,
      );
      if (expiredPlayerId === HUMAN_ID) setSelectedIds([]);
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "playing" ||
          latest.publicAction ||
          latest.players[latest.currentIndex]?.id !== expiredPlayerId
        ) {
          return latest;
        }
        return timeoutPassTurn(latest, expiredPlayerId);
      });
    }, TURN_LIMIT_MS);

    return () => {
      window.clearTimeout(beginTimer);
      window.clearInterval(clockTimer);
      window.clearTimeout(expirationTimer);
    };
  }, [turnTimerPlayerId]);

  useEffect(() => {
    const action = game?.publicAction;
    if (!action) return;
    const actionId = action.id;
    const playedSet =
      action.kind === "play" ? normalizedSet(action.cards) : null;
    const accelerateAction =
      humanFinished && action.player.id !== HUMAN_ID;
    const duration = accelerateAction
      ? action.kind === "pass"
        ? FAST_PASS_ACTION_DURATION_MS
        : playedSet?.rank === 1
          ? FAST_DALMUTI_ACTION_DURATION_MS
          : FAST_PUBLIC_ACTION_DURATION_MS
      : action.kind === "pass"
        ? PASS_ACTION_DURATION_MS
        : playedSet?.rank === 1
          ? DALMUTI_ACTION_DURATION_MS
          : PUBLIC_ACTION_DURATION_MS;

    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (!latest || latest.publicAction?.id !== actionId) return latest;
        return { ...latest, publicAction: null };
      });
    }, duration);

    return () => window.clearTimeout(timer);
  }, [game?.publicAction, humanFinished]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "rank-intro" ||
      !game.openingRankSelection
    ) {
      return;
    }
    const introRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "rank-intro" ||
          latest.revision !== introRevision ||
          !latest.openingRankSelection
        ) {
          return latest;
        }

        if (latest.openingRankSelection.countdown > 1) {
          return {
            ...latest,
            revision: latest.revision + 1,
            openingRankSelection: {
              ...latest.openingRankSelection,
              countdown: latest.openingRankSelection.countdown - 1,
            },
          };
        }

        return {
          ...latest,
          phase: "rank-selection",
          revision: latest.revision + 1,
          openingRankSelection: {
            ...latest.openingRankSelection,
            countdown: 0,
          },
          log: ["계급 카드 선택이 시작되었습니다.", ...latest.log].slice(0, 12),
        };
      });
    }, RANK_COUNTDOWN_STEP_MS);

    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "rank-selection" ||
      !game.openingRankSelection ||
      game.openingRankSelection.selectedBy.every(Boolean)
    ) {
      return;
    }

    const unpickedBot = game.players.find(
      (player) =>
        !player.isHuman &&
        !game.openingRankSelection!.pickOrder.includes(player.id),
    );
    if (!unpickedBot) return;

    const selectionRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "rank-selection" ||
          latest.revision !== selectionRevision ||
          !latest.openingRankSelection
        ) {
          return latest;
        }
        const bot = latest.players.find(
          (player) =>
            !player.isHuman &&
            !latest.openingRankSelection!.pickOrder.includes(player.id),
        );
        if (!bot) return latest;
        const availableIndexes = latest.openingRankSelection.selectedBy
          .map((selectedPlayerId, index) => (selectedPlayerId ? -1 : index))
          .filter((index) => index >= 0);
        if (!availableIndexes.length) return latest;
        const cardIndex =
          availableIndexes[Math.floor(Math.random() * availableIndexes.length)];
        const selectedBy = [...latest.openingRankSelection.selectedBy];
        selectedBy[cardIndex] = bot.id;

        const nextState: GameState = {
          ...latest,
          revision: latest.revision + 1,
          openingRankSelection: {
            ...latest.openingRankSelection,
            selectedBy,
            pickOrder: [...latest.openingRankSelection.pickOrder, bot.id],
          },
          log: [
            `${subjectLabel(bot.name)} 계급 카드 한 장을 골랐습니다.`,
            ...latest.log,
          ].slice(0, 12),
        };
        return autoAssignFinalOpeningRankCard(nextState);
      });
    }, BOT_RANK_PICK_DELAY_MS);

    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "rank-selection" ||
      !game.openingRankSelection ||
      !game.openingRankSelection.selectedBy.every(Boolean)
    ) {
      return;
    }
    const selectionRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "rank-selection" ||
          latest.revision !== selectionRevision ||
          !latest.openingRankSelection ||
          !latest.openingRankSelection.selectedBy.every(Boolean)
        ) {
          return latest;
        }
        return {
          ...latest,
          phase: "rank-reveal",
          revision: latest.revision + 1,
          log: ["선택한 계급 카드를 공개합니다.", ...latest.log].slice(0, 12),
        };
      });
    }, RANK_ALL_SELECTED_PAUSE_MS);

    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "rank-reveal" ||
      !game.openingRankSelection
    ) {
      return;
    }
    const revealRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "rank-reveal" ||
          latest.revision !== revealRevision ||
          !latest.openingRankSelection
        ) {
          return latest;
        }
        return {
          ...latest,
          phase: "rank-confirm",
          revision: latest.revision + 1,
          log: ["나의 확정 서열을 확인합니다.", ...latest.log].slice(0, 12),
        };
      });
    }, RANK_REVEAL_DURATION_MS);

    return () => window.clearTimeout(timer);
  }, [game]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "rank-confirm" ||
      !game.openingRankSelection
    ) {
      return;
    }
    const confirmRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "rank-confirm" ||
          latest.revision !== confirmRevision ||
          !latest.openingRankSelection
        ) {
          return latest;
        }
        return completeOpeningRankSelection(latest);
      });
    }, RANK_CONFIRM_DURATION_MS);

    return () => window.clearTimeout(timer);
  }, [game]);

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
    if (!game || game.phase !== "revolution-intro") return;
    const revolutionRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "revolution-intro" ||
          latest.revision !== revolutionRevision
        ) {
          return latest;
        }
        return {
          ...latest,
          phase: "play-intro",
          revision: latest.revision + 1,
        };
      });
    }, REVOLUTION_INTRO_DURATION_MS);

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
      !game.table
    ) {
      return;
    }
    const player = game.players[game.currentIndex];
    const handCount = game.hands[player.id]?.length ?? 0;
    if (handCount === 0 || handCount >= game.table.count) return;

    const turnRevision = game.revision;
    const playerId = player.id;
    const timer = window.setTimeout(() => {
      if (playerId === HUMAN_ID) setSelectedIds([]);
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "playing" ||
          latest.revision !== turnRevision ||
          latest.publicAction ||
          !latest.table ||
          latest.players[latest.currentIndex]?.id !== playerId
        ) {
          return latest;
        }
        return insufficientCardsPassTurn(latest, playerId);
      });
    }, humanFinished ? 70 : 170);

    return () => window.clearTimeout(timer);
  }, [game, humanFinished]);

  useEffect(() => {
    if (
      !game ||
      game.phase !== "playing" ||
      game.publicAction ||
      currentPlayer?.isHuman ||
      currentPlayerMustAutoPass
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
    }, humanFinished ? FAST_BOT_THINK_MS : 760);
    return () => window.clearTimeout(timer);
  }, [
    currentPlayer?.id,
    currentPlayer?.isHuman,
    currentPlayerMustAutoPass,
    game,
    humanFinished,
  ]);

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
    setRevealedRoundResultKey(null);
    setGame(createOpeningRound(BASE_PLAYERS, scores));
  };

  const returnToModeSelection = () => {
    setSelectedIds([]);
    setShowRules(false);
    setRevealedRoundResultKey(null);
    setTaxAnchors({ players: {}, midpoint: null });
    setGame(null);
  };

  const beginHostedGame = () => {
    setSelectedIds([]);
    setGame((current) => {
      if (!current || current.phase !== "ready") return current;
      const alreadyDealt = Object.values(current.hands).some(
        (hand) => hand.length > 0,
      );
      if (current.round > 1 || alreadyDealt) {
        return {
          ...current,
          phase: "reveal-intro",
          revision: current.revision + 1,
          log: [
            "방장이 패 공개를 시작했습니다.",
            ...current.log,
          ].slice(0, 12),
        };
      }
      const cards = shuffle(
        Array.from({ length: current.players.length }, (_, index) => index + 1),
      );
      return {
        ...current,
        phase: "rank-intro",
        revision: current.revision + 1,
        openingRankSelection: {
          cards,
          selectedBy: cards.map(() => null),
          pickOrder: [],
          countdown: 3,
        },
        log: [
          "첫 게임의 계급은 선착순 카드 선택으로 정합니다.",
          ...current.log,
        ].slice(0, 12),
      };
    });
  };

  const chooseOpeningRankCard = (cardIndex: number) => {
    setGame((current) => {
      if (
        !current ||
        current.phase !== "rank-selection" ||
        !current.openingRankSelection ||
        current.openingRankSelection.pickOrder.includes(HUMAN_ID) ||
        current.openingRankSelection.selectedBy[cardIndex]
      ) {
        return current;
      }

      const selectedBy = [...current.openingRankSelection.selectedBy];
      selectedBy[cardIndex] = HUMAN_ID;
      const nextState: GameState = {
        ...current,
        revision: current.revision + 1,
        openingRankSelection: {
          ...current.openingRankSelection,
          selectedBy,
          pickOrder: [...current.openingRankSelection.pickOrder, HUMAN_ID],
        },
        log: ["내 계급 카드 한 장을 골랐습니다.", ...current.log].slice(0, 12),
      };
      return autoAssignFinalOpeningRankCard(nextState);
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
      let revolutionAnnouncement = current.revolutionAnnouncement;
      const holder = current.players.find(
        (player) => player.id === current.revolutionHolder,
      );

      if (declare && holder?.role === "great-peon") {
        players = assignRoles([...current.players].reverse());
        phase = "revolution-intro";
        revolutionAnnouncement = createRevolutionAnnouncement(
          holder,
          "great-revolution",
        );
        log = ["당신의 대혁명으로 모든 계급이 뒤집혔습니다.", ...log];
      } else if (declare && holder) {
        phase = "revolution-intro";
        revolutionAnnouncement = createRevolutionAnnouncement(
          holder,
          "revolution",
        );
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
        revolutionAnnouncement,
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

  const skipRemainingPlayers = () => {
    if (!canSkipRemainingBots) return;
    setSelectedIds([]);
    setRevealedRoundResultKey(null);
    setGame((current) =>
      current &&
      current.phase === "playing" &&
      current.finishOrder.includes(HUMAN_ID)
        ? skipRemainingBotTurns(current)
        : current,
    );
  };

  const nextRound = () => {
    if (!game || game.phase !== "round-end" || game.publicAction) return;
    const ordered = game.finishOrder.map(
      (id) => game.players.find((player) => player.id === id)!,
    );
    setSelectedIds([]);
    setRevealedRoundResultKey(null);
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
        ? publicPlayedSet.rank === 1
          ? `${subjectLabel(game.publicAction.player.name)} 달무티를 내 모두 자동 패스합니다`
          : `${subjectLabel(game.publicAction.player.name)} ${RANK_NAMES[publicPlayedSet.rank]} 카드 ${publicPlayedSet.count}장을 내는 중`
        : game.publicAction.automaticReason === "insufficient-cards"
          ? `${subjectLabel(game.publicAction.player.name)} 필요한 장수가 부족해 자동 패스합니다`
          : `${subjectLabel(game.publicAction.player.name)} 패스했습니다`
      : game.phase === "ready"
        ? game.round === 1 && !hasDealtHands
          ? "방장이 PLAY를 누르면 계급 정하기를 시작합니다"
          : "방장이 PLAY를 누르면 패를 공개합니다"
        : game.phase === "rank-intro"
          ? `계급 카드 선택까지 ${game.openingRankSelection?.countdown ?? 3}`
          : game.phase === "rank-selection"
            ? game.openingRankSelection?.selectedBy.every(Boolean)
              ? "모든 선택 완료 · 카드를 공개합니다"
              : game.openingRankSelection?.pickOrder.includes(HUMAN_ID)
                ? "다른 플레이어가 계급 카드를 고르는 중"
                : "선착순으로 계급 카드 한 장을 고르세요"
            : game.phase === "rank-reveal"
              ? "숫자가 낮은 카드부터 높은 계급을 얻습니다"
              : game.phase === "rank-confirm"
                ? humanOpeningRole
                  ? `나의 서열은 ${ROLE_LABELS[humanOpeningRole]}입니다`
                  : "나의 확정 서열을 확인합니다"
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
                          : game.phase === "revolution-intro"
                            ? `${subjectLabel(game.revolutionAnnouncement?.playerName ?? "")} ${
                                game.revolutionAnnouncement?.kind === "great-revolution"
                                  ? "대혁명을 일으켰습니다"
                                  : "혁명을 일으켰습니다"
                              }`
                            : game.phase === "taxation"
                              ? game.taxStage === "selection"
                                ? `${humanTaxRecipientLabel}에게 돌려줄 카드 ${humanTaxSelectionCount}장을 선택하세요`
                                : focusedTaxRoute
                                  ? `${focusedTaxRoute.from.name} → ${focusedTaxRoute.to.name} · 카드 ${focusedTaxRoute.cards.length}장 전달 중`
                                  : "당사자끼리 비공개 세금 교환 중"
                              : isHumanTurn
                                ? game.table
                                  ? `${game.table.rank}보다 낮은 숫자의 카드 ${game.table.count}장을 내세요`
                                  : "새로운 묶음을 시작하세요"
                                : `${currentPlayer?.name}의 선택을 기다리는 중`;

  const tablePreview = visibleTable?.cards ?? [];
  const tableCardStep =
    tablePreview.length <= 1
      ? 0
      : Math.min(54, 460 / Math.max(1, tablePreview.length - 1));
  const mobileTableCardStep =
    tablePreview.length <= 1
      ? 0
      : Math.min(32, 160 / Math.max(1, tablePreview.length - 1));

  return (
    <main className="game-shell">
      <div className="paper-grain" aria-hidden="true" />

      <header className="topbar">
        <button
          type="button"
          className="brand brand-button"
          onClick={returnToModeSelection}
          aria-label="초기 모드 선택 화면으로 돌아가기"
        >
          <span className="brand-seal" aria-hidden="true" />
          <div>
            <strong>DALMUTI</strong>
            <small>DCLab의 계급전</small>
          </div>
        </button>

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
            <span>서열</span>
            <small>누적 점수</small>
          </div>
          <ol>
            {scoreRailPlayers.map((player) => {
              const rankLabel = openingRankRolesHidden
                ? "계급 미정"
                : ROLE_LABELS[player.role];
              const rankMark = openingRankRolesHidden
                ? "·"
                : ROLE_MARKS[player.role];
              const score = game?.scores[player.id] ?? 0;
              const chipCount = scoreChipCount(score, highestScore);
              return (
                <li
                  key={player.id}
                  className={player.id === HUMAN_ID ? "is-you" : ""}
                >
                  <span>{rankMark}</span>
                  <div>
                    <b>{player.name}</b>
                    <small>{rankLabel}</small>
                  </div>
                  <em
                    className="score-display"
                    aria-label={`${player.name} 누적 점수 ${score}점`}
                  >
                    <span className="score-chip-stack" aria-hidden="true">
                      {Array.from({ length: chipCount }, (_, chipIndex) => (
                        <i
                          key={chipIndex}
                          style={
                            {
                              "--score-chip-index": chipIndex,
                            } as React.CSSProperties
                          }
                        />
                      ))}
                    </span>
                    <span className="score-number">{score}</span>
                  </em>
                </li>
              );
            })}
          </ol>
          <div className="rail-note">
            <span>계급의 법칙</span>
            <p>숫자가 낮을수록 강합니다. 더 강하게 맞서세요.</p>
          </div>
        </aside>

        <div
          className={`table-column ${isHumanTurn ? "is-human-turn" : ""} ${
            isHumanTurn && turnUrgency > 0 ? "is-turn-urgent" : ""
          } ${isRankTransitioning ? "is-rank-transitioning" : ""} ${
            isGreatRevolutionActive ? "has-great-revolution" : ""
          }`}
          style={
            {
              "--turn-urgency": turnUrgency,
              "--turn-accent-hue": turnAccentHue,
            } as React.CSSProperties
          }
          ref={tableColumnRef}
        >
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
              players={game.players}
              fastForward={
                humanFinished && game.publicAction.player.id !== HUMAN_ID
              }
            />
          )}

          {turnSecondsRemaining !== null &&
            currentPlayer &&
            !isFastForwardingBots && (
            <div
              className={`turn-countdown ${
                currentPlayer.id === HUMAN_ID ? "is-mine" : ""
              } ${turnSecondsRemaining <= 10 ? "is-urgent" : ""}`}
              style={
                {
                  "--turn-angle": `${turnProgress * 360}deg`,
                  "--turn-urgency": turnUrgency,
                  "--turn-accent-hue": turnAccentHue,
                } as React.CSSProperties
              }
              role="timer"
              aria-live={turnSecondsRemaining <= 10 ? "polite" : "off"}
              aria-label={`${currentPlayer.name}의 남은 시간 ${turnSecondsRemaining}초`}
            >
              <div>
                <span>
                  <b>{turnSecondsRemaining}</b>
                  <small>SEC</small>
                </span>
              </div>
              <p>
                {currentPlayer.id === HUMAN_ID
                  ? "내 차례"
                  : `${currentPlayer.name}의 차례`}
              </p>
            </div>
          )}

          {isRankTransitioning && (
            <div className="rank-transition-effect" aria-hidden="true">
              <i />
              <i />
              {Array.from({ length: 10 }, (_, index) => (
                <span
                  key={`rank-transition-spark-${index}`}
                  style={
                    {
                      "--transition-spark-index": index,
                      "--transition-spark-y": `${16 + index * 7}%`,
                      "--transition-spark-delay": `${index * 90}ms`,
                    } as React.CSSProperties
                  }
                />
              ))}
            </div>
          )}

          <div
            className={`felt-table ${
              isRevolutionActive ? "is-revolution" : ""
            } ${isGreatRevolutionActive ? "is-great-revolution" : ""}`}
            ref={feltCenterRef}
          >
            <div className="table-ring" aria-hidden="true">
              <span>♜</span>
              <i />
              <span>♞</span>
              <i />
              <span>♝</span>
            </div>

            {isGreatRevolutionActive && (
              <div className="great-revolution-field-effect" aria-hidden="true">
                <i />
                <i />
                {Array.from({ length: 14 }, (_, index) => (
                  <span
                    key={`great-revolution-ember-${index}`}
                    style={
                      {
                        "--ember-index": index,
                        "--ember-x": `${4 + ((index * 37) % 92)}%`,
                        "--ember-y": `${12 + ((index * 29) % 74)}%`,
                        "--ember-size": `${3 + (index % 3)}px`,
                        "--ember-delay": `${(index % 7) * -0.52}s`,
                        "--ember-duration": `${3.2 + (index % 5) * 0.42}s`,
                      } as React.CSSProperties
                    }
                  />
                ))}
              </div>
            )}

            <div
              className={`opponent-row ${
                isHandRevealing ? "is-revealing" : ""
              }`}
            >
              {(orderedOpponents.length
                ? orderedOpponents
                : visibleRankPlayers.filter((player) => !player.isHuman)
              ).map((player) => {
                const rankSeat =
                  visibleRankPlayers.findIndex(
                    (candidate) => candidate.id === player.id,
                  ) + 1;
                const route = activeTaxRoutes.find(
                  (candidate) =>
                    candidate.from.id === player.id ||
                    candidate.to.id === player.id,
                );
                const taxDirection: TaxDirection | null = route
                  ? route.from.id === player.id
                    ? "source"
                    : "destination"
                  : null;
                const rankSelectionLabel = openingRankRolesHidden
                  ? "계급 미정"
                  : undefined;
                const rankSelectionMark = openingRankRolesHidden
                  ? "·"
                  : undefined;
                const finishIndex =
                  game?.finishOrder.indexOf(player.id) ?? -1;

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
                    finishRank={finishIndex >= 0 ? finishIndex + 1 : null}
                    taxDirection={taxDirection}
                    isFocusedTaxParty={Boolean(route?.reveal)}
                    showHandBacks={hasDealtHands}
                    isHandRevealing={isHandRevealing}
                    rankSelectionLabel={rankSelectionLabel}
                    rankSelectionMark={rankSelectionMark}
                    rankSeat={rankSeat}
                    isDalmutiHighlighted={
                      dalmutiHighlightPlayerId === player.id
                    }
                    seatRef={(node) => {
                      seatRefs.current[player.id] = node;
                    }}
                  />
                );
              })}
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
                  <span>
                    {game.round === 1 && !hasDealtHands
                      ? "PLAY를 누르면 제1막의 계급 정하기를 시작합니다"
                      : "PLAY를 누르면 새로운 계급의 패를 공개합니다"}
                  </span>
                  <button
                    type="button"
                    className="ready-play-button"
                    onClick={beginHostedGame}
                  >
                    <i>▶</i>
                    PLAY
                  </button>
                </div>
              ) : game?.phase === "rank-intro" &&
                game.openingRankSelection ? (
                <div className="opening-rank-intro">
                  <small>ACT I · RANK DRAW</small>
                  <strong>계급 정하기</strong>
                  <span>
                    첫 게임은 선착순으로 카드를 한 장씩 골라 계급을 정합니다
                  </span>
                  <b
                    key={`rank-countdown-${game.openingRankSelection.countdown}`}
                    aria-label={`${game.openingRankSelection.countdown}초 후 선택 시작`}
                  >
                    {game.openingRankSelection.countdown}
                  </b>
                  <em>숫자가 낮을수록 높은 계급입니다</em>
                </div>
              ) : (game?.phase === "rank-selection" ||
                  game?.phase === "rank-reveal") &&
                game.openingRankSelection ? (
                <div
                  className={`opening-rank-board ${
                    game.phase === "rank-reveal" ? "is-revealed" : ""
                  }`}
                >
                  <div className="opening-rank-heading">
                    <small>ACT I · RANK DRAW</small>
                    <strong>
                      {game.phase === "rank-reveal"
                        ? "계급 카드 공개"
                        : game.openingRankSelection.selectedBy.every(Boolean)
                          ? "모든 선택 완료"
                          : game.openingRankSelection.pickOrder.includes(HUMAN_ID)
                            ? "다른 플레이어를 기다리는 중"
                            : "계급 카드를 고르세요"}
                    </strong>
                    <span>
                      {game.phase === "rank-reveal"
                        ? "낮은 숫자의 카드를 뽑은 순서로 서열이 정해집니다"
                        : game.openingRankSelection.selectedBy.every(Boolean)
                          ? "1초 뒤에 선택한 카드를 공개합니다"
                          : "빛나는 카드는 이미 다른 플레이어가 선택했습니다"}
                    </span>
                  </div>
                  <div
                    className="opening-rank-cards"
                    aria-label="계급 선택 카드"
                  >
                    {game.openingRankSelection.cards.map((rank, cardIndex) => {
                      const selectedPlayerId =
                        game.openingRankSelection!.selectedBy[cardIndex];
                      const selectedPlayer = game.players.find(
                        (player) => player.id === selectedPlayerId,
                      );
                      const humanAlreadyPicked =
                        game.openingRankSelection!.pickOrder.includes(HUMAN_ID);
                      const canChoose =
                        game.phase === "rank-selection" &&
                        !humanAlreadyPicked &&
                        !selectedPlayerId;
                      return (
                        <button
                          key={`opening-rank-${cardIndex}`}
                          type="button"
                          className={`opening-rank-card ${
                            selectedPlayerId ? "is-selected" : ""
                          } ${selectedPlayerId === HUMAN_ID ? "is-yours" : ""}`}
                          disabled={!canChoose}
                          onClick={() => chooseOpeningRankCard(cardIndex)}
                          aria-label={
                            game.phase === "rank-reveal"
                              ? `${selectedPlayer?.name ?? "선택자"}의 계급 카드, ${RANK_NAMES[rank]} ${rank}`
                              : selectedPlayer
                                ? `${selectedPlayer.name}이(가) 선택한 카드`
                                : `${cardIndex + 1}번째 뒤집힌 계급 카드 선택`
                          }
                        >
                          <span className="opening-rank-card-inner">
                            <span
                              className="opening-rank-card-back"
                              aria-hidden="true"
                            />
                            <span className="opening-rank-card-front">
                              <img
                                src={`/cards/${String(rank).padStart(2, "0")}.webp?v=${CARD_ART_VERSION}`}
                                alt=""
                                aria-hidden="true"
                              />
                            </span>
                          </span>
                          <em>
                            {selectedPlayer
                              ? `${selectedPlayer.name} 선택`
                              : "선택 가능"}
                          </em>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : game?.phase === "rank-confirm" &&
                humanOpeningRank &&
                humanOpeningRole ? (
                <div
                  key={`rank-confirm-${game.revision}`}
                  className={`opening-rank-confirmation role-${humanOpeningRole}`}
                  role="status"
                  aria-live="assertive"
                >
                  <small>YOUR RANK · ACT I</small>
                  <div className="opening-rank-confirmation-body">
                    <div
                      className="opening-rank-confirmation-card"
                      aria-hidden="true"
                    >
                      <img
                        src={`/cards/${String(humanOpeningRank).padStart(2, "0")}.webp?v=${CARD_ART_VERSION}`}
                        alt=""
                      />
                    </div>
                    <div className="opening-rank-confirmation-copy">
                      <span>나의 서열</span>
                      <strong>{ROLE_LABELS[humanOpeningRole]}</strong>
                      <em>
                        {RANK_NAMES[humanOpeningRank]}({humanOpeningRank}) 카드를
                        선택했습니다
                      </em>
                    </div>
                  </div>
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
              ) : game?.phase === "revolution-intro" &&
                game.revolutionAnnouncement ? (
                <div
                  key={game.revolutionAnnouncement.id}
                  className={`revolution-announcement is-${game.revolutionAnnouncement.kind}`}
                  role="status"
                  aria-live="assertive"
                >
                  <div className="revolution-joker-pair" aria-hidden="true">
                    <div className="revolution-joker-card is-left">
                      <PlayingCard
                        card={{ id: "revolution-joker-left", rank: 13 }}
                        displayOnly
                      />
                    </div>
                    <div className="revolution-joker-card is-right">
                      <PlayingCard
                        card={{ id: "revolution-joker-right", rank: 13 }}
                        displayOnly
                      />
                    </div>
                    <i />
                    <i />
                  </div>
                  <small>
                    {game.revolutionAnnouncement.kind === "great-revolution"
                      ? "GREAT REVOLUTION"
                      : "REVOLUTION"}
                  </small>
                  <strong>
                    {game.revolutionAnnouncement.kind === "great-revolution"
                      ? "대혁명"
                      : "혁명"}
                  </strong>
                  <span>
                    {subjectLabel(game.revolutionAnnouncement.playerName)}{" "}
                    {game.revolutionAnnouncement.kind === "great-revolution"
                      ? "대혁명을 일으켰습니다"
                      : "혁명을 일으켰습니다"}
                  </span>
                  <em>
                    {game.revolutionAnnouncement.kind === "great-revolution"
                      ? "모든 계급이 뒤집히고 이번 막의 세금이 사라집니다"
                      : "이번 막의 세금이 사라집니다"}
                  </em>
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
                    <strong>
                      {humanTaxRecipientLabel}에게 돌려줄 카드를 고르세요
                    </strong>
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
                          ? `${ROLE_LABELS[focusedTaxRoute.from.role]}의 세금 카드 전달`
                          : `${ROLE_LABELS[focusedTaxRoute.from.role]}의 반환 카드 전달`
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
                          style={
                            {
                              "--card-index": index,
                              "--table-card-offset":
                                index - (tablePreview.length - 1) / 2,
                              "--table-card-lift": `${
                                Math.abs(
                                  index - (tablePreview.length - 1) / 2,
                                ) * 0.9
                              }px`,
                              "--table-card-overlap": `${tableCardStep - 140}px`,
                              "--table-card-overlap-mobile": `${mobileTableCardStep - 108}px`,
                            } as React.CSSProperties
                          }
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
            } ${dalmutiHighlightPlayerId === HUMAN_ID ? "is-dalmuti-highlighted" : ""
            }`}
          >
            <div className="human-status" ref={humanAnchorRef}>
              <div className="human-avatar">나</div>
              <div>
                <span>
                  {openingRankRolesHidden
                    ? "계급 미정"
                    : game
                      ? ROLE_LABELS[
                          game.players.find((p) => p.id === HUMAN_ID)!.role
                        ]
                      : "상인"}
                </span>
                <strong>
                  {humanFinished
                    ? `이번 막 완료 · ${humanFinishRank + 1}위`
                    : isOpeningRankEvent
                      ? game?.phase === "rank-selection" &&
                        !game.openingRankSelection?.pickOrder.includes(HUMAN_ID)
                        ? "계급 카드를 고르세요"
                        : game?.phase === "rank-reveal"
                          ? "선택한 계급을 확인하는 중"
                          : game?.phase === "rank-confirm" && humanOpeningRole
                            ? `${ROLE_LABELS[humanOpeningRole]} 확정`
                          : "계급 정하기 진행 중"
                    : isHumanTaxSelecting
                      ? `반환 카드 ${humanTaxSelectionCount}장을 선택하세요`
                    : isHandConcealed
                      ? "패 미정"
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
                    : currentPlayer
                      ? `${currentPlayer.name}의 차례`
                      : "나의 차례"}
                </strong>
              </div>
              <em>
                {humanFinished
                  ? `${humanFinishRank + 1}위`
                  : isOpeningRankEvent
                    ? "선택"
                    : `${game ? humanHand.length : 16}장`}
              </em>
              {humanTaxDirection && (
                <i className={`human-tax-flag is-${humanTaxDirection}`}>
                  {humanTaxDirection === "source" ? "보냄" : "받음"}
                </i>
              )}
            </div>

            <div className="hand-wrap">
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
                    <span className="finished-hand-medal">
                      <b>{humanFinishRank + 1}</b>
                      <i>PLACE</i>
                    </span>
                    <span className="finished-hand-copy">
                      <small>ROUND COMPLETE</small>
                      <strong>먼저 모든 카드를 냈습니다</strong>
                      <em>
                        {humanFinishRank + 1}위 확정 · 남은 경기를 관전하는 중
                      </em>
                    </span>
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
              {isOpeningRankEvent ? (
                <div className="selection-hint is-valid opening-rank-control-state">
                  <span>
                    {game?.phase === "rank-selection" &&
                    !game.openingRankSelection?.pickOrder.includes(HUMAN_ID)
                      ? "필드의 카드 한 장을 선택하세요"
                      : game?.phase === "rank-reveal"
                        ? humanOpeningRank
                          ? `${RANK_NAMES[humanOpeningRank]}(${humanOpeningRank}) 선택`
                          : "계급 확인 중"
                        : game?.phase === "rank-confirm" && humanOpeningRole
                          ? `나의 서열 · ${ROLE_LABELS[humanOpeningRole]}`
                        : "계급 정하기 진행 중"}
                  </span>
                  <small>
                    {game?.phase === "rank-selection"
                      ? "이미 빛나는 카드는 다른 플레이어가 먼저 선택했습니다"
                      : "계급이 확정되면 패를 나누고 공개합니다"}
                  </small>
                </div>
              ) : humanFinished ? (
                <div className="selection-hint is-valid finished-control-state">
                  <span>
                    {canSkipRemainingBots ? "배속으로 순위 결정 중" : "이번 막 완료"}
                  </span>
                  <small>
                    {canSkipRemainingBots
                      ? "남은 플레이를 빠르게 진행합니다"
                      : "모든 순위가 확정되었습니다"}
                  </small>
                  {canSkipRemainingBots && (
                    <button
                      type="button"
                      className="skip-round-button"
                      onClick={skipRemainingPlayers}
                    >
                      스킵
                    </button>
                  )}
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
                        `선택한 ${humanTaxSelectionCount}장은 ${humanTaxRecipientLabel}에게 전달됩니다`}
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
                    제출
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
              "첫 판은 계급 카드를 직접 골라 서열을 정합니다.",
              "방장의 PLAY 이후 계급 정하기가 시작됩니다.",
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
            <span className="welcome-crown" aria-hidden="true" />
            <span className="eyebrow">PLAYABLE PROTOTYPE · 5 PLAYERS</span>
            <h1 id="welcome-title">DALMUTI</h1>
            <p>
              약한 패부터 영리하게 털어내고, 계급을 뒤집으세요.
            </p>
            <div className="welcome-features">
              <span>80장 정식 덱</span>
              <span>세금과 혁명</span>
              <span>연속 라운드</span>
            </div>
            <button type="button" className="start-button" onClick={startGame}>
              <span>빠른 대전(5인)</span>
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
            <h2 id="revolution-title">
              {game.players.find((player) => player.id === game.revolutionHolder)
                ?.role === "great-peon"
                ? "대혁명을 선포하시겠습니까?"
                : "혁명을 선포하시겠습니까?"}
            </h2>
            <p>
              혁명을 선포하면 이번 막의 세금이 사라집니다.
              농노라면 모든 계급까지 뒤집힙니다.
            </p>
            <div>
              <button type="button" className="secondary-button" onClick={() => resolveRevolution(false)}>
                조용히 지나간다
              </button>
              <button type="button" className="play-button" onClick={() => resolveRevolution(true)}>
                {game.players.find((player) => player.id === game.revolutionHolder)
                  ?.role === "great-peon"
                  ? "대혁명 선포"
                  : "혁명 선포"}
              </button>
            </div>
          </section>
        </div>
      )}

      {game?.phase === "round-end" &&
        !game.publicAction &&
        roundResultReady && (
        <div className="modal-layer">
          <section className="result-card" role="dialog" aria-labelledby="result-title">
            <span className="eyebrow">THE COURT HAS SPOKEN</span>
            <h2 id="result-title">제 {game.round}막의 새로운 계급</h2>
            <ol>
              {game.finishOrder.map((id, index) => {
                const player = game.players.find((candidate) => candidate.id === id)!;
                const nextRole = roleForIndex(index, game.players.length);
                const previousIndex = game.players.findIndex(
                  (candidate) => candidate.id === id,
                );
                const rankMovement =
                  index < previousIndex
                    ? "up"
                    : index > previousIndex
                      ? "down"
                      : "same";
                return (
                  <li
                    key={id}
                    className={`${id === HUMAN_ID ? "is-you" : ""} ${
                      index === 0
                        ? "is-first-place"
                        : index === 1
                          ? "is-second-place"
                          : ""
                    }`}
                  >
                    <span>{index + 1}</span>
                    <div className="result-player-copy">
                      <div className="result-player-name">
                        <b>{player.name}</b>
                        <span
                          className={`result-rank-shift is-${rankMovement}`}
                        >
                          {ROLE_LABELS[player.role]} → {ROLE_LABELS[nextRole]}
                        </span>
                      </div>
                      <small>
                        {rankMovement === "up"
                          ? "서열 상승"
                          : rankMovement === "down"
                            ? "서열 하락"
                            : "서열 유지"}
                      </small>
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
              카드로 취급합니다. 하위 계급은 광대를 먼저 바치고, 상위 계급은 일반
              카드부터 돌려줍니다.
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
