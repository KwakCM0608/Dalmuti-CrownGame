import type {
  OnlineCard,
  OnlineCommand,
  OnlineEngineDeps,
  OnlineEvent,
  OnlineEventType,
  OnlinePhaseDurations,
  OnlinePlayerInput,
  OnlinePlayerState,
  OnlineRole,
  OnlineRoomState,
  OnlineSnapshot,
  OnlineTable,
  OnlineTaxExchange,
} from "./types.ts";
import {
  BOT_DIFFICULTIES,
  chooseBotCardIds,
  chooseBotRevolution,
  chooseBotTaxReturn,
  chooseFacedownRankSlot,
  type BotDifficulty,
} from "../bot-strategy.ts";

const MIN_PLAYERS = 4;
const MAX_PLAYERS = 8;
const MAX_EVENTS = 240;
const MAX_PROCESSED_COMMANDS = 512;
const PASS_ACTION_LOCK_MS = 1_500;
const PLAY_ACTION_LOCK_MS = 2_250;
const DALMUTI_ACTION_LOCK_MS = 3_300;
const TURN_DURATION_MS = 30_000;
const BOT_ACTION_DELAY_MS = 750;
const EMPTY_TABLE_TIMEOUT_CYCLE_MS =
  PASS_ACTION_LOCK_MS + TURN_DURATION_MS;

const DEFAULT_DURATIONS: OnlinePhaseDurations = {
  rankChoiceIntroMs: 3_300,
  rankRevealDelayMs: 1_500,
  rankRevealMs: 3_400,
  rankConfirmMs: 2_600,
  revealIntroMs: 2_400,
  handRevealMs: 1_400,
  revolutionDecisionMs: 20_000,
  revolutionIntroMs: 3_300,
  greatRevolutionSwapMs: 2_600,
  taxIntroMs: 2_400,
  taxSelectionMs: 45_000,
  taxTributeMs: 6_000,
  taxReturnMs: 6_000,
  playIntroMs: 2_600,
};

export class OnlineGameError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "OnlineGameError";
    this.code = code;
  }
}

function fail(code: string, message: string): never {
  throw new OnlineGameError(code, message);
}

function assertNow(now: number): void {
  if (!Number.isFinite(now) || now < 0) {
    fail("INVALID_TIME", "now must be a non-negative finite timestamp");
  }
}

function roleForIndex(index: number, total: number): OnlineRole {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === total - 2) return "lesser-peon";
  if (index === total - 1) return "great-peon";
  return "merchant";
}

function withAssignedRoles(players: OnlinePlayerState[]): OnlinePlayerState[] {
  return players.map((player, index) => ({
    ...player,
    role: roleForIndex(index, players.length),
  }));
}

function normalizePlayer(
  player: OnlinePlayerInput,
  role: OnlineRole,
  joinedAt: number,
): OnlinePlayerState {
  const id = player.id?.trim();
  const name = player.name?.trim();
  if (!id || id.length > 64) {
    fail("INVALID_PLAYER_ID", "player id must contain 1 to 64 characters");
  }
  if (!name || name.length > 30) {
    fail("INVALID_PLAYER_NAME", "player name must contain 1 to 30 characters");
  }

  const monogram =
    player.monogram?.trim().slice(0, 4) ?? Array.from(name).slice(0, 1).join("");

  return {
    id,
    name,
    monogram,
    isBot: false,
    botDifficulty: null,
    role,
    ready: false,
    connected: true,
    joinedAt,
    score: 0,
  };
}

function createBotPlayer(
  state: OnlineRoomState,
  joinedAt: number,
  botDifficulty: BotDifficulty,
): OnlinePlayerState {
  let botNumber = 1;
  while (
    state.players.some(
      (player) =>
        player.id === `bot-${botNumber}` ||
        player.name === `봇 ${botNumber}`,
    )
  ) {
    botNumber += 1;
  }

  return {
    id: `bot-${botNumber}`,
    name: `봇 ${botNumber}`,
    monogram: "AI",
    isBot: true,
    botDifficulty,
    role: "merchant",
    ready: true,
    connected: true,
    joinedAt,
    score: 0,
  };
}

function cloneRoom(state: OnlineRoomState): OnlineRoomState {
  return {
    ...state,
    turnDeadline:
      typeof state.turnDeadline === "number" &&
      Number.isFinite(state.turnDeadline)
        ? state.turnDeadline
        : null,
    players: state.players.map((player) => ({
      ...player,
      isBot: player.isBot === true,
      botDifficulty:
        player.isBot === true &&
        BOT_DIFFICULTIES.includes(
          player.botDifficulty as BotDifficulty,
        )
          ? (player.botDifficulty as BotDifficulty)
          : player.isBot === true
            ? "normal"
            : null,
      ready: player.isBot === true ? true : player.ready,
      connected: player.isBot === true ? true : player.connected,
    })),
    hands: Object.fromEntries(
      Object.entries(state.hands).map(([id, hand]) => [id, [...hand]]),
    ),
    table: state.table
      ? { ...state.table, cards: [...state.table.cards] }
      : null,
    passedPlayerIds: [...state.passedPlayerIds],
    finishOrder: [...state.finishOrder],
    rankSelection: state.rankSelection
      ? {
          ...state.rankSelection,
          cards: state.rankSelection.cards.map((card) => ({ ...card })),
        }
      : null,
    declaredRevolution: state.declaredRevolution
      ? { ...state.declaredRevolution }
      : null,
    taxExchanges: state.taxExchanges.map((exchange) => ({
      ...exchange,
      peonCardIds: [...exchange.peonCardIds],
      nobleCardIds: exchange.nobleCardIds
        ? [...exchange.nobleCardIds]
        : null,
    })),
    botActionAt:
      typeof state.botActionAt === "number" &&
      Number.isFinite(state.botActionAt)
        ? state.botActionAt
        : null,
    events: [...state.events],
    processedCommandIds: [...state.processedCommandIds],
    durations: { ...state.durations },
  };
}

function resolveDurations(
  base: OnlinePhaseDurations,
  deps?: OnlineEngineDeps,
): OnlinePhaseDurations {
  // Persisted rooms from an older deployment do not contain newly introduced
  // duration keys. A legacy timing profile must be replaced as one unit so a
  // room created before deployment still follows the same opening timeline as
  // quick match instead of mixing old and new animation lengths.
  const isLegacyProfile =
    !Number.isFinite(base.rankConfirmMs) ||
    !Number.isFinite(base.revolutionIntroMs) ||
    !Number.isFinite(base.greatRevolutionSwapMs);
  const persistedDurations = isLegacyProfile
    ? { ...DEFAULT_DURATIONS }
    : { ...DEFAULT_DURATIONS, ...base };
  const durations = {
    ...persistedDurations,
    ...deps?.durations,
  };
  for (const [key, value] of Object.entries(durations)) {
    if (!Number.isFinite(value) || value < 0) {
      fail("INVALID_DURATION", `${key} must be a non-negative finite number`);
    }
  }
  return durations;
}

function appendEvent(
  state: OnlineRoomState,
  type: OnlineEventType,
  at: number,
  payload: Record<string, unknown> = {},
  playerIds?: string[],
): void {
  const event: OnlineEvent = {
    seq: state.nextEventSeq,
    type,
    at,
    visibility: playerIds ? "private" : "public",
    ...(playerIds ? { playerIds: [...new Set(playerIds)] } : {}),
    payload,
  };
  state.nextEventSeq += 1;
  state.events = [...state.events, event].slice(-MAX_EVENTS);
}

function commit(
  state: OnlineRoomState,
  now: number,
  commandId?: string,
): OnlineRoomState {
  state.revision += 1;
  state.updatedAt = now;
  if (commandId) {
    state.processedCommandIds = [
      ...state.processedCommandIds,
      commandId,
    ].slice(-MAX_PROCESSED_COMMANDS);
  }
  return state;
}

function createDeck(): OnlineCard[] {
  const deck: OnlineCard[] = [];
  for (let rank = 1; rank <= 12; rank += 1) {
    for (let copy = 0; copy < rank; copy += 1) {
      deck.push({ id: `${rank}-${copy}`, rank });
    }
  }
  deck.push({ id: "joker-1", rank: 13 });
  deck.push({ id: "joker-2", rank: 13 });
  return deck;
}

function secureRandomInt(maxExclusive: number): number {
  if (!Number.isInteger(maxExclusive) || maxExclusive <= 0) {
    fail("INVALID_RANDOM_RANGE", "random range must be a positive integer");
  }
  if (!globalThis.crypto?.getRandomValues) {
    fail("SECURE_RANDOM_UNAVAILABLE", "secure random generation is unavailable");
  }

  const range = 0x1_0000_0000;
  const limit = range - (range % maxExclusive);
  const values = new Uint32Array(1);
  do {
    globalThis.crypto.getRandomValues(values);
  } while (values[0] >= limit);
  return values[0] % maxExclusive;
}

function shuffle(
  cards: OnlineCard[],
  injectedRandomInt?: (maxExclusive: number) => number,
): OnlineCard[] {
  const shuffled = [...cards];
  const randomInt = injectedRandomInt ?? secureRandomInt;

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const target = randomInt(index + 1);
    if (!Number.isInteger(target) || target < 0 || target > index) {
      fail(
        "INVALID_RANDOM_RESULT",
        `randomInt(${index + 1}) returned an out-of-range value`,
      );
    }
    [shuffled[index], shuffled[target]] = [
      shuffled[target],
      shuffled[index],
    ];
  }
  return shuffled;
}

function rankedDealCounts(totalCards: number, playerCount: number): number[] {
  const baseCount = Math.floor(totalCards / playerCount);
  const remainder = totalCards % playerCount;
  const bonusStart = playerCount - remainder;
  return Array.from(
    { length: playerCount },
    (_, index) => baseCount + (index >= bonusStart ? 1 : 0),
  );
}

function sortHand(cards: OnlineCard[]): OnlineCard[] {
  return [...cards].sort(
    (left, right) =>
      right.rank - left.rank || left.id.localeCompare(right.id),
  );
}

function deal(
  players: OnlinePlayerState[],
  randomInt?: (maxExclusive: number) => number,
): Record<string, OnlineCard[]> {
  const deck = shuffle(createDeck(), randomInt);
  const counts = rankedDealCounts(deck.length, players.length);
  const hands: Record<string, OnlineCard[]> = {};
  let cursor = 0;

  players.forEach((player, index) => {
    hands[player.id] = sortHand(deck.slice(cursor, cursor + counts[index]));
    cursor += counts[index];
  });
  return hands;
}

function clearSealedDeal(state: OnlineRoomState): void {
  state.hands = {};
  state.dealSealed = false;
}

function resetRoomToLobby(state: OnlineRoomState): void {
  state.players = withAssignedRoles(
    [...state.players].sort(
      (left, right) =>
        left.joinedAt - right.joinedAt || left.id.localeCompare(right.id),
    ),
  ).map((player) => ({
    ...player,
    ready: player.isBot,
    connected: player.isBot ? true : player.connected,
    score: 0,
  }));
  state.hands = {};
  state.dealSealed = false;
  state.phase = "lobby";
  state.phaseEndsAt = null;
  state.turnDeadline = null;
  state.round = 0;
  state.currentIndex = 0;
  state.table = null;
  state.lastPlayedId = null;
  state.passedPlayerIds = [];
  state.finishOrder = [];
  state.rankSelection = null;
  state.revolutionHolderId = null;
  state.declaredRevolution = null;
  state.taxExchanges = [];
  state.actionLockUntil = null;
  state.botActionAt = null;
  // A reset is a new client timeline. Keep the sequence monotonic so an
  // existing cursor still receives the reset event, but discard stale private
  // hand/tax events and animation instructions from the abandoned match.
  state.events = [];
}

function sealDeal(
  state: OnlineRoomState,
  at: number,
  deps?: OnlineEngineDeps,
): void {
  state.players = withAssignedRoles(state.players);
  state.hands = deal(state.players, deps?.randomInt);
  state.dealSealed = true;
  appendEvent(state, "DEAL_SEALED", at, {
    handCounts: Object.fromEntries(
      state.players.map((player) => [
        player.id,
        state.hands[player.id]?.length ?? 0,
      ]),
    ),
  });
}

function beginRankSelection(
  state: OnlineRoomState,
  at: number,
  deps?: OnlineEngineDeps,
): void {
  state.durations = resolveDurations(state.durations, deps);
  clearSealedDeal(state);

  const countdownEndsAt = at + state.durations.rankChoiceIntroMs;
  const countdownStartsAt = at;
  const shuffledRankCards = shuffle(
    state.players.map((_, index) => ({
      id: `rank-card-${index + 1}`,
      rank: index + 1,
    })),
    deps?.randomInt,
  );

  state.round = 1;
  state.phase = "rank-intro";
  state.phaseEndsAt = countdownEndsAt;
  state.turnDeadline = null;
  state.rankSelection = {
    cards: shuffledRankCards.map((card, slotIndex) => ({
      slotIndex,
      rank: card.rank,
      claimedByPlayerId: null,
      claimedAt: null,
    })),
    introStartedAt: at,
    countdownStartsAt,
    countdownEndsAt,
    revealAt: null,
    revealEndsAt: null,
  };
  state.currentIndex = 0;
  state.table = null;
  state.lastPlayedId = null;
  state.passedPlayerIds = [];
  state.finishOrder = [];
  state.revolutionHolderId = null;
  state.declaredRevolution = null;
  state.taxExchanges = [];
  state.actionLockUntil = null;
  state.botActionAt = null;

  appendEvent(state, "RANK_CHOICE_INTRO_STARTED", at, {
    playerCount: state.players.length,
    introStartedAt: at,
    countdownStartsAt,
    countdownEndsAt,
  });
}

function enterRankSelection(state: OnlineRoomState, at: number): void {
  const rankSelection = state.rankSelection;
  if (!rankSelection) {
    fail("ROOM_INVARIANT", "rank selection state is missing");
  }
  state.phase = "rank-selection";
  state.phaseEndsAt = null;
  appendEvent(state, "RANK_CHOICE_STARTED", at, {
    cards: rankSelection.cards.map((card) => ({
      slotIndex: card.slotIndex,
    })),
  });
  scheduleBotAction(state, at);
}

function allRankCardsChosen(state: OnlineRoomState): boolean {
  return (
    state.rankSelection?.cards.length === state.players.length &&
    state.rankSelection.cards.every(
      (card) => card.claimedByPlayerId !== null,
    )
  );
}

function lockRankChoices(state: OnlineRoomState, at: number): void {
  const rankSelection = state.rankSelection;
  if (!rankSelection || !allRankCardsChosen(state)) {
    fail("ROOM_INVARIANT", "rank choices cannot lock before every player chooses");
  }
  rankSelection.revealAt = at + state.durations.rankRevealDelayMs;
  state.phaseEndsAt = rankSelection.revealAt;
  state.botActionAt = null;
  appendEvent(state, "RANK_CHOICES_LOCKED", at, {
    revealAt: rankSelection.revealAt,
  });
}

function claimRankCard(
  state: OnlineRoomState,
  actorId: string,
  slotIndex: number,
  at: number,
  automatic: boolean,
): void {
  if (
    !Number.isInteger(slotIndex) ||
    slotIndex < 0 ||
    slotIndex >= state.players.length
  ) {
    fail("INVALID_RANK_SLOT", "slotIndex must identify an available rank card");
  }
  const rankSelection = state.rankSelection;
  if (!rankSelection) {
    fail("ROOM_INVARIANT", "rank selection state is missing");
  }
  if (
    rankSelection.cards.some(
      (card) => card.claimedByPlayerId === actorId,
    )
  ) {
    fail("RANK_ALREADY_CHOSEN", "each player may choose exactly one rank card");
  }
  const card = rankSelection.cards.find(
    (candidate) => candidate.slotIndex === slotIndex,
  );
  if (!card) {
    fail("INVALID_RANK_SLOT", "the selected rank card does not exist");
  }
  if (card.claimedByPlayerId) {
    fail("RANK_CARD_CLAIMED", "that rank card has already been chosen");
  }

  card.claimedByPlayerId = actorId;
  card.claimedAt = at;
  appendEvent(state, "RANK_CARD_CHOSEN", at, {
    slotIndex: card.slotIndex,
    playerId: actorId,
    automatic,
  });

  const remainingCards = rankSelection.cards.filter(
    (candidate) => candidate.claimedByPlayerId === null,
  );
  const assignedPlayerIds = new Set(
    rankSelection.cards.flatMap((candidate) =>
      candidate.claimedByPlayerId
        ? [candidate.claimedByPlayerId]
        : [],
    ),
  );
  const remainingPlayers = state.players.filter(
    (player) => !assignedPlayerIds.has(player.id),
  );
  if (remainingCards.length === 1 && remainingPlayers.length === 1) {
    const finalCard = remainingCards[0];
    const finalPlayer = remainingPlayers[0];
    finalCard.claimedByPlayerId = finalPlayer.id;
    finalCard.claimedAt = at;
    appendEvent(state, "RANK_CARD_CHOSEN", at, {
      slotIndex: finalCard.slotIndex,
      playerId: finalPlayer.id,
      automatic: true,
    });
  }

  if (allRankCardsChosen(state)) {
    lockRankChoices(state, at);
  } else {
    scheduleBotAction(state, at);
  }
}

function enterRankReveal(state: OnlineRoomState, at: number): void {
  const rankSelection = state.rankSelection;
  if (!rankSelection || !allRankCardsChosen(state)) {
    fail("ROOM_INVARIANT", "rank cards cannot reveal before every player chooses");
  }

  rankSelection.revealAt = at;
  rankSelection.revealEndsAt = at + state.durations.rankRevealMs;
  state.phase = "rank-reveal";
  state.phaseEndsAt = rankSelection.revealEndsAt;
  state.botActionAt = null;

  const assignments = [...rankSelection.cards]
    .sort((left, right) => left.slotIndex - right.slotIndex)
    .map((card) => ({
      slotIndex: card.slotIndex,
      playerId: card.claimedByPlayerId,
      rank: card.rank,
    }));
  appendEvent(state, "RANK_CARDS_REVEALED", at, {
    cards: assignments,
    endsAt: state.phaseEndsAt,
  });
}

function enterRankConfirm(
  state: OnlineRoomState,
  at: number,
): void {
  state.phase = "rank-confirm";
  state.phaseEndsAt = at + state.durations.rankConfirmMs;
  state.botActionAt = null;
  appendEvent(state, "RANK_CONFIRM_STARTED", at, {
    playerIds: state.players.map((player) => player.id),
    endsAt: state.phaseEndsAt,
  });
}

function finalizeRankOrder(state: OnlineRoomState, at: number): void {
  const rankSelection = state.rankSelection;
  if (!rankSelection || !allRankCardsChosen(state)) {
    fail("ROOM_INVARIANT", "rank order cannot finalize before every player chooses");
  }
  const playerById = new Map(
    state.players.map((player) => [player.id, player]),
  );
  const orderedCards = [...rankSelection.cards].sort(
    (left, right) => left.rank - right.rank,
  );
  const assignedPlayerIds = orderedCards.map(
    (card) => card.claimedByPlayerId,
  );
  if (
    assignedPlayerIds.some((playerId) => !playerId) ||
    new Set(assignedPlayerIds).size !== state.players.length
  ) {
    fail("ROOM_INVARIANT", "rank card claims do not assign every player once");
  }

  state.players = withAssignedRoles(
    assignedPlayerIds.map((playerId) => {
      const player = playerById.get(playerId!);
      if (!player) {
        fail("ROOM_INVARIANT", "a rank card belongs to an unknown player");
      }
      return player;
    }),
  );
  const assignments = [...rankSelection.cards]
    .sort((left, right) => left.slotIndex - right.slotIndex)
    .map((card) => ({
      slotIndex: card.slotIndex,
      playerId: card.claimedByPlayerId,
      rank: card.rank,
    }));
  appendEvent(state, "RANK_ORDER_ASSIGNED", at, {
    playerIds: state.players.map((player) => player.id),
    assignments,
  });
}

function taxationPriority(card: OnlineCard): number {
  return card.rank === 13 ? 0 : card.rank;
}

function selectPeonTaxCards(
  hand: OnlineCard[],
  count: number,
): OnlineCard[] {
  return [...hand]
    .sort(
      (left, right) =>
        taxationPriority(left) - taxationPriority(right) ||
        left.id.localeCompare(right.id),
    )
    .slice(0, count);
}

function selectAutomaticNobleReturns(
  hand: OnlineCard[],
  count: number,
): OnlineCard[] {
  return [...hand]
    .sort(
      (left, right) =>
        taxationPriority(right) - taxationPriority(left) ||
        left.id.localeCompare(right.id),
    )
    .slice(0, count);
}

function removeCardIds(hand: OnlineCard[], cardIds: string[]): OnlineCard[] {
  const removing = new Set(cardIds);
  return hand.filter((card) => !removing.has(card.id));
}

function cardsByIds(
  hand: OnlineCard[],
  cardIds: string[],
): OnlineCard[] {
  const byId = new Map(hand.map((card) => [card.id, card]));
  return cardIds.map((id) => {
    const card = byId.get(id);
    if (!card) {
      fail("CARD_NOT_OWNED", `card ${id} is not in the player's hand`);
    }
    return card;
  });
}

function assertUniqueCardIds(cardIds: string[]): void {
  if (new Set(cardIds).size !== cardIds.length) {
    fail("DUPLICATE_CARD_ID", "a card can only be selected once");
  }
}

function normalizedSet(
  cards: OnlineCard[],
): { rank: number; count: number } | null {
  if (cards.length === 0) return null;
  const normalCards = cards.filter((card) => card.rank !== 13);
  if (normalCards.length === 0) {
    return cards.length === 1 ? { rank: 13, count: 1 } : null;
  }
  const rank = normalCards[0].rank;
  if (normalCards.some((card) => card.rank !== rank)) return null;
  return { rank, count: cards.length };
}

function findRole(
  state: OnlineRoomState,
  role: OnlineRole,
): OnlinePlayerState {
  const player = state.players.find((candidate) => candidate.role === role);
  if (!player) fail("ROOM_INVARIANT", `room has no ${role}`);
  return player;
}

function startRound(
  state: OnlineRoomState,
  round: number,
  at: number,
  deps?: OnlineEngineDeps,
  useSealedDeal = false,
): void {
  state.durations = resolveDurations(state.durations, deps);
  state.players = withAssignedRoles(state.players).map((player) => ({
    ...player,
    ready: true,
  }));
  if (round > 1) state.rankSelection = null;
  if (!useSealedDeal || !state.dealSealed) {
    sealDeal(state, at, deps);
  }
  state.round = round;
  state.phase = "reveal-intro";
  state.phaseEndsAt = at + state.durations.revealIntroMs;
  state.turnDeadline = null;
  state.currentIndex = 0;
  state.table = null;
  state.lastPlayedId = null;
  state.passedPlayerIds = [];
  state.finishOrder = [];
  state.revolutionHolderId = null;
  state.declaredRevolution = null;
  state.taxExchanges = [];
  state.actionLockUntil = null;
  state.botActionAt = null;

  appendEvent(state, "MATCH_STARTED", at, {
    round,
    playerIds: state.players.map((player) => player.id),
    handCounts: Object.fromEntries(
      state.players.map((player) => [player.id, state.hands[player.id].length]),
    ),
    endsAt: state.phaseEndsAt,
  });
}

function enterHandReveal(state: OnlineRoomState, at: number): void {
  state.phase = "hand-reveal";
  state.phaseEndsAt = at + state.durations.handRevealMs;
  appendEvent(state, "HAND_REVEAL_STARTED", at, {
    endsAt: state.phaseEndsAt,
  });

  for (const player of state.players) {
    appendEvent(
      state,
      "HAND_REVEALED",
      at,
      { cards: [...state.hands[player.id]], endsAt: state.phaseEndsAt },
      [player.id],
    );
  }
}

function enterRevolutionDecision(
  state: OnlineRoomState,
  holderId: string,
  at: number,
): void {
  state.phase = "revolution";
  state.phaseEndsAt = at + state.durations.revolutionDecisionMs;
  state.revolutionHolderId = holderId;
  appendEvent(state, "REVOLUTION_DECISION_STARTED", at, {
    endsAt: state.phaseEndsAt,
  });
  appendEvent(
    state,
    "REVOLUTION_DECISION_STARTED",
    at,
    { holderId, canChoose: true, endsAt: state.phaseEndsAt },
    [holderId],
  );
  scheduleBotAction(state, at);
}

function buildTaxExchanges(state: OnlineRoomState): OnlineTaxExchange[] {
  const pairs: Array<{
    nobleRole: OnlineRole;
    peonRole: OnlineRole;
    count: number;
  }> = [
    {
      nobleRole: "great-dalmuti",
      peonRole: "great-peon",
      count: 2,
    },
    {
      nobleRole: "lesser-dalmuti",
      peonRole: "lesser-peon",
      count: 1,
    },
  ];

  return pairs.map(({ nobleRole, peonRole, count }) => {
    const noble = findRole(state, nobleRole);
    const peon = findRole(state, peonRole);
    const peonCards = selectPeonTaxCards(state.hands[peon.id], count);
    if (peonCards.length !== count) {
      fail("ROOM_INVARIANT", "a peon does not have enough cards for tax");
    }
    return {
      nobleId: noble.id,
      peonId: peon.id,
      count,
      peonCardIds: peonCards.map((card) => card.id),
      nobleCardIds: null,
    };
  });
}

function publicTaxRoutes(
  exchanges: OnlineTaxExchange[],
  direction: "tribute" | "return" = "tribute",
) {
  return exchanges.map((exchange) => ({
    nobleId: exchange.nobleId,
    peonId: exchange.peonId,
    fromPlayerId:
      direction === "tribute" ? exchange.peonId : exchange.nobleId,
    toPlayerId:
      direction === "tribute" ? exchange.nobleId : exchange.peonId,
    count: exchange.count,
  }));
}

function enterTaxIntro(state: OnlineRoomState, at: number): void {
  state.revolutionHolderId = null;
  state.botActionAt = null;
  state.taxExchanges = buildTaxExchanges(state);
  state.phase = "tax-intro";
  state.phaseEndsAt = at + state.durations.taxIntroMs;
  appendEvent(state, "TAX_INTRO_STARTED", at, {
    routes: publicTaxRoutes(state.taxExchanges),
    endsAt: state.phaseEndsAt,
  });
}

function enterTaxSelection(state: OnlineRoomState, at: number): void {
  state.phase = "tax-selection";
  state.phaseEndsAt = null;
  state.botActionAt = null;
  autoSelectTaxReturns(state, at, true);

  if (allTaxReturnsSelected(state)) {
    enterTaxTribute(state, at);
    return;
  }

  const waitingForPlayerIds = state.taxExchanges
    .filter((exchange) => !exchange.nobleCardIds)
    .map((exchange) => exchange.nobleId);
  state.phaseEndsAt = at + state.durations.taxSelectionMs;
  appendEvent(state, "TAX_SELECTION_STARTED", at, {
    waitingForPlayerIds,
    endsAt: state.phaseEndsAt,
  });

  for (const exchange of state.taxExchanges.filter(
    (candidate) => !candidate.nobleCardIds,
  )) {
    appendEvent(
      state,
      "TAX_SELECTION_STARTED",
      at,
      {
        requiredReturnCount: exchange.count,
        endsAt: state.phaseEndsAt,
      },
      [exchange.nobleId],
    );
  }
}

function allTaxReturnsSelected(state: OnlineRoomState): boolean {
  return state.taxExchanges.every(
    (exchange) => exchange.nobleCardIds?.length === exchange.count,
  );
}

function selectTaxReturn(
  state: OnlineRoomState,
  actorId: string,
  cardIds: string[],
  at: number,
  automatic: boolean,
): void {
  const exchangeIndex = state.taxExchanges.findIndex(
    (exchange) => exchange.nobleId === actorId,
  );
  if (exchangeIndex < 0) {
    fail("NOT_TAX_NOBLE", "this player does not choose a tax return");
  }
  const exchange = state.taxExchanges[exchangeIndex];
  if (exchange.nobleCardIds) {
    fail("TAX_RETURN_ALREADY_SELECTED", "the return cards are already locked");
  }
  if (!Array.isArray(cardIds) || cardIds.length !== exchange.count) {
    fail(
      "WRONG_TAX_CARD_COUNT",
      `exactly ${exchange.count} return cards are required`,
    );
  }
  assertUniqueCardIds(cardIds);
  cardsByIds(state.hands[actorId], cardIds);
  state.taxExchanges[exchangeIndex] = {
    ...exchange,
    nobleCardIds: [...cardIds],
  };
  appendEvent(
    state,
    "TAX_RETURN_SELECTED",
    at,
    { cardIds: [...cardIds], automatic },
    [actorId],
  );
  if (allTaxReturnsSelected(state)) {
    enterTaxTribute(state, at);
  } else {
    scheduleBotAction(state, at);
  }
}

function enterTaxTribute(state: OnlineRoomState, at: number): void {
  if (!allTaxReturnsSelected(state)) {
    fail("ROOM_INVARIANT", "tax tribute cannot start before all returns are selected");
  }

  for (const exchange of state.taxExchanges) {
    const peonHand = state.hands[exchange.peonId];
    const tributeCards = cardsByIds(peonHand, exchange.peonCardIds);
    state.hands[exchange.peonId] = sortHand(
      removeCardIds(peonHand, exchange.peonCardIds),
    );
    state.hands[exchange.nobleId] = sortHand([
      ...state.hands[exchange.nobleId],
      ...tributeCards,
    ]);
  }

  state.phase = "tax-tribute";
  state.phaseEndsAt = at + state.durations.taxTributeMs;
  state.botActionAt = null;
  appendEvent(state, "TAX_TRIBUTE_STARTED", at, {
    routes: publicTaxRoutes(state.taxExchanges, "tribute"),
    endsAt: state.phaseEndsAt,
  });

  for (const exchange of state.taxExchanges) {
    const cards = cardsByIds(
      state.hands[exchange.nobleId],
      exchange.peonCardIds,
    );
    appendEvent(
      state,
      "TAX_TRIBUTE",
      at,
      {
        fromPlayerId: exchange.peonId,
        toPlayerId: exchange.nobleId,
        cards,
        routes: publicTaxRoutes(state.taxExchanges, "tribute"),
        endsAt: state.phaseEndsAt,
      },
      [exchange.peonId, exchange.nobleId],
    );
  }
}

function enterTaxReturn(state: OnlineRoomState, at: number): void {
  for (const exchange of state.taxExchanges) {
    const cardIds = exchange.nobleCardIds;
    if (!cardIds || cardIds.length !== exchange.count) {
      fail("ROOM_INVARIANT", "a noble return selection is missing");
    }
    const nobleHand = state.hands[exchange.nobleId];
    const returnCards = cardsByIds(nobleHand, cardIds);
    state.hands[exchange.nobleId] = sortHand(
      removeCardIds(nobleHand, cardIds),
    );
    state.hands[exchange.peonId] = sortHand([
      ...state.hands[exchange.peonId],
      ...returnCards,
    ]);
  }

  state.phase = "tax-return";
  state.phaseEndsAt = at + state.durations.taxReturnMs;
  state.botActionAt = null;
  appendEvent(state, "TAX_RETURN_STARTED", at, {
    routes: publicTaxRoutes(state.taxExchanges, "return"),
    endsAt: state.phaseEndsAt,
  });

  for (const exchange of state.taxExchanges) {
    const cards = cardsByIds(
      state.hands[exchange.peonId],
      exchange.nobleCardIds ?? [],
    );
    appendEvent(
      state,
      "TAX_RETURN",
      at,
      {
        fromPlayerId: exchange.nobleId,
        toPlayerId: exchange.peonId,
        cards,
        routes: publicTaxRoutes(state.taxExchanges, "return"),
        endsAt: state.phaseEndsAt,
      },
      [exchange.peonId, exchange.nobleId],
    );
  }
}

function enterPlayIntro(state: OnlineRoomState, at: number): void {
  state.phase = "play-intro";
  state.phaseEndsAt = at + state.durations.playIntroMs;
  state.turnDeadline = null;
  state.botActionAt = null;
  state.taxExchanges = [];
  state.revolutionHolderId = null;
  state.currentIndex = 0;
  appendEvent(state, "PLAY_INTRO_STARTED", at, {
    round: state.round,
    firstPlayerId: state.players[0].id,
    endsAt: state.phaseEndsAt,
  });
}

function enterRevolutionIntro(
  state: OnlineRoomState,
  holderId: string,
  kind: "great" | "normal",
  at: number,
): void {
  state.phase = "revolution-intro";
  state.phaseEndsAt = at + state.durations.revolutionIntroMs;
  state.turnDeadline = null;
  state.botActionAt = null;
  state.revolutionHolderId = null;
  appendEvent(state, "REVOLUTION_DECLARED", at, {
    round: state.round,
    playerId: holderId,
    kind,
    endsAt: state.phaseEndsAt,
  });
  appendEvent(state, "REVOLUTION_INTRO_STARTED", at, {
    round: state.round,
    playerId: holderId,
    kind,
    endsAt: state.phaseEndsAt,
  });
}

function enterGreatRevolutionSwap(
  state: OnlineRoomState,
  at: number,
): void {
  const declaration = state.declaredRevolution;
  if (!declaration || declaration.kind !== "great-revolution") {
    fail("ROOM_INVARIANT", "great revolution rank swap has no declaration");
  }
  const previousPlayerIds = state.players.map((player) => player.id);
  state.players = withAssignedRoles([...state.players].reverse());
  state.currentIndex = 0;
  state.phase = "great-revolution-swap";
  state.phaseEndsAt = at + state.durations.greatRevolutionSwapMs;
  state.turnDeadline = null;
  state.botActionAt = null;
  appendEvent(state, "GREAT_REVOLUTION_RANK_SWAP_STARTED", at, {
    round: state.round,
    playerId: declaration.playerId,
    previousPlayerIds,
    playerIds: state.players.map((player) => player.id),
    endsAt: state.phaseEndsAt,
  });
}

function enterPlaying(state: OnlineRoomState, at: number): void {
  state.phase = "playing";
  state.phaseEndsAt = null;
  state.actionLockUntil = null;
  state.currentIndex = 0;
  state.turnDeadline = at + TURN_DURATION_MS;
  appendEvent(state, "TURN_STARTED", at, {
    playerId: state.players[state.currentIndex].id,
    endsAt: state.turnDeadline,
  });
  scheduleBotAction(state, at);
}

function chooseRevolution(
  state: OnlineRoomState,
  declare: boolean,
  at: number,
): void {
  const holderId = state.revolutionHolderId;
  if (!holderId) fail("ROOM_INVARIANT", "revolution has no holder");
  const holder = state.players.find((player) => player.id === holderId);
  if (!holder) fail("ROOM_INVARIANT", "revolution holder has left the room");

  if (!declare) {
    appendEvent(state, "REVOLUTION_DECLINED", at, {});
    enterTaxIntro(state, at);
    return;
  }

  const isGreatRevolution = holder.role === "great-peon";
  state.declaredRevolution = {
    round: state.round,
    playerId: holderId,
    kind: isGreatRevolution ? "great-revolution" : "revolution",
  };
  enterRevolutionIntro(
    state,
    holderId,
    isGreatRevolution ? "great" : "normal",
    at,
  );
}

function autoSelectTaxReturns(
  state: OnlineRoomState,
  at: number,
  botNoblesOnly = false,
): void {
  const botPlayerIds = botNoblesOnly
    ? new Set(
        state.players
          .filter((player) => player.isBot)
          .map((player) => player.id),
      )
    : null;
  state.taxExchanges = state.taxExchanges.map((exchange) => {
    if (
      exchange.nobleCardIds?.length === exchange.count ||
      (botPlayerIds && !botPlayerIds.has(exchange.nobleId))
    ) {
      return exchange;
    }
    const noble = state.players.find(
      (player) => player.id === exchange.nobleId,
    );
    const cardIds = noble?.isBot
      ? chooseBotTaxReturn(
          state.hands[exchange.nobleId],
          exchange.count,
          noble.botDifficulty ?? "normal",
        ).cardIds
      : selectAutomaticNobleReturns(
          state.hands[exchange.nobleId],
          exchange.count,
        ).map((card) => card.id);
    const cards = cardsByIds(state.hands[exchange.nobleId], cardIds);
    appendEvent(
      state,
      "TAX_RETURN_SELECTED",
      at,
      { cardIds: cards.map((card) => card.id), automatic: true },
      [exchange.nobleId],
    );
    return {
      ...exchange,
      nobleCardIds: cards.map((card) => card.id),
    };
  });
}

function advanceOneTimedPhase(
  state: OnlineRoomState,
  at: number,
  deps?: OnlineEngineDeps,
): void {
  switch (state.phase) {
    case "rank-intro":
      enterRankSelection(state, at);
      return;
    case "rank-selection":
      enterRankReveal(state, at);
      return;
    case "rank-reveal":
      enterRankConfirm(state, at);
      return;
    case "rank-confirm":
      finalizeRankOrder(state, at);
      startRound(state, 1, at, deps);
      return;
    case "reveal-intro":
      enterHandReveal(state, at);
      return;
    case "hand-reveal": {
      const holder = state.players.find(
        (player) =>
          state.hands[player.id].filter((card) => card.rank === 13).length ===
          2,
      );
      if (!holder) {
        enterTaxIntro(state, at);
      } else if (holder.isBot) {
        state.revolutionHolderId = holder.id;
        const decision = chooseBotRevolution(
          {
            hand: state.hands[holder.id] ?? [],
            role: holder.role,
            playerCount: state.players.length,
          },
          holder.botDifficulty ?? "normal",
        );
        chooseRevolution(state, decision.declare, at);
      } else {
        enterRevolutionDecision(state, holder.id, at);
      }
      return;
    }
    case "revolution":
      chooseRevolution(state, false, at);
      return;
    case "revolution-intro":
      if (state.declaredRevolution?.kind === "great-revolution") {
        enterGreatRevolutionSwap(state, at);
      } else {
        enterPlayIntro(state, at);
      }
      return;
    case "great-revolution-swap":
      enterPlayIntro(state, at);
      return;
    case "tax-intro":
      enterTaxSelection(state, at);
      return;
    case "tax-selection":
      autoSelectTaxReturns(state, at);
      enterTaxTribute(state, at);
      return;
    case "tax-tribute":
      enterTaxReturn(state, at);
      return;
    case "tax-return":
      enterPlayIntro(state, at);
      return;
    case "play-intro":
      enterPlaying(state, at);
      return;
    default:
      fail("ROOM_INVARIANT", `${state.phase} is not a timed phase`);
  }
}

function nextActiveIndex(state: OnlineRoomState, fromIndex: number): number {
  for (let step = 1; step <= state.players.length; step += 1) {
    const index = (fromIndex + step + state.players.length) % state.players.length;
    if ((state.hands[state.players[index].id]?.length ?? 0) > 0) return index;
  }
  return fromIndex;
}

function addTurnStartedEvent(state: OnlineRoomState, at: number): void {
  const player = state.players[state.currentIndex];
  if (!player) {
    state.turnDeadline = null;
    state.botActionAt = null;
    return;
  }
  state.turnDeadline = at + TURN_DURATION_MS;
  appendEvent(state, "TURN_STARTED", at, {
    playerId: player.id,
    endsAt: state.turnDeadline,
  });
  scheduleBotAction(state, at);
}

function finishRoundIfNeeded(
  state: OnlineRoomState,
  at: number,
): boolean {
  if (state.finishOrder.length < state.players.length - 1) return false;
  const last = state.players.find(
    (player) => !state.finishOrder.includes(player.id),
  );
  if (last) state.finishOrder.push(last.id);
  state.phase = "round-end";
  state.phaseEndsAt = null;
  state.turnDeadline = null;
  state.botActionAt = null;
  appendEvent(state, "ROUND_ENDED", at, {
    round: state.round,
    finishOrder: [...state.finishOrder],
  });
  return true;
}

function handlePlayCards(
  state: OnlineRoomState,
  actorId: string,
  cardIds: string[],
  at: number,
): void {
  if (state.phase !== "playing") {
    fail("WRONG_PHASE", "cards can only be played during the playing phase");
  }
  if (state.actionLockUntil !== null && at < state.actionLockUntil) {
    fail("ACTION_IN_PROGRESS", "the previous table animation is still playing");
  }
  const current = state.players[state.currentIndex];
  if (current?.id !== actorId) fail("NOT_YOUR_TURN", "it is not your turn");
  if (!Array.isArray(cardIds) || cardIds.length === 0) {
    fail("INVALID_CARD_SET", "at least one card must be selected");
  }
  assertUniqueCardIds(cardIds);

  const hand = state.hands[actorId];
  const cards = cardsByIds(hand, cardIds);
  const normalized = normalizedSet(cards);
  if (!normalized) {
    fail(
      "INVALID_CARD_SET",
      "cards must share one rank; jokers may accompany that rank",
    );
  }
  if (state.table) {
    if (normalized.count !== state.table.count) {
      fail("WRONG_CARD_COUNT", `exactly ${state.table.count} cards are required`);
    }
    if (normalized.rank >= state.table.rank) {
      fail("CARD_NOT_STRONG_ENOUGH", "a lower numbered rank is required");
    }
  }

  const previousTable = state.table
    ? { ...state.table, cards: [...state.table.cards] }
    : null;
  state.hands[actorId] = sortHand(removeCardIds(hand, cardIds));
  const table: OnlineTable = {
    rank: normalized.rank,
    count: normalized.count,
    playerId: actorId,
    cards: [...cards],
  };
  state.table = table;
  state.lastPlayedId = actorId;
  state.passedPlayerIds = [];
  appendEvent(state, "CARDS_PLAYED", at, {
    ...table,
    previousTable,
  });
  const isDalmutiEffect = normalized.rank === 1;
  state.actionLockUntil =
    at + (isDalmutiEffect ? DALMUTI_ACTION_LOCK_MS : PLAY_ACTION_LOCK_MS);
  const automaticallyPassedPlayerIds = isDalmutiEffect
    ? state.players
        .filter(
          (player) =>
            player.id !== actorId && state.hands[player.id].length > 0,
        )
        .map((player) => player.id)
    : [];

  if (isDalmutiEffect) {
    appendEvent(state, "DALMUTI_EFFECT", at, {
      playerId: actorId,
      cards: [...cards],
      rank: normalized.rank,
      count: normalized.count,
      autoPassedPlayerIds: automaticallyPassedPlayerIds,
      previousTable,
    });
    for (const playerId of automaticallyPassedPlayerIds) {
      appendEvent(state, "PLAYER_PASSED", at, {
        playerId,
        automatic: true,
        reason: "dalmuti",
      });
    }
  }

  if (state.hands[actorId].length === 0) {
    state.finishOrder.push(actorId);
    const place = state.finishOrder.length;
    const awardedScore = state.players.length - place;
    state.players = state.players.map((player) =>
      player.id === actorId
        ? { ...player, score: player.score + awardedScore }
        : player,
    );
    appendEvent(state, "PLAYER_FINISHED", at, {
      playerId: actorId,
      place,
      awardedScore,
    });
  }

  if (finishRoundIfNeeded(state, at)) return;
  if (isDalmutiEffect) {
    const actorIndex = state.players.findIndex(
      (player) => player.id === actorId,
    );
    const actorStillActive = state.hands[actorId].length > 0;
    state.table = null;
    state.passedPlayerIds = [];
    state.currentIndex = actorStillActive
      ? actorIndex
      : nextActiveIndex(state, actorIndex);
    appendEvent(state, "TRICK_CLEARED", at, {
      previousLeaderId: actorId,
      nextPlayerId: state.players[state.currentIndex].id,
      reason: "dalmuti",
      automatic: true,
    });
    addTurnStartedEvent(state, state.actionLockUntil);
    return;
  }
  state.currentIndex = nextActiveIndex(state, state.currentIndex);
  addTurnStartedEvent(state, state.actionLockUntil);
}

function handlePass(
  state: OnlineRoomState,
  actorId: string,
  at: number,
  options: {
    allowEmptyTable?: boolean;
    automatic?: boolean;
    reason?: string;
  } = {},
): void {
  if (state.phase !== "playing") {
    fail("WRONG_PHASE", "passing is only allowed during the playing phase");
  }
  if (state.actionLockUntil !== null && at < state.actionLockUntil) {
    fail("ACTION_IN_PROGRESS", "the previous table animation is still playing");
  }
  const current = state.players[state.currentIndex];
  if (current?.id !== actorId) fail("NOT_YOUR_TURN", "it is not your turn");
  if (!state.table && !options.allowEmptyTable) {
    fail("CANNOT_PASS", "the leading player cannot pass");
  }

  state.passedPlayerIds = [
    ...new Set([...state.passedPlayerIds, actorId]),
  ];
  appendEvent(state, "PLAYER_PASSED", at, {
    playerId: actorId,
    ...(options.automatic ? { automatic: true } : {}),
    ...(options.reason ? { reason: options.reason } : {}),
  });
  state.actionLockUntil = at + PASS_ACTION_LOCK_MS;

  if (!state.table) {
    state.currentIndex = nextActiveIndex(state, state.currentIndex);
    addTurnStartedEvent(state, state.actionLockUntil);
    return;
  }

  const active = state.players.filter(
    (player) => state.hands[player.id].length > 0,
  );
  const requiredToPass = active.filter(
    (player) => player.id !== state.lastPlayedId,
  );
  const trickIsOver = requiredToPass.every((player) =>
    state.passedPlayerIds.includes(player.id),
  );

  if (trickIsOver) {
    const previousLeaderId = state.lastPlayedId;
    const previousLeaderIndex = state.players.findIndex(
      (player) => player.id === previousLeaderId,
    );
    const leaderStillActive =
      previousLeaderIndex >= 0 &&
      state.hands[state.players[previousLeaderIndex].id].length > 0;
    state.table = null;
    state.passedPlayerIds = [];
    state.currentIndex = leaderStillActive
      ? previousLeaderIndex
      : nextActiveIndex(state, previousLeaderIndex);
    appendEvent(state, "TRICK_CLEARED", at, {
      previousLeaderId,
      nextPlayerId: state.players[state.currentIndex].id,
    });
    addTurnStartedEvent(state, state.actionLockUntil);
    return;
  }

  state.currentIndex = nextActiveIndex(state, state.currentIndex);
  addTurnStartedEvent(state, state.actionLockUntil);
}

function handleTurnTimeout(state: OnlineRoomState, at: number): void {
  if (state.phase !== "playing") {
    state.turnDeadline = null;
    return;
  }
  const current = state.players[state.currentIndex];
  if (!current || (state.hands[current.id]?.length ?? 0) === 0) {
    state.currentIndex = nextActiveIndex(state, state.currentIndex);
    addTurnStartedEvent(state, at);
    return;
  }
  handlePass(state, current.id, at, {
    allowEmptyTable: true,
    automatic: true,
    reason: "timeout",
  });
}

function pendingBotPlayer(
  state: OnlineRoomState,
): OnlinePlayerState | null {
  if (state.phase === "rank-selection" && state.phaseEndsAt === null) {
    const claimedPlayerIds = new Set(
      state.rankSelection?.cards.flatMap((card) =>
        card.claimedByPlayerId ? [card.claimedByPlayerId] : [],
      ) ?? [],
    );
    return (
      state.players.find(
        (player) => player.isBot && !claimedPlayerIds.has(player.id),
      ) ?? null
    );
  }
  if (state.phase === "revolution") {
    return (
      state.players.find(
        (player) =>
          player.isBot && player.id === state.revolutionHolderId,
      ) ?? null
    );
  }
  if (state.phase === "tax-selection") {
    const pendingBotExchange = state.taxExchanges.find(
      (exchange) =>
        !exchange.nobleCardIds &&
        state.players.some(
          (player) => player.isBot && player.id === exchange.nobleId,
        ),
    );
    return pendingBotExchange
      ? state.players.find(
          (player) => player.id === pendingBotExchange.nobleId,
        ) ?? null
      : null;
  }
  if (state.phase === "playing") {
    const current = state.players[state.currentIndex];
    return current?.isBot && (state.hands[current.id]?.length ?? 0) > 0
      ? current
      : null;
  }
  return null;
}

function scheduleBotAction(state: OnlineRoomState, at: number): void {
  const bot = pendingBotPlayer(state);
  if (!bot) {
    state.botActionAt = null;
    return;
  }
  const animationUnlock =
    state.phase === "playing" &&
    typeof state.actionLockUntil === "number" &&
    Number.isFinite(state.actionLockUntil)
      ? state.actionLockUntil
      : at;
  state.botActionAt =
    Math.max(at, animationUnlock) + BOT_ACTION_DELAY_MS;
}

function chooseOnlineBotCards(
  state: OnlineRoomState,
  playerId: string,
): string[] | null {
  const bot = state.players.find((player) => player.id === playerId);
  const roundStartedAt = [...state.events]
    .reverse()
    .find((event) => event.type === "MATCH_STARTED")?.at;
  const publicCounts = new Map<number, number>();
  for (const event of state.events) {
    if (
      event.type !== "CARDS_PLAYED" ||
      (roundStartedAt !== undefined && event.at < roundStartedAt)
    ) {
      continue;
    }
    const cards = Array.isArray(event.payload.cards)
      ? (event.payload.cards as OnlineCard[])
      : [];
    for (const card of cards) {
      publicCounts.set(card.rank, (publicCounts.get(card.rank) ?? 0) + 1);
    }
  }

  return chooseBotCardIds(
    {
      actorId: playerId,
      hand: state.hands[playerId] ?? [],
      table: state.table
        ? {
            rank: state.table.rank,
            count: state.table.count,
            playerId: state.table.playerId,
          }
        : null,
      players: state.players.map((player) => ({
        id: player.id,
        handCount: state.hands[player.id]?.length ?? 0,
        finished: state.finishOrder.includes(player.id),
      })),
      passedPlayerIds: state.passedPlayerIds,
      publicPlayedCards: [...publicCounts].map(([rank, count]) => ({
        rank,
        count,
      })),
    },
    bot?.botDifficulty ?? "normal",
  );
}

function performBotAction(
  state: OnlineRoomState,
  at: number,
  deps?: OnlineEngineDeps,
): void {
  const bot = pendingBotPlayer(state);
  state.botActionAt = null;
  if (!bot) return;

  if (state.phase === "rank-selection") {
    const openSlots =
      state.rankSelection?.cards.filter(
        (card) => card.claimedByPlayerId === null,
      ) ?? [];
    if (!openSlots.length) return;
    const randomInt = deps?.randomInt ?? secureRandomInt;
    const selectedSlot = chooseFacedownRankSlot(
      openSlots.map((card) => card.slotIndex),
      randomInt,
    );
    const selectedCard = openSlots.find(
      (card) => card.slotIndex === selectedSlot,
    );
    if (!selectedCard) return;
    claimRankCard(state, bot.id, selectedCard.slotIndex, at, true);
    return;
  }

  if (state.phase === "revolution") {
    const decision = chooseBotRevolution(
      {
        hand: state.hands[bot.id] ?? [],
        role: bot.role,
        playerCount: state.players.length,
      },
      bot.botDifficulty ?? "normal",
    );
    chooseRevolution(state, decision.declare, at);
    return;
  }

  if (state.phase === "tax-selection") {
    const exchange = state.taxExchanges.find(
      (candidate) =>
        candidate.nobleId === bot.id && !candidate.nobleCardIds,
    );
    if (!exchange) return;
    const cardIds = chooseBotTaxReturn(
      state.hands[bot.id],
      exchange.count,
      bot.botDifficulty ?? "normal",
    ).cardIds;
    selectTaxReturn(state, bot.id, cardIds, at, true);
    return;
  }

  if (state.phase === "playing") {
    const cardIds = chooseOnlineBotCards(state, bot.id);
    if (cardIds) {
      handlePlayCards(state, bot.id, cardIds, at);
    } else {
      handlePass(state, bot.id, at);
    }
  }
}

export function createOnlineRoom(
  code: string,
  firstPlayer: OnlinePlayerInput,
  now: number,
): OnlineRoomState {
  assertNow(now);
  const normalizedCode = code?.trim().toUpperCase();
  if (!normalizedCode || normalizedCode.length > 32) {
    fail("INVALID_ROOM_CODE", "room code must contain 1 to 32 characters");
  }
  const host = normalizePlayer(firstPlayer, "great-dalmuti", now);
  const state: OnlineRoomState = {
    code: normalizedCode,
    revision: 0,
    phase: "lobby",
    phaseEndsAt: null,
    turnDeadline: null,
    round: 0,
    hostId: host.id,
    players: [host],
    hands: {},
    dealSealed: false,
    currentIndex: 0,
    table: null,
    lastPlayedId: null,
    passedPlayerIds: [],
    finishOrder: [],
    rankSelection: null,
    revolutionHolderId: null,
    declaredRevolution: null,
    taxExchanges: [],
    actionLockUntil: null,
    botActionAt: null,
    events: [],
    nextEventSeq: 1,
    processedCommandIds: [],
    durations: { ...DEFAULT_DURATIONS },
    createdAt: now,
    updatedAt: now,
  };
  appendEvent(state, "ROOM_CREATED", now, {
    code: normalizedCode,
    hostId: host.id,
  });
  return state;
}

export function joinOnlineRoom(
  state: OnlineRoomState,
  player: OnlinePlayerInput,
  now: number,
): OnlineRoomState {
  assertNow(now);
  if (state.phase !== "lobby") {
    fail("ROOM_ALREADY_STARTED", "players can only join in the lobby");
  }
  if (state.players.length >= MAX_PLAYERS) {
    fail("ROOM_FULL", `a room can contain at most ${MAX_PLAYERS} players`);
  }
  const normalized = normalizePlayer(
    player,
    roleForIndex(state.players.length, state.players.length + 1),
    now,
  );
  if (state.players.some((candidate) => candidate.id === normalized.id)) {
    fail("PLAYER_ID_TAKEN", "that player id is already in the room");
  }

  const next = cloneRoom(state);
  next.players = withAssignedRoles([...next.players, normalized]);
  clearSealedDeal(next);
  appendEvent(next, "PLAYER_JOINED", now, {
    playerId: normalized.id,
    name: normalized.name,
  });
  return commit(next, now);
}

function finiteDeadline(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function insufficientCardsPassAt(
  state: OnlineRoomState,
): number | null {
  if (state.phase !== "playing" || !state.table) return null;
  const current = state.players[state.currentIndex];
  if (!current) return null;
  const handCount = state.hands[current.id]?.length ?? 0;
  if (handCount === 0 || handCount >= state.table.count) return null;
  return (
    finiteDeadline(state.actionLockUntil) ??
    finiteDeadline(state.updatedAt) ??
    0
  );
}

function fastForwardExpiredEmptyTurns(
  state: OnlineRoomState,
  now: number,
): void {
  const deadline = finiteDeadline(state.turnDeadline);
  if (
    state.phase !== "playing" ||
    state.table !== null ||
    deadline === null ||
    deadline > now
  ) {
    return;
  }

  const overdueTimeouts =
    Math.floor((now - deadline) / EMPTY_TABLE_TIMEOUT_CYCLE_MS) + 1;
  // Two public events are emitted per timeout (PASS and the next turn). Retain
  // enough recent turns to fill the bounded event window, while skipping old
  // cycles that no connected client could receive from the retained history.
  const retainedTimeouts = Math.ceil(MAX_EVENTS / 2);
  const skippedTimeouts = overdueTimeouts - retainedTimeouts;
  if (skippedTimeouts <= 0) return;

  const activeIndices = state.players.flatMap((player, index) =>
    (state.hands[player.id]?.length ?? 0) > 0 ? [index] : [],
  );
  const currentPosition = activeIndices.indexOf(state.currentIndex);
  if (activeIndices.length === 0 || currentPosition < 0) return;

  const timedOutIds =
    skippedTimeouts >= activeIndices.length
      ? activeIndices.map((index) => state.players[index].id)
      : Array.from({ length: skippedTimeouts }, (_, offset) => {
          const index =
            activeIndices[
              (currentPosition + offset) % activeIndices.length
            ];
          return state.players[index].id;
        });
  state.passedPlayerIds = [
    ...new Set([...state.passedPlayerIds, ...timedOutIds]),
  ];
  state.currentIndex =
    activeIndices[
      (currentPosition + skippedTimeouts) % activeIndices.length
    ];

  const lastSkippedAt =
    deadline + (skippedTimeouts - 1) * EMPTY_TABLE_TIMEOUT_CYCLE_MS;
  state.turnDeadline =
    deadline + skippedTimeouts * EMPTY_TABLE_TIMEOUT_CYCLE_MS;
  state.actionLockUntil = lastSkippedAt + PASS_ACTION_LOCK_MS;
  state.revision += skippedTimeouts;
  state.updatedAt = lastSkippedAt;

  const skippedEventCount = skippedTimeouts * 2;
  state.nextEventSeq += skippedEventCount;
  const retainedOldEventCount = Math.max(0, MAX_EVENTS - skippedEventCount);
  state.events =
    retainedOldEventCount > 0
      ? state.events.slice(-retainedOldEventCount)
      : [];
}

export function advanceOnlineRoom(
  state: OnlineRoomState,
  now: number,
  deps?: OnlineEngineDeps,
): OnlineRoomState {
  assertNow(now);
  const phaseDeadline = finiteDeadline(state.phaseEndsAt);
  const persistedTurnDeadline = finiteDeadline(state.turnDeadline);
  const persistedBotActionAt =
    pendingBotPlayer(state) === null
      ? null
      : finiteDeadline(state.botActionAt);
  const persistedInsufficientPassAt = insufficientCardsPassAt(state);
  const needsTurnDeadline =
    state.phase === "playing" && persistedTurnDeadline === null;
  const hasStaleTurnDeadline =
    state.phase !== "playing" && persistedTurnDeadline !== null;
  const hasExpiredPhase =
    phaseDeadline !== null && phaseDeadline <= now;
  const hasExpiredTurn =
    state.phase === "playing" &&
    persistedTurnDeadline !== null &&
    persistedTurnDeadline <= now;
  const hasExpiredBotAction =
    persistedBotActionAt !== null && persistedBotActionAt <= now;
  const hasInsufficientCardsPass =
    persistedInsufficientPassAt !== null &&
    persistedInsufficientPassAt <= now;
  if (
    !needsTurnDeadline &&
    !hasStaleTurnDeadline &&
    !hasExpiredPhase &&
    !hasExpiredTurn &&
    !hasExpiredBotAction &&
    !hasInsufficientCardsPass
  ) {
    return state;
  }

  const next = cloneRoom(state);
  next.durations = resolveDurations(next.durations, deps);
  if (needsTurnDeadline) {
    // Give rooms persisted by the pre-timer release a full first turn instead
    // of timing them out immediately on their first request after deployment.
    next.turnDeadline = now + TURN_DURATION_MS;
    next.updatedAt = now;
  } else if (hasStaleTurnDeadline) {
    next.turnDeadline = null;
    next.updatedAt = now;
  }

  let transitions = 0;
  while (true) {
    fastForwardExpiredEmptyTurns(next, now);
    const nextPhaseDeadline = finiteDeadline(next.phaseEndsAt);
    const nextTurnDeadline =
      next.phase === "playing"
        ? finiteDeadline(next.turnDeadline)
        : null;
    const nextBotActionAt =
      pendingBotPlayer(next) === null
        ? null
        : finiteDeadline(next.botActionAt);
    const nextInsufficientPassAt = insufficientCardsPassAt(next);
    const transitionAt = [
      nextPhaseDeadline,
      nextTurnDeadline,
      nextBotActionAt,
      nextInsufficientPassAt,
    ]
      .filter((deadline): deadline is number => deadline !== null)
      .reduce<number | null>(
        (earliest, deadline) =>
          earliest === null ? deadline : Math.min(earliest, deadline),
        null,
      );
    if (transitionAt === null || transitionAt > now) break;

    if (transitions >= 256) {
      fail("TRANSITION_LOOP", "too many timed room transitions");
    }
    if (
      nextPhaseDeadline !== null &&
      nextPhaseDeadline <= (nextTurnDeadline ?? Number.POSITIVE_INFINITY) &&
      nextPhaseDeadline <= (nextBotActionAt ?? Number.POSITIVE_INFINITY) &&
      nextPhaseDeadline <=
        (nextInsufficientPassAt ?? Number.POSITIVE_INFINITY)
    ) {
      advanceOneTimedPhase(next, transitionAt, deps);
    } else if (
      nextInsufficientPassAt !== null &&
      nextInsufficientPassAt <=
        (nextBotActionAt ?? Number.POSITIVE_INFINITY) &&
      nextInsufficientPassAt <=
        (nextTurnDeadline ?? Number.POSITIVE_INFINITY)
    ) {
      const current = next.players[next.currentIndex];
      if (!current) {
        fail("ROOM_INVARIANT", "an automatic pass has no current player");
      }
      handlePass(next, current.id, transitionAt, {
        automatic: true,
        reason: "insufficient-cards",
      });
    } else if (
      nextBotActionAt !== null &&
      nextBotActionAt <= (nextTurnDeadline ?? Number.POSITIVE_INFINITY)
    ) {
      performBotAction(next, transitionAt, deps);
    } else {
      handleTurnTimeout(next, transitionAt);
    }
    commit(next, transitionAt);
    transitions += 1;
  }
  if (next.updatedAt < now) next.updatedAt = now;
  return next;
}

export function applyOnlineCommand(
  state: OnlineRoomState,
  actorId: string,
  command: OnlineCommand,
  now: number,
  deps?: OnlineEngineDeps,
): OnlineRoomState {
  assertNow(now);
  if (!command?.id?.trim() || command.id.length > 128) {
    fail("INVALID_COMMAND_ID", "command id must contain 1 to 128 characters");
  }

  const advanced = advanceOnlineRoom(state, now, deps);
  if (advanced.processedCommandIds.includes(command.id)) return advanced;
  if (!advanced.players.some((player) => player.id === actorId)) {
    fail("PLAYER_NOT_FOUND", "the actor is not a member of this room");
  }
  if (
    command.type !== "SET_READY" &&
    command.type !== "CHOOSE_RANK_CARD" &&
    command.expectedRevision !== undefined &&
    command.expectedRevision !== advanced.revision
  ) {
    fail(
      "REVISION_MISMATCH",
      `expected revision ${command.expectedRevision}, current revision is ${advanced.revision}`,
    );
  }

  const next = cloneRoom(advanced);
  switch (command.type) {
    case "SET_READY": {
      if (next.phase !== "lobby") {
        fail("WRONG_PHASE", "readiness can only change in the lobby");
      }
      if (typeof command.ready !== "boolean") {
        fail("INVALID_READY_VALUE", "ready must be a boolean");
      }
      const player = next.players.find((candidate) => candidate.id === actorId)!;
      if (player.ready === command.ready) {
        next.processedCommandIds = [
          ...next.processedCommandIds,
          command.id,
        ].slice(-MAX_PROCESSED_COMMANDS);
        return next;
      }
      next.players = next.players.map((candidate) =>
        candidate.id === actorId
          ? { ...candidate, ready: command.ready }
          : candidate,
      );
      // The opening deal depends on the rank-card result, so it cannot be
      // assigned by temporary lobby order. Clear a legacy pre-sealed deal and
      // deal only after the rank reveal has established the real order.
      if (next.dealSealed) {
        clearSealedDeal(next);
      }
      appendEvent(next, "PLAYER_READY_CHANGED", now, {
        playerId: actorId,
        ready: command.ready,
      });
      break;
    }
    case "ADD_BOT": {
      if (next.phase !== "lobby") {
        fail("WRONG_PHASE", "bots can only be changed in the lobby");
      }
      if (actorId !== next.hostId) {
        fail("HOST_ONLY", "only the host can add a bot");
      }
      if (next.players.length >= MAX_PLAYERS) {
        fail("ROOM_FULL", `a room can contain at most ${MAX_PLAYERS} players`);
      }
      const difficulty = command.difficulty ?? "normal";
      if (!BOT_DIFFICULTIES.includes(difficulty)) {
        fail(
          "INVALID_BOT_DIFFICULTY",
          "bot difficulty must be easy, normal, or hard",
        );
      }
      const bot = createBotPlayer(next, now, difficulty);
      next.players = withAssignedRoles([...next.players, bot]);
      clearSealedDeal(next);
      appendEvent(next, "BOT_ADDED", now, {
        playerId: bot.id,
        name: bot.name,
        difficulty,
        byPlayerId: actorId,
      });
      break;
    }
    case "REMOVE_BOT": {
      if (next.phase !== "lobby") {
        fail("WRONG_PHASE", "bots can only be changed in the lobby");
      }
      if (actorId !== next.hostId) {
        fail("HOST_ONLY", "only the host can remove a bot");
      }
      if (typeof command.botId !== "string" || !command.botId.trim()) {
        fail("INVALID_BOT_ID", "botId must identify a bot in the room");
      }
      const bot = next.players.find(
        (player) => player.id === command.botId,
      );
      if (!bot) {
        fail("BOT_NOT_FOUND", "that bot is no longer in the room");
      }
      if (!bot.isBot) {
        fail("NOT_A_BOT", "only bot slots can be removed this way");
      }
      next.players = withAssignedRoles(
        next.players.filter((player) => player.id !== bot.id),
      );
      clearSealedDeal(next);
      appendEvent(next, "BOT_REMOVED", now, {
        playerId: bot.id,
        name: bot.name,
        byPlayerId: actorId,
      });
      break;
    }
    case "START_MATCH": {
      if (next.phase !== "lobby") {
        fail("WRONG_PHASE", "a match can only start from the lobby");
      }
      if (actorId !== next.hostId) {
        fail("HOST_ONLY", "only the host can start the match");
      }
      if (next.players.length < MIN_PLAYERS) {
        fail(
          "NOT_ENOUGH_PLAYERS",
          `at least ${MIN_PLAYERS} players are required`,
        );
      }
      if (next.players.some((player) => !player.ready)) {
        fail("PLAYERS_NOT_READY", "every player must be ready");
      }
      beginRankSelection(next, now, deps);
      break;
    }
    case "CHOOSE_RANK_CARD": {
      if (next.phase !== "rank-selection") {
        fail("WRONG_PHASE", "rank cards can only be chosen during rank selection");
      }
      if (next.phaseEndsAt !== null) {
        fail("RANK_CHOICES_LOCKED", "rank choices are already locked");
      }
      claimRankCard(next, actorId, command.slotIndex, now, false);
      break;
    }
    case "CHOOSE_REVOLUTION": {
      if (next.phase !== "revolution") {
        fail("WRONG_PHASE", "there is no revolution decision in progress");
      }
      if (actorId !== next.revolutionHolderId) {
        fail("NOT_REVOLUTION_HOLDER", "only the joker holder may decide");
      }
      if (typeof command.declare !== "boolean") {
        fail("INVALID_REVOLUTION_VALUE", "declare must be a boolean");
      }
      chooseRevolution(next, command.declare, now);
      break;
    }
    case "SELECT_TAX_RETURN": {
      if (next.phase !== "tax-selection") {
        fail("WRONG_PHASE", "tax returns are not being selected");
      }
      selectTaxReturn(next, actorId, command.cardIds, now, false);
      break;
    }
    case "PLAY_CARDS":
      handlePlayCards(next, actorId, command.cardIds, now);
      break;
    case "PASS":
      handlePass(next, actorId, now);
      break;
    case "START_NEXT_ROUND": {
      if (next.phase !== "round-end") {
        fail("WRONG_PHASE", "the current round has not ended");
      }
      if (actorId !== next.hostId) {
        fail("HOST_ONLY", "only the host can start the next round");
      }
      const byId = new Map(next.players.map((player) => [player.id, player]));
      next.players = next.finishOrder.map((id) => {
        const player = byId.get(id);
        if (!player) fail("ROOM_INVARIANT", "finish order contains an unknown player");
        return player;
      });
      startRound(next, next.round + 1, now, deps);
      break;
    }
    case "RESET_ROOM": {
      if (actorId !== next.hostId) {
        fail("HOST_ONLY", "only the host can reset the room");
      }
      resetRoomToLobby(next);
      appendEvent(next, "ROOM_RESET", now, {
        byPlayerId: actorId,
        reason: "host-reset",
      });
      break;
    }
    case "LEAVE_ROOM": {
      if (actorId === next.hostId) {
        fail(
          "HOST_CANNOT_LEAVE",
          "the host must reset the room instead of leaving it",
        );
      }
      const leavingPlayer = next.players.find(
        (player) => player.id === actorId,
      )!;
      next.players = next.players.filter((player) => player.id !== actorId);
      resetRoomToLobby(next);
      appendEvent(next, "PLAYER_LEFT", now, {
        playerId: actorId,
        name: leavingPlayer.name,
      });
      appendEvent(next, "ROOM_RESET", now, {
        reason: "player-left",
      });
      break;
    }
    default: {
      fail("INVALID_COMMAND_TYPE", "unknown online game command");
    }
  }

  return commit(next, now, command.id);
}

function handIsVisible(phase: OnlineRoomState["phase"]): boolean {
  return ![
    "lobby",
    "rank-intro",
    "rank-selection",
    "rank-reveal",
    "rank-confirm",
  ].includes(phase);
}

function projectEventForPlayer(
  event: OnlineEvent,
  actorId: string,
): OnlineEvent | null {
  if (
    event.visibility === "private" &&
    !event.playerIds?.includes(actorId)
  ) {
    return null;
  }
  return {
    ...event,
    ...(event.playerIds ? { playerIds: [...event.playerIds] } : {}),
    payload: { ...event.payload },
  };
}

export function projectOnlineRoom(
  state: OnlineRoomState,
  actorId: string,
  sinceEventSeq = 0,
): OnlineSnapshot {
  if (!state.players.some((player) => player.id === actorId)) {
    fail("PLAYER_NOT_FOUND", "the viewer is not a member of this room");
  }
  if (!Number.isInteger(sinceEventSeq) || sinceEventSeq < 0) {
    fail("INVALID_EVENT_CURSOR", "event cursor must be a non-negative integer");
  }

  const finishPlaces = new Map(
    state.finishOrder.map((playerId, index) => [playerId, index + 1]),
  );
  const viewerExchange = state.taxExchanges.find(
    (exchange) => exchange.nobleId === actorId,
  );
  const waitingForPlayerIds =
    state.phase === "tax-selection"
      ? state.taxExchanges
          .filter((exchange) => !exchange.nobleCardIds)
          .map((exchange) => exchange.nobleId)
      : [];
  const events = state.events
    .filter((event) => event.seq > sinceEventSeq)
    .map((event) => projectEventForPlayer(event, actorId))
    .filter((event): event is OnlineEvent => event !== null);
  const rankSelection = state.rankSelection ?? null;
  const selectedRankCard =
    rankSelection?.cards.find(
      (card) => card.claimedByPlayerId === actorId,
    ) ?? null;
  const rankChoicesLocked =
    state.phase === "rank-selection" &&
    rankSelection?.revealAt !== null &&
    rankSelection?.revealAt !== undefined;
  const rankCardsRevealed =
    rankSelection?.revealEndsAt !== null &&
    rankSelection?.revealEndsAt !== undefined;

  return {
    code: state.code,
    revision: state.revision,
    phase: state.phase,
    phaseEndsAt: state.phaseEndsAt,
    turnDeadline:
      state.phase === "playing"
        ? finiteDeadline(state.turnDeadline)
        : null,
    round: state.round,
    viewerId: actorId,
    hostId: state.hostId,
    dealSealed: state.dealSealed,
    minPlayers: MIN_PLAYERS,
    maxPlayers: MAX_PLAYERS,
    players: state.players.map((player) => ({
      id: player.id,
      name: player.name,
      monogram: player.monogram,
      isBot: player.isBot === true,
      botDifficulty: player.isBot
        ? player.botDifficulty ?? "normal"
        : null,
      role: player.role,
      ready: player.ready,
      connected: player.connected,
      handCount: state.hands[player.id]?.length ?? 0,
      finishedPlace: finishPlaces.get(player.id) ?? null,
      score: player.score,
    })),
    hand: handIsVisible(state.phase) ? [...state.hands[actorId]] : null,
    table: state.table
      ? { ...state.table, cards: [...state.table.cards] }
      : null,
    currentPlayerId:
      state.phase === "playing"
        ? state.players[state.currentIndex]?.id ?? null
        : null,
    lastPlayedId: state.lastPlayedId,
    actionLockUntil: state.actionLockUntil,
    passedPlayerIds: [...state.passedPlayerIds],
    finishOrder: [...state.finishOrder],
    events,
    latestEventSeq: state.nextEventSeq - 1,
    rankSelection: rankSelection
      ? {
          stage:
            state.phase === "rank-intro"
              ? "intro"
              : state.phase === "rank-selection"
                ? rankChoicesLocked
                  ? "locked"
                  : "selecting"
                : state.phase === "rank-confirm"
                  ? "confirmed"
                  : "revealed",
          cards: rankSelection.cards.map((card) => ({
            slotIndex: card.slotIndex,
            claimedByPlayerId: card.claimedByPlayerId,
            revealedRank: rankCardsRevealed ? card.rank : null,
          })),
          introStartedAt: rankSelection.introStartedAt,
          countdownStartsAt: rankSelection.countdownStartsAt,
          countdownEndsAt: rankSelection.countdownEndsAt,
          revealAt: rankSelection.revealAt,
          revealEndsAt: rankSelection.revealEndsAt,
          canChoose:
            state.phase === "rank-selection" &&
            !rankChoicesLocked &&
            selectedRankCard === null,
          selectedSlotIndex: selectedRankCard?.slotIndex ?? null,
        }
      : null,
    declaredRevolution: state.declaredRevolution
      ? { ...state.declaredRevolution }
      : null,
    tax:
      state.phase.startsWith("tax-") && viewerExchange
        ? {
            requiredReturnCount: viewerExchange.count,
            selectedReturnCount: viewerExchange.nobleCardIds?.length ?? 0,
            waitingForPlayerIds,
          }
        : state.phase.startsWith("tax-")
          ? {
              requiredReturnCount: 0,
              selectedReturnCount: 0,
              waitingForPlayerIds,
            }
          : null,
    revolution:
      state.phase === "revolution"
        ? {
            holderId:
              state.revolutionHolderId === actorId
                ? state.revolutionHolderId
                : null,
            canChoose: state.revolutionHolderId === actorId,
          }
        : null,
  };
}
