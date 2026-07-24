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

const MIN_PLAYERS = 4;
const MAX_PLAYERS = 8;
const MAX_EVENTS = 240;
const MAX_PROCESSED_COMMANDS = 512;
const PUBLIC_ACTION_LOCK_MS = 1_500;

const DEFAULT_DURATIONS: OnlinePhaseDurations = {
  revealIntroMs: 1_600,
  handRevealMs: 900,
  revolutionDecisionMs: 15_000,
  taxIntroMs: 1_500,
  taxSelectionMs: 30_000,
  taxTributeMs: 4_000,
  taxReturnMs: 4_000,
  playIntroMs: 1_800,
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
    role,
    ready: false,
    connected: true,
    joinedAt,
    score: 0,
  };
}

function cloneRoom(state: OnlineRoomState): OnlineRoomState {
  return {
    ...state,
    players: state.players.map((player) => ({ ...player })),
    hands: Object.fromEntries(
      Object.entries(state.hands).map(([id, hand]) => [id, [...hand]]),
    ),
    table: state.table
      ? { ...state.table, cards: [...state.table.cards] }
      : null,
    passedPlayerIds: [...state.passedPlayerIds],
    finishOrder: [...state.finishOrder],
    taxExchanges: state.taxExchanges.map((exchange) => ({
      ...exchange,
      peonCardIds: [...exchange.peonCardIds],
      nobleCardIds: exchange.nobleCardIds
        ? [...exchange.nobleCardIds]
        : null,
    })),
    events: [...state.events],
    processedCommandIds: [...state.processedCommandIds],
    durations: { ...state.durations },
  };
}

function resolveDurations(
  base: OnlinePhaseDurations,
  deps?: OnlineEngineDeps,
): OnlinePhaseDurations {
  const durations = { ...base, ...deps?.durations };
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
  if (!useSealedDeal || !state.dealSealed) {
    sealDeal(state, at, deps);
  }
  state.round = round;
  state.phase = "reveal-intro";
  state.phaseEndsAt = at + state.durations.revealIntroMs;
  state.currentIndex = 0;
  state.table = null;
  state.lastPlayedId = null;
  state.passedPlayerIds = [];
  state.finishOrder = [];
  state.revolutionHolderId = null;
  state.taxExchanges = [];
  state.actionLockUntil = null;

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
      { cards: [...state.hands[player.id]] },
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

function publicTaxRoutes(exchanges: OnlineTaxExchange[]) {
  return exchanges.map((exchange) => ({
    nobleId: exchange.nobleId,
    peonId: exchange.peonId,
    count: exchange.count,
  }));
}

function enterTaxIntro(state: OnlineRoomState, at: number): void {
  state.revolutionHolderId = null;
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
  state.phaseEndsAt = at + state.durations.taxSelectionMs;
  appendEvent(state, "TAX_SELECTION_STARTED", at, {
    waitingForPlayerIds: state.taxExchanges.map(
      (exchange) => exchange.nobleId,
    ),
    endsAt: state.phaseEndsAt,
  });

  for (const exchange of state.taxExchanges) {
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
  appendEvent(state, "TAX_TRIBUTE_STARTED", at, {
    routes: publicTaxRoutes(state.taxExchanges),
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
  appendEvent(state, "TAX_RETURN_STARTED", at, {
    routes: publicTaxRoutes(state.taxExchanges),
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
      },
      [exchange.peonId, exchange.nobleId],
    );
  }
}

function enterPlayIntro(state: OnlineRoomState, at: number): void {
  state.phase = "play-intro";
  state.phaseEndsAt = at + state.durations.playIntroMs;
  state.taxExchanges = [];
  state.revolutionHolderId = null;
  state.currentIndex = 0;
  appendEvent(state, "PLAY_INTRO_STARTED", at, {
    firstPlayerId: state.players[0].id,
    endsAt: state.phaseEndsAt,
  });
}

function enterPlaying(state: OnlineRoomState, at: number): void {
  state.phase = "playing";
  state.phaseEndsAt = null;
  state.actionLockUntil = null;
  state.currentIndex = 0;
  appendEvent(state, "TURN_STARTED", at, {
    playerId: state.players[state.currentIndex].id,
  });
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
  if (isGreatRevolution) {
    state.players = withAssignedRoles([...state.players].reverse());
  }
  appendEvent(state, "REVOLUTION_DECLARED", at, {
    playerId: holderId,
    kind: isGreatRevolution ? "great" : "normal",
  });
  enterPlayIntro(state, at);
}

function autoSelectTaxReturns(state: OnlineRoomState, at: number): void {
  state.taxExchanges = state.taxExchanges.map((exchange) => {
    if (exchange.nobleCardIds?.length === exchange.count) return exchange;
    const cards = selectAutomaticNobleReturns(
      state.hands[exchange.nobleId],
      exchange.count,
    );
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

function advanceOneTimedPhase(state: OnlineRoomState, at: number): void {
  switch (state.phase) {
    case "reveal-intro":
      enterHandReveal(state, at);
      return;
    case "hand-reveal": {
      const holder = state.players.find(
        (player) =>
          state.hands[player.id].filter((card) => card.rank === 13).length ===
          2,
      );
      if (holder) enterRevolutionDecision(state, holder.id, at);
      else enterTaxIntro(state, at);
      return;
    }
    case "revolution":
      chooseRevolution(state, false, at);
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
  if (player) appendEvent(state, "TURN_STARTED", at, { playerId: player.id });
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
  appendEvent(state, "CARDS_PLAYED", at, { ...table });
  state.actionLockUntil = at + PUBLIC_ACTION_LOCK_MS;

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
  state.currentIndex = nextActiveIndex(state, state.currentIndex);
  addTurnStartedEvent(state, at);
}

function handlePass(
  state: OnlineRoomState,
  actorId: string,
  at: number,
): void {
  if (state.phase !== "playing") {
    fail("WRONG_PHASE", "passing is only allowed during the playing phase");
  }
  if (state.actionLockUntil !== null && at < state.actionLockUntil) {
    fail("ACTION_IN_PROGRESS", "the previous table animation is still playing");
  }
  const current = state.players[state.currentIndex];
  if (current?.id !== actorId) fail("NOT_YOUR_TURN", "it is not your turn");
  if (!state.table) fail("CANNOT_PASS", "the leading player cannot pass");

  state.passedPlayerIds = [
    ...new Set([...state.passedPlayerIds, actorId]),
  ];
  appendEvent(state, "PLAYER_PASSED", at, { playerId: actorId });
  state.actionLockUntil = at + PUBLIC_ACTION_LOCK_MS;

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
    addTurnStartedEvent(state, at);
    return;
  }

  state.currentIndex = nextActiveIndex(state, state.currentIndex);
  addTurnStartedEvent(state, at);
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
    revolutionHolderId: null,
    taxExchanges: [],
    actionLockUntil: null,
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

export function advanceOnlineRoom(
  state: OnlineRoomState,
  now: number,
  deps?: OnlineEngineDeps,
): OnlineRoomState {
  assertNow(now);
  if (state.phaseEndsAt === null || state.phaseEndsAt > now) return state;

  const next = cloneRoom(state);
  next.durations = resolveDurations(next.durations, deps);
  let transitions = 0;
  while (next.phaseEndsAt !== null && next.phaseEndsAt <= now) {
    if (transitions >= 16) {
      fail("TRANSITION_LOOP", "too many timed room transitions");
    }
    const transitionAt = next.phaseEndsAt;
    advanceOneTimedPhase(next, transitionAt);
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
      if (
        next.players.length >= MIN_PLAYERS &&
        next.players.every((candidate) => candidate.ready)
      ) {
        sealDeal(next, now, deps);
      } else if (next.dealSealed) {
        clearSealedDeal(next);
      }
      appendEvent(next, "PLAYER_READY_CHANGED", now, {
        playerId: actorId,
        ready: command.ready,
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
      if (!next.dealSealed) {
        fail("DEAL_NOT_SEALED", "the hidden deal is not ready");
      }
      startRound(next, 1, now, deps, true);
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
      const exchangeIndex = next.taxExchanges.findIndex(
        (exchange) => exchange.nobleId === actorId,
      );
      if (exchangeIndex < 0) {
        fail("NOT_TAX_NOBLE", "this player does not choose a tax return");
      }
      const exchange = next.taxExchanges[exchangeIndex];
      if (exchange.nobleCardIds) {
        fail("TAX_RETURN_ALREADY_SELECTED", "the return cards are already locked");
      }
      if (
        !Array.isArray(command.cardIds) ||
        command.cardIds.length !== exchange.count
      ) {
        fail(
          "WRONG_TAX_CARD_COUNT",
          `exactly ${exchange.count} return cards are required`,
        );
      }
      assertUniqueCardIds(command.cardIds);
      cardsByIds(next.hands[actorId], command.cardIds);
      next.taxExchanges[exchangeIndex] = {
        ...exchange,
        nobleCardIds: [...command.cardIds],
      };
      appendEvent(
        next,
        "TAX_RETURN_SELECTED",
        now,
        { cardIds: [...command.cardIds], automatic: false },
        [actorId],
      );
      if (allTaxReturnsSelected(next)) enterTaxTribute(next, now);
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
    default: {
      fail("INVALID_COMMAND_TYPE", "unknown online game command");
    }
  }

  return commit(next, now, command.id);
}

function handIsVisible(phase: OnlineRoomState["phase"]): boolean {
  return phase !== "lobby" && phase !== "reveal-intro";
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

  return {
    code: state.code,
    revision: state.revision,
    phase: state.phase,
    phaseEndsAt: state.phaseEndsAt,
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
