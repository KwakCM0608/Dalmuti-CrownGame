import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const {
  OnlineGameError,
  advanceOnlineRoom,
  applyOnlineCommand,
  createOnlineRoom,
  joinOnlineRoom,
  projectOnlineRoom,
} = await import(new URL("../lib/online-game/index.ts", import.meta.url));

function command(state, actorId, type, payload = {}, now = state.updatedAt + 1) {
  return applyOnlineCommand(
    state,
    actorId,
    {
      id: `${type}-${actorId}-${now}`,
      expectedRevision: state.revision,
      type,
      ...payload,
    },
    now,
    { randomInt: () => 0 },
  );
}

function createFourPlayerLobby() {
  let state = createOnlineRoom(
    "ABC234",
    { id: "p1", name: "하나" },
    1,
  );
  state = joinOnlineRoom(state, { id: "p2", name: "둘" }, 2);
  state = joinOnlineRoom(state, { id: "p3", name: "셋" }, 3);
  state = joinOnlineRoom(state, { id: "p4", name: "넷" }, 4);
  return state;
}

function createSixPlayerLobby() {
  let state = createFourPlayerLobby();
  state = joinOnlineRoom(state, { id: "p5", name: "p5" }, 5);
  state = joinOnlineRoom(state, { id: "p6", name: "p6" }, 6);
  return state;
}

function createThreeHumanOneBotLobby() {
  let state = createOnlineRoom(
    "BOT234",
    { id: "p1", name: "p1" },
    1,
  );
  state = joinOnlineRoom(state, { id: "p2", name: "p2" }, 2);
  state = joinOnlineRoom(state, { id: "p3", name: "p3" }, 3);
  state = command(state, "p1", "ADD_BOT", {}, 4);
  return state;
}

test("player count assigns the official five-tier rank structure", () => {
  const rolesFor = (playerCount) => {
    let state = createOnlineRoom(
      `ROLE${playerCount}`,
      { id: "p1", name: "p1" },
      1,
    );
    for (let index = 2; index <= playerCount; index += 1) {
      state = joinOnlineRoom(
        state,
        { id: `p${index}`, name: `p${index}` },
        index,
      );
    }
    return state.players.map((player) => player.role);
  };

  assert.deepEqual(rolesFor(4), [
    "great-dalmuti",
    "lesser-dalmuti",
    "lesser-peon",
    "great-peon",
  ]);
  assert.deepEqual(rolesFor(5), [
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
  ]);
  assert.deepEqual(rolesFor(6), [
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "merchant",
    "lesser-peon",
    "great-peon",
  ]);
  assert.deepEqual(rolesFor(8), [
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "merchant",
    "merchant",
    "merchant",
    "lesser-peon",
    "great-peon",
  ]);
});

test("a host can fill a three-human room with a ready connected bot and start PLAY", () => {
  let state = createOnlineRoom(
    "BOTLOBBY",
    { id: "p1", name: "p1" },
    1,
  );
  state = joinOnlineRoom(state, { id: "p2", name: "p2" }, 2);
  state = joinOnlineRoom(state, { id: "p3", name: "p3" }, 3);

  assert.throws(
    () => command(state, "p2", "ADD_BOT", {}, 4),
    (error) =>
      error instanceof OnlineGameError && error.code === "HOST_ONLY",
  );

  state = command(state, "p1", "ADD_BOT", { difficulty: "hard" }, 5);
  const bot = state.players.find((player) => player.isBot);
  assert.ok(bot);
  assert.equal(bot.ready, true);
  assert.equal(bot.connected, true);
  assert.equal(bot.botDifficulty, "hard");
  assert.equal(bot.role, "great-peon");
  assert.equal(
    state.events.findLast((event) => event.type === "BOT_ADDED")?.payload
      .playerId,
    bot.id,
  );

  const botView = projectOnlineRoom(state, "p2").players.find(
    (player) => player.id === bot.id,
  );
  assert.deepEqual(
    {
      isBot: botView?.isBot,
      ready: botView?.ready,
      connected: botView?.connected,
      botDifficulty: botView?.botDifficulty,
    },
    {
      isBot: true,
      ready: true,
      connected: true,
      botDifficulty: "hard",
    },
  );

  assert.throws(
    () =>
      command(
        state,
        "p1",
        "ADD_BOT",
        { difficulty: "impossible" },
        6,
      ),
    (error) =>
      error instanceof OnlineGameError &&
      error.code === "INVALID_BOT_DIFFICULTY",
  );

  assert.throws(
    () =>
      command(
        state,
        "p2",
        "REMOVE_BOT",
        { botId: bot.id },
        6,
      ),
    (error) =>
      error instanceof OnlineGameError && error.code === "HOST_ONLY",
  );
  assert.throws(
    () =>
      command(
        state,
        "p1",
        "REMOVE_BOT",
        { botId: "p2" },
        7,
      ),
    (error) =>
      error instanceof OnlineGameError && error.code === "NOT_A_BOT",
  );

  const withoutBot = command(
    state,
    "p1",
    "REMOVE_BOT",
    { botId: bot.id },
    8,
  );
  assert.equal(withoutBot.players.length, 3);
  assert.equal(withoutBot.players.some((player) => player.isBot), false);
  assert.equal(
    withoutBot.events.findLast((event) => event.type === "BOT_REMOVED")
      ?.payload.playerId,
    bot.id,
  );

  state = command(withoutBot, "p1", "ADD_BOT", {}, 9);
  for (const [offset, playerId] of ["p2", "p3"].entries()) {
    state = command(
      state,
      playerId,
      "SET_READY",
      { ready: true },
      10 + offset,
    );
  }
  assert.equal(
    state.players.find((player) => player.id === state.hostId)?.ready,
    false,
    "the host does not need to ready up",
  );
  assert.equal(
    state.players
      .filter((player) => player.id !== state.hostId)
      .every((player) => player.ready && player.connected),
    true,
  );

  state = applyOnlineCommand(
    state,
    "p1",
    {
      id: "three-humans-one-bot-start",
      expectedRevision: state.revision,
      type: "START_MATCH",
    },
    20,
    {
      randomInt: () => 0,
      durations: { rankChoiceIntroMs: 1 },
    },
  );
  assert.equal(state.players.length, 4);
  assert.equal(state.phase, "rank-intro");
  assert.equal(state.round, 1);
});

test("PLAY requires every connected non-host player to be ready", () => {
  let state = createFourPlayerLobby();
  state = command(state, "p2", "SET_READY", { ready: true }, 10);
  state = command(state, "p3", "SET_READY", { ready: true }, 11);

  assert.throws(
    () => command(state, "p1", "START_MATCH", {}, 12),
    (error) =>
      error instanceof OnlineGameError &&
      error.code === "PLAYERS_NOT_READY",
    "an unready guest must keep PLAY disabled on the server",
  );

  state = command(state, "p4", "SET_READY", { ready: true }, 13);
  const disconnectedGuest = state.players.find(
    (player) => player.id === "p3",
  );
  assert.ok(disconnectedGuest);
  disconnectedGuest.connected = false;

  assert.throws(
    () => command(state, "p1", "START_MATCH", {}, 14),
    (error) =>
      error instanceof OnlineGameError &&
      error.code === "PLAYERS_NOT_READY",
    "a disconnected guest must keep PLAY disabled on the server",
  );

  disconnectedGuest.connected = true;
  state = command(state, "p1", "START_MATCH", {}, 15);
  assert.equal(state.phase, "rank-intro");
  assert.equal(
    state.players.find((player) => player.id === state.hostId)?.ready,
    false,
  );
});

test("server bots act automatically during rank choice, revolution, tax, and play", () => {
  const botDeps = {
    randomInt: () => 0,
    durations: {
      rankChoiceIntroMs: 0,
      rankRevealDelayMs: 1,
      rankRevealMs: 1,
      rankConfirmMs: 1,
      revolutionIntroMs: 1,
      playIntroMs: 1,
    },
  };

  let rankState = createThreeHumanOneBotLobby();
  for (const [offset, playerId] of ["p1", "p2", "p3"].entries()) {
    rankState = command(
      rankState,
      playerId,
      "SET_READY",
      { ready: true },
      10 + offset,
    );
  }
  rankState = applyOnlineCommand(
    rankState,
    "p1",
    {
      id: "start-bot-rank-choice",
      expectedRevision: rankState.revision,
      type: "START_MATCH",
    },
    100,
    botDeps,
  );
  rankState = advanceOnlineRoom(rankState, 100, botDeps);
  assert.equal(rankState.phase, "rank-selection");
  assert.equal(rankState.botActionAt, 850);

  rankState = advanceOnlineRoom(rankState, rankState.botActionAt, botDeps);
  const rankBot = rankState.players.find((player) => player.isBot);
  const botRankChoice = rankState.events.findLast(
    (event) =>
      event.type === "RANK_CARD_CHOSEN" &&
      event.payload.playerId === rankBot.id,
  );
  assert.equal(botRankChoice?.payload.automatic, true);
  assert.equal(
    rankState.rankSelection.cards.some(
      (card) => card.claimedByPlayerId === rankBot.id,
    ),
    true,
  );

  let revolutionState = createThreeHumanOneBotLobby();
  const revolutionBot = revolutionState.players.find(
    (player) => player.isBot,
  );
  assert.equal(revolutionBot.role, "great-peon");
  revolutionState.phase = "hand-reveal";
  revolutionState.phaseEndsAt = 200;
  revolutionState.round = 1;
  revolutionState.hands = Object.fromEntries(
    revolutionState.players.map((player, index) => [
      player.id,
      player.id === revolutionBot.id
        ? [
            { id: "bot-joker-1", rank: 13 },
            { id: "bot-joker-2", rank: 13 },
          ]
        : [{ id: `human-${index + 1}`, rank: index + 2 }],
    ]),
  );
  revolutionState = advanceOnlineRoom(revolutionState, 200, botDeps);
  assert.equal(revolutionState.phase, "revolution-intro");
  assert.deepEqual(revolutionState.declaredRevolution, {
    round: 1,
    playerId: revolutionBot.id,
    kind: "great-revolution",
  });
  assert.equal(
    revolutionState.events.findLast(
      (event) => event.type === "REVOLUTION_DECLARED",
    )?.payload.playerId,
    revolutionBot.id,
  );
  assert.equal(
    revolutionState.events.some(
      (event) => event.type === "REVOLUTION_DECISION_STARTED",
    ),
    false,
  );

  let taxState = createOnlineRoom(
    "BOTTAX",
    { id: "p1", name: "p1" },
    1,
  );
  taxState = command(taxState, "p1", "ADD_BOT", {}, 2);
  taxState = joinOnlineRoom(taxState, { id: "p2", name: "p2" }, 3);
  taxState = joinOnlineRoom(taxState, { id: "p3", name: "p3" }, 4);
  const taxBot = taxState.players.find((player) => player.isBot);
  assert.equal(taxBot.role, "lesser-dalmuti");
  taxState.phase = "tax-intro";
  taxState.phaseEndsAt = 300;
  taxState.botActionAt = null;
  taxState.hands = {
    p1: [
      { id: "p1-return-a", rank: 12 },
      { id: "p1-return-b", rank: 11 },
    ],
    [taxBot.id]: [
      { id: "bot-return", rank: 12 },
      { id: "bot-keep", rank: 3 },
    ],
    p2: [
      { id: "p2-tax", rank: 2 },
      { id: "p2-keep", rank: 8 },
    ],
    p3: [
      { id: "p3-tax-a", rank: 1 },
      { id: "p3-tax-b", rank: 2 },
      { id: "p3-keep", rank: 9 },
    ],
  };
  taxState.taxExchanges = [
    {
      nobleId: "p1",
      peonId: "p3",
      count: 2,
      peonCardIds: ["p3-tax-a", "p3-tax-b"],
      nobleCardIds: null,
    },
    {
      nobleId: taxBot.id,
      peonId: "p2",
      count: 1,
      peonCardIds: ["p2-tax"],
      nobleCardIds: null,
    },
  ];

  let completedByBotState = structuredClone(taxState);
  completedByBotState.taxExchanges[0].nobleCardIds = [
    "p1-return-a",
    "p1-return-b",
  ];
  completedByBotState = advanceOnlineRoom(
    completedByBotState,
    300,
    botDeps,
  );
  assert.equal(completedByBotState.phase, "tax-tribute");
  assert.deepEqual(
    completedByBotState.taxExchanges[1].nobleCardIds,
    ["bot-return"],
  );
  assert.equal(
    completedByBotState.events.some(
      (event) => event.type === "TAX_SELECTION_STARTED",
    ),
    false,
  );

  taxState = advanceOnlineRoom(taxState, 300, botDeps);
  assert.equal(taxState.phase, "tax-selection");
  assert.deepEqual(taxState.taxExchanges[1].nobleCardIds, ["bot-return"]);
  assert.equal(taxState.taxExchanges[0].nobleCardIds, null);
  assert.equal(taxState.botActionAt, null);
  const taxSelectionStarted = taxState.events.findLast(
    (event) =>
      event.type === "TAX_SELECTION_STARTED" &&
      event.visibility === "public",
  );
  assert.deepEqual(taxSelectionStarted?.payload.waitingForPlayerIds, ["p1"]);
  assert.equal(
    taxState.events.some(
      (event) =>
        event.type === "TAX_SELECTION_STARTED" &&
        event.playerIds?.includes(taxBot.id),
    ),
    false,
  );
  const automaticTaxChoice = taxState.events.findLast(
    (event) =>
      event.type === "TAX_RETURN_SELECTED" &&
      event.playerIds?.includes(taxBot.id),
  );
  assert.equal(automaticTaxChoice?.payload.automatic, true);
  assert.equal(automaticTaxChoice?.at, 300);

  taxState = command(
    taxState,
    "p1",
    "SELECT_TAX_RETURN",
    { cardIds: ["p1-return-a", "p1-return-b"] },
    301,
  );
  assert.equal(taxState.phase, "tax-tribute");

  let playingState = createThreeHumanOneBotLobby();
  const playingBot = playingState.players.find((player) => player.isBot);
  playingState.phase = "playing";
  playingState.phaseEndsAt = null;
  playingState.currentIndex = playingState.players.findIndex(
    (player) => player.id === playingBot.id,
  );
  playingState.actionLockUntil = null;
  playingState.turnDeadline = 30_000;
  playingState.botActionAt = 400;
  playingState.table = {
    rank: 8,
    count: 1,
    playerId: "p3",
    cards: [{ id: "table-8", rank: 8 }],
  };
  playingState.lastPlayedId = "p3";
  playingState.hands = {
    p1: [{ id: "p1-12", rank: 12 }],
    p2: [{ id: "p2-11", rank: 11 }],
    p3: [{ id: "p3-10", rank: 10 }],
    [playingBot.id]: [{ id: "bot-7", rank: 7 }],
  };
  playingState = advanceOnlineRoom(playingState, 400, botDeps);
  assert.equal(playingState.table?.playerId, playingBot.id);
  assert.equal(playingState.table?.rank, 7);
  assert.equal(playingState.hands[playingBot.id].length, 0);
  assert.equal(
    playingState.events.findLast(
      (event) => event.type === "CARDS_PLAYED",
    )?.payload.playerId,
    playingBot.id,
  );
});

function readyEveryone(state) {
  let next = state;
  for (const [index, player] of next.players.entries()) {
    next = command(next, player.id, "SET_READY", { ready: true }, 10 + index);
  }
  return next;
}

const instantRankDurations = {
  rankChoiceIntroMs: 0,
  rankRevealDelayMs: 1,
  rankRevealMs: 1,
  rankConfirmMs: 1,
};

function startAndAssignJoinOrder(
  state,
  startAt = 20,
  durations = {},
  engineDeps = {},
) {
  const helperDurations = {
    ...instantRankDurations,
    ...durations,
    // Keep the helper parked at reveal-intro even when the tested downstream
    // flow uses zero-length intro phases.
    ...(durations.revealIntroMs === 0 ? { revealIntroMs: 1 } : {}),
  };
  const deps = {
    randomInt: () => 0,
    ...engineDeps,
    durations: helperDurations,
  };
  let next = applyOnlineCommand(
    state,
    state.hostId,
    {
      id: `start-rank-choice-${startAt}`,
      expectedRevision: state.revision,
      type: "START_MATCH",
    },
    startAt,
    deps,
  );
  next = advanceOnlineRoom(next, next.phaseEndsAt, deps);
  assert.equal(next.phase, "rank-selection");

  const joinedPlayers = [...next.players].sort(
    (left, right) =>
      left.joinedAt - right.joinedAt || left.id.localeCompare(right.id),
  );
  for (const [index, player] of joinedPlayers.slice(0, -1).entries()) {
    const card = next.rankSelection.cards.find(
      (candidate) => candidate.rank === index + 1,
    );
    next = applyOnlineCommand(
      next,
      player.id,
      {
        id: `choose-rank-${player.id}-${startAt}`,
        expectedRevision: next.revision,
        type: "CHOOSE_RANK_CARD",
        slotIndex: card.slotIndex,
      },
      startAt + index + 1,
      deps,
    );
  }
  const finalPlayer = joinedPlayers.at(-1);
  assert.equal(
    next.rankSelection.cards.some(
      (card) => card.claimedByPlayerId === finalPlayer.id,
    ),
    true,
  );
  assert.equal(next.phase, "rank-selection");
  assert.notEqual(next.phaseEndsAt, null);

  next = advanceOnlineRoom(next, next.phaseEndsAt, deps);
  assert.equal(next.phase, "rank-reveal");
  next = advanceOnlineRoom(next, next.phaseEndsAt, deps);
  assert.equal(next.phase, "rank-confirm");
  assert.equal(projectOnlineRoom(next, next.hostId).hand, null);
  next = advanceOnlineRoom(next, next.phaseEndsAt, deps);
  assert.equal(next.phase, "reveal-intro");
  return next;
}

test("normal online setup preserves the host's selected opening rank", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state = startAndAssignJoinOrder(state, 150);

  assert.equal(state.players.at(0).id, state.hostId);
  assert.equal(state.players.at(0).role, "great-dalmuti");
});

test("the first act presents hand reveal, no-tax notice, and game start on one exact timeline", () => {
  const durations = {
    revealIntroMs: 120,
    handRevealMs: 140,
    taxIntroMs: 240,
    playIntroMs: 260,
  };
  let state = readyEveryone(createFourPlayerLobby());
  state = startAndAssignJoinOrder(state, 180, durations);
  state.hands = Object.fromEntries(
    state.players.map((player, index) => [
      player.id,
      [
        { id: `${player.id}-normal-a`, rank: index + 3 },
        { id: `${player.id}-normal-b`, rank: index + 7 },
      ],
    ]),
  );

  const revealIntroEndsAt = state.phaseEndsAt;
  assert.notEqual(revealIntroEndsAt, null);
  state = advanceOnlineRoom(state, revealIntroEndsAt - 1, { durations });
  assert.equal(state.phase, "reveal-intro");
  state = advanceOnlineRoom(state, revealIntroEndsAt, { durations });
  assert.equal(state.phase, "hand-reveal");

  const handRevealEvent = state.events.findLast(
    (event) => event.type === "HAND_REVEAL_STARTED",
  );
  const handsBeforeNoTax = structuredClone(state.hands);
  const handRevealEndsAt = state.phaseEndsAt;
  assert.equal(handRevealEvent?.payload.endsAt, handRevealEndsAt);

  state = advanceOnlineRoom(state, handRevealEndsAt - 1, { durations });
  assert.equal(state.phase, "hand-reveal");
  state = advanceOnlineRoom(state, handRevealEndsAt, { durations });
  assert.equal(state.phase, "tax-intro");
  assert.deepEqual(state.taxExchanges, []);
  assert.deepEqual(state.hands, handsBeforeNoTax);

  const noTaxEvent = state.events.findLast(
    (event) => event.type === "TAX_INTRO_STARTED",
  );
  assert.equal(noTaxEvent?.at, handRevealEndsAt);
  assert.equal(noTaxEvent?.payload.skipped, true);
  assert.equal(noTaxEvent?.payload.round, 1);
  assert.deepEqual(noTaxEvent?.payload.routes, []);
  assert.equal(
    projectOnlineRoom(state, state.players.at(-1).id).events.findLast(
      (event) => event.type === "TAX_INTRO_STARTED",
    )?.payload.skipped,
    true,
  );

  const noTaxEndsAt = state.phaseEndsAt;
  assert.equal(noTaxEvent?.payload.endsAt, noTaxEndsAt);
  state = advanceOnlineRoom(state, noTaxEndsAt - 1, { durations });
  assert.equal(state.phase, "tax-intro");
  state = advanceOnlineRoom(state, noTaxEndsAt, { durations });
  assert.equal(state.phase, "play-intro");

  const playIntroEvent = state.events.findLast(
    (event) => event.type === "PLAY_INTRO_STARTED",
  );
  assert.equal(playIntroEvent?.at, noTaxEndsAt);
  const playIntroEndsAt = state.phaseEndsAt;
  assert.equal(playIntroEvent?.payload.endsAt, playIntroEndsAt);

  state = advanceOnlineRoom(state, playIntroEndsAt - 1, { durations });
  assert.equal(state.phase, "play-intro");
  state = advanceOnlineRoom(state, playIntroEndsAt, { durations });
  assert.equal(state.phase, "playing");
  assert.equal(
    state.events.findLast((event) => event.type === "TURN_STARTED")?.at,
    playIntroEndsAt,
  );
  assert.equal(
    state.events.some((event) =>
      [
        "TAX_SELECTION_STARTED",
        "TAX_TRIBUTE_STARTED",
        "TAX_TRIBUTE",
        "TAX_RETURN_STARTED",
        "TAX_RETURN",
      ].includes(event.type),
    ),
    false,
  );
  assert.deepEqual(state.hands, handsBeforeNoTax);
});

test("the opening PLAY runs a hidden, server-authoritative rank choice before dealing", () => {
  let state = readyEveryone(createFourPlayerLobby());

  assert.equal(state.phase, "lobby");
  assert.equal(state.dealSealed, false);
  assert.deepEqual(state.hands, {});

  const durations = {
    rankChoiceIntroMs: 3_000,
    rankRevealDelayMs: 1_000,
    rankRevealMs: 1_500,
    rankConfirmMs: 1_000,
  };
  state = applyOnlineCommand(
    state,
    state.hostId,
    {
      id: "opening-rank-choice",
      expectedRevision: state.revision,
      type: "START_MATCH",
    },
    100,
    { randomInt: () => 0, durations },
  );
  assert.equal(state.phase, "rank-intro");
  assert.equal(state.phaseEndsAt, 3_100);
  assert.equal(state.rankSelection.cards.length, 4);
  assert.deepEqual(
    [...state.rankSelection.cards].map((card) => card.rank).sort(),
    [1, 2, 3, 4],
  );
  assert.notDeepEqual(
    state.rankSelection.cards.map((card) => card.rank),
    [1, 2, 3, 4],
  );

  let view = projectOnlineRoom(state, "p1");
  assert.equal(view.rankSelection.stage, "intro");
  assert.equal(view.rankSelection.countdownEndsAt, 3_100);
  assert.equal(
    view.rankSelection.cards.every((card) => card.revealedRank === null),
    true,
  );
  state = advanceOnlineRoom(state, 3_100, {
    randomInt: () => 0,
    durations,
  });
  assert.equal(state.phase, "rank-selection");

  const p1Slot = state.rankSelection.cards[0].slotIndex;
  state = command(
    state,
    "p1",
    "CHOOSE_RANK_CARD",
    { slotIndex: p1Slot },
    3_101,
  );
  assert.throws(
    () =>
      command(
        state,
        "p1",
        "CHOOSE_RANK_CARD",
        { slotIndex: state.rankSelection.cards[1].slotIndex },
        3_102,
      ),
    (error) =>
      error instanceof OnlineGameError && error.code === "RANK_ALREADY_CHOSEN",
  );
  assert.throws(
    () =>
      command(
        state,
        "p2",
        "CHOOSE_RANK_CARD",
        { slotIndex: p1Slot },
        3_103,
      ),
    (error) =>
      error instanceof OnlineGameError && error.code === "RANK_CARD_CLAIMED",
  );

  for (const [offset, playerId] of ["p2", "p3"].entries()) {
    const card = state.rankSelection.cards.find(
      (candidate) => candidate.claimedByPlayerId === null,
    );
    state = command(
      state,
      playerId,
      "CHOOSE_RANK_CARD",
      { slotIndex: card.slotIndex },
      3_104 + offset,
    );
  }
  assert.equal(state.phase, "rank-selection");
  assert.equal(state.phaseEndsAt, 4_105);
  assert.equal(
    state.rankSelection.cards.find(
      (card) => card.claimedByPlayerId === "p4",
    )?.claimedAt,
    3_105,
  );
  const finalChoiceEvent = state.events.findLast(
    (event) =>
      event.type === "RANK_CARD_CHOSEN" &&
      event.payload.playerId === "p4",
  );
  assert.equal(finalChoiceEvent?.payload.automatic, true);
  assert.equal("rank" in finalChoiceEvent.payload, false);
  assert.throws(
    () =>
      command(
        state,
        "p4",
        "CHOOSE_RANK_CARD",
        {
          slotIndex: state.rankSelection.cards.find(
            (card) => card.claimedByPlayerId === "p4",
          ).slotIndex,
        },
        3_106,
      ),
    (error) =>
      error instanceof OnlineGameError &&
      error.code === "RANK_CHOICES_LOCKED",
  );

  view = projectOnlineRoom(state, "p3");
  assert.equal(view.rankSelection.stage, "locked");
  assert.equal(view.rankSelection.canChoose, false);
  assert.equal(
    view.rankSelection.cards.every(
      (card) =>
        card.claimedByPlayerId !== null && card.revealedRank === null,
    ),
    true,
  );

  state = advanceOnlineRoom(state, 4_105, {
    randomInt: () => 0,
    durations,
  });
  assert.equal(state.phase, "rank-reveal");
  view = projectOnlineRoom(state, "p3");
  assert.equal(view.rankSelection.stage, "revealed");
  assert.deepEqual(
    view.rankSelection.cards
      .map((card) => card.revealedRank)
      .sort(),
    [1, 2, 3, 4],
  );
  const revealedOrder = [...state.rankSelection.cards]
    .sort((left, right) => left.rank - right.rank)
    .map((card) => card.claimedByPlayerId);
  // During the flip animation, the rank-card result is visible only through
  // rankSelection. Seat order and role labels must not jump ahead and spoil it.
  assert.deepEqual(
    state.players.map((player) => player.id),
    ["p1", "p2", "p3", "p4"],
  );

  state = advanceOnlineRoom(state, 5_605, {
    randomInt: () => 0,
    durations,
  });
  assert.equal(state.phase, "rank-confirm");
  view = projectOnlineRoom(state, "p3");
  assert.equal(view.rankSelection.stage, "confirmed");
  assert.equal(view.hand, null);
  assert.deepEqual(
    state.players.map((player) => player.id),
    ["p1", "p2", "p3", "p4"],
  );
  assert.equal(
    state.events.some((event) => event.type === "RANK_ORDER_ASSIGNED"),
    false,
  );

  state = advanceOnlineRoom(state, 6_605, {
    randomInt: () => 0,
    durations,
  });
  assert.equal(state.phase, "reveal-intro");
  assert.equal(state.dealSealed, true);
  assert.deepEqual(
    state.players.map((player) => player.id),
    revealedOrder,
  );
  assert.equal(
    state.events.findLast(
      (event) => event.type === "RANK_ORDER_ASSIGNED",
    )?.at,
    6_605,
  );
  const allCards = state.players.flatMap((player) => state.hands[player.id]);
  assert.equal(allCards.length, 80);
  assert.equal(new Set(allCards.map((card) => card.id)).size, 80);
  assert.deepEqual(
    state.players.map((player) => state.hands[player.id].length),
    [20, 20, 20, 20],
  );
});

test("stale rank-choice revisions allow distinct slots but reject a claimed slot", () => {
  let state = readyEveryone(createFourPlayerLobby());
  const deps = {
    randomInt: () => 0,
    durations: { ...instantRankDurations },
  };
  state = applyOnlineCommand(
    state,
    "p1",
    {
      id: "start-concurrent-rank-choice",
      expectedRevision: state.revision,
      type: "START_MATCH",
    },
    100,
    deps,
  );
  state = advanceOnlineRoom(state, state.phaseEndsAt, deps);
  const sharedRevision = state.revision;
  const [firstSlot, secondSlot] = state.rankSelection.cards;

  state = applyOnlineCommand(
    state,
    "p1",
    {
      id: "concurrent-rank-p1",
      expectedRevision: sharedRevision,
      type: "CHOOSE_RANK_CARD",
      slotIndex: firstSlot.slotIndex,
    },
    101,
    deps,
  );
  assert.throws(
    () =>
      applyOnlineCommand(
        state,
        "p2",
        {
          id: "concurrent-rank-same-slot",
          expectedRevision: sharedRevision,
          type: "CHOOSE_RANK_CARD",
          slotIndex: firstSlot.slotIndex,
        },
        102,
        deps,
      ),
    (error) =>
      error instanceof OnlineGameError && error.code === "RANK_CARD_CLAIMED",
  );

  state = applyOnlineCommand(
    state,
    "p2",
    {
      id: "concurrent-rank-distinct-slot",
      expectedRevision: sharedRevision,
      type: "CHOOSE_RANK_CARD",
      slotIndex: secondSlot.slotIndex,
    },
    103,
    deps,
  );
  assert.equal(
    state.rankSelection.cards.find(
      (card) => card.slotIndex === secondSlot.slotIndex,
    ).claimedByPlayerId,
    "p2",
  );
});

test("rank order controls remainder dealing and later rounds skip rank choice", () => {
  let state = readyEveryone(createSixPlayerLobby());
  state = startAndAssignJoinOrder(state, 200);

  assert.equal(state.round, 1);
  assert.deepEqual(
    state.players.map((player) => player.id),
    ["p1", "p2", "p3", "p4", "p5", "p6"],
  );
  assert.deepEqual(
    state.players.map((player) => state.hands[player.id].length),
    [13, 13, 13, 13, 14, 14],
  );

  state.phase = "round-end";
  state.phaseEndsAt = null;
  state.finishOrder = ["p6", "p5", "p4", "p3", "p2", "p1"];
  state = command(state, "p1", "START_NEXT_ROUND", {}, 500);

  assert.equal(state.round, 2);
  assert.equal(state.phase, "reveal-intro");
  assert.equal(state.rankSelection, null);
  assert.deepEqual(
    state.players.map((player) => player.id),
    ["p6", "p5", "p4", "p3", "p2", "p1"],
  );
  assert.equal(
    state.events
      .filter((event) => event.type === "RANK_CHOICE_INTRO_STARTED")
      .length,
    1,
  );
});

test("the concealed reveal intro already projects only the viewer's stable hand", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state = startAndAssignJoinOrder(state);

  assert.equal(state.phase, "reveal-intro");
  const firstView = projectOnlineRoom(state, "p1");
  const firstCardIds = new Set(state.hands.p1.map((card) => card.id));
  const secondCardIds = new Set(state.hands.p2.map((card) => card.id));
  assert.deepEqual(
    new Set(firstView.hand.map((card) => card.id)),
    firstCardIds,
  );
  assert.equal(
    firstView.hand.some((card) => secondCardIds.has(card.id)),
    false,
  );

  const serialized = JSON.stringify(firstView);
  for (const cardId of secondCardIds) {
    assert.equal(serialized.includes(`"id":"${cardId}"`), false);
  }

  state = advanceOnlineRoom(state, state.phaseEndsAt, {
    randomInt: () => 0,
  });
  assert.equal(state.phase, "hand-reveal");
  assert.deepEqual(projectOnlineRoom(state, "p1").hand, firstView.hand);
});

test("lobby readiness never deals before opening ranks are assigned", () => {
  let state = readyEveryone(createFourPlayerLobby());
  assert.equal(state.dealSealed, false);
  assert.deepEqual(state.hands, {});

  state = command(state, "p4", "SET_READY", { ready: false }, 30);
  assert.equal(state.dealSealed, false);
  assert.deepEqual(state.hands, {});

  state = command(state, "p4", "SET_READY", { ready: true }, 31);
  assert.equal(state.dealSealed, false);
  assert.deepEqual(state.hands, {});
});

test("the server validates actions, locks animation time, and deduplicates commands", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "playing";
  state.phaseEndsAt = null;
  state.turnDeadline = 30_000;
  state.currentIndex = 0;
  state.actionLockUntil = null;
  state.table = null;
  state.hands = {
    p1: [{ id: "p1-12", rank: 12 }, { id: "p1-1", rank: 1 }],
    p2: [{ id: "p2-11", rank: 11 }],
    p3: [{ id: "p3-10", rank: 10 }],
    p4: [{ id: "p4-9", rank: 9 }],
  };

  const play = {
    id: "play-once",
    expectedRevision: state.revision,
    type: "PLAY_CARDS",
    cardIds: ["p1-12"],
  };
  const played = applyOnlineCommand(state, "p1", play, 100);
  assert.equal(played.table.rank, 12);
  assert.equal(played.currentIndex, 1);
  assert.equal(played.actionLockUntil, 2_650);
  assert.equal(played.turnDeadline, 32_650);
  assert.equal(played.events.at(-1).type, "TURN_STARTED");
  assert.equal(played.events.at(-1).at, played.actionLockUntil);
  assert.equal(played.events.at(-1).payload.endsAt, played.turnDeadline);
  assert.equal(
    played.events.findLast((event) => event.type === "CARDS_PLAYED")?.payload
      .previousTable,
    null,
  );

  assert.throws(
    () =>
      applyOnlineCommand(
        played,
        "p2",
        {
          id: "too-fast",
          expectedRevision: played.revision,
          type: "PLAY_CARDS",
          cardIds: ["p2-11"],
        },
        200,
      ),
    (error) =>
      error instanceof OnlineGameError && error.code === "ACTION_IN_PROGRESS",
  );

  const duplicate = applyOnlineCommand(played, "p1", play, 300);
  assert.equal(duplicate.revision, played.revision);
  assert.deepEqual(duplicate.hands, played.hands);

  const continued = applyOnlineCommand(
    played,
    "p2",
    {
      id: "play-after-animation",
      expectedRevision: played.revision,
      type: "PLAY_CARDS",
      cardIds: ["p2-11"],
    },
    3_700,
  );
  assert.equal(continued.table.rank, 11);
  assert.equal(continued.actionLockUntil, 6_250);
  assert.equal(continued.turnDeadline, 36_250);
  assert.deepEqual(
    continued.events.findLast((event) => event.type === "CARDS_PLAYED")
      ?.payload.previousTable,
    {
      rank: 12,
      count: 1,
      playerId: "p1",
      cards: [{ id: "p1-12", rank: 12 }],
    },
  );
});

test("playing turns use a server-authoritative 30 second deadline and timeout PASS even on an empty table", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "play-intro";
  state.phaseEndsAt = 100;
  state.turnDeadline = null;
  state.currentIndex = 0;
  state.actionLockUntil = null;
  state.table = null;
  state.lastPlayedId = null;
  state.hands = {
    p1: [{ id: "p1-12", rank: 12 }],
    p2: [{ id: "p2-11", rank: 11 }],
    p3: [{ id: "p3-10", rank: 10 }],
    p4: [{ id: "p4-9", rank: 9 }],
  };

  state = advanceOnlineRoom(state, 100);
  assert.equal(state.phase, "playing");
  assert.equal(state.players[state.currentIndex].id, "p1");
  assert.equal(state.turnDeadline, 30_100);
  assert.equal(projectOnlineRoom(state, "p2").turnDeadline, 30_100);

  const handsBeforeTimeout = structuredClone(state.hands);
  const beforeDeadline = advanceOnlineRoom(state, 30_099);
  assert.equal(beforeDeadline, state);

  state = advanceOnlineRoom(state, 30_100);
  assert.deepEqual(state.hands, handsBeforeTimeout);
  assert.equal(state.table, null);
  assert.equal(state.players[state.currentIndex].id, "p2");
  assert.equal(state.actionLockUntil, 31_900);
  assert.equal(state.turnDeadline, 61_900);
  assert.deepEqual(state.passedPlayerIds, ["p1"]);
  const timedOut = state.events.at(-2);
  assert.equal(timedOut.type, "PLAYER_PASSED");
  assert.deepEqual(timedOut.payload, {
    playerId: "p1",
    automatic: true,
    reason: "timeout",
  });
  assert.equal(timedOut.at, 30_100);
  const nextTurn = state.events.at(-1);
  assert.equal(nextTurn.type, "TURN_STARTED");
  assert.equal(nextTurn.at, 31_900);
  assert.equal(nextTurn.payload.endsAt, 61_900);

  state = advanceOnlineRoom(state, 125_500);
  assert.equal(state.players[state.currentIndex].id, "p1");
  assert.equal(state.turnDeadline, 157_300);
  assert.deepEqual(
    new Set(state.passedPlayerIds),
    new Set(["p1", "p2", "p3", "p4"]),
  );
  const timeoutPasses = state.events.filter(
    (event) =>
      event.type === "PLAYER_PASSED" &&
      event.payload.reason === "timeout",
  );
  assert.deepEqual(
    timeoutPasses.map((event) => event.payload.playerId),
    ["p1", "p2", "p3", "p4"],
  );
});

test("a player with fewer cards than the occupied table requires is automatically passed", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "playing";
  state.phaseEndsAt = null;
  state.turnDeadline = 30_000;
  state.currentIndex = 0;
  state.actionLockUntil = 100;
  state.botActionAt = null;
  state.table = {
    rank: 8,
    count: 2,
    playerId: "p4",
    cards: [
      { id: "table-8-a", rank: 8 },
      { id: "table-8-b", rank: 8 },
    ],
  };
  state.lastPlayedId = "p4";
  state.hands = {
    p1: [{ id: "p1-only-card", rank: 1 }],
    p2: [
      { id: "p2-a", rank: 7 },
      { id: "p2-b", rank: 7 },
    ],
    p3: [
      { id: "p3-a", rank: 6 },
      { id: "p3-b", rank: 6 },
    ],
    p4: [
      { id: "p4-a", rank: 12 },
      { id: "p4-b", rank: 12 },
    ],
  };

  const handBeforePass = structuredClone(state.hands.p1);
  state = advanceOnlineRoom(state, 100);

  assert.deepEqual(state.hands.p1, handBeforePass);
  assert.equal(state.players[state.currentIndex].id, "p2");
  assert.deepEqual(state.passedPlayerIds, ["p1"]);
  assert.equal(state.actionLockUntil, 1_900);
  const automaticPass = state.events.findLast(
    (event) =>
      event.type === "PLAYER_PASSED" &&
      event.payload.playerId === "p1",
  );
  assert.deepEqual(automaticPass?.payload, {
    playerId: "p1",
    automatic: true,
    reason: "insufficient-cards",
    previousTable: {
      rank: 8,
      count: 2,
      playerId: "p4",
      cards: [
        { id: "table-8-a", rank: 8 },
        { id: "table-8-b", rank: 8 },
      ],
    },
  });
  assert.equal(automaticPass?.at, 100);
});

test("timeout PASS clears an occupied trick and resets the next leader's full deadline", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "playing";
  state.phaseEndsAt = null;
  state.turnDeadline = 30_000;
  state.currentIndex = 0;
  state.actionLockUntil = null;
  state.table = {
    rank: 8,
    count: 1,
    playerId: "p4",
    cards: [{ id: "p4-table", rank: 8 }],
  };
  state.lastPlayedId = "p4";
  state.hands = {
    p1: [{ id: "p1-12", rank: 12 }],
    p2: [{ id: "p2-11", rank: 11 }],
    p3: [{ id: "p3-10", rank: 10 }],
    p4: [{ id: "p4-9", rank: 9 }],
  };

  state = advanceOnlineRoom(state, 93_600);
  assert.equal(state.table, null);
  assert.equal(state.players[state.currentIndex].id, "p4");
  assert.equal(state.actionLockUntil, 95_400);
  assert.equal(state.turnDeadline, 125_400);
  assert.deepEqual(state.passedPlayerIds, []);
  assert.equal(
    state.events.some(
      (event) =>
        event.type === "TRICK_CLEARED" &&
        event.payload.nextPlayerId === "p4",
    ),
    true,
  );
  assert.deepEqual(
    state.events.findLast((event) => event.type === "PLAYER_PASSED")
      ?.payload.previousTable,
    {
      rank: 8,
      count: 1,
      playerId: "p4",
      cards: [{ id: "p4-table", rank: 8 }],
    },
  );
});

test("round end and room reset clear the playing turn deadline", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "playing";
  state.phaseEndsAt = null;
  state.turnDeadline = 30_000;
  state.currentIndex = 0;
  state.actionLockUntil = null;
  state.table = null;
  state.lastPlayedId = null;
  state.finishOrder = ["p2", "p3"];
  state.hands = {
    p1: [{ id: "p1-last", rank: 12 }],
    p2: [],
    p3: [],
    p4: [{ id: "p4-last", rank: 11 }],
  };

  state = command(
    state,
    "p1",
    "PLAY_CARDS",
    { cardIds: ["p1-last"] },
    100,
  );
  assert.equal(state.phase, "round-end");
  assert.equal(state.turnDeadline, null);

  state = command(state, "p1", "RESET_ROOM", {}, 101);
  assert.equal(state.phase, "lobby");
  assert.equal(state.turnDeadline, null);
});

test("unknown and malformed online commands cannot corrupt room state", () => {
  const state = createFourPlayerLobby();

  assert.throws(
    () =>
      applyOnlineCommand(
        state,
        "p1",
        {
          id: "malicious",
          expectedRevision: state.revision,
          type: "DELETE_ROOM",
        },
        100,
      ),
    (error) =>
      error instanceof OnlineGameError &&
      error.code === "INVALID_COMMAND_TYPE",
  );

  assert.throws(
    () =>
      applyOnlineCommand(
        state,
        "p1",
        {
          id: "bad-ready",
          expectedRevision: state.revision,
          type: "SET_READY",
          ready: "yes",
        },
        101,
      ),
    (error) =>
      error instanceof OnlineGameError &&
      error.code === "INVALID_READY_VALUE",
  );
});

test("tax card identities are visible only to each exchange pair", () => {
  let state = readyEveryone(createFourPlayerLobby());
  const shortIntros = {
    ...instantRankDurations,
    revealIntroMs: 0,
    handRevealMs: 0,
    taxIntroMs: 0,
    taxSelectionMs: 10_000,
    taxTributeMs: 1_000,
    taxReturnMs: 1_000,
    playIntroMs: 0,
  };
  state = startAndAssignJoinOrder(state, 100, shortIntros);
  state.hands = {
    p1: [
      { id: "noble-a", rank: 12 },
      { id: "noble-b", rank: 11 },
    ],
    p2: [{ id: "lesser-noble", rank: 10 }],
    p3: [
      { id: "lesser-peon-best", rank: 2 },
      { id: "lesser-peon-other", rank: 8 },
    ],
    p4: [
      { id: "great-peon-joker", rank: 13 },
      { id: "great-peon-one", rank: 1 },
      { id: "great-peon-two", rank: 2 },
    ],
  };
  state.round = 2;
  state = advanceOnlineRoom(state, state.phaseEndsAt, {
    durations: shortIntros,
  });
  assert.equal(state.phase, "tax-selection");
  assert.deepEqual(state.taxExchanges[0].peonCardIds, [
    "great-peon-one",
    "great-peon-two",
  ]);
  assert.deepEqual(state.taxExchanges[1].peonCardIds, ["lesser-peon-best"]);

  state = applyOnlineCommand(
    state,
    "p1",
    {
      id: "great-return",
      expectedRevision: state.revision,
      type: "SELECT_TAX_RETURN",
      cardIds: ["noble-a", "noble-b"],
    },
    200,
  );
  state = applyOnlineCommand(
    state,
    "p2",
    {
      id: "lesser-return",
      expectedRevision: state.revision,
      type: "SELECT_TAX_RETURN",
      cardIds: ["lesser-noble"],
    },
    201,
  );
  assert.equal(state.phase, "tax-tribute");

  const greatDalmutiView = projectOnlineRoom(state, "p1");
  const greatPeonView = projectOnlineRoom(state, "p4");
  const unrelatedView = projectOnlineRoom(state, "p3");
  for (const view of [greatDalmutiView, greatPeonView]) {
    const payload = JSON.stringify(
      view.events.findLast((event) => event.type === "TAX_TRIBUTE"),
    );
    assert.match(payload, /great-peon-one/);
    assert.match(payload, /great-peon-two/);
    assert.doesNotMatch(payload, /great-peon-joker/);
  }
  const unrelatedPayload = JSON.stringify(unrelatedView.events);
  assert.doesNotMatch(
    unrelatedPayload,
    /great-peon-one|great-peon-two/,
  );

  state = advanceOnlineRoom(state, 1_201);
  assert.equal(state.phase, "tax-return");
  const returnRoutes = state.events.findLast(
    (event) => event.type === "TAX_RETURN_STARTED",
  )?.payload.routes;
  assert.deepEqual(
    returnRoutes.map((route) => [route.fromPlayerId, route.toPlayerId]),
    [
      ["p1", "p4"],
      ["p2", "p3"],
    ],
  );
  assert.deepEqual(
    state.hands.p1.map((card) => card.id).sort(),
    ["great-peon-one", "great-peon-two"].sort(),
  );
  assert.deepEqual(
    state.hands.p4.map((card) => card.id).sort(),
    ["great-peon-joker", "noble-a", "noble-b"].sort(),
  );

  state = advanceOnlineRoom(state, state.phaseEndsAt, {
    durations: shortIntros,
  });
  assert.equal(
    state.events.findLast(
      (event) => event.type === "PLAY_INTRO_STARTED",
    )?.payload.round,
    2,
  );
});

test("tax-selection timeout keeps submitted returns and automatically fills only missing choices", () => {
  let state = readyEveryone(createFourPlayerLobby());
  const durations = {
    ...instantRankDurations,
    revealIntroMs: 0,
    handRevealMs: 0,
    taxIntroMs: 0,
    taxSelectionMs: 250,
    taxTributeMs: 10,
    taxReturnMs: 10,
    playIntroMs: 0,
  };
  state = startAndAssignJoinOrder(state, 100, durations);
  state.hands = {
    p1: [
      { id: "p1-manual-a", rank: 12 },
      { id: "p1-manual-b", rank: 11 },
    ],
    p2: [
      { id: "p2-auto", rank: 10 },
      { id: "p2-keep", rank: 3 },
    ],
    p3: [
      { id: "p3-tax", rank: 2 },
      { id: "p3-other", rank: 8 },
    ],
    p4: [
      { id: "p4-tax-a", rank: 1 },
      { id: "p4-tax-b", rank: 2 },
      { id: "p4-other", rank: 9 },
    ],
  };
  state.round = 2;
  state = advanceOnlineRoom(state, state.phaseEndsAt, { durations });
  assert.equal(state.phase, "tax-selection");
  const selectionTimeoutAt = state.phaseEndsAt;

  state = applyOnlineCommand(
    state,
    "p1",
    {
      id: "manual-return-before-timeout",
      expectedRevision: state.revision,
      type: "SELECT_TAX_RETURN",
      cardIds: ["p1-manual-a", "p1-manual-b"],
    },
    selectionTimeoutAt - 1,
    { durations },
  );
  assert.equal(state.phase, "tax-selection");

  state = advanceOnlineRoom(state, selectionTimeoutAt, { durations });
  assert.equal(state.phase, "tax-tribute");
  assert.deepEqual(state.taxExchanges[0].nobleCardIds, [
    "p1-manual-a",
    "p1-manual-b",
  ]);
  assert.deepEqual(state.taxExchanges[1].nobleCardIds, ["p2-auto"]);

  const automaticSelections = state.events.filter(
    (event) =>
      event.type === "TAX_RETURN_SELECTED" &&
      event.payload.automatic === true,
  );
  assert.equal(automaticSelections.length, 1);
  assert.deepEqual(automaticSelections[0].playerIds, ["p2"]);
  assert.equal(automaticSelections[0].at, selectionTimeoutAt);
  assert.equal(state.phaseEndsAt, selectionTimeoutAt + durations.taxTributeMs);
});

test("playing rank 1 auto-passes every other active player and starts a new trick", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "playing";
  state.phaseEndsAt = null;
  state.currentIndex = 0;
  state.actionLockUntil = null;
  state.table = {
    rank: 2,
    count: 2,
    playerId: "p4",
    cards: [
      { id: "old-2-a", rank: 2 },
      { id: "old-2-b", rank: 2 },
    ],
  };
  state.lastPlayedId = "p4";
  state.hands = {
    p1: [
      { id: "dalmuti", rank: 1 },
      { id: "dalmuti-joker", rank: 13 },
      { id: "p1-12", rank: 12 },
    ],
    p2: [{ id: "p2-11", rank: 11 }],
    p3: [{ id: "p3-10", rank: 10 }],
    p4: [{ id: "p4-9", rank: 9 }],
  };

  state = command(
    state,
    "p1",
    "PLAY_CARDS",
    { cardIds: ["dalmuti", "dalmuti-joker"] },
    100,
  );

  assert.equal(state.table, null);
  assert.equal(state.currentIndex, 0);
  assert.equal(state.players[state.currentIndex].id, "p1");
  assert.equal(state.actionLockUntil, 3_700);
  assert.equal(state.turnDeadline, 33_700);
  assert.equal(state.events.at(-1).type, "TURN_STARTED");
  assert.equal(state.events.at(-1).at, state.actionLockUntil);
  assert.equal(state.events.at(-1).payload.endsAt, state.turnDeadline);
  assert.deepEqual(state.passedPlayerIds, []);
  const effect = state.events.find((event) => event.type === "DALMUTI_EFFECT");
  assert.deepEqual(effect.payload.cards, [
    { id: "dalmuti", rank: 1 },
    { id: "dalmuti-joker", rank: 13 },
  ]);
  assert.equal(effect.payload.rank, 1);
  assert.equal(effect.payload.count, 2);
  assert.deepEqual(effect.payload.previousTable, {
    rank: 2,
    count: 2,
    playerId: "p4",
    cards: [
      { id: "old-2-a", rank: 2 },
      { id: "old-2-b", rank: 2 },
    ],
  });
  assert.deepEqual(effect.payload.autoPassedPlayerIds, ["p2", "p3", "p4"]);
  const automaticPasses = state.events.filter(
    (event) =>
      event.type === "PLAYER_PASSED" &&
      event.payload.reason === "dalmuti",
  );
  assert.deepEqual(
    automaticPasses.map((event) => event.payload.playerId),
    ["p2", "p3", "p4"],
  );
  assert.equal(
    state.events.some(
      (event) =>
        event.type === "TRICK_CLEARED" &&
        event.payload.reason === "dalmuti",
    ),
    true,
  );
});

test("the opening round skips taxation after revolution decisions", () => {
  let state = readyEveryone(createFourPlayerLobby());
  const durations = {
    ...instantRankDurations,
    revealIntroMs: 0,
    handRevealMs: 0,
    revolutionDecisionMs: 10_000,
    revolutionIntroMs: 1,
    taxIntroMs: 0,
    taxSelectionMs: 10_000,
    taxTributeMs: 1_000,
    taxReturnMs: 1_000,
    playIntroMs: 0,
  };
  state = startAndAssignJoinOrder(state, 100, durations);
  state.hands = {
    p1: [
      { id: "joker-1", rank: 13 },
      { id: "joker-2", rank: 13 },
      { id: "p1-return-a", rank: 12 },
      { id: "p1-return-b", rank: 11 },
    ],
    p2: [{ id: "p2-return", rank: 10 }],
    p3: [
      { id: "p3-tax", rank: 2 },
      { id: "p3-other", rank: 9 },
    ],
    p4: [
      { id: "p4-tax-a", rank: 1 },
      { id: "p4-tax-b", rank: 2 },
      { id: "p4-other", rank: 8 },
    ],
  };
  state = advanceOnlineRoom(state, state.phaseEndsAt, { durations });

  assert.equal(state.round, 1);
  assert.equal(state.phase, "revolution");
  assert.equal(state.revolutionHolderId, "p1");
  assert.equal(
    state.events.some((event) => event.type === "REVOLUTION_DECISION_STARTED"),
    true,
  );
  const pendingRevolution = state;
  const declineAt = state.updatedAt + 1;

  state = applyOnlineCommand(
    state,
    "p1",
    {
      id: "decline-opening-revolution",
      expectedRevision: state.revision,
      type: "CHOOSE_REVOLUTION",
      declare: false,
    },
    declineAt,
    { durations },
  );
  assert.equal(state.phase, "tax-intro");
  assert.deepEqual(state.taxExchanges, []);
  const skippedTaxEvent = state.events.findLast(
    (event) => event.type === "TAX_INTRO_STARTED",
  );
  assert.equal(skippedTaxEvent?.payload.skipped, true);
  assert.deepEqual(skippedTaxEvent?.payload.routes, []);
  state = advanceOnlineRoom(state, declineAt, { durations });

  assert.equal(state.phase, "playing");
  assert.equal(
    state.events.some((event) => event.type === "TAX_SELECTION_STARTED"),
    false,
  );

  const timedOut = advanceOnlineRoom(
    pendingRevolution,
    pendingRevolution.phaseEndsAt,
    { durations },
  );
  assert.equal(timedOut.phase, "playing");
  assert.equal(
    timedOut.events.some((event) => event.type === "REVOLUTION_DECLINED"),
    true,
  );
  assert.equal(
    timedOut.events.findLast(
      (event) => event.type === "TAX_INTRO_STARTED",
    )?.payload.skipped,
    true,
  );

  const declared = applyOnlineCommand(
    pendingRevolution,
    "p1",
    {
      id: "declare-opening-revolution",
      expectedRevision: pendingRevolution.revision,
      type: "CHOOSE_REVOLUTION",
      declare: true,
    },
    pendingRevolution.updatedAt + 1,
    { durations },
  );
  assert.deepEqual(declared.declaredRevolution, {
    round: 1,
    playerId: "p1",
    kind: "revolution",
  });
  assert.deepEqual(
    projectOnlineRoom(declared, "p4").declaredRevolution,
    declared.declaredRevolution,
  );
  const declaredEvent = declared.events.findLast(
    (event) => event.type === "REVOLUTION_DECLARED",
  );
  assert.equal(declaredEvent.payload.endsAt, declared.phaseEndsAt);

  const playingWithRevolution = advanceOnlineRoom(
    declared,
    declared.phaseEndsAt,
    { durations },
  );
  assert.equal(playingWithRevolution.phase, "playing");
  assert.deepEqual(
    playingWithRevolution.declaredRevolution,
    declared.declaredRevolution,
  );
  assert.equal(
    playingWithRevolution.events.findLast(
      (event) => event.type === "TAX_INTRO_STARTED",
    )?.payload.skipped,
    true,
  );

  const endedRound = structuredClone(playingWithRevolution);
  endedRound.phase = "round-end";
  endedRound.phaseEndsAt = null;
  endedRound.turnDeadline = null;
  endedRound.actionLockUntil = null;
  endedRound.finishOrder = endedRound.players.map((player) => player.id);
  const nextRound = command(
    endedRound,
    endedRound.hostId,
    "START_NEXT_ROUND",
    {},
    endedRound.updatedAt + 1,
  );
  assert.equal(nextRound.round, 2);
  assert.equal(nextRound.declaredRevolution, null);
});

test("a first-act great revolution reverses ranks before the no-tax intro", () => {
  let state = createFourPlayerLobby();
  const originalOrder = state.players.map((player) => player.id);
  const originalRoles = state.players.map((player) => player.role);
  const greatPeon = state.players.find(
    (player) => player.role === "great-peon",
  );
  assert.ok(greatPeon);

  const durations = {
    revolutionIntroMs: 300,
    greatRevolutionSwapMs: 200,
    taxIntroMs: 240,
    playIntroMs: 100,
  };
  state.phase = "revolution";
  state.phaseEndsAt = 10_000;
  state.round = 1;
  state.revolutionHolderId = greatPeon.id;
  state.durations = { ...state.durations, ...durations };

  const declaredAt = 100;
  state = applyOnlineCommand(
    state,
    greatPeon.id,
    {
      id: "declare-staged-great-revolution",
      expectedRevision: state.revision,
      type: "CHOOSE_REVOLUTION",
      declare: true,
    },
    declaredAt,
    { durations },
  );

  assert.equal(state.phase, "revolution-intro");
  assert.equal(state.phaseEndsAt, declaredAt + durations.revolutionIntroMs);
  assert.deepEqual(
    state.players.map((player) => player.id),
    originalOrder,
    "the declaration animation must not move seats yet",
  );
  assert.deepEqual(
    state.players.map((player) => player.role),
    originalRoles,
    "the declaration animation must preserve every rank",
  );
  assert.deepEqual(state.declaredRevolution, {
    round: 1,
    playerId: greatPeon.id,
    kind: "great-revolution",
  });

  const declarationEvent = state.events.findLast(
    (event) => event.type === "REVOLUTION_INTRO_STARTED",
  );
  assert.equal(declarationEvent?.payload.playerId, greatPeon.id);
  assert.equal(declarationEvent?.payload.kind, "great");
  assert.equal(declarationEvent?.payload.round, 1);

  state = advanceOnlineRoom(state, state.phaseEndsAt - 1, { durations });
  assert.equal(state.phase, "revolution-intro");
  assert.deepEqual(
    state.players.map((player) => player.id),
    originalOrder,
  );

  state = advanceOnlineRoom(state, state.phaseEndsAt, { durations });
  assert.equal(state.phase, "great-revolution-swap");
  assert.equal(
    state.phaseEndsAt,
    declaredAt +
      durations.revolutionIntroMs +
      durations.greatRevolutionSwapMs,
  );
  assert.deepEqual(
    state.players.map((player) => player.id),
    [...originalOrder].reverse(),
    "players move only when the rank-swap announcement starts",
  );
  assert.deepEqual(
    state.players.map((player) => player.role),
    originalRoles,
    "roles are reassigned in rank order after reversing the players",
  );

  const swapEvent = state.events.findLast(
    (event) => event.type === "GREAT_REVOLUTION_RANK_SWAP_STARTED",
  );
  assert.equal(swapEvent?.payload.playerId, greatPeon.id);
  assert.equal(swapEvent?.payload.round, 1);
  assert.equal(swapEvent?.payload.endsAt, state.phaseEndsAt);
  assert.ok(swapEvent.seq > declarationEvent.seq);

  state = advanceOnlineRoom(state, state.phaseEndsAt - 1, { durations });
  assert.equal(state.phase, "great-revolution-swap");

  state = advanceOnlineRoom(state, state.phaseEndsAt, { durations });
  assert.equal(state.phase, "tax-intro");
  assert.equal(
    state.events.findLast(
      (event) => event.type === "TAX_INTRO_STARTED",
    )?.payload.skipped,
    true,
  );
  assert.deepEqual(
    state.players.map((player) => player.id),
    [...originalOrder].reverse(),
  );

  state = advanceOnlineRoom(state, state.phaseEndsAt, { durations });
  assert.equal(state.phase, "play-intro");
  assert.ok(
    state.events.findLast((event) => event.type === "PLAY_INTRO_STARTED").seq >
      swapEvent.seq,
  );
});

test("legacy persisted room JSON hydrates new optional ranking fields safely", () => {
  const state = readyEveryone(createFourPlayerLobby());
  delete state.rankSelection;
  delete state.declaredRevolution;
  delete state.durations.rankChoiceIntroMs;
  delete state.durations.rankRevealDelayMs;
  delete state.durations.rankConfirmMs;
  delete state.durations.revolutionIntroMs;
  delete state.durations.greatRevolutionSwapMs;
  state.durations.rankRevealMs = 2_800;

  const view = projectOnlineRoom(state, "p1");
  assert.equal(view.rankSelection, null);
  assert.equal(view.declaredRevolution, null);

  const started = applyOnlineCommand(
    state,
    "p1",
    {
      id: "legacy-start",
      expectedRevision: state.revision,
      type: "START_MATCH",
    },
    100,
    { randomInt: () => 0 },
  );
  assert.equal(started.phase, "rank-intro");
  assert.equal(started.durations.rankChoiceIntroMs, 3_300);
  assert.equal(started.durations.rankRevealDelayMs, 1_500);
  assert.equal(started.durations.rankRevealMs, 3_400);
  assert.equal(started.durations.rankConfirmMs, 2_600);
  assert.equal(started.durations.revealIntroMs, 2_400);
  assert.equal(started.durations.handRevealMs, 1_400);
  assert.equal(started.durations.revolutionDecisionMs, 20_000);
  assert.equal(started.durations.revolutionIntroMs, 3_300);
  assert.equal(started.durations.greatRevolutionSwapMs, 2_600);
  assert.equal(started.durations.taxIntroMs, 2_400);
  assert.equal(started.durations.taxSelectionMs, 45_000);
  assert.equal(started.durations.taxTributeMs, 6_000);
  assert.equal(started.durations.taxReturnMs, 6_000);
  assert.equal(started.durations.playIntroMs, 2_600);
});

test("the host can reset a room and a non-host can leave without stale match data", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "playing";
  state.round = 3;
  state.players = state.players.map((player, index) => ({
    ...player,
    score: index + 2,
  }));

  state = command(state, "p1", "RESET_ROOM", {}, 200);
  assert.equal(state.phase, "lobby");
  assert.equal(state.round, 0);
  assert.equal(state.dealSealed, false);
  assert.deepEqual(state.hands, {});
  assert.equal(state.players.every((player) => !player.ready), true);
  assert.equal(state.players.every((player) => player.score === 0), true);
  assert.equal(state.events.at(-1).type, "ROOM_RESET");

  state = command(state, "p4", "LEAVE_ROOM", {}, 201);
  assert.deepEqual(
    state.players.map((player) => player.id),
    ["p1", "p2", "p3"],
  );
  assert.equal(state.phase, "lobby");
  assert.equal(state.events.some((event) => event.type === "PLAYER_LEFT"), true);

  assert.throws(
    () => command(state, "p1", "LEAVE_ROOM", {}, 202),
    (error) =>
      error instanceof OnlineGameError && error.code === "HOST_CANNOT_LEAVE",
  );
});

test("host reset APIs dispose the room and all stored memberships", async () => {
  const [store, resetRoute, commandRoute, leaveRoute] = await Promise.all([
    readFile(
      new URL("../lib/online-room-store.ts", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL(
        "../app/api/online/rooms/[code]/reset/route.ts",
        import.meta.url,
      ),
      "utf8",
    ),
    readFile(
      new URL(
        "../app/api/online/rooms/[code]/commands/route.ts",
        import.meta.url,
      ),
      "utf8",
    ),
    readFile(
      new URL(
        "../app/api/online/rooms/[code]/leave/route.ts",
        import.meta.url,
      ),
      "utf8",
    ),
  ]);

  assert.match(store, /export async function deleteStoredOnlineRoom/);
  assert.match(
    store,
    /DELETE FROM online_room_members WHERE room_code = \?/,
  );
  assert.match(store, /DELETE FROM online_rooms WHERE code = \?/);
  assert.match(resetRoute, /applyOnlineCommand\(/);
  assert.match(resetRoute, /await deleteStoredOnlineRoom\(code\)/);
  assert.match(resetRoute, /reset: true/);
  assert.doesNotMatch(resetRoute, /projectOnlineRoom/);
  assert.match(commandRoute, /command\.type === "RESET_ROOM"/);
  assert.match(commandRoute, /await deleteStoredOnlineRoom\(code\)/);
  assert.match(leaveRoute, /await removeOnlineRoomMember\(code, member\.playerId\)/);
});
