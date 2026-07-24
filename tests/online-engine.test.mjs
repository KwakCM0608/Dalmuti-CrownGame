import assert from "node:assert/strict";
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

function readyEveryone(state) {
  let next = state;
  for (const [index, player] of next.players.entries()) {
    next = command(next, player.id, "SET_READY", { ready: true }, 10 + index);
  }
  return next;
}

test("online lobby seals one private 80-card deal before PLAY", () => {
  const state = readyEveryone(createFourPlayerLobby());

  assert.equal(state.phase, "lobby");
  assert.equal(state.dealSealed, true);
  assert.deepEqual(
    state.players.map((player) => state.hands[player.id].length),
    [20, 20, 20, 20],
  );

  const allCards = state.players.flatMap((player) => state.hands[player.id]);
  assert.equal(allCards.length, 80);
  assert.equal(new Set(allCards.map((card) => card.id)).size, 80);

  const firstView = projectOnlineRoom(state, "p1");
  assert.equal(firstView.hand, null);
  assert.equal(firstView.dealSealed, true);
  assert.deepEqual(
    firstView.players.map((player) => player.handCount),
    [20, 20, 20, 20],
  );

  const hiddenCardId = state.hands.p2[0].id;
  assert.equal(JSON.stringify(firstView).includes(`"id":"${hiddenCardId}"`), false);
});

test("PLAY reveals only the viewer hand and never an opponent hand", () => {
  let state = readyEveryone(createFourPlayerLobby());
  const host = state.hostId;
  state = command(state, host, "START_MATCH", {}, 20);

  assert.equal(state.phase, "reveal-intro");
  assert.equal(projectOnlineRoom(state, "p1").hand, null);

  state = advanceOnlineRoom(state, state.phaseEndsAt, {
    randomInt: () => 0,
  });
  assert.equal(state.phase, "hand-reveal");

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
});

test("unreadying discards the sealed deal and readying reseals it", () => {
  let state = readyEveryone(createFourPlayerLobby());
  const previousCards = state.players.flatMap((player) =>
    state.hands[player.id].map((card) => card.id),
  );

  state = command(state, "p4", "SET_READY", { ready: false }, 30);
  assert.equal(state.dealSealed, false);
  assert.deepEqual(state.hands, {});

  state = command(state, "p4", "SET_READY", { ready: true }, 31);
  assert.equal(state.dealSealed, true);
  assert.equal(
    state.players.flatMap((player) => state.hands[player.id]).length,
    80,
  );
  assert.deepEqual(
    state.players.flatMap((player) =>
      state.hands[player.id].map((card) => card.id),
    ),
    previousCards,
  );
});

test("the server validates actions, locks animation time, and deduplicates commands", () => {
  let state = readyEveryone(createFourPlayerLobby());
  state.phase = "playing";
  state.phaseEndsAt = null;
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
  assert.equal(played.actionLockUntil, 1_600);

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
    1_600,
  );
  assert.equal(continued.table.rank, 11);
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
  const shortIntros = {
    revealIntroMs: 0,
    handRevealMs: 0,
    taxIntroMs: 0,
    taxSelectionMs: 10_000,
    taxTributeMs: 1_000,
    taxReturnMs: 1_000,
    playIntroMs: 0,
  };

  state = applyOnlineCommand(
    state,
    "p1",
    {
      id: "start-tax-test",
      expectedRevision: state.revision,
      type: "START_MATCH",
    },
    100,
    { durations: shortIntros },
  );
  state = advanceOnlineRoom(state, 100, { durations: shortIntros });
  assert.equal(state.phase, "tax-selection");
  assert.deepEqual(state.taxExchanges[0].peonCardIds, [
    "great-peon-joker",
    "great-peon-one",
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
    const payload = JSON.stringify(view.events);
    assert.match(payload, /great-peon-joker/);
    assert.match(payload, /great-peon-one/);
  }
  const unrelatedPayload = JSON.stringify(unrelatedView.events);
  assert.doesNotMatch(unrelatedPayload, /great-peon-joker|great-peon-one/);

  state = advanceOnlineRoom(state, 1_201);
  assert.equal(state.phase, "tax-return");
  assert.deepEqual(
    state.hands.p1.map((card) => card.id).sort(),
    ["great-peon-joker", "great-peon-one"].sort(),
  );
  assert.deepEqual(
    state.hands.p4.map((card) => card.id).sort(),
    ["great-peon-two", "noble-a", "noble-b"].sort(),
  );
});
