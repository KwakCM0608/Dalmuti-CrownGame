import assert from "node:assert/strict";
import test from "node:test";

const {
  V4_MAX_PUBLIC_HISTORY_EVENTS,
  V4_PUBLIC_OBSERVATION_SCHEMA_VERSION,
  buildV4ActorVisibleObservation,
  encodeV4ActorVisibleObservationBytes,
} = await import(
  new URL("../training/v4-public-history.ts", import.meta.url)
);

const ROLE = {
  dalmuti: "great-dalmuti",
  lesserDalmuti: "lesser-dalmuti",
  merchant: "merchant",
  lesserPeon: "lesser-peon",
  peon: "great-peon",
};

function roleAt(index, playerCount) {
  if (index === 0) return ROLE.dalmuti;
  if (index === 1) return ROLE.lesserDalmuti;
  if (index === playerCount - 2) return ROLE.lesserPeon;
  if (index === playerCount - 1) return ROLE.peon;
  return ROLE.merchant;
}

function firstDeckCards(count) {
  const deck = [];
  for (let rank = 1; rank <= 12; rank += 1) {
    for (let copy = 0; copy < rank; copy += 1) deck.push({ rank });
  }
  deck.push({ rank: 13 }, { rank: 13 });
  return deck.slice(0, count);
}

function makeObservation(playerCount, actorIndex = 0) {
  const baseCount = Math.floor(80 / playerCount);
  const extraCount = 80 % playerCount;
  const players = Array.from({ length: playerCount }, (_, index) => {
    const receivesExtra = index >= playerCount - extraCount;
    const handCount = baseCount + (receivesExtra ? 1 : 0);
    return {
      id: `player-${index}`,
      handCount,
      finished: false,
      passed: index % 3 === 1,
      role: roleAt(index, playerCount),
      score: index * 2,
    };
  });
  return {
    actorId: players[actorIndex].id,
    act: 1,
    revolution: null,
    ownHand: firstDeckCards(players[actorIndex].handCount),
    publicPlayedCounts: Array.from({ length: 13 }, () => 0),
    players,
    table: null,
    history: [],
  };
}

function passEvent(sequence, actorId, handCount, reason = "manual") {
  return {
    type: "pass",
    sequence,
    actorId,
    handCountBefore: handCount,
    handCountAfter: handCount,
    reason,
  };
}

test("V4 validates and canonicalizes play, pass, clear, and finish events", () => {
  const input = makeObservation(4, 2);
  input.history = [
    {
      type: "play",
      sequence: 7,
      actorId: "player-0",
      handCountBefore: 20,
      handCountAfter: 19,
      rank: 1,
      naturalCount: 1,
      jokerCount: 0,
      totalCount: 1,
    },
    passEvent(8, "player-1", 20, "timeout"),
    {
      type: "clear",
      sequence: 9,
      actorId: "player-0",
      handCountBefore: 19,
      handCountAfter: 19,
      rank: 1,
      naturalCount: 1,
      jokerCount: 0,
      totalCount: 1,
      reason: "all-passed",
      nextLeaderId: "player-0",
    },
    {
      type: "finish",
      sequence: 10,
      actorId: "player-0",
      handCountBefore: 0,
      handCountAfter: 0,
      place: 1,
    },
  ];

  const output = buildV4ActorVisibleObservation(input);
  assert.equal(output.schemaVersion, V4_PUBLIC_OBSERVATION_SCHEMA_VERSION);
  assert.deepEqual(
    output.historyTokens.map((event) => event.type),
    [0, 1, 2, 3],
  );
  assert.deepEqual(
    output.historyTokens.map((event) => event.actorOffset),
    [2, 3, 2, 2],
  );
  assert.equal(output.historyTokens[1].passReason, 2);
  assert.equal(output.historyTokens[2].nextLeaderOffset, 2);
  assert.equal(output.historyTokens[3].finishPlace, 1);
  assert.equal(JSON.stringify(output).includes("player-"), false);
});

test("V4 keeps 192 recent events and converts the older prefix to memory traces", () => {
  const input = makeObservation(10, 7);
  input.history = Array.from({ length: 205 }, (_, sequence) => {
    const player = input.players[sequence % input.players.length];
    return passEvent(
      sequence,
      player.id,
      player.handCount,
      sequence % 2 === 0 ? "manual" : "timeout",
    );
  });

  const output = buildV4ActorVisibleObservation(input);
  assert.equal(output.historyTokens.length, V4_MAX_PUBLIC_HISTORY_EVENTS);
  assert.equal(output.historyTokens[0].sequence, 13);
  assert.equal(output.historyTokens.at(-1).sequence, 204);
  assert.equal(output.truncatedHistoryCount, 13);
  assert.equal(output.memoryTraceVectors.length, 4);
  assert.equal(output.memoryTraceVectors.every((trace) => trace.length === 20), true);
  assert.equal(
    output.memoryTraceVectors.every(
      (trace) => trace.every(Number.isFinite) && trace.some((value) => value > 0),
    ),
    true,
  );
});

test("V4 actor bytes are invariant to hidden-hand permutations and ID names", () => {
  const visibleA = makeObservation(4, 1);
  const hiddenWorldA = {
    "player-0": [1, 2, 3],
    "player-2": [12, 12, 13],
  };
  const hiddenWorldB = {
    "player-0": [12, 12, 13],
    "player-2": [1, 2, 3],
  };
  assert.notDeepEqual(hiddenWorldA, hiddenWorldB);

  const visibleB = structuredClone(visibleA);
  const rename = new Map(
    visibleB.players.map((player, index) => [player.id, `renamed-${index}`]),
  );
  visibleB.actorId = rename.get(visibleB.actorId);
  visibleB.players = visibleB.players.map((player) => ({
    ...player,
    id: rename.get(player.id),
  }));

  assert.deepEqual(
    encodeV4ActorVisibleObservationBytes(visibleA),
    encodeV4ActorVisibleObservationBytes(visibleB),
  );
});

test("V4 rejects malformed events and every attempted private payload", () => {
  const badTransition = makeObservation(4);
  badTransition.history = [
    {
      type: "play",
      sequence: 1,
      actorId: "player-0",
      handCountBefore: 20,
      handCountAfter: 18,
      rank: 1,
      naturalCount: 1,
      jokerCount: 0,
      totalCount: 1,
    },
  ];
  assert.throws(
    () => buildV4ActorVisibleObservation(badTransition),
    /hand-count transition/,
  );

  const leakedHand = makeObservation(4);
  leakedHand.players[1].hand = [{ rank: 1 }];
  assert.throws(
    () => buildV4ActorVisibleObservation(leakedHand),
    /unknown or private field: hand/,
  );

  const leakedTax = makeObservation(4);
  leakedTax.privateTax = { from: "player-3", cards: [1, 2] };
  assert.throws(
    () => buildV4ActorVisibleObservation(leakedTax),
    /unknown or private field: privateTax/,
  );

  const leakedCardId = makeObservation(4);
  leakedCardId.ownHand[0].id = "private-card-id";
  assert.throws(
    () => buildV4ActorVisibleObservation(leakedCardId),
    /unknown or private field: id/,
  );

  const badSequence = makeObservation(4);
  badSequence.history = [
    passEvent(2, "player-0", 20),
    passEvent(2, "player-1", 20),
  ];
  assert.throws(
    () => buildV4ActorVisibleObservation(badSequence),
    /strictly increasing/,
  );
});

test("V4 emits deterministic bytes regardless of source object key order", () => {
  const original = makeObservation(10, 9);
  original.act = 6;
  original.revolution = "great-revolution";
  const reordered = {
    history: original.history,
    table: original.table,
    players: original.players.map((player) => ({
      score: player.score,
      role: player.role,
      passed: player.passed,
      finished: player.finished,
      handCount: player.handCount,
      id: player.id,
    })),
    publicPlayedCounts: original.publicPlayedCounts,
    ownHand: original.ownHand.map(({ rank }) => ({ rank })),
    revolution: original.revolution,
    act: original.act,
    actorId: original.actorId,
  };

  const first = encodeV4ActorVisibleObservationBytes(original);
  const second = encodeV4ActorVisibleObservationBytes(reordered);
  assert.deepEqual(first, second);
  assert.equal(new TextDecoder().decode(first), new TextDecoder().decode(second));
});

test("V4 covers minimum p4 and maximum p10 relative layouts", () => {
  for (const [playerCount, actorIndex] of [
    [4, 3],
    [10, 6],
  ]) {
    const output = buildV4ActorVisibleObservation(
      makeObservation(playerCount, actorIndex),
    );
    assert.equal(output.playerCount, playerCount);
    assert.equal(output.playerTokens.length, playerCount);
    assert.equal(output.memoryTraceVectors.length, 4);
    assert.equal(
      output.memoryTraceVectors.every(
        (trace) => trace.length === 20 && trace.every((value) => value === 0),
      ),
      true,
    );
    assert.equal(output.playerTokens[0].self, 1);
    assert.deepEqual(
      output.playerTokens.map((player) => player.relativeOffset),
      Array.from({ length: playerCount }, (_, index) => index),
    );
    assert.equal(output.ownHandCounts.length, 13);
    assert.equal(output.publicPlayedCounts.length, 13);
  }
});
