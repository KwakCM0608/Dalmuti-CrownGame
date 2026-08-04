import assert from "node:assert/strict";
import test from "node:test";

const {
  BOT_DIFFICULTIES,
  chooseBotCardIds,
  chooseBotPlay,
  chooseBotRevolution,
  chooseBotTaxReturn,
  chooseFacedownRankSlot,
  enumerateLegalBotPlays,
  selectForcedBotTribute,
} = await import(new URL("../lib/bot-strategy.ts", import.meta.url));

function card(id, rank) {
  return { id, rank };
}

function observation({
  hand,
  table = null,
  leaderHandCount = 5,
  nextHandCount = 5,
  publicPlayedCards,
}) {
  return {
    actorId: "bot",
    hand,
    table:
      table === null
        ? null
        : {
            ...table,
            playerId: table.playerId ?? "leader",
          },
    players: [
      { id: "leader", handCount: leaderHandCount },
      { id: "bot", handCount: hand.length },
      { id: "next", handCount: nextHandCount },
    ],
    publicPlayedCards,
  };
}

test("exposes the three supported bot difficulties", () => {
  assert.deepEqual(BOT_DIFFICULTIES, ["easy", "normal", "hard"]);
});

test("enumerates every legal response by card id without using a joker pair as a response", () => {
  const plays = enumerateLegalBotPlays(
    observation({
      hand: [
        card("seven-a", 7),
        card("seven-b", 7),
        card("nine", 9),
        card("joker-a", 13),
        card("joker-b", 13),
      ],
      table: { rank: 8, count: 2 },
    }),
  );

  assert.equal(plays.length, 5);
  assert.ok(
    plays.some(
      (play) =>
        play.rank === 7 &&
        play.jokerCount === 0 &&
        play.cardIds.join(",") === "seven-a,seven-b",
    ),
  );
  assert.equal(
    plays.filter((play) => play.rank === 7 && play.jokerCount === 1).length,
    4,
  );
  assert.equal(
    plays.some(
      (play) =>
        play.cardIds.includes("joker-a") &&
        play.cardIds.includes("joker-b"),
    ),
    false,
  );
  assert.equal(plays.some((play) => play.rank === 9), false);
});

test("allows either joker singly or both jokers together as an empty-table lead", () => {
  const plays = enumerateLegalBotPlays(
    observation({
      hand: [card("joker-a", 13), card("joker-b", 13)],
    }),
  );

  assert.deepEqual(
    plays.map((play) => play.cardIds),
    [["joker-a"], ["joker-b"], ["joker-a", "joker-b"]],
  );
});

test("normal and hard bots always prioritize an immediate finish", () => {
  const state = observation({
    hand: [card("six-a", 6), card("six-b", 6)],
    table: { rank: 7, count: 2 },
    leaderHandCount: 4,
  });

  for (const difficulty of ["normal", "hard"]) {
    const decision = chooseBotPlay(state, difficulty);
    assert.equal(decision.action.type, "play");
    assert.deepEqual(decision.action.cardIds, ["six-a", "six-b"]);
    assert.ok(decision.reasons.includes("즉시 완주"));
  }
});

test("normal bot preserves a joker when a natural pair can answer", () => {
  const state = observation({
    hand: [
      card("eight-a", 8),
      card("eight-b", 8),
      card("seven", 7),
      card("joker", 13),
      card("twelve", 12),
    ],
    table: { rank: 9, count: 2 },
  });
  const decision = chooseBotPlay(state, "normal");

  assert.equal(decision.action.type, "play");
  assert.deepEqual(decision.action.cardIds, ["eight-a", "eight-b"]);
  assert.equal(decision.action.jokerCount, 0);
});

test("normal bot sheds an isolated card instead of breaking a useful pair", () => {
  const state = observation({
    hand: [
      card("two-a", 2),
      card("two-b", 2),
      card("eight", 8),
      card("twelve-a", 12),
      card("twelve-b", 12),
    ],
    table: { rank: 13, count: 1 },
  });

  assert.deepEqual(chooseBotCardIds(state, "normal"), ["eight"]);
});

test("normal bot strategically passes rather than split a strong pair", () => {
  const state = observation({
    hand: [card("two-a", 2), card("two-b", 2), card("twelve", 12)],
    table: { rank: 3, count: 1 },
    leaderHandCount: 5,
  });

  assert.equal(chooseBotPlay(state, "normal").action.type, "pass");
});

test("an opponent close to finishing makes the bot spend cards to block them", () => {
  const safeState = observation({
    hand: [card("two-a", 2), card("two-b", 2), card("twelve", 12)],
    table: { rank: 3, count: 1 },
    leaderHandCount: 5,
  });
  const threatState = {
    ...safeState,
    players: safeState.players.map((player) =>
      player.id === "leader" ? { ...player, handCount: 1 } : player,
    ),
  };

  assert.equal(chooseBotPlay(safeState, "hard").action.type, "pass");
  assert.equal(chooseBotPlay(threatState, "hard").action.type, "play");
});

test("easy mode preserves the former deterministic joker-first lead", () => {
  const state = observation({
    hand: [
      card("twelve-a", 12),
      card("twelve-b", 12),
      card("joker", 13),
    ],
  });

  assert.deepEqual(chooseBotCardIds(state, "easy"), ["joker"]);
  assert.notDeepEqual(chooseBotCardIds(state, "normal"), ["joker"]);
});

test("forced peon tribute excludes jokers and takes the lowest ranks", () => {
  assert.deepEqual(
    selectForcedBotTribute(
      [card("twelve", 12), card("one", 1), card("joker", 13), card("two", 2)],
      2,
    ),
    ["one", "two"],
  );
});

test("normal tax return preserves a weak pair and returns isolated weak cards", () => {
  const hand = [
    card("twelve-a", 12),
    card("twelve-b", 12),
    card("eleven", 11),
    card("ten", 10),
    card("joker", 13),
  ];

  assert.deepEqual(
    chooseBotTaxReturn(hand, 2, "easy").cardIds,
    ["twelve-a", "twelve-b"],
  );
  assert.deepEqual(
    chooseBotTaxReturn(hand, 2, "normal").cardIds.sort(),
    ["eleven", "ten"],
  );
});

test("revolution policy weighs public role instead of always declaring", () => {
  const hand = [card("joker-a", 13), card("joker-b", 13), card("ten", 10)];

  assert.equal(
    chooseBotRevolution(
      { hand, role: "great-dalmuti", playerCount: 5 },
      "easy",
    ).declare,
    true,
  );
  assert.equal(
    chooseBotRevolution(
      { hand, role: "great-dalmuti", playerCount: 5 },
      "normal",
    ).declare,
    false,
  );
  assert.equal(
    chooseBotRevolution(
      { hand, role: "lesser-peon", playerCount: 5 },
      "normal",
    ).declare,
    true,
  );

  const great = chooseBotRevolution(
    { hand, role: "great-peon", playerCount: 5 },
    "hard",
  );
  assert.equal(great.declare, true);
  assert.equal(great.kind, "great-revolution");
});

test("facedown rank selection remains fair and uses the injected random source", () => {
  assert.equal(chooseFacedownRankSlot([2, 7, 9], () => 1), 7);
  assert.equal(chooseFacedownRankSlot([], () => 0), null);
  assert.throws(
    () => chooseFacedownRankSlot([2, 7, 9], () => 3),
    /invalid value/,
  );
});

test("rejects observations that omit the actor from public turn order", () => {
  assert.throws(
    () =>
      chooseBotPlay(
        {
          actorId: "bot",
          hand: [card("one", 1)],
          table: null,
          players: [{ id: "someone-else", handCount: 1 }],
        },
        "normal",
      ),
    /acting bot/,
  );
});
