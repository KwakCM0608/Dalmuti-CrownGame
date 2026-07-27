import assert from "node:assert/strict";
import test from "node:test";

const {
  TEMP_GREAT_REVOLUTION_TEST_MODE,
  forceClaimedPlayerToLastRank,
  forceTwoJokersIntoHand,
} = await import(
  new URL("../lib/temporary-test-mode.ts", import.meta.url)
);

test("temporary great-revolution mode is disabled for normal play", () => {
  assert.equal(TEMP_GREAT_REVOLUTION_TEST_MODE, false);
});

test("temporary great-revolution mode preserves rank uniqueness", () => {
  const original = [
    { rank: 1, claimedByPlayerId: "host" },
    { rank: 2, claimedByPlayerId: "p2" },
    { rank: 3, claimedByPlayerId: "p3" },
    { rank: 4, claimedByPlayerId: "p4" },
    { rank: 5, claimedByPlayerId: "p5" },
  ];
  const result = forceClaimedPlayerToLastRank(original, "host");

  assert.deepEqual(
    result.map((card) => card.rank).sort((left, right) => left - right),
    [1, 2, 3, 4, 5],
  );
  assert.equal(
    result.find((card) => card.claimedByPlayerId === "host").rank,
    TEMP_GREAT_REVOLUTION_TEST_MODE ? 5 : 1,
  );
  assert.equal(original[0].rank, 1);
});

test("temporary great-revolution mode swaps both jokers into the target hand", () => {
  const original = {
    host: [
      { id: "host-12", rank: 12 },
      { id: "host-11", rank: 11 },
    ],
    p2: [
      { id: "joker-1", rank: 13 },
      { id: "p2-9", rank: 9 },
    ],
    p3: [
      { id: "joker-2", rank: 13 },
      { id: "p3-8", rank: 8 },
    ],
  };
  const result = forceTwoJokersIntoHand(
    original,
    ["host", "p2", "p3"],
    "host",
  );

  assert.deepEqual(
    Object.values(result).map((hand) => hand.length),
    [2, 2, 2],
  );
  assert.deepEqual(
    Object.values(result)
      .flat()
      .map((card) => card.id)
      .sort(),
    Object.values(original)
      .flat()
      .map((card) => card.id)
      .sort(),
  );
  assert.equal(
    result.host.filter((card) => card.rank === 13).length,
    TEMP_GREAT_REVOLUTION_TEST_MODE ? 2 : 0,
  );
  assert.equal(
    original.host.filter((card) => card.rank === 13).length,
    0,
  );
});
