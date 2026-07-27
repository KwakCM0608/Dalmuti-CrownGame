import assert from "node:assert/strict";
import test from "node:test";

const { resolveQuickDalmutiAutoPass } = await import(
  new URL("../lib/quick-dalmuti.ts", import.meta.url)
);

const players = [
  { id: "p1" },
  { id: "p2" },
  { id: "p3" },
  { id: "p4" },
  { id: "p5" },
];

test("quick Dalmuti auto-passes every other active player and returns the lead", () => {
  const result = resolveQuickDalmutiAutoPass(
    players,
    {
      p1: [{ id: "still-has-a-card" }],
      p2: [{ id: "two" }],
      p3: [],
      p4: [{ id: "four" }],
      p5: [{ id: "five" }],
    },
    "p1",
    0,
  );

  assert.deepEqual(result.autoPassedPlayerIds, ["p2", "p4", "p5"]);
  assert.equal(result.nextPlayerIndex, 0);
});

test("quick Dalmuti gives the next active player the lead when its actor finishes", () => {
  const result = resolveQuickDalmutiAutoPass(
    players,
    {
      p1: [],
      p2: [],
      p3: [{ id: "three" }],
      p4: [{ id: "four" }],
      p5: [],
    },
    "p1",
    0,
  );

  assert.deepEqual(result.autoPassedPlayerIds, ["p3", "p4"]);
  assert.equal(result.nextPlayerIndex, 2);
});

test("quick Dalmuti lead selection wraps around the rank order", () => {
  const result = resolveQuickDalmutiAutoPass(
    players,
    {
      p1: [{ id: "one" }],
      p2: [],
      p3: [],
      p4: [],
      p5: [],
    },
    "p4",
    3,
  );

  assert.deepEqual(result.autoPassedPlayerIds, ["p1"]);
  assert.equal(result.nextPlayerIndex, 0);
});
