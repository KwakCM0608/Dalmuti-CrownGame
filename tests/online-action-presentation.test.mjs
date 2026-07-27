import assert from "node:assert/strict";
import test from "node:test";

const {
  REMOTE_ACTION_REPLAY_WINDOW_MS,
  collectRemoteActionPresentations,
  isOnlineTableActionEvent,
} = await import(
  new URL("../lib/online-action-presentation.ts", import.meta.url)
);

function action({
  id,
  seq,
  type,
  startsAt = 9_000,
  actorPlayerId = "opponent",
  data = {},
}) {
  return {
    id,
    seq,
    type,
    startsAt,
    durationMs: type === "PLAYER_PASSED" ? 1_380 : 2_080,
    actorPlayerId,
    data,
  };
}

test("collects unseen opponent card and PASS actions in sequence order", () => {
  const result = collectRemoteActionPresentations(
    [
      action({
        id: "own-card",
        seq: 1,
        type: "CARDS_PLAYED",
        actorPlayerId: "viewer",
      }),
      action({ id: "remote-pass", seq: 3, type: "PLAYER_PASSED" }),
      action({ id: "remote-card", seq: 2, type: "CARDS_PLAYED" }),
    ],
    "viewer",
    10_000,
    new Set(),
  );

  assert.deepEqual(
    result.events.map((event) => event.id),
    ["remote-card", "remote-pass"],
  );
  assert.deepEqual(result.seenIds, ["remote-card", "remote-pass"]);
});

test("collapses one Dalmuti server batch into its spectacle animation", () => {
  const result = collectRemoteActionPresentations(
    [
      action({ id: "cards", seq: 1, type: "CARDS_PLAYED" }),
      action({ id: "dalmuti", seq: 2, type: "DALMUTI_EFFECT" }),
      action({
        id: "auto-pass",
        seq: 3,
        type: "PLAYER_PASSED",
        actorPlayerId: "another-opponent",
        data: { reason: "dalmuti", automatic: true },
      }),
    ],
    "viewer",
    10_000,
    new Set(),
  );

  assert.deepEqual(
    result.events.map((event) => event.id),
    ["dalmuti"],
  );
  assert.deepEqual(result.seenIds, ["cards", "dalmuti", "auto-pass"]);
});

test("marks stale actions seen without replaying a reconnect backlog", () => {
  const stale = action({
    id: "stale-pass",
    seq: 1,
    type: "PLAYER_PASSED",
    startsAt: 10_000 - REMOTE_ACTION_REPLAY_WINDOW_MS - 1,
  });
  const result = collectRemoteActionPresentations(
    [stale],
    "viewer",
    10_000,
    new Set(),
  );

  assert.deepEqual(result.events, []);
  assert.deepEqual(result.seenIds, ["stale-pass"]);
});

test("never re-enqueues an event already observed through another response", () => {
  const duplicate = action({
    id: "same-response-event",
    seq: 8,
    type: "CARDS_PLAYED",
  });
  const result = collectRemoteActionPresentations(
    [duplicate],
    "viewer",
    10_000,
    new Set([duplicate.id]),
  );

  assert.deepEqual(result.events, []);
  assert.deepEqual(result.seenIds, []);
});

test("recognizes only public table-motion event types", () => {
  assert.equal(
    isOnlineTableActionEvent(action({ id: "pass", seq: 1, type: "PLAYER_PASSED" })),
    true,
  );
  assert.equal(
    isOnlineTableActionEvent(action({ id: "turn", seq: 2, type: "TURN_STARTED" })),
    false,
  );
});
