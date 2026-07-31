import assert from "node:assert/strict";
import test from "node:test";

const {
  ACTION_SPACE_SIZE,
  decodeSemanticAction,
  encodeSemanticAction,
  legalSemanticActionIndices,
  resolveSemanticAction,
  semanticActionIndexFromBotAction,
} = await import(
  new URL("../training/action-space.ts", import.meta.url)
);
const { enumerateLegalBotPlays } = await import(
  new URL("../lib/bot-strategy.ts", import.meta.url)
);
const {
  OBSERVATION_FEATURE_COUNT,
  encodeTrainingObservation,
} = await import(
  new URL("../training/observation.ts", import.meta.url)
);
const { simulateMatch } = await import(
  new URL("../training/simulator.ts", import.meta.url)
);
const {
  parseMlpPolicyModel,
  selectMaskedMlpAction,
} = await import(
  new URL("../training/model-policy.ts", import.meta.url)
);

function card(id, rank) {
  return { id, rank };
}

function playObservation(hand, table = null) {
  return {
    actorId: "actor",
    hand,
    table,
    players: [
      { id: "leader", handCount: 5 },
      { id: "actor", handCount: hand.length },
      { id: "next", handCount: 4 },
      { id: "last", handCount: 6 },
    ],
    passedPlayerIds: ["next"],
    publicPlayedCards: [{ rank: 12, count: 2 }],
  };
}

test("V1 semantic action indices round-trip across all 506 choices", () => {
  assert.equal(ACTION_SPACE_SIZE, 506);
  for (let actionIndex = 0; actionIndex < ACTION_SPACE_SIZE; actionIndex += 1) {
    assert.equal(
      encodeSemanticAction(decodeSemanticAction(actionIndex)),
      actionIndex,
    );
  }
});

test("semantic mask merges physical copies and excludes a two-joker-only pair", () => {
  const observation = playObservation([
    card("seven-a", 7),
    card("seven-b", 7),
    card("joker-a", 13),
    card("joker-b", 13),
  ]);
  const legal = legalSemanticActionIndices(observation);

  assert.ok(legal.includes(1));
  assert.ok(
    legal.includes(
      encodeSemanticAction({
        type: "play",
        rank: 7,
        count: 4,
        jokerCount: 2,
      }),
    ),
  );
  assert.equal(
    legal.includes(
      encodeSemanticAction({
        type: "play",
        rank: 7,
        count: 2,
        jokerCount: 2,
      }),
    ),
    false,
  );
});

test("semantic response must match count and beat the table rank", () => {
  const observation = playObservation(
    [
      card("six-a", 6),
      card("six-b", 6),
      card("eight-a", 8),
      card("joker", 13),
    ],
    { rank: 8, count: 2, playerId: "leader" },
  );
  const legal = legalSemanticActionIndices(observation);
  const sixPair = encodeSemanticAction({
    type: "play",
    rank: 6,
    count: 2,
    jokerCount: 0,
  });

  assert.ok(legal.includes(0));
  assert.ok(legal.includes(sixPair));
  assert.equal(
    legal.some((actionIndex) => {
      const action = decodeSemanticAction(actionIndex);
      return action.type === "play" && action.rank >= 8;
    }),
    false,
  );
  assert.deepEqual(
    resolveSemanticAction(observation, sixPair).cardIds,
    ["six-a", "six-b"],
  );
});

test("semantic masks exactly match the production bot legal-play rules", () => {
  const observations = [
    playObservation([
      card("one", 1),
      card("six-a", 6),
      card("six-b", 6),
      card("twelve", 12),
      card("joker-a", 13),
      card("joker-b", 13),
    ]),
    playObservation(
      [
        card("two-a", 2),
        card("two-b", 2),
        card("seven-a", 7),
        card("seven-b", 7),
        card("joker", 13),
      ],
      { rank: 8, count: 3, playerId: "leader" },
    ),
    playObservation(
      [card("ten", 10), card("twelve", 12)],
      { rank: 9, count: 2, playerId: "leader" },
    ),
  ];

  for (const observation of observations) {
    const expected = new Set(
      enumerateLegalBotPlays(observation).map(
        semanticActionIndexFromBotAction,
      ),
    );
    if (observation.table) expected.add(0);
    assert.deepEqual(
      legalSemanticActionIndices(observation),
      [...expected].sort((left, right) => left - right),
    );
  }
});

test("observation V2 has fixed size and only accepts the actor hand", () => {
  const observation = playObservation(
    [card("one", 1), card("joker", 13)],
    { rank: 10, count: 1, playerId: "leader" },
  );
  const encoded = encodeTrainingObservation({
    observation,
    round: 2,
    rolesByPlayerId: {
      leader: "great-dalmuti",
      actor: "lesser-dalmuti",
      next: "lesser-peon",
      last: "great-peon",
    },
    scoresByPlayerId: { leader: 4, actor: 3, next: 1, last: 0 },
    revolution: null,
  });

  assert.equal(OBSERVATION_FEATURE_COUNT, 172);
  assert.equal(encoded.length, OBSERVATION_FEATURE_COUNT);
  assert.equal(encoded.every(Number.isFinite), true);
  assert.equal("hands" in observation.players[0], false);
});

test("seeded multi-act simulation is reproducible and emits only legal actions", () => {
  const config = {
    playerCount: 4,
    acts: 2,
    seed: 20260731,
    difficulties: ["easy", "normal", "hard", "hard"],
  };
  const first = simulateMatch(config);
  const second = simulateMatch(config);

  assert.deepEqual(first, second);
  assert.equal(first.acts.length, 2);
  assert.ok(first.steps.length > 0);
  for (const act of first.acts) {
    assert.equal(new Set(act.finishOrder).size, 4);
    assert.equal(
      Object.values(act.chipAwards).reduce(
        (total, award) => total + award,
        0,
      ),
      8,
    );
  }
  for (const step of first.steps) {
    assert.equal(step.observation.length, OBSERVATION_FEATURE_COUNT);
    assert.ok(step.legalActionIndices.includes(step.actionIndex));
    assert.ok([-1, -0.5, 0, 0.5, 1].includes(step.reward));
    if (!step.actorTerminal) assert.equal(step.reward, 0);
  }
  for (const actorId of Object.keys(first.finalScores)) {
    const actorSteps = first.steps.filter(
      (step) => step.round === 2 && step.actorId === actorId,
    );
    assert.equal(
      actorSteps.filter((step) => step.actorTerminal).length,
      1,
    );
  }
});

test("headless simulation completes for every supported quick-match player count", () => {
  for (let playerCount = 4; playerCount <= 10; playerCount += 1) {
    const match = simulateMatch({
      playerCount,
      acts: 1,
      seed: 8_000 + playerCount,
      difficulties: ["hard"],
    });
    assert.equal(match.acts[0].finishOrder.length, playerCount);
    assert.equal(
      new Set(match.acts[0].finishOrder).size,
      playerCount,
    );
    assert.equal(
      match.steps.every((step) =>
        step.legalActionIndices.includes(step.actionIndex),
      ),
      true,
    );
  }
});

test("MLP policy inference applies ReLU and never selects a masked action", () => {
  const model = parseMlpPolicyModel({
    format: "dalmuti-mlp-policy",
    version: 1,
    observationFeatures: 2,
    actionCount: 3,
    hiddenSizes: [2],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    layers: [
      {
        inFeatures: 2,
        outFeatures: 2,
        weight: [1, 0, 0, 1],
        bias: [0, 0],
      },
      {
        inFeatures: 2,
        outFeatures: 3,
        weight: [1, 0, 0, 1, 2, 2],
        bias: [0, 0, 0],
      },
    ],
  });

  assert.equal(selectMaskedMlpAction(model, [2, 1], [0, 1, 2]), 2);
  assert.equal(selectMaskedMlpAction(model, [2, 1], [0, 1]), 0);
});

test("simulation can mix a custom policy with baseline players", () => {
  let candidateDecisions = 0;
  const candidatePolicy = ({ legalActionIndices }) => {
    candidateDecisions += 1;
    return legalActionIndices[0];
  };
  const match = simulateMatch({
    playerCount: 4,
    acts: 1,
    seed: 9191,
    difficulties: ["normal"],
    policyByPlayerId: {
      "player-1": candidatePolicy,
    },
  });

  assert.ok(candidateDecisions > 0);
  assert.equal(match.acts[0].finishOrder.length, 4);
  assert.equal(
    match.steps
      .filter((step) => step.actorId === "player-1")
      .every((step) => step.behaviorPolicy === "custom"),
    true,
  );
  assert.equal(
    match.steps
      .filter((step) => step.actorId !== "player-1")
      .every((step) => step.behaviorPolicy === "normal"),
    true,
  );
});

test("DAgger supervision labels candidate states without changing behavior", () => {
  const firstLegalPolicy = ({ legalActionIndices }) =>
    legalActionIndices[0];
  const lastLegalSupervisor = ({ legalActionIndices }) =>
    legalActionIndices.at(-1);
  const match = simulateMatch({
    playerCount: 4,
    acts: 1,
    seed: 9292,
    difficulties: ["normal"],
    policy: firstLegalPolicy,
    supervisionPolicy: lastLegalSupervisor,
  });

  assert.equal(
    match.steps.every(
      (step) =>
        step.legalActionIndices.includes(step.actionIndex) &&
        step.supervisedActionIndex !== null &&
        step.legalActionIndices.includes(step.supervisedActionIndex),
    ),
    true,
  );
  assert.equal(
    match.steps.some(
      (step) => step.actionIndex !== step.supervisedActionIndex,
    ),
    true,
  );
});
