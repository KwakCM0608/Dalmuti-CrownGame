import assert from "node:assert/strict";
import test from "node:test";

const observationModule = await import(
  new URL("../training/non-card-observation.ts", import.meta.url)
);
const actionModule = await import(
  new URL("../training/non-card-action-space.ts", import.meta.url)
);
const searchModule = await import(
  new URL("../training/non-card-search-targets.ts", import.meta.url)
);

const {
  REVOLUTION_OBSERVATION_FEATURE_COUNT,
  TAX_RETURN_OBSERVATION_FEATURE_COUNT,
  encodeRevolutionObservation,
  encodeTaxReturnObservation,
} = observationModule;

const {
  REVOLUTION_DECLARE_ACTION_INDEX,
  TAX_RETURN_ACTION_CATALOGUE,
  TAX_RETURN_ACTION_COUNT,
  decodeTaxReturnAction,
  encodeTaxReturnAction,
  encodeTaxReturnActionFeatures,
  enumerateLegalRevolutionActions,
  legalTaxReturnActionIndices,
  resolveTaxReturnAction,
} = actionModule;

const {
  buildPairedCounterfactualTargets,
  buildRevolutionCounterfactualTargets,
  selectExactRevolutionAction,
  selectExactTaxReturnAction,
} = searchModule;

function roleAt(index, playerCount) {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === playerCount - 2) return "lesser-peon";
  if (index === playerCount - 1) return "great-peon";
  return "merchant";
}

function players(playerCount, actorIndex, actorHandCount, prefix = "p") {
  return Array.from({ length: playerCount }, (_, index) => ({
    id: `${prefix}${index}`,
    role: roleAt(index, playerCount),
    handCount: index === actorIndex ? actorHandCount : 8 + (index % 3),
    score: index * 2,
  }));
}

function taxObservation(prefix = "p") {
  const hand = [
    { id: `${prefix}-r1`, rank: 1 },
    { id: `${prefix}-r2b`, rank: 2 },
    { id: `${prefix}-r2a`, rank: 2 },
    { id: `${prefix}-r5b`, rank: 5 },
    { id: `${prefix}-r5a`, rank: 5 },
    { id: `${prefix}-r9`, rank: 9 },
    { id: `${prefix}-joker-b`, rank: 13 },
    { id: `${prefix}-joker-a`, rank: 13 },
  ];
  return {
    actorId: `${prefix}0`,
    hand,
    players: players(10, 0, hand.length, prefix),
    round: 3,
    returnCount: 2,
  };
}

function revolutionObservation(prefix = "r") {
  const hand = [
    ...Array.from({ length: 12 }, (_, index) => ({
      id: `${prefix}-r12-${index}`,
      rank: 12,
    })),
    ...Array.from({ length: 6 }, (_, index) => ({
      id: `${prefix}-r6-${index}`,
      rank: 6,
    })),
    { id: `${prefix}-joker-a`, rank: 13 },
    { id: `${prefix}-joker-b`, rank: 13 },
  ];
  return {
    actorId: `${prefix}3`,
    hand,
    players: players(4, 3, hand.length, prefix),
    round: 1,
  };
}

test("tax catalogue is stable, exact, and round-trips semantic rank multisets", () => {
  assert.equal(TAX_RETURN_ACTION_COUNT, 103);
  assert.equal(TAX_RETURN_ACTION_CATALOGUE.length, 103);
  const keys = new Set();
  for (let actionIndex = 0; actionIndex < TAX_RETURN_ACTION_COUNT; actionIndex += 1) {
    const action = decodeTaxReturnAction(actionIndex);
    assert.equal(encodeTaxReturnAction(action.ranks), actionIndex);
    assert.deepEqual(action.ranks, [...action.ranks].sort((a, b) => a - b));
    keys.add(action.ranks.join("+"));
  }
  assert.equal(keys.size, 103);
  assert.throws(() => encodeTaxReturnAction([1, 1]), /structurally impossible/);
  assert.equal(
    encodeTaxReturnAction([13, 2]),
    encodeTaxReturnAction([2, 13]),
  );
});

test("tax legal mask uses only own ranks and resolves duplicate IDs deterministically", () => {
  const observation = taxObservation();
  const legalIndices = legalTaxReturnActionIndices(observation);
  assert.equal(legalIndices.length, 13);
  assert.equal(
    legalIndices.every(
      (actionIndex) => decodeTaxReturnAction(actionIndex).ranks.length === 2,
    ),
    true,
  );

  const repeatedFive = encodeTaxReturnAction([5, 5]);
  assert.equal(legalIndices.includes(repeatedFive), true);
  assert.deepEqual(resolveTaxReturnAction(observation, repeatedFive), [
    "p-r5a",
    "p-r5b",
  ]);
  assert.equal(
    legalIndices.includes(encodeTaxReturnAction([9, 9])),
    false,
  );

  const actionFeatures = encodeTaxReturnActionFeatures(repeatedFive);
  assert.equal(actionFeatures.length, 15);
  assert.deepEqual(actionFeatures.slice(0, 2), [0, 1]);
  assert.equal(actionFeatures[6], 1);
});

test("both encoders cover every player count from 4 through 10", () => {
  for (let playerCount = 4; playerCount <= 10; playerCount += 1) {
    const tax = taxObservation(`t${playerCount}-`);
    tax.actorId = `t${playerCount}-0`;
    tax.players = players(
      playerCount,
      0,
      tax.hand.length,
      `t${playerCount}-`,
    );
    const revolution = revolutionObservation(`r${playerCount}-`);
    revolution.actorId = `r${playerCount}-${playerCount - 1}`;
    revolution.players = players(
      playerCount,
      playerCount - 1,
      revolution.hand.length,
      `r${playerCount}-`,
    );

    const taxFeatures = encodeTaxReturnObservation(tax);
    const revolutionFeatures = encodeRevolutionObservation(revolution);
    assert.equal(taxFeatures.length, TAX_RETURN_OBSERVATION_FEATURE_COUNT);
    assert.equal(
      revolutionFeatures.length,
      REVOLUTION_OBSERVATION_FEATURE_COUNT,
    );
    assert.equal(taxFeatures.every(Number.isFinite), true);
    assert.equal(revolutionFeatures.every(Number.isFinite), true);
    assert.equal(
      revolutionFeatures.at(-1),
      0,
      "opening act has no tax to cancel",
    );
  }
});

test("encoders are ID-invariant and reject opponent card fields", () => {
  assert.deepEqual(
    encodeTaxReturnObservation(taxObservation("p")),
    encodeTaxReturnObservation(taxObservation("renamed")),
  );

  const leaked = taxObservation();
  leaked.players[4].hand = [{ id: "secret", rank: 1 }];
  assert.throws(
    () => encodeTaxReturnObservation(leaked),
    /must not contain hand/,
  );
});

test("observation validators enforce role, tax count, and two-joker rules", () => {
  const wrongTaxCount = { ...taxObservation(), returnCount: 1 };
  assert.throws(
    () => encodeTaxReturnObservation(wrongTaxCount),
    /must return exactly 2/,
  );

  const wrongRoleOrder = taxObservation();
  wrongRoleOrder.players[2] = {
    ...wrongRoleOrder.players[2],
    role: "great-peon",
  };
  assert.throws(
    () => encodeTaxReturnObservation(wrongRoleOrder),
    /rank seat 2 must be merchant/,
  );

  const missingJoker = revolutionObservation();
  missingJoker.hand[19] = { id: "replacement", rank: 7 };
  assert.throws(
    () => encodeRevolutionObservation(missingJoker),
    /exactly two jokers/,
  );
});

test("revolution catalogue exposes normal versus great declaration consequences", () => {
  const great = enumerateLegalRevolutionActions(revolutionObservation());
  assert.equal(great.length, 2);
  assert.equal(great[0].declarationKind, null);
  assert.equal(great[1].declarationKind, "great-revolution");
  assert.deepEqual(great[1].actionFeatures, [0, 0, 1]);

  const normalObservation = {
    ...revolutionObservation("n"),
    actorId: "n1",
    players: players(4, 1, 20, "n"),
  };
  const normal = enumerateLegalRevolutionActions(normalObservation);
  assert.equal(normal[1].declarationKind, "revolution");
  assert.deepEqual(normal[1].actionFeatures, [0, 1, 0]);
});

test("exact action-conditioned selection scores all legal actions with no raw state", () => {
  const observation = taxObservation();
  const targetIndex = encodeTaxReturnAction([5, 13]);
  let scoreCalls = 0;
  const selected = selectExactTaxReturnAction(observation, (input) => {
    scoreCalls += 1;
    assert.deepEqual(
      Object.keys(input).sort(),
      [
        "actionFeatures",
        "actionIndex",
        "decision",
        "observationFeatures",
        "observationSchemaVersion",
      ].sort(),
    );
    return input.actionIndex === targetIndex ? 10 : 0;
  });
  assert.equal(scoreCalls, legalTaxReturnActionIndices(observation).length);
  assert.equal(selected.candidate.actionIndex, targetIndex);
  assert.deepEqual(selected.candidate.cardIds, ["p-r5a", "p-joker-a"]);

  const revolution = selectExactRevolutionAction(
    revolutionObservation(),
    ({ actionIndex }) => actionIndex,
  );
  assert.equal(
    revolution.candidate.actionIndex,
    REVOLUTION_DECLARE_ACTION_INDEX,
  );
  assert.throws(
    () => selectExactRevolutionAction(revolutionObservation(), () => NaN),
    /non-finite score/,
  );
});

test("paired targets require complete same-world action coverage", () => {
  const targets = buildPairedCounterfactualTargets(
    [false, true, true],
    [
      {
        sampleId: "world-a",
        outcomes: [
          { actionIndex: 1, utility: 2 },
          { actionIndex: 2, utility: 1 },
        ],
      },
      {
        sampleId: "world-b",
        outcomes: [
          { actionIndex: 1, utility: 4 },
          { actionIndex: 2, utility: 5 },
        ],
      },
    ],
  );
  assert.equal(targets.sampleCount, 2);
  assert.equal(targets.bestActionIndex, 1, "ties prefer lower stable index");
  assert.deepEqual(
    targets.actions.map((action) => action.meanUtility),
    [3, 3],
  );
  assert.deepEqual(
    targets.actions.map((action) => action.centeredUtility),
    [0, 0],
  );
  assert.ok(Math.abs(targets.actions[0].sampleStandardDeviation - Math.SQRT2) < 1e-12);
  assert.ok(
    Math.abs(
      targets.actions.reduce(
        (total, action) => total + action.policyProbability,
        0,
      ) - 1,
    ) < 1e-12,
  );

  assert.throws(
    () =>
      buildPairedCounterfactualTargets([true, true], [
        {
          sampleId: "incomplete",
          outcomes: [{ actionIndex: 0, utility: 1 }],
        },
      ]),
    /cover every legal action/,
  );
  assert.throws(
    () =>
      buildPairedCounterfactualTargets([true, true], [
        {
          sampleId: "duplicate",
          outcomes: [
            { actionIndex: 0, utility: 1 },
            { actionIndex: 0, utility: 2 },
          ],
        },
      ]),
    /repeats action/,
  );
});

test("revolution target wrapper uses the exact two-action legal mask", () => {
  const targets = buildRevolutionCounterfactualTargets(
    revolutionObservation(),
    [
      {
        sampleId: "seed-1",
        outcomes: [
          { actionIndex: 0, utility: 1 },
          { actionIndex: 1, utility: 4 },
        ],
      },
    ],
    { policyTemperature: 0.5 },
  );
  assert.equal(targets.bestActionIndex, 1);
  assert.ok(targets.actions[1].policyProbability > 0.99);
});
