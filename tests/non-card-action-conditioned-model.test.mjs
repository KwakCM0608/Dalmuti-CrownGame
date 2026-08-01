import assert from "node:assert/strict";
import test from "node:test";

const actionModule = await import(
  new URL("../training/non-card-action-space.ts", import.meta.url)
);
const modelModule = await import(
  new URL(
    "../training/non-card-action-conditioned-model.ts",
    import.meta.url,
  )
);
const observationModule = await import(
  new URL("../training/non-card-observation.ts", import.meta.url)
);
const simulatorModule = await import(
  new URL("../training/simulator.ts", import.meta.url)
);

const {
  REVOLUTION_ACTION_COUNT,
  REVOLUTION_ACTION_FEATURE_COUNT,
  REVOLUTION_ACTION_FEATURE_LAYOUT,
  TAX_RETURN_ACTION_COUNT,
  TAX_RETURN_ACTION_FEATURE_COUNT,
  TAX_RETURN_ACTION_FEATURE_LAYOUT,
  encodeTaxReturnAction,
  legalTaxReturnActionIndices,
} = actionModule;
const {
  createRevolutionModelTrainingPolicy,
  createTaxReturnModelTrainingPolicy,
  evaluateRevolutionActionConditionedActorCritic,
  evaluateTaxReturnActionConditionedActorCritic,
  parseRevolutionActionConditionedActorCriticModel,
  parseTaxReturnActionConditionedActorCriticModel,
  selectBaselineGatedNonCardAction,
  selectRevolutionActionConditionedAction,
  selectTaxReturnActionConditionedAction,
} = modelModule;
const {
  REVOLUTION_OBSERVATION_FEATURE_COUNT,
  TAX_RETURN_OBSERVATION_FEATURE_COUNT,
  encodeTaxReturnObservation,
} = observationModule;
const { simulateMatch } = simulatorModule;

function zeroes(length) {
  return Array.from({ length }, () => 0);
}

function layer(inFeatures, outFeatures, weight, bias) {
  return { inFeatures, outFeatures, weight, bias };
}

function commonPayload({
  format,
  decisionKind,
  observationFeatures,
  actionCount,
  actionFeatures,
  actionFeatureLayout,
  actionWeights,
  scorerWeights,
}) {
  return {
    format,
    version: 1,
    decisionKind,
    observationSchemaVersion: 1,
    observationFeatures,
    actionCatalogueVersion: 1,
    actionCount,
    actionFeatures,
    actionFeatureLayout,
    actorObservationHiddenSizes: [1],
    actorActionHiddenSizes: [1],
    actorScorerHiddenSizes: [],
    valueHiddenSizes: [1],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    actorObservationLayers: [
      layer(observationFeatures, 1, zeroes(observationFeatures), [1]),
    ],
    actorActionLayers: [
      layer(actionFeatures, 1, actionWeights, [0]),
    ],
    actorScorerLayers: [layer(2, 1, scorerWeights, [0.25])],
    valueLayers: [
      layer(observationFeatures, 1, zeroes(observationFeatures), [2]),
      layer(1, 1, [3], [-0.5]),
    ],
  };
}

function taxPayload() {
  const actionWeights = zeroes(TAX_RETURN_ACTION_FEATURE_COUNT);
  actionWeights[6] = 4;
  return commonPayload({
    format: "dalmuti-tax-return-action-conditioned-actor-critic",
    decisionKind: "tax-return",
    observationFeatures: TAX_RETURN_OBSERVATION_FEATURE_COUNT,
    actionCount: TAX_RETURN_ACTION_COUNT,
    actionFeatures: TAX_RETURN_ACTION_FEATURE_COUNT,
    actionFeatureLayout: TAX_RETURN_ACTION_FEATURE_LAYOUT,
    actionWeights,
    scorerWeights: [2, 3],
  });
}

function revolutionPayload() {
  return {
    ...commonPayload({
      format: "dalmuti-revolution-action-conditioned-actor-critic",
      decisionKind: "revolution",
      observationFeatures: REVOLUTION_OBSERVATION_FEATURE_COUNT,
      actionCount: REVOLUTION_ACTION_COUNT,
      actionFeatures: REVOLUTION_ACTION_FEATURE_COUNT,
      actionFeatureLayout: REVOLUTION_ACTION_FEATURE_LAYOUT,
      actionWeights: [1, 2, 0.25],
      scorerWeights: [0, 1],
    }),
    greatPeonRoleFeatureIndex: 7,
  };
}

function roleAt(index, playerCount) {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === playerCount - 2) return "lesser-peon";
  if (index === playerCount - 1) return "great-peon";
  return "merchant";
}

function rankedPlayers(actorIndex, actorHandCount, prefix = "p") {
  return Array.from({ length: 4 }, (_, index) => ({
    id: `${prefix}${index}`,
    role: roleAt(index, 4),
    handCount: index === actorIndex ? actorHandCount : 6 + index,
    score: index * 2,
  }));
}

function taxObservation() {
  const hand = [
    { id: "tax-r2", rank: 2 },
    { id: "tax-r5", rank: 5 },
    { id: "tax-joker", rank: 13 },
  ];
  return {
    actorId: "tax1",
    hand,
    players: rankedPlayers(1, hand.length, "tax"),
    round: 2,
    returnCount: 1,
  };
}

function revolutionObservation(actorIndex, prefix) {
  const hand = [
    { id: `${prefix}-joker-a`, rank: 13 },
    { id: `${prefix}-joker-b`, rank: 13 },
  ];
  return {
    actorId: `${prefix}${actorIndex}`,
    hand,
    players: rankedPlayers(actorIndex, hand.length, prefix),
    round: 1,
  };
}

test("tax model parses, scores only hand-legal actions, and ties by lowest index", () => {
  const model = parseTaxReturnActionConditionedActorCriticModel(taxPayload());
  const observation = taxObservation();
  const output = evaluateTaxReturnActionConditionedActorCritic(
    model,
    observation,
  );

  assert.deepEqual(output.actionIndices, [1, 4, 12]);
  assert.deepEqual([...output.logits], [2.25, 8.25, 2.25]);
  assert.equal(output.value, 5.5);
  assert.equal(
    selectTaxReturnActionConditionedAction(model, observation),
    encodeTaxReturnAction([5]),
  );
  assert.equal(output.actionIndices.includes(encodeTaxReturnAction([1])), false);

  const tiedPayload = taxPayload();
  tiedPayload.actorActionLayers[0].weight.fill(0);
  const tiedModel = parseTaxReturnActionConditionedActorCriticModel(tiedPayload);
  assert.equal(
    selectTaxReturnActionConditionedAction(tiedModel, observation),
    1,
    "equal logits choose the lowest stable legal action index",
  );
});

test("conservative non-card gate preserves argmax at zero and falls back below threshold", () => {
  const output = {
    actionIndices: [1, 4, 12],
    logits: Float64Array.from([0, 0, -1]),
    value: 0,
  };
  assert.deepEqual(selectBaselineGatedNonCardAction(output, 4, 0), {
    actionIndex: 1,
    modelActionIndex: 1,
    baselineActionIndex: 4,
    modelActionLogit: 0,
    baselineActionLogit: 0,
    predictedAdvantage: 0,
    minimumAdvantage: 0,
    routing: "learnedAction",
  });
  assert.deepEqual(selectBaselineGatedNonCardAction(output, 4, 0.01), {
    actionIndex: 4,
    modelActionIndex: 1,
    baselineActionIndex: 4,
    modelActionLogit: 0,
    baselineActionLogit: 0,
    predictedAdvantage: 0,
    minimumAdvantage: 0.01,
    routing: "safetyFallback",
  });
  assert.equal(
    selectBaselineGatedNonCardAction(output, 1, 100).routing,
    "agreedWithBaseline",
  );
  assert.throws(
    () => selectBaselineGatedNonCardAction(output, 4, -0.01),
    /non-negative finite/,
  );
  assert.throws(
    () => selectBaselineGatedNonCardAction(output, 99, 0),
    /not legal/,
  );
});

test("tax training policy rejects context legality that does not match the hand", () => {
  const observation = taxObservation();
  const legalActionIndicesValue = legalTaxReturnActionIndices(observation);
  const policy = createTaxReturnModelTrainingPolicy(taxPayload(), "tax-model-v1");
  const context = {
    decision: "tax-return",
    episodeId: "context-test",
    round: observation.round,
    actorId: observation.actorId,
    actorSeat: 1,
    actorRole: "lesser-dalmuti",
    actorScore: 2,
    decisionKey: "context-key",
    observation,
    encodedObservation: encodeTaxReturnObservation(observation),
    legalActionIndices: legalActionIndicesValue,
    random: () => 0.5,
  };

  assert.deepEqual(policy(context), {
    actionIndex: encodeTaxReturnAction([5]),
    logProbability: 0,
    valueEstimate: 5.5,
    policyVersion: "tax-model-v1",
  });
  assert.throws(
    () =>
      policy({
        ...context,
        legalActionIndices: [...legalActionIndicesValue, encodeTaxReturnAction([1])],
      }),
    /exact hand-derived legality/,
  );
  assert.throws(
    () =>
      policy({
        ...context,
        encodedObservation: [
          context.encodedObservation[0] + 0.01,
          ...context.encodedObservation.slice(1),
        ],
      }),
    /encodedObservation does not match/,
  );
});

test("revolution action features condition declaration scoring on actor role", () => {
  const model = parseRevolutionActionConditionedActorCriticModel(
    revolutionPayload(),
  );
  const normal = revolutionObservation(0, "normal");
  const great = revolutionObservation(3, "great");
  const normalOutput = evaluateRevolutionActionConditionedActorCritic(
    model,
    normal,
  );
  const greatOutput = evaluateRevolutionActionConditionedActorCritic(
    model,
    great,
  );

  assert.deepEqual(normalOutput.actionIndices, [0, 1]);
  assert.deepEqual([...normalOutput.logits], [1.25, 2.25]);
  assert.deepEqual([...greatOutput.logits], [1.25, 0.5]);
  assert.equal(normalOutput.value, 5.5);
  assert.equal(greatOutput.value, 5.5);
  assert.equal(selectRevolutionActionConditionedAction(model, normal), 1);
  assert.equal(selectRevolutionActionConditionedAction(model, great), 0);
});

test("non-card parsers reject schema, catalogue, connectivity, and finite-value drift", () => {
  assert.throws(
    () =>
      parseTaxReturnActionConditionedActorCriticModel({
        ...taxPayload(),
        decisionKind: "revolution",
      }),
    /unsupported non-card actor-critic model format/,
  );
  assert.throws(
    () =>
      parseTaxReturnActionConditionedActorCriticModel({
        ...taxPayload(),
        observationFeatures: TAX_RETURN_OBSERVATION_FEATURE_COUNT - 1,
      }),
    /observation contract mismatch/,
  );
  assert.throws(
    () =>
      parseTaxReturnActionConditionedActorCriticModel({
        ...taxPayload(),
        actionFeatureLayout: TAX_RETURN_ACTION_FEATURE_LAYOUT.slice(1),
      }),
    /action catalogue contract mismatch/,
  );

  const disconnected = taxPayload();
  disconnected.actorScorerLayers[0] = {
    ...disconnected.actorScorerLayers[0],
    inFeatures: 3,
  };
  assert.throws(
    () => parseTaxReturnActionConditionedActorCriticModel(disconnected),
    /dimensions do not connect/,
  );

  const nonFinite = taxPayload();
  nonFinite.valueLayers[0].weight[0] = Number.NaN;
  assert.throws(
    () => parseTaxReturnActionConditionedActorCriticModel(nonFinite),
    /finite numbers/,
  );
  assert.throws(
    () =>
      parseRevolutionActionConditionedActorCriticModel({
        ...revolutionPayload(),
        greatPeonRoleFeatureIndex: 6,
      }),
    /role-conditioned action contract mismatch/,
  );
});

test("model policies plug into simulator non-card hooks and record value metadata", () => {
  const taxMatch = simulateMatch({
    playerCount: 6,
    acts: 3,
    seed: 20260801,
    difficulties: ["normal"],
    nonCard: {
      taxReturnPolicy: createTaxReturnModelTrainingPolicy(
        taxPayload(),
        "tax-simulator-v1",
      ),
    },
  });
  const taxSteps = taxMatch.nonCardSteps.filter(
    (step) => step.decision === "tax-return",
  );
  assert.ok(taxSteps.length > 0);
  assert.equal(
    taxSteps.every(
      (step) =>
        step.behaviorPolicy === "custom" &&
        step.behaviorPolicyVersion === "tax-simulator-v1" &&
        step.behaviorLogProbability === 0 &&
        step.behaviorValueEstimate === 5.5 &&
        step.legalActionIndices.includes(step.actionIndex),
    ),
    true,
  );

  const revolutionMatch = simulateMatch({
    playerCount: 4,
    acts: 1,
    seed: 6,
    difficulties: ["easy"],
    nonCard: {
      revolutionPolicy: createRevolutionModelTrainingPolicy(
        revolutionPayload(),
        "revolution-simulator-v1",
      ),
    },
  });
  const revolutionSteps = revolutionMatch.nonCardSteps.filter(
    (step) => step.decision === "revolution",
  );
  assert.equal(revolutionSteps.length, 1);
  assert.equal(revolutionSteps[0].actorRole, "great-peon");
  assert.equal(revolutionSteps[0].actionIndex, 0);
  assert.equal(revolutionSteps[0].behaviorValueEstimate, 5.5);
  assert.equal(
    revolutionSteps[0].behaviorPolicyVersion,
    "revolution-simulator-v1",
  );
});
