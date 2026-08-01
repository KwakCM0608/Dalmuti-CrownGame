import assert from "node:assert/strict";
import test from "node:test";

const {
  V3_ACTION_CATALOGUE,
  V3_ACTION_COUNT,
  V3_ACTION_FEATURE_COUNT,
  V3_ACTION_FEATURE_LAYOUT,
  V3_ACTION_FEATURES,
  decodeV3SemanticAction,
  encodeV3ActionFeatures,
  encodeV3SemanticAction,
} = await import(
  new URL("../training/v3-action-catalogue.ts", import.meta.url)
);
const {
  evaluateV3ActionConditionedActorCritic,
  parseV3ActionConditionedActorCriticModel,
  selectV3ActionConditionedAction,
} = await import(
  new URL("../training/v3-action-conditioned-model.ts", import.meta.url)
);

function zeroes(length) {
  return Array.from({ length }, () => 0);
}

function layer(inFeatures, outFeatures, weight, bias) {
  return { inFeatures, outFeatures, weight, bias };
}

function tinyModel() {
  const actionWeights = zeroes(V3_ACTION_FEATURE_COUNT * 2);
  actionWeights[0] = 1;
  actionWeights[V3_ACTION_FEATURE_COUNT + 18] = 1;
  return parseV3ActionConditionedActorCriticModel({
    format: "dalmuti-action-conditioned-actor-critic",
    version: 1,
    observationSchemaVersion: 3,
    observationFeatures: 3,
    actionCatalogueVersion: 1,
    actionCount: V3_ACTION_COUNT,
    actionFeatures: V3_ACTION_FEATURE_COUNT,
    actionFeatureLayout: V3_ACTION_FEATURE_LAYOUT,
    actorObservationHiddenSizes: [2],
    actorActionHiddenSizes: [2],
    actorScorerHiddenSizes: [],
    valueHiddenSizes: [2],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    actorObservationLayers: [
      layer(3, 2, [1, 0, 0, 0, 1, 0], [0, 0]),
    ],
    actorActionLayers: [
      layer(V3_ACTION_FEATURE_COUNT, 2, actionWeights, [0, 0]),
    ],
    actorScorerLayers: [layer(4, 1, [1, 1, 3, 4], [0.5])],
    valueLayers: [
      layer(3, 2, [1, 0, 0, 0, 1, 0], [0, 0]),
      layer(2, 1, [0.5, 2], [1]),
    ],
  });
}

test("V3 catalogue has exactly 236 structurally possible stable actions", () => {
  assert.equal(V3_ACTION_COUNT, 236);
  assert.equal(V3_ACTION_CATALOGUE.length, V3_ACTION_COUNT);
  const entriesByRank = Array.from({ length: 12 }, () => 0);
  for (let actionIndex = 0; actionIndex < V3_ACTION_COUNT; actionIndex += 1) {
    const action = decodeV3SemanticAction(actionIndex);
    assert.equal(encodeV3SemanticAction(action), actionIndex);
    if (action.type !== "play") continue;
    const naturalCount = action.count - action.jokerCount;
    assert.ok(naturalCount >= 1 && naturalCount <= action.rank);
    entriesByRank[action.rank - 1] += 1;
  }
  assert.deepEqual(
    entriesByRank,
    Array.from({ length: 12 }, (_, index) => 3 * (index + 1)),
  );
  assert.throws(
    () =>
      encodeV3SemanticAction({
        type: "play",
        rank: 1,
        count: 4,
        jokerCount: 2,
      }),
    /natural-card count/,
  );
});

test("V3 action features expose stable type, rank, count, and joker signals", () => {
  assert.equal(V3_ACTION_FEATURE_COUNT, 22);
  assert.equal(V3_ACTION_FEATURE_LAYOUT[18], "rank-strength");
  assert.equal(V3_ACTION_FEATURES.length, 236);
  assert.equal(
    V3_ACTION_FEATURES.every(
      (features) =>
        features.length === V3_ACTION_FEATURE_COUNT &&
        features.every(Number.isFinite),
    ),
    true,
  );

  const rankSix = encodeV3ActionFeatures({
    type: "play",
    rank: 6,
    count: 4,
    jokerCount: 1,
  });
  assert.equal(rankSix[2], 1);
  assert.equal(rankSix[8], 1);
  assert.equal(rankSix[16], 1);
  assert.equal(rankSix[18], 7 / 12);
  assert.equal(rankSix[19], 3 / 12);
  assert.equal(rankSix[20], 4 / 14);
  assert.equal(rankSix[21], 1 / 4);
});

test("V3 action-conditioned actor evaluates only legal catalogue entries", () => {
  const model = tinyModel();
  const rankOne = encodeV3SemanticAction({
    type: "play",
    rank: 1,
    count: 1,
    jokerCount: 0,
  });
  const legal = [rankOne, 0, 1];
  const output = evaluateV3ActionConditionedActorCritic(
    model,
    [2, 1, -100],
    legal,
  );

  assert.deepEqual(output.actionIndices, legal);
  assert.deepEqual([...output.logits], [7.5, 6.5, 3.5]);
  assert.equal(output.value, 4);
  assert.equal(
    selectV3ActionConditionedAction(model, [2, 1, -100], legal),
    rankOne,
  );
  assert.throws(
    () => evaluateV3ActionConditionedActorCritic(model, [2, 1, 0], [0, 0]),
    /duplicate/,
  );
});

test("V3 equal-logit tie breaking is independent of incoming legal order", () => {
  const base = tinyModel();
  const model = parseV3ActionConditionedActorCriticModel({
    ...base,
    actorScorerLayers: [
      layer(
        base.actorScorerLayers[0].inFeatures,
        1,
        zeroes(base.actorScorerLayers[0].inFeatures),
        [0],
      ),
    ],
  });
  assert.equal(
    selectV3ActionConditionedAction(model, [0, 0, 0], [8, 5, 6]),
    5,
  );
});

test("V3 parser rejects catalogue drift and connected-layer mistakes", () => {
  const model = tinyModel();
  assert.throws(
    () =>
      parseV3ActionConditionedActorCriticModel({
        ...model,
        actionCount: 235,
      }),
    /catalogue contract/,
  );
  assert.throws(
    () =>
      parseV3ActionConditionedActorCriticModel({
        ...model,
        actorScorerLayers: [
          { ...model.actorScorerLayers[0], inFeatures: 5 },
        ],
      }),
    /dimensions do not connect/,
  );
});
