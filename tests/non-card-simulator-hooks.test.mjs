import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

const simulatorModule = await import(
  new URL("../training/simulator.ts", import.meta.url)
);
const observationModule = await import(
  new URL("../training/non-card-observation.ts", import.meta.url)
);

const {
  createTrainingNonCardDecisionKey,
  simulateMatch,
} = simulatorModule;
const {
  NON_CARD_OBSERVATION_SCHEMA_VERSION,
  REVOLUTION_OBSERVATION_FEATURE_COUNT,
  TAX_RETURN_OBSERVATION_FEATURE_COUNT,
} = observationModule;

const LEGACY_CONFIG = {
  playerCount: 6,
  acts: 3,
  seed: 20260801,
  difficulties: ["normal"],
};

function sha256(value) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function withoutNonCardSteps(match) {
  const legacyShape = { ...match };
  delete legacyShape.nonCardSteps;
  return legacyShape;
}

function forcedOverrideConfig(step, actionIndex) {
  const namespace =
    step.decision === "tax-return" ? "taxReturn" : "revolution";
  return {
    forcedOverrides: {
      [namespace]: { [step.decisionKey]: actionIndex },
    },
  };
}

test("omitting non-card hooks preserves the exact legacy result and RNG path", () => {
  const first = simulateMatch(LEGACY_CONFIG);
  const second = simulateMatch(LEGACY_CONFIG);

  assert.deepEqual(first, second);
  assert.deepEqual(Object.keys(first), [
    "episodeId",
    "seed",
    "playerCount",
    "acts",
    "steps",
    "finalScores",
  ]);
  assert.equal(Object.hasOwn(first, "nonCardSteps"), false);
  assert.equal(
    sha256(first),
    "6baefb4ac76ba3bd8fafac58ec94c1ec980fe966a1b1c8e897ac46c2e9ff2bbc",
  );

  const recordedBaseline = simulateMatch({ ...LEGACY_CONFIG, nonCard: {} });
  assert.deepEqual(withoutNonCardSteps(recordedBaseline), first);
  assert.ok(recordedBaseline.nonCardSteps.length > 0);
});

test("tax and revolution policies receive own-hand/public-only observations and record metadata", () => {
  let taxCalls = 0;
  const taxMatch = simulateMatch({
    ...LEGACY_CONFIG,
    nonCard: {
      taxReturnPolicy(context) {
        taxCalls += 1;
        assert.equal(context.decision, "tax-return");
        assert.equal(
          context.encodedObservation.length,
          TAX_RETURN_OBSERVATION_FEATURE_COUNT,
        );
        assert.equal(
          context.observation.players.every(
            (player) =>
              JSON.stringify(Object.keys(player).sort()) ===
              JSON.stringify(["handCount", "id", "role", "score"]),
          ),
          true,
        );
        assert.equal(
          context.observation.players.some(
            (player) =>
              "hand" in player || "cards" in player || "cardIds" in player,
          ),
          false,
        );
        return {
          actionIndex: context.legalActionIndices.at(-1),
          logProbability: -0.25,
          valueEstimate: 0.75,
          policyVersion: "tax-test-v1",
        };
      },
    },
  });
  assert.equal(taxCalls, taxMatch.nonCardSteps.length);
  assert.equal(
    taxMatch.nonCardSteps.every(
      (step) =>
        step.decision === "tax-return" &&
        step.observationSchemaVersion ===
          NON_CARD_OBSERVATION_SCHEMA_VERSION &&
        step.behaviorPolicy === "custom" &&
        step.behaviorLogProbability === -0.25 &&
        step.behaviorValueEstimate === 0.75 &&
        step.behaviorPolicyVersion === "tax-test-v1" &&
        step.legalActionIndices.includes(step.actionIndex) &&
        step.observation.every(Number.isFinite) &&
        Object.keys(step.metadata).sort().join(",") ===
          "actorHandCount,playerCount,returnCount",
    ),
    true,
  );

  let revolutionContext;
  const revolutionMatch = simulateMatch({
    playerCount: 4,
    acts: 1,
    seed: 6,
    difficulties: ["easy"],
    nonCard: {
      revolutionPolicy(context) {
        revolutionContext = context;
        return {
          actionIndex: 0,
          logProbability: -0.5,
          valueEstimate: -0.25,
          policyVersion: "revolution-test-v1",
        };
      },
    },
  });
  assert.equal(
    revolutionContext.encodedObservation.length,
    REVOLUTION_OBSERVATION_FEATURE_COUNT,
  );
  assert.equal(
    revolutionContext.observation.players.some(
      (player) => "hand" in player || "cards" in player,
    ),
    false,
  );
  const [revolutionStep] = revolutionMatch.nonCardSteps;
  assert.equal(revolutionStep.decision, "revolution");
  assert.equal(revolutionStep.actorRole, "great-peon");
  assert.equal(revolutionStep.actionIndex, 0);
  assert.equal(revolutionStep.metadata.declarationKind, null);
  assert.equal(revolutionStep.behaviorPolicyVersion, "revolution-test-v1");
  assert.equal(revolutionMatch.acts[0].revolution, null);
  assert.equal("observation" in revolutionStep.metadata, false);
  assert.equal("hand" in revolutionStep.metadata, false);
});

test("forced overrides validate exact keys and legal actions", () => {
  const baseline = simulateMatch({ ...LEGACY_CONFIG, nonCard: {} });
  const taxStep = baseline.nonCardSteps.find(
    (step) => step.decision === "tax-return",
  );
  assert.ok(taxStep);
  assert.equal(
    taxStep.decisionKey,
    createTrainingNonCardDecisionKey({
      episodeId: taxStep.episodeId,
      round: taxStep.round,
      actorId: taxStep.actorId,
    }),
  );

  assert.throws(
    () =>
      simulateMatch({
        ...LEGACY_CONFIG,
        nonCard: forcedOverrideConfig(taxStep, 999),
      }),
    /tax-return forced override selected illegal action 999/,
  );
  assert.throws(
    () =>
      simulateMatch({
        ...LEGACY_CONFIG,
        nonCard: {
          forcedOverrides: {
            taxReturn: { '["wrong-episode",2,"player-3"]': 0 },
          },
        },
      }),
    /unused tax-return forced override/,
  );
  assert.throws(
    () =>
      simulateMatch({
        ...LEGACY_CONFIG,
        nonCard: {
          forcedOverrides: {
            taxReturn: { [taxStep.decisionKey]: 1.5 },
          },
        },
      }),
    /non-negative integers/,
  );
});

test("same-seed revolution counterfactuals share the world and replay exactly", () => {
  const baseConfig = {
    playerCount: 4,
    acts: 1,
    seed: 6,
    episodeId: "paired-revolution",
    difficulties: ["easy"],
  };
  const captured = simulateMatch({ ...baseConfig, nonCard: {} });
  const revolutionStep = captured.nonCardSteps.find(
    (step) => step.decision === "revolution",
  );
  assert.ok(revolutionStep);
  assert.equal(revolutionStep.actorRole, "great-peon");

  const declineConfig = {
    ...baseConfig,
    nonCard: forcedOverrideConfig(revolutionStep, 0),
  };
  const declareConfig = {
    ...baseConfig,
    nonCard: forcedOverrideConfig(revolutionStep, 1),
  };
  const decline = simulateMatch(declineConfig);
  const declineReplay = simulateMatch(declineConfig);
  const declare = simulateMatch(declareConfig);
  const declareReplay = simulateMatch(declareConfig);

  assert.deepEqual(decline, declineReplay);
  assert.deepEqual(declare, declareReplay);
  const declineStep = decline.nonCardSteps[0];
  const declareStep = declare.nonCardSteps[0];
  assert.equal(declineStep.decisionKey, declareStep.decisionKey);
  assert.deepEqual(declineStep.observation, declareStep.observation);
  assert.deepEqual(
    declineStep.legalActionIndices,
    declareStep.legalActionIndices,
  );
  assert.equal(declineStep.forcedOverride, true);
  assert.equal(declareStep.forcedOverride, true);
  assert.equal(decline.acts[0].revolution, null);
  assert.equal(declare.acts[0].revolution, "great-revolution");
  assert.notDeepEqual(
    decline.acts[0].playerOrder,
    declare.acts[0].playerOrder,
  );
});

test("same-seed tax counterfactuals keep the pre-decision encoding fixed", () => {
  const captured = simulateMatch({ ...LEGACY_CONFIG, nonCard: {} });
  const taxStep = captured.nonCardSteps.find(
    (step) =>
      step.decision === "tax-return" && step.legalActionIndices.length > 1,
  );
  assert.ok(taxStep);
  const firstAction = taxStep.legalActionIndices[0];
  const lastAction = taxStep.legalActionIndices.at(-1);
  const firstConfig = {
    ...LEGACY_CONFIG,
    nonCard: forcedOverrideConfig(taxStep, firstAction),
  };
  const lastConfig = {
    ...LEGACY_CONFIG,
    nonCard: forcedOverrideConfig(taxStep, lastAction),
  };
  const first = simulateMatch(firstConfig);
  const last = simulateMatch(lastConfig);
  assert.deepEqual(first, simulateMatch(firstConfig));
  assert.deepEqual(last, simulateMatch(lastConfig));

  const firstStep = first.nonCardSteps.find(
    (step) => step.decisionKey === taxStep.decisionKey,
  );
  const lastStep = last.nonCardSteps.find(
    (step) => step.decisionKey === taxStep.decisionKey,
  );
  assert.deepEqual(firstStep.observation, lastStep.observation);
  assert.deepEqual(firstStep.legalActionIndices, lastStep.legalActionIndices);
  assert.equal(firstStep.actionIndex, firstAction);
  assert.equal(lastStep.actionIndex, lastAction);
  assert.equal(firstStep.behaviorPolicy, "forced-override");
  assert.equal(lastStep.behaviorPolicy, "forced-override");
});
