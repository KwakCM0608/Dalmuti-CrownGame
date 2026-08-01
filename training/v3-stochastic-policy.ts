import {
  OBSERVATION_FEATURE_COUNT,
  OBSERVATION_SCHEMA_VERSION,
} from "./observation.ts";
import type { TrainingPolicy } from "./simulator.ts";
import {
  evaluateV3ActionConditionedActorCritic,
  parseV3ActionConditionedActorCriticModel,
  type V3ActionConditionedActorCriticModel,
} from "./v3-action-conditioned-model.ts";
import {
  legacyLegalActionIndicesToV3,
  v3ActionIndexToLegacy,
} from "./v3-action-bridge.ts";

function parseRuntimeModel(value: unknown): V3ActionConditionedActorCriticModel {
  const model = parseV3ActionConditionedActorCriticModel(value);
  if (
    model.observationSchemaVersion !== OBSERVATION_SCHEMA_VERSION ||
    model.observationFeatures !== OBSERVATION_FEATURE_COUNT
  ) {
    throw new TypeError(
      `V3 play model must use observation schema ${OBSERVATION_SCHEMA_VERSION} ` +
        `with ${OBSERVATION_FEATURE_COUNT} features`,
    );
  }
  return model;
}

function sampleLocalLogits(
  logits: ArrayLike<number>,
  random: () => number,
  temperature: number,
): { localIndex: number; logProbability: number } {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("policy temperature must be finite and positive");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.length; index += 1) {
    if (!Number.isFinite(logits[index])) {
      throw new RangeError("V3 policy logits must be finite");
    }
    maximum = Math.max(maximum, logits[index] / temperature);
  }
  const weights = Array.from(
    { length: logits.length },
    (_, index) => Math.exp(logits[index] / temperature - maximum),
  );
  const total = weights.reduce((sum, value) => sum + value, 0);
  const draw = random();
  if (!Number.isFinite(draw) || draw < 0 || draw >= 1) {
    throw new RangeError("policy random source must return [0, 1)");
  }
  const threshold = draw * total;
  let cumulative = 0;
  let localIndex = weights.length - 1;
  for (let index = 0; index < weights.length; index += 1) {
    cumulative += weights[index];
    if (threshold < cumulative) {
      localIndex = index;
      break;
    }
  }
  return {
    localIndex,
    logProbability:
      logits[localIndex] / temperature - maximum - Math.log(total),
  };
}

export function createStochasticV3TrainingPolicy(
  modelValue: unknown,
  policyVersion: string,
  temperature = 1,
): TrainingPolicy {
  if (typeof policyVersion !== "string" || policyVersion.length < 1) {
    throw new TypeError("policyVersion must be non-empty");
  }
  const model = parseRuntimeModel(modelValue);
  return ({ encodedObservation, legalActionIndices, random }) => {
    const v3Legal = legacyLegalActionIndicesToV3(legalActionIndices);
    const output = evaluateV3ActionConditionedActorCritic(
      model,
      encodedObservation,
      v3Legal,
    );
    const sampled = sampleLocalLogits(output.logits, random, temperature);
    return {
      actionIndex: v3ActionIndexToLegacy(
        output.actionIndices[sampled.localIndex],
      ),
      logProbability: sampled.logProbability,
      valueEstimate: output.value,
      policyVersion,
    };
  };
}

export function createGreedyV3TrainingPolicy(
  modelValue: unknown,
): TrainingPolicy {
  const model = parseRuntimeModel(modelValue);
  return ({ encodedObservation, legalActionIndices }) => {
    const v3Legal = legacyLegalActionIndicesToV3(legalActionIndices);
    const output = evaluateV3ActionConditionedActorCritic(
      model,
      encodedObservation,
      v3Legal,
    );
    let localBest = 0;
    for (let index = 1; index < output.logits.length; index += 1) {
      if (
        output.logits[index] > output.logits[localBest] ||
        (output.logits[index] === output.logits[localBest] &&
          output.actionIndices[index] < output.actionIndices[localBest])
      ) {
        localBest = index;
      }
    }
    return v3ActionIndexToLegacy(output.actionIndices[localBest]);
  };
}
