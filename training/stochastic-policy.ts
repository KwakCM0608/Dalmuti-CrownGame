import {
  evaluateActorCritic,
  parseActorCriticModel,
  type ActorCriticModel,
} from "./actor-critic.ts";
import {
  evaluateMlpPolicy,
  parseMlpPolicyModel,
  selectMaskedMlpAction,
  type MlpPolicyModel,
} from "./model-policy.ts";
import type {
  TrainingPolicy,
  TrainingPolicyDecision,
} from "./simulator.ts";

export type InferenceModel = MlpPolicyModel | ActorCriticModel;

export type SampledPolicyDecision = Required<
  Pick<
    TrainingPolicyDecision,
    "actionIndex" | "logProbability" | "valueEstimate"
  >
>;

export type StochasticPolicyOptions = {
  /**
   * Softmax temperature used by the behavior policy. Values above one
   * increase exploration while values below one sharpen the distribution.
   */
  temperature?: number;
};

export function parseInferenceModel(value: unknown): InferenceModel {
  if (
    value &&
    typeof value === "object" &&
    (value as { format?: unknown }).format === "dalmuti-actor-critic"
  ) {
    return parseActorCriticModel(value);
  }
  return parseMlpPolicyModel(value);
}

function validateLegalActions(
  legalActionIndices: readonly number[],
  actionCount: number,
): void {
  if (legalActionIndices.length < 1) {
    throw new RangeError("at least one legal action is required");
  }
  const unique = new Set<number>();
  for (const actionIndex of legalActionIndices) {
    if (
      !Number.isInteger(actionIndex) ||
      actionIndex < 0 ||
      actionIndex >= actionCount
    ) {
      throw new RangeError(`invalid legal action index ${actionIndex}`);
    }
    if (unique.has(actionIndex)) {
      throw new RangeError(`duplicate legal action index ${actionIndex}`);
    }
    unique.add(actionIndex);
  }
}

export function sampleMaskedLogits(
  logits: ArrayLike<number>,
  legalActionIndices: readonly number[],
  random: () => number,
  valueEstimate = 0,
  temperature = 1,
): SampledPolicyDecision {
  validateLegalActions(legalActionIndices, logits.length);
  if (!Number.isFinite(valueEstimate)) {
    throw new RangeError("value estimate must be finite");
  }
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("policy temperature must be finite and positive");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (const actionIndex of legalActionIndices) {
    const logit = logits[actionIndex];
    if (!Number.isFinite(logit)) {
      throw new RangeError(`logit ${actionIndex} must be finite`);
    }
    maximum = Math.max(maximum, logit / temperature);
  }
  const weights = legalActionIndices.map((actionIndex) =>
    Math.exp(logits[actionIndex] / temperature - maximum),
  );
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new RangeError("masked policy distribution is invalid");
  }
  const randomValue = random();
  if (
    !Number.isFinite(randomValue) ||
    randomValue < 0 ||
    randomValue >= 1
  ) {
    throw new RangeError("policy random source must return [0, 1)");
  }
  const threshold = randomValue * total;
  let cumulative = 0;
  let selectedPosition = weights.length - 1;
  for (let index = 0; index < weights.length; index += 1) {
    cumulative += weights[index];
    if (threshold < cumulative) {
      selectedPosition = index;
      break;
    }
  }
  return {
    actionIndex: legalActionIndices[selectedPosition],
    logProbability:
      logits[legalActionIndices[selectedPosition]] / temperature -
      maximum -
      Math.log(total),
    valueEstimate,
  };
}

export function sampleInferenceModel(
  model: InferenceModel,
  observation: readonly number[],
  legalActionIndices: readonly number[],
  random: () => number,
  temperature = 1,
): SampledPolicyDecision {
  if (model.format === "dalmuti-actor-critic") {
    const output = evaluateActorCritic(model, observation);
    return sampleMaskedLogits(
      output.logits,
      legalActionIndices,
      random,
      output.value,
      temperature,
    );
  }
  return sampleMaskedLogits(
    evaluateMlpPolicy(model, observation),
    legalActionIndices,
    random,
    0,
    temperature,
  );
}

export function createStochasticTrainingPolicy(
  modelValue: unknown,
  policyVersion: string,
  options: StochasticPolicyOptions = {},
): TrainingPolicy {
  if (!policyVersion) {
    throw new TypeError("policyVersion must be non-empty");
  }
  const model = parseInferenceModel(modelValue);
  const temperature = options.temperature ?? 1;
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("policy temperature must be finite and positive");
  }
  return ({
    encodedObservation,
    legalActionIndices,
    random,
  }) => ({
    ...sampleInferenceModel(
      model,
      encodedObservation,
      legalActionIndices,
      random,
      temperature,
    ),
    policyVersion,
  });
}

export function createGreedyInferenceTrainingPolicy(
  modelValue: unknown,
): TrainingPolicy {
  const model = parseInferenceModel(modelValue);
  if (model.format === "dalmuti-mlp-policy") {
    return ({ encodedObservation, legalActionIndices }) =>
      selectMaskedMlpAction(
        model,
        encodedObservation,
        legalActionIndices,
      );
  }
  return ({ encodedObservation, legalActionIndices }) => {
    const { logits } = evaluateActorCritic(model, encodedObservation);
    let bestAction = legalActionIndices[0];
    let bestLogit = Number.NEGATIVE_INFINITY;
    for (const actionIndex of legalActionIndices) {
      const logit = logits[actionIndex];
      if (
        logit > bestLogit ||
        (logit === bestLogit && actionIndex < bestAction)
      ) {
        bestAction = actionIndex;
        bestLogit = logit;
      }
    }
    return bestAction;
  };
}
