import {
  ACTION_SPACE_SIZE,
} from "./action-space.ts";
import {
  OBSERVATION_FEATURE_COUNT,
} from "./observation.ts";
import type {
  TrainingPolicy,
} from "./simulator.ts";

export type MlpPolicyLayer = {
  inFeatures: number;
  outFeatures: number;
  weight: readonly number[];
  bias: readonly number[];
};

export type MlpPolicyModel = {
  format: "dalmuti-mlp-policy";
  version: 1;
  observationFeatures: number;
  actionCount: number;
  hiddenSizes: readonly number[];
  activation: "relu";
  weightLayout: "row-major [out_features, in_features]";
  layers: readonly MlpPolicyLayer[];
};

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
}

function assertFiniteArray(
  values: ArrayLike<number>,
  expectedLength: number,
  label: string,
): void {
  let valid = values.length === expectedLength;
  for (let index = 0; valid && index < values.length; index += 1) {
    valid = Number.isFinite(values[index]);
  }
  if (!valid) {
    throw new TypeError(
      `${label} must contain ${expectedLength} finite numbers`,
    );
  }
}

export function parseMlpPolicyModel(value: unknown): MlpPolicyModel {
  if (!value || typeof value !== "object") {
    throw new TypeError("policy model must be an object");
  }
  const candidate = value as Partial<MlpPolicyModel>;
  if (
    candidate.format !== "dalmuti-mlp-policy" ||
    candidate.version !== 1 ||
    candidate.activation !== "relu" ||
    candidate.weightLayout !== "row-major [out_features, in_features]"
  ) {
    throw new TypeError("unsupported policy model format");
  }
  assertPositiveInteger(
    candidate.observationFeatures ?? 0,
    "observationFeatures",
  );
  assertPositiveInteger(candidate.actionCount ?? 0, "actionCount");
  if (!Array.isArray(candidate.layers) || candidate.layers.length < 1) {
    throw new TypeError("policy model must contain at least one layer");
  }

  let expectedInput = candidate.observationFeatures!;
  for (const [index, layer] of candidate.layers.entries()) {
    assertPositiveInteger(layer.inFeatures, `layer ${index} inFeatures`);
    assertPositiveInteger(layer.outFeatures, `layer ${index} outFeatures`);
    if (layer.inFeatures !== expectedInput) {
      throw new TypeError(`layer ${index} input size does not connect`);
    }
    assertFiniteArray(
      layer.weight,
      layer.inFeatures * layer.outFeatures,
      `layer ${index} weight`,
    );
    assertFiniteArray(
      layer.bias,
      layer.outFeatures,
      `layer ${index} bias`,
    );
    expectedInput = layer.outFeatures;
  }
  if (expectedInput !== candidate.actionCount) {
    throw new TypeError("final layer does not match actionCount");
  }
  return candidate as MlpPolicyModel;
}

function runLayer(
  inputs: ArrayLike<number>,
  layer: MlpPolicyLayer,
  applyRelu: boolean,
): Float64Array {
  const outputs = new Float64Array(layer.outFeatures);
  for (let output = 0; output < layer.outFeatures; output += 1) {
    let value = layer.bias[output];
    const weightOffset = output * layer.inFeatures;
    for (let input = 0; input < layer.inFeatures; input += 1) {
      value += layer.weight[weightOffset + input] * inputs[input];
    }
    outputs[output] = applyRelu ? Math.max(0, value) : value;
  }
  return outputs;
}

export function evaluateMlpPolicy(
  model: MlpPolicyModel,
  observation: readonly number[],
): Float64Array {
  assertFiniteArray(
    observation,
    model.observationFeatures,
    "observation",
  );
  let values: ArrayLike<number> = observation;
  let outputs: Float64Array<ArrayBufferLike> = new Float64Array(0);
  model.layers.forEach((layer, index) => {
    outputs = runLayer(
      values,
      layer,
      index < model.layers.length - 1,
    );
    values = outputs;
  });
  return outputs;
}

export function selectMaskedMlpAction(
  model: MlpPolicyModel,
  observation: readonly number[],
  legalActionIndices: readonly number[],
): number {
  if (legalActionIndices.length < 1) {
    throw new RangeError("at least one legal action is required");
  }
  const logits = evaluateMlpPolicy(model, observation);
  let bestAction = legalActionIndices[0];
  let bestLogit = Number.NEGATIVE_INFINITY;
  for (const actionIndex of legalActionIndices) {
    if (
      !Number.isInteger(actionIndex) ||
      actionIndex < 0 ||
      actionIndex >= model.actionCount
    ) {
      throw new RangeError(`invalid legal action index ${actionIndex}`);
    }
    const logit = logits[actionIndex];
    if (
      logit > bestLogit ||
      (logit === bestLogit && actionIndex < bestAction)
    ) {
      bestLogit = logit;
      bestAction = actionIndex;
    }
  }
  return bestAction;
}

export function createMlpTrainingPolicy(
  modelValue: unknown,
): TrainingPolicy {
  const model = parseMlpPolicyModel(modelValue);
  if (
    model.observationFeatures !== OBSERVATION_FEATURE_COUNT ||
    model.actionCount !== ACTION_SPACE_SIZE
  ) {
    throw new TypeError(
      `policy must use ${OBSERVATION_FEATURE_COUNT} observations and ` +
        `${ACTION_SPACE_SIZE} actions`,
    );
  }
  return ({ encodedObservation, legalActionIndices }) =>
    selectMaskedMlpAction(
      model,
      encodedObservation,
      legalActionIndices,
    );
}
