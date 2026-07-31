import {
  ACTION_SPACE_SIZE,
} from "./action-space.ts";
import {
  type MlpPolicyLayer,
} from "./model-policy.ts";
import {
  OBSERVATION_FEATURE_COUNT,
} from "./observation.ts";

export type ActorCriticModel = {
  format: "dalmuti-actor-critic";
  version: 1;
  observationFeatures: number;
  actionCount: number;
  hiddenSizes: readonly number[];
  activation: "relu";
  weightLayout: "row-major [out_features, in_features]";
  trunkLayers: readonly MlpPolicyLayer[];
  policyLayer: MlpPolicyLayer;
  valueLayer: MlpPolicyLayer;
};

export type ActorCriticOutput = {
  logits: Float64Array;
  value: number;
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

function validateLayer(
  layer: MlpPolicyLayer,
  expectedInput: number,
  label: string,
): void {
  assertPositiveInteger(layer.inFeatures, `${label} inFeatures`);
  assertPositiveInteger(layer.outFeatures, `${label} outFeatures`);
  if (layer.inFeatures !== expectedInput) {
    throw new TypeError(`${label} input size does not connect`);
  }
  assertFiniteArray(
    layer.weight,
    layer.inFeatures * layer.outFeatures,
    `${label} weight`,
  );
  assertFiniteArray(layer.bias, layer.outFeatures, `${label} bias`);
}

export function parseActorCriticModel(value: unknown): ActorCriticModel {
  if (!value || typeof value !== "object") {
    throw new TypeError("actor-critic model must be an object");
  }
  const candidate = value as Partial<ActorCriticModel>;
  if (
    candidate.format !== "dalmuti-actor-critic" ||
    candidate.version !== 1 ||
    candidate.activation !== "relu" ||
    candidate.weightLayout !== "row-major [out_features, in_features]"
  ) {
    throw new TypeError("unsupported actor-critic model format");
  }
  if (
    candidate.observationFeatures !== OBSERVATION_FEATURE_COUNT ||
    candidate.actionCount !== ACTION_SPACE_SIZE
  ) {
    throw new TypeError(
      `actor-critic model must use ${OBSERVATION_FEATURE_COUNT} ` +
        `observations and ${ACTION_SPACE_SIZE} actions`,
    );
  }
  if (
    !Array.isArray(candidate.hiddenSizes) ||
    candidate.hiddenSizes.length < 1 ||
    !Array.isArray(candidate.trunkLayers) ||
    candidate.trunkLayers.length !== candidate.hiddenSizes.length ||
    !candidate.policyLayer ||
    !candidate.valueLayer
  ) {
    throw new TypeError("actor-critic model structure is incomplete");
  }

  let expectedInput = candidate.observationFeatures;
  candidate.trunkLayers.forEach((layer, index) => {
    const expectedOutput = candidate.hiddenSizes![index];
    assertPositiveInteger(expectedOutput, `hiddenSizes[${index}]`);
    validateLayer(layer, expectedInput, `trunk layer ${index}`);
    if (layer.outFeatures !== expectedOutput) {
      throw new TypeError(`trunk layer ${index} output size mismatch`);
    }
    expectedInput = layer.outFeatures;
  });
  validateLayer(candidate.policyLayer, expectedInput, "policy layer");
  validateLayer(candidate.valueLayer, expectedInput, "value layer");
  if (candidate.policyLayer.outFeatures !== ACTION_SPACE_SIZE) {
    throw new TypeError("policy layer does not match the action count");
  }
  if (candidate.valueLayer.outFeatures !== 1) {
    throw new TypeError("value layer must contain exactly one output");
  }
  return candidate as ActorCriticModel;
}

function runLayer(
  inputs: ArrayLike<number>,
  layer: MlpPolicyLayer,
  applyRelu: boolean,
): Float64Array {
  const outputs = new Float64Array(layer.outFeatures);
  for (let output = 0; output < layer.outFeatures; output += 1) {
    let value = layer.bias[output];
    const offset = output * layer.inFeatures;
    for (let input = 0; input < layer.inFeatures; input += 1) {
      value += layer.weight[offset + input] * inputs[input];
    }
    outputs[output] = applyRelu ? Math.max(0, value) : value;
  }
  return outputs;
}

export function evaluateActorCritic(
  model: ActorCriticModel,
  observation: readonly number[],
): ActorCriticOutput {
  assertFiniteArray(
    observation,
    model.observationFeatures,
    "observation",
  );
  let hidden: ArrayLike<number> = observation;
  for (const layer of model.trunkLayers) {
    hidden = runLayer(hidden, layer, true);
  }
  return {
    logits: runLayer(hidden, model.policyLayer, false),
    value: runLayer(hidden, model.valueLayer, false)[0],
  };
}
