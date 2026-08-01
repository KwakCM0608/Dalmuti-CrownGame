import type { MlpPolicyLayer } from "./model-policy.ts";
import {
  V3_ACTION_CATALOGUE_VERSION,
  V3_ACTION_COUNT,
  V3_ACTION_FEATURE_COUNT,
  V3_ACTION_FEATURE_LAYOUT,
  V3_ACTION_FEATURES,
} from "./v3-action-catalogue.ts";

export type V3ActionConditionedActorCriticModel = {
  readonly format: "dalmuti-action-conditioned-actor-critic";
  readonly version: 1;
  readonly observationSchemaVersion: number;
  readonly observationFeatures: number;
  readonly actionCatalogueVersion: number;
  readonly actionCount: number;
  readonly actionFeatures: number;
  readonly actionFeatureLayout: readonly string[];
  readonly actorObservationHiddenSizes: readonly number[];
  readonly actorActionHiddenSizes: readonly number[];
  readonly actorScorerHiddenSizes: readonly number[];
  readonly valueHiddenSizes: readonly number[];
  readonly activation: "relu";
  readonly weightLayout: "row-major [out_features, in_features]";
  readonly actorObservationLayers: readonly MlpPolicyLayer[];
  readonly actorActionLayers: readonly MlpPolicyLayer[];
  readonly actorScorerLayers: readonly MlpPolicyLayer[];
  readonly valueLayers: readonly MlpPolicyLayer[];
};

export type V3LegalActorCriticOutput = {
  readonly actionIndices: readonly number[];
  readonly logits: Float64Array;
  readonly value: number;
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
  expectedOutput: number,
  label: string,
): void {
  if (!layer || typeof layer !== "object") {
    throw new TypeError(`${label} must be a layer object`);
  }
  assertPositiveInteger(layer.inFeatures, `${label} inFeatures`);
  assertPositiveInteger(layer.outFeatures, `${label} outFeatures`);
  if (
    layer.inFeatures !== expectedInput ||
    layer.outFeatures !== expectedOutput
  ) {
    throw new TypeError(`${label} dimensions do not connect`);
  }
  assertFiniteArray(
    layer.weight,
    layer.inFeatures * layer.outFeatures,
    `${label} weight`,
  );
  assertFiniteArray(layer.bias, layer.outFeatures, `${label} bias`);
}

function validateHiddenStack(
  layers: readonly MlpPolicyLayer[],
  sizes: readonly number[],
  inputSize: number,
  label: string,
  requireLayer = true,
): number {
  if (requireLayer && sizes.length < 1) {
    throw new TypeError(`${label} must contain at least one hidden layer`);
  }
  if (layers.length !== sizes.length) {
    throw new TypeError(`${label} layer count mismatch`);
  }
  let expectedInput = inputSize;
  sizes.forEach((size, index) => {
    assertPositiveInteger(size, `${label} hiddenSizes[${index}]`);
    validateLayer(
      layers[index],
      expectedInput,
      size,
      `${label} layer ${index}`,
    );
    expectedInput = size;
  });
  return expectedInput;
}

function sameStrings(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

export function parseV3ActionConditionedActorCriticModel(
  value: unknown,
): V3ActionConditionedActorCriticModel {
  if (!value || typeof value !== "object") {
    throw new TypeError("V3 actor-critic model must be an object");
  }
  const candidate = value as Partial<V3ActionConditionedActorCriticModel>;
  if (
    candidate.format !== "dalmuti-action-conditioned-actor-critic" ||
    candidate.version !== 1 ||
    candidate.activation !== "relu" ||
    candidate.weightLayout !== "row-major [out_features, in_features]"
  ) {
    throw new TypeError("unsupported V3 actor-critic model format");
  }
  assertPositiveInteger(
    candidate.observationSchemaVersion ?? 0,
    "observationSchemaVersion",
  );
  assertPositiveInteger(
    candidate.observationFeatures ?? 0,
    "observationFeatures",
  );
  if (
    candidate.actionCatalogueVersion !== V3_ACTION_CATALOGUE_VERSION ||
    candidate.actionCount !== V3_ACTION_COUNT ||
    candidate.actionFeatures !== V3_ACTION_FEATURE_COUNT ||
    !Array.isArray(candidate.actionFeatureLayout) ||
    !sameStrings(candidate.actionFeatureLayout, V3_ACTION_FEATURE_LAYOUT)
  ) {
    throw new TypeError("V3 action catalogue contract mismatch");
  }

  const sizeGroups = [
    candidate.actorObservationHiddenSizes,
    candidate.actorActionHiddenSizes,
    candidate.actorScorerHiddenSizes,
    candidate.valueHiddenSizes,
  ];
  const layerGroups = [
    candidate.actorObservationLayers,
    candidate.actorActionLayers,
    candidate.actorScorerLayers,
    candidate.valueLayers,
  ];
  if (
    sizeGroups.some((group) => !Array.isArray(group)) ||
    layerGroups.some((group) => !Array.isArray(group))
  ) {
    throw new TypeError("V3 actor-critic model structure is incomplete");
  }

  const actorObservationOutput = validateHiddenStack(
    candidate.actorObservationLayers!,
    candidate.actorObservationHiddenSizes!,
    candidate.observationFeatures!,
    "actor observation trunk",
  );
  const actorActionOutput = validateHiddenStack(
    candidate.actorActionLayers!,
    candidate.actorActionHiddenSizes!,
    V3_ACTION_FEATURE_COUNT,
    "actor action trunk",
  );

  if (
    candidate.actorScorerLayers!.length !==
    candidate.actorScorerHiddenSizes!.length + 1
  ) {
    throw new TypeError("actor scorer layer count mismatch");
  }
  let scorerInput = actorObservationOutput + actorActionOutput;
  candidate.actorScorerHiddenSizes!.forEach((size, index) => {
    assertPositiveInteger(size, `actor scorer hiddenSizes[${index}]`);
    validateLayer(
      candidate.actorScorerLayers![index],
      scorerInput,
      size,
      `actor scorer layer ${index}`,
    );
    scorerInput = size;
  });
  validateLayer(
    candidate.actorScorerLayers!.at(-1)!,
    scorerInput,
    1,
    "actor scorer output layer",
  );

  if (
    candidate.valueLayers!.length !== candidate.valueHiddenSizes!.length + 1
  ) {
    throw new TypeError("value network layer count mismatch");
  }
  let valueInput = candidate.observationFeatures!;
  candidate.valueHiddenSizes!.forEach((size, index) => {
    assertPositiveInteger(size, `value hiddenSizes[${index}]`);
    validateLayer(
      candidate.valueLayers![index],
      valueInput,
      size,
      `value layer ${index}`,
    );
    valueInput = size;
  });
  validateLayer(
    candidate.valueLayers!.at(-1)!,
    valueInput,
    1,
    "value output layer",
  );
  return candidate as V3ActionConditionedActorCriticModel;
}

function runLayer(
  inputs: ArrayLike<number>,
  layer: MlpPolicyLayer,
  applyRelu: boolean,
): Float64Array {
  const output = new Float64Array(layer.outFeatures);
  for (let row = 0; row < layer.outFeatures; row += 1) {
    let value = layer.bias[row];
    const offset = row * layer.inFeatures;
    for (let column = 0; column < layer.inFeatures; column += 1) {
      value += layer.weight[offset + column] * inputs[column];
    }
    output[row] = applyRelu ? Math.max(0, value) : value;
  }
  return output;
}

function runHiddenStack(
  inputs: ArrayLike<number>,
  layers: readonly MlpPolicyLayer[],
): Float64Array {
  let hidden: ArrayLike<number> = inputs;
  let output: Float64Array<ArrayBufferLike> = new Float64Array(0);
  for (const layer of layers) {
    output = runLayer(hidden, layer, true);
    hidden = output;
  }
  return output;
}

function validateLegalActionIndices(
  actionIndices: readonly number[],
): number[] {
  if (actionIndices.length < 1) {
    throw new RangeError("at least one legal V3 action is required");
  }
  const result = [...actionIndices];
  const unique = new Set<number>();
  for (const actionIndex of result) {
    if (
      !Number.isInteger(actionIndex) ||
      actionIndex < 0 ||
      actionIndex >= V3_ACTION_COUNT
    ) {
      throw new RangeError(`invalid legal V3 action index ${actionIndex}`);
    }
    if (unique.has(actionIndex)) {
      throw new RangeError(`duplicate legal V3 action index ${actionIndex}`);
    }
    unique.add(actionIndex);
  }
  return result;
}

export function evaluateV3ActionConditionedActorCritic(
  model: V3ActionConditionedActorCriticModel,
  observation: readonly number[],
  legalActionIndices: readonly number[],
): V3LegalActorCriticOutput {
  assertFiniteArray(
    observation,
    model.observationFeatures,
    "observation",
  );
  const actionIndices = validateLegalActionIndices(legalActionIndices);
  const actorObservation = runHiddenStack(
    observation,
    model.actorObservationLayers,
  );
  const logits = new Float64Array(actionIndices.length);

  actionIndices.forEach((actionIndex, outputIndex) => {
    const actorAction = runHiddenStack(
      V3_ACTION_FEATURES[actionIndex],
      model.actorActionLayers,
    );
    let scorer: ArrayLike<number> = Float64Array.from([
      ...actorObservation,
      ...actorAction,
    ]);
    model.actorScorerLayers.forEach((layer, index) => {
      scorer = runLayer(
        scorer,
        layer,
        index < model.actorScorerLayers.length - 1,
      );
    });
    logits[outputIndex] = scorer[0];
  });

  let valueOutput: ArrayLike<number> = observation;
  model.valueLayers.forEach((layer, index) => {
    valueOutput = runLayer(
      valueOutput,
      layer,
      index < model.valueLayers.length - 1,
    );
  });
  return { actionIndices, logits, value: valueOutput[0] };
}

export function selectV3ActionConditionedAction(
  model: V3ActionConditionedActorCriticModel,
  observation: readonly number[],
  legalActionIndices: readonly number[],
): number {
  const output = evaluateV3ActionConditionedActorCritic(
    model,
    observation,
    legalActionIndices,
  );
  let bestAction = output.actionIndices[0];
  let bestLogit = output.logits[0];
  for (let index = 1; index < output.actionIndices.length; index += 1) {
    const actionIndex = output.actionIndices[index];
    const logit = output.logits[index];
    if (logit > bestLogit || (logit === bestLogit && actionIndex < bestAction)) {
      bestAction = actionIndex;
      bestLogit = logit;
    }
  }
  return bestAction;
}
