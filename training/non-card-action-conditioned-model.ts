import type { MlpPolicyLayer } from "./model-policy.ts";
import {
  REVOLUTION_ACTION_CATALOGUE_VERSION,
  REVOLUTION_ACTION_COUNT,
  REVOLUTION_ACTION_FEATURE_COUNT,
  REVOLUTION_ACTION_FEATURE_LAYOUT,
  TAX_RETURN_ACTION_CATALOGUE_VERSION,
  TAX_RETURN_ACTION_COUNT,
  TAX_RETURN_ACTION_FEATURE_COUNT,
  TAX_RETURN_ACTION_FEATURE_LAYOUT,
  TAX_RETURN_ACTION_FEATURES,
  encodeRevolutionActionFeatures,
  legalRevolutionActionIndices,
  legalTaxReturnActionIndices,
} from "./non-card-action-space.ts";
import {
  NON_CARD_OBSERVATION_SCHEMA_VERSION,
  REVOLUTION_OBSERVATION_FEATURE_COUNT,
  TAX_RETURN_OBSERVATION_FEATURE_COUNT,
  encodeRevolutionObservation,
  encodeTaxReturnObservation,
  type RevolutionObservation,
  type TaxReturnObservation,
} from "./non-card-observation.ts";
import type {
  TrainingRevolutionPolicy,
  TrainingRevolutionPolicyContext,
  TrainingTaxReturnPolicy,
  TrainingTaxReturnPolicyContext,
} from "./simulator.ts";

export const TAX_RETURN_ACTION_CONDITIONED_MODEL_FORMAT =
  "dalmuti-tax-return-action-conditioned-actor-critic" as const;
export const REVOLUTION_ACTION_CONDITIONED_MODEL_FORMAT =
  "dalmuti-revolution-action-conditioned-actor-critic" as const;
export const NON_CARD_ACTION_CONDITIONED_MODEL_VERSION = 1 as const;

/** Actor-role one-hot starts at offset 3 and great-peon is its fifth item. */
export const REVOLUTION_GREAT_PEON_ROLE_FEATURE_INDEX = 7;

type NonCardActionConditionedActorCriticModelBase = {
  readonly version: typeof NON_CARD_ACTION_CONDITIONED_MODEL_VERSION;
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

export type TaxReturnActionConditionedActorCriticModel =
  NonCardActionConditionedActorCriticModelBase & {
    readonly format: typeof TAX_RETURN_ACTION_CONDITIONED_MODEL_FORMAT;
    readonly decisionKind: "tax-return";
  };

export type RevolutionActionConditionedActorCriticModel =
  NonCardActionConditionedActorCriticModelBase & {
    readonly format: typeof REVOLUTION_ACTION_CONDITIONED_MODEL_FORMAT;
    readonly decisionKind: "revolution";
    readonly greatPeonRoleFeatureIndex: number;
  };

export type NonCardLegalActorCriticOutput = {
  readonly actionIndices: readonly number[];
  readonly logits: Float64Array;
  readonly value: number;
};

export type NonCardBaselineGateRouting =
  | "learnedAction"
  | "agreedWithBaseline"
  | "safetyFallback";

export type NonCardBaselineGatedDecision = {
  readonly actionIndex: number;
  readonly modelActionIndex: number;
  readonly baselineActionIndex: number;
  readonly modelActionLogit: number;
  readonly baselineActionLogit: number;
  /** Model argmax logit minus the exact normal-baseline action logit. */
  readonly predictedAdvantage: number;
  readonly minimumAdvantage: number;
  readonly routing: NonCardBaselineGateRouting;
};

type ModelContract = {
  readonly format: string;
  readonly decisionKind: "tax-return" | "revolution";
  readonly observationFeatures: number;
  readonly actionCatalogueVersion: number;
  readonly actionCount: number;
  readonly actionFeatures: number;
  readonly actionFeatureLayout: readonly string[];
};

function assertPositiveInteger(value: unknown, label: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
}

function requireHiddenSizes(
  value: unknown,
  label: string,
  allowEmpty = false,
): number[] {
  if (!Array.isArray(value) || (!allowEmpty && value.length < 1)) {
    throw new TypeError(
      `${label} must ${allowEmpty ? "be a list" : "contain at least one hidden layer"}`,
    );
  }
  value.forEach((size, index) =>
    assertPositiveInteger(size, `${label}[${index}]`),
  );
  return value as number[];
}

function assertFiniteNumberArray(
  value: unknown,
  expectedLength: number,
  label: string,
): asserts value is number[] {
  if (
    !Array.isArray(value) ||
    value.length !== expectedLength ||
    value.some((entry) => !Number.isFinite(entry))
  ) {
    throw new TypeError(
      `${label} must contain ${expectedLength} finite numbers`,
    );
  }
}

function validateLayer(
  value: unknown,
  expectedInput: number,
  expectedOutput: number,
  label: string,
): asserts value is MlpPolicyLayer {
  if (!value || typeof value !== "object") {
    throw new TypeError(`${label} must be a layer object`);
  }
  const layer = value as Partial<MlpPolicyLayer>;
  assertPositiveInteger(layer.inFeatures, `${label} inFeatures`);
  assertPositiveInteger(layer.outFeatures, `${label} outFeatures`);
  if (
    layer.inFeatures !== expectedInput ||
    layer.outFeatures !== expectedOutput
  ) {
    throw new TypeError(`${label} dimensions do not connect`);
  }
  assertFiniteNumberArray(
    layer.weight,
    layer.inFeatures * layer.outFeatures,
    `${label} weight`,
  );
  assertFiniteNumberArray(layer.bias, layer.outFeatures, `${label} bias`);
}

function requireLayerList(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${label} layers must be a list`);
  }
  return value;
}

function validateHiddenStack(
  layers: readonly unknown[],
  sizes: readonly number[],
  inputSize: number,
  label: string,
): number {
  if (layers.length !== sizes.length) {
    throw new TypeError(`${label} layer count mismatch`);
  }
  let expectedInput = inputSize;
  sizes.forEach((size, index) => {
    validateLayer(layers[index], expectedInput, size, `${label} layer ${index}`);
    expectedInput = size;
  });
  return expectedInput;
}

function sameStrings(left: unknown, right: readonly string[]): boolean {
  return (
    Array.isArray(left) &&
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function parseNonCardActionConditionedModel(
  value: unknown,
  contract: ModelContract,
): NonCardActionConditionedActorCriticModelBase & Record<string, unknown> {
  if (!value || typeof value !== "object") {
    throw new TypeError("non-card actor-critic model must be an object");
  }
  const candidate = value as Record<string, unknown>;
  if (
    candidate.format !== contract.format ||
    candidate.version !== NON_CARD_ACTION_CONDITIONED_MODEL_VERSION ||
    candidate.decisionKind !== contract.decisionKind ||
    candidate.activation !== "relu" ||
    candidate.weightLayout !== "row-major [out_features, in_features]"
  ) {
    throw new TypeError("unsupported non-card actor-critic model format");
  }
  if (
    candidate.observationSchemaVersion !== NON_CARD_OBSERVATION_SCHEMA_VERSION ||
    candidate.observationFeatures !== contract.observationFeatures
  ) {
    throw new TypeError("non-card observation contract mismatch");
  }
  if (
    candidate.actionCatalogueVersion !== contract.actionCatalogueVersion ||
    candidate.actionCount !== contract.actionCount ||
    candidate.actionFeatures !== contract.actionFeatures ||
    !sameStrings(candidate.actionFeatureLayout, contract.actionFeatureLayout)
  ) {
    throw new TypeError("non-card action catalogue contract mismatch");
  }

  const actorObservationSizes = requireHiddenSizes(
    candidate.actorObservationHiddenSizes,
    "actor observation hiddenSizes",
  );
  const actorActionSizes = requireHiddenSizes(
    candidate.actorActionHiddenSizes,
    "actor action hiddenSizes",
  );
  const actorScorerSizes = requireHiddenSizes(
    candidate.actorScorerHiddenSizes,
    "actor scorer hiddenSizes",
    true,
  );
  const valueSizes = requireHiddenSizes(
    candidate.valueHiddenSizes,
    "value hiddenSizes",
  );
  const actorObservationLayers = requireLayerList(
    candidate.actorObservationLayers,
    "actor observation trunk",
  );
  const actorActionLayers = requireLayerList(
    candidate.actorActionLayers,
    "actor action trunk",
  );
  const actorScorerLayers = requireLayerList(
    candidate.actorScorerLayers,
    "actor scorer",
  );
  const valueLayers = requireLayerList(candidate.valueLayers, "value network");

  const actorObservationOutput = validateHiddenStack(
    actorObservationLayers,
    actorObservationSizes,
    contract.observationFeatures,
    "actor observation trunk",
  );
  const actorActionOutput = validateHiddenStack(
    actorActionLayers,
    actorActionSizes,
    contract.actionFeatures,
    "actor action trunk",
  );

  if (actorScorerLayers.length !== actorScorerSizes.length + 1) {
    throw new TypeError("actor scorer layer count mismatch");
  }
  let scorerInput = actorObservationOutput + actorActionOutput;
  actorScorerSizes.forEach((size, index) => {
    validateLayer(
      actorScorerLayers[index],
      scorerInput,
      size,
      `actor scorer layer ${index}`,
    );
    scorerInput = size;
  });
  validateLayer(
    actorScorerLayers.at(-1),
    scorerInput,
    1,
    "actor scorer output layer",
  );

  if (valueLayers.length !== valueSizes.length + 1) {
    throw new TypeError("value network layer count mismatch");
  }
  let valueInput = contract.observationFeatures;
  valueSizes.forEach((size, index) => {
    validateLayer(valueLayers[index], valueInput, size, `value layer ${index}`);
    valueInput = size;
  });
  validateLayer(
    valueLayers.at(-1),
    valueInput,
    1,
    "value output layer",
  );

  return candidate as NonCardActionConditionedActorCriticModelBase &
    Record<string, unknown>;
}

const TAX_RETURN_MODEL_CONTRACT: ModelContract = {
  format: TAX_RETURN_ACTION_CONDITIONED_MODEL_FORMAT,
  decisionKind: "tax-return",
  observationFeatures: TAX_RETURN_OBSERVATION_FEATURE_COUNT,
  actionCatalogueVersion: TAX_RETURN_ACTION_CATALOGUE_VERSION,
  actionCount: TAX_RETURN_ACTION_COUNT,
  actionFeatures: TAX_RETURN_ACTION_FEATURE_COUNT,
  actionFeatureLayout: TAX_RETURN_ACTION_FEATURE_LAYOUT,
};

const REVOLUTION_MODEL_CONTRACT: ModelContract = {
  format: REVOLUTION_ACTION_CONDITIONED_MODEL_FORMAT,
  decisionKind: "revolution",
  observationFeatures: REVOLUTION_OBSERVATION_FEATURE_COUNT,
  actionCatalogueVersion: REVOLUTION_ACTION_CATALOGUE_VERSION,
  actionCount: REVOLUTION_ACTION_COUNT,
  actionFeatures: REVOLUTION_ACTION_FEATURE_COUNT,
  actionFeatureLayout: REVOLUTION_ACTION_FEATURE_LAYOUT,
};

export function parseTaxReturnActionConditionedActorCriticModel(
  value: unknown,
): TaxReturnActionConditionedActorCriticModel {
  return parseNonCardActionConditionedModel(
    value,
    TAX_RETURN_MODEL_CONTRACT,
  ) as TaxReturnActionConditionedActorCriticModel;
}

export function parseRevolutionActionConditionedActorCriticModel(
  value: unknown,
): RevolutionActionConditionedActorCriticModel {
  const model = parseNonCardActionConditionedModel(
    value,
    REVOLUTION_MODEL_CONTRACT,
  );
  if (
    model.greatPeonRoleFeatureIndex !==
    REVOLUTION_GREAT_PEON_ROLE_FEATURE_INDEX
  ) {
    throw new TypeError("revolution role-conditioned action contract mismatch");
  }
  return model as RevolutionActionConditionedActorCriticModel;
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
  let values: ArrayLike<number> = inputs;
  let output: Float64Array<ArrayBufferLike> = new Float64Array(0);
  for (const layer of layers) {
    output = runLayer(values, layer, true);
    values = output;
  }
  return output;
}

function assertFiniteObservation(
  observation: readonly number[],
  expectedLength: number,
): void {
  if (
    observation.length !== expectedLength ||
    observation.some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError(
      `observation must contain ${expectedLength} finite numbers`,
    );
  }
}

function validateLegalActionIndices(
  actionIndices: readonly number[],
  actionCount: number,
): number[] {
  if (actionIndices.length < 1) {
    throw new RangeError("at least one legal non-card action is required");
  }
  const result = [...actionIndices];
  const unique = new Set<number>();
  for (const actionIndex of result) {
    if (
      !Number.isInteger(actionIndex) ||
      actionIndex < 0 ||
      actionIndex >= actionCount
    ) {
      throw new RangeError(`invalid legal non-card action index ${actionIndex}`);
    }
    if (unique.has(actionIndex)) {
      throw new RangeError(`duplicate legal non-card action index ${actionIndex}`);
    }
    unique.add(actionIndex);
  }
  return result;
}

function evaluateNonCardActionConditionedActorCritic(
  model: NonCardActionConditionedActorCriticModelBase,
  observation: readonly number[],
  legalActionIndicesValue: readonly number[],
  actionFeatures: (actionIndex: number) => readonly number[],
): NonCardLegalActorCriticOutput {
  assertFiniteObservation(observation, model.observationFeatures);
  const actionIndices = validateLegalActionIndices(
    legalActionIndicesValue,
    model.actionCount,
  );
  const actorObservation = runHiddenStack(
    observation,
    model.actorObservationLayers,
  );
  const logits = new Float64Array(actionIndices.length);

  actionIndices.forEach((actionIndex, outputIndex) => {
    const features = actionFeatures(actionIndex);
    assertFiniteObservation(features, model.actionFeatures);
    const actorAction = runHiddenStack(features, model.actorActionLayers);
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

export function evaluateTaxReturnActionConditionedActorCritic(
  model: TaxReturnActionConditionedActorCriticModel,
  observation: TaxReturnObservation,
): NonCardLegalActorCriticOutput {
  const encodedObservation = encodeTaxReturnObservation(observation);
  const legalActionIndicesValue = legalTaxReturnActionIndices(observation);
  return evaluateNonCardActionConditionedActorCritic(
    model,
    encodedObservation,
    legalActionIndicesValue,
    (actionIndex) => TAX_RETURN_ACTION_FEATURES[actionIndex],
  );
}

export function evaluateRevolutionActionConditionedActorCritic(
  model: RevolutionActionConditionedActorCriticModel,
  observation: RevolutionObservation,
): NonCardLegalActorCriticOutput {
  const encodedObservation = encodeRevolutionObservation(observation);
  const legalActionIndicesValue = legalRevolutionActionIndices(observation);
  return evaluateNonCardActionConditionedActorCritic(
    model,
    encodedObservation,
    legalActionIndicesValue,
    (actionIndex) => encodeRevolutionActionFeatures(observation, actionIndex),
  );
}

function selectLowestIndexArgmax(output: NonCardLegalActorCriticOutput): number {
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

/**
 * Training/evaluation-only conservative routing for a learned non-card policy.
 * A zero threshold is deliberately equivalent to ordinary deterministic
 * argmax, including a tied lower-index model action that differs from the
 * baseline. Production integration makes its own separately reviewed choice.
 */
export function selectBaselineGatedNonCardAction(
  output: NonCardLegalActorCriticOutput,
  baselineActionIndex: number,
  minimumAdvantage = 0,
): NonCardBaselineGatedDecision {
  if (!Number.isFinite(minimumAdvantage) || minimumAdvantage < 0) {
    throw new RangeError("minimumAdvantage must be a non-negative finite number");
  }
  if (output.actionIndices.length !== output.logits.length) {
    throw new TypeError("non-card action indices and logits must have equal length");
  }
  const baselineOffset = output.actionIndices.indexOf(baselineActionIndex);
  if (baselineOffset < 0) {
    throw new RangeError(
      `baseline action ${baselineActionIndex} is not legal for this observation`,
    );
  }
  output.logits.forEach((logit, index) => {
    if (!Number.isFinite(logit)) {
      throw new RangeError(
        `non-card model produced a non-finite logit at legal offset ${index}`,
      );
    }
  });

  const modelActionIndex = selectLowestIndexArgmax(output);
  const modelOffset = output.actionIndices.indexOf(modelActionIndex);
  const modelActionLogit = output.logits[modelOffset];
  const baselineActionLogit = output.logits[baselineOffset];
  const rawAdvantage = modelActionLogit - baselineActionLogit;
  const predictedAdvantage = Object.is(rawAdvantage, -0) ? 0 : rawAdvantage;
  if (predictedAdvantage < 0) {
    throw new Error("non-card argmax logit is below the baseline action logit");
  }

  const agreesWithBaseline = modelActionIndex === baselineActionIndex;
  const safetyFallback =
    !agreesWithBaseline && predictedAdvantage < minimumAdvantage;
  const routing: NonCardBaselineGateRouting = agreesWithBaseline
    ? "agreedWithBaseline"
    : safetyFallback
      ? "safetyFallback"
      : "learnedAction";
  return {
    actionIndex: safetyFallback ? baselineActionIndex : modelActionIndex,
    modelActionIndex,
    baselineActionIndex,
    modelActionLogit,
    baselineActionLogit,
    predictedAdvantage,
    minimumAdvantage,
    routing,
  };
}

export function selectTaxReturnActionConditionedAction(
  model: TaxReturnActionConditionedActorCriticModel,
  observation: TaxReturnObservation,
): number {
  return selectLowestIndexArgmax(
    evaluateTaxReturnActionConditionedActorCritic(model, observation),
  );
}

export function selectRevolutionActionConditionedAction(
  model: RevolutionActionConditionedActorCriticModel,
  observation: RevolutionObservation,
): number {
  return selectLowestIndexArgmax(
    evaluateRevolutionActionConditionedActorCritic(model, observation),
  );
}

function sameNumbers(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => Object.is(value, right[index]))
  );
}

function validatePolicyVersion(policyVersion: string | undefined): void {
  if (
    policyVersion !== undefined &&
    (typeof policyVersion !== "string" || policyVersion.length < 1)
  ) {
    throw new TypeError("policyVersion must be a non-empty string");
  }
}

function validatePolicyContext(
  context: TrainingTaxReturnPolicyContext | TrainingRevolutionPolicyContext,
  expectedDecision: "tax-return" | "revolution",
  encodedObservation: readonly number[],
  legalActionIndicesValue: readonly number[],
): void {
  const actorIndex = context.observation.players.findIndex(
    (player) => player.id === context.observation.actorId,
  );
  const actor = context.observation.players[actorIndex];
  if (
    context.decision !== expectedDecision ||
    context.actorId !== context.observation.actorId ||
    context.round !== context.observation.round ||
    context.actorSeat !== actorIndex ||
    context.actorRole !== actor?.role ||
    context.actorScore !== actor?.score
  ) {
    throw new TypeError(`${expectedDecision} policy context identity mismatch`);
  }
  if (!sameNumbers(context.encodedObservation, encodedObservation)) {
    throw new TypeError(
      `${expectedDecision} encodedObservation does not match the exact observation`,
    );
  }
  if (!sameNumbers(context.legalActionIndices, legalActionIndicesValue)) {
    throw new TypeError(
      `${expectedDecision} legalActionIndices do not match exact hand-derived legality`,
    );
  }
}

function policyDecision(
  output: NonCardLegalActorCriticOutput,
  policyVersion: string | undefined,
) {
  const result = {
    actionIndex: selectLowestIndexArgmax(output),
    logProbability: 0,
    valueEstimate: output.value,
  };
  return policyVersion === undefined
    ? result
    : { ...result, policyVersion };
}

export function createTaxReturnModelTrainingPolicy(
  modelValue: unknown,
  policyVersion?: string,
): TrainingTaxReturnPolicy {
  validatePolicyVersion(policyVersion);
  const model = parseTaxReturnActionConditionedActorCriticModel(modelValue);
  return (context) => {
    const encodedObservation = encodeTaxReturnObservation(context.observation);
    const legalActionIndicesValue = legalTaxReturnActionIndices(
      context.observation,
    );
    validatePolicyContext(
      context,
      "tax-return",
      encodedObservation,
      legalActionIndicesValue,
    );
    return policyDecision(
      evaluateNonCardActionConditionedActorCritic(
        model,
        encodedObservation,
        legalActionIndicesValue,
        (actionIndex) => TAX_RETURN_ACTION_FEATURES[actionIndex],
      ),
      policyVersion,
    );
  };
}

export function createRevolutionModelTrainingPolicy(
  modelValue: unknown,
  policyVersion?: string,
): TrainingRevolutionPolicy {
  validatePolicyVersion(policyVersion);
  const model = parseRevolutionActionConditionedActorCriticModel(modelValue);
  return (context) => {
    const encodedObservation = encodeRevolutionObservation(context.observation);
    const legalActionIndicesValue = legalRevolutionActionIndices(
      context.observation,
    );
    validatePolicyContext(
      context,
      "revolution",
      encodedObservation,
      legalActionIndicesValue,
    );
    return policyDecision(
      evaluateNonCardActionConditionedActorCritic(
        model,
        encodedObservation,
        legalActionIndicesValue,
        (actionIndex) =>
          encodeRevolutionActionFeatures(context.observation, actionIndex),
      ),
      policyVersion,
    );
  };
}
