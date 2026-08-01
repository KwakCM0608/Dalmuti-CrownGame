import type { MlpPolicyLayer } from "./model-policy.ts";
import {
  TAX_RETURN_ACTION_CATALOGUE_VERSION,
  TAX_RETURN_ACTION_COUNT,
  TAX_RETURN_ACTION_FEATURE_COUNT,
  TAX_RETURN_ACTION_FEATURE_LAYOUT,
  TAX_RETURN_ACTION_FEATURES,
  legalTaxReturnActionIndices,
} from "./non-card-action-space.ts";
import {
  NON_CARD_OBSERVATION_SCHEMA_VERSION,
  TAX_RETURN_OBSERVATION_FEATURE_COUNT,
  encodeTaxReturnObservation,
  type TaxReturnObservation,
} from "./non-card-observation.ts";

export const TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT =
  "dalmuti-tax-return-bilinear-residual-ensemble" as const;
export const TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION = 2 as const;
export const TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS =
  "chip-advantage-vs-normal-baseline" as const;
export const TAX_RETURN_ADVANTAGE_ENSEMBLE_MEMBER_COUNT = 5 as const;
export const TAX_RETURN_ADVANTAGE_ENSEMBLE_Z_VALUE = 1.645 as const;
export const TAX_RETURN_ADVANTAGE_DEFAULT_MINIMUM_CHIPS = 0.5 as const;
export const TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT =
  "row-major [context_features, action_features]" as const;
export const TAX_RETURN_DETERMINIZATION_ALGORITHM =
  "target-act-opponent-physical-card-fisher-yates-v1" as const;
export const TAX_RETURN_DETERMINIZATION_ALGORITHM_VERSION = 1 as const;
export const TAX_RETURN_DETERMINIZATION_ALGORITHM_CONTRACT_SHA256 =
  "368240f14f2e5d84bb3085610a176ad4519bc6e5ae288b70de549f63212905c4" as const;
export const TAX_RETURN_DETERMINIZATION_CANDIDATE_SEED_DERIVATION =
  "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,attempt)))" as const;
export const TAX_RETURN_DETERMINIZATION_CONTINUATION_SEED_DERIVATION =
  "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,continuationIndex,continuation)))" as const;

/** SHA-256 of the canonical JSON baseline provenance object in the trainer. */
export const TAX_RETURN_NORMAL_BASELINE_PROVENANCE_SHA256 =
  "99228c7c28a500dd54c17707270729b08ce78c15ae7385037789c018f871b57e" as const;

export const TAX_RETURN_NORMAL_BASELINE_PROVENANCE = Object.freeze({
  implementation: "lib/bot-strategy.ts#chooseBotTaxReturn",
  semanticEncoding:
    "training/non-card-action-space.ts#encodeTaxReturnAction",
  difficulty: "normal",
});

type TaxReturnAdvantageMember = {
  readonly memberIndex: number;
  readonly seed: number;
  readonly checkpointEpoch: number;
  readonly validationPairedLoss: number;
  readonly parametersSha256: string;
  readonly contextLayer: MlpPolicyLayer;
  readonly bilinearWeight: readonly number[];
};

export type TaxReturnAdvantageEnsembleModel = {
  readonly format: typeof TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT;
  readonly version: typeof TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION;
  readonly decisionKind: "tax-return";
  readonly scoreSemantics: typeof TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS;
  readonly observationSchemaVersion: number;
  readonly observationFeatures: number;
  readonly actionCatalogueVersion: number;
  readonly actionCount: number;
  readonly actionFeatures: number;
  readonly actionFeatureLayout: readonly string[];
  readonly trainingData: {
    readonly sourceFormatVersions: readonly [1] | readonly [2];
    readonly groupSplitKey:
      | "canonicalWorldKey"
      | "canonicalInformationStateKey";
    readonly determinizationSchema:
      | null
      | "world-clustered-paired-baseline-advantages-v2";
    readonly worldCountPerInformationState: number;
    readonly continuationCountPerHiddenWorld: number;
    readonly effectiveIndependentWorldsPerInformationState: number;
    readonly rawContinuationEvaluationsPerInformationState: number;
    readonly standardErrorEstimable: boolean;
    readonly determinizationAlgorithm:
      | null
      | typeof TAX_RETURN_DETERMINIZATION_ALGORITHM;
    readonly determinizationAlgorithmVersion:
      | null
      | typeof TAX_RETURN_DETERMINIZATION_ALGORITHM_VERSION;
    readonly determinizationAlgorithmContractSha256:
      | null
      | typeof TAX_RETURN_DETERMINIZATION_ALGORITHM_CONTRACT_SHA256;
    readonly candidateSeedDerivation:
      | null
      | typeof TAX_RETURN_DETERMINIZATION_CANDIDATE_SEED_DERIVATION;
    readonly continuationSeedDerivation:
      | null
      | typeof TAX_RETURN_DETERMINIZATION_CONTINUATION_SEED_DERIVATION;
    readonly targetField: string;
    readonly targetTransform: {
      readonly scoreUnit: "chip-units";
      readonly sourceUnit: "(roundChipAward-2)/2";
      readonly operation: "multiply-source-baseline-advantage-by-2";
      readonly multiplier: 2;
    };
    readonly stateWeighting: "one-per-information-state-independent-of-worldCount";
  };
  readonly architecture: {
    readonly contextFeatures: number;
    readonly contextActivation: "tanh";
    readonly score: "raw(s,a)-raw(s,normalBaselineAction)";
    readonly weightLayout: typeof TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT;
  };
  readonly baseline: {
    readonly provenance: typeof TAX_RETURN_NORMAL_BASELINE_PROVENANCE;
    readonly provenanceSha256: string;
    readonly score: "exactly-zero-by-residualization";
  };
  readonly objective: {
    readonly utilityTarget: "decision-act-current-chip-advantage";
    readonly utilityScale: "chip-units";
    readonly weighting: "equal-per-state";
    readonly regression: Readonly<Record<string, unknown>>;
    readonly tieAwareSign: Readonly<Record<string, unknown>>;
    readonly checkpointSelection: "paired-validation-loss";
    readonly bootstrapUnit:
      | "canonicalWorldKey"
      | "canonicalInformationStateKey";
  };
  readonly routing: {
    readonly returnCountOne: "exact-normal-fallback";
    readonly returnCountTwo: "ensemble-lower-confidence-bound";
    readonly roleRouting: {
      readonly "great-dalmuti": "ensemble-lower-confidence-bound";
      readonly "lesser-dalmuti": "exact-normal-fallback";
      readonly "other-roles": "not-applicable";
    };
    readonly memberCount: typeof TAX_RETURN_ADVANTAGE_ENSEMBLE_MEMBER_COUNT;
    readonly unanimityRule: "all-member-advantages-strictly-positive";
    readonly lowerConfidenceBound: "mean-minus-z-times-sample-sd";
    readonly zValue: typeof TAX_RETURN_ADVANTAGE_ENSEMBLE_Z_VALUE;
    readonly defaultMinimumChipAdvantage: number;
    readonly selection: "maximum-eligible-lcb";
    readonly tieBreak: "baseline-then-lowest-action-index";
  };
  readonly members: readonly TaxReturnAdvantageMember[];
};

export type TaxReturnAdvantageActionScore = {
  readonly actionIndex: number;
  readonly memberAdvantages: readonly number[];
  readonly meanAdvantage: number;
  readonly sampleStandardDeviation: number;
  readonly lowerConfidenceBound: number;
  readonly unanimousPositive: boolean;
  readonly eligible: boolean;
};

export type TaxReturnAdvantageEnsembleDecision = {
  readonly actionIndex: number;
  readonly modelActionIndex: number;
  readonly baselineActionIndex: number;
  readonly scoreSemantics: typeof TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS;
  readonly minimumChipAdvantage: number;
  readonly returnCount: 1 | 2;
  readonly routing: "learnedAction" | "safetyFallback";
  readonly fallback: boolean;
  readonly fallbackReason:
    | "return-count-one-exact-normal"
    | "no-unanimous-positive-action"
    | "lower-confidence-bound-not-above-threshold"
    | null;
  readonly selectedScore: TaxReturnAdvantageActionScore | null;
  readonly actionScores: readonly TaxReturnAdvantageActionScore[];
};

const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function hasExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}

function requireObject(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireExactObject(
  value: unknown,
  expected: readonly string[],
  label: string,
): Record<string, unknown> {
  const result = requireObject(value, label);
  if (!hasExactKeys(result, expected)) {
    throw new TypeError(`${label} fields mismatch`);
  }
  return result;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number`);
  }
  return value;
}

function nonNegativeNumber(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (result < 0) throw new RangeError(`${label} must be non-negative`);
  return result;
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value as number;
}

function nonNegativeInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new RangeError(`${label} must be a non-negative integer`);
  }
  return value as number;
}

function finiteArray(
  value: unknown,
  length: number,
  label: string,
): number[] {
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    value.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))
  ) {
    throw new TypeError(`${label} must contain ${length} finite numbers`);
  }
  return value as number[];
}

function sameStrings(value: unknown, expected: readonly string[]): boolean {
  return (
    Array.isArray(value) &&
    value.length === expected.length &&
    value.every((entry, index) => entry === expected[index])
  );
}

export function isTaxReturnAdvantageEnsembleArtifact(
  value: unknown,
): boolean {
  return (
    Boolean(value) &&
    typeof value === "object" &&
    (value as { format?: unknown }).format ===
      TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT
  );
}

export function parseTaxReturnAdvantageEnsemble(
  value: unknown,
): TaxReturnAdvantageEnsembleModel {
  const candidate = requireExactObject(
    value,
    [
      "format",
      "version",
      "decisionKind",
      "scoreSemantics",
      "observationSchemaVersion",
      "observationFeatures",
      "actionCatalogueVersion",
      "actionCount",
      "actionFeatures",
      "actionFeatureLayout",
      "trainingData",
      "architecture",
      "baseline",
      "objective",
      "routing",
      "members",
    ],
    "tax-return advantage ensemble",
  );
  if (
    candidate.format !== TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT ||
    candidate.version !== TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION ||
    candidate.decisionKind !== "tax-return" ||
    candidate.scoreSemantics !== TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS
  ) {
    throw new TypeError("unsupported tax-return advantage ensemble format");
  }
  if (
    candidate.observationSchemaVersion !==
      NON_CARD_OBSERVATION_SCHEMA_VERSION ||
    candidate.observationFeatures !== TAX_RETURN_OBSERVATION_FEATURE_COUNT ||
    candidate.actionCatalogueVersion !== TAX_RETURN_ACTION_CATALOGUE_VERSION ||
    candidate.actionCount !== TAX_RETURN_ACTION_COUNT ||
    candidate.actionFeatures !== TAX_RETURN_ACTION_FEATURE_COUNT ||
    !sameStrings(candidate.actionFeatureLayout, TAX_RETURN_ACTION_FEATURE_LAYOUT)
  ) {
    throw new TypeError("tax-return advantage feature contract mismatch");
  }
  const trainingData = requireExactObject(
    candidate.trainingData,
    [
      "sourceFormatVersions",
      "groupSplitKey",
      "determinizationSchema",
      "worldCountPerInformationState",
      "continuationCountPerHiddenWorld",
      "effectiveIndependentWorldsPerInformationState",
      "rawContinuationEvaluationsPerInformationState",
      "standardErrorEstimable",
      "determinizationAlgorithm",
      "determinizationAlgorithmVersion",
      "determinizationAlgorithmContractSha256",
      "candidateSeedDerivation",
      "continuationSeedDerivation",
      "targetField",
      "targetTransform",
      "stateWeighting",
    ],
    "tax-return training data",
  );
  if (
    !Array.isArray(trainingData.sourceFormatVersions) ||
    trainingData.sourceFormatVersions.length !== 1 ||
    (trainingData.sourceFormatVersions[0] !== 1 &&
      trainingData.sourceFormatVersions[0] !== 2)
  ) {
    throw new TypeError("tax-return source format versions must be [1] or [2]");
  }
  const sourceVersion = trainingData.sourceFormatVersions[0];
  const expectedGroupKey =
    sourceVersion === 1
      ? "canonicalWorldKey"
      : "canonicalInformationStateKey";
  const expectedDeterminizationSchema =
    sourceVersion === 1
      ? null
      : "world-clustered-paired-baseline-advantages-v2";
  if (
    trainingData.groupSplitKey !== expectedGroupKey ||
    trainingData.determinizationSchema !== expectedDeterminizationSchema ||
    trainingData.stateWeighting !==
      "one-per-information-state-independent-of-worldCount"
  ) {
    throw new TypeError("tax-return training-data provenance mismatch");
  }
  const expectedDeterminizationContract =
    sourceVersion === 1
      ? {
          determinizationAlgorithm: null,
          determinizationAlgorithmVersion: null,
          determinizationAlgorithmContractSha256: null,
          candidateSeedDerivation: null,
          continuationSeedDerivation: null,
        }
      : {
          determinizationAlgorithm: TAX_RETURN_DETERMINIZATION_ALGORITHM,
          determinizationAlgorithmVersion:
            TAX_RETURN_DETERMINIZATION_ALGORITHM_VERSION,
          determinizationAlgorithmContractSha256:
            TAX_RETURN_DETERMINIZATION_ALGORITHM_CONTRACT_SHA256,
          candidateSeedDerivation:
            TAX_RETURN_DETERMINIZATION_CANDIDATE_SEED_DERIVATION,
          continuationSeedDerivation:
            TAX_RETURN_DETERMINIZATION_CONTINUATION_SEED_DERIVATION,
        };
  if (
    Object.entries(expectedDeterminizationContract).some(
      ([key, expected]) => trainingData[key] !== expected,
    )
  ) {
    throw new TypeError(
      "tax-return determinization algorithm contract mismatch",
    );
  }
  const worldCount = positiveInteger(
    trainingData.worldCountPerInformationState,
    "worldCountPerInformationState",
  );
  const continuationCount = positiveInteger(
    trainingData.continuationCountPerHiddenWorld,
    "continuationCountPerHiddenWorld",
  );
  const effectiveWorlds = positiveInteger(
    trainingData.effectiveIndependentWorldsPerInformationState,
    "effectiveIndependentWorldsPerInformationState",
  );
  const rawContinuationEvaluations = positiveInteger(
    trainingData.rawContinuationEvaluationsPerInformationState,
    "rawContinuationEvaluationsPerInformationState",
  );
  if (
    effectiveWorlds !== worldCount ||
    rawContinuationEvaluations !== worldCount * continuationCount ||
    (sourceVersion === 1 &&
      (worldCount !== 1 ||
        continuationCount !== 1 ||
        effectiveWorlds !== 1 ||
        rawContinuationEvaluations !== 1))
  ) {
    throw new TypeError("tax-return hidden-world/continuation binding mismatch");
  }
  if (trainingData.standardErrorEstimable !== (worldCount > 1)) {
    throw new TypeError("tax-return standard-error estimability mismatch");
  }
  const expectedTargetField =
    sourceVersion === 1
      ? "actions[].decisionActUtility-minus-baseline.decisionActUtility"
      : "actions[].pairedDecisionActBaselineAdvantage.mean";
  if (trainingData.targetField !== expectedTargetField) {
    throw new TypeError("tax-return training target field mismatch");
  }
  const targetTransform = requireExactObject(
    trainingData.targetTransform,
    ["scoreUnit", "sourceUnit", "operation", "multiplier"],
    "tax-return target transform",
  );
  if (
    targetTransform.scoreUnit !== "chip-units" ||
    targetTransform.sourceUnit !== "(roundChipAward-2)/2" ||
    targetTransform.operation !==
      "multiply-source-baseline-advantage-by-2" ||
    targetTransform.multiplier !== 2
  ) {
    throw new TypeError("tax-return target unit transform mismatch");
  }
  const architecture = requireExactObject(
    candidate.architecture,
    ["contextFeatures", "contextActivation", "score", "weightLayout"],
    "tax-return advantage architecture",
  );
  const contextFeatures = positiveInteger(
    architecture.contextFeatures,
    "contextFeatures",
  );
  if (
    architecture.contextActivation !== "tanh" ||
    architecture.score !== "raw(s,a)-raw(s,normalBaselineAction)" ||
    architecture.weightLayout !== TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT
  ) {
    throw new TypeError("unsupported tax-return advantage architecture");
  }
  const baseline = requireExactObject(
    candidate.baseline,
    ["provenance", "provenanceSha256", "score"],
    "tax-return normal baseline",
  );
  const provenance = requireExactObject(
    baseline.provenance,
    ["implementation", "semanticEncoding", "difficulty"],
    "tax-return normal baseline provenance",
  );
  if (
    provenance.implementation !==
      TAX_RETURN_NORMAL_BASELINE_PROVENANCE.implementation ||
    provenance.semanticEncoding !==
      TAX_RETURN_NORMAL_BASELINE_PROVENANCE.semanticEncoding ||
    provenance.difficulty !== TAX_RETURN_NORMAL_BASELINE_PROVENANCE.difficulty ||
    baseline.provenanceSha256 !==
      TAX_RETURN_NORMAL_BASELINE_PROVENANCE_SHA256 ||
    baseline.score !== "exactly-zero-by-residualization"
  ) {
    throw new TypeError("tax-return normal-baseline provenance mismatch");
  }
  const objective = requireExactObject(
    candidate.objective,
    [
      "utilityTarget",
      "utilityScale",
      "weighting",
      "regression",
      "tieAwareSign",
      "checkpointSelection",
      "bootstrapUnit",
    ],
    "tax-return objective",
  );
  if (
    objective.utilityTarget !== "decision-act-current-chip-advantage" ||
    objective.utilityScale !== "chip-units" ||
    objective.weighting !== "equal-per-state" ||
    objective.checkpointSelection !== "paired-validation-loss" ||
    objective.bootstrapUnit !== expectedGroupKey
  ) {
    throw new TypeError("tax-return objective contract mismatch");
  }
  const regression = requireExactObject(
    objective.regression,
    ["loss", "coefficient", "deltaChips"],
    "tax-return regression objective",
  );
  if (
    regression.loss !== "huber-paired-action-vs-baseline" ||
    finiteNumber(regression.coefficient, "regression coefficient") <= 0 ||
    finiteNumber(regression.deltaChips, "Huber delta") <= 0
  ) {
    throw new TypeError("tax-return regression objective mismatch");
  }
  const tieAwareSign = requireExactObject(
    objective.tieAwareSign,
    [
      "loss",
      "coefficient",
      "temperatureChips",
      "tieTarget",
      "tieEpsilonChips",
    ],
    "tax-return tie-aware sign objective",
  );
  if (
    tieAwareSign.loss !== "binary-cross-entropy-with-logits" ||
    nonNegativeNumber(tieAwareSign.coefficient, "sign coefficient") < 0 ||
    finiteNumber(tieAwareSign.temperatureChips, "sign temperature") <= 0 ||
    tieAwareSign.tieTarget !== 0.5 ||
    nonNegativeNumber(tieAwareSign.tieEpsilonChips, "tie epsilon") < 0
  ) {
    throw new TypeError("tax-return tie-aware sign objective mismatch");
  }
  const routing = requireExactObject(
    candidate.routing,
    [
      "returnCountOne",
      "returnCountTwo",
      "roleRouting",
      "memberCount",
      "unanimityRule",
      "lowerConfidenceBound",
      "zValue",
      "defaultMinimumChipAdvantage",
      "selection",
      "tieBreak",
    ],
    "tax-return routing",
  );
  if (
    routing.returnCountOne !== "exact-normal-fallback" ||
    routing.returnCountTwo !== "ensemble-lower-confidence-bound" ||
    routing.memberCount !== TAX_RETURN_ADVANTAGE_ENSEMBLE_MEMBER_COUNT ||
    routing.unanimityRule !== "all-member-advantages-strictly-positive" ||
    routing.lowerConfidenceBound !== "mean-minus-z-times-sample-sd" ||
    routing.zValue !== TAX_RETURN_ADVANTAGE_ENSEMBLE_Z_VALUE ||
    routing.selection !== "maximum-eligible-lcb" ||
    routing.tieBreak !== "baseline-then-lowest-action-index"
  ) {
    throw new TypeError("unsupported tax-return advantage routing contract");
  }
  const roleRouting = requireExactObject(
    routing.roleRouting,
    ["great-dalmuti", "lesser-dalmuti", "other-roles"],
    "tax-return role routing",
  );
  if (
    roleRouting["great-dalmuti"] !==
      "ensemble-lower-confidence-bound" ||
    roleRouting["lesser-dalmuti"] !== "exact-normal-fallback" ||
    roleRouting["other-roles"] !== "not-applicable"
  ) {
    throw new TypeError("unsupported tax-return role routing");
  }
  nonNegativeNumber(
    routing.defaultMinimumChipAdvantage,
    "defaultMinimumChipAdvantage",
  );
  if (
    !Array.isArray(candidate.members) ||
    candidate.members.length !== TAX_RETURN_ADVANTAGE_ENSEMBLE_MEMBER_COUNT
  ) {
    throw new TypeError("tax-return advantage ensemble requires five members");
  }
  const seeds = new Set<number>();
  candidate.members.forEach((rawMember, index) => {
    const member = requireExactObject(
      rawMember,
      [
        "memberIndex",
        "seed",
        "checkpointEpoch",
        "validationPairedLoss",
        "parametersSha256",
        "contextLayer",
        "bilinearWeight",
      ],
      `tax-return member ${index}`,
    );
    if (member.memberIndex !== index) {
      throw new TypeError("tax-return member indices must be canonical");
    }
    const seed = nonNegativeInteger(member.seed, `member ${index} seed`);
    if (seeds.has(seed)) {
      throw new TypeError("tax-return member seeds must be distinct");
    }
    seeds.add(seed);
    positiveInteger(member.checkpointEpoch, `member ${index} checkpointEpoch`);
    nonNegativeNumber(
      member.validationPairedLoss,
      `member ${index} validationPairedLoss`,
    );
    if (
      typeof member.parametersSha256 !== "string" ||
      !SHA256_PATTERN.test(member.parametersSha256)
    ) {
      throw new TypeError(`member ${index} parametersSha256 is invalid`);
    }
    const layer = requireExactObject(
      member.contextLayer,
      ["inFeatures", "outFeatures", "weight", "bias"],
      `member ${index} contextLayer`,
    );
    if (
      layer.inFeatures !== TAX_RETURN_OBSERVATION_FEATURE_COUNT ||
      layer.outFeatures !== contextFeatures
    ) {
      throw new TypeError(`member ${index} context layer dimensions mismatch`);
    }
    finiteArray(
      layer.weight,
      contextFeatures * TAX_RETURN_OBSERVATION_FEATURE_COUNT,
      `member ${index} context weight`,
    );
    finiteArray(layer.bias, contextFeatures, `member ${index} context bias`);
    finiteArray(
      member.bilinearWeight,
      contextFeatures * TAX_RETURN_ACTION_FEATURE_COUNT,
      `member ${index} bilinear weight`,
    );
  });
  return candidate as unknown as TaxReturnAdvantageEnsembleModel;
}

function rawMemberScores(
  member: TaxReturnAdvantageMember,
  contextFeatures: number,
  observation: readonly number[],
  actionIndices: readonly number[],
): Float64Array {
  const context = new Float64Array(contextFeatures);
  for (let row = 0; row < contextFeatures; row += 1) {
    let value = member.contextLayer.bias[row];
    const offset = row * TAX_RETURN_OBSERVATION_FEATURE_COUNT;
    for (
      let column = 0;
      column < TAX_RETURN_OBSERVATION_FEATURE_COUNT;
      column += 1
    ) {
      value += member.contextLayer.weight[offset + column] * observation[column];
    }
    context[row] = Math.tanh(value);
  }
  const scores = new Float64Array(actionIndices.length);
  actionIndices.forEach((actionIndex, outputIndex) => {
    const action = TAX_RETURN_ACTION_FEATURES[actionIndex];
    let score = 0;
    for (let contextIndex = 0; contextIndex < contextFeatures; contextIndex += 1) {
      let projectedAction = 0;
      const offset = contextIndex * TAX_RETURN_ACTION_FEATURE_COUNT;
      for (
        let featureIndex = 0;
        featureIndex < TAX_RETURN_ACTION_FEATURE_COUNT;
        featureIndex += 1
      ) {
        projectedAction +=
          member.bilinearWeight[offset + featureIndex] * action[featureIndex];
      }
      score += context[contextIndex] * projectedAction;
    }
    if (!Number.isFinite(score)) {
      throw new RangeError("tax-return ensemble produced a non-finite score");
    }
    scores[outputIndex] = score;
  });
  return scores;
}

function meanAndSampleStandardDeviation(values: readonly number[]): {
  mean: number;
  sampleStandardDeviation: number;
} {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const sumSquares = values.reduce(
    (total, value) => total + (value - mean) ** 2,
    0,
  );
  return {
    mean: Object.is(mean, -0) ? 0 : mean,
    sampleStandardDeviation: Math.sqrt(sumSquares / (values.length - 1)),
  };
}

export function selectTaxReturnAdvantageEnsembleAction(
  modelValue: unknown,
  observation: TaxReturnObservation,
  baselineActionIndex: number,
  minimumChipAdvantage?: number,
): TaxReturnAdvantageEnsembleDecision {
  const model = parseTaxReturnAdvantageEnsemble(modelValue);
  const threshold = nonNegativeNumber(
    minimumChipAdvantage ?? model.routing.defaultMinimumChipAdvantage,
    "minimumChipAdvantage",
  );
  const encodedObservation = encodeTaxReturnObservation(observation);
  const legalActionIndices = legalTaxReturnActionIndices(observation);
  if (!legalActionIndices.includes(baselineActionIndex)) {
    throw new RangeError("normal baseline tax-return action must be legal");
  }
  if (observation.returnCount === 1) {
    return {
      actionIndex: baselineActionIndex,
      modelActionIndex: baselineActionIndex,
      baselineActionIndex,
      scoreSemantics: TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
      minimumChipAdvantage: threshold,
      returnCount: 1,
      routing: "safetyFallback",
      fallback: true,
      fallbackReason: "return-count-one-exact-normal",
      selectedScore: null,
      actionScores: [],
    };
  }
  const memberScores = model.members.map((member) =>
    rawMemberScores(
      member,
      model.architecture.contextFeatures,
      encodedObservation,
      legalActionIndices,
    ),
  );
  const baselineOffset = legalActionIndices.indexOf(baselineActionIndex);
  const actionScores = legalActionIndices.map((actionIndex, actionOffset) => {
    const advantages = memberScores.map((scores) =>
      actionIndex === baselineActionIndex
        ? 0
        : scores[actionOffset] - scores[baselineOffset],
    );
    const { mean, sampleStandardDeviation } =
      meanAndSampleStandardDeviation(advantages);
    const lowerConfidenceBound =
      mean - model.routing.zValue * sampleStandardDeviation;
    const unanimousPositive = advantages.every((value) => value > 0);
    return {
      actionIndex,
      memberAdvantages: advantages,
      meanAdvantage: mean,
      sampleStandardDeviation,
      lowerConfidenceBound: Object.is(lowerConfidenceBound, -0)
        ? 0
        : lowerConfidenceBound,
      unanimousPositive,
      eligible:
        actionIndex !== baselineActionIndex &&
        unanimousPositive &&
        lowerConfidenceBound > threshold,
    };
  });
  const nonBaselineScores = actionScores.filter(
    (score) => score.actionIndex !== baselineActionIndex,
  );
  const diagnosticBest = nonBaselineScores.reduce<
    TaxReturnAdvantageActionScore | null
  >((best, score) => {
    if (
      best === null ||
      score.lowerConfidenceBound > best.lowerConfidenceBound ||
      (score.lowerConfidenceBound === best.lowerConfidenceBound &&
        score.actionIndex < best.actionIndex)
    ) {
      return score;
    }
    return best;
  }, null);
  const selected = nonBaselineScores
    .filter((score) => score.eligible)
    .reduce<TaxReturnAdvantageActionScore | null>((best, score) => {
      if (
        best === null ||
        score.lowerConfidenceBound > best.lowerConfidenceBound ||
        (score.lowerConfidenceBound === best.lowerConfidenceBound &&
          score.actionIndex < best.actionIndex)
      ) {
        return score;
      }
      return best;
    }, null);
  if (selected) {
    return {
      actionIndex: selected.actionIndex,
      modelActionIndex: selected.actionIndex,
      baselineActionIndex,
      scoreSemantics: TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
      minimumChipAdvantage: threshold,
      returnCount: 2,
      routing: "learnedAction",
      fallback: false,
      fallbackReason: null,
      selectedScore: selected,
      actionScores,
    };
  }
  const anyUnanimous = nonBaselineScores.some(
    (score) => score.unanimousPositive,
  );
  return {
    actionIndex: baselineActionIndex,
    modelActionIndex: diagnosticBest?.actionIndex ?? baselineActionIndex,
    baselineActionIndex,
    scoreSemantics: TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
    minimumChipAdvantage: threshold,
    returnCount: 2,
    routing: "safetyFallback",
    fallback: true,
    fallbackReason: anyUnanimous
      ? "lower-confidence-bound-not-above-threshold"
      : "no-unanimous-positive-action",
    selectedScore: diagnosticBest,
    actionScores,
  };
}
