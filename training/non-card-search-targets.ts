import {
  enumerateLegalRevolutionActions,
  enumerateLegalTaxReturnActions,
  legalRevolutionActionMask,
  legalTaxReturnActionMask,
  type RevolutionActionCandidate,
  type TaxReturnActionCandidate,
} from "./non-card-action-space.ts";
import {
  NON_CARD_OBSERVATION_SCHEMA_VERSION,
  encodeRevolutionObservation,
  encodeTaxReturnObservation,
  type RevolutionObservation,
  type TaxReturnObservation,
} from "./non-card-observation.ts";

export type NonCardDecisionKind = "tax-return" | "revolution";

/**
 * A scorer receives only model-ready observation/action features. It never
 * receives the authoritative room, opponent hands, or private tax routes.
 */
export type NonCardActionScoreInput = {
  readonly decision: NonCardDecisionKind;
  readonly observationSchemaVersion: number;
  readonly actionIndex: number;
  readonly observationFeatures: readonly number[];
  readonly actionFeatures: readonly number[];
};

export type NonCardActionScorer = (
  input: NonCardActionScoreInput,
) => number;

export type ScoredAction = {
  readonly actionIndex: number;
  readonly score: number;
};

export type ExactTaxReturnSelection = {
  readonly candidate: TaxReturnActionCandidate;
  readonly scores: readonly ScoredAction[];
};

export type ExactRevolutionSelection = {
  readonly candidate: RevolutionActionCandidate;
  readonly scores: readonly ScoredAction[];
};

function scoreCandidates<T extends { actionIndex: number; actionFeatures: readonly number[] }>(
  decision: NonCardDecisionKind,
  observationFeatures: readonly number[],
  candidates: readonly T[],
  scorer: NonCardActionScorer,
): { readonly candidate: T; readonly scores: readonly ScoredAction[] } {
  if (typeof scorer !== "function") {
    throw new TypeError("non-card action scorer must be a function");
  }
  if (candidates.length === 0) {
    throw new Error("exact action enumeration produced no legal candidate");
  }
  const immutableObservation = Object.freeze([...observationFeatures]);
  let selectedCandidate = candidates[0];
  let selectedScore = Number.NEGATIVE_INFINITY;
  const scores = candidates.map((candidate) => {
    const score = scorer(
      Object.freeze({
        decision,
        observationSchemaVersion: NON_CARD_OBSERVATION_SCHEMA_VERSION,
        actionIndex: candidate.actionIndex,
        observationFeatures: immutableObservation,
        actionFeatures: candidate.actionFeatures,
      }),
    );
    if (!Number.isFinite(score)) {
      throw new RangeError(
        `non-card scorer returned a non-finite score for action ${candidate.actionIndex}`,
      );
    }
    // Candidates are in ascending catalogue order, making equal-score
    // selection reproducible without depending on object or card ID order.
    if (score > selectedScore) {
      selectedCandidate = candidate;
      selectedScore = score;
    }
    return Object.freeze({ actionIndex: candidate.actionIndex, score });
  });
  return Object.freeze({
    candidate: selectedCandidate,
    scores: Object.freeze(scores),
  });
}

/** Exhaustively scores every legal semantic return action. */
export function selectExactTaxReturnAction(
  observation: TaxReturnObservation,
  scorer: NonCardActionScorer,
): ExactTaxReturnSelection {
  const result = scoreCandidates(
    "tax-return",
    encodeTaxReturnObservation(observation),
    enumerateLegalTaxReturnActions(observation),
    scorer,
  );
  return result;
}

/** Exhaustively scores decline and declare. */
export function selectExactRevolutionAction(
  observation: RevolutionObservation,
  scorer: NonCardActionScorer,
): ExactRevolutionSelection {
  const result = scoreCandidates(
    "revolution",
    encodeRevolutionObservation(observation),
    enumerateLegalRevolutionActions(observation),
    scorer,
  );
  return result;
}

export type PairedActionUtility = {
  readonly actionIndex: number;
  /** Caller-defined terminal utility, normally the actor's chip outcome. */
  readonly utility: number;
};

/**
 * One determinized hidden world and continuation random seed. Every legal
 * action must be forced once in the same batch, providing common-random-number
 * counterfactuals without putting hidden cards into model features.
 */
export type PairedCounterfactualBatch = {
  readonly sampleId: string;
  readonly outcomes: readonly PairedActionUtility[];
};

export type CounterfactualTargetOptions = {
  readonly policyTemperature?: number;
};

export type ActionValueTarget = {
  readonly actionIndex: number;
  readonly meanUtility: number;
  readonly centeredUtility: number;
  readonly sampleStandardDeviation: number;
  readonly standardError: number;
  readonly policyProbability: number;
};

export type PairedCounterfactualTargets = {
  readonly sampleCount: number;
  readonly bestActionIndex: number;
  readonly actions: readonly ActionValueTarget[];
};

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

/**
 * Aggregates complete paired counterfactual batches into action-value and
 * soft policy targets. Full hidden-state game-tree search is intentionally not
 * attempted; the simulator supplies sampled worlds while this function
 * enforces unbiased, same-world action coverage.
 */
export function buildPairedCounterfactualTargets(
  legalMask: readonly boolean[],
  batches: readonly PairedCounterfactualBatch[],
  options: CounterfactualTargetOptions = {},
): PairedCounterfactualTargets {
  if (!Array.isArray(legalMask) || legalMask.length === 0) {
    throw new TypeError("legalMask must be a non-empty boolean array");
  }
  if (legalMask.some((legal) => typeof legal !== "boolean")) {
    throw new TypeError("legalMask entries must be booleans");
  }
  const legalActionIndices = legalMask.flatMap((legal, actionIndex) =>
    legal ? [actionIndex] : [],
  );
  if (legalActionIndices.length === 0) {
    throw new RangeError("legalMask must contain at least one legal action");
  }
  if (!Array.isArray(batches) || batches.length === 0) {
    throw new RangeError("at least one counterfactual batch is required");
  }
  const temperature = options.policyTemperature ?? 1;
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("policyTemperature must be finite and greater than zero");
  }

  const utilitiesByAction = new Map<number, number[]>(
    legalActionIndices.map((actionIndex) => [actionIndex, []]),
  );
  const sampleIds = new Set<string>();
  for (const batch of batches) {
    if (typeof batch.sampleId !== "string" || batch.sampleId.length === 0) {
      throw new TypeError("counterfactual sampleId must be non-empty");
    }
    if (sampleIds.has(batch.sampleId)) {
      throw new TypeError(`duplicate counterfactual sampleId: ${batch.sampleId}`);
    }
    sampleIds.add(batch.sampleId);
    if (
      !Array.isArray(batch.outcomes) ||
      batch.outcomes.length !== legalActionIndices.length
    ) {
      throw new TypeError(
        `sample ${batch.sampleId} must cover every legal action exactly once`,
      );
    }
    const seenActions = new Set<number>();
    for (const outcome of batch.outcomes) {
      if (
        !Number.isInteger(outcome.actionIndex) ||
        outcome.actionIndex < 0 ||
        outcome.actionIndex >= legalMask.length ||
        !legalMask[outcome.actionIndex]
      ) {
        throw new RangeError(
          `sample ${batch.sampleId} contains illegal action ${outcome.actionIndex}`,
        );
      }
      if (seenActions.has(outcome.actionIndex)) {
        throw new TypeError(
          `sample ${batch.sampleId} repeats action ${outcome.actionIndex}`,
        );
      }
      if (!Number.isFinite(outcome.utility)) {
        throw new RangeError(
          `sample ${batch.sampleId} has non-finite utility for action ${outcome.actionIndex}`,
        );
      }
      seenActions.add(outcome.actionIndex);
      utilitiesByAction.get(outcome.actionIndex)?.push(outcome.utility);
    }
  }

  const means = legalActionIndices.map((actionIndex) =>
    mean(utilitiesByAction.get(actionIndex) ?? []),
  );
  const center = mean(means);
  const maximumMean = Math.max(...means);
  const exponentials = means.map((value) =>
    Math.exp((value - maximumMean) / temperature),
  );
  const exponentialTotal = exponentials.reduce(
    (total, value) => total + value,
    0,
  );

  const actions = legalActionIndices.map((actionIndex, position) => {
    const values = utilitiesByAction.get(actionIndex) ?? [];
    const actionMean = means[position];
    const variance =
      values.length > 1
        ? values.reduce(
            (total, value) => total + (value - actionMean) ** 2,
            0,
          ) /
          (values.length - 1)
        : 0;
    const sampleStandardDeviation = Math.sqrt(variance);
    return Object.freeze({
      actionIndex,
      meanUtility: actionMean,
      centeredUtility: actionMean - center,
      sampleStandardDeviation,
      standardError: sampleStandardDeviation / Math.sqrt(values.length),
      policyProbability: exponentials[position] / exponentialTotal,
    });
  });

  let bestActionIndex = actions[0].actionIndex;
  let bestMean = actions[0].meanUtility;
  for (const action of actions.slice(1)) {
    if (action.meanUtility > bestMean) {
      bestActionIndex = action.actionIndex;
      bestMean = action.meanUtility;
    }
  }
  return Object.freeze({
    sampleCount: batches.length,
    bestActionIndex,
    actions: Object.freeze(actions),
  });
}

export function buildTaxReturnCounterfactualTargets(
  observation: TaxReturnObservation,
  batches: readonly PairedCounterfactualBatch[],
  options: CounterfactualTargetOptions = {},
): PairedCounterfactualTargets {
  return buildPairedCounterfactualTargets(
    legalTaxReturnActionMask(observation),
    batches,
    options,
  );
}

export function buildRevolutionCounterfactualTargets(
  observation: RevolutionObservation,
  batches: readonly PairedCounterfactualBatch[],
  options: CounterfactualTargetOptions = {},
): PairedCounterfactualTargets {
  return buildPairedCounterfactualTargets(
    legalRevolutionActionMask(observation),
    batches,
    options,
  );
}
