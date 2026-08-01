import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import {
  chooseBotRevolution,
  chooseBotTaxReturn,
} from "../../lib/bot-strategy.ts";
import {
  REVOLUTION_DECLARE_ACTION_INDEX,
  REVOLUTION_DECLINE_ACTION_INDEX,
  encodeTaxReturnAction,
} from "../../training/non-card-action-space.ts";
import {
  createRevolutionModelTrainingPolicy,
  createTaxReturnModelTrainingPolicy,
  evaluateRevolutionActionConditionedActorCritic,
  evaluateTaxReturnActionConditionedActorCritic,
  parseRevolutionActionConditionedActorCriticModel,
  parseTaxReturnActionConditionedActorCriticModel,
  selectBaselineGatedNonCardAction,
} from "../../training/non-card-action-conditioned-model.ts";
import {
  TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
  isTaxReturnAdvantageEnsembleArtifact,
  parseTaxReturnAdvantageEnsemble,
  selectTaxReturnAdvantageEnsembleAction,
} from "../../training/tax-return-advantage-ensemble.ts";

const MARGIN_ACCUMULATORS = Symbol("non-card benchmark margin accumulators");
const ENSEMBLE_ACCUMULATORS = Symbol(
  "tax-return advantage ensemble accumulators",
);

function taxAdvantageMemberParametersSha256(member) {
  const arrays = [
    member.contextLayer.weight,
    member.contextLayer.bias,
    member.bilinearWeight,
  ];
  const dimensions = Buffer.alloc(20);
  dimensions.writeUInt32LE(member.contextLayer.inFeatures, 0);
  dimensions.writeUInt32LE(member.contextLayer.outFeatures, 4);
  dimensions.writeUInt32LE(arrays[0].length, 8);
  dimensions.writeUInt32LE(arrays[1].length, 12);
  dimensions.writeUInt32LE(arrays[2].length, 16);
  const digest = createHash("sha256")
    .update("dalmuti-tax-return-bilinear-residual-member-v1\0")
    .update(dimensions);
  for (const values of arrays) {
    const bytes = Buffer.alloc(values.length * 8);
    values.forEach((value, index) => bytes.writeDoubleLE(value, index * 8));
    digest.update(bytes);
  }
  return digest.digest("hex");
}

export const NON_CARD_BASELINE_PROVENANCE = Object.freeze({
  taxReturn: Object.freeze({
    implementation: "lib/bot-strategy.ts#chooseBotTaxReturn",
    semanticEncoding:
      "training/non-card-action-space.ts#encodeTaxReturnAction",
    difficulty: "normal",
  }),
  revolution: Object.freeze({
    implementation: "lib/bot-strategy.ts#chooseBotRevolution",
    semanticEncoding:
      "training/non-card-action-space.ts revolution action indices",
    difficulty: "normal",
  }),
});

function cardRanksForIds(hand, cardIds) {
  const cardsById = new Map(hand.map((card) => [card.id, card]));
  return cardIds.map((cardId) => {
    const card = cardsById.get(cardId);
    if (!card) {
      throw new Error(`normal tax heuristic selected unknown card ${cardId}`);
    }
    return card.rank;
  });
}

function normalTaxReturnDecision(context) {
  const selected = chooseBotTaxReturn(
    context.observation.hand,
    context.observation.returnCount,
    "normal",
  );
  return encodeTaxReturnAction(
    cardRanksForIds(context.observation.hand, selected.cardIds),
  );
}

function normalRevolutionDecision(context) {
  const selected = chooseBotRevolution(
    {
      hand: context.observation.hand,
      role: context.actorRole,
      playerCount: context.observation.players.length,
    },
    "normal",
  );
  return selected.declare
    ? REVOLUTION_DECLARE_ACTION_INDEX
    : REVOLUTION_DECLINE_ACTION_INDEX;
}

async function readModelArtifact(pathValue, parseModel, createPolicy) {
  const path = resolve(pathValue);
  const bytes = await readFile(path);
  let source;
  try {
    source = JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new TypeError(
      `non-card model is not valid JSON: ${path}`,
      { cause: error },
    );
  }
  const model = parseModel(source);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const policyVersion = `benchmark-${sha256.slice(0, 12)}`;
  return {
    modelKind: "action-conditioned-actor-critic",
    model,
    policy: createPolicy(model, policyVersion),
    policyVersion,
    metadata: {
      path,
      sha256,
      format: model.format,
      version: model.version,
      decisionKind: model.decisionKind,
      observationSchemaVersion: model.observationSchemaVersion,
      actionCatalogueVersion: model.actionCatalogueVersion,
      policyVersion,
    },
  };
}

export async function loadTaxReturnBenchmarkModel(pathValue) {
  const path = resolve(pathValue);
  const bytes = await readFile(path);
  let source;
  try {
    source = JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new TypeError(`non-card model is not valid JSON: ${path}`, {
      cause: error,
    });
  }
  if (!isTaxReturnAdvantageEnsembleArtifact(source)) {
    return readModelArtifact(
      pathValue,
      parseTaxReturnActionConditionedActorCriticModel,
      createTaxReturnModelTrainingPolicy,
    );
  }
  const model = parseTaxReturnAdvantageEnsemble(source);
  for (const member of model.members) {
    if (
      taxAdvantageMemberParametersSha256(member) !==
      member.parametersSha256
    ) {
      throw new TypeError(
        `tax-return advantage member ${member.memberIndex} parameter hash mismatch`,
      );
    }
  }
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const policyVersion = `benchmark-${sha256.slice(0, 12)}`;
  return {
    modelKind: "baseline-advantage-ensemble",
    model,
    policy: null,
    policyVersion,
    defaultMinimumAdvantage:
      model.routing.defaultMinimumChipAdvantage,
    metadata: {
      path,
      sha256,
      format: model.format,
      version: model.version,
      decisionKind: model.decisionKind,
      observationSchemaVersion: model.observationSchemaVersion,
      actionCatalogueVersion: model.actionCatalogueVersion,
      policyVersion,
      scoreSemantics: model.scoreSemantics,
      memberCount: model.members.length,
      memberSeeds: model.members.map((member) => member.seed),
      memberParameterHashes: model.members.map(
        (member) => member.parametersSha256,
      ),
      trainingData: model.trainingData,
      routing: model.routing,
    },
  };
}

export async function loadRevolutionBenchmarkModel(pathValue) {
  return readModelArtifact(
    pathValue,
    parseRevolutionActionConditionedActorCriticModel,
    createRevolutionModelTrainingPolicy,
  );
}

/**
 * Candidate seats receive the optional learned policy. Every other actor is
 * explicitly routed through the exact current `normal` heuristic. This keeps
 * tax-only, revolution-only, and combined ablations comparable without ever
 * giving the normal control group a candidate non-card model.
 */
export function createCandidateOnlyNonCardHooks({
  candidateIds,
  taxReturn,
  revolution,
  taxMinAdvantage,
  revolutionMinAdvantage = 0,
  decisionTelemetry = [],
}) {
  if (!(candidateIds instanceof Set)) {
    throw new TypeError("candidateIds must be a Set");
  }
  const resolvedTaxMinAdvantage =
    taxMinAdvantage ?? taxReturn?.defaultMinimumAdvantage ?? 0;
  validateMinimumAdvantage(
    resolvedTaxMinAdvantage,
    "taxMinAdvantage",
  );
  validateMinimumAdvantage(
    revolutionMinAdvantage,
    "revolutionMinAdvantage",
  );
  if (!Array.isArray(decisionTelemetry)) {
    throw new TypeError("decisionTelemetry must be an array");
  }
  if (!taxReturn && resolvedTaxMinAdvantage !== 0) {
    throw new TypeError("taxMinAdvantage requires a tax-return model");
  }
  if (!revolution && revolutionMinAdvantage !== 0) {
    throw new TypeError("revolutionMinAdvantage requires a revolution model");
  }
  if (!taxReturn && !revolution) return undefined;
  return {
    ...(taxReturn
      ? {
          taxReturnPolicy(context) {
            if (!candidateIds.has(context.actorId)) {
              return normalTaxReturnDecision(context);
            }
            const baselineActionIndex = normalTaxReturnDecision(context);
            if (taxReturn.modelKind === "baseline-advantage-ensemble") {
              return taxAdvantageEnsembleCandidateDecision({
                context,
                benchmarkModel: taxReturn,
                baselineActionIndex,
                minimumAdvantage: resolvedTaxMinAdvantage,
                decisionTelemetry,
              });
            }
            return gatedCandidateDecision({
              context,
              benchmarkModel: taxReturn,
              baselineActionIndex,
              minimumAdvantage: resolvedTaxMinAdvantage,
              evaluate: evaluateTaxReturnActionConditionedActorCritic,
              decisionTelemetry,
            });
          },
        }
      : {}),
    ...(revolution
      ? {
          revolutionPolicy(context) {
            if (!candidateIds.has(context.actorId)) {
              return normalRevolutionDecision(context);
            }
            return gatedCandidateDecision({
              context,
              benchmarkModel: revolution,
              baselineActionIndex: normalRevolutionDecision(context),
              minimumAdvantage: revolutionMinAdvantage,
              evaluate: evaluateRevolutionActionConditionedActorCritic,
              decisionTelemetry,
            });
          },
        }
      : {}),
  };
}

function taxAdvantageEnsembleCandidateDecision({
  context,
  benchmarkModel,
  baselineActionIndex,
  minimumAdvantage,
  decisionTelemetry,
}) {
  const selected = selectTaxReturnAdvantageEnsembleAction(
    benchmarkModel.model,
    context.observation,
    baselineActionIndex,
    minimumAdvantage,
  );
  const score = selected.selectedScore;
  const decision = {
    actionIndex: selected.actionIndex,
    logProbability: 0,
    policyVersion: benchmarkModel.policyVersion,
  };
  decisionTelemetry.push({
    decision: context.decision,
    decisionKey: context.decisionKey,
    actorId: context.actorId,
    actionIndex: decision.actionIndex,
    modelActionIndex: selected.modelActionIndex,
    baselineActionIndex: selected.baselineActionIndex,
    predictedAdvantage: Math.max(0, score?.lowerConfidenceBound ?? 0),
    minimumAdvantage: selected.minimumChipAdvantage,
    routing: selected.routing,
    policyVersion: benchmarkModel.policyVersion,
    scoreSemantics: selected.scoreSemantics,
    ensemble: {
      memberCount: benchmarkModel.model.members.length,
      memberAdvantages: score?.memberAdvantages ?? null,
      meanAdvantage: score?.meanAdvantage ?? null,
      sampleStandardDeviation: score?.sampleStandardDeviation ?? null,
      lowerConfidenceBound: score?.lowerConfidenceBound ?? null,
      unanimousPositive: score?.unanimousPositive ?? false,
      fallback: selected.fallback,
      fallbackReason: selected.fallbackReason,
      returnCount: selected.returnCount,
      actionScores: selected.actionScores,
    },
  });
  return decision;
}

function validateMinimumAdvantage(value, label) {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative finite number`);
  }
}

function gatedCandidateDecision({
  context,
  benchmarkModel,
  baselineActionIndex,
  minimumAdvantage,
  evaluate,
  decisionTelemetry,
}) {
  // The standard model policy validates that the simulator-supplied encoded
  // observation and legal mask exactly match the semantic observation.
  const ordinaryDecision = benchmarkModel.policy(context);
  if (!ordinaryDecision || typeof ordinaryDecision !== "object") {
    throw new TypeError("benchmark model policy returned no decision metadata");
  }
  const output = evaluate(benchmarkModel.model, context.observation);
  const gated = selectBaselineGatedNonCardAction(
    output,
    baselineActionIndex,
    minimumAdvantage,
  );
  if (
    ordinaryDecision.actionIndex !== gated.modelActionIndex ||
    ordinaryDecision.policyVersion !== benchmarkModel.policyVersion
  ) {
    throw new Error("benchmark model argmax or policyVersion drifted during gating");
  }
  const decision = {
    actionIndex: gated.actionIndex,
    logProbability: ordinaryDecision.logProbability,
    valueEstimate: ordinaryDecision.valueEstimate,
    policyVersion: benchmarkModel.policyVersion,
  };
  decisionTelemetry.push({
    decision: context.decision,
    decisionKey: context.decisionKey,
    actorId: context.actorId,
    actionIndex: decision.actionIndex,
    modelActionIndex: gated.modelActionIndex,
    baselineActionIndex: gated.baselineActionIndex,
    predictedAdvantage: gated.predictedAdvantage,
    minimumAdvantage: gated.minimumAdvantage,
    routing: gated.routing,
    policyVersion: benchmarkModel.policyVersion,
  });
  return decision;
}

function marginAccumulator() {
  return {
    count: 0,
    sum: 0,
    sumSquares: 0,
    minimum: null,
    maximum: null,
  };
}

function addMargin(accumulator, value) {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError("predicted non-card advantage must be non-negative and finite");
  }
  accumulator.count += 1;
  accumulator.sum += value;
  accumulator.sumSquares += value * value;
  accumulator.minimum =
    accumulator.minimum === null ? value : Math.min(accumulator.minimum, value);
  accumulator.maximum =
    accumulator.maximum === null ? value : Math.max(accumulator.maximum, value);
}

function addFiniteValue(accumulator, value, label) {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite`);
  }
  accumulator.count += 1;
  accumulator.sum += value;
  accumulator.sumSquares += value * value;
  accumulator.minimum =
    accumulator.minimum === null
      ? value
      : Math.min(accumulator.minimum, value);
  accumulator.maximum =
    accumulator.maximum === null
      ? value
      : Math.max(accumulator.maximum, value);
}

function routingGroup() {
  const group = {
    candidateModel: 0,
    candidateNormalHeuristic: 0,
    normalNormalHeuristic: 0,
    learnedAction: 0,
    agreedWithBaseline: 0,
    safetyFallback: 0,
    validatedPolicyVersionSteps: 0,
  };
  Object.defineProperty(group, MARGIN_ACCUMULATORS, {
    value: {
      allCandidateModelDecisions: marginAccumulator(),
      modelDiffersFromBaseline: marginAccumulator(),
      learnedAction: marginAccumulator(),
      safetyFallback: marginAccumulator(),
    },
  });
  Object.defineProperty(group, ENSEMBLE_ACCUMULATORS, {
    value: {
      decisions: 0,
      returnCountOneExactNormalFallback: 0,
      returnCountTwoEvaluated: 0,
      unanimousPositive: 0,
      notUnanimousPositive: 0,
      fallback: 0,
      learned: 0,
      fallbackReasons: {},
      meanAdvantage: marginAccumulator(),
      sampleStandardDeviation: marginAccumulator(),
      lowerConfidenceBound: marginAccumulator(),
    },
  });
  return group;
}

export function createNonCardRoutingTotals() {
  return {
    taxReturn: routingGroup(),
    revolution: routingGroup(),
  };
}

export function recordNonCardRouting(
  totals,
  steps,
  candidateIds,
  enabledModels,
  decisionTelemetry = [],
) {
  if (!Array.isArray(decisionTelemetry)) {
    throw new TypeError("decisionTelemetry must be an array");
  }
  const telemetryByKey = new Map();
  for (const decision of decisionTelemetry) {
    const key = `${decision.decision}\u0000${decision.decisionKey}`;
    if (telemetryByKey.has(key)) {
      throw new Error(`duplicate non-card benchmark telemetry ${key}`);
    }
    telemetryByKey.set(key, decision);
  }
  const consumedTelemetry = new Set();
  for (const step of steps ?? []) {
    const group =
      step.decision === "tax-return"
        ? totals.taxReturn
        : step.decision === "revolution"
          ? totals.revolution
          : null;
    if (!group) {
      throw new TypeError(`unknown non-card decision ${String(step.decision)}`);
    }
    const model =
      step.decision === "tax-return"
        ? enabledModels.taxReturn
        : enabledModels.revolution;
    const candidateActor = candidateIds.has(step.actorId);
    if (candidateActor && model) {
      if (
        step.behaviorPolicy !== "custom" ||
        step.behaviorPolicyVersion !== model.policyVersion
      ) {
        throw new Error(
          `${step.decision} candidate was not routed through its model`,
        );
      }
      const key = `${step.decision}\u0000${step.decisionKey}`;
      const decision = telemetryByKey.get(key);
      if (!decision) {
        throw new Error(`${step.decision} candidate has no safety-gate telemetry`);
      }
      if (
        decision.actorId !== step.actorId ||
        decision.actionIndex !== step.actionIndex ||
        decision.policyVersion !== step.behaviorPolicyVersion
      ) {
        throw new Error(
          `${step.decision} simulator step does not match safety-gate telemetry`,
        );
      }
      if (
        decision.routing !== "learnedAction" &&
        decision.routing !== "agreedWithBaseline" &&
        decision.routing !== "safetyFallback"
      ) {
        throw new TypeError(`${step.decision} has unknown safety-gate routing`);
      }
      consumedTelemetry.add(key);
      group[decision.routing] += 1;
      group.validatedPolicyVersionSteps += 1;
      const margins = group[MARGIN_ACCUMULATORS];
      addMargin(
        margins.allCandidateModelDecisions,
        decision.predictedAdvantage,
      );
      if (decision.modelActionIndex !== decision.baselineActionIndex) {
        addMargin(
          margins.modelDiffersFromBaseline,
          decision.predictedAdvantage,
        );
      }
      if (decision.routing === "learnedAction") {
        addMargin(margins.learnedAction, decision.predictedAdvantage);
      } else if (decision.routing === "safetyFallback") {
        addMargin(margins.safetyFallback, decision.predictedAdvantage);
      }
      if (decision.ensemble !== undefined) {
        if (
          decision.scoreSemantics !== TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS ||
          step.decision !== "tax-return" ||
          !decision.ensemble ||
          typeof decision.ensemble !== "object"
        ) {
          throw new TypeError("invalid tax-return ensemble telemetry");
        }
        const ensemble = group[ENSEMBLE_ACCUMULATORS];
        ensemble.decisions += 1;
        if (decision.ensemble.returnCount === 1) {
          if (
            !decision.ensemble.fallback ||
            decision.ensemble.fallbackReason !==
              "return-count-one-exact-normal" ||
            decision.actionIndex !== decision.baselineActionIndex ||
            decision.ensemble.meanAdvantage !== null ||
            decision.ensemble.sampleStandardDeviation !== null ||
            decision.ensemble.lowerConfidenceBound !== null
          ) {
            throw new Error(
              "returnCount=1 ensemble telemetry did not use exact normal fallback",
            );
          }
          ensemble.returnCountOneExactNormalFallback += 1;
        } else if (decision.ensemble.returnCount === 2) {
          ensemble.returnCountTwoEvaluated += 1;
          addFiniteValue(
            ensemble.meanAdvantage,
            decision.ensemble.meanAdvantage,
            "ensemble mean advantage",
          );
          addFiniteValue(
            ensemble.sampleStandardDeviation,
            decision.ensemble.sampleStandardDeviation,
            "ensemble sample standard deviation",
          );
          addFiniteValue(
            ensemble.lowerConfidenceBound,
            decision.ensemble.lowerConfidenceBound,
            "ensemble lower confidence bound",
          );
          if (decision.ensemble.sampleStandardDeviation < 0) {
            throw new RangeError(
              "ensemble sample standard deviation must be non-negative",
            );
          }
          if (decision.ensemble.unanimousPositive) {
            ensemble.unanimousPositive += 1;
          } else {
            ensemble.notUnanimousPositive += 1;
          }
        } else {
          throw new TypeError("unknown tax-return ensemble returnCount");
        }
        if (decision.ensemble.fallback) {
          ensemble.fallback += 1;
          const reason = decision.ensemble.fallbackReason;
          if (typeof reason !== "string" || reason.length < 1) {
            throw new TypeError("ensemble fallback requires a reason");
          }
          ensemble.fallbackReasons[reason] =
            (ensemble.fallbackReasons[reason] ?? 0) + 1;
        } else {
          ensemble.learned += 1;
        }
      }
    } else if (!candidateActor && model) {
      if (
        step.behaviorPolicy !== "custom" ||
        step.behaviorPolicyVersion !== null
      ) {
        throw new Error(
          `${step.decision} normal actor was not routed through the normal heuristic`,
        );
      }
    } else if (step.behaviorPolicy !== "normal") {
      throw new Error(
        `${step.decision} omitted-model fallback was not the normal heuristic`,
      );
    }

    if (!candidateActor) {
      group.normalNormalHeuristic += 1;
    } else if (model) {
      group.candidateModel += 1;
    } else {
      group.candidateNormalHeuristic += 1;
    }
  }
  if (consumedTelemetry.size !== telemetryByKey.size) {
    const unused = [...telemetryByKey.keys()].filter(
      (key) => !consumedTelemetry.has(key),
    );
    throw new Error(
      `non-card safety-gate telemetry was not observed in simulator steps: ${unused.join(", ")}`,
    );
  }
}

function summarizeMargin(accumulator) {
  if (accumulator.count === 0) {
    return {
      count: 0,
      minimum: null,
      maximum: null,
      mean: null,
      populationStandardDeviation: null,
    };
  }
  const mean = accumulator.sum / accumulator.count;
  const variance = Math.max(
    0,
    accumulator.sumSquares / accumulator.count - mean * mean,
  );
  return {
    count: accumulator.count,
    minimum: accumulator.minimum,
    maximum: accumulator.maximum,
    mean,
    populationStandardDeviation: Math.sqrt(variance),
  };
}

function summarizeRoutingGroup(group) {
  const margins = group[MARGIN_ACCUMULATORS];
  const ensemble = group[ENSEMBLE_ACCUMULATORS];
  return {
    candidateModel: group.candidateModel,
    candidateNormalHeuristic: group.candidateNormalHeuristic,
    normalNormalHeuristic: group.normalNormalHeuristic,
    learnedAction: group.learnedAction,
    agreedWithBaseline: group.agreedWithBaseline,
    safetyFallback: group.safetyFallback,
    validatedPolicyVersionSteps: group.validatedPolicyVersionSteps,
    marginSummaries: {
      unit:
        ensemble.decisions === 0
          ? "actor-logit"
          : "chip-lower-confidence-bound-clamped-at-zero",
      allCandidateModelDecisions: summarizeMargin(
        margins.allCandidateModelDecisions,
      ),
      modelDiffersFromBaseline: summarizeMargin(
        margins.modelDiffersFromBaseline,
      ),
      learnedAction: summarizeMargin(margins.learnedAction),
      safetyFallback: summarizeMargin(margins.safetyFallback),
    },
    ensembleAdvantageTelemetry:
      ensemble.decisions === 0
        ? null
        : {
            scoreSemantics: TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
            unit: "chip",
            decisions: ensemble.decisions,
            returnCountOneExactNormalFallback:
              ensemble.returnCountOneExactNormalFallback,
            returnCountTwoEvaluated: ensemble.returnCountTwoEvaluated,
            unanimousPositive: ensemble.unanimousPositive,
            notUnanimousPositive: ensemble.notUnanimousPositive,
            fallback: ensemble.fallback,
            learned: ensemble.learned,
            fallbackReasons: ensemble.fallbackReasons,
            meanAdvantage: summarizeMargin(ensemble.meanAdvantage),
            sampleStandardDeviation: summarizeMargin(
              ensemble.sampleStandardDeviation,
            ),
            lowerConfidenceBound: summarizeMargin(
              ensemble.lowerConfidenceBound,
            ),
          },
  };
}

export function summarizeNonCardRoutingTotals(totals) {
  return {
    taxReturn: summarizeRoutingGroup(totals.taxReturn),
    revolution: summarizeRoutingGroup(totals.revolution),
  };
}

export function nonCardSafetyGateProvenance({
  taxReturn,
  revolution,
  taxMinAdvantage,
  revolutionMinAdvantage,
}) {
  const taxAdvantageEnsemble =
    taxReturn?.modelKind === "baseline-advantage-ensemble";
  return {
    score: taxAdvantageEnsemble
      ? revolution
        ? "mixed-tax-chip-advantage-and-revolution-actor-logit"
        : TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS
      : "actor-logit",
    decisionRule: taxAdvantageEnsemble
      ? "returnCount=1 always uses exact normal; for returnCount=2 choose " +
        "the maximum LCB action only when every member advantage is > 0 " +
        "and mean - 1.645 * sampleSD > minimumChipAdvantage"
      : "when model argmax differs from baseline, use baseline iff " +
        "(model argmax logit - baseline action logit) < minimumAdvantage",
    tieBreak: taxAdvantageEnsemble
      ? "baseline, then lowest action index"
      : "lowest legal action index",
    defaultZeroPreservesModelArgmax: !taxAdvantageEnsemble,
    taxReturn: taxReturn
      ? {
          minimumAdvantage: taxMinAdvantage,
          unit: taxAdvantageEnsemble ? "chip" : "actor-logit",
          defaultMinimumAdvantage:
            taxReturn.defaultMinimumAdvantage ?? 0,
          memberCount: taxAdvantageEnsemble
            ? taxReturn.model.members.length
            : null,
          zValue: taxAdvantageEnsemble
            ? taxReturn.model.routing.zValue
            : null,
          baseline: NON_CARD_BASELINE_PROVENANCE.taxReturn,
        }
      : null,
    revolution: revolution
      ? {
          minimumAdvantage: revolutionMinAdvantage,
          unit: "actor-logit",
          baseline: NON_CARD_BASELINE_PROVENANCE.revolution,
        }
      : null,
  };
}

export function nonCardAblationName(taxReturn, revolution) {
  if (taxReturn && revolution) return "tax-return+revolution";
  if (taxReturn) return "tax-return-only";
  if (revolution) return "revolution-only";
  return null;
}
