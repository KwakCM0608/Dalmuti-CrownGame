import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, open } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import {
  REVOLUTION_ACTION_CATALOGUE_VERSION,
  REVOLUTION_ACTION_COUNT,
  REVOLUTION_ACTION_FEATURE_COUNT,
  TAX_RETURN_ACTION_CATALOGUE_VERSION,
  TAX_RETURN_ACTION_COUNT,
  TAX_RETURN_ACTION_FEATURE_COUNT,
  TAX_RETURN_ACTION_FEATURES,
} from "../../training/non-card-action-space.ts";
import {
  NON_CARD_OBSERVATION_SCHEMA_VERSION,
  REVOLUTION_OBSERVATION_FEATURE_COUNT,
  TAX_RETURN_OBSERVATION_FEATURE_COUNT,
} from "../../training/non-card-observation.ts";
import { buildPairedCounterfactualTargets } from "../../training/non-card-search-targets.ts";
import {
  NonCardDeterminizationRejectedError,
  simulateMatch,
} from "../../training/simulator.ts";

export const NON_CARD_COUNTERFACTUAL_FORMAT =
  "dalmuti-non-card-counterfactual-ndjson";
export const NON_CARD_COUNTERFACTUAL_FORMAT_VERSION = 1;
export const NON_CARD_COUNTERFACTUAL_DETERMINIZATION_FORMAT_VERSION = 2;
export const NON_CARD_DETERMINIZATION_ALGORITHM_VERSION = 1;
export const NON_CARD_DETERMINIZATION_ALGORITHM =
  "target-act-opponent-physical-card-fisher-yates-v1";
export const NON_CARD_DETERMINIZATION_SCHEMA =
  "world-clustered-paired-baseline-advantages-v2";
const NON_CARD_DETERMINIZATION_CONTRACT = Object.freeze({
  algorithm: NON_CARD_DETERMINIZATION_ALGORITHM,
  version: NON_CARD_DETERMINIZATION_ALGORITHM_VERSION,
  actorHand: "original-replay-hand-fixed",
  opponents:
    "all-non-actor-physical-cards-shuffled-then-dealt-to-original-public-hand-counts-in-rank-order",
  environmentRng: "not-consumed",
  candidateSeed:
    "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,attempt)))",
  continuationSeed:
    "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,continuationIndex,continuation)))",
});
export const NON_CARD_DETERMINIZATION_CONTRACT_SHA256 = sha256Json(
  NON_CARD_DETERMINIZATION_CONTRACT,
);
export const DEFAULT_NON_CARD_PLAYER_COUNTS = Object.freeze(
  Array.from({ length: 7 }, (_, index) => index + 4),
);
export const ALL_NON_CARD_DECISION_KINDS = Object.freeze([
  "tax-return",
  "revolution",
]);

const MAX_UINT32 = 0xffff_ffff;

function sha256Json(value) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

async function sha256File(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

export async function writeAllUtf8(handle, value) {
  if (!handle || typeof handle.write !== "function") {
    throw new TypeError("writeAllUtf8 requires a writable file handle");
  }
  if (typeof value !== "string") {
    throw new TypeError("writeAllUtf8 value must be a string");
  }
  const bytes = Buffer.from(value, "utf8");
  let offset = 0;
  while (offset < bytes.length) {
    const result = await handle.write(
      bytes,
      offset,
      bytes.length - offset,
      null,
    );
    const bytesWritten = result?.bytesWritten;
    if (
      !Number.isInteger(bytesWritten) ||
      bytesWritten < 1 ||
      bytesWritten > bytes.length - offset
    ) {
      throw new Error("file write made invalid or zero byte progress");
    }
    offset += bytesWritten;
  }
}

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

function nonNegativeInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new RangeError(`${label} must be a non-negative integer`);
  }
  return parsed;
}

function positiveFiniteNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new RangeError(`${label} must be finite and greater than zero`);
  }
  return parsed;
}

export function parseNonCardPlayerCounts(value) {
  const rawValues = Array.isArray(value)
    ? value
    : String(value ?? "4,5,6,7,8,9,10").split(",");
  const counts = rawValues.flatMap((entry) => String(entry).split(","));
  if (counts.length === 0) throw new RangeError("players cannot be empty");
  const parsed = counts.map((entry) => positiveInteger(entry.trim(), "players"));
  for (const playerCount of parsed) {
    if (playerCount < 4 || playerCount > 10) {
      throw new RangeError("players must contain only values from 4 to 10");
    }
  }
  if (new Set(parsed).size !== parsed.length) {
    throw new TypeError("players must not contain duplicates");
  }
  return parsed.sort((left, right) => left - right);
}

export function parseNonCardDecisionKinds(value) {
  const raw = String(value ?? "all").trim();
  if (raw === "all") return [...ALL_NON_CARD_DECISION_KINDS];
  const kinds = raw.split(",").map((entry) => entry.trim());
  if (kinds.length === 0 || kinds.some((entry) => entry.length === 0)) {
    throw new TypeError("decision must be tax-return, revolution, or all");
  }
  for (const kind of kinds) {
    if (!ALL_NON_CARD_DECISION_KINDS.includes(kind)) {
      throw new TypeError(`unsupported non-card decision kind: ${kind}`);
    }
  }
  if (new Set(kinds).size !== kinds.length) {
    throw new TypeError("decision kinds must not contain duplicates");
  }
  return ALL_NON_CARD_DECISION_KINDS.filter((kind) => kinds.includes(kind));
}

function actionCount(decision) {
  return decision === "tax-return"
    ? TAX_RETURN_ACTION_COUNT
    : REVOLUTION_ACTION_COUNT;
}

function legalMaskForStep(step) {
  const mask = Array.from({ length: actionCount(step.decision) }, () => false);
  let previous = -1;
  for (const actionIndex of step.legalActionIndices) {
    if (
      !Number.isInteger(actionIndex) ||
      actionIndex < 0 ||
      actionIndex >= mask.length
    ) {
      throw new RangeError(
        `${step.decision} step contains invalid legal action ${String(actionIndex)}`,
      );
    }
    if (actionIndex <= previous) {
      throw new TypeError(
        `${step.decision} legal action indices must be strictly increasing`,
      );
    }
    previous = actionIndex;
    mask[actionIndex] = true;
  }
  if (!mask.some(Boolean)) {
    throw new RangeError(`${step.decision} step has no legal action`);
  }
  if (!mask[step.actionIndex]) {
    throw new RangeError(`${step.decision} baseline selected an illegal action`);
  }
  return mask;
}

function stablePreDecisionMetadata(step) {
  const common = {
    playerCount: step.metadata.playerCount,
    actorHandCount: step.metadata.actorHandCount,
  };
  return step.decision === "tax-return"
    ? { ...common, returnCount: step.metadata.returnCount }
    : common;
}

function preDecisionState(step) {
  return {
    decision: step.decision,
    episodeId: step.episodeId,
    round: step.round,
    step: step.step,
    actorId: step.actorId,
    actorSeat: step.actorSeat,
    actorRole: step.actorRole,
    decisionKey: step.decisionKey,
    observationSchemaVersion: step.observationSchemaVersion,
    actionCatalogueVersion: step.actionCatalogueVersion,
    observation: step.observation,
    legalActionIndices: step.legalActionIndices,
    metadata: stablePreDecisionMetadata(step),
  };
}

function assertBaselineStep(step, baseConfig) {
  if (!step || typeof step !== "object") {
    throw new TypeError("baseline non-card step must be an object");
  }
  if (!ALL_NON_CARD_DECISION_KINDS.includes(step.decision)) {
    throw new TypeError(`unsupported baseline decision: ${String(step.decision)}`);
  }
  if (step.episodeId !== baseConfig.episodeId) {
    throw new Error("baseline decision episodeId does not match the match config");
  }
  if (step.observationSchemaVersion !== NON_CARD_OBSERVATION_SCHEMA_VERSION) {
    throw new Error("baseline observation schema version is unsupported");
  }
  const expectedCatalogueVersion =
    step.decision === "tax-return"
      ? TAX_RETURN_ACTION_CATALOGUE_VERSION
      : REVOLUTION_ACTION_CATALOGUE_VERSION;
  if (step.actionCatalogueVersion !== expectedCatalogueVersion) {
    throw new Error("baseline action catalogue version is unsupported");
  }
  if (step.forcedOverride) {
    throw new Error("baseline decision must not be a forced override");
  }
  if (
    !Array.isArray(step.observation) ||
    step.observation.some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError("baseline encoded observation must be finite");
  }
  legalMaskForStep(step);
}

function forcedOverrideFor(step, actionIndex) {
  const namespace =
    step.decision === "tax-return" ? "taxReturn" : "revolution";
  return {
    forcedOverrides: {
      [namespace]: {
        [step.decisionKey]: actionIndex,
      },
    },
  };
}

function actionFeatures(step, actionIndex) {
  if (step.decision === "tax-return") {
    return [...TAX_RETURN_ACTION_FEATURES[actionIndex]];
  }
  if (actionIndex === 0) return [1, 0, 0];
  return step.actorRole === "great-peon" ? [0, 0, 1] : [0, 1, 0];
}

function findForcedTargetStep(match, baselineStep) {
  const matches = (match.nonCardSteps ?? []).filter(
    (step) =>
      step.decision === baselineStep.decision &&
      step.decisionKey === baselineStep.decisionKey,
  );
  if (matches.length !== 1) {
    throw new Error(
      `forced rerun found ${matches.length} targeted decisions; expected exactly one`,
    );
  }
  return matches[0];
}

function assertStableTargetedDecision(baselineStep, forcedStep, actionIndex) {
  if (!isDeepStrictEqual(preDecisionState(forcedStep), preDecisionState(baselineStep))) {
    throw new Error(
      `${baselineStep.decision} targeted decision changed before forced action ${actionIndex}`,
    );
  }
  if (
    forcedStep.actionIndex !== actionIndex ||
    forcedStep.behaviorPolicy !== "forced-override" ||
    forcedStep.forcedOverride !== true
  ) {
    throw new Error(
      `${baselineStep.decision} action ${actionIndex} was not consumed as the targeted forced override`,
    );
  }
}

/**
 * Rerun one hidden world once for every legal root action. The match seed,
 * episode ID, normal continuation policies, and environment RNG are identical
 * in every clone. Only the correctly namespaced root override differs.
 */
function evaluateNonCardDecisionCounterfactualsLegacy({
  baseConfig,
  baselineStep,
  temperature = 1,
  simulate = simulateMatch,
}) {
  if (typeof simulate !== "function") {
    throw new TypeError("simulate must be a function");
  }
  const policyTemperature = positiveFiniteNumber(temperature, "temperature");
  assertBaselineStep(baselineStep, baseConfig);
  const legalMask = legalMaskForStep(baselineStep);
  const worldIdentity = {
    playerCount: baseConfig.playerCount,
    acts: baseConfig.acts,
    matchSeed: baseConfig.seed,
    episodeId: baseConfig.episodeId,
  };
  const pairedWorldId = `sha256:${sha256Json(worldIdentity)}`;
  const preDecisionSha256 = sha256Json(preDecisionState(baselineStep));
  const rawActions = [];

  for (const actionIndex of baselineStep.legalActionIndices) {
    const forcedMatch = simulate({
      ...baseConfig,
      nonCard: forcedOverrideFor(baselineStep, actionIndex),
    });
    const forcedStep = findForcedTargetStep(forcedMatch, baselineStep);
    assertStableTargetedDecision(baselineStep, forcedStep, actionIndex);
    const terminalActorUtility = forcedMatch.finalScores[baselineStep.actorId];
    if (!Number.isFinite(terminalActorUtility)) {
      throw new Error(
        `forced match has no finite terminal score for ${baselineStep.actorId}`,
      );
    }
    rawActions.push({
      actionIndex,
      actionFeatures: actionFeatures(baselineStep, actionIndex),
      pairedWorldId,
      terminalActorUtility,
      decisionActUtility: forcedStep.reward,
      terminalFinishPlaceInDecisionAct: forcedStep.finishPlace,
    });
  }

  const targets = buildPairedCounterfactualTargets(
    legalMask,
    [
      {
        sampleId: pairedWorldId,
        outcomes: rawActions.map((action) => ({
          actionIndex: action.actionIndex,
          utility: action.terminalActorUtility,
        })),
      },
    ],
    { policyTemperature },
  );
  const targetByAction = new Map(
    targets.actions.map((target) => [target.actionIndex, target]),
  );
  const actions = rawActions.map((action) => {
    const target = targetByAction.get(action.actionIndex);
    if (!target) throw new Error(`missing target for action ${action.actionIndex}`);
    return {
      ...action,
      meanUtility: target.meanUtility,
      centeredUtility: target.centeredUtility,
      uncertainty: {
        sampleStandardDeviation: target.sampleStandardDeviation,
        standardError: target.standardError,
      },
      softTargetProbability: target.policyProbability,
    };
  });

  return {
    type: "counterfactual-decision",
    sampleId: `${baselineStep.decision}:${preDecisionSha256}`,
    decision: baselineStep.decision,
    episodeId: baselineStep.episodeId,
    matchSeed: baseConfig.seed,
    playerCount: baseConfig.playerCount,
    acts: baseConfig.acts,
    round: baselineStep.round,
    actorId: baselineStep.actorId,
    actorSeat: baselineStep.actorSeat,
    actorRole: baselineStep.actorRole,
    decisionKey: baselineStep.decisionKey,
    observationSchemaVersion: baselineStep.observationSchemaVersion,
    actionCatalogueVersion: baselineStep.actionCatalogueVersion,
    observation: [...baselineStep.observation],
    legalMask,
    legalActionIndices: [...baselineStep.legalActionIndices],
    baselineActionIndex: baselineStep.actionIndex,
    metadata: stablePreDecisionMetadata(baselineStep),
    pairing: {
      pairedWorldId,
      preDecisionSha256,
      continuationPolicy: "normal-deterministic",
      forcedOverrideNamespace:
        baselineStep.decision === "tax-return" ? "taxReturn" : "revolution",
      rootActionCoverage: "all-legal-actions-exactly-once",
    },
    utility: {
      definition: "terminal-cumulative-chip-score",
      centeredAcrossLegalActions: true,
    },
    targetBuilder:
      "training/non-card-search-targets.ts#buildPairedCounterfactualTargets",
    targetSampleCount: targets.sampleCount,
    bestActionIndex: targets.bestActionIndex,
    actions,
  };
}

function canonicalInformationStateKey(step) {
  const material = {
    decision: step.decision,
    observationSchemaVersion: step.observationSchemaVersion,
    actionCatalogueVersion: step.actionCatalogueVersion,
    observation: step.observation,
    legalActionIndices: step.legalActionIndices,
    baselineActionIndex: step.actionIndex,
    metadata: stablePreDecisionMetadata(step),
  };
  return `sha256:${sha256Json(material)}`;
}

function determinizationCandidateSeed({
  rootSeed,
  informationStateKey,
  worldIndex,
  attempt,
}) {
  const digest = sha256Json({
    rootSeed,
    informationStateKey,
    worldIndex,
    attempt,
    algorithmVersion: NON_CARD_DETERMINIZATION_ALGORITHM_VERSION,
    algorithmContractSha256: NON_CARD_DETERMINIZATION_CONTRACT_SHA256,
  });
  return Number.parseInt(digest.slice(0, 8), 16) >>> 0;
}

function continuationCandidateSeed({
  rootSeed,
  informationStateKey,
  worldIndex,
  continuationIndex,
}) {
  const digest = sha256Json({
    rootSeed,
    informationStateKey,
    worldIndex,
    continuationIndex,
    purpose: "continuation",
    algorithmVersion: NON_CARD_DETERMINIZATION_ALGORITHM_VERSION,
    algorithmContractSha256: NON_CARD_DETERMINIZATION_CONTRACT_SHA256,
  });
  return Number.parseInt(digest.slice(0, 8), 16) >>> 0;
}

function sampleStatistics(values) {
  if (!Array.isArray(values) || values.length < 1) {
    throw new RangeError("sample statistics require at least one value");
  }
  if (values.some((value) => !Number.isFinite(value))) {
    throw new TypeError("sample statistics require finite values");
  }
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const sampleStandardDeviation =
    values.length === 1
      ? 0
      : Math.sqrt(
          values.reduce(
            (total, value) => total + (value - mean) ** 2,
            0,
          ) /
            (values.length - 1),
        );
  return {
    count: values.length,
    mean,
    sampleStandardDeviation,
    standardError: sampleStandardDeviation / Math.sqrt(values.length),
    standardErrorEstimable: values.length > 1,
  };
}

function meanAcrossContinuations(world, actionPosition, utilityField) {
  return (
    world.reduce((total, continuation) => {
      const action = continuation[actionPosition];
      if (!action || !Number.isFinite(action[utilityField])) {
        throw new Error(
          `hidden-world continuation omitted finite ${utilityField}`,
        );
      }
      return total + action[utilityField];
    }, 0) / world.length
  );
}

function assertWorldContinuationContract(world, legalActionIndices) {
  if (!Array.isArray(world) || world.length < 1) {
    throw new Error("hidden world must contain at least one continuation");
  }
  const reference = world[0];
  if (reference.length !== legalActionIndices.length) {
    throw new Error("hidden-world continuation has incomplete action coverage");
  }
  for (const [continuationIndex, continuation] of world.entries()) {
    if (continuation.length !== reference.length) {
      throw new Error("hidden-world continuations have inconsistent action coverage");
    }
    for (let position = 0; position < reference.length; position += 1) {
      const expectedActionIndex = legalActionIndices[position];
      const referenceAction = reference[position];
      const action = continuation[position];
      if (
        referenceAction.actionIndex !== expectedActionIndex ||
        action.actionIndex !== expectedActionIndex
      ) {
        throw new Error("hidden-world continuations changed legal action order");
      }
      // The collector's normal deterministic card-play policy consumes no
      // continuation RNG in the decision act. A difference here means that
      // contract drifted and C would be invalid pseudo-replication.
      if (!Object.is(action.decisionActUtility, referenceAction.decisionActUtility)) {
        throw new Error(
          `decision-act utility changed across continuations in one hidden world at continuation ${continuationIndex}`,
        );
      }
    }
  }
}

function publicPlayersForExpectedState(step) {
  if (!Array.isArray(step.publicPlayers) || step.publicPlayers.length < 4) {
    throw new TypeError(
      "augmented counterfactual collection requires public player snapshots",
    );
  }
  return step.publicPlayers.map((player) => ({
    id: player.id,
    role: player.role,
    handCount: player.handCount,
    score: player.score,
  }));
}

function assertDeterminizationAudit(
  audit,
  expectedSeed,
  baselineStep,
  { resampleOpponents, replaceContinuationRng },
) {
  if (!audit || typeof audit !== "object") {
    throw new Error("determinized match omitted its strict audit");
  }
  const expectedTaxTransfer = baselineStep.decision === "tax-return";
  if (
    audit.algorithmVersion !== NON_CARD_DETERMINIZATION_ALGORITHM_VERSION ||
    audit.hiddenWorldSeed !== expectedSeed ||
    audit.targetDecision !== baselineStep.decision ||
    audit.targetRound !== baselineStep.round ||
    audit.targetActorId !== baselineStep.actorId ||
    audit.physicalCardCount !== 80 ||
    audit.uniquePhysicalCardCount !== 80 ||
    audit.actorHandPreserved !== true ||
    audit.publicHandCountsPreserved !== true ||
    audit.publicHistoryPreserved !== true ||
    audit.environmentRandomDrawsConsumed !== 0 ||
    audit.hiddenWorldKind !==
      (resampleOpponents ? "resampled" : "original-replay") ||
    !Number.isInteger(audit.opponentCardsResampled) ||
    audit.opponentCardsResampled < 0 ||
    !Number.isInteger(audit.changedOpponentOwnershipCards) ||
    audit.changedOpponentOwnershipCards < 0 ||
    (resampleOpponents &&
      (audit.opponentCardsResampled < 1 ||
        audit.changedOpponentOwnershipCards < 1)) ||
    (!resampleOpponents &&
      (audit.opponentCardsResampled !== 0 ||
        audit.changedOpponentOwnershipCards !== 0)) ||
    audit.continuationRngReplaced !== replaceContinuationRng ||
    audit.targetEncodedObservationPreserved !== true ||
    audit.targetLegalActionsPreserved !== true ||
    audit.targetBaselineActionPreserved !== true ||
    audit.targetReturnCountPreserved !== true ||
    audit.targetPublicPlayersPreserved !== true ||
    (expectedTaxTransfer &&
      (!audit.taxTransfer ||
        audit.taxTransfer.exchangeCount !== 2 ||
        audit.taxTransfer.tributeCardCount !== 3 ||
        audit.taxTransfer.returnCardCount !== 3 ||
        audit.taxTransfer.ownershipCompleteAfterTransfer !== true ||
        audit.taxTransfer.publicHandCountsRestoredAfterTransfer !== true))
  ) {
    throw new Error("determinized match failed its strict invariant audit");
  }
}

function pairedWorldAuditProjection(audit) {
  const pairedWorld = { ...audit };
  delete pairedWorld.taxTransfer;
  return pairedWorld;
}

function evaluateControlledHiddenWorld({
  baseConfig,
  baselineMatch,
  baselineStep,
  hiddenWorldSeed,
  resampleOpponents,
  continuationSeed,
  simulate,
}) {
  const expected = {
    decision: baselineStep.decision,
    decisionKey: baselineStep.decisionKey,
    round: baselineStep.round,
    decisionStep: baselineStep.step,
    actorId: baselineStep.actorId,
    encodedObservation: [...baselineStep.observation],
    legalActionIndices: [...baselineStep.legalActionIndices],
    baselineActionIndex: baselineStep.actionIndex,
    returnCount:
      baselineStep.decision === "tax-return"
        ? baselineStep.metadata.returnCount
        : null,
    publicPlayers: publicPlayersForExpectedState(baselineStep),
    publicHistory: baselineMatch.acts
      .slice(0, baselineStep.round - 1)
      .map((act) => structuredClone(act)),
  };
  let invariantAudit = null;
  const actions = [];
  for (const actionIndex of baselineStep.legalActionIndices) {
    const forcedMatch = simulate({
      ...baseConfig,
      nonCard: {
        ...forcedOverrideFor(baselineStep, actionIndex),
        determinization: {
          algorithmVersion: NON_CARD_DETERMINIZATION_ALGORITHM_VERSION,
          hiddenWorldSeed,
          resampleOpponents,
          continuationSeed,
          expected,
        },
      },
    });
    const forcedStep = findForcedTargetStep(forcedMatch, baselineStep);
    assertStableTargetedDecision(baselineStep, forcedStep, actionIndex);
    if (!isDeepStrictEqual(forcedStep.publicPlayers, baselineStep.publicPlayers)) {
      throw new Error("determinized target public players drifted");
    }
    assertDeterminizationAudit(
      forcedMatch.nonCardDeterminizationAudit,
      hiddenWorldSeed,
      baselineStep,
      {
        resampleOpponents,
        replaceContinuationRng: continuationSeed !== null,
      },
    );
    if (invariantAudit === null) {
      invariantAudit = pairedWorldAuditProjection(
        forcedMatch.nonCardDeterminizationAudit,
      );
    } else if (
      !isDeepStrictEqual(
        invariantAudit,
        pairedWorldAuditProjection(forcedMatch.nonCardDeterminizationAudit),
      )
    ) {
      throw new Error(
        "paired root actions did not share one determinized hidden world",
      );
    }
    const terminalActorUtility = forcedMatch.finalScores[baselineStep.actorId];
    if (!Number.isFinite(terminalActorUtility)) {
      throw new Error("determinized match omitted finite actor utility");
    }
    actions.push({
      actionIndex,
      terminalActorUtility,
      decisionActUtility: forcedStep.reward,
    });
  }
  return actions;
}

function reasonCounts(reasons) {
  const counts = {};
  for (const reason of reasons) counts[reason] = (counts[reason] ?? 0) + 1;
  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function evaluateAugmentedNonCardDecisionCounterfactuals({
  baseConfig,
  baselineMatch,
  baselineStep,
  temperature,
  simulate,
  determinizationWorlds,
  continuationCount,
  determinizationRootSeed,
  maxDeterminizationAttempts,
}) {
  if (!baselineMatch || typeof baselineMatch !== "object") {
    throw new TypeError("augmented collection requires the baseline match");
  }
  const legacy = evaluateNonCardDecisionCounterfactualsLegacy({
    baseConfig,
    baselineStep,
    temperature,
    simulate,
  });
  const informationStateKey = canonicalInformationStateKey(baselineStep);
  const hiddenWorlds = [
    [
      legacy.actions.map((action) => ({
        actionIndex: action.actionIndex,
        terminalActorUtility: action.terminalActorUtility,
        decisionActUtility: action.decisionActUtility,
      })),
    ],
  ];
  const attempts = [];

  for (
    let continuationIndex = 1;
    continuationIndex < continuationCount;
    continuationIndex += 1
  ) {
    hiddenWorlds[0].push(
      evaluateControlledHiddenWorld({
        baseConfig,
        baselineMatch,
        baselineStep,
        hiddenWorldSeed: determinizationCandidateSeed({
          rootSeed: determinizationRootSeed,
          informationStateKey,
          worldIndex: 0,
          attempt: 0,
        }),
        resampleOpponents: false,
        continuationSeed: continuationCandidateSeed({
          rootSeed: determinizationRootSeed,
          informationStateKey,
          worldIndex: 0,
          continuationIndex,
        }),
        simulate,
      }),
    );
  }

  for (let worldIndex = 1; worldIndex < determinizationWorlds; worldIndex += 1) {
    const rejectedReasons = [];
    let accepted = null;
    let attempt = 0;
    for (attempt = 1; attempt <= maxDeterminizationAttempts; attempt += 1) {
      const hiddenWorldSeed = determinizationCandidateSeed({
        rootSeed: determinizationRootSeed,
        informationStateKey,
        worldIndex,
        attempt,
      });
      try {
        accepted = [];
        for (
          let continuationIndex = 0;
          continuationIndex < continuationCount;
          continuationIndex += 1
        ) {
          accepted.push(
            evaluateControlledHiddenWorld({
              baseConfig,
              baselineMatch,
              baselineStep,
              hiddenWorldSeed,
              resampleOpponents: true,
              continuationSeed:
                continuationIndex === 0
                  ? null
                  : continuationCandidateSeed({
                      rootSeed: determinizationRootSeed,
                      informationStateKey,
                      worldIndex,
                      continuationIndex,
                    }),
              simulate,
            }),
          );
        }
        break;
      } catch (error) {
        if (!(error instanceof NonCardDeterminizationRejectedError)) throw error;
        rejectedReasons.push(error.reasonCode);
        accepted = null;
      }
    }
    if (accepted === null) {
      throw new Error(
        `could not accept determinized world ${worldIndex} in ${maxDeterminizationAttempts} attempts; rejections=${JSON.stringify(reasonCounts(rejectedReasons))}`,
      );
    }
    hiddenWorlds.push(accepted);
    attempts.push({
      worldIndex,
      attemptCount: attempt,
      rejectedAttemptCount: rejectedReasons.length,
      rejectedReasonCounts: reasonCounts(rejectedReasons),
    });
  }

  if (
    hiddenWorlds.length !== determinizationWorlds ||
    hiddenWorlds.some((world) => world.length !== continuationCount)
  ) {
    throw new Error("hidden-world continuation grid is incomplete");
  }
  for (const world of hiddenWorlds) {
    assertWorldContinuationContract(world, baselineStep.legalActionIndices);
  }

  const legalMask = legalMaskForStep(baselineStep);
  const terminalBatches = hiddenWorlds.map((world, worldIndex) => ({
    sampleId: `hidden-world-${worldIndex}`,
    outcomes: baselineStep.legalActionIndices.map((actionIndex, position) => ({
      actionIndex,
      utility: meanAcrossContinuations(
        world,
        position,
        "terminalActorUtility",
      ),
    })),
  }));
  const decisionActBatches = hiddenWorlds.map((world, worldIndex) => ({
    sampleId: `hidden-world-${worldIndex}`,
    outcomes: baselineStep.legalActionIndices.map((actionIndex, position) => ({
      actionIndex,
      utility: world[0][position].decisionActUtility,
    })),
  }));
  const terminalTargets = buildPairedCounterfactualTargets(
    legalMask,
    terminalBatches,
    { policyTemperature: temperature },
  );
  const decisionActTargets = buildPairedCounterfactualTargets(
    legalMask,
    decisionActBatches,
    { policyTemperature: temperature },
  );
  const terminalByAction = new Map(
    terminalTargets.actions.map((target) => [target.actionIndex, target]),
  );
  const decisionActByAction = new Map(
    decisionActTargets.actions.map((target) => [target.actionIndex, target]),
  );
  const baselineActionPosition = baselineStep.legalActionIndices.indexOf(
    baselineStep.actionIndex,
  );
  if (baselineActionPosition < 0) {
    throw new Error("baseline action is absent from augmented worlds");
  }
  const actions = baselineStep.legalActionIndices.map((actionIndex, position) => {
    const terminal = terminalByAction.get(actionIndex);
    const decisionAct = decisionActByAction.get(actionIndex);
    if (!terminal || !decisionAct) {
      throw new Error(`missing augmented target for action ${actionIndex}`);
    }
    const terminalAdvantages = hiddenWorlds.map(
      (world) =>
        meanAcrossContinuations(world, position, "terminalActorUtility") -
        meanAcrossContinuations(
          world,
          baselineActionPosition,
          "terminalActorUtility",
        ),
    );
    const decisionActAdvantages = hiddenWorlds.map(
      (world) =>
        world[0][position].decisionActUtility -
        world[0][baselineActionPosition].decisionActUtility,
    );
    return {
      actionIndex,
      actionFeatures: actionFeatures(baselineStep, actionIndex),
      meanUtility: terminal.meanUtility,
      centeredUtility: terminal.centeredUtility,
      uncertainty: {
        count: determinizationWorlds,
        sampleStandardDeviation: terminal.sampleStandardDeviation,
        standardError: terminal.standardError,
        standardErrorEstimable: determinizationWorlds > 1,
      },
      softTargetProbability: terminal.policyProbability,
      pairedBaselineAdvantage: sampleStatistics(terminalAdvantages),
      decisionActUtilityAggregate: {
        meanUtility: decisionAct.meanUtility,
        centeredUtility: decisionAct.centeredUtility,
        uncertainty: {
          count: determinizationWorlds,
          sampleStandardDeviation: decisionAct.sampleStandardDeviation,
          standardError: decisionAct.standardError,
          standardErrorEstimable: determinizationWorlds > 1,
        },
        softTargetProbability: decisionAct.policyProbability,
      },
      pairedDecisionActBaselineAdvantage: sampleStatistics(
        decisionActAdvantages,
      ),
    };
  });

  return {
    type: "counterfactual-decision",
    sampleId: legacy.sampleId,
    canonicalInformationStateKey: informationStateKey,
    decision: legacy.decision,
    playerCount: legacy.playerCount,
    acts: legacy.acts,
    round: legacy.round,
    actorId: legacy.actorId,
    actorSeat: legacy.actorSeat,
    actorRole: legacy.actorRole,
    observationSchemaVersion: legacy.observationSchemaVersion,
    actionCatalogueVersion: legacy.actionCatalogueVersion,
    observation: legacy.observation,
    legalMask: legacy.legalMask,
    legalActionIndices: legacy.legalActionIndices,
    baselineActionIndex: legacy.baselineActionIndex,
    metadata: legacy.metadata,
    pairing: {
      canonicalInformationStateKey: informationStateKey,
      preDecisionSha256: legacy.pairing.preDecisionSha256,
      continuationPolicy: legacy.pairing.continuationPolicy,
      forcedOverrideNamespace: legacy.pairing.forcedOverrideNamespace,
      rootActionCoverage: "all-legal-actions-in-every-accepted-hidden-world",
      continuationRngPairing:
        "same-environment-stream-and-hidden-world-seed-for-every-root-action",
    },
    determinization: {
      worldCount: determinizationWorlds,
      continuationCount,
      rawContinuationEvaluations:
        determinizationWorlds * continuationCount,
      effectiveIndependentWorlds: determinizationWorlds,
      standardErrorEstimable: determinizationWorlds > 1,
      originalReplayWorldIncluded: true,
      resampledWorldCount: determinizationWorlds - 1,
      rootSeed: determinizationRootSeed,
      maxAttemptsPerResampledWorld: maxDeterminizationAttempts,
      algorithm: NON_CARD_DETERMINIZATION_ALGORITHM,
      algorithmVersion: NON_CARD_DETERMINIZATION_ALGORITHM_VERSION,
      algorithmContractSha256: NON_CARD_DETERMINIZATION_CONTRACT_SHA256,
      candidateSeedDerivation: NON_CARD_DETERMINIZATION_CONTRACT.candidateSeed,
      continuationSeedDerivation:
        NON_CARD_DETERMINIZATION_CONTRACT.continuationSeed,
      acceptedWorldAttempts: attempts,
      individualReplaySeedsIncluded: false,
      explicitIndividualSeedsIncluded: false,
      individualSeedsDerivableFromRestrictedRootProvenance: true,
      individualWorldUtilitiesIncluded: false,
      distribution: "restricted-training-only",
    },
    utility: {
      terminalDefinition: "terminal-cumulative-chip-score",
      decisionActDefinition: "centered-round-chip-award",
      centeredAcrossLegalActions: true,
      pairedBaselineAdvantagesBeforeAggregation: true,
    },
    targetBuilder:
      "training/non-card-search-targets.ts#buildPairedCounterfactualTargets",
    targetSampleCount: terminalTargets.sampleCount,
    bestActionIndex: terminalTargets.bestActionIndex,
    bestDecisionActActionIndex: decisionActTargets.bestActionIndex,
    forcedActionEvaluations:
      determinizationWorlds * continuationCount * actions.length,
    actions,
  };
}

export function evaluateNonCardDecisionCounterfactuals({
  baseConfig,
  baselineMatch,
  baselineStep,
  temperature = 1,
  simulate = simulateMatch,
  determinizationWorlds = 1,
  continuationCount = 1,
  determinizationRootSeed = baseConfig?.seed ?? 0,
  maxDeterminizationAttempts = 32,
  forceAugmentedFormat = false,
}) {
  const worldCount = positiveInteger(
    determinizationWorlds,
    "determinizationWorlds",
  );
  const continuations = positiveInteger(
    continuationCount,
    "continuationCount",
  );
  if (worldCount === 1 && continuations === 1 && !forceAugmentedFormat) {
    return evaluateNonCardDecisionCounterfactualsLegacy({
      baseConfig,
      baselineStep,
      temperature,
      simulate,
    });
  }
  const rootSeed = nonNegativeInteger(
    determinizationRootSeed,
    "determinizationRootSeed",
  );
  if (rootSeed > MAX_UINT32) {
    throw new RangeError("determinizationRootSeed must fit unsigned 32-bit");
  }
  const attempts = positiveInteger(
    maxDeterminizationAttempts,
    "maxDeterminizationAttempts",
  );
  return evaluateAugmentedNonCardDecisionCounterfactuals({
    baseConfig,
    baselineMatch,
    baselineStep,
    temperature: positiveFiniteNumber(temperature, "temperature"),
    simulate,
    determinizationWorlds: worldCount,
    continuationCount: continuations,
    determinizationRootSeed: rootSeed,
    maxDeterminizationAttempts: attempts,
  });
}

function normalizeGenerationOptions(options) {
  if (!options || typeof options !== "object") {
    throw new TypeError("generation options must be an object");
  }
  if (typeof options.outputPath !== "string" || options.outputPath.length === 0) {
    throw new TypeError("outputPath must be a non-empty string");
  }
  const playerCounts = parseNonCardPlayerCounts(
    options.playerCounts ?? DEFAULT_NON_CARD_PLAYER_COUNTS,
  );
  const episodes = positiveInteger(options.episodes ?? 10, "episodes");
  const acts = positiveInteger(options.acts ?? 3, "acts");
  const seed = nonNegativeInteger(options.seed ?? 710_001, "seed");
  const totalMatches = playerCounts.length * episodes;
  if (seed > MAX_UINT32 || seed + totalMatches - 1 > MAX_UINT32) {
    throw new RangeError("seed range must fit unique unsigned 32-bit match seeds");
  }
  const decisionKinds = Array.isArray(options.decisionKinds)
    ? parseNonCardDecisionKinds(options.decisionKinds.join(","))
    : parseNonCardDecisionKinds(options.decisionKinds ?? "all");
  const temperature = positiveFiniteNumber(options.temperature ?? 1, "temperature");
  const determinizationWorlds = positiveInteger(
    options.determinizationWorlds ?? 1,
    "determinizationWorlds",
  );
  const continuationCount = positiveInteger(
    options.continuationCount ?? 1,
    "continuationCount",
  );
  const determinizationRootSeed = nonNegativeInteger(
    options.determinizationRootSeed ?? seed,
    "determinizationRootSeed",
  );
  if (determinizationRootSeed > MAX_UINT32) {
    throw new RangeError("determinizationRootSeed must fit unsigned 32-bit");
  }
  const maxDeterminizationAttempts = positiveInteger(
    options.maxDeterminizationAttempts ?? 32,
    "maxDeterminizationAttempts",
  );
  let taxReturnCounts = null;
  if (
    options.taxReturnCounts !== undefined &&
    options.taxReturnCounts !== null &&
    String(options.taxReturnCounts).trim() !== "all"
  ) {
    const values = (Array.isArray(options.taxReturnCounts)
      ? options.taxReturnCounts
      : String(options.taxReturnCounts).split(",")
    ).map((value) => positiveInteger(String(value).trim(), "taxReturnCounts"));
    if (
      values.length < 1 ||
      values.some((value) => value !== 1 && value !== 2) ||
      new Set(values).size !== values.length
    ) {
      throw new TypeError("taxReturnCounts must be all, 1, 2, or 1,2");
    }
    taxReturnCounts = [...values].sort((left, right) => left - right);
    if (
      decisionKinds.length !== 1 ||
      decisionKinds[0] !== "tax-return"
    ) {
      throw new TypeError(
        "taxReturnCounts requires decisionKinds to contain only tax-return",
      );
    }
  }
  const maxDecisions =
    options.maxDecisions === undefined || options.maxDecisions === null
      ? null
      : positiveInteger(options.maxDecisions, "maxDecisions");
  const createdAt = options.createdAt ?? new Date().toISOString();
  if (typeof createdAt !== "string" || createdAt.length === 0) {
    throw new TypeError("createdAt must be a non-empty string");
  }
  if (options.simulate !== undefined && typeof options.simulate !== "function") {
    throw new TypeError("simulate must be a function");
  }
  return {
    outputPath: resolve(options.outputPath),
    playerCounts,
    episodes,
    acts,
    seed,
    decisionKinds,
    temperature,
    determinizationWorlds,
    continuationCount,
    determinizationRootSeed,
    maxDeterminizationAttempts,
    taxReturnCounts,
    augmentedFormat:
      determinizationWorlds > 1 ||
      continuationCount > 1 ||
      taxReturnCounts !== null,
    maxDecisions,
    createdAt,
    simulate: options.simulate ?? simulateMatch,
  };
}

function emptyCountRecord(playerCounts) {
  return {
    byDecision: Object.fromEntries(
      ALL_NON_CARD_DECISION_KINDS.map((decision) => [
        decision,
        { discovered: 0, written: 0, actionEvaluations: 0 },
      ]),
    ),
    byPlayerCount: Object.fromEntries(
      playerCounts.map((playerCount) => [
        playerCount,
        { baselineMatches: 0, decisionsWritten: 0, actionEvaluations: 0 },
      ]),
    ),
  };
}

/**
 * Stream an exclusive, non-resumable NDJSON dataset. If collection fails, the
 * claimed file remains visibly incomplete and a later run must use a new path.
 */
export async function generateNonCardCounterfactualDataset(options) {
  const normalized = normalizeGenerationOptions(options);
  await mkdir(dirname(normalized.outputPath), { recursive: true });
  const handle = await open(normalized.outputPath, "wx");
  const checksumPath = `${normalized.outputPath}.sha256`;
  let checksumHandle;
  try {
    checksumHandle = await open(checksumPath, "wx");
  } catch (error) {
    await handle.close();
    throw error;
  }
  const contentHash = createHash("sha256");
  let bytesBeforeSummary = 0;
  let closed = false;
  let checksumClosed = false;

  async function writeRecord(record, includeInContentHash = true) {
    const line = `${JSON.stringify(record)}\n`;
    if (includeInContentHash) {
      contentHash.update(line);
      bytesBeforeSummary += Buffer.byteLength(line);
    }
    await writeAllUtf8(handle, line);
  }

  const manifest = {
    type: "manifest",
    format: NON_CARD_COUNTERFACTUAL_FORMAT,
    version:
      !normalized.augmentedFormat
        ? NON_CARD_COUNTERFACTUAL_FORMAT_VERSION
        : NON_CARD_COUNTERFACTUAL_DETERMINIZATION_FORMAT_VERSION,
    createdAt: normalized.createdAt,
    observationSchemaVersion: NON_CARD_OBSERVATION_SCHEMA_VERSION,
    actionCatalogueVersions: {
      taxReturn: TAX_RETURN_ACTION_CATALOGUE_VERSION,
      revolution: REVOLUTION_ACTION_CATALOGUE_VERSION,
    },
    featureDimensions: {
      taxReturn: {
        observation: TAX_RETURN_OBSERVATION_FEATURE_COUNT,
        action: TAX_RETURN_ACTION_FEATURE_COUNT,
        catalogue: TAX_RETURN_ACTION_COUNT,
      },
      revolution: {
        observation: REVOLUTION_OBSERVATION_FEATURE_COUNT,
        action: REVOLUTION_ACTION_FEATURE_COUNT,
        catalogue: REVOLUTION_ACTION_COUNT,
      },
    },
    collection: {
      playerCounts: normalized.playerCounts,
      episodesPerPlayerCount: normalized.episodes,
      acts: normalized.acts,
      initialSeed: normalized.seed,
      matchSeedDerivation:
        "initialSeed + zero-based index over ascending playerCount then episode",
      decisionKinds: normalized.decisionKinds,
      policyTemperature: normalized.temperature,
      maxDecisions: normalized.maxDecisions,
      baselineNonCardHooks: {},
      continuationPolicy: "normal-deterministic",
      resumeAllowed: false,
      ...(normalized.augmentedFormat
        ? {
            taxReturnCounts: normalized.taxReturnCounts,
            determinization: {
              worldCountPerInformationState:
                normalized.determinizationWorlds,
              continuationCountPerHiddenWorld:
                normalized.continuationCount,
              rawContinuationEvaluationsPerInformationState:
                normalized.determinizationWorlds *
                normalized.continuationCount,
              effectiveIndependentWorldsPerInformationState:
                normalized.determinizationWorlds,
              standardErrorEstimable:
                normalized.determinizationWorlds > 1,
              originalReplayWorldIncluded: true,
              rootSeed: normalized.determinizationRootSeed,
              maxAttemptsPerResampledWorld:
                normalized.maxDeterminizationAttempts,
              algorithm: NON_CARD_DETERMINIZATION_ALGORITHM,
              algorithmVersion:
                NON_CARD_DETERMINIZATION_ALGORITHM_VERSION,
              algorithmContractSha256:
                NON_CARD_DETERMINIZATION_CONTRACT_SHA256,
              candidateSeedDerivation:
                NON_CARD_DETERMINIZATION_CONTRACT.candidateSeed,
              continuationSeedDerivation:
                NON_CARD_DETERMINIZATION_CONTRACT.continuationSeed,
            },
          }
        : {}),
    },
    privacy: {
      observation: "encoded-actor-hand-and-public-state-only",
      opponentCardIdentitiesIncluded: false,
      physicalCardIdsIncluded: false,
      ...(normalized.augmentedFormat
        ? {
            individualReplaySeedsIncluded: false,
            explicitIndividualSeedsIncluded: false,
            individualSeedsDerivableFromRestrictedRootProvenance: true,
            individualWorldUtilitiesIncluded: false,
            aggregateTargetsOnly: true,
            distribution: "restricted-training-only",
          }
        : {}),
    },
    ...(normalized.augmentedFormat
      ? {
          groupSplitKey: "canonicalInformationStateKey",
          determinizationSchema: NON_CARD_DETERMINIZATION_SCHEMA,
        }
      : {}),
  };
  const counts = emptyCountRecord(normalized.playerCounts);
  let baselineMatches = 0;
  let decisionsDiscovered = 0;
  let decisionsWritten = 0;
  let actionEvaluations = 0;
  let stoppedAtMaxDecisions = false;
  let determinizedWorldsAccepted = 0;
  let determinizationAttempts = 0;
  let determinizationRejectedAttempts = 0;

  try {
    await writeRecord(manifest);
    outer: for (const playerCount of normalized.playerCounts) {
      for (let episode = 0; episode < normalized.episodes; episode += 1) {
        const matchIndex = baselineMatches;
        const matchSeed = normalized.seed + matchIndex;
        const episodeId =
          `non-card-p${playerCount}-episode-${episode + 1}-seed-${matchSeed}`;
        const baseConfig = {
          playerCount,
          acts: normalized.acts,
          seed: matchSeed,
          episodeId,
          difficulties: ["normal"],
        };
        const baseline = normalized.simulate({
          ...baseConfig,
          nonCard: {},
        });
        baselineMatches += 1;
        counts.byPlayerCount[playerCount].baselineMatches += 1;
        for (const step of baseline.nonCardSteps ?? []) {
          decisionsDiscovered += 1;
          counts.byDecision[step.decision].discovered += 1;
          if (!normalized.decisionKinds.includes(step.decision)) continue;
          if (
            step.decision === "tax-return" &&
            normalized.taxReturnCounts !== null &&
            !normalized.taxReturnCounts.includes(step.metadata.returnCount)
          ) {
            continue;
          }
          if (
            normalized.maxDecisions !== null &&
            decisionsWritten >= normalized.maxDecisions
          ) {
            stoppedAtMaxDecisions = true;
            break outer;
          }
          const record = evaluateNonCardDecisionCounterfactuals({
            baseConfig,
            baselineMatch: baseline,
            baselineStep: step,
            temperature: normalized.temperature,
            simulate: normalized.simulate,
            determinizationWorlds: normalized.determinizationWorlds,
            continuationCount: normalized.continuationCount,
            determinizationRootSeed: normalized.determinizationRootSeed,
            maxDeterminizationAttempts:
              normalized.maxDeterminizationAttempts,
            forceAugmentedFormat: normalized.augmentedFormat,
          });
          await writeRecord(record);
          decisionsWritten += 1;
          const recordActionEvaluations =
            record.forcedActionEvaluations ?? record.actions.length;
          actionEvaluations += recordActionEvaluations;
          counts.byDecision[step.decision].written += 1;
          counts.byDecision[step.decision].actionEvaluations +=
            recordActionEvaluations;
          counts.byPlayerCount[playerCount].decisionsWritten += 1;
          counts.byPlayerCount[playerCount].actionEvaluations +=
            recordActionEvaluations;
          if (record.determinization) {
            determinizedWorldsAccepted +=
              record.determinization.resampledWorldCount;
            for (const attempt of record.determinization.acceptedWorldAttempts) {
              determinizationAttempts += attempt.attemptCount;
              determinizationRejectedAttempts +=
                attempt.rejectedAttemptCount;
            }
          }
        }
      }
    }
    const contentSha256 = contentHash.digest("hex");
    const summary = {
      type: "summary",
      baselineMatches,
      decisionsDiscovered,
      decisionsWritten,
      actionEvaluations,
      stoppedAtMaxDecisions,
      counts,
      hashes: {
        algorithm: "sha256",
        contentBeforeSummary: contentSha256,
        contentBeforeSummaryBytes: bytesBeforeSummary,
        scope: "UTF-8 NDJSON bytes for manifest and decision records, including newlines",
      },
      ...(normalized.augmentedFormat
        ? {
            determinization: {
              resampledWorldsAccepted: determinizedWorldsAccepted,
              attempts: determinizationAttempts,
              rejectedAttempts: determinizationRejectedAttempts,
            },
          }
        : {}),
    };
    await writeRecord(summary, false);
    await handle.close();
    closed = true;
    const fileSha256 = await sha256File(normalized.outputPath);
    await writeAllUtf8(
      checksumHandle,
      `${fileSha256}  ${basename(normalized.outputPath)}\n`,
    );
    await checksumHandle.close();
    checksumClosed = true;
    return {
      outputPath: normalized.outputPath,
      checksumPath,
      fileSha256,
      summary,
    };
  } finally {
    if (!closed) await handle.close();
    if (!checksumClosed) await checksumHandle.close();
  }
}
