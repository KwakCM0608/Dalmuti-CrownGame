import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, open, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";

import {
  REVOLUTION_ACTION_CATALOGUE_VERSION,
  REVOLUTION_ACTION_COUNT,
  REVOLUTION_ACTION_FEATURE_COUNT,
  REVOLUTION_DECLARE_ACTION_INDEX,
  TAX_RETURN_ACTION_CATALOGUE,
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
import { confidenceInterval95 } from "../rl-evaluation-statistics.mjs";

const DATASET_FORMAT = "dalmuti-non-card-counterfactual-ndjson";
const DATASET_VERSION = 1;
const REPORT_FORMAT = "dalmuti-non-card-counterfactual-diagnostics";
const REPORT_VERSION = 1;
const DECISIONS = Object.freeze(["tax-return", "revolution"]);
const ROLES = new Set([
  "great-dalmuti",
  "lesser-dalmuti",
  "merchant",
  "lesser-peon",
  "great-peon",
]);
const SUM_TOLERANCE = 1e-9;

function fail(lineNumber, message) {
  const prefix = lineNumber === null ? "dataset" : `line ${lineNumber}`;
  throw new Error(`${prefix}: ${message}`);
}

function requireObject(value, label, lineNumber) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(lineNumber, `${label} must be an object`);
  }
  return value;
}

function requireNonEmptyString(value, label, lineNumber) {
  if (typeof value !== "string" || value.length === 0) {
    fail(lineNumber, `${label} must be a non-empty string`);
  }
  return value;
}

function requireInteger(value, label, lineNumber, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    fail(lineNumber, `${label} must be an integer >= ${minimum}`);
  }
  return value;
}

function requireFinite(value, label, lineNumber) {
  if (!Number.isFinite(value)) {
    fail(lineNumber, `${label} must be finite`);
  }
  return value;
}

function nearlyEqual(left, right, tolerance = SUM_TOLERANCE) {
  return Math.abs(left - right) <= tolerance * Math.max(1, Math.abs(left), Math.abs(right));
}

function validateManifest(record, lineNumber) {
  requireObject(record, "manifest", lineNumber);
  if (record.type !== "manifest") fail(lineNumber, "first record must be a manifest");
  if (record.format !== DATASET_FORMAT || record.version !== DATASET_VERSION) {
    fail(lineNumber, `unsupported format/version ${String(record.format)}/${String(record.version)}`);
  }
  if (record.observationSchemaVersion !== NON_CARD_OBSERVATION_SCHEMA_VERSION) {
    fail(lineNumber, "unsupported observation schema version");
  }
  const catalogues = requireObject(
    record.actionCatalogueVersions,
    "manifest.actionCatalogueVersions",
    lineNumber,
  );
  if (
    catalogues.taxReturn !== TAX_RETURN_ACTION_CATALOGUE_VERSION ||
    catalogues.revolution !== REVOLUTION_ACTION_CATALOGUE_VERSION
  ) {
    fail(lineNumber, "unsupported action catalogue version");
  }
  const dimensions = requireObject(
    record.featureDimensions,
    "manifest.featureDimensions",
    lineNumber,
  );
  const expectedDimensions = {
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
  };
  for (const [key, expected] of Object.entries(expectedDimensions)) {
    const actual = requireObject(dimensions[key], `manifest.featureDimensions.${key}`, lineNumber);
    for (const [field, value] of Object.entries(expected)) {
      if (actual[field] !== value) fail(lineNumber, `${key}.${field} dimension mismatch`);
    }
  }
  const collection = requireObject(record.collection, "manifest.collection", lineNumber);
  if (!Array.isArray(collection.playerCounts) || collection.playerCounts.length === 0) {
    fail(lineNumber, "manifest playerCounts must be non-empty");
  }
  const seenPlayers = new Set();
  for (const playerCount of collection.playerCounts) {
    requireInteger(playerCount, "manifest playerCount", lineNumber, 4);
    if (playerCount > 10 || seenPlayers.has(playerCount)) {
      fail(lineNumber, "manifest playerCounts must be unique values from 4 to 10");
    }
    seenPlayers.add(playerCount);
  }
  requireInteger(collection.episodesPerPlayerCount, "episodesPerPlayerCount", lineNumber, 1);
  requireInteger(collection.acts, "acts", lineNumber, 1);
  if (
    !Array.isArray(collection.decisionKinds) ||
    collection.decisionKinds.length === 0 ||
    collection.decisionKinds.some((decision) => !DECISIONS.includes(decision))
  ) {
    fail(lineNumber, "manifest decisionKinds are invalid");
  }
  return record;
}

function expectedActionShape(decision) {
  return decision === "tax-return"
    ? {
        catalogueVersion: TAX_RETURN_ACTION_CATALOGUE_VERSION,
        actionCount: TAX_RETURN_ACTION_COUNT,
        actionFeatures: TAX_RETURN_ACTION_FEATURE_COUNT,
        observationFeatures: TAX_RETURN_OBSERVATION_FEATURE_COUNT,
      }
    : {
        catalogueVersion: REVOLUTION_ACTION_CATALOGUE_VERSION,
        actionCount: REVOLUTION_ACTION_COUNT,
        actionFeatures: REVOLUTION_ACTION_FEATURE_COUNT,
        observationFeatures: REVOLUTION_OBSERVATION_FEATURE_COUNT,
      };
}

function expectedRevolutionFeatures(actionIndex, actorRole) {
  if (actionIndex === 0) return [1, 0, 0];
  return actorRole === "great-peon" ? [0, 0, 1] : [0, 1, 0];
}

function validateDecision(record, manifest, lineNumber) {
  requireObject(record, "decision record", lineNumber);
  if (record.type !== "counterfactual-decision") {
    fail(lineNumber, `unexpected record type ${String(record.type)}`);
  }
  if (!DECISIONS.includes(record.decision)) {
    fail(lineNumber, `unsupported decision ${String(record.decision)}`);
  }
  if (!manifest.collection.decisionKinds.includes(record.decision)) {
    fail(lineNumber, "decision was not requested by the manifest");
  }
  const shape = expectedActionShape(record.decision);
  if (record.observationSchemaVersion !== NON_CARD_OBSERVATION_SCHEMA_VERSION) {
    fail(lineNumber, "decision observation schema version mismatch");
  }
  if (record.actionCatalogueVersion !== shape.catalogueVersion) {
    fail(lineNumber, "decision action catalogue version mismatch");
  }
  requireNonEmptyString(record.sampleId, "sampleId", lineNumber);
  requireNonEmptyString(record.episodeId, "episodeId", lineNumber);
  requireNonEmptyString(record.actorId, "actorId", lineNumber);
  requireNonEmptyString(record.decisionKey, "decisionKey", lineNumber);
  requireInteger(record.matchSeed, "matchSeed", lineNumber);
  requireInteger(record.playerCount, "playerCount", lineNumber, 4);
  if (
    record.playerCount > 10 ||
    !manifest.collection.playerCounts.includes(record.playerCount)
  ) {
    fail(lineNumber, "playerCount is outside the manifest collection");
  }
  requireInteger(record.acts, "acts", lineNumber, 1);
  if (record.acts !== manifest.collection.acts) fail(lineNumber, "acts mismatch");
  requireInteger(record.round, "round", lineNumber, 1);
  if (record.round > record.acts) fail(lineNumber, "round exceeds acts");
  if (!ROLES.has(record.actorRole)) fail(lineNumber, "actorRole is invalid");

  if (
    !Array.isArray(record.observation) ||
    record.observation.length !== shape.observationFeatures ||
    record.observation.some((value) => !Number.isFinite(value))
  ) {
    fail(lineNumber, "observation has an invalid shape or non-finite value");
  }
  if (
    !Array.isArray(record.legalMask) ||
    record.legalMask.length !== shape.actionCount ||
    record.legalMask.some((value) => typeof value !== "boolean")
  ) {
    fail(lineNumber, "legalMask has an invalid shape");
  }
  const indicesFromMask = record.legalMask.flatMap((legal, actionIndex) =>
    legal ? [actionIndex] : [],
  );
  if (
    !Array.isArray(record.legalActionIndices) ||
    record.legalActionIndices.length !== indicesFromMask.length ||
    record.legalActionIndices.some((value, index) => value !== indicesFromMask[index])
  ) {
    fail(lineNumber, "legalActionIndices do not exactly match legalMask");
  }
  if (indicesFromMask.length === 0) fail(lineNumber, "decision has no legal action");
  if (!record.legalMask[record.baselineActionIndex]) {
    fail(lineNumber, "baselineActionIndex is illegal");
  }
  if (!record.legalMask[record.bestActionIndex]) {
    fail(lineNumber, "bestActionIndex is illegal");
  }
  requireInteger(record.targetSampleCount, "targetSampleCount", lineNumber, 1);
  if (!Array.isArray(record.actions) || record.actions.length !== indicesFromMask.length) {
    fail(lineNumber, "actions do not cover every legal action");
  }
  const pairing = requireObject(record.pairing, "pairing", lineNumber);
  const pairedWorldId = requireNonEmptyString(
    pairing.pairedWorldId,
    "pairing.pairedWorldId",
    lineNumber,
  );
  let centeredSum = 0;
  let probabilitySum = 0;
  let bestActionIndex = null;
  let bestMeanUtility = Number.NEGATIVE_INFINITY;
  for (let position = 0; position < record.actions.length; position += 1) {
    const action = requireObject(record.actions[position], `actions[${position}]`, lineNumber);
    const expectedIndex = indicesFromMask[position];
    if (action.actionIndex !== expectedIndex) {
      fail(lineNumber, "actions must be in complete ascending legal-index order");
    }
    if (
      !Array.isArray(action.actionFeatures) ||
      action.actionFeatures.length !== shape.actionFeatures ||
      action.actionFeatures.some((value) => !Number.isFinite(value))
    ) {
      fail(lineNumber, `action ${expectedIndex} has invalid features`);
    }
    const expectedFeatures =
      record.decision === "tax-return"
        ? TAX_RETURN_ACTION_FEATURES[expectedIndex]
        : expectedRevolutionFeatures(expectedIndex, record.actorRole);
    if (action.actionFeatures.some((value, index) => value !== expectedFeatures[index])) {
      fail(lineNumber, `action ${expectedIndex} features do not match the catalogue`);
    }
    if (action.pairedWorldId !== pairedWorldId) {
      fail(lineNumber, `action ${expectedIndex} pairedWorldId mismatch`);
    }
    requireFinite(action.meanUtility, `action ${expectedIndex} meanUtility`, lineNumber);
    requireFinite(action.centeredUtility, `action ${expectedIndex} centeredUtility`, lineNumber);
    requireFinite(
      action.softTargetProbability,
      `action ${expectedIndex} softTargetProbability`,
      lineNumber,
    );
    if (action.softTargetProbability < 0 || action.softTargetProbability > 1) {
      fail(lineNumber, `action ${expectedIndex} probability is outside [0,1]`);
    }
    centeredSum += action.centeredUtility;
    probabilitySum += action.softTargetProbability;
    if (action.meanUtility > bestMeanUtility) {
      bestMeanUtility = action.meanUtility;
      bestActionIndex = expectedIndex;
    }
  }
  if (!nearlyEqual(centeredSum, 0)) fail(lineNumber, "centered utilities do not sum to zero");
  if (!nearlyEqual(probabilitySum, 1)) fail(lineNumber, "soft target probabilities do not sum to one");
  if (record.bestActionIndex !== bestActionIndex) {
    fail(lineNumber, "bestActionIndex is not the canonical maximum-utility action");
  }
  const metadata = requireObject(record.metadata, "metadata", lineNumber);
  if (metadata.playerCount !== record.playerCount) fail(lineNumber, "metadata playerCount mismatch");
  if (record.decision === "tax-return") {
    if (metadata.returnCount !== 1 && metadata.returnCount !== 2) {
      fail(lineNumber, "tax returnCount must be 1 or 2");
    }
    for (const actionIndex of indicesFromMask) {
      if (TAX_RETURN_ACTION_CATALOGUE[actionIndex].ranks.length !== metadata.returnCount) {
        fail(lineNumber, "tax legal action disagrees with returnCount");
      }
    }
  }
  return {
    pairedWorldId,
    actionsByIndex: new Map(record.actions.map((action) => [action.actionIndex, action])),
  };
}

function createTaxSelectionTotals() {
  return {
    actions: 0,
    cards: 0,
    normalCards: 0,
    jokers: 0,
    normalRankSum: 0,
    normalStrengthSum: 0,
    strongestNormalRankSum: 0,
    strongestNormalRankActions: 0,
    weakestNormalRankSum: 0,
    weakestNormalRankActions: 0,
    rankCounts: Array.from({ length: 13 }, () => 0),
  };
}

function recordTaxSelection(totals, actionIndex) {
  const ranks = TAX_RETURN_ACTION_CATALOGUE[actionIndex].ranks;
  totals.actions += 1;
  totals.cards += ranks.length;
  const normals = ranks.filter((rank) => rank <= 12);
  for (const rank of ranks) {
    totals.rankCounts[rank - 1] += 1;
    if (rank === 13) {
      totals.jokers += 1;
    } else {
      totals.normalCards += 1;
      totals.normalRankSum += rank;
      totals.normalStrengthSum += (12 - rank) / 11;
    }
  }
  if (normals.length > 0) {
    totals.strongestNormalRankSum += Math.min(...normals);
    totals.strongestNormalRankActions += 1;
    totals.weakestNormalRankSum += Math.max(...normals);
    totals.weakestNormalRankActions += 1;
  }
}

function createAccumulator(dimensions) {
  return {
    dimensions,
    records: 0,
    legalActions: 0,
    baselineBest: 0,
    baselineUtilityOptimal: 0,
    oracleTieActions: 0,
    uniqueOracleBest: 0,
    baselineCentered: 0,
    lowestCentered: 0,
    oracleAdvantage: 0,
    targetEntropy: 0,
    normalizedTargetEntropy: 0,
    baselineTargetProbability: 0,
    maximumTargetProbability: 0,
    targetSamples: 0,
    singleWorldTargets: 0,
    actionCounts: {
      baseline: new Map(),
      best: new Map(),
      lowest: new Map(),
    },
    revolutionRecords: 0,
    revolutionDeclare: { baseline: 0, best: 0, lowest: 0 },
    tax: {
      records: 0,
      baseline: createTaxSelectionTotals(),
      best: createTaxSelectionTotals(),
      lowest: createTaxSelectionTotals(),
    },
    worldClusters: new Map(),
  };
}

function incrementActionCount(map, decision, actionIndex) {
  const key = `${decision}:${actionIndex}`;
  map.set(key, (map.get(key) ?? 0) + 1);
}

function addRecord(accumulator, record, validation) {
  const actions = validation.actionsByIndex;
  const baseline = actions.get(record.baselineActionIndex);
  const best = actions.get(record.bestActionIndex);
  const lowestIndex = record.legalActionIndices[0];
  const lowest = actions.get(lowestIndex);
  const maximumUtility = best.meanUtility;
  const entropy = record.actions.reduce(
    (sum, action) =>
      action.softTargetProbability > 0
        ? sum - action.softTargetProbability * Math.log(action.softTargetProbability)
        : sum,
    0,
  );
  accumulator.records += 1;
  accumulator.legalActions += record.actions.length;
  accumulator.baselineBest += Number(record.baselineActionIndex === record.bestActionIndex);
  accumulator.baselineUtilityOptimal += Number(nearlyEqual(baseline.meanUtility, maximumUtility));
  const oracleTieCount = record.actions.filter((action) =>
    nearlyEqual(action.meanUtility, maximumUtility),
  ).length;
  accumulator.oracleTieActions += oracleTieCount;
  accumulator.uniqueOracleBest += Number(oracleTieCount === 1);
  accumulator.baselineCentered += baseline.centeredUtility;
  accumulator.lowestCentered += lowest.centeredUtility;
  accumulator.oracleAdvantage += maximumUtility - baseline.meanUtility;
  accumulator.targetEntropy += entropy;
  accumulator.normalizedTargetEntropy +=
    record.actions.length === 1 ? 0 : entropy / Math.log(record.actions.length);
  accumulator.baselineTargetProbability += baseline.softTargetProbability;
  accumulator.maximumTargetProbability += Math.max(
    ...record.actions.map((action) => action.softTargetProbability),
  );
  accumulator.targetSamples += record.targetSampleCount;
  accumulator.singleWorldTargets += Number(record.targetSampleCount === 1);
  incrementActionCount(
    accumulator.actionCounts.baseline,
    record.decision,
    record.baselineActionIndex,
  );
  incrementActionCount(accumulator.actionCounts.best, record.decision, record.bestActionIndex);
  incrementActionCount(accumulator.actionCounts.lowest, record.decision, lowestIndex);
  if (record.decision === "revolution") {
    accumulator.revolutionRecords += 1;
    accumulator.revolutionDeclare.baseline += Number(
      record.baselineActionIndex === REVOLUTION_DECLARE_ACTION_INDEX,
    );
    accumulator.revolutionDeclare.best += Number(
      record.bestActionIndex === REVOLUTION_DECLARE_ACTION_INDEX,
    );
    accumulator.revolutionDeclare.lowest += Number(
      lowestIndex === REVOLUTION_DECLARE_ACTION_INDEX,
    );
  } else {
    accumulator.tax.records += 1;
    recordTaxSelection(accumulator.tax.baseline, record.baselineActionIndex);
    recordTaxSelection(accumulator.tax.best, record.bestActionIndex);
    recordTaxSelection(accumulator.tax.lowest, lowestIndex);
  }
  const cluster = accumulator.worldClusters.get(validation.pairedWorldId) ?? {
    sum: 0,
    count: 0,
  };
  cluster.sum += baseline.centeredUtility;
  cluster.count += 1;
  accumulator.worldClusters.set(validation.pairedWorldId, cluster);
}

function ratesFromActionCounts(map, total) {
  return [...map.entries()]
    .map(([key, count]) => {
      const separator = key.lastIndexOf(":");
      return {
        decision: key.slice(0, separator),
        actionIndex: Number(key.slice(separator + 1)),
        count,
        rate: count / total,
      };
    })
    .sort(
      (left, right) =>
        left.decision.localeCompare(right.decision) || left.actionIndex - right.actionIndex,
    );
}

function summarizeTaxSelection(totals) {
  if (totals.actions === 0) return null;
  return {
    selectedActions: totals.actions,
    selectedCards: totals.cards,
    meanCardsPerAction: totals.cards / totals.actions,
    meanNormalRank: totals.normalCards === 0 ? null : totals.normalRankSum / totals.normalCards,
    meanNormalRankStrength:
      totals.normalCards === 0 ? null : totals.normalStrengthSum / totals.normalCards,
    normalRankStrengthDefinition: "(12-rank)/11; rank 1 = 1, rank 12 = 0; jokers excluded",
    meanStrongestNormalRankPerEligibleAction:
      totals.strongestNormalRankActions === 0
        ? null
        : totals.strongestNormalRankSum / totals.strongestNormalRankActions,
    meanWeakestNormalRankPerEligibleAction:
      totals.weakestNormalRankActions === 0
        ? null
        : totals.weakestNormalRankSum / totals.weakestNormalRankActions,
    jokerCards: totals.jokers,
    jokerCardRate: totals.cards === 0 ? null : totals.jokers / totals.cards,
    rankCardRates: totals.rankCounts.map((count, index) => ({
      rank: index + 1,
      label: index === 12 ? "joker" : String(index + 1),
      count,
      rate: totals.cards === 0 ? null : count / totals.cards,
    })),
  };
}

function summarizeAccumulator(accumulator) {
  const count = accumulator.records;
  const worldMeans = [...accumulator.worldClusters.values()].map(
    (cluster) => cluster.sum / cluster.count,
  );
  const interval = confidenceInterval95(worldMeans);
  return {
    dimensions: accumulator.dimensions,
    records: count,
    pairedWorldClusters: worldMeans.length,
    meanLegalActionCount: accumulator.legalActions / count,
    baselineActionBestLabelAgreementRate: accumulator.baselineBest / count,
    baselineActionUtilityOptimalTieAgreementRate:
      accumulator.baselineUtilityOptimal / count,
    meanOracleMaximumUtilityTieCount: accumulator.oracleTieActions / count,
    uniqueOracleBestRate: accumulator.uniqueOracleBest / count,
    meanBaselineCenteredUtility: accumulator.baselineCentered / count,
    baselineCenteredUtilityWorldClustered95: {
      estimand: "equal-weighted mean of each paired world's within-group record mean",
      mean: interval.mean,
      low: interval.low,
      high: interval.high,
      clusters: interval.count,
      inference: {
        method: interval.method,
        sampleStandardDeviation: interval.sampleStandardDeviation,
        standardError: interval.standardError,
        criticalValue: interval.criticalValue,
      },
    },
    meanLowestIndexCenteredUtility: accumulator.lowestCentered / count,
    meanOracleAdvantageOverBaseline: accumulator.oracleAdvantage / count,
    meanTargetEntropyNats: accumulator.targetEntropy / count,
    meanNormalizedTargetEntropy: accumulator.normalizedTargetEntropy / count,
    meanBaselineSoftTargetProbability:
      accumulator.baselineTargetProbability / count,
    meanMaximumSoftTargetProbability:
      accumulator.maximumTargetProbability / count,
    meanTargetSampleCount: accumulator.targetSamples / count,
    singleWorldTargetRate: accumulator.singleWorldTargets / count,
    actionIndexRates: {
      baseline: ratesFromActionCounts(accumulator.actionCounts.baseline, count),
      best: ratesFromActionCounts(accumulator.actionCounts.best, count),
      lowestLegal: ratesFromActionCounts(accumulator.actionCounts.lowest, count),
    },
    revolutionDeclareRates:
      accumulator.revolutionRecords === 0
        ? null
        : {
            records: accumulator.revolutionRecords,
            baseline:
              accumulator.revolutionDeclare.baseline / accumulator.revolutionRecords,
            best: accumulator.revolutionDeclare.best / accumulator.revolutionRecords,
            lowestLegal:
              accumulator.revolutionDeclare.lowest / accumulator.revolutionRecords,
          },
    taxReturnRankStrength:
      accumulator.tax.records === 0
        ? null
        : {
            records: accumulator.tax.records,
            baseline: summarizeTaxSelection(accumulator.tax.baseline),
            best: summarizeTaxSelection(accumulator.tax.best),
            lowestLegal: summarizeTaxSelection(accumulator.tax.lowest),
          },
  };
}

function createGroupingStore() {
  return {
    overall: createAccumulator({}),
    byDecision: new Map(),
    byPlayerCount: new Map(),
    byRound: new Map(),
    byActorRole: new Map(),
    byReturnCount: new Map(),
    byDecisionPlayerCount: new Map(),
    byDecisionRound: new Map(),
    byDecisionActorRole: new Map(),
    cells: new Map(),
  };
}

function addToMap(map, dimensions, record, validation) {
  const key = JSON.stringify(dimensions);
  let accumulator = map.get(key);
  if (!accumulator) {
    accumulator = createAccumulator(dimensions);
    map.set(key, accumulator);
  }
  addRecord(accumulator, record, validation);
}

function addToGroupings(groupings, record, validation) {
  addRecord(groupings.overall, record, validation);
  addToMap(groupings.byDecision, { decision: record.decision }, record, validation);
  addToMap(groupings.byPlayerCount, { playerCount: record.playerCount }, record, validation);
  addToMap(groupings.byRound, { round: record.round }, record, validation);
  addToMap(groupings.byActorRole, { actorRole: record.actorRole }, record, validation);
  addToMap(
    groupings.byDecisionPlayerCount,
    { decision: record.decision, playerCount: record.playerCount },
    record,
    validation,
  );
  addToMap(
    groupings.byDecisionRound,
    { decision: record.decision, round: record.round },
    record,
    validation,
  );
  addToMap(
    groupings.byDecisionActorRole,
    { decision: record.decision, actorRole: record.actorRole },
    record,
    validation,
  );
  if (record.decision === "tax-return") {
    addToMap(
      groupings.byReturnCount,
      { decision: record.decision, returnCount: record.metadata.returnCount },
      record,
      validation,
    );
  }
  addToMap(
    groupings.cells,
    {
      decision: record.decision,
      playerCount: record.playerCount,
      round: record.round,
      actorRole: record.actorRole,
      returnCount:
        record.decision === "tax-return" ? record.metadata.returnCount : null,
    },
    record,
    validation,
  );
}

function summarizeMap(map) {
  return [...map.values()]
    .map(summarizeAccumulator)
    .sort((left, right) =>
      JSON.stringify(left.dimensions).localeCompare(JSON.stringify(right.dimensions), "en", {
        numeric: true,
      }),
    );
}

function validateSummary(summary, manifest, recomputed, contentHash, contentBytes, lineNumber) {
  requireObject(summary, "summary", lineNumber);
  if (summary.type !== "summary") fail(lineNumber, "final record must be a summary");
  if (summary.decisionsWritten !== recomputed.decisionsWritten) {
    fail(lineNumber, "summary decisionsWritten mismatch");
  }
  if (summary.actionEvaluations !== recomputed.actionEvaluations) {
    fail(lineNumber, "summary actionEvaluations mismatch");
  }
  requireInteger(summary.baselineMatches, "summary baselineMatches", lineNumber);
  requireInteger(summary.decisionsDiscovered, "summary decisionsDiscovered", lineNumber);
  if (summary.decisionsDiscovered < summary.decisionsWritten) {
    fail(lineNumber, "summary decisionsDiscovered is below decisionsWritten");
  }
  if (!summary.stoppedAtMaxDecisions) {
    const expectedMatches =
      manifest.collection.playerCounts.length *
      manifest.collection.episodesPerPlayerCount;
    if (summary.baselineMatches !== expectedMatches) {
      fail(lineNumber, "complete summary baselineMatches do not match the manifest plan");
    }
    if (
      manifest.collection.decisionKinds.length === DECISIONS.length &&
      summary.decisionsDiscovered !== summary.decisionsWritten
    ) {
      fail(lineNumber, "untruncated all-decision dataset omitted discovered decisions");
    }
  }
  const counts = requireObject(summary.counts, "summary.counts", lineNumber);
  const byDecision = requireObject(counts.byDecision, "summary.counts.byDecision", lineNumber);
  let discoveredByDecision = 0;
  let writtenByDecision = 0;
  let actionsByDecision = 0;
  for (const decision of DECISIONS) {
    const actual = requireObject(byDecision[decision], `summary count ${decision}`, lineNumber);
    const expected = recomputed.byDecision[decision];
    if (actual.written !== expected.written || actual.actionEvaluations !== expected.actions) {
      fail(lineNumber, `summary ${decision} counts mismatch`);
    }
    if (!Number.isSafeInteger(actual.discovered) || actual.discovered < actual.written) {
      fail(lineNumber, `summary ${decision} discovered count is invalid`);
    }
    discoveredByDecision += actual.discovered;
    writtenByDecision += actual.written;
    actionsByDecision += actual.actionEvaluations;
  }
  if (
    discoveredByDecision !== summary.decisionsDiscovered ||
    writtenByDecision !== summary.decisionsWritten ||
    actionsByDecision !== summary.actionEvaluations
  ) {
    fail(lineNumber, "summary decision subtotals do not match totals");
  }
  const byPlayerCount = requireObject(
    counts.byPlayerCount,
    "summary.counts.byPlayerCount",
    lineNumber,
  );
  let baselineMatchesByPlayer = 0;
  let writtenByPlayer = 0;
  let actionsByPlayer = 0;
  for (const playerCount of manifest.collection.playerCounts) {
    const actual = requireObject(
      byPlayerCount[playerCount],
      `summary player ${playerCount}`,
      lineNumber,
    );
    const expected = recomputed.byPlayerCount[playerCount] ?? { written: 0, actions: 0 };
    if (actual.decisionsWritten !== expected.written || actual.actionEvaluations !== expected.actions) {
      fail(lineNumber, `summary player ${playerCount} counts mismatch`);
    }
    if (!summary.stoppedAtMaxDecisions && actual.baselineMatches !== manifest.collection.episodesPerPlayerCount) {
      fail(lineNumber, `summary player ${playerCount} baselineMatches mismatch`);
    }
    baselineMatchesByPlayer += actual.baselineMatches;
    writtenByPlayer += actual.decisionsWritten;
    actionsByPlayer += actual.actionEvaluations;
  }
  if (
    baselineMatchesByPlayer !== summary.baselineMatches ||
    writtenByPlayer !== summary.decisionsWritten ||
    actionsByPlayer !== summary.actionEvaluations
  ) {
    fail(lineNumber, "summary player-count subtotals do not match totals");
  }
  const hashes = requireObject(summary.hashes, "summary.hashes", lineNumber);
  if (hashes.algorithm !== "sha256") fail(lineNumber, "summary hash algorithm is not sha256");
  if (hashes.contentBeforeSummary !== contentHash) {
    fail(lineNumber, "contentBeforeSummary SHA-256 mismatch");
  }
  if (hashes.contentBeforeSummaryBytes !== contentBytes) {
    fail(lineNumber, "contentBeforeSummary byte count mismatch");
  }
}

async function requireTrailingNewline(inputPath) {
  const info = await stat(inputPath);
  if (info.size === 0) fail(null, "input is empty");
  const handle = await open(inputPath, "r");
  try {
    const byte = Buffer.alloc(1);
    await handle.read(byte, 0, 1, info.size - 1);
    if (byte[0] !== 0x0a) fail(null, "NDJSON must end with a newline");
  } finally {
    await handle.close();
  }
  return info;
}

/**
 * Validate and diagnose a completed counterfactual NDJSON stream. Memory is
 * bounded by aggregate group/world cells and fixed action catalogues; decision
 * records and action arrays are released after each line.
 */
export async function diagnoseNonCardCounterfactualDataset(inputPath) {
  const resolvedInput = resolve(inputPath);
  const fileInfo = await requireTrailingNewline(resolvedInput);
  const input = createReadStream(resolvedInput, { encoding: "utf8" });
  const lines = createInterface({ input, crlfDelay: Infinity });
  const contentHasher = createHash("sha256");
  let contentBytes = 0;
  let lineNumber = 0;
  let manifest = null;
  let summary = null;
  const groupings = createGroupingStore();
  const recomputed = {
    decisionsWritten: 0,
    actionEvaluations: 0,
    byDecision: Object.fromEntries(DECISIONS.map((decision) => [decision, { written: 0, actions: 0 }])),
    byPlayerCount: {},
    pairedWorlds: new Set(),
  };

  for await (const line of lines) {
    lineNumber += 1;
    if (line.length === 0) fail(lineNumber, "blank lines are not allowed");
    let record;
    try {
      record = JSON.parse(line);
    } catch (error) {
      fail(lineNumber, `invalid JSON (${error.message})`);
    }
    if (summary !== null) fail(lineNumber, "records after the summary are not allowed");
    if (lineNumber === 1) {
      manifest = validateManifest(record, lineNumber);
      const bytes = Buffer.from(`${line}\n`, "utf8");
      contentHasher.update(bytes);
      contentBytes += bytes.length;
      continue;
    }
    if (record.type === "summary") {
      summary = record;
      continue;
    }
    if (manifest === null) fail(lineNumber, "missing manifest");
    const validation = validateDecision(record, manifest, lineNumber);
    const bytes = Buffer.from(`${line}\n`, "utf8");
    contentHasher.update(bytes);
    contentBytes += bytes.length;
    recomputed.decisionsWritten += 1;
    recomputed.actionEvaluations += record.actions.length;
    recomputed.byDecision[record.decision].written += 1;
    recomputed.byDecision[record.decision].actions += record.actions.length;
    const player = (recomputed.byPlayerCount[record.playerCount] ??= { written: 0, actions: 0 });
    player.written += 1;
    player.actions += record.actions.length;
    recomputed.pairedWorlds.add(validation.pairedWorldId);
    addToGroupings(groupings, record, validation);
  }
  if (manifest === null) fail(null, "missing manifest");
  if (summary === null) fail(null, "incomplete dataset: missing final summary");
  if (recomputed.decisionsWritten === 0) fail(null, "dataset contains no decision records");
  const contentSha256 = contentHasher.digest("hex");
  validateSummary(
    summary,
    manifest,
    recomputed,
    contentSha256,
    contentBytes,
    lineNumber,
  );

  return {
    format: REPORT_FORMAT,
    version: REPORT_VERSION,
    source: {
      path: resolvedInput,
      bytes: fileInfo.size,
      datasetFormat: manifest.format,
      datasetVersion: manifest.version,
      createdAt: manifest.createdAt,
      collection: manifest.collection,
    },
    validation: {
      complete: true,
      records: recomputed.decisionsWritten,
      actionEvaluations: recomputed.actionEvaluations,
      pairedWorlds: recomputed.pairedWorlds.size,
      contentBeforeSummaryBytes: contentBytes,
      contentBeforeSummarySha256: contentSha256,
      summaryCountsVerified: true,
      actionCoverageVerified: true,
      targetNormalizationVerified: true,
      memoryModel: "streaming records; O(group x paired-world + fixed action-catalogue) aggregates",
    },
    metrics: {
      overall: summarizeAccumulator(groupings.overall),
      byDecision: summarizeMap(groupings.byDecision),
      byPlayerCount: summarizeMap(groupings.byPlayerCount),
      byRound: summarizeMap(groupings.byRound),
      byActorRole: summarizeMap(groupings.byActorRole),
      byReturnCount: summarizeMap(groupings.byReturnCount),
      byDecisionPlayerCount: summarizeMap(groupings.byDecisionPlayerCount),
      byDecisionRound: summarizeMap(groupings.byDecisionRound),
      byDecisionActorRole: summarizeMap(groupings.byDecisionActorRole),
      cells: summarizeMap(groupings.cells),
    },
  };
}

export async function writeNonCardDiagnosticReportExclusive(report, outputPath) {
  const resolvedOutput = resolve(outputPath);
  await mkdir(dirname(resolvedOutput), { recursive: true });
  const handle = await open(resolvedOutput, "wx");
  try {
    await handle.writeFile(`${JSON.stringify(report, null, 2)}\n`, "utf8");
  } finally {
    await handle.close();
  }
  return resolvedOutput;
}
