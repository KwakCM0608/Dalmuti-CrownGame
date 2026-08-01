import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

import { chooseBotRevolution } from "../../lib/bot-strategy.ts";
import {
  REVOLUTION_DECLARE_ACTION_INDEX,
  REVOLUTION_DECLINE_ACTION_INDEX,
} from "../../training/non-card-action-space.ts";
import {
  NON_CARD_DETERMINIZATION_ALGORITHM,
  NON_CARD_DETERMINIZATION_ALGORITHM_VERSION,
  NON_CARD_DETERMINIZATION_CONTRACT_SHA256,
  NON_CARD_DETERMINIZATION_SCHEMA,
} from "./non-card-counterfactual-data.mjs";
import { confidenceInterval95 } from "../rl-evaluation-statistics.mjs";
import {
  EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
  REVOLUTION_MERCHANT_SOURCE_DATA,
  selectExperimentalMerchantRevolution,
} from "./experimental-revolution-merchant-candidate.mjs";

const EXPECTED_PLAYER_COUNTS = Object.freeze([4, 5, 6, 7, 8, 9, 10]);
const EXPECTED_ROLES = Object.freeze([
  "great-dalmuti",
  "lesser-dalmuti",
  "merchant",
  "lesser-peon",
  "great-peon",
]);
const EXPECTED_CANDIDATE_SEED_DERIVATION =
  "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,attempt)))";
const EXPECTED_CONTINUATION_SEED_DERIVATION =
  "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,continuationIndex,continuation)))";
const TOLERANCE = 1e-9;

function fail(label, message) {
  throw new Error(`${label}: ${message}`);
}

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(label, "must be an object");
  }
  return value;
}

function integer(value, label, minimum = Number.MIN_SAFE_INTEGER) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    fail(label, `must be a safe integer >= ${minimum}`);
  }
  return value;
}

function finite(value, label) {
  if (!Number.isFinite(value)) fail(label, "must be finite");
  return value;
}

function nearlyEqual(left, right) {
  return (
    Math.abs(left - right) <=
    TOLERANCE * Math.max(1, Math.abs(left), Math.abs(right))
  );
}

function exactArray(actual, expected, label) {
  if (
    !Array.isArray(actual) ||
    actual.length !== expected.length ||
    actual.some((value, index) => value !== expected[index])
  ) {
    fail(label, `expected ${JSON.stringify(expected)}`);
  }
}

function expectedRoleAt(seat, playerCount) {
  if (seat === 0) return "great-dalmuti";
  if (seat === 1) return "lesser-dalmuti";
  if (seat === playerCount - 2) return "lesser-peon";
  if (seat === playerCount - 1) return "great-peon";
  return "merchant";
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function canonicalInformationStateKey(record) {
  const material = {
    decision: record.decision,
    observationSchemaVersion: record.observationSchemaVersion,
    actionCatalogueVersion: record.actionCatalogueVersion,
    observation: record.observation,
    legalActionIndices: record.legalActionIndices,
    baselineActionIndex: record.baselineActionIndex,
    metadata: record.metadata,
  };
  return `sha256:${sha256(JSON.stringify(material))}`;
}

function handFromObservation(observation, label) {
  const cards = [];
  for (let rank = 1; rank <= 13; rank += 1) {
    const copies = rank === 13 ? 2 : rank;
    const rawCount = finite(
      observation[8 + rank - 1],
      `${label}.observation own-hand rank ${rank}`,
    );
    const count = Math.round(rawCount * copies);
    if (
      count < 0 ||
      count > copies ||
      !nearlyEqual(rawCount, count / copies)
    ) {
      fail(label, `rank ${rank} own-hand count is not physically representable`);
    }
    for (let copy = 0; copy < count; copy += 1) {
      cards.push({ id: `rank-${rank}-copy-${copy + 1}`, rank });
    }
  }
  return cards;
}

function reconstructedObservation(record, label) {
  if (!Array.isArray(record.observation) || record.observation.length !== 102) {
    fail(label, "revolution observation must contain 102 features");
  }
  for (const [index, value] of record.observation.entries()) {
    finite(value, `${label}.observation[${index}]`);
  }
  const expectedPlayerFeature = (record.playerCount - 4) / 6;
  const expectedRoundFeature = Math.min(1, (record.round - 1) / 19);
  if (!nearlyEqual(record.observation[0], expectedPlayerFeature)) {
    fail(label, "player-count feature mismatch");
  }
  if (!nearlyEqual(record.observation[1], expectedRoundFeature)) {
    fail(label, "round feature mismatch");
  }
  const roleIndex = EXPECTED_ROLES.indexOf(record.actorRole);
  if (roleIndex < 0) fail(label, `unknown actor role ${String(record.actorRole)}`);
  exactArray(
    record.observation.slice(3, 8),
    EXPECTED_ROLES.map((_, index) => Number(index === roleIndex)),
    `${label}.actorRole features`,
  );
  const hand = handFromObservation(record.observation, label);
  if (!nearlyEqual(record.observation[2], hand.length / 20)) {
    fail(label, "own-hand size feature mismatch");
  }
  if (record.metadata.actorHandCount !== hand.length) {
    fail(label, "metadata actorHandCount does not match encoded hand");
  }
  if (hand.filter((card) => card.rank === 13).length !== 2) {
    fail(label, "revolution actor does not hold both jokers");
  }

  const publicHandCounts = [];
  for (let relativeSeat = 0; relativeSeat < 10; relativeSeat += 1) {
    const offset = 21 + relativeSeat * 8;
    const slot = record.observation.slice(offset, offset + 8);
    if (relativeSeat >= record.playerCount) {
      exactArray(slot, [0, 0, 0, 0, 0, 0, 0, 0], `${label}.empty public slot`);
      continue;
    }
    if (slot[0] !== 1) fail(label, `public slot ${relativeSeat} is not occupied`);
    const handCount = Math.round(slot[1] * 20);
    if (handCount < 0 || handCount > 20 || !nearlyEqual(slot[1], handCount / 20)) {
      fail(label, `public slot ${relativeSeat} hand count is invalid`);
    }
    if (slot[2] <= -1 || slot[2] >= 1) {
      fail(label, `public slot ${relativeSeat} score feature is not invertible`);
    }
    const absoluteSeat = (record.actorSeat + relativeSeat) % record.playerCount;
    const expectedRole = expectedRoleAt(absoluteSeat, record.playerCount);
    const expectedRoleIndex = EXPECTED_ROLES.indexOf(expectedRole);
    exactArray(
      slot.slice(3),
      EXPECTED_ROLES.map((_, index) => Number(index === expectedRoleIndex)),
      `${label}.public slot ${relativeSeat} role`,
    );
    publicHandCounts[absoluteSeat] = handCount;
  }
  if (publicHandCounts[record.actorSeat] !== hand.length) {
    fail(label, "actor public hand count does not match private hand");
  }
  if (record.observation[101] !== Number(record.round > 1)) {
    fail(label, "tax-applies feature mismatch");
  }

  const players = Array.from({ length: record.playerCount }, (_, seat) => ({
    id: seat === record.actorSeat ? record.actorId : `public-seat-${seat}`,
    role: expectedRoleAt(seat, record.playerCount),
    handCount: publicHandCounts[seat],
    score: 0,
  }));
  return {
    actorId: record.actorId,
    hand,
    players,
    round: record.round,
  };
}

function validateManifest(manifest) {
  object(manifest, "manifest");
  if (
    manifest.type !== "manifest" ||
    manifest.format !== REVOLUTION_MERCHANT_SOURCE_DATA.format ||
    manifest.version !== REVOLUTION_MERCHANT_SOURCE_DATA.version
  ) {
    fail("manifest", "format/version mismatch");
  }
  if (manifest.observationSchemaVersion !== 1) {
    fail("manifest", "observation schema version must be 1");
  }
  if (
    manifest.actionCatalogueVersions?.revolution !== 1 ||
    manifest.featureDimensions?.revolution?.observation !== 102 ||
    manifest.featureDimensions?.revolution?.action !== 3 ||
    manifest.featureDimensions?.revolution?.catalogue !== 2
  ) {
    fail("manifest", "revolution catalogue/dimensions mismatch");
  }
  const collection = object(manifest.collection, "manifest.collection");
  exactArray(collection.playerCounts, EXPECTED_PLAYER_COUNTS, "manifest playerCounts");
  exactArray(collection.decisionKinds, ["revolution"], "manifest decisionKinds");
  if (
    collection.episodesPerPlayerCount !== 100 ||
    collection.acts !== 5 ||
    collection.initialSeed !== 733000001 ||
    collection.continuationPolicy !== "normal-deterministic" ||
    collection.resumeAllowed !== false
  ) {
    fail("manifest", "collection contract mismatch");
  }
  const determinization = object(
    collection.determinization,
    "manifest.collection.determinization",
  );
  if (
    determinization.worldCountPerInformationState !== 8 ||
    determinization.continuationCountPerHiddenWorld !== 1 ||
    determinization.rawContinuationEvaluationsPerInformationState !== 8 ||
    determinization.effectiveIndependentWorldsPerInformationState !== 8 ||
    determinization.standardErrorEstimable !== true ||
    determinization.originalReplayWorldIncluded !== true ||
    determinization.rootSeed !== 733100001 ||
    determinization.maxAttemptsPerResampledWorld !== 64 ||
    determinization.algorithm !== NON_CARD_DETERMINIZATION_ALGORITHM ||
    determinization.algorithmVersion !== NON_CARD_DETERMINIZATION_ALGORITHM_VERSION ||
    determinization.algorithmContractSha256 !==
      NON_CARD_DETERMINIZATION_CONTRACT_SHA256 ||
    determinization.candidateSeedDerivation !==
      EXPECTED_CANDIDATE_SEED_DERIVATION ||
    determinization.continuationSeedDerivation !==
      EXPECTED_CONTINUATION_SEED_DERIVATION
  ) {
    fail("manifest", "determinization contract mismatch");
  }
  if (
    manifest.determinizationSchema !== NON_CARD_DETERMINIZATION_SCHEMA ||
    manifest.groupSplitKey !== "canonicalInformationStateKey"
  ) {
    fail("manifest", "determinization schema/group key mismatch");
  }
  if (
    manifest.privacy?.opponentCardIdentitiesIncluded !== false ||
    manifest.privacy?.physicalCardIdsIncluded !== false ||
    manifest.privacy?.aggregateTargetsOnly !== true ||
    manifest.privacy?.distribution !== "restricted-training-only"
  ) {
    fail("manifest", "privacy contract mismatch");
  }
}

function validateAction(action, expectedIndex, actorRole, label) {
  object(action, label);
  if (action.actionIndex !== expectedIndex) {
    fail(label, `expected actionIndex ${expectedIndex}`);
  }
  const features =
    expectedIndex === REVOLUTION_DECLINE_ACTION_INDEX
      ? [1, 0, 0]
      : actorRole === "great-peon"
        ? [0, 0, 1]
        : [0, 1, 0];
  exactArray(action.actionFeatures, features, `${label}.actionFeatures`);
  for (const [name, statistic] of [
    ["pairedBaselineAdvantage", action.pairedBaselineAdvantage],
    ["pairedDecisionActBaselineAdvantage", action.pairedDecisionActBaselineAdvantage],
  ]) {
    object(statistic, `${label}.${name}`);
    if (
      statistic.count !== 8 ||
      statistic.standardErrorEstimable !== true ||
      !Number.isFinite(statistic.mean) ||
      !Number.isFinite(statistic.sampleStandardDeviation) ||
      !Number.isFinite(statistic.standardError) ||
      statistic.sampleStandardDeviation < 0 ||
      statistic.standardError < 0
    ) {
      fail(label, `${name} is invalid`);
    }
  }
}

function validateDecision(record, index, canonicalKeys) {
  const label = `decision ${index}`;
  object(record, label);
  if (
    record.type !== "counterfactual-decision" ||
    record.decision !== "revolution" ||
    record.observationSchemaVersion !== 1 ||
    record.actionCatalogueVersion !== 1 ||
    record.acts !== 5
  ) {
    fail(label, "decision contract mismatch");
  }
  integer(record.playerCount, `${label}.playerCount`, 4);
  integer(record.round, `${label}.round`, 1);
  integer(record.actorSeat, `${label}.actorSeat`, 0);
  if (
    record.playerCount > 10 ||
    record.round > 5 ||
    record.actorSeat >= record.playerCount ||
    record.actorRole !== expectedRoleAt(record.actorSeat, record.playerCount)
  ) {
    fail(label, "player/round/role coordinates are invalid");
  }
  if (record.metadata?.playerCount !== record.playerCount) {
    fail(label, "metadata playerCount mismatch");
  }
  exactArray(record.legalMask, [true, true], `${label}.legalMask`);
  exactArray(record.legalActionIndices, [0, 1], `${label}.legalActionIndices`);
  if (
    record.baselineActionIndex !== REVOLUTION_DECLINE_ACTION_INDEX &&
    record.baselineActionIndex !== REVOLUTION_DECLARE_ACTION_INDEX
  ) {
    fail(label, "baseline action is invalid");
  }
  const recomputedKey = canonicalInformationStateKey(record);
  if (record.canonicalInformationStateKey !== recomputedKey) {
    fail(label, "canonicalInformationStateKey mismatch");
  }
  if (canonicalKeys.has(recomputedKey)) {
    fail(label, "duplicate canonicalInformationStateKey");
  }
  canonicalKeys.add(recomputedKey);
  if (
    record.pairing?.canonicalInformationStateKey !== recomputedKey ||
    record.pairing?.continuationPolicy !== "normal-deterministic" ||
    record.pairing?.forcedOverrideNamespace !== "revolution" ||
    record.pairing?.rootActionCoverage !==
      "all-legal-actions-in-every-accepted-hidden-world" ||
    record.pairing?.continuationRngPairing !==
      "same-environment-stream-and-hidden-world-seed-for-every-root-action"
  ) {
    fail(label, "pairing contract mismatch");
  }
  const determinization = record.determinization;
  if (
    determinization?.worldCount !== 8 ||
    determinization?.continuationCount !== 1 ||
    determinization?.rawContinuationEvaluations !== 8 ||
    determinization?.effectiveIndependentWorlds !== 8 ||
    determinization?.standardErrorEstimable !== true ||
    determinization?.algorithm !== NON_CARD_DETERMINIZATION_ALGORITHM ||
    determinization?.algorithmVersion !== NON_CARD_DETERMINIZATION_ALGORITHM_VERSION ||
    determinization?.algorithmContractSha256 !==
      NON_CARD_DETERMINIZATION_CONTRACT_SHA256 ||
    determinization?.candidateSeedDerivation !==
      EXPECTED_CANDIDATE_SEED_DERIVATION ||
    determinization?.continuationSeedDerivation !==
      EXPECTED_CONTINUATION_SEED_DERIVATION ||
    determinization?.individualWorldUtilitiesIncluded !== false ||
    determinization?.distribution !== "restricted-training-only"
  ) {
    fail(label, "determinization record contract mismatch");
  }
  if (
    record.utility?.terminalDefinition !== "terminal-cumulative-chip-score" ||
    record.utility?.decisionActDefinition !== "centered-round-chip-award" ||
    record.utility?.pairedBaselineAdvantagesBeforeAggregation !== true ||
    record.targetSampleCount !== 8 ||
    record.forcedActionEvaluations !== 16 ||
    !Array.isArray(record.actions) ||
    record.actions.length !== 2
  ) {
    fail(label, "utility/action coverage contract mismatch");
  }
  validateAction(record.actions[0], 0, record.actorRole, `${label}.actions[0]`);
  validateAction(record.actions[1], 1, record.actorRole, `${label}.actions[1]`);
  const baseline = record.actions[record.baselineActionIndex];
  if (
    !nearlyEqual(baseline.pairedBaselineAdvantage.mean, 0) ||
    !nearlyEqual(baseline.pairedDecisionActBaselineAdvantage.mean, 0)
  ) {
    fail(label, "baseline action advantages are not zero");
  }

  const observation = reconstructedObservation(record, label);
  const actor = observation.players[record.actorSeat];
  const normal = chooseBotRevolution(
    {
      hand: observation.hand,
      role: actor.role,
      playerCount: record.playerCount,
    },
    "normal",
  );
  const expectedBaseline = normal.declare ? 1 : 0;
  if (record.baselineActionIndex !== expectedBaseline) {
    fail(label, "baselineActionIndex does not match current normal");
  }
  const candidate = selectExperimentalMerchantRevolution(observation);
  const selectedAction = record.actions[candidate.actionIndex];
  return {
    playerCount: record.playerCount,
    round: record.round,
    actorRole: record.actorRole,
    baselineActionIndex: record.baselineActionIndex,
    actionIndex: candidate.actionIndex,
    changedFromBaseline: candidate.changedFromBaseline,
    routing: candidate.routing,
    decisionActActualChipAdvantage:
      selectedAction.pairedDecisionActBaselineAdvantage.mean *
      REVOLUTION_MERCHANT_SOURCE_DATA.normalizedRewardToActualChipMultiplier,
    terminalCumulativeChipAdvantage:
      selectedAction.pairedBaselineAdvantage.mean,
  };
}

function summary(samples) {
  const interval = confidenceInterval95(samples);
  return {
    clusters: interval.count,
    unit: "canonical-information-state",
    mean: interval.mean,
    confidence95: { low: interval.low, high: interval.high },
    inference: {
      method: interval.method,
      sampleStandardDeviation: interval.sampleStandardDeviation,
      standardError: interval.standardError,
      criticalValue: interval.criticalValue,
    },
    positive: samples.filter((sample) => sample > 0).length,
    tied: samples.filter((sample) => sample === 0).length,
    negative: samples.filter((sample) => sample < 0).length,
  };
}

function summarizeRows(rows) {
  return {
    records: rows.length,
    changedFromBaseline: rows.filter((row) => row.changedFromBaseline).length,
    decisionActActualChipAdvantage: summary(
      rows.map((row) => row.decisionActActualChipAdvantage),
    ),
    terminalCumulativeChipAdvantage: summary(
      rows.map((row) => row.terminalCumulativeChipAdvantage),
    ),
  };
}

function validateSummaryRecord(record, rows, contentBytes, manifestAndDecisions) {
  object(record, "summary");
  if (
    record.type !== "summary" ||
    record.baselineMatches !== 700 ||
    record.decisionsWritten !== rows.length ||
    record.decisionsWritten !== 502 ||
    record.actionEvaluations !== rows.length * 16 ||
    record.stoppedAtMaxDecisions !== false
  ) {
    fail("summary", "top-level counts mismatch");
  }
  if (
    record.hashes?.algorithm !== "sha256" ||
    record.hashes?.contentBeforeSummaryBytes !== contentBytes ||
    record.hashes?.contentBeforeSummary !== sha256(manifestAndDecisions)
  ) {
    fail("summary", "content-before-summary hash mismatch");
  }
  for (const playerCount of EXPECTED_PLAYER_COUNTS) {
    const actual = rows.filter((row) => row.playerCount === playerCount).length;
    if (
      record.counts?.byPlayerCount?.[playerCount]?.baselineMatches !== 100 ||
      record.counts?.byPlayerCount?.[playerCount]?.decisionsWritten !== actual ||
      record.counts?.byPlayerCount?.[playerCount]?.actionEvaluations !== actual * 16
    ) {
      fail("summary", `p${playerCount} counts mismatch`);
    }
  }
}

export async function auditRevolutionMerchantSourceData(
  inputPath,
  { expectedSha256 = REVOLUTION_MERCHANT_SOURCE_DATA.sha256 } = {},
) {
  const path = resolve(inputPath);
  const bytes = await readFile(path);
  if (bytes.length === 0 || bytes.at(-1) !== 0x0a) {
    fail("dataset", "must be non-empty and end with a newline");
  }
  const fileSha256 = sha256(bytes);
  if (fileSha256 !== expectedSha256) {
    fail("dataset", `sha256 mismatch: ${fileSha256}`);
  }
  const sidecarPath = `${path}.sha256`;
  const sidecar = (await readFile(sidecarPath, "utf8")).trim();
  if (sidecar !== `${fileSha256}  ${basename(path)}`) {
    fail("dataset", "sha256 sidecar mismatch");
  }
  const text = bytes.toString("utf8");
  const lines = text.slice(0, -1).split("\n").map((line) =>
    line.endsWith("\r") ? line.slice(0, -1) : line,
  );
  if (lines.some((line) => line.length === 0)) {
    fail("dataset", "blank NDJSON lines are not allowed");
  }
  const records = lines.map((line, index) => {
    try {
      return JSON.parse(line);
    } catch (error) {
      throw new Error(`line ${index + 1}: invalid JSON`, { cause: error });
    }
  });
  if (records.length < 3) fail("dataset", "manifest/decisions/summary are required");
  validateManifest(records[0]);
  const canonicalKeys = new Set();
  const rows = records
    .slice(1, -1)
    .map((record, index) => validateDecision(record, index + 1, canonicalKeys));
  const contentText = `${lines.slice(0, -1).join("\n")}\n`;
  const contentBytes = Buffer.byteLength(contentText, "utf8");
  validateSummaryRecord(
    records.at(-1),
    rows,
    contentBytes,
    Buffer.from(contentText, "utf8"),
  );

  const eligible = rows.filter((row) => row.changedFromBaseline);
  const byPlayerCount = Object.fromEntries(
    EXPECTED_PLAYER_COUNTS.map((playerCount) => [
      playerCount,
      summarizeRows(rows.filter((row) => row.playerCount === playerCount)),
    ]),
  );
  const byRole = Object.fromEntries(
    EXPECTED_ROLES.map((role) => [
      role,
      summarizeRows(rows.filter((row) => row.actorRole === role)),
    ]),
  );
  const eligibleByPlayerCount = Object.fromEntries(
    EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS.map((playerCount) => [
      playerCount,
      summarizeRows(
        eligible.filter((row) => row.playerCount === playerCount),
      ),
    ]),
  );
  const evidenceSupportedCounts = Object.entries(eligibleByPlayerCount)
    .filter(
      ([, cell]) =>
        cell.decisionActActualChipAdvantage.confidence95.low > 0,
    )
    .map(([playerCount]) => Number(playerCount));
  exactArray(
    evidenceSupportedCounts,
    EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
    "evidence-supported player counts",
  );

  return {
    format: "dalmuti-experimental-merchant-revolution-data-audit",
    version: 1,
    trainingOnly: true,
    source: {
      path,
      bytes: bytes.length,
      sha256: fileSha256,
      sidecarPath,
      sidecarVerified: true,
      records: rows.length,
    },
    contract: {
      actionSemantics: {
        [REVOLUTION_DECLINE_ACTION_INDEX]: "decline",
        [REVOLUTION_DECLARE_ACTION_INDEX]: "declare",
      },
      normalBaselineRecomputed: true,
      canonicalInformationStateKeysRecomputed: true,
      canonicalInformationStateKeysUnique: true,
      observationAndRoleCoordinatesRecomputed: true,
      deterministicCandidate: {
        role: "merchant",
        playerCounts: [...EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS],
        otherCases: "exact-current-normal-fallback",
      },
      normalizedRewardToActualChipMultiplier:
        REVOLUTION_MERCHANT_SOURCE_DATA.normalizedRewardToActualChipMultiplier,
    },
    evidence: {
      allRecords: summarizeRows(rows),
      eligibleChangedRecords: summarizeRows(eligible),
      byPlayerCount,
      byRole,
      eligibleByPlayerCount,
      evidenceSupportedCounts,
      selectionRule:
        "merchant player-count cells whose exploratory information-state-clustered two-sided 95% CI lower bound for current-act actual-chip advantage is above zero",
      limitation:
        "the same completed determinization dataset selected and describes the rule; an independent fresh-seed paired simulator benchmark is required before any promotion claim",
    },
  };
}
