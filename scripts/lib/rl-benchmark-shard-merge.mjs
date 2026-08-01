import { isDeepStrictEqual } from "node:util";

import {
  BENCHMARK_SHARD_FORMAT,
  BENCHMARK_SHARD_VERSION,
  benchmarkShardMatchIndexes,
  summarizeBenchmarkMatchRecords,
  validateBenchmarkMatchRecord,
} from "./rl-benchmark-aggregation.mjs";

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sameContract(actual, expected, label) {
  if (stableJson(actual) !== stableJson(expected)) {
    throw new Error(`${label} does not match the merge plan`);
  }
}

function resultWithoutMatchSummaries(result) {
  const copy = structuredClone(result);
  delete copy.matchSummaries;
  return copy;
}

function validateReportEnvelope(entry, expected) {
  const { report } = entry;
  const label = entry.path ?? "benchmark shard";
  if (report?.format !== "dalmuti-model-benchmark" || report.version !== 2) {
    throw new TypeError(`${label}: unsupported benchmark report`);
  }
  const shard = report.distributedShard;
  if (
    shard?.format !== BENCHMARK_SHARD_FORMAT ||
    shard.version !== BENCHMARK_SHARD_VERSION ||
    shard.strategy !== "zero-based-match-index-modulo"
  ) {
    throw new TypeError(`${label}: unsupported distributed shard contract`);
  }
  if (report.modelSha256 !== expected.modelSha256) {
    throw new Error(`${label}: model SHA-256 does not match the merge plan`);
  }
  if (report.seed !== expected.seed || report.actsPerMatch !== expected.acts) {
    throw new Error(`${label}: seed or act count does not match the merge plan`);
  }
  sameContract(
    report.promotionThresholds,
    expected.promotionThresholds,
    `${label}: promotion thresholds`,
  );
  if (report.roleRegressionMargin !== expected.roleRegressionMargin) {
    throw new Error(`${label}: role regression margin does not match the merge plan`);
  }
  if (
    report.evaluationDesign?.finalMatchCountPreset !== false ||
    report.nonCardEvaluation ||
    report.candidatePlayPolicy
  ) {
    throw new Error(`${label}: only non-final play-model shards can be merged`);
  }
  if (
    !Array.isArray(report.playerCounts) ||
    report.playerCounts.length < 1 ||
    new Set(report.playerCounts).size !== report.playerCounts.length ||
    report.playerCounts.some((count) => !expected.playerCounts.includes(count))
  ) {
    throw new Error(`${label}: shard player counts do not match the merge plan`);
  }
  if (
    !Number.isSafeInteger(shard.shardCount) ||
    shard.shardCount < 2 ||
    !Number.isSafeInteger(shard.shardIndex) ||
    shard.shardIndex < 0 ||
    shard.shardIndex >= shard.shardCount
  ) {
    throw new Error(`${label}: invalid shard index/count`);
  }
  if (!Array.isArray(shard.records) || shard.records.length < 1) {
    throw new Error(`${label}: shard aggregation records are missing`);
  }
  const byPlayerCount = new Map();
  for (const [recordIndex, record] of shard.records.entries()) {
    validateBenchmarkMatchRecord(record, {
      label: `${label}: record ${recordIndex + 1}`,
      seed: expected.seed,
    });
    if (!report.playerCounts.includes(record.playerCount)) {
      throw new Error(`${label}: record player count is outside the shard envelope`);
    }
    if (record.matchIndex % shard.shardCount !== shard.shardIndex) {
      throw new Error(`${label}: record is assigned to the wrong modulo shard`);
    }
    const records = byPlayerCount.get(record.playerCount) ?? [];
    records.push(record);
    byPlayerCount.set(record.playerCount, records);
  }
  if (byPlayerCount.size !== report.playerCounts.length) {
    throw new Error(`${label}: shard is missing a player-count partition`);
  }
  for (const playerCount of report.playerCounts) {
    const plannedMatches = expected.matchCountsByPlayerCount[playerCount];
    if (
      shard.plannedMatchCountsByPlayerCount?.[playerCount] !== plannedMatches
    ) {
      throw new Error(`${label}: planned p${playerCount} match count is wrong`);
    }
    const expectedIndexes = benchmarkShardMatchIndexes(
      plannedMatches,
      shard.shardIndex,
      shard.shardCount,
    );
    const records = byPlayerCount.get(playerCount)
      .sort((left, right) => left.matchIndex - right.matchIndex);
    if (
      records.length !== expectedIndexes.length ||
      records.some((record, index) => record.matchIndex !== expectedIndexes[index])
    ) {
      throw new Error(`${label}: p${playerCount} shard has missing or extra match seeds`);
    }
    if (report.matchCountsByPlayerCount?.[playerCount] !== records.length) {
      throw new Error(`${label}: p${playerCount} executed match count is wrong`);
    }
    const recomputed = summarizeBenchmarkMatchRecords(records, {
      playerCount,
      acts: expected.acts,
      promotionThresholds: expected.promotionThresholds,
      roleRegressionMargin: expected.roleRegressionMargin,
    });
    const reportedResult = report.results?.find(
      (result) => result.playerCount === playerCount,
    );
    if (!isDeepStrictEqual(recomputed, resultWithoutMatchSummaries(reportedResult))) {
      throw new Error(`${label}: p${playerCount} summary does not match raw shard records`);
    }
  }
  return { shard, byPlayerCount };
}

export function mergeBenchmarkShardReports(entries, expected) {
  if (!Array.isArray(entries) || entries.length < 2) {
    throw new RangeError("at least two benchmark shard reports are required");
  }
  if (
    !Array.isArray(expected.playerCounts) ||
    expected.playerCounts.length < 1 ||
    new Set(expected.playerCounts).size !== expected.playerCounts.length
  ) {
    throw new TypeError("merge plan player counts must be unique");
  }
  const validated = entries.map((entry) => ({
    entry,
    ...validateReportEnvelope(entry, expected),
  }));
  const recordsByPlayerCount = new Map(
    expected.playerCounts.map((playerCount) => [playerCount, []]),
  );
  const shardKeysByPlayerCount = new Map(
    expected.playerCounts.map((playerCount) => [playerCount, new Set()]),
  );
  const shardCountsByPlayerCount = new Map();
  for (const item of validated) {
    for (const [playerCount, records] of item.byPlayerCount) {
      const previousCount = shardCountsByPlayerCount.get(playerCount);
      if (previousCount !== undefined && previousCount !== item.shard.shardCount) {
        throw new Error(`p${playerCount} mixes incompatible shard counts`);
      }
      shardCountsByPlayerCount.set(playerCount, item.shard.shardCount);
      const key = item.shard.shardIndex;
      const keys = shardKeysByPlayerCount.get(playerCount);
      if (keys.has(key)) {
        throw new Error(`p${playerCount} contains duplicate shard index ${key}`);
      }
      keys.add(key);
      recordsByPlayerCount.get(playerCount).push(...records);
    }
  }
  const results = [];
  const pooledRecords = [];
  for (const playerCount of expected.playerCounts) {
    const records = recordsByPlayerCount.get(playerCount);
    const expectedMatches = expected.matchCountsByPlayerCount[playerCount];
    records.sort((left, right) => left.matchIndex - right.matchIndex);
    if (
      records.length !== expectedMatches ||
      records.some((record, index) => record.matchIndex !== index)
    ) {
      throw new Error(`p${playerCount} has duplicate or missing deterministic match seeds`);
    }
    const shardCount = shardCountsByPlayerCount.get(playerCount);
    const shardIndexes = shardKeysByPlayerCount.get(playerCount);
    if (
      !Number.isSafeInteger(shardCount) ||
      shardIndexes.size !== shardCount ||
      Array.from({ length: shardCount }, (_, index) => index).some(
        (index) => !shardIndexes.has(index),
      )
    ) {
      throw new Error(`p${playerCount} has a missing modulo shard`);
    }
    results.push(summarizeBenchmarkMatchRecords(records, {
      playerCount,
      acts: expected.acts,
      promotionThresholds: expected.promotionThresholds,
      roleRegressionMargin: expected.roleRegressionMargin,
    }));
    pooledRecords.push(...records);
  }
  const pooled = summarizeBenchmarkMatchRecords(pooledRecords, {
    playerCounts: expected.playerCounts,
    acts: expected.acts,
    promotionThresholds: expected.promotionThresholds,
    roleRegressionMargin: expected.roleRegressionMargin,
  });
  const uniformMatchCounts =
    new Set(Object.values(expected.matchCountsByPlayerCount)).size === 1;
  const report = {
    format: "dalmuti-model-benchmark",
    version: 2,
    modelPath: expected.modelPath,
    modelSha256: expected.modelSha256,
    seed: expected.seed,
    matchesPerPlayerCount: uniformMatchCounts
      ? Object.values(expected.matchCountsByPlayerCount)[0]
      : expected.matchCountsByPlayerCount,
    matchCountsByPlayerCount: expected.matchCountsByPlayerCount,
    actsPerMatch: expected.acts,
    playerCounts: expected.playerCounts,
    elapsedSeconds: entries.reduce(
      (total, entry) => total + (Number(entry.report.elapsedSeconds) || 0),
      0,
    ),
    evaluationDesign: {
      confidenceLevel: 0.95,
      confidenceUnit: "match",
      smallSampleMethod: "student-t below 30 matches",
      largeSampleMethod: "normal at 30 or more matches",
      candidateSeatAssignment: "cyclically rotated by match",
      matchDataIncluded: false,
      finalMatchCountPreset: false,
      seedSource: "cli",
      distributedMerge: true,
    },
    promotionThresholds: expected.promotionThresholds,
    roleRegressionMargin: expected.roleRegressionMargin,
    roleRegressionRule:
      "Descriptive audit only: a role is materially regressed when the " +
      "match-clustered 95% confidence interval high bound is below " +
      "-roleRegressionMargin; unavailable roles pass as not applicable",
    promotionRule:
      "For every player count: mean chip difference >= minPointDifference, " +
      "match-clustered 95% lower bound >= minLowerBound, and candidate-before-normal " +
      "pairwise rate >= minPairwiseRate",
    promotionPassed: results.every((result) => result.effectSizeGate.passed),
    roleRegressionAuditPassed:
      results.every((result) => result.roleRegressionAuditPassed) &&
      pooled.roleRegressionAuditPassed,
    pooled,
    results,
    distributedMerge: {
      format: "dalmuti-model-benchmark-shard-merge",
      version: 1,
      strategy: "candidate x player-count x deterministic-match-seed shards",
      shardReports: validated
        .map(({ entry, shard }) => ({
          path: entry.path,
          bytes: entry.bytes,
          sha256: entry.sha256,
          playerCounts: entry.report.playerCounts,
          shardIndex: shard.shardIndex,
          shardCount: shard.shardCount,
        }))
        .sort((left, right) =>
          left.playerCounts.join(",").localeCompare(right.playerCounts.join(",")) ||
          left.shardIndex - right.shardIndex ||
          left.sha256.localeCompare(right.sha256)),
      seedCoverage:
        "every planned zero-based match index occurs exactly once; seed = " +
        "base seed + playerCount * 1,000,000 + match index",
    },
  };
  return report;
}
