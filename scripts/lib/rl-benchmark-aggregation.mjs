import {
  EVALUATION_ROLES,
  confidenceInterval95,
  createOutcomeTotals,
  evaluateEffectSizeGates,
  mergeOutcomeTotals,
  summarizeOutcome,
  summarizeRoleDifferenceAudit,
} from "../rl-evaluation-statistics.mjs";

export const BENCHMARK_SHARD_FORMAT = "dalmuti-model-benchmark-shard";
export const BENCHMARK_SHARD_VERSION = 1;

export function createBenchmarkRoleTotals() {
  return Object.fromEntries(
    EVALUATION_ROLES.map((role) => [
      role,
      {
        candidate: createOutcomeTotals(),
        normal: createOutcomeTotals(),
      },
    ]),
  );
}

function mergeRoleTotals(target, source) {
  for (const role of EVALUATION_ROLES) {
    mergeOutcomeTotals(target[role].candidate, source[role].candidate);
    mergeOutcomeTotals(target[role].normal, source[role].normal);
  }
}

function confidenceSummary(interval) {
  return { low: interval.low, high: interval.high };
}

function detailedConfidenceSummary(interval) {
  return {
    unit: "match",
    method: interval.method,
    clusters: interval.count,
    sampleStandardDeviation: interval.sampleStandardDeviation,
    standardError: interval.standardError,
    criticalValue: interval.criticalValue,
    low: interval.low,
    high: interval.high,
  };
}

function pairwiseSummary(candidateBefore, comparisons, rates) {
  const interval = confidenceInterval95(rates);
  return {
    candidateBefore,
    comparisons,
    rate: interval.mean,
    rawPairWeightedRate: candidateBefore / comparisons,
    confidence95: detailedConfidenceSummary(interval),
  };
}

function summarizeRoleTotals(
  roleTotals,
  roleDifferenceSamples,
  totalMatches,
  roleRegressionMargin,
) {
  return Object.fromEntries(
    EVALUATION_ROLES.map((role) => {
      const candidate = summarizeOutcome(roleTotals[role].candidate);
      const normal = summarizeOutcome(roleTotals[role].normal);
      return [
        role,
        {
          candidate,
          normal,
          meanChipDifference:
            candidate.meanChip === null || normal.meanChip === null
              ? null
              : candidate.meanChip - normal.meanChip,
          matchClusteredChipDifference: summarizeRoleDifferenceAudit(
            roleDifferenceSamples[role],
            {
              totalMatches,
              regressionMargin: roleRegressionMargin,
            },
          ),
        },
      ];
    }),
  );
}

function assertOutcomeTotals(value, label) {
  if (!value || typeof value !== "object") {
    throw new TypeError(`${label} must be outcome totals`);
  }
  for (const field of ["chips", "places", "firsts", "lasts", "seatActs"]) {
    if (!Number.isSafeInteger(value[field]) || value[field] < 0) {
      throw new TypeError(`${label}.${field} must be a non-negative integer`);
    }
  }
}

export function validateBenchmarkMatchRecord(record, expected = {}) {
  const label = expected.label ?? "benchmark match record";
  if (!record || typeof record !== "object") {
    throw new TypeError(`${label} must be an object`);
  }
  for (const [field, minimum] of [
    ["playerCount", 4],
    ["matchIndex", 0],
    ["seed", 1],
    ["decisions", 1],
  ]) {
    if (!Number.isSafeInteger(record[field]) || record[field] < minimum) {
      throw new TypeError(`${label}.${field} is invalid`);
    }
  }
  if (record.playerCount > 10) {
    throw new RangeError(`${label}.playerCount is invalid`);
  }
  if (
    expected.playerCount !== undefined &&
    record.playerCount !== expected.playerCount
  ) {
    throw new Error(`${label}.playerCount does not match its shard`);
  }
  if (
    expected.seed !== undefined &&
    record.seed !== expected.seed + record.playerCount * 1_000_000 + record.matchIndex
  ) {
    throw new Error(`${label}.seed does not match its deterministic schedule`);
  }
  assertOutcomeTotals(record.groups?.candidate, `${label}.groups.candidate`);
  assertOutcomeTotals(record.groups?.normal, `${label}.groups.normal`);
  const roleSums = {
    candidate: createOutcomeTotals(),
    normal: createOutcomeTotals(),
  };
  for (const role of EVALUATION_ROLES) {
    assertOutcomeTotals(
      record.roleTotals?.[role]?.candidate,
      `${label}.roleTotals.${role}.candidate`,
    );
    assertOutcomeTotals(
      record.roleTotals?.[role]?.normal,
      `${label}.roleTotals.${role}.normal`,
    );
    mergeOutcomeTotals(roleSums.candidate, record.roleTotals[role].candidate);
    mergeOutcomeTotals(roleSums.normal, record.roleTotals[role].normal);
    const difference = record.roleDifferences?.[role];
    if (difference !== null && !Number.isFinite(difference)) {
      throw new TypeError(`${label}.roleDifferences.${role} is invalid`);
    }
    const candidateRole = summarizeOutcome(record.roleTotals[role].candidate);
    const normalRole = summarizeOutcome(record.roleTotals[role].normal);
    const expectedDifference =
      candidateRole.meanChip === null || normalRole.meanChip === null
        ? null
        : candidateRole.meanChip - normalRole.meanChip;
    if (!Object.is(difference, expectedDifference)) {
      throw new Error(`${label}.roleDifferences.${role} does not match totals`);
    }
  }
  for (const group of ["candidate", "normal"]) {
    if (!isSameOutcomeTotals(roleSums[group], record.groups[group])) {
      throw new Error(`${label}.roleTotals do not add up to ${group} totals`);
    }
  }
  if (!Number.isFinite(record.meanChipDifference)) {
    throw new TypeError(`${label}.meanChipDifference is invalid`);
  }
  const candidate = summarizeOutcome(record.groups.candidate);
  const normal = summarizeOutcome(record.groups.normal);
  if (
    candidate.meanChip === null ||
    normal.meanChip === null ||
    record.meanChipDifference !== candidate.meanChip - normal.meanChip
  ) {
    throw new Error(`${label}.meanChipDifference does not match totals`);
  }
  for (const field of ["candidateBefore", "comparisons"]) {
    if (!Number.isSafeInteger(record.pairwise?.[field]) || record.pairwise[field] < 0) {
      throw new TypeError(`${label}.pairwise.${field} is invalid`);
    }
  }
  if (
    record.pairwise.comparisons < 1 ||
    record.pairwise.candidateBefore > record.pairwise.comparisons ||
    !Number.isFinite(record.pairwise.rate) ||
    record.pairwise.rate !==
      record.pairwise.candidateBefore / record.pairwise.comparisons
  ) {
    throw new Error(`${label}.pairwise totals do not agree`);
  }
  return record;
}

function isSameOutcomeTotals(left, right) {
  return ["chips", "places", "firsts", "lasts", "seatActs"].every(
    (field) => left[field] === right[field],
  );
}

export function summarizeBenchmarkMatchRecords(
  records,
  {
    playerCount,
    playerCounts,
    acts,
    promotionThresholds,
    roleRegressionMargin,
  },
) {
  if (!Array.isArray(records) || records.length < 1) {
    throw new RangeError("benchmark aggregation needs at least one match record");
  }
  const groups = {
    candidate: createOutcomeTotals(),
    normal: createOutcomeTotals(),
  };
  const roleTotals = createBenchmarkRoleTotals();
  const roleDifferenceSamples = Object.fromEntries(
    EVALUATION_ROLES.map((role) => [role, []]),
  );
  const differences = [];
  const pairwiseRates = [];
  let candidateBefore = 0;
  let comparisons = 0;
  let decisions = 0;
  for (const [index, record] of records.entries()) {
    validateBenchmarkMatchRecord(record, {
      label: `benchmark match record ${index + 1}`,
      ...(playerCount === undefined ? {} : { playerCount }),
    });
    mergeOutcomeTotals(groups.candidate, record.groups.candidate);
    mergeOutcomeTotals(groups.normal, record.groups.normal);
    mergeRoleTotals(roleTotals, record.roleTotals);
    for (const role of EVALUATION_ROLES) {
      if (record.roleDifferences[role] !== null) {
        roleDifferenceSamples[role].push(record.roleDifferences[role]);
      }
    }
    differences.push(record.meanChipDifference);
    pairwiseRates.push(record.pairwise.rate);
    candidateBefore += record.pairwise.candidateBefore;
    comparisons += record.pairwise.comparisons;
    decisions += record.decisions;
  }
  const differenceInterval = confidenceInterval95(differences);
  const roles = summarizeRoleTotals(
    roleTotals,
    roleDifferenceSamples,
    records.length,
    roleRegressionMargin,
  );
  const result = {
    ...(playerCount === undefined
      ? { playerCounts: [...playerCounts] }
      : { playerCount }),
    matches: records.length,
    actsPerMatch: acts,
    decisions,
    candidate: summarizeOutcome(groups.candidate),
    normal: summarizeOutcome(groups.normal),
    meanChipDifference: differenceInterval.mean,
    meanChipDifference95: confidenceSummary(differenceInterval),
    meanChipDifferenceInference: detailedConfidenceSummary(differenceInterval),
    pairwiseCandidateBeforeNormal: pairwiseSummary(
      candidateBefore,
      comparisons,
      pairwiseRates,
    ),
    roles,
    statisticallyAboveNormal: differenceInterval.low > 0,
  };
  result.effectSizeGate = evaluateEffectSizeGates(result, promotionThresholds);
  result.roleRegressionAuditPassed = EVALUATION_ROLES.every(
    (role) => roles[role].matchClusteredChipDifference.auditPassed,
  );
  return result;
}

export function benchmarkShardMatchIndexes(totalMatches, shardIndex, shardCount) {
  if (!Number.isSafeInteger(totalMatches) || totalMatches < 1) {
    throw new RangeError("totalMatches must be a positive integer");
  }
  if (!Number.isSafeInteger(shardCount) || shardCount < 1) {
    throw new RangeError("shardCount must be a positive integer");
  }
  if (
    !Number.isSafeInteger(shardIndex) ||
    shardIndex < 0 ||
    shardIndex >= shardCount
  ) {
    throw new RangeError("shardIndex must be from zero to shardCount - 1");
  }
  const result = [];
  for (let matchIndex = shardIndex; matchIndex < totalMatches; matchIndex += shardCount) {
    result.push(matchIndex);
  }
  if (result.length < 1) {
    throw new RangeError("every benchmark shard must contain at least one match");
  }
  return result;
}
