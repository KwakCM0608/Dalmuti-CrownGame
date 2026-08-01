const NORMAL_95_CRITICAL_VALUE = 1.959963984540054;

// Two-sided 95% Student-t critical values for 1 through 29 degrees of freedom.
const STUDENT_T_95_CRITICAL_VALUES = [
  undefined,
  12.7062047364,
  4.30265272975,
  3.18244630528,
  2.7764451052,
  2.57058183564,
  2.44691185114,
  2.36462425101,
  2.30600413503,
  2.26215716285,
  2.22813885196,
  2.20098516008,
  2.17881282966,
  2.16036865646,
  2.14478668792,
  2.13144954556,
  2.11990529922,
  2.10981557783,
  2.10092204024,
  2.09302405441,
  2.08596344727,
  2.07961384473,
  2.0738730679,
  2.06865761042,
  2.06389856163,
  2.05953855275,
  2.05552943864,
  2.05183051648,
  2.0484071418,
  2.04522964213,
];

export const FINAL_MATCH_COUNTS = Object.freeze({
  4: 2500,
  5: 1700,
  6: 900,
  7: 600,
  8: 400,
  9: 400,
  10: 300,
});

export const EVALUATION_ROLES = Object.freeze([
  "great-dalmuti",
  "lesser-dalmuti",
  "merchant",
  "lesser-peon",
  "great-peon",
]);

function requireFiniteNumber(value, label) {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${label} must be finite`);
  }
  return value;
}

export function confidenceInterval95(samples) {
  if (!Array.isArray(samples) || samples.length < 1) {
    throw new RangeError("confidence interval needs at least one sample");
  }
  samples.forEach((sample, index) =>
    requireFiniteNumber(sample, `samples[${index}]`),
  );
  const count = samples.length;
  const mean = samples.reduce((total, sample) => total + sample, 0) / count;
  if (count === 1) {
    return {
      count,
      mean,
      sampleStandardDeviation: 0,
      standardError: 0,
      criticalValue: null,
      method: "degenerate",
      low: mean,
      high: mean,
    };
  }
  const variance =
    samples.reduce(
      (total, sample) => total + (sample - mean) ** 2,
      0,
    ) /
    (count - 1);
  const sampleStandardDeviation = Math.sqrt(variance);
  const standardError = sampleStandardDeviation / Math.sqrt(count);
  const usesStudentT = count < 30;
  const criticalValue = usesStudentT
    ? STUDENT_T_95_CRITICAL_VALUES[count - 1]
    : NORMAL_95_CRITICAL_VALUE;
  const margin = criticalValue * standardError;
  return {
    count,
    mean,
    sampleStandardDeviation,
    standardError,
    criticalValue,
    method: usesStudentT ? "student-t" : "normal",
    low: mean - margin,
    high: mean + margin,
  };
}

export function summarizeRoleDifferenceAudit(
  samples,
  { totalMatches, regressionMargin = 0.1 },
) {
  if (!Array.isArray(samples)) {
    throw new TypeError("samples must be an array");
  }
  if (!Number.isInteger(totalMatches) || totalMatches < 1) {
    throw new RangeError("totalMatches must be a positive integer");
  }
  requireFiniteNumber(regressionMargin, "regressionMargin");
  if (regressionMargin < 0) {
    throw new RangeError("regressionMargin must be non-negative");
  }

  const clusters = samples.length;
  if (clusters > totalMatches) {
    throw new RangeError("samples cannot contain more than one cluster per match");
  }
  const coverage = {
    matchedMatches: clusters,
    totalMatches,
    rate: clusters / totalMatches,
  };
  if (clusters === 0) {
    return {
      unit: "match",
      status: "not-applicable",
      clusters,
      meanChipDifference: null,
      confidence95: null,
      inference: null,
      coverage,
      regressionMargin,
      statisticallyEvidencedMaterialRegression: false,
      auditPassed: true,
    };
  }

  const interval = confidenceInterval95(samples);
  const statisticallyEvidencedMaterialRegression =
    interval.high < -regressionMargin;
  return {
    unit: "match",
    status: "available",
    clusters,
    meanChipDifference: interval.mean,
    confidence95: {
      low: interval.low,
      high: interval.high,
    },
    inference: {
      method: interval.method,
      sampleStandardDeviation: interval.sampleStandardDeviation,
      standardError: interval.standardError,
      criticalValue: interval.criticalValue,
    },
    coverage,
    regressionMargin,
    statisticallyEvidencedMaterialRegression,
    auditPassed: !statisticallyEvidencedMaterialRegression,
  };
}

export function roleForSeat(seatIndex, playerCount) {
  if (!Number.isInteger(seatIndex) || seatIndex < 0) {
    throw new RangeError("seatIndex must be a non-negative integer");
  }
  if (!Number.isInteger(playerCount) || playerCount < 4 || seatIndex >= playerCount) {
    throw new RangeError("seatIndex must identify a player at the table");
  }
  if (seatIndex === 0) return "great-dalmuti";
  if (seatIndex === 1) return "lesser-dalmuti";
  if (seatIndex === playerCount - 2) return "lesser-peon";
  if (seatIndex === playerCount - 1) return "great-peon";
  return "merchant";
}

export function createOutcomeTotals() {
  return {
    chips: 0,
    places: 0,
    firsts: 0,
    lasts: 0,
    seatActs: 0,
  };
}

export function recordOutcome(totals, { chips, place, playerCount }) {
  requireFiniteNumber(chips, "chips");
  if (!Number.isInteger(place) || place < 1 || place > playerCount) {
    throw new RangeError("place must be valid for playerCount");
  }
  totals.chips += chips;
  totals.places += place;
  totals.seatActs += 1;
  if (place === 1) totals.firsts += 1;
  if (place === playerCount) totals.lasts += 1;
  return totals;
}

export function mergeOutcomeTotals(target, source) {
  target.chips += source.chips;
  target.places += source.places;
  target.firsts += source.firsts;
  target.lasts += source.lasts;
  target.seatActs += source.seatActs;
  return target;
}

export function summarizeOutcome(totals) {
  if (totals.seatActs === 0) {
    return {
      meanChip: null,
      meanPlace: null,
      firstRate: null,
      lastRate: null,
      seatActs: 0,
    };
  }
  return {
    meanChip: totals.chips / totals.seatActs,
    meanPlace: totals.places / totals.seatActs,
    firstRate: totals.firsts / totals.seatActs,
    lastRate: totals.lasts / totals.seatActs,
    seatActs: totals.seatActs,
  };
}

export function candidateBeforeNormal(finishOrder, candidateIds) {
  const candidateSet =
    candidateIds instanceof Set ? candidateIds : new Set(candidateIds);
  let candidateBefore = 0;
  let comparisons = 0;
  for (let left = 0; left < finishOrder.length; left += 1) {
    const leftIsCandidate = candidateSet.has(finishOrder[left]);
    for (let right = left + 1; right < finishOrder.length; right += 1) {
      const rightIsCandidate = candidateSet.has(finishOrder[right]);
      if (leftIsCandidate === rightIsCandidate) continue;
      comparisons += 1;
      if (leftIsCandidate) candidateBefore += 1;
    }
  }
  return {
    candidateBefore,
    comparisons,
    rate: comparisons === 0 ? null : candidateBefore / comparisons,
  };
}

export function rotatingCandidateIds(playerCount, candidateCount, matchIndex) {
  if (
    !Number.isInteger(candidateCount) ||
    candidateCount < 1 ||
    candidateCount >= playerCount
  ) {
    throw new RangeError("candidateCount must split the table into two groups");
  }
  if (!Number.isInteger(matchIndex) || matchIndex < 0) {
    throw new RangeError("matchIndex must be a non-negative integer");
  }
  const start = matchIndex % playerCount;
  return Array.from({ length: candidateCount }, (_, offset) =>
    `player-${((start + offset) % playerCount) + 1}`,
  );
}

export function parseMatchCounts(specification, playerCounts) {
  const entries = specification.split(",").map((entry) => entry.trim());
  if (entries.length === 0 || entries.some((entry) => entry.length === 0)) {
    throw new TypeError("match-counts must contain player:matches entries");
  }
  const parsed = {};
  for (const entry of entries) {
    const match = /^(\d+):(\d+)$/.exec(entry);
    if (!match) {
      throw new TypeError(`invalid match-counts entry: ${entry}`);
    }
    const playerCount = Number(match[1]);
    const matches = Number(match[2]);
    if (playerCount < 4 || playerCount > 10 || matches < 1) {
      throw new RangeError(`invalid match-counts entry: ${entry}`);
    }
    if (Object.hasOwn(parsed, playerCount)) {
      throw new RangeError(`duplicate match-counts player: ${playerCount}`);
    }
    parsed[playerCount] = matches;
  }
  for (const playerCount of playerCounts) {
    if (!Object.hasOwn(parsed, playerCount)) {
      throw new RangeError(`match-counts is missing player ${playerCount}`);
    }
  }
  return Object.fromEntries(
    playerCounts.map((playerCount) => [playerCount, parsed[playerCount]]),
  );
}

export function evaluateEffectSizeGates(
  result,
  { minPointDifference, minLowerBound, minPairwiseRate },
) {
  const pointDifferencePassed =
    result.meanChipDifference >= minPointDifference;
  const lowerBoundPassed =
    result.meanChipDifference95.low >= minLowerBound;
  const pairwiseRatePassed =
    result.pairwiseCandidateBeforeNormal.rate >= minPairwiseRate;
  return {
    pointDifferencePassed,
    lowerBoundPassed,
    pairwiseRatePassed,
    passed:
      pointDifferencePassed && lowerBoundPassed && pairwiseRatePassed,
  };
}
