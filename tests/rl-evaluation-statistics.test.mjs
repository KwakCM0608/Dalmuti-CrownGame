import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  FINAL_MATCH_COUNTS,
  candidateBeforeNormal,
  confidenceInterval95,
  createOutcomeTotals,
  evaluateEffectSizeGates,
  parseMatchCounts,
  recordOutcome,
  roleForSeat,
  rotatingCandidateIds,
  summarizeRoleDifferenceAudit,
  summarizeOutcome,
} from "../scripts/rl-evaluation-statistics.mjs";

const OBSERVATION_FEATURES = 172;
const ACTION_COUNT = 506;

function createZeroActorCriticModel() {
  return {
    format: "dalmuti-actor-critic",
    version: 1,
    observationFeatures: OBSERVATION_FEATURES,
    actionCount: ACTION_COUNT,
    hiddenSizes: [1],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    trunkLayers: [
      {
        inFeatures: OBSERVATION_FEATURES,
        outFeatures: 1,
        weight: Array(OBSERVATION_FEATURES).fill(0),
        bias: [0],
      },
    ],
    policyLayer: {
      inFeatures: 1,
      outFeatures: ACTION_COUNT,
      weight: Array(ACTION_COUNT).fill(0),
      bias: Array(ACTION_COUNT).fill(0),
    },
    valueLayer: {
      inFeatures: 1,
      outFeatures: 1,
      weight: [0],
      bias: [0],
    },
  };
}

test("confidence intervals use Student-t for small match samples", () => {
  const interval = confidenceInterval95([1, 3]);

  assert.equal(interval.method, "student-t");
  assert.equal(interval.count, 2);
  assert.equal(interval.mean, 2);
  assert.ok(Math.abs(interval.standardError - 1) < 1e-12);
  assert.ok(Math.abs(interval.low - (2 - 12.7062047364)) < 1e-10);
  assert.ok(Math.abs(interval.high - (2 + 12.7062047364)) < 1e-10);
});

test("confidence intervals use a normal critical value at 30 matches", () => {
  const interval = confidenceInterval95(
    Array.from({ length: 30 }, () => 2),
  );

  assert.equal(interval.method, "normal");
  assert.equal(interval.count, 30);
  assert.equal(interval.low, 2);
  assert.equal(interval.high, 2);
});

test("pairwise rate counts every candidate-normal finishing comparison", () => {
  const result = candidateBeforeNormal(
    ["candidate-1", "normal-1", "normal-2", "candidate-2"],
    new Set(["candidate-1", "candidate-2"]),
  );

  assert.deepEqual(result, {
    candidateBefore: 2,
    comparisons: 4,
    rate: 0.5,
  });
});

test("cyclic candidate assignment balances odd tables over two cycles", () => {
  const appearances = new Map(
    Array.from({ length: 5 }, (_, index) => [`player-${index + 1}`, 0]),
  );
  for (let matchIndex = 0; matchIndex < 10; matchIndex += 1) {
    const candidateCount = matchIndex % 2 === 0 ? 2 : 3;
    for (const playerId of rotatingCandidateIds(
      5,
      candidateCount,
      matchIndex,
    )) {
      appearances.set(playerId, appearances.get(playerId) + 1);
    }
  }

  assert.deepEqual([...appearances.values()], [5, 5, 5, 5, 5]);
});

test("role and outcome helpers preserve the five social ranks", () => {
  assert.deepEqual(
    Array.from({ length: 6 }, (_, seat) => roleForSeat(seat, 6)),
    [
      "great-dalmuti",
      "lesser-dalmuti",
      "merchant",
      "merchant",
      "lesser-peon",
      "great-peon",
    ],
  );
  const totals = createOutcomeTotals();
  recordOutcome(totals, { chips: 4, place: 1, playerCount: 4 });
  recordOutcome(totals, { chips: 0, place: 4, playerCount: 4 });
  assert.deepEqual(summarizeOutcome(totals), {
    meanChip: 2,
    meanPlace: 2.5,
    firstRate: 0.5,
    lastRate: 0.5,
    seatActs: 2,
  });
});

test("match-count maps and inclusive effect-size gates are configurable", () => {
  assert.deepEqual(FINAL_MATCH_COUNTS, {
    4: 2500,
    5: 1700,
    6: 900,
    7: 600,
    8: 400,
    9: 400,
    10: 300,
  });
  assert.deepEqual(parseMatchCounts("4:2500,6:900", [4, 6]), {
    4: 2500,
    6: 900,
  });
  const gates = evaluateEffectSizeGates(
    {
      meanChipDifference: 0.25,
      meanChipDifference95: { low: 0.15, high: 0.35 },
      pairwiseCandidateBeforeNormal: { rate: 0.55 },
    },
    {
      minPointDifference: 0.25,
      minLowerBound: 0.15,
      minPairwiseRate: 0.55,
    },
  );

  assert.deepEqual(gates, {
    pointDifferencePassed: true,
    lowerBoundPassed: true,
    pairwiseRatePassed: true,
    passed: true,
  });
});

test("role audit uses matched match clusters and flags only evidenced material regression", () => {
  const regression = summarizeRoleDifferenceAudit(
    [-0.3, -0.3, -0.3],
    { totalMatches: 5, regressionMargin: 0.1 },
  );

  assert.equal(regression.clusters, 3);
  assert.deepEqual(regression.coverage, {
    matchedMatches: 3,
    totalMatches: 5,
    rate: 0.6,
  });
  assert.equal(regression.meanChipDifference, -0.3);
  assert.deepEqual(regression.confidence95, { low: -0.3, high: -0.3 });
  assert.equal(regression.statisticallyEvidencedMaterialRegression, true);
  assert.equal(regression.auditPassed, false);

  const boundary = summarizeRoleDifferenceAudit([-0.1, -0.1], {
    totalMatches: 2,
    regressionMargin: 0.1,
  });
  assert.equal(boundary.statisticallyEvidencedMaterialRegression, false);
  assert.equal(boundary.auditPassed, true);
});

test("role audit represents an unavailable role without treating it as regression", () => {
  const notApplicable = summarizeRoleDifferenceAudit([], {
    totalMatches: 4,
    regressionMargin: 0.1,
  });

  assert.deepEqual(notApplicable, {
    unit: "match",
    status: "not-applicable",
    clusters: 0,
    meanChipDifference: null,
    confidence95: null,
    inference: null,
    coverage: {
      matchedMatches: 0,
      totalMatches: 4,
      rate: 0,
    },
    regressionMargin: 0.1,
    statisticallyEvidencedMaterialRegression: false,
    auditPassed: true,
  });
});

test("benchmark computes role audit without match data and marks p4 merchant N/A", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-role-audit-"));
  const modelPath = join(directory, "model.json");
  const outputPath = join(directory, "report.json");
  await writeFile(
    modelPath,
    `${JSON.stringify(createZeroActorCriticModel())}\n`,
    "utf8",
  );
  const script = new URL(
    "../scripts/rl-benchmark-model.mjs",
    import.meta.url,
  );
  const result = spawnSync(
    process.execPath,
    [
      fileURLToPath(script),
      "--model",
      modelPath,
      "--players",
      "4",
      "--matches",
      "2",
      "--acts",
      "2",
      "--seed",
      "20260802",
      "--omit-match-data",
      "--role-regression-margin",
      "0.2",
      "--output",
      outputPath,
    ],
    { encoding: "utf8" },
  );

  assert.equal(result.status, 0, result.stderr || result.stdout);
  const report = JSON.parse(await readFile(outputPath, "utf8"));
  assert.equal(report.evaluationDesign.matchDataIncluded, false);
  assert.equal(report.roleRegressionMargin, 0.2);
  assert.equal(typeof report.roleRegressionAuditPassed, "boolean");
  assert.equal(Object.hasOwn(report.results[0], "matchSummaries"), false);
  const merchant =
    report.results[0].roles.merchant.matchClusteredChipDifference;
  assert.equal(merchant.status, "not-applicable");
  assert.equal(merchant.clusters, 0);
  assert.equal(merchant.meanChipDifference, null);
  assert.equal(merchant.confidence95, null);
  assert.deepEqual(merchant.coverage, {
    matchedMatches: 0,
    totalMatches: 2,
    rate: 0,
  });
  assert.equal(merchant.auditPassed, true);
  assert.equal(
    report.pooled.roles.merchant.matchClusteredChipDifference.status,
    "not-applicable",
  );
});

test("final evaluation refuses a subset of supported player counts", () => {
  const script = new URL(
    "../scripts/rl-benchmark-model.mjs",
    import.meta.url,
  );
  const result = spawnSync(
    process.execPath,
    [
      fileURLToPath(script),
      "--model",
      "unused.json",
      "--final",
      "--players",
      "4",
      "--seed",
      "20260801",
    ],
    { encoding: "utf8" },
  );

  assert.notEqual(result.status, 0);
  assert.match(
    result.stderr,
    /--final requires every player count in order/,
  );
});
