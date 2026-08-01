import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { simulateMatch } from "../training/simulator.ts";
import {
  createGreedyInferenceTrainingPolicy,
} from "../training/stochastic-policy.ts";
import {
  createGreedyV3TrainingPolicy,
} from "../training/v3-stochastic-policy.ts";
import {
  EVALUATION_ROLES,
  FINAL_MATCH_COUNTS,
  candidateBeforeNormal,
  confidenceInterval95,
  createOutcomeTotals,
  evaluateEffectSizeGates,
  mergeOutcomeTotals,
  parseMatchCounts,
  recordOutcome,
  roleForSeat,
  rotatingCandidateIds,
  summarizeRoleDifferenceAudit,
  summarizeOutcome,
} from "./rl-evaluation-statistics.mjs";
import {
  createCandidateOnlyNonCardHooks,
  createNonCardRoutingTotals,
  loadRevolutionBenchmarkModel,
  loadTaxReturnBenchmarkModel,
  nonCardAblationName,
  nonCardSafetyGateProvenance,
  recordNonCardRouting,
  summarizeNonCardRoutingTotals,
} from "./lib/non-card-benchmark-policies.mjs";
import {
  BENCHMARK_SHARD_FORMAT,
  BENCHMARK_SHARD_VERSION,
  benchmarkShardMatchIndexes,
} from "./lib/rl-benchmark-aggregation.mjs";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const seedWasExplicit = cliArgs.some(
  (argument) => argument === "--seed" || argument.startsWith("--seed="),
);
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    "candidate-play": { type: "string", default: "model" },
    "tax-model": { type: "string" },
    "revolution-model": { type: "string" },
    "tax-min-advantage": { type: "string" },
    "revolution-min-advantage": { type: "string" },
    matches: { type: "string", default: "100" },
    "match-counts": { type: "string" },
    final: { type: "boolean", default: false },
    acts: { type: "string", default: "5" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    seed: { type: "string", default: "800001" },
    "min-point-diff": { type: "string", default: "0.25" },
    "min-lower-bound": { type: "string", default: "0.15" },
    "min-pairwise-rate": { type: "string", default: "0.55" },
    "role-regression-margin": { type: "string", default: "0.10" },
    "omit-match-data": { type: "boolean", default: false },
    "shard-index": { type: "string", default: "0" },
    "shard-count": { type: "string", default: "1" },
    output: { type: "string", short: "o" },
  },
});

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

function finiteNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new RangeError(`${label} must be a finite number`);
  }
  return parsed;
}

function createRoleTotals() {
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

function createRoleDifferenceSamples() {
  return Object.fromEntries(EVALUATION_ROLES.map((role) => [role, []]));
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
      const summary = {
        candidate,
        normal,
        meanChipDifference:
          candidate.meanChip === null || normal.meanChip === null
            ? null
            : candidate.meanChip - normal.meanChip,
      };
      if (roleDifferenceSamples) {
        summary.matchClusteredChipDifference = summarizeRoleDifferenceAudit(
          roleDifferenceSamples[role],
          {
            totalMatches,
            regressionMargin: roleRegressionMargin,
          },
        );
      }
      return [
        role,
        summary,
      ];
    }),
  );
}

function mergeRoleTotals(target, source) {
  for (const role of EVALUATION_ROLES) {
    mergeOutcomeTotals(target[role].candidate, source[role].candidate);
    mergeOutcomeTotals(target[role].normal, source[role].normal);
  }
}

function confidenceSummary(interval) {
  return {
    low: interval.low,
    high: interval.high,
  };
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

const candidatePlayMode = values["candidate-play"];
if (candidatePlayMode !== "model" && candidatePlayMode !== "normal") {
  throw new TypeError("--candidate-play must be model or normal");
}
if (candidatePlayMode === "model" && !values.model) {
  throw new TypeError("--model is required when --candidate-play is model");
}
if (candidatePlayMode === "normal" && values.model) {
  throw new TypeError(
    "--model must be omitted when --candidate-play is normal",
  );
}
if (
  candidatePlayMode === "normal" &&
  !values["tax-model"] &&
  !values["revolution-model"]
) {
  throw new TypeError(
    "--candidate-play normal requires --tax-model or --revolution-model",
  );
}
if (values["tax-min-advantage"] !== undefined && !values["tax-model"]) {
  throw new TypeError("--tax-min-advantage requires --tax-model");
}
if (
  values["revolution-min-advantage"] !== undefined &&
  !values["revolution-model"]
) {
  throw new TypeError(
    "--revolution-min-advantage requires --revolution-model",
  );
}
if (values.final && values["match-counts"]) {
  throw new TypeError("--final and --match-counts cannot be combined");
}
if (values.final && !seedWasExplicit) {
  throw new TypeError("--final requires an explicit fresh --seed");
}
const matches = positiveInteger(values.matches, "matches");
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
const shardCount = positiveInteger(values["shard-count"], "shard-count");
const shardIndex = Number(values["shard-index"]);
if (
  !Number.isSafeInteger(shardIndex) ||
  shardIndex < 0 ||
  shardIndex >= shardCount
) {
  throw new RangeError("shard-index must be from zero to shard-count - 1");
}
const distributedShardEnabled = shardCount > 1;
if (!distributedShardEnabled && shardIndex !== 0) {
  throw new RangeError("shard-index must be zero when shard-count is one");
}
if (values.final && distributedShardEnabled) {
  throw new TypeError("--final cannot be executed as an unmerged shard");
}
const playerCounts = values.players.split(",").map((value) =>
  positiveInteger(value.trim(), "players"),
);
if (
  playerCounts.length < 1 ||
  new Set(playerCounts).size !== playerCounts.length ||
  playerCounts.some((count) => count < 4 || count > 10)
) {
  throw new RangeError("players must be unique counts from 4 to 10");
}
if (
  values.final &&
  (playerCounts.length !== 7 ||
    playerCounts.some((count, index) => count !== index + 4))
) {
  throw new RangeError(
    "--final requires every player count in order: 4,5,6,7,8,9,10",
  );
}
const matchCountsByPlayerCount = values.final
  ? Object.fromEntries(
      playerCounts.map((playerCount) => [
        playerCount,
        FINAL_MATCH_COUNTS[playerCount],
      ]),
    )
  : values["match-counts"]
    ? parseMatchCounts(values["match-counts"], playerCounts)
    : Object.fromEntries(
        playerCounts.map((playerCount) => [playerCount, matches]),
      );
const promotionThresholds = {
  minPointDifference: finiteNumber(
    values["min-point-diff"],
    "min-point-diff",
  ),
  minLowerBound: finiteNumber(
    values["min-lower-bound"],
    "min-lower-bound",
  ),
  minPairwiseRate: finiteNumber(
    values["min-pairwise-rate"],
    "min-pairwise-rate",
  ),
};
const roleRegressionMargin = finiteNumber(
  values["role-regression-margin"],
  "role-regression-margin",
);
if (roleRegressionMargin < 0) {
  throw new RangeError("role-regression-margin must be non-negative");
}
const requestedTaxMinAdvantage =
  values["tax-min-advantage"] === undefined
    ? null
    : finiteNumber(
        values["tax-min-advantage"],
        "tax-min-advantage",
      );
const revolutionMinAdvantage = finiteNumber(
  values["revolution-min-advantage"] ?? "0",
  "revolution-min-advantage",
);
if (requestedTaxMinAdvantage !== null && requestedTaxMinAdvantage < 0) {
  throw new RangeError("tax-min-advantage must be non-negative");
}
if (revolutionMinAdvantage < 0) {
  throw new RangeError("revolution-min-advantage must be non-negative");
}
if (
  promotionThresholds.minPairwiseRate < 0 ||
  promotionThresholds.minPairwiseRate > 1
) {
  throw new RangeError("min-pairwise-rate must be between 0 and 1");
}

const modelPath =
  candidatePlayMode === "model" ? resolve(values.model) : null;
const modelBytes = modelPath === null ? null : await readFile(modelPath);
const modelValue =
  modelBytes === null ? null : JSON.parse(modelBytes.toString("utf8"));
const modelSha256 =
  modelBytes === null
    ? null
    : createHash("sha256").update(modelBytes).digest("hex");
if (modelPath !== null) {
  const filename = basename(modelPath);
  const expectedFormat =
    filename === "v3-actor-critic-weights.json"
      ? "dalmuti-action-conditioned-actor-critic"
      : filename === "actor-critic-weights.json"
        ? "dalmuti-actor-critic"
        : null;
  if (expectedFormat !== null && modelValue?.format !== expectedFormat) {
    throw new TypeError(
      `${filename} must contain ${expectedFormat}; refusing ambiguous renamed model`,
    );
  }
}
const candidatePolicy =
  modelPath === null
    ? null
    : modelValue.format === "dalmuti-action-conditioned-actor-critic"
      ? createGreedyV3TrainingPolicy(modelValue)
      : createGreedyInferenceTrainingPolicy(modelValue);
const taxReturnBenchmarkModel = values["tax-model"]
  ? await loadTaxReturnBenchmarkModel(values["tax-model"])
  : null;
const revolutionBenchmarkModel = values["revolution-model"]
  ? await loadRevolutionBenchmarkModel(values["revolution-model"])
  : null;
const taxMinAdvantage =
  requestedTaxMinAdvantage ??
  taxReturnBenchmarkModel?.defaultMinimumAdvantage ??
  0;
const nonCardEvaluationEnabled = Boolean(
  taxReturnBenchmarkModel || revolutionBenchmarkModel,
);
if (distributedShardEnabled && (modelPath === null || nonCardEvaluationEnabled)) {
  throw new TypeError(
    "distributed benchmark shards currently require --candidate-play model " +
      "without non-card ablations",
  );
}
const nonCardRouting = nonCardEvaluationEnabled
  ? createNonCardRoutingTotals()
  : null;
const normalCandidatePlayRouting =
  candidatePlayMode === "normal"
    ? {
        candidateNormalHeuristic: 0,
        normalNormalHeuristic: 0,
      }
    : null;
const results = [];
const internals = [];
const startedAt = performance.now();

for (const playerCount of playerCounts) {
  const plannedPlayerCountMatches = matchCountsByPlayerCount[playerCount];
  const matchIndexes = benchmarkShardMatchIndexes(
    plannedPlayerCountMatches,
    shardIndex,
    shardCount,
  );
  const playerCountMatches = matchIndexes.length;
  const groups = {
    candidate: createOutcomeTotals(),
    normal: createOutcomeTotals(),
  };
  const roleTotals = createRoleTotals();
  const roleDifferenceSamples = createRoleDifferenceSamples();
  const differences = [];
  const pairwiseRates = [];
  const matchSummaries = [];
  const shardMatchRecords = [];
  let pairwiseCandidateBefore = 0;
  let pairwiseComparisons = 0;
  let decisions = 0;
  for (const matchIndex of matchIndexes) {
    const lowerHalf = Math.floor(playerCount / 2);
    const candidateCount =
      playerCount % 2 === 0 || matchIndex % 2 === 0
        ? lowerHalf
        : lowerHalf + 1;
    const candidatePlayerIds = rotatingCandidateIds(
      playerCount,
      candidateCount,
      matchIndex,
    );
    const candidateIds = new Set(candidatePlayerIds);
    const normalPlayerIds = Array.from(
      { length: playerCount },
      (_, index) => `player-${index + 1}`,
    ).filter((playerId) => !candidateIds.has(playerId));
    const policyByPlayerId = candidatePolicy
      ? Object.fromEntries(
          candidatePlayerIds.map((playerId) => [playerId, candidatePolicy]),
        )
      : null;
    const nonCardDecisionTelemetry = [];
    const nonCard = createCandidateOnlyNonCardHooks({
      candidateIds,
      taxReturn: taxReturnBenchmarkModel,
      revolution: revolutionBenchmarkModel,
      taxMinAdvantage,
      revolutionMinAdvantage,
      decisionTelemetry: nonCardDecisionTelemetry,
    });
    const matchSeed = seed + playerCount * 1_000_000 + matchIndex;
    const match = simulateMatch({
      playerCount,
      acts,
      seed: matchSeed,
      episodeId: `benchmark-p${playerCount}-${matchIndex + 1}`,
      difficulties: ["normal"],
      ...(policyByPlayerId ? { policyByPlayerId } : {}),
      ...(nonCard ? { nonCard } : {}),
    });
    if (normalCandidatePlayRouting) {
      for (const step of match.steps) {
        if (step.behaviorPolicy !== "normal") {
          throw new Error(
            `candidate-play normal produced ${step.behaviorPolicy} play behavior`,
          );
        }
        if (candidateIds.has(step.actorId)) {
          normalCandidatePlayRouting.candidateNormalHeuristic += 1;
        } else {
          normalCandidatePlayRouting.normalNormalHeuristic += 1;
        }
      }
    }
    if (nonCardRouting) {
      recordNonCardRouting(
        nonCardRouting,
        match.nonCardSteps,
        candidateIds,
        {
          taxReturn: taxReturnBenchmarkModel,
          revolution: revolutionBenchmarkModel,
        },
        nonCardDecisionTelemetry,
      );
    }
    decisions += match.steps.length;
    const matchGroups = {
      candidate: createOutcomeTotals(),
      normal: createOutcomeTotals(),
    };
    const matchRoleTotals = createRoleTotals();
    let matchCandidateBefore = 0;
    let matchComparisons = 0;
    for (const act of match.acts) {
      const rolesByPlayerId = Object.fromEntries(
        act.playerOrder.map((playerId, seatIndex) => [
          playerId,
          roleForSeat(seatIndex, playerCount),
        ]),
      );
      act.finishOrder.forEach((playerId, index) => {
        const groupName = candidateIds.has(playerId)
          ? "candidate"
          : "normal";
        const outcome = {
          chips: act.chipAwards[playerId],
          place: index + 1,
          playerCount,
        };
        recordOutcome(groups[groupName], outcome);
        recordOutcome(matchGroups[groupName], outcome);
        const role = rolesByPlayerId[playerId];
        recordOutcome(roleTotals[role][groupName], outcome);
        recordOutcome(matchRoleTotals[role][groupName], outcome);
      });
      const pairwise = candidateBeforeNormal(
        act.finishOrder,
        candidateIds,
      );
      matchCandidateBefore += pairwise.candidateBefore;
      matchComparisons += pairwise.comparisons;
    }
    const candidateMatch = summarizeOutcome(matchGroups.candidate);
    const normalMatch = summarizeOutcome(matchGroups.normal);
    const meanChipDifference =
      candidateMatch.meanChip - normalMatch.meanChip;
    const pairwiseRate = matchCandidateBefore / matchComparisons;
    const matchRoles = summarizeRoleTotals(matchRoleTotals);
    const roleDifferences = Object.fromEntries(
      EVALUATION_ROLES.map((role) => {
        const candidateRole = matchRoles[role].candidate;
        const normalRole = matchRoles[role].normal;
        return [
          role,
          candidateRole.seatActs > 0 && normalRole.seatActs > 0
            ? candidateRole.meanChip - normalRole.meanChip
            : null,
        ];
      }),
    );
    for (const role of EVALUATION_ROLES) {
      if (roleDifferences[role] !== null) {
        roleDifferenceSamples[role].push(roleDifferences[role]);
      }
    }
    differences.push(meanChipDifference);
    pairwiseRates.push(pairwiseRate);
    pairwiseCandidateBefore += matchCandidateBefore;
    pairwiseComparisons += matchComparisons;
    if (distributedShardEnabled) {
      shardMatchRecords.push({
        playerCount,
        matchIndex,
        seed: matchSeed,
        decisions: match.steps.length,
        groups: matchGroups,
        roleTotals: matchRoleTotals,
        roleDifferences,
        meanChipDifference,
        pairwise: {
          candidateBefore: matchCandidateBefore,
          comparisons: matchComparisons,
          rate: pairwiseRate,
        },
      });
    }
    if (!values["omit-match-data"]) {
      matchSummaries.push({
        matchIndex: matchIndex + 1,
        seed: matchSeed,
        candidatePlayerIds,
        normalPlayerIds,
        candidate: candidateMatch,
        normal: normalMatch,
        meanChipDifference,
        pairwiseCandidateBeforeNormal: {
          candidateBefore: matchCandidateBefore,
          comparisons: matchComparisons,
          rate: pairwiseRate,
        },
        roles: matchRoles,
      });
    }
  }

  const differenceInterval = confidenceInterval95(differences);
  const result = {
    playerCount,
    matches: playerCountMatches,
    actsPerMatch: acts,
    decisions,
    candidate: summarizeOutcome(groups.candidate),
    normal: summarizeOutcome(groups.normal),
    meanChipDifference: differenceInterval.mean,
    meanChipDifference95: confidenceSummary(differenceInterval),
    meanChipDifferenceInference: detailedConfidenceSummary(
      differenceInterval,
    ),
    pairwiseCandidateBeforeNormal: pairwiseSummary(
      pairwiseCandidateBefore,
      pairwiseComparisons,
      pairwiseRates,
    ),
    roles: summarizeRoleTotals(
      roleTotals,
      roleDifferenceSamples,
      playerCountMatches,
      roleRegressionMargin,
    ),
    statisticallyAboveNormal: differenceInterval.low > 0,
  };
  result.effectSizeGate = evaluateEffectSizeGates(
    result,
    promotionThresholds,
  );
  result.roleRegressionAuditPassed = EVALUATION_ROLES.every(
    (role) => result.roles[role].matchClusteredChipDifference.auditPassed,
  );
  if (!values["omit-match-data"]) {
    result.matchSummaries = matchSummaries;
  }
  results.push(result);
  internals.push({
    groups,
    roleTotals,
    roleDifferenceSamples,
    differences,
    pairwiseRates,
    pairwiseCandidateBefore,
    pairwiseComparisons,
    shardMatchRecords,
  });
  console.log(
    `p${playerCount}: candidate ${result.candidate.meanChip.toFixed(4)} ` +
      `vs normal ${result.normal.meanChip.toFixed(4)} | ` +
      `diff ${result.meanChipDifference.toFixed(4)} ` +
      `[${result.meanChipDifference95.low.toFixed(4)}, ` +
      `${result.meanChipDifference95.high.toFixed(4)}] | ` +
      `before ${result.pairwiseCandidateBeforeNormal.rate.toFixed(4)} ` +
      `(${result.effectSizeGate.passed ? "PASS" : "FAIL"})`,
  );
}

const pooledGroups = {
  candidate: createOutcomeTotals(),
  normal: createOutcomeTotals(),
};
const pooledRoleTotals = createRoleTotals();
const pooledRoleDifferenceSamples = createRoleDifferenceSamples();
const pooledDifferences = [];
const pooledPairwiseRates = [];
let pooledCandidateBefore = 0;
let pooledComparisons = 0;
for (const internal of internals) {
  mergeOutcomeTotals(pooledGroups.candidate, internal.groups.candidate);
  mergeOutcomeTotals(pooledGroups.normal, internal.groups.normal);
  mergeRoleTotals(pooledRoleTotals, internal.roleTotals);
  for (const role of EVALUATION_ROLES) {
    pooledRoleDifferenceSamples[role].push(
      ...internal.roleDifferenceSamples[role],
    );
  }
  pooledDifferences.push(...internal.differences);
  pooledPairwiseRates.push(...internal.pairwiseRates);
  pooledCandidateBefore += internal.pairwiseCandidateBefore;
  pooledComparisons += internal.pairwiseComparisons;
}
const pooledDifferenceInterval = confidenceInterval95(pooledDifferences);
const pooled = {
  playerCounts,
  matches: pooledDifferences.length,
  actsPerMatch: acts,
  decisions: results.reduce((total, result) => total + result.decisions, 0),
  candidate: summarizeOutcome(pooledGroups.candidate),
  normal: summarizeOutcome(pooledGroups.normal),
  meanChipDifference: pooledDifferenceInterval.mean,
  meanChipDifference95: confidenceSummary(pooledDifferenceInterval),
  meanChipDifferenceInference: detailedConfidenceSummary(
    pooledDifferenceInterval,
  ),
  pairwiseCandidateBeforeNormal: pairwiseSummary(
    pooledCandidateBefore,
    pooledComparisons,
    pooledPairwiseRates,
  ),
  roles: summarizeRoleTotals(
    pooledRoleTotals,
    pooledRoleDifferenceSamples,
    pooledDifferences.length,
    roleRegressionMargin,
  ),
  statisticallyAboveNormal: pooledDifferenceInterval.low > 0,
};
pooled.effectSizeGate = evaluateEffectSizeGates(
  pooled,
  promotionThresholds,
);
pooled.roleRegressionAuditPassed = EVALUATION_ROLES.every(
  (role) => pooled.roles[role].matchClusteredChipDifference.auditPassed,
);

const executedMatchCountsByPlayerCount = Object.fromEntries(
  playerCounts.map((playerCount) => [
    playerCount,
    benchmarkShardMatchIndexes(
      matchCountsByPlayerCount[playerCount],
      shardIndex,
      shardCount,
    ).length,
  ]),
);
const uniformMatchCounts = new Set(
  Object.values(executedMatchCountsByPlayerCount),
).size === 1;
const report = {
  format: "dalmuti-model-benchmark",
  version: 2,
  ...(modelPath ? { modelPath } : {}),
  ...(modelSha256 ? { modelSha256 } : {}),
  ...(normalCandidatePlayRouting
    ? {
        candidatePlayPolicy: {
          mode: "normal",
          provenance: {
            implementation: "lib/bot-strategy.ts#chooseBotPlay",
            difficulty: "normal",
            appliesTo: ["candidate", "normal-control"],
          },
          routing: normalCandidatePlayRouting,
          validation:
            "every recorded play TrainingStep.behaviorPolicy was normal",
        },
      }
    : {}),
  seed,
  matchesPerPlayerCount: uniformMatchCounts
    ? Object.values(executedMatchCountsByPlayerCount)[0]
    : executedMatchCountsByPlayerCount,
  matchCountsByPlayerCount: executedMatchCountsByPlayerCount,
  actsPerMatch: acts,
  playerCounts,
  elapsedSeconds: (performance.now() - startedAt) / 1000,
  evaluationDesign: {
    confidenceLevel: 0.95,
    confidenceUnit: "match",
    smallSampleMethod: "student-t below 30 matches",
    largeSampleMethod: "normal at 30 or more matches",
    candidateSeatAssignment: "cyclically rotated by match",
    matchDataIncluded: !values["omit-match-data"],
    finalMatchCountPreset: values.final,
    seedSource: seedWasExplicit ? "cli" : "legacy-default",
  },
  promotionThresholds,
  roleRegressionMargin,
  roleRegressionRule:
    "Descriptive audit only: a role is materially regressed when the " +
    "match-clustered 95% confidence interval high bound is below " +
    "-roleRegressionMargin; unavailable roles pass as not applicable",
  promotionRule:
    "For every player count: mean chip difference >= minPointDifference, " +
    "match-clustered 95% lower bound >= minLowerBound, and candidate-before-normal " +
    "pairwise rate >= minPairwiseRate",
  promotionPassed: results.every(
    (result) => result.effectSizeGate.passed,
  ),
  roleRegressionAuditPassed:
    results.every((result) => result.roleRegressionAuditPassed) &&
    pooled.roleRegressionAuditPassed,
  ...(nonCardRouting
    ? {
        nonCardEvaluation: {
          ablation: nonCardAblationName(
            taxReturnBenchmarkModel,
            revolutionBenchmarkModel,
          ),
          candidateOnlyRouting: true,
          normalControlPolicy:
            "exact current normal tax-return and revolution heuristics",
          models: {
            taxReturn: taxReturnBenchmarkModel?.metadata ?? null,
            revolution: revolutionBenchmarkModel?.metadata ?? null,
          },
          safetyGate: nonCardSafetyGateProvenance({
            taxReturn: taxReturnBenchmarkModel,
            revolution: revolutionBenchmarkModel,
            taxMinAdvantage,
            revolutionMinAdvantage,
          }),
          routing: summarizeNonCardRoutingTotals(nonCardRouting),
        },
      }
    : {}),
  pooled,
  results,
  ...(distributedShardEnabled
    ? {
        distributedShard: {
          format: BENCHMARK_SHARD_FORMAT,
          version: BENCHMARK_SHARD_VERSION,
          strategy: "zero-based-match-index-modulo",
          shardIndex,
          shardCount,
          plannedMatchCountsByPlayerCount: matchCountsByPlayerCount,
          records: internals.flatMap((internal) => internal.shardMatchRecords),
        },
      }
    : {}),
};
console.log(
  `Promotion gate: ${report.promotionPassed ? "PASS" : "FAIL"} ` +
    `(${report.elapsedSeconds.toFixed(2)}s)`,
);
console.log(
  `Role regression audit: ${report.roleRegressionAuditPassed ? "PASS" : "FAIL"}`,
);
if (values.output) {
  const outputPath = resolve(values.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(
    outputPath,
    `${JSON.stringify(report, null, 2)}\n`,
    "utf8",
  );
  console.log(`Saved benchmark report to ${outputPath}`);
}
