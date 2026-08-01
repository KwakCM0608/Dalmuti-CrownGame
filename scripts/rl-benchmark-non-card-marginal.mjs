import { access, mkdir, open } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { simulateMatch } from "../training/simulator.ts";
import {
  confidenceInterval95,
  createOutcomeTotals,
  mergeOutcomeTotals,
  parseMatchCounts,
  recordOutcome,
  rotatingCandidateIds,
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

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const seedWasExplicit = cliArgs.some(
  (argument) => argument === "--seed" || argument.startsWith("--seed="),
);
const matchesWasExplicit = cliArgs.some(
  (argument) =>
    argument === "--matches" || argument.startsWith("--matches="),
);
const { values } = parseArgs({
  args: cliArgs,
  options: {
    "tax-model": { type: "string" },
    "revolution-model": { type: "string" },
    "tax-min-advantage": { type: "string" },
    "revolution-min-advantage": { type: "string" },
    matches: { type: "string", default: "100" },
    "match-counts": { type: "string" },
    acts: { type: "string", default: "5" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    seed: { type: "string" },
    "omit-match-data": { type: "boolean", default: false },
    output: { type: "string", short: "o" },
  },
  strict: true,
});

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

function finiteNonNegative(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new RangeError(`${label} must be a finite number`);
  }
  if (parsed < 0) {
    throw new RangeError(`${label} must be non-negative`);
  }
  return parsed;
}

function arraysEqual(left, right) {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
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

function pairedMetric(samples, betterDirection) {
  const interval = confidenceInterval95(samples);
  return {
    mean: interval.mean,
    confidence95: {
      low: interval.low,
      high: interval.high,
    },
    inference: detailedConfidenceSummary(interval),
    betterDirection,
  };
}

function createComparisonTotals() {
  return {
    interventionBetter: 0,
    tied: 0,
    interventionWorse: 0,
  };
}

function mergeComparisonTotals(target, source) {
  target.interventionBetter += source.interventionBetter;
  target.tied += source.tied;
  target.interventionWorse += source.interventionWorse;
  return target;
}

function summarizeComparisonTotals(totals) {
  const comparisons =
    totals.interventionBetter + totals.tied + totals.interventionWorse;
  const nonTies = totals.interventionBetter + totals.interventionWorse;
  return {
    ...totals,
    comparisons,
    interventionBetterRate:
      comparisons === 0 ? null : totals.interventionBetter / comparisons,
    tiedRate: comparisons === 0 ? null : totals.tied / comparisons,
    interventionWorseRate:
      comparisons === 0 ? null : totals.interventionWorse / comparisons,
    interventionBetterRateAmongNonTies:
      nonTies === 0 ? null : totals.interventionBetter / nonTies,
  };
}

function createMetricSamples() {
  return {
    chipDifference: [],
    finishPlaceDifference: [],
    firstRateDifference: [],
    lastRateDifference: [],
    finalScoreDifference: [],
  };
}

function appendMetricSamples(target, source) {
  for (const key of Object.keys(target)) {
    target[key].push(...source[key]);
  }
}

function summarizeMetrics(samples) {
  return {
    chipDifference: pairedMetric(samples.chipDifference, "positive"),
    finishPlaceDifference: pairedMetric(
      samples.finishPlaceDifference,
      "negative",
    ),
    firstRateDifference: pairedMetric(
      samples.firstRateDifference,
      "positive",
    ),
    lastRateDifference: pairedMetric(
      samples.lastRateDifference,
      "negative",
    ),
    finalScoreDifference: pairedMetric(
      samples.finalScoreDifference,
      "positive",
    ),
  };
}

function summarizePairedMatch(
  baseline,
  intervention,
  candidatePlayerIds,
) {
  const baselineTotals = createOutcomeTotals();
  const interventionTotals = createOutcomeTotals();
  const finishComparison = createComparisonTotals();
  let chipDifference = 0;
  let finishPlaceDifference = 0;
  let firstRateDifference = 0;
  let lastRateDifference = 0;
  const rounds = [];

  for (let actIndex = 0; actIndex < baseline.acts.length; actIndex += 1) {
    const baselineAct = baseline.acts[actIndex];
    const interventionAct = intervention.acts[actIndex];
    if (
      baselineAct.round !== interventionAct.round ||
      baselineAct.round !== actIndex + 1
    ) {
      throw new Error("paired simulations have misaligned act numbers");
    }
    const baselinePlaces = new Map(
      baselineAct.finishOrder.map((playerId, index) => [playerId, index + 1]),
    );
    const interventionPlaces = new Map(
      interventionAct.finishOrder.map((playerId, index) => [
        playerId,
        index + 1,
      ]),
    );
    const candidateOutcomes = [];
    for (const playerId of candidatePlayerIds) {
      const baselinePlace = baselinePlaces.get(playerId);
      const interventionPlace = interventionPlaces.get(playerId);
      const baselineChips = baselineAct.chipAwards[playerId];
      const interventionChips = interventionAct.chipAwards[playerId];
      if (
        baselinePlace === undefined ||
        interventionPlace === undefined ||
        !Number.isFinite(baselineChips) ||
        !Number.isFinite(interventionChips)
      ) {
        throw new Error(`paired simulation is missing outcome for ${playerId}`);
      }
      recordOutcome(baselineTotals, {
        chips: baselineChips,
        place: baselinePlace,
        playerCount: baseline.playerCount,
      });
      recordOutcome(interventionTotals, {
        chips: interventionChips,
        place: interventionPlace,
        playerCount: intervention.playerCount,
      });
      chipDifference += interventionChips - baselineChips;
      finishPlaceDifference += interventionPlace - baselinePlace;
      firstRateDifference +=
        Number(interventionPlace === 1) - Number(baselinePlace === 1);
      lastRateDifference +=
        Number(interventionPlace === intervention.playerCount) -
        Number(baselinePlace === baseline.playerCount);
      if (interventionPlace < baselinePlace) {
        finishComparison.interventionBetter += 1;
      } else if (interventionPlace > baselinePlace) {
        finishComparison.interventionWorse += 1;
      } else {
        finishComparison.tied += 1;
      }
      candidateOutcomes.push({
        playerId,
        baseline: { chips: baselineChips, place: baselinePlace },
        intervention: {
          chips: interventionChips,
          place: interventionPlace,
        },
        chipDifference: interventionChips - baselineChips,
        finishPlaceDifference: interventionPlace - baselinePlace,
      });
    }
    rounds.push({
      round: baselineAct.round,
      candidateOutcomes,
    });
  }

  const candidateSeatActs = candidatePlayerIds.length * baseline.acts.length;
  const finalScoreDifferences = Object.fromEntries(
    candidatePlayerIds.map((playerId) => [
      playerId,
      intervention.finalScores[playerId] - baseline.finalScores[playerId],
    ]),
  );
  const meanFinalScoreDifference =
    Object.values(finalScoreDifferences).reduce(
      (total, difference) => total + difference,
      0,
    ) / candidatePlayerIds.length;
  return {
    baselineTotals,
    interventionTotals,
    finishComparison,
    samples: {
      chipDifference: chipDifference / candidateSeatActs,
      finishPlaceDifference: finishPlaceDifference / candidateSeatActs,
      firstRateDifference: firstRateDifference / candidateSeatActs,
      lastRateDifference: lastRateDifference / candidateSeatActs,
      finalScoreDifference: meanFinalScoreDifference,
    },
    summary: {
      candidateSeatActs,
      baseline: summarizeOutcome(baselineTotals),
      intervention: summarizeOutcome(interventionTotals),
      chipDifference: chipDifference / candidateSeatActs,
      finishPlaceDifference: finishPlaceDifference / candidateSeatActs,
      firstRateDifference: firstRateDifference / candidateSeatActs,
      lastRateDifference: lastRateDifference / candidateSeatActs,
      meanFinalScoreDifference,
      finalScoreDifferences,
      finishComparison: summarizeComparisonTotals(finishComparison),
      rounds,
    },
  };
}

if (!seedWasExplicit) {
  throw new TypeError("--seed is required and must be explicit");
}
if (!values.output) {
  throw new TypeError("--output is required");
}
if (!values["tax-model"] && !values["revolution-model"]) {
  throw new TypeError("--tax-model or --revolution-model is required");
}
if (matchesWasExplicit && values["match-counts"]) {
  throw new TypeError("--matches and --match-counts cannot be combined");
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

const seed = positiveInteger(values.seed, "seed");
const matches = positiveInteger(values.matches, "matches");
const acts = positiveInteger(values.acts, "acts");
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
const matchCountsByPlayerCount = values["match-counts"]
  ? parseMatchCounts(values["match-counts"], playerCounts)
  : Object.fromEntries(
      playerCounts.map((playerCount) => [playerCount, matches]),
    );
const requestedTaxMinAdvantage =
  values["tax-min-advantage"] === undefined
    ? null
    : finiteNonNegative(
        values["tax-min-advantage"],
        "tax-min-advantage",
      );
const revolutionMinAdvantage = finiteNonNegative(
  values["revolution-min-advantage"] ?? "0",
  "revolution-min-advantage",
);
const outputPath = resolve(values.output);
await mkdir(dirname(outputPath), { recursive: true });
try {
  await access(outputPath);
  throw new Error(`output must not already exist: ${outputPath}`);
} catch (error) {
  if (error?.code !== "ENOENT") throw error;
}

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
const nonCardRouting = createNonCardRoutingTotals();
const baselineNonCardRouting = {
  taxReturn: {
    candidateNormalHeuristic: 0,
    otherNormalHeuristic: 0,
  },
  revolution: {
    candidateNormalHeuristic: 0,
    otherNormalHeuristic: 0,
  },
};
const playRouting = {
  baselineNormalHeuristic: 0,
  interventionNormalHeuristic: 0,
};
const results = [];
const internals = [];
const startedAt = performance.now();

for (const playerCount of playerCounts) {
  const playerCountMatches = matchCountsByPlayerCount[playerCount];
  const baselineTotals = createOutcomeTotals();
  const interventionTotals = createOutcomeTotals();
  const finishComparison = createComparisonTotals();
  const metricSamples = createMetricSamples();
  const matchSummaries = [];
  let candidateSeatActs = 0;
  let matchedInitialSeatOrders = 0;

  for (let matchIndex = 0; matchIndex < playerCountMatches; matchIndex += 1) {
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
    const matchSeed = seed + playerCount * 1_000_000 + matchIndex;
    if (!Number.isSafeInteger(matchSeed)) {
      throw new RangeError("derived match seed must be a safe integer");
    }
    const episodeId = `paired-noncard-p${playerCount}-${matchIndex + 1}`;
    const baseline = simulateMatch({
      playerCount,
      acts,
      seed: matchSeed,
      episodeId,
      difficulties: ["normal"],
      nonCard: {},
    });
    const decisionTelemetry = [];
    const nonCard = createCandidateOnlyNonCardHooks({
      candidateIds,
      taxReturn: taxReturnBenchmarkModel,
      revolution: revolutionBenchmarkModel,
      taxMinAdvantage,
      revolutionMinAdvantage,
      decisionTelemetry,
    });
    const intervention = simulateMatch({
      playerCount,
      acts,
      seed: matchSeed,
      episodeId,
      difficulties: ["normal"],
      nonCard,
    });

    if (
      baseline.seed !== intervention.seed ||
      baseline.episodeId !== intervention.episodeId ||
      baseline.playerCount !== intervention.playerCount ||
      baseline.acts.length !== intervention.acts.length
    ) {
      throw new Error("paired simulations drifted from their shared config");
    }
    if (
      !arraysEqual(
        baseline.acts[0].playerOrder,
        intervention.acts[0].playerOrder,
      )
    ) {
      throw new Error("paired simulations did not start from the same seats");
    }
    matchedInitialSeatOrders += 1;
    for (const step of baseline.steps) {
      if (step.behaviorPolicy !== "normal") {
        throw new Error("baseline card play was not current normal");
      }
      playRouting.baselineNormalHeuristic += 1;
    }
    for (const step of intervention.steps) {
      if (step.behaviorPolicy !== "normal") {
        throw new Error("intervention card play was not current normal");
      }
      playRouting.interventionNormalHeuristic += 1;
    }
    for (const step of baseline.nonCardSteps ?? []) {
      if (
        step.behaviorPolicy !== "normal" ||
        step.behaviorPolicyVersion !== null ||
        step.forcedOverride
      ) {
        throw new Error("baseline non-card behavior was not current normal");
      }
      const group = baselineNonCardRouting[
        step.decision === "tax-return"
          ? "taxReturn"
          : step.decision === "revolution"
            ? "revolution"
            : ""
      ];
      if (!group) {
        throw new TypeError(`unknown baseline decision ${String(step.decision)}`);
      }
      if (candidateIds.has(step.actorId)) {
        group.candidateNormalHeuristic += 1;
      } else {
        group.otherNormalHeuristic += 1;
      }
    }
    recordNonCardRouting(
      nonCardRouting,
      intervention.nonCardSteps,
      candidateIds,
      {
        taxReturn: taxReturnBenchmarkModel,
        revolution: revolutionBenchmarkModel,
      },
      decisionTelemetry,
    );

    const paired = summarizePairedMatch(
      baseline,
      intervention,
      candidatePlayerIds,
    );
    mergeOutcomeTotals(baselineTotals, paired.baselineTotals);
    mergeOutcomeTotals(interventionTotals, paired.interventionTotals);
    mergeComparisonTotals(finishComparison, paired.finishComparison);
    candidateSeatActs += paired.summary.candidateSeatActs;
    for (const [key, value] of Object.entries(paired.samples)) {
      metricSamples[key].push(value);
    }
    if (!values["omit-match-data"]) {
      matchSummaries.push({
        matchIndex: matchIndex + 1,
        seed: matchSeed,
        episodeId,
        candidatePlayerIds,
        pairedWorldValidation: {
          sameSeed: true,
          sameEpisodeId: true,
          sameInitialPlayerOrder: true,
          initialPlayerOrder: [...baseline.acts[0].playerOrder],
        },
        ...paired.summary,
      });
    }
  }

  const result = {
    playerCount,
    matches: playerCountMatches,
    actsPerMatch: acts,
    candidateSeatActs,
    baseline: summarizeOutcome(baselineTotals),
    intervention: summarizeOutcome(interventionTotals),
    pairedMarginal: summarizeMetrics(metricSamples),
    finishComparison: summarizeComparisonTotals(finishComparison),
    worldPairing: {
      matchedSeedPairs: playerCountMatches,
      matchedEpisodeIds: playerCountMatches,
      matchedInitialSeatOrders,
    },
    ...(values["omit-match-data"] ? {} : { matchSummaries }),
  };
  results.push(result);
  internals.push({
    baselineTotals,
    interventionTotals,
    finishComparison,
    metricSamples,
    candidateSeatActs,
  });
  const chip = result.pairedMarginal.chipDifference;
  console.log(
    `p${playerCount}: paired marginal chip ${chip.mean.toFixed(4)} ` +
      `[${chip.confidence95.low.toFixed(4)}, ` +
      `${chip.confidence95.high.toFixed(4)}]`,
  );
}

const pooledBaselineTotals = createOutcomeTotals();
const pooledInterventionTotals = createOutcomeTotals();
const pooledFinishComparison = createComparisonTotals();
const pooledMetricSamples = createMetricSamples();
let pooledCandidateSeatActs = 0;
for (const internal of internals) {
  mergeOutcomeTotals(pooledBaselineTotals, internal.baselineTotals);
  mergeOutcomeTotals(pooledInterventionTotals, internal.interventionTotals);
  mergeComparisonTotals(pooledFinishComparison, internal.finishComparison);
  appendMetricSamples(pooledMetricSamples, internal.metricSamples);
  pooledCandidateSeatActs += internal.candidateSeatActs;
}
const pooledMatches = Object.values(matchCountsByPlayerCount).reduce(
  (total, count) => total + count,
  0,
);
const pooled = {
  playerCounts,
  matches: pooledMatches,
  actsPerMatch: acts,
  candidateSeatActs: pooledCandidateSeatActs,
  baseline: summarizeOutcome(pooledBaselineTotals),
  intervention: summarizeOutcome(pooledInterventionTotals),
  pairedMarginal: summarizeMetrics(pooledMetricSamples),
  finishComparison: summarizeComparisonTotals(pooledFinishComparison),
  worldPairing: {
    matchedSeedPairs: pooledMatches,
    matchedEpisodeIds: pooledMatches,
    matchedInitialSeatOrders: pooledMatches,
  },
};

const uniformMatchCounts =
  new Set(Object.values(matchCountsByPlayerCount)).size === 1;
const report = {
  format: "dalmuti-non-card-paired-marginal-benchmark",
  version: 1,
  trainingOnly: true,
  seed,
  matchesPerPlayerCount: uniformMatchCounts
    ? Object.values(matchCountsByPlayerCount)[0]
    : matchCountsByPlayerCount,
  matchCountsByPlayerCount,
  actsPerMatch: acts,
  playerCounts,
  elapsedSeconds: (performance.now() - startedAt) / 1000,
  evaluationDesign: {
    estimand:
      "same-player marginal outcome change from replacing only selected " +
      "candidate non-card decisions while every card-play decision stays " +
      "on the exact current normal heuristic",
    pairUnit: "same match seed, episode id, and selected candidate player IDs",
    confidenceLevel: 0.95,
    confidenceUnit: "match",
    clusterValue:
      "mean intervention-minus-baseline outcome over selected candidate " +
      "player-act observations within one paired match",
    smallSampleMethod: "student-t below 30 paired matches",
    largeSampleMethod: "normal at 30 or more paired matches",
    candidateSeatAssignment: "cyclically rotated by match",
    matchSeedFormula:
      "base seed + playerCount * 1,000,000 + zero-based match index",
    matchDataIncluded: !values["omit-match-data"],
    initialWorldValidation:
      "same simulator seed/config and identical initial player order",
    trajectoryNote:
      "later roles and deals may differ causally after an intervention changes " +
      "a prior finish order; the environment RNG stream remains seed-matched",
    promotionGatesApplied: false,
    promotionGateNote:
      "training diagnostic only; this report does not redefine or replace the " +
      "three automatic promotion gates",
  },
  cardPlayPolicy: {
    implementation: "lib/bot-strategy.ts#chooseBotPlay",
    difficulty: "normal",
    appliesTo: ["baseline", "intervention", "candidate", "other players"],
    routing: playRouting,
    validation:
      "every recorded baseline and intervention TrainingStep.behaviorPolicy was normal",
  },
  nonCardEvaluation: {
    ablation: nonCardAblationName(
      taxReturnBenchmarkModel,
      revolutionBenchmarkModel,
    ),
    candidateOnlyRouting: true,
    baselinePolicy: "exact current normal tax-return and revolution heuristics",
    baselineRouting: baselineNonCardRouting,
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
    interventionRouting: summarizeNonCardRoutingTotals(nonCardRouting),
  },
  pooled,
  results,
};

const outputHandle = await open(outputPath, "wx");
try {
  await outputHandle.writeFile(`${JSON.stringify(report, null, 2)}\n`, "utf8");
} finally {
  await outputHandle.close();
}
console.log(
  `Saved paired non-card marginal report to ${outputPath} ` +
    `(${report.elapsedSeconds.toFixed(2)}s)`,
);
