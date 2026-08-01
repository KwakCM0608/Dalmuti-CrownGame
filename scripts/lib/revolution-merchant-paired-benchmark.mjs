import { isDeepStrictEqual } from "node:util";

import { simulateMatch } from "../../training/simulator.ts";
import {
  confidenceInterval95,
  roleForSeat,
  rotatingCandidateIds,
} from "../rl-evaluation-statistics.mjs";
import {
  EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
  EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION,
  REVOLUTION_MERCHANT_SOURCE_DATA,
  createExperimentalMerchantRevolutionHook,
} from "./experimental-revolution-merchant-candidate.mjs";

const ROLES = Object.freeze([
  "great-dalmuti",
  "lesser-dalmuti",
  "merchant",
  "lesser-peon",
  "great-peon",
]);

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
  return value;
}

function arraysEqual(left, right) {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function emptyMetricSamples() {
  return {
    chipDifference: [],
    finishPlaceDifference: [],
    firstRateDifference: [],
    lastRateDifference: [],
    finalScoreDifference: [],
  };
}

function appendMetricSamples(target, source) {
  for (const key of Object.keys(target)) target[key].push(...source[key]);
}

function inference(samples, betterDirection) {
  if (samples.length === 0) {
    return {
      status: "not-applicable",
      clusters: 0,
      unit: "match",
      mean: null,
      confidence95: null,
      inference: null,
      betterDirection,
    };
  }
  const interval = confidenceInterval95(samples);
  return {
    status: "available",
    clusters: interval.count,
    unit: "match",
    mean: interval.mean,
    confidence95: { low: interval.low, high: interval.high },
    inference: {
      method: interval.method,
      sampleStandardDeviation: interval.sampleStandardDeviation,
      standardError: interval.standardError,
      criticalValue: interval.criticalValue,
    },
    betterDirection,
  };
}

function summarizeMetrics(samples) {
  return {
    chipDifference: inference(samples.chipDifference, "positive"),
    finishPlaceDifference: inference(
      samples.finishPlaceDifference,
      "negative",
    ),
    firstRateDifference: inference(samples.firstRateDifference, "positive"),
    lastRateDifference: inference(samples.lastRateDifference, "negative"),
    finalScoreDifference: inference(
      samples.finalScoreDifference,
      "positive",
    ),
  };
}

function emptyFinishComparison() {
  return { interventionBetter: 0, tied: 0, interventionWorse: 0 };
}

function mergeFinishComparison(target, source) {
  target.interventionBetter += source.interventionBetter;
  target.tied += source.tied;
  target.interventionWorse += source.interventionWorse;
}

function summarizeFinishComparison(value) {
  const comparisons =
    value.interventionBetter + value.tied + value.interventionWorse;
  const nonTies = value.interventionBetter + value.interventionWorse;
  return {
    ...value,
    comparisons,
    interventionBetterRate:
      comparisons === 0 ? null : value.interventionBetter / comparisons,
    tiedRate: comparisons === 0 ? null : value.tied / comparisons,
    interventionWorseRate:
      comparisons === 0 ? null : value.interventionWorse / comparisons,
    interventionBetterRateAmongNonTies:
      nonTies === 0 ? null : value.interventionBetter / nonTies,
  };
}

function emptyRouting() {
  const createRoleCounts = () =>
    Object.fromEntries(
      ROLES.map((role) => [
        role,
        {
          candidate: 0,
          nonCandidate: 0,
          changedFromNormal: 0,
          exactNormal: 0,
        },
      ]),
    );
  return {
    revolutionDecisions: 0,
    candidateDecisions: 0,
    nonCandidateDecisions: 0,
    changedFromNormal: 0,
    candidateExactNormalFallback: 0,
    nonCandidateExactNormal: 0,
    byRole: createRoleCounts(),
  };
}

function addRouting(target, telemetry) {
  for (const decision of telemetry) {
    target.revolutionDecisions += 1;
    const role = target.byRole[decision.actorRole];
    if (!role) throw new Error(`unknown revolution telemetry role ${decision.actorRole}`);
    if (decision.candidateActor) {
      target.candidateDecisions += 1;
      role.candidate += 1;
      if (decision.changedFromBaseline) {
        target.changedFromNormal += 1;
        role.changedFromNormal += 1;
      } else {
        target.candidateExactNormalFallback += 1;
        role.exactNormal += 1;
      }
    } else {
      target.nonCandidateDecisions += 1;
      target.nonCandidateExactNormal += 1;
      role.nonCandidate += 1;
      role.exactNormal += 1;
    }
  }
}

function validateEvidence(sourceEvidence) {
  if (!sourceEvidence || typeof sourceEvidence !== "object") {
    throw new TypeError("sourceEvidence from the strict data audit is required");
  }
  if (
    sourceEvidence.source?.sha256 !== REVOLUTION_MERCHANT_SOURCE_DATA.sha256 ||
    sourceEvidence.contract?.normalBaselineRecomputed !== true ||
    sourceEvidence.contract?.canonicalInformationStateKeysRecomputed !== true ||
    sourceEvidence.contract?.canonicalInformationStateKeysUnique !== true ||
    !arraysEqual(
      sourceEvidence.evidence?.evidenceSupportedCounts ?? [],
      EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
    )
  ) {
    throw new TypeError("sourceEvidence does not prove the frozen candidate contract");
  }
}

function validateInterventionRouting(match, telemetry, candidateIds) {
  const byKey = new Map();
  for (const decision of telemetry) {
    if (byKey.has(decision.decisionKey)) {
      throw new Error(`duplicate revolution telemetry ${decision.decisionKey}`);
    }
    byKey.set(decision.decisionKey, decision);
  }
  let observedRevolutionSteps = 0;
  for (const step of match.nonCardSteps ?? []) {
    if (step.forcedOverride) {
      throw new Error("paired benchmark must not contain a forced override");
    }
    if (step.decision === "tax-return") {
      if (
        step.behaviorPolicy !== "normal" ||
        step.behaviorPolicyVersion !== null
      ) {
        throw new Error("tax return drifted from exact current normal");
      }
      continue;
    }
    if (step.decision !== "revolution") {
      throw new Error(`unknown non-card decision ${String(step.decision)}`);
    }
    observedRevolutionSteps += 1;
    const decision = byKey.get(step.decisionKey);
    if (!decision) {
      throw new Error(`missing revolution telemetry ${step.decisionKey}`);
    }
    if (
      step.behaviorPolicy !== "custom" ||
      step.behaviorPolicyVersion !==
        EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION ||
      step.actorId !== decision.actorId ||
      step.actorRole !== decision.actorRole ||
      step.actionIndex !== decision.actionIndex ||
      candidateIds.has(step.actorId) !== decision.candidateActor ||
      step.actionIndex !==
        (decision.candidateActor
          ? decision.actionIndex
          : decision.baselineActionIndex)
    ) {
      throw new Error(`revolution step/telemetry mismatch ${step.decisionKey}`);
    }
  }
  if (observedRevolutionSteps !== telemetry.length) {
    throw new Error("not every revolution telemetry record reached the simulator");
  }
}

function summarizePairedMatch(baseline, intervention, candidatePlayerIds) {
  const samples = emptyMetricSamples();
  const finishComparison = emptyFinishComparison();
  const roleCells = Object.fromEntries(
    ROLES.map((role) => [role, { sum: 0, count: 0 }]),
  );
  let chipDifference = 0;
  let finishPlaceDifference = 0;
  let firstRateDifference = 0;
  let lastRateDifference = 0;

  for (let actIndex = 0; actIndex < baseline.acts.length; actIndex += 1) {
    const baselineAct = baseline.acts[actIndex];
    const interventionAct = intervention.acts[actIndex];
    if (
      baselineAct.round !== actIndex + 1 ||
      interventionAct.round !== baselineAct.round
    ) {
      throw new Error("paired act numbers are misaligned");
    }
    const baselinePlaces = new Map(
      baselineAct.finishOrder.map((playerId, index) => [playerId, index + 1]),
    );
    const interventionPlaces = new Map(
      interventionAct.finishOrder.map((playerId, index) => [playerId, index + 1]),
    );
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
        throw new Error(`paired outcome is missing ${playerId}`);
      }
      const chip = interventionChips - baselineChips;
      const finish = interventionPlace - baselinePlace;
      chipDifference += chip;
      finishPlaceDifference += finish;
      firstRateDifference +=
        Number(interventionPlace === 1) - Number(baselinePlace === 1);
      lastRateDifference +=
        Number(interventionPlace === intervention.playerCount) -
        Number(baselinePlace === baseline.playerCount);
      if (finish < 0) finishComparison.interventionBetter += 1;
      else if (finish > 0) finishComparison.interventionWorse += 1;
      else finishComparison.tied += 1;
      const baselineSeat = baselineAct.playerOrder.indexOf(playerId);
      if (baselineSeat < 0) throw new Error(`baseline act omitted ${playerId}`);
      const role = roleForSeat(baselineSeat, baseline.playerCount);
      roleCells[role].sum += chip;
      roleCells[role].count += 1;
    }
  }
  const candidateSeatActs = candidatePlayerIds.length * baseline.acts.length;
  samples.chipDifference.push(chipDifference / candidateSeatActs);
  samples.finishPlaceDifference.push(finishPlaceDifference / candidateSeatActs);
  samples.firstRateDifference.push(firstRateDifference / candidateSeatActs);
  samples.lastRateDifference.push(lastRateDifference / candidateSeatActs);
  samples.finalScoreDifference.push(
    candidatePlayerIds.reduce(
      (sum, playerId) =>
        sum + intervention.finalScores[playerId] - baseline.finalScores[playerId],
      0,
    ) / candidatePlayerIds.length,
  );
  return {
    candidateSeatActs,
    samples,
    finishComparison,
    roleSamples: Object.fromEntries(
      Object.entries(roleCells).map(([role, value]) => [
        role,
        value.count === 0 ? null : value.sum / value.count,
      ]),
    ),
    matchSummary: {
      candidatePlayerIds,
      chipDifference: samples.chipDifference[0],
      finishPlaceDifference: samples.finishPlaceDifference[0],
      firstRateDifference: samples.firstRateDifference[0],
      lastRateDifference: samples.lastRateDifference[0],
      finalScoreDifference: samples.finalScoreDifference[0],
      finishComparison: summarizeFinishComparison(finishComparison),
    },
  };
}

function summarizeRoleSamples(roleSamples) {
  return Object.fromEntries(
    ROLES.map((role) => [role, inference(roleSamples[role], "positive")]),
  );
}

function createInternal() {
  return {
    metricSamples: emptyMetricSamples(),
    roleSamples: Object.fromEntries(ROLES.map((role) => [role, []])),
    changedDecisionActChipSamples: [],
    finishComparison: emptyFinishComparison(),
    routing: emptyRouting(),
    candidateSeatActs: 0,
    exactTrajectoryMatches: 0,
    changedDecisionMatches: 0,
  };
}

function summarizeInternal(internal, matches) {
  return {
    candidateSeatActs: internal.candidateSeatActs,
    pairedMarginal: summarizeMetrics(internal.metricSamples),
    baselineRoleChipDifference: summarizeRoleSamples(internal.roleSamples),
    changedRevolutionActorActChipDifference: inference(
      internal.changedDecisionActChipSamples,
      "positive",
    ),
    finishComparison: summarizeFinishComparison(internal.finishComparison),
    interventionRouting: internal.routing,
    trajectoryParity: {
      exactMatches: internal.exactTrajectoryMatches,
      totalMatches: matches,
      exactRate: internal.exactTrajectoryMatches / matches,
      changedDecisionMatches: internal.changedDecisionMatches,
    },
  };
}

export function runExperimentalMerchantRevolutionPairedBenchmark({
  sourceEvidence,
  playerCounts = [4, 5, 6, 7, 8, 9, 10],
  matchCountsByPlayerCount,
  acts = 5,
  seed,
  includeMatchData = false,
}) {
  validateEvidence(sourceEvidence);
  positiveInteger(acts, "acts");
  positiveInteger(seed, "seed");
  if (
    !Array.isArray(playerCounts) ||
    playerCounts.length < 1 ||
    new Set(playerCounts).size !== playerCounts.length ||
    playerCounts.some(
      (playerCount) =>
        !Number.isInteger(playerCount) || playerCount < 4 || playerCount > 10,
    )
  ) {
    throw new RangeError("playerCounts must be unique integers from 4 to 10");
  }
  if (!matchCountsByPlayerCount || typeof matchCountsByPlayerCount !== "object") {
    throw new TypeError("matchCountsByPlayerCount is required");
  }
  for (const playerCount of playerCounts) {
    positiveInteger(
      matchCountsByPlayerCount[playerCount],
      `matches for p${playerCount}`,
    );
  }
  if (typeof includeMatchData !== "boolean") {
    throw new TypeError("includeMatchData must be boolean");
  }

  const startedAt = performance.now();
  const pooled = createInternal();
  const results = [];
  let totalMatches = 0;
  let baselineCardSteps = 0;
  let interventionCardSteps = 0;
  let baselineNonCardSteps = 0;

  for (const playerCount of playerCounts) {
    const matches = matchCountsByPlayerCount[playerCount];
    const internal = createInternal();
    const matchSummaries = [];
    for (let matchIndex = 0; matchIndex < matches; matchIndex += 1) {
      const candidateCount =
        playerCount % 2 === 0 || matchIndex % 2 === 0
          ? Math.floor(playerCount / 2)
          : Math.floor(playerCount / 2) + 1;
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
      const episodeId = `paired-revolution-merchant-p${playerCount}-${matchIndex + 1}`;
      const baseline = simulateMatch({
        playerCount,
        acts,
        seed: matchSeed,
        episodeId,
        difficulties: ["normal"],
        nonCard: {},
      });
      const telemetry = [];
      const intervention = simulateMatch({
        playerCount,
        acts,
        seed: matchSeed,
        episodeId,
        difficulties: ["normal"],
        nonCard: createExperimentalMerchantRevolutionHook({
          candidateIds,
          telemetry,
        }),
      });
      if (
        baseline.seed !== intervention.seed ||
        baseline.episodeId !== intervention.episodeId ||
        baseline.playerCount !== intervention.playerCount ||
        baseline.acts.length !== intervention.acts.length ||
        !arraysEqual(
          baseline.acts[0].playerOrder,
          intervention.acts[0].playerOrder,
        )
      ) {
        throw new Error("paired simulations drifted from their shared initial world");
      }
      for (const step of baseline.steps) {
        if (step.behaviorPolicy !== "normal") {
          throw new Error("baseline card play drifted from current normal");
        }
        baselineCardSteps += 1;
      }
      for (const step of intervention.steps) {
        if (step.behaviorPolicy !== "normal") {
          throw new Error("intervention card play drifted from current normal");
        }
        interventionCardSteps += 1;
      }
      for (const step of baseline.nonCardSteps ?? []) {
        if (
          step.behaviorPolicy !== "normal" ||
          step.behaviorPolicyVersion !== null ||
          step.forcedOverride
        ) {
          throw new Error("baseline non-card play drifted from current normal");
        }
        baselineNonCardSteps += 1;
      }
      validateInterventionRouting(intervention, telemetry, candidateIds);
      addRouting(internal.routing, telemetry);
      addRouting(pooled.routing, telemetry);

      const changed = telemetry.filter((decision) => decision.changedFromBaseline);
      const exactTrajectory =
        isDeepStrictEqual(baseline.acts, intervention.acts) &&
        isDeepStrictEqual(baseline.finalScores, intervention.finalScores);
      if (changed.length === 0 && !exactTrajectory) {
        throw new Error("an exact-normal fallback changed the match trajectory");
      }
      if (exactTrajectory) {
        internal.exactTrajectoryMatches += 1;
        pooled.exactTrajectoryMatches += 1;
      }
      if (changed.length > 0) {
        internal.changedDecisionMatches += 1;
        pooled.changedDecisionMatches += 1;
        const changedActorDifferences = changed.map((decision) => {
          const round = (intervention.nonCardSteps ?? []).find(
            (step) =>
              step.decision === "revolution" &&
              step.decisionKey === decision.decisionKey,
          )?.round;
          if (!Number.isInteger(round)) {
            throw new Error(`changed decision omitted round ${decision.decisionKey}`);
          }
          return (
            intervention.acts[round - 1].chipAwards[decision.actorId] -
            baseline.acts[round - 1].chipAwards[decision.actorId]
          );
        });
        const changedActorMean =
          changedActorDifferences.reduce((sum, value) => sum + value, 0) /
          changedActorDifferences.length;
        internal.changedDecisionActChipSamples.push(changedActorMean);
        pooled.changedDecisionActChipSamples.push(changedActorMean);
      }

      const paired = summarizePairedMatch(
        baseline,
        intervention,
        candidatePlayerIds,
      );
      appendMetricSamples(internal.metricSamples, paired.samples);
      appendMetricSamples(pooled.metricSamples, paired.samples);
      mergeFinishComparison(internal.finishComparison, paired.finishComparison);
      mergeFinishComparison(pooled.finishComparison, paired.finishComparison);
      internal.candidateSeatActs += paired.candidateSeatActs;
      pooled.candidateSeatActs += paired.candidateSeatActs;
      for (const role of ROLES) {
        const sample = paired.roleSamples[role];
        if (sample !== null) {
          internal.roleSamples[role].push(sample);
          pooled.roleSamples[role].push(sample);
        }
      }
      if (includeMatchData) {
        matchSummaries.push({
          matchIndex: matchIndex + 1,
          seed: matchSeed,
          episodeId,
          pairedWorldValidation: {
            sameSeed: true,
            sameEpisodeId: true,
            sameInitialPlayerOrder: true,
            initialPlayerOrder: [...baseline.acts[0].playerOrder],
          },
          revolutionTelemetry: telemetry,
          exactTrajectory,
          ...paired.matchSummary,
        });
      }
    }
    totalMatches += matches;
    results.push({
      playerCount,
      matches,
      actsPerMatch: acts,
      enabledCandidateCount:
        EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS.includes(playerCount),
      ...summarizeInternal(internal, matches),
      ...(includeMatchData ? { matchSummaries } : {}),
    });
  }

  return {
    format: "dalmuti-experimental-merchant-revolution-paired-benchmark",
    version: 1,
    trainingOnly: true,
    seed,
    actsPerMatch: acts,
    playerCounts: [...playerCounts],
    matchCountsByPlayerCount: Object.fromEntries(
      playerCounts.map((playerCount) => [
        playerCount,
        matchCountsByPlayerCount[playerCount],
      ]),
    ),
    elapsedSeconds: (performance.now() - startedAt) / 1000,
    sourceEvidence: {
      dataSha256: sourceEvidence.source.sha256,
      dataAuditFormat: sourceEvidence.format,
      dataAuditVersion: sourceEvidence.version,
      evidenceSupportedCounts: [
        ...sourceEvidence.evidence.evidenceSupportedCounts,
      ],
    },
    candidate: {
      policyVersion: EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION,
      role: "merchant",
      enabledPlayerCounts: [
        ...EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
      ],
      action: "declare",
      fallback: "exact current normal chooseBotRevolution",
      deterministic: true,
    },
    evaluationDesign: {
      estimand:
        "same-player marginal outcome change from replacing only selected candidate p6 merchant revolution decisions while all card play, tax returns, non-candidate revolutions, and every other candidate revolution use exact current normal",
      pairUnit: "same match seed, episode id, and selected candidate player IDs",
      confidenceLevel: 0.95,
      confidenceUnit: "match",
      clusterValue:
        "mean intervention-minus-baseline outcome over selected candidate player-act observations within one paired match",
      smallSampleMethod: "student-t below 30 paired matches",
      largeSampleMethod: "normal at 30 or more paired matches",
      candidateSeatAssignment: "cyclically rotated by match",
      matchSeedFormula:
        "base seed + playerCount * 1,000,000 + zero-based match index",
      initialWorldValidation:
        "same simulator seed/config and identical initial player order",
      unchangedTrajectoryAssertion:
        "any pair with no changed revolution action must have byte-equivalent acts and final scores",
      causalTrajectoryNote:
        "later roles and deals may differ after a changed revolution; the environment stream remains seed-matched",
      promotionGatesApplied: false,
      promotionGateNote:
        "isolated revolution diagnostic only; this cannot promote or deploy a hard bot",
      matchDataIncluded: includeMatchData,
    },
    policyRoutingValidation: {
      cardPlay: {
        baseline: "current normal",
        intervention: "current normal",
        baselineValidatedSteps: baselineCardSteps,
        interventionValidatedSteps: interventionCardSteps,
      },
      baselineNonCard: {
        policy: "current normal",
        validatedSteps: baselineNonCardSteps,
      },
      intervention:
        "tax returns remain normal; every revolution step is cross-checked against deterministic routing telemetry and policyVersion",
    },
    pooled: {
      matches: totalMatches,
      ...summarizeInternal(pooled, totalMatches),
    },
    results,
  };
}
