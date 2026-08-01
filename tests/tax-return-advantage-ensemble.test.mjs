import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  TAX_RETURN_ACTION_FEATURE_COUNT,
  TAX_RETURN_ACTION_FEATURE_LAYOUT,
  encodeTaxReturnAction,
} from "../training/non-card-action-space.ts";
import {
  TAX_RETURN_NORMAL_BASELINE_PROVENANCE,
  TAX_RETURN_NORMAL_BASELINE_PROVENANCE_SHA256,
  parseTaxReturnAdvantageEnsemble,
  selectTaxReturnAdvantageEnsembleAction,
} from "../training/tax-return-advantage-ensemble.ts";
import { simulateMatch } from "../training/simulator.ts";
import {
  createCandidateOnlyNonCardHooks,
  createNonCardRoutingTotals,
  loadTaxReturnBenchmarkModel,
  nonCardSafetyGateProvenance,
  recordNonCardRouting,
  summarizeNonCardRoutingTotals,
} from "../scripts/lib/non-card-benchmark-policies.mjs";

function roleAt(index, playerCount) {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === playerCount - 2) return "lesser-peon";
  if (index === playerCount - 1) return "great-peon";
  return "merchant";
}

function observation(returnCount) {
  const actorIndex = returnCount === 2 ? 0 : 1;
  const hand = [
    { id: "r2", rank: 2 },
    { id: "r5", rank: 5 },
    { id: "r7", rank: 7 },
    { id: "r9", rank: 9 },
  ];
  return {
    actorId: `p${actorIndex}`,
    hand,
    players: Array.from({ length: 4 }, (_, index) => ({
      id: `p${index}`,
      role: roleAt(index, 4),
      handCount: index === actorIndex ? hand.length : 8 + index,
      score: index,
    })),
    round: 2,
    returnCount,
  };
}

function memberHash(member) {
  const arrays = [
    member.contextLayer.weight,
    member.contextLayer.bias,
    member.bilinearWeight,
  ];
  const dimensions = Buffer.alloc(20);
  dimensions.writeUInt32LE(member.contextLayer.inFeatures, 0);
  dimensions.writeUInt32LE(member.contextLayer.outFeatures, 4);
  dimensions.writeUInt32LE(arrays[0].length, 8);
  dimensions.writeUInt32LE(arrays[1].length, 12);
  dimensions.writeUInt32LE(arrays[2].length, 16);
  const digest = createHash("sha256")
    .update("dalmuti-tax-return-bilinear-residual-member-v1\0")
    .update(dimensions);
  for (const values of arrays) {
    const bytes = Buffer.alloc(values.length * 8);
    values.forEach((value, index) => bytes.writeDoubleLE(value, index * 8));
    digest.update(bytes);
  }
  return digest.digest("hex");
}

export function taxAdvantageEnsembleFixture({
  memberWeights = [2, 2, 2, 2, 2],
  defaultMinimumChipAdvantage = 0.5,
  sourceVersion = 2,
} = {}) {
  const contextFeatures = 1;
  return {
    format: "dalmuti-tax-return-bilinear-residual-ensemble",
    version: 2,
    decisionKind: "tax-return",
    scoreSemantics: "chip-advantage-vs-normal-baseline",
    observationSchemaVersion: 1,
    observationFeatures: 103,
    actionCatalogueVersion: 1,
    actionCount: 103,
    actionFeatures: TAX_RETURN_ACTION_FEATURE_COUNT,
    actionFeatureLayout: [...TAX_RETURN_ACTION_FEATURE_LAYOUT],
    trainingData: {
      sourceFormatVersions: [sourceVersion],
      groupSplitKey:
        sourceVersion === 1
          ? "canonicalWorldKey"
          : "canonicalInformationStateKey",
      determinizationSchema:
        sourceVersion === 1
          ? null
          : "world-clustered-paired-baseline-advantages-v2",
      worldCountPerInformationState: sourceVersion === 1 ? 1 : 8,
      continuationCountPerHiddenWorld: sourceVersion === 1 ? 1 : 4,
      effectiveIndependentWorldsPerInformationState:
        sourceVersion === 1 ? 1 : 8,
      rawContinuationEvaluationsPerInformationState:
        sourceVersion === 1 ? 1 : 32,
      standardErrorEstimable: sourceVersion !== 1,
      determinizationAlgorithm:
        sourceVersion === 1
          ? null
          : "target-act-opponent-physical-card-fisher-yates-v1",
      determinizationAlgorithmVersion: sourceVersion === 1 ? null : 1,
      determinizationAlgorithmContractSha256:
        sourceVersion === 1
          ? null
          : "368240f14f2e5d84bb3085610a176ad4519bc6e5ae288b70de549f63212905c4",
      candidateSeedDerivation:
        sourceVersion === 1
          ? null
          : "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,attempt)))",
      continuationSeedDerivation:
        sourceVersion === 1
          ? null
          : "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,worldIndex,continuationIndex,continuation)))",
      targetField:
        sourceVersion === 1
          ? "actions[].decisionActUtility-minus-baseline.decisionActUtility"
          : "actions[].pairedDecisionActBaselineAdvantage.mean",
      targetTransform: {
        scoreUnit: "chip-units",
        sourceUnit: "(roundChipAward-2)/2",
        operation: "multiply-source-baseline-advantage-by-2",
        multiplier: 2,
      },
      stateWeighting: "one-per-information-state-independent-of-worldCount",
    },
    architecture: {
      contextFeatures,
      contextActivation: "tanh",
      score: "raw(s,a)-raw(s,normalBaselineAction)",
      weightLayout: "row-major [context_features, action_features]",
    },
    baseline: {
      provenance: TAX_RETURN_NORMAL_BASELINE_PROVENANCE,
      provenanceSha256: TAX_RETURN_NORMAL_BASELINE_PROVENANCE_SHA256,
      score: "exactly-zero-by-residualization",
    },
    objective: {
      utilityTarget: "decision-act-current-chip-advantage",
      utilityScale: "chip-units",
      weighting: "equal-per-state",
      regression: {
        loss: "huber-paired-action-vs-baseline",
        coefficient: 1,
        deltaChips: 0.5,
      },
      tieAwareSign: {
        loss: "binary-cross-entropy-with-logits",
        coefficient: 0.25,
        temperatureChips: 0.25,
        tieTarget: 0.5,
        tieEpsilonChips: 1e-9,
      },
      checkpointSelection: "paired-validation-loss",
      bootstrapUnit:
        sourceVersion === 1
          ? "canonicalWorldKey"
          : "canonicalInformationStateKey",
    },
    routing: {
      returnCountOne: "exact-normal-fallback",
      returnCountTwo: "ensemble-lower-confidence-bound",
      roleRouting: {
        "great-dalmuti": "ensemble-lower-confidence-bound",
        "lesser-dalmuti": "exact-normal-fallback",
        "other-roles": "not-applicable",
      },
      memberCount: 5,
      unanimityRule: "all-member-advantages-strictly-positive",
      lowerConfidenceBound: "mean-minus-z-times-sample-sd",
      zValue: 1.645,
      defaultMinimumChipAdvantage,
      selection: "maximum-eligible-lcb",
      tieBreak: "baseline-then-lowest-action-index",
    },
    members: memberWeights.map((rankSevenWeight, memberIndex) => {
      const bilinearWeight = Array(TAX_RETURN_ACTION_FEATURE_COUNT).fill(0);
      bilinearWeight[8] = rankSevenWeight;
      const member = {
        memberIndex,
        seed: 1000 + memberIndex,
        checkpointEpoch: 10 + memberIndex,
        validationPairedLoss: 0.1 + memberIndex * 0.01,
        parametersSha256: "",
        contextLayer: {
          inFeatures: 103,
          outFeatures: contextFeatures,
          weight: Array(103).fill(0),
          bias: [1],
        },
        bilinearWeight,
      };
      member.parametersSha256 = memberHash(member);
      return member;
    }),
  };
}

test("v2 ensemble parser binds target scaling, hidden-world counts, and five distinct members", () => {
  const payload = taxAdvantageEnsembleFixture();
  assert.equal(parseTaxReturnAdvantageEnsemble(payload), payload);
  const v1Payload = taxAdvantageEnsembleFixture({ sourceVersion: 1 });
  assert.equal(parseTaxReturnAdvantageEnsemble(v1Payload), v1Payload);

  const wrongCount = structuredClone(payload);
  wrongCount.trainingData.rawContinuationEvaluationsPerInformationState = 31;
  assert.throws(
    () => parseTaxReturnAdvantageEnsemble(wrongCount),
    /hidden-world\/continuation binding/,
  );

  const wrongTransform = structuredClone(payload);
  wrongTransform.trainingData.targetTransform.multiplier = 1;
  assert.throws(
    () => parseTaxReturnAdvantageEnsemble(wrongTransform),
    /target unit transform/,
  );

  const wrongDeterminizationContract = structuredClone(payload);
  wrongDeterminizationContract.trainingData.determinizationAlgorithmContractSha256 =
    "f".repeat(64);
  assert.throws(
    () => parseTaxReturnAdvantageEnsemble(wrongDeterminizationContract),
    /determinization algorithm contract/,
  );

  const duplicateSeed = structuredClone(payload);
  duplicateSeed.members[4].seed = duplicateSeed.members[0].seed;
  assert.throws(
    () => parseTaxReturnAdvantageEnsemble(duplicateSeed),
    /seeds must be distinct/,
  );
});

test("returnCount=1 always uses the exact normal action without scoring", () => {
  const baselineActionIndex = encodeTaxReturnAction([5]);
  const decision = selectTaxReturnAdvantageEnsembleAction(
    taxAdvantageEnsembleFixture(),
    observation(1),
    baselineActionIndex,
  );
  assert.equal(decision.actionIndex, baselineActionIndex);
  assert.equal(decision.baselineActionIndex, baselineActionIndex);
  assert.equal(decision.routing, "safetyFallback");
  assert.equal(decision.fallback, true);
  assert.equal(decision.fallbackReason, "return-count-one-exact-normal");
  assert.equal(decision.selectedScore, null);
  assert.deepEqual(decision.actionScores, []);
});

test("returnCount=2 selects only unanimous positive max-LCB residuals", () => {
  const baselineActionIndex = encodeTaxReturnAction([2, 5]);
  const decision = selectTaxReturnAdvantageEnsembleAction(
    taxAdvantageEnsembleFixture(),
    observation(2),
    baselineActionIndex,
  );
  assert.equal(decision.actionIndex, encodeTaxReturnAction([2, 7]));
  assert.equal(decision.routing, "learnedAction");
  assert.equal(decision.selectedScore.unanimousPositive, true);
  assert.equal(decision.selectedScore.sampleStandardDeviation, 0);
  assert.ok(decision.selectedScore.lowerConfidenceBound > 0.5);
  const baselineScore = decision.actionScores.find(
    (score) => score.actionIndex === baselineActionIndex,
  );
  assert.deepEqual(baselineScore.memberAdvantages, [0, 0, 0, 0, 0]);
  assert.equal(baselineScore.meanAdvantage, 0);
  assert.equal(baselineScore.lowerConfidenceBound, 0);
  assert.equal(baselineScore.eligible, false);
});

test("ensemble LCB uses five-member sample SD rather than population SD or SE", () => {
  const baselineActionIndex = encodeTaxReturnAction([2, 5]);
  const decision = selectTaxReturnAdvantageEnsembleAction(
    taxAdvantageEnsembleFixture({ memberWeights: [1, 2, 3, 4, 5] }),
    observation(2),
    baselineActionIndex,
    0,
  );
  const values = decision.selectedScore.memberAdvantages;
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const sumSquares = values.reduce(
    (total, value) => total + (value - mean) ** 2,
    0,
  );
  const sampleSd = Math.sqrt(sumSquares / (values.length - 1));
  const populationSd = Math.sqrt(sumSquares / values.length);
  assert.equal(decision.selectedScore.sampleStandardDeviation, sampleSd);
  assert.notEqual(decision.selectedScore.sampleStandardDeviation, populationSd);
  assert.notEqual(
    decision.selectedScore.sampleStandardDeviation,
    sampleSd / Math.sqrt(values.length),
  );
  assert.equal(
    decision.selectedScore.lowerConfidenceBound,
    mean - 1.645 * sampleSd,
  );
});

test("one dissenting member or an LCB equal to the threshold falls back to normal", () => {
  const baselineActionIndex = encodeTaxReturnAction([2, 5]);
  const dissent = selectTaxReturnAdvantageEnsembleAction(
    taxAdvantageEnsembleFixture({ memberWeights: [2, 2, 2, 2, -0.1] }),
    observation(2),
    baselineActionIndex,
    0,
  );
  assert.equal(dissent.actionIndex, baselineActionIndex);
  assert.equal(dissent.fallbackReason, "no-unanimous-positive-action");
  assert.equal(dissent.selectedScore.unanimousPositive, false);

  const positive = selectTaxReturnAdvantageEnsembleAction(
    taxAdvantageEnsembleFixture(),
    observation(2),
    baselineActionIndex,
    0,
  );
  const exactThreshold = positive.selectedScore.lowerConfidenceBound;
  const tied = selectTaxReturnAdvantageEnsembleAction(
    taxAdvantageEnsembleFixture(),
    observation(2),
    baselineActionIndex,
    exactThreshold,
  );
  assert.equal(tied.actionIndex, baselineActionIndex);
  assert.equal(
    tied.fallbackReason,
    "lower-confidence-bound-not-above-threshold",
  );
});

test("shared benchmark hooks load the ensemble default and expose conservative telemetry", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-tax-advantage-"));
  const modelPath = join(directory, "model.json");
  await writeFile(
    modelPath,
    `${JSON.stringify(taxAdvantageEnsembleFixture())}\n`,
    "utf8",
  );
  const benchmarkModel = await loadTaxReturnBenchmarkModel(modelPath);
  assert.equal(benchmarkModel.modelKind, "baseline-advantage-ensemble");
  assert.equal(benchmarkModel.defaultMinimumAdvantage, 0.5);
  assert.equal(benchmarkModel.metadata.memberCount, 5);
  assert.equal(
    benchmarkModel.metadata.scoreSemantics,
    "chip-advantage-vs-normal-baseline",
  );

  const candidateIds = new Set([
    "player-1",
    "player-2",
    "player-3",
    "player-4",
  ]);
  const telemetry = [];
  const nonCard = createCandidateOnlyNonCardHooks({
    candidateIds,
    taxReturn: benchmarkModel,
    revolution: null,
    decisionTelemetry: telemetry,
  });
  const match = simulateMatch({
    playerCount: 4,
    acts: 3,
    seed: 839000001,
    episodeId: "tax-advantage-shared-hook",
    difficulties: ["normal"],
    nonCard,
  });
  const totals = createNonCardRoutingTotals();
  recordNonCardRouting(
    totals,
    match.nonCardSteps,
    candidateIds,
    { taxReturn: benchmarkModel, revolution: null },
    telemetry,
  );
  const summary = summarizeNonCardRoutingTotals(totals).taxReturn;
  assert.ok(summary.candidateModel > 0);
  assert.equal(
    summary.candidateModel,
    summary.learnedAction + summary.safetyFallback,
  );
  assert.equal(
    summary.ensembleAdvantageTelemetry.decisions,
    summary.candidateModel,
  );
  assert.ok(
    summary.ensembleAdvantageTelemetry.returnCountOneExactNormalFallback > 0,
  );
  assert.equal(
    summary.ensembleAdvantageTelemetry.returnCountTwoEvaluated,
    summary.ensembleAdvantageTelemetry.meanAdvantage.count,
  );
  assert.equal(
    summary.ensembleAdvantageTelemetry.returnCountTwoEvaluated,
    summary.ensembleAdvantageTelemetry.sampleStandardDeviation.count,
  );
  assert.equal(
    summary.ensembleAdvantageTelemetry.returnCountTwoEvaluated,
    summary.ensembleAdvantageTelemetry.lowerConfidenceBound.count,
  );
  assert.equal(
    summary.ensembleAdvantageTelemetry.fallback +
      summary.ensembleAdvantageTelemetry.learned,
    summary.candidateModel,
  );

  const provenance = nonCardSafetyGateProvenance({
    taxReturn: benchmarkModel,
    revolution: null,
    taxMinAdvantage: benchmarkModel.defaultMinimumAdvantage,
    revolutionMinAdvantage: 0,
  });
  assert.equal(provenance.score, "chip-advantage-vs-normal-baseline");
  assert.equal(provenance.taxReturn.unit, "chip");
  assert.equal(provenance.taxReturn.minimumAdvantage, 0.5);
  assert.equal(provenance.taxReturn.memberCount, 5);
  assert.equal(provenance.taxReturn.zValue, 1.645);

  // Ordinary and seed-paired evaluators both construct this same hook. Two
  // independent constructions over the same simulator world must therefore
  // produce byte-for-byte equivalent routing telemetry.
  const pairedTelemetry = [];
  const pairedHook = createCandidateOnlyNonCardHooks({
    candidateIds,
    taxReturn: benchmarkModel,
    revolution: null,
    decisionTelemetry: pairedTelemetry,
  });
  const pairedIntervention = simulateMatch({
    playerCount: 4,
    acts: 3,
    seed: 839000001,
    episodeId: "tax-advantage-shared-hook",
    difficulties: ["normal"],
    nonCard: pairedHook,
  });
  const pairedTotals = createNonCardRoutingTotals();
  recordNonCardRouting(
    pairedTotals,
    pairedIntervention.nonCardSteps,
    candidateIds,
    { taxReturn: benchmarkModel, revolution: null },
    pairedTelemetry,
  );
  assert.deepEqual(
    summarizeNonCardRoutingTotals(pairedTotals),
    summarizeNonCardRoutingTotals(totals),
  );

  const corruptPath = join(directory, "corrupt-model.json");
  const corrupt = taxAdvantageEnsembleFixture();
  corrupt.members[0].bilinearWeight[0] += 1;
  await writeFile(corruptPath, JSON.stringify(corrupt), "utf8");
  await assert.rejects(
    () => loadTaxReturnBenchmarkModel(corruptPath),
    /member 0 parameter hash mismatch/,
  );
});
