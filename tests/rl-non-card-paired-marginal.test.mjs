import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { TAX_RETURN_ACTION_FEATURE_LAYOUT } from "../training/non-card-action-space.ts";

const BENCHMARK_SCRIPT = fileURLToPath(
  new URL("../scripts/rl-benchmark-non-card-marginal.mjs", import.meta.url),
);

function layer(inFeatures, outFeatures) {
  return {
    inFeatures,
    outFeatures,
    weight: Array(inFeatures * outFeatures).fill(0),
    bias: Array(outFeatures).fill(0),
  };
}

function zeroTaxReturnModel() {
  return {
    format: "dalmuti-tax-return-action-conditioned-actor-critic",
    version: 1,
    decisionKind: "tax-return",
    observationSchemaVersion: 1,
    observationFeatures: 103,
    actionCatalogueVersion: 1,
    actionCount: 103,
    actionFeatures: 15,
    actionFeatureLayout: [...TAX_RETURN_ACTION_FEATURE_LAYOUT],
    actorObservationHiddenSizes: [1],
    actorActionHiddenSizes: [1],
    actorScorerHiddenSizes: [],
    valueHiddenSizes: [1],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    actorObservationLayers: [layer(103, 1)],
    actorActionLayers: [layer(15, 1)],
    actorScorerLayers: [layer(2, 1)],
    valueLayers: [layer(103, 1), layer(1, 1)],
  };
}

async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-paired-noncard-"));
  const taxPath = join(directory, "tax.json");
  const bytes = `${JSON.stringify(zeroTaxReturnModel())}\n`;
  await writeFile(taxPath, bytes, "utf8");
  return {
    directory,
    taxPath,
    taxSha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

function runBenchmark({
  taxPath,
  outputPath,
  seed = "202608041",
  matches = "6",
  acts = "3",
  threshold = "0.01",
  omitMatchData = false,
  extraArgs = [],
}) {
  const args = [
    BENCHMARK_SCRIPT,
    "--tax-model",
    taxPath,
    "--tax-min-advantage",
    threshold,
    "--players",
    "4",
    "--matches",
    matches,
    "--acts",
    acts,
    "--seed",
    seed,
    "--output",
    outputPath,
    ...extraArgs,
  ];
  if (omitMatchData) args.push("--omit-match-data");
  return spawnSync(process.execPath, args, {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  });
}

async function readReport(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

test("paired marginal evaluator compares the same players in seed-matched worlds", async () => {
  const data = await fixture();
  const outputPath = join(data.directory, "nested", "paired.json");
  const result = runBenchmark({ ...data, outputPath });
  assert.equal(result.status, 0, result.stderr || result.stdout);

  const report = await readReport(outputPath);
  assert.equal(
    report.format,
    "dalmuti-non-card-paired-marginal-benchmark",
  );
  assert.equal(report.version, 1);
  assert.equal(report.trainingOnly, true);
  assert.equal(report.evaluationDesign.confidenceUnit, "match");
  assert.equal(report.evaluationDesign.promotionGatesApplied, false);
  assert.equal(Object.hasOwn(report, "promotionThresholds"), false);
  assert.equal(Object.hasOwn(report, "promotionPassed"), false);
  assert.equal(report.nonCardEvaluation.ablation, "tax-return-only");
  assert.equal(
    report.nonCardEvaluation.models.taxReturn.sha256,
    data.taxSha256,
  );
  assert.equal(
    report.nonCardEvaluation.models.taxReturn.path,
    data.taxPath,
  );
  assert.equal(
    report.nonCardEvaluation.safetyGate.taxReturn.minimumAdvantage,
    0.01,
  );
  assert.ok(
    report.nonCardEvaluation.interventionRouting.taxReturn.candidateModel > 0,
  );
  assert.ok(
    report.nonCardEvaluation.baselineRouting.taxReturn
      .candidateNormalHeuristic > 0,
  );
  assert.equal(
    report.nonCardEvaluation.interventionRouting.taxReturn.candidateModel,
    report.nonCardEvaluation.interventionRouting.taxReturn.learnedAction +
      report.nonCardEvaluation.interventionRouting.taxReturn
        .agreedWithBaseline +
      report.nonCardEvaluation.interventionRouting.taxReturn.safetyFallback,
  );
  assert.equal(
    report.nonCardEvaluation.interventionRouting.taxReturn
      .validatedPolicyVersionSteps,
    report.nonCardEvaluation.interventionRouting.taxReturn.candidateModel,
  );
  assert.ok(report.cardPlayPolicy.routing.baselineNormalHeuristic > 0);
  assert.ok(report.cardPlayPolicy.routing.interventionNormalHeuristic > 0);

  const playerCount = report.results[0];
  assert.equal(playerCount.matches, 6);
  assert.equal(playerCount.worldPairing.matchedSeedPairs, 6);
  assert.equal(playerCount.worldPairing.matchedEpisodeIds, 6);
  assert.equal(playerCount.worldPairing.matchedInitialSeatOrders, 6);
  assert.equal(playerCount.matchSummaries.length, 6);
  for (const match of playerCount.matchSummaries) {
    assert.equal(match.pairedWorldValidation.sameSeed, true);
    assert.equal(match.pairedWorldValidation.sameEpisodeId, true);
    assert.equal(match.pairedWorldValidation.sameInitialPlayerOrder, true);
    assert.ok(match.candidatePlayerIds.length >= 2);
    assert.equal(
      Object.keys(match.finalScoreDifferences).length,
      match.candidatePlayerIds.length,
    );
    for (const round of match.rounds) {
      assert.deepEqual(
        round.candidateOutcomes.map((outcome) => outcome.playerId),
        match.candidatePlayerIds,
      );
    }
  }
  assert.equal(
    playerCount.pairedMarginal.chipDifference.inference.unit,
    "match",
  );
  assert.equal(
    playerCount.pairedMarginal.finishPlaceDifference.betterDirection,
    "negative",
  );
});

test("a safety-gated tied model reproduces the baseline exactly", async () => {
  const data = await fixture();
  const outputPath = join(data.directory, "neutral.json");
  const result = runBenchmark({
    ...data,
    outputPath,
    matches: "8",
    acts: "4",
    omitMatchData: true,
  });
  assert.equal(result.status, 0, result.stderr || result.stdout);

  const report = await readReport(outputPath);
  const summaries = [...report.results, report.pooled];
  for (const summary of summaries) {
    assert.deepEqual(summary.intervention, summary.baseline);
    for (const metric of Object.values(summary.pairedMarginal)) {
      assert.equal(metric.mean, 0);
      assert.equal(metric.confidence95.low, 0);
      assert.equal(metric.confidence95.high, 0);
    }
    assert.equal(summary.finishComparison.interventionBetter, 0);
    assert.equal(summary.finishComparison.interventionWorse, 0);
    assert.equal(
      summary.finishComparison.tied,
      summary.finishComparison.comparisons,
    );
  }
  const taxRouting =
    report.nonCardEvaluation.interventionRouting.taxReturn;
  assert.equal(taxRouting.learnedAction, 0);
  assert.equal(
    taxRouting.candidateModel,
    taxRouting.agreedWithBaseline + taxRouting.safetyFallback,
  );
});

test("same seed produces deterministic paired results", async () => {
  const data = await fixture();
  const firstPath = join(data.directory, "first.json");
  const secondPath = join(data.directory, "second.json");
  const first = runBenchmark({
    ...data,
    outputPath: firstPath,
    matches: "3",
    omitMatchData: true,
  });
  const second = runBenchmark({
    ...data,
    outputPath: secondPath,
    matches: "3",
    omitMatchData: true,
  });
  assert.equal(first.status, 0, first.stderr || first.stdout);
  assert.equal(second.status, 0, second.stderr || second.stdout);
  const firstReport = await readReport(firstPath);
  const secondReport = await readReport(secondPath);
  delete firstReport.elapsedSeconds;
  delete secondReport.elapsedSeconds;
  assert.deepEqual(firstReport, secondReport);
});

test("paired marginal CLI requires explicit provenance args and a fresh output", async () => {
  const data = await fixture();
  const outputPath = join(data.directory, "fresh.json");
  const first = runBenchmark({
    ...data,
    outputPath,
    matches: "1",
    acts: "2",
    omitMatchData: true,
  });
  assert.equal(first.status, 0, first.stderr || first.stdout);
  const repeated = runBenchmark({
    ...data,
    outputPath,
    matches: "1",
    acts: "2",
    omitMatchData: true,
  });
  assert.notEqual(repeated.status, 0);
  assert.match(repeated.stderr, /output must not already exist/);

  const missingSeed = spawnSync(
    process.execPath,
    [
      BENCHMARK_SCRIPT,
      "--tax-model",
      data.taxPath,
      "--output",
      join(data.directory, "missing-seed.json"),
    ],
    { encoding: "utf8" },
  );
  assert.notEqual(missingSeed.status, 0);
  assert.match(missingSeed.stderr, /--seed is required and must be explicit/);

  const missingModel = spawnSync(
    process.execPath,
    [
      BENCHMARK_SCRIPT,
      "--seed",
      "1",
      "--output",
      join(data.directory, "missing-model.json"),
    ],
    { encoding: "utf8" },
  );
  assert.notEqual(missingModel.status, 0);
  assert.match(
    missingModel.stderr,
    /--tax-model or --revolution-model is required/,
  );

  const missingOutput = spawnSync(
    process.execPath,
    [BENCHMARK_SCRIPT, "--tax-model", data.taxPath, "--seed", "1"],
    { encoding: "utf8" },
  );
  assert.notEqual(missingOutput.status, 0);
  assert.match(missingOutput.stderr, /--output is required/);

  const ambiguousCounts = runBenchmark({
    ...data,
    outputPath: join(data.directory, "ambiguous-counts.json"),
    matches: "1",
    extraArgs: ["--match-counts", "4:1"],
  });
  assert.notEqual(ambiguousCounts.status, 0);
  assert.match(
    ambiguousCounts.stderr,
    /--matches and --match-counts cannot be combined/,
  );
});
