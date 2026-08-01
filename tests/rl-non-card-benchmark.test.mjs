import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  REVOLUTION_ACTION_FEATURE_LAYOUT,
  TAX_RETURN_ACTION_FEATURE_LAYOUT,
} from "../training/non-card-action-space.ts";

const BENCHMARK_SCRIPT = fileURLToPath(
  new URL("../scripts/rl-benchmark-model.mjs", import.meta.url),
);
const PLAY_OBSERVATION_FEATURES = 172;
const PLAY_ACTION_COUNT = 506;

function layer(inFeatures, outFeatures) {
  return {
    inFeatures,
    outFeatures,
    weight: Array(inFeatures * outFeatures).fill(0),
    bias: Array(outFeatures).fill(0),
  };
}

function zeroPlayModel() {
  return {
    format: "dalmuti-actor-critic",
    version: 1,
    observationFeatures: PLAY_OBSERVATION_FEATURES,
    actionCount: PLAY_ACTION_COUNT,
    hiddenSizes: [1],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    trunkLayers: [layer(PLAY_OBSERVATION_FEATURES, 1)],
    policyLayer: layer(1, PLAY_ACTION_COUNT),
    valueLayer: layer(1, 1),
  };
}

function zeroNonCardModel(decisionKind) {
  const taxReturn = decisionKind === "tax-return";
  const observationFeatures = taxReturn ? 103 : 102;
  const actionFeatures = taxReturn ? 15 : 3;
  const actionCount = taxReturn ? 103 : 2;
  return {
    format: taxReturn
      ? "dalmuti-tax-return-action-conditioned-actor-critic"
      : "dalmuti-revolution-action-conditioned-actor-critic",
    version: 1,
    decisionKind,
    observationSchemaVersion: 1,
    observationFeatures,
    actionCatalogueVersion: 1,
    actionCount,
    actionFeatures,
    actionFeatureLayout: taxReturn
      ? [...TAX_RETURN_ACTION_FEATURE_LAYOUT]
      : [...REVOLUTION_ACTION_FEATURE_LAYOUT],
    actorObservationHiddenSizes: [1],
    actorActionHiddenSizes: [1],
    actorScorerHiddenSizes: [],
    valueHiddenSizes: [1],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    actorObservationLayers: [layer(observationFeatures, 1)],
    actorActionLayers: [layer(actionFeatures, 1)],
    actorScorerLayers: [layer(2, 1)],
    valueLayers: [layer(observationFeatures, 1), layer(1, 1)],
    ...(taxReturn ? {} : { greatPeonRoleFeatureIndex: 7 }),
  };
}

async function fixtureDirectory() {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-noncard-bench-"));
  const playPath = join(directory, "play.json");
  const taxPath = join(directory, "tax.json");
  const revolutionPath = join(directory, "revolution.json");
  const contents = {
    play: `${JSON.stringify(zeroPlayModel())}\n`,
    tax: `${JSON.stringify(zeroNonCardModel("tax-return"))}\n`,
    revolution: `${JSON.stringify(zeroNonCardModel("revolution"))}\n`,
  };
  await Promise.all([
    writeFile(playPath, contents.play, "utf8"),
    writeFile(taxPath, contents.tax, "utf8"),
    writeFile(revolutionPath, contents.revolution, "utf8"),
  ]);
  return {
    directory,
    playPath,
    taxPath,
    revolutionPath,
    hashes: Object.fromEntries(
      Object.entries(contents).map(([name, content]) => [
        name,
        createHash("sha256").update(content).digest("hex"),
      ]),
    ),
  };
}

function runBenchmark({
  playPath,
  candidatePlay,
  taxPath,
  revolutionPath,
  taxMinAdvantage,
  revolutionMinAdvantage,
  outputPath,
  seed = "20260803",
  matches = "4",
  acts = "3",
}) {
  const args = [
    BENCHMARK_SCRIPT,
    "--players",
    "4",
    "--matches",
    matches,
    "--acts",
    acts,
    "--seed",
    seed,
    "--omit-match-data",
    "--output",
    outputPath,
  ];
  if (playPath) args.push("--model", playPath);
  if (candidatePlay) args.push("--candidate-play", candidatePlay);
  if (taxPath) args.push("--tax-model", taxPath);
  if (revolutionPath) args.push("--revolution-model", revolutionPath);
  if (taxMinAdvantage !== undefined) {
    args.push(`--tax-min-advantage=${taxMinAdvantage}`);
  }
  if (revolutionMinAdvantage !== undefined) {
    args.push(`--revolution-min-advantage=${revolutionMinAdvantage}`);
  }
  return spawnSync(process.execPath, args, {
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
  });
}

async function readReport(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

test("tax-only CLI benchmark routes the model only to candidate actors", async () => {
  const fixture = await fixtureDirectory();
  const outputPath = join(fixture.directory, "tax-only-report.json");
  const result = runBenchmark({
    ...fixture,
    revolutionPath: undefined,
    outputPath,
  });

  assert.equal(result.status, 0, result.stderr || result.stdout);
  const report = await readReport(outputPath);
  assert.equal(report.nonCardEvaluation.ablation, "tax-return-only");
  assert.equal(report.nonCardEvaluation.candidateOnlyRouting, true);
  assert.equal(report.nonCardEvaluation.models.taxReturn.path, fixture.taxPath);
  assert.equal(
    report.nonCardEvaluation.models.taxReturn.sha256,
    fixture.hashes.tax,
  );
  assert.equal(
    report.nonCardEvaluation.models.taxReturn.format,
    "dalmuti-tax-return-action-conditioned-actor-critic",
  );
  assert.equal(report.nonCardEvaluation.models.revolution, null);
  assert.ok(
    report.nonCardEvaluation.routing.taxReturn.candidateModel > 0,
  );
  assert.equal(
    report.nonCardEvaluation.routing.taxReturn.candidateModel,
    report.nonCardEvaluation.routing.taxReturn.learnedAction +
      report.nonCardEvaluation.routing.taxReturn.agreedWithBaseline +
      report.nonCardEvaluation.routing.taxReturn.safetyFallback,
  );
  assert.equal(
    report.nonCardEvaluation.routing.taxReturn.validatedPolicyVersionSteps,
    report.nonCardEvaluation.routing.taxReturn.candidateModel,
  );
  assert.equal(
    report.nonCardEvaluation.models.taxReturn.policyVersion,
    `benchmark-${fixture.hashes.tax.slice(0, 12)}`,
  );
  assert.equal(report.nonCardEvaluation.safetyGate.score, "actor-logit");
  assert.equal(
    report.nonCardEvaluation.safetyGate.taxReturn.minimumAdvantage,
    0,
  );
  assert.equal(
    report.nonCardEvaluation.routing.taxReturn.safetyFallback,
    0,
    "the default zero gate preserves ordinary model argmax",
  );
  assert.equal(
    report.nonCardEvaluation.safetyGate.defaultZeroPreservesModelArgmax,
    true,
  );
  assert.ok(
    report.nonCardEvaluation.routing.taxReturn.normalNormalHeuristic > 0,
  );
  assert.equal(
    report.nonCardEvaluation.routing.taxReturn.candidateNormalHeuristic,
    0,
  );
  assert.equal(report.promotionThresholds.minPointDifference, 0.25);
  assert.equal(report.promotionThresholds.minLowerBound, 0.15);
  assert.equal(report.promotionThresholds.minPairwiseRate, 0.55);
  assert.equal(typeof report.roleRegressionAuditPassed, "boolean");
});

test("revolution-only and combined CLI ablations record independent models", async () => {
  const fixture = await fixtureDirectory();
  const revolutionOutput = join(fixture.directory, "revolution-only.json");
  const revolutionResult = runBenchmark({
    ...fixture,
    taxPath: undefined,
    outputPath: revolutionOutput,
    matches: "2",
    acts: "2",
  });
  assert.equal(
    revolutionResult.status,
    0,
    revolutionResult.stderr || revolutionResult.stdout,
  );
  const revolutionReport = await readReport(revolutionOutput);
  assert.equal(revolutionReport.nonCardEvaluation.ablation, "revolution-only");
  assert.equal(revolutionReport.nonCardEvaluation.models.taxReturn, null);
  assert.equal(
    revolutionReport.nonCardEvaluation.models.revolution.sha256,
    fixture.hashes.revolution,
  );

  const combinedOutput = join(fixture.directory, "combined.json");
  const combinedResult = runBenchmark({
    ...fixture,
    outputPath: combinedOutput,
    matches: "2",
    acts: "2",
  });
  assert.equal(
    combinedResult.status,
    0,
    combinedResult.stderr || combinedResult.stdout,
  );
  const combinedReport = await readReport(combinedOutput);
  assert.equal(
    combinedReport.nonCardEvaluation.ablation,
    "tax-return+revolution",
  );
  assert.equal(
    combinedReport.nonCardEvaluation.models.taxReturn.decisionKind,
    "tax-return",
  );
  assert.equal(
    combinedReport.nonCardEvaluation.models.revolution.decisionKind,
    "revolution",
  );
});

test("same-seed non-card benchmark output is deterministic", async () => {
  const fixture = await fixtureDirectory();
  const firstPath = join(fixture.directory, "first.json");
  const secondPath = join(fixture.directory, "second.json");
  const calibrated = {
    ...fixture,
    taxMinAdvantage: "0.01",
    revolutionMinAdvantage: "0.01",
  };
  const first = runBenchmark({ ...calibrated, outputPath: firstPath });
  const second = runBenchmark({ ...calibrated, outputPath: secondPath });
  assert.equal(first.status, 0, first.stderr || first.stdout);
  assert.equal(second.status, 0, second.stderr || second.stdout);
  const firstReport = await readReport(firstPath);
  const secondReport = await readReport(secondPath);
  delete firstReport.elapsedSeconds;
  delete secondReport.elapsedSeconds;
  assert.deepEqual(firstReport, secondReport);
});

test("positive tax threshold neutralizes a tied harmful low-index model", async () => {
  const fixture = await fixtureDirectory();
  const outputPath = join(fixture.directory, "tax-safety-fallback.json");
  const result = runBenchmark({
    ...fixture,
    playPath: undefined,
    candidatePlay: "normal",
    revolutionPath: undefined,
    taxMinAdvantage: "0.01",
    outputPath,
    matches: "6",
    acts: "3",
  });

  assert.equal(result.status, 0, result.stderr || result.stdout);
  const report = await readReport(outputPath);
  const routing = report.nonCardEvaluation.routing.taxReturn;
  assert.ok(routing.safetyFallback > 0);
  assert.equal(routing.learnedAction, 0);
  assert.equal(
    routing.candidateModel,
    routing.agreedWithBaseline + routing.safetyFallback,
  );
  assert.equal(routing.validatedPolicyVersionSteps, routing.candidateModel);
  assert.equal(
    routing.marginSummaries.modelDiffersFromBaseline.maximum,
    0,
  );
  assert.equal(
    routing.marginSummaries.safetyFallback.count,
    routing.safetyFallback,
  );
  assert.deepEqual(
    report.nonCardEvaluation.safetyGate.taxReturn.baseline,
    {
      implementation: "lib/bot-strategy.ts#chooseBotTaxReturn",
      semanticEncoding:
        "training/non-card-action-space.ts#encodeTaxReturnAction",
      difficulty: "normal",
    },
  );
});

test("model-free normal-play mode isolates a tax model to candidate non-card decisions", async () => {
  const fixture = await fixtureDirectory();
  const outputPath = join(fixture.directory, "normal-play-tax-only.json");
  const result = runBenchmark({
    ...fixture,
    playPath: undefined,
    candidatePlay: "normal",
    revolutionPath: undefined,
    outputPath,
  });

  assert.equal(result.status, 0, result.stderr || result.stdout);
  const report = await readReport(outputPath);
  assert.equal(Object.hasOwn(report, "modelPath"), false);
  assert.equal(report.candidatePlayPolicy.mode, "normal");
  assert.deepEqual(report.candidatePlayPolicy.provenance, {
    implementation: "lib/bot-strategy.ts#chooseBotPlay",
    difficulty: "normal",
    appliesTo: ["candidate", "normal-control"],
  });
  assert.ok(
    report.candidatePlayPolicy.routing.candidateNormalHeuristic > 0,
  );
  assert.ok(
    report.candidatePlayPolicy.routing.normalNormalHeuristic > 0,
  );
  assert.match(
    report.candidatePlayPolicy.validation,
    /behaviorPolicy was normal/,
  );
  assert.equal(report.nonCardEvaluation.ablation, "tax-return-only");
  assert.ok(report.nonCardEvaluation.routing.taxReturn.candidateModel > 0);
  assert.ok(
    report.nonCardEvaluation.routing.taxReturn.normalNormalHeuristic > 0,
  );
});

test("model-free normal-play isolation is deterministic for the same seed", async () => {
  const fixture = await fixtureDirectory();
  const firstPath = join(fixture.directory, "normal-first.json");
  const secondPath = join(fixture.directory, "normal-second.json");
  const common = {
    ...fixture,
    playPath: undefined,
    candidatePlay: "normal",
    revolutionPath: undefined,
  };
  const first = runBenchmark({ ...common, outputPath: firstPath });
  const second = runBenchmark({ ...common, outputPath: secondPath });
  assert.equal(first.status, 0, first.stderr || first.stdout);
  assert.equal(second.status, 0, second.stderr || second.stdout);
  const firstReport = await readReport(firstPath);
  const secondReport = await readReport(secondPath);
  delete firstReport.elapsedSeconds;
  delete secondReport.elapsedSeconds;
  assert.deepEqual(firstReport, secondReport);
});

test("candidate-play mode rejects ambiguous or incomplete model combinations", async () => {
  const fixture = await fixtureDirectory();
  const normalWithPlayModel = runBenchmark({
    ...fixture,
    candidatePlay: "normal",
    revolutionPath: undefined,
    outputPath: join(fixture.directory, "ambiguous.json"),
    matches: "1",
    acts: "1",
  });
  assert.notEqual(normalWithPlayModel.status, 0);
  assert.match(
    normalWithPlayModel.stderr,
    /--model must be omitted when --candidate-play is normal/,
  );

  const modelWithoutPlayModel = runBenchmark({
    ...fixture,
    playPath: undefined,
    candidatePlay: "model",
    revolutionPath: undefined,
    outputPath: join(fixture.directory, "missing.json"),
    matches: "1",
    acts: "1",
  });
  assert.notEqual(modelWithoutPlayModel.status, 0);
  assert.match(
    modelWithoutPlayModel.stderr,
    /--model is required when --candidate-play is model/,
  );

  const normalWithoutNonCardModel = runBenchmark({
    ...fixture,
    playPath: undefined,
    candidatePlay: "normal",
    taxPath: undefined,
    revolutionPath: undefined,
    outputPath: join(fixture.directory, "no-ablation.json"),
    matches: "1",
    acts: "1",
  });
  assert.notEqual(normalWithoutNonCardModel.status, 0);
  assert.match(
    normalWithoutNonCardModel.stderr,
    /requires --tax-model or --revolution-model/,
  );
});

test("minimum-advantage options require matching models and finite nonnegative values", async () => {
  const fixture = await fixtureDirectory();
  const withoutTaxModel = runBenchmark({
    ...fixture,
    taxPath: undefined,
    taxMinAdvantage: "0.1",
    outputPath: join(fixture.directory, "threshold-without-tax.json"),
    matches: "1",
    acts: "1",
  });
  assert.notEqual(withoutTaxModel.status, 0);
  assert.match(
    withoutTaxModel.stderr,
    /--tax-min-advantage requires --tax-model/,
  );

  const negativeTax = runBenchmark({
    ...fixture,
    revolutionPath: undefined,
    taxMinAdvantage: "-0.1",
    outputPath: join(fixture.directory, "negative-tax.json"),
    matches: "1",
    acts: "1",
  });
  assert.notEqual(negativeTax.status, 0);
  assert.match(negativeTax.stderr, /tax-min-advantage must be non-negative/);

  const infiniteRevolution = runBenchmark({
    ...fixture,
    revolutionMinAdvantage: "Infinity",
    outputPath: join(fixture.directory, "infinite-revolution.json"),
    matches: "1",
    acts: "1",
  });
  assert.notEqual(infiniteRevolution.status, 0);
  assert.match(
    infiniteRevolution.stderr,
    /revolution-min-advantage must be a finite number/,
  );
});

test("CLI rejects invalid and decision-mismatched non-card models", async () => {
  const fixture = await fixtureDirectory();
  const mismatchedOutputPath = join(fixture.directory, "mismatched.json");
  const mismatched = runBenchmark({
    ...fixture,
    taxPath: fixture.revolutionPath,
    revolutionPath: undefined,
    outputPath: mismatchedOutputPath,
    matches: "1",
    acts: "1",
  });

  assert.notEqual(mismatched.status, 0);
  assert.match(
    mismatched.stderr,
    /unsupported non-card actor-critic model format/,
  );

  const invalidTaxPath = join(fixture.directory, "invalid-tax.json");
  const invalidTax = zeroNonCardModel("tax-return");
  invalidTax.observationFeatures = 102;
  await writeFile(invalidTaxPath, JSON.stringify(invalidTax), "utf8");
  const invalid = runBenchmark({
    ...fixture,
    taxPath: invalidTaxPath,
    revolutionPath: undefined,
    outputPath: join(fixture.directory, "invalid.json"),
    matches: "1",
    acts: "1",
  });
  assert.notEqual(invalid.status, 0);
  assert.match(invalid.stderr, /non-card observation contract mismatch/);
});

test("omitting non-card flags retains the legacy report schema", async () => {
  const fixture = await fixtureDirectory();
  const outputPath = join(fixture.directory, "legacy.json");
  const result = runBenchmark({
    ...fixture,
    taxPath: undefined,
    revolutionPath: undefined,
    outputPath,
    matches: "1",
    acts: "1",
  });

  assert.equal(result.status, 0, result.stderr || result.stdout);
  const report = await readReport(outputPath);
  assert.equal(Object.hasOwn(report, "nonCardEvaluation"), false);
  assert.equal(Object.hasOwn(report, "candidatePlayPolicy"), false);
  assert.equal(report.modelPath, fixture.playPath);
  assert.equal(report.format, "dalmuti-model-benchmark");
  assert.equal(report.version, 2);
  assert.equal(Object.hasOwn(report.results[0], "nonCardSteps"), false);
});
