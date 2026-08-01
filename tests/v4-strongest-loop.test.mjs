import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import {
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  V4_DEVELOPMENT_GATES,
  V4_FINAL_GATES,
  V4_FINAL_MATCH_COUNTS,
  bindV4Evaluation,
  buildV4AttemptPlan,
  chooseNextFinalSeed,
  createV4Attempt,
  finalMatchSeedRanges,
  interventionForV4Failure,
  nextDirectiveFromFinalEvaluation,
  readV4AttemptSnapshot,
  recommendV4DevelopmentInterventions,
  recordBoundV4Evaluation,
  recordV4Exit,
  recordV4Heartbeat,
  reserveNextFinalSeed,
  sealFinalV4Evaluation,
  transitionV4Attempt,
} from "../scripts/lib/v4-strongest-loop.mjs";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const bindings = Object.freeze({
  artifactSha256: "a".repeat(64),
  modelSha256: "b".repeat(64),
  observationSchemaSha256: "c".repeat(64),
  normalBaselineSha256:
    "aa44743c64a23ac002d7faf09867bdb3e06232320f8efeb1df0e42724037bb61",
  normalBaselineSourceCommit: "e0c52b0462d86756cf40b90f19d35a3e26b0f674",
});

async function temporaryDirectory(t, label) {
  const path = await mkdtemp(join(tmpdir(), `dalmuti-v4-${label}-`));
  t.after(() => rm(path, { recursive: true, force: true }));
  return path;
}

function runNode(args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, args, {
      cwd: projectRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("exit", (code) => resolvePromise({ code, stdout, stderr }));
  });
}

function benchmark({
  gates = V4_DEVELOPMENT_GATES,
  mean = gates.minMeanChipDifference,
  lower = gates.minLowerConfidenceBound,
  pairwise = gates.minPairwiseRate,
  final = false,
  seed = 12345,
} = {}) {
  const results = Array.from({ length: 7 }, (_, index) => ({
    playerCount: index + 4,
    matches: final ? V4_FINAL_MATCH_COUNTS[index + 4] : 60,
    actsPerMatch: 5,
    meanChipDifference: mean,
    meanChipDifference95: { low: lower, high: lower + 0.1 },
    pairwiseCandidateBeforeNormal: { rate: pairwise },
    effectSizeGate: { passed: true },
  }));
  return {
    format: "dalmuti-model-benchmark",
    version: 2,
    evaluationMode: final ? "final" : "development",
    modelSha256: bindings.modelSha256,
    bindings: { ...bindings },
    bindingEvidence: {
      format: "dalmuti-v4-actual-input-binding-evidence",
      version: 1,
      actualFilesVerified: true,
      actorBundleArtifactSha256: bindings.artifactSha256,
      actorModelSha256: bindings.modelSha256,
      observationContractSha256: bindings.observationSchemaSha256,
      normalSourceSha256: bindings.normalBaselineSha256,
      normalSourceCommit: bindings.normalBaselineSourceCommit,
      normalCommitBlobMatchesWorkingSource: true,
    },
    candidatePolicy: {
      actorCount: 1,
      bundleActorSha256s: [bindings.modelSha256],
      bundleManifestSha256s: [bindings.artifactSha256],
      bundleArtifactSha256: bindings.artifactSha256,
    },
    seed,
    playerCounts: [4, 5, 6, 7, 8, 9, 10],
    matchCountsByPlayerCount: final
      ? { ...V4_FINAL_MATCH_COUNTS }
      : { 4: 60, 5: 60, 6: 60, 7: 60, 8: 60, 9: 60, 10: 60 },
    actsPerMatch: 5,
    evaluationDesign: { finalMatchCountPreset: final },
    promotionThresholds: {
      minPointDifference: gates.minMeanChipDifference,
      minLowerBound: gates.minLowerConfidenceBound,
      minPairwiseRate: gates.minPairwiseRate,
    },
    promotionPassed: results.every((result) => result.effectSizeGate.passed),
    results,
  };
}

test("attempt plan binds every artifact and explicitly excludes deployment", () => {
  const plan = buildV4AttemptPlan({
    root: "results/v4",
    attemptNumber: 7,
    label: "transformer-ensemble",
    bindings,
  });
  assert.equal(plan.attemptId, "v4-strongest-attempt-007-transformer-ensemble");
  assert.deepEqual(plan.bindings, bindings);
  assert.equal(plan.evaluationContract.finalGates.minMeanChipDifference, 0.25);
  assert.equal(plan.evaluationContract.developmentGates.minMeanChipDifference, 0.3);
  assert.match(plan.outOfScope.join(" "), /Sites deployment/);
  assert.match(plan.prohibitedTrainingInputs.join(" "), /final evaluation metrics/);
});

test("attempt directories and exit records are never reused", async (t) => {
  const root = await temporaryDirectory(t, "attempt");
  const plan = buildV4AttemptPlan({ root, attemptNumber: 1, bindings });
  await createV4Attempt(plan, { now: new Date("2026-08-01T00:00:00Z") });
  await assert.rejects(createV4Attempt(plan), /must not already exist/);
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "failed",
    reason: "test failure",
  });
  await recordV4Exit({
    attemptDirectory: plan.attemptDirectory,
    outcome: "failed",
    failureClass: "integrity",
  });
  await assert.rejects(
    recordV4Exit({
      attemptDirectory: plan.attemptDirectory,
      outcome: "failed",
      failureClass: "integrity",
    }),
    /EEXIST|already exists/,
  );
});

test("state and heartbeat journals form ordered immutable chains", async (t) => {
  const root = await temporaryDirectory(t, "journals");
  const plan = buildV4AttemptPlan({ root, attemptNumber: 2, bindings });
  await createV4Attempt(plan);
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "training",
    reason: "contracts passed",
  });
  await recordV4Heartbeat({
    attemptDirectory: plan.attemptDirectory,
    phase: "gpu-training",
    progress: 0.25,
    message: "epoch 1",
  });
  await recordV4Heartbeat({
    attemptDirectory: plan.attemptDirectory,
    phase: "gpu-training",
    progress: 0.5,
    message: "epoch 2",
  });
  const snapshot = await readV4AttemptSnapshot(plan.attemptDirectory);
  assert.equal(snapshot.currentState, "training");
  assert.equal(snapshot.stateSequence, 2);
  assert.equal(snapshot.latestHeartbeat.sequence, 2);
  assert.equal(snapshot.latestHeartbeat.previousRecordSha256.length, 64);
  await assert.rejects(
    transitionV4Attempt({
      attemptDirectory: plan.attemptDirectory,
      to: "passed",
      reason: "illegal shortcut",
    }),
    /invalid attempt transition/,
  );
});

test("failure classes map to deterministic fresh-attempt interventions", () => {
  assert.deepEqual(interventionForV4Failure("oom"), {
    failureClass: "oom",
    action: "reduce-microbatch-and-increase-gradient-accumulation",
    retry: "fresh-attempt",
    feedbackAllowed: true,
  });
  assert.equal(
    interventionForV4Failure("sealed-final-failure").feedbackAllowed,
    false,
  );
  assert.throws(() => interventionForV4Failure("guess"), /unsupported/);
});

test("final seed schedule starts at 900000001 then 920000001 and skips collisions", () => {
  assert.equal(chooseNextFinalSeed().baseSeed, 900_000_001);
  assert.equal(
    chooseNextFinalSeed({ reservations: [{ baseSeed: 900_000_001 }] }).baseSeed,
    920_000_001,
  );
  const collision = finalMatchSeedRanges(920_000_001)[3];
  assert.equal(
    chooseNextFinalSeed({
      reservations: [{ baseSeed: 900_000_001 }],
      collisionRanges: [collision],
    }).baseSeed,
    940_000_001,
  );
  assert.equal(
    chooseNextFinalSeed({ collisionRanges: [900_000_001] }).baseSeed,
    920_000_001,
  );
});

test("atomic seed reservation consumes a seed forever", async (t) => {
  const root = await temporaryDirectory(t, "registry");
  const first = await reserveNextFinalSeed({
    registryDirectory: root,
    attemptId: "v4-strongest-attempt-001-strongest",
    bindings,
  });
  const second = await reserveNextFinalSeed({
    registryDirectory: root,
    attemptId: "v4-strongest-attempt-002-strongest",
    bindings,
  });
  assert.equal(first.baseSeed, 900_000_001);
  assert.equal(second.baseSeed, 920_000_001);
  assert.notEqual(first.reservation.attemptId, second.reservation.attemptId);
});

test("development gates are exact for every p4 through p10", () => {
  const passing = bindV4Evaluation({
    stage: "development",
    benchmark: benchmark(),
    bindings,
  });
  assert.equal(passing.gateSummary.passed, true);
  const failingBenchmark = benchmark({ mean: 0.299 });
  for (const result of failingBenchmark.results) result.effectSizeGate.passed = false;
  failingBenchmark.promotionPassed = false;
  const failing = bindV4Evaluation({
    stage: "development",
    benchmark: failingBenchmark,
    bindings,
  });
  assert.equal(failing.gateSummary.passed, false);
  assert.equal(
    recommendV4DevelopmentInterventions(failing)[0].failureClass,
    "mean-chip-gap",
  );
});

test("development intervention cannot inspect a sealed final evaluation", () => {
  const reservation = {
    format: "dalmuti-v4-final-seed-reservation",
    version: 1,
    baseSeed: 900_000_001,
    attemptId: "v4-strongest-attempt-001-strongest",
    bindings,
    bindingSha256: buildV4AttemptPlan({ root: ".", attemptNumber: 1, bindings }).bindingSha256,
    matchSeedRanges: finalMatchSeedRanges(900_000_001),
    reuseForbidden: true,
    finalFeedbackPolicy: "sealed-holdout-not-a-training-input",
  };
  const final = bindV4Evaluation({
    stage: "final",
    benchmark: benchmark({ gates: V4_FINAL_GATES, final: true, seed: 900_000_001 }),
    bindings,
    finalSeedReservation: reservation,
  });
  assert.throws(
    () => recommendV4DevelopmentInterventions(final),
    /only development metrics/,
  );
});

test("final evaluation enforces counts, five acts, bindings, and reserved seed", () => {
  const bindingDigest = buildV4AttemptPlan({ root: ".", attemptNumber: 1, bindings }).bindingSha256;
  const reservation = {
    format: "dalmuti-v4-final-seed-reservation",
    version: 1,
    baseSeed: 900_000_001,
    attemptId: "v4-strongest-attempt-001-strongest",
    bindings,
    bindingSha256: bindingDigest,
    matchSeedRanges: finalMatchSeedRanges(900_000_001),
    reuseForbidden: true,
    finalFeedbackPolicy: "sealed-holdout-not-a-training-input",
  };
  const report = benchmark({
    gates: V4_FINAL_GATES,
    final: true,
    seed: reservation.baseSeed,
  });
  const bound = bindV4Evaluation({
    stage: "final",
    benchmark: report,
    bindings,
    finalSeedReservation: reservation,
  });
  assert.equal(bound.gateSummary.passed, true);
  assert.equal(nextDirectiveFromFinalEvaluation(bound).loopComplete, true);
  const wrongCounts = structuredClone(report);
  wrongCounts.matchCountsByPlayerCount[10] = 299;
  assert.throws(
    () => bindV4Evaluation({
      stage: "final",
      benchmark: wrongCounts,
      bindings,
      finalSeedReservation: reservation,
    }),
    /p10 result does not match|p10 must be 300/,
  );
  const wrongModel = structuredClone(report);
  wrongModel.modelSha256 = "d".repeat(64);
  assert.throws(
    () => bindV4Evaluation({
      stage: "final",
      benchmark: wrongModel,
      bindings,
      finalSeedReservation: reservation,
    }),
    /model SHA-256/,
  );
  const forgedObservation = structuredClone(report);
  forgedObservation.bindings.observationSchemaSha256 = "d".repeat(64);
  forgedObservation.bindingEvidence.observationContractSha256 = "d".repeat(64);
  assert.throws(
    () => bindV4Evaluation({
      stage: "final",
      benchmark: forgedObservation,
      bindings,
      finalSeedReservation: reservation,
    }),
    /bindings do not exactly match/,
  );
});

test("a failed final directive contains no metric feedback", () => {
  const report = benchmark({
    gates: V4_FINAL_GATES,
    final: true,
    seed: 900_000_001,
    pairwise: 0.549,
  });
  for (const result of report.results) result.effectSizeGate.passed = false;
  report.promotionPassed = false;
  const plan = buildV4AttemptPlan({ root: ".", attemptNumber: 1, bindings });
  const bound = bindV4Evaluation({
    stage: "final",
    benchmark: report,
    bindings,
    finalSeedReservation: {
      format: "dalmuti-v4-final-seed-reservation",
      version: 1,
      baseSeed: 900_000_001,
      attemptId: plan.attemptId,
      bindings,
      bindingSha256: plan.bindingSha256,
      matchSeedRanges: finalMatchSeedRanges(900_000_001),
      reuseForbidden: true,
      finalFeedbackPolicy: "sealed-holdout-not-a-training-input",
    },
  });
  const directive = nextDirectiveFromFinalEvaluation(bound);
  assert.equal(directive.loopComplete, false);
  assert.equal(directive.feedbackAllowed, false);
  assert.equal(Object.hasOwn(directive, "gateSummary"), false);
  assert.equal(JSON.stringify(directive).includes("0.549"), false);
});

test("passed requires the actual sealed report and every final gate", async (t) => {
  const root = await temporaryDirectory(t, "evaluation");
  const plan = buildV4AttemptPlan({ root, attemptNumber: 1, bindings });
  await createV4Attempt(plan);
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "development-evaluation",
    reason: "candidate frozen",
  });
  const development = bindV4Evaluation({
    stage: "development",
    benchmark: benchmark(),
    bindings,
  });
  await recordBoundV4Evaluation({
    attemptDirectory: plan.attemptDirectory,
    boundEvaluation: development,
  });
  const forgedBound = structuredClone(development);
  forgedBound.gateSummary.passed = false;
  await assert.rejects(
    recordBoundV4Evaluation({
      attemptDirectory: plan.attemptDirectory,
      boundEvaluation: forgedBound,
    }),
    /exact canonical evaluation/,
  );
  await assert.rejects(
    recordBoundV4Evaluation({
      attemptDirectory: plan.attemptDirectory,
      boundEvaluation: development,
    }),
    /EEXIST|already exists/,
  );

  const registry = join(root, "final-seed-registry");
  const reserved = await reserveNextFinalSeed({
    registryDirectory: registry,
    attemptId: plan.attemptId,
    bindings,
  });
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "ready-for-final",
    reason: "both development families passed",
  });
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "final-evaluation",
    reason: "fresh final seed reserved",
  });
  const final = bindV4Evaluation({
    stage: "final",
    benchmark: benchmark({
      gates: V4_FINAL_GATES,
      final: true,
      seed: reserved.baseSeed,
    }),
    bindings,
    finalSeedReservation: reserved.reservation,
  });
  const recorded = await recordBoundV4Evaluation({
    attemptDirectory: plan.attemptDirectory,
    boundEvaluation: final,
  });
  await assert.rejects(
    transitionV4Attempt({
      attemptDirectory: plan.attemptDirectory,
      to: "passed",
      reason: "must not pass before sealing",
      finalSeedRegistryDirectory: registry,
    }),
    /seal is missing/,
  );
  const sealed = await sealFinalV4Evaluation({
    registryDirectory: registry,
    attemptDirectory: plan.attemptDirectory,
    boundEvaluation: final,
    reportSha256: recorded.sha256,
  });
  assert.equal(sealed.seal.containsEvaluationMetrics, false);
  assert.equal(JSON.stringify(sealed.seal).includes("meanChipDifference"), false);
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "passed",
    reason: "sealed final gates passed",
    finalSeedRegistryDirectory: registry,
  });
  const snapshot = await readV4AttemptSnapshot(plan.attemptDirectory);
  assert.equal(snapshot.currentState, "passed");
  const finalPath = join(plan.attemptDirectory, "evaluations", "final.json");
  const beforeTamper = await readFile(finalPath, "utf8");
  const match = /("benchmarkSha256": ")([a-f0-9])/.exec(beforeTamper);
  assert.ok(match);
  const replacement = match[2] === "0" ? "1" : "0";
  await writeFile(
    finalPath,
    beforeTamper.replace(match[0], `${match[1]}${replacement}`),
  );
  await assert.rejects(
    readV4AttemptSnapshot(plan.attemptDirectory),
    /canonical evaluation|attest|no longer verifies/,
  );
});

test("a sealed failed gate can never transition the attempt to passed", async (t) => {
  const root = await temporaryDirectory(t, "failed-final");
  const plan = buildV4AttemptPlan({ root, attemptNumber: 2, bindings });
  await createV4Attempt(plan);
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "development-evaluation",
    reason: "test development evaluation",
  });
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "ready-for-final",
    reason: "test development gate",
  });
  await transitionV4Attempt({
    attemptDirectory: plan.attemptDirectory,
    to: "final-evaluation",
    reason: "test final evaluation",
  });
  const registry = join(root, "final-seed-registry");
  const reserved = await reserveNextFinalSeed({
    registryDirectory: registry,
    attemptId: plan.attemptId,
    bindings,
  });
  const report = benchmark({
    gates: V4_FINAL_GATES,
    final: true,
    seed: reserved.baseSeed,
    pairwise: 0.549,
  });
  for (const result of report.results) result.effectSizeGate.passed = false;
  report.promotionPassed = false;
  const final = bindV4Evaluation({
    stage: "final",
    benchmark: report,
    bindings,
    finalSeedReservation: reserved.reservation,
  });
  await recordBoundV4Evaluation({
    attemptDirectory: plan.attemptDirectory,
    boundEvaluation: final,
  });
  await sealFinalV4Evaluation({
    registryDirectory: registry,
    attemptDirectory: plan.attemptDirectory,
    boundEvaluation: final,
  });
  await assert.rejects(
    transitionV4Attempt({
      attemptDirectory: plan.attemptDirectory,
      to: "passed",
      reason: "forged passing outcome",
      finalSeedRegistryDirectory: registry,
    }),
    /every final p4-p10 gate/,
  );
});

test("CLI dry-run is deterministic and writes no attempt directory", async (t) => {
  const root = await temporaryDirectory(t, "cli");
  const outputRoot = join(root, "runs");
  const args = [
    "scripts/rl-v4-strongest-loop.mjs",
    "--root", outputRoot,
    "--attempt", "3",
    "--label", "transformer-ensemble",
    "--artifact-sha256", bindings.artifactSha256,
    "--model-sha256", bindings.modelSha256,
    "--schema-sha256", bindings.observationSchemaSha256,
    "--baseline-sha256", bindings.normalBaselineSha256,
    "--baseline-source-commit", bindings.normalBaselineSourceCommit,
    "--dry-run",
  ];
  const first = await runNode(args);
  const second = await runNode(args);
  assert.equal(first.code, 0, first.stderr);
  assert.equal(second.code, 0, second.stderr);
  assert.equal(first.stdout, second.stdout);
  const output = JSON.parse(first.stdout);
  assert.equal(output.dryRun, true);
  assert.equal(output.plan.attemptId, "v4-strongest-attempt-003-transformer-ensemble");
  await assert.rejects(readFile(output.plan.attemptDirectory));
});
