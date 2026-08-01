import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdir,
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
  V3_ACTION_COUNT,
  V3_ACTION_FEATURE_COUNT,
  V3_ACTION_FEATURE_LAYOUT,
} from "../training/v3-action-catalogue.ts";

import {
  buildBenchmarkArguments,
  buildScreeningSeedSchedule,
  deterministicConcurrentMap,
  discoverCheckpointCandidates,
  rankScreeningResults,
  screenCheckpointDirectory,
  screeningConcurrency,
} from "../scripts/lib/rl-checkpoint-screening.mjs";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const benchmarkPath = fileURLToPath(
  new URL("../scripts/rl-benchmark-model.mjs", import.meta.url),
);

async function temporaryDirectory(t, label) {
  const path = await mkdtemp(join(tmpdir(), `dalmuti-${label}-`));
  t.after(() => rm(path, { recursive: true, force: true }));
  return path;
}

function model(tag) {
  return `${JSON.stringify({
    format: "dalmuti-actor-critic",
    version: 1,
    observationFeatures: 172,
    actionCount: 506,
    hiddenSizes: [256, 256],
    tag,
  })}\n`;
}

function v3Model(tag) {
  return `${JSON.stringify({
    format: "dalmuti-action-conditioned-actor-critic",
    version: 1,
    observationSchemaVersion: 2,
    observationFeatures: 172,
    actionCatalogueVersion: 1,
    actionCount: 236,
    actionFeatures: 22,
    actionFeatureLayout: ["test-layout"],
    actorObservationHiddenSizes: [256],
    actorActionHiddenSizes: [64],
    actorScorerHiddenSizes: [128],
    valueHiddenSizes: [256],
    tag,
  })}\n`;
}

function layer(inFeatures, outFeatures) {
  return {
    inFeatures,
    outFeatures,
    weight: Array.from({ length: inFeatures * outFeatures }, () => 0),
    bias: Array.from({ length: outFeatures }, () => 0),
  };
}

function runnableV3Model() {
  return {
    format: "dalmuti-action-conditioned-actor-critic",
    version: 1,
    observationSchemaVersion: 2,
    observationFeatures: 172,
    actionCatalogueVersion: 1,
    actionCount: V3_ACTION_COUNT,
    actionFeatures: V3_ACTION_FEATURE_COUNT,
    actionFeatureLayout: V3_ACTION_FEATURE_LAYOUT,
    actorObservationHiddenSizes: [1],
    actorActionHiddenSizes: [1],
    actorScorerHiddenSizes: [],
    valueHiddenSizes: [1],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    actorObservationLayers: [layer(172, 1)],
    actorActionLayers: [layer(V3_ACTION_FEATURE_COUNT, 1)],
    actorScorerLayers: [layer(2, 1)],
    valueLayers: [layer(172, 1), layer(1, 1)],
  };
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
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", reject);
    child.on("exit", (code) => resolvePromise({ code, stdout, stderr }));
  });
}

async function writeReturnedResult(root) {
  const epochOne = join(root, "checkpoints", "epoch-01");
  const epochTwo = join(root, "checkpoints", "epoch-02");
  await mkdir(epochOne, { recursive: true });
  await mkdir(epochTwo, { recursive: true });
  await writeFile(join(epochOne, "actor-critic-weights.json"), model("one"));
  await writeFile(join(epochTwo, "actor-critic-weights.json"), model("two"));
  await writeFile(join(root, "actor-critic-weights.json"), model("two"));
}

async function writeReturnedV3Result(root) {
  const epochOne = join(root, "checkpoints", "epoch-01");
  const epochTwo = join(root, "checkpoints", "epoch-02");
  await mkdir(epochOne, { recursive: true });
  await mkdir(epochTwo, { recursive: true });
  await writeFile(
    join(epochOne, "v3-actor-critic-weights.json"),
    v3Model("one"),
  );
  await writeFile(
    join(epochTwo, "v3-actor-critic-weights.json"),
    v3Model("two"),
  );
  await writeFile(
    join(root, "v3-actor-critic-weights.json"),
    v3Model("two"),
  );
}

function benchmarkReport({ candidate, seed, playerCounts, strength = 0 }) {
  return {
    format: "dalmuti-model-benchmark",
    version: 2,
    modelPath: candidate.canonicalPath,
    modelSha256: candidate.sha256,
    seed,
    playerCounts,
    evaluationDesign: { finalMatchCountPreset: false },
    results: playerCounts.map((playerCount, index) => ({
      playerCount,
      meanChipDifference: strength + 0.3 - index * 0.01,
      meanChipDifference95: {
        low: strength + 0.2 - index * 0.01,
        high: strength + 0.4,
      },
      pairwiseCandidateBeforeNormal: { rate: strength + 0.6 - index * 0.01 },
      effectSizeGate: { passed: strength >= 0 },
    })),
  };
}

test("checkpoint discovery hashes every source and de-duplicates final aliases", async (t) => {
  const root = await temporaryDirectory(t, "checkpoint-discovery");
  await writeReturnedResult(root);
  const discovery = await discoverCheckpointCandidates(root);
  assert.equal(discovery.sourceCount, 3);
  assert.equal(discovery.uniqueHashCount, 2);
  assert.equal(discovery.duplicateSourceCount, 1);
  assert.deepEqual(discovery.candidates.map((candidate) => candidate.id), [
    "epoch-01",
    "final",
  ]);
  assert.deepEqual(discovery.candidates[1].labels, ["epoch-02", "final"]);
  assert.equal(
    discovery.candidates[1].sha256,
    createHash("sha256").update(model("two")).digest("hex"),
  );
  assert.equal(
    new Set(discovery.candidates.map((candidate) => candidate.sha256)).size,
    2,
  );
});

test("checkpoint discovery supports V3 filenames without changing legacy de-duplication", async (t) => {
  const root = await temporaryDirectory(t, "v3-checkpoint-discovery");
  await writeReturnedV3Result(root);
  const discovery = await discoverCheckpointCandidates(root);
  assert.equal(discovery.sourceCount, 3);
  assert.equal(discovery.uniqueHashCount, 2);
  assert.equal(discovery.duplicateSourceCount, 1);
  assert.deepEqual(discovery.candidates.map((candidate) => candidate.id), [
    "epoch-01",
    "final",
  ]);
  assert.equal(
    discovery.candidates[1].canonicalRelativePath,
    "v3-actor-critic-weights.json",
  );
  assert.equal(
    discovery.candidates[1].model.format,
    "dalmuti-action-conditioned-actor-critic",
  );
  assert.deepEqual(
    discovery.candidates[1].model.actorObservationHiddenSizes,
    [256],
  );
});

test("checkpoint discovery rejects ambiguous or renamed model families", async (t) => {
  const renamed = await temporaryDirectory(t, "renamed-v3-checkpoint");
  const renamedEpoch = join(renamed, "checkpoints", "epoch-01");
  await mkdir(renamedEpoch, { recursive: true });
  await writeFile(
    join(renamedEpoch, "actor-critic-weights.json"),
    v3Model("epoch"),
  );
  await writeFile(
    join(renamed, "actor-critic-weights.json"),
    v3Model("final"),
  );
  await assert.rejects(
    discoverCheckpointCandidates(renamed),
    /refusing ambiguous renamed model/,
  );

  const reverseRenamed = await temporaryDirectory(
    t,
    "renamed-legacy-checkpoint",
  );
  const reverseEpoch = join(reverseRenamed, "checkpoints", "epoch-01");
  await mkdir(reverseEpoch, { recursive: true });
  await writeFile(
    join(reverseEpoch, "v3-actor-critic-weights.json"),
    model("epoch"),
  );
  await writeFile(
    join(reverseRenamed, "v3-actor-critic-weights.json"),
    model("final"),
  );
  await assert.rejects(
    discoverCheckpointCandidates(reverseRenamed),
    /refusing ambiguous renamed model/,
  );

  const ambiguous = await temporaryDirectory(t, "ambiguous-checkpoint");
  await writeReturnedResult(ambiguous);
  await writeFile(
    join(ambiguous, "v3-actor-critic-weights.json"),
    v3Model("also-final"),
  );
  await assert.rejects(
    discoverCheckpointCandidates(ambiguous),
    /both legacy and V3 checkpoint filenames/,
  );

  const mixed = await temporaryDirectory(t, "mixed-checkpoint");
  const mixedEpoch = join(mixed, "checkpoints", "epoch-01");
  await mkdir(mixedEpoch, { recursive: true });
  await writeFile(
    join(mixedEpoch, "actor-critic-weights.json"),
    model("epoch"),
  );
  await writeFile(
    join(mixed, "v3-actor-critic-weights.json"),
    v3Model("final"),
  );
  await assert.rejects(
    discoverCheckpointCandidates(mixed),
    /mixed checkpoint model families/,
  );
});

test("benchmark routes a V3 checkpoint through the V3 greedy runtime", async (t) => {
  const root = await temporaryDirectory(t, "v3-benchmark-runtime");
  const modelPath = join(root, "v3-actor-critic-weights.json");
  const reportPath = join(root, "benchmark.json");
  await writeFile(
    modelPath,
    `${JSON.stringify(runnableV3Model())}\n`,
    "utf8",
  );
  const execution = await runNode([
    benchmarkPath,
    "--model",
    modelPath,
    "--matches",
    "1",
    "--acts",
    "1",
    "--players",
    "4",
    "--seed",
    "810001",
    "--omit-match-data",
    "--output",
    reportPath,
  ]);
  assert.equal(execution.code, 0, execution.stderr);
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  assert.equal(report.modelPath, modelPath);
  assert.deepEqual(report.playerCounts, [4]);
  assert.ok(report.results[0].decisions > 0);

  const renamedPath = join(root, "actor-critic-weights.json");
  await writeFile(
    renamedPath,
    `${JSON.stringify(runnableV3Model())}\n`,
    "utf8",
  );
  const renamedExecution = await runNode([
    benchmarkPath,
    "--model",
    renamedPath,
    "--matches",
    "1",
    "--acts",
    "1",
    "--players",
    "4",
    "--seed",
    "810002",
  ]);
  assert.notEqual(renamedExecution.code, 0);
  assert.match(
    renamedExecution.stderr,
    /refusing ambiguous renamed model/,
  );
});

test("screening seed schedules are deterministic, disjoint, and exclude final seeds", () => {
  const options = {
    candidateCount: 3,
    playerCounts: [4, 6],
    matchCountsByPlayerCount: { 4: 30, 6: 20 },
    seedBase: 1_600_001,
    seedStride: 11_000_003,
  };
  const first = buildScreeningSeedSchedule(options);
  const second = buildScreeningSeedSchedule(options);
  assert.deepEqual(first, second);
  assert.deepEqual(first.map((entry) => entry.seed), [
    1_600_001,
    12_600_004,
    23_600_007,
  ]);
  assert.throws(
    () => buildScreeningSeedSchedule({
      ...options,
      explicitSeeds: [100, 100, 200],
    }),
    /distinct/,
  );
  assert.throws(
    () => buildScreeningSeedSchedule({
      ...options,
      explicitSeeds: [100, 200, 300],
      reservedFinalSeeds: [200],
    }),
    /reserved final seeds/,
  );
});

test("ranking uses each candidate's worst player-count metrics conservatively", () => {
  const candidate = (sha256) => ({ id: sha256, sha256 });
  const report = (values, passed = true) => ({
    results: values.map(([playerCount, low, difference, pairwise]) => ({
      playerCount,
      meanChipDifference: difference,
      meanChipDifference95: { low, high: difference + 0.1 },
      pairwiseCandidateBeforeNormal: { rate: pairwise },
      effectSizeGate: { passed },
    })),
  });
  const ranked = rankScreeningResults([
    {
      candidate: candidate("b"),
      benchmark: report([[4, 0.18, 0.35, 0.58], [6, 0.17, 0.4, 0.6]]),
    },
    {
      candidate: candidate("a"),
      benchmark: report([[4, 0.2, 0.31, 0.56], [6, 0.19, 0.3, 0.59]]),
    },
    {
      candidate: candidate("c"),
      benchmark: report([[4, 0.4, 0.5, 0.7]], false),
    },
  ]);
  assert.deepEqual(ranked.map((entry) => entry.candidate.sha256), ["a", "b", "c"]);
  assert.deepEqual(ranked[0].conservative.worstMeanChipDifference, {
    playerCount: 6,
    value: 0.3,
  });
});

test("benchmark arguments use the evaluator's player:matches syntax without final mode", () => {
  const args = buildBenchmarkArguments({
    candidate: { canonicalPath: "model.json" },
    reportPath: "report.json",
    playerCounts: [4, 6],
    matches: 30,
    matchCountsByPlayerCount: { 4: 40, 6: 20 },
    acts: 5,
    seed: 1_600_001,
    thresholds: {
      minPointDifference: 0.25,
      minLowerBound: 0.15,
      minPairwiseRate: 0.55,
    },
  });
  assert.equal(args[args.indexOf("--match-counts") + 1], "4:40,6:20");
  assert.equal(args.includes("--final"), false);
});

test("candidate worker pool is bounded and preserves deterministic input ordering", async () => {
  let active = 0;
  let maximumActive = 0;
  const completionOrder = [];
  const results = await deterministicConcurrentMap(
    [0, 1, 2, 3, 4, 5],
    3,
    async (value) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      await new Promise((resolvePromise) =>
        setTimeout(resolvePromise, (6 - value) * 3),
      );
      completionOrder.push(value);
      active -= 1;
      return `candidate-${value}`;
    },
  );
  assert.equal(maximumActive, 3);
  assert.notDeepEqual(completionOrder, [0, 1, 2, 3, 4, 5]);
  assert.deepEqual(results, [
    "candidate-0",
    "candidate-1",
    "candidate-2",
    "candidate-3",
    "candidate-4",
    "candidate-5",
  ]);
  await assert.rejects(
    deterministicConcurrentMap([1], 33, async (value) => value),
    /must not exceed 32/,
  );
});

test("screening concurrency accepts calibrated high parallelism up to its safety ceiling", () => {
  assert.equal(screeningConcurrency(10), 10);
  assert.equal(screeningConcurrency(32), 32);
  assert.throws(
    () => screeningConcurrency(33),
    /concurrency must not exceed 32/,
  );
});

test("screening invokes non-final benchmarks and writes exclusive manifest/report", async (t) => {
  const root = await temporaryDirectory(t, "checkpoint-screen");
  const returned = join(root, "returned");
  const output = join(root, "screening", "run-1");
  await mkdir(returned);
  await writeReturnedResult(returned);
  const invocations = [];
  let active = 0;
  let maximumActive = 0;
  const runner = async (options) => {
    invocations.push(options);
    active += 1;
    maximumActive = Math.max(maximumActive, active);
    assert.equal(options.args.includes("--final"), false);
    assert.equal(options.args.includes("--omit-match-data"), true);
    const strength = options.candidate.id === "final" ? 0.1 : 0;
    await new Promise((resolvePromise) =>
      setTimeout(resolvePromise, options.candidate.id === "final" ? 2 : 10),
    );
    await writeFile(
      options.reportPath,
      `${JSON.stringify(benchmarkReport({
        candidate: options.candidate,
        seed: options.seed,
        playerCounts: [4, 6],
        strength,
      }))}\n`,
      "utf8",
    );
    active -= 1;
    return { code: 0, stdout: "", stderr: "" };
  };
  const result = await screenCheckpointDirectory({
    directory: returned,
    output,
    playerCounts: [4, 6],
    matches: 2,
    matchCountsByPlayerCount: { 4: 2, 6: 3 },
    acts: 3,
    concurrency: 2,
    seedBase: 2_000_001,
    seedStride: 11_000_003,
    processRunner: runner,
    now: () => new Date("2026-08-01T00:00:00.000Z"),
  });
  assert.equal(invocations.length, 2);
  assert.equal(maximumActive, 2);
  assert.equal(new Set(invocations.map(({ seed }) => seed)).size, 2);
  const manifest = JSON.parse(await readFile(result.manifestPath, "utf8"));
  const report = JSON.parse(await readFile(result.reportPath, "utf8"));
  assert.equal(manifest.evaluation.finalPresetUsed, false);
  assert.equal(manifest.evaluation.concurrency, 2);
  assert.match(manifest.evaluation.finalEvaluationPolicy, /forbidden/);
  assert.equal(report.duplicateSourcesSkipped, 1);
  assert.equal(report.winner.id, "final");
  assert.deepEqual(report.finalEvaluationSeedsForbidden, [
    2_000_001,
    13_000_004,
  ]);
  await assert.rejects(
    screenCheckpointDirectory({
      directory: returned,
      output,
      playerCounts: [4, 6],
      matches: 2,
      matchCountsByPlayerCount: { 4: 2, 6: 3 },
      acts: 3,
      processRunner: runner,
    }),
    /must not already exist/,
  );
  assert.equal(invocations.length, 2);
});

test("a parallel child failure leaves no final report and the run remains non-overwritable", async (t) => {
  const root = await temporaryDirectory(t, "checkpoint-screen-failure");
  const returned = join(root, "returned");
  const output = join(root, "screening", "failed-run");
  await mkdir(returned);
  await writeReturnedResult(returned);
  const configuration = {
    directory: returned,
    output,
    playerCounts: [4],
    matches: 1,
    matchCountsByPlayerCount: { 4: 1 },
    acts: 1,
    concurrency: 2,
  };
  await assert.rejects(
    screenCheckpointDirectory({
      ...configuration,
      processRunner: async () => ({
        code: 7,
        stdout: "",
        stderr: "synthetic child failure",
      }),
    }),
    /synthetic child failure/,
  );
  assert.equal(
    JSON.parse(
      await readFile(join(output, "screening-manifest.json"), "utf8"),
    ).evaluation.concurrency,
    2,
  );
  await assert.rejects(
    readFile(join(output, "screening-report.json")),
    /ENOENT/,
  );
  await assert.rejects(
    screenCheckpointDirectory({
      ...configuration,
      processRunner: async () => ({ code: 0, stdout: "", stderr: "" }),
    }),
    /must not already exist/,
  );
});
