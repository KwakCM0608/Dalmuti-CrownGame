import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  V3_ACTION_COUNT,
  V3_ACTION_FEATURE_COUNT,
  V3_ACTION_FEATURE_LAYOUT,
} from "../training/v3-action-catalogue.ts";
import { mergeBenchmarkShardReports } from "../scripts/lib/rl-benchmark-shard-merge.mjs";
import { screenCheckpointDirectory } from "../scripts/lib/rl-checkpoint-screening.mjs";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const benchmarkPath = join(projectRoot, "scripts", "rl-benchmark-model.mjs");
const mergePath = join(projectRoot, "scripts", "rl-merge-benchmark-shards.mjs");

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

async function tempRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "dalmuti-distributed-benchmark-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function readEntry(path) {
  const bytes = await readFile(path);
  return {
    path,
    bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    report: JSON.parse(bytes.toString("utf8")),
  };
}

test("candidate/player/seed shards merge exactly and independently of input order", async (t) => {
  const root = await tempRoot(t);
  const modelPath = join(root, "v3-actor-critic-weights.json");
  const monoPath = join(root, "monolithic.json");
  const shardZeroPath = join(root, "shard-0.json");
  const shardOnePath = join(root, "shard-1.json");
  const mergedPath = join(root, "merged.json");
  const reversedPath = join(root, "merged-reversed.json");
  await writeFile(modelPath, `${JSON.stringify(runnableV3Model())}\n`);
  const common = [
    "--model", modelPath,
    "--matches", "4",
    "--match-counts", "4:4,5:4",
    "--acts", "1",
    "--players", "4,5",
    "--seed", "187000001",
    "--omit-match-data",
  ];
  const monolithic = await runNode([
    benchmarkPath,
    ...common,
    "--output", monoPath,
  ]);
  assert.equal(monolithic.code, 0, monolithic.stderr);
  for (const [index, output] of [[0, shardZeroPath], [1, shardOnePath]]) {
    const execution = await runNode([
      benchmarkPath,
      ...common,
      "--shard-index", String(index),
      "--shard-count", "2",
      "--output", output,
    ]);
    assert.equal(execution.code, 0, execution.stderr);
  }
  const mergeArgs = [
    "--model", modelPath,
    "--players", "4,5",
    "--match-counts", "4:4,5:4",
    "--acts", "1",
    "--seed", "187000001",
  ];
  const merged = await runNode([
    mergePath,
    ...mergeArgs,
    "--shard", shardZeroPath,
    "--shard", shardOnePath,
    "--output", mergedPath,
  ]);
  assert.equal(merged.code, 0, merged.stderr);
  const reversed = await runNode([
    mergePath,
    ...mergeArgs,
    "--shard", shardOnePath,
    "--shard", shardZeroPath,
    "--output", reversedPath,
  ]);
  assert.equal(reversed.code, 0, reversed.stderr);
  const monolithicReport = JSON.parse(await readFile(monoPath, "utf8"));
  const mergedReport = JSON.parse(await readFile(mergedPath, "utf8"));
  const reversedReport = JSON.parse(await readFile(reversedPath, "utf8"));
  assert.deepEqual(mergedReport.results, monolithicReport.results);
  assert.deepEqual(mergedReport.pooled, monolithicReport.pooled);
  assert.deepEqual(reversedReport, mergedReport);
  assert.match(mergedReport.distributedMerge.seedCoverage, /exactly once/);
});

test("strict merge rejects duplicate shards, missing seeds, and model hash drift", async (t) => {
  const root = await tempRoot(t);
  const modelPath = join(root, "v3-actor-critic-weights.json");
  const shardZeroPath = join(root, "shard-0.json");
  const shardOnePath = join(root, "shard-1.json");
  const modelBytes = Buffer.from(`${JSON.stringify(runnableV3Model())}\n`);
  await writeFile(modelPath, modelBytes);
  for (const [index, output] of [[0, shardZeroPath], [1, shardOnePath]]) {
    const execution = await runNode([
      benchmarkPath,
      "--model", modelPath,
      "--matches", "4",
      "--acts", "1",
      "--players", "4",
      "--seed", "188000001",
      "--shard-index", String(index),
      "--shard-count", "2",
      "--omit-match-data",
      "--output", output,
    ]);
    assert.equal(execution.code, 0, execution.stderr);
  }
  const first = await readEntry(shardZeroPath);
  const second = await readEntry(shardOnePath);
  const expected = {
    modelPath,
    modelSha256: createHash("sha256").update(modelBytes).digest("hex"),
    playerCounts: [4],
    matchCountsByPlayerCount: { 4: 4 },
    acts: 1,
    seed: 188000001,
    promotionThresholds: {
      minPointDifference: 0.25,
      minLowerBound: 0.15,
      minPairwiseRate: 0.55,
    },
    roleRegressionMargin: 0.1,
  };
  assert.throws(
    () => mergeBenchmarkShardReports([
      first,
      { ...first, path: `${first.path}.copy` },
    ], expected),
    /duplicate shard index/,
  );
  const missing = structuredClone(second);
  missing.report.distributedShard.records.pop();
  assert.throws(
    () => mergeBenchmarkShardReports([first, missing], expected),
    /missing or extra match seeds/,
  );
  assert.throws(
    () => mergeBenchmarkShardReports([first, second], {
      ...expected,
      modelSha256: "0".repeat(64),
    }),
    /model SHA-256/,
  );
});

test("checkpoint screening parallelizes candidate x player x seed jobs and hashes merged reports", async (t) => {
  const root = await tempRoot(t);
  const returned = join(root, "returned");
  const epochOne = join(returned, "checkpoints", "epoch-01");
  const epochTwo = join(returned, "checkpoints", "epoch-02");
  const output = join(root, "screening");
  await mkdir(epochOne, { recursive: true });
  await mkdir(epochTwo, { recursive: true });
  const first = runnableV3Model();
  const second = runnableV3Model();
  second.actorScorerLayers[0].bias[0] = 0.1;
  await writeFile(
    join(epochOne, "v3-actor-critic-weights.json"),
    `${JSON.stringify(first)}\n`,
  );
  await writeFile(
    join(epochTwo, "v3-actor-critic-weights.json"),
    `${JSON.stringify(second)}\n`,
  );
  await writeFile(
    join(returned, "v3-actor-critic-weights.json"),
    `${JSON.stringify(second)}\n`,
  );
  const result = await screenCheckpointDirectory({
    directory: returned,
    output,
    playerCounts: [4],
    matches: 4,
    matchCountsByPlayerCount: { 4: 4 },
    acts: 1,
    concurrency: 4,
    benchmarkShards: 2,
    seedBase: 189000001,
    seedStride: 11000003,
    reservedFinalSeeds: [900000001],
  });
  assert.equal(result.manifest.evaluation.benchmarkShards, 2);
  assert.equal(result.report.ranking.length, 2);
  for (const entry of result.report.ranking) {
    assert.match(entry.benchmarkReportSha256, /^[0-9a-f]{64}$/);
    assert.equal(entry.benchmark.distributedMerge.shardReports.length, 2);
    assert.equal(entry.benchmark.results[0].matches, 4);
  }
});
