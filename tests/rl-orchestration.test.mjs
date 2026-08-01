import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
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
import { inflateRawSync } from "node:zlib";

import {
  assertMissingDirectory,
  createPortableZip,
  parsePlayerCountOverrides,
  readRolloutEnvelope,
} from "../scripts/lib/rl-orchestration.mjs";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

async function temporaryDirectory(t, label) {
  const path = await mkdtemp(join(tmpdir(), `dalmuti-${label}-`));
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
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      resolvePromise({ code, signal, stdout, stderr });
    });
  });
}

function readZipEntries(bytes) {
  let endOffset = bytes.length - 22;
  while (endOffset >= 0 && bytes.readUInt32LE(endOffset) !== 0x06054b50) {
    endOffset -= 1;
  }
  assert.ok(endOffset >= 0, "ZIP end-of-central-directory record is missing");
  const entryCount = bytes.readUInt16LE(endOffset + 10);
  let offset = bytes.readUInt32LE(endOffset + 16);
  const entries = [];
  for (let index = 0; index < entryCount; index += 1) {
    assert.equal(bytes.readUInt32LE(offset), 0x02014b50);
    const method = bytes.readUInt16LE(offset + 10);
    const compressedSize = bytes.readUInt32LE(offset + 20);
    const nameLength = bytes.readUInt16LE(offset + 28);
    const extraLength = bytes.readUInt16LE(offset + 30);
    const commentLength = bytes.readUInt16LE(offset + 32);
    const localOffset = bytes.readUInt32LE(offset + 42);
    const name = bytes
      .subarray(offset + 46, offset + 46 + nameLength)
      .toString("utf8");
    assert.equal(bytes.readUInt32LE(localOffset), 0x04034b50);
    const localNameLength = bytes.readUInt16LE(localOffset + 26);
    const localExtraLength = bytes.readUInt16LE(localOffset + 28);
    const dataOffset =
      localOffset + 30 + localNameLength + localExtraLength;
    const compressed = bytes.subarray(
      dataOffset,
      dataOffset + compressedSize,
    );
    assert.equal(method, 8);
    entries.push({ name, content: inflateRawSync(compressed).toString("utf8") });
    offset += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}

test("player-count overrides are explicit and reject duplicates", () => {
  assert.deepEqual(
    [...parsePlayerCountOverrides(["4=900,5=300", "10=200"], "episodes")],
    [
      [4, 900],
      [5, 300],
      [10, 200],
    ],
  );
  assert.throws(
    () => parsePlayerCountOverrides(["4=10", "4=20"], "episodes"),
    /repeats player count 4/,
  );
  assert.throws(
    () => parsePlayerCountOverrides(["3=10"], "episodes"),
    /from 4 to 10/,
  );
});

test("all pre-existing run directories are rejected", async (t) => {
  const root = await temporaryDirectory(t, "nonempty-run");
  const missing = join(root, "missing");
  await assert.doesNotReject(() =>
    assertMissingDirectory(missing, "run directory"),
  );
  await writeFile(join(root, "existing.txt"), "keep", "utf8");
  await assert.rejects(
    assertMissingDirectory(root, "run directory"),
    /must not already exist/,
  );
});

test("portable ZIP streams valid files with forward-slash entry names", async (t) => {
  const root = await temporaryDirectory(t, "portable-zip");
  const source = join(root, "gpu-bundle");
  await mkdir(join(source, "nested"), { recursive: true });
  await writeFile(join(source, "alpha.txt"), "alpha", "utf8");
  await writeFile(join(source, "nested", "beta.txt"), "beta", "utf8");
  const archivePath = join(root, "bundle.zip");
  const result = await createPortableZip({
    sourceDirectory: source,
    archivePath,
  });
  assert.deepEqual(result.entries, [
    "gpu-bundle/alpha.txt",
    "gpu-bundle/nested/beta.txt",
  ]);
  assert.equal(result.entries.some((name) => name.includes("\\")), false);
  assert.deepEqual(readZipEntries(await readFile(archivePath)), [
    { name: "gpu-bundle/alpha.txt", content: "alpha" },
    { name: "gpu-bundle/nested/beta.txt", content: "beta" },
  ]);
});

test("rollout envelope exposes reproducible collection counts", async (t) => {
  const root = await temporaryDirectory(t, "rollout-envelope");
  const path = join(root, "rollout.ndjson");
  const sha256 = "a".repeat(64);
  await writeFile(
    path,
    [
      JSON.stringify({
        type: "manifest",
        format: "dalmuti-ppo-ndjson",
        behaviorModel: { sha256 },
        environment: { playerCount: 4, initialSeed: 123 },
      }),
      JSON.stringify({ type: "sample" }),
      JSON.stringify({
        type: "summary",
        episodes: 2,
        learnerSamples: 11,
        forcedSamples: 7,
        nonForcedSamples: 4,
        environmentDecisions: 20,
        behaviorModelSha256: sha256,
      }),
      "",
    ].join("\n"),
    "utf8",
  );
  assert.deepEqual((await readRolloutEnvelope(path)).counts, {
    episodes: 2,
    learnerSamples: 11,
    forcedSamples: 7,
    nonForcedSamples: 4,
    environmentDecisions: 20,
  });
});

test("target collection stops on non-forced decisions and removes its partial file", async (t) => {
  const root = await temporaryDirectory(t, "target-rollout");
  const modelPath = join(root, "uniform-policy.json");
  const rolloutPath = join(root, "target.ndjson");
  const observationFeatures = 172;
  const actionCount = 506;
  await writeFile(
    modelPath,
    JSON.stringify({
      format: "dalmuti-mlp-policy",
      version: 1,
      observationFeatures,
      actionCount,
      hiddenSizes: [],
      activation: "relu",
      weightLayout: "row-major [out_features, in_features]",
      layers: [
        {
          inFeatures: observationFeatures,
          outFeatures: actionCount,
          weight: Array.from(
            { length: observationFeatures * actionCount },
            () => 0,
          ),
          bias: Array.from({ length: actionCount }, () => 0),
        },
      ],
    }),
    "utf8",
  );
  const result = await runNode([
    "scripts/rl-generate-league-rollouts.mjs",
    "--model",
    modelPath,
    "--players",
    "4",
    "--acts",
    "1",
    "--seed",
    "810001",
    "--temperature",
    "1.25",
    "--target-non-forced-decisions",
    "1",
    "--max-episodes",
    "1",
    "--output",
    rolloutPath,
  ]);
  assert.equal(result.code, 0, result.stderr);
  const envelope = await readRolloutEnvelope(rolloutPath);
  assert.equal(envelope.manifest.environment.episodes, 1);
  assert.deepEqual(envelope.manifest.behaviorPolicy, {
    sampling: "softmax",
    temperature: 1.25,
  });
  assert.ok(envelope.counts.nonForcedSamples >= 1);
  await assert.rejects(readFile(`${rolloutPath}.partial`));
});

test("bundle preparation records hashes, counts, mix, and GPU temperature", async (t) => {
  const root = await temporaryDirectory(t, "bundle-smoke");
  const modelPath = join(root, "parent.json");
  const rolloutPath = join(root, "p4.ndjson");
  const outputPath = join(root, "bundle");
  const model = JSON.stringify({ format: "test-model", version: 1 });
  await writeFile(modelPath, model, "utf8");
  const sha256 = createHash("sha256").update(model).digest("hex");
  await writeFile(
    rolloutPath,
    [
      JSON.stringify({
        type: "manifest",
        format: "dalmuti-ppo-ndjson",
        behaviorModel: { sha256 },
        behaviorPolicy: { sampling: "softmax", temperature: 1.25 },
        environment: {
          playerCount: 4,
          actsPerEpisode: 3,
          initialSeed: 9001,
          opponentMix: { normalFraction: 0.75 },
        },
      }),
      JSON.stringify({
        type: "summary",
        episodes: 2,
        learnerSamples: 11,
        forcedSamples: 7,
        nonForcedSamples: 4,
        environmentDecisions: 20,
        behaviorModelSha256: sha256,
        opponentSeatAssignments: { normal: 3, byModelSha256: {} },
      }),
      "",
    ].join("\n"),
    "utf8",
  );
  const result = await runNode([
    "scripts/rl-prepare-ppo-bundle.mjs",
    "--model",
    modelPath,
    "--rollout",
    rolloutPath,
    "--output",
    outputPath,
  ]);
  assert.equal(result.code, 0, result.stderr);
  const manifest = JSON.parse(
    await readFile(join(outputPath, "bundle-manifest.json"), "utf8"),
  );
  assert.equal(manifest.parentModel.sha256, sha256);
  assert.equal(manifest.rollouts[0].temperature, 1.25);
  assert.equal(manifest.rollouts[0].opponentMix.normalFraction, 0.75);
  assert.equal(manifest.dataCounts.nonForcedSamples, 4);
  const gpuConfig = JSON.parse(
    await readFile(join(outputPath, "gpu-run-config.json"), "utf8"),
  );
  assert.deepEqual(gpuConfig.correctedPpoContract, {
    epochs: 12,
    batchSize: 4096,
    learningRate: 0.0001,
    gamma: 1,
    gaeLambda: 1,
    skipForcedPolicyTime: true,
    clipCoefficient: 0.2,
    valueCoefficient: 0.5,
    entropyCoefficient: 0.01,
    targetKl: 0.015,
    terminalRankAuxiliaryVariants: [0, 0.05],
  });
  assert.deepEqual(gpuConfig.requiredRunGpuPpoArguments, [
    "--epochs",
    "12",
    "--batch-size",
    "4096",
    "--learning-rate",
    "0.0001",
    "--gamma",
    "1",
    "--gae-lambda",
    "1",
    "--skip-forced-policy-time",
    "--rollout-temperature",
    "1.25",
    "--clip-coefficient",
    "0.2",
    "--value-coefficient",
    "0.5",
    "--entropy-coefficient",
    "0.01",
    "--target-kl",
    "0.015",
  ]);

  const repeated = await runNode([
    "scripts/rl-prepare-ppo-bundle.mjs",
    "--model",
    modelPath,
    "--rollout",
    rolloutPath,
    "--output",
    outputPath,
  ]);
  assert.notEqual(repeated.code, 0);
  assert.match(repeated.stderr, /must not already exist/);
});

test("iteration dry-run is deterministic and does not create a run directory", async (t) => {
  const root = await temporaryDirectory(t, "iteration-dry-run");
  const modelPath = join(root, "parent.json");
  const outputRoot = join(root, "runs");
  await writeFile(
    modelPath,
    JSON.stringify({ format: "test-model", version: 1 }),
    "utf8",
  );
  const result = await runNode([
    "scripts/rl-prepare-next-ppo.mjs",
    "--model",
    modelPath,
    "--iteration",
    "51",
    "--run-label",
    "temperature-sweep",
    "--episodes",
    "300",
    "--episodes-by-player",
    "4=900,10=200",
    "--temperature",
    "1.25",
    "--seed",
    "700001",
    "--output",
    outputRoot,
    "--dry-run",
  ]);
  assert.equal(result.code, 0, result.stderr);
  const manifest = JSON.parse(result.stdout);
  assert.equal(manifest.runId, "ppo-iteration-51-temperature-sweep");
  assert.equal(manifest.configuration.seedSchedule[0].seed, 1_100_001);
  assert.equal(manifest.configuration.seedSchedule[0].episodes, 900);
  assert.equal(manifest.configuration.seedSchedule.at(-1).episodes, 200);
  await assert.rejects(readFile(join(outputRoot, manifest.runId, "run-manifest.json")));
});
