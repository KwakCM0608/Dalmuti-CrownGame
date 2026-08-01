import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  decodeV3LegalMaskHex,
  encodeV3LegalMaskHex,
  legacyActionIndexToV3,
  legacyLegalActionIndicesToV3,
  v3ActionIndexToLegacy,
} from "../training/v3-action-bridge.ts";
import {
  V3_ACTION_COUNT,
  V3_ACTION_FEATURE_COUNT,
  V3_ACTION_FEATURE_LAYOUT,
} from "../training/v3-action-catalogue.ts";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function layer(inFeatures, outFeatures) {
  return {
    inFeatures,
    outFeatures,
    weight: Array.from({ length: inFeatures * outFeatures }, () => 0),
    bias: Array.from({ length: outFeatures }, () => 0),
  };
}

function uniformV3Model() {
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

function runCommand(command, args, environment = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: projectRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, ...environment },
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

function runNode(args) {
  return runCommand(process.execPath, args);
}

test("V3 bridge is bijective for every compact action and mask encoding", () => {
  const all = Array.from({ length: V3_ACTION_COUNT }, (_, index) => index);
  for (const v3Index of all) {
    assert.equal(legacyActionIndexToV3(v3ActionIndexToLegacy(v3Index)), v3Index);
  }
  assert.deepEqual(decodeV3LegalMaskHex(encodeV3LegalMaskHex(all)), all);
  assert.deepEqual(
    legacyLegalActionIndicesToV3(all.map(v3ActionIndexToLegacy)),
    all,
  );
});

test("V3 bridge canonicalizes joker mixes whose catalogue orders differ", () => {
  // Legacy orders rank-2 plays by total count: 1 natural, 2 naturals,
  // then 1 natural + 1 joker. V3 orders the same semantics by natural count.
  const legacy = [44, 47, 48];
  const v3 = legacyLegalActionIndicesToV3(legacy);
  assert.deepEqual(v3, [5, 6, 8]);
  assert.deepEqual(
    v3.map(v3ActionIndexToLegacy).sort((left, right) => left - right),
    legacy,
  );
  assert.throws(
    () => legacyLegalActionIndicesToV3([44, 44]),
    /non-empty set of unique indices/,
  );
});

test("tiny V3 rollout binds observation, catalogue, legal mask, and behavior logprob", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "dalmuti-v3-rollout-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const modelPath = join(root, "uniform-v3.json");
  const rolloutPath = join(root, "v3-p4.ndjson");
  const modelText = JSON.stringify(uniformV3Model());
  await writeFile(modelPath, modelText, "utf8");
  const modelSha256 = createHash("sha256").update(modelText).digest("hex");
  const result = await runNode([
    "scripts/rl-generate-v3-league-rollouts.mjs",
    "--model",
    modelPath,
    "--players",
    "4",
    "--acts",
    "1",
    "--episodes",
    "1",
    "--seed",
    "860001001",
    "--temperature",
    "1.25",
    "--output",
    rolloutPath,
  ]);
  assert.equal(result.code, 0, result.stderr);
  const records = (await readFile(rolloutPath, "utf8"))
    .trim()
    .split(/\r?\n/)
    .map(JSON.parse);
  const manifest = records[0];
  const samples = records.filter((record) => record.type === "sample");
  const summary = records.at(-1);
  assert.equal(manifest.format, "dalmuti-v3-ppo-ndjson");
  assert.equal(manifest.behaviorModel.sha256, modelSha256);
  assert.equal(manifest.observation.version, 2);
  assert.equal(manifest.observation.featureCount, 172);
  assert.equal(manifest.actionSpace.catalogueVersion, 1);
  assert.equal(manifest.actionSpace.size, 236);
  assert.equal(manifest.actionSpace.catalogue.length, 236);
  assert.equal(manifest.actionSpace.encodedActionFeatures.length, 236);
  assert.ok(samples.length > 0);
  for (const sample of samples) {
    assert.equal(sample.observationSchemaVersion, 2);
    assert.equal(sample.actionCatalogueVersion, 1);
    assert.equal(sample.policyVersion, `sha256:${modelSha256}`);
    assert.equal(sample.observation.length, 172);
    assert.deepEqual(
      decodeV3LegalMaskHex(sample.legalMaskHex),
      sample.legalActionIndices,
    );
    assert.ok(
      Math.abs(
        sample.oldLogProbability + Math.log(sample.legalActionIndices.length),
      ) < 1e-12,
    );
    assert.equal(sample.oldValue, 0);
    assert.equal(sample.forced, sample.legalActionIndices.length === 1);
  }
  assert.equal(summary.learnerSamples, samples.length);
  assert.equal(
    summary.forcedSamples + summary.nonForcedSamples,
    summary.learnerSamples,
  );

  const rolloutPaths = [rolloutPath];
  let totalLearnerSamples = summary.learnerSamples;
  for (let playerCount = 5; playerCount <= 10; playerCount += 1) {
    const additionalPath = join(root, `v3-p${playerCount}.ndjson`);
    const additional = await runNode([
      "scripts/rl-generate-v3-league-rollouts.mjs",
      "--model",
      modelPath,
      "--players",
      String(playerCount),
      "--acts",
      "1",
      "--episodes",
      "1",
      "--seed",
      String(860001000 + playerCount),
      "--temperature",
      "1.25",
      "--output",
      additionalPath,
    ]);
    assert.equal(additional.code, 0, additional.stderr);
    const additionalRecords = (await readFile(additionalPath, "utf8"))
      .trim()
      .split(/\r?\n/)
      .map(JSON.parse);
    totalLearnerSamples += additionalRecords.at(-1).learnerSamples;
    rolloutPaths.push(additionalPath);
  }

  const incompleteBundle = await runNode([
    "scripts/rl-prepare-v3-ppo-bundle.mjs",
    "--model",
    modelPath,
    "--rollout",
    rolloutPath,
    "--output",
    join(root, "incomplete-v3-bundle"),
  ]);
  assert.notEqual(incompleteBundle.code, 0);
  assert.match(incompleteBundle.stderr, /each player count 4\.\.10/);

  const bundlePath = join(root, "fresh-v3-bundle");
  const bundle = await runNode([
    "scripts/rl-prepare-v3-ppo-bundle.mjs",
    "--model",
    modelPath,
    ...rolloutPaths.flatMap((path) => ["--rollout", path]),
    "--output",
    bundlePath,
  ]);
  assert.equal(bundle.code, 0, bundle.stderr);
  const bundleManifest = JSON.parse(
    await readFile(join(bundlePath, "bundle-manifest.json"), "utf8"),
  );
  assert.equal(bundleManifest.format, "dalmuti-v3-ppo-gpu-bundle");
  assert.equal(bundleManifest.parentModel.sha256, modelSha256);
  assert.equal(bundleManifest.observationSchemaVersion, 2);
  assert.equal(bundleManifest.actionCatalogueVersion, 1);
  assert.equal(bundleManifest.actionCount, 236);
  assert.equal(bundleManifest.rollouts.length, 7);
  assert.equal(bundleManifest.dataCounts.learnerSamples, totalLearnerSamples);
  const runConfig = JSON.parse(
    await readFile(join(bundlePath, "gpu-run-config.json"), "utf8"),
  );
  assert.equal(runConfig.version, 2);
  assert.equal(runConfig.rolloutTemperature, 1.25);
  assert.deepEqual(runConfig.allowedTerminalRankAuxiliaryCoefficients, [0, 0.05]);
  assert.deepEqual(runConfig.algorithm, {
    epochs: 12,
    batchSize: 4096,
    learningRate: 0.0001,
    weightDecay: 0.00001,
    gamma: 1,
    gaeLambda: 1,
    skipForcedPolicyTime: true,
    rolloutTemperature: 1.25,
    clipCoefficient: 0.2,
    valueCoefficient: 0.5,
    entropyCoefficient: 0.01,
    maxGradientNorm: 0.5,
    targetKl: 0.015,
    bindingTolerance: 0.00002,
    behaviorBindingBatchSize: 8192,
    loaderWorkers: 7,
    device: "cuda",
    seed: 202608061,
  });
  assert.equal(runConfig.determinism.cublasWorkspaceConfig, ":4096:8");
  assert.equal(runConfig.determinism.pythonDontWriteBytecode, true);
  assert.equal(runConfig.determinism.torchDeterministicAlgorithms, true);
  assert.equal(runConfig.pathPolicy.outputRoot, "models");
  assert.equal(runConfig.pathPolicy.resultsRoot, "returned");
  assert.equal(runConfig.pathPolicy.protectBundleInputs, true);
  assert.ok(
    bundleManifest.files.some(
      (entry) => entry.path === "v3_ppo_result_contract.py",
    ),
  );
  assert.ok(
    bundleManifest.files.some(
      (entry) => entry.path === "test_v3_ppo_result_contract.py",
    ),
  );
  assert.ok(
    bundleManifest.files.some(
      (entry) => entry.path === "verify_v3_ppo_data.py",
    ),
  );
  assert.ok(
    runConfig.requiredCommandArguments.includes("--rollout-temperature"),
  );
  assert.ok(
    runConfig.requiredCommandArguments.includes(
      "--behavior-binding-batch-size",
    ),
  );
  assert.equal(
    runConfig.requiredCommandArguments[
      runConfig.requiredCommandArguments.indexOf(
        "--behavior-binding-batch-size",
      ) + 1
    ],
    "8192",
  );
  assert.equal(
    runConfig.requiredCommandArguments[
      runConfig.requiredCommandArguments.indexOf("--loader-workers") + 1
    ],
    "7",
  );
  const gpuPrompt = await readFile(
    join(bundlePath, "PROMPT_FOR_GPU_V3_PPO.md"),
    "utf8",
  );
  assert.ok(
    gpuPrompt.indexOf("export PYTHONDONTWRITEBYTECODE=1") <
      gpuPrompt.indexOf("python verify_bundle.py"),
  );
  assert.doesNotMatch(gpuPrompt, /python verify_v3_ppo_data\.py/);
  assert.match(gpuPrompt, /single dataset load/);
  assert.match(gpuPrompt, /batches of 8192/);
  const bundledRunner = await readFile(
    join(bundlePath, "run_gpu_v3_ppo.py"),
    "utf8",
  );
  assert.ok(!bundledRunner.includes('str(root / "verify_v3_ppo_data.py")'));
  assert.ok(bundledRunner.includes('"--data-verification-output"'));
  assert.ok(bundledRunner.includes('"--behavior-binding-batch-size"'));
  const bundledTrainer = await readFile(
    join(bundlePath, "train_v3_ppo.py"),
    "utf8",
  );
  assert.ok(bundledTrainer.includes('"--data-verification-output"'));
  assert.ok(bundledTrainer.includes('"--behavior-binding-batch-size"'));
  if (process.env.DALMUTI_TEST_PYTHON) {
    const pythonEnvironment = process.env.DALMUTI_TEST_PYTHONPATH
      ? {
          PYTHONPATH: process.env.DALMUTI_TEST_PYTHONPATH,
          PYTHONDONTWRITEBYTECODE: "1",
        }
      : { PYTHONDONTWRITEBYTECODE: "1" };
    const verifiedBundle = await runCommand(
      process.env.DALMUTI_TEST_PYTHON,
      [join(bundlePath, "verify_bundle.py")],
      pythonEnvironment,
    );
    assert.equal(verifiedBundle.code, 0, verifiedBundle.stderr);
    const verificationPath = join(root, "cross-language-verification.json");
    const verified = await runCommand(
      process.env.DALMUTI_TEST_PYTHON,
      [
        "gpu-training/verify_v3_ppo_data.py",
        "--data",
        rolloutPath,
        "--behavior-model",
        modelPath,
        "--rollout-temperature",
        "1.25",
        "--output",
        verificationPath,
      ],
      pythonEnvironment,
    );
    assert.equal(verified.code, 0, verified.stderr);
    const verification = JSON.parse(
      await readFile(verificationPath, "utf8"),
    );
    assert.equal(
      verification.behaviorBinding.logProbability,
      "recomputed-and-verified",
    );
    assert.deepEqual(verification.legalMaskShape.slice(1), [236]);
    const sourceContract = await runCommand(
      process.env.DALMUTI_TEST_PYTHON,
      [
        "-c",
        [
          "import pathlib,sys",
          `root=pathlib.Path(${JSON.stringify(bundlePath)})`,
          "sys.path.insert(0,str(root))",
          "from v3_ppo_result_contract import load_source_contract",
          "load_source_contract(root/'bundle-manifest.json',root/'gpu-run-config.json',verify_source_files=True)",
        ].join(";"),
      ],
      process.env.DALMUTI_TEST_PYTHONPATH
        ? {
            PYTHONPATH: process.env.DALMUTI_TEST_PYTHONPATH,
            PYTHONDONTWRITEBYTECODE: "1",
          }
        : { PYTHONDONTWRITEBYTECODE: "1" },
    );
    assert.equal(sourceContract.code, 0, sourceContract.stderr);
    await assert.rejects(stat(join(bundlePath, "__pycache__")), {
      code: "ENOENT",
    });
  }

  const repeated = await runNode([
    "scripts/rl-generate-v3-league-rollouts.mjs",
    "--model",
    modelPath,
    "--players",
    "4",
    "--acts",
    "1",
    "--episodes",
    "1",
    "--seed",
    "860001001",
    "--output",
    rolloutPath,
  ]);
  assert.notEqual(repeated.code, 0);
});
