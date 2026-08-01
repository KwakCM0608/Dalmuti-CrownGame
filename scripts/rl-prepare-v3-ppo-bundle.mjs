import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import {
  assertMissingDirectory,
  createNewDirectory,
  portableRelativePath,
  readRolloutEnvelope,
  sha256File,
} from "./lib/rl-orchestration.mjs";
import {
  V3_ACTION_CATALOGUE_VERSION,
  V3_ACTION_COUNT,
} from "../training/v3-action-catalogue.ts";
import {
  OBSERVATION_FEATURE_COUNT,
  OBSERVATION_SCHEMA_VERSION,
} from "../training/observation.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    rollout: { type: "string", multiple: true },
    output: { type: "string", required: true },
    "dry-run": { type: "boolean", default: false },
  },
});
if (!values.model) throw new TypeError("--model is required");
if (!values.rollout?.length) {
  throw new TypeError("at least one --rollout is required");
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const packageRoot = join(projectRoot, "gpu-training");
const bundleRoot = resolve(projectRoot, values.output);
const dataRoot = join(bundleRoot, "data");
const modelPath = resolve(projectRoot, values.model);
const modelBytes = await readFile(modelPath);
const model = JSON.parse(modelBytes.toString("utf8"));
if (
  model.format !== "dalmuti-action-conditioned-actor-critic" ||
  model.version !== 1 ||
  model.observationSchemaVersion !== OBSERVATION_SCHEMA_VERSION ||
  model.observationFeatures !== OBSERVATION_FEATURE_COUNT ||
  model.actionCatalogueVersion !== V3_ACTION_CATALOGUE_VERSION ||
  model.actionCount !== V3_ACTION_COUNT
) {
  throw new TypeError("behavior model does not satisfy the V3 runtime contract");
}
const modelSha256 = await sha256File(modelPath);
const rolloutPaths = values.rollout.map((value) => resolve(projectRoot, value));
if (
  new Set(rolloutPaths.map((value) => value.toLowerCase())).size !==
  rolloutPaths.length
) {
  throw new Error("the same V3 rollout was supplied more than once");
}
await assertMissingDirectory(bundleRoot, "V3 bundle output directory");

const rollouts = [];
const usedNames = new Set();
for (const path of rolloutPaths) {
  const filename = basename(path);
  if (usedNames.has(filename.toLowerCase())) {
    throw new Error(`duplicate V3 rollout filename: ${filename}`);
  }
  usedNames.add(filename.toLowerCase());
  const envelope = await readRolloutEnvelope(path, "dalmuti-v3-ppo-ndjson");
  if (envelope.manifest.behaviorModel?.sha256 !== modelSha256) {
    throw new Error(`${path} does not match the supplied V3 behavior model`);
  }
  if (
    envelope.manifest.observation?.version !== OBSERVATION_SCHEMA_VERSION ||
    envelope.manifest.observation?.featureCount !== OBSERVATION_FEATURE_COUNT ||
    envelope.manifest.actionSpace?.catalogueVersion !==
      V3_ACTION_CATALOGUE_VERSION ||
    envelope.manifest.actionSpace?.size !== V3_ACTION_COUNT
  ) {
    throw new Error(`${path} has a mismatched V3 observation/action contract`);
  }
  rollouts.push({
    path,
    filename,
    bytes: (await stat(path)).size,
    sha256: await sha256File(path),
    playerCount: envelope.manifest.environment?.playerCount,
    acts: envelope.manifest.environment?.actsPerEpisode,
    seed: envelope.manifest.environment?.initialSeed,
    temperature: envelope.manifest.behaviorPolicy?.temperature,
    ...envelope.counts,
  });
}
const temperatures = new Set(rollouts.map((rollout) => rollout.temperature));
const requiredPlayerCounts = Array.from({ length: 7 }, (_, index) => index + 4);
const suppliedPlayerCounts = rollouts
  .map((rollout) => rollout.playerCount)
  .sort((left, right) => left - right);
if (
  suppliedPlayerCounts.length !== requiredPlayerCounts.length ||
  suppliedPlayerCounts.some(
    (playerCount, index) => playerCount !== requiredPlayerCounts[index],
  )
) {
  throw new Error(
    "strict V3 PPO bundles require exactly one rollout for each player count 4..10",
  );
}
if (temperatures.size !== 1) {
  throw new Error("all V3 rollouts in a bundle must use one temperature");
}
const rolloutTemperature = rollouts[0].temperature;
if (!Number.isFinite(rolloutTemperature) || rolloutTemperature <= 0) {
  throw new Error("V3 rollout temperature is invalid");
}
function manifestRollout(rollout) {
  const result = { ...rollout };
  delete result.path;
  return result;
}
const preview = {
  format: "dalmuti-v3-ppo-bundle-plan",
  version: 1,
  output: bundleRoot,
  parentModel: {
    filename: basename(modelPath),
    format: model.format,
    bytes: modelBytes.length,
    sha256: modelSha256,
  },
  rollouts: rollouts.map(manifestRollout),
};
if (values["dry-run"]) {
  console.log(JSON.stringify(preview, null, 2));
  process.exit(0);
}

const packageFiles = [
  "requirements.txt",
  "preflight.py",
  "verify_bundle.py",
  "v3-ppo-schema.json",
  "actor_critic.py",
  "ppo_dataset.py",
  "train_ppo.py",
  "v3_action_conditioned.py",
  "test_v3_action_conditioned.py",
  "v3_ppo_dataset.py",
  "verify_v3_ppo_data.py",
  "train_v3_ppo.py",
  "package_v3_ppo_results.py",
  "verify_v3_ppo_results.py",
  "v3_ppo_result_contract.py",
  "run_gpu_v3_ppo.py",
  "test_v3_ppo_pipeline.py",
  "test_v3_ppo_result_contract.py",
  "PROMPT_FOR_GPU_V3_PPO.md",
];
const promptPath = join(packageRoot, "PROMPT_FOR_GPU_V3_PPO.md");
const prompt = await readFile(promptPath, "utf8");
const pythonBytecodeGuard = "export PYTHONDONTWRITEBYTECODE=1";
const guardPosition = prompt.indexOf(pythonBytecodeGuard);
const firstPythonProcess = prompt.indexOf("python verify_bundle.py");
if (
  guardPosition < 0 ||
  firstPythonProcess < 0 ||
  guardPosition > firstPythonProcess
) {
  throw new Error(
    "V3 PPO prompt must disable Python bytecode before its first Python process",
  );
}
await createNewDirectory(bundleRoot, "V3 bundle output directory");
await mkdir(dataRoot);
const files = [];
async function copyVerified(source, destination) {
  await copyFile(source, destination);
  files.push({
    path: portableRelativePath(bundleRoot, destination),
    bytes: (await stat(destination)).size,
    sha256: await sha256File(destination),
  });
}
for (const filename of packageFiles) {
  await copyVerified(join(packageRoot, filename), join(bundleRoot, filename));
}
await copyVerified(modelPath, join(bundleRoot, "behavior-model.json"));
for (const rollout of rollouts) {
  await copyVerified(rollout.path, join(dataRoot, rollout.filename));
}
const runConfig = {
  format: "dalmuti-v3-ppo-gpu-run-config",
  version: 2,
  parentModelSha256: modelSha256,
  rolloutTemperature,
  algorithm: {
    epochs: 12,
    batchSize: 4096,
    learningRate: 0.0001,
    weightDecay: 0.00001,
    gamma: 1,
    gaeLambda: 1,
    skipForcedPolicyTime: true,
    rolloutTemperature,
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
  },
  allowedTerminalRankAuxiliaryCoefficients: [0, 0.05],
  determinism: {
    required: true,
    pythonDontWriteBytecode: true,
    torchDeterministicAlgorithms: true,
    warnOnly: false,
    cublasWorkspaceConfig: ":4096:8",
    cudnnDeterministic: true,
    cudnnBenchmark: false,
    cudaMatmulAllowTf32: false,
    cudnnAllowTf32: false,
  },
  pathPolicy: {
    bundleRoot: ".",
    behaviorModel: "behavior-model.json",
    dataRoot: "data",
    outputRoot: "models",
    resultsRoot: "returned",
    requireFreshRunDirectories: true,
    requireMatchingRunIds: true,
    requireDisjointPaths: true,
    protectBundleInputs: true,
    rejectSymbolicLinks: true,
  },
  requiredCommandArguments: [
    "--output",
    "models/<fresh-v3-run>",
    "--results-dir",
    "returned/<fresh-v3-run>",
    "--epochs",
    "12",
    "--batch-size",
    "4096",
    "--learning-rate",
    "0.0001",
    "--weight-decay",
    "0.00001",
    "--gamma",
    "1",
    "--gae-lambda",
    "1",
    "--skip-forced-policy-time",
    "--terminal-rank-auxiliary-coefficient",
    "<0-or-0.05>",
    "--rollout-temperature",
    String(rolloutTemperature),
    "--clip-coefficient",
    "0.2",
    "--value-coefficient",
    "0.5",
    "--entropy-coefficient",
    "0.01",
    "--max-gradient-norm",
    "0.5",
    "--target-kl",
    "0.015",
    "--binding-tolerance",
    "0.00002",
    "--behavior-binding-batch-size",
    "8192",
    "--loader-workers",
    "7",
    "--seed",
    "202608061",
    "--device",
    "cuda",
  ],
};
const configPath = join(bundleRoot, "gpu-run-config.json");
await writeFile(configPath, `${JSON.stringify(runConfig, null, 2)}\n`, {
  encoding: "utf8",
  flag: "wx",
});
files.push({
  path: portableRelativePath(bundleRoot, configPath),
  bytes: (await stat(configPath)).size,
  sha256: await sha256File(configPath),
});
const totals = rollouts.reduce(
  (sum, rollout) => ({
    episodes: sum.episodes + rollout.episodes,
    learnerSamples: sum.learnerSamples + rollout.learnerSamples,
    forcedSamples: sum.forcedSamples + rollout.forcedSamples,
    nonForcedSamples: sum.nonForcedSamples + rollout.nonForcedSamples,
    environmentDecisions:
      sum.environmentDecisions + rollout.environmentDecisions,
  }),
  {
    episodes: 0,
    learnerSamples: 0,
    forcedSamples: 0,
    nonForcedSamples: 0,
    environmentDecisions: 0,
  },
);
const bundleManifest = {
  format: "dalmuti-v3-ppo-gpu-bundle",
  version: 1,
  createdAt: new Date().toISOString(),
  parentModel: preview.parentModel,
  behaviorModelSha256: modelSha256,
  observationSchemaVersion: OBSERVATION_SCHEMA_VERSION,
  observationFeatures: OBSERVATION_FEATURE_COUNT,
  actionCatalogueVersion: V3_ACTION_CATALOGUE_VERSION,
  actionCount: V3_ACTION_COUNT,
  rollouts: preview.rollouts,
  dataCounts: totals,
  files,
  totalBytes: files.reduce((sum, file) => sum + file.bytes, 0),
};
await writeFile(
  join(bundleRoot, "bundle-manifest.json"),
  `${JSON.stringify(bundleManifest, null, 2)}\n`,
  { encoding: "utf8", flag: "wx" },
);
console.log(`Strict V3 PPO GPU bundle ready: ${bundleRoot}`);
