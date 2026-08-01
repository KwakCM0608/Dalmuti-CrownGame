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

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    rollout: { type: "string", multiple: true },
    output: {
      type: "string",
      default: "artifacts/rl/ppo-gpu-bundle-v1",
    },
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
const bundleDataRoot = join(bundleRoot, "data");
const modelPath = resolve(projectRoot, values.model);
const modelStat = await stat(modelPath);
if (!modelStat.isFile()) throw new TypeError("--model must name a file");
const modelBytes = await readFile(modelPath);
const modelValue = JSON.parse(modelBytes.toString("utf8"));
if (typeof modelValue?.format !== "string") {
  throw new TypeError("behavior model format is missing");
}
const modelSha256 = await sha256File(modelPath);
const rolloutPaths = values.rollout.map((path) =>
  resolve(projectRoot, path),
);
if (new Set(rolloutPaths.map((path) => path.toLowerCase())).size !== rolloutPaths.length) {
  throw new Error("the same rollout path was supplied more than once");
}
const packageFiles = [
  "requirements.txt",
  "preflight.py",
  "verify_bundle.py",
  "ppo-schema.json",
  "actor_critic.py",
  "ppo_dataset.py",
  "verify_ppo_data.py",
  "test_ppo.py",
  "test_ppo_core_upgrades.py",
  "v3_action_conditioned.py",
  "test_v3_action_conditioned.py",
  "non_card_action_conditioned.py",
  "test_non_card_action_conditioned.py",
  "train_ppo.py",
  "package_ppo_results.py",
  "run_gpu_ppo.py",
  "PROMPT_FOR_GPU_PPO.md",
];

await assertMissingDirectory(bundleRoot, "bundle output directory");

const rollouts = [];
const usedNames = new Set();
for (const rolloutPath of rolloutPaths) {
  const filename = basename(rolloutPath);
  if (usedNames.has(filename.toLowerCase())) {
    throw new Error(`duplicate rollout filename: ${filename}`);
  }
  usedNames.add(filename.toLowerCase());
  const envelope = await readRolloutEnvelope(rolloutPath);
  if (envelope.manifest.behaviorModel?.sha256 !== modelSha256) {
    throw new Error(
      `${rolloutPath} does not match the supplied behavior model`,
    );
  }
  rollouts.push({
    path: rolloutPath,
    filename,
    sha256: await sha256File(rolloutPath),
    bytes: (await stat(rolloutPath)).size,
    playerCount: envelope.manifest.environment?.playerCount,
    acts: envelope.manifest.environment?.actsPerEpisode,
    seed: envelope.manifest.environment?.initialSeed,
    temperature: envelope.manifest.behaviorPolicy?.temperature ?? 1,
    opponentMix: envelope.manifest.environment?.opponentMix ?? null,
    opponentSeatAssignments:
      envelope.summary.opponentSeatAssignments ?? null,
    ...envelope.counts,
  });
}
const rolloutTemperatures = new Set(
  rollouts.map((rollout) => rollout.temperature),
);
if (rolloutTemperatures.size !== 1) {
  throw new Error("all rollouts in one bundle must use the same temperature");
}
const rolloutTemperature = rollouts[0].temperature;
function manifestRollout(rollout) {
  const result = { ...rollout };
  delete result.path;
  return result;
}

const preview = {
  format: "dalmuti-ppo-bundle-plan",
  version: 1,
  output: bundleRoot,
  parentModel: {
    filename: basename(modelPath),
    format: modelValue.format,
    bytes: modelStat.size,
    sha256: modelSha256,
  },
  rollouts: rollouts.map(manifestRollout),
};
if (values["dry-run"]) {
  console.log(JSON.stringify(preview, null, 2));
  process.exit(0);
}

await createNewDirectory(bundleRoot, "bundle output directory");
await mkdir(bundleDataRoot);
const manifestFiles = [];
async function copyVerified(source, destination) {
  await copyFile(source, destination);
  const fileStat = await stat(destination);
  const sha256 = await sha256File(destination);
  manifestFiles.push({
    path: portableRelativePath(bundleRoot, destination),
    bytes: fileStat.size,
    sha256,
  });
}

for (const filename of packageFiles) {
  await copyVerified(join(packageRoot, filename), join(bundleRoot, filename));
}
await copyVerified(modelPath, join(bundleRoot, "behavior-model.json"));
for (const rollout of rollouts) {
  await copyVerified(
    rollout.path,
    join(bundleDataRoot, rollout.filename),
  );
}
const gpuRunConfigPath = join(bundleRoot, "gpu-run-config.json");
const correctedPpoContract = {
  epochs: 12,
  batchSize: 4096,
  learningRate: 1.0e-4,
  gamma: 1,
  gaeLambda: 1,
  skipForcedPolicyTime: true,
  clipCoefficient: 0.2,
  valueCoefficient: 0.5,
  entropyCoefficient: 0.01,
  targetKl: 0.015,
  terminalRankAuxiliaryVariants: [0, 0.05],
};
await writeFile(
  gpuRunConfigPath,
  `${JSON.stringify(
    {
      format: "dalmuti-ppo-gpu-run-config",
      version: 1,
      parentModelSha256: modelSha256,
      rolloutTemperature,
      correctedPpoContract,
      requiredRunGpuPpoArguments: [
        "--epochs",
        String(correctedPpoContract.epochs),
        "--batch-size",
        String(correctedPpoContract.batchSize),
        "--learning-rate",
        String(correctedPpoContract.learningRate),
        "--gamma",
        String(correctedPpoContract.gamma),
        "--gae-lambda",
        String(correctedPpoContract.gaeLambda),
        "--skip-forced-policy-time",
        "--rollout-temperature",
        String(rolloutTemperature),
        "--clip-coefficient",
        String(correctedPpoContract.clipCoefficient),
        "--value-coefficient",
        String(correctedPpoContract.valueCoefficient),
        "--entropy-coefficient",
        String(correctedPpoContract.entropyCoefficient),
        "--target-kl",
        String(correctedPpoContract.targetKl),
      ],
    },
    null,
    2,
  )}\n`,
  "utf8",
);
manifestFiles.push({
  path: portableRelativePath(bundleRoot, gpuRunConfigPath),
  bytes: (await stat(gpuRunConfigPath)).size,
  sha256: await sha256File(gpuRunConfigPath),
});

const totals = rollouts.reduce(
  (result, rollout) => ({
    episodes: result.episodes + rollout.episodes,
    learnerSamples: result.learnerSamples + rollout.learnerSamples,
    forcedSamples: result.forcedSamples + rollout.forcedSamples,
    nonForcedSamples:
      result.nonForcedSamples + rollout.nonForcedSamples,
    environmentDecisions:
      result.environmentDecisions + rollout.environmentDecisions,
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
  format: "dalmuti-ppo-gpu-bundle",
  version: 1,
  createdAt: new Date().toISOString(),
  parentModel: {
    filename: basename(modelPath),
    format: modelValue.format,
    bytes: modelStat.size,
    sha256: modelSha256,
  },
  // Kept for compatibility with the GPU verifier and older result tooling.
  behaviorModelSha256: modelSha256,
  rollouts: rollouts.map(manifestRollout),
  dataCounts: totals,
  files: manifestFiles,
  totalBytes: manifestFiles.reduce((total, file) => total + file.bytes, 0),
};
await writeFile(
  join(bundleRoot, "bundle-manifest.json"),
  `${JSON.stringify(bundleManifest, null, 2)}\n`,
  "utf8",
);
console.log(`PPO GPU bundle ready: ${bundleRoot}`);
console.log(
  `${manifestFiles.length} files, ` +
    `${(bundleManifest.totalBytes / 1024 / 1024).toFixed(2)} MiB, ` +
    `${totals.nonForcedSamples.toLocaleString()} non-forced samples`,
);
