import { spawn } from "node:child_process";
import { readFile, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import {
  assertMissingDirectory,
  createNewDirectory,
  createPortableZip,
  finiteNumber,
  normalizeRunLabel,
  parsePlayerCountOverrides,
  positiveInteger,
  sha256File,
} from "./lib/rl-orchestration.mjs";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    "opponent-model": { type: "string", multiple: true },
    "normal-opponent-fraction": { type: "string", default: "0.5" },
    iteration: { type: "string", default: "1" },
    "run-label": { type: "string" },
    episodes: { type: "string", default: "200" },
    "episodes-by-player": { type: "string", multiple: true },
    "target-non-forced-decisions": { type: "string" },
    "max-episodes": { type: "string", default: "1000000" },
    temperature: { type: "string", default: "1" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "500001" },
    output: { type: "string", default: "artifacts/rl" },
    "skip-archive": { type: "boolean", default: false },
    "dry-run": { type: "boolean", default: false },
  },
});

if (!values.model) throw new TypeError("--model is required");
const iteration = positiveInteger(values.iteration, "iteration");
const runLabel = normalizeRunLabel(values["run-label"]);
const defaultEpisodes = positiveInteger(values.episodes, "episodes");
const episodeOverrides = parsePlayerCountOverrides(
  values["episodes-by-player"],
  "episodes-by-player",
);
const targetNonForcedDecisions =
  values["target-non-forced-decisions"] === undefined
    ? null
    : positiveInteger(
        values["target-non-forced-decisions"],
        "target-non-forced-decisions",
      );
if (targetNonForcedDecisions !== null && episodeOverrides.size > 0) {
  throw new Error(
    "--episodes-by-player and --target-non-forced-decisions are mutually exclusive",
  );
}
const maxEpisodes = positiveInteger(values["max-episodes"], "max-episodes");
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
const temperature = finiteNumber(values.temperature, "temperature", {
  minimum: 0.05,
  maximum: 10,
});
const normalOpponentFraction = finiteNumber(
  values["normal-opponent-fraction"],
  "normal-opponent-fraction",
  { minimum: 0, maximum: 1 },
);

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const modelPath = resolve(projectRoot, values.model);
const opponentModelPaths = (values["opponent-model"] ?? []).map((path) =>
  resolve(projectRoot, path),
);
const outputRoot = resolve(projectRoot, values.output);
const runId = `ppo-iteration-${iteration}${runLabel ? `-${runLabel}` : ""}`;
const iterationRoot = join(outputRoot, runId);
const rolloutRoot = join(iterationRoot, "rollouts");
const bundleRoot = join(iterationRoot, "gpu-bundle");
const runManifestPath = join(iterationRoot, "run-manifest.json");
const archivePath = join(iterationRoot, `dalmuti-${runId}-gpu.zip`);
const checksumPath = `${archivePath}.sha256`;

await assertMissingDirectory(iterationRoot, "iteration run directory");

async function describeModel(path, role) {
  const fileStat = await stat(path);
  if (!fileStat.isFile()) throw new TypeError(`${role} must name a file`);
  const value = JSON.parse(await readFile(path, "utf8"));
  if (typeof value?.format !== "string") {
    throw new TypeError(`${role} model format is missing`);
  }
  return {
    filename: basename(path),
    format: value.format,
    bytes: fileStat.size,
    sha256: await sha256File(path),
  };
}

const parentModel = await describeModel(modelPath, "behavior");
const opponentModels = [];
const modelHashes = new Set([parentModel.sha256]);
for (const path of opponentModelPaths) {
  const description = await describeModel(path, "opponent");
  if (modelHashes.has(description.sha256)) {
    throw new Error(
      description.sha256 === parentModel.sha256
        ? "an opponent model is identical to the behavior model"
        : `duplicate opponent model SHA-256: ${description.sha256}`,
    );
  }
  modelHashes.add(description.sha256);
  opponentModels.push(description);
}

const seedSchedule = [];
for (let playerCount = 4; playerCount <= 10; playerCount += 1) {
  const rolloutSeed = seed + playerCount * 100_000;
  if (!Number.isSafeInteger(rolloutSeed)) {
    throw new RangeError("derived rollout seed exceeds safe integer range");
  }
  seedSchedule.push({
    playerCount,
    seed: rolloutSeed,
    episodes:
      targetNonForcedDecisions === null
        ? (episodeOverrides.get(playerCount) ?? defaultEpisodes)
        : null,
    targetNonForcedDecisions,
  });
}

const runManifest = {
  format: "dalmuti-ppo-local-run",
  version: 1,
  runId,
  status: values["dry-run"] ? "dry-run" : "collecting-rollouts",
  createdAt: new Date().toISOString(),
  parentModel,
  opponentModels,
  configuration: {
    iteration,
    runLabel: runLabel || null,
    playerCounts: [4, 5, 6, 7, 8, 9, 10],
    acts,
    temperature,
    normalOpponentFraction:
      opponentModels.length === 0 ? 1 : normalOpponentFraction,
    defaultEpisodes:
      targetNonForcedDecisions === null ? defaultEpisodes : null,
    episodeOverrides: Object.fromEntries(episodeOverrides),
    targetNonForcedDecisions,
    maxEpisodes: targetNonForcedDecisions === null ? null : maxEpisodes,
    baseSeed: seed,
    seedDerivation: "baseSeed + playerCount * 100000",
    seedSchedule,
  },
  output: {
    directory: iterationRoot,
    bundleDirectory: bundleRoot,
    archive: values["skip-archive"] ? null : archivePath,
  },
};

if (values["dry-run"]) {
  console.log(JSON.stringify(runManifest, null, 2));
  process.exit(0);
}

await createNewDirectory(iterationRoot, "iteration run directory");
await createNewDirectory(rolloutRoot, "rollout directory");
await writeFile(
  runManifestPath,
  `${JSON.stringify(runManifest, null, 2)}\n`,
  { encoding: "utf8", flag: "wx" },
);

function runNode(script, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [join(projectRoot, script), ...args], {
      cwd: projectRoot,
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) resolvePromise();
      else {
        reject(new Error(`${script} failed with code ${code} signal ${signal}`));
      }
    });
  });
}

try {
  const rolloutPaths = [];
  for (const schedule of seedSchedule) {
    const rolloutPath = join(
      rolloutRoot,
      `ppo-i${iteration}${runLabel ? `-${runLabel}` : ""}-p${schedule.playerCount}.ndjson`,
    );
    rolloutPaths.push(rolloutPath);
    const rolloutArguments = [
      "--model",
      modelPath,
      "--players",
      String(schedule.playerCount),
      "--acts",
      String(acts),
      "--seed",
      String(schedule.seed),
      "--temperature",
      String(temperature),
      "--output",
      rolloutPath,
      "--normal-opponent-fraction",
      String(normalOpponentFraction),
    ];
    if (targetNonForcedDecisions === null) {
      rolloutArguments.push("--episodes", String(schedule.episodes));
    } else {
      rolloutArguments.push(
        "--target-non-forced-decisions",
        String(targetNonForcedDecisions),
        "--max-episodes",
        String(maxEpisodes),
      );
    }
    for (const opponentModelPath of opponentModelPaths) {
      rolloutArguments.push("--opponent-model", opponentModelPath);
    }
    await runNode("scripts/rl-generate-league-rollouts.mjs", rolloutArguments);
  }

  runManifest.status = "preparing-bundle";
  await writeFile(
    runManifestPath,
    `${JSON.stringify(runManifest, null, 2)}\n`,
    "utf8",
  );
  const bundleArguments = ["--model", modelPath, "--output", bundleRoot];
  for (const rolloutPath of rolloutPaths) {
    bundleArguments.push("--rollout", rolloutPath);
  }
  await runNode("scripts/rl-prepare-ppo-bundle.mjs", bundleArguments);

  const bundleManifest = JSON.parse(
    await readFile(join(bundleRoot, "bundle-manifest.json"), "utf8"),
  );
  runManifest.dataCounts = bundleManifest.dataCounts;
  runManifest.rollouts = bundleManifest.rollouts;
  runManifest.bundle = {
    files: bundleManifest.files.length,
    totalBytes: bundleManifest.totalBytes,
    manifestSha256: await sha256File(join(bundleRoot, "bundle-manifest.json")),
  };
  if (!values["skip-archive"]) {
    console.log(`Creating portable ZIP: ${archivePath}`);
    const archive = await createPortableZip({
      sourceDirectory: bundleRoot,
      archivePath,
    });
    const sha256 = await sha256File(archivePath);
    await writeFile(
      checksumPath,
      `${sha256}  ${basename(archivePath)}\n`,
      { encoding: "utf8", flag: "wx" },
    );
    runManifest.archive = {
      filename: basename(archivePath),
      bytes: archive.bytes,
      entries: archive.entries.length,
      entryPathSeparator: "/",
      sha256,
      checksumFilename: basename(checksumPath),
    };
  }
  runManifest.status = "ready-for-gpu";
  runManifest.completedAt = new Date().toISOString();
  await writeFile(
    runManifestPath,
    `${JSON.stringify(runManifest, null, 2)}\n`,
    "utf8",
  );
  console.log(`Prepared PPO run ${runId}: ${iterationRoot}`);
} catch (error) {
  runManifest.status = "failed";
  runManifest.failedAt = new Date().toISOString();
  runManifest.failure = {
    name: error?.name ?? "Error",
    message: error?.message ?? String(error),
  };
  await writeFile(
    runManifestPath,
    `${JSON.stringify(runManifest, null, 2)}\n`,
    "utf8",
  );
  throw error;
}
