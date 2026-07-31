import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    "opponent-model": { type: "string", multiple: true },
    "normal-opponent-fraction": { type: "string", default: "0.5" },
    iteration: { type: "string", default: "1" },
    episodes: { type: "string", default: "200" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "500001" },
    output: {
      type: "string",
      default: "artifacts/rl",
    },
  },
});

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

if (!values.model) throw new TypeError("--model is required");
const iteration = positiveInteger(values.iteration, "iteration");
const episodes = positiveInteger(values.episodes, "episodes");
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
const normalOpponentFraction = Number(
  values["normal-opponent-fraction"],
);
if (
  !Number.isFinite(normalOpponentFraction) ||
  normalOpponentFraction < 0 ||
  normalOpponentFraction > 1
) {
  throw new RangeError("normal-opponent-fraction must be from 0 to 1");
}
const projectRoot = resolve(new URL("..", import.meta.url).pathname.slice(1));
const modelPath = resolve(projectRoot, values.model);
const opponentModelPaths = (values["opponent-model"] ?? []).map((path) =>
  resolve(projectRoot, path),
);
const outputRoot = resolve(projectRoot, values.output);
const iterationRoot = join(outputRoot, `ppo-iteration-${iteration}`);
const rolloutRoot = join(iterationRoot, "rollouts");
const bundleRoot = join(iterationRoot, "gpu-bundle");
await mkdir(rolloutRoot, { recursive: true });

function runNode(script, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(
      process.execPath,
      [join(projectRoot, script), ...args],
      {
        cwd: projectRoot,
        stdio: "inherit",
      },
    );
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolvePromise();
      } else {
        reject(
          new Error(
            `${script} failed with code ${code} signal ${signal}`,
          ),
        );
      }
    });
  });
}

const rolloutPaths = [];
for (let playerCount = 4; playerCount <= 10; playerCount += 1) {
  const rolloutPath = join(
    rolloutRoot,
    `ppo-i${iteration}-p${playerCount}.ndjson`,
  );
  rolloutPaths.push(rolloutPath);
  const rolloutArguments = [
    "--model",
    modelPath,
    "--episodes",
    String(episodes),
    "--players",
    String(playerCount),
    "--acts",
    String(acts),
    "--seed",
    String(seed + playerCount * 100_000),
    "--output",
    rolloutPath,
    "--normal-opponent-fraction",
    String(normalOpponentFraction),
  ];
  for (const opponentModelPath of opponentModelPaths) {
    rolloutArguments.push("--opponent-model", opponentModelPath);
  }
  await runNode(
    "scripts/rl-generate-league-rollouts.mjs",
    rolloutArguments,
  );
}

const bundleArguments = [
  "--model",
  modelPath,
  "--output",
  bundleRoot,
];
for (const rolloutPath of rolloutPaths) {
  bundleArguments.push("--rollout", rolloutPath);
}
await runNode(
  "scripts/rl-prepare-ppo-bundle.mjs",
  bundleArguments,
);
console.log(`Prepared PPO iteration ${iteration}: ${iterationRoot}`);
