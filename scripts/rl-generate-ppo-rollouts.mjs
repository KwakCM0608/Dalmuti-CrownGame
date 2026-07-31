import { createHash } from "node:crypto";
import { once } from "node:events";
import { createWriteStream } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { createPpoRolloutManifest } from "../training/ppo-dataset.ts";
import { simulateMatch } from "../training/simulator.ts";
import {
  createStochasticTrainingPolicy,
  parseInferenceModel,
} from "../training/stochastic-policy.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();

const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    episodes: { type: "string", default: "100" },
    players: { type: "string", default: "4" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "300001" },
    output: {
      type: "string",
      default: "artifacts/rl/ppo-rollouts-p4.ndjson",
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
const episodes = positiveInteger(values.episodes, "episodes");
const playerCount = positiveInteger(values.players, "players");
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
if (playerCount < 4 || playerCount > 10) {
  throw new RangeError("players must be from 4 to 10");
}

const modelPath = resolve(values.model);
const modelBytes = await readFile(modelPath);
const modelValue = JSON.parse(modelBytes.toString("utf8"));
const model = parseInferenceModel(modelValue);
const modelSha256 = createHash("sha256")
  .update(modelBytes)
  .digest("hex");
const policyVersion = `sha256:${modelSha256}`;
const policy = createStochasticTrainingPolicy(model, policyVersion);

const outputPath = resolve(values.output);
await mkdir(dirname(outputPath), { recursive: true });
const output = createWriteStream(outputPath, { encoding: "utf8" });

async function writeRecord(record) {
  if (!output.write(`${JSON.stringify(record)}\n`)) {
    await once(output, "drain");
  }
}

await writeRecord(
  createPpoRolloutManifest({
    createdAt: new Date().toISOString(),
    episodes,
    playerCount,
    acts,
    seed,
    behaviorModelSha256: modelSha256,
    behaviorModelFormat: model.format,
  }),
);

let samples = 0;
let forcedSamples = 0;
for (let episode = 0; episode < episodes; episode += 1) {
  const episodeNumber = episode + 1;
  const match = simulateMatch({
    playerCount,
    acts,
    seed: seed + episode,
    episodeId: `ppo-p${playerCount}-episode-${episodeNumber}`,
    difficulties: ["normal"],
    policy,
  });
  for (const step of match.steps) {
    if (
      step.behaviorLogProbability === null ||
      step.behaviorValueEstimate === null ||
      step.behaviorPolicyVersion !== policyVersion
    ) {
      throw new Error("stochastic policy metadata is missing");
    }
    await writeRecord({
      type: "sample",
      trajectoryId:
        `${step.episodeId}:round-${step.round}:${step.actorId}`,
      episodeId: step.episodeId,
      round: step.round,
      step: step.step,
      actorId: step.actorId,
      actorSeat: step.actorSeat,
      actorRole: step.actorRole,
      observation: step.observation,
      legalActionIndices: step.legalActionIndices,
      actionIndex: step.actionIndex,
      oldLogProbability: step.behaviorLogProbability,
      oldValue: step.behaviorValueEstimate,
      reward: step.reward,
      terminal: step.actorTerminal,
      forced: step.forced,
      finishPlace: step.finishPlace,
      policyVersion: step.behaviorPolicyVersion,
    });
    samples += 1;
    if (step.forced) forcedSamples += 1;
  }
  if (
    episodeNumber === episodes ||
    episodeNumber % Math.max(1, Math.floor(episodes / 10)) === 0
  ) {
    console.log(
      `Generated ${episodeNumber}/${episodes} episodes ` +
        `(${samples.toLocaleString()} decisions)`,
    );
  }
}

await writeRecord({
  type: "summary",
  episodes,
  samples,
  forcedSamples,
  behaviorModelSha256: modelSha256,
});
output.end();
await once(output, "finish");

console.log(`Wrote ${samples} PPO samples to ${outputPath}`);
console.log(`Forced-action samples retained for value learning: ${forcedSamples}`);
