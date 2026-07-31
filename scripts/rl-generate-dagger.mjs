import { once } from "node:events";
import { createWriteStream } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { createRolloutManifest } from "../training/dataset.ts";
import { createMlpTrainingPolicy } from "../training/model-policy.ts";
import {
  createBaselineTrainingPolicy,
  simulateMatch,
} from "../training/simulator.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    episodes: { type: "string", default: "50" },
    players: { type: "string", default: "4" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "1" },
    output: {
      type: "string",
      default: "artifacts/rl/dagger-p4-v2.ndjson",
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

const model = JSON.parse(
  await readFile(resolve(values.model), "utf8"),
);
const candidatePolicy = createMlpTrainingPolicy(model);
const supervisorPolicy = createBaselineTrainingPolicy("normal");
const outputPath = resolve(values.output);
await mkdir(dirname(outputPath), { recursive: true });
const output = createWriteStream(outputPath, { encoding: "utf8" });

async function writeRecord(record) {
  if (!output.write(`${JSON.stringify(record)}\n`)) {
    await once(output, "drain");
  }
}

await writeRecord(
  createRolloutManifest({
    createdAt: new Date().toISOString(),
    episodes,
    playerCount,
    acts,
    seed,
    difficulties: ["candidate-mlp", "normal-supervisor"],
  }),
);

let samples = 0;
let disagreements = 0;
let forcedSamples = 0;
for (let episode = 0; episode < episodes; episode += 1) {
  const match = simulateMatch({
    playerCount,
    acts,
    seed: seed + episode,
    episodeId: `dagger-${episode + 1}`,
    difficulties: ["normal"],
    policy: candidatePolicy,
    supervisionPolicy: supervisorPolicy,
  });
  for (const step of match.steps) {
    await writeRecord({ type: "sample", ...step });
    samples += 1;
    if (step.forced) forcedSamples += 1;
    if (step.actionIndex !== step.supervisedActionIndex) {
      disagreements += 1;
    }
  }
}

await writeRecord({
  type: "summary",
  episodes,
  samples,
  forcedSamples,
  disagreements,
});
output.end();
await once(output, "finish");

console.log(`Wrote ${samples} DAgger samples to ${outputPath}`);
console.log(
  `Supervisor disagreements: ${disagreements} ` +
    `(${((disagreements / samples) * 100).toFixed(2)}%)`,
);

