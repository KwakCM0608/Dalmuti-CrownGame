import { once } from "node:events";
import { createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { BOT_DIFFICULTIES } from "../lib/bot-strategy.ts";
import { createRolloutManifest } from "../training/dataset.ts";
import { simulateMatch } from "../training/simulator.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();

const { values } = parseArgs({
  args: cliArgs,
  options: {
    episodes: { type: "string", default: "100" },
    players: { type: "string", default: "4" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "1" },
    difficulty: { type: "string", default: "hard" },
    output: {
      type: "string",
      default: "artifacts/rl/rollouts-v1.ndjson",
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

const episodes = positiveInteger(values.episodes, "episodes");
const playerCount = positiveInteger(values.players, "players");
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
if (playerCount < 4 || playerCount > 10) {
  throw new RangeError("players must be from 4 to 10");
}
if (!BOT_DIFFICULTIES.includes(values.difficulty)) {
  throw new RangeError(`unknown difficulty: ${values.difficulty}`);
}

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
    difficulties: [values.difficulty],
  }),
);

let samples = 0;
let forcedSamples = 0;
for (let episode = 0; episode < episodes; episode += 1) {
  const match = simulateMatch({
    playerCount,
    acts,
    seed: seed + episode,
    episodeId: `episode-${episode + 1}`,
    difficulties: [values.difficulty],
  });
  for (const step of match.steps) {
    await writeRecord({ type: "sample", ...step });
    samples += 1;
    if (step.forced) forcedSamples += 1;
  }
}

await writeRecord({
  type: "summary",
  episodes,
  samples,
  forcedSamples,
});
output.end();
await once(output, "finish");

console.log(`Wrote ${samples} samples to ${outputPath}`);
console.log(`Forced-action samples: ${forcedSamples}`);
