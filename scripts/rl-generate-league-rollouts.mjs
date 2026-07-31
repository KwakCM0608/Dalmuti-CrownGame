import { createHash } from "node:crypto";
import { once } from "node:events";
import { createWriteStream } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { createPpoRolloutManifest } from "../training/ppo-dataset.ts";
import { SeededRandom } from "../training/random.ts";
import { simulateMatch } from "../training/simulator.ts";
import {
  createGreedyInferenceTrainingPolicy,
  createStochasticTrainingPolicy,
  parseInferenceModel,
} from "../training/stochastic-policy.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    "opponent-model": { type: "string", multiple: true },
    "normal-opponent-fraction": { type: "string", default: "0.5" },
    episodes: { type: "string", default: "100" },
    players: { type: "string", default: "4" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "600001" },
    output: {
      type: "string",
      default: "artifacts/rl/ppo-league-p4.ndjson",
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
const normalOpponentFraction = Number(
  values["normal-opponent-fraction"],
);
if (playerCount < 4 || playerCount > 10) {
  throw new RangeError("players must be from 4 to 10");
}
if (
  !Number.isFinite(normalOpponentFraction) ||
  normalOpponentFraction < 0 ||
  normalOpponentFraction > 1
) {
  throw new RangeError("normal-opponent-fraction must be from 0 to 1");
}

const modelPath = resolve(values.model);
const modelBytes = await readFile(modelPath);
const modelValue = JSON.parse(modelBytes.toString("utf8"));
const model = parseInferenceModel(modelValue);
const modelSha256 = createHash("sha256")
  .update(modelBytes)
  .digest("hex");
const policyVersion = `sha256:${modelSha256}`;
const learnerPolicy = createStochasticTrainingPolicy(
  model,
  policyVersion,
);

const opponentModels = [];
for (const pathValue of values["opponent-model"] ?? []) {
  const path = resolve(pathValue);
  const bytes = await readFile(path);
  const value = JSON.parse(bytes.toString("utf8"));
  const parsed = parseInferenceModel(value);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 === modelSha256) continue;
  opponentModels.push({
    path,
    sha256,
    policy: createGreedyInferenceTrainingPolicy(parsed),
  });
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
  createPpoRolloutManifest({
    createdAt: new Date().toISOString(),
    episodes,
    playerCount,
    acts,
    seed,
    behaviorModelSha256: modelSha256,
    behaviorModelFormat: model.format,
    mode: "league",
    opponentPolicies: [
      "normal",
      ...opponentModels.map((entry) => `sha256:${entry.sha256}`),
    ],
  }),
);

let learnerSamples = 0;
let environmentDecisions = 0;
let forcedSamples = 0;
for (let episode = 0; episode < episodes; episode += 1) {
  const episodeNumber = episode + 1;
  const assignmentRandom = new SeededRandom(
    seed + episode * 97 + playerCount * 10_000,
  );
  const playerIds = assignmentRandom.shuffle(
    Array.from(
      { length: playerCount },
      (_, index) => `player-${index + 1}`,
    ),
  );
  const lowerHalf = Math.floor(playerCount / 2);
  const learnerCount =
    playerCount % 2 === 0 || episode % 2 === 0
      ? lowerHalf
      : lowerHalf + 1;
  const learnerIds = new Set(playerIds.slice(0, learnerCount));
  const policyByPlayerId = {};
  for (const playerId of learnerIds) {
    policyByPlayerId[playerId] = learnerPolicy;
  }
  for (const playerId of playerIds.slice(learnerCount)) {
    if (
      opponentModels.length === 0 ||
      assignmentRandom.next() < normalOpponentFraction
    ) {
      continue;
    }
    const opponentIndex = Math.floor(
      assignmentRandom.next() * opponentModels.length,
    );
    policyByPlayerId[playerId] =
      opponentModels[opponentIndex].policy;
  }

  const match = simulateMatch({
    playerCount,
    acts,
    seed: seed + episode,
    episodeId: `league-p${playerCount}-episode-${episodeNumber}`,
    difficulties: ["normal"],
    policyByPlayerId,
  });
  environmentDecisions += match.steps.length;
  for (const step of match.steps) {
    if (step.behaviorPolicyVersion !== policyVersion) continue;
    if (
      step.behaviorLogProbability === null ||
      step.behaviorValueEstimate === null
    ) {
      throw new Error("learner policy metadata is missing");
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
    learnerSamples += 1;
    if (step.forced) forcedSamples += 1;
  }
  if (
    episodeNumber === episodes ||
    episodeNumber % Math.max(1, Math.floor(episodes / 10)) === 0
  ) {
    console.log(
      `Generated ${episodeNumber}/${episodes} league episodes ` +
        `(${learnerSamples.toLocaleString()} learner decisions)`,
    );
  }
}

await writeRecord({
  type: "summary",
  episodes,
  learnerSamples,
  environmentDecisions,
  forcedSamples,
  behaviorModelSha256: modelSha256,
  opponentModelSha256: opponentModels.map((entry) => entry.sha256),
});
output.end();
await once(output, "finish");
console.log(`Wrote ${learnerSamples} learner samples to ${outputPath}`);
console.log(`Full environment decisions: ${environmentDecisions}`);
