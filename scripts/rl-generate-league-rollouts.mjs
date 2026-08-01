import { createHash } from "node:crypto";
import { once } from "node:events";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, unlink } from "node:fs/promises";
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
import {
  finiteNumber,
  positiveInteger,
} from "./lib/rl-orchestration.mjs";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    "opponent-model": { type: "string", multiple: true },
    "normal-opponent-fraction": { type: "string", default: "0.5" },
    episodes: { type: "string", default: "100" },
    "target-non-forced-decisions": { type: "string" },
    "max-episodes": { type: "string", default: "1000000" },
    temperature: { type: "string", default: "1" },
    players: { type: "string", default: "4" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "600001" },
    output: {
      type: "string",
      default: "artifacts/rl/ppo-league-p4.ndjson",
    },
  },
});

if (!values.model) throw new TypeError("--model is required");
const requestedEpisodes = positiveInteger(values.episodes, "episodes");
const targetNonForcedDecisions =
  values["target-non-forced-decisions"] === undefined
    ? null
    : positiveInteger(
        values["target-non-forced-decisions"],
        "target-non-forced-decisions",
      );
const maxEpisodes = positiveInteger(values["max-episodes"], "max-episodes");
const playerCount = positiveInteger(values.players, "players");
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
if (playerCount < 4 || playerCount > 10) {
  throw new RangeError("players must be from 4 to 10");
}

const modelPath = resolve(values.model);
const modelBytes = await readFile(modelPath);
const modelValue = JSON.parse(modelBytes.toString("utf8"));
const model = parseInferenceModel(modelValue);
const modelSha256 = createHash("sha256").update(modelBytes).digest("hex");
const policyVersion = `sha256:${modelSha256}`;
const learnerPolicy = createStochasticTrainingPolicy(
  model,
  policyVersion,
  { temperature },
);

const opponentModels = [];
const opponentHashes = new Set();
for (const pathValue of values["opponent-model"] ?? []) {
  const path = resolve(pathValue);
  const bytes = await readFile(path);
  const value = JSON.parse(bytes.toString("utf8"));
  const parsed = parseInferenceModel(value);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 === modelSha256) {
    throw new Error("an opponent model is identical to the behavior model");
  }
  if (opponentHashes.has(sha256)) {
    throw new Error(`duplicate opponent model SHA-256: ${sha256}`);
  }
  opponentHashes.add(sha256);
  opponentModels.push({
    path,
    sha256,
    policy: createGreedyInferenceTrainingPolicy(parsed),
  });
}
const effectiveNormalFraction =
  opponentModels.length === 0 ? 1 : normalOpponentFraction;

const outputPath = resolve(values.output);
const partialPath = `${outputPath}.partial`;
await mkdir(dirname(outputPath), { recursive: true });
const createdAt = new Date().toISOString();

function buildManifest(episodes) {
  const manifest = createPpoRolloutManifest({
    createdAt,
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
  });
  manifest.behaviorPolicy = {
    sampling: "softmax",
    temperature,
  };
  manifest.environment.opponentMix = {
    normalFraction: effectiveNormalFraction,
    trainedModelFraction: 1 - effectiveNormalFraction,
    trainedModelSelection: "uniform",
    trainedModels: opponentModels.map((entry) => ({
      sha256: entry.sha256,
      conditionalFraction: 1 / opponentModels.length,
    })),
  };
  manifest.environment.collection =
    targetNonForcedDecisions === null
      ? { mode: "fixed-episodes", requestedEpisodes }
      : {
          mode: "target-non-forced-decisions",
          targetNonForcedDecisions,
          maxEpisodes,
        };
  return manifest;
}

async function openExclusiveOutput(path) {
  const output = createWriteStream(path, { encoding: "utf8", flags: "wx" });
  await once(output, "open");
  return output;
}

async function writeRecord(output, record) {
  if (!output.write(`${JSON.stringify(record)}\n`)) {
    await once(output, "drain");
  }
}

async function finishOutput(output) {
  output.end();
  await once(output, "finish");
}

const samplesPath =
  targetNonForcedDecisions === null ? outputPath : partialPath;
const samplesOutput = await openExclusiveOutput(samplesPath);
if (targetNonForcedDecisions === null) {
  await writeRecord(samplesOutput, buildManifest(requestedEpisodes));
}

let learnerSamples = 0;
let environmentDecisions = 0;
let forcedSamples = 0;
let nonForcedSamples = 0;
let actualEpisodes = 0;
const opponentSeatAssignments = {
  normal: 0,
  byModelSha256: Object.fromEntries(
    opponentModels.map((entry) => [entry.sha256, 0]),
  ),
};
const episodeLimit =
  targetNonForcedDecisions === null ? requestedEpisodes : maxEpisodes;
const progressInterval =
  targetNonForcedDecisions === null
    ? Math.max(1, Math.floor(episodeLimit / 10))
    : 100;
for (let episode = 0; episode < episodeLimit; episode += 1) {
  const episodeNumber = episode + 1;
  const assignmentRandom = new SeededRandom(
    seed + episode * 97 + playerCount * 10_000,
  );
  const playerIds = assignmentRandom.shuffle(
    Array.from({ length: playerCount }, (_, index) => `player-${index + 1}`),
  );
  const lowerHalf = Math.floor(playerCount / 2);
  const learnerCount =
    playerCount % 2 === 0 || episode % 2 === 0
      ? lowerHalf
      : lowerHalf + 1;
  const learnerIds = new Set(playerIds.slice(0, learnerCount));
  const policyByPlayerId = {};
  for (const playerId of learnerIds) policyByPlayerId[playerId] = learnerPolicy;
  for (const playerId of playerIds.slice(learnerCount)) {
    if (assignmentRandom.next() < effectiveNormalFraction) {
      opponentSeatAssignments.normal += 1;
      continue;
    }
    const opponentIndex = Math.floor(
      assignmentRandom.next() * opponentModels.length,
    );
    const opponent = opponentModels[opponentIndex];
    policyByPlayerId[playerId] = opponent.policy;
    opponentSeatAssignments.byModelSha256[opponent.sha256] += 1;
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
    await writeRecord(samplesOutput, {
      type: "sample",
      trajectoryId: `${step.episodeId}:round-${step.round}:${step.actorId}`,
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
    else nonForcedSamples += 1;
  }
  actualEpisodes = episodeNumber;
  if (
    episodeNumber % progressInterval === 0 ||
    (targetNonForcedDecisions !== null &&
      nonForcedSamples >= targetNonForcedDecisions) ||
    episodeNumber === episodeLimit
  ) {
    console.log(
      `Generated ${episodeNumber}${
        targetNonForcedDecisions === null ? `/${episodeLimit}` : ""
      } league episodes (${learnerSamples.toLocaleString()} learner, ` +
        `${nonForcedSamples.toLocaleString()} non-forced decisions)`,
    );
  }
  if (
    targetNonForcedDecisions !== null &&
    nonForcedSamples >= targetNonForcedDecisions
  ) {
    break;
  }
}

if (
  targetNonForcedDecisions !== null &&
  nonForcedSamples < targetNonForcedDecisions
) {
  await finishOutput(samplesOutput);
  throw new Error(
    `max-episodes ${maxEpisodes} reached with only ` +
      `${nonForcedSamples} non-forced learner decisions`,
  );
}

const summary = {
  type: "summary",
  episodes: actualEpisodes,
  learnerSamples,
  environmentDecisions,
  forcedSamples,
  nonForcedSamples,
  behaviorModelSha256: modelSha256,
  samplingTemperature: temperature,
  targetNonForcedDecisions,
  opponentModelSha256: opponentModels.map((entry) => entry.sha256),
  opponentSeatAssignments,
};
if (targetNonForcedDecisions === null) {
  await writeRecord(samplesOutput, summary);
  await finishOutput(samplesOutput);
} else {
  await finishOutput(samplesOutput);
  const output = await openExclusiveOutput(outputPath);
  await writeRecord(output, buildManifest(actualEpisodes));
  for await (const chunk of createReadStream(partialPath)) {
    if (!output.write(chunk)) await once(output, "drain");
  }
  await writeRecord(output, summary);
  await finishOutput(output);
  await unlink(partialPath);
}

console.log(`Wrote ${learnerSamples} learner samples to ${outputPath}`);
console.log(`Non-forced policy samples: ${nonForcedSamples}`);
console.log(`Full environment decisions: ${environmentDecisions}`);
