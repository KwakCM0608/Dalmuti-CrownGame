import { createHash } from "node:crypto";
import { once } from "node:events";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, unlink } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import {
  encodeV3LegalMaskHex,
  legacyActionIndexToV3,
  legacyLegalActionIndicesToV3,
} from "../training/v3-action-bridge.ts";
import { createV3PpoRolloutManifest } from "../training/v3-ppo-dataset.ts";
import { SeededRandom } from "../training/random.ts";
import { simulateMatch } from "../training/simulator.ts";
import {
  createGreedyV3TrainingPolicy,
  createStochasticV3TrainingPolicy,
} from "../training/v3-stochastic-policy.ts";
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
    "normal-opponent-fraction": { type: "string", default: "0.75" },
    episodes: { type: "string", default: "100" },
    "target-non-forced-decisions": { type: "string" },
    "max-episodes": { type: "string", default: "1000000" },
    temperature: { type: "string", default: "1" },
    players: { type: "string", default: "4" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "860000001" },
    output: {
      type: "string",
      default: "artifacts/rl/v3-ppo-league-p4.ndjson",
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

async function readModel(pathValue) {
  const path = resolve(pathValue);
  const bytes = await readFile(path);
  const value = JSON.parse(bytes.toString("utf8"));
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  return { path, value, sha256, filename: basename(path) };
}

const behavior = await readModel(values.model);
const policyVersion = `sha256:${behavior.sha256}`;
const learnerPolicy = createStochasticV3TrainingPolicy(
  behavior.value,
  policyVersion,
  temperature,
);
const opponents = [];
const modelHashes = new Set([behavior.sha256]);
for (const pathValue of values["opponent-model"] ?? []) {
  const opponent = await readModel(pathValue);
  if (modelHashes.has(opponent.sha256)) {
    throw new Error("opponent models must be unique and differ from behavior");
  }
  modelHashes.add(opponent.sha256);
  opponents.push({
    ...opponent,
    policy: createGreedyV3TrainingPolicy(opponent.value),
  });
}
const effectiveNormalFraction =
  opponents.length === 0 ? 1 : normalOpponentFraction;

const outputPath = resolve(values.output);
const partialPath = `${outputPath}.partial`;
await mkdir(dirname(outputPath), { recursive: true });
const createdAt = new Date().toISOString();

function manifest(episodes) {
  const value = createV3PpoRolloutManifest({
    createdAt,
    episodes,
    playerCount,
    acts,
    seed,
    behaviorModelSha256: behavior.sha256,
    temperature,
    mode: "league",
    opponentPolicies: [
      "normal",
      ...opponents.map((entry) => `sha256:${entry.sha256}`),
    ],
  });
  value.environment.learnerSeats =
    "approximately half; only behavior-model decisions are samples";
  value.environment.opponentMix = {
    normalFraction: effectiveNormalFraction,
    trainedModelFraction: 1 - effectiveNormalFraction,
    trainedModelSelection: "uniform",
    trainedModels: opponents.map((entry) => ({ sha256: entry.sha256 })),
  };
  value.environment.collection =
    targetNonForcedDecisions === null
      ? { mode: "fixed-episodes", requestedEpisodes }
      : {
          mode: "target-non-forced-decisions",
          targetNonForcedDecisions,
          maxEpisodes,
        };
  return value;
}

async function openExclusive(path) {
  const stream = createWriteStream(path, { encoding: "utf8", flags: "wx" });
  await once(stream, "open");
  return stream;
}
async function writeRecord(stream, record) {
  if (!stream.write(`${JSON.stringify(record)}\n`)) await once(stream, "drain");
}
async function finish(stream) {
  stream.end();
  await once(stream, "finish");
}

const samplesPath = targetNonForcedDecisions === null ? outputPath : partialPath;
const output = await openExclusive(samplesPath);
if (targetNonForcedDecisions === null) {
  await writeRecord(output, manifest(requestedEpisodes));
}

let learnerSamples = 0;
let forcedSamples = 0;
let nonForcedSamples = 0;
let environmentDecisions = 0;
let actualEpisodes = 0;
const opponentSeatAssignments = {
  normal: 0,
  byModelSha256: Object.fromEntries(opponents.map((entry) => [entry.sha256, 0])),
};
const episodeLimit = targetNonForcedDecisions === null
  ? requestedEpisodes
  : maxEpisodes;
for (let episode = 0; episode < episodeLimit; episode += 1) {
  const episodeNumber = episode + 1;
  const assignmentRandom = new SeededRandom(
    seed + episode * 97 + playerCount * 10_000,
  );
  const playerIds = assignmentRandom.shuffle(
    Array.from({ length: playerCount }, (_, index) => `player-${index + 1}`),
  );
  const learnerCount =
    playerCount % 2 === 0 || episode % 2 === 0
      ? Math.floor(playerCount / 2)
      : Math.floor(playerCount / 2) + 1;
  const learnerIds = new Set(playerIds.slice(0, learnerCount));
  const policyByPlayerId = {};
  for (const playerId of learnerIds) policyByPlayerId[playerId] = learnerPolicy;
  for (const playerId of playerIds.slice(learnerCount)) {
    if (assignmentRandom.next() < effectiveNormalFraction) {
      opponentSeatAssignments.normal += 1;
    } else {
      const opponent = opponents[
        Math.floor(assignmentRandom.next() * opponents.length)
      ];
      policyByPlayerId[playerId] = opponent.policy;
      opponentSeatAssignments.byModelSha256[opponent.sha256] += 1;
    }
  }
  const match = simulateMatch({
    playerCount,
    acts,
    seed: seed + episode,
    episodeId: `v3-league-p${playerCount}-episode-${episodeNumber}`,
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
      throw new Error("V3 behavior policy metadata is missing");
    }
    const legalActionIndices = legacyLegalActionIndicesToV3(
      step.legalActionIndices,
    );
    await writeRecord(output, {
      type: "sample",
      trajectoryId: `${step.episodeId}:round-${step.round}:${step.actorId}`,
      episodeId: step.episodeId,
      round: step.round,
      step: step.step,
      actorId: step.actorId,
      actorSeat: step.actorSeat,
      actorRole: step.actorRole,
      observationSchemaVersion: 2,
      actionCatalogueVersion: 1,
      observation: step.observation,
      legalActionIndices,
      legalMaskHex: encodeV3LegalMaskHex(legalActionIndices),
      actionIndex: legacyActionIndexToV3(step.actionIndex),
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
    episodeNumber === episodeLimit ||
    episodeNumber % 100 === 0 ||
    (targetNonForcedDecisions !== null &&
      nonForcedSamples >= targetNonForcedDecisions)
  ) {
    console.log(
      `Generated ${episodeNumber} V3 episodes; ` +
        `${nonForcedSamples.toLocaleString()} non-forced learner decisions`,
    );
  }
  if (
    targetNonForcedDecisions !== null &&
    nonForcedSamples >= targetNonForcedDecisions
  ) break;
}
if (
  targetNonForcedDecisions !== null &&
  nonForcedSamples < targetNonForcedDecisions
) {
  await finish(output);
  throw new Error("max-episodes reached before the V3 sample target");
}
const summary = {
  type: "summary",
  episodes: actualEpisodes,
  learnerSamples,
  environmentDecisions,
  forcedSamples,
  nonForcedSamples,
  behaviorModelSha256: behavior.sha256,
  samplingTemperature: temperature,
  targetNonForcedDecisions,
  opponentModelSha256: opponents.map((entry) => entry.sha256),
  opponentSeatAssignments,
};
if (targetNonForcedDecisions === null) {
  await writeRecord(output, summary);
  await finish(output);
} else {
  await finish(output);
  const finalOutput = await openExclusive(outputPath);
  await writeRecord(finalOutput, manifest(actualEpisodes));
  for await (const chunk of createReadStream(partialPath)) {
    if (!finalOutput.write(chunk)) await once(finalOutput, "drain");
  }
  await writeRecord(finalOutput, summary);
  await finish(finalOutput);
  await unlink(partialPath);
}
console.log(`Wrote ${learnerSamples} V3 learner samples to ${outputPath}`);
