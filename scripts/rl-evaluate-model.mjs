import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { createMlpTrainingPolicy } from "../training/model-policy.ts";
import { simulateMatch } from "../training/simulator.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    matches: { type: "string", default: "30" },
    acts: { type: "string", default: "3" },
    players: { type: "string", default: "4" },
    seed: { type: "string", default: "70001" },
    json: { type: "boolean", default: false },
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
const matches = positiveInteger(values.matches, "matches");
const acts = positiveInteger(values.acts, "acts");
const playerCount = positiveInteger(values.players, "players");
const seed = positiveInteger(values.seed, "seed");
if (playerCount < 4 || playerCount > 10 || playerCount % 2 !== 0) {
  throw new RangeError("players must be one of 4, 6, 8, or 10");
}

const modelPath = resolve(values.model);
const model = JSON.parse(await readFile(modelPath, "utf8"));
const candidatePolicy = createMlpTrainingPolicy(model);
const candidateCount = playerCount / 2;
const policyByPlayerId = Object.fromEntries(
  Array.from({ length: candidateCount }, (_, index) => [
    `player-${index + 1}`,
    candidatePolicy,
  ]),
);
const groups = {
  candidate: {
    group: "candidate",
    acts: 0,
    chips: 0,
    places: 0,
    firsts: 0,
    lasts: 0,
  },
  normal: {
    group: "normal",
    acts: 0,
    chips: 0,
    places: 0,
    firsts: 0,
    lasts: 0,
  },
};

let totalSteps = 0;
const startedAt = performance.now();
for (let matchIndex = 0; matchIndex < matches; matchIndex += 1) {
  const match = simulateMatch({
    playerCount,
    acts,
    seed: seed + matchIndex,
    episodeId: `model-evaluation-${matchIndex + 1}`,
    difficulties: ["normal"],
    policyByPlayerId,
  });
  totalSteps += match.steps.length;
  for (const act of match.acts) {
    act.finishOrder.forEach((playerId, index) => {
      const group =
        Number(playerId.slice("player-".length)) <= candidateCount
          ? groups.candidate
          : groups.normal;
      group.acts += 1;
      group.chips += act.chipAwards[playerId];
      group.places += index + 1;
      if (index === 0) group.firsts += 1;
      if (index === playerCount - 1) group.lasts += 1;
    });
  }
}

const rows = Object.values(groups).map((group) => ({
  group: group.group,
  meanChip: Number((group.chips / group.acts).toFixed(4)),
  meanPlace: Number((group.places / group.acts).toFixed(4)),
  firstRate: Number((group.firsts / group.acts).toFixed(4)),
  lastRate: Number((group.lasts / group.acts).toFixed(4)),
  acts: group.acts,
}));
const result = {
  modelPath,
  matches,
  actsPerMatch: acts,
  playerCount,
  candidatePlayers: candidateCount,
  normalPlayers: candidateCount,
  seed,
  totalSteps,
  elapsedSeconds: Number(
    ((performance.now() - startedAt) / 1000).toFixed(3),
  ),
  rows,
};

if (values.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(
    `Model evaluation: ${matches} matches × ${acts} acts, ` +
      `${playerCount} players`,
  );
  console.table(rows);
  console.log(
    `${totalSteps} decisions in ${result.elapsedSeconds.toFixed(3)}s`,
  );
}

