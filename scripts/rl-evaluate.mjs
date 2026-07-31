import { parseArgs } from "node:util";

import { BOT_DIFFICULTIES } from "../lib/bot-strategy.ts";
import { simulateMatch } from "../training/simulator.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();

const { values } = parseArgs({
  args: cliArgs,
  options: {
    matches: { type: "string", default: "100" },
    acts: { type: "string", default: "3" },
    seed: { type: "string", default: "1" },
    lineup: {
      type: "string",
      default: "easy,normal,hard,hard",
    },
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

const matches = positiveInteger(values.matches, "matches");
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
const lineup = values.lineup.split(",").map((value) => value.trim());
if (lineup.length < 4 || lineup.length > 10) {
  throw new RangeError("lineup must contain 4 to 10 difficulties");
}
for (const difficulty of lineup) {
  if (!BOT_DIFFICULTIES.includes(difficulty)) {
    throw new RangeError(`unknown difficulty in lineup: ${difficulty}`);
  }
}

const byPlayer = new Map(
  lineup.map((difficulty, index) => [
    `player-${index + 1}`,
    {
      playerId: `player-${index + 1}`,
      difficulty,
      acts: 0,
      chips: 0,
      places: 0,
      firsts: 0,
      lasts: 0,
    },
  ]),
);

let totalSteps = 0;
for (let matchIndex = 0; matchIndex < matches; matchIndex += 1) {
  const match = simulateMatch({
    playerCount: lineup.length,
    acts,
    seed: seed + matchIndex,
    episodeId: `evaluation-${matchIndex + 1}`,
    difficulties: lineup,
  });
  totalSteps += match.steps.length;
  for (const act of match.acts) {
    act.finishOrder.forEach((playerId, index) => {
      const row = byPlayer.get(playerId);
      row.acts += 1;
      row.chips += act.chipAwards[playerId];
      row.places += index + 1;
      if (index === 0) row.firsts += 1;
      if (index === lineup.length - 1) row.lasts += 1;
    });
  }
}

const rows = [...byPlayer.values()].map((row) => ({
  player: row.playerId,
  difficulty: row.difficulty,
  meanChip: Number((row.chips / row.acts).toFixed(4)),
  meanPlace: Number((row.places / row.acts).toFixed(4)),
  firstRate: Number((row.firsts / row.acts).toFixed(4)),
  lastRate: Number((row.lasts / row.acts).toFixed(4)),
  acts: row.acts,
}));
const result = {
  matches,
  actsPerMatch: acts,
  playerCount: lineup.length,
  seed,
  totalSteps,
  rows,
};

if (values.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(
    `DALMUTI baseline evaluation: ${matches} matches × ${acts} acts, seed ${seed}`,
  );
  console.table(rows);
  console.log(`Recorded policy decisions: ${totalSteps}`);
}
