import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { simulateMatch } from "../training/simulator.ts";
import {
  createGreedyInferenceTrainingPolicy,
} from "../training/stochastic-policy.ts";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    candidate: { type: "string", short: "c" },
    reference: { type: "string", short: "r" },
    matches: { type: "string", default: "100" },
    acts: { type: "string", default: "5" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    seed: { type: "string", default: "830001" },
    output: { type: "string", short: "o" },
  },
});

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

if (!values.candidate) throw new TypeError("--candidate is required");
if (!values.reference) throw new TypeError("--reference is required");
const matches = positiveInteger(values.matches, "matches");
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
const playerCounts = values.players.split(",").map((value) =>
  positiveInteger(value.trim(), "players"),
);
if (
  playerCounts.length < 1 ||
  new Set(playerCounts).size !== playerCounts.length ||
  playerCounts.some((count) => count < 4 || count > 10)
) {
  throw new RangeError("players must be unique counts from 4 to 10");
}

const candidatePath = resolve(values.candidate);
const referencePath = resolve(values.reference);
const candidatePolicy = createGreedyInferenceTrainingPolicy(
  JSON.parse(await readFile(candidatePath, "utf8")),
);
const referencePolicy = createGreedyInferenceTrainingPolicy(
  JSON.parse(await readFile(referencePath, "utf8")),
);
const results = [];
const startedAt = performance.now();

for (const playerCount of playerCounts) {
  let candidateChips = 0;
  let candidateSeatActs = 0;
  let referenceChips = 0;
  let referenceSeatActs = 0;
  let decisions = 0;
  const differences = [];
  for (let matchIndex = 0; matchIndex < matches; matchIndex += 1) {
    const lowerHalf = Math.floor(playerCount / 2);
    const candidateCount =
      playerCount % 2 === 0 || matchIndex % 2 === 0
        ? lowerHalf
        : lowerHalf + 1;
    const candidateIds = new Set(
      Array.from(
        { length: candidateCount },
        (_, index) => `player-${index + 1}`,
      ),
    );
    const policyByPlayerId = Object.fromEntries(
      Array.from({ length: playerCount }, (_, index) => {
        const playerId = `player-${index + 1}`;
        return [
          playerId,
          candidateIds.has(playerId)
            ? candidatePolicy
            : referencePolicy,
        ];
      }),
    );
    const match = simulateMatch({
      playerCount,
      acts,
      seed: seed + playerCount * 1_000_000 + matchIndex,
      episodeId: `comparison-p${playerCount}-${matchIndex + 1}`,
      difficulties: ["normal"],
      policyByPlayerId,
    });
    decisions += match.steps.length;
    let candidateMatchChips = 0;
    let referenceMatchChips = 0;
    for (const act of match.acts) {
      for (const playerId of act.finishOrder) {
        const award = act.chipAwards[playerId];
        if (candidateIds.has(playerId)) {
          candidateChips += award;
          candidateSeatActs += 1;
          candidateMatchChips += award;
        } else {
          referenceChips += award;
          referenceSeatActs += 1;
          referenceMatchChips += award;
        }
      }
    }
    differences.push(
      candidateMatchChips / (candidateCount * acts) -
        referenceMatchChips / ((playerCount - candidateCount) * acts),
    );
  }
  const meanDifference =
    differences.reduce((total, value) => total + value, 0) /
    differences.length;
  const variance =
    differences.length > 1
      ? differences.reduce(
          (total, value) =>
            total + (value - meanDifference) ** 2,
          0,
        ) /
        (differences.length - 1)
      : 0;
  const margin95 = 1.96 * Math.sqrt(variance / differences.length);
  const result = {
    playerCount,
    matches,
    actsPerMatch: acts,
    decisions,
    candidateMeanChip: candidateChips / candidateSeatActs,
    referenceMeanChip: referenceChips / referenceSeatActs,
    meanChipDifference: meanDifference,
    meanChipDifference95: {
      low: meanDifference - margin95,
      high: meanDifference + margin95,
    },
  };
  results.push(result);
  console.log(
    `p${playerCount}: candidate ${result.candidateMeanChip.toFixed(4)} ` +
      `vs reference ${result.referenceMeanChip.toFixed(4)} | ` +
      `diff ${meanDifference.toFixed(4)} ` +
      `[${result.meanChipDifference95.low.toFixed(4)}, ` +
      `${result.meanChipDifference95.high.toFixed(4)}]`,
  );
}

const report = {
  format: "dalmuti-model-comparison",
  version: 1,
  candidatePath,
  referencePath,
  seed,
  matchesPerPlayerCount: matches,
  actsPerMatch: acts,
  playerCounts,
  elapsedSeconds: (performance.now() - startedAt) / 1000,
  results,
};
if (values.output) {
  const outputPath = resolve(values.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(
    outputPath,
    `${JSON.stringify(report, null, 2)}\n`,
    "utf8",
  );
  console.log(`Saved model comparison to ${outputPath}`);
}
