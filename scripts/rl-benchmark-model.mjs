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
    model: { type: "string", short: "m" },
    matches: { type: "string", default: "100" },
    acts: { type: "string", default: "5" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    seed: { type: "string", default: "800001" },
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

if (!values.model) throw new TypeError("--model is required");
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

const modelPath = resolve(values.model);
const model = JSON.parse(await readFile(modelPath, "utf8"));
const candidatePolicy = createGreedyInferenceTrainingPolicy(model);
const results = [];
const startedAt = performance.now();

for (const playerCount of playerCounts) {
  const groups = {
    candidate: {
      chips: 0,
      places: 0,
      firsts: 0,
      lasts: 0,
      acts: 0,
    },
    normal: {
      chips: 0,
      places: 0,
      firsts: 0,
      lasts: 0,
      acts: 0,
    },
  };
  const differences = [];
  let decisions = 0;
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
      [...candidateIds].map((playerId) => [
        playerId,
        candidatePolicy,
      ]),
    );
    const match = simulateMatch({
      playerCount,
      acts,
      seed:
        seed +
        playerCount * 1_000_000 +
        matchIndex,
      episodeId: `benchmark-p${playerCount}-${matchIndex + 1}`,
      difficulties: ["normal"],
      policyByPlayerId,
    });
    decisions += match.steps.length;
    let candidateMatchChips = 0;
    let normalMatchChips = 0;
    for (const act of match.acts) {
      act.finishOrder.forEach((playerId, index) => {
        const group = candidateIds.has(playerId)
          ? groups.candidate
          : groups.normal;
        group.chips += act.chipAwards[playerId];
        group.places += index + 1;
        group.acts += 1;
        if (index === 0) group.firsts += 1;
        if (index === playerCount - 1) group.lasts += 1;
        if (candidateIds.has(playerId)) {
          candidateMatchChips += act.chipAwards[playerId];
        } else {
          normalMatchChips += act.chipAwards[playerId];
        }
      });
    }
    differences.push(
      candidateMatchChips / (candidateCount * acts) -
        normalMatchChips / ((playerCount - candidateCount) * acts),
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
  const margin95 =
    1.96 * Math.sqrt(variance / differences.length);
  function summarize(group) {
    return {
      meanChip: group.chips / group.acts,
      meanPlace: group.places / group.acts,
      firstRate: group.firsts / group.acts,
      lastRate: group.lasts / group.acts,
      seatActs: group.acts,
    };
  }
  const result = {
    playerCount,
    matches,
    actsPerMatch: acts,
    decisions,
    candidate: summarize(groups.candidate),
    normal: summarize(groups.normal),
    meanChipDifference: meanDifference,
    meanChipDifference95: {
      low: meanDifference - margin95,
      high: meanDifference + margin95,
    },
    statisticallyAboveNormal:
      meanDifference - margin95 > 0,
  };
  results.push(result);
  console.log(
    `p${playerCount}: candidate ${result.candidate.meanChip.toFixed(4)} ` +
      `vs normal ${result.normal.meanChip.toFixed(4)} | ` +
      `diff ${meanDifference.toFixed(4)} ` +
      `[${result.meanChipDifference95.low.toFixed(4)}, ` +
      `${result.meanChipDifference95.high.toFixed(4)}]`,
  );
}

const report = {
  format: "dalmuti-model-benchmark",
  version: 1,
  modelPath,
  seed,
  matchesPerPlayerCount: matches,
  actsPerMatch: acts,
  playerCounts,
  elapsedSeconds: (performance.now() - startedAt) / 1000,
  promotionRule:
    "95% lower confidence bound of mean chip difference > 0 for every player count",
  promotionPassed: results.every(
    (result) => result.statisticallyAboveNormal,
  ),
  results,
};
console.log(
  `Promotion gate: ${report.promotionPassed ? "PASS" : "FAIL"} ` +
    `(${report.elapsedSeconds.toFixed(2)}s)`,
);
if (values.output) {
  const outputPath = resolve(values.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(
    outputPath,
    `${JSON.stringify(report, null, 2)}\n`,
    "utf8",
  );
  console.log(`Saved benchmark report to ${outputPath}`);
}
