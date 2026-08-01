import { parseArgs } from "node:util";

import { parseMatchCounts } from "./rl-evaluation-statistics.mjs";
import {
  DEFAULT_SCREENING_SEED_BASE,
  DEFAULT_SCREENING_SEED_STRIDE,
  finiteNumber,
  parseIntegerList,
  parsePlayerCounts,
  positiveInteger,
  runBenchmarkProcess,
  screeningConcurrency,
  screenCheckpointDirectory,
} from "./lib/rl-checkpoint-screening.mjs";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    directory: { type: "string", short: "d" },
    output: { type: "string", short: "o" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    matches: { type: "string", default: "300" },
    "match-counts": { type: "string" },
    acts: { type: "string", default: "5" },
    concurrency: { type: "string", default: "1" },
    "benchmark-shards": { type: "string", default: "1" },
    seeds: { type: "string" },
    "seed-base": {
      type: "string",
      default: String(DEFAULT_SCREENING_SEED_BASE),
    },
    "seed-stride": {
      type: "string",
      default: String(DEFAULT_SCREENING_SEED_STRIDE),
    },
    "reserved-final-seeds": { type: "string" },
    "min-point-diff": { type: "string", default: "0.25" },
    "min-lower-bound": { type: "string", default: "0.15" },
    "min-pairwise-rate": { type: "string", default: "0.55" },
  },
});

if (!values.directory) throw new TypeError("--directory is required");
if (!values.output) throw new TypeError("--output is required");
const playerCounts = parsePlayerCounts(values.players);
const matches = positiveInteger(values.matches, "matches");
const acts = positiveInteger(values.acts, "acts");
const matchCountsByPlayerCount = values["match-counts"]
  ? parseMatchCounts(values["match-counts"], playerCounts)
  : Object.fromEntries(
      playerCounts.map((playerCount) => [playerCount, matches]),
    );
const thresholds = {
  minPointDifference: finiteNumber(
    values["min-point-diff"],
    "min-point-diff",
  ),
  minLowerBound: finiteNumber(
    values["min-lower-bound"],
    "min-lower-bound",
  ),
  minPairwiseRate: finiteNumber(
    values["min-pairwise-rate"],
    "min-pairwise-rate",
  ),
};
if (thresholds.minPairwiseRate < 0 || thresholds.minPairwiseRate > 1) {
  throw new RangeError("min-pairwise-rate must be between 0 and 1");
}

const result = await screenCheckpointDirectory({
  directory: values.directory,
  output: values.output,
  playerCounts,
  matches,
  matchCountsByPlayerCount,
  acts,
  concurrency: screeningConcurrency(values.concurrency),
  benchmarkShards: positiveInteger(
    values["benchmark-shards"],
    "benchmark-shards",
  ),
  explicitSeeds: parseIntegerList(values.seeds, "screening seed"),
  seedBase: positiveInteger(values["seed-base"], "seed-base"),
  seedStride: positiveInteger(values["seed-stride"], "seed-stride"),
  reservedFinalSeeds: parseIntegerList(
    values["reserved-final-seeds"],
    "reserved final seed",
  ),
  thresholds,
  processRunner: (options) =>
    runBenchmarkProcess({
      ...options,
      onStdout: (chunk) => process.stdout.write(chunk),
      onStderr: (chunk) => process.stderr.write(chunk),
    }),
  onProgress: (event) => {
    if (event.type === "candidate-start") {
      console.log(
        `Screening ${event.index + 1}/${event.total}: ` +
          `${event.candidate.labels.join(" + ")} ` +
          `(seed ${event.seed})`,
      );
    }
  },
});

console.log(`Checkpoint screening report: ${result.reportPath}`);
console.log(
  `Selected: ${result.report.winner.id} ` +
    `(${result.report.winner.sha256.slice(0, 12)})`,
);
