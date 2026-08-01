import { mkdir, open } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { parseMatchCounts } from "./rl-evaluation-statistics.mjs";
import { auditRevolutionMerchantSourceData } from "./lib/revolution-merchant-data-audit.mjs";
import { runExperimentalMerchantRevolutionPairedBenchmark } from "./lib/revolution-merchant-paired-benchmark.mjs";

const args = process.argv.slice(2);
if (args[0] === "--") args.shift();
const seedWasExplicit = args.some(
  (argument) => argument === "--seed" || argument.startsWith("--seed="),
);
const matchesWasExplicit = args.some(
  (argument) => argument === "--matches" || argument.startsWith("--matches="),
);
const { values } = parseArgs({
  args,
  options: {
    "source-data": { type: "string" },
    output: { type: "string", short: "o" },
    seed: { type: "string" },
    matches: { type: "string", default: "100" },
    "match-counts": { type: "string" },
    acts: { type: "string", default: "5" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    "include-match-data": { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(
    "Usage: node scripts/rl-benchmark-revolution-merchant-candidate.mjs " +
      "--source-data <v2-revolution.ndjson> --seed <positive-int> " +
      "--output <new-report.json> [--matches 100 | --match-counts 4:100,...] " +
      "[--acts 5] [--players 4,5,6,7,8,9,10] [--include-match-data]",
  );
  process.exit(0);
}

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
  return parsed;
}

if (!values["source-data"]) throw new TypeError("--source-data is required");
if (!values.output) throw new TypeError("--output is required");
if (!seedWasExplicit) throw new TypeError("--seed is required and must be explicit");
if (matchesWasExplicit && values["match-counts"]) {
  throw new TypeError("--matches and --match-counts cannot be combined");
}

const playerCounts = values.players.split(",").map((value) =>
  positiveInteger(value.trim(), "players"),
);
if (
  playerCounts.length < 1 ||
  new Set(playerCounts).size !== playerCounts.length ||
  playerCounts.some((playerCount) => playerCount < 4 || playerCount > 10)
) {
  throw new RangeError("players must be unique counts from 4 to 10");
}
const matches = positiveInteger(values.matches, "matches");
const matchCountsByPlayerCount = values["match-counts"]
  ? parseMatchCounts(values["match-counts"], playerCounts)
  : Object.fromEntries(
      playerCounts.map((playerCount) => [playerCount, matches]),
    );
const sourceEvidence = await auditRevolutionMerchantSourceData(
  values["source-data"],
);
const report = runExperimentalMerchantRevolutionPairedBenchmark({
  sourceEvidence,
  playerCounts,
  matchCountsByPlayerCount,
  acts: positiveInteger(values.acts, "acts"),
  seed: positiveInteger(values.seed, "seed"),
  includeMatchData: values["include-match-data"],
});

const outputPath = resolve(values.output);
await mkdir(dirname(outputPath), { recursive: true });
const handle = await open(outputPath, "wx");
try {
  await handle.writeFile(`${JSON.stringify(report, null, 2)}\n`, "utf8");
} finally {
  await handle.close();
}

for (const result of report.results) {
  const chip = result.pairedMarginal.chipDifference;
  console.log(
    `p${result.playerCount}: changed=${result.interventionRouting.changedFromNormal}, ` +
      `chip=${chip.mean.toFixed(4)} ` +
      `[${chip.confidence95.low.toFixed(4)}, ` +
      `${chip.confidence95.high.toFixed(4)}], ` +
      `exact=${result.trajectoryParity.exactMatches}/${result.matches}`,
  );
}
const pooledChip = report.pooled.pairedMarginal.chipDifference;
console.log(
  `pooled: chip=${pooledChip.mean.toFixed(4)} ` +
    `[${pooledChip.confidence95.low.toFixed(4)}, ` +
    `${pooledChip.confidence95.high.toFixed(4)}]`,
);
console.log(`Saved paired benchmark to ${outputPath}`);
