import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { parseMatchCounts } from "./rl-evaluation-statistics.mjs";
import { mergeBenchmarkShardReports } from "./lib/rl-benchmark-shard-merge.mjs";
import {
  finiteNumber,
  parseIntegerList,
  parsePlayerCounts,
  positiveInteger,
} from "./lib/rl-checkpoint-screening.mjs";

const args = process.argv.slice(2);
if (args[0] === "--") args.shift();
const { values } = parseArgs({
  args,
  options: {
    model: { type: "string", short: "m" },
    shard: { type: "string", multiple: true },
    output: { type: "string", short: "o" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    "match-counts": { type: "string" },
    acts: { type: "string", default: "5" },
    seed: { type: "string" },
    "reserved-final-seeds": { type: "string" },
    "min-point-diff": { type: "string", default: "0.25" },
    "min-lower-bound": { type: "string", default: "0.15" },
    "min-pairwise-rate": { type: "string", default: "0.55" },
    "role-regression-margin": { type: "string", default: "0.10" },
  },
});

if (!values.model) throw new TypeError("--model is required");
if (!values.output) throw new TypeError("--output is required");
if (!values.shard || values.shard.length < 2) {
  throw new TypeError("at least two --shard reports are required");
}
if (values.seed === undefined) {
  throw new TypeError("an explicit --seed is required");
}
if (!values["match-counts"]) {
  throw new TypeError("an explicit --match-counts plan is required");
}
const modelPath = resolve(values.model);
const modelBytes = await readFile(modelPath);
const model = JSON.parse(modelBytes.toString("utf8"));
if (model?.format !== "dalmuti-action-conditioned-actor-critic") {
  throw new TypeError("distributed checkpoint evaluation requires a V3 model");
}
const modelSha256 = createHash("sha256").update(modelBytes).digest("hex");
const playerCounts = parsePlayerCounts(values.players);
const matchCountsByPlayerCount = parseMatchCounts(
  values["match-counts"],
  playerCounts,
);
const seed = positiveInteger(values.seed, "seed");
const reservedFinalSeeds = parseIntegerList(
  values["reserved-final-seeds"],
  "reserved final seed",
);
if (reservedFinalSeeds.includes(seed)) {
  throw new Error("development shard seed is reserved for final evaluation");
}
const acts = positiveInteger(values.acts, "acts");
const promotionThresholds = {
  minPointDifference: finiteNumber(values["min-point-diff"], "min-point-diff"),
  minLowerBound: finiteNumber(values["min-lower-bound"], "min-lower-bound"),
  minPairwiseRate: finiteNumber(values["min-pairwise-rate"], "min-pairwise-rate"),
};
if (
  promotionThresholds.minPairwiseRate < 0 ||
  promotionThresholds.minPairwiseRate > 1
) {
  throw new RangeError("min-pairwise-rate must be between 0 and 1");
}
const roleRegressionMargin = finiteNumber(
  values["role-regression-margin"],
  "role-regression-margin",
);
if (roleRegressionMargin < 0) {
  throw new RangeError("role-regression-margin must be non-negative");
}

const shardEntries = [];
const seenPaths = new Set();
for (const value of values.shard) {
  const path = resolve(value);
  if (seenPaths.has(path.toLowerCase())) {
    throw new Error(`duplicate shard report path: ${path}`);
  }
  seenPaths.add(path.toLowerCase());
  const bytes = await readFile(path);
  shardEntries.push({
    path,
    bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    report: JSON.parse(bytes.toString("utf8")),
  });
}
const report = mergeBenchmarkShardReports(shardEntries, {
  modelPath,
  modelSha256,
  playerCounts,
  matchCountsByPlayerCount,
  acts,
  seed,
  promotionThresholds,
  roleRegressionMargin,
});
const outputPath = resolve(values.output);
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, {
  encoding: "utf8",
  flag: "wx",
});
console.log(`Merged ${shardEntries.length} strict benchmark shards: ${outputPath}`);
console.log(`Promotion gate: ${report.promotionPassed ? "PASS" : "FAIL"}`);
