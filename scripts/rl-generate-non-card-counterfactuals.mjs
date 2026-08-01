import { parseArgs } from "node:util";

import {
  generateNonCardCounterfactualDataset,
  parseNonCardDecisionKinds,
  parseNonCardPlayerCounts,
} from "./lib/non-card-counterfactual-data.mjs";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();

const { values } = parseArgs({
  args: cliArgs,
  options: {
    decision: { type: "string", default: "all" },
    episodes: { type: "string", default: "10" },
    acts: { type: "string", default: "3" },
    players: { type: "string", default: "4,5,6,7,8,9,10" },
    seed: { type: "string", default: "710001" },
    temperature: { type: "string", default: "1" },
    "hidden-worlds": { type: "string", default: "1" },
    continuations: { type: "string", default: "1" },
    "determinization-root-seed": { type: "string" },
    "max-determinization-attempts": { type: "string", default: "32" },
    "tax-return-count": { type: "string", default: "all" },
    "max-decisions": { type: "string" },
    "created-at": { type: "string" },
    output: {
      type: "string",
      short: "o",
      default: "artifacts/rl/non-card-counterfactuals.ndjson",
    },
  },
});

const result = await generateNonCardCounterfactualDataset({
  outputPath: values.output,
  playerCounts: parseNonCardPlayerCounts(values.players),
  decisionKinds: parseNonCardDecisionKinds(values.decision),
  episodes: values.episodes,
  acts: values.acts,
  seed: values.seed,
  temperature: values.temperature,
  determinizationWorlds: values["hidden-worlds"],
  continuationCount: values.continuations,
  determinizationRootSeed: values["determinization-root-seed"],
  maxDeterminizationAttempts: values["max-determinization-attempts"],
  taxReturnCounts: values["tax-return-count"],
  maxDecisions: values["max-decisions"],
  createdAt: values["created-at"],
});

console.log(`Wrote ${result.summary.decisionsWritten} paired decisions`);
console.log(`Forced action evaluations: ${result.summary.actionEvaluations}`);
console.log(`Dataset: ${result.outputPath}`);
console.log(`SHA-256: ${result.fileSha256}`);
console.log(`Checksum: ${result.checksumPath}`);
