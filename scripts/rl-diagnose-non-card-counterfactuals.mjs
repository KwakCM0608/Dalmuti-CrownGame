import { parseArgs } from "node:util";

import {
  diagnoseNonCardCounterfactualDataset,
  writeNonCardDiagnosticReportExclusive,
} from "./lib/non-card-data-diagnostics.mjs";

const args = process.argv.slice(2);
if (args[0] === "--") args.shift();

const { values } = parseArgs({
  args,
  options: {
    input: { type: "string", short: "i" },
    output: { type: "string", short: "o" },
    "json-stdout": { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(
    "Usage: node scripts/rl-diagnose-non-card-counterfactuals.mjs --input <dataset.ndjson> [--output <new-report.json>] [--json-stdout]",
  );
  process.exit(0);
}
if (!values.input) throw new TypeError("--input is required");

const report = await diagnoseNonCardCounterfactualDataset(values.input);
const byDecision = Object.fromEntries(
  report.metrics.byDecision.map((entry) => [entry.dimensions.decision, entry]),
);

function percentage(value) {
  return `${(value * 100).toFixed(1)}%`;
}

console.log(
  `Validated ${report.validation.records} decisions / ${report.validation.actionEvaluations} actions in ${report.validation.pairedWorlds} paired worlds.`,
);
for (const decision of ["tax-return", "revolution"]) {
  const metric = byDecision[decision];
  if (!metric) continue;
  const ci = metric.baselineCenteredUtilityWorldClustered95;
  const declare = metric.revolutionDeclareRates;
  console.log(
    [
      decision,
      `records=${metric.records}`,
      `legal=${metric.meanLegalActionCount.toFixed(2)}`,
      `baseline-best=${percentage(metric.baselineActionBestLabelAgreementRate)}`,
      `baseline-optimal-tie=${percentage(metric.baselineActionUtilityOptimalTieAgreementRate)}`,
      `oracle-ties=${metric.meanOracleMaximumUtilityTieCount.toFixed(2)}`,
      `unique-best=${percentage(metric.uniqueOracleBestRate)}`,
      `baseline-centered=${metric.meanBaselineCenteredUtility.toFixed(4)}`,
      `world-CI=[${ci.low.toFixed(4)}, ${ci.high.toFixed(4)}]`,
      `oracle-gap=${metric.meanOracleAdvantageOverBaseline.toFixed(4)}`,
      declare
        ? `declare baseline/best=${percentage(declare.baseline)}/${percentage(declare.best)}`
        : null,
    ]
      .filter(Boolean)
      .join(" | "),
  );
}
const tax = byDecision["tax-return"]?.taxReturnRankStrength;
if (tax) {
  for (const name of ["baseline", "best", "lowestLegal"]) {
    const metric = tax[name];
    console.log(
      `tax ${name}: mean-normal-rank=${metric.meanNormalRank?.toFixed(3) ?? "n/a"}, ` +
        `strength=${metric.meanNormalRankStrength?.toFixed(3) ?? "n/a"}, ` +
        `joker-rate=${percentage(metric.jokerCardRate)}`,
    );
  }
}
if (values.output) {
  const output = await writeNonCardDiagnosticReportExclusive(report, values.output);
  console.log(`Report: ${output}`);
}
if (values["json-stdout"]) console.log(JSON.stringify(report, null, 2));
