import { mkdir, open } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

import { auditRevolutionMerchantSourceData } from "./lib/revolution-merchant-data-audit.mjs";

const args = process.argv.slice(2);
if (args[0] === "--") args.shift();
const { values } = parseArgs({
  args,
  options: {
    input: { type: "string", short: "i" },
    output: { type: "string", short: "o" },
    "expected-sha256": { type: "string" },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(
    "Usage: node scripts/rl-audit-revolution-merchant-candidate.mjs " +
      "--input <v2-revolution.ndjson> --output <new-report.json> " +
      "[--expected-sha256 <hex>]",
  );
  process.exit(0);
}
if (!values.input) throw new TypeError("--input is required");
if (!values.output) throw new TypeError("--output is required");
if (
  values["expected-sha256"] !== undefined &&
  !/^[0-9a-f]{64}$/.test(values["expected-sha256"])
) {
  throw new TypeError("--expected-sha256 must be 64 lowercase hexadecimal characters");
}

const report = await auditRevolutionMerchantSourceData(values.input, {
  ...(values["expected-sha256"]
    ? { expectedSha256: values["expected-sha256"] }
    : {}),
});
const outputPath = resolve(values.output);
await mkdir(dirname(outputPath), { recursive: true });
const handle = await open(outputPath, "wx");
try {
  await handle.writeFile(`${JSON.stringify(report, null, 2)}\n`, "utf8");
} finally {
  await handle.close();
}

const eligible = report.evidence.eligibleChangedRecords
  .decisionActActualChipAdvantage;
console.log(
  `Validated ${report.source.records} decisions; candidate changes ` +
    `${report.evidence.eligibleChangedRecords.records}.`,
);
console.log(
  `Eligible merchant current-act advantage: ${eligible.mean.toFixed(4)} ` +
    `[${eligible.confidence95.low.toFixed(4)}, ` +
    `${eligible.confidence95.high.toFixed(4)}] actual chips.`,
);
for (const [playerCount, cell] of Object.entries(
  report.evidence.eligibleByPlayerCount,
)) {
  const metric = cell.decisionActActualChipAdvantage;
  console.log(
    `p${playerCount}: n=${metric.clusters}, ${metric.mean.toFixed(4)} ` +
      `[${metric.confidence95.low.toFixed(4)}, ` +
      `${metric.confidence95.high.toFixed(4)}]`,
  );
}
console.log(`Saved audit report to ${outputPath}`);
