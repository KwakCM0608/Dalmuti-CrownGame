import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  diagnoseNonCardCounterfactualDataset,
  writeNonCardDiagnosticReportExclusive,
} from "../scripts/lib/non-card-data-diagnostics.mjs";
import { generateNonCardCounterfactualDataset } from "../scripts/lib/non-card-counterfactual-data.mjs";
import { TAX_RETURN_ACTION_CATALOGUE } from "../training/non-card-action-space.ts";

async function withTemp(run) {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-noncard-diagnostic-"));
  try {
    return await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

async function createTaxFixture(path) {
  return generateNonCardCounterfactualDataset({
    outputPath: path,
    playerCounts: [10],
    episodes: 1,
    acts: 2,
    seed: 47,
    decisionKinds: ["tax-return"],
    maxDecisions: 1,
    temperature: 1,
    createdAt: "2026-08-01T00:00:00.000Z",
  });
}

test("streaming diagnostic validates targets and computes requested tax metrics", async () => {
  await withTemp(async (directory) => {
    const dataset = join(directory, "tax.ndjson");
    await createTaxFixture(dataset);
    const records = (await readFile(dataset, "utf8"))
      .trimEnd()
      .split("\n")
      .map(JSON.parse);
    const decision = records.find((record) => record.type === "counterfactual-decision");
    const actions = new Map(decision.actions.map((action) => [action.actionIndex, action]));
    const report = await diagnoseNonCardCounterfactualDataset(dataset);
    const metric = report.metrics.overall;

    assert.equal(report.validation.complete, true);
    assert.equal(report.validation.records, 1);
    assert.equal(metric.records, 1);
    assert.equal(metric.meanLegalActionCount, decision.actions.length);
    assert.equal(
      metric.baselineActionBestLabelAgreementRate,
      Number(decision.baselineActionIndex === decision.bestActionIndex),
    );
    assert.equal(
      metric.meanBaselineCenteredUtility,
      actions.get(decision.baselineActionIndex).centeredUtility,
    );
    assert.equal(
      metric.meanLowestIndexCenteredUtility,
      actions.get(decision.legalActionIndices[0]).centeredUtility,
    );
    assert.equal(
      metric.meanOracleAdvantageOverBaseline,
      actions.get(decision.bestActionIndex).meanUtility -
        actions.get(decision.baselineActionIndex).meanUtility,
    );
    const maximum = actions.get(decision.bestActionIndex).meanUtility;
    assert.equal(
      metric.meanOracleMaximumUtilityTieCount,
      decision.actions.filter((action) => action.meanUtility === maximum).length,
    );
    assert.equal(metric.meanTargetSampleCount, 1);
    assert.equal(metric.singleWorldTargetRate, 1);
    assert.equal(metric.baselineCenteredUtilityWorldClustered95.clusters, 1);
    assert.equal(metric.taxReturnRankStrength.records, 1);
    const baselineRanks = TAX_RETURN_ACTION_CATALOGUE[decision.baselineActionIndex].ranks;
    const expectedJokers = baselineRanks.filter((rank) => rank === 13).length;
    assert.equal(
      metric.taxReturnRankStrength.baseline.jokerCards,
      expectedJokers,
    );
    assert.equal(report.metrics.byPlayerCount[0].dimensions.playerCount, 10);
    assert.equal(report.metrics.byReturnCount.length, 1);
    assert.deepEqual(report.metrics.byDecisionPlayerCount[0].dimensions, {
      decision: "tax-return",
      playerCount: 10,
    });
    assert.deepEqual(report.metrics.byDecisionActorRole[0].dimensions, {
      decision: "tax-return",
      actorRole: decision.actorRole,
    });
    assert.equal(report.metrics.cells.length, 1);
  });
});

test("diagnostic rejects missing summaries and corrupted summary hashes", async () => {
  await withTemp(async (directory) => {
    const dataset = join(directory, "source.ndjson");
    await createTaxFixture(dataset);
    const lines = (await readFile(dataset, "utf8")).trimEnd().split("\n");

    const incomplete = join(directory, "incomplete.ndjson");
    await writeFile(incomplete, `${lines.slice(0, -1).join("\n")}\n`, "utf8");
    await assert.rejects(
      diagnoseNonCardCounterfactualDataset(incomplete),
      /incomplete dataset: missing final summary/,
    );

    const summary = JSON.parse(lines.at(-1));
    summary.hashes.contentBeforeSummary = "0".repeat(64);
    const corrupt = join(directory, "corrupt.ndjson");
    await writeFile(
      corrupt,
      `${[...lines.slice(0, -1), JSON.stringify(summary)].join("\n")}\n`,
      "utf8",
    );
    await assert.rejects(
      diagnoseNonCardCounterfactualDataset(corrupt),
      /SHA-256 mismatch/,
    );
  });
});

test("diagnostic rejects incomplete action coverage and report output is exclusive", async () => {
  await withTemp(async (directory) => {
    const dataset = join(directory, "source.ndjson");
    await createTaxFixture(dataset);
    const lines = (await readFile(dataset, "utf8")).trimEnd().split("\n");
    const decision = JSON.parse(lines[1]);
    decision.actions.pop();
    const incompleteActions = join(directory, "actions.ndjson");
    await writeFile(
      incompleteActions,
      `${[lines[0], JSON.stringify(decision), lines.at(-1)].join("\n")}\n`,
      "utf8",
    );
    await assert.rejects(
      diagnoseNonCardCounterfactualDataset(incompleteActions),
      /actions do not cover every legal action/,
    );

    const report = await diagnoseNonCardCounterfactualDataset(dataset);
    const output = join(directory, "nested", "report.json");
    assert.equal(await writeNonCardDiagnosticReportExclusive(report, output), output);
    await assert.rejects(
      writeNonCardDiagnosticReportExclusive(report, output),
      /EEXIST/,
    );
  });
});
