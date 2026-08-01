import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  evaluateNonCardDecisionCounterfactuals,
  generateNonCardCounterfactualDataset,
  parseNonCardDecisionKinds,
  parseNonCardPlayerCounts,
  writeAllUtf8,
} from "../scripts/lib/non-card-counterfactual-data.mjs";
import {
  NonCardDeterminizationRejectedError,
  simulateMatch,
} from "../training/simulator.ts";

const CREATED_AT = "2026-08-01T00:00:00.000Z";

async function withTemporaryDirectory(run) {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-non-card-"));
  try {
    return await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

async function readNdjson(path) {
  const bytes = await readFile(path);
  return {
    bytes,
    records: bytes
      .toString("utf8")
      .trimEnd()
      .split("\n")
      .map((line) => JSON.parse(line)),
  };
}

async function assertChecksumSidecar(result, fileName) {
  assert.equal(
    await readFile(result.checksumPath, "utf8"),
    `${result.fileSha256}  ${fileName}\n`,
  );
  assert.equal(
    createHash("sha256")
      .update(await readFile(result.outputPath))
      .digest("hex"),
    result.fileSha256,
  );
}

// This checks direct record fields only. Replay seeds deliberately remain in
// the provenance artifact and can reconstruct a hidden world in the simulator.
function assertNoDirectHiddenCardFields(value) {
  const forbiddenKeys = new Set([
    "hand",
    "hands",
    "cards",
    "cardIds",
    "hiddenCards",
    "hiddenHand",
  ]);
  function visit(current) {
    if (!current || typeof current !== "object") return;
    for (const [key, child] of Object.entries(current)) {
      assert.equal(forbiddenKeys.has(key), false, `forbidden field: ${key}`);
      visit(child);
    }
  }
  visit(value);
}

test("CLI value parsers canonicalize deterministic player counts and decisions", () => {
  assert.deepEqual(parseNonCardPlayerCounts("10,4,7"), [4, 7, 10]);
  assert.deepEqual(parseNonCardDecisionKinds("revolution,tax-return"), [
    "tax-return",
    "revolution",
  ]);
  assert.throws(() => parseNonCardPlayerCounts("3"), /4 to 10/);
  assert.throws(() => parseNonCardPlayerCounts("4,4"), /duplicates/);
  assert.throws(() => parseNonCardDecisionKinds("tribute"), /unsupported/);
});

test("UTF-8 writer completes injected short writes and rejects zero progress", async () => {
  const chunks = [];
  const requestedWriteSizes = [1, 2, 3];
  const shortWritingHandle = {
    async write(bytes, offset, length, position) {
      assert.equal(Buffer.isBuffer(bytes), true);
      assert.equal(position, null);
      const bytesWritten = Math.min(
        requestedWriteSizes.shift() ?? length,
        length,
      );
      chunks.push(Buffer.from(bytes.subarray(offset, offset + bytesWritten)));
      return { bytesWritten, buffer: bytes };
    },
  };
  const value = "달무티 UTF-8\n";
  await writeAllUtf8(shortWritingHandle, value);
  assert.equal(Buffer.concat(chunks).toString("utf8"), value);
  assert.ok(chunks.length > 1);

  await assert.rejects(
    writeAllUtf8(
      {
        async write(bytes) {
          return { bytesWritten: 0, buffer: bytes };
        },
      },
      "x",
    ),
    /invalid or zero byte progress/,
  );
});

test("tax dataset exhausts legal roots in one world and is byte deterministic", async () => {
  await withTemporaryDirectory(async (directory) => {
    const firstPath = join(directory, "first.ndjson");
    const secondPath = join(directory, "second.ndjson");
    const calls = [];
    const simulate = (config) => {
      calls.push(config);
      return simulateMatch(config);
    };
    const options = {
      playerCounts: [10],
      episodes: 1,
      acts: 2,
      seed: 47,
      decisionKinds: ["tax-return"],
      maxDecisions: 1,
      temperature: 0.75,
      createdAt: CREATED_AT,
    };
    const first = await generateNonCardCounterfactualDataset({
      ...options,
      outputPath: firstPath,
      simulate,
    });
    const second = await generateNonCardCounterfactualDataset({
      ...options,
      outputPath: secondPath,
      determinizationWorlds: 1,
      determinizationRootSeed: 4_294_967_295,
      maxDeterminizationAttempts: 1,
    });
    const firstDataset = await readNdjson(firstPath);
    const secondDataset = await readNdjson(secondPath);
    assert.deepEqual(firstDataset.bytes, secondDataset.bytes);

    const [manifest, decision, summary] = firstDataset.records;
    assert.equal(manifest.collection.baselineNonCardHooks.constructor, Object);
    assert.deepEqual(manifest.collection.baselineNonCardHooks, {});
    assert.equal(manifest.collection.resumeAllowed, false);
    assert.equal(manifest.privacy.opponentCardIdentitiesIncluded, false);
    assert.equal(decision.decision, "tax-return");
    assert.equal(decision.targetSampleCount, 1);
    assert.equal(decision.actions.length, decision.legalActionIndices.length);
    assert.deepEqual(
      decision.actions.map((action) => action.actionIndex),
      decision.legalActionIndices,
    );
    assert.deepEqual(
      decision.legalMask.flatMap((legal, index) => (legal ? [index] : [])),
      decision.legalActionIndices,
    );
    assert.equal(
      new Set(decision.actions.map((action) => action.pairedWorldId)).size,
      1,
    );
    assert.equal(
      decision.actions.every(
        (action) =>
          Number.isFinite(action.terminalActorUtility) &&
          action.uncertainty.sampleStandardDeviation === 0 &&
          action.uncertainty.standardError === 0,
      ),
      true,
    );
    const centeredTotal = decision.actions.reduce(
      (total, action) => total + action.centeredUtility,
      0,
    );
    const probabilityTotal = decision.actions.reduce(
      (total, action) => total + action.softTargetProbability,
      0,
    );
    assert.ok(Math.abs(centeredTotal) < 1e-9);
    assert.ok(Math.abs(probabilityTotal - 1) < 1e-12);
    assertNoDirectHiddenCardFields(decision);

    assert.deepEqual(calls[0].nonCard, {});
    assert.equal(calls.length, 1 + decision.legalActionIndices.length);
    for (const call of calls.slice(1)) {
      assert.equal(call.seed, calls[0].seed);
      assert.equal(call.episodeId, calls[0].episodeId);
      assert.deepEqual(Object.keys(call.nonCard.forcedOverrides), ["taxReturn"]);
      const override = call.nonCard.forcedOverrides.taxReturn;
      assert.deepEqual(Object.keys(override), [decision.decisionKey]);
      assert.equal(decision.legalActionIndices.includes(override[decision.decisionKey]), true);
    }

    const preSummaryBytes = firstDataset.bytes
      .toString("utf8")
      .split("\n")
      .slice(0, -2)
      .map((line) => `${line}\n`)
      .join("");
    assert.equal(
      createHash("sha256").update(preSummaryBytes).digest("hex"),
      summary.hashes.contentBeforeSummary,
    );
    assert.equal(summary.decisionsWritten, 1);
    assert.equal(summary.actionEvaluations, decision.actions.length);
    assert.equal(first.fileSha256, second.fileSha256);
    await assertChecksumSidecar(first, "first.ndjson");
    await assertChecksumSidecar(second, "second.ndjson");

    await assert.rejects(
      generateNonCardCounterfactualDataset({
        ...options,
        outputPath: firstPath,
      }),
      /EEXIST/,
    );
  });
});

test("K=1 determinization options preserve legacy NDJSON bytes and file hash exactly", async () => {
  await withTemporaryDirectory(async (directory) => {
    const legacyPath = join(directory, "legacy.ndjson");
    const explicitPath = join(directory, "explicit-k1.ndjson");
    const options = {
      playerCounts: [4],
      episodes: 1,
      acts: 1,
      seed: 6,
      decisionKinds: ["revolution"],
      maxDecisions: 1,
      temperature: 1,
      createdAt: CREATED_AT,
    };
    const legacy = await generateNonCardCounterfactualDataset({
      ...options,
      outputPath: legacyPath,
    });
    const explicit = await generateNonCardCounterfactualDataset({
      ...options,
      outputPath: explicitPath,
      determinizationWorlds: 1,
      determinizationRootSeed: 999_999_999,
      maxDeterminizationAttempts: 99,
    });
    assert.deepEqual(
      (await readNdjson(legacyPath)).bytes,
      (await readNdjson(explicitPath)).bytes,
    );
    assert.equal(legacy.fileSha256, explicit.fileSha256);
    await assertChecksumSidecar(legacy, "legacy.ndjson");
    await assertChecksumSidecar(explicit, "explicit-k1.ndjson");
  });
});

test("an existing checksum sidecar rejects a fresh output without overwriting it", async () => {
  await withTemporaryDirectory(async (directory) => {
    const outputPath = join(directory, "blocked.ndjson");
    const checksumPath = `${outputPath}.sha256`;
    await writeFile(checksumPath, "owner-data\n", "utf8");
    await assert.rejects(
      generateNonCardCounterfactualDataset({
        outputPath,
        playerCounts: [4],
        episodes: 1,
        acts: 1,
        seed: 6,
        decisionKinds: ["revolution"],
        maxDecisions: 1,
        createdAt: CREATED_AT,
      }),
      /EEXIST/,
    );
    assert.equal(await readFile(checksumPath, "utf8"), "owner-data\n");
  });
});

test("K>1 aggregates paired hidden-world advantages without exposing worlds", async () => {
  await withTemporaryDirectory(async (directory) => {
    const firstPath = join(directory, "augmented-first.ndjson");
    const secondPath = join(directory, "augmented-second.ndjson");
    const options = {
      playerCounts: [4],
      episodes: 1,
      acts: 1,
      seed: 6,
      decisionKinds: ["revolution"],
      maxDecisions: 1,
      temperature: 0.8,
      determinizationWorlds: 3,
      continuationCount: 2,
      determinizationRootSeed: 123_456_789,
      maxDeterminizationAttempts: 8,
      createdAt: CREATED_AT,
    };
    const first = await generateNonCardCounterfactualDataset({
      ...options,
      outputPath: firstPath,
    });
    const second = await generateNonCardCounterfactualDataset({
      ...options,
      outputPath: secondPath,
    });
    const firstDataset = await readNdjson(firstPath);
    const secondDataset = await readNdjson(secondPath);
    assert.deepEqual(firstDataset.bytes, secondDataset.bytes);
    assert.equal(first.fileSha256, second.fileSha256);
    await assertChecksumSidecar(first, "augmented-first.ndjson");
    await assertChecksumSidecar(second, "augmented-second.ndjson");

    const [manifest, decision, summary] = firstDataset.records;
    assert.equal(manifest.version, 2);
    assert.equal(manifest.groupSplitKey, "canonicalInformationStateKey");
    assert.equal(
      manifest.collection.determinization.worldCountPerInformationState,
      3,
    );
    assert.equal(
      manifest.collection.determinization.continuationCountPerHiddenWorld,
      2,
    );
    assert.equal(
      manifest.collection.determinization.rawContinuationEvaluationsPerInformationState,
      6,
    );
    assert.equal(
      manifest.collection.determinization.effectiveIndependentWorldsPerInformationState,
      3,
    );
    assert.equal(
      manifest.determinizationSchema,
      "world-clustered-paired-baseline-advantages-v2",
    );
    assert.equal(manifest.privacy.individualReplaySeedsIncluded, false);
    assert.equal(manifest.privacy.explicitIndividualSeedsIncluded, false);
    assert.equal(
      manifest.privacy.individualSeedsDerivableFromRestrictedRootProvenance,
      true,
    );
    assert.equal(manifest.privacy.individualWorldUtilitiesIncluded, false);
    assert.equal(manifest.privacy.distribution, "restricted-training-only");
    assert.match(decision.canonicalInformationStateKey, /^sha256:[0-9a-f]{64}$/);
    assert.equal(decision.targetSampleCount, 3);
    assert.equal(decision.forcedActionEvaluations, decision.actions.length * 6);
    assert.equal(decision.determinization.rootSeed, 123_456_789);
    assert.equal(decision.determinization.acceptedWorldAttempts.length, 2);
    assert.equal(decision.determinization.individualReplaySeedsIncluded, false);
    assert.equal(
      decision.determinization.explicitIndividualSeedsIncluded,
      false,
    );
    assert.equal(
      decision.determinization
        .individualSeedsDerivableFromRestrictedRootProvenance,
      true,
    );
    assert.equal(decision.determinization.individualWorldUtilitiesIncluded, false);
    assert.equal(decision.determinization.continuationCount, 2);
    assert.equal(decision.determinization.rawContinuationEvaluations, 6);
    assert.equal(decision.determinization.effectiveIndependentWorlds, 3);
    assert.equal(decision.determinization.standardErrorEstimable, true);
    assert.equal(summary.actionEvaluations, decision.actions.length * 6);
    assert.equal(summary.determinization.resampledWorldsAccepted, 2);

    const baseline = decision.actions.find(
      (action) => action.actionIndex === decision.baselineActionIndex,
    );
    assert.ok(baseline);
    assert.deepEqual(baseline.pairedBaselineAdvantage, {
      count: 3,
      mean: 0,
      sampleStandardDeviation: 0,
      standardError: 0,
      standardErrorEstimable: true,
    });
    for (const action of decision.actions) {
      assert.equal(Object.hasOwn(action, "pairedWorldId"), false);
      assert.equal(Object.hasOwn(action, "terminalActorUtility"), false);
      assert.equal(Object.hasOwn(action, "decisionActUtility"), false);
      assert.equal(Object.hasOwn(action, "terminalFinishPlaceInDecisionAct"), false);
      assert.equal(Number.isFinite(action.meanUtility), true);
      assert.equal(Number.isFinite(action.pairedBaselineAdvantage.mean), true);
      assert.equal(
        action.uncertainty.standardError,
        action.uncertainty.sampleStandardDeviation / Math.sqrt(3),
      );
      assert.equal(action.uncertainty.count, 3);
      assert.equal(action.uncertainty.standardErrorEstimable, true);
      assert.equal(action.pairedBaselineAdvantage.count, 3);
      assert.equal(
        action.pairedDecisionActBaselineAdvantage.count,
        3,
      );
    }
    const serialized = JSON.stringify(firstDataset.records);
    assert.equal(serialized.includes("hiddenWorldSeed"), false);
    assert.equal(serialized.includes("terminalFinishPlaceInDecisionAct"), false);
    assertNoDirectHiddenCardFields(decision);
  });
});

test("duplicating continuations cannot shrink decision-act world uncertainty", () => {
  const baseConfig = {
    playerCount: 4,
    acts: 1,
    seed: 6,
    episodeId: "continuation-cluster-test",
    difficulties: ["normal"],
  };
  const baselineMatch = simulateMatch({ ...baseConfig, nonCard: {} });
  const baselineStep = baselineMatch.nonCardSteps.find(
    (step) => step.decision === "revolution",
  );
  assert.ok(baselineStep);
  const common = {
    baseConfig,
    baselineMatch,
    baselineStep,
    determinizationWorlds: 3,
    determinizationRootSeed: 987_654_321,
    maxDeterminizationAttempts: 8,
  };
  const oneContinuation = evaluateNonCardDecisionCounterfactuals({
    ...common,
    continuationCount: 1,
  });
  const fourContinuations = evaluateNonCardDecisionCounterfactuals({
    ...common,
    continuationCount: 4,
  });

  assert.equal(oneContinuation.targetSampleCount, 3);
  assert.equal(fourContinuations.targetSampleCount, 3);
  assert.equal(oneContinuation.determinization.rawContinuationEvaluations, 3);
  assert.equal(fourContinuations.determinization.rawContinuationEvaluations, 12);
  assert.deepEqual(
    fourContinuations.actions.map((action) => ({
      aggregate: action.decisionActUtilityAggregate,
      advantage: action.pairedDecisionActBaselineAdvantage,
    })),
    oneContinuation.actions.map((action) => ({
      aggregate: action.decisionActUtilityAggregate,
      advantage: action.pairedDecisionActBaselineAdvantage,
    })),
  );
});

test("finite determinization retries record rejection reasons without replay seeds", async () => {
  await withTemporaryDirectory(async (directory) => {
    const outputPath = join(directory, "retry.ndjson");
    let rejected = false;
    await generateNonCardCounterfactualDataset({
      outputPath,
      playerCounts: [4],
      episodes: 1,
      acts: 1,
      seed: 6,
      decisionKinds: ["revolution"],
      maxDecisions: 1,
      determinizationWorlds: 2,
      determinizationRootSeed: 77,
      maxDeterminizationAttempts: 3,
      createdAt: CREATED_AT,
      simulate(config) {
        if (config.nonCard?.determinization && !rejected) {
          rejected = true;
          throw new NonCardDeterminizationRejectedError(
            "synthetic-rejection",
            "synthetic deterministic retry",
          );
        }
        return simulateMatch(config);
      },
    });
    const { records } = await readNdjson(outputPath);
    const decision = records[1];
    assert.deepEqual(decision.determinization.acceptedWorldAttempts, [
      {
        worldIndex: 1,
        attemptCount: 2,
        rejectedAttemptCount: 1,
        rejectedReasonCounts: { "synthetic-rejection": 1 },
      },
    ]);
    assert.equal(JSON.stringify(decision).includes("hiddenWorldSeed"), false);
  });
});

test("tax augmentation filters return count and validates the full transfer", async () => {
  await withTemporaryDirectory(async (directory) => {
    const outputPath = join(directory, "tax-k2.ndjson");
    await generateNonCardCounterfactualDataset({
      outputPath,
      playerCounts: [10],
      episodes: 1,
      acts: 2,
      seed: 47,
      decisionKinds: ["tax-return"],
      taxReturnCounts: [2],
      maxDecisions: 1,
      determinizationWorlds: 2,
      determinizationRootSeed: 88,
      maxDeterminizationAttempts: 32,
      createdAt: CREATED_AT,
    });
    const { records } = await readNdjson(outputPath);
    const decision = records[1];
    assert.equal(decision.decision, "tax-return");
    assert.equal(decision.metadata.returnCount, 2);
    assert.equal(decision.targetSampleCount, 2);
    assert.equal(decision.determinization.resampledWorldCount, 1);
  });
});

test("strict public-history drift exhausts the finite retry cap", async () => {
  await withTemporaryDirectory(async (directory) => {
    const outputPath = join(directory, "history-drift.ndjson");
    let determinizedCalls = 0;
    await assert.rejects(
      generateNonCardCounterfactualDataset({
        outputPath,
        playerCounts: [10],
        episodes: 1,
        acts: 2,
        seed: 47,
        decisionKinds: ["tax-return"],
        taxReturnCounts: [2],
        maxDecisions: 1,
        determinizationWorlds: 2,
        continuationCount: 1,
        determinizationRootSeed: 99,
        maxDeterminizationAttempts: 2,
        createdAt: CREATED_AT,
        simulate(config) {
          const request = config.nonCard?.determinization;
          if (!request) return simulateMatch(config);
          determinizedCalls += 1;
          const publicHistory = request.expected.publicHistory.map((act) => ({
            ...act,
            finishOrder: [...act.finishOrder],
          }));
          publicHistory[0].finishOrder.reverse();
          return simulateMatch({
            ...config,
            nonCard: {
              ...config.nonCard,
              determinization: {
                ...request,
                expected: {
                  ...request.expected,
                  publicHistory,
                },
              },
            },
          });
        },
      }),
      /could not accept determinized world 1 in 2 attempts;.*public-history-drift/,
    );
    assert.equal(determinizedCalls, 2);
  });
});

test("a later-act revolution pairs declare against decline despite action-dependent tax", () => {
  let fixture = null;
  for (let seed = 1; seed <= 300 && fixture === null; seed += 1) {
    const baseConfig = {
      playerCount: 4,
      acts: 2,
      seed,
      episodeId: `later-revolution-${seed}`,
      difficulties: ["normal"],
    };
    const baselineMatch = simulateMatch({ ...baseConfig, nonCard: {} });
    const baselineStep = baselineMatch.nonCardSteps.find(
      (step) => step.decision === "revolution" && step.round === 2,
    );
    if (baselineStep) fixture = { baseConfig, baselineMatch, baselineStep };
  }
  assert.ok(fixture, "expected a deterministic round-two revolution fixture");
  const record = evaluateNonCardDecisionCounterfactuals({
    ...fixture,
    determinizationWorlds: 2,
    continuationCount: 1,
    determinizationRootSeed: 1234,
    maxDeterminizationAttempts: 8,
  });
  assert.equal(record.decision, "revolution");
  assert.equal(record.round, 2);
  assert.equal(record.targetSampleCount, 2);
  assert.deepEqual(record.legalActionIndices, [0, 1]);
});

test("revolution counterfactuals cover decline and declare with stable legality", async () => {
  await withTemporaryDirectory(async (directory) => {
    const outputPath = join(directory, "revolution.ndjson");
    await generateNonCardCounterfactualDataset({
      outputPath,
      playerCounts: [4],
      episodes: 1,
      acts: 1,
      seed: 6,
      decisionKinds: ["revolution"],
      maxDecisions: 1,
      createdAt: CREATED_AT,
    });
    const { records } = await readNdjson(outputPath);
    const decision = records.find(
      (record) => record.type === "counterfactual-decision",
    );
    assert.equal(decision.actorRole, "great-peon");
    assert.deepEqual(decision.legalMask, [true, true]);
    assert.deepEqual(
      decision.actions.map((action) => [action.actionIndex, action.actionFeatures]),
      [
        [0, [1, 0, 0]],
        [1, [0, 0, 1]],
      ],
    );
    assertNoDirectHiddenCardFields(decision);
  });
});

test("unused or mistargeted overrides fail instead of creating unpaired data", () => {
  const baseConfig = {
    playerCount: 10,
    acts: 2,
    seed: 47,
    episodeId: "unused-override-test",
    difficulties: ["normal"],
  };
  const baseline = simulateMatch({ ...baseConfig, nonCard: {} });
  const step = baseline.nonCardSteps.find(
    (candidate) => candidate.decision === "tax-return",
  );
  assert.ok(step);
  assert.throws(
    () =>
      evaluateNonCardDecisionCounterfactuals({
        baseConfig,
        baselineStep: {
          ...step,
          decisionKey: `${step.decisionKey}-never-occurs`,
        },
      }),
    /unused tax-return forced override/,
  );
});

test("a forced rerun is rejected if the targeted pre-action state drifts", () => {
  const baseConfig = {
    playerCount: 4,
    acts: 1,
    seed: 6,
    episodeId: "pre-action-drift-test",
    difficulties: ["normal"],
  };
  const baseline = simulateMatch({ ...baseConfig, nonCard: {} });
  const step = baseline.nonCardSteps.find(
    (candidate) => candidate.decision === "revolution",
  );
  assert.ok(step);
  assert.throws(
    () =>
      evaluateNonCardDecisionCounterfactuals({
        baseConfig,
        baselineStep: step,
        simulate(config) {
          const result = simulateMatch(config);
          const target = result.nonCardSteps.find(
            (candidate) =>
              candidate.decision === step.decision &&
              candidate.decisionKey === step.decisionKey,
          );
          target.observation = [...target.observation];
          target.observation[0] += 0.01;
          return result;
        },
      }),
    /targeted decision changed before forced action/,
  );
});
