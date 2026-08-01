import assert from "node:assert/strict";
import test from "node:test";

import {
  EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
  EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION,
  createExperimentalMerchantRevolutionHook,
  selectExperimentalMerchantRevolution,
} from "../scripts/lib/experimental-revolution-merchant-candidate.mjs";
import { runExperimentalMerchantRevolutionPairedBenchmark } from "../scripts/lib/revolution-merchant-paired-benchmark.mjs";
import {
  REVOLUTION_DECLARE_ACTION_INDEX,
  REVOLUTION_DECLINE_ACTION_INDEX,
} from "../training/non-card-action-space.ts";

function roleAt(seat, playerCount) {
  if (seat === 0) return "great-dalmuti";
  if (seat === 1) return "lesser-dalmuti";
  if (seat === playerCount - 2) return "lesser-peon";
  if (seat === playerCount - 1) return "great-peon";
  return "merchant";
}

function observationFor(playerCount, actorSeat = 2) {
  const hand = [
    { id: "joker-a", rank: 13 },
    { id: "joker-b", rank: 13 },
    { id: "rank-5-a", rank: 5 },
    { id: "rank-9-a", rank: 9 },
  ];
  const players = Array.from({ length: playerCount }, (_, seat) => ({
    id: `player-${seat + 1}`,
    role: roleAt(seat, playerCount),
    handCount: seat === actorSeat ? hand.length : 12,
    score: seat - 2,
  }));
  return {
    actorId: players[actorSeat].id,
    hand,
    players,
    round: 3,
  };
}

test("the frozen candidate changes only merchant p6 decisions", () => {
  assert.deepEqual(EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS, [6]);
  for (const playerCount of [5, 6, 7, 8, 9, 10]) {
    const selected = selectExperimentalMerchantRevolution(
      observationFor(playerCount),
    );
    assert.equal(selected.baselineActionIndex, REVOLUTION_DECLINE_ACTION_INDEX);
    if (playerCount === 6) {
      assert.equal(selected.actionIndex, REVOLUTION_DECLARE_ACTION_INDEX);
      assert.equal(selected.changedFromBaseline, true);
      assert.equal(selected.routing, "merchant-declare");
    } else {
      assert.equal(selected.actionIndex, REVOLUTION_DECLINE_ACTION_INDEX);
      assert.equal(selected.changedFromBaseline, false);
      assert.equal(selected.routing, "exact-normal-fallback");
    }
  }
});

test("every non-merchant role is routed through the exact current normal decision", () => {
  const expectedBySeat = new Map([
    [0, REVOLUTION_DECLINE_ACTION_INDEX],
    [1, REVOLUTION_DECLINE_ACTION_INDEX],
    [4, REVOLUTION_DECLARE_ACTION_INDEX],
    [5, REVOLUTION_DECLARE_ACTION_INDEX],
  ]);
  for (const [actorSeat, expected] of expectedBySeat) {
    const selected = selectExperimentalMerchantRevolution(
      observationFor(6, actorSeat),
    );
    assert.equal(selected.actionIndex, expected);
    assert.equal(selected.baselineActionIndex, expected);
    assert.equal(selected.changedFromBaseline, false);
    assert.equal(selected.routing, "exact-normal-fallback");
    assert.equal(selected.reason, "non-merchant-role");
  }
});

test("the simulator hook applies the candidate only to selected player IDs", () => {
  const telemetry = [];
  const hook = createExperimentalMerchantRevolutionHook({
    candidateIds: new Set(["player-3"]),
    telemetry,
  });
  const observation = observationFor(6);
  const context = {
    decision: "revolution",
    decisionKey: "episode\u00003\u0000player-3",
    actorId: "player-3",
    actorRole: "merchant",
    observation,
  };
  const candidate = hook.revolutionPolicy(context);
  assert.equal(candidate.actionIndex, REVOLUTION_DECLARE_ACTION_INDEX);
  assert.equal(candidate.policyVersion, EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION);
  assert.equal(telemetry[0].changedFromBaseline, true);

  const otherTelemetry = [];
  const otherHook = createExperimentalMerchantRevolutionHook({
    candidateIds: new Set(["player-4"]),
    telemetry: otherTelemetry,
  });
  const other = otherHook.revolutionPolicy(context);
  assert.equal(other.actionIndex, REVOLUTION_DECLINE_ACTION_INDEX);
  assert.equal(otherTelemetry[0].routing, "non-candidate-exact-normal");
  assert.equal(otherTelemetry[0].changedFromBaseline, false);
});

test("candidate validation rejects malformed count contracts and hidden public cards", () => {
  const valid = observationFor(6);
  assert.throws(
    () =>
      selectExperimentalMerchantRevolution(valid, {
        enabledPlayerCounts: [6, 6],
      }),
    /duplicate enabled player count 6/,
  );
  assert.throws(
    () =>
      createExperimentalMerchantRevolutionHook({
        candidateIds: new Set(["player-3"]),
        enabledPlayerCounts: [3],
      }),
    /integers from 4 to 10/,
  );
  const leaked = observationFor(6);
  leaked.players[0].hand = [{ id: "secret", rank: 1 }];
  assert.throws(
    () => selectExperimentalMerchantRevolution(leaked),
    /must not contain hand/,
  );
});

function verifiedSourceEvidence() {
  return {
    format: "dalmuti-experimental-merchant-revolution-data-audit",
    version: 1,
    source: {
      sha256: "b861bc857e4e9d845f224ab06b3f9b2c503f9e3721e8e6979a99ef694dc96a05",
    },
    contract: {
      normalBaselineRecomputed: true,
      canonicalInformationStateKeysRecomputed: true,
      canonicalInformationStateKeysUnique: true,
    },
    evidence: {
      evidenceSupportedCounts: [6],
    },
  };
}

test("paired benchmark proves disabled counts are exact no-ops and changes only p6 merchants", () => {
  const report = runExperimentalMerchantRevolutionPairedBenchmark({
    sourceEvidence: verifiedSourceEvidence(),
    playerCounts: [4, 6],
    matchCountsByPlayerCount: { 4: 8, 6: 40 },
    acts: 5,
    seed: 941000001,
    includeMatchData: false,
  });
  const p4 = report.results.find((result) => result.playerCount === 4);
  const p6 = report.results.find((result) => result.playerCount === 6);
  assert.equal(p4.enabledCandidateCount, false);
  assert.equal(p4.trajectoryParity.exactRate, 1);
  assert.equal(p4.pairedMarginal.chipDifference.mean, 0);
  assert.equal(p4.interventionRouting.changedFromNormal, 0);

  assert.equal(p6.enabledCandidateCount, true);
  assert.ok(p6.interventionRouting.changedFromNormal > 0);
  assert.equal(
    p6.interventionRouting.byRole.merchant.changedFromNormal,
    p6.interventionRouting.changedFromNormal,
  );
  for (const role of [
    "great-dalmuti",
    "lesser-dalmuti",
    "lesser-peon",
    "great-peon",
  ]) {
    assert.equal(p6.interventionRouting.byRole[role].changedFromNormal, 0);
  }
  assert.equal(report.evaluationDesign.promotionGatesApplied, false);
});

test("paired benchmark refuses unverified or mismatched source evidence", () => {
  const evidence = verifiedSourceEvidence();
  evidence.evidence.evidenceSupportedCounts = [6, 9];
  assert.throws(
    () =>
      runExperimentalMerchantRevolutionPairedBenchmark({
        sourceEvidence: evidence,
        playerCounts: [6],
        matchCountsByPlayerCount: { 6: 1 },
        acts: 1,
        seed: 1,
      }),
    /does not prove the frozen candidate contract/,
  );
});
