import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const {
  V4_PRIVILEGED_CRITIC_FEATURE_COUNT,
  V4_PRIVILEGED_CRITIC_LAYOUT,
  simulateMatch,
} = await import(new URL("../training/simulator.ts", import.meta.url));
const { decodeV3LegalMaskHex } = await import(
  new URL("../training/v3-action-bridge.ts", import.meta.url)
);

function baseConfig(playerCount = 4) {
  return {
    playerCount,
    acts: 1,
    seed: 12345,
    episodeId: `capture-p${playerCount}`,
    difficulties: ["normal"],
  };
}

function withoutV4(match) {
  const result = structuredClone(match);
  for (const step of result.steps) {
    delete step.v4ActorObservation;
    delete step.v4PrivilegedCriticState;
    delete step.v4EventsAfterAction;
  }
  return result;
}

test("recordV4 is opt-in and preserves every legacy result field and value", () => {
  const omitted = simulateMatch(baseConfig());
  const explicitFalse = simulateMatch({ ...baseConfig(), recordV4: false });
  const captured = simulateMatch({ ...baseConfig(), recordV4: true });

  assert.equal(
    omitted.steps.every(
      (step) =>
        !("v4ActorObservation" in step) &&
        !("v4PrivilegedCriticState" in step) &&
        !("v4EventsAfterAction" in step),
    ),
    true,
  );
  assert.deepEqual(explicitFalse, omitted);
  assert.deepEqual(withoutV4(captured), omitted);
});

test("V4 capture covers p4 and p10 with a 512-float separated critic", () => {
  for (const playerCount of [4, 10]) {
    const match = simulateMatch({
      ...baseConfig(playerCount),
      recordV4: true,
    });
    assert.ok(match.steps.length > 0);
    for (const step of match.steps) {
      const actor = step.v4ActorObservation;
      const critic = step.v4PrivilegedCriticState;
      assert.equal(actor.playerCount, playerCount);
      assert.equal(actor.playerTokens.length, playerCount);
      assert.equal(critic.playerCount, playerCount);
      assert.equal(critic.players.length, playerCount);
      assert.equal(
        critic.features.length,
        V4_PRIVILEGED_CRITIC_FEATURE_COUNT,
      );
      assert.equal(critic.features.every(Number.isFinite), true);
      assert.equal(
        critic.features
          .slice(V4_PRIVILEGED_CRITIC_LAYOUT.reservedZeroTail.offset)
          .every((value) => value === 0),
        true,
      );
    }
  }
});

test("V4 public history advances after actions and retains trick, clear, and finish detail", () => {
  const match = simulateMatch({ ...baseConfig(), recordV4: true });
  const priorEvents = [];
  for (const step of match.steps) {
    const observation = step.v4ActorObservation;
    assert.equal(
      observation.truncatedHistoryCount + observation.historyTokens.length,
      priorEvents.length,
    );
    if (priorEvents.length > 0) {
      assert.equal(
        observation.historyTokens.at(-1).sequence,
        priorEvents.at(-1).sequence,
      );
    }
    for (const event of step.v4EventsAfterAction) {
      assert.equal(event.sequence, priorEvents.length);
      assert.equal(Number.isInteger(event.handCountAfter), true);
      if (event.type === "play" || event.type === "clear") {
        assert.equal(Number.isInteger(event.jokerCount), true);
        assert.equal(
          event.totalCount,
          event.naturalCount + event.jokerCount,
        );
      }
      priorEvents.push(event);
    }
  }

  assert.deepEqual(
    [...new Set(priorEvents.map((event) => event.type))].sort(),
    ["clear", "finish", "pass", "play"],
  );
  assert.equal(
    priorEvents.some(
      (event) => event.type === "clear" && event.reason === "act-ended",
    ),
    true,
  );
  const occupiedTrick = match.steps.find(
    (step) => step.v4ActorObservation.table !== null,
  )?.v4ActorObservation.table;
  assert.ok(occupiedTrick);
  assert.equal(
    occupiedTrick.totalCount,
    occupiedTrick.naturalCount + occupiedTrick.jokerCount,
  );
  assert.equal(
    priorEvents.some(
      (event) =>
        event.type === "finish" &&
        event.handCountBefore === 0 &&
        event.handCountAfter === 0,
    ),
    true,
  );
});

test("actor capture excludes hidden hands while the critic contains every rank count", () => {
  const match = simulateMatch({ ...baseConfig(10), recordV4: true });
  for (const step of match.steps) {
    const actor = step.v4ActorObservation;
    const critic = step.v4PrivilegedCriticState;
    const actorJson = JSON.stringify(actor);
    assert.doesNotMatch(actorJson, /"actorId"|"handRankCounts"|player-\d+/);
    assert.equal(actor.ownHandCounts.length, 13);
    assert.equal(
      Object.prototype.hasOwnProperty.call(actor, "privilegedCriticState"),
      false,
    );

    let hiddenAndOwnCards = 0;
    for (const player of critic.players) {
      assert.equal(player.handRankCounts.length, 13);
      const playerHandCount = player.handRankCounts.reduce(
        (sum, count) => sum + count,
        0,
      );
      hiddenAndOwnCards += playerHandCount;
      const vectorOffset =
        V4_PRIVILEGED_CRITIC_LAYOUT.players.offset +
        player.relativeOffset * V4_PRIVILEGED_CRITIC_LAYOUT.players.stride +
        12;
      assert.deepEqual(
        critic.features.slice(vectorOffset, vectorOffset + 13),
        player.handRankCounts,
      );
    }
    assert.equal(
      hiddenAndOwnCards +
        critic.publicPlayedCounts.reduce((sum, count) => sum + count, 0),
      80,
    );
  }
});

async function generateTinyRollout(outputPath, seed = 701) {
  return execFileAsync(
    process.execPath,
    [
      "scripts/rl-generate-v4-rollouts.mjs",
      "--players",
      "4",
      "--acts",
      "1",
      "--seed",
      String(seed),
      "--target-non-forced-decisions",
      "1",
      "--max-episodes",
      "2",
      "--output",
      outputPath,
    ],
    { cwd: new URL("..", import.meta.url), maxBuffer: 1024 * 1024 },
  );
}

test("V4 Normal NDJSON is deterministic, hash-bound, exact-236, and exclusively promoted", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-v4-rollout-"));
  t.after(async () => {
    const { rm } = await import("node:fs/promises");
    await rm(directory, { recursive: true, force: true });
  });
  const firstPath = join(directory, "first.ndjson");
  const secondPath = join(directory, "second.ndjson");
  const disjointPath = join(directory, "disjoint.ndjson");
  await generateTinyRollout(firstPath);
  await generateTinyRollout(secondPath);
  await generateTinyRollout(disjointPath, 702);

  const firstBytes = await readFile(firstPath);
  const secondBytes = await readFile(secondPath);
  assert.deepEqual(firstBytes, secondBytes);
  await assert.rejects(readFile(`${firstPath}.partial`), /ENOENT/);
  const expectedSha256 = createHash("sha256")
    .update(firstBytes)
    .digest("hex");
  assert.equal(
    (await readFile(`${firstPath}.sha256`, "ascii")).trim(),
    expectedSha256,
  );

  const lines = firstBytes.toString("utf8").trimEnd().split("\n");
  const records = lines.map((line) => JSON.parse(line));
  const manifest = records[0];
  const summary = records.at(-1);
  const samples = records.slice(1, -1);
  assert.equal(manifest.format, "dalmuti-v4-normal-warmstart-ndjson");
  assert.equal(manifest.environment.playerCount, 4);
  assert.equal(manifest.environment.behaviorPolicy, "normal");
  assert.equal(manifest.actorObservation.schemaVersion, 4);
  assert.equal(manifest.privilegedCritic.featureCount, 512);
  assert.equal(manifest.actionSpace.size, 236);
  assert.equal(manifest.actionSpace.catalogue.length, 236);
  assert.equal(
    Object.values(manifest.sourceHashes).every((hash) =>
      /^[0-9a-f]{64}$/.test(hash),
    ),
    true,
  );
  assert.ok(summary.nonForcedSamples >= summary.targetNonForcedDecisions);
  assert.equal(summary.samples, samples.length);
  assert.equal(
    createHash("sha256")
      .update(`${lines.slice(0, -1).join("\n")}\n`)
      .digest("hex"),
    summary.recordsBeforeSummarySha256,
  );
  for (const sample of samples) {
    assert.deepEqual(
      decodeV3LegalMaskHex(sample.legalMaskHex),
      sample.legalActionIndices,
    );
    assert.equal(sample.legalActionIndices.includes(sample.actionIndex), true);
    assert.equal(sample.forced, sample.legalActionIndices.length === 1);
    assert.equal(sample.privilegedCriticState.features.length, 512);
    assert.equal(Number.isFinite(sample.reward), true);
  }

  const disjointRecords = (await readFile(disjointPath, "utf8"))
    .trimEnd()
    .split("\n")
    .map((line) => JSON.parse(line));
  const disjointEpisodeIds = new Set(
    disjointRecords
      .filter((record) => record.type === "sample")
      .map((record) => record.episodeId),
  );
  assert.equal(
    samples.some((sample) => disjointEpisodeIds.has(sample.episodeId)),
    false,
  );
  assert.equal(
    samples.every((sample) => sample.episodeId.includes("-seed-701-")),
    true,
  );

  const beforeRejectedRewrite = await readFile(firstPath);
  await assert.rejects(generateTinyRollout(firstPath), /already exists/);
  assert.deepEqual(await readFile(firstPath), beforeRejectedRewrite);
});
