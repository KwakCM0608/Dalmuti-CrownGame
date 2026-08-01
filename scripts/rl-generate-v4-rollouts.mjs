import { createHash } from "node:crypto";
import { once } from "node:events";
import { createReadStream, createWriteStream } from "node:fs";
import {
  access,
  link,
  mkdir,
  readFile,
  unlink,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import {
  V3_ACTION_CATALOGUE,
  V3_ACTION_CATALOGUE_VERSION,
} from "../training/v3-action-catalogue.ts";
import {
  encodeV3LegalMaskHex,
  legacyActionIndexToV3,
  legacyLegalActionIndicesToV3,
} from "../training/v3-action-bridge.ts";
import { simulateMatch } from "../training/simulator.ts";
import { createV4NormalWarmstartManifest } from "../training/v4-rollout-dataset.ts";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, "..");
const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();

const { values } = parseArgs({
  args: cliArgs,
  options: {
    players: { type: "string", default: "4" },
    acts: { type: "string", default: "5" },
    seed: { type: "string", default: "20260801" },
    "target-non-forced-decisions": {
      type: "string",
      default: "1000",
    },
    "max-episodes": { type: "string", default: "1000000" },
    output: {
      type: "string",
      default: "artifacts/rl/v4-normal-warmstart-p4.ndjson",
    },
  },
});

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

const playerCount = positiveInteger(values.players, "players");
if (playerCount < 4 || playerCount > 10) {
  throw new RangeError("players must be from 4 to 10");
}
const acts = positiveInteger(values.acts, "acts");
const seed = positiveInteger(values.seed, "seed");
const targetNonForcedDecisions = positiveInteger(
  values["target-non-forced-decisions"],
  "target-non-forced-decisions",
);
const maxEpisodes = positiveInteger(values["max-episodes"], "max-episodes");
const outputPath = resolve(values.output);
const partialPath = `${outputPath}.partial`;
const checksumPath = `${outputPath}.sha256`;

async function assertAbsent(path, label) {
  try {
    await access(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`${label} already exists: ${path}`);
}

async function sha256File(path) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

async function sha256Source(relativePath) {
  return createHash("sha256")
    .update(await readFile(resolve(repositoryRoot, relativePath)))
    .digest("hex");
}

const sourceHashes = {
  actorObservationContract: await sha256Source(
    "training/v4-public-history.ts",
  ),
  privilegedCriticContract: await sha256Source("training/simulator.ts"),
  actionCatalogue: createHash("sha256")
    .update(
      JSON.stringify({
        version: V3_ACTION_CATALOGUE_VERSION,
        catalogue: V3_ACTION_CATALOGUE,
      }),
    )
    .digest("hex"),
  normalPolicy: await sha256Source("lib/bot-strategy.ts"),
  generator: await sha256Source("scripts/rl-generate-v4-rollouts.mjs"),
  datasetManifest: await sha256Source("training/v4-rollout-dataset.ts"),
};

await mkdir(dirname(outputPath), { recursive: true });
await assertAbsent(outputPath, "output");
await assertAbsent(partialPath, "partial output");
await assertAbsent(checksumPath, "checksum output");

const output = createWriteStream(partialPath, {
  encoding: "utf8",
  flags: "wx",
});
await once(output, "open");
const recordDigest = createHash("sha256");

async function writeRecord(record, includeInRecordDigest = true) {
  const line = `${JSON.stringify(record)}\n`;
  if (includeInRecordDigest) recordDigest.update(line);
  if (!output.write(line)) await once(output, "drain");
}

await writeRecord(
  createV4NormalWarmstartManifest({
    playerCount,
    acts,
    initialSeed: seed,
    targetNonForcedDecisions,
    maxEpisodes,
    sourceHashes,
  }),
);

let episodes = 0;
let samples = 0;
let forcedSamples = 0;
let nonForcedSamples = 0;
for (let episode = 0; episode < maxEpisodes; episode += 1) {
  const episodeNumber = episode + 1;
  const match = simulateMatch({
    playerCount,
    acts,
    seed: seed + episode,
    // Bind the shard seed into public identities so independently generated
    // CPU shards can be merged without ambiguous episode/trajectory IDs.
    episodeId:
      `v4-normal-p${playerCount}-seed-${seed}-episode-${episodeNumber}`,
    difficulties: ["normal"],
    recordV4: true,
  });
  for (const step of match.steps) {
    if (
      !step.v4ActorObservation ||
      !step.v4PrivilegedCriticState ||
      !step.v4EventsAfterAction
    ) {
      throw new Error("recordV4 did not produce the complete V4 contract");
    }
    const legalActionIndices = legacyLegalActionIndicesToV3(
      step.legalActionIndices,
    );
    const actionIndex = legacyActionIndexToV3(step.actionIndex);
    if (!legalActionIndices.includes(actionIndex)) {
      throw new Error("Normal selected an action outside the V4 legal mask");
    }
    await writeRecord({
      type: "sample",
      trajectoryId: `${step.episodeId}:act-${step.round}:${step.actorId}`,
      episodeId: step.episodeId,
      act: step.round,
      step: step.step,
      actorId: step.actorId,
      actorSeat: step.actorSeat,
      actorRole: step.actorRole,
      actorObservation: step.v4ActorObservation,
      privilegedCriticState: step.v4PrivilegedCriticState,
      legalActionIndices,
      legalMaskHex: encodeV3LegalMaskHex(legalActionIndices),
      actionIndex,
      reward: step.reward,
      actorTerminal: step.actorTerminal,
      environmentTerminal: step.environmentTerminal,
      finishPlace: step.finishPlace,
      forced: legalActionIndices.length === 1,
      eventsAfterAction: step.v4EventsAfterAction,
    });
    samples += 1;
    if (legalActionIndices.length === 1) forcedSamples += 1;
    else nonForcedSamples += 1;
  }
  episodes = episodeNumber;
  if (nonForcedSamples >= targetNonForcedDecisions) break;
}

if (nonForcedSamples < targetNonForcedDecisions) {
  output.destroy();
  throw new Error(
    `max-episodes ${maxEpisodes} reached with only ` +
      `${nonForcedSamples} non-forced decisions`,
  );
}

const recordsBeforeSummarySha256 = recordDigest.digest("hex");
await writeRecord(
  {
    type: "summary",
    episodes,
    samples,
    forcedSamples,
    nonForcedSamples,
    targetNonForcedDecisions,
    recordsBeforeSummarySha256,
  },
  false,
);
output.end();
await once(output, "finish");

// A hard link is an atomic, no-overwrite promotion on the same filesystem.
// Removing the link named `.partial` leaves the exact completed bytes at the
// final path and cannot replace an existing dataset.
await link(partialPath, outputPath);
await unlink(partialPath);
const outputSha256 = await sha256File(outputPath);
await writeFile(checksumPath, `${outputSha256}\n`, {
  encoding: "ascii",
  flag: "wx",
});

console.log(`Wrote ${samples} samples to ${outputPath}`);
console.log(
  `Non-forced Normal decisions: ${nonForcedSamples} ` +
    `(target ${targetNonForcedDecisions})`,
);
console.log(`SHA-256: ${outputSha256}`);
