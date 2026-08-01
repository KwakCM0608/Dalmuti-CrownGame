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
  V3_LEGAL_MASK_HEX_LENGTH,
  encodeV3LegalMaskHex,
  legacyActionIndexToV3,
  legacyLegalActionIndicesToV3,
} from "../training/v3-action-bridge.ts";
import { simulateMatch } from "../training/simulator.ts";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, "..");
const FORMAT = "dalmuti-v4-env-parity-ndjson";
const VERSION = 1;
const DEFAULT_BASE_SEED = 610_000_001;
const RESERVED_FINAL_SEED_START = 900_000_001;

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    players: { type: "string", default: "4-10" },
    acts: { type: "string", default: "5" },
    "seeds-per-player": { type: "string", default: "5" },
    "seed-base": { type: "string", default: String(DEFAULT_BASE_SEED) },
    output: {
      type: "string",
      default: "artifacts/rl/v4-env-parity-fixtures.ndjson",
    },
    "allow-small-test-fixture": { type: "boolean", default: false },
  },
});

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

function parsePlayers(value) {
  const result = value === "4-10"
    ? [4, 5, 6, 7, 8, 9, 10]
    : value.split(",").map((entry) => positiveInteger(entry, "players"));
  if (
    result.length < 1 ||
    new Set(result).size !== result.length ||
    result.some((count) => count < 4 || count > 10)
  ) {
    throw new RangeError("players must be unique values from 4 through 10");
  }
  return result;
}

const players = parsePlayers(values.players);
const acts = positiveInteger(values.acts, "acts");
const seedsPerPlayer = positiveInteger(
  values["seeds-per-player"],
  "seeds-per-player",
);
const seedBase = positiveInteger(values["seed-base"], "seed-base");
const testOnly = values["allow-small-test-fixture"];
if (
  !testOnly &&
  (players.join(",") !== "4,5,6,7,8,9,10" ||
    acts !== 5 ||
    seedsPerPlayer < 5)
) {
  throw new Error(
    "production parity fixtures require p4-p10, five acts, and at least five seeds per player",
  );
}

function seedsForPlayer(playerCount) {
  const result = Array.from(
    { length: seedsPerPlayer },
    (_, index) => seedBase + playerCount * 1_000_000 + index,
  );
  if (
    result.some(
      (seed) =>
        !Number.isSafeInteger(seed) ||
        seed < 1 ||
        seed >= RESERVED_FINAL_SEED_START,
    )
  ) {
    throw new RangeError(
      "parity fixture seeds must remain below the reserved final-seed range",
    );
  }
  return result;
}

const seedSchedule = Object.fromEntries(
  players.map((playerCount) => [playerCount, seedsForPlayer(playerCount)]),
);
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

function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

function playerIndex(playerId) {
  const match = /^player-(\d+)$/.exec(playerId);
  if (!match) throw new TypeError(`unexpected simulator player id: ${playerId}`);
  return Number(match[1]) - 1;
}

function revolutionId(value) {
  if (value === null) return 0;
  if (value === "revolution") return 1;
  if (value === "great-revolution") return 2;
  throw new TypeError(`unknown revolution state: ${String(value)}`);
}

function memoryTraceFloat64Bytes(memoryTraceVectors) {
  const flattened = memoryTraceVectors.flat();
  const bytes = Buffer.allocUnsafe(flattened.length * 8);
  flattened.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      throw new TypeError("memory trace contains a non-finite value");
    }
    bytes.writeDoubleLE(value, index * 8);
  });
  return bytes;
}

function observationCore(observation) {
  const historyBytes = Buffer.from(
    JSON.stringify(observation.historyTokens),
    "utf8",
  );
  const memoryBytes = memoryTraceFloat64Bytes(
    observation.memoryTraceVectors,
  );
  return {
    schemaVersion: observation.schemaVersion,
    playerCount: observation.playerCount,
    act: observation.act,
    actorRole: observation.actorRole,
    revolution: observation.revolution,
    ownHandCounts: observation.ownHandCounts,
    publicPlayedCounts: observation.publicPlayedCounts,
    table: observation.table,
    playerTokens: observation.playerTokens,
    truncatedHistoryCount: observation.truncatedHistoryCount,
    historyTokenCount: observation.historyTokens.length,
    historyFirstSequence:
      observation.historyTokens.length === 0
        ? -1
        : observation.historyTokens[0].sequence,
    historyLastSequence:
      observation.historyTokens.length === 0
        ? -1
        : observation.historyTokens.at(-1).sequence,
    historyTokenBytesLength: historyBytes.length,
    historyTokenBytesSha256: sha256Bytes(historyBytes),
    memoryTraceFloat64BytesLength: memoryBytes.length,
    memoryTraceFloat64Sha256: sha256Bytes(memoryBytes),
  };
}

function normalizeEvent(event) {
  const base = {
    type: event.type,
    sequence: event.sequence,
    actorIndex: playerIndex(event.actorId),
    handCountBefore: event.handCountBefore,
    handCountAfter: event.handCountAfter,
  };
  if (event.type === "play") {
    return {
      ...base,
      rank: event.rank,
      naturalCount: event.naturalCount,
      jokerCount: event.jokerCount,
      totalCount: event.totalCount,
    };
  }
  if (event.type === "pass") return { ...base, reason: event.reason };
  if (event.type === "clear") {
    return {
      ...base,
      rank: event.rank,
      naturalCount: event.naturalCount,
      jokerCount: event.jokerCount,
      totalCount: event.totalCount,
      reason: event.reason,
      nextLeaderIndex:
        event.nextLeaderId === null ? null : playerIndex(event.nextLeaderId),
    };
  }
  return { ...base, place: event.place };
}

function normalizedActSummary(act, playerCount) {
  const taxationApplied = act.round > 1 && act.revolution === null;
  return {
    act: act.round,
    playerOrder: act.playerOrder.map(playerIndex),
    finishOrder: act.finishOrder.map(playerIndex),
    revolution: revolutionId(act.revolution),
    taxation: {
      applied: taxationApplied,
      exchangeCounts: taxationApplied ? [2, 1] : [],
      transferredEachDirection: taxationApplied ? 3 : 0,
    },
    chipAwardsByPlayer: Array.from(
      { length: playerCount },
      (_, index) => act.chipAwards[`player-${index + 1}`],
    ),
    transitions: act.transitions,
  };
}

const sourcePaths = [
  "training/simulator.ts",
  "lib/bot-strategy.ts",
  "training/v3-action-catalogue.ts",
  "training/v3-action-bridge.ts",
  "training/v4-public-history.ts",
  "gpu-training/v4_env.py",
  "scripts/rl-generate-v4-env-parity-fixtures.mjs",
];
const sourceHashes = Object.fromEntries(
  await Promise.all(
    sourcePaths.map(async (path) => [path, await sha256Source(path)]),
  ),
);
const catalogueSha256 = sha256Bytes(
  Buffer.from(JSON.stringify(V3_ACTION_CATALOGUE), "utf8"),
);

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
let recordCount = 0;

async function writeRecord(record, includeInDigest = true) {
  const line = `${JSON.stringify(record)}\n`;
  if (includeInDigest) recordDigest.update(line);
  if (!output.write(line)) await once(output, "drain");
  recordCount += 1;
}

await writeRecord({
  type: "manifest",
  format: FORMAT,
  version: VERSION,
  testOnly,
  environment: {
    players,
    acts,
    behaviorPolicy: "normal",
    seedsPerPlayer,
    seedBase,
    seedSchedule,
    reservedFinalSeedFloor: RESERVED_FINAL_SEED_START,
  },
  actionSpace: {
    version: V3_ACTION_CATALOGUE_VERSION,
    size: V3_ACTION_CATALOGUE.length,
    legalMaskHexLength: V3_LEGAL_MASK_HEX_LENGTH,
    catalogueSha256,
  },
  actorObservation: {
    schemaVersion: 4,
    maxHistoryEvents: 192,
    historyTokenBytesEncoding: "utf8-json-stringify",
    memoryTraceBytesEncoding: "float64-little-endian",
  },
  sourceHashes,
});

let matchCount = 0;
let decisionCount = 0;
for (const playerCount of players) {
  for (const seed of seedSchedule[playerCount]) {
    const matchId = `p${playerCount}-seed-${seed}`;
    const match = simulateMatch({
      playerCount,
      acts,
      seed,
      episodeId: `v4-env-parity-${matchId}`,
      difficulties: ["normal"],
      recordV4: true,
    });
    await writeRecord({
      type: "match-start",
      matchId,
      playerCount,
      seed,
      acts,
    });
    for (const act of match.acts) {
      const steps = match.steps.filter((step) => step.round === act.round);
      if (steps.length < 1 || !steps[0].v4ActorObservation) {
        throw new Error(`${matchId} act ${act.round} has no V4 decision`);
      }
      const actSummary = normalizedActSummary(act, playerCount);
      await writeRecord({
        type: "act-start",
        matchId,
        act: act.round,
        playerOrder: actSummary.playerOrder,
        revolution: actSummary.revolution,
        taxation: actSummary.taxation,
        firstActorIndex: playerIndex(steps[0].actorId),
        initialObservationCore: observationCore(
          steps[0].v4ActorObservation,
        ),
      });
      for (const step of steps) {
        if (!step.v4ActorObservation || !step.v4EventsAfterAction) {
          throw new Error(`${matchId} act ${act.round} has incomplete V4 capture`);
        }
        const legalActionIndices = legacyLegalActionIndicesToV3(
          step.legalActionIndices,
        );
        const actionIndex = legacyActionIndexToV3(step.actionIndex);
        if (!legalActionIndices.includes(actionIndex)) {
          throw new Error(`${matchId} Normal action escaped its legal mask`);
        }
        await writeRecord({
          type: "decision",
          matchId,
          act: act.round,
          decision: step.step,
          actorIndex: playerIndex(step.actorId),
          actorSeat: step.actorSeat,
          actorRole: step.actorRole,
          observationCore: observationCore(step.v4ActorObservation),
          legalMaskHex: encodeV3LegalMaskHex(legalActionIndices),
          normalActionIndex: actionIndex,
          eventsAfterAction: step.v4EventsAfterAction.map(normalizeEvent),
        });
        decisionCount += 1;
      }
      await writeRecord({
        type: "act-summary",
        matchId,
        ...actSummary,
      });
    }
    await writeRecord({
      type: "match-summary",
      matchId,
      finalScoresByPlayer: Array.from(
        { length: playerCount },
        (_, index) => match.finalScores[`player-${index + 1}`],
      ),
    });
    matchCount += 1;
  }
}

const recordsBeforeSummarySha256 = recordDigest.digest("hex");
await writeRecord(
  {
    type: "summary",
    matches: matchCount,
    decisions: decisionCount,
    recordsBeforeSummary: recordCount,
    recordsBeforeSummarySha256,
  },
  false,
);
output.end();
await once(output, "finish");
await link(partialPath, outputPath);
await unlink(partialPath);
const outputSha256 = await sha256File(outputPath);
await writeFile(checksumPath, `${outputSha256}\n`, {
  encoding: "ascii",
  flag: "wx",
});

console.log(
  `Wrote ${matchCount} matches and ${decisionCount} decisions to ${outputPath}`,
);
console.log(`SHA-256: ${outputSha256}`);
