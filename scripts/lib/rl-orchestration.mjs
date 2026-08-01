import { createHash } from "node:crypto";
import {
  createReadStream,
  createWriteStream,
} from "node:fs";
import {
  mkdir,
  open,
  readdir,
  stat,
} from "node:fs/promises";
import { once } from "node:events";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import { createDeflateRaw } from "node:zlib";

export function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

export function finiteNumber(value, label, { minimum, maximum } = {}) {
  const parsed = Number(value);
  if (
    !Number.isFinite(parsed) ||
    (minimum !== undefined && parsed < minimum) ||
    (maximum !== undefined && parsed > maximum)
  ) {
    const range =
      minimum !== undefined && maximum !== undefined
        ? ` from ${minimum} to ${maximum}`
        : minimum !== undefined
          ? ` >= ${minimum}`
          : maximum !== undefined
            ? ` <= ${maximum}`
            : "";
    throw new RangeError(`${label} must be a finite number${range}`);
  }
  return parsed;
}

export function parsePlayerCountOverrides(values, label) {
  const overrides = new Map();
  for (const value of values ?? []) {
    for (const pair of value.split(",")) {
      const match = /^(\d+)=(\d+)$/.exec(pair.trim());
      if (!match) {
        throw new TypeError(
          `${label} entries must use PLAYER_COUNT=VALUE`,
        );
      }
      const playerCount = positiveInteger(match[1], `${label} player count`);
      const count = positiveInteger(match[2], `${label} value`);
      if (playerCount < 4 || playerCount > 10) {
        throw new RangeError(`${label} player count must be from 4 to 10`);
      }
      if (overrides.has(playerCount)) {
        throw new Error(`${label} repeats player count ${playerCount}`);
      }
      overrides.set(playerCount, count);
    }
  }
  return overrides;
}

export function normalizeRunLabel(value) {
  if (value === undefined || value === "") return "";
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value)) {
    throw new TypeError(
      "run-label must contain lowercase letters, digits, and single hyphens",
    );
  }
  return value;
}

export async function directoryState(path) {
  try {
    const fileStat = await stat(path);
    if (!fileStat.isDirectory()) return "not-directory";
    return (await readdir(path)).length === 0 ? "empty" : "nonempty";
  } catch (error) {
    if (error?.code === "ENOENT") return "missing";
    throw error;
  }
}

export async function assertMissingDirectory(path, label) {
  const state = await directoryState(path);
  if (state !== "missing") {
    throw new Error(`${label} must not already exist: ${path}`);
  }
  return state;
}

export async function createNewDirectory(path, label) {
  await assertMissingDirectory(path, label);
  await mkdir(dirname(path), { recursive: true });
  await mkdir(path);
}

export async function sha256File(path) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

async function readFirstLine(path, maximumBytes = 1024 * 1024) {
  const handle = await open(path, "r");
  try {
    let offset = 0;
    let text = "";
    const buffer = Buffer.alloc(16 * 1024);
    while (offset < maximumBytes) {
      const { bytesRead } = await handle.read(
        buffer,
        0,
        Math.min(buffer.length, maximumBytes - offset),
        offset,
      );
      if (bytesRead === 0) break;
      text += buffer.subarray(0, bytesRead).toString("utf8");
      const newline = text.indexOf("\n");
      if (newline !== -1) return text.slice(0, newline).replace(/\r$/, "");
      offset += bytesRead;
    }
    if (text.length > 0 && offset < maximumBytes) return text.replace(/\r$/, "");
    throw new Error(`${path}: first NDJSON record exceeds ${maximumBytes} bytes`);
  } finally {
    await handle.close();
  }
}

async function readLastNonEmptyLine(path, maximumBytes = 1024 * 1024) {
  const handle = await open(path, "r");
  try {
    const { size } = await handle.stat();
    let length = Math.min(size, 16 * 1024);
    while (length <= Math.min(size, maximumBytes)) {
      const buffer = Buffer.alloc(length);
      await handle.read(buffer, 0, length, size - length);
      const lines = buffer
        .toString("utf8")
        .split(/\r?\n/)
        .filter((line) => line.trim().length > 0);
      if (lines.length >= 2 || length === size) return lines.at(-1) ?? "";
      if (length === Math.min(size, maximumBytes)) break;
      length = Math.min(length * 2, size, maximumBytes);
    }
    throw new Error(`${path}: final NDJSON record exceeds ${maximumBytes} bytes`);
  } finally {
    await handle.close();
  }
}

export async function readRolloutEnvelope(
  path,
  expectedFormat = "dalmuti-ppo-ndjson",
) {
  let manifest;
  let summary;
  try {
    manifest = JSON.parse(await readFirstLine(path));
    summary = JSON.parse(await readLastNonEmptyLine(path));
  } catch (error) {
    throw new Error(`${path}: invalid NDJSON envelope: ${error.message}`, {
      cause: error,
    });
  }
  if (
    manifest.type !== "manifest" ||
    manifest.format !== expectedFormat
  ) {
    throw new Error(`${path}: unsupported rollout manifest`);
  }
  if (summary.type !== "summary") {
    throw new Error(`${path}: final record is not a rollout summary`);
  }
  const learnerSamples = summary.learnerSamples ?? summary.samples;
  const forcedSamples = summary.forcedSamples;
  const nonForcedSamples =
    summary.nonForcedSamples ?? learnerSamples - forcedSamples;
  for (const [label, value] of Object.entries({
    learnerSamples,
    forcedSamples,
    nonForcedSamples,
  })) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`${path}: summary ${label} is invalid`);
    }
  }
  if (forcedSamples + nonForcedSamples !== learnerSamples) {
    throw new Error(`${path}: rollout sample counts do not add up`);
  }
  if (!Number.isSafeInteger(summary.episodes) || summary.episodes < 1) {
    throw new Error(`${path}: summary episodes is invalid`);
  }
  const environmentDecisions =
    summary.environmentDecisions ?? learnerSamples;
  if (
    !Number.isSafeInteger(environmentDecisions) ||
    environmentDecisions < learnerSamples
  ) {
    throw new Error(`${path}: summary environmentDecisions is invalid`);
  }
  if (summary.behaviorModelSha256 !== manifest.behaviorModel?.sha256) {
    throw new Error(`${path}: summary behavior model does not match manifest`);
  }
  return {
    manifest,
    summary,
    counts: {
      episodes: summary.episodes,
      learnerSamples,
      forcedSamples,
      nonForcedSamples,
      environmentDecisions,
    },
  };
}

export function portableRelativePath(root, path) {
  const value = relative(root, path);
  if (!value || value === ".." || value.startsWith(`..${sep}`)) {
    throw new RangeError(`path is not a child of root: ${path}`);
  }
  return value.split(sep).join("/");
}

const CRC_TABLE = Array.from({ length: 256 }, (_, value) => {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) {
    crc = (crc & 1) === 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return crc >>> 0;
});

function updateCrc32(crc, chunk) {
  let value = crc;
  for (const byte of chunk) {
    value = CRC_TABLE[(value ^ byte) & 0xff] ^ (value >>> 8);
  }
  return value >>> 0;
}

function uint16(value) {
  const buffer = Buffer.allocUnsafe(2);
  buffer.writeUInt16LE(value);
  return buffer;
}

function uint32(value) {
  const buffer = Buffer.allocUnsafe(4);
  buffer.writeUInt32LE(value >>> 0);
  return buffer;
}

async function writeChunk(output, chunk) {
  if (!output.write(chunk)) await once(output, "drain");
}

async function listFiles(root, current = root) {
  const result = [];
  const entries = await readdir(current, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name, "en"));
  for (const entry of entries) {
    const path = join(current, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`ZIP input must not contain symbolic links: ${path}`);
    }
    if (entry.isDirectory()) result.push(...(await listFiles(root, path)));
    else if (entry.isFile()) result.push(path);
  }
  return result;
}

/**
 * Create a standard, non-ZIP64 archive without loading large files into memory.
 * Every entry name is UTF-8 and uses '/' regardless of the host OS.
 */
export async function createPortableZip({ sourceDirectory, archivePath }) {
  const sourceRoot = resolve(sourceDirectory);
  const sourceName = basename(sourceRoot);
  const files = await listFiles(sourceRoot);
  if (files.length > 0xffff) throw new RangeError("ZIP has too many entries");
  const output = createWriteStream(archivePath, { flags: "wx" });
  await once(output, "open");
  let position = 0;
  const centralEntries = [];
  try {
    for (const path of files) {
      const relativeName = portableRelativePath(sourceRoot, path);
      const name = Buffer.from(`${sourceName}/${relativeName}`, "utf8");
      const localOffset = position;
      const flags = 0x0808;
      const method = 8;
      const header = Buffer.concat([
        uint32(0x04034b50),
        uint16(20),
        uint16(flags),
        uint16(method),
        uint16(0),
        uint16(0x21),
        uint32(0),
        uint32(0),
        uint32(0),
        uint16(name.length),
        uint16(0),
        name,
      ]);
      await writeChunk(output, header);
      position += header.length;

      let crc = 0xffffffff;
      let uncompressedSize = 0;
      let compressedSize = 0;
      const input = createReadStream(path);
      input.on("data", (chunk) => {
        crc = updateCrc32(crc, chunk);
        uncompressedSize += chunk.length;
      });
      const compressed = input.pipe(createDeflateRaw({ level: 6 }));
      for await (const chunk of compressed) {
        await writeChunk(output, chunk);
        compressedSize += chunk.length;
      }
      crc = (crc ^ 0xffffffff) >>> 0;
      if (
        compressedSize > 0xffffffff ||
        uncompressedSize > 0xffffffff ||
        position + compressedSize > 0xffffffff
      ) {
        throw new RangeError("ZIP64 archives are not supported");
      }
      position += compressedSize;
      const descriptor = Buffer.concat([
        uint32(0x08074b50),
        uint32(crc),
        uint32(compressedSize),
        uint32(uncompressedSize),
      ]);
      await writeChunk(output, descriptor);
      position += descriptor.length;
      centralEntries.push({
        name,
        flags,
        method,
        crc,
        compressedSize,
        uncompressedSize,
        localOffset,
      });
    }

    const centralOffset = position;
    for (const entry of centralEntries) {
      const header = Buffer.concat([
        uint32(0x02014b50),
        uint16(0x0314),
        uint16(20),
        uint16(entry.flags),
        uint16(entry.method),
        uint16(0),
        uint16(0x21),
        uint32(entry.crc),
        uint32(entry.compressedSize),
        uint32(entry.uncompressedSize),
        uint16(entry.name.length),
        uint16(0),
        uint16(0),
        uint16(0),
        uint16(0),
        uint32(0),
        uint32(entry.localOffset),
        entry.name,
      ]);
      await writeChunk(output, header);
      position += header.length;
    }
    const centralSize = position - centralOffset;
    const end = Buffer.concat([
      uint32(0x06054b50),
      uint16(0),
      uint16(0),
      uint16(centralEntries.length),
      uint16(centralEntries.length),
      uint32(centralSize),
      uint32(centralOffset),
      uint16(0),
    ]);
    await writeChunk(output, end);
    position += end.length;
    output.end();
    await once(output, "finish");
  } catch (error) {
    output.destroy();
    throw error;
  }
  return {
    path: resolve(archivePath),
    entries: centralEntries.map((entry) => entry.name.toString("utf8")),
    bytes: position,
  };
}
