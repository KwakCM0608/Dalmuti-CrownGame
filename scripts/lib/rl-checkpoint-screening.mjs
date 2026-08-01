import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  writeFile,
} from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { mergeBenchmarkShardReports } from "./rl-benchmark-shard-merge.mjs";

const BENCHMARK_PATH = fileURLToPath(
  new URL("../rl-benchmark-model.mjs", import.meta.url),
);

export const DEFAULT_SCREENING_SEED_BASE = 1_600_001;
export const DEFAULT_SCREENING_SEED_STRIDE = 11_000_003;
export const MAX_SCREENING_CONCURRENCY = 32;

const CHECKPOINT_MODEL_FAMILIES = Object.freeze([
  Object.freeze({
    id: "legacy",
    filename: "actor-critic-weights.json",
    format: "dalmuti-actor-critic",
    version: 1,
  }),
  Object.freeze({
    id: "v3",
    filename: "v3-actor-critic-weights.json",
    format: "dalmuti-action-conditioned-actor-critic",
    version: 1,
  }),
]);

function portablePath(path) {
  return path.split(sep).join("/");
}

export function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

export function finiteNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new RangeError(`${label} must be a finite number`);
  }
  return parsed;
}

export function parseIntegerList(value, label) {
  if (typeof value !== "string" || value.trim() === "") return [];
  const parsed = value
    .split(",")
    .map((item) => positiveInteger(item.trim(), label));
  if (new Set(parsed).size !== parsed.length) {
    throw new RangeError(`${label} must not contain duplicates`);
  }
  return parsed;
}

export function parsePlayerCounts(value) {
  const playerCounts = parseIntegerList(value, "players");
  if (
    playerCounts.length < 1 ||
    playerCounts.some((playerCount) => playerCount < 4 || playerCount > 10)
  ) {
    throw new RangeError("players must be unique counts from 4 to 10");
  }
  return playerCounts;
}

export function screeningConcurrency(value) {
  const parsed = positiveInteger(value, "concurrency");
  if (parsed > MAX_SCREENING_CONCURRENCY) {
    throw new RangeError(
      `concurrency must not exceed ${MAX_SCREENING_CONCURRENCY}`,
    );
  }
  return parsed;
}

export async function deterministicConcurrentMap(
  items,
  concurrencyValue,
  task,
) {
  if (!Array.isArray(items)) {
    throw new TypeError("concurrent map items must be an array");
  }
  if (typeof task !== "function") {
    throw new TypeError("concurrent map task must be a function");
  }
  if (items.length === 0) return [];
  const concurrency = Math.min(
    screeningConcurrency(concurrencyValue),
    items.length,
  );
  const results = new Array(items.length);
  let nextIndex = 0;
  let failure = null;

  async function worker() {
    while (failure === null) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= items.length) return;
      try {
        results[index] = await task(items[index], index);
      } catch (error) {
        if (failure === null || index < failure.index) {
          failure = { index, error };
        }
      }
    }
  }

  await Promise.all(
    Array.from({ length: concurrency }, () => worker()),
  );
  if (failure !== null) throw failure.error;
  return results;
}

async function pathExists(path) {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

async function readModelSource(directory, source, family) {
  let bytes;
  try {
    bytes = await readFile(source.absolutePath);
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new Error(`checkpoint model is missing: ${source.relativePath}`);
    }
    throw error;
  }
  let model;
  try {
    model = JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new SyntaxError(
      `checkpoint model is not valid JSON: ${source.relativePath}`,
      { cause: error },
    );
  }
  if (model?.format !== family.format || model.version !== family.version) {
    throw new TypeError(
      `${family.filename} must contain ${family.format} version ${family.version}; ` +
        `refusing ambiguous renamed model: ${source.relativePath}`,
    );
  }
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  return {
    ...source,
    absolutePath: resolve(source.absolutePath),
    relativePath: portablePath(relative(directory, source.absolutePath)),
    bytes: bytes.length,
    sha256,
    model: {
      format: model.format,
      version: model.version,
      observationFeatures: model.observationFeatures,
      actionCount: model.actionCount,
      ...(family.id === "legacy"
        ? { hiddenSizes: model.hiddenSizes }
        : {
            observationSchemaVersion: model.observationSchemaVersion,
            actionCatalogueVersion: model.actionCatalogueVersion,
            actionFeatures: model.actionFeatures,
            actionFeatureLayout: model.actionFeatureLayout,
            actorObservationHiddenSizes: model.actorObservationHiddenSizes,
            actorActionHiddenSizes: model.actorActionHiddenSizes,
            actorScorerHiddenSizes: model.actorScorerHiddenSizes,
            valueHiddenSizes: model.valueHiddenSizes,
          }),
    },
  };
}

async function selectCheckpointModelFamily(directory, label, expectedFamily) {
  const presentFamilies = [];
  for (const family of CHECKPOINT_MODEL_FAMILIES) {
    if (await pathExists(join(directory, family.filename))) {
      presentFamilies.push(family);
    }
  }
  if (presentFamilies.length > 1) {
    throw new Error(
      `${label} contains both legacy and V3 checkpoint filenames; ` +
        "checkpoint model family is ambiguous",
    );
  }
  if (presentFamilies.length === 0) {
    throw new Error(
      `${label} is missing ${CHECKPOINT_MODEL_FAMILIES.map(({ filename }) => filename).join(" or ")}`,
    );
  }
  const family = presentFamilies[0];
  if (expectedFamily && family.id !== expectedFamily.id) {
    throw new Error(
      `${label} uses ${family.filename}, but the returned result uses ` +
        `${expectedFamily.filename}; mixed checkpoint model families are not allowed`,
    );
  }
  return family;
}

function sourceSortKey(source) {
  return source.kind === "epoch" ? source.epoch : Number.MAX_SAFE_INTEGER;
}

function createUniqueCandidates(sources) {
  const byHash = new Map();
  for (const source of sources) {
    const existing = byHash.get(source.sha256);
    if (existing) {
      existing.sources.push(source);
    } else {
      byHash.set(source.sha256, {
        sha256: source.sha256,
        bytes: source.bytes,
        model: source.model,
        sources: [source],
      });
    }
  }
  const candidates = [...byHash.values()].map((candidate) => {
    candidate.sources.sort(
      (left, right) => sourceSortKey(left) - sourceSortKey(right),
    );
    const finalSource = candidate.sources.find(
      (source) => source.kind === "final",
    );
    const canonicalSource = finalSource ?? candidate.sources[0];
    return {
      ...candidate,
      id: finalSource ? "final" : candidate.sources[0].label,
      canonicalPath: canonicalSource.absolutePath,
      canonicalRelativePath: canonicalSource.relativePath,
      labels: candidate.sources.map((source) => source.label),
      sources: candidate.sources.map((source) => ({
        kind: source.kind,
        epoch: source.epoch,
        label: source.label,
        relativePath: source.relativePath,
        bytes: source.bytes,
        sha256: source.sha256,
      })),
    };
  });
  const hashes = candidates.map((candidate) => candidate.sha256);
  if (new Set(hashes).size !== hashes.length) {
    throw new Error("checkpoint hash de-duplication failed");
  }
  return candidates;
}

export async function discoverCheckpointCandidates(directoryValue) {
  const directory = resolve(directoryValue);
  const checkpointDirectory = join(directory, "checkpoints");
  let entries;
  try {
    entries = await readdir(checkpointDirectory, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new Error(`checkpoint directory is missing: ${checkpointDirectory}`);
    }
    throw error;
  }
  const epochs = entries
    .filter((entry) => entry.isDirectory() && /^epoch-\d+$/.test(entry.name))
    .map((entry) => ({
      entry,
      epoch: Number(entry.name.slice("epoch-".length)),
    }))
    .sort((left, right) => left.epoch - right.epoch);
  if (epochs.length < 1) {
    throw new Error("returned PPO result contains no epoch checkpoints");
  }
  if (
    new Set(epochs.map(({ epoch }) => epoch)).size !== epochs.length ||
    epochs.some(({ epoch }) => !Number.isSafeInteger(epoch) || epoch < 1)
  ) {
    throw new Error("returned PPO result contains invalid epoch directories");
  }
  const modelFamily = await selectCheckpointModelFamily(
    directory,
    "returned PPO result",
  );
  const sourceDefinitions = [
    ...epochs.map(({ entry, epoch }) => ({
      kind: "epoch",
      epoch,
      label: `epoch-${String(epoch).padStart(2, "0")}`,
      directory: join(checkpointDirectory, entry.name),
      relativePath: "",
    })),
    {
      kind: "final",
      epoch: null,
      label: "final",
      directory,
      relativePath: "",
    },
  ];
  const sources = [];
  for (const source of sourceDefinitions) {
    const sourceFamily = await selectCheckpointModelFamily(
      source.directory,
      source.kind === "final" ? "returned PPO result" : source.label,
      modelFamily,
    );
    sources.push(await readModelSource(
      directory,
      {
        ...source,
        absolutePath: join(source.directory, sourceFamily.filename),
      },
      sourceFamily,
    ));
  }
  const candidates = createUniqueCandidates(sources);
  return {
    directory,
    sources: sources.map((source) => ({
      kind: source.kind,
      epoch: source.epoch,
      label: source.label,
      relativePath: source.relativePath,
      bytes: source.bytes,
      sha256: source.sha256,
      model: source.model,
    })),
    candidates,
    sourceCount: sources.length,
    uniqueHashCount: candidates.length,
    duplicateSourceCount: sources.length - candidates.length,
  };
}

function createMatchSeedRanges(seed, playerCounts, matchCountsByPlayerCount) {
  return playerCounts.map((playerCount) => {
    const matches = positiveInteger(
      matchCountsByPlayerCount[playerCount],
      `matches for p${playerCount}`,
    );
    const start = seed + playerCount * 1_000_000;
    const end = start + matches - 1;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) {
      throw new RangeError("derived benchmark match seed exceeds safe range");
    }
    return { playerCount, matches, start, end };
  });
}

function rangesOverlap(left, right) {
  return left.start <= right.end && right.start <= left.end;
}

export function buildScreeningSeedSchedule({
  candidateCount,
  playerCounts,
  matchCountsByPlayerCount,
  explicitSeeds = [],
  seedBase = DEFAULT_SCREENING_SEED_BASE,
  seedStride = DEFAULT_SCREENING_SEED_STRIDE,
  reservedFinalSeeds = [],
}) {
  positiveInteger(candidateCount, "candidate count");
  const base = positiveInteger(seedBase, "seed base");
  const stride = positiveInteger(seedStride, "seed stride");
  const reserved = reservedFinalSeeds.map((seed) =>
    positiveInteger(seed, "reserved final seed"),
  );
  if (new Set(reserved).size !== reserved.length) {
    throw new RangeError("reserved final seeds must not contain duplicates");
  }
  const seeds = explicitSeeds.length > 0
    ? explicitSeeds.map((seed) => positiveInteger(seed, "screening seed"))
    : Array.from(
        { length: candidateCount },
        (_, index) => base + index * stride,
      );
  if (seeds.length !== candidateCount) {
    throw new RangeError(
      `screening seed schedule needs ${candidateCount} seeds; got ${seeds.length}`,
    );
  }
  if (
    seeds.some((seed) => !Number.isSafeInteger(seed)) ||
    new Set(seeds).size !== seeds.length
  ) {
    throw new RangeError("screening seeds must be distinct safe integers");
  }
  const reservedSet = new Set(reserved);
  if (seeds.some((seed) => reservedSet.has(seed))) {
    throw new RangeError("screening seeds overlap reserved final seeds");
  }
  const schedule = seeds.map((seed, candidateIndex) => ({
    candidateIndex,
    seed,
    matchSeedRanges: createMatchSeedRanges(
      seed,
      playerCounts,
      matchCountsByPlayerCount,
    ),
  }));
  const screeningRanges = schedule.flatMap((entry) =>
    entry.matchSeedRanges.map((range) => ({
      ...range,
      candidateIndex: entry.candidateIndex,
      baseSeed: entry.seed,
    })),
  );
  for (let index = 0; index < screeningRanges.length; index += 1) {
    for (let otherIndex = index + 1; otherIndex < screeningRanges.length; otherIndex += 1) {
      const left = screeningRanges[index];
      const right = screeningRanges[otherIndex];
      if (rangesOverlap(left, right)) {
        throw new RangeError(
          `screening match seed ranges overlap: candidate ${left.candidateIndex + 1} ` +
            `p${left.playerCount} and candidate ${right.candidateIndex + 1} ` +
            `p${right.playerCount}`,
        );
      }
    }
  }
  for (const reservedSeed of reserved) {
    const reservedRanges = createMatchSeedRanges(
      reservedSeed,
      playerCounts,
      matchCountsByPlayerCount,
    );
    for (const screeningRange of screeningRanges) {
      if (reservedRanges.some((range) => rangesOverlap(screeningRange, range))) {
        throw new RangeError(
          `screening match seeds overlap reserved final seed ${reservedSeed}`,
        );
      }
    }
  }
  return schedule;
}

function metricWithPlayerCount(results, selector) {
  return results.reduce((worst, result) => {
    const value = selector(result);
    if (!Number.isFinite(value)) {
      throw new TypeError(
        `benchmark result for p${result.playerCount} contains a non-finite metric`,
      );
    }
    if (worst === null || value < worst.value) {
      return { playerCount: result.playerCount, value };
    }
    return worst;
  }, null);
}

export function conservativeMetrics(benchmark) {
  if (!Array.isArray(benchmark?.results) || benchmark.results.length < 1) {
    throw new TypeError("benchmark report contains no per-player-count results");
  }
  return {
    promotionPassed: benchmark.results.every(
      (result) => result.effectSizeGate?.passed === true,
    ),
    passedPlayerCounts: benchmark.results.filter(
      (result) => result.effectSizeGate?.passed === true,
    ).length,
    totalPlayerCounts: benchmark.results.length,
    worstLowerConfidenceBound: metricWithPlayerCount(
      benchmark.results,
      (result) => result.meanChipDifference95?.low,
    ),
    worstMeanChipDifference: metricWithPlayerCount(
      benchmark.results,
      (result) => result.meanChipDifference,
    ),
    worstPairwiseRate: metricWithPlayerCount(
      benchmark.results,
      (result) => result.pairwiseCandidateBeforeNormal?.rate,
    ),
  };
}

function compareDescending(left, right) {
  if (left === right) return 0;
  return left > right ? -1 : 1;
}

export function rankScreeningResults(entries) {
  const ranked = entries.map((entry) => ({
    ...entry,
    conservative: conservativeMetrics(entry.benchmark),
  }));
  ranked.sort((left, right) => {
    const leftMetrics = left.conservative;
    const rightMetrics = right.conservative;
    return (
      compareDescending(
        Number(leftMetrics.promotionPassed),
        Number(rightMetrics.promotionPassed),
      ) ||
      compareDescending(
        leftMetrics.worstLowerConfidenceBound.value,
        rightMetrics.worstLowerConfidenceBound.value,
      ) ||
      compareDescending(
        leftMetrics.worstMeanChipDifference.value,
        rightMetrics.worstMeanChipDifference.value,
      ) ||
      compareDescending(
        leftMetrics.worstPairwiseRate.value,
        rightMetrics.worstPairwiseRate.value,
      ) ||
      left.candidate.sha256.localeCompare(right.candidate.sha256)
    );
  });
  return ranked.map((entry, index) => ({ rank: index + 1, ...entry }));
}

export function buildBenchmarkArguments({
  candidate,
  reportPath,
  playerCounts,
  matches,
  matchCountsByPlayerCount,
  acts,
  seed,
  thresholds,
  shardIndex = 0,
  shardCount = 1,
}) {
  const matchCounts = playerCounts
    .map((playerCount) => `${playerCount}:${matchCountsByPlayerCount[playerCount]}`)
    .join(",");
  const args = [
    BENCHMARK_PATH,
    "--model",
    candidate.canonicalPath,
    "--matches",
    String(matches),
    "--match-counts",
    matchCounts,
    "--acts",
    String(acts),
    "--players",
    playerCounts.join(","),
    "--seed",
    String(seed),
    "--min-point-diff",
    String(thresholds.minPointDifference),
    "--min-lower-bound",
    String(thresholds.minLowerBound),
    "--min-pairwise-rate",
    String(thresholds.minPairwiseRate),
    "--omit-match-data",
    "--output",
    reportPath,
  ];
  if (shardCount > 1) {
    args.push(
      "--shard-index",
      String(shardIndex),
      "--shard-count",
      String(shardCount),
    );
  }
  return args;
}

export function runBenchmarkProcess({
  args,
  cwd,
  onStdout,
  onStderr,
}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      onStdout?.(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      onStderr?.(chunk);
    });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      resolvePromise({ code, signal, stdout, stderr });
    });
  });
}

async function verifyCandidateHash(candidate) {
  const content = await readFile(candidate.canonicalPath);
  const sha256 = createHash("sha256").update(content).digest("hex");
  if (sha256 !== candidate.sha256) {
    throw new Error(
      `checkpoint changed during screening: ${candidate.canonicalRelativePath}`,
    );
  }
}

function validateBenchmarkReport(report, expected) {
  if (report?.format !== "dalmuti-model-benchmark" || report.version !== 2) {
    throw new TypeError("screening benchmark returned an unsupported report");
  }
  if (resolve(report.modelPath) !== resolve(expected.candidate.canonicalPath)) {
    throw new Error("screening benchmark model path does not match its candidate");
  }
  if (report.modelSha256 !== expected.candidate.sha256) {
    throw new Error("screening benchmark model SHA-256 does not match its candidate");
  }
  if (report.seed !== expected.seed) {
    throw new Error("screening benchmark seed does not match its schedule");
  }
  if (report.evaluationDesign?.finalMatchCountPreset !== false) {
    throw new Error("checkpoint screening must never use the final preset");
  }
  if (
    !Array.isArray(report.playerCounts) ||
    report.playerCounts.join(",") !== expected.playerCounts.join(",")
  ) {
    throw new Error("screening benchmark player counts do not match its plan");
  }
  conservativeMetrics(report);
}

async function writeJsonExclusive(path, value) {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    flag: "wx",
  });
}

export async function screenCheckpointDirectory({
  directory: directoryValue,
  output: outputValue,
  playerCounts,
  matches,
  matchCountsByPlayerCount,
  acts,
  concurrency = 1,
  benchmarkShards = 1,
  explicitSeeds = [],
  seedBase = DEFAULT_SCREENING_SEED_BASE,
  seedStride = DEFAULT_SCREENING_SEED_STRIDE,
  reservedFinalSeeds = [],
  thresholds = {
    minPointDifference: 0.25,
    minLowerBound: 0.15,
    minPairwiseRate: 0.55,
  },
  processRunner = runBenchmarkProcess,
  onProgress,
  now = () => new Date(),
}) {
  const inputDirectory = resolve(directoryValue);
  const outputDirectory = resolve(outputValue);
  const parsedMatches = positiveInteger(matches, "matches");
  const parsedActs = positiveInteger(acts, "acts");
  const parsedConcurrency = screeningConcurrency(concurrency);
  const parsedBenchmarkShards = positiveInteger(
    benchmarkShards,
    "benchmark shards",
  );
  if (parsedBenchmarkShards > MAX_SCREENING_CONCURRENCY) {
    throw new RangeError(
      `benchmark shards must not exceed ${MAX_SCREENING_CONCURRENCY}`,
    );
  }
  if (
    parsedBenchmarkShards > 1 &&
    playerCounts.some(
      (playerCount) =>
        matchCountsByPlayerCount[playerCount] < parsedBenchmarkShards,
    )
  ) {
    throw new RangeError(
      "benchmark shards cannot exceed any player-count match count",
    );
  }
  if (
    outputDirectory === inputDirectory ||
    outputDirectory.startsWith(`${inputDirectory}${sep}`)
  ) {
    throw new RangeError("screening output must be outside the returned result directory");
  }
  const discovery = await discoverCheckpointCandidates(inputDirectory);
  const schedule = buildScreeningSeedSchedule({
    candidateCount: discovery.candidates.length,
    playerCounts,
    matchCountsByPlayerCount,
    explicitSeeds,
    seedBase,
    seedStride,
    reservedFinalSeeds,
  });
  if (await pathExists(outputDirectory)) {
    throw new Error(`screening output must not already exist: ${outputDirectory}`);
  }
  await mkdir(dirname(outputDirectory), { recursive: true });
  try {
    await mkdir(outputDirectory);
  } catch (error) {
    if (error?.code === "EEXIST") {
      throw new Error(`screening output must not already exist: ${outputDirectory}`);
    }
    throw error;
  }
  const benchmarkDirectory = join(outputDirectory, "benchmarks");
  await mkdir(benchmarkDirectory);
  const manifest = {
    format: "dalmuti-checkpoint-screening-manifest",
    version: 1,
    createdAt: now().toISOString(),
    inputDirectory,
    outputDirectory,
    discovery: {
      sourceCount: discovery.sourceCount,
      uniqueHashCount: discovery.uniqueHashCount,
      duplicateSourceCount: discovery.duplicateSourceCount,
      sources: discovery.sources,
      candidates: discovery.candidates.map((candidate) => ({
        id: candidate.id,
        sha256: candidate.sha256,
        bytes: candidate.bytes,
        model: candidate.model,
        canonicalRelativePath: candidate.canonicalRelativePath,
        labels: candidate.labels,
        sources: candidate.sources,
      })),
    },
    evaluation: {
      purpose: "development-checkpoint-screening",
      finalPresetUsed: false,
      playerCounts,
      matches: parsedMatches,
      matchCountsByPlayerCount,
      acts: parsedActs,
      concurrency: parsedConcurrency,
      benchmarkShards: parsedBenchmarkShards,
      thresholds,
      seedDerivation:
        explicitSeeds.length > 0
          ? "explicit per-unique-checkpoint screening seeds"
          : "seedBase + uniqueCheckpointIndex * seedStride",
      seedBase,
      seedStride,
      seedSchedule: schedule,
      reservedFinalSeeds,
      finalEvaluationPolicy:
        "Every screening base seed and derived match seed is forbidden in final evaluation.",
      rankingRule:
        "gate pass, then worst player-count 95% lower bound, mean chip difference, " +
        "and candidate-before-normal pairwise rate; all descending",
    },
  };
  const manifestPath = join(outputDirectory, "screening-manifest.json");
  await writeJsonExclusive(manifestPath, manifest);

  let completed;
  if (parsedBenchmarkShards === 1) {
    completed = await deterministicConcurrentMap(
      discovery.candidates,
      parsedConcurrency,
      async (candidate, index) => {
      const scheduleEntry = schedule[index];
      const safeId = candidate.id.replace(/[^a-zA-Z0-9._-]+/g, "-");
      const reportPath = join(
        benchmarkDirectory,
        `${String(index + 1).padStart(2, "0")}-${safeId}-${candidate.sha256.slice(0, 12)}.json`,
      );
      await verifyCandidateHash(candidate);
      const args = buildBenchmarkArguments({
        candidate,
        reportPath,
        playerCounts,
        matches: parsedMatches,
        matchCountsByPlayerCount,
        acts: parsedActs,
        seed: scheduleEntry.seed,
        thresholds,
      });
      if (args.includes("--final")) {
        throw new Error(
          "checkpoint screening attempted to use the final preset",
        );
      }
      onProgress?.({
        type: "candidate-start",
        index,
        total: discovery.candidates.length,
        candidate,
        seed: scheduleEntry.seed,
      });
      const execution = await processRunner({
        args,
        cwd: resolve(dirname(BENCHMARK_PATH), ".."),
        reportPath,
        candidate,
        seed: scheduleEntry.seed,
      });
      if (execution?.code !== 0) {
        throw new Error(
          `checkpoint benchmark failed for ${candidate.id} ` +
            `(code ${execution?.code ?? "unknown"}): ${execution?.stderr ?? ""}`,
        );
      }
      await verifyCandidateHash(candidate);
      const benchmark = JSON.parse(await readFile(reportPath, "utf8"));
      validateBenchmarkReport(benchmark, {
        candidate,
        seed: scheduleEntry.seed,
        playerCounts,
      });
      const completedEntry = {
        candidate: {
          id: candidate.id,
          sha256: candidate.sha256,
          bytes: candidate.bytes,
          canonicalRelativePath: candidate.canonicalRelativePath,
          labels: candidate.labels,
          sources: candidate.sources,
        },
        seed: scheduleEntry.seed,
        benchmarkReport: portablePath(
          relative(outputDirectory, reportPath),
        ),
        benchmark,
      };
      onProgress?.({
        type: "candidate-complete",
        index,
        total: discovery.candidates.length,
        candidate,
        seed: scheduleEntry.seed,
      });
        return completedEntry;
      },
    );
  } else {
    const shardDirectory = join(benchmarkDirectory, "shards");
    await mkdir(shardDirectory);
    const jobs = discovery.candidates.flatMap((candidate, candidateIndex) =>
      playerCounts.flatMap((playerCount) =>
        Array.from({ length: parsedBenchmarkShards }, (_, shardIndex) => ({
          candidate,
          candidateIndex,
          playerCount,
          shardIndex,
        })),
      ),
    );
    for (const [index, candidate] of discovery.candidates.entries()) {
      onProgress?.({
        type: "candidate-start",
        index,
        total: discovery.candidates.length,
        candidate,
        seed: schedule[index].seed,
      });
    }
    const shardResults = await deterministicConcurrentMap(
      jobs,
      parsedConcurrency,
      async (job) => {
        const { candidate, candidateIndex, playerCount, shardIndex } = job;
        const scheduleEntry = schedule[candidateIndex];
        const safeId = candidate.id.replace(/[^a-zA-Z0-9._-]+/g, "-");
        const reportPath = join(
          shardDirectory,
          `${String(candidateIndex + 1).padStart(2, "0")}-${safeId}-` +
            `${candidate.sha256.slice(0, 12)}-p${playerCount}-` +
            `s${shardIndex + 1}-of-${parsedBenchmarkShards}.json`,
        );
        await verifyCandidateHash(candidate);
        const args = buildBenchmarkArguments({
          candidate,
          reportPath,
          playerCounts: [playerCount],
          matches: parsedMatches,
          matchCountsByPlayerCount,
          acts: parsedActs,
          seed: scheduleEntry.seed,
          thresholds,
          shardIndex,
          shardCount: parsedBenchmarkShards,
        });
        const execution = await processRunner({
          args,
          cwd: resolve(dirname(BENCHMARK_PATH), ".."),
          reportPath,
          candidate,
          candidateIndex,
          playerCount,
          shardIndex,
          shardCount: parsedBenchmarkShards,
          seed: scheduleEntry.seed,
        });
        if (execution?.code !== 0) {
          throw new Error(
            `checkpoint benchmark shard failed for ${candidate.id} p${playerCount} ` +
              `${shardIndex + 1}/${parsedBenchmarkShards} ` +
              `(code ${execution?.code ?? "unknown"}): ${execution?.stderr ?? ""}`,
          );
        }
        await verifyCandidateHash(candidate);
        const bytes = await readFile(reportPath);
        return {
          candidateIndex,
          path: reportPath,
          bytes: bytes.length,
          sha256: createHash("sha256").update(bytes).digest("hex"),
          report: JSON.parse(bytes.toString("utf8")),
        };
      },
    );
    completed = [];
    for (const [index, candidate] of discovery.candidates.entries()) {
      await verifyCandidateHash(candidate);
      const scheduleEntry = schedule[index];
      const safeId = candidate.id.replace(/[^a-zA-Z0-9._-]+/g, "-");
      const reportPath = join(
        benchmarkDirectory,
        `${String(index + 1).padStart(2, "0")}-${safeId}-${candidate.sha256.slice(0, 12)}.json`,
      );
      const candidateShards = shardResults
        .filter((entry) => entry.candidateIndex === index)
        .map((entry) => ({
          ...entry,
          path: portablePath(relative(outputDirectory, entry.path)),
        }));
      const benchmark = mergeBenchmarkShardReports(candidateShards, {
        modelPath: candidate.canonicalPath,
        modelSha256: candidate.sha256,
        playerCounts,
        matchCountsByPlayerCount,
        acts: parsedActs,
        seed: scheduleEntry.seed,
        promotionThresholds: thresholds,
        roleRegressionMargin: 0.1,
      });
      await writeJsonExclusive(reportPath, benchmark);
      await verifyCandidateHash(candidate);
      validateBenchmarkReport(benchmark, {
        candidate,
        seed: scheduleEntry.seed,
        playerCounts,
      });
      const reportBytes = await readFile(reportPath);
      completed.push({
        candidate: {
          id: candidate.id,
          sha256: candidate.sha256,
          bytes: candidate.bytes,
          canonicalRelativePath: candidate.canonicalRelativePath,
          labels: candidate.labels,
          sources: candidate.sources,
        },
        seed: scheduleEntry.seed,
        benchmarkReport: portablePath(relative(outputDirectory, reportPath)),
        benchmarkReportSha256: createHash("sha256")
          .update(reportBytes)
          .digest("hex"),
        benchmark,
      });
      onProgress?.({
        type: "candidate-complete",
        index,
        total: discovery.candidates.length,
        candidate,
        seed: scheduleEntry.seed,
      });
    }
  }
  const ranking = rankScreeningResults(completed);
  const report = {
    format: "dalmuti-checkpoint-screening-report",
    version: 1,
    completedAt: now().toISOString(),
    manifest: "screening-manifest.json",
    inputDirectory,
    candidateSourceCount: discovery.sourceCount,
    uniqueCandidateCount: discovery.uniqueHashCount,
    duplicateSourcesSkipped: discovery.duplicateSourceCount,
    finalEvaluationSeedsForbidden: schedule.map((entry) => entry.seed),
    winner: ranking[0]
      ? {
          id: ranking[0].candidate.id,
          sha256: ranking[0].candidate.sha256,
          canonicalRelativePath: ranking[0].candidate.canonicalRelativePath,
          labels: ranking[0].candidate.labels,
          conservative: ranking[0].conservative,
        }
      : null,
    ranking,
  };
  const reportPath = join(outputDirectory, "screening-report.json");
  await writeJsonExclusive(reportPath, report);
  return { manifestPath, reportPath, manifest, report };
}
