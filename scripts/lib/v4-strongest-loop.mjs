import { createHash, randomUUID } from "node:crypto";
import {
  link,
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  rm,
} from "node:fs/promises";
import { basename, join, resolve } from "node:path";

export const V4_PLAYER_COUNTS = Object.freeze([4, 5, 6, 7, 8, 9, 10]);
export const V4_FINAL_ACTS_PER_MATCH = 5;
export const V4_FINAL_MATCH_COUNTS = Object.freeze({
  4: 2500,
  5: 1700,
  6: 900,
  7: 600,
  8: 400,
  9: 400,
  10: 300,
});
export const V4_DEVELOPMENT_GATES = Object.freeze({
  minMeanChipDifference: 0.3,
  minLowerConfidenceBound: 0.2,
  minPairwiseRate: 0.57,
});
export const V4_FINAL_GATES = Object.freeze({
  minMeanChipDifference: 0.25,
  minLowerConfidenceBound: 0.15,
  minPairwiseRate: 0.55,
});
export const V4_FINAL_SEED_START = 900_000_001;
export const V4_FINAL_SEED_STEP = 20_000_000;

const MAX_UINT32 = 0xffff_ffff;
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const GIT_COMMIT_PATTERN = /^[a-f0-9]{40}$/;
const ATTEMPT_LABEL_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const ATTEMPT_STATES = Object.freeze({
  created: Object.freeze([
    "data-generation",
    "training",
    "development-evaluation",
    "paused",
    "failed",
  ]),
  "data-generation": Object.freeze(["training", "paused", "failed"]),
  training: Object.freeze([
    "development-evaluation",
    "paused",
    "failed",
  ]),
  "development-evaluation": Object.freeze([
    "ready-for-final",
    "rejected",
    "paused",
    "failed",
  ]),
  "ready-for-final": Object.freeze([
    "final-evaluation",
    "paused",
    "failed",
  ]),
  "final-evaluation": Object.freeze([
    "passed",
    "rejected",
    "paused",
    "failed",
  ]),
  paused: Object.freeze([]),
  failed: Object.freeze([]),
  rejected: Object.freeze([]),
  passed: Object.freeze([]),
});

export const V4_FAILURE_INTERVENTIONS = Object.freeze({
  "mean-chip-gap": Object.freeze({
    action: "increase-weak-player-count-search-and-rollouts",
    retry: "fresh-attempt",
    feedbackAllowed: true,
  }),
  "confidence-gap": Object.freeze({
    action: "increase-independent-rollouts-and-variance-control",
    retry: "fresh-attempt",
    feedbackAllowed: true,
  }),
  "pairwise-gap": Object.freeze({
    action: "increase-endgame-and-immediate-finish-curriculum",
    retry: "fresh-attempt",
    feedbackAllowed: true,
  }),
  "role-regression": Object.freeze({
    action: "rebalance-role-and-seat-curriculum",
    retry: "fresh-attempt",
    feedbackAllowed: true,
  }),
  plateau: Object.freeze({
    action: "escalate-search-exploiter-ensemble-then-model-capacity",
    retry: "fresh-attempt",
    feedbackAllowed: true,
  }),
  oom: Object.freeze({
    action: "reduce-microbatch-and-increase-gradient-accumulation",
    retry: "fresh-attempt",
    feedbackAllowed: true,
  }),
  "transient-infrastructure": Object.freeze({
    action: "retry-from-last-verified-checkpoint",
    retry: "fresh-attempt",
    feedbackAllowed: false,
  }),
  "invalid-artifact": Object.freeze({
    action: "quarantine-artifact-and-rebuild-from-verified-inputs",
    retry: "fresh-attempt",
    feedbackAllowed: false,
  }),
  integrity: Object.freeze({
    action: "quarantine-attempt-and-run-contract-audit",
    retry: "fresh-attempt",
    feedbackAllowed: false,
  }),
  disk: Object.freeze({
    action: "pause-and-prune-only-locally-verified-artifacts",
    retry: "manual-resume-with-fresh-attempt",
    feedbackAllowed: false,
  }),
  thermal: Object.freeze({
    action: "pause-for-cooling-and-hardware-check",
    retry: "manual-resume-with-fresh-attempt",
    feedbackAllowed: false,
  }),
  "sealed-final-failure": Object.freeze({
    action: "train-a-fresh-candidate-without-final-metric-feedback",
    retry: "fresh-attempt",
    feedbackAllowed: false,
  }),
});

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

export function sha256Value(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function sha256Bytes(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

function finiteNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new RangeError(`${label} must be finite`);
  }
  return parsed;
}

function normalizeSha256(value, label) {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    throw new TypeError(`${label} must be a lowercase SHA-256 digest`);
  }
  return value;
}

function normalizeCommit(value, label) {
  if (typeof value !== "string" || !GIT_COMMIT_PATTERN.test(value)) {
    throw new TypeError(`${label} must be a full lowercase Git commit`);
  }
  return value;
}

function isoTimestamp(value, label = "timestamp") {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    throw new TypeError(`${label} must be a valid date`);
  }
  return date.toISOString();
}

export function normalizeV4Bindings(value) {
  if (value === null || typeof value !== "object") {
    throw new TypeError("bindings must be an object");
  }
  const bindings = {
    artifactSha256: normalizeSha256(
      value.artifactSha256,
      "artifact SHA-256",
    ),
    modelSha256: normalizeSha256(value.modelSha256, "model SHA-256"),
    observationSchemaSha256: normalizeSha256(
      value.observationSchemaSha256,
      "observation schema SHA-256",
    ),
    normalBaselineSha256: normalizeSha256(
      value.normalBaselineSha256,
      "Normal baseline SHA-256",
    ),
    normalBaselineSourceCommit: normalizeCommit(
      value.normalBaselineSourceCommit,
      "Normal baseline source commit",
    ),
  };
  return Object.freeze(bindings);
}

export function bindingSha256(bindings) {
  return sha256Value(normalizeV4Bindings(bindings));
}

export function buildV4AttemptId(attemptNumberValue, label = "strongest") {
  const attemptNumber = positiveInteger(attemptNumberValue, "attempt number");
  if (!ATTEMPT_LABEL_PATTERN.test(label)) {
    throw new TypeError(
      "attempt label must contain lowercase letters, digits, and single hyphens",
    );
  }
  return `v4-strongest-attempt-${String(attemptNumber).padStart(3, "0")}-${label}`;
}

export function buildV4AttemptPlan({
  root,
  attemptNumber,
  label = "strongest",
  bindings: bindingValue,
}) {
  if (typeof root !== "string" || root.trim() === "") {
    throw new TypeError("root is required");
  }
  const bindings = normalizeV4Bindings(bindingValue);
  const attemptId = buildV4AttemptId(attemptNumber, label);
  const rootDirectory = resolve(root);
  return {
    format: "dalmuti-v4-strongest-attempt-plan",
    version: 1,
    attemptId,
    attemptNumber: positiveInteger(attemptNumber, "attempt number"),
    rootDirectory,
    attemptDirectory: join(rootDirectory, attemptId),
    immutableAttemptDirectory: true,
    pathReusePolicy: "never-reuse-even-after-failure",
    bindings,
    bindingSha256: bindingSha256(bindings),
    evaluationContract: {
      playerCounts: [...V4_PLAYER_COUNTS],
      actsPerMatch: V4_FINAL_ACTS_PER_MATCH,
      developmentGates: { ...V4_DEVELOPMENT_GATES },
      finalGates: { ...V4_FINAL_GATES },
      finalMatchCounts: { ...V4_FINAL_MATCH_COUNTS },
      finalFeedbackPolicy: "sealed-holdout-not-a-training-input",
    },
    finalSeedContract: {
      firstBaseSeed: V4_FINAL_SEED_START,
      secondBaseSeed: V4_FINAL_SEED_START + V4_FINAL_SEED_STEP,
      stride: V4_FINAL_SEED_STEP,
      reuse: "forbidden",
      collisionPolicy: "skip-candidate-and-consume-only-atomic-reservations",
    },
    continuationPolicy:
      "a failed model creates a new attempt directory; it never ends the loop",
    prohibitedTrainingInputs: [
      "sealed final evaluation metrics",
      "opponent hidden hands in the deployed actor",
    ],
    outOfScope: [
      "quick-match integration",
      "online-mode integration",
      "PWA or Android changes",
      "Sites deployment",
    ],
  };
}

async function exists(path) {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

async function publishJsonExclusive(path, value) {
  const directory = resolve(join(path, ".."));
  await mkdir(directory, { recursive: true });
  const temporaryPath = join(
    directory,
    `.${basename(path)}.${process.pid}.${randomUUID()}.partial`,
  );
  const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`, "utf8");
  let handle;
  try {
    handle = await open(temporaryPath, "wx");
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.close();
    handle = undefined;
    await link(temporaryPath, path);
  } finally {
    await handle?.close().catch(() => {});
    await rm(temporaryPath, { force: true }).catch(() => {});
  }
  return { path: resolve(path), bytes: bytes.length, sha256: sha256Bytes(bytes) };
}

async function readJson(path, label) {
  let bytes;
  try {
    bytes = await readFile(path);
  } catch (error) {
    if (error?.code === "ENOENT") throw new Error(`${label} is missing: ${path}`);
    throw error;
  }
  let value;
  try {
    value = JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new SyntaxError(`${label} is not valid JSON: ${path}`, { cause: error });
  }
  return { value, bytes, sha256: sha256Bytes(bytes) };
}

function journalFilename(sequence, kind) {
  return `${String(sequence).padStart(6, "0")}-${kind}.json`;
}

async function readJournal(directory, kind) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
  const pattern = new RegExp(`^(\\d{6})-${kind}\\.json$`);
  const files = entries
    .filter((entry) => entry.isFile() && pattern.test(entry.name))
    .map((entry) => ({ name: entry.name, sequence: Number(pattern.exec(entry.name)[1]) }))
    .sort((left, right) => left.sequence - right.sequence);
  const records = [];
  let previousSha256 = null;
  for (const [index, file] of files.entries()) {
    if (file.sequence !== index + 1) {
      throw new Error(`${kind} journal has a sequence gap at ${file.name}`);
    }
    const parsed = await readJson(join(directory, file.name), `${kind} record`);
    if (
      parsed.value.sequence !== file.sequence ||
      parsed.value.previousRecordSha256 !== previousSha256
    ) {
      throw new Error(`${kind} journal chain is invalid at ${file.name}`);
    }
    records.push({ ...parsed, path: join(directory, file.name) });
    previousSha256 = parsed.sha256;
  }
  return records;
}

async function appendJournalRecord(directory, kind, value) {
  const records = await readJournal(directory, kind);
  const previous = records.at(-1) ?? null;
  const sequence = records.length + 1;
  const record = {
    ...value,
    sequence,
    previousRecordSha256: previous?.sha256 ?? null,
  };
  return publishJsonExclusive(
    join(directory, journalFilename(sequence, kind)),
    record,
  );
}

async function readAttemptContext(attemptDirectoryValue) {
  const attemptDirectory = resolve(attemptDirectoryValue);
  const manifest = await readJson(
    join(attemptDirectory, "attempt-manifest.json"),
    "attempt manifest",
  );
  if (
    manifest.value?.format !== "dalmuti-v4-strongest-attempt-manifest" ||
    manifest.value.version !== 1
  ) {
    throw new Error("unsupported V4 attempt manifest");
  }
  if (
    resolve(manifest.value.attemptDirectory) !== attemptDirectory ||
    manifest.value.attemptId !== basename(attemptDirectory)
  ) {
    throw new Error("attempt manifest path identity does not match its directory");
  }
  const bindings = normalizeV4Bindings(manifest.value.bindings);
  if (bindingSha256(bindings) !== manifest.value.bindingSha256) {
    throw new Error("attempt binding SHA-256 is invalid");
  }
  const states = await readJournal(join(attemptDirectory, "state"), "state");
  if (states.length < 1) throw new Error("attempt state journal is empty");
  for (const [index, state] of states.entries()) {
    if (
      state.value.format !== "dalmuti-v4-attempt-state" ||
      state.value.version !== 1 ||
      state.value.manifestSha256 !== manifest.sha256 ||
      state.value.bindingSha256 !== manifest.value.bindingSha256
    ) {
      throw new Error("attempt state is not bound to its manifest");
    }
    const previousState = states[index - 1]?.value.to ?? null;
    if (state.value.from !== previousState) {
      throw new Error("attempt state journal contains a discontinuous transition");
    }
    if (index === 0) {
      if (state.value.to !== "created") {
        throw new Error("attempt state journal must begin in created state");
      }
    } else if (!ATTEMPT_STATES[previousState]?.includes(state.value.to)) {
      throw new Error(
        `attempt state journal contains an invalid transition: ` +
          `${previousState} -> ${state.value.to}`,
      );
    }
  }
  const context = {
    attemptDirectory,
    manifest,
    bindings,
    states,
    currentState: states.at(-1).value.to,
  };
  if (context.currentState === "passed") {
    const declaredEvidence = states.at(-1).value.passingEvidence;
    if (
      declaredEvidence === null ||
      typeof declaredEvidence !== "object" ||
      typeof declaredEvidence.finalSeedRegistryDirectory !== "string"
    ) {
      throw new Error("passed state is missing sealed final evidence");
    }
    const verifiedEvidence = await verifyStoredPassingFinalEvidence({
      context,
      registryDirectory: declaredEvidence.finalSeedRegistryDirectory,
    });
    if (canonicalJson(declaredEvidence) !== canonicalJson(verifiedEvidence)) {
      throw new Error("passed state final evidence no longer verifies exactly");
    }
  }
  return context;
}

export async function createV4Attempt(planValue, { now = new Date() } = {}) {
  const expected = buildV4AttemptPlan({
    root: planValue.rootDirectory,
    attemptNumber: planValue.attemptNumber,
    label: planValue.attemptId.replace(
      /^v4-strongest-attempt-\d+-/,
      "",
    ),
    bindings: planValue.bindings,
  });
  if (canonicalJson(expected) !== canonicalJson(planValue)) {
    throw new Error("attempt plan is not canonical");
  }
  await mkdir(expected.rootDirectory, { recursive: true });
  try {
    await mkdir(expected.attemptDirectory);
  } catch (error) {
    if (error?.code === "EEXIST") {
      throw new Error(
        `attempt directory must not already exist: ${expected.attemptDirectory}`,
      );
    }
    throw error;
  }
  const createdAt = isoTimestamp(now);
  const manifestValue = {
    ...expected,
    format: "dalmuti-v4-strongest-attempt-manifest",
    createdAt,
  };
  const manifest = await publishJsonExclusive(
    join(expected.attemptDirectory, "attempt-manifest.json"),
    manifestValue,
  );
  const initialState = await appendJournalRecord(
    join(expected.attemptDirectory, "state"),
    "state",
    {
      format: "dalmuti-v4-attempt-state",
      version: 1,
      manifestSha256: manifest.sha256,
      bindingSha256: expected.bindingSha256,
      recordedAt: createdAt,
      from: null,
      to: "created",
      reason: "immutable attempt directory created",
    },
  );
  return {
    attemptId: expected.attemptId,
    attemptDirectory: expected.attemptDirectory,
    manifest,
    initialState,
  };
}

export async function transitionV4Attempt({
  attemptDirectory,
  to,
  reason,
  finalSeedRegistryDirectory = null,
  now = new Date(),
}) {
  const context = await readAttemptContext(attemptDirectory);
  if (await exists(join(context.attemptDirectory, "exit-record.json"))) {
    throw new Error("cannot transition an attempt after its exit record");
  }
  const allowed = ATTEMPT_STATES[context.currentState];
  if (!allowed || !allowed.includes(to)) {
    throw new Error(
      `invalid attempt transition: ${context.currentState} -> ${to}`,
    );
  }
  if (typeof reason !== "string" || reason.trim() === "") {
    throw new TypeError("transition reason is required");
  }
  let passingEvidence = null;
  if (to === "passed") {
    if (
      typeof finalSeedRegistryDirectory !== "string" ||
      finalSeedRegistryDirectory.trim() === ""
    ) {
      throw new Error("passed transition requires a final seed registry");
    }
    passingEvidence = await verifyStoredPassingFinalEvidence({
      context,
      registryDirectory: finalSeedRegistryDirectory,
    });
  }
  return appendJournalRecord(
    join(context.attemptDirectory, "state"),
    "state",
    {
      format: "dalmuti-v4-attempt-state",
      version: 1,
      manifestSha256: context.manifest.sha256,
      bindingSha256: context.manifest.value.bindingSha256,
      recordedAt: isoTimestamp(now),
      from: context.currentState,
      to,
      reason: reason.trim(),
      ...(passingEvidence === null ? {} : { passingEvidence }),
    },
  );
}

export async function recordV4Heartbeat({
  attemptDirectory,
  phase,
  progress,
  message = "",
  now = new Date(),
}) {
  const context = await readAttemptContext(attemptDirectory);
  if (await exists(join(context.attemptDirectory, "exit-record.json"))) {
    throw new Error("cannot heartbeat an attempt after its exit record");
  }
  const parsedProgress = finiteNumber(progress, "heartbeat progress");
  if (parsedProgress < 0 || parsedProgress > 1) {
    throw new RangeError("heartbeat progress must be from 0 to 1");
  }
  if (typeof phase !== "string" || phase.trim() === "") {
    throw new TypeError("heartbeat phase is required");
  }
  const state = context.states.at(-1);
  return appendJournalRecord(
    join(context.attemptDirectory, "heartbeats"),
    "heartbeat",
    {
      format: "dalmuti-v4-attempt-heartbeat",
      version: 1,
      manifestSha256: context.manifest.sha256,
      bindingSha256: context.manifest.value.bindingSha256,
      recordedAt: isoTimestamp(now),
      stateSequence: state.value.sequence,
      stateRecordSha256: state.sha256,
      state: context.currentState,
      phase: phase.trim(),
      progress: parsedProgress,
      message: String(message),
    },
  );
}

export function interventionForV4Failure(failureClass) {
  const intervention = V4_FAILURE_INTERVENTIONS[failureClass];
  if (!intervention) {
    throw new RangeError(`unsupported V4 failure class: ${failureClass}`);
  }
  return { failureClass, ...intervention };
}

export async function recordV4Exit({
  attemptDirectory,
  outcome,
  failureClass = null,
  exitCode = null,
  signal = null,
  now = new Date(),
}) {
  const context = await readAttemptContext(attemptDirectory);
  if (!new Set(["passed", "rejected", "failed", "paused"]).has(outcome)) {
    throw new RangeError("exit outcome must be passed, rejected, failed, or paused");
  }
  if (context.currentState !== outcome) {
    throw new Error(
      `exit outcome ${outcome} does not match state ${context.currentState}`,
    );
  }
  if (outcome !== "passed" && !failureClass) {
    throw new TypeError("non-passing exits require a failure class");
  }
  const state = context.states.at(-1);
  const record = {
    format: "dalmuti-v4-attempt-exit",
    version: 1,
    manifestSha256: context.manifest.sha256,
    bindingSha256: context.manifest.value.bindingSha256,
    recordedAt: isoTimestamp(now),
    stateSequence: state.value.sequence,
    stateRecordSha256: state.sha256,
    outcome,
    exitCode,
    signal,
    intervention:
      outcome === "passed" ? null : interventionForV4Failure(failureClass),
    deploymentTriggered: false,
  };
  return publishJsonExclusive(
    join(context.attemptDirectory, "exit-record.json"),
    record,
  );
}

export async function readV4AttemptSnapshot(attemptDirectory) {
  const context = await readAttemptContext(attemptDirectory);
  const heartbeats = await readJournal(
    join(context.attemptDirectory, "heartbeats"),
    "heartbeat",
  );
  for (const heartbeat of heartbeats) {
    const state = context.states[heartbeat.value.stateSequence - 1];
    if (
      heartbeat.value.format !== "dalmuti-v4-attempt-heartbeat" ||
      heartbeat.value.version !== 1 ||
      heartbeat.value.manifestSha256 !== context.manifest.sha256 ||
      heartbeat.value.bindingSha256 !==
        context.manifest.value.bindingSha256 ||
      heartbeat.value.stateRecordSha256 !== state?.sha256 ||
      heartbeat.value.state !== state?.value.to
    ) {
      throw new Error("attempt heartbeat is not bound to a valid state record");
    }
  }
  const exitPath = join(context.attemptDirectory, "exit-record.json");
  const exit = (await exists(exitPath))
    ? await readJson(exitPath, "attempt exit record")
    : null;
  return {
    attemptId: context.manifest.value.attemptId,
    attemptDirectory: context.attemptDirectory,
    bindings: context.bindings,
    bindingSha256: context.manifest.value.bindingSha256,
    currentState: context.currentState,
    stateSequence: context.states.at(-1).value.sequence,
    latestHeartbeat: heartbeats.at(-1)?.value ?? null,
    exit: exit?.value ?? null,
  };
}

export function finalMatchSeedRanges(baseSeedValue) {
  const baseSeed = positiveInteger(baseSeedValue, "final base seed");
  return V4_PLAYER_COUNTS.map((playerCount) => {
    const start = baseSeed + playerCount * 1_000_000;
    const end = start + V4_FINAL_MATCH_COUNTS[playerCount] - 1;
    if (end > MAX_UINT32) {
      throw new RangeError("final match seed exceeds uint32 range");
    }
    return {
      playerCount,
      matches: V4_FINAL_MATCH_COUNTS[playerCount],
      start,
      end,
    };
  });
}

function normalizeCollisionRange(value) {
  if (Number.isSafeInteger(value)) {
    return { start: value, end: value, label: `seed-${value}` };
  }
  if (value === null || typeof value !== "object") {
    throw new TypeError("collision range must be an integer or range object");
  }
  const start = positiveInteger(value.start, "collision start");
  const end = positiveInteger(value.end, "collision end");
  if (end < start) throw new RangeError("collision range end precedes start");
  return { start, end, label: String(value.label ?? `${start}:${end}`) };
}

function overlaps(left, right) {
  return left.start <= right.end && right.start <= left.end;
}

function finalSeedAt(index) {
  const seed = V4_FINAL_SEED_START + index * V4_FINAL_SEED_STEP;
  if (!Number.isSafeInteger(seed)) {
    throw new RangeError("final seed registry exhausted safe integers");
  }
  return seed;
}

export async function listFinalSeedReservations(registryDirectoryValue) {
  const registryDirectory = resolve(registryDirectoryValue);
  let entries;
  try {
    entries = await readdir(registryDirectory, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
  const pattern = /^seed-(\d+)-reservation\.json$/;
  const reservations = [];
  for (const entry of entries) {
    const match = entry.isFile() && pattern.exec(entry.name);
    if (!match) continue;
    const parsed = await readJson(
      join(registryDirectory, entry.name),
      "final seed reservation",
    );
    if (
      parsed.value?.format !== "dalmuti-v4-final-seed-reservation" ||
      parsed.value.version !== 1 ||
      parsed.value.baseSeed !== Number(match[1])
    ) {
      throw new Error(`invalid final seed reservation: ${entry.name}`);
    }
    finalMatchSeedRanges(parsed.value.baseSeed);
    reservations.push({ ...parsed.value, path: parsed.path, sha256: parsed.sha256 });
  }
  reservations.sort((left, right) => left.baseSeed - right.baseSeed);
  if (new Set(reservations.map(({ baseSeed }) => baseSeed)).size !== reservations.length) {
    throw new Error("final seed registry contains duplicate seeds");
  }
  return reservations;
}

export function chooseNextFinalSeed({
  reservations = [],
  collisionRanges = [],
} = {}) {
  const consumedSeeds = new Set(
    reservations.map((entry) => positiveInteger(entry.baseSeed, "reserved seed")),
  );
  const collisions = [
    ...collisionRanges.map(normalizeCollisionRange),
    ...reservations.flatMap((entry) => finalMatchSeedRanges(entry.baseSeed)),
  ];
  for (let index = 0; ; index += 1) {
    const baseSeed = finalSeedAt(index);
    const ranges = finalMatchSeedRanges(baseSeed);
    if (consumedSeeds.has(baseSeed)) continue;
    const candidateRanges = [
      { start: baseSeed, end: baseSeed },
      ...ranges,
    ];
    if (
      candidateRanges.some((range) =>
        collisions.some((collision) => overlaps(range, collision)),
      )
    ) {
      continue;
    }
    return { baseSeed, matchSeedRanges: ranges, skippedCandidateCount: index };
  }
}

export async function reserveNextFinalSeed({
  registryDirectory: registryDirectoryValue,
  attemptId,
  bindings: bindingValue,
  collisionRanges = [],
  now = new Date(),
}) {
  const registryDirectory = resolve(registryDirectoryValue);
  const bindings = normalizeV4Bindings(bindingValue);
  if (typeof attemptId !== "string" || !/^v4-strongest-attempt-\d{3,}-/.test(attemptId)) {
    throw new TypeError("attemptId must be a V4 strongest attempt ID");
  }
  await mkdir(registryDirectory, { recursive: true });
  for (;;) {
    const reservations = await listFinalSeedReservations(registryDirectory);
    const selection = chooseNextFinalSeed({ reservations, collisionRanges });
    const record = {
      format: "dalmuti-v4-final-seed-reservation",
      version: 1,
      reservedAt: isoTimestamp(now),
      baseSeed: selection.baseSeed,
      matchSeedRanges: selection.matchSeedRanges,
      attemptId,
      bindings,
      bindingSha256: bindingSha256(bindings),
      reuseForbidden: true,
      finalFeedbackPolicy: "sealed-holdout-not-a-training-input",
    };
    try {
      const published = await publishJsonExclusive(
        join(registryDirectory, `seed-${selection.baseSeed}-reservation.json`),
        record,
      );
      return { ...selection, reservation: record, published };
    } catch (error) {
      if (error?.code === "EEXIST") continue;
      throw error;
    }
  }
}

function exactObjectNumbers(actual, expected, label) {
  for (const playerCount of V4_PLAYER_COUNTS) {
    if (actual?.[playerCount] !== expected[playerCount]) {
      throw new Error(
        `${label} p${playerCount} must be ${expected[playerCount]}`,
      );
    }
  }
  const keys = Object.keys(actual ?? {}).map(Number).sort((a, b) => a - b);
  if (keys.join(",") !== V4_PLAYER_COUNTS.join(",")) {
    throw new Error(`${label} must contain exactly p4 through p10`);
  }
}

function metric(result, path, label) {
  let value = result;
  for (const key of path) value = value?.[key];
  return finiteNumber(value, label);
}

function evaluatePerPlayerCount(resultsValue, gates) {
  if (!Array.isArray(resultsValue) || resultsValue.length !== V4_PLAYER_COUNTS.length) {
    throw new Error("evaluation must contain exactly seven player-count results");
  }
  const byPlayerCount = new Map();
  for (const result of resultsValue) {
    const playerCount = positiveInteger(result.playerCount, "result player count");
    if (!V4_PLAYER_COUNTS.includes(playerCount) || byPlayerCount.has(playerCount)) {
      throw new Error(`invalid or duplicate player-count result: p${playerCount}`);
    }
    const meanChipDifference = metric(
      result,
      ["meanChipDifference"],
      `p${playerCount} mean chip difference`,
    );
    const lowerConfidenceBound = metric(
      result,
      ["meanChipDifference95", "low"],
      `p${playerCount} confidence lower bound`,
    );
    const pairwiseRate = metric(
      result,
      ["pairwiseCandidateBeforeNormal", "rate"],
      `p${playerCount} pairwise rate`,
    );
    const gate = {
      meanPassed: meanChipDifference >= gates.minMeanChipDifference,
      lowerBoundPassed:
        lowerConfidenceBound >= gates.minLowerConfidenceBound,
      pairwisePassed: pairwiseRate >= gates.minPairwiseRate,
    };
    gate.passed = gate.meanPassed && gate.lowerBoundPassed && gate.pairwisePassed;
    byPlayerCount.set(playerCount, {
      playerCount,
      meanChipDifference,
      lowerConfidenceBound,
      pairwiseRate,
      gate,
    });
  }
  const results = V4_PLAYER_COUNTS.map((playerCount) => byPlayerCount.get(playerCount));
  return { passed: results.every((result) => result.gate.passed), results };
}

function benchmarkThresholds(gates) {
  return {
    minPointDifference: gates.minMeanChipDifference,
    minLowerBound: gates.minLowerConfidenceBound,
    minPairwiseRate: gates.minPairwiseRate,
  };
}

function thresholdsEqual(actual, expected) {
  return (
    actual?.minPointDifference === expected.minPointDifference &&
    actual?.minLowerBound === expected.minLowerBound &&
    actual?.minPairwiseRate === expected.minPairwiseRate
  );
}

export function bindV4Evaluation({
  stage,
  benchmark,
  bindings: bindingValue,
  finalSeedReservation = null,
}) {
  if (!new Set(["development", "final"]).has(stage)) {
    throw new RangeError("evaluation stage must be development or final");
  }
  const bindings = normalizeV4Bindings(bindingValue);
  if (benchmark?.format !== "dalmuti-model-benchmark" || benchmark.version !== 2) {
    throw new Error("V4 evaluation requires a benchmark version 2 report");
  }
  if (benchmark.evaluationMode !== stage) {
    throw new Error("benchmark evaluation mode does not match the bound stage");
  }
  let benchmarkBindings;
  try {
    benchmarkBindings = normalizeV4Bindings(benchmark.bindings);
  } catch (error) {
    throw new Error("benchmark is missing complete V4 attempt bindings", {
      cause: error,
    });
  }
  if (
    canonicalJson(benchmark.bindings) !== canonicalJson(benchmarkBindings) ||
    canonicalJson(benchmarkBindings) !== canonicalJson(bindings)
  ) {
    throw new Error("benchmark bindings do not exactly match the V4 attempt");
  }
  const expectedBindingEvidence = {
    format: "dalmuti-v4-actual-input-binding-evidence",
    version: 1,
    actualFilesVerified: true,
    actorBundleArtifactSha256: bindings.artifactSha256,
    actorModelSha256: bindings.modelSha256,
    observationContractSha256: bindings.observationSchemaSha256,
    normalSourceSha256: bindings.normalBaselineSha256,
    normalSourceCommit: bindings.normalBaselineSourceCommit,
    normalCommitBlobMatchesWorkingSource: true,
  };
  if (canonicalJson(benchmark.bindingEvidence) !== canonicalJson(expectedBindingEvidence)) {
    throw new Error("benchmark inputs were not verified from the actual bound files");
  }
  const actorInventory = benchmark.candidatePolicy?.bundleActorSha256s;
  const manifestInventory = benchmark.candidatePolicy?.bundleManifestSha256s;
  if (
    benchmark.candidatePolicy?.bundleArtifactSha256 !== bindings.artifactSha256 ||
    !Array.isArray(actorInventory) ||
    ![1, 3].includes(actorInventory.length) ||
    actorInventory.some((digest) => !SHA256_PATTERN.test(digest)) ||
    !Array.isArray(manifestInventory) ||
    ![1, 3].includes(manifestInventory.length) ||
    manifestInventory.some((digest) => !SHA256_PATTERN.test(digest)) ||
    (actorInventory.length === 1 && actorInventory[0] !== bindings.modelSha256)
  ) {
    throw new Error("benchmark actor bundle inventory does not match the V4 attempt");
  }
  if (benchmark.modelSha256 !== bindings.modelSha256) {
    throw new Error("benchmark model SHA-256 does not match the V4 binding");
  }
  if (benchmark.actsPerMatch !== V4_FINAL_ACTS_PER_MATCH) {
    throw new Error(`benchmark must use ${V4_FINAL_ACTS_PER_MATCH} acts per match`);
  }
  if (benchmark.playerCounts?.join(",") !== V4_PLAYER_COUNTS.join(",")) {
    throw new Error("benchmark must evaluate p4 through p10 in canonical order");
  }
  const countKeys = Object.keys(benchmark.matchCountsByPlayerCount ?? {})
    .map(Number)
    .sort((left, right) => left - right);
  if (countKeys.join(",") !== V4_PLAYER_COUNTS.join(",")) {
    throw new Error("benchmark match counts must contain exactly p4 through p10");
  }
  for (const result of benchmark.results ?? []) {
    const expectedMatches = benchmark.matchCountsByPlayerCount[result.playerCount];
    if (!Number.isSafeInteger(expectedMatches) || expectedMatches < 1) {
      throw new Error(`benchmark p${result.playerCount} match count is invalid`);
    }
    if (
      result.matches !== expectedMatches ||
      result.actsPerMatch !== V4_FINAL_ACTS_PER_MATCH
    ) {
      throw new Error(
        `benchmark p${result.playerCount} result does not match its count/acts contract`,
      );
    }
  }
  const gates = stage === "final" ? V4_FINAL_GATES : V4_DEVELOPMENT_GATES;
  if (!thresholdsEqual(benchmark.promotionThresholds, benchmarkThresholds(gates))) {
    throw new Error(`${stage} benchmark promotion thresholds are not exact`);
  }
  const gateSummary = evaluatePerPlayerCount(benchmark.results, gates);
  if (benchmark.promotionPassed !== gateSummary.passed) {
    throw new Error("benchmark promotionPassed disagrees with independently checked gates");
  }
  let seedReservation = null;
  if (stage === "final") {
    exactObjectNumbers(
      benchmark.matchCountsByPlayerCount,
      V4_FINAL_MATCH_COUNTS,
      "final match counts",
    );
    if (benchmark.evaluationDesign?.finalMatchCountPreset !== true) {
      throw new Error("final benchmark must use the final match-count preset");
    }
    if (
      finalSeedReservation?.format !== "dalmuti-v4-final-seed-reservation" ||
      finalSeedReservation.version !== 1
    ) {
      throw new Error("final evaluation requires an atomic final seed reservation");
    }
    if (
      finalSeedReservation.baseSeed !== benchmark.seed ||
      finalSeedReservation.bindingSha256 !== bindingSha256(bindings) ||
      canonicalJson(finalSeedReservation.bindings) !== canonicalJson(bindings) ||
      canonicalJson(finalSeedReservation.matchSeedRanges) !==
        canonicalJson(finalMatchSeedRanges(finalSeedReservation.baseSeed)) ||
      finalSeedReservation.reuseForbidden !== true ||
      finalSeedReservation.finalFeedbackPolicy !==
        "sealed-holdout-not-a-training-input"
    ) {
      throw new Error("final benchmark does not match its seed reservation");
    }
    seedReservation = canonicalize(finalSeedReservation);
  } else if (benchmark.evaluationDesign?.finalMatchCountPreset !== false) {
    throw new Error("development benchmark must not consume the final preset");
  }
  return {
    format: "dalmuti-v4-bound-evaluation",
    version: 1,
    stage,
    bindings,
    bindingSha256: bindingSha256(bindings),
    finalSeedReservation: seedReservation,
    feedbackPolicy:
      stage === "final"
        ? "sealed-holdout-not-a-training-input"
        : "development-metrics-may-guide-next-attempt",
    gates: { ...gates },
    gateSummary,
    benchmarkSha256: sha256Value(benchmark),
    benchmark,
    deploymentTriggered: false,
  };
}

export function validateBoundV4Evaluation(boundEvaluation, bindingValue) {
  const bindings = normalizeV4Bindings(bindingValue);
  if (
    boundEvaluation?.format !== "dalmuti-v4-bound-evaluation" ||
    boundEvaluation.version !== 1
  ) {
    throw new Error("unsupported bound V4 evaluation");
  }
  const rebuilt = bindV4Evaluation({
    stage: boundEvaluation.stage,
    benchmark: boundEvaluation.benchmark,
    bindings,
    finalSeedReservation:
      boundEvaluation.stage === "final"
        ? boundEvaluation.finalSeedReservation
        : null,
  });
  if (canonicalJson(boundEvaluation) !== canonicalJson(rebuilt)) {
    throw new Error("bound V4 evaluation is not the exact canonical evaluation");
  }
  return rebuilt;
}

export function recommendV4DevelopmentInterventions(boundEvaluation, {
  consecutivePlateaus = 0,
} = {}) {
  if (boundEvaluation?.stage !== "development") {
    throw new Error("only development metrics may select training interventions");
  }
  if (boundEvaluation.gateSummary.passed) {
    return [{ action: "reserve-fresh-final-seed", feedbackAllowed: true }];
  }
  const failures = new Set();
  for (const result of boundEvaluation.gateSummary.results) {
    if (!result.gate.meanPassed) failures.add("mean-chip-gap");
    if (!result.gate.lowerBoundPassed) failures.add("confidence-gap");
    if (!result.gate.pairwisePassed) failures.add("pairwise-gap");
  }
  if (positiveInteger(consecutivePlateaus + 1, "plateau count") >= 3) {
    failures.add("plateau");
  }
  return [...failures].map(interventionForV4Failure);
}

export function nextDirectiveFromFinalEvaluation(boundEvaluation) {
  if (boundEvaluation?.stage !== "final") {
    throw new Error("a final directive requires a final evaluation");
  }
  if (boundEvaluation.gateSummary.passed) {
    return {
      loopComplete: true,
      action: "freeze-passing-artifact",
      bindingSha256: boundEvaluation.bindingSha256,
      feedbackAllowed: false,
      deploymentTriggered: false,
    };
  }
  return {
    loopComplete: false,
    ...interventionForV4Failure("sealed-final-failure"),
    bindingSha256: boundEvaluation.bindingSha256,
    sealedResult: "failed",
    metrics: undefined,
    deploymentTriggered: false,
  };
}

export async function recordBoundV4Evaluation({
  attemptDirectory,
  boundEvaluation,
}) {
  const context = await readAttemptContext(attemptDirectory);
  validateBoundV4Evaluation(boundEvaluation, context.bindings);
  if (boundEvaluation?.bindingSha256 !== context.manifest.value.bindingSha256) {
    throw new Error("evaluation is not bound to this attempt");
  }
  if (
    boundEvaluation.stage === "final" &&
    boundEvaluation.finalSeedReservation?.attemptId !==
      context.manifest.value.attemptId
  ) {
    throw new Error("final seed was not reserved for this attempt");
  }
  const expectedState =
    boundEvaluation.stage === "final"
      ? "final-evaluation"
      : "development-evaluation";
  if (context.currentState !== expectedState) {
    throw new Error(
      `${boundEvaluation.stage} evaluation cannot be recorded in state ${context.currentState}`,
    );
  }
  return publishJsonExclusive(
    join(context.attemptDirectory, "evaluations", `${boundEvaluation.stage}.json`),
    boundEvaluation,
  );
}

export async function sealFinalV4Evaluation({
  registryDirectory: registryDirectoryValue,
  attemptDirectory,
  boundEvaluation,
  reportSha256 = null,
  now = new Date(),
}) {
  if (boundEvaluation?.stage !== "final") {
    throw new Error("only a final evaluation can seal a final seed");
  }
  const context = await readAttemptContext(attemptDirectory);
  if (context.currentState !== "final-evaluation") {
    throw new Error("final evaluation can only be sealed in final-evaluation state");
  }
  validateBoundV4Evaluation(boundEvaluation, context.bindings);
  const storedReport = await readJson(
    join(context.attemptDirectory, "evaluations", "final.json"),
    "stored final evaluation report",
  );
  if (canonicalJson(storedReport.value) !== canonicalJson(boundEvaluation)) {
    throw new Error("stored final evaluation does not match the report being sealed");
  }
  const reportDigest = storedReport.sha256;
  if (
    reportSha256 !== null &&
    normalizeSha256(reportSha256, "final report SHA-256") !== reportDigest
  ) {
    throw new Error("declared final report SHA-256 does not match stored report bytes");
  }
  const reservation = boundEvaluation.finalSeedReservation;
  if (reservation.attemptId !== context.manifest.value.attemptId) {
    throw new Error("final seed was not reserved for this attempt");
  }
  const registryDirectory = resolve(registryDirectoryValue);
  const reservationPath = join(
    registryDirectory,
    `seed-${reservation.baseSeed}-reservation.json`,
  );
  const stored = await readJson(reservationPath, "final seed reservation");
  if (
    stored.value.attemptId !== reservation.attemptId ||
    stored.value.bindingSha256 !== boundEvaluation.bindingSha256 ||
    canonicalJson(stored.value) !== canonicalJson(reservation)
  ) {
    throw new Error("stored final seed reservation does not match evaluation");
  }
  const directive = nextDirectiveFromFinalEvaluation(boundEvaluation);
  const seal = {
    format: "dalmuti-v4-final-seed-seal",
    version: 1,
    sealedAt: isoTimestamp(now),
    baseSeed: reservation.baseSeed,
    attemptId: reservation.attemptId,
    bindingSha256: boundEvaluation.bindingSha256,
    reportSha256: reportDigest,
    benchmarkSha256: boundEvaluation.benchmarkSha256,
    reservationSha256: stored.sha256,
    promotionPassed: boundEvaluation.gateSummary.passed,
    feedbackPolicy: "sealed-holdout-not-a-training-input",
    nextDirective: directive,
    containsEvaluationMetrics: false,
    deploymentTriggered: false,
  };
  const published = await publishJsonExclusive(
    join(registryDirectory, `seed-${reservation.baseSeed}-seal.json`),
    seal,
  );
  return { seal, published };
}

async function verifyStoredPassingFinalEvidence({
  context,
  registryDirectory: registryDirectoryValue,
}) {
  const registryDirectory = resolve(registryDirectoryValue);
  const report = await readJson(
    join(context.attemptDirectory, "evaluations", "final.json"),
    "stored final evaluation report",
  );
  const boundEvaluation = validateBoundV4Evaluation(
    report.value,
    context.bindings,
  );
  if (
    boundEvaluation.stage !== "final" ||
    boundEvaluation.gateSummary.passed !== true ||
    boundEvaluation.gateSummary.results.some(
      (result) => result.gate.passed !== true,
    )
  ) {
    throw new Error("passed transition requires every final p4-p10 gate to pass");
  }
  const reservation = boundEvaluation.finalSeedReservation;
  if (reservation.attemptId !== context.manifest.value.attemptId) {
    throw new Error("passing final seed was not reserved for this attempt");
  }
  const reservationRecord = await readJson(
    join(
      registryDirectory,
      `seed-${reservation.baseSeed}-reservation.json`,
    ),
    "stored final seed reservation",
  );
  if (canonicalJson(reservationRecord.value) !== canonicalJson(reservation)) {
    throw new Error("stored final reservation does not match passing evaluation");
  }
  const sealRecord = await readJson(
    join(registryDirectory, `seed-${reservation.baseSeed}-seal.json`),
    "stored final evaluation seal",
  );
  const sealedAt = isoTimestamp(sealRecord.value?.sealedAt, "final seal timestamp");
  const expectedSeal = {
    format: "dalmuti-v4-final-seed-seal",
    version: 1,
    sealedAt,
    baseSeed: reservation.baseSeed,
    attemptId: context.manifest.value.attemptId,
    bindingSha256: context.manifest.value.bindingSha256,
    reportSha256: report.sha256,
    benchmarkSha256: boundEvaluation.benchmarkSha256,
    reservationSha256: reservationRecord.sha256,
    promotionPassed: true,
    feedbackPolicy: "sealed-holdout-not-a-training-input",
    nextDirective: nextDirectiveFromFinalEvaluation(boundEvaluation),
    containsEvaluationMetrics: false,
    deploymentTriggered: false,
  };
  if (canonicalJson(sealRecord.value) !== canonicalJson(expectedSeal)) {
    throw new Error("final seal does not exactly attest the stored passing report");
  }
  return {
    finalSeedRegistryDirectory: registryDirectory,
    baseSeed: reservation.baseSeed,
    finalReportSha256: report.sha256,
    finalBenchmarkSha256: boundEvaluation.benchmarkSha256,
    finalReservationSha256: reservationRecord.sha256,
    finalSealSha256: sealRecord.sha256,
    everyPlayerCountGatePassed: true,
  };
}
