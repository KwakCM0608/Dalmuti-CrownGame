import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import {
  bindV4Evaluation,
  buildV4AttemptPlan,
  chooseNextFinalSeed,
  createV4Attempt,
  listFinalSeedReservations,
  nextDirectiveFromFinalEvaluation,
  normalizeV4Bindings,
  readV4AttemptSnapshot,
  recordBoundV4Evaluation,
  recordV4Exit,
  recordV4Heartbeat,
  reserveNextFinalSeed,
  sealFinalV4Evaluation,
  sha256Value,
  transitionV4Attempt,
} from "./lib/v4-strongest-loop.mjs";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    action: { type: "string", default: "create-attempt" },
    root: { type: "string" },
    "attempt-directory": { type: "string" },
    attempt: { type: "string", default: "1" },
    label: { type: "string", default: "strongest" },
    "artifact-sha256": { type: "string" },
    "model-sha256": { type: "string" },
    "schema-sha256": { type: "string" },
    "baseline-sha256": { type: "string" },
    "baseline-source-commit": { type: "string" },
    to: { type: "string" },
    reason: { type: "string" },
    phase: { type: "string" },
    progress: { type: "string", default: "0" },
    message: { type: "string", default: "" },
    outcome: { type: "string" },
    "failure-class": { type: "string" },
    "exit-code": { type: "string" },
    signal: { type: "string" },
    report: { type: "string" },
    stage: { type: "string" },
    "seed-reservation": { type: "string" },
    "collision-range": { type: "string", multiple: true },
    "dry-run": { type: "boolean", default: false },
  },
});

function requireString(value, option) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`--${option} is required`);
  }
  return value;
}

function bindingsFromValues() {
  return normalizeV4Bindings({
    artifactSha256: requireString(values["artifact-sha256"], "artifact-sha256"),
    modelSha256: requireString(values["model-sha256"], "model-sha256"),
    observationSchemaSha256: requireString(values["schema-sha256"], "schema-sha256"),
    normalBaselineSha256: requireString(values["baseline-sha256"], "baseline-sha256"),
    normalBaselineSourceCommit: requireString(
      values["baseline-source-commit"],
      "baseline-source-commit",
    ),
  });
}

function parseCollisionRanges(entries = []) {
  return entries.map((entry) => {
    const match = /^(\d+):(\d+)$/.exec(entry);
    if (!match) throw new TypeError("--collision-range must use START:END");
    return { start: Number(match[1]), end: Number(match[2]), label: "cli" };
  });
}

async function jsonFile(path, label) {
  try {
    const bytes = await readFile(resolve(path));
    return {
      value: JSON.parse(bytes.toString("utf8")),
      sha256: createHash("sha256").update(bytes).digest("hex"),
    };
  } catch (error) {
    throw new Error(`${label} could not be read: ${path}: ${error.message}`, {
      cause: error,
    });
  }
}

function print(value) {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

const action = values.action;
if (action === "create-attempt") {
  const plan = buildV4AttemptPlan({
    root: requireString(values.root, "root"),
    attemptNumber: values.attempt,
    label: values.label,
    bindings: bindingsFromValues(),
  });
  print(values["dry-run"] ? { dryRun: true, plan } : await createV4Attempt(plan));
} else if (action === "snapshot") {
  print(await readV4AttemptSnapshot(requireString(values["attempt-directory"], "attempt-directory")));
} else if (action === "transition") {
  const input = {
    attemptDirectory: requireString(values["attempt-directory"], "attempt-directory"),
    to: requireString(values.to, "to"),
    reason: requireString(values.reason, "reason"),
  };
  print(
    values["dry-run"]
      ? { dryRun: true, action, input }
      : await transitionV4Attempt(input),
  );
} else if (action === "heartbeat") {
  const input = {
    attemptDirectory: requireString(values["attempt-directory"], "attempt-directory"),
    phase: requireString(values.phase, "phase"),
    progress: Number(values.progress),
    message: values.message,
  };
  print(
    values["dry-run"]
      ? { dryRun: true, action, input }
      : await recordV4Heartbeat(input),
  );
} else if (action === "exit") {
  const input = {
    attemptDirectory: requireString(values["attempt-directory"], "attempt-directory"),
    outcome: requireString(values.outcome, "outcome"),
    failureClass: values["failure-class"] ?? null,
    exitCode: values["exit-code"] === undefined ? null : Number(values["exit-code"]),
    signal: values.signal ?? null,
  };
  print(values["dry-run"] ? { dryRun: true, action, input } : await recordV4Exit(input));
} else if (action === "reserve-final-seed") {
  const root = resolve(requireString(values.root, "root"));
  const registryDirectory = join(root, "final-seed-registry");
  const collisionRanges = parseCollisionRanges(values["collision-range"]);
  const attemptDirectory = requireString(
    values["attempt-directory"],
    "attempt-directory",
  );
  const snapshot = await readV4AttemptSnapshot(attemptDirectory);
  const attemptId = snapshot.attemptId;
  const bindings = snapshot.bindings;
  if (values["dry-run"]) {
    const reservations = await listFinalSeedReservations(registryDirectory);
    print({
      dryRun: true,
      action,
      registryDirectory,
      attemptId,
      selection: chooseNextFinalSeed({ reservations, collisionRanges }),
      bindingSha256: sha256Value(bindings),
    });
  } else {
    print(await reserveNextFinalSeed({
      registryDirectory,
      attemptId,
      bindings,
      collisionRanges,
    }));
  }
} else if (action === "record-evaluation") {
  const attemptDirectory = requireString(values["attempt-directory"], "attempt-directory");
  const snapshot = await readV4AttemptSnapshot(attemptDirectory);
  const reportFile = await jsonFile(
    requireString(values.report, "report"),
    "benchmark report",
  );
  const benchmark = reportFile.value;
  const stage = requireString(values.stage, "stage");
  const seedReservation = values["seed-reservation"]
    ? (await jsonFile(values["seed-reservation"], "seed reservation")).value
    : null;
  const boundEvaluation = bindV4Evaluation({
    stage,
    benchmark,
    bindings: snapshot.bindings,
    finalSeedReservation: seedReservation,
  });
  if (values["dry-run"]) {
    print({
      dryRun: true,
      action,
      gateSummary: boundEvaluation.gateSummary,
      directive:
        stage === "final"
          ? nextDirectiveFromFinalEvaluation(boundEvaluation)
          : null,
    });
  } else {
    const recorded = await recordBoundV4Evaluation({ attemptDirectory, boundEvaluation });
    let sealed = null;
    if (stage === "final") {
      const root = resolve(requireString(values.root, "root"));
      sealed = await sealFinalV4Evaluation({
        registryDirectory: join(root, "final-seed-registry"),
        boundEvaluation,
        reportSha256: reportFile.sha256,
      });
    }
    print({ recorded, sealed, gateSummary: boundEvaluation.gateSummary });
  }
} else {
  throw new RangeError(`unsupported --action: ${action}`);
}
