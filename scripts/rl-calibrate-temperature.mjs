import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { ACTION_SPACE_SIZE } from "../training/action-space.ts";
import {
  evaluateActorCritic,
} from "../training/actor-critic.ts";
import {
  evaluateMlpPolicy,
} from "../training/model-policy.ts";
import { OBSERVATION_FEATURE_COUNT } from "../training/observation.ts";
import { SeededRandom } from "../training/random.ts";
import {
  parseInferenceModel,
} from "../training/stochastic-policy.ts";

const DEFAULT_TEMPERATURES = [1, 1.25, 1.5, 2];
const DEFAULT_SAMPLES_PER_FILE = 2_000;
const TOP_ACTION_PROBABILITY_TARGET = 0.85;

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return parsed;
}

function explicitSeed(value) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new RangeError("seed must be a non-negative safe integer");
  }
  return parsed;
}

export function parseTemperatures(value) {
  const rawValues = value === undefined
    ? DEFAULT_TEMPERATURES
    : String(value).split(",").map((entry) => entry.trim());
  if (rawValues.length < 1 || rawValues.some((entry) => entry === "")) {
    throw new RangeError("temperatures must contain at least one value");
  }
  const temperatures = rawValues.map(Number);
  if (temperatures.some(
    (temperature) => !Number.isFinite(temperature) || temperature <= 0,
  )) {
    throw new RangeError("temperatures must be finite and positive");
  }
  return [...new Set(temperatures)].sort((left, right) => left - right);
}

function deriveFileSeed(seed, fileIndex) {
  return (
    (seed >>> 0) ^ Math.imul(fileIndex + 1, 0x9e37_79b9)
  ) >>> 0;
}

function assertFiniteObservation(observation, expectedLength, label) {
  if (
    !Array.isArray(observation) ||
    observation.length !== expectedLength ||
    observation.some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError(
      `${label} observation must contain ${expectedLength} finite numbers`,
    );
  }
}

function assertLegalActions(legalActionIndices, actionCount, label) {
  if (!Array.isArray(legalActionIndices) || legalActionIndices.length < 1) {
    throw new TypeError(`${label} must contain at least one legal action`);
  }
  const unique = new Set();
  for (const actionIndex of legalActionIndices) {
    if (
      !Number.isInteger(actionIndex) ||
      actionIndex < 0 ||
      actionIndex >= actionCount
    ) {
      throw new RangeError(`${label} contains invalid action ${actionIndex}`);
    }
    if (unique.has(actionIndex)) {
      throw new RangeError(`${label} contains duplicate action ${actionIndex}`);
    }
    unique.add(actionIndex);
  }
}

function validateManifest(
  manifest,
  observationFeatures,
  actionCount,
  label,
) {
  if (!manifest || typeof manifest !== "object" || manifest.type !== "manifest") {
    throw new TypeError(`${label} first non-empty record must be a manifest`);
  }
  if (manifest.observation?.featureCount !== observationFeatures) {
    throw new TypeError(
      `${label} observation feature count must be ${observationFeatures}`,
    );
  }
  if (manifest.actionSpace?.size !== actionCount) {
    throw new TypeError(`${label} action space size must be ${actionCount}`);
  }
}

function validateSample(sample, observationFeatures, actionCount, label) {
  if (!sample || typeof sample !== "object" || sample.type !== "sample") {
    throw new TypeError(`${label} must be a sample record`);
  }
  if (typeof sample.forced !== "boolean") {
    throw new TypeError(`${label} forced must be boolean`);
  }
  assertFiniteObservation(sample.observation, observationFeatures, label);
  assertLegalActions(sample.legalActionIndices, actionCount, label);
  if (
    !Number.isInteger(sample.actionIndex) ||
    !sample.legalActionIndices.includes(sample.actionIndex)
  ) {
    throw new RangeError(`${label} actionIndex must be a legal action`);
  }
}

export async function reservoirSampleRolloutFile(
  pathValue,
  {
    observationFeatures,
    actionCount,
    samplesPerFile,
    seed,
  },
) {
  const path = resolve(pathValue);
  const random = new SeededRandom(seed);
  const reservoir = [];
  const input = createReadStream(path, { encoding: "utf8" });
  const lines = createInterface({ input, crlfDelay: Infinity });
  let manifest = null;
  let lineNumber = 0;
  let records = 0;
  let eligibleStates = 0;
  let summarySeen = false;

  try {
    for await (const line of lines) {
      lineNumber += 1;
      if (line.trim() === "") continue;
      let record;
      try {
        record = JSON.parse(line);
      } catch (error) {
        throw new SyntaxError(
          `${path}:${lineNumber} is not valid JSON: ${error.message}`,
        );
      }
      const label = `${path}:${lineNumber}`;
      if (manifest === null) {
        validateManifest(
          record,
          observationFeatures,
          actionCount,
          label,
        );
        manifest = record;
        continue;
      }
      if (record?.type === "summary") {
        if (summarySeen) {
          throw new TypeError(`${label} duplicates the rollout summary`);
        }
        summarySeen = true;
        continue;
      }
      if (summarySeen) {
        throw new TypeError(`${label} appears after the rollout summary`);
      }
      validateSample(record, observationFeatures, actionCount, label);
      records += 1;
      if (record.forced || record.legalActionIndices.length <= 1) continue;

      eligibleStates += 1;
      const sampledState = {
        observation: record.observation,
        legalActionIndices: record.legalActionIndices,
      };
      if (reservoir.length < samplesPerFile) {
        reservoir.push(sampledState);
      } else {
        const replacementIndex = random.int(eligibleStates);
        if (replacementIndex < samplesPerFile) {
          reservoir[replacementIndex] = sampledState;
        }
      }
    }
  } finally {
    lines.close();
  }

  if (manifest === null) {
    throw new TypeError(`${path} does not contain a manifest`);
  }
  if (eligibleStates === 0) {
    throw new RangeError(`${path} does not contain any non-forced states`);
  }
  return {
    path,
    records,
    eligibleStates,
    sampledStates: reservoir.length,
    samples: reservoir,
  };
}

function median(values) {
  if (values.length < 1) {
    throw new RangeError("median requires at least one value");
  }
  values.sort((left, right) => left - right);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0
    ? (values[middle - 1] + values[middle]) / 2
    : values[middle];
}

export function maskedTemperatureMetrics(
  logits,
  legalActionIndices,
  temperature,
) {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("temperature must be finite and positive");
  }
  if (legalActionIndices.length <= 1) {
    throw new RangeError("temperature metrics require multiple legal actions");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (const actionIndex of legalActionIndices) {
    const logit = logits[actionIndex];
    if (!Number.isFinite(logit)) {
      throw new RangeError(`logit ${actionIndex} must be finite`);
    }
    maximum = Math.max(maximum, logit / temperature);
  }
  const weights = legalActionIndices.map((actionIndex) =>
    Math.exp(logits[actionIndex] / temperature - maximum),
  );
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new RangeError("masked policy distribution is invalid");
  }
  let topActionProbability = 0;
  let entropy = 0;
  for (const weight of weights) {
    const probability = weight / total;
    topActionProbability = Math.max(topActionProbability, probability);
    if (probability > 0) entropy -= probability * Math.log(probability);
  }
  const normalizedEntropy = entropy / Math.log(legalActionIndices.length);
  if (
    !Number.isFinite(topActionProbability) ||
    !Number.isFinite(normalizedEntropy)
  ) {
    throw new RangeError("temperature metrics are not finite");
  }
  return { topActionProbability, normalizedEntropy };
}

function evaluateInferenceLogits(model, observation) {
  const logits = model.format === "dalmuti-actor-critic"
    ? evaluateActorCritic(model, observation).logits
    : evaluateMlpPolicy(model, observation);
  if (
    logits.length !== model.actionCount ||
    Array.from(logits).some((logit) => !Number.isFinite(logit))
  ) {
    throw new RangeError(
      `inference must produce ${model.actionCount} finite logits`,
    );
  }
  return logits;
}

export async function calibrateTemperature({
  model: modelPathValue,
  data,
  temperatures = DEFAULT_TEMPERATURES,
  samplesPerFile = DEFAULT_SAMPLES_PER_FILE,
  seed,
}) {
  if (!Array.isArray(data) || data.length < 1) {
    throw new TypeError("at least one --data file is required");
  }
  const parsedSeed = explicitSeed(seed);
  const parsedSamplesPerFile = positiveInteger(
    samplesPerFile,
    "samples-per-file",
  );
  const parsedTemperatures = parseTemperatures(
    Array.isArray(temperatures) ? temperatures.join(",") : temperatures,
  );
  const modelPath = resolve(modelPathValue);
  const modelBytes = await readFile(modelPath);
  const model = parseInferenceModel(JSON.parse(modelBytes.toString("utf8")));
  if (
    model.observationFeatures !== OBSERVATION_FEATURE_COUNT ||
    model.actionCount !== ACTION_SPACE_SIZE
  ) {
    throw new TypeError(
      `model must use ${OBSERVATION_FEATURE_COUNT} observations and ` +
        `${ACTION_SPACE_SIZE} actions`,
    );
  }

  const topProbabilities = parsedTemperatures.map(() => []);
  const normalizedEntropies = parsedTemperatures.map(() => []);
  const files = [];
  for (const [fileIndex, dataPath] of data.entries()) {
    const fileSeed = deriveFileSeed(parsedSeed, fileIndex);
    const sampled = await reservoirSampleRolloutFile(dataPath, {
      observationFeatures: model.observationFeatures,
      actionCount: model.actionCount,
      samplesPerFile: parsedSamplesPerFile,
      seed: fileSeed,
    });
    files.push({
      path: sampled.path,
      seed: fileSeed,
      records: sampled.records,
      eligibleStates: sampled.eligibleStates,
      sampledStates: sampled.sampledStates,
    });
    for (const sample of sampled.samples) {
      const logits = evaluateInferenceLogits(model, sample.observation);
      parsedTemperatures.forEach((temperature, temperatureIndex) => {
        const metrics = maskedTemperatureMetrics(
          logits,
          sample.legalActionIndices,
          temperature,
        );
        topProbabilities[temperatureIndex].push(
          metrics.topActionProbability,
        );
        normalizedEntropies[temperatureIndex].push(
          metrics.normalizedEntropy,
        );
      });
    }
  }

  const metrics = parsedTemperatures.map((temperature, index) => ({
    temperature,
    samples: topProbabilities[index].length,
    medianTopActionProbability: median(topProbabilities[index]),
    medianNormalizedEntropy: median(normalizedEntropies[index]),
  }));
  const selected = metrics.find(
    (entry) =>
      entry.medianTopActionProbability <= TOP_ACTION_PROBABILITY_TARGET,
  ) ?? metrics.at(-1);

  return {
    format: "dalmuti-temperature-calibration",
    version: 1,
    model: {
      path: modelPath,
      sha256: createHash("sha256").update(modelBytes).digest("hex"),
      format: model.format,
      observationFeatures: model.observationFeatures,
      actionCount: model.actionCount,
    },
    seed: parsedSeed,
    samplesPerFile: parsedSamplesPerFile,
    files,
    sampledStates: files.reduce(
      (total, file) => total + file.sampledStates,
      0,
    ),
    metrics,
    selection: {
      topActionProbabilityTarget: TOP_ACTION_PROBABILITY_TARGET,
      temperature: selected.temperature,
      rule:
        "smallest temperature with median top-action probability <= target; " +
        "otherwise largest temperature",
    },
  };
}

export async function main(args = process.argv.slice(2)) {
  const cliArgs = [...args];
  if (cliArgs[0] === "--") cliArgs.shift();
  const seedWasExplicit = cliArgs.some(
    (argument) => argument === "--seed" || argument.startsWith("--seed="),
  );
  const { values } = parseArgs({
    args: cliArgs,
    options: {
      model: { type: "string", short: "m" },
      data: { type: "string", multiple: true },
      temperatures: {
        type: "string",
        default: DEFAULT_TEMPERATURES.join(","),
      },
      "samples-per-file": {
        type: "string",
        default: String(DEFAULT_SAMPLES_PER_FILE),
      },
      seed: { type: "string" },
      output: { type: "string", short: "o" },
    },
  });
  if (!values.model) throw new TypeError("--model is required");
  if (!values.data?.length) {
    throw new TypeError("at least one --data file is required");
  }
  if (!seedWasExplicit || values.seed === undefined) {
    throw new TypeError("an explicit --seed is required");
  }

  const result = await calibrateTemperature({
    model: values.model,
    data: values.data,
    temperatures: values.temperatures,
    samplesPerFile: values["samples-per-file"],
    seed: values.seed,
  });
  if (values.output) {
    const outputPath = resolve(values.output);
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
    console.log(
      `Selected temperature ${result.selection.temperature}; wrote ${outputPath}`,
    );
  } else {
    console.log(JSON.stringify(result, null, 2));
  }
  return result;
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : null;
if (invokedPath === fileURLToPath(import.meta.url)) {
  await main();
}
