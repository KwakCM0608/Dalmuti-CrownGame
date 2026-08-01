import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const {
  calibrateTemperature,
  main,
  maskedTemperatureMetrics,
  reservoirSampleRolloutFile,
} = await import(
  new URL("../scripts/rl-calibrate-temperature.mjs", import.meta.url)
);

const OBSERVATION_FEATURES = 172;
const ACTION_COUNT = 506;

function createModel() {
  const policyWeight = Array(ACTION_COUNT).fill(0);
  policyWeight[0] = Math.log(9);
  return {
    format: "dalmuti-actor-critic",
    version: 1,
    observationFeatures: OBSERVATION_FEATURES,
    actionCount: ACTION_COUNT,
    hiddenSizes: [1],
    activation: "relu",
    weightLayout: "row-major [out_features, in_features]",
    trunkLayers: [
      {
        inFeatures: OBSERVATION_FEATURES,
        outFeatures: 1,
        weight: Array(OBSERVATION_FEATURES).fill(0),
        bias: [1],
      },
    ],
    policyLayer: {
      inFeatures: 1,
      outFeatures: ACTION_COUNT,
      weight: policyWeight,
      bias: Array(ACTION_COUNT).fill(0),
    },
    valueLayer: {
      inFeatures: 1,
      outFeatures: 1,
      weight: [0],
      bias: [0],
    },
  };
}

function manifest(overrides = {}) {
  return {
    type: "manifest",
    observation: { featureCount: OBSERVATION_FEATURES },
    actionSpace: { size: ACTION_COUNT },
    ...overrides,
  };
}

function sample(id, { forced = false, legal = [0, 1] } = {}) {
  const observation = Array(OBSERVATION_FEATURES).fill(0);
  observation[0] = id;
  return {
    type: "sample",
    observation,
    legalActionIndices: legal,
    actionIndex: legal[0],
    forced,
  };
}

async function writeNdjson(path, records) {
  await writeFile(
    path,
    `${records.map((record) => JSON.stringify(record)).join("\n")}\n`,
  );
}

test("masked metrics use exact tempered softmax and normalized entropy", () => {
  const metrics = maskedTemperatureMetrics([Math.log(9), 0], [0, 1], 1);
  const expectedEntropy = -(
    0.9 * Math.log(0.9) + 0.1 * Math.log(0.1)
  ) / Math.log(2);

  assert.ok(Math.abs(metrics.topActionProbability - 0.9) < 1e-12);
  assert.ok(Math.abs(metrics.normalizedEntropy - expectedEntropy) < 1e-12);
});

test("reservoir sampling is deterministic and excludes forced states", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-temperature-"));
  const dataPath = join(directory, "rollout.ndjson");
  await writeNdjson(dataPath, [
    manifest(),
    sample(1),
    sample(2),
    sample(3),
    sample(4),
    sample(100, { forced: true }),
    sample(200, { legal: [0] }),
    { type: "summary", samples: 6 },
  ]);
  const options = {
    observationFeatures: OBSERVATION_FEATURES,
    actionCount: ACTION_COUNT,
    samplesPerFile: 2,
    seed: 4123,
  };

  const first = await reservoirSampleRolloutFile(dataPath, options);
  const second = await reservoirSampleRolloutFile(dataPath, options);

  assert.equal(first.records, 6);
  assert.equal(first.eligibleStates, 4);
  assert.equal(first.sampledStates, 2);
  assert.deepEqual(first.samples, second.samples);
  assert.ok(first.samples.every((entry) => entry.observation[0] < 100));
});

test("calibration selects the first temperature at or below the target", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dalmuti-temperature-"));
  const modelPath = join(directory, "model.json");
  const firstDataPath = join(directory, "p4.ndjson");
  const secondDataPath = join(directory, "p8.ndjson");
  await writeFile(modelPath, JSON.stringify(createModel()));
  await writeNdjson(firstDataPath, [manifest(), sample(1), sample(2)]);
  await writeNdjson(secondDataPath, [manifest(), sample(3), sample(4)]);

  const result = await calibrateTemperature({
    model: modelPath,
    data: [firstDataPath, secondDataPath],
    temperatures: [1, 1.25, 1.5, 2],
    samplesPerFile: 2,
    seed: 9876,
  });

  assert.equal(result.sampledStates, 4);
  assert.equal(result.selection.temperature, 1.5);
  assert.deepEqual(
    result.metrics.map((entry) => entry.samples),
    [4, 4, 4, 4],
  );
  assert.ok(result.metrics[0].medianTopActionProbability > 0.85);
  assert.ok(result.metrics[1].medianTopActionProbability > 0.85);
  assert.ok(result.metrics[2].medianTopActionProbability <= 0.85);
});

test("CLI requires an explicit seed and rollout dimensions must match", async () => {
  await assert.rejects(
    () => main(["--model", "unused", "--data", "unused"]),
    /explicit --seed is required/,
  );

  const directory = await mkdtemp(join(tmpdir(), "dalmuti-temperature-"));
  const modelPath = join(directory, "model.json");
  const dataPath = join(directory, "bad.ndjson");
  await writeFile(modelPath, JSON.stringify(createModel()));
  await writeNdjson(dataPath, [
    manifest({ observation: { featureCount: OBSERVATION_FEATURES - 1 } }),
    sample(1),
  ]);

  await assert.rejects(
    () => calibrateTemperature({
      model: modelPath,
      data: [dataPath],
      seed: 1,
    }),
    /observation feature count must be 172/,
  );
});
