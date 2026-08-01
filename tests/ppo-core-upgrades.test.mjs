import assert from "node:assert/strict";
import test from "node:test";

const { sampleMaskedLogits } = await import(
  new URL("../training/stochastic-policy.ts", import.meta.url)
);

test("temperature sampling reports the exact tempered behavior probability", () => {
  const logits = [Math.log(1), Math.log(9), 1_000];
  const first = sampleMaskedLogits(
    logits,
    [0, 1],
    () => 0.1,
    -0.25,
    2,
  );
  const second = sampleMaskedLogits(
    logits,
    [0, 1],
    () => 0.9,
    -0.25,
    2,
  );

  assert.equal(first.actionIndex, 0);
  assert.ok(Math.abs(first.logProbability - Math.log(0.25)) < 1e-12);
  assert.equal(first.valueEstimate, -0.25);
  assert.equal(second.actionIndex, 1);
  assert.ok(Math.abs(second.logProbability - Math.log(0.75)) < 1e-12);
});

test("temperature validation rejects non-positive and non-finite values", () => {
  for (const temperature of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(
      () => sampleMaskedLogits([0], [0], () => 0, 0, temperature),
      /temperature must be finite and positive/,
    );
  }
});
