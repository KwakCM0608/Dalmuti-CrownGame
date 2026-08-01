# Conservative tax-return advantage ensemble

This pipeline is training and evaluation infrastructure only. It does not
change the quick-game or online production bot and it does not change the
three automatic promotion gates.

## Scope and fallback

- `returnCount=1` (lesser Dalmuti) is never used for a gradient or checkpoint
  selection. Inference always returns the exact current `normal` heuristic
  action.
- `returnCount=2` (great Dalmuti) is the only learned role/decision route.
- The learned score is an actual-chip advantage against the exact current
  `normal` return action. The baseline action is residualized to exactly zero.

The source simulator reward is `(roundChipAward - 2) / 2`. Both accepted data
schemas therefore apply the mandatory transform `source advantage * 2` before
training. The exported model binds this transform under `trainingData` and the
Python and TypeScript strict parsers reject another unit or multiplier.

## Accepted counterfactual data

The adapter accepts one source schema per run; v1 and v2 files cannot be mixed.

- v1 derives each target from
  `decisionActUtility[action] - decisionActUtility[baseline]`. Its split key is
  `canonicalWorldKey` and its bound sample count is one.
- v2 consumes
  `pairedDecisionActBaselineAdvantage.mean`. It verifies the restricted
  training-data privacy contract, checksum sidecar, content-before-summary
  hash, a recomputed `canonicalInformationStateKey`, the exact known
  determinization algorithm/version/contract hash and seed derivations,
  `worldCount`, `continuationCount`, effective independent-world count,
  uncertainty, and an exactly-zero baseline advantage. The schema is explicitly
  bound as `world-clustered-paired-baseline-advantages-v2`. Continuations are averaged inside
  each hidden world before uncertainty is computed: `count` and the standard
  error use the `K` independent world means, never the raw `K*C` continuation
  evaluations.

V2 binds `effectiveIndependentWorlds=K` separately from
`rawContinuationEvaluations=K*C`. For terminal-play targets, continuations only
reduce within-world rollout noise; for deterministic decision-act targets the
collector verifies that all `C` results within a world are identical. In both
cases, continuations are not independent samples and do not increase a state’s
optimization weight. Every canonical information state has equal weight.
Train/validation splitting and deterministic bootstrap use the same canonical
state key, so one information state cannot leak across the split.

## Objective and ensemble

Each of five deterministic bootstrap members uses the compact score

```text
c(s) = tanh(A s + b)
raw(s, a) = c(s)^T W phi(a)
advantage(s, a | baseline) = raw(s, a) - raw(s, baseline)
```

Training averages over legal non-baseline actions inside each state, then
averages states equally. Its paired objective combines:

- Huber regression on actual-chip action-vs-baseline advantage;
- tie-aware sign BCE, with positive/negative labels and a `0.5` target for
  utility ties.

Every member is selected independently by paired validation loss. Member seeds,
checkpoint epochs, validation losses, and canonical parameter hashes are bound
into strict model JSON version 2.

## Conservative inference

For every legal two-card action the evaluator computes five member advantages,
their mean, sample standard deviation, and
`LCB = mean - 1.645 * sampleSD`. An action may replace `normal` only when:

1. all five member advantages are strictly positive; and
2. its LCB is strictly greater than the chip-unit threshold.

The eligible action with the highest LCB is selected. A tie with the zero
baseline stays on the baseline; remaining action ties use the lowest stable
catalogue index. The conservative model default threshold is `0.5` actual chips,
and a
benchmark CLI `--tax-min-advantage` value overrides it.

Benchmark reports expose model/member provenance and aggregate telemetry for
mean advantage, sample standard deviation, LCB, unanimity, learned routing,
fallback count, and fallback reason. The seed-paired marginal evaluator uses
the same shared hook, so ordinary and paired reports cannot drift in routing.

## Training and verification

Use a brand-new result directory for every attempt:

```powershell
python gpu-training/train_tax_return_advantage.py `
  --input artifacts/rl/noncard-determinization-v2.ndjson `
  --result-dir artifacts/rl/tax-advantage-v2-seed-202608051-run-001 `
  --device cuda `
  --epochs 500 `
  --batch-size 256 `
  --seed 202608051
```

Verify and create an exclusive package:

```powershell
python gpu-training/verify_tax_return_advantage_results.py `
  --result-dir artifacts/rl/tax-advantage-v2-seed-202608051-run-001

python gpu-training/package_tax_return_advantage_results.py `
  --result-dir artifacts/rl/tax-advantage-v2-seed-202608051-run-001 `
  --output artifacts/rl/tax-advantage-v2-seed-202608051-run-001.zip

python gpu-training/package_tax_return_advantage_results.py `
  --verify-archive artifacts/rl/tax-advantage-v2-seed-202608051-run-001.zip `
  --checksum artifacts/rl/tax-advantage-v2-seed-202608051-run-001.zip.sha256
```

The result verifier checks the exclusive file inventory (including nested
same-named files), every file hash,
dataset/model unit binding, five seeds, member metrics, best checkpoints, and
member parameter hashes. The package verifier extracts into a temporary
directory and repeats the full result verification after first binding the ZIP
to its external checksum sidecar.

## Training-only evaluation

Use the same model in the seed-paired marginal evaluator. Omitting the threshold
uses the model's explicit `0.5`-chip default:

```powershell
pnpm rl:benchmark-noncard-marginal -- `
  --tax-model artifacts/rl/tax-advantage-v2-seed-202608051-run-001/model.json `
  --players 4,5,6,7,8,9,10 `
  --matches 100 `
  --acts 5 `
  --seed 840000001 `
  --output artifacts/rl/benchmarks/tax-advantage-paired-840000001.json
```

This report is a training diagnostic. It does not apply, redefine, or replace
the final play-policy promotion gates.
