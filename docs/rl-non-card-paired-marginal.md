# Paired marginal benchmark for non-card policies

`scripts/rl-benchmark-non-card-marginal.mjs` is a training-only diagnostic for
tax-return and revolution models. It removes the seat-composition noise in the
mixed candidate-versus-normal benchmark by simulating every match twice:

1. the baseline run uses the exact current `normal` heuristic for card play,
   tax return, and revolution;
2. the intervention run reuses the same match seed, episode ID, player IDs,
   player count, act count, and initial seat order;
3. only the selected candidate IDs may receive the supplied non-card models;
4. the same candidate players' chip awards and finish places are subtracted as
   `intervention - baseline`.

The selected IDs rotate cyclically between matches. Both simulations use the
simulator's deterministic environment stream, while model-local randomness is
isolated from deals and card play. If an early model decision changes a finish
order, later roles and card ownership are allowed to diverge: that divergence
is part of the policy's causal effect, not a pairing failure. The report checks
that each pair starts with the same seed, episode ID, and initial player order.

## Running an ablation

The seed and fresh output path are mandatory. Supply at least one model:

```powershell
pnpm rl:benchmark-noncard-marginal -- `
  --tax-model artifacts/tax.json `
  --tax-min-advantage 0.50 `
  --players 4,5,6,7,8,9,10 `
  --matches 100 `
  --acts 5 `
  --seed 820000001 `
  --output artifacts/rl/benchmarks/tax-paired-820000001.json
```

Use `--revolution-model` and `--revolution-min-advantage` for a revolution
ablation, or provide both model paths for a combined run. `--match-counts`
accepts the same `player:matches` comma-separated form as the full benchmark.
`--omit-match-data` removes per-match and per-round details from large reports.

The output is opened exclusively and is never overwritten. Reusing an output
path fails. Model provenance includes the absolute source path, SHA-256,
format/schema versions, and the policy version verified against simulator
steps. The report also records safety thresholds, learned/baseline agreement,
fallback routing, actor-logit margins, and exact-normal routing counts.

## Reading the report

For each player count and for the pooled sample, `pairedMarginal` contains
match-clustered 95% confidence intervals for:

- chip difference per selected player-act (positive is better);
- finish-place difference per selected player-act (negative is better);
- first-place-rate difference (positive is better);
- last-place-rate difference (negative is better);
- final cumulative-score difference per selected player (positive is better).

Each match contributes one cluster value, so repeated acts and multiple
candidate players in a match are not treated as independent observations.
`finishComparison` separately counts how often the same candidate finished
better, tied, or worse in the intervention run.

This report deliberately contains no promotion thresholds or pass/fail result.
It is useful for model and safety-threshold selection, but it does not alter or
replace the three full automatic promotion gates in
`scripts/rl-benchmark-model.mjs`. Use development seeds here; keep the reserved
final evaluation seed unseen until the final candidate is fixed.

The report contains outcomes and routing metadata, not hands or encoded private
observations. As with all training artifacts, keep model and benchmark files
out of production bundles until the final integration decision.
