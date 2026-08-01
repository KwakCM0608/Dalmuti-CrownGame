# Non-card counterfactual diagnostics

`scripts/rl-diagnose-non-card-counterfactuals.mjs` is a training-only,
read-only diagnostic for completed paired counterfactual NDJSON datasets. It
does not load a model and does not change either production bot.

The reader streams one record at a time. Memory is limited to fixed action
catalogue counters and aggregate cells by paired world; it never retains the
dataset's decision/action records. Before reporting metrics it requires:

- the supported manifest as the first line and a final summary as the last;
- a trailing newline, no blank lines, and no records after the summary;
- exact observation, action-catalogue, mask, feature, and legal-root coverage;
- finite utilities, centered targets summing to zero, and probabilities
  summing to one;
- the canonical maximum-utility `bestActionIndex`;
- summary decision/action counts and the exact pre-summary byte count/hash.

An incomplete or internally inconsistent dataset is rejected instead of
producing a partial report.

## Usage

```powershell
node scripts/rl-diagnose-non-card-counterfactuals.mjs `
  --input artifacts/rl/non-card-counterfactuals-run-001.ndjson
```

Add `--output new-report.json` to write the complete JSON report. The output
uses exclusive creation and must not already exist. `--json-stdout` prints the
same JSON without creating a file.

## Report interpretation

The report includes overall, single-dimension, decision-plus-dimension, and
fully crossed cells for decision kind, player count, act/round, actor role,
and tax return count. Each cell contains:

- record and mean legal-action counts;
- exact baseline-versus-best-label agreement, plus an agreement measure that
  treats tied maximum-utility actions as optimal;
- the mean number of tied oracle actions, unique-best rate, target sample
  count, and baseline/maximum soft-target mass, which expose fragile
  single-world argmax labels;
- baseline and lowest-legal-index centered utility;
- oracle maximum utility minus baseline utility;
- raw and legal-action-normalized target entropy;
- baseline, target-best, and lowest-legal action-index rates;
- revolution declaration rates where applicable;
- tax return rank summaries for baseline, best, and lowest-legal selections.

Tax strength excludes jokers and maps rank 1 to 1 and rank 12 to 0 using
`(12 - rank) / 11`. Joker frequency is reported separately. This avoids
pretending that a joker has an ordinary numeric strength.

The 95% interval for baseline centered utility first averages records within
each paired hidden world, then treats worlds as independent clusters. Its
estimand is therefore an equal-weighted mean of world means; the ordinary
`meanBaselineCenteredUtility` remains the record-weighted descriptive mean.
Student-t intervals are used below 30 worlds and normal intervals otherwise.
