# Experimental merchant revolution candidate

## Status

This is a training-only, production-isolated experiment. Its current verdict
is **rejected for integration**. It does not modify quick-match or online bot
dispatch, and it is not a hard-bot promotion result.

The candidate is intentionally simple and deterministic:

- at six players, a candidate-controlled merchant holding both jokers declares
  a normal revolution;
- every other role and player count executes the exact current `normal`
  `chooseBotRevolution` result;
- non-candidate players always execute the exact current `normal` result;
- card play and tax returns stay on current `normal` in both arms.

The implementation lives under `scripts/` so it cannot be reached by either
production game mode.

## Action and source-data audit

The strict audit command is:

```powershell
node scripts/rl-audit-revolution-merchant-candidate.mjs `
  --input artifacts/rl/noncard-revolution-determinization-p4-10-e100-a5-k8-c1-seed733000001-run-001/data.ndjson `
  --output artifacts/rl/revolution-merchant-candidate-v1-data-audit-run-001/report.json
```

It independently verifies the full-file SHA-256 and sidecar, v2
determinization contract, schema and action dimensions, canonical information
state hashes and uniqueness, role/player-count coordinates, physical own-hand
counts, both-joker precondition, action coverage, paired-world counts, summary
hashes, and the current `normal` baseline action.

The audited source has SHA-256
`b861bc857e4e9d845f224ab06b3f9b2c503f9e3721e8e6979a99ef694dc96a05`.
Action `0` is decline and action `1` is declare. Simulator non-card rewards are
`(round chips - 2) / 2`, so decision-act differences are multiplied by two to
report actual chips.

The source contains 502 revolution information states. Among merchants, only
the p6 cell has a positive two-sided 95% lower bound when the repository's
small-sample Student-t rule is applied:

| Players | Merchant states | Declare minus normal, actual chips | 95% CI |
| --- | ---: | ---: | ---: |
| 5 | 18 | +0.0556 | [-0.1960, +0.3071] |
| 6 | 26 | +0.3317 | [+0.1365, +0.5270] |
| 7 | 20 | +0.1187 | [-0.1156, +0.3531] |
| 8 | 34 | +0.0956 | [-0.0369, +0.2281] |
| 9 | 17 | +0.2279 | [-0.0085, +0.4643] |
| 10 | 20 | +0.0063 | [-0.1672, +0.1797] |

The p9 normal-approximation lower bound is barely positive, but the required
small-sample Student-t lower bound is negative. Therefore p9 is excluded rather
than rounded into the candidate.

The audit report SHA-256 is
`ccafb2dc5b0c1a9c209be1101cbb05d6859572a45d17f90fc32068ff99fe7fdc`.
This table is exploratory because the same dataset selected and describes the
rule; it is not independent confirmation.

## Fresh-seed paired simulator evaluation

The independent benchmark used seed `945000001`, five acts, 1,000 p6 match
pairs, and 20 exact-no-op pairs at every other player count:

```powershell
node scripts/rl-benchmark-revolution-merchant-candidate.mjs `
  --source-data artifacts/rl/noncard-revolution-determinization-p4-10-e100-a5-k8-c1-seed733000001-run-001/data.ndjson `
  --seed 945000001 `
  --match-counts 4:20,5:20,6:1000,7:20,8:20,9:20,10:20 `
  --acts 5 `
  --output artifacts/rl/revolution-merchant-candidate-v1-paired-seed945000001-run-001/report.json
```

Each pair uses the same seed, episode ID, initial seats, candidate IDs, card
policy, and environment stream. Every intervention action is checked against
routing telemetry and the frozen policy version. A pair with no changed action
must have exactly equal acts and final scores.

At p6 there were 141 changed merchant decisions in 135 of 1,000 match pairs.
The direct declaring actor's chip result in the changed act improved by
`+0.2407`, 95% CI `[+0.0150, +0.4665]`, clustered by match. This independently
confirms that the narrow action can help the merchant who declares.

It did not improve the balanced candidate group:

| p6 paired metric | Estimate | 95% CI |
| --- | ---: | ---: |
| Candidate chip difference per player-act | -0.0025 | [-0.0087, +0.0036] |
| Candidate finish-place difference | +0.0025 | [-0.0057, +0.0106] |
| Candidate final-score difference | -0.0127 | [-0.0433, +0.0180] |

Candidate finishes were better 318 times and worse 319 times among 15,000
candidate player-acts; the non-tie better rate was 49.92%. Baseline-role
analysis also found negative externalities for candidate noble seats:

- great Dalmuti: `-0.0364`, 95% CI `[-0.0569, -0.0159]` chips;
- lesser Dalmuti: `-0.0178`, 95% CI `[-0.0327, -0.0029]` chips;
- merchant: `+0.0128`, 95% CI `[+0.0013, +0.0243]` chips.

All p4, p5, p7, p8, p9, and p10 pairs were exact no-ops with zero metric
differences. The paired report SHA-256 is
`f2d4e66c5d65ecc8ac85dc82e65d98df5866eea2886e3f3aa9b2e3a050be69ee`.

## Interpretation

The p6 declaration is individually useful when the candidate itself is the
merchant, but revolution is a table-wide, zero-sum intervention. In the
balanced candidate-versus-normal design, its benefit is offset by harm to
candidate nobles. The candidate therefore fails the stated goal of improving
over current `normal` without regressions and must not be integrated.

The result is still useful as a supervised/RL signal: a future shared policy
can learn that a p6 merchant declaration may be locally valuable, while final
promotion must continue to be decided by full mixed-table gates rather than by
the decision-state advantage alone.
