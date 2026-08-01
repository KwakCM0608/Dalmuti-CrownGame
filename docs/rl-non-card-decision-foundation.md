# RL foundation for tax returns and revolutions

This foundation is deliberately disconnected from production. It defines the
information boundary, stable semantic actions, exact legal masks, exhaustive
root-action scoring, and paired counterfactual training targets for the two
pre-play decisions.

## Current production boundary

- `chooseBotTaxReturn(hand, count, difficulty)` receives only the noble's own
  pre-tribute hand and the required return count.
- Quick play and the authoritative online engine lock both nobles' returns
  before tribute cards move. The incoming tribute identities therefore cannot
  be features of the return decision under the current rules implementation.
- `chooseBotRevolution({ hand, role, playerCount }, difficulty)` receives the
  two-joker holder's own hand plus public role and player count.
- The online snapshot exposes a tax route's details only to its participant and
  exposes the revolution chooser only to the holder. Opponent hands remain in
  the authoritative state and are not policy inputs.

`training/non-card-observation.ts` preserves this boundary while adding public
act number, rank-ordered player roles, public hand counts, and public cumulative
scores. Public-player objects that contain card-level fields are rejected.

## Stable observations and actions

- Both observation encoders support 4 through 10 players with ten fixed public
  slots rotated so the actor is first. Opaque player/card IDs are not features.
- The tax catalogue has 103 actions: 13 one-rank returns followed by the 90
  structurally possible sorted two-rank multisets. `[1, 1]` is omitted because
  the deck contains one rank-1 card. Physical cards of the same rank are
  gameplay-equivalent and resolve by stable card-ID order.
- The revolution catalogue has exactly two actions: decline and declare. The
  declaration action feature identifies its public consequence as a normal or
  great revolution from the holder's role.
- State-specific legal masks are exact. A tax action must have the required
  size and cannot use more copies of a rank than the actor holds. Both valid
  revolution actions are legal.

## Search and target design

The root action spaces are small enough to enumerate completely. The scorer in
`training/non-card-search-targets.ts` sees only encoded observation and action
features, and ties resolve to the lower stable catalogue index.

An exact full game tree is neither tractable nor information-safe because
future deals, hidden opponent cards, and other policies are stochastic. The
implemented target design instead uses paired Monte Carlo counterfactuals:

1. Construct or sample a hidden world consistent with the public observation.
2. Clone it once per legal root action.
3. Force every legal action and continue each clone with the same continuation
   random seed and opponent policies.
4. Record the actor's terminal utility (normally chip outcome) without adding
   the sampled hidden world to model features.
5. Repeat across worlds. The target builder requires every batch to cover every
   legal action exactly once, then returns action means, centered values,
   uncertainty, a soft policy target, and a deterministic best action.

This is exhaustive over the decision being learned and paired over downstream
chance, while remaining an honest imperfect-information policy.

That guarantee applies to policy inputs, not to distribution of the raw
counterfactual dataset. Replay seeds are retained for provenance and can
reconstruct hidden deals when combined with the deterministic simulator. The
dataset is consequently a restricted training artifact and must not be
published as an anonymized or public-safe dataset.

## Remaining integration work

The simulator hooks and CPU paired collector are now implemented. See
`docs/rl-non-card-counterfactual-data.md` for the exact same-world contract,
privacy boundary, NDJSON format, and collection command.

1. Add matching GPU action-conditioned heads (a separate head per decision is
   simplest because observation and action feature sizes differ), export them
   with schema metadata, and verify TypeScript/Python score parity.
2. Evaluate tax, revolution, and play-policy ablations separately before using
   a joint candidate in the full automatic gate and human blind A/B test.
3. Only after those gates pass, adapt the selected model to both quick play and
   the online authoritative engine in one final production integration.

## Automatic ablation benchmark

By default, `scripts/rl-benchmark-model.mjs` keeps `--model` as the candidate
card-play policy and optionally accepts either or both independent pre-play
policies:

```powershell
node scripts/rl-benchmark-model.mjs `
  --model artifacts/play.json `
  --tax-model artifacts/tax.json `
  --tax-min-advantage 0.15 `
  --revolution-model artifacts/revolution.json `
  --revolution-min-advantage 0.10 `
  --players 4,5,6,7,8,9,10 `
  --seed 900000001 `
  --output artifacts/joint-benchmark.json
```

Using only `--tax-model` or only `--revolution-model` runs the corresponding
ablation. Candidate seats use each supplied model; every normal seat is routed
through the exact current `normal` tax/revolution heuristic. An omitted
non-card model leaves that decision on the normal heuristic for both groups.
Opponent hidden cards are never added to either policy context.

The optional `--tax-min-advantage` and `--revolution-min-advantage` flags are
training-benchmark safety gates. Each must be a finite nonnegative number and
is accepted only when its matching model is supplied. The default is `0`,
which preserves the existing deterministic model argmax exactly. For every
candidate decision the evaluator also computes the semantic action selected
by the exact current `normal` heuristic, scores every legal action, and
measures:

```text
predicted advantage = model-argmax actor logit - baseline-action actor logit
```

If the model action differs from the baseline and this margin is strictly less
than the configured minimum, the evaluator uses the baseline action. Equality
passes the model action, and every argmax tie uses the lowest stable legal
action index. These gates are evaluation/training infrastructure only; they do
not alter either production bot path.

To measure a non-card policy without mixing in a learned card-play advantage,
use the isolated normal-play mode and omit `--model`:

```powershell
node scripts/rl-benchmark-model.mjs `
  --candidate-play normal `
  --tax-model artifacts/tax.json `
  --players 4,5,6,7,8,9,10 `
  --seed 900000002 `
  --output artifacts/tax-only-normal-play.json
```

This mode routes card play for both candidate and control seats through the
exact current `normal` heuristic, while only candidate actors receive the
specified tax and/or revolution model. Every recorded card-play step is
checked to have `behaviorPolicy: "normal"`. The report's optional
`candidatePlayPolicy` object records the source implementation and verified
candidate/control routing counts. To prevent ambiguous experiments,
`--candidate-play normal` rejects `--model` and requires at least one non-card
model; the default `--candidate-play model` requires `--model`.

When at least one non-card flag is present, the report adds the optional
`nonCardEvaluation` object. It records the ablation name, candidate-only
routing counters, and each supplied model's absolute path, source-file SHA-256,
format, version, decision kind, observation schema version, and action
catalogue version. It also records the model-derived `policyVersion` checked
against the actual simulator steps, the exact baseline implementation and
semantic encoding, the thresholds and decision rule, and routing counts split
into `learnedAction`, `agreedWithBaseline`, and `safetyFallback`. Actor-logit
margin summaries report count, minimum, maximum, mean, and population standard
deviation for all candidate decisions, model/baseline disagreements, learned
actions, and fallbacks. This makes threshold calibration reproducible without
emitting private observations or raw hands. With both model flags omitted,
`nonCardEvaluation` remains absent and the legacy benchmark simulation and
report schema remain unchanged. The three promotion gates and the descriptive
role-regression audit are identical for all ablations.

For lower-variance training diagnostics, use the paired same-seed evaluator
described in `docs/rl-non-card-paired-marginal.md`. It compares each selected
candidate player against that same player in an all-`normal` baseline run. It
does not define or apply a new promotion gate.
