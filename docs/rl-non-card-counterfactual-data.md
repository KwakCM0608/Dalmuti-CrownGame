# Paired non-card counterfactual data

`scripts/rl-generate-non-card-counterfactuals.mjs` collects CPU training data
for tax-return and revolution decisions. It is training-only and does not
change either production bot.

## What is paired

For every requested player count and match seed, the collector first runs the
normal deterministic bots with `nonCard: {}`. That baseline discovers the real
tax and revolution decision points without changing their behavior. At each
selected decision the collector then:

1. enumerates every action in the recorded legal mask;
2. reruns the complete match with the same seed and episode ID;
3. supplies exactly one forced override in the `taxReturn` or `revolution`
   namespace;
4. verifies that decision identity, encoded observation, legal mask, actor,
   schema, catalogue, and public pre-decision metadata are unchanged;
5. records the actor's terminal cumulative chip score for that action.

The continuation remains the deterministic normal policy. Forced non-card
actions use no environment RNG, so every action clone starts in the identical
dealt hidden world and shares the same downstream random stream. A missing,
illegal, duplicated, or unused override aborts collection.

The target fields come from
`training/non-card-search-targets.ts#buildPairedCounterfactualTargets`. Each
record contains terminal and centered utilities, a deterministic best action,
sample standard deviation, standard error, and a temperature-controlled soft
policy target. A single hidden-world record reports numeric SD/SE as zero for
serialization compatibility but marks `standardErrorEstimable: false`; one
world is not evidence of zero sampling uncertainty. At least two independently
sampled hidden worlds are required to estimate between-world uncertainty.

## Conditional hidden-world augmentation

The default `--hidden-worlds 1 --continuations 1` path is the original
version-1 collector. Supplying those values explicitly produces the exact same
NDJSON bytes and SHA-256 as omitting them. A run switches to the aggregate
version-2 schema when hidden worlds or continuations exceed one, or when a tax
return-count filter is requested.

For version 2, the original replay is hidden world zero. In each additional
world the target actor keeps the exact original physical hand. All other
physical cards are shuffled with a decision-local RNG and dealt back to the
other players at their unchanged public hand counts. The simulator consumes
zero draws from the environment stream for this operation. Before accepting a
world it strictly verifies:

- all 80 physical cards have exactly one owner, with no missing, duplicated,
  or rank-drifted card;
- the actor's encoded observation, legal mask, normal baseline action,
  tax-return count, and ordered public-player snapshot exactly match;
- every public act before the target act is byte-for-byte identical;
- the target decision appears at the same non-card decision position;
- tax tribute and return selections have the required `2 + 1` cards, preserve
  the full deck through both transfer directions, and restore public hand
  counts.

A sampled tax world can legitimately stop reaching the target, for example if
the redistributed jokers cause a revolution. Such a world is rejected. Each
requested world has a finite retry cap; accepted records include attempt counts
and rejection-reason counts, but never the candidate seeds.

`--continuations C` optionally replaces the environment stream only after the
root non-card phase. This leaves the complete information state and hidden
world fixed while varying later deals. Every action in one `(hidden world,
continuation)` pair receives the same stream. Continuations are nested Monte
Carlo evaluations, not independent hidden-world samples: terminal utilities
and paired terminal advantages are averaged across the `C` continuations
inside each world first, then the `K` world means receive equal weight. SD and
SE are computed only across those `K` world means.

The normal deterministic card-play policy consumes no continuation RNG during
the decision act. The collector verifies that decision-act utility is exactly
identical across all continuations of a world, discards those duplicates for
statistics, and again computes SD/SE from the `K` hidden worlds only. Records
separate `rawContinuationEvaluations = K * C` from
`effectiveIndependentWorlds = K`; `targetSampleCount` is `K`. Individual
cards, replay seeds, finishes, and per-world utilities are absent.

All worlds for a record share `canonicalInformationStateKey`, which is the
version-2 train/validation grouping key. Reproduction is bound by the root
seed, algorithm name/version, and SHA-256 of the algorithm contract. Derived
individual seeds are omitted explicitly, but they can be reconstructed from
the restricted root provenance, information-state key, accepted attempt, and
published derivation contract. The manifest states both facts and marks the
artifact `restricted-training-only`.

## Privacy boundary

The model-input fields in each NDJSON decision record contain only the encoded
observation, semantic action features, legal mask, public actor metadata, and
terminal outcomes. Opponent hands and physical card IDs are not included
directly in those fields. The actor's own rank counts are already part of the
approved non-card observation schema.

This is a model-feature information boundary, not anonymization of the dataset.
Version 1 intentionally includes per-match replay identity. Version 2 removes
per-world replay seeds and individual outcomes, retaining only collection and
determinization root seeds plus algorithm provenance. A matching deterministic
simulator can still reconstruct the run from that root provenance. Both
versions are therefore restricted training artifacts: do not publish them,
attach them to public releases, or treat them as safe telemetry. Store and
transfer them only with the same access controls as other private training
artifacts.

## Usage

Run the script directly with Node 22 or newer:

```powershell
node scripts/rl-generate-non-card-counterfactuals.mjs `
  --decision all `
  --players 4,5,6,7,8,9,10 `
  --episodes 10 `
  --acts 3 `
  --seed 710001 `
  --temperature 1 `
  --hidden-worlds 1 `
  --continuations 1 `
  --output artifacts/rl/non-card-counterfactuals-run-001.ndjson
```

Options:

- `--decision tax-return`, `revolution`, a comma-separated pair, or `all`;
- `--players` is a unique comma-separated subset of 4 through 10;
- `--episodes` is the number of baseline matches per player count;
- `--acts` controls the complete match horizon;
- `--seed` starts a collision-free sequential seed range;
- `--temperature` controls the soft target distribution;
- `--hidden-worlds K` selects the total conditional worlds per information
  state, including the original replay;
- `--continuations C` selects paired downstream RNG streams per hidden world;
- `--determinization-root-seed` controls hidden-world and continuation seed
  derivation without exposing derived seeds;
- `--max-determinization-attempts` is the finite rejection cap per additional
  hidden world;
- `--tax-return-count all`, `1`, `2`, or `1,2` filters tax pilots. A non-`all`
  value requires `--decision tax-return`;
- `--max-decisions` stops only between complete decision batches;
- `--created-at` makes the sole time-varying manifest field injectable for
  byte-for-byte reproducibility tests;
- `--output` must not exist.

The collector also claims `<output>.sha256` exclusively and writes the
conventional `<digest>  <filename>` line after the NDJSON closes. Either an
existing dataset path or existing checksum path rejects the fresh run. Validate
the sidecar before moving a dataset into a GPU bundle.

Resume and overwrite are intentionally unsupported. A failed run leaves its
exclusive output and checksum paths claimed. Depending on whether failure came
before or after the NDJSON closed, the dataset may lack its final summary or may
already be structurally complete; the checksum sidecar remains incomplete,
empty, or otherwise unverified. In every case, discard both claimed paths and
restart at a new path.

## Format and verification

The first NDJSON line is a versioned manifest. Version 2 uses the
`world-clustered-paired-baseline-advantages-v2` determinization schema. Each
middle line is one complete
paired decision and contains all legal action outcomes. The final summary
reports baseline-match, decision, and forced-action counts by decision kind and
player count. Its SHA-256 covers the exact UTF-8 manifest and decision-record
bytes, including newlines. The CLI also prints the SHA-256 of the complete
file.

Collection cost is approximately:

```text
baseline matches + sum(legal root actions for every written decision)
```

For version 2, multiply each written decision's legal-root count by
`hidden-worlds * continuations` to obtain raw simulator work. This raw count
does not increase the independent statistical sample count beyond
`hidden-worlds`. Rejected hidden worlds add only their attempted root
evaluations before the strict pre-decision check aborts.

Revolution always costs two forced reruns. Tax is the expensive part: a
one-card return has one legal action per distinct held rank, while a two-card
return can have several dozen. A measured one-episode, three-act 4–10 sweep on
the current CPU wrote 30 decisions (537 forced reruns, 286 KB) in 4.9 seconds.
Runtime is close to linear in `actionEvaluations`, so ten episodes per count is
roughly 50 seconds on the same machine before filesystem and workload variance;
100 episodes is roughly 8–10 minutes. Use `--max-decisions` for a fresh pilot
when acts or hardware change. Collection is CPU-only and streams records, so
memory use stays bounded by one match and one decision.
