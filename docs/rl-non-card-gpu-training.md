# Non-card counterfactual GPU training

The non-card pipeline trains two independent action-conditioned policies:

- `tax-return`: exactly 103 semantic return actions, masked by the noble's
  encoded pre-tribute hand;
- `revolution`: decline or declare, with the declare action feature conditioned
  on whether the actor is the Great Peon.

It is training-only. It does not modify the quick-match or online production
bot until a separately evaluated model is promoted.

## Data preflight

The input is the version 1
`dalmuti-non-card-counterfactual-ndjson` produced by
`scripts/rl-generate-non-card-counterfactuals.mjs`. The Python loader reads it
line by line and rejects:

- incomplete files without a final summary;
- a wrong content hash, count, schema, feature dimension, privacy declaration,
  observation-derived legal mask, action catalogue, or action feature;
- incomplete or duplicate legal-action coverage;
- non-finite values, inconsistent centered utilities or uncertainty, and soft
  probabilities that disagree with the manifest temperature;
- duplicate `sampleId` values across input files.

Run a preflight before using GPU time:

```bash
python verify_non_card_counterfactual_data.py \
  --data data/non-card-counterfactual-*.ndjson \
  --validation-fraction 0.2 \
  --split-seed 20260801 \
  --json
```

Train/validation assignment hashes a canonical world key containing
`(playerCount, acts, matchSeed, continuationPolicy)`, not an individual
decision or a collector-local episode label. Consequently every tax and
revolution decision from the same hidden world remains in one partition even
across separately named collection runs. Reusing a canonical world in two
input files is rejected instead of double-counting it. All files must use the
same act horizon. The dataset manifest and training metrics record
`groupSplitKey: "canonicalWorldKey"`, the split seed, sample counts, unique
world and episode counts, and sorted sample-ID hashes.

Without a temperature override, all source files must also use the same policy
temperature. Mixed source temperatures are accepted only when
`--policy-temperature` is present, because that causes every target
probability and corresponding expected value target to be recomputed from the
validated utilities.

## Objectives

Every state contains exhaustive outcomes for all legal root actions. Training
uses that coverage directly:

1. confidence-weighted soft-target policy cross entropy (equivalent to the
   corresponding weighted KL objective up to fixed target entropy);
2. confidence-weighted Huber regression from mean-centered legal actor scores
   to `centeredUtility`;
3. Huber regression from the value head to the soft target policy's expected
   terminal cumulative chip score.
4. optionally, behavior-cloning NLL on the collector's
   `baselineActionIndex`, anchoring the actor to the deterministic current
   normal policy while noisy one-world labels are explored.

The per-action confidence is `1 / (1 + standardError^2)`. A record's sample
weight is `sqrt(targetSampleCount)`. Illegal actions receive no policy or value
loss. Metrics include policy KL, best-action accuracy, and chosen-action
counterfactual regret in addition to all loss terms.

The default coefficients are policy `0.5`, centered action value `1.0`, and
state value `0.25`. Current records contain one paired hidden world, so their
soft policy is a temperature-shaped preference from a noisy single-world
sample and reports zero within-record standard error. The exhaustive centered
action differences still carry the most direct within-world ranking signal;
they are therefore the primary objective. Policy imitation remains a useful
distributional regularizer, and state value remains auxiliary. Multi-world
aggregates with meaningful uncertainty can justify retuning these coefficients
in a later ablation.

Behavior cloning defaults to `0.0`, preserving the pre-anchor objective and
reproducibility. Fresh, otherwise identical ablations can use:

```bash
--behavior-cloning-coefficient 0.5
--behavior-cloning-coefficient 1
--behavior-cloning-coefficient 2
```

The coefficient contributes
`coefficient * NLL(model, baselineActionIndex)` to
`actorSelectionLoss`, so it affects both optimization and actor-checkpoint
selection. Metrics report behavior-cloning loss, model/baseline action
agreement, the fixed rate at which the target best action equals baseline,
predicted logit/probability margins over baseline, and the chosen action's
target-utility margin over baseline. Config, PT checkpoints, training metrics,
result manifest, and package manifest all bind the coefficient.

By default the trainer uses each record's `softTargetProbability`, which the
loader has checked against its manifest temperature. Expensive simulations can
be reused for a temperature ablation:

```bash
--policy-temperature 0.5
--policy-temperature 1
--policy-temperature 2
```

An override recomputes legal soft targets from `centeredUtility`; it also
recomputes the value target. The chosen temperature and all source manifest
temperatures are written to config, metrics, dataset binding, and result
manifest files.

The utility horizon is independently selectable:

```bash
--utility-target terminal
--utility-target decision-act
```

`terminal` is the default and preserves all historical behavior. In
`decision-act` mode the loader retains every action's normalized immediate act
reward and verifies it exactly against
`(roundChipAward(finishPlace, playerCount) - 2) / 2`. For each decision, the
trainer then centers those immediate utilities over legal actions, recomputes
the soft policy using the requested temperature (or that record's source
temperature), recomputes the catalogue-first best action, and sets the state
value target to the policy-weighted expected immediate utility. BC anchoring
and actor checkpoint selection are unchanged.

Utility-target experiments use result/checkpoint/package contract version 3,
and `utilityTarget` is bound in config, dataset metadata, every PT checkpoint,
training metrics, result manifest, and package manifest. Versions 1 and 2 are
normalized to `terminal` by the verifier.

## Fresh GPU run

Always create a new work and output directory. The trainer claims `--output`
exclusively and never resumes or overwrites it. The ZIP and checksum paths are
also exclusive.

From the `gpu-training` directory:

```bash
python run_gpu_non_card_counterfactual.py \
  --data /new/work/data/non-card-counterfactual-*.ndjson \
  --output /new/work/non-card-models \
  --result-zip /new/work/non-card-models-result.zip \
  --decision all \
  --epochs 500 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --behavior-cloning-coefficient 1 \
  --utility-target decision-act \
  --early-stopping-patience 30 \
  --validation-fraction 0.2 \
  --split-seed 20260801 \
  --seed 20260801 \
  --device cuda
```

`--decision tax-return` and `--decision revolution` support isolated ablations.
Training is reproducibly seeded, uses deterministic Torch algorithms by
default, saves PT plus portable JSON for every epoch, and selects each
decision's actor checkpoint independently. The selection/early-stopping metric
is `validation.actorSelectionLoss` (policy + centered action value + optional
behavior-cloning NLL - entropy),
because production consumes actor argmax and not the state-value prediction.
Total loss and value loss are still trained and recorded; metrics separately
report the actor-best epoch, value loss at that epoch, and the independently
best value epoch/loss. Early stopping never removes completed checkpoints.

The 500-epoch, batch-256 defaults are deliberate for the current roughly
5,800-decision dataset: the earlier 50-epoch/batch-1024 pilot provided too few
optimizer updates and was still improving at epoch 50. Patience 30 permits a
meaningful plateau while retaining every epoch for post-hoc inspection.

## Result layout

```text
non-card-models/
  training-config.json
  dataset-manifest.json
  training-metrics.json
  training-manifest.json
  tax-return/
    metrics.json
    best/checkpoint.pt
    best/model.json
    checkpoints/epoch-001/checkpoint.pt
    checkpoints/epoch-001/model.json
    checkpoints/epoch-001/metrics.json
    ...
  revolution/
    ...
```

`training-manifest.json` binds every file by canonical relative path, byte
count, and SHA-256. Result verification checks all hashes, config/metrics/split
metadata, consecutive epochs, checkpoint schemas, portable JSON model schemas,
and numerical PT/JSON weight parity.

Historical anchored results used contract version 2; current utility-bound
results use version 3. Verifiers retain read compatibility with both
historical version-1 layouts: the original total-loss-selected summary and the
later actor-loss-selected summary. Missing historical BC metadata is
interpreted strictly as coefficient `0.0`, and versions 1/2 normalize the
utility target to `terminal`.

Verify an unpacked run:

```bash
python verify_non_card_results.py --result-dir /new/work/non-card-models
```

Package separately if the training-only command was used:

```bash
python package_non_card_results.py \
  --result-dir /new/work/non-card-models \
  --output /new/work/non-card-models-result.zip
```

This creates `non-card-models-result.zip` and
`non-card-models-result.zip.sha256`. Packaging first runs the full directory
verifier, writes a second hash manifest inside the ZIP, reopens and verifies
every compressed entry, extracts into a temporary directory, and repeats the
full schema/PT/JSON verification before writing the checksum sidecar.

Verify a downloaded archive before extraction or remote cleanup:

```bash
python package_non_card_results.py \
  --verify-archive non-card-models-result.zip
```

Only after the local ZIP, checksum, internal hashes, and model schemas all pass
should the exact completed remote run directory be deleted.

## CPU tests

The pipeline test uses tiny synthetic exhaustive datasets and CPU Torch:

```bash
python -m unittest -v \
  test_non_card_action_conditioned.py \
  test_non_card_counterfactual_pipeline.py
```

It covers strict streaming validation, canonical-world no-leakage splits,
temperature override consistency, exhaustive losses and gradients,
BC-anchor loss/metrics, deterministic replay, early stopping/checkpoint
outputs, legacy/current directory verification, exclusive ZIP/checksum
creation, archive extraction verification, and refusal to overwrite completed
output.
