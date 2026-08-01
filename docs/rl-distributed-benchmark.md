# Distributed RL benchmark contract

Development checkpoint evaluation can be split across candidates, player
counts, and deterministic match-seed shards. A shard preserves the exact
monolithic schedule:

```text
matchIndex belongs to shard when matchIndex % shardCount == shardIndex
matchSeed = baseSeed + playerCount * 1,000,000 + matchIndex
```

The global match index is also used for candidate-seat rotation. Therefore a
strictly merged result is numerically identical to a single-process run.

## One-host screening

`--benchmark-shards` splits every candidate and player count by match seed.
`--concurrency` is the process-pool limit across all candidate, player-count,
and seed jobs. The host-agnostic safety ceiling is 32 processes; use the
calibrated throughput optimum rather than treating that ceiling as a target.

```powershell
pnpm rl:screen-checkpoints -- `
  --directory <verified-result> `
  --output <fresh-screening-directory> `
  --players 4,5,6,7,8,9,10 `
  --match-counts 4:300,5:300,6:300,7:300,8:300,9:300,10:300 `
  --benchmark-shards 4 `
  --concurrency 8 `
  --seed-base <development-seed> `
  --reserved-final-seeds <reserved-final-seed>
```

Each shard JSON stores the behavior-model SHA-256, deterministic seed
provenance, and raw integer aggregation totals. Screening hashes every shard,
recomputes its summary, rejects contract drift, merges the complete seed set,
and only then ranks checkpoints.

## Two-host execution

Run `rl-benchmark-model` independently for each candidate/player/shard job on
either host. Model files may live at different absolute paths, but their bytes
must have the same SHA-256.

```powershell
pnpm rl:benchmark-model -- `
  --model <candidate-model> `
  --players 6 `
  --match-counts 6:300 `
  --acts 5 `
  --seed <development-seed> `
  --shard-index 0 `
  --shard-count 4 `
  --omit-match-data `
  --output <fresh-shard-json>
```

After transferring the independent JSON reports, merge one candidate with an
explicit plan:

```powershell
pnpm rl:merge-benchmark-shards -- `
  --model <local-byte-identical-candidate-model> `
  --players 4,5,6,7,8,9,10 `
  --match-counts 4:300,5:300,6:300,7:300,8:300,9:300,10:300 `
  --acts 5 `
  --seed <development-seed> `
  --reserved-final-seeds <reserved-final-seed> `
  --shard <report-1.json> `
  --shard <report-2.json> `
  --output <fresh-merged-report.json>
```

Merge is fail-closed for a changed model, incompatible thresholds, duplicate
shard index, missing shard, duplicate match index, missing match seed, summary
drift, or an explicitly reserved final seed. Input ordering does not change the
merged result.

## Two-host resource policy

Every new iteration uses both available hosts rather than leaving rollout or
evaluation work on a single machine:

- use the exact same Node.js runtime on both hosts (currently v24.14.0) and
  verify every transferred model and source snapshot by SHA-256;
- calibrate CPU concurrency with a short fixed-seed development run before the
  full job. The current starting values are 6 processes on the 12-logical-CPU
  local host and 10 processes on the 20-logical-CPU GPU host, but measured
  throughput decides the final values;
- dispatch benchmark shards through a shared queue so the faster host receives
  the next unfinished shard instead of waiting for a fixed half-and-half split;
- collect V3 rollouts as whole player-count files. Start with `p4`-`p6` on the
  local host and `p7`-`p10` on the GPU host, then move one whole file if the
  calibration shows that the opposite split finishes sooner;
- on the GPU host, use seven loader worker processes for strict NDJSON parsing
  and memmap filling, CUDA for behavior-model binding, and CUDA for PPO tensor
  optimization;
- keep at least 12 GiB of free space before a multi-worker load because its
  temporary memmaps are approximately the size of the decoded training arrays;
- use a fresh attempt directory and run ID for every collection, training, and
  evaluation run. Never resume by overwriting a partial directory.

The TypeScript game simulator is sequential and stateful, so GPU execution is
not currently faster for rollout generation or match simulation. Those stages
scale through independent CPU processes across both hosts. GPU acceleration is
reserved for large batched tensor work where it reduces elapsed time.

## Rollout boundary

V3 PPO rollout collection is distributed between hosts by whole player-count
files (`p4` through `p10`). Do not split one player count into multiple files:
the current strict Python contract requires sequential per-file episode IDs and
the GPU bundle requires exactly one rollout file for every player count. The
coordinator verifies transferred SHA-256 sidecars and the bundle verifier then
checks exactly seven contract-compatible files before training.
