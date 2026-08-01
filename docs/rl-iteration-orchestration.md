# RL iteration orchestration

`rl-prepare-next-ppo.mjs` creates one self-contained local run directory. A
path that already exists, even if empty, is never reused, so a failed or
completed experiment cannot be overwritten accidentally. Add `--run-label`
when the same numerical iteration has multiple variants.

## Validate before collecting

Dry-run resolves every model, hashes it, validates the run name and collection
settings, checks the destination, and prints the exact seed schedule without
creating files.

```powershell
pnpm rl:ppo-prepare -- --model artifacts/rl/returned/ppo-iteration-4/actor-critic-weights.json --iteration 5 --run-label temperature-125 --temperature 1.25 --episodes-by-player 4=900,5=500,6=400,7=350,8=300,9=300,10=300 --seed 700001 --dry-run
```

Remove `--dry-run` to collect data. The result is written under
`artifacts/rl/ppo-iteration-5-temperature-125/`; choose a new iteration or
label for every subsequent attempt.

For balanced policy decisions instead of fixed episode counts, replace
`--episodes-by-player` with `--target-non-forced-decisions`. Collection stops
after a complete episode once every player-count rollout reaches the target.

```powershell
pnpm rl:ppo-prepare -- --model artifacts/rl/returned/ppo-iteration-4/actor-critic-weights.json --iteration 5 --run-label balanced --temperature 1.25 --target-non-forced-decisions 150000 --max-episodes 100000 --seed 700001
```

The base seed and each derived player-count seed are stored in
`run-manifest.json`. Given the same model hashes and arguments, simulation and
opponent assignment are reproducible. Opponent models are selected uniformly
within the non-`normal` share:

```powershell
pnpm rl:ppo-prepare -- --model path/to/champion.json --opponent-model path/to/history-a.json --opponent-model path/to/history-b.json --normal-opponent-fraction 0.75 --iteration 6 --target-non-forced-decisions 150000 --temperature 1.25 --seed 800001
```

## Handoff contract

Each completed run contains:

- `run-manifest.json`: parent/opponent hashes, full collection configuration,
  seed schedule, actual data counts, bundle hash, and archive hash.
- `gpu-bundle/bundle-manifest.json`: forward-slash file paths, per-rollout
  opponent mix and actual seat assignments, sample counts, and SHA-256 values.
- `gpu-bundle/gpu-run-config.json`: the mandatory matching
  `--rollout-temperature` argument for `run_gpu_ppo.py`.
- `dalmuti-ppo-iteration-...-gpu.zip` and `.sha256`: a streaming portable ZIP;
  entries use `/` on Windows and Unix.

On the GPU host, extract every archive into a new directory such as
`/home/pangmin/dalmuti/ppo-iteration-5-balanced-work`. Only the shared Python
environment may be reused. Read `gpu-run-config.json` and pass its exact
`--rollout-temperature` value when running `run_gpu_ppo.py`; the loader rejects
a mismatch. Remote cleanup is deliberately outside these local scripts and is
only performed after the returned result has been downloaded and verified.
