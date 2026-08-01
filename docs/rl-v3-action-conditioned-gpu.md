# V3 action-conditioned play-policy GPU handoff

V3 is an isolated fallback for the card-play PPO path. It does not change the
legacy V1/V2 behavior-cloning or 506-action PPO files and it is not imported by
the production bot.

## Runtime contract

- observation schema: version 2, 172 features;
- action catalogue: version 1, exactly 236 structurally possible actions;
- action features: 22, with the complete catalogue and feature table embedded
  in every rollout manifest;
- legal mask: exactly 59 lowercase hexadecimal digits per sample. Action `i`
  is bit `i % 4` of digit `floor(i / 4)`;
- behavior binding: model SHA-256, `policyVersion`, observation and catalogue
  versions, observation-derived legality, selected action, log probability,
  and value estimate are all checked before training;
- rollout-semantics binding: the verifier and `v3-ppo-schema.json` share the
  canonical contract SHA-256
  `d7d249a24153ecc204add53f3d3ab352fabfd5ab175001b51f8ed2ba1296e275`.
  It fixes the game and house-rule identifiers, normal-policy non-card
  decisions, league seat policy, observation privacy/encoder, trajectory and
  terminal meanings, chip-award reward formula, and summary-count meanings;
- provenance and count binding: UTC collection time, initial seed, sequential
  episode IDs, acts, learner-seat sets, trajectory terminals, step ordering,
  learner/forced/non-forced totals, opponent-seat totals, temperature, and
  collection targets are checked. `environmentDecisions` includes hidden
  opponent decisions and therefore cannot be reconstructed exactly from the
  learner-only records; it is source-declared, but must cover every sampled
  step and remain below the simulator's declared per-act transition cap;
- reward binding: every non-terminal reward must be zero, and each terminal
  reward is recomputed from `finishPlace` and `playerCount` using the project
  chip curve. Finish places and terminal placement are also checked per
  trajectory;
- observation binding: all 172 values must be finite and canonical. Player
  count, act number, actor seat/role, table state, physical-card counts,
  relative public-player slots, score encoding, role order, and revolution
  one-hot state are range- and consistency-checked where the public encoding
  permits reconstruction;
- the GPU run config binds every optimizer/PPO argument, seed, CUDA device,
  deterministic-algorithm setting, and the two allowed terminal-rank A/B
  coefficients (`0` and `0.05`);
- output must be a new direct child of `models/`, results must be a new direct
  child of `returned/`, and both must use the same safe run ID. Absolute paths,
  traversal, symbolic links, containment, overlap, or reused paths are rejected;
- every manifest-bound bundle source (including the exact behavior model and
  every rollout), plus the bundle manifest itself, is hash-checked before the
  output directory is created and again between every stage. Input mutation
  aborts the run;
- result ZIPs, checksum files, and extraction directories must not exist before
  a run.

The Python verifier reconstructs legal actions from the private-hand and table
features in each encoded observation. It then loads the exact behavior-model
JSON and recomputes the masked softmax log probability at the manifest
temperature and the critic value. The default absolute cross-runtime tolerance
is `2e-5` (TypeScript float64 inference versus PyTorch float32 inference).
The loader additionally rejects any non-finite raw or computed float32 array;
the CLI writes a verification artifact only after that check, and exits
nonzero on every contract failure. Verification report version 2 includes
`sourceFiles`, an ordered list of absolute path, byte length, and lowercase
SHA-256 for every loaded rollout. Each source must be a stable regular,
non-symlink file and its hash must remain unchanged across both loader passes,
behavior-model verification, and the final report hash.

## Local rollout and bundle commands

The input model must already have format
`dalmuti-action-conditioned-actor-critic`, observation schema `2`, 172
observation features, catalogue version `1`, and 236 actions.

```powershell
pnpm rl:v3-ppo-rollouts -- `
  --model artifacts/rl/<v3-parent>/v3-actor-critic-weights.json `
  --players 4 `
  --acts 3 `
  --temperature 2.5 `
  --target-non-forced-decisions 150000 `
  --seed 861000001 `
  --output artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p4.ndjson
```

Repeat with fresh output files for player counts 5 through 10 and distinct
seeds. Older V3 models can be supplied repeatedly with `--opponent-model`; if
none are supplied all opponent seats use the current `normal` heuristic.

```powershell
pnpm rl:v3-ppo-bundle -- `
  --model artifacts/rl/<v3-parent>/v3-actor-critic-weights.json `
  --rollout artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p4.ndjson `
  --rollout artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p5.ndjson `
  --rollout artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p6.ndjson `
  --rollout artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p7.ndjson `
  --rollout artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p8.ndjson `
  --rollout artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p9.ndjson `
  --rollout artifacts/rl/<fresh-v3-local-run>/rollouts/v3-p10.ndjson `
  --output artifacts/rl/<fresh-v3-local-run>/gpu-bundle
```

The resulting directory is the exact GPU bundle input. Archive that directory
without changing its contents; `verify_bundle.py` checks every listed byte and
SHA-256 on the GPU computer.

## Exact GPU command

Run this inside a newly extracted bundle directory. Both output arguments must
name paths that do not exist.

```bash
python -m unittest test_v3_action_conditioned.py test_v3_ppo_pipeline.py \
  test_v3_ppo_result_contract.py
python run_gpu_v3_ppo.py \
  --data data/*.ndjson \
  --behavior-model behavior-model.json \
  --output models/v3-ppo-i1-rank0-run-001 \
  --results-dir returned/v3-ppo-i1-rank0-run-001 \
  --epochs 12 \
  --batch-size 4096 \
  --learning-rate 0.0001 \
  --weight-decay 0.00001 \
  --gamma 1 \
  --gae-lambda 1 \
  --skip-forced-policy-time \
  --terminal-rank-auxiliary-coefficient 0 \
  --rollout-temperature 2.5 \
  --clip-coefficient 0.2 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0.01 \
  --max-gradient-norm 0.5 \
  --target-kl 0.015 \
  --loader-workers 7 \
  --behavior-binding-batch-size 8192 \
  --binding-tolerance 0.00002 \
  --seed 202608061 \
  --device cuda
```

Use `0.05` instead of `0` only for the paired rank-auxiliary branch. All other
arguments are immutable. The runner launches every child process with
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=202608061`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, enables strict PyTorch deterministic
algorithms, disables cuDNN benchmarking and TF32, and records the seed,
deterministic flags, CUDA/PyTorch/cuDNN versions, and exact GPU identity in the
returned metadata. The bundle prompt exports the bytecode guard before its
first Python process; an unmanifested `__pycache__` remains a hard source
inventory failure. A failed attempt consumes its run ID; retry only with a
completely new output/results name.

Return the generated `*-result.zip` and `*.zip.sha256`. Verify them locally
before screening:

```bash
python verify_v3_ppo_results.py \
  --archive <fresh-run>-result.zip \
  --checksum <fresh-run>-result.zip.sha256 \
  --expected-bundle-manifest <original-gpu-bundle>/bundle-manifest.json \
  --expected-run-config <original-gpu-bundle>/gpu-run-config.json \
  --extract-dir <fresh-local-extraction>
```

## Bootstrap from the legacy champion

The missing bootstrap bridge is implemented in
`docs/rl-v3-distillation-warmstart.md`. It semantically distills the PPO4
506-action teacher over existing PPO5 observations into a strict 236-action
V3 behavior JSON, preserves the teacher critic exactly, partitions by source
episode, exports every epoch, and verifies the portable result package and
SHA-256. A real-data CPU smoke also passed the V3 rollout verifier. Run the
documented full warm start before collecting the first strength-bearing V3 PPO
iteration; the tiny smoke model is only contract evidence.
