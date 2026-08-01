# V3 legacy-policy distillation warm start

This training-only bridge turns the strongest available legacy 506-action
actor-critic into a strict 236-action V3 action-conditioned behavior model.
It does not import a model into either production bot.

## Contracts

- Teacher: `dalmuti-actor-critic` version 1, 172 observations, 506 actions.
- Student: `dalmuti-action-conditioned-actor-critic` version 1, observation
  schema 2, 172 observation features, catalogue version 1, 236 actions.
- Every legacy legal action is decoded semantically, mapped one-to-one, sorted
  in V3 catalogue order, and round-tripped back to the exact legacy set.
- Legality is independently reconstructed from the encoded private-hand/public
  observation before a sample is accepted.
- Targets contain the teacher's legal logits, temperature-2.5 distribution,
  deterministic V3 argmax, and value.
- The copied legacy critic is topology-identical and remains frozen, so value
  transfer is lossless. The new shared V3 action scorer learns the policy
  distribution.
- The train/validation partition is a SHA-256 assignment of
  `sourceSha256:episodeId`; an episode can never enter both partitions.
- Dataset, teacher, every epoch PT/JSON pair, selected-best model, result ZIP,
  and checksum are all bound and verified.
- Dataset and output directories must be new. Reusing any completed run path is
  rejected.

The distillation dataset contains no opponent hidden hand. It does contain the
acting player's private observation and is therefore a restricted training
artifact, not a public downloadable dataset.

## Exact full local warm-start command

The following command uses 20,000 non-forced samples from each player count
(140,000 total), all seven PPO5 rollout files as observation sources, PPO4 as
the teacher, and temperature 2.5. Change the `run-001` suffix for every new
attempt; do not overwrite or reuse it.

```powershell
$repo = (Resolve-Path ".").Path
$python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = Join-Path $repo "artifacts\rl\python-deps"
Set-Location (Join-Path $repo "gpu-training")

& $python prepare_v3_distillation_data.py `
  --rollout `
    ..\artifacts\rl\ppo-iteration-5-mc-temp25\rollouts\ppo-i5-mc-temp25-p4.ndjson `
    ..\artifacts\rl\ppo-iteration-5-mc-temp25\rollouts\ppo-i5-mc-temp25-p5.ndjson `
    ..\artifacts\rl\ppo-iteration-5-mc-temp25\rollouts\ppo-i5-mc-temp25-p6.ndjson `
    ..\artifacts\rl\ppo-iteration-5-mc-temp25\rollouts\ppo-i5-mc-temp25-p7.ndjson `
    ..\artifacts\rl\ppo-iteration-5-mc-temp25\rollouts\ppo-i5-mc-temp25-p8.ndjson `
    ..\artifacts\rl\ppo-iteration-5-mc-temp25\rollouts\ppo-i5-mc-temp25-p9.ndjson `
    ..\artifacts\rl\ppo-iteration-5-mc-temp25\rollouts\ppo-i5-mc-temp25-p10.ndjson `
  --teacher-model ..\artifacts\rl\returned\ppo-iteration-4\actor-critic-weights.json `
  --output-dir ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\dataset `
  --temperature 2.5 `
  --max-samples-per-source 20000 `
  --batch-size 1024 `
  --device cpu

& $python train_v3_distillation.py `
  --data ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\dataset\v3-distillation.ndjson `
  --teacher-model ..\artifacts\rl\returned\ppo-iteration-4\actor-critic-weights.json `
  --output ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\model `
  --epochs 50 `
  --batch-size 512 `
  --learning-rate 0.0003 `
  --weight-decay 0.00001 `
  --value-coefficient 0.25 `
  --validation-fraction 0.15 `
  --split-seed 20260801 `
  --seed 202608071 `
  --patience 8 `
  --max-gradient-norm 1 `
  --binding-tolerance 0.00002 `
  --device cpu `
  --deterministic

& $python package_v3_distillation_results.py `
  --result-dir ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\model `
  --output ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\package\v3-warmstart-result.zip `
  --teacher-model ..\artifacts\rl\returned\ppo-iteration-4\actor-critic-weights.json `
  --data ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\dataset\v3-distillation.ndjson

& $python verify_v3_distillation_results.py `
  --archive ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\package\v3-warmstart-result.zip `
  --checksum ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\package\v3-warmstart-result.zip.sha256 `
  --extract-dir ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\verified `
  --teacher-model ..\artifacts\rl\returned\ppo-iteration-4\actor-critic-weights.json `
  --data ..\artifacts\rl\v3-warmstart-distill-ppo4-t25-seed-202608071-run-001\dataset\v3-distillation.ndjson
```

The selected portable behavior model is:

```text
artifacts/rl/v3-warmstart-distill-ppo4-t25-seed-202608071-run-001/model/v3-actor-critic-weights.json
```

Pass that file directly to `pnpm rl:v3-ppo-rollouts -- --model ...` or to
`scripts/rl-prepare-v3-ppo-bundle.mjs --model ...`.

## Prepared full GPU handoff (run-004)

The full CPU-side preparation used the fresh local run
`v3-warmstart-distill-ppo4-t25-seed-202608071-run-002`. It generated and then
independently reloaded all 140,000 records before packaging. The final bundle
was extracted into a different fresh directory and its verifier recomputed all
teacher logits, temperature-2.5 probabilities, argmax actions, values, legal
masks, semantic legacy/V3 round trips, and the episode-group split.

- teacher: 2,732,440 bytes, SHA-256
  `3a8bc15ee05305e4cd8f9e6710cb8e927a54e0a3acf6ae0927ffabe50318535f`;
- dataset: 220,218,496 bytes, SHA-256
  `cac0fa2c98592c48c3f0fffe94a77f193f0e56d833fd794c7d056b7afeb373bb`;
- dataset contract: 140,000 non-forced samples, exactly 20,000 each for
  player counts 4 through 10, 1,793 episode groups;
- fixed split: 117,737 train samples in 1,510 groups and 22,263 validation
  samples in 283 groups, with zero overlap;
- handoff ZIP: 24,912,510 bytes, SHA-256
  `aa420d2062091a75020dbd10547e16ab7a2d28801cd152bc647370e4c231a3f0`;
- bundle manifest SHA-256:
  `315f1461344642c48420b572d209d5091d088ef8eb90aea514b45b15557be5f0`;
- GPU run config SHA-256:
  `a61381ce43aa47f7e2f5e8e2433ed29abcd137e9c863d20156e50ec405eb0461`;
- `handoff-files.sha256` SHA-256:
  `4d2854a0d669a89cf639c43e1b89f896c7048cfbd3a43bf3c07c64f0e4249516`;
- SHA-256 of the fourteen sorted `code/` checksum lines joined with a final
  newline:
  `8fc7c5b559b0d8a7a0169eb559942b5e02b26ec9c974df7104d5ba36e62791f8`.

The handoff files are:

```text
artifacts/rl/v3-warmstart-distill-ppo4-t25-seed-202608071-run-002/dalmuti-v3-warmstart-distill-ppo4-t25-seed-202608071-gpu-handoff-run-004.zip
artifacts/rl/v3-warmstart-distill-ppo4-t25-seed-202608071-run-002/dalmuti-v3-warmstart-distill-ppo4-t25-seed-202608071-gpu-handoff-run-004.zip.sha256
```

The adjacent checksum was independently compared with the ZIP after creation.
The embedded prompt uses fresh remote bundle/work directories, requires CUDA
and deterministic PyTorch algorithms, records Python/PyTorch/CUDA/cuDNN/GPU
identity, and rejects missing, extra, or symbolic-link result files. Exact
bitwise identity across different hosts still requires matching hardware and
library versions.

The default result verifier requires the strict inventory. Legacy local
artifacts without it are accepted only through the explicit
`--allow-legacy-inventory` opt-in. GPU packaging and verification must pass
`--expected-handoff <extracted-run-004-root>`; that path additionally binds all
five provenance files, manifest/config hashes and run ID, CUDA identity,
deterministic/CUBLAS settings, every fixed training argument, and the fixed
train/validation sample and group counts. The embedded GPU prompt already uses
this strict path for both packaging and post-package verification.

Handoffs generated after the bytecode-cache hardening export
`PYTHONDONTWRITEBYTECODE=1` before the first Python process. This prevents test,
verification, training, and packaging imports from adding `code/__pycache__`
to the immutable bundle; the verifier continues to reject every unmanifested
file rather than permitting cache files.

## Verification commands

```powershell
& $python -m unittest `
  test_v3_action_conditioned.py `
  test_v3_ppo_pipeline.py `
  test_v3_distillation_pipeline.py
```

The focused real-data smoke used 256 PPO5 p4 observations, three CPU epochs,
and a completely fresh run directory. This is an interface smoke, not a
strength-qualified model:

- teacher SHA-256:
  `3a8bc15ee05305e4cd8f9e6710cb8e927a54e0a3acf6ae0927ffabe50318535f`;
- dataset SHA-256:
  `7372bb5d1b62e98411f0118007d03f097216f2f869c16b6360340858aa155c72`;
- final model SHA-256:
  `255feae7d06659dfba83bb89980242c52ef645a4c3de36d6944a3b9113a6cd06`;
- result ZIP SHA-256:
  `19f1a3ada927ea853526cc460e8ee6e1fa346e4f16dbc31b48f5508a5f8bd784`;
- best epoch: 3; validation policy KL: `0.885084`; teacher-argmax
  agreement: `51.61%`; critic RMSE: exactly `0`;
- the exported JSON then collected one real four-player V3 act: 51 learner
  samples, 27 non-forced, with observation, 236-action legal mask,
  log-probability, and value bindings all independently verified;
- `train_v3_ppo.py` accepted that same JSON and completed a one-epoch CPU PPO
  update over the verified rollout, proving the full `--behavior-model`
  interface rather than only the JSON parser.

The smoke metrics are deliberately not a promotion claim. Use the full run,
screen the distilled model against PPO4/normal, and then begin V3 PPO from the
verified selected-best JSON.
