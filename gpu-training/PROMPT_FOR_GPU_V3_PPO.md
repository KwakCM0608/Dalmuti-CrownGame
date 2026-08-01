# DALMUTI V3 PPO GPU handoff

Work only inside this newly extracted bundle directory. Do not reuse an old
model or result directory and do not edit the input data or behavior model.

Before running any Python process inside the bundle, set and retain this
environment guard for the entire shell session:

```bash
export PYTHONDONTWRITEBYTECODE=1
test "$PYTHONDONTWRITEBYTECODE" = 1
```

1. Run `python verify_bundle.py`.
2. Run `python -m unittest test_v3_action_conditioned.py test_v3_ppo_pipeline.py
   test_v3_ppo_result_contract.py`.
3. Read `gpu-run-config.json`; its parent SHA-256, complete `algorithm`,
   `determinism`, `pathPolicy`, and allowed rank-auxiliary variants are
   mandatory machine bindings. Do not change any fixed algorithm value.
4. Run `python run_gpu_v3_ppo.py` with all arguments listed in
   `requiredCommandArguments`, replacing both `<fresh-v3-run>` placeholders
   with one new safe run-specific name and choose the instructed `0` or `0.05`
   rank-auxiliary branch. The two paths must be direct children of `models/`
   and `returned/` and must use the same run ID.
5. Do not retry in the same output or results path. On failure, use a new
   directory name.
6. Do not edit, move, replace, or symlink the behavior model, rollout data,
   bundle manifest, or run config. The runner checks their hashes throughout.
7. Return only the result ZIP and adjacent `.zip.sha256` after the runner
   succeeds. Do not manually change or repackage them.

The runner must not launch `verify_v3_ppo_data.py` as a separate preprocessing
pass. The standalone verifier remains in the bundle only for backward
compatibility and diagnostics; do not run it during the standard handoff.
Instead, the single dataset load inside `train_v3_ppo.py` must recompute and
verify the observation, catalogue, legal-mask, behavior log-probability, and
critic-value bindings on the selected CUDA training device in batches of 8192,
while the seven rollout files are parsed with the bound seven-worker loader,
then exclusively write `<fresh output>/data-verification.json` before the first
optimizer update. The report format and version must remain unchanged.
