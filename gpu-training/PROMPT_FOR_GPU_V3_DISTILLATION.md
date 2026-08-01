# DALMUTI V3 distillation GPU handoff

Use only the newly extracted bundle and a newly allocated work directory.
Never edit anything under `input/`, `bundle-manifest.json`,
`gpu-run-config.json`, or `handoff-files.sha256`. Never reuse a failed or
completed attempt directory.

The bundle is bound to:

- legacy PPO4 teacher SHA-256
  `3a8bc15ee05305e4cd8f9e6710cb8e927a54e0a3acf6ae0927ffabe50318535f`;
- 140,000 PPO5 observations, exactly 20,000 for every player count 4 through
  10;
- observation schema 2 with 172 features;
- the exact 236-action V3 catalogue version 1;
- teacher softmax temperature 2.5;
- episode/world split seed 20260801 and training seed 202608071.

Upload the ZIP and adjacent `.zip.sha256` to
`/home/pangmin/dalmuti/incoming/`. Run these commands exactly. The loop
automatically chooses a fresh attempt suffix without changing the logical run
ID recorded in the immutable handoff.

```bash
set -Eeuo pipefail

ROOT=/home/pangmin/dalmuti
PY="$ROOT/gpu-bundle-v3/.venv/bin/python"
INCOMING="$ROOT/incoming"
ARCHIVE="$INCOMING/dalmuti-v3-warmstart-distill-ppo4-t25-seed-202608071-gpu-handoff-run-004.zip"
RUN_ID=v3-warmstart-distill-ppo4-t25-seed-202608071-gpu-run-001
export PYTHONDONTWRITEBYTECODE=1
test "$PYTHONDONTWRITEBYTECODE" = 1
ATTEMPT=1
while :; do
  TAG=$(printf '%s-attempt-%03d' "$RUN_ID" "$ATTEMPT")
  BUNDLE="$ROOT/$TAG-bundle"
  WORK="$ROOT/$TAG-work"
  if test ! -e "$BUNDLE" && test ! -e "$WORK"; then
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
done

cd "$INCOMING"
sha256sum -c "$(basename "$ARCHIVE").sha256"
test ! -e "$BUNDLE"
test ! -e "$WORK"
mkdir "$BUNDLE"
"$PY" -m zipfile -e "$ARCHIVE" "$BUNDLE"
mkdir "$WORK"
cd "$BUNDLE"

"$PY" code/verify_v3_distillation_bundle.py
"$PY" code/preflight.py --device cuda --output "$WORK/hardware-report.json"
(
  cd code
  "$PY" -m unittest test_v3_action_conditioned.py test_v3_distillation_pipeline.py
)

"$PY" code/train_v3_distillation.py \
  --data input/v3-distillation.ndjson \
  --teacher-model input/ppo4-actor-critic-weights.json \
  --output "$WORK/model" \
  --epochs 50 \
  --batch-size 512 \
  --learning-rate 0.0003 \
  --weight-decay 0.00001 \
  --value-coefficient 0.25 \
  --validation-fraction 0.15 \
  --split-seed 20260801 \
  --seed 202608071 \
  --patience 8 \
  --max-gradient-norm 1 \
  --binding-tolerance 0.00002 \
  --device cuda \
  --deterministic 2>&1 | tee "$WORK/training.log"

TRAIN_STATUS=${PIPESTATUS[0]}
test "$TRAIN_STATUS" -eq 0
mkdir "$WORK/model/provenance"
cp bundle-manifest.json handoff-files.sha256 gpu-run-config.json \
  "$WORK/hardware-report.json" "$WORK/training.log" \
  "$WORK/model/provenance/"
mkdir "$WORK/returned"
"$PY" code/package_v3_distillation_results.py \
  --result-dir "$WORK/model" \
  --output "$WORK/returned/$TAG-result.zip" \
  --teacher-model input/ppo4-actor-critic-weights.json \
  --data input/v3-distillation.ndjson \
  --expected-handoff "$BUNDLE"

"$PY" code/verify_v3_distillation_results.py \
  --archive "$WORK/returned/$TAG-result.zip" \
  --checksum "$WORK/returned/$TAG-result.zip.sha256" \
  --extract-dir "$WORK/verified-result" \
  --teacher-model input/ppo4-actor-critic-weights.json \
  --data input/v3-distillation.ndjson \
  --expected-handoff "$BUNDLE"
```

Return only the result ZIP and its adjacent `.zip.sha256`. The selected model
inside the archive is `v3-actor-critic-weights.json` and can be supplied
directly as the first V3 PPO `--behavior-model`.

If any command fails, preserve both failed attempt directories for diagnosis
and rerun the whole block. The allocator will select a different attempt. Do
not overwrite, delete, or resume within a failed attempt.
