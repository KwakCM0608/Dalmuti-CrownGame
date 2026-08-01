# DALMUTI RL experiment ledger

This ledger records development evidence. It is not a promotion report and it
does not relax the final gates in `scripts/rl-evaluation-statistics.mjs`.
Production bot code remains unchanged until one play policy passes every gate
and the human blind A/B check.

## Final automatic gates

Every player count from 4 through 10 must independently satisfy all three:

- mean chip difference versus current `normal` at least `+0.25` per act;
- match-clustered 95% confidence lower bound at least `+0.15`;
- candidate-before-normal pairwise finish rate at least `55%`.

The descriptive role audit and non-card marginal diagnostics do not replace
these gates.

## Play-policy lineage

### PPO iteration 4

- actor JSON:
  `artifacts/rl/returned/ppo-iteration-4/actor-critic-weights.json`
- SHA-256:
  `3a8bc15ee05305e4cd8f9e6710cb8e927a54e0a3acf6ae0927ffabe50318535f`
- strengthened screen: 300 matches x 5 acts per player count
- result: failed every final player-count gate
- observed mean chip differences ranged from `+0.0538` to `+0.1372`; all
  were below the required `+0.25`, and no confidence lower bound reached
  `+0.15`.

### PPO iteration 5 rollout bundle

- directory: `artifacts/rl/ppo-iteration-5-mc-temp25`
- parent: PPO iteration 4
- opponent mixture: current `normal` 75%, PPO iteration 3 25%
- behavior temperature: `2.5`
- rollout seed: `202608060`
- samples: 2,885,942 total decisions; 1,050,237 non-forced decisions
- ZIP SHA-256:
  `effb779537631ace332072736ca21f902add9096d5ef87682e49bbfbc4b407d2`
- status: exact rank-auxiliary A/B GPU training is pending GPU-host recovery.

### V3 action-conditioned warm start

- action catalogue: version 1, 236 semantic actions
- teacher: PPO iteration 4 at temperature `2.5`
- teacher SHA-256:
  `3a8bc15ee05305e4cd8f9e6710cb8e927a54e0a3acf6ae0927ffabe50318535f`
- distillation data: 140,000 non-forced PPO5 observations, exactly 20,000
  for every player count from 4 through 10
- data SHA-256:
  `cac0fa2c98592c48c3f0fffe94a77f193f0e56d833fd794c7d056b7afeb373bb`
- episode groups: 1,793 total; 1,510 train and 283 validation with zero
  group overlap
- strict run-004 GPU handoff ZIP SHA-256:
  `aa420d2062091a75020dbd10547e16ab7a2d28801cd152bc647370e4c231a3f0`
- bundle manifest SHA-256:
  `315f1461344642c48420b572d209d5091d088ef8eb90aea514b45b15557be5f0`
- status: the independent verifier audit is complete. The return verifier now
  hard-binds the immutable handoff, CUDA/GPU identity, deterministic runtime,
  exact training arguments, and fixed episode-group split. The fresh run-004
  ZIP passed a full 140,000-sample revalidation; GPU training is pending host
  recovery.

### V3 PPO execution contract

- rollout semantics contract SHA-256:
  `d7d249a24153ecc204add53f3d3ab352fabfd5ab175001b51f8ed2ba1296e275`
- the source verifier recomputes observation legality, terminal placement,
  reward, trajectory, and summary semantics and binds every rollout byte
- strict bundles require exactly one rollout for each player count 4 through
  10
- the GPU runner requires fresh disjoint model/result paths, deterministic
  CUDA/CUBLAS/cuDNN settings, complete GPU identity, and stable source hashes
- the returned archive binds the selected final checkpoint, training metrics,
  hardware, source bundle/run config, exact inventory, and external ZIP
  checksum; legacy CPU smoke results require an explicit opt-in and cannot be
  accepted as GPU evidence
- status: implementation audit found no remaining blocking issue; Python
  integration 22/22 and Node integration 10/10 passed before the full project
  regression.

### V3 PPO iteration 1 and iteration 2 parent selection

- iteration-1 behavior/warmstart model SHA-256:
  `029333858dd177e2c6e6380e93a0efdad1752d0e942647dcadc285eb42f1e63c`
- iteration-1 rollouts: exactly one file for every player count 4 through 10,
  1,050,419 non-forced samples, 1,822,062 forced samples, and 2,872,481
  learner samples in total
- rank-coefficient `0` result ZIP SHA-256:
  `ca7f942cacec888bdbfb130ac81ffe5b2b584d76561773b299e20c5ec869718b`
- rank-coefficient `0.05` result ZIP SHA-256:
  `d4ec6d9243f24651d0006f403065382fb3e871b2b3fd05a6269c6377a470629f`
- both branches stopped after three epochs under the bound KL safeguard and
  passed strict local source/result verification
- every checkpoint failed the development promotion gate; no iteration-1
  model is eligible for product integration

The two conservative parent finalists were rerun on the same development seed
`610000001`, with 120 matches for every player count and 5 acts per match:

| Candidate | SHA-256 prefix | Pooled chip delta | 95% CI | Weakest player-count delta | Worst lower bound |
| --- | --- | ---: | ---: | ---: | ---: |
| rank `0.05`, epoch 1 | `5847b61b308a` | +0.12310 | [+0.07326, +0.17294] | +0.04764 (p7) | -0.11460 (p4) |
| rank `0`, epoch 2 | `df901f34c013` | +0.12623 | [+0.07626, +0.17620] | +0.02361 (p5) | -0.13812 (p5) |

The rank-`0.05` epoch-1 checkpoint is the iteration-2 parent because the
promotion contract is an all-player-count gate: its weakest player-count mean
and worst confidence lower bound are both more conservative, despite the other
candidate's slightly higher pooled mean. Its full SHA-256 is
`5847b61b308a6c84586f67f16687edd10a56786a455c84611c0fe4d6f1c05f96`.

Iteration-2 rollouts use temperature `2.5`, 3 acts, and 150,000 non-forced
decisions per player count. Opponent seats are 75% exact `normal`; the
remaining 25% are split uniformly between the original warmstart and rank-0
epoch-2 finalist. Seeds `871000001` through `877000001` map to p4 through p10.
Collection is distributed by whole files: p4-p6 on the local CPU and p7-p10 on
the GPU host CPU. The final reserved seed remains untouched.

## Non-card counterfactual source

- data:
  `artifacts/rl/noncard-counterfactuals-seed-720000001-p4-10-e100-a5.ndjson`
- SHA-256:
  `0ef81d2ccb7a43d4114866ef19f3f80911d75c7e4cfa58c1a3a8e08041c04aac`
- 700 matches, 5 acts, 5,803 decisions, 96,578 action continuations
- tax return: 5,328 decisions / 95,628 actions
- revolution: 475 decisions / 950 actions

Important limitation: each information state/action has only one hidden-world
and continuation sample. A stored standard error of zero therefore means
"not estimable from repeats", not "certain". Single-world oracle labels have
many ties and substantial winner's-curse noise.

## Non-card supervised experiments

### V1, terminal target, no behavior-cloning anchor

- ZIP SHA-256:
  `6eab0e04011e95016837c860947323a027681e1d9d9747acbb92300d0cfa6a7f`
- result: rejected; tax-only play was materially below `normal`.

### V2, terminal target, actor-selected checkpoint

- ZIP SHA-256:
  `8edc8de31f97aceda80d858acad99c89e2405e3e02459608ba7c70586396f953`
- result: rejected; safety thresholds only caused fallback to `normal` and
  did not demonstrate improvement.

### V3, terminal target with behavior cloning

- BC 1 ZIP SHA-256:
  `46fd3221228c786c49239c3aeec3853b480b3c57fb31bdd095cc9481bbba597e`
- BC 2 ZIP SHA-256:
  `2d7ef8781aade8e23d164f27c8350df69e781132711ee128303f999bdd73be49`
- result: rejected; BC 2 pooled tax effect was approximately `-0.0155` and
  statistically indistinguishable from zero.

### V4, decision-act target with behavior cloning

- BC 1 ZIP SHA-256:
  `59d1094ad9f268d4d01e1167222a3b60929909e8b9c4666a30ea0aa9fb910422`
- BC 2 ZIP SHA-256:
  `f42a634f752e28cdbd1e0efc9df76853278a7138bf535b75ebffac2af258759c`
- paired marginal screen: 700 seed-matched baseline/intervention pairs,
  5 acts, development seed `830000001`

| Candidate | Pooled chip delta | 95% CI | Tax actions changed |
| --- | ---: | ---: | ---: |
| BC 1 raw | +0.00163 | [-0.01443, +0.01769] | 485 |
| BC 1, actor-logit threshold 0.25 | +0.00394 | [-0.01001, +0.01790] | 364 |
| BC 2 raw | -0.00255 | [-0.01715, +0.01206] | 416 |
| BC 2, actor-logit threshold 0.25 | +0.00112 | [-0.01090, +0.01315] | 335 |

Result: rejected. The best V4 variant is effectively tied with `normal`, not a
demonstrated improvement. Actor-logit thresholds are not calibrated chip
advantages and cannot serve as a safety proof.

## Corrective non-card direction

The next tax policy predicts action advantage directly against the exact
`normal` baseline in current-act chip units. Return-count-one decisions stay
on `normal`. Return-count-two changes require a five-model ensemble to be
unanimously positive and its conservative lower score to exceed an explicit
chip-unit threshold. Hidden-world determinization must aggregate paired
action-minus-baseline advantages before this policy can be considered for
promotion.

### V2 hidden-world tax pilot

- source SHA-256:
  `67be65214a81b068508654cce37bad2d2dd199b9a7aec12820817bbfde37c2dd`
- 70 baseline matches, 264 two-card return states, 8 independently
  resampled hidden worlds per state and one continuation per world
- raw action evaluations: 60,832
- purpose: contract and signal pilot only; it is too small to select a
  production candidate
- status: accepted as a valid training-pipeline pilot, not as promotion
  evidence.

### V2 conservative tax-return ensemble

- training sources: 700 baseline matches across player counts 4 through 10;
  the four-player source is four disjoint 25-match files and every other
  player count is one 100-match file
- 2,653 two-card return information states, 604,464 paired action
  evaluations, and 8 independently resampled hidden worlds per state
- strict split: 2,151 train / 502 validation states, zero canonical-state
  overlap
- model SHA-256:
  `6f6fc82bd787c2e0dc4dbe199e625985d457e3de9e647bd7db91e9f56c8b0f7e`
- training manifest SHA-256:
  `46c4bb124a2a0f7234a73230c246949f24ba302714ebc3a2e448c8ad74f9d5d3`
- strict result ZIP SHA-256:
  `523ecd378f36b089b9adca685d07c3ade08417f997f51fe700ffc94d80b2ab73`
- paired development screen: 700 independent seed-matched pairs, 5 acts,
  seed `840000001`

| Minimum LCB | Learned decisions | Pooled chip delta | 95% CI | Result |
| ---: | ---: | ---: | ---: | --- |
| 0.00 chips | 287 | -0.00520 | [-0.01904, +0.00863] | reject |
| 0.25 chips | 100 | -0.00129 | [-0.00872, +0.00615] | reject |
| 0.50 chips | 33 | -0.00004 | [-0.00452, +0.00444] | reject |

The zero threshold was significantly harmful at seven and ten players. The
precommitted 0.5-chip gate avoided that regression but was indistinguishable
from exact `normal`, so the ensemble is retained only as training evidence and
is not integrated into the product bot. The 0.5 report SHA-256 is
`921c78432e0579bba8a500b2e4a31721086b18030353135a8b030cb0646582e5`.

### Conservative merchant-revolution candidate

- exploratory source SHA-256:
  `b861bc857e4e9d845f224ab06b3f9b2c503f9e3721e8e6979a99ef694dc96a05`
- strict source audit report:
  `artifacts/rl/revolution-merchant-candidate-v1-data-audit-run-001/report.json`
  (SHA-256
  `ccafb2dc5b0c1a9c209be1101cbb05d6859572a45d17f90fc32068ff99fe7fdc`)
- rule tested: declare only for a merchant in a six-player game and use exact
  current `normal` everywhere else
- source-data six-player merchant effect: `+0.3317` current-act chips, 95%
  CI `[+0.1365, +0.5270]`
- independent paired screen: 1,000 six-player matches, 5 acts, seed
  `945000001`
- declaring actor's affected-round delta: `+0.2407` chips, 95% CI
  `[+0.0150, +0.4665]`
- balanced candidate delta: `-0.00253` chip per player-act, 95% CI
  `[-0.00867, +0.00360]`; pairwise finish better 318, worse 319
- paired report:
  `artifacts/rl/revolution-merchant-candidate-v1-paired-seed945000001-run-001/report.json`
  (SHA-256
  `f2d4e66c5d65ecc8ac85dc82e65d98df5866eea2886e3f3aa9b2e3a050be69ee`)
- result: rejected. The actor-level gain did not become a whole-policy gain
  and produced a statistically adverse externality for candidate noble seats.
  Player counts 4, 5, 7, 8, 9, and 10 were verified exact no-ops.

## Reserved seeds

- development screens have used the documented `810000001`, `820000001`,
  `830000001`, `840000001`, and `945000001` families;
- training ablations use the documented `2026080xx` family;
- `900000001` is reserved for the final never-used automatic evaluation and
  must not be used for model selection or threshold tuning.
