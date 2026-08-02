# V4 strongest-model execution ledger

This ledger is the durable source of truth for the new V4 line. The stopped
V3/i2 line is preserved for audit only and must never be resumed as a V4 run.

## Fixed objective

- Train a public-information actor that beats the exact frozen Normal policy.
- Continue with fresh immutable attempts until every player count from 4 to 10
  independently passes all three promotion gates:
  - mean chip difference per act at least `+0.25`;
  - match-clustered 95% confidence lower bound at least `+0.15`;
  - candidate-before-Normal pairwise finish rate at least `0.55`.
- Preserve the passing model and report only. Product integration and Sites
  deployment are explicitly out of scope until a new user request.

## Pre-V4 preservation

- Pre-V4 source commit: `e0c52b0` (`Preserve completed RL infrastructure before V4`).
- Verification before that commit: production build and all 230 tests passed;
  lint reported zero errors and the existing seven image warnings.
- The local and remote V3/i2 rollout files for p4-p10 were compared by SHA-256
  before remote cleanup and all seven pairs matched.
- The previously missing V3/i2 rank0 result was copied locally and strictly
  verified against its original bundle manifest and run config.

Preserved archive hashes copied from the GPU host:

| Archive | SHA-256 |
| --- | --- |
| `v3-ppo-i2-rank0-run-001-result.zip` | `28492f406934796706aed3b1a7c5b4bf15f86000b0e5b1ff5cf695545386d6b8` |
| `v3-ppo-i2-rank005-run-001-result.zip` | `25a164ad590b50f7617ebfaefab0ced978bdac24ed20fc2dd8edb5fe43443177` |
| `gpu-bundle-run-001-source-only.tar.gz` | `8ee31c514fa0dc5d97d65b4fe84dfa9438f86db2b7c5e8228d63750d2d32be4f` |
| `rank005-coarse60-remote-shards-run-001.tar.gz` | `8ac8e6ed4fcbb960bbe595e425f1c398294216bb5431ef7125889007cb972ec8` |
| `dalmuti-ppo-iteration-2-gpu.zip` | `0b98bbf460149a165dc7d22a55837c33483de366e0dadb637195554207bc537c` |
| `dalmuti-v3-warmstart-distill-ppo4-t25-seed-202608071-gpu-handoff-run-004.zip` | `aa420d2062091a75020dbd10547e16ab7a2d28801cd152bc647370e4c231a3f0` |
| `dalmuti-eval-source.tar.gz` | `64853dd45f44611fbc9ad4cb1ac2c8d634a4ef5ebf544a1ce5ee68de83ea5e15` |

The verified duplicate V3 work directories were then removed only from
`/home/pangmin/dalmuti`. Free space increased from 9.7 GiB to 14 GiB. The
shared Node runtime and the CUDA-enabled PyTorch environment were retained.

## Hardware allocation

- GPU host: RTX 3080 (10,495,852,544 bytes), PyTorch 2.7.1+cu118, CUDA active.
- GPU host CPU and local CPU consume disjoint rollout/search/evaluation shards.
- The RTX 3080 is reserved for Transformer updates and batched leaf inference.
- Every attempt uses a new `v4-strongest-attempt-NNN-*` directory and records
  checksums, heartbeat, exit code, configuration, source commit, and seeds.

## Holdout policy

- Development gates are `+0.30`, `+0.20`, and `0.57` on two disjoint
  certification seed families before a final holdout is eligible.
- The first final base seed is `900000001`; a frozen model may consume it once.
- Consumed final seeds are never reused and final reports are never training or
  tuning input.
- Final match counts are p4 2500, p5 1700, p6 900, p7 600, p8 400, p9 400,
  and p10 300, with five acts per match.

## Initial throughput measurements

- A production-shape 8-layer, 384-wide Actor plus 512-wide privileged Critic
  completed three CUDA update/checkpoint cycles on the RTX 3080 in 5.11 seconds.
  The deliberately disposable benchmark directory occupied 478 MiB. Its
  candidate and metadata (excluding disposable optimizer checkpoints) were
  preserved locally as `v4-default-benchmark-run-001-preserved.tar.gz` with
  SHA-256 `8e8a04745d1c7217ae85c37771e9a4a18af8fd3f7019a6d80200843ea34243b5`
  before the exact remote directory was removed.
- Seven local CPU workers generated complete five-act Normal shards for p4-p10
  in 5.6 seconds wall time: 40,551 samples, including 15,027 non-forced
  decisions. The strict raw audit files occupy 0.84 GiB before NPZ conversion.
- Seven concurrent jobs on the remote i9-10900K generated a disjoint p4-p10
  set of 38,795 samples (14,339 non-forced decisions); the slowest p10 shard
  finished in 3.2 seconds after SSH startup. Local and remote collection seeds
  are disjoint and outside every reserved final-evaluation family.
- These are infrastructure measurements, not model-quality evidence. A real
  trajectory conversion and training pass is required before revising the
  end-to-end estimate.

## First real warmstart candidate

- A collision-free mix of local p5/p7/p9 shards and remote p4/p6/p8/p10
  shards produced 38,753 samples in 1,890 actor trajectories. Strict combined
  conversion took 98.1 seconds and yielded dataset fingerprint
  `735fffc325cbd73a30bea31d91694dd4b22862d127e8cc5025779fb7d007a4f6`.
- Seed `330000101` trained four BC/critic epochs in 240.9 seconds. Normal BC
  loss moved from `0.295812` to `0.261323`.
- Candidate Actor SHA-256:
  `98c98d34ae90111772122cf357121f4540eb970b83f52f372720c7c4712a2d2f`.
- The candidate and non-optimizer metadata were copied locally with archive
  SHA-256 `286d3286e9be6bc563b0363e4d0c314e451c7506ba7706363a15787a4a3a4223`.
- This is only a warmstart candidate. Promotion evidence remains absent until
  its p4-p10 Normal screening report is complete.

The first screening completed and failed every player count. Report SHA-256:
`6279ccb84035c40ad7dd561e5331666c4e4ba8ef1f03abdc237da2e924c95fe1`.

| Players | Mean chip diff/act | Clustered 95% LCB | Pairwise before Normal |
| ---: | ---: | ---: | ---: |
| 4 | -1.0900 | -1.3700 | 0.3067 |
| 5 | -0.8361 | -1.0584 | 0.3328 |
| 6 | -0.4289 | -0.6444 | 0.4037 |
| 7 | -0.4842 | -0.6319 | 0.3772 |
| 8 | -0.3367 | -0.4567 | 0.4010 |
| 9 | -0.2805 | -0.3960 | 0.4183 |
| 10 | -0.1787 | -0.2680 | 0.4455 |

Interpretation: pure offline Normal BC suffers severe compounding-error state
distribution shift, especially at small tables. Repeating the same BC recipe
is not an eligible intervention; the next attempt must add candidate-state
DAgger/search supervision and an exact-Normal safety fallback.

## First-candidate action diagnostic

- The verified seed `330000101` Actor was evaluated on the strict combined
  Normal dataset with public Actor tensors only. The immutable report is
  `artifacts/rl/v4-accuracy-seed330000101-run-001/accuracy.json`, SHA-256
  `bf20813b732debcab3fcdc9025fef9700d7cee2b4daf0b3b739387cace9d8404`.
- Agreement with exact Normal is `0.918613` across all decisions, but forced
  decisions inflate that number. Agreement is only `0.782887` on non-forced
  decisions (`14,527` samples).
- The weakest non-forced strata are p4 (`0.7552`), great peon (`0.7194`),
  legal-action count 3-4 (`0.7036`), and legal-action count 5-8 (`0.7214`).
  DAgger/search sampling must therefore prioritize those strata instead of
  adding another undifferentiated Normal-only BC epoch.
- Fresh training initialization from a hash-verified Actor bundle was tested
  on the CUDA host in isolated directory `v4-train-init-smoke-run-001`; all
  seven pipeline tests passed. This enables the next DAgger candidate to start
  from the preserved first Actor without treating it as a resume checkpoint.

## Attempt 002 DAgger collection

- A candidate-state DAgger shard was collected in the isolated CUDA-host
  directory `v4-dagger-attempt002-seed370000001-run-001` with ten five-act
  matches at each of p4-p10, a deterministic candidate-behaviour probability
  of `0.5`, and exact Normal expert labels at every decision.
- The dataset contains `2,450` complete actor-act trajectories and `52,189`
  decisions. Collection and deterministic NPZ publication took `346.6`
  seconds and peaked at `7,927,092` KiB RSS.
- NPZ SHA-256:
  `9a4b749ac96e5b27ddee9635756b93ca24486ee23d070ac7f5ff754ee5cfb771`.
  External metadata SHA-256:
  `4bdd2c0e64ab39a031ee5b7212b78743502b16f884c930f9f0a327633512163d`.
  Both are preserved locally and remotely.
- There are `33,565` forced decisions. On the remaining `18,624` decisions,
  the first Actor disagrees with exact Normal `4,218` times (`22.65%`). The
  disagreement remains concentrated at p4 (`23.81%` of non-forced decisions),
  validating that this shard covers the deployment-state error exposed by the
  first screening rather than merely duplicating the offline Normal corpus.

## Search-teacher stability pilot

- The same complete p4 Actor trajectory was labelled with 2, 8, 32, and 64
  public-information determinizations per legal root action. Exact terminal
  Normal continuation was used; learned leaf values were forbidden.
- Wall times on the remote CPU were `34.1`, `108.3`, `417.7`, and `852.8`
  seconds respectively. Local CPU PyTorch was installed only under the ignored
  artifact dependency directory, allowing four disjoint h8 pilots to run in
  parallel in 24-27 seconds each and bringing the local CPU into the loop.
- Teacher agreement between h32 and h64 was only `0.7059` on the 17 non-forced
  states. The lower-budget h2/h8 agreement was `0.3529`. Search labels are
  therefore not yet eligible for blind bulk distillation. Direct search-vs-
  Normal outcome screening and episode-disjoint multi-core sharding are being
  added before a search-labelled training corpus is admitted.

## Attempts 002 and 003 training

- Attempt 002 started from Actor `98c98d34...` and used the 52,189-decision
  DAgger shard for four BC/critic epochs at actor LR `1e-4`. Training took
  `336.0` seconds. Its Actor SHA-256 is
  `54ab2ff459fc8c170ec2fab685a781fc361aaf21cdee0d39032da5698b83e711`.
  Strict Normal non-forced agreement rose from `0.782887` to `0.813933`.
- A deterministic merger combined the 38,753-decision original Normal corpus
  and the 52,189-decision DAgger corpus: 90,942 decisions and 4,340 complete
  trajectories. NPZ SHA-256 is
  `70c885a5b33a0f765135cc9433dfc1268a7fc43607c3709a65086614a0ed2f55`;
  merged fingerprint is
  `e8557ca8beaf797eed98aeb8f07dd4a2cc6f4baf66e108f3b79ec40ad5b0ecf5`.
- Attempt 003 started from attempt 002 and used the merged corpus for three
  BC/critic epochs at actor LR `5e-5`. Training took `481.2` seconds. Its Actor
  SHA-256 is
  `0166d0592b0c02b803e169c707871c24788c4703d9b776216dca79f1944f944d`.
  Strict Normal non-forced agreement rose again to `0.833345`; p4 improved
  from the first Actor's `0.7552` to `0.8090`. This remains a diagnostic, not
  promotion evidence. Attempts 002 and 003 both completed paired 420-match
  screening and failed promotion.

## Critic-layout correction and attempt 002 screening

- A contract audit found that the Python DAgger environment's privileged
  512-value critic tensor did not match the canonical TypeScript raw layout.
  The Python environment now uses the exact TS global/rank/player offsets and
  binds layout ID `dalmuti-v4-ts-privileged-critic-raw-v1` with SHA-256
  `be332c07e1753b6e87082917bbf5528faef8fed3cda794c853f655d3ade0110f`.
- Consequently the old attempt 002 DAgger and Normal+DAgger merged corpora are
  permanently ineligible for critic or PPO training. Their Actor-only BC
  results remain valid diagnostics because Actor and Critic parameters were
  isolated and Q-boost was disabled. Fresh PPO/DAgger collection must use the
  corrected, checksum-bound layout.
- The audit also found that variable table widths shifted Transformer memory
  and history token positions. Fixed maximum-player segment offsets now make
  a p4 observation's logits bit-identical when padded to p10 width (observed
  maximum absolute logit delta `0.0`).
- Attempt 002 completed the same 60-match-per-player-count screening and failed
  all player counts. Report SHA-256:
  `ab5d5343c2bf5cf8fc0707f79fa66372124a3354b1f728b612a6bfb7537d2d4a`.

| Players | Mean chip diff/act | Clustered 95% LCB | Pairwise before Normal |
| ---: | ---: | ---: | ---: |
| 4 | -1.2933 | -1.5567 | 0.2725 |
| 5 | -0.9722 | -1.1806 | 0.3056 |
| 6 | -0.6111 | -0.8156 | 0.3626 |
| 7 | -0.5114 | -0.6573 | 0.3683 |
| 8 | -0.2967 | -0.4318 | 0.4152 |
| 9 | -0.2595 | -0.3750 | 0.4230 |
| 10 | -0.2013 | -0.2894 | 0.4437 |

Normal action agreement improved while actual outcomes regressed. This is
direct evidence that imitation accuracy is not the optimization target and
that further pure DAgger/BC attempts should not displace on-policy learning.

Attempt 003 was the strongest completed imitation candidate but also remained
below Normal at every table size. Report SHA-256:
`5f75d9b1dc67bc12328eab4778f02db2b784de8fc1124a01104c524ed67fb1c4`.

| Players | Mean chip diff/act | Clustered 95% LCB | Pairwise before Normal |
| ---: | ---: | ---: | ---: |
| 4 | -0.5333 | -0.8300 | 0.4025 |
| 5 | -0.7083 | -0.9361 | 0.3583 |
| 6 | -0.5667 | -0.7179 | 0.3763 |
| 7 | -0.4044 | -0.5289 | 0.3944 |
| 8 | -0.2633 | -0.3900 | 0.4294 |
| 9 | -0.1695 | -0.2880 | 0.4453 |
| 10 | -0.0987 | -0.2053 | 0.4767 |

## Exact-search outcome diagnostic

- Direct public-information search was tested at p4 with exact frozen-Normal
  continuation and every legal root action. It is a bounded research
  diagnostic, not promotion evidence.
- One-hypothesis h1 produced mean chip difference `-2.0` and pairwise rate
  `0.15` in one match. H8 produced `-1.0` and `0.325` across two matches; H16
  produced `-1.0` and `0.30` in one match.
- The h8 and h16 reports are preserved with SHA-256
  `07b8c6c01641b591b008659fcde850b825addf0807b7b6446190b84a6af5b6f4`
  and `14921c2f695b1b0c963a3f9b4d04b39b4d591daeea320957c58ffa08a69f55e9`.
  Search is both unstable and computationally expensive at the tested budgets,
  so it is not the primary path for the next candidate.

## PPO transition

- A new on-policy league collector now runs p4-p10 CPU environments in
  parallel and batches public-only Actor inference on CUDA. It records exact
  behaviour log probabilities, terminal chip rewards, leak-free leave-one-
  match-out advantages, exact Normal labels, and the separately bound TS
  privileged critic state.
- The collector passed 12/12 tests on the remote RTX 3080, including a
  p4-p10 21-lane CUDA rollout and one-epoch PPO training smoke test.
- The first corrected PPO league corpus was collected in the isolated
  directory `v4-ppo-iteration-001-collect-seed430000001-run-001`. It uses the
  attempt 003 Actor only as a warm start, with a 75% Normal / 25% frozen-
  candidate opponent mixture. Six disjoint GPU-host shards completed in
  `357.4` to `393.7` seconds wall time and contain `2,240` complete actor-act
  trajectories with `48,828` public decisions.
- The six shard NPZ SHA-256 values are, by shard index 0 through 5:
  `78dc06bcda6bc0d07130b2ea8d273f2b3e9fad9840497802b0be122e222e46c3`,
  `637738b705212ae8574e9456f5f4fda3552059c041576dd3931d636c1b36da80`,
  `130684e81a933c8065e098a9f8bb2b4a8dee48eb840f69e8d58867f709baf860`,
  `56c034695e8507b671ef308a84bb119e86a50dc6ffbc518684e611a4aa403b24`,
  `b5d2a3613719ee688525f3661524c14dd699e62b50e4eb9f9aba135866ddeaa7`,
  and `4063f690b238fe9cde353aa32c8ba852b62fcd0cab8f9bb3ef7ab4970094adb1`.
- All collection files remain remote and immutable. Their strict merge
  completed in `13.89` seconds with `5,069,812` KiB maximum RSS. The merged
  corpus contains the same `2,240` trajectories and `48,828` decisions;
  fingerprint:
  `5dee3eda4b5a1f745de33724c5ca0b225fb07f2df95c3786385f4f855f910369`.
- Merged NPZ SHA-256:
  `faee2aee7d3b7cbef39e41ca3147ad5a7a2dd860a7b4d68ebd1e5962ccb3b6a5`.
  External metadata SHA-256:
  `d68100e63948a8e5ff08711896f37844a8ef6eff0ff0c535da307d25ae6e0a22`.
  Both are preserved locally and remotely.
- Every valid sample is explicitly BC/PPO/critic eligible, the canonical
  critic layout SHA is `be332c...`, and the sole bound behaviour Actor SHA is
  attempt 003's
  `0166d0592b0c02b803e169c707871c24788c4703d9b776216dca79f1944f944d`.
  No product integration or deployment is authorized by this run.

## Attempt 004: first real PPO candidate

- Attempt 004 initialized the exact Actor that generated iteration 001 and
  trained four on-policy epochs with PPO weight `1.0`, Normal-label BC anchor
  `0.1`, critic weight `0.1`, entropy `0.001`, clip ratio `0.1`, Actor LR
  `1e-5`, and Q-boost disabled. All `48,828` eligible samples were seen once
  per epoch by each requested loss.
- CUDA training completed in `301.52` seconds and reached global step `4,480`.
  Approximate KL remained small (`0.000604` to `0.000912`) and clip fraction
  remained `0.0233` to `0.0322`; critic loss declined from `0.08599` to
  `0.06483` without a runaway policy update.
- Actor SHA-256:
  `cea22f67c1b0e6aeee82338642efb0c16e83000b5e72aa0b6c3fb2786ec1350f`.
  Candidate manifest SHA-256:
  `db8507172c959bf12cfbbf342ad513ab60ac725b099e32231b46f1d9f268c6ed`.
  The public Actor and non-optimizer evidence are preserved locally and
  remotely.
- Screening uses a real Git bundle of frozen commit
  `e0c52b0462d86756cf40b90f19d35a3e26b0f674`; bundle SHA-256 is
  `9ea0b9eb4200ac369fbc3ffb1493efe59625b34f5f994359f8a01d4b5610db4d`.
  The evaluator checks the commit blob's Normal source hash `aa44743c...` and
  the actual observation contract hash `13dc7e48...` before any match runs.
- Attempt 004 completed the same 60-match-per-player-count screening and did
  not pass promotion. Report SHA-256:
  `ffcdb63c57f2d2486bf44efb85b304f905e951f95857c5877f05831378dc4951`.

| Players | Mean chip diff/act | Clustered 95% LCB | Pairwise before Normal |
| ---: | ---: | ---: | ---: |
| 4 | -0.7767 | -1.1200 | 0.3558 |
| 5 | -0.5417 | -0.7694 | 0.3917 |
| 6 | -0.4911 | -0.6600 | 0.3922 |
| 7 | -0.2256 | -0.3792 | 0.4303 |
| 8 | -0.2267 | -0.3634 | 0.4310 |
| 9 | -0.2955 | -0.4020 | 0.4177 |
| 10 | -0.0827 | -0.1880 | 0.4649 |

The first PPO update improved p5-p8 and p10 point estimates over attempt 003,
but regressed at p4 and p9 and remained below Normal at every table size. It
is therefore evidence of a usable but weak learning signal, not a promotable
candidate.

## PPO iteration 002

- Four disjoint CUDA-host rollout shards generated `2,240` complete actor-act
  trajectories and `48,686` decisions from attempt 004. Their decision counts
  were `12,297`, `12,041`, `12,267`, and `12,081`.
- The strict merge fingerprint is
  `4650051048bed3fb1572a82f4497802844f1a9dc3357e3e519aa4ee589415289`.
  Merged NPZ SHA-256 is
  `bd047aa0eed8c534a5d43022128a420c000e6b20895c3b2de3e28dea6ce91c9a`;
  external metadata SHA-256 is
  `bca17ad03b35d8bf0f1503e615752f03176cae1ae9a1ad2bc326376cc8861b54`.
- A training audit found that roughly `62.7%` of rollout rows have only one
  legal action. Those rows correctly have zero direct policy gradient, but
  attempt 004's minibatch advantage normalization and policy-loss denominator
  still let them dilute and distort the non-forced policy signal. The trainer
  now restricts BC/PPO Actor losses to eligible rows with more than one legal
  action and uses the collector-bound advantage without a second minibatch
  normalization. Critic learning may continue to use all eligible rows.
- A second audit found that the first merge retained independently calculated
  shard-local baselines. Their exact p/role/act reference count had median
  `4`, and `20.5%` of iteration 002 advantages changed sign when calculated
  from the complete population. Merged PPO advantage contract v2 now excludes
  the target's entire match cluster, requires at least `16` references before
  accepting a hierarchy tier, and uses one population scale across all PPO
  trajectories with floor `0.5` and no recentering.
- The fresh v2 merge completed in `15.1` seconds and preserves `2,240`
  trajectories, `48,686` samples, and `18,174` effective non-forced policy
  decisions. Reference counts are min/median/max `16 / 37 / 129`; global scale
  is `0.5758417692757455`, and maximum absolute training advantage is
  `3.1355063915252686` rather than the old `8.66`.
- V2 NPZ SHA-256:
  `73a34ad7c9d435ef3081d6f2c1cea4060a369207e1cc72f9ea69b18c79fd5c75`.
  External metadata SHA-256:
  `6e08ab6aaa9f62affeb16aaeca14c947dde0d41d05ab3eecd9721c7d54641ba0`.
  Dataset fingerprint:
  `ea6408372b0c6ac888625fe2e4f97f0b5fbbb09f46c8a62b6e825979151bc9ae`.
  Advantage-array binding:
  `b97c3ab2b5aefea613576cfb1441d439c06f8b1c79b402005db10384a848a206`.
  The artifact and sidecars are preserved and checksum-verified locally and
  remotely. Old merged-v1 PPO artifacts fail closed before training, while
  original direct PPO shards remain valid inputs for a new v2 merge.

## Evaluation throughput and regression verification

- The V4 evaluator can now shard complete player counts across at most four
  subprocesses. The default balanced allocation is p10; p9+p4; p8+p5; and
  p7+p6. Every worker independently verifies the Actor bundle, frozen Normal
  source, observation contract, and final reservation where applicable.
- Shards are immutable canonical JSON plus SHA-256. Strict merging requires
  p4-p10 exactly once and calls the same report assembler as the serial path;
  serial and parallel reports are byte- and SHA-identical for the same inputs.
- Evaluator tests passed `23/23` locally and on the remote Linux/PyTorch
  environment. The complete remote V4 suite passed `139/139` with two optional
  environment skips. Product regression passed `254/254`; TypeScript checking
  passed; lint reports zero errors and the pre-existing seven image warnings.

## Attempt 005: corrected PPO signal diagnostic

- Attempt 005 initialized attempt 004's exact behavior Actor and trained one
  epoch on iteration 002 global-advantage v2 with PPO `1.0`, non-forced Normal
  BC `0.05`, entropy `0.001`, Actor LR `1e-5`, and clip ratio `0.10`. Q-boost
  remained disabled. CUDA training took `81.27` seconds for `1,120` optimizer
  steps.
- The epoch saw all `48,686` raw PPO rows, excluded `30,512` singleton-legal
  rows from both Actor objectives, and used `18,174` effective decisions.
  Approximate KL was `0.002048` and clip fraction `0.07559`, both below the
  precommitted stop limits `0.012` and `0.15`.
- Epoch-1 Actor SHA-256:
  `32f7f366c0a65d7b2b67baf5aeb2e33c49c87ddf4bcac513317bf710fc351466`.
  Candidate manifest SHA-256:
  `6485004cfc936f1c711e84bbf6cdfe365eddf7055db6abb6a2780a24ed1c3b5c`.
  The Actor, manifest, training result, run manifest, metrics, and sidecars are
  checksum-verified and preserved locally and remotely.
- Common-random-number screening reused attempt 004's exact family
  `attempt004-screening-seed450000001`. Four whole-player-count GPU workers
  completed all 420 matches in `258.04` seconds. Report SHA-256:
  `f8f2087b9c1cf5262ed7f80e926a46fcd804d3aa9e918f7f92f9641b29d98905`.

| Players | Attempt 005 chip diff | Gain vs 004 | 95% LCB | Pairwise | Pair gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | -0.6300 | +0.1467 | -0.9567 | 0.3817 | +0.0258 |
| 5 | -0.5417 | +0.0000 | -0.7833 | 0.3917 | +0.0000 |
| 6 | -0.3578 | +0.1333 | -0.5556 | 0.4237 | +0.0315 |
| 7 | -0.3092 | -0.0836 | -0.4550 | 0.4094 | -0.0208 |
| 8 | -0.2000 | +0.0267 | -0.3217 | 0.4373 | +0.0063 |
| 9 | -0.1845 | +0.1110 | -0.3015 | 0.4385 | +0.0208 |
| 10 | -0.0387 | +0.0440 | -0.1440 | 0.4827 | +0.0177 |

The corrected signal improved five table sizes, but failed every part of the
precommitted permission rule for one more rotating-ID iteration: p4 gain was
below `+0.25`, p5 below `+0.15`, fewer than six table sizes improved, median
gain was only `+0.044`, p7 regressed beyond `-0.08`, and both pairwise-gain
requirements failed. Training therefore stops at epoch 1 rather than reusing
the mismatched collector. The next corpus must keep evaluator-equivalent
candidate physical IDs fixed across all five acts and assign suffix-of-match
outcomes to each learner act segment.

## Fixed-identity PPO iteration 001: precommitted production plan

This plan was sealed before any production trajectory was generated. It does
not authorize product integration or deployment.

- Fresh remote run directory:
  `/home/pangmin/dalmuti/v4-fixedid-ppo-i001-s530000001-run-001`.
- Run namespace: `v4-fixedid-ppo-i001-s530000001`.
- Environment seed base: `530000001`; training seed: `540000001`.
- Complete five-act match counts are p4 `320`, p5 `256`, p6 `192`, p7
  `160`, p8 `128`, p9 `112`, and p10 `96`: `1,264` complete matches and
  `6,320` learner act trajectories in total.
- The corpus is partitioned into exactly 12 modulo shards, match start `0`.
  Every shard must use CUDA, requested lanes `16`, temperature `1.0`, epsilon
  floor `0.0`, pairwise coefficient `0.25`, standardized advantages,
  deterministic algorithms, TF32 disabled, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Shards 0 through 5 run concurrently, followed only after complete success
  by shards 6 through 11. Collection, merge, training, and evaluation never
  overlap.
- The immutable behaviour Actor is attempt 005 epoch 1. Actor SHA-256:
  `32f7f366c0a65d7b2b67baf5aeb2e33c49c87ddf4bcac513317bf710fc351466`;
  manifest SHA-256:
  `6485004cfc936f1c711e84bbf6cdfe365eddf7055db6abb6a2780a24ed1c3b5c`.
- Before training, all 12 direct artifacts and the strict merged artifact
  must prove the same canonical collection plan, exact shard indices
  `0..11`, exact match-index ranges and deterministic seeds, the settings
  above, identical execution metadata, identical source hashes, and the
  attempt 005 bindings. The trainer receives the sole plan's
  `canonicalSha256`, never the coverage SHA.
- Epoch 1 uses Actor LR `2e-5`, critic LR `2e-4`, PPO `1.0`, Normal BC anchor
  `0.05`, critic `0.2`, entropy `0.0005`, clip `0.12`, batch size `2`, weight
  decay `1e-4`, max gradient norm `1.0`, and Q-boost `0.0`. The Actor stays
  FP32 with autocast disabled; only the critic may use AMP.
- Training is forbidden unless a full pre-update replay of every PPO-eligible
  row reproduces the stored selected-action log probability with maximum
  absolute error at most `2e-5`.
- Epoch 1 stops immediately on non-finite metrics, approximate KL above
  `0.020`, clip fraction above `0.25`, or entropy collapse beyond 30%. Values
  in KL `0.012..0.020` or clip `0.15..0.25` allow screening but forbid a
  second epoch. A second epoch is considered only when KL is below `0.0015`,
  clip fraction below `0.03`, and the common-random-number screen improves.
- Screening reuses family `attempt004-screening-seed450000001`, base seed
  `450000001`, 60 complete five-act matches per p4-p10, four whole-player-
  count workers, pure Actor policy, and no fallback. Each epoch candidate is
  checksum-preserved before any resume.
- No final-reservation seed is consumed until a candidate passes two fresh
  development families with the stricter all-p gates `+0.30` mean chip
  difference, `+0.20` clustered 95% lower bound, and `0.57` pairwise rate.

## Fixed-identity PPO preflight evidence and remote-run amendment

The following work validates the production plan without changing its data,
training, screening, or promotion thresholds.

- The first remote directory,
  `/home/pangmin/dalmuti/v4-fixedid-ppo-i001-s530000001-run-001`, was created,
  but its initial source transfer reset before extraction. No rollout, merge,
  training, or evaluation process started there. Ten subsequent key-based SSH
  probes to `220.70.2.226:2222` all timed out before authentication. The
  directory is preserved as abandoned evidence and must not be reused. The
  next remote attempt uses the fresh suffix `run-002`.
- A real p4-p10 local smoke exposed that JSON `sort_keys=True` serializes
  `matchCounts` as p10,p4,... while the strict loader previously trusted map
  insertion order. Commit `6da5a8d` now validates the exact p4-p10 key set
  independently of input order, reconstructs numeric order, and reloads its
  own strict merged artifact. The focused collector suite passed `19/19`; the
  then-current V4 suite passed `174/174` with five optional skips.
- The local end-to-end CPU smoke preserved 7 complete matches, 35 learner-act
  trajectories, and 769 samples. Merged NPZ SHA-256 is
  `9c83ae70f6efb8f809704071d5a83911d309fb0967a3b80a83b2808dab37b333`;
  canonical plan SHA-256 is
  `62c7a2236b4f3e3e7414b313179cf10c0064d99ba2423b18554c025cd2b64573`.
  One 18-step CPU epoch passed the full initial-policy replay with maximum
  absolute selected-log-probability error `4.0531158e-6`.
- A larger local CPU pilot preserved 28 complete matches, 140 learner-act
  trajectories, and 3,007 samples. Merged NPZ SHA-256 is
  `692969eac71b7c557168d9ca82be3376e222e008e44eed7c8e685641d9c0b221`;
  canonical plan SHA-256 is
  `b262becc4bf0b25d3c08823f6e03f23e4de64d7223e72fe7a549fd1ad46b0d7e`.
  Its production-hyperparameter CPU epoch completed 70 optimizer steps. An
  independent full-policy replay measured final KL `0.00226305`, clip
  fraction `0.0789652`, entropy `0.472475`, and maximum absolute log ratio
  `0.489123`, demonstrating that the earlier streaming minibatch diagnostics
  were not a sufficient final-policy gate.
- Fixed-identity training now performs a full FP32 Actor replay after every
  epoch. Non-forced PPO rows use the exact p4-p10 equal-weight reduction in
  float64; forced singleton rows are reported separately. The final Actor
  KL, clip fraction, entropy, entropy retention, per-player-count evidence,
  and maximum log ratio replace the streaming values used by stop decisions.
  The old streaming values remain only under `optimizationPassDiagnostics`.
- The versioned post-epoch audit is bound through metrics, checkpoint,
  `latest.json`, resume, candidate manifest, and result. It includes a
  deterministic, canonical little-endian semantic Actor-state SHA-256; exact
  forced-row counts by player count; CUDA/CPU audit batch selection; and the
  fixed checkpoint RNG contract. Every new fixed checkpoint has an atomically
  written `.pt.sha256` sidecar, and both explicit and `latest` resume verify
  the whole checkpoint before deserialization. Fixed checkpoints preserve CPU
  Torch, all CUDA-device, NumPy, and Python RNG state. Resume rejects
  Actor-state splicing, stale checkpoint bytes, missing or type-changed
  training settings, altered p-specific forced evidence, and incompatible
  CUDA RNG state. Failed resume and full replay restore Actor mode, model
  weights, and every optimizer RNG source transactionally. Legacy BC/DAGGER
  checkpoint shape and explicit-resume compatibility remain unchanged.
  `resume=latest` additionally accepts only the exact canonical
  `checkpoints/epoch-N.pt` path derived from the recorded epoch and proves the
  resolved file remains inside that run's checkpoint directory before any
  checksum lookup or PyTorch deserialization.
- A same-seed local diagnostic rerun completed in `1,291.8` seconds. Its
  optimization-pass metrics matched the earlier pilot within approximately
  `2e-8`, and its final replay measured KL `0.002263054`, clip fraction
  `0.0789652`, and entropy `0.472475`. The two CPU runs used different thread
  execution environments and therefore were not tensor-byte identical
  (maximum parameter difference `0.00013943`). The rerun also began before
  the final Actor-state/RNG binding patch loaded, so it is diagnostic evidence
  only and cannot be promoted or resumed as the production candidate.
- After the final audit and resume hardening, the focused balance/training
  tests pass `17/17` and the complete V4 suite passes `180/180` with five
  CUDA-only skips on this CPU host. Python static compilation and
  `git diff --check` pass.
- The source archive prepared from commit `6da5a8d` predates the final audit
  hardening and is stale. A new compact archive and checksum must be generated
  from the final committed source before any `run-002` transfer. No candidate
  from this preflight has been evaluated against Normal, integrated into the
  game, or deployed.

## Mixed-host amendment before production collection

The CUDA-only `run-002` controller reached the public address again after its
local integrity checks, but TCP `220.70.2.226:2222` timed out before SSH
authentication. It did not create the remote run directory or transfer a
payload. Two subsequent local full-shard attempts also produced no artifact:
the first failed before process creation because Windows PowerShell inherited
duplicate `Path`/`PATH` entries, and the second was terminated with its two
collectors when the 15-minute controller limit expired. Their fresh local
directories remain failure evidence and are never resumed or promoted.

No production trajectory for the CUDA-only 12-shard plan exists. Before any
production data was admitted, that plan was therefore superseded by this
fresh mixed-host plan so the user's local CPU and the remote CPU/GPU can work
concurrently without weakening provenance:

- Run namespace: `v4-fixedid-ppo-i001-mixed-s580000001`.
- Environment seed base: `580000001`; training seed: `590000001`.
- Match counts and all model, reward, optimization, hard-stop, screening, and
  promotion thresholds remain exactly those of the preceding fixed-identity
  plan: p4 `320`, p5 `256`, p6 `192`, p7 `160`, p8 `128`, p9 `112`, and p10
  `96`, for `1,264` complete matches and `6,320` learner-act trajectories.
- The corpus uses exactly 14 modulo shards. Shards `0` and `1` are collected
  on the local CPU; shards `2` through `13` are collected on the remote CUDA
  host in two waves of six. Every remote collector uses rolling CPU
  environments and batched CUDA Actor inference, so the remote CPU and GPU are
  both active while the local CPU produces disjoint match clusters.
- The exact backend map is
  `cpu,cpu,cuda,cuda,cuda,cuda,cuda,cuda,cuda,cuda,cuda,cuda,cuda,cuda`.
  It is part of the canonical
  `fixed-complete-mixed-backend-shard-plan-v2` fields and SHA-256. A shard run
  on the wrong backend, a missing/extra index, a reordered map, or a version-1
  downgrade is rejected before merge and again by the trainer's expected-plan
  binding. The canonical SHA-256 of the required p4-p10 CPU/CUDA calibration
  report is a second mandatory v2 plan field; collection cannot start with a
  missing, stale, differently bound, or merely invented report hash. Every
  mixed collector must also receive the exact CPU NPZ and CUDA NPZ named by
  that report. It reloads their canonical metadata and checksum sidecars,
  copies their arrays from one hashed byte snapshot, recreates the complete
  comparison report in memory, and requires byte-for-byte equality with the
  supplied report before collection or resume.
- The immutable behaviour Actor remains attempt 005 epoch 1, Actor SHA-256
  `32f7f366c0a65d7b2b67baf5aeb2e33c49c87ddf4bcac513317bf710fc351466`
  and candidate manifest SHA-256
  `6485004cfc936f1c711e84bbf6cdfe365eddf7055db6abb6a2780a24ed1c3b5c`.
- A separate calibration family
  `v4-fixedid-mixed-calibration-s575000001`, seed base `575000001`, collects
  the same one complete five-act match at every p4-p10 once on local CPU and
  once on remote CUDA. Admission requires exact equality of identity,
  trajectory coverage, actions, legal masks, public and privileged states,
  Normal labels, finish orders, chips, rewards, and every other unclassified
  array. Only selected-action old log probability, selected probability, and
  policy entropy may differ, each by at most `2e-5`. Both runtime versions and
  all input/checksum hashes are preserved in the canonical calibration report.
- The calibration and production processes run only from the same extracted
  Git-blob archive, not from a platform-specific checkout. This keeps every
  source hash identical across Windows CPU and Linux CUDA despite checkout
  line-ending settings. The archive includes this ledger and the immutable
  execution recipe, and its binding verifies every source file directly.
- Screening deliberately uses two separately sealed baselines: Normal comes
  from `lib/bot-strategy.ts` at frozen commit `e0c52b0`, while the observation
  contract comes from the current sealed package source at
  `training/v4-public-history.ts`. The Git bundle, frozen commit and Normal
  hash do not imply that the observation file belongs to that old commit;
  both paths and both hashes are verified independently before evaluation.
- After all 14 shards are strictly merged, CUDA must independently replay
  every PPO-eligible row with the initial Actor and reproduce the stored old
  selected-action log probability within `2e-5`, both through a sealed
  pre-training audit and through the trainer's own mandatory replay. CPU/CUDA,
  shard, and p4-p10 strata are reported separately. Any calibration or replay
  failure invalidates both the CPU and CUDA artifacts and requires a fresh
  mixed-host namespace with new local and remote run directories; it never
  relaxes a tolerance.
- The first fresh local production directory is
  `v4-fixedid-ppo-i001-mixed-s580000001-local-run-001`; the first fresh remote
  directory is
  `/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixed-s580000001-run-001`.
  Every later retry increments its own suffix and never deletes or reuses a
  failed directory.

This amendment still does not authorize product integration, deployment, or
use of a final-reservation seed.

## Mixed-host run-001 calibration failure and deterministic-math retry

The first mixed-host execution is preserved as an immutable failed run. The
local directory is
`artifacts/rl/v4-fixedid-ppo-i001-mixed-s580000001-local-run-001`, the remote
directory is
`/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixed-s580000001-run-001`, and its
local terminal failure record has SHA-256
`8a49b3b766e810ac40465174e08b443e6732e134578e32ef349b3cad811abc41`.
Neither directory is deleted, reused, resumed, or promoted.

Both sides completed the isolated calibration schedule: one complete
five-act match at each of p4-p10, seven complete matches, 35 learner-act
trajectories, and 741 learner decision samples per backend. The CPU artifact
used Torch `2.13.0+cpu` and has NPZ SHA-256
`e9f74beadb86d02b0fba0a5eb316ef151186be69803c02069171efe0d42934ff`;
the CUDA artifact used Torch `2.7.1+cu118` and has NPZ SHA-256
`aa05bb562a42736ddf39f0a851ecc779e0393e69a67c43bf0eb627505bcee92e`.
Schedule, identity, actions, legal masks, states, Normal labels, finish orders,
chips, rewards, and every other exact array matched before the comparator
reached the three permitted policy-float arrays. The actual CPU/CUDA maximum
absolute differences were:

| Array | Maximum absolute difference | Required maximum |
| --- | ---: | ---: |
| `old_action_log_probs` | `0.0007748603820800781` | `0.00002` |
| `selected_action_probabilities` | `0.00007956765152261625` | `0.00002` |
| `policy_entropies` | `0.00015497207641601562` | `0.00002` |

The comparator therefore stopped at calibration admission with the exact
reported `old_action_log_probs` difference `0.00077486038`, approximately
`38.743` times the fixed tolerance. This exposed backend-dependent attention
math rather than a schedule or game-state mismatch. No admitted calibration
report, production shard, merged dataset, pre-training replay, PPO training
run, candidate model, screening report, or Normal comparison was produced by
this execution.

The retry uses deterministic math attention and wholly new identities while
retaining the behavior Actor, frozen Normal baseline, sealed observation
contract, collection counts/topology, training settings, screening family,
hard gates, and promotion gates:

- Package ID and production namespace:
  `v4-fixedid-ppo-i001-mixedmath-s600000001`.
- Calibration namespace and seed:
  `v4-fixedid-mixedmath-calibration-s595000001` and `595000001`.
- Environment seed base: `600000001`; training seed: `610000001`.
- The run contract requires the exact FP32 MHA-slowpath/math-SDP policy
  numerics contract, SHA-256
  `a08de79f95df089fb5c525bb12a14f0fa28985d294f9fa3b2942e5db46df1ca3`.
  Collection, pre-training replay, training, and screening evidence each fail
  closed if this contract is missing or altered.
- Fresh local run directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmath-s600000001-local-run-001`.
- Fresh remote run directory:
  `/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmath-s600000001-run-001`.
- Fresh package directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmath-s600000001-package-run-001`.

The retry package and all package/source hashes remain intentionally
unassigned until the deterministic-math changes and this identity amendment
are committed and the package is built from that exact commit. This retry
still prohibits V3/i2 resumption, final-reservation seeds, product
integration, and deployment.

## Deterministic-math run collection success and float32-binding retry

The deterministic-math run passed cross-backend calibration before starting
production collection. CPU and CUDA produced the same 7 complete matches, 35
learner-act trajectories, and 771 samples. All exact arrays matched. The
maximum absolute differences were `1.9073486328125e-6` for selected-action old
log probability, `4.0316981358978765e-7` for selected probability, and
`7.152557373046875e-7` for entropy, all below the immutable `2e-5` limit. The
calibration report bound to production has SHA-256
`dc5333c8c8e647b09f48164c410f83f22f7352e2d8865e24dbcd9ebab5ec7782`.

All 14 production shards then completed and were transferred successfully:
`1,264` complete matches, `6,320` learner-act trajectories, and `135,788`
valid decision samples. Every NPZ, metadata, completion, retrieval, and status
sidecar matched its recorded SHA-256. Merge nevertheless failed closed before
training because 284 rows in 8 shards did not satisfy the exact serialized
advantage derivation. The terminal failure record SHA-256 is
`60cd90c766995a2d5b99a06bc23c8c4a3776ab429777376aba50ef4792b32ad3`.

The failure was a serialization-contract defect, not a game, reward, policy,
or CPU/CUDA rollout mismatch. The collector independently computed return,
baseline, raw advantage, scale, and standardized advantage with Python
float64 intermediates and then stored each as float32. The loader recomputed
the derivation from the stored float32 operands. For 14 trajectories with
shard-local scales between `0.001` and `0.00625`, float32 quantization error
was amplified to as much as two output ULPs past the strict `2e-6` derivation
tolerance. The affected 284 rows are
`0.20915%` of the corpus and the maximum discrepancy is
`3.0517578125e-5`. The failed local and remote directories remain immutable
and are never resumed, rewritten, or promoted.

The collector now stores return, baseline, and scale first and derives both
advantage arrays exclusively from those serialized float32 operands. It also
runs the unchanged strict fixed-match validator before publishing any shard;
validation failure leaves no artifact. The tolerance is not relaxed. The
retry uses a new source package, seeds, namespace, and directories:

- Package ID and production namespace:
  `v4-fixedid-ppo-i001-mixedmathfp32-s620000001`.
- Calibration namespace and seed:
  `v4-fixedid-mixedmathfp32-calibration-s615000001` and `615000001`.
- Environment seed base: `620000001`; training seed: `630000001`.
- Fresh local run directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-local-run-001`.
- Fresh remote run directory:
  `/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-run-001`.
- Fresh package directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-package-run-001`.

All behavior-Actor, frozen-Normal, observation, deterministic policy-math,
collection topology, optimization, hard-gate, screening, and promotion
contracts remain unchanged. Package and source hashes remain unassigned until
this fix and amendment are committed and built from that exact commit. Product
integration and deployment remain prohibited.

## Float32-binding launcher preflight failure and environment retry

The first float32-binding package was built from commit `9312381` with package
manifest SHA-256
`6d6c0e543724cc83ee9f10b5bee85518dcd6be3c23c0b8d0d6bacf0e12d83e28`,
source archive SHA-256
`8d38135a63e0399e816e0f522028e4b3c57238c79e089a632fe7bb7e9e7acb35`,
and source inventory SHA-256
`fda2213cdb0257124a037d6f37579ad6c9b54a38bfc12166de9430847c3307b9`.
Windows and Linux package verification, extraction, and the canonical 26-phase
dry-run plan all passed with plan SHA-256
`015fc003db4a871aa67e85d5d46370378174d530000a24fcef25363982bd55c4`.

The actual controller then failed during the first local Actor verification,
before remote staging, calibration, collection, merge, replay, training, or
screening. The selected local Python executable depends on the external
`artifacts/rl/python-deps` package directory, but the launcher process did not
inherit that directory through `PYTHONPATH`; importing `torch` therefore
raised `ModuleNotFoundError`. The immutable local failure directory is
`artifacts/rl/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-local-run-001` and
its failure record has SHA-256
`fd3a19f3037d0ef7d0925f23ecb76c7afb6455fe5f30dc28d7b1fe5a38eefe5f`.
The corresponding remote run directory was never created. This local run is
not reused or rewritten.

The next controller invocation must set `PYTHONPATH` to the absolute local
`artifacts/rl/python-deps` directory before process creation. It uses wholly
new execution identities even though the failure occurred before rollout:

- Package ID and production namespace:
  `v4-fixedid-ppo-i001-mixedmathfp32env-s640000001`.
- Calibration namespace and seed:
  `v4-fixedid-mixedmathfp32env-calibration-s635000001` and `635000001`.
- Environment seed base: `640000001`; training seed: `650000001`.
- Fresh local run directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmathfp32env-s640000001-local-run-001`.
- Fresh remote run directory:
  `/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmathfp32env-s640000001-run-001`.
- Fresh package directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmathfp32env-s640000001-package-run-001`.

All semantic, numerical, training, screening, promotion, and prohibition
contracts remain unchanged. The new package hashes remain unassigned until
this identity amendment is committed and built from that exact commit.

## Environment retry collection success and CUDA replay OOM retry

The environment-corrected package was built from commit `a2fe1c8` with package
manifest SHA-256
`aca6aba47608cfcef161c7292bd423f6c2c6583e3358bc5fe233a6adaf53aae0`,
source archive SHA-256
`04178a49cde26f29a440b2ac30ac2e6576e6299ec55916d1b468e100240830b9`,
source binding SHA-256
`e2ecc3f76ff04bc0b25f5c61dfb2ed9786d126d3a7d3b37e741e70819a92ecb2`,
and source inventory SHA-256
`927a1daf792d1c34be705bfdf608dc1bbd802f4a079fb31bfbe576601f645885`.
Windows and Linux verification, extraction, and the canonical 26-phase dry
run passed. The workflow plan SHA-256 was
`3121a459e7c2656f4606ae3ceb4dc48e3805365f5eb79c491225828fac5fd6e7`.

Cross-backend calibration passed with 7 complete matches, 35 learner-act
trajectories, 775 decision samples, and exact agreement for all 61 exact
arrays. The largest selected-action old-log-probability difference was
`1.6689300537109375e-6`, below the immutable `2e-5` limit. The calibration
report SHA-256 is
`6cfc366b466491b0effa2522873f54dc84a415dfdd16afaae603fc29260863e7`.

All 14 production shards passed their pre-publication validators and checksum
checks. They contain 1,264 complete matches, 6,320 learner-act trajectories,
136,940 valid decision samples, and 814,576 environment decisions. Strict
merge also passed, directly proving that the serialized-float32 advantage fix
removed the preceding derivation failure. The merged NPZ SHA-256 is
`8e9327c0a32ce0a2f39ce858b1b32f83377fce296532fce223b8a54f7ef88641`
and its metadata SHA-256 is
`4b98bf35d1183a769a845af82113c1a8bf1404f36e5a423559f3e9a31827b6ea`.

The run then failed closed before training during the first full CUDA policy
replay. This was not a policy-probability, dataset, reward, or fingerprint
failure. The replay hard-coded an audit trajectory batch of 64; the first
Transformer attention forward requested another 4.61 GiB when the RTX 3080
had 2.29 GiB free out of 9.78 GiB. PyTorch reported 4.81 GiB allocated and
1.36 GiB reserved but unused, so allocator tuning alone could not satisfy the
request. No replay report, checkpoint, candidate, screening report, or Normal
comparison was created. The local terminal failure record SHA-256 is
`0ffaa7be6198786f3a4f9b329d44ddb11e19273eb970e644a99f96c0cf175d26`.
The complete remote failure evidence was archived locally and remotely with
SHA-256
`6b6370ba213e52adff2d0e065e5d0fcbef8665e76b465a326edeeb185d127d73`.
The failed local and remote run directories remain immutable and are never
resumed, rewritten, or promoted.

The retry binds every independent and trainer pre/post-update CUDA policy
audit to a fixed trajectory batch of 4. The sealed runtime verifier rejects
the former batch of 64. This changes only memory shape: actor eval mode, FP32
forward, disabled autocast, deterministic policy numerics, full-dataset
coverage, and the `2e-5` tolerance remain unchanged. A new loader also
materializes each unique NPZ member exactly once before running the unchanged
strict semantic and fingerprint validators. This avoids repeatedly inflating
the 37 MB compressed merged file, whose arrays occupy 4.115 GB, without
copying the 3.970 GB core tensor storage. Duplicate logical NPZ members fail
closed.

Before a new production controller is allowed to run, the preserved s640
merged dataset must pass a disposable full CUDA replay with the committed
batch-4 implementation. The actual retry then uses wholly new execution
identities and directories:

- Package ID and production namespace:
  `v4-fixedid-ppo-i001-mixedmathfp32env-s660000001`.
- Calibration namespace and seed:
  `v4-fixedid-mixedmathfp32env-calibration-s655000001` and `655000001`.
- Environment seed base: `660000001`; training seed: `670000001`.
- Fresh local run directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmathfp32env-s660000001-local-run-001`.
- Fresh remote run directory:
  `/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmathfp32env-s660000001-run-001`.
- Fresh package directory:
  `artifacts/rl/v4-fixedid-ppo-i001-mixedmathfp32env-s660000001-package-run-001`.

All behavior-Actor, frozen-Normal, observation, reward, collection topology,
optimization, hard-gate, screening, promotion, and prohibition contracts
remain unchanged. Package and source hashes remain unassigned until this fix
and identity amendment are committed and built from that exact commit.
