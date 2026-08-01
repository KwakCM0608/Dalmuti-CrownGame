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
- Attempt 004 p4-p10 screening is active. While it runs, iteration 002 is
  collecting four disjoint on-policy shards from attempt 004 so a failed
  screening does not leave the GPU/CPU pipeline idle.
