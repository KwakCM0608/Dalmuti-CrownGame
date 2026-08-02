# DALMUTI V5 immutable execution runbook

이 문서는 V5의 `source seal -> initialization -> calibration -> measured
parallel collection -> verified dataset -> one epoch train -> screening -> two
certifications -> one-shot final` 경로만 다룬다. 제품 코드 통합과 배포는 별도 승인
전까지 금지한다.

## 1. 고정 규칙

- 매 학습은 반드시 새로운 디렉터리에서 시작한다. 기존 run 디렉터리를 재사용하거나
  덮어쓰지 않는다.
- 첫 run 이름과 namespace는 둘 다
  `v5-mappo-normalresidual-i001-s840000001-run-001`이다.
- iteration/run별 개발 seed는 `v5_workflow.py describe`가 기록한 값만 쓴다.
  첫 run은 initialization `830000001`, calibration `835000001`, collection
  `840000001`, training `850000001`, screening `860000001`이다.
- final seed namespace는 `900000001` 이상이며 development 수집, 튜닝, 재학습에
  절대 사용하지 않는다.
- 원본 저장소의 clean full-40 HEAD를 commit한 후에만 bootstrap한다. 실행 코드는
  반드시 run의 `source-checkout`에서 불러온다.
- promotion registry는 run 내부가 아니라 모든 sibling run이 공유하는 단일
  디렉터리다.
- immutable JSON/디렉터리가 이미 있으면 삭제하거나 덮지 않는다. 같은 요청의
  검증 가능한 재시도만 허용한다.

## 2. 디렉터리와 I/O

첫 run의 표준 경로는 다음과 같다.

| 역할 | 표준 경로 |
| --- | --- |
| 로컬 run | `artifacts/rl/v5-runs/<runNamespace>` |
| 로컬 sealed checkout | `<run>/source-checkout` |
| 원격 run | `/home/pangmin/dalmuti/<runNamespace>` |
| 원격 임시 raw shard | `/dev/shm/<runNamespace>/raw-shards` |
| 원격 영구 spool | `/home/pangmin/dalmuti/<runNamespace>/spool-bundles` |
| 로컬 수신 spool | `<local-independent>/<runNamespace>/incoming-spool-bundles` |
| 로컬 canonical shard | `<local-independent>/<runNamespace>/canonical-shards` |
| 로컬 copy receipt | `<local-independent>/<runNamespace>/verified-copy-receipts` |
| GPU 영구 shard tier | `/home/pangmin/dalmuti/<runNamespace>/persistent-shards` |
| GPU 휘발 shard tier | `/dev/shm/<runNamespace>/volatile-shards` |
| GPU promotion receipt | `/home/pangmin/dalmuti/<runNamespace>/low-disk-promotion-receipts` |
| 공유 promotion registry | `/home/pangmin/dalmuti/v5-promotion-registry` |

`v5_spool.py`와 `v5_low_disk_stage.py`의 root basename/parent 검사는 계약의
일부다. 표에 적힌 basename을 바꾸지 않는다.

## 3. source seal과 초기 pair

PowerShell에서 모든 V5 변경을 commit하고 clean 상태를 확인한다.

```powershell
$Repo = 'C:\Users\byj01\Documents\Dalmuti'
$Py = 'C:\Users\byj01\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$Run = 'v5-mappo-normalresidual-i001-s840000001-run-001'
$RunRoot = Join-Path $Repo "artifacts\rl\v5-runs\$Run"
$Head = (git -C $Repo rev-parse HEAD).Trim()
git -C $Repo status --short
& $Py "$Repo\gpu-training\v5_workflow.py" bootstrap `
  --run-root $RunRoot `
  --repository-root $Repo `
  --source-commit $Head `
  --iteration 1 `
  --run-number 1
& $Py "$Repo\gpu-training\v5_workflow.py" materialize-source `
  --run-root $RunRoot `
  --output "$RunRoot\source-checkout"
```

출력은 `workflow.json[.sha256]`, `source-seal/source.{bundle,tar}`,
`source-seal/manifest.json[.sha256]`, `initialization/actor-bundle`,
`initialization/critic.pt`, `initialization/pair-manifest.json[.sha256]`이다.
remote에는 이 run을 새 디렉터리 그대로 복사한다. remote 작업도 다음 checkout에서
실행한다.

```bash
RUN=v5-mappo-normalresidual-i001-s840000001-run-001
ROOT=/home/pangmin/dalmuti/$RUN
cd "$ROOT/source-checkout"
python gpu-training/v5_workflow.py describe --run-root "$ROOT"
git status --short
git rev-parse HEAD
```

## 4. CPU/CUDA calibration

CPU와 CUDA는 같은 pair, source inventory, namespace, seed, 기본 `32 match/p`
schedule을 쓴다. `<CHECKOUT>`은 각 host의 sealed checkout이다.

```text
python <CHECKOUT>/gpu-training/v5_collect_cli.py calibrate-collect
  --actor-bundle <RUN>/initialization/actor-bundle
  --critic-checkpoint <RUN>/initialization/critic.pt
  --behavior-pair <RUN>/initialization
  --source-root <CHECKOUT>
  --output <RUN>/calibration/cpu
  --backend cpu --device cpu
  --run-namespace <runNamespace> --seed-base 835000001 --lanes 7

python <CHECKOUT>/gpu-training/v5_collect_cli.py calibrate-collect
  --actor-bundle <RUN>/initialization/actor-bundle
  --critic-checkpoint <RUN>/initialization/critic.pt
  --behavior-pair <RUN>/initialization
  --source-root <CHECKOUT>
  --output <RUN>/calibration/cuda
  --backend cuda --device cuda:0
  --run-namespace <runNamespace> --seed-base 835000001 --lanes 7

python <CHECKOUT>/gpu-training/v5_collect_cli.py calibrate-compare
  --cpu-snapshot <RUN>/calibration/cpu
  --cuda-snapshot <RUN>/calibration/cuda
  --output <RUN>/calibration/cpu-cuda.json
```

`cpu-cuda.json[.sha256]`가 PASS하지 않으면 다음 단계로 가지 않는다. production
plan의 자동 stratified rate estimate는 이 calibration의 p4..p10 각각 최소
20 match 이상 측정치만 사용한다.

두 host 사이의 이동 순서는 고정한다. CPU snapshot은 로컬에서, CUDA snapshot은 GPU
host에서 만든다. CUDA 완료 후 `<GPU_RUN>/calibration/cuda` 디렉터리 전체를
`<LOCAL_RUN>/calibration/cuda`로 상대 구조 그대로 복사하고 manifest/sidecar와 모든
파일의 checksum을 로컬 loader로 재검증한다. 그 뒤에만 로컬의 CPU snapshot과 수신한
CUDA snapshot으로 `calibrate-compare` 및 production collection plan을 만든다.

Collection 전에 다음 산출물을 GPU run의 같은 상대 경로로 다시 복사하고 checksum을
검증한다. 일부 파일만 복사하거나 JSON 경로를 평탄화하지 않는다.

- `<LOCAL_RUN>/calibration/cpu` → `<GPU_RUN>/calibration/cpu`
- `<LOCAL_RUN>/calibration/cuda` → `<GPU_RUN>/calibration/cuda`
- `<LOCAL_RUN>/calibration/cpu-cuda.json[.sha256]` →
  `<GPU_RUN>/calibration/cpu-cuda.json[.sha256]`
- `<LOCAL_RUN>/collection/plan` → `<GPU_RUN>/collection/plan`

Remote `collect-shard`는 이 네 산출물을 다시 열어 exact pair/source/plan binding을
검증한 뒤에만 시작한다.

## 5. GPU admission을 두 단계로 분리

1. 합성 최대형 preflight는 RTX 3080/PyTorch/AMP 구성과 대략적인 VRAM/속도를
   조사하는 추천 자료다. 실제 production admission은 아니다.
2. canonical shards의 merged index가 생긴 뒤, exact initial pair와 실제 데이터셋을
   사용해 committed `v5_gpu_memory_preflight.py`를 다시 실행한다. RTX 3080
   synthetic admission에서 안전한 `audit=64`, Actor physical microbatch `32`,
   accumulation `1`, 그리고 trainer와 동일한
   `critic_batch_size=256`을 검사한 PASS report/sha가 train의 필수 입력이다.

실제 admission 명령의 형태는 다음과 같다. source seal 이후 trainer의 확정
`critic_batch_size`를 명시한다.

```text
python <CHECKOUT>/gpu-training/v5_gpu_memory_preflight.py
  --dataset <RUN>/collection/index
  --model-pair <RUN>/initialization
  --output <RUN>/preflight/gpu-memory.json
  --device cuda:0
  --audit-batch-size 64
  --microbatch-size 32
  --gradient-accumulation 1
  --critic-batch-size 256
  --warmup-iterations 2
  --timing-iterations 7
```

Report는 p10 nonforced에서 history 최대, legal-action 수 최대, 두 값의 곱 최대인
audit 후보를 각각 실행한다. 각 phase의 allocator peak와 `cuda.mem_get_info()` 직접
관측 free bytes를 기록하고, 관측 free 최솟값 1 GiB 이상 및 allocator reserved
fraction 90% 이하를 각각 통과해야 한다. Audit/Actor/Critic timing median 및 p95와
1.5M/2.0M decision 한 epoch GPU-compute 예상 시간도 포함한다. 이 예상값은 forced
semantic vector scan과 dataset I/O를 제외한다고 명시한다. OOM이나 config 불일치는
report를 보존하고 train을 차단한다.

## 6. worker scaling probe와 production plan

Collector process 하나는 거의 single-core이므로 local CPU와 remote CUDA 각각
`1, 2, 4, 6` process 후보를 측정한다.

측정 seed는 서로 겹치지 않게 아래 값으로 고정한다.

- CPU single-process: `835100001`
- CUDA single-process: `835200001`
- CPU scaling의 worker 수 `W`, 반복 `j`: `835300001 + W*1000 + j`
- CUDA scaling의 worker 수 `W`, 반복 `j`: `835400001 + W*1000 + j`

- 모든 후보에서 각 process는 동일한 p4..p10 mix와 `20 match/p`를 쓴다.
- 후보 안에서는 lane count와 Torch thread/process를 고정한다.
- wall time, aggregate nonforced decisions/s, peak RSS, CUDA peak VRAM을 기록한다.
- OOM/swap/thrash 없이 headroom을 만족하는 최고 aggregate 처리량의 worker 수를
  선택한다.
- 다른 worker 수를 동시에 시험하지 않는다. 각 output/scratch는 새 경로다.

단일-process p별 속도 보고서는 다음 명령으로 만든다.

```text
python <CHECKOUT>/gpu-training/v5_collect_cli.py benchmark-throughput
  --actor-bundle <RUN>/initialization/actor-bundle
  --critic-checkpoint <RUN>/initialization/critic.pt
  --behavior-pair <RUN>/initialization
  --source-root <CHECKOUT>
  --output <RUN>/throughput/<backend>-single.json
  --backend <cpu|cuda> --device <cpu|cuda:0>
  --run-namespace <runNamespace>
  --seed-base <위 규칙으로 계산한 고정 subseed>
  --scratch-root <new-existing-empty-scratch-root>
  --matches-per-player-count 20 --lanes <fixed-lanes>
```

`cpu-single.json`과 `cuda-single.json`의 p4..p10 `secondsPerMatch`를 각각
`4:s,...,10:s` 형식으로 옮긴다. Plan은 각 p에서
`n_cpu = N * t_cuda/workers_cuda / (t_cpu/workers_cpu + t_cuda/workers_cuda)`
에 가장 가까운 양 backend 최소 1 match 배분을 만들고 양 backend 모두
`maxMatchesPerShard` 단위로 나눈다.

```text
python <CHECKOUT>/gpu-training/v5_collect_cli.py plan
  --actor-bundle <RUN>/initialization/actor-bundle
  --critic-checkpoint <RUN>/initialization/critic.pt
  --behavior-pair <RUN>/initialization
  --source-root <CHECKOUT>
  --calibration-report <RUN>/calibration/cpu-cuda.json
  --calibration-cpu-snapshot <RUN>/calibration/cpu
  --calibration-cuda-snapshot <RUN>/calibration/cuda
  --run-namespace <runNamespace> --seed-base 840000001
  --output <RUN>/collection/plan
  --target-nonforced-decisions 1600000
  --minimum-nonforced-decisions 1500000
  --maximum-nonforced-decisions 2000000
  --cpu-seconds-per-match 4:s,...,10:s
  --cuda-seconds-per-match 4:s,...,10:s
  --cpu-worker-count <selected-local-workers>
  --cuda-worker-count <selected-remote-workers>
  --cpu-torch-threads-per-worker <fixed-local-threads>
  --cuda-torch-threads-per-worker <fixed-remote-threads>
```

## 7. parallel shard collection과 stale slot 복구

Plan의 shard backend를 바꾸지 않는다. Host별 동시 process 수는 plan에 봉인된
worker 수와 정확히 같거나 작아야 한다.

```text
python <CHECKOUT>/gpu-training/v5_collect_cli.py collect-shard
  --plan <RUN>/collection/plan
  --shard-index <I>
  --shards-root <host-root>
  --actor-bundle <RUN>/initialization/actor-bundle
  --critic-checkpoint <RUN>/initialization/critic.pt
  --behavior-pair <RUN>/initialization
  --source-root <CHECKOUT>
  --calibration-report <RUN>/calibration/cpu-cuda.json
  --calibration-cpu-snapshot <RUN>/calibration/cpu
  --calibration-cuda-snapshot <RUN>/calibration/cuda
  --device <cpu|cuda:0> --lanes <sealed-lanes>
```

Worker lock에는 pid, hostname, kernel boot-id, process-start ticks, plan SHA,
backend, slot이 들어간다. Crash 후 lock을 임의 삭제하지 않는다. 같은 host+boot에서
pid/start가 일치하면 active로 거절한다. 다른 hostname도 원격 생존 여부를 증명할
수 없어 거절한다. 같은 host에서 process 소멸/PID reuse/재부팅이 확인된 경우에만
다음 명령이 O_EXCL receipt를 만들고 원 lock을 no-replace retirement한 후 slot을
다시 연다.

```text
python <CHECKOUT>/gpu-training/v5_collect_cli.py recover-worker-slot
  --plan <RUN>/collection/plan
  --shards-root <exact-host-root>
  --backend <cpu|cuda>
  --slot-index <I>
  --reason "operator-audited reason"
```

## 8. `/dev/shm` spool, transfer, receipt, retirement

Remote CUDA shard 하나가 완성될 때마다 즉시 export한다.

```text
python <CHECKOUT>/gpu-training/v5_spool.py export
  --plan <RUN>/collection/plan --shard-index <I>
  --raw-root /dev/shm/<runNamespace>/raw-shards
  --spool-root /home/pangmin/dalmuti/<runNamespace>/spool-bundles
```

반환된 exact bundle directory를 local
`<runNamespace>/incoming-spool-bundles`로 복사한 뒤 import한다.

```text
python <CHECKOUT>/gpu-training/v5_spool.py import
  --plan <RUN>/collection/plan --shard-index <I>
  --bundle <incoming-spool-bundles>/<exact-bundle>
  --canonical-root <runNamespace>/canonical-shards
  --receipt-root <runNamespace>/verified-copy-receipts
```

생성된 exact receipt를 remote에 복사한다. 단순 SCP 성공만으로는 아무것도 지우지
않는다. Remote에서 bundle과 receipt를 다시 검증한 다음 raw, spool 순으로만
retire한다.

```text
python <CHECKOUT>/gpu-training/v5_spool.py retire-raw
  --plan <RUN>/collection/plan --shard-index <I>
  --raw-root /dev/shm/<runNamespace>/raw-shards
  --bundle <remote-exact-bundle> --receipt <remote-exact-receipt>

python <CHECKOUT>/gpu-training/v5_spool.py retire-spool
  --plan <RUN>/collection/plan --shard-index <I>
  --raw-root /dev/shm/<runNamespace>/raw-shards
  --spool-root /home/pangmin/dalmuti/<runNamespace>/spool-bundles
  --bundle <remote-exact-bundle> --receipt <remote-exact-receipt>
```

모든 shard가 canonical root에 모이면 zero-copy verified index를 발행한다.

```text
python <CHECKOUT>/gpu-training/v5_collect_cli.py publish-index
  --plan <RUN>/collection/plan
  --shards-root <runNamespace>/canonical-shards
  --output <RUN>/collection/index
```

GPU 호스트의 영구 디스크만으로 corpus 전체를 담을 수 없으므로 실제 GPU 학습에서는
hybrid stage를 사용한다. 로컬 canonical index의 상대 구조와 각 shard 디렉터리 내부
구조를 바꾸거나 파일을 합치지 않는다. 먼저 로컬에서 raw source-index record와
capacity plan을 고정한다. 기본 reserve는 영구 tier 6 GiB, 휘발 tier 2 GiB이며,
`*-free-bytes`에는 plan 직전 GPU 호스트에서 직접 측정한 값을 넣는다.

```text
python <LOCAL_CHECKOUT>/gpu-training/v5_low_disk_stage.py record-source-index
  --source-index <LOCAL_RUN>/collection/index
  --output <LOCAL_RUN>/collection/source-index-record

python <LOCAL_CHECKOUT>/gpu-training/v5_low_disk_stage.py plan
  --source-index <LOCAL_RUN>/collection/index
  --output <LOCAL_RUN>/collection/low-disk-stage-plan
  --run-namespace <runNamespace>
  --persistent-free-bytes <MEASURED_HOME_FREE_BYTES>
  --volatile-free-bytes <MEASURED_DEV_SHM_FREE_BYTES>
  --persistent-reserve-bytes 6442450944
  --volatile-reserve-bytes 2147483648
```

다음 세 control directory를 상대 경로 그대로 GPU run으로 복사하고 각 manifest와
sidecar를 byte-for-byte 확인한다.

- `<LOCAL_RUN>/collection/plan` → `<GPU_RUN>/collection/plan`
- `<LOCAL_RUN>/collection/source-index-record` →
  `<GPU_RUN>/collection/source-index-record`
- `<LOCAL_RUN>/collection/low-disk-stage-plan` →
  `<GPU_RUN>/collection/low-disk-stage-plan`

Stage plan의 각 shard record가 지정한 tier와 `stagedName`을 읽는다. 원본 canonical
shard 전체를 해당 tier root 안의
`.<stagedName>.incoming-<8..64자리 nonce>`로 복사한 뒤에만 다음 명령을 실행한다.
SCP 종료만으로 완성으로 간주하지 않는다. `promote-shard`가 모든 파일, manifest,
planned shard index와 checksum을 검증한 뒤 같은 filesystem에서 no-replace rename하고
immutable receipt를 발행한다.

```text
python <GPU_CHECKOUT>/gpu-training/v5_low_disk_stage.py promote-shard
  --plan <GPU_RUN>/collection/low-disk-stage-plan
  --shard-index <I>
  --incoming <TIER_ROOT>/.<stagedName>.incoming-<nonce>
  --tier-root <TIER_ROOT>
  --receipt-root <GPU_RUN>/low-disk-promotion-receipts
```

Process가 atomic rename 직후 receipt 발행 전에 죽었으면 target이나 lock을 임의로
삭제하지 않는다. 동일한 plan, shard index, incoming 경로, tier root, receipt root로
`promote-shard`를 그대로 재실행한다. 구현이 inactive owner를 검증하고 exact target을
재검사한 뒤 deterministic receipt를 발행하거나 재사용한다. 다른 bytes/target이면
fail-closed한다.

모든 shard가 승격된 뒤에만 cross-filesystem index를 발행하고 즉시 다시 검증한다.
Plan, source-index record, output index는 반드시 같은 `<GPU_RUN>/collection` parent에
있어야 한다. 영구/휘발 tier는 서로 다른 `st_dev`여야 하고 symlink는 금지된다.

```text
python <GPU_CHECKOUT>/gpu-training/v5_low_disk_stage.py verify-index
  --plan <GPU_RUN>/collection/low-disk-stage-plan
  --persistent-root <GPU_RUN>/persistent-shards
  --volatile-root /dev/shm/<runNamespace>/volatile-shards
  --source-index-record <GPU_RUN>/collection/source-index-record
  --receipt-root <GPU_RUN>/low-disk-promotion-receipts
  --collection-plan <GPU_RUN>/collection/plan
  --output-index <GPU_RUN>/collection/index

python <GPU_CHECKOUT>/gpu-training/v5_low_disk_stage.py verify-stage
  --plan <GPU_RUN>/collection/low-disk-stage-plan
  --persistent-root <GPU_RUN>/persistent-shards
  --volatile-root /dev/shm/<runNamespace>/volatile-shards
  --source-index-record <GPU_RUN>/collection/source-index-record
  --receipt-root <GPU_RUN>/low-disk-promotion-receipts
  --collection-plan <GPU_RUN>/collection/plan
  --hybrid-index <GPU_RUN>/collection/index
```

로컬 canonical corpus는 이 과정 이후에도 보존한다. `/dev/shm` 복제본은 휘발성이라
GPU 재부팅 후 같은 plan과 원본으로 다시 stage해야 하며 학습 결과로 복구하지 않는다.

## 9. 실제 corpus admission과 one-epoch train

Section 5의 실제-corpus GPU preflight PASS/sha를 먼저 만든다. Train은 다음 모두를
즉시 재검증해야 한다.

- 실행 중인 `v5_workflow.py`/`v5_train.py`가 sealed checkout 파일인지
- clean exact source commit과 source inventory가 run source seal과 같은지
- dataset identity가 preflight report와 같은지
- initial model pair ID/hash가 report와 같은지
- policy numerics와 audit/Actor/Critic batch config가 report와 같은지
- low-disk plan/source record/receipt와 두 tier의 실제 bytes가 hybrid index 및 production
  collection plan과 같은지
- training output이 아직 존재하지 않는지

```text
python <CHECKOUT>/gpu-training/v5_workflow.py train
  --run-root <RUN>
  --repository-root <CHECKOUT>
  --dataset-index <RUN>/collection/index
  --gpu-memory-preflight <RUN>/preflight/gpu-memory.json
  --device cuda:0
  --audit-batch-size 64
  --microbatch-size 32
  --gradient-accumulation 1
  --critic-batch-size 256
  --low-disk-persistent-root <GPU_RUN>/persistent-shards
  --low-disk-volatile-root /dev/shm/<runNamespace>/volatile-shards
  --low-disk-promotion-receipt-root <GPU_RUN>/low-disk-promotion-receipts
```

외부 `--initial-model-pair`는 허용하지 않는다. 새로운 carry-forward pair가 필요하면
새 run bootstrap 계약으로 pair를 source seal에 포함시켜야 한다. Epoch는 정확히
1회이며 manifest에 effective Actor batch, Critic batch, forced-row audit, preflight
SHA를 보존한다. 세 low-disk root 중 하나라도 생략하거나, run의 fixed control
directory 없이 root만 넘기거나, 두 tier가 같은 filesystem이면 admission은
fail-closed한다. 이 검증을 통과한 동일 `<RUN>/collection/index`만 GPU preflight
report 검증과 trainer 양쪽에 전달한다.

## 10. screening, certification, final

Production screening은 Actor의 functional tensor identity당 전역에서 정확히 한 번만
허용한다. 평가 전에 shared registry에 screening을 먼저 예약한다. 예약이 반환한
`reservationPath`와 `outputPath` 외의 seed/family/output으로 실행하지 않는다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py reserve-screening
  --run-root <RUN> --registry <SHARED_REGISTRY>
  --bundle <RUN>/training/actor-bundle
  --repository-root <CHECKOUT> --device cuda:0

python <CHECKOUT>/gpu-training/v5_workflow.py evaluate
  --run-root <RUN> --bundle <RUN>/training/actor-bundle
  --stage screening --device cuda:0 --lanes 32
  --repository-root <CHECKOUT>
  --screening-reservation <screeningReservationPath>
  --output <SHARED_REGISTRY>/screening-results/<reservationId>/report.json
```

같은 tensor를 metadata만 바꿔 다시 export하거나 provenance/seed를 바꿔도 두 번째
production screening은 거절된다. Screening PASS 후에만 두 certification coordinate를
예약한다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py reserve-certification
  --run-root <RUN> --registry <SHARED_REGISTRY>
  --bundle <RUN>/training/actor-bundle
  --screening-reservation <screeningReservationPath>
  --screening-report <registry-screening-report.json>
  --repository-root <CHECKOUT> --device cuda:0
```

반환된 `reservationPath` 하나가 a/b 둘을 모두 고정한다. Certification은 각각
60 match/p, 10000 bootstrap resample, unsharded `1/0`이며 output은 registry의
`certification-results/<reservationId>/{a,b}.json`으로 자동 결정된다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py evaluate
  --run-root <RUN> --bundle <RUN>/training/actor-bundle
  --stage certification-a --device cuda:0 --lanes 32
  --repository-root <CHECKOUT>
  --screening-report <registry-screening-report.json>
  --certification-reservation <reservationPath>

python <CHECKOUT>/gpu-training/v5_workflow.py evaluate
  --run-root <RUN> --bundle <RUN>/training/actor-bundle
  --stage certification-b --device cuda:0 --lanes 32
  --repository-root <CHECKOUT>
  --screening-report <registry-screening-report.json>
  --certification-reservation <reservationPath>
```

두 certification이 stricter development gate까지 PASS한 경우에만 final을
예약한다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py reserve-final
  --registry <SHARED_REGISTRY>
  --bundle <RUN>/training/actor-bundle
  --certification-report <registry-cert-a.json>
  --certification-report <registry-cert-b.json>
  --final-shards <FROZEN_FINAL_SHARD_COUNT>
```

Final match plan은 p4 `2500`, p5 `1700`, p6 `900`, p7 `600`, p8 `400`, p9
`400`, p10 `300`으로 총 `6800` match다. 각 shard는 gameplay 전에 claim하고
claim이 예약한 canonical output만 사용한다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py claim-final
  --run-root <RUN> --plan <promotion-plan> --bundle <RUN>/training/actor-bundle
  --repository-root <CHECKOUT> --device cuda:0
  --match-shard-count <FROZEN_FINAL_SHARD_COUNT>
  --match-shard-index <I>

python <CHECKOUT>/gpu-training/v5_workflow.py evaluate
  --run-root <RUN> --bundle <RUN>/training/actor-bundle
  --stage final --device cuda:0 --lanes 32
  --repository-root <CHECKOUT>
  --promotion-plan <promotion-plan> --final-claim <claimPath>
  --match-shard-count <FROZEN_FINAL_SHARD_COUNT>
  --match-shard-index <I>
```

모든 final shard report를 한 번만 merge하고 승인한다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py merge-evaluations
  --output <SHARED_REGISTRY>/final-results/<reservationId>/merged.json
  <all-final-shard-reports-in-index-order>

python <CHECKOUT>/gpu-training/v5_workflow.py approve-final
  --plan <promotion-plan> --bundle <RUN>/training/actor-bundle
  --final-report <SHARED_REGISTRY>/final-results/<reservationId>/merged.json
```

Certification/final canonical report가 이미 있으면 coordinator는 gameplay를 다시
실행하지 않고 report, Actor, source provenance, reservation/claim을 재검증해 기존
SHA만 반환한다. Execution-start marker만 있고 report가 없으면 결과가 모호하므로
자동 replay를 거절한다. 동일 host/boot의 기록된 process가 더 이상 active가 아니고
retired target/temp 증거를 검증한 경우에만 원래 evaluate 명령에 명시적 사유를 더해
결정론적 다음 attempt를 만들거나 exact-valid orphan output을 복구한다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py evaluate
  <원래 예약/claim/evaluate 인수 전체>
  --recover-crashed-attempt-reason "operator-audited inactive execution"
```

Promotion registry lock도 임의 삭제하지 않는다. 같은 host에서 process 소멸, PID
reuse 또는 재부팅이 증명된 경우에만 다음 명령이 lock과 recovery receipt를
evidence-bound 방식으로 처리한다.

```text
python <CHECKOUT>/gpu-training/v5_workflow.py recover-promotion-lock
  --registry <SHARED_REGISTRY>
  --reason "operator-audited stale promotion lock"
```

Final 결과를 본 뒤 학습, seed 변경, hyperparameter 변경, 모델 선택을 하면 해당
final namespace는 소모된 것으로 간주한다.

## 11. 완료 전 검증

```text
python -m unittest discover -s gpu-training -p "test_v5*.py" -v
pnpm test
pnpm run lint
git diff --check
```

승인 산출물이 생겨도 이번 작업 범위에서는 빠른대전/온라인 봇에 연결하거나
배포하지 않는다.
