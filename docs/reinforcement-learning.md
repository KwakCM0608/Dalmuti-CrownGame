# DALMUTI 강화학습 작업 분리

## 목표

현재 봇을 즉시 교체하지 않고, 게임 규칙과 개인정보 경계를 고정한
학습 환경부터 만든다. 학습 대상 V2는 매 턴의 `카드 제출/PASS` 결정이다.
세금 반환과 혁명 선언은 환경에 포함되지만, 우선 기존 봇 정책을 사용한다.

## 이 컴퓨터(CPU·프로젝트 파일 보유)

- 실제 프로젝트 규칙을 사용하는 결정론적 헤드리스 시뮬레이션
- 고정 행동 공간과 합법 행동 마스크 생성
- 자기 패와 공개 정보만 포함하는 관측값 생성
- 현재 쉬움·보통·어려움 봇의 기준 성능 평가
- 합성 셀프플레이 rollout 생성
- 추후 GPU가 학습한 모델의 합법성·성능·응답시간 회귀 검증

웹 UI, 애니메이션, D1, 네트워크, 실제 사용자 데이터는 학습 데이터 생성에
사용하지 않는다.

## GPU 컴퓨터(프로젝트 파일 불필요)

GPU 컴퓨터에는 다음만 전달한다.

1. 생성된 NDJSON 또는 이를 변환한 NPZ/Parquet 숫자 데이터
2. 이 문서에 고정된 관측값·행동 manifest
3. 게임 코드와 분리된 일반 PyTorch 학습 스크립트

첫 단계는 기존 봇 rollout으로 행동 모방(behavior cloning)을 해 네트워크를
안정적으로 초기화한다. 실제 기준전 결과 `normal`이 `hard`보다 강했으므로
V2의 교사는 `normal`이다. 교사 분포를 벗어난 상태는 DAgger로
다시 정답 표시한다. 그 다음 action-masked PPO와 league
self-play로 넘어간다. PPO는 on-policy이므로 GPU가 만든 최신 가중치를 이
컴퓨터의 시뮬레이터로 되돌려 새 rollout을 생성하는 반복 과정이 필요하다.

## 규격 V2

### 행동

총 506개 고정 인덱스다.

- `0`: PASS
- `1`: 조커 한 장 단독 제출
- `2..505`: `(일반 숫자 1..12, 총 장수 1..14, 조커 장수 0..2)`

현재 패와 필드에서 불가능한 인덱스는 `legalActionIndices`에서 제외한다.
동일 숫자의 물리적 카드 ID 조합은 하나의 의미 행동으로 합치며, 실제 엔진에
넣을 때 정렬된 카드 ID로 결정론적으로 변환한다.

### 관측

172개 실수 특성이다. 정확한 offset은 rollout 첫 줄의 manifest에 들어간다.

- 인원, 막, 현재 계급 좌석
- 본인의 사회 계급
- 필드 숫자와 장수
- 본인 패의 숫자별 장수
- 공개되어 제출된 카드의 숫자별 장수
- 최대 10명 기준 공개 손패 장수, 완주, PASS, 누적 칩, 사회 계급
- 현재 필드 제출자 표시
- 혁명 상태

상대의 숨은 패와 당사자가 아닌 세금 카드 정체는 입력 타입 자체에 없다.

### 보상

프로젝트의 막 종료 칩을 그대로 정규화한다.

```text
마지막 결정의 reward = (막 획득 칩 - 2) / 2
그 이전 결정의 reward = 0
```

따라서 각 플레이어 trajectory의 마지막 보상은 1위부터
`+1, +0.5, 0, -0.5, -1`이다. 인원수에 따라 가운데 순위가 늘어나도
모두 0이라 서로 비교할 수 있다. 결과 보상을 매 행동에 반복하지 않으므로
게임을 오래 끄는 행동에 잘못된 누적 이득이 생기지 않는다.

## 실행

Node.js와 pnpm만 사용하며 GPU는 필요 없다.

```powershell
pnpm run rl:evaluate -- --matches 100 --acts 3 --lineup easy,normal,hard,hard
pnpm run rl:rollouts -- --episodes 1000 --players 4 --acts 3 --difficulty normal
pnpm run rl:evaluate-model -- --model artifacts/rl/models/bc-v2/policy-weights.json
pnpm run rl:gpu-bundle
```

기본 rollout 경로는 `artifacts/rl/rollouts-v2.ndjson`이며 Git에 포함되지
않는다. 각 실행은 초기 seed와 episode 순서가 같으면 동일한 게임 결과를
만든다. `createdAt`만 실행 시각이라 파일 전체 바이트 비교에서는 제외한다.

행동 모방 모델이 교사와 다른 상태로 이동하면서 오차가 누적되면
`rl:dagger`로 그 모델이 실제 방문한 상태를 normal 교사가 다시 표시한다.
GPU 번들은 행동 모방 데이터와 두 차례 DAgger 데이터를 함께 포함한다.

## 다음 단계

1. normal 봇 rollout으로 GPU에서 행동 모방 워밍업 모델 학습
2. 반환 모델을 4~10인 `normal` 상대 기준전으로 평가
3. 최신 모델의 `logProbability`와 `valueEstimate`를 포함한 on-policy
   rollout 생성
4. `normal`과 과거 체크포인트를 상대 좌석에 섞어 action-masked PPO 반복
5. 모든 인원수에서 평균 칩 차이의 95% 신뢰구간 하한이 0보다 큰 모델만 승격
6. 카드 제출 정책이 안정된 뒤 혁명·세금 반환 head를 별도로 학습
7. 불법 행동 0건, 평균 칩, 1위/꼴찌 비율, 4~10인·계급별 편향과
   한 수 추론 지연시간을 통과한 모델만 실제 게임에 적용

기존 V2 BC/DAgger 파일에는 `logProbability`와 `valueEstimate`가 없으므로
그 자체는 PPO 데이터가 아니다. PPO용 파일은 반드시
`rl:ppo-rollouts` 또는 `rl:ppo-prepare`로 최신 behavior model에서 새로
생성한다. 전체 반복 절차는
[`ppo-self-play-pipeline.md`](ppo-self-play-pipeline.md)에 정리되어 있다.
