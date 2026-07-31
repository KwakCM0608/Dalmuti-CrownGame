# DALMUTI PPO 리그 학습 파이프라인

## 목적

행동 모방 모델은 `normal`의 결정을 따라 배우는 워밍업이다. `normal`보다 강한
후보를 만들기 위해 그 모델을 actor-critic으로 전환하고, action-masked PPO와
리그 대전을 반복한다.

학습 대상은 카드 제출과 PASS 결정이다. 혁명 선언과 세금 반환은 현재
`normal` 정책을 사용한다. 관측에는 자기 손패와 공개 정보만 들어가며 상대의
숨은 패는 포함되지 않는다.

## CPU 컴퓨터의 역할

- GPU 반환 ZIP과 파일별 SHA-256 검증
- 4~10인 `normal` 상대 기준전
- 최신 모델로 on-policy rollout 생성
- `normal`과 과거 모델을 섞은 리그 구성
- GPU가 필요한 숫자 데이터와 학습 코드만 전달
- 기준전을 통과한 모델만 승격

GPU 워밍업 결과를 풀었다고 가정하면 먼저 다음을 실행한다.

```powershell
pnpm run rl:verify-result -- --directory <결과를 푼 폴더>
pnpm run rl:benchmark-model -- \
  --model <policy-weights.json> \
  --matches 100 \
  --acts 5 \
  --output artifacts/rl/benchmarks/bc-warmstart.json
```

PPO 1차 rollout과 GPU 폴더는 다음 명령으로 한 번에 만든다.

```powershell
pnpm run rl:ppo-prepare -- \
  --model <policy-weights.json> \
  --iteration 1 \
  --episodes 200 \
  --acts 3
```

4~10인 각각 200 episode가 생성된다. 각 게임에서 대략 절반의 좌석만 최신
확률 정책이 맡고 나머지는 `normal`이 맡는다. GPU 번들에는 최신 정책이 실제로
선택한 행동의 관측, 합법 행동, 로그확률, 가치 추정, 보상만 포함된다.

2차부터는 과거 승격 모델을 리그 상대에 넣는다.

```powershell
pnpm run rl:ppo-prepare -- \
  --model <최신 actor-critic-weights.json> \
  --opponent-model <과거 actor-critic-weights.json> \
  --normal-opponent-fraction 0.5 \
  --iteration 2 \
  --episodes 200 \
  --acts 3
```

`normal-opponent-fraction 0.5`는 상대 좌석 중 절반을 `normal`, 나머지를 과거
모델 풀에서 뽑는다는 뜻이다. 과거 모델이 없으면 상대 좌석은 모두 `normal`이다.

## GPU 컴퓨터의 역할

GPU 폴더에는 `PROMPT_FOR_GPU_PPO.md`가 들어 있다. GPU Codex가 다음을 수행한다.

- 번들·behavior model SHA-256 검증
- CUDA 사전점검
- GAE 분리 단위 테스트
- 4 epoch PPO 업데이트
- actor-critic JSON과 체크포인트 생성
- 결과 ZIP과 SHA-256 생성

GPU는 게임 강도를 판정하지 않는다. 결과는 CPU 컴퓨터로 돌아와 다시
`rl:benchmark-model`을 거친다.

## 승격 조건

기본 기준전은 4~10인 각각 동일한 수의 match를 실행하고, 홀수 인원에서는
후보와 `normal`의 인원 우위를 번갈아 배정한다. 매 match의 후보 평균 칩과
`normal` 평균 칩 차이로 95% 신뢰구간을 계산한다.

다음 조건을 모두 만족해야 승격한다.

- 4~10인 모든 인원수에서 평균 칩 차이가 양수
- 모든 인원수에서 차이의 95% 신뢰구간 하한도 0보다 큼
- 불법 행동 0건
- 상대 숨은 패 입력 0건
- 이전 승격 모델과의 리그에서도 심각한 퇴행 없음

초기 100 match 기준전은 빠른 선별용이다. 승격 직전에는 match 수를 늘려 다시
검증한다. 통과하지 못한 모델은 프로덕션에 적용하지 않고 다음 rollout의
탐색 후보나 리그 상대로만 보관한다.
