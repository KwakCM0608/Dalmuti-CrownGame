# DALMUTI Crown Game — Reinforcement Learning Research

> **현재 브랜치: `research/rl` — 봇 학습·평가·GPU 연구용**

이 브랜치는 배포용 브랜치가 아닙니다. 행동 공간, 관측 인코더, 시뮬레이터, 모방학습·PPO·MAPPO 파이프라인과 통계 평가를 개발합니다. 제품에 사용할 수 있다고 검증된 모델 파일과 런타임 변경만 `develop`으로 옮깁니다.

## 브랜치 역할

| 브랜치 | 역할 |
| --- | --- |
| `main` | 현재 프로덕션 배포 기준 |
| `develop` | 다음 제품 버전 통합 |
| `research/rl` | 학습·대규모 평가·모델 후보 선별 |

`research/rl`에서 Sites나 앱을 직접 배포하지 않습니다. 모델 승격은 `research/rl` → `develop` → 전체 회귀 검증 → `main` 순서로 진행합니다.

## 연구 원칙

- 목표는 현재 Normal 봇보다 실제 플레이에서 체감될 정도로 강한 모델입니다.
- 봇은 자기 패, 공개된 행동, 상대 남은 장수와 완주 정보만 사용합니다.
- 상대의 숨은 패나 서버 전용 상태를 actor 입력에 포함하지 않습니다.
- 학습 승률만으로 승격하지 않고 여러 인원수·좌석·계급·막에서 평가합니다.
- 최종 평가는 개발 중 사용하지 않은 고정 seed 묶음으로 봉인합니다.
- 통계적으로 불확실하거나 특정 계급에서 크게 퇴보한 후보는 승격하지 않습니다.
- 학습과 평가마다 새 디렉터리를 만들며 기존 실행 결과를 덮어쓰지 않습니다.

## 환경 준비

공통 소스 준비:

```bash
git clone git@github.com:KwakCM0608/Dalmuti-CrownGame.git
cd Dalmuti-CrownGame
git checkout research/rl
pnpm install --frozen-lockfile
```

Node.js 22.13 이상과 pnpm이 필요합니다. GPU 학습 컴퓨터에는 해당 학습 스크립트가 요구하는 Python·PyTorch·CUDA 환경을 별도로 구성합니다. Python 가상환경과 설치된 패키지는 저장소에 커밋하지 않습니다.

## 주요 연구 경로

| 경로 | 내용 |
| --- | --- |
| `training/` | 모델·데이터 구조와 학습 기반 코드 |
| `gpu-training/` | GPU 학습 진입점, 설정과 보조 코드 |
| `scripts/` | rollout, bundle, benchmark, calibration, screening 도구 |
| `tests/*rl*`, `tests/v4-*` | 행동·관측·학습·평가 회귀 테스트 |
| `docs/reinforcement-learning.md` | 전체 RL 구조와 CPU/GPU 작업 분리 |
| `docs/ppo-self-play-pipeline.md` | PPO/self-play 반복 절차 |
| `lib/bot-models/` | 제품 런타임에서 사용할 수 있는 승격 모델 |

현재 프로덕션 어려움 난이도 모델은 `lib/bot-models/hard-ppo5-epoch11.json`입니다. 이 파일을 바꾸는 것만으로 승격이 완료되는 것은 아니며 모델 해시, 로더 호환성, 난이도 매핑과 비교 평가가 함께 검증되어야 합니다.

## 기본 명령

전체 연구 회귀 테스트:

```bash
pnpm run test:rl
```

기준 봇 평가와 rollout 생성 예시:

```bash
pnpm run rl:evaluate -- --matches 100 --acts 3 --lineup easy,normal,hard,hard
pnpm run rl:rollouts -- --episodes 1000 --players 4 --acts 3 --difficulty normal
pnpm run rl:gpu-bundle
```

모델 검증과 비교:

```bash
pnpm run rl:verify-result -- --directory <extracted-result-directory>
pnpm run rl:benchmark-model -- --model <policy-weights.json>
pnpm run rl:compare-models -- --candidate <candidate.json> --reference <reference.json>
pnpm run rl:calibrate-temperature -- --model <policy-weights.json> --data <rollout.ndjson> --seed <seed>
pnpm run rl:screen-checkpoints -- --directory <checkpoint-directory> --output <report-directory>
```

세부 인수와 출력 계약은 각 스크립트의 `--help`, 관련 테스트와 `docs/` 문서를 우선합니다. 명령 이름만 맞추기 위해 결과 디렉터리나 모델 메타데이터 검증을 우회하지 않습니다.

## 권장 반복 절차

1. `main` 또는 최신 `develop` 변경을 연구 브랜치에 병합합니다.
2. CPU에서 결정론적 기준전과 학습 입력 bundle을 생성합니다.
3. bundle의 SHA-256, 소스 커밋, 관측·행동 스키마를 기록합니다.
4. GPU 컴퓨터의 새 실행 디렉터리에서 학습합니다.
5. 결과 ZIP과 SHA-256을 원본 그대로 회수합니다.
6. 결과 구조·모델 해시·카탈로그 호환성을 사전 검증합니다.
7. 여러 인원수와 독립 seed를 병렬 평가합니다.
8. Normal, 현재 프로덕션 Hard와 직접 비교합니다.
9. 계급별·좌석별·인원별 퇴보와 신뢰구간을 확인합니다.
10. 승격 기준을 통과한 경우에만 런타임 모델과 검증 테스트를 `develop`에 반영합니다.

## Git에 올리는 것

- 학습·평가 소스코드
- 작고 재현 가능한 설정과 manifest 예시
- 관측·행동 스키마 및 모델 로더
- 최종 승격 모델과 해시 메타데이터
- 재현 가능한 테스트와 문서

## Git에 올리지 않는 것

- `artifacts/`, `results/`, `outputs/`, `work/`
- 원본 rollout과 대규모 NDJSON 데이터
- 중간 checkpoint와 optimizer state
- Python 가상환경, PyTorch/CUDA 설치 파일과 캐시
- SSH 키, 비밀번호, 토큰과 `.env`
- 다른 컴퓨터에서 생성한 임시 제어·heartbeat 파일

대용량 결과를 전달할 때는 ZIP과 별도 `.sha256` 파일을 함께 사용하고, 검증이 끝날 때까지 원본을 수정하지 않습니다. Git LFS를 임의의 학습 결과 창고로 사용하지 않습니다.

## 제품 승격 체크리스트

- 모델 형식과 observation/action catalogue가 현재 런타임과 정확히 일치
- 모든 선택이 실제 게임에서 합법
- 상대 숨은 패가 입력·로그·critic 외부 출력에 노출되지 않음
- Normal 대비 목표 효과 크기와 신뢰구간 통과
- 지원 인원수 4~10명과 주요 사회 계급별 중대한 퇴보 없음
- 빠른대전과 온라인이 같은 모델·난이도 매핑 사용
- 런타임 추론 시간과 번들 크기가 실제 기기에서 허용 범위
- `pnpm test`, typecheck와 lint 통과
- 모델 SHA-256과 출처 메타데이터를 테스트로 고정

## 저장소 보안과 권리

이 저장소에는 원작 카드 이미지와 상표 관련 자산이 포함되어 있으므로 비공개로 유지합니다. Android 서명키, 서버 자격 증명과 운영 데이터도 이 브랜치에 올리지 않습니다.
