# DALMUTI GPU 워밍업 학습 번들

이 폴더는 DALMUTI 전체 프로젝트가 없는 GPU 컴퓨터에서도 행동 모방(BC) 모델을
학습할 수 있는 독립 실행 번들이다. 입력 데이터의 교사 행동은 `normal` 봇이
만들었으며, DAgger 데이터도 약한 모델이 방문한 상태에 대해 `normal` 봇이 다시
정답을 붙인 것이다.

이 단계의 목적은 PPO 자기대전을 시작할 정책 워밍업 모델을 만드는 것이다.
검증 정확도가 높더라도 `normal`보다 강하다는 뜻은 아니다. 실제 게임 강도는
학습 결과를 원래 DALMUTI 컴퓨터로 돌려보낸 뒤 다수의 기준전으로 판정한다.

## 전달받은 사람이 가장 먼저 할 일

1. 압축을 풀고 `PROMPT_FOR_GPU_CODEX.md`를 GPU 컴퓨터의 Codex에 그대로 전달한다.
2. Codex가 CUDA 지원 PyTorch 설치와 사전점검을 마치게 한다.
3. 학습이 끝나면 `results` 폴더의 결과 ZIP과 SHA-256 파일을 원래 컴퓨터로
   돌려보낸다.

## 권장 환경

- NVIDIA CUDA 지원 GPU
- 최신 NVIDIA 드라이버
- Python 3.11 또는 3.12 64비트
- 시스템 메모리 16GB 이상 권장
- 여유 저장공간 8GB 이상

PyTorch 설치 명령은 GPU 드라이버와 운영체제에 따라 달라질 수 있으므로
`requirements.txt`만 무조건 실행하지 말고, GPU 컴퓨터에서 공식 PyTorch 설치
방식을 확인해 CUDA 빌드를 설치한다. `torch.cuda.is_available()`가 `True`가
되기 전에는 학습을 시작하지 않는다.

## 수동 실행 요약

번들 무결성 검증:

```powershell
python verify_bundle.py
```

CUDA 사전점검:

```powershell
python preflight.py --device cuda --output hardware-report.json
```

전체 학습·검증·결과 패키징:

```powershell
python run_gpu_training.py `
  --output "models\bc-warmstart-v3" `
  --epochs 80 `
  --batch-size 4096 `
  --learning-rate 0.0003 `
  --hidden-sizes 256,256 `
  --supervised-weight 5 `
  --patience 10 `
  --seed 20260731
```

Linux의 경우 줄바꿈 문자만 셸 형식에 맞게 바꾸면 된다.

GPU 메모리가 부족하면 `--batch-size`를 2048, 1024 순서로 낮춘다. 모델 구조와
시드, 학습률 등 다른 설정은 임의로 바꾸지 않는다. 오류가 발생하면 원인을
수정한 뒤 동일한 출력 폴더로 처음부터 다시 학습한다.

## 결과물

성공하면 `results`에 다음 두 파일이 생성된다.

- `bc-warmstart-v3-result.zip`
- `bc-warmstart-v3-result.zip.sha256`

ZIP에는 다음이 포함된다.

- `checkpoint.pt`: PPO 초기화와 추가 PyTorch 학습용 체크포인트
- `policy-weights.json`: TypeScript 게임 시뮬레이터용 순수 MLP 가중치
- `policy-metadata.json`: 환경·데이터·학습 설정
- `training-metrics.json`: epoch별 손실과 정확도
- `hardware-report.json`: CUDA 및 GPU 정보
- `data-verification.json`: 입력 데이터 검증 결과
- `training.log`: 전체 실행 기록
- `result-manifest.json`: 결과 파일별 크기와 SHA-256

결과 ZIP만으로 실제 게임 강도를 판정할 수 없다. 원래 DALMUTI 컴퓨터에서
`normal` 상대 기준전을 통과하기 전에는 프로덕션 게임에 적용하지 않는다.
