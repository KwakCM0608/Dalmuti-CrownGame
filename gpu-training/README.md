# DALMUTI GPU 행동 모방 학습 패키지

이 폴더는 DALMUTI 웹 프로젝트 없이 독립 실행된다. 필요한 입력은
`data/bc-p4-v2.ndjson`부터 `data/bc-p10-v2.ndjson`까지의 합성 숫자
rollout뿐이다. 카드 이미지, 사용자 데이터, 서버 데이터베이스는 사용하지
않는다.

## 1. GPU 컴퓨터로 복사

현재 컴퓨터에서 만들어진 `artifacts/rl/gpu-bundle-v2` 폴더 전체를 GPU
컴퓨터로 복사한다. 권장 위치 예시는 `C:\DalmutiTraining`이다.

## 2. Python 환경

Python 3.11 또는 3.12의 64비트 버전을 권장한다.

압축을 푼 직후 먼저 전송 무결성을 확인한다.

```powershell
python verify_bundle.py
```

```powershell
cd C:\DalmutiTraining
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치 후 CUDA 인식 여부를 확인한다.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

`False`가 나오면 학습을 시작하지 말고 GPU 드라이버와 설치한 PyTorch
빌드가 맞는지 확인한다.

## 3. 데이터 검증

```powershell
python verify_data.py --data "data\*-p*-v2.ndjson"
```

V2 초기 데이터는 강제 행동을 제외한 학습·검증 샘플을 읽어야 한다.
선택 행동이 합법 행동 목록에 없거나 관측값 길이가 172가 아니면 즉시
실패한다.

## 4. GPU 학습

```powershell
python train_bc.py `
  --data "data\*-p*-v2.ndjson" `
  --output "models\bc-v2" `
  --device cuda `
  --epochs 60 `
  --batch-size 4096 `
  --learning-rate 0.0003 `
  --hidden-sizes 256,256 `
  --supervised-weight 5 `
  --patience 8
```

GPU 메모리가 부족하면 `--batch-size 2048` 또는 `1024`로 낮춘다.

## 5. 반환할 파일

학습이 끝나면 `models/bc-v2`의 다음 파일을 현재 DALMUTI 컴퓨터로
복사한다.

- `policy-weights.json`: 브라우저와 서버에서 공통으로 읽을 순수 MLP 가중치
- `policy-metadata.json`: 학습 환경과 데이터 정보
- `training-metrics.json`: epoch별 손실과 일치율
- `checkpoint.pt`: GPU에서 추가 학습할 PyTorch 체크포인트

행동 모방의 검증 정확도는 교사 정책 재현율이지 실제 게임 실력 자체가
아니다. 반환된 모델은 반드시 CPU 시뮬레이터에서 기존 normal/hard 봇과
대전 평가한 후에만 게임에 적용한다.
