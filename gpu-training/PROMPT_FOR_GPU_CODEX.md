# GPU 컴퓨터의 Codex에 전달할 프롬프트

아래 내용을 이 파일이 들어 있는 압축을 GPU 컴퓨터에 푼 다음, 그 폴더를 Codex
작업공간으로 열고 새 작업에 그대로 붙여 넣으세요.

---

이 폴더는 DALMUTI 봇의 GPU 워밍업 학습을 위한 독립 실행 번들이다. 전체 게임
프로젝트는 이 컴퓨터에 없으며 필요하지 않다. 네가 직접 환경 점검, CUDA 지원
PyTorch 설치, 데이터 검증, 학습, 결과 패키징까지 끝까지 진행해라. 나에게 명령을
대신 실행하라고 하지 말고, 권한 승인이 필요한 경우에만 승인을 요청해라.

목표와 제한:

- 이번 단계는 `normal` 봇보다 강하다고 판정하는 단계가 아니라, 이후 PPO
  자기대전에 사용할 BC+DAgger 워밍업 모델을 생성하는 단계다.
- 입력 정답은 `normal` 정책이다. DAgger 파일의 행동 상태는 이전 후보 모델이
  만들었지만, 학습 정답은 `supervisedActionIndex`에 기록된 `normal` 행동이다.
- 모델 구조, 관측 172개, 행동 506개, 데이터 파일은 임의로 변경하지 마라.
- CUDA가 실제로 활성화되지 않으면 CPU 학습으로 대체하지 말고 원인을 해결해라.
- 검증 정확도만 보고 `normal`보다 강해졌다고 주장하지 마라.
- 전체 게임 코드나 카드 이미지, 서버, 웹 배포 작업은 하지 마라.

진행 순서:

1. `README.md`, `schema.json`, `bundle-manifest.json`을 읽어 구성과 계약을
   확인한다.
2. `python verify_bundle.py`를 실행해 모든 파일 크기와 SHA-256을 검사한다.
   실패하면 학습하지 말고 손상된 파일을 정확히 보고한다.
3. `nvidia-smi`, Python 버전, GPU 모델과 VRAM, NVIDIA 드라이버를 확인한다.
4. 프로젝트 폴더 안에 `.venv`를 만들고 활성화한다. Python 3.11 또는 3.12를
   사용한다.
5. 공식 PyTorch 설치 방법을 기준으로 이 GPU와 드라이버에 맞는 CUDA 지원
   PyTorch를 설치한다. `numpy`도 설치한다. 단순히 CPU 전용 torch를 설치하면
   안 된다.
6. 다음을 실행해 `torch.cuda.is_available()`가 `True`이고 GPU 이름이 실제로
   출력되는지 확인한다.

   `python preflight.py --device cuda --output hardware-report.json`

7. 다음 명령으로 검증, 학습, 로그 저장, 결과 ZIP 생성을 한 번에 수행한다.

   `python run_gpu_training.py --output models/bc-warmstart-v3 --epochs 80 --batch-size 4096 --learning-rate 0.0003 --hidden-sizes 256,256 --supervised-weight 5 --patience 10 --seed 20260731`

8. CUDA out-of-memory가 발생한 경우에만 배치 크기를 2048, 그래도 실패하면
   1024로 낮춰 같은 설정으로 처음부터 재실행한다. 다른 하이퍼파라미터는
   바꾸지 않는다.
9. 학습이 끝나면 `results/bc-warmstart-v3-result.zip`과
   `results/bc-warmstart-v3-result.zip.sha256`이 생성되었는지 확인한다.
10. SHA-256을 다시 계산해 `.sha256` 파일과 일치하는지 검증한다.
11. 최종 답변에는 다음만 명확히 정리한다.
    - GPU와 CUDA/PyTorch 버전
    - 사용한 최종 배치 크기
    - best epoch
    - train/validation sample 수
    - validation top-1/top-3 정확도
    - 결과 ZIP의 절대 경로, 크기, SHA-256
    - 오류나 설정 변경이 있었다면 정확한 내용

실제 `normal` 상대 기준전과 PPO 자기대전은 결과 ZIP을 원래 DALMUTI 컴퓨터로
돌려보낸 후 별도로 수행한다. 웹사이트나 프로덕션에는 아무것도 배포하지 마라.

---
