# GPU 컴퓨터의 Codex에 전달할 PPO 프롬프트

이 파일은 CPU 컴퓨터가 생성한 PPO 전달 ZIP 안에 자동으로 포함된다. 압축을 푼
폴더를 GPU 컴퓨터의 Codex 작업공간으로 연 뒤 아래 내용을 그대로 전달한다.

---

이 폴더에는 DALMUTI의 한 번 분량 on-policy PPO rollout, 그 행동을 생성한
behavior model, action-masked PPO 학습 코드가 들어 있다. 전체 게임 프로젝트는
필요하지 않다. 네가 직접 번들 검증, CUDA 점검, PPO 업데이트, 결과 패키징까지
완료해라. 사용자에게 명령 실행을 대신 시키지 말고 권한 승인만 요청해라.

반드시 지킬 조건:

- rollout과 `behavior-model.json`의 SHA-256이 다르면 학습하지 않는다.
- CPU 전용 PyTorch로 대체하지 않는다. `torch.cuda.is_available()`가 `True`여야
  한다.
- 행동 마스크를 제거하거나 불법 행동을 학습 대상으로 추가하지 않는다.
- 관측은 172개, 행동은 506개 계약을 그대로 유지한다.
- 기본 하이퍼파라미터를 임의로 튜닝하지 않는다.
- 한 번의 PPO 업데이트 결과만으로 `normal`보다 강하다고 주장하지 않는다.
- 웹사이트·게임 코드·프로덕션 배포는 건드리지 않는다.

진행:

1. `ppo-schema.json`, `bundle-manifest.json`을 읽는다.
2. `python verify_bundle.py`로 파일 크기와 SHA-256을 검사한다.
3. 기존 `.venv`가 있으면 CUDA PyTorch가 정상인지 확인하고, 없으면 Python
   3.11/3.12 가상환경과 GPU에 맞는 공식 CUDA PyTorch를 설치한다.
4. `python preflight.py --device cuda --output hardware-report.json`을 실행한다.
5. `python -m unittest test_ppo.py test_ppo_core_upgrades.py test_v3_action_conditioned.py test_non_card_action_conditioned.py`를 실행해 trajectory 분리, 보정된 PPO 계약, V3·비카드 모델 계약이 통과하는지
   확인한다.
6. 다음 명령을 실행한다.

   먼저 `gpu-run-config.json`을 읽고, 그 파일의 정확한 `rolloutTemperature` 값을 `--rollout-temperature`로 전달해야 합니다. 예:

   `python run_gpu_ppo.py --output models/ppo-iteration --epochs 12 --batch-size 4096 --learning-rate 0.0001 --gamma 1 --gae-lambda 1 --skip-forced-policy-time --terminal-rank-auxiliary-coefficient 0.05 --rollout-temperature <gpu-run-config.json의 값> --clip-coefficient 0.2 --value-coefficient 0.5 --entropy-coefficient 0.01 --target-kl 0.015 --seed 20260801`

   `gpu-run-config.json`의 `requiredRunGpuPpoArguments`를 그대로 사용하고, 비교 실험은 서로 다른 새 작업 디렉터리에서 `--terminal-rank-auxiliary-coefficient 0`과 `0.05`를 각각 한 번씩 실행한다. 기존 output 또는 result 경로는 재사용하지 않는다.

7. CUDA 메모리 부족일 때만 배치를 2048, 1024 순서로 낮춘다.
8. `results/ppo-iteration-result.zip`과 `.sha256`을 다시 검증한다.
9. 최종 답변에는 GPU/CUDA/PyTorch, 샘플·trajectory 수, 실행 epoch,
   policy/value loss, entropy, KL, clip fraction, explained variance, 결과 ZIP
   절대 경로·크기·SHA-256을 보고한다.

결과 ZIP은 CPU DALMUTI 컴퓨터에서 실제 기준전을 거친다. 기준전 결과 없이
성능 향상을 단정하거나 프로덕션에 적용하지 마라.

---
