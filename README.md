# DALMUTI Crown Game — Development

> **현재 브랜치: `develop` — 다음 업데이트 통합·검증용**

이 브랜치는 빠른대전, 온라인, PWA와 Android 변경을 함께 통합하고 전체 회귀 테스트를 실행하는 곳입니다. 실제 서비스 배포 기준은 `main`이며, `develop`을 프로덕션에 직접 배포하지 않습니다.

## 브랜치 흐름

```text
feature/* 또는 fix/* → develop → 전체 검증 → main
research/rl → 검증된 런타임 모델만 develop → main
```

- `main`: 현재 프로덕션과 동일한 배포 기준
- `develop`: 다음 버전 후보 통합
- `research/rl`: AI 학습·평가와 GPU 실험
- `feature/<name>`: 새로운 기능
- `fix/<name>`: 일반 버그 수정
- `hotfix/<name>`: 현재 프로덕션의 긴급 수정

하나의 기능은 가능한 한 하나의 임시 브랜치에서 완성하고, 검증 후 `develop`에 병합합니다. 빠른대전·온라인·Android를 서로 다른 영구 브랜치로 분리하지 않습니다. 세 환경이 게임 규칙과 표시 코드를 공유하므로 영구 분리는 동작 차이를 만들기 쉽습니다.

## 개발 환경 준비

```bash
git clone git@github.com:KwakCM0608/Dalmuti-CrownGame.git
cd Dalmuti-CrownGame
git checkout develop
pnpm install --frozen-lockfile
pnpm dev
```

모바일 실기기에서 같은 네트워크로 확인할 때:

```bash
pnpm dev:lan
```

필수 버전:

- Node.js 22.13 이상
- pnpm
- Next.js App Router + vinext
- Cloudflare Workers와 D1 호환 환경

## 변경 전 확인사항

1. 수정 범위를 빠른대전/온라인과 PC/모바일 웹/설치 앱으로 구분합니다.
2. 빠른대전의 UI·애니메이션을 공통 기준으로 사용합니다.
3. 한 모드만 요청받은 수정은 반대 모드에 유출시키지 않습니다.
4. 온라인 행동은 서버 snapshot을 즉시 표시하지 않고 공개 애니메이션 큐가 끝난 뒤 정착시킵니다.
5. 봇에는 자기 패와 공개 정보만 전달합니다.
6. 세금 카드 정체는 교환 당사자에게만 공개합니다.
7. 임시 대혁명 테스트 모드와 기타 강제 QA 설정은 병합 전에 해제합니다.
8. 커스텀 스플래시는 제거 상태를 유지합니다.

누적된 반응형 CSS의 우선순위가 민감하므로 `app/globals.css`와 `app/online/online.module.css`는 요청 범위를 지정한 마지막 override 방식을 우선합니다.

## 검증

변경을 `develop`에 병합하기 전에 다음을 모두 통과시킵니다.

```bash
pnpm test
pnpm run typecheck
pnpm run lint
git diff --check
```

lint의 기존 `<img>` 최적화 경고는 오류가 아니지만, 새로운 오류나 경고가 추가되면 원인을 확인합니다. 실제 기기 화면은 정적 테스트만으로 확정하지 않고 PC·모바일 웹·설치 앱 범위를 구분해 최종 확인합니다.

## D1과 Sites

- 논리 D1 바인딩 이름은 `DB`입니다.
- `.openai/hosting.json`의 기존 Sites 프로젝트 ID를 임의로 변경하지 않습니다.
- 스키마 변경은 `drizzle/` 마이그레이션을 생성하고 함께 검토합니다.
- 운영 D1 데이터와 자격 증명은 저장소에 포함하지 않습니다.
- `develop`은 테스트용이며 실제 배포는 검증 후 `main`에서 진행합니다.

## AI 모델 반영

학습과 대규모 평가는 `research/rl`에서 진행합니다. 새 모델을 제품에 넣을 때는 다음 자료를 함께 가져옵니다.

- 런타임에서 직접 읽는 모델 파일
- SHA-256과 모델 출처 메타데이터
- Normal 및 현재 최강 모델과의 비교 보고서
- 난이도 매핑과 숨은 정보 비사용을 검증하는 테스트

현재 어려움 난이도 모델은 `lib/bot-models/hard-ppo5-epoch11.json`입니다.

## `main` 병합 조건

- 전체 빌드와 테스트 통과
- typecheck 오류 0건
- lint 오류 0건
- D1 마이그레이션 검토 완료
- 테스트용 강제 모드 비활성화
- PC·모바일·설치 앱 영향 범위 확인
- 사용자 화면 검수 완료
- 릴리스 README와 버전 정보 확인

## 저장소 보안과 권리

`.env`, 토큰, Android 서명키, Sites 자격 증명, 학습 원본 데이터와 대용량 체크포인트는 커밋하지 않습니다. 이 프로젝트에는 원작 카드 이미지와 상표 관련 자산이 포함되어 있으므로 저장소는 비공개로 유지합니다.
