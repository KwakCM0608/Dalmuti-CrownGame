# DALMUTI Crown Game

> **현재 브랜치: `main` — 프로덕션 배포 기준**

달무티를 빠른대전과 온라인 모드로 플레이할 수 있도록 만든 비공식·비상업 팬 프로젝트입니다. `main`은 실제 서비스에 배포할 수 있는 검증된 소스만 유지합니다.

## 현재 배포 기준

- 웹: OpenAI Sites 버전 35
- Android TWA: 1.0.8 (`versionCode 9`)
- Node.js: 22.13 이상
- 패키지 관리자: pnpm
- 런타임: Next.js App Router + vinext + Cloudflare Workers
- 데이터베이스: Cloudflare D1 (`DB` 바인딩)
- 어려움 봇: `lib/bot-models/hard-ppo5-epoch11.json`

## 브랜치 정책

| 브랜치 | 용도 | 직접 배포 |
| --- | --- | --- |
| `main` | 검증을 통과한 프로덕션 소스 | 가능 |
| `develop` | 다음 업데이트 통합과 회귀 검증 | 금지 |
| `research/rl` | 봇 학습·평가·GPU 연구 | 금지 |

일반 기능은 `feature/<name>`, 버그 수정은 `fix/<name>`, 긴급 수정은 `hotfix/<name>` 임시 브랜치에서 작업합니다. 검증된 변경만 `develop`을 거쳐 `main`으로 병합합니다.

## 새 컴퓨터에서 복원

```bash
git clone git@github.com:KwakCM0608/Dalmuti-CrownGame.git
cd Dalmuti-CrownGame
git checkout main
pnpm install --frozen-lockfile
```

로컬 실행:

```bash
pnpm dev
```

같은 네트워크의 모바일 기기에서 확인할 때:

```bash
pnpm dev:lan
```

## 필수 검증

```bash
pnpm test
pnpm run typecheck
pnpm run lint
git diff --check
```

`pnpm test`는 프로덕션 빌드 후 빠른대전·온라인 엔진, UI 표시 계약, PWA, 테마, 봇과 학습 기반 회귀 테스트를 실행합니다.

## OpenAI Sites에 동일 프로젝트로 재배포

이 저장소에는 다음 Sites 연결 정보가 포함되어 있습니다.

```json
{
  "project_id": "appgprj_6a61b5d3d50c8191ac800c16dc1421d5",
  "d1": "DB",
  "r2": null
}
```

다른 컴퓨터에서 동일 프로젝트에 배포하려면:

1. 동일한 OpenAI Sites 프로젝트에 접근할 수 있는 계정으로 Codex에 로그인합니다.
2. 이 저장소의 `main`을 받고 의존성을 설치합니다.
3. 위의 전체 검증을 통과시킵니다.
4. `.openai/hosting.json`의 기존 `project_id`를 유지한 채 Sites 배포를 요청합니다.
5. 새 Sites 프로젝트를 만들지 말고 기존 프로젝트에 새 버전을 저장·배포합니다.
6. 배포 후 상태가 `succeeded`인지 확인하고 빠른대전과 온라인 입장을 확인합니다.

주의사항:

- Git에는 운영 D1 데이터, Sites 로그인 정보와 배포 자격 증명이 저장되지 않습니다.
- 다른 계정으로 같은 `project_id`에 배포하려면 해당 Sites 프로젝트 권한이 별도로 필요합니다.
- D1 스키마를 변경했다면 `drizzle/`의 마이그레이션도 반드시 함께 검토·배포합니다.
- 온라인 API는 서비스워커에서 캐시하면 안 됩니다.

## 주요 디렉터리

| 경로 | 내용 |
| --- | --- |
| `app/` | 빠른대전·온라인 화면과 API 라우트 |
| `lib/` | 게임 엔진, 봇, D1 저장소, 공통 규칙 |
| `public/` | 카드·테마·PWA·브랜드 자산 |
| `drizzle/`, `db/` | D1 스키마와 마이그레이션 |
| `tests/` | 전체 회귀 테스트 |
| `scripts/` | 검증·봇 학습·평가 도구 |
| `android-twa/` | Android Bubblewrap/TWA 프로젝트 |
| `docs/` | UI 계약, 규칙과 연구 문서 |
| `.openai/hosting.json` | 기존 Sites 프로젝트와 D1 논리 바인딩 |

## 저장소에 포함하지 않는 파일

다음 파일은 새 컴퓨터에서 다시 생성하거나 별도로 안전하게 전달해야 합니다.

- `node_modules/`, `.next/`, `.vinext/`, `dist/`, `.wrangler/`
- `artifacts/`, `results/`, 학습 중간 체크포인트와 원본 데이터
- `.env*`, 토큰, 로그인 정보와 기타 비밀값
- Android 서명키(`*.jks`, `*.keystore`)와 로컬 빌드 결과
- 로컬 캐시와 임시 출력물

Android 스토어 업데이트에 사용하는 기존 서명키는 GitHub에 올리지 말고 별도 암호화 백업으로 전달해야 합니다. 해당 키가 없으면 같은 패키지의 기존 앱을 업데이트할 수 없습니다.

## 제품 불변 조건

- 빠른대전의 UI와 애니메이션을 공통 기준으로 사용합니다.
- 온라인은 서버 권위형이며 상대방의 숨은 패를 클라이언트나 봇에 공개하지 않습니다.
- 세금 교환 카드 정체는 교환 당사자에게만 공개합니다.
- PC·모바일 웹·설치형 앱의 CSS 범위를 구분합니다.
- 임시 대혁명 강제 테스트 모드는 배포 전에 반드시 꺼져 있어야 합니다.
- 커스텀 스플래시는 제거 상태를 유지합니다.

## 라이선스와 공개 범위

이 저장소에는 원작 게임명, 카드 이미지와 상표 관련 자산이 포함되어 있습니다. 권리자의 공식 승인을 받은 오픈소스 배포물이 아니므로 저장소를 **비공개로 유지**하고, 공개 배포·재배포·상업 이용 전에는 별도의 권리 확인이 필요합니다.

게임 원작과 카드 일러스트의 권리는 각 권리자에게 있으며, 이 프로젝트의 크레딧과 팬 구현 면책 문구가 사용 허가를 대신하지 않습니다.
