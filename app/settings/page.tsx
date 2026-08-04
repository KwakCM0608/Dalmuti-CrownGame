"use client";

/* eslint-disable @next/next/no-img-element -- theme previews use the exact card artwork */

import { useAppPreferences } from "@/app/components/AppPreferencesProvider";
import Link from "next/link";
import {
  APP_THEMES,
  cardArtPath,
  cardProfessionName,
  type AppTheme,
} from "@/lib/app-preferences";
import styles from "./settings.module.css";

const THEME_COPY: Record<
  AppTheme,
  { eyebrow: string; title: string; description: string }
> = {
  original: {
    eyebrow: "CLASSIC TABLE",
    title: "오리지널",
    description: "와인색과 금빛, 초록 펠트의 기존 달무티 테마",
  },
  halloween: {
    eyebrow: "DARK COURT",
    title: "할로윈",
    description: "검은 궁정과 회색 필드, 할로윈 카드로 바뀌는 테마",
  },
};

export default function SettingsPage() {
  const { preferences, updatePreferences } = useAppPreferences();

  return (
    <main className={styles.shell}>
      <div className={styles.grain} aria-hidden="true" />
      <header className={styles.header}>
        <Link className={styles.brand} href="/" aria-label="메인 화면으로 돌아가기">
          <span className={styles.brandSeal} aria-hidden="true" />
          <strong>DALMUTI</strong>
        </Link>
        <Link className={styles.back} href="/">
          <span aria-hidden="true">←</span>
          메인으로
        </Link>
      </header>

      <section className={styles.panel} aria-labelledby="settings-title">
        <div className={styles.titleBlock}>
          <span className={styles.gear} aria-hidden="true">⚙</span>
          <div>
            <small>GAME PREFERENCES</small>
            <h1 id="settings-title">환경설정</h1>
            <p>이 기기에만 저장되며 게임 규칙과 진행에는 영향을 주지 않습니다.</p>
          </div>
        </div>

        <section className={styles.section} aria-labelledby="bgm-title">
          <div className={styles.sectionHeading}>
            <div>
              <small>SOUND</small>
              <h2 id="bgm-title">배경음악</h2>
            </div>
            <span className={styles.pending}>음원 연결 준비 중</span>
          </div>

          <div className={styles.soundRow}>
            <label className={styles.switchRow}>
              <span>
                <strong>BGM</strong>
                <small>{preferences.bgmEnabled ? "켜짐" : "꺼짐"}</small>
              </span>
              <input
                type="checkbox"
                checked={preferences.bgmEnabled}
                onChange={(event) =>
                  updatePreferences({ bgmEnabled: event.currentTarget.checked })
                }
              />
              <i aria-hidden="true" />
            </label>

            <label className={styles.volumeRow}>
              <span>
                <strong>BGM 소리 크기</strong>
                <output>{preferences.bgmVolume}%</output>
              </span>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={preferences.bgmVolume}
                onChange={(event) =>
                  updatePreferences({
                    bgmVolume: Number(event.currentTarget.value),
                  })
                }
                aria-label="BGM 소리 크기"
              />
            </label>
          </div>

          <p className={styles.soundNotice}>
            현재는 설정값만 저장됩니다. BGM 파일이 추가되면 이 스위치와 소리 크기가
            그대로 재생에 연결됩니다.
          </p>
        </section>

        <section className={styles.section} aria-labelledby="theme-title">
          <div className={styles.sectionHeading}>
            <div>
              <small>APPEARANCE</small>
              <h2 id="theme-title">카드와 테이블 테마</h2>
            </div>
            <span className={styles.saved}>변경 즉시 저장</span>
          </div>

          <div className={styles.themeGrid} role="radiogroup" aria-label="테마 선택">
            {APP_THEMES.map((theme) => {
              const selected = preferences.theme === theme;
              const copy = THEME_COPY[theme];
              return (
                <button
                  key={theme}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  className={`${styles.themeOption} ${
                    selected ? styles.selected : ""
                  } ${theme === "halloween" ? styles.halloweenOption : ""}`}
                  onClick={() => updatePreferences({ theme })}
                  aria-label={`${copy.title} 테마 · ${cardProfessionName(theme, 1)} 카드 미리보기`}
                >
                  <span className={styles.cardPreview}>
                    <img
                      src={cardArtPath(theme, 1)}
                      alt={`${cardProfessionName(theme, 1)} 카드 앞면`}
                    />
                    <i aria-hidden="true" />
                  </span>
                  <span className={styles.themeCopy}>
                    <small>{copy.eyebrow}</small>
                    <strong>{copy.title}</strong>
                    <span>{copy.description}</span>
                  </span>
                  <b aria-hidden="true">{selected ? "✓" : ""}</b>
                </button>
              );
            })}
          </div>
        </section>

        <footer className={styles.footer}>
          <span>설정은 브라우저와 설치 앱에서 각각 이 기기에 보관됩니다.</span>
          <Link href="/">완료</Link>
        </footer>
      </section>
    </main>
  );
}
