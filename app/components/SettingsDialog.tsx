"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./SettingsDialog.module.css";

export const SETTINGS_DIALOG_ID = "dalmuti-settings-dialog";

type EffectMode = "rich" | "light";

const EFFECT_MODE_STORAGE_KEY = "dalmuti.preferences.effects";

function storedEffectMode(): EffectMode {
  if (typeof window === "undefined") return "rich";
  return window.localStorage.getItem(EFFECT_MODE_STORAGE_KEY) === "light"
    ? "light"
    : "rich";
}

function applyEffectMode(mode: EffectMode) {
  document.documentElement.dataset.dalmutiEffects = mode;
}

export function PreferenceRuntime() {
  useEffect(() => {
    applyEffectMode(storedEffectMode());
  }, []);

  return null;
}

export function SettingsDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [effectMode, setEffectMode] = useState<EffectMode>(() =>
    storedEffectMode(),
  );
  const dialogRef = useRef<HTMLElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const focusTimer = window.requestAnimationFrame(() => {
      dialogRef.current?.focus();
    });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.cancelAnimationFrame(focusTimer);
      window.removeEventListener("keydown", handleKeyDown);
      restoreFocusRef.current?.focus();
    };
  }, [onClose, open]);

  const chooseEffectMode = (mode: EffectMode) => {
    setEffectMode(mode);
    window.localStorage.setItem(EFFECT_MODE_STORAGE_KEY, mode);
    applyEffectMode(mode);
  };

  if (!open) return null;

  return (
    <div
      className={styles.layer}
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        ref={dialogRef}
        id={SETTINGS_DIALOG_ID}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        aria-describedby="settings-description"
        tabIndex={-1}
      >
        <button
          type="button"
          className={styles.close}
          onClick={onClose}
          aria-label="환경설정 닫기"
        >
          ×
        </button>
        <small>GAME PREFERENCES</small>
        <h2 id="settings-title">환경설정</h2>
        <p id="settings-description">
          이 기기에서 사용할 화면 연출 수준을 선택하세요.
        </p>

        <fieldset className={styles.options}>
          <legend>화면 연출</legend>
          <button
            type="button"
            className={effectMode === "rich" ? styles.selected : ""}
            aria-pressed={effectMode === "rich"}
            onClick={() => chooseEffectMode("rich")}
          >
            <strong>풍부한 연출</strong>
            <span>모든 조명, 입자 효과와 장면 전환을 표시합니다.</span>
          </button>
          <button
            type="button"
            className={effectMode === "light" ? styles.selected : ""}
            aria-pressed={effectMode === "light"}
            onClick={() => chooseEffectMode("light")}
          >
            <strong>가벼운 연출</strong>
            <span>저사양 휴대전화에서 장식 효과를 줄여 플레이합니다.</span>
          </button>
        </fieldset>

        <p className={styles.note}>선택한 설정은 이 기기에 자동 저장됩니다.</p>
        <button type="button" className={styles.done} onClick={onClose}>
          완료
        </button>
      </section>
    </div>
  );
}
