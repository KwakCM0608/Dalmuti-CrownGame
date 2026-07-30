"use client";

import { useEffect, useRef } from "react";
import styles from "./CreditsDialog.module.css";

export const CREDITS_DIALOG_ID = "dalmuti-credits-dialog";

const CREDIT_ITEMS = [
  {
    label: "DEVELOPER",
    value: "Kwak Changmin",
    detail: "Game direction, design, and development",
  },
  {
    label: "ASSISTANT",
    value: "Kang Donghyeon",
    detail: "Project assistance and playtesting support",
  },
  {
    label: "LABORATORY",
    value: "Distributed Computing Lab (DCLab)",
    detail: "Project affiliation",
  },
  {
    label: "AI DEVELOPMENT SUPPORT",
    value: "GPT-5.6 Sol",
    detail: "Built with OpenAI GPT-5.6 Sol",
  },
] as const;

export function CreditsDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;

    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;

    const focusFrame = window.requestAnimationFrame(() => {
      dialogRef.current?.focus();
    });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("keydown", handleKeyDown);
      restoreFocusRef.current?.focus();
    };
  }, [onClose, open]);

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
        id={CREDITS_DIALOG_ID}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="credits-title"
        aria-describedby="credits-description"
        tabIndex={-1}
      >
        <button
          type="button"
          className={styles.close}
          onClick={onClose}
          aria-label="Close credits"
        >
          ×
        </button>

        <span className={styles.crown} aria-hidden="true" />
        <small>PROJECT CREDITS</small>
        <h2 id="credits-title">DALMUTI</h2>
        <p id="credits-description">
          A browser and Android multiplayer card game project.
        </p>

        <dl className={styles.list}>
          {CREDIT_ITEMS.map((item) => (
            <div key={item.label}>
              <dt>{item.label}</dt>
              <dd>
                <strong>{item.value}</strong>
                <span>{item.detail}</span>
              </dd>
            </div>
          ))}
        </dl>

        <p className={styles.notice}>
          This is an independent digital implementation. Referenced game
          artwork and trademarks remain the property of their respective
          rights holders.
        </p>

        <button type="button" className={styles.done} onClick={onClose}>
          CLOSE
        </button>
      </section>
    </div>
  );
}
