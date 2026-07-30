"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./CreditsDialog.module.css";

export const CREDITS_DIALOG_ID = "dalmuti-credits-dialog";
const CREDITS_CLOSE_DURATION_MS = 180;

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
    value: "DCLab(Distributed Computing Lab)",
    detail: "Project affiliation",
  },
  {
    label: "AI DEVELOPMENT SUPPORT",
    value: "GPT-5.6 Sol",
    detail: "Built with OpenAI GPT-5.6 Sol",
  },
  {
    label: "ORIGINAL GAME DESIGN",
    value: "Richard Garfield",
    detail: "The Great Dalmuti",
  },
  {
    label: "ORIGINAL CARD ILLUSTRATIONS",
    value: "Margaret Organ-Kean",
    detail: "Original tabletop artwork",
  },
  {
    label: "ORIGINAL EDITION",
    value: "Wizards of the Coast, Inc.",
    detail: "Original 1995 edition © 1995 Wizards of the Coast, Inc.",
  },
] as const;

export function CreditsDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [closing, setClosing] = useState(false);
  const dialogRef = useRef<HTMLElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const closingRef = useRef(false);
  const closeTimerRef = useRef<number | null>(null);

  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      onClose();
      return;
    }

    closingRef.current = true;
    setClosing(true);
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null;
      closingRef.current = false;
      setClosing(false);
      onClose();
    }, CREDITS_CLOSE_DURATION_MS);
  }, [onClose]);

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
      if (event.key === "Escape") requestClose();
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("keydown", handleKeyDown);
      restoreFocusRef.current?.focus();
    };
  }, [open, requestClose]);

  useEffect(
    () => () => {
      if (closeTimerRef.current !== null) {
        window.clearTimeout(closeTimerRef.current);
      }
    },
    [],
  );

  if (!open) return null;

  return (
    <div
      className={`${styles.layer} ${closing ? styles.closing : ""}`}
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) requestClose();
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
          onClick={requestClose}
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
          This is an independent, non-commercial fan implementation and is
          not affiliated with, endorsed by, or sponsored by Wizards of the
          Coast. The Great Dalmuti and Wizards names and logos are trademarks
          of Wizards of the Coast. All referenced game artwork, trademarks,
          and other rights remain with their respective owners.
        </p>

        <button type="button" className={styles.done} onClick={requestClose}>
          CLOSE
        </button>
      </section>
    </div>
  );
}
