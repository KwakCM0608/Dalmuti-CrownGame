"use client";

/* eslint-disable @next/next/no-img-element -- preprocessed card art is rendered at several animated CSS sizes */

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
  RULEBOOK_DECK,
  RULEBOOK_ROLES,
  RULEBOOK_SECTIONS,
  type RulebookSectionId,
} from "@/lib/rulebook-content";
import {
  cardArtPath,
  cardProfessionName,
  type AppTheme,
} from "@/lib/app-preferences";
import { useAppPreferences } from "./AppPreferencesProvider";
import styles from "./RulebookDialog.module.css";

export const RULEBOOK_DIALOG_ID = "dalmuti-rulebook-dialog";

function themedRulebookCopy(copy: string, theme: AppTheme): string {
  if (theme === "original") return copy;
  return copy
    .replaceAll("달무티(1)", `${cardProfessionName(theme, 1)}(1)`)
    .replaceAll("달무티를 낸", `${cardProfessionName(theme, 1)}를 낸`)
    .replaceAll("농노(12)", `${cardProfessionName(theme, 12)}(12)`)
    .replaceAll("어릿광대", cardProfessionName(theme, 13))
    .replaceAll("조커", cardProfessionName(theme, 13));
}

function useCardImage(): (rank: number) => string {
  const { preferences } = useAppPreferences();
  return useCallback(
    (rank: number) => cardArtPath(preferences.theme, rank),
    [preferences.theme],
  );
}

function RuleCard({
  rank,
  label,
  tone,
}: {
  rank: number;
  label: string;
  tone?: "strong" | "weak" | "joker";
}) {
  const cardImage = useCardImage();
  return (
    <figure className={`${styles.ruleCard} ${tone ? styles[tone] : ""}`}>
      <img src={cardImage(rank)} alt="" loading="lazy" draggable={false} />
      <figcaption>{label}</figcaption>
    </figure>
  );
}

function SectionVisual({ id }: { id: RulebookSectionId }) {
  const { preferences } = useAppPreferences();
  const theme = preferences.theme;
  const cardImage = useCardImage();
  if (id === "goal") {
    return (
      <div className={styles.finishVisual} aria-hidden="true">
        <span><b>1</b><small>달무티</small></span>
        <span><b>2</b><small>총리대신</small></span>
        <span><b>3</b><small>상인</small></span>
        <i>완주 순서가 다음 계급</i>
      </div>
    );
  }

  if (id === "cards") {
    return (
      <div className={styles.strengthVisual} aria-hidden="true">
        <RuleCard
          rank={1}
          label={`${cardProfessionName(theme, 1)} · 가장 강함`}
          tone="strong"
        />
        <span className={styles.strengthLine}>
          <b>강함</b><i>1 → 12</i><b>약함</b>
        </span>
        <RuleCard
          rank={12}
          label={`${cardProfessionName(theme, 12)} · 가장 약한 일반 카드`}
          tone="weak"
        />
        <RuleCard
          rank={13}
          label={cardProfessionName(theme, 13)}
          tone="joker"
        />
      </div>
    );
  }

  if (id === "turn") {
    return (
      <div className={styles.turnExample} aria-hidden="true">
        <div>
          <small>필드</small>
          <span>
            <img src={cardImage(7)} alt="" />
            <img src={cardImage(7)} alt="" />
          </span>
          <b>{cardProfessionName(theme, 7)}(7) × 2장</b>
        </div>
        <i>→</i>
        <div className={styles.validPlay}>
          <small>낼 수 있음</small>
          <span>
            <img src={cardImage(5)} alt="" />
            <img src={cardImage(5)} alt="" />
          </span>
          <b>{cardProfessionName(theme, 5)}(5) × 2장</b>
        </div>
      </div>
    );
  }

  if (id === "trick") {
    return (
      <div className={styles.dalmutiVisual} aria-hidden="true">
        <img src={cardImage(1)} alt="" />
        <strong>DALMUTI</strong>
        <span><i>마르코 PASS</i><i>루나 PASS</i><i>세라핀 PASS</i></span>
      </div>
    );
  }

  if (id === "opening") {
    return (
      <div className={styles.rankDrawVisual} aria-hidden="true">
        {[1, 2, 3, 4, 5].map((rank) => (
          <span
            key={rank}
            style={{ "--rule-rank-index": rank - 1 } as CSSProperties}
          >
            <i className={styles.rankBack} />
            <img src={cardImage(rank)} alt="" />
          </span>
        ))}
        <b>선택 완료 → 계급 공개</b>
      </div>
    );
  }

  if (id === "tax") {
    return (
      <div className={styles.taxVisual} aria-hidden="true">
        <span className={styles.taxPerson}><b>♟</b><small>농노</small></span>
        <span className={styles.taxCards}>
          <img src={cardImage(2)} alt="" />
          <img src={cardImage(4)} alt="" />
          <i>
            {theme === "halloween" ? cardProfessionName(theme, 13) : "조커"}는
            보호
          </i>
        </span>
        <span className={styles.taxArrow}>2장 전달 →</span>
        <span className={styles.taxPerson}><b>♛</b><small>달무티</small></span>
      </div>
    );
  }

  if (id === "revolution") {
    return (
      <div className={styles.revolutionVisual} aria-hidden="true">
        <img src={cardImage(13)} alt="" />
        <strong>REVOLUTION</strong>
        <img src={cardImage(13)} alt="" />
      </div>
    );
  }

  if (id === "roles") {
    return (
      <div className={styles.roleVisual}>
        {RULEBOOK_ROLES.map((role, index) => (
          <span key={role.id}>
            <b>{index + 1}</b>
            <small>{role.name}</small>
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className={styles.controlVisual} aria-hidden="true">
      <span><i>1×</i><b>한 장 선택</b></span>
      <span><i>2×</i><b>같은 숫자 전체</b></span>
      <span><i>30</i><b>초 안에 행동</b></span>
    </div>
  );
}

export function RulebookDialog({
  open,
  onClose,
  gameInProgress = false,
}: {
  open: boolean;
  onClose: () => void;
  gameInProgress?: boolean;
}) {
  const { preferences } = useAppPreferences();
  const dialogRef = useRef<HTMLElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  const [openSectionIds, setOpenSectionIds] = useState<Set<RulebookSectionId>>(
    () => new Set(["goal", "cards", "turn", "trick"]),
  );

  const setSectionOpen = (id: RulebookSectionId, nextOpen = true) => {
    setOpenSectionIds((current) => {
      const next = new Set(current);
      if (nextOpen) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!open) return;

    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusFrame = window.requestAnimationFrame(() =>
      closeButtonRef.current?.focus(),
    );

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;

      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), details > summary, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => !element.hasAttribute("hidden"));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable.at(-1)!;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      restoreFocusRef.current?.focus();
    };
  }, [open]);

  if (!open) return null;

  return (
    <div
      className={styles.layer}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        ref={dialogRef}
        id={RULEBOOK_DIALOG_ID}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="rulebook-title"
        aria-describedby="rulebook-description"
      >
        <header className={styles.header}>
          <span className={styles.crown} aria-hidden="true" />
          <div>
            <small>HOW TO PLAY · BEGINNER GUIDE</small>
            <h2 id="rulebook-title">달무티 완전 가이드</h2>
            <p id="rulebook-description">
              처음 하는 사람도 순서대로 읽으면 바로 플레이할 수 있습니다.
            </p>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className={styles.close}
            aria-label="규칙 닫기"
            onClick={onClose}
          >
            <span aria-hidden="true">×</span>
          </button>
        </header>

        {gameInProgress && (
          <div className={styles.liveNotice} role="status">
            <span aria-hidden="true">●</span>
            <b>게임은 계속 진행 중입니다.</b>
            <small>차례와 제한시간은 룰북을 보는 동안에도 흐릅니다.</small>
          </div>
        )}

        <div className={styles.quickStart}>
          <span>
            <b>1</b>
            <strong>같은 숫자를 묶기</strong>
            <small>한 장 또는 같은 숫자 여러 장</small>
          </span>
          <i>→</i>
          <span>
            <b>2</b>
            <strong>더 낮은 숫자로 응수</strong>
            <small>장수는 반드시 똑같이</small>
          </span>
          <i>→</i>
          <span>
            <b>3</b>
            <strong>손패를 먼저 비우기</strong>
            <small>완주 순서가 다음 계급</small>
          </span>
        </div>

        <div className={styles.body}>
          <nav className={styles.toc} aria-label="룰북 목차">
            <strong>CONTENTS</strong>
            {RULEBOOK_SECTIONS.map((section) => (
              <a
                key={section.id}
                href={`#rulebook-${section.id}`}
                onClick={() => setSectionOpen(section.id)}
              >
                <span>{section.number}</span>
                {section.title}
              </a>
            ))}
            <div className={styles.deckNote}>
              <b>{RULEBOOK_DECK.totalCards}장</b>
              <span>정식 덱</span>
              <small>
                {themedRulebookCopy(RULEBOOK_DECK.composition, preferences.theme)}
              </small>
            </div>
          </nav>

          <article className={styles.sections}>
            {RULEBOOK_SECTIONS.map((section) => (
              <details
                id={`rulebook-${section.id}`}
                className={styles.section}
                key={section.id}
                open={openSectionIds.has(section.id)}
                onToggle={(event) =>
                  setSectionOpen(section.id, event.currentTarget.open)
                }
              >
                <summary>
                  <span>{section.number}</span>
                  <div>
                    <small>{section.eyebrow}</small>
                    <strong>{section.title}</strong>
                  </div>
                  <i aria-hidden="true">＋</i>
                </summary>
                <div className={styles.sectionBody}>
                  <p>{themedRulebookCopy(section.summary, preferences.theme)}</p>
                  <SectionVisual id={section.id} />
                  <ul>
                    {section.points.map((point) => (
                      <li key={point}>
                        {themedRulebookCopy(point, preferences.theme)}
                      </li>
                    ))}
                  </ul>
                  {section.id === "tax" && (
                    <div className={styles.taxPrivacy}>
                      <strong>세금 교환 핵심</strong>
                      <span>
                        {preferences.theme === "halloween"
                          ? cardProfessionName(preferences.theme, 13)
                          : "조커"}
                        는 빼앗기지 않습니다. 교환되는 카드의 정체는 두
                        당사자만 확인할 수 있습니다.
                      </span>
                    </div>
                  )}
                </div>
              </details>
            ))}
          </article>
        </div>

        <footer className={styles.footer}>
          <span>
            <b>한 줄 요약</b>
            같은 장수의 더 낮은 숫자를 내고, 누구보다 먼저 손패를 비우세요.
          </span>
          <button type="button" onClick={onClose}>게임으로 돌아가기</button>
        </footer>
      </section>
    </div>
  );
}
