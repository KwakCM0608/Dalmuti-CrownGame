import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const {
  RULEBOOK_DECK,
  RULEBOOK_PLAYER_COUNTS,
  RULEBOOK_ROLES,
  RULEBOOK_SECTIONS,
} = await import(new URL("../lib/rulebook-content.ts", import.meta.url));

test("beginner rulebook covers the complete implemented ruleset", () => {
  assert.equal(
    Array.from({ length: RULEBOOK_DECK.numberedRanks }, (_, index) => index + 1)
      .reduce((sum, rank) => sum + rank, 0) +
      RULEBOOK_DECK.jokerCount,
    RULEBOOK_DECK.totalCards,
  );
  assert.deepEqual(RULEBOOK_PLAYER_COUNTS, {
    quick: "4~10인",
    online: "4~8인",
  });
  assert.deepEqual(
    RULEBOOK_ROLES.map((role) => role.name),
    ["달무티", "총리대신", "상인", "소작농", "농노"],
  );

  const sectionIds = RULEBOOK_SECTIONS.map((section) => section.id);
  assert.equal(new Set(sectionIds).size, sectionIds.length);
  assert.deepEqual(sectionIds, [
    "goal",
    "cards",
    "turn",
    "trick",
    "opening",
    "tax",
    "revolution",
    "roles",
    "controls",
  ]);

  const copy = RULEBOOK_SECTIONS.flatMap((section) => [
    section.title,
    section.summary,
    ...section.points,
  ]).join("\n");
  for (const expected of [
    "같은 장수",
    "더 낮은 숫자",
    "어릿광대만 한 장 또는 두 장",
    "달무티(1)를 내면 나머지 플레이어가 즉시 자동 PASS",
    "제1막은 세금 교환 없이",
    "농노 → 달무티",
    "소작농 → 총리대신",
    "조커를 제외",
    "원하는 카드를 골라 돌려줍니다",
    "어릿광대 두 장",
    "농노의 대혁명",
    "모든 플레이어의 계급 순서가 뒤집힙니다",
    "30초 안에 행동하지 않으면 자동으로 PASS",
    "1위는 4칩",
    "2위는 3칩",
    "꼴찌는 0칩",
    "모든 플레이어는 순위와 관계없이 2칩",
    "누적 칩",
    "4인 게임에는 상인이 없습니다",
    "6인 이상은 인원이 늘 때마다 상인이 한 명씩",
    "더블클릭",
  ]) {
    assert.match(copy, new RegExp(expected.replace(/[()]/g, "\\$&")));
  }
});

test("quick and online modes share an accessible visual rulebook", async () => {
  const [component, styles, quickPage, onlinePage] = await Promise.all([
    readFile(
      new URL("../app/components/RulebookDialog.tsx", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL("../app/components/RulebookDialog.module.css", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
  ]);

  for (const page of [quickPage, onlinePage]) {
    assert.match(page, /RulebookDialog/);
    assert.match(page, /RULEBOOK_DIALOG_ID/);
    assert.match(page, /aria-haspopup="dialog"/);
    assert.match(page, /aria-controls=\{RULEBOOK_DIALOG_ID\}/);
  }
  assert.equal(
    (onlinePage.match(/aria-haspopup="dialog"/g) ?? []).length,
    3,
    "online entry, lobby, and game headers should all expose the same rulebook",
  );

  assert.match(component, /role="dialog"/);
  assert.match(component, /aria-modal="true"/);
  assert.match(component, /aria-labelledby="rulebook-title"/);
  assert.match(component, /aria-describedby="rulebook-description"/);
  assert.match(component, /aria-label="규칙 닫기"/);
  assert.match(component, /event\.key === "Escape"/);
  assert.match(component, /event\.key !== "Tab"/);
  assert.match(component, /restoreFocusRef\.current\?\.focus\(\)/);
  assert.match(component, /cardArtPath\(preferences\.theme, rank\)/);
  assert.match(styles, /var\(--dalmuti-card-back-image\)/);
  assert.match(component, /게임은 계속 진행 중입니다/);

  assert.match(styles, /\.layer\s*\{[^}]*z-index: 120;/s);
  assert.match(styles, /max-height: calc\(100dvh - 44px\)/);
  assert.match(styles, /overflow-y: auto/);
  assert.match(styles, /overscroll-behavior: contain/);
  assert.match(styles, /@media \(max-width: 760px\)/);
  assert.match(styles, /@media \(prefers-reduced-motion: reduce\)/);
});
