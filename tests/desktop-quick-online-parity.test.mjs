import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const {
  GAME_PRESENTATION_TIMING_MS,
} = await import(
  new URL("../lib/game-presentation-parity.ts", import.meta.url)
);

const [quickStyles, onlinePage, onlineStyles] =
  await Promise.all([
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
  ]);

function numericConst(source, name) {
  const match = source.match(
    new RegExp(`const ${name} = ([0-9_]+);`),
  );
  assert.ok(match, `missing numeric constant ${name}`);
  return Number(match[1].replaceAll("_", ""));
}

function desktopParityLayer() {
  const marker = "Desktop quick/online presentation parity";
  const markerIndex = onlineStyles.lastIndexOf(marker);
  assert.notEqual(
    markerIndex,
    -1,
    `missing terminal CSS marker: ${marker}`,
  );
  const mediaIndex = onlineStyles.indexOf(
    "@media (min-width: 821px)",
    markerIndex,
  );
  assert.notEqual(mediaIndex, -1, "desktop parity marker has no media block");
  const openingBrace = onlineStyles.indexOf("{", mediaIndex);
  assert.notEqual(openingBrace, -1, "desktop parity media block is malformed");

  let depth = 0;
  for (let index = openingBrace; index < onlineStyles.length; index += 1) {
    if (onlineStyles[index] === "{") depth += 1;
    if (onlineStyles[index] === "}") {
      depth -= 1;
      if (depth === 0) return onlineStyles.slice(mediaIndex, index + 1);
    }
  }
  assert.fail("desktop parity media block has no closing brace");
}

test("desktop online hand owns the full dock and recentres after every mutation", () => {
  assert.match(onlinePage, /const handScrollerRef = useRef<HTMLDivElement/);
  assert.match(
    onlinePage,
    /const desktopHandLayoutKey[\s\S]*?useLayoutEffect\(\(\) => \{[\s\S]*?\(scroller\.scrollWidth - scroller\.clientWidth\) \/ 2[\s\S]*?\[desktopHandLayoutKey\]/,
  );
  assert.match(
    onlinePage,
    /className=\{styles\.handScroller\} ref=\{handScrollerRef\}/,
  );
  assert.match(
    onlinePage,
    /new ResizeObserver\([\s\S]*?observer\.observe\(scroller\)[\s\S]*?observer\.observe\(scroller\.firstElementChild\)/,
  );

  const parityCss = desktopParityLayer();
  assert.match(parityCss, /@media \(min-width: 821px\)/);
  assert.match(
    parityCss,
    /\.gameShell \.ownDock\s*\{[\s\S]*?grid-template-columns: minmax\(0, 1fr\);/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.ownStatus\s*\{[\s\S]*?position: absolute;[\s\S]*?left: 13px;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.handScroller\s*\{[\s\S]*?width: 100%;[\s\S]*?justify-content: center;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.hand\s*\{[\s\S]*?min-width: 100%;[\s\S]*?justify-content: center;/,
  );
});

test("desktop online round-end and great-revolution movement start without a visual dead gap", () => {
  assert.equal(
    numericConst(onlinePage, "GREAT_REVOLUTION_MOVE_PRELUDE_MS"),
    0,
  );
  assert.equal(
    numericConst(onlinePage, "ROUND_END_MOVE_PRELUDE_MS"),
    0,
  );
  assert.equal(
    numericConst(onlinePage, "ROUND_END_MOVE_SETTLE_MS"),
    GAME_PRESENTATION_TIMING_MS.rankResultDelay,
  );
  assert.equal(
    numericConst(onlinePage, "RANK_MOVE_DURATION_MS"),
    GAME_PRESENTATION_TIMING_MS.rankMove,
  );

  assert.match(
    onlinePage,
    /duration: RANK_MOVE_DURATION_MS,/,
  );
  assert.match(
    onlinePage,
    /rankMoveTimerRef\.current = window\.setTimeout\(\(\) => \{[\s\S]*?setRoundEndResultReady\(true\)[\s\S]*?ROUND_END_MOVE_SETTLE_MS/,
  );
});

test("terminal desktop parity layer removes duplicate fades and pins quick-match geometry", () => {
  const parityCss = desktopParityLayer();

  assert.match(
    parityCss,
    /\.gameShell \.eventOverlay\s*\{[\s\S]*?animation: none;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.tableCenter\s*\{[\s\S]*?width: min\(560px, 72%\);/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.hand \.card \+ \.card\s*\{[\s\S]*?margin-left: -34px;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.hand\s*\{[\s\S]*?padding: 18px 38px 0;/,
  );

  assert.match(
    quickStyles,
    /\.game-shell \.welcome-layer,[\s\S]*?backdrop-filter: blur\(12px\);/,
  );
  assert.match(parityCss, /backdrop-filter: blur\(12px\);/);
  assert.match(
    parityCss,
    /\.gameShell \.resultCard[\s\S]*?rgba\(222, 188, 110, 0\.28\)/,
  );
});

test("desktop online self seat, timer, and compact dock status match quick play", () => {
  const parityCss = desktopParityLayer();

  assert.match(
    parityCss,
    /\.gameShell \.playerSeat\.playerSeatSelf\s*\{[\s\S]*?rgba\(244, 203, 105, 0\.9\)[\s\S]*?#d1a447/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.turnCountdownRing b\s*\{[^}]*transform: none;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.turnCountdownRing small\s*\{[^}]*margin: 3px 0 0;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.ownStatus\s*\{[^}]*width: 154px;[^}]*max-width: 154px;/,
  );
});

test("quick desktop score rail is pinned to the online rail contract", () => {
  assert.match(
    quickStyles,
    /Desktop rail contract shared with online play[\s\S]*?@media \(min-width: 1121px\)[\s\S]*?grid-template-columns: 178px minmax\(620px, 1fr\) 184px;[\s\S]*?width: 178px;/,
  );
  assert.match(
    quickStyles,
    /@media \(min-width: 821px\) and \(max-width: 1120px\)[\s\S]*?grid-template-columns: 158px minmax\(570px, 1fr\);[\s\S]*?width: 158px;/,
  );
  assert.match(
    quickStyles,
    /Quick-match hierarchy rail keeps vertical scrolling[\s\S]*?\.game-shell\.game-shell \.score-rail\s*\{[^}]*overflow-x: hidden;[^}]*\}[\s\S]*?\.game-shell\.game-shell \.score-rail ol\s*\{[^}]*margin-inline: -5px;[^}]*padding-inline: 5px;[^}]*overflow-x: hidden;/,
  );
  assert.doesNotMatch(
    quickStyles,
    /\.game-shell(?:\.game-shell)? \.score-rail(?: ol)?\s*\{[^}]*overflow-x:\s*auto;/,
  );
});

test("desktop opening and phase overlays use one quick-match rhythm", () => {
  const parityCss = desktopParityLayer();

  assert.match(
    parityCss,
    /\.gameShell \.taxIntroOverlay \.eventCenterCopy[\s\S]*?animation-duration: 2250ms;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.playIntroOverlay \.eventCenterCopy[\s\S]*?animation-duration: 2450ms;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.rankConfirmation[\s\S]*?animation-duration: 2500ms;/,
  );
  assert.match(
    parityCss,
    /\.gameShell \.phaseIntroOverlay \.eventCenterCopy > strong[\s\S]*?clamp\(40px, 6vw, 68px\)/,
  );

  assert.match(
    onlinePage,
    /className=\{styles\.revolutionJokers\}[\s\S]*?<i \/>[\s\S]*?<i \/>/,
  );
  assert.match(parityCss, /\.gameShell \.revolutionJokers > i/);
});

test("desktop-only parity rules do not replace installed-app opening geometry", () => {
  assert.match(
    quickStyles,
    /--installed-pregame-rank-card-width: clamp\(46px, 13vw, 54px\);/,
  );
  for (const source of [quickStyles, onlineStyles]) {
    assert.match(
      source,
      /display-mode: standalone[\s\S]*?var\(--installed-pregame-rank-card-width\)/,
    );
  }

  assert.match(
    quickStyles,
    /\.game-shell\.game-shell[\s\S]{0,120}\.opening-rank-cards\[data-card-count\][\s\S]{0,80}\.opening-rank-card\s*\{[\s\S]{0,160}width: var\(--installed-pregame-rank-card-width\);[\s\S]{0,160}max-width: var\(--installed-pregame-rank-card-width\);/,
  );
  assert.match(
    onlineStyles,
    /\.gameShell[\s\S]{0,120}\.rankChoiceCards\[data-card-count\][\s\S]{0,80}> \.rankChoiceSlot\s*\{[\s\S]{0,160}width: var\(--installed-pregame-rank-card-width\);[\s\S]{0,160}max-width: var\(--installed-pregame-rank-card-width\);/,
  );
  assert.match(
    onlineStyles,
    /\.rankChoiceCards\[data-card-count\][\s\S]{0,80}\.rankChoiceCard\s*\{[\s\S]{0,120}width: var\(--installed-pregame-rank-card-width\);[\s\S]{0,120}max-width: var\(--installed-pregame-rank-card-width\);/,
  );

  const parityCss = desktopParityLayer();
  assert.match(parityCss, /@media \(min-width: 821px\)/);
  assert.doesNotMatch(parityCss, /display-mode: standalone/);
  assert.doesNotMatch(parityCss, /--installed-pregame-rank-card-width:/);
});
