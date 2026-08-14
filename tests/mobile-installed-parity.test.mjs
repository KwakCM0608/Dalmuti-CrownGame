import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const {
  GAME_PRESENTATION_TIMING_MS,
  INSTALLED_MOBILE_PRESENTATION,
  cappedPresentationStep,
} = await import(
  new URL("../lib/game-presentation-parity.ts", import.meta.url)
);

const [quickPage, quickStyles, onlinePage, onlineStyles, onlineEngine] =
  await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../lib/online-game/engine.ts", import.meta.url), "utf8"),
  ]);

function numericConst(source, name) {
  const match = source.match(
    new RegExp(`const ${name} = ([0-9_]+);`),
  );
  assert.ok(match, `missing numeric constant ${name}`);
  return Number(match[1].replaceAll("_", ""));
}

test("shared presentation constants describe the installed phone contract", () => {
  assert.deepEqual(
    {
      tableCardWidth: INSTALLED_MOBILE_PRESENTATION.tableCardWidth,
      tableCardHeight: INSTALLED_MOBILE_PRESENTATION.tableCardHeight,
      tableCardMaxStep: INSTALLED_MOBILE_PRESENTATION.tableCardMaxStep,
      tableCardSpread: INSTALLED_MOBILE_PRESENTATION.tableCardSpread,
      actionCardWidth: INSTALLED_MOBILE_PRESENTATION.actionCardWidth,
      actionCardHeight: INSTALLED_MOBILE_PRESENTATION.actionCardHeight,
      actionExpandedMaxStep:
        INSTALLED_MOBILE_PRESENTATION.actionExpandedMaxStep,
      actionSettledMaxStep:
        INSTALLED_MOBILE_PRESENTATION.actionSettledMaxStep,
      dalmutiAutoPassOffset:
        INSTALLED_MOBILE_PRESENTATION.dalmutiAutoPassOffset,
    },
    {
      tableCardWidth: 88,
      tableCardHeight: 135,
      tableCardMaxStep: 32,
      tableCardSpread: 160,
      actionCardWidth: 92,
      actionCardHeight: 142,
      actionExpandedMaxStep: 54,
      actionSettledMaxStep: 24,
      dalmutiAutoPassOffset: 34,
    },
  );

  assert.equal(cappedPresentationStep(1, 32, 160), 0);
  assert.equal(cappedPresentationStep(5, 32, 160), 32);
  assert.equal(cappedPresentationStep(10, 32, 160), 160 / 9);
});

test("quick, online client, and online engine pin the same visible timings", () => {
  const quickTimingConstants = {
    rankCountdownStep: "RANK_COUNTDOWN_STEP_MS",
    rankSelectionPause: "RANK_ALL_SELECTED_PAUSE_MS",
    rankReveal: "RANK_REVEAL_DURATION_MS",
    rankConfirm: "RANK_CONFIRM_DURATION_MS",
    revealIntro: "REVEAL_INTRO_DURATION_MS",
    handReveal: "HAND_REVEAL_DURATION_MS",
    taxIntro: "TAX_INTRO_DURATION_MS",
    taxStage: "TAX_STAGE_DURATION_MS",
    playIntro: "PLAY_INTRO_DURATION_MS",
    revolutionIntro: "REVOLUTION_INTRO_DURATION_MS",
    greatRevolutionSwap: "GREAT_REVOLUTION_SWAP_DURATION_MS",
    publicPlay: "PUBLIC_ACTION_DURATION_MS",
    publicPass: "PASS_ACTION_DURATION_MS",
    dalmuti: "DALMUTI_ACTION_DURATION_MS",
    turn: "TURN_LIMIT_MS",
    rankMove: "RANK_TRANSITION_DURATION_MS",
    rankResultDelay: "RANK_RESULT_REVEAL_DELAY_MS",
  };

  for (const [sharedName, localName] of Object.entries(
    quickTimingConstants,
  )) {
    assert.equal(
      numericConst(quickPage, localName),
      GAME_PRESENTATION_TIMING_MS[sharedName],
      `quick ${localName} drifted`,
    );
  }

  assert.equal(
    numericConst(onlinePage, "HAND_REVEAL_PRESENTATION_MS"),
    GAME_PRESENTATION_TIMING_MS.handReveal,
  );
  assert.equal(
    numericConst(onlinePage, "TURN_DURATION_MS"),
    GAME_PRESENTATION_TIMING_MS.turn,
  );
  assert.equal(
    numericConst(onlinePage, "RANK_MOVE_DURATION_MS"),
    GAME_PRESENTATION_TIMING_MS.rankMove,
  );
  assert.match(
    onlinePage,
    /countdownElapsed \/\s*GAME_PRESENTATION_TIMING_MS\.rankCountdownStep/,
  );
  assert.equal(
    numericConst(onlineEngine, "PASS_ACTION_LOCK_MS"),
    GAME_PRESENTATION_TIMING_MS.publicPass,
  );
  assert.equal(
    numericConst(onlineEngine, "PLAY_ACTION_LOCK_MS"),
    GAME_PRESENTATION_TIMING_MS.publicPlay,
  );
  assert.equal(
    numericConst(onlineEngine, "DALMUTI_ACTION_LOCK_MS"),
    GAME_PRESENTATION_TIMING_MS.dalmuti,
  );
  assert.equal(
    numericConst(onlineEngine, "TURN_DURATION_MS"),
    GAME_PRESENTATION_TIMING_MS.turn,
  );

  assert.match(
    onlineEngine,
    /rankChoiceIntroMs: 3_300[\s\S]*rankRevealDelayMs: 1_500[\s\S]*rankRevealMs: 3_400[\s\S]*rankConfirmMs: 2_600/,
  );
  assert.match(
    onlineEngine,
    /revealIntroMs: 2_400[\s\S]*handRevealMs: 1_400[\s\S]*revolutionIntroMs: 3_300[\s\S]*greatRevolutionSwapMs: 2_600/,
  );
  assert.match(
    onlineEngine,
    /taxIntroMs: 2_400[\s\S]*taxTributeMs: 6_000[\s\S]*taxReturnMs: 6_000[\s\S]*playIntroMs: 2_600/,
  );
});

test("both clients derive installed card fans and DALMUTI passes from shared geometry", () => {
  for (const source of [quickPage, onlinePage]) {
    assert.match(source, /INSTALLED_MOBILE_PRESENTATION/);
    assert.match(source, /cappedPresentationStep\(/);
    assert.match(
      source,
      /INSTALLED_MOBILE_PRESENTATION\.actionExpandedMaxStep/,
    );
    assert.match(
      source,
      /INSTALLED_MOBILE_PRESENTATION\.actionSettledMaxStep/,
    );
    assert.match(
      source,
      /INSTALLED_MOBILE_PRESENTATION\.dalmutiAutoPassOffset/,
    );
    assert.match(
      source,
      /INSTALLED_MOBILE_PRESENTATION\.dalmutiAutoPassInitialDelay/,
    );
    assert.match(
      source,
      /INSTALLED_MOBILE_PRESENTATION\.dalmutiAutoPassStagger/,
    );
  }

  assert.match(
    quickPage,
    /anchors\.players\[action\.player\.id\][\s\S]*anchors\.midpoint/,
  );
  assert.match(
    onlinePage,
    /stableAnchors\.players\[fromId \|\| actorId \|\| ""\] \?\? center/,
  );
  assert.match(
    onlinePage,
    /motionAnchors\.players\[queuedRemoteAction\.actorPlayerId\]/,
  );
});

test("installed online CSS matches quick table, action, PASS, and dock geometry", () => {
  assert.match(
    quickStyles,
    /\.game-shell\.game-shell \.table-cards \.playing-card\s*\{\s*width: 88px;\s*height: 135px;/,
  );
  assert.match(
    onlineStyles,
    /\.gameShell \.tableCards \{[\s\S]{0,240}--table-card-width: 88px;[\s\S]{0,240}min-height: 147px;/,
  );
  assert.match(
    onlineStyles,
    /\.gameShell \.tableCards \.card\s*\{\s*width: 88px;\s*height: 135px;/,
  );
  assert.match(
    onlineStyles,
    /\.gameShell \.playOverlay \.eventCards \.card\s*\{\s*width: 92px;\s*height: 142px;/,
  );

  for (const [quickMotion, onlineMotion] of [
    ["publicCardPlay", "onlinePublicCardPlay"],
    ["publicPassToTable", "onlinePublicPassToTable"],
  ]) {
    assert.match(quickStyles, new RegExp(`@keyframes ${quickMotion}`));
    assert.match(onlineStyles, new RegExp(`@keyframes ${onlineMotion}`));
  }

  assert.match(quickStyles, /animation: publicCardPlay 2\.08s/);
  assert.match(onlineStyles, /animation: onlinePublicCardPlay 2\.08s/);
  assert.match(quickStyles, /animation: publicPassToTable 1\.38s/);
  assert.match(onlineStyles, /animation: onlinePublicPassToTable 1\.38s/);
  assert.match(quickStyles, /animation: taxSeatTransfer 5\.55s/);
  assert.match(onlineStyles, /animation: taxCardTransfer 5\.55s/);

  assert.match(
    quickStyles,
    /\.game-shell \.playing-card,[\s\S]{0,100}width: 53px;\s*height: 81px;/,
  );
  assert.match(
    onlineStyles,
    /@media \(max-width: 520px\)[\s\S]*?\.card\s*\{\s*width: 53px;\s*height: 81px;/,
  );
  assert.match(quickStyles, /\.game-shell \.turn-controls\s*\{[\s\S]*?min-height: 44px;/);
  assert.match(onlineStyles, /\.actionBar\s*\{\s*min-height: 44px;/);
});

test("installed online chat opens as a narrower and taller touch viewport", () => {
  assert.match(
    onlineStyles,
    /@media \(display-mode: standalone\)[\s\S]*?\.gameChatPanel:not\(\.chatPanelCollapsed\)\s*\{[\s\S]*?width: min\(78vw, 306px\);[\s\S]*?height: clamp\(260px, 46dvh, 380px\);/,
  );
  assert.match(
    onlineStyles,
    /\.chatPanel \.chatMessages\s*\{[\s\S]{0,220}pointer-events: auto;[\s\S]{0,120}touch-action: pan-y;/,
  );
});

test("mobile controls reserve Android navigation space and expose private auto PASS", () => {
  for (const styles of [quickStyles, onlineStyles]) {
    assert.match(
      styles,
      /--mobile-system-bottom-inset: max\(env\(safe-area-inset-bottom, 0px\), 40px\)/,
    );
    assert.match(
      styles,
      /padding-bottom: calc\(4px \+ var\(--mobile-system-bottom-inset\)\)/,
    );
    assert.match(styles, /grid-template-rows: minmax\(0, 1fr\) 124px/);
  }
  assert.match(
    quickStyles,
    /\.game-shell \.selection-hint > span\s*\{[\s\S]{0,180}white-space: nowrap;[\s\S]{0,80}word-break: keep-all;/,
  );
  assert.match(
    quickStyles,
    /\.game-shell \.game-stage,\s*\.game-shell \.table-column\s*\{[\s\S]{0,120}height: calc\(100dvh - var\(--app-header-block\)\);[\s\S]{0,80}min-height: 0;/,
  );
  assert.match(
    onlineStyles,
    /\.selectionCopy > strong,[\s\S]{0,120}\.selectionCopy > small\s*\{[\s\S]{0,100}white-space: nowrap;[\s\S]{0,80}word-break: keep-all;/,
  );
  assert.match(
    onlineStyles,
    /\.gameChatPanel\s*\{\s*bottom: calc\(var\(--mobile-system-bottom-inset\) \+ 106px\);/,
  );
  assert.match(
    quickStyles,
    /\.turn-controls:has\(\.auto-pass-toggle\)\s*\{\s*grid-template-columns: minmax\(190px, 1fr\) 70px 64px 90px;/,
  );
  assert.match(
    onlineStyles,
    /\.actionBar:has\(\.autoPassToggle\)\s*\{\s*grid-template-columns: minmax\(190px, 1fr\) 70px 64px 90px;/,
  );
  for (const styles of [quickStyles, onlineStyles]) {
    assert.match(
      styles,
      /\(orientation: portrait\)[\s\S]*?height: 98px;[\s\S]*?min-height: 98px;[\s\S]*?max-height: 98px;/,
    );
    assert.match(
      styles,
      /grid-template-columns: 78px minmax\(70px, 1fr\) minmax\(96px, 1\.25fr\)/,
    );
    assert.match(
      styles,
      /grid-template-rows: 18px 14px;/,
    );
    assert.match(
      styles,
      /position: fixed;[\s\S]{0,180}bottom: var\(--mobile-system-bottom-inset\);[\s\S]{0,180}z-index: 76;/,
    );
    assert.match(
      styles,
      /padding-bottom: 4px;/,
    );
  }
  assert.match(quickPage, /className="auto-pass-toggle"/);
  assert.match(onlinePage, /className=\{styles\.autoPassToggle\}/);
  assert.match(quickPage, /hasLegalCardPlay\(hand, game\.table\)/);
  assert.match(onlinePage, /type: "SET_AUTO_PASS"/);
  assert.match(onlineEngine, /current\.autoPassEnabled === false/);
  assert.match(
    onlineEngine,
    /publicly indistinguishable from a manual PASS/,
  );
});

test("installed quick and online seats use matching exterior rails and count alignment", () => {
  for (const styles of [quickStyles, onlineStyles]) {
    assert.match(styles, /margin-block: 43px 45px;/);
    assert.match(styles, /inset: -36px 5px -38px;/);
    assert.match(
      styles,
      /grid-template-rows: 30px minmax\(0, 1fr\) 30px;/,
    );
    assert.match(
      styles,
      /grid-template-columns: minmax\(0, 1fr\) max-content;/,
    );
    assert.match(styles, /justify-self: end;/);
    assert.match(styles, /text-align: right;/);
    assert.match(styles, /var\(--mobile-seat-panel\)/);
  }

  assert.match(
    quickStyles,
    /Terminal installed-app seat rail[\s\S]*?\.player-seat\.is-human-seat[\s\S]*?inset 3px 0 #d5aa4e/,
  );
  assert.match(
    onlineStyles,
    /Terminal installed-app seat rail[\s\S]*?\.playerSeatSelf[\s\S]*?inset 3px 0 #d5aa4e/,
  );
});

test("installed quick and online openings share fixed card geometry for every player count", () => {
  const sharedVariables = {
    "--installed-pregame-rank-card-width": "clamp\\(46px, 13vw, 54px\\)",
    "--installed-pregame-rank-gap": "5px",
    "--installed-pregame-confirm-card-width": "42px",
    "--installed-pregame-reveal-card-width": "40px",
    "--installed-pregame-reveal-card-height": "62px",
    "--installed-pregame-hand-card-width": "56px",
    "--installed-pregame-hand-card-height": "86px",
    "--installed-pregame-tax-public-width": "72px",
    "--installed-pregame-tax-public-height": "111px",
    "--installed-pregame-tax-private-width": "92px",
    "--installed-pregame-tax-private-height": "142px",
    "--installed-pregame-joker-width": "46px",
    "--installed-pregame-joker-height": "71px",
  };

  for (const [name, value] of Object.entries(sharedVariables)) {
    assert.match(
      quickStyles,
      new RegExp(`${name}: ${value};`),
      `missing shared installed opening variable ${name}`,
    );
    for (const source of [quickStyles, onlineStyles]) {
      assert.match(
        source,
        new RegExp(`var\\(${name}\\)`),
        `${name} is not consumed by both installed clients`,
      );
    }
  }

  assert.match(
    quickStyles,
    /Installed-app opening presentation[\s\S]*?\.opening-rank-cards\[data-card-count\][\s\S]*?var\(--installed-pregame-rank-card-width\)/,
  );
  assert.match(
    onlineStyles,
    /Installed-app opening presentation[\s\S]*?\.rankChoiceCards\[data-card-count\][\s\S]*?var\(--installed-pregame-rank-card-width\)/,
  );

  for (const count of [4, 5, 6, 7, 8, 9, 10]) {
    assert.match(
      quickStyles,
      new RegExp(
        `opening-rank-cards\\[data-card-count="${count}"\\][\\s\\S]{0,120}--installed-rank-columns: ${count};`,
      ),
      `quick short-landscape rank draw is not one fixed row at ${count} players`,
    );
    assert.match(
      onlineStyles,
      new RegExp(
        `rankChoiceCards\\[data-card-count="${count}"\\][\\s\\S]{0,120}--installed-rank-columns: ${count};`,
      ),
      `online short-landscape rank draw is not one fixed row at ${count} players`,
    );
  }

  for (const source of [quickStyles, onlineStyles]) {
    assert.match(
      source,
      /display-mode: standalone[\s\S]*?display-mode: fullscreen[\s\S]*?hover: none[\s\S]*?pointer: coarse/,
    );
  }
});

test("desktop online hand recentres on its full dock after every hand mutation", () => {
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
    onlineStyles,
    /Desktop hand parity with quick match[\s\S]*?\.gameShell \.ownDock \{[\s\S]*?grid-template-columns: minmax\(0, 1fr\);/,
  );
  assert.match(
    onlineStyles,
    /\.gameShell \.ownStatus \{[\s\S]*?position: absolute;[\s\S]*?left: 13px;/,
  );
  assert.match(
    onlineStyles,
    /\.gameShell \.ownDock > \.handScroller \.hand \{[\s\S]*?min-width: 100%;[\s\S]*?justify-content: center;/,
  );
});

test("installed online rank draw stays mounted and centred while choices lock", () => {
  assert.match(
    onlinePage,
    /stage === "selecting" \|\|\s*snapshot\.rankSelection\.stage === "locked"[\s\S]{0,180}\? `choice:\$\{[\s\S]{0,100}countdownEndsAt/,
  );

  const marker = "End-of-file installed portrait rank draw";
  const markerIndex = onlineStyles.lastIndexOf(marker);
  assert.notEqual(markerIndex, -1);
  const terminalRankCss = onlineStyles.slice(markerIndex);

  assert.match(
    terminalRankCss,
    /\.tableRankSelection \.tableCenter\s*\{[^}]*right: 0;[^}]*left: 0;[^}]*width: 100%;[^}]*transform: translateY\(-50%\);/,
  );
  assert.match(
    terminalRankCss,
    /:is\(\.rankChoiceField, \.rankChoiceFieldRevealed\)\s*\{[^}]*width: 100%;[^}]*justify-self: center;[^}]*margin-inline: auto;/,
  );
  assert.match(terminalRankCss, /display: flex;/);
  assert.match(terminalRankCss, /flex-wrap: wrap;/);
  assert.match(terminalRankCss, /justify-content: center;/);
  assert.match(terminalRankCss, /margin-inline: auto;/);
  assert.match(
    terminalRankCss,
    /\.rankChoiceCards\[data-card-count\][\s\S]*?> \.rankChoiceSlot\s*\{[\s\S]*?flex: 0 0 var\(--installed-pregame-rank-card-width\);[\s\S]*?transform: none;/,
  );
  assert.doesNotMatch(
    terminalRankCss,
    /nth-child[\s\S]{0,100}translateX/,
  );
  assert.match(
    onlineStyles,
    /@media \(hover: hover\) and \(pointer: fine\)[\s\S]*?translateY\(-9px\)/,
  );
});
