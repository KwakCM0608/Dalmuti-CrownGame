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
