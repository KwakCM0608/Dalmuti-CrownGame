import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const projectRoot = new URL("../", import.meta.url);

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the playable Dalmuti prototype", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="ko">/i);
  assert.match(html, /달무티 — DCLab의 계급전/i);
  assert.match(html, /DALMUTI/);
  assert.doesNotMatch(html, /왕관은/);
  assert.doesNotMatch(html, /네 명의 AI와(?: 함께)? 바로 한 판을 시작합니다/);
  assert.match(html, /빠른 대전\(5인\)/);
  assert.match(html, /<link rel="icon" href="\/brand-dalmuti-crown\.png"\/>/);
  assert.match(html, /친구들과 온라인/);
  assert.match(html, /랩실 서열/);
  assert.match(html, />기록</);
  assert.doesNotMatch(html, /궁정 서열|궁정 기록|5인 궁정|CROWN/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("server-renders the online room entry surface", async () => {
  const response = await render("/online");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /온라인 계급전/);
  assert.match(html, /방 만들기/);
  assert.match(html, /코드로 참가/);
  assert.match(html, /4–8 PLAYERS/);
});

test("ships without the disposable starter preview", async () => {
  const [page, layout, styles, packageJson, cardAssetBuilder] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../scripts/build_card_assets.py", import.meta.url), "utf8"),
  ]);

  assert.match(page, /"use client"/);
  assert.match(page, /DCLab의 계급전/);
  assert.match(styles, /brand-dalmuti-crown\.png/);
  assert.match(page, /function createDeck/);
  assert.match(page, /function applyTax/);
  assert.match(page, /selectPeonTaxCards\(sourceHands\[peon\.id\], count\)/);
  assert.match(
    page,
    /selectDalmutiReturnCards\(sourceHands\[noble\.id\], count\)/,
  );
  assert.match(page, /세금 계산에 한해 광대를 가장 강한/);
  assert.match(page, /function chooseBotCards/);
  assert.match(page, /type TaxStage = "selection" \| "tribute" \| "return"/);
  assert.match(page, /\| "reveal-intro"/);
  assert.match(page, /\| "hand-reveal"/);
  assert.match(page, /REVEAL_INTRO_DURATION_MS = 2400/);
  assert.match(page, /HAND_REVEAL_DURATION_MS = 1400/);
  assert.match(page, /function advanceAfterHandReveal/);
  assert.match(page, /phase: "reveal-intro"/);
  assert.match(page, /className="hand-reveal-intro"/);
  assert.match(page, /concealed=\{isHandConcealed\}/);
  assert.match(page, /toggleWholeRankSelection\(current, sameRankIds\)/);
  assert.match(page, /type PublicTurnAction/);
  assert.match(page, /publicAction: PublicTurnAction \| null/);
  assert.match(page, /function TaxTransferLayer/);
  assert.match(page, /function PublicTurnActionLayer/);
  assert.match(page, /function PrivateCardBack/);
  assert.match(page, /activeTaxRoutes/);
  assert.match(page, /private-tax-state/);
  assert.match(page, /taxAnimationId/);
  assert.match(page, /TAX_STAGE_DURATION_MS = 6000/);
  assert.match(page, /TAX_INTRO_DURATION_MS = 2400/);
  assert.match(page, /PUBLIC_ACTION_DURATION_MS = 2250/);
  assert.match(page, /PASS_ACTION_DURATION_MS = 1500/);
  assert.match(page, /DALMUTI_ACTION_DURATION_MS = 3300/);
  assert.match(page, /playedSet\?\.rank === 1/);
  assert.match(page, /isDalmuti \? "달무티" : "공개 플레이"/);
  assert.doesNotMatch(page, /달무티 효과/);
  assert.match(page, /kind: "play"/);
  assert.match(page, /kind: "pass"/);
  assert.match(page, /previousTable: state\.table/);
  assert.match(page, /visibleTable\?\.cards \?\? \[\]/);
  assert.match(
    page,
    /humanFinished[\s\S]{0,80}\$\{humanFinishRank \+ 1\}위/,
  );
  assert.match(page, /createOpeningRound\(BASE_PLAYERS, scores\)/);
  assert.match(page, /\| "rank-intro"/);
  assert.match(page, /\| "rank-selection"/);
  assert.match(page, /\| "rank-reveal"/);
  assert.match(page, /\| "rank-confirm"/);
  assert.match(
    page,
    /shuffle\(\s*Array\.from\(\{ length: current\.players\.length \}, \(_.*, index\) => index \+ 1\)/,
  );
  assert.match(page, /RANK_COUNTDOWN_STEP_MS = 1100/);
  assert.match(page, /RANK_ALL_SELECTED_PAUSE_MS = 1500/);
  assert.match(page, /RANK_REVEAL_DURATION_MS = 3400/);
  assert.match(page, /RANK_CONFIRM_DURATION_MS = 2600/);
  assert.match(page, /function autoAssignFinalOpeningRankCard/);
  assert.match(
    page,
    /availableIndexes\.length !== 1 \|\| unassignedPlayers\.length !== 1/,
  );
  assert.match(page, /남은 계급 카드를 자동으로 받았습니다/);
  assert.match(page, /return autoAssignFinalOpeningRankCard\(nextState\)/);
  assert.match(page, /className="brand brand-button"/);
  assert.match(page, /초기 모드 선택 화면으로 돌아가기/);
  assert.doesNotMatch(page, /왕관은/);
  assert.match(page, /className="opening-rank-intro"/);
  assert.match(page, /selectedPlayerId \? "is-selected"/);
  assert.match(page, /revolution-announcement is-\$\{/);
  assert.match(page, /const beginHostedGame/);
  assert.match(page, /className="ready-play-button"/);
  assert.match(page, />세금 교환</);
  assert.match(page, />게임 시작</);
  assert.match(page, /const confirmTaxReturn/);
  assert.match(page, /반환 카드 확정/);
  assert.match(page, /onDoubleClick=\{onDoubleClick\}/);
  assert.match(page, /onDoubleClick=\{\(\) => selectAllOfRank\(card\)\}/);
  assert.match(page, /모든 카드를 냈습니다/);
  assert.match(page, /return `\$\{name\}이\(가\)`/);
  assert.match(page, /taxSourcePlaceholder=\{humanSourceIds\.has\(card\.id\)\}/);
  assert.doesNotMatch(
    page,
    /humanIncomingCards|incoming-tax-flight|card-back-transfer|motion=/,
  );
  assert.doesNotMatch(page, /confirmTaxation|tax-exchange|tax-mini-hand/);
  assert.match(layout, /lang="ko"/);
  assert.doesNotMatch(page, /_sites-preview|SkeletonPreview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(
    styles,
    /\.tax-transfer-card\.is-face-up \.playing-card\s*\{[^}]*width: 134px;[^}]*height: 206px;/s,
  );
  assert.match(page, /RANK_NAMES\[visibleTable\.rank\]/);
  assert.match(page, /보다 낮은 숫자의 카드 \$\{game\.table\.count\}장을 내세요/);
  assert.match(
    styles,
    /\.table-cards \.playing-card\s*\{[^}]*width: 140px;[^}]*height: 216px;/s,
  );
  assert.match(page, /className=\{`table-column \$\{isHumanTurn \? "is-human-turn"/);
  assert.match(page, /className=\{`opening-rank-confirmation role-\$\{/);
  assert.match(page, /previousSeatRectsRef/);
  assert.match(
    page,
    /game\.phase === "round-end"[\s\S]*game\.finishOrder\.map/,
  );
  assert.match(page, /160 \/ Math\.max\(1, tablePreview\.length - 1\)/);
  assert.match(page, /index - \(tablePreview\.length - 1\) \/ 2/);
  assert.match(styles, /rotate\(calc\(var\(--table-card-offset\) \* 1\.1deg\)\)/);
  assert.match(page, /data-rank-seat=\{rankSeat\}/);
  assert.match(styles, /\.opponent-row\s*\{[^}]*grid-template-columns: repeat\(5,/s);
  assert.match(page, /is-first-place/);
  assert.match(page, /is-second-place/);
  assert.match(styles, /animation: publicPassToTable 1\.38s/);
  assert.match(styles, /\.public-turn-action-layer\s*\{/);
  assert.match(styles, /@keyframes publicCardPlay/);
  assert.match(styles, /@keyframes publicPassToTable/);
  assert.match(styles, /@keyframes dalmutiTableGlow/);
  assert.match(styles, /animation: taxSeatTransfer 5\.55s/);
  assert.match(styles, /animation: publicCardPlay 2\.08s/);
  assert.match(styles, /animation: revolutionAnnouncement 3\.1s/);
  assert.match(page, /const TURN_LIMIT_MS = 30_000/);
  assert.match(page, /function timeoutPassTurn/);
  assert.match(page, /allowEmptyTable|previousTable: null/);
  assert.match(page, /automatic: true/);
  assert.match(page, /className=\{`turn-countdown/);
  assert.match(page, /"--turn-angle": `\$\{turnProgress \* 360\}deg`/);
  assert.match(page, /function seatPosition/);
  assert.match(page, /style=\{seatPosition\(rankSeat - 1, 5\)\}/);
  assert.match(page, /"--seat-grid-column": rankIndex \+ 1/);
  assert.match(page, /className="human-status" ref=\{humanAnchorRef\}/);
  assert.doesNotMatch(page, /className="hand-wrap" ref=\{humanAnchorRef\}/);
  assert.match(page, /isDalmutiHighlighted/);
  assert.match(page, /className="welcome-crown"/);
  assert.match(
    styles,
    /Shared online visual system for the five-player quick match/,
  );
  assert.match(
    styles,
    /\.game-shell \.welcome-crown\s*\{[^}]*brand-dalmuti-crown\.png/s,
  );
  assert.match(
    styles,
    /@keyframes taxSeatTransfer\s*\{[\s\S]*top: var\(--from-y\)[\s\S]*top: var\(--mid-y\)[\s\S]*top: var\(--to-y\)/,
  );
  assert.match(
    page,
    /"--from-x"[\s\S]*"--mid-x"[\s\S]*"--to-x"/,
  );
  assert.match(layout, /icons:\s*\{[\s\S]*brand-dalmuti-crown\.png/);
  assert.doesNotMatch(
    styles,
    /\.tax-transfer-card\.is-face-up\s*\{[^}]*filter:/s,
  );
  assert.match(cardAssetBuilder, /OUTPUT_SIZE = \(1040, 1600\)/);
  assert.match(
    cardAssetBuilder,
    /ROTATED_CARD_NAMES = \{"06", "07", "08", "11", "joker"\}/,
  );
  assert.match(cardAssetBuilder, /name in \{"01", "joker"\}/);
  assert.match(cardAssetBuilder, /def add_black_outer_border/);
  assert.doesNotMatch(cardAssetBuilder, /normalize_outer_frame/);
  for (const [rank, name] of [
    [1, "달무티"],
    [2, "대주교"],
    [3, "시종장"],
    [4, "남작부인"],
    [5, "수녀원장"],
    [6, "기사"],
    [7, "재봉사"],
    [8, "석공"],
    [9, "요리사"],
    [10, "양치기"],
    [11, "광부"],
    [12, "농노"],
    [13, "어릿광대"],
  ]) {
    assert.match(page, new RegExp(`${rank}: "${name}"`));
  }
  await assert.rejects(access(new URL("../app/_sites-preview", projectRoot)));
});

test("house taxation makes Jesters strongest for both rank pairs", async () => {
  const { selectDalmutiReturnCards, selectPeonTaxCards } = await import(
    new URL("../lib/taxation.ts", import.meta.url)
  );
  const hand = [
    { id: "dalmuti", rank: 1 },
    { id: "archbishop", rank: 2 },
    { id: "jester", rank: 13 },
  ];

  assert.deepEqual(
    selectPeonTaxCards(hand, 2).map((card) => card.id),
    ["jester", "dalmuti"],
  );
  assert.deepEqual(
    selectPeonTaxCards(hand, 1).map((card) => card.id),
    ["jester"],
  );
  assert.deepEqual(
    selectDalmutiReturnCards(hand, 2).map((card) => card.id),
    ["archbishop", "dalmuti"],
  );
  assert.deepEqual(
    selectDalmutiReturnCards(hand, 1).map((card) => card.id),
    ["archbishop"],
  );
});

test("ships one normalized artwork asset for every card rank", async () => {
  const assetNames = [
    ...Array.from({ length: 12 }, (_, index) =>
      String(index + 1).padStart(2, "0"),
    ),
    "joker",
  ];

  await Promise.all(
    assetNames.map((name) =>
      access(new URL(`../public/cards/${name}.webp`, import.meta.url)),
    ),
  );
  await access(new URL("../public/cards/back.webp", import.meta.url));
});

test("online mode exposes synchronized reveal, tax, Dalmuti, and exit states", async () => {
  const [page, styles] = await Promise.all([
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/online/online.module.css", import.meta.url), "utf8"),
  ]);

  assert.match(page, /selectedReturnCount/);
  assert.match(page, /waitingTaxPlayerIds/);
  assert.match(page, /className=\{styles\.taxDecisionField\}/);
  assert.match(page, /이\(가\) 세금 교환 중/);
  assert.match(page, /한 플레이어가 혁명 여부를 결정 중/);
  assert.match(page, /type === "DALMUTI_EFFECT"/);
  assert.match(page, /나머지 플레이어 자동 PASS/);
  assert.match(page, /isHandRevealing \? styles\.handRevealing/);
  assert.match(page, /isHost \? "reset" : "leave"/);
  assert.match(page, /clearSavedSession/);
  assert.match(page, /function RankSelectionField/);
  assert.match(page, /sendCommand\("CHOOSE_RANK_CARD", \{ slotIndex \}\)/);
  assert.match(page, /첫 게임은 선착순으로 카드를 선택해/);
  assert.match(page, /계급 미정/);
  assert.match(page, /declaredKind === "great-revolution"/);
  assert.match(page, /대혁명을 선포하시겠습니까/);
  assert.match(
    page,
    /if \(label\.includes\("PASS"\)\) return 1500;[\s\S]*if \(label\.includes\("PLAY"\)\) return 3600;/,
  );
  assert.match(page, /showMyTurnHighlight/);
  assert.match(page, /dalmutiActorIdFromEvent/);
  assert.match(page, /isDalmutiHighlighted/);
  assert.match(page, /player\.id === dalmutiHighlightPlayerId/);
  assert.match(page, /turnDeadline/);
  assert.match(page, /TURN_DURATION_MS = 30_000/);
  assert.match(page, /className=\{`\$\{styles\.turnCountdown\}/);
  assert.doesNotMatch(page, /event\.durationMs \+ 220/);
  assert.match(styles, /\.turnCountdownRing/);
  assert.match(page, /!actionLocked/);
  assert.match(page, /rankedOpponents/);
  assert.match(page, /seatRankOverrides/);
  assert.match(page, /rankMovingPlayerIds/);
  assert.match(page, /pendingRoundEndMoveIds/);
  assert.match(page, /roundEndResultReady/);
  assert.match(page, /next\.finishOrder\.map/);
  assert.match(page, /element\.animate/);
  assert.match(page, /Boolean\(activeEvent\)/);
  assert.match(page, /당신의 첫 서열은 \{viewerRank\}위입니다/);
  assert.match(page, /styles\.resultFirst/);
  assert.match(page, /styles\.resultSecond/);
  assert.match(page, /--table-card-step-wide/);
  assert.doesNotMatch(page, /sendCommand\("PASS", \{ automatic: true \}\)/);
  assert.match(styles, /@keyframes onlineHandCardReveal/);
  assert.match(styles, /\.dalmutiEffectOverlay/);
  assert.match(styles, /\.playerSeatDalmuti/);
  assert.match(styles, /\.dalmutiEffectActorSeat/);
  assert.match(
    styles,
    /\.dalmutiEffectOverlay\s*\{[^}]*overflow-y: auto;[^}]*pointer-events: auto;/s,
  );
  assert.match(
    styles,
    /@media \(prefers-reduced-motion: reduce\)[\s\S]*\.eventOverlay\s*\{[^}]*animation: none !important;/,
  );
  assert.match(styles, /\.rankChoiceSlotClaimed/);
  assert.match(styles, /\.rankConfirmation/);
  assert.match(styles, /\.tableMyTurn/);
  assert.match(styles, /\.playerSeatRankMoving/);
  assert.match(styles, /grid-column: var\(--seat-grid-column\)/);
  assert.match(styles, /\.resultCard \.resultFirst/);
  assert.match(styles, /\.resultCard \.resultSecond/);
  assert.match(styles, /\.tableRevolution/);
});

test("deals remainder cards to the lowest-ranked players", async () => {
  const { rankedDealCounts } = await import(
    new URL("../lib/dealing.ts", import.meta.url)
  );

  assert.deepEqual(rankedDealCounts(80, 5), [16, 16, 16, 16, 16]);
  assert.deepEqual(rankedDealCounts(80, 6), [13, 13, 13, 13, 14, 14]);
  assert.deepEqual(rankedDealCounts(80, 7), [11, 11, 11, 11, 12, 12, 12]);
});

test("double-clicking a selected rank clears that whole rank", async () => {
  const { toggleWholeRankSelection } = await import(
    new URL("../lib/selection.ts", import.meta.url)
  );
  const rankIds = ["8-1", "8-2", "8-3"];

  assert.deepEqual(toggleWholeRankSelection([], rankIds), rankIds);
  assert.deepEqual(toggleWholeRankSelection(rankIds, rankIds), []);
  assert.deepEqual(
    toggleWholeRankSelection(["joker-1", ...rankIds], rankIds),
    ["joker-1"],
  );
});

test("quick and online tables expose the enhanced timed and rank feedback", async () => {
  const [quickPage, quickStyles, onlinePage, onlineStyles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
  ]);

  assert.match(quickPage, /RANK_TRANSITION_DURATION_MS = 2300/);
  assert.match(quickPage, /turnSecondsRemaining <= 10/);
  assert.match(quickPage, /className="dalmuti-action-effects"/);
  assert.match(quickPage, /finishRank=\{finishIndex >= 0 \? finishIndex \+ 1/);
  assert.match(quickPage, /className="opening-rank-confirmation-body"/);
  assert.match(quickPage, /className="great-revolution-field-effect"/);
  assert.match(quickPage, /className=\{`result-rank-shift is-\$\{rankMovement\}`\}/);
  assert.match(quickStyles, /\.table-column > \.turn-countdown/);
  assert.match(quickStyles, /\.result-rank-shift\.is-up/);
  assert.match(quickStyles, /\.result-rank-shift\.is-down/);
  assert.match(quickStyles, /\.felt-table\.is-great-revolution/);
  assert.match(
    quickStyles,
    /@media \(prefers-reduced-motion: reduce\)[\s\S]*\.opening-rank-confirmation-card\s*\{[^}]*transform: rotate\(-7deg\) !important;/,
  );

  assert.match(onlinePage, /turnRemainingMs <= 10_000/);
  assert.match(onlinePage, /styles\.boardColumnTurnUrgent/);
  assert.match(onlinePage, /styles\.tableDalmutiBurst/);
  assert.match(onlinePage, /styles\.rankShiftEffect/);
  assert.match(onlinePage, /styles\.resultRoleChange/);
  assert.match(onlinePage, /greatRevolutionActive/);
  assert.match(onlineStyles, /\.turnCountdown\s*\{[^}]*z-index:\s*28/s);
  assert.match(onlineStyles, /\.tableGreatRevolution/);
  assert.match(onlineStyles, /\.resultRoleUp/);
  assert.match(onlineStyles, /\.resultRoleDown/);
  assert.match(
    onlineStyles,
    /@media \(max-width: 820px\)[\s\S]*\.turnCountdown\s*\{[^}]*position: relative;[^}]*top: auto;[^}]*animation-name: turnCountdownArrivalMobile;/,
  );
  assert.doesNotMatch(
    onlineStyles,
    /@media \(max-width: 820px\)[\s\S]{0,300}\.turnCountdown\s*\{[^}]*top: (?:80|136|158)px;/,
  );
});
