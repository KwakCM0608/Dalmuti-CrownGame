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
  assert.match(html, /<title>DALMUTI<\/title>/i);
  assert.match(html, /DALMUTI/);
  assert.doesNotMatch(html, /왕관은/);
  assert.doesNotMatch(html, /네 명의 AI와(?: 함께)? 바로 한 판을 시작합니다/);
  assert.match(html, />빠른 대전</);
  assert.match(html, />온라인 모드</);
  assert.match(html, />게임 규칙</);
  assert.match(html, />크레딧</);
  assert.doesNotMatch(html, />환경설정</);
  assert.doesNotMatch(html, /빠른 대전 플레이 인원/);
  assert.match(html, /<link rel="icon" href="\/brand-dalmuti-crown\.png"\/>/);
  assert.match(html, /누적 점수/);
  assert.match(html, />기록</);
  assert.doesNotMatch(html, /궁정 서열|궁정 기록|5인 궁정|CROWN/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("server-renders the online room entry surface", async () => {
  const response = await render("/online");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, />DALMUTI</);
  assert.doesNotMatch(html, /DCLab의 계급전/);
  assert.match(html, /방 만들기/);
  assert.match(html, /코드로 참가/);
  assert.match(html, /새로운 달무티/);
  assert.match(html, /게임에서 사용할 닉네임/);
  assert.match(html, /4–8 PLAYERS/);
  assert.match(html, />규칙</);
});

test("ships without the disposable starter preview", async () => {
  const [
    page,
    layout,
    styles,
    packageJson,
    cardAssetBuilder,
    rulebookContent,
  ] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../scripts/build_card_assets.py", import.meta.url), "utf8"),
    readFile(new URL("../lib/rulebook-content.ts", import.meta.url), "utf8"),
  ]);

  assert.match(page, /"use client"/);
  assert.doesNotMatch(page, /DCLab의 계급전/);
  assert.match(page, /type LandingView = "main" \| "quick-setup"/);
  assert.match(page, />온라인 모드</);
  assert.match(page, />게임 규칙</);
  assert.match(page, />크레딧</);
  assert.match(page, /CreditsDialog/);
  assert.doesNotMatch(page, /SettingsDialog|환경설정/);
  assert.match(styles, /brand-dalmuti-crown\.png/);
  assert.match(page, /function createDeck/);
  assert.match(page, /function applyTax/);
  assert.match(page, /selectPeonTaxCards\(sourceHands\[peon\.id\], count\)/);
  assert.match(
    page,
    /chooseBotTaxReturn\([\s\S]{0,240}sourceHands\[noble\.id\]/,
  );
  assert.match(rulebookContent, /조커를 제외한 일반 카드/);
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
  assert.match(page, /isDalmuti \? "DALMUTI" : "공개 플레이"/);
  assert.doesNotMatch(page, /달무티 효과/);
  assert.match(page, /resolveQuickDalmutiAutoPass/);
  assert.match(page, /set\.rank === 1/);
  assert.match(page, /nextState\.table = null/);
  assert.match(page, /nextState\.passed = \[\]/);
  assert.match(page, /나머지 플레이어 자동 PASS/);
  assert.match(styles, /@keyframes quickDalmutiAutoPass/);
  assert.match(page, /kind: "play"/);
  assert.match(page, /kind: "pass"/);
  assert.match(page, /previousTable: state\.table/);
  assert.match(page, /visibleTable\?\.cards \?\? \[\]/);
  assert.match(
    page,
    /humanFinished[\s\S]{0,80}\$\{humanFinishRank \+ 1\}위/,
  );
  assert.match(
    page,
    /createOpeningRound\([\s\S]{0,120}quickPlayers,[\s\S]{0,80}scores,[\s\S]{0,80}quickBotDifficulty/,
  );
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
  assert.match(page, /if \(name === "나"\) return "내가"/);
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
  assert.match(
    page,
    /style=\{seatPosition\(rankSeat - 1, totalPlayers\)\}/,
  );
  assert.match(page, /"--seat-grid-column": \(rankIndex % compactColumns\) \+ 1/);
  assert.match(page, /className="human-status" ref=\{humanAnchorRef\}/);
  assert.doesNotMatch(page, /className="hand-wrap" ref=\{humanAnchorRef\}/);
  assert.match(page, /isDalmutiHighlighted/);
  assert.match(page, /className="welcome-crown"/);
  assert.match(
    styles,
    /Shared online visual system for the configurable 4–10-player quick match/,
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

test("tribute excludes Jesters while noble returns keep them", async () => {
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
    ["dalmuti", "archbishop"],
  );
  assert.deepEqual(
    selectPeonTaxCards(hand, 1).map((card) => card.id),
    ["dalmuti"],
  );
  assert.deepEqual(
    selectPeonTaxCards([{ id: "jester", rank: 13 }], 1),
    [],
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
  assert.match(page, /첫 게임은 선착순으로 카드를 한 장씩 골라 계급을 정합니다/);
  assert.match(page, /계급 미정/);
  assert.match(page, /declaredKind === "great-revolution"/);
  assert.match(page, /대혁명을 선포하시겠습니까/);
  assert.match(
    page,
    /if \(label\.includes\("PASS"\)\) return 1500;[\s\S]*if \(label\.includes\("PLAY"\)\) return 2250;/,
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
  assert.match(page, /<span>나의 서열<\/span>/);
  assert.match(page, /RANK_NAMES\[viewerRank\][\s\S]{0,100}카드를[\s\S]{0,100}선택했습니다/);
  assert.match(page, /styles\.resultFirst/);
  assert.match(page, /styles\.resultSecond/);
  assert.match(page, /--table-card-step-wide/);
  assert.doesNotMatch(page, /sendCommand\("PASS", \{ automatic: true \}\)/);
  assert.match(styles, /@keyframes onlineHandCardReveal/);
  assert.match(styles, /\.dalmutiEffectOverlay/);
  assert.match(styles, /\.playerSeatDalmuti/);
  assert.match(
    styles,
    /\.eventOverlay\s*\{[^}]*position: absolute;[^}]*pointer-events: none;/s,
  );
  assert.match(styles, /@keyframes eventCardSeatToTable/);
  assert.match(styles, /@keyframes taxCardTransfer/);
  assert.match(styles, /@keyframes passSeatToTable/);
  assert.match(page, /key=\{activeEvent\.id\}/);
  assert.match(page, /const \[initialElapsed\] = useState/);
  assert.match(page, /className=\{styles\.eventCenterCopy\}/);
  assert.match(
    page,
    /length: Math\.max\(1, displayedMe\?\.handCount \?\? 14\)/,
  );
  assert.match(page, /"--card-index": index/);
  assert.match(styles, /\.ownDockFinished \.playerSeatSelf/);
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

  assert.deepEqual(rankedDealCounts(80, 4), [20, 20, 20, 20]);
  assert.deepEqual(rankedDealCounts(80, 5), [16, 16, 16, 16, 16]);
  assert.deepEqual(rankedDealCounts(80, 6), [13, 13, 13, 13, 14, 14]);
  assert.deepEqual(rankedDealCounts(80, 7), [11, 11, 11, 11, 12, 12, 12]);
  assert.deepEqual(rankedDealCounts(80, 10), [
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  ]);
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
    /@media \(max-width: 820px\)[\s\S]*\.turnCountdown\s*\{[^}]*position: absolute;[^}]*top: 104px;[^}]*right: 13px;[^}]*left: auto;/,
  );
});

test("quick match delays its great-revolution seat reversal until the dedicated swap announcement", async () => {
  const quickPage = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  const decisionStart = quickPage.indexOf(
    "const resolveRevolution = (declare: boolean) =>",
  );
  const decisionEnd = quickPage.indexOf(
    "const toggleCard =",
    decisionStart,
  );
  assert.ok(decisionStart >= 0 && decisionEnd > decisionStart);
  const decisionBlock = quickPage.slice(decisionStart, decisionEnd);
  assert.match(
    decisionBlock,
    /phase = "revolution-intro";[\s\S]*"great-revolution"/,
  );
  assert.doesNotMatch(
    decisionBlock,
    /\.reverse\(\)/,
    "declaring a great revolution must not reverse seats before its first announcement ends",
  );

  const introEffectStart = quickPage.indexOf(
    'if (!game || game.phase !== "revolution-intro") return;',
  );
  const swapEffectStart = quickPage.indexOf(
    'if (!game || game.phase !== "great-revolution-swap") return;',
    introEffectStart,
  );
  assert.ok(introEffectStart >= 0 && swapEffectStart > introEffectStart);
  const introEffect = quickPage.slice(introEffectStart, swapEffectStart);
  assert.match(
    introEffect,
    /kind === "great-revolution"[\s\S]*phase: "great-revolution-swap"[\s\S]*players: assignRoles\(\[\.\.\.latest\.players\]\.reverse\(\)\)/,
  );
  assert.match(
    introEffect,
    /대혁명으로 인해 모두의 계급이 뒤바뀝니다/,
  );

  const swapEffectEnd = quickPage.indexOf(
    'if (!game || game.phase !== "hand-reveal") return;',
    swapEffectStart,
  );
  assert.ok(swapEffectEnd > swapEffectStart);
  const swapEffect = quickPage.slice(swapEffectStart, swapEffectEnd);
  assert.match(
    swapEffect,
    /phase: latest\.round === 1 \? "tax-intro" : "play-intro"/,
  );
  assert.match(
    swapEffect,
    /GREAT_REVOLUTION_SWAP_DURATION_MS/,
  );
});

test("online mode renders the dedicated great-revolution rank-swap announcement", async () => {
  const onlinePage = await readFile(
    new URL("../app/online/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    onlinePage,
    /GREAT_REVOLUTION_RANK_SWAP_STARTED|snapshot\.phase === "great-revolution-swap"/,
    "the online client must render the server's separate rank-swap phase or event",
  );
  assert.match(
    onlinePage,
    /대혁명으로 인해 모두의 계급이 뒤바뀝니다/,
  );
  assert.match(
    onlinePage,
    /const eventRound = numberValue\(event\.data\.round, Number\.NaN\);[\s\S]{0,80}eventRound !== round/,
  );
});

test("quick and online modes wire configurable bot difficulty into the shared policy", async () => {
  const [quickPage, quickStyles, onlinePage, engine, strategy] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/online-game/engine.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/bot-strategy.ts", import.meta.url), "utf8"),
  ]);

  assert.match(
    quickPage,
    /Array\.from\(\{ length: 7 \}, \(_, index\) => index \+ 4\)/,
  );
  assert.match(quickPage, /BASE_PLAYERS\.slice\(0, quickPlayerCount\)/);
  assert.match(quickPage, /chooseBotCardIds\(/);
  assert.match(
    quickPage,
    /publicPlayedCards: \[\.\.\.state\.publicPlayedCards, \.\.\.selected\]/,
  );
  assert.match(
    quickPage,
    /publicPlayedCards: state\.publicPlayedCards\.map\(\(card\) =>/,
  );
  assert.match(quickPage, /"--opening-rank-columns": Math\.min\(/);
  assert.match(quickPage, /"--rank-reveal-delay": `\$\{cardIndex \* 125\}ms`/);
  assert.match(
    quickStyles,
    /grid-template-columns: repeat\([\s\S]{0,100}var\(--opening-rank-columns, 5\)/,
  );
  assert.match(onlinePage, /BOT_DIFFICULTIES\.map\(\(difficulty\) =>/);
  assert.match(
    onlinePage,
    /sendCommand\("ADD_BOT",\s*\{\s*difficulty\s*\}\)/,
  );
  assert.match(engine, /command\.difficulty \?\? "normal"/);
  assert.match(engine, /INVALID_BOT_DIFFICULTY/);
  assert.match(strategy, /export const BOT_DIFFICULTIES = \["easy", "normal", "hard"\]/);
});

test("online play keeps the previous table visible during the submit animation", async () => {
  const [onlinePage, engine, onlineStyles] = await Promise.all([
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../lib/online-game/engine.ts", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
  ]);

  assert.match(engine, /const previousTable = state\.table/);
  assert.match(
    engine,
    /appendEvent\(state, "CARDS_PLAYED"[\s\S]{0,160}previousTable/,
  );
  assert.match(
    onlinePage,
    /const visibleTable = useMemo(?:<TableView>)?\([\s\S]{0,900}activeEvent\.data\.previousTable/,
  );
  assert.match(
    onlinePage,
    /\["CARDS_PLAYED", "DALMUTI_EFFECT", "PLAYER_PASSED"\]/,
  );
  assert.match(onlinePage, /<strong>DALMUTI<\/strong>/);
  assert.match(onlinePage, /visibleTable\?\.cards/);
  assert.match(
    onlineStyles,
    /\.dalmutiEffectOverlay > strong\s*\{[^}]*top: calc\(var\(--center-y\) \+ 100px\);[^}]*line-height: 0\.88;/s,
  );
  assert.match(
    onlineStyles,
    /\.dalmutiEffectOverlay > span\s*\{[^}]*top: calc\(var\(--center-y\) \+ 178px\);[^}]*line-height: 1\.45;/s,
  );
  assert.match(
    onlineStyles,
    /\.revealOverlay \.eventCenterCopy > strong\s*\{[^}]*white-space:\s*nowrap;[^}]*word-break:\s*keep-all;/s,
  );
});

test("online phase locks and visual animation timing mirror quick match", async () => {
  const [quickPage, onlinePage, onlineStyles, engine] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../lib/online-game/engine.ts", import.meta.url), "utf8"),
  ]);

  for (const expected of [
    /rankChoiceIntroMs: 3_300/,
    /rankRevealDelayMs: 1_500/,
    /rankRevealMs: 3_400/,
    /rankConfirmMs: 2_600/,
    /revealIntroMs: 2_400/,
    /handRevealMs: 1_400/,
    /revolutionIntroMs: 3_300/,
    /taxIntroMs: 2_400/,
    /taxTributeMs: 6_000/,
    /taxReturnMs: 6_000/,
    /playIntroMs: 2_600/,
    /BOT_ACTION_DELAY_MS = 750/,
    /PASS_ACTION_LOCK_MS = 1_500/,
    /PLAY_ACTION_LOCK_MS = 2_250/,
    /DALMUTI_ACTION_LOCK_MS = 3_300/,
    /ACTION_SETTLE_MS = 300/,
  ]) {
    assert.match(engine, expected);
  }

  for (const expected of [
    /animation: onlineOpeningRankCountdown 1\.05s/,
    /animation: rankCardReveal 1\.25s/,
    /animation: onlineRankConfirmation 2\.6s/,
    /animation: onlineHandCardReveal 1\.2s/,
    /animation: seatHandReveal 1\.15s/,
    /animation: onlinePhaseIntroReveal var\(--event-duration, 2400ms\)/,
    /\.playIntroOverlay \.eventCenterCopy\s*\{[^}]*animation-duration: var\(--event-duration, 2600ms\);/s,
    /\.taxSkippedIntroOverlay \.eventCenterCopy\s*\{[^}]*animation-duration: var\(--event-duration, 2400ms\);/s,
    /animation: onlineHandRevealIntro var\(--event-duration, 2400ms\)/,
    /animation: onlineRevealIntroCard 1\.8s/,
    /animation: onlineRevolutionAnnouncement 3\.1s/,
    /animation: onlinePublicCardPlay 2\.08s/,
    /animation: onlinePublicPlayCaption 2\.08s/,
    /animation: onlinePublicPassToTable 1\.38s/,
    /animation: taxCardTransfer 5\.55s/,
    /animation: onlineDalmutiFieldSpectacle 3\.25s/,
    /animation: onlineDalmutiShockwave 3\.1s/,
    /animation: onlineDalmutiAutoPass 2\.55s/,
    /animation: onlineDalmutiAutoPassBanner 3\.05s 240ms/,
    /\.rankShiftEffect\s*\{[^}]*animation-duration: 2\.3s;/s,
    /animation: myTurnBoardPulse 1\.3s ease-in-out infinite alternate/,
    /animation: greatRevolutionFieldBreath 2\.8s ease-in-out infinite alternate/,
    /animation: onlineGreatRevolutionOrbit 9s linear infinite/,
    /animation: onlineGreatRevolutionOrbit 14s linear infinite reverse/,
    /animation: onlineGreatRevolutionOrbit 7s linear infinite/,
  ]) {
    assert.match(onlineStyles, expected);
  }

  assert.match(onlinePage, /const EventOverlay = memo\(EventOverlayView/);
  assert.match(onlinePage, /const PLAYING_POLL_INTERVAL_MS = 180/);
  assert.match(onlinePage, /const LOBBY_POLL_INTERVAL_MS = 420/);
  assert.match(onlinePage, /const MAX_EVENT_CATCHUP_MS = 120/);
  assert.match(onlinePage, /const HAND_REVEAL_PRESENTATION_MS = 1_400/);
  assert.match(onlinePage, /TRANSITION_DEADLINE_POLL_PADDING_MS = 24/);
  assert.match(onlinePage, /MIN_TRANSITION_POLL_INTERVAL_MS = 48/);
  assert.match(
    onlinePage,
    /effectiveClock <\s*eventPresentationStartsAt\(event\) \+ event\.durationMs/,
  );
  assert.match(
    onlinePage,
    /current\.phaseEndsAt -\s*estimatedServerNow \+\s*TRANSITION_DEADLINE_POLL_PADDING_MS/,
  );
  assert.match(
    onlinePage,
    /REMOTE_ACTION_PRESENTATION_GRACE_MS = 300/,
  );
  assert.match(
    onlinePage,
    /collectRemoteActionPresentations\(/,
  );
  assert.match(onlinePage, /const \[remoteActionQueue, setRemoteActionQueue\]/);
  assert.match(
    onlinePage,
    /const localPlaybackStartedAt = Date\.now\(\)/,
  );
  assert.match(
    onlinePage,
    /Date\.now\(\) - event\.localPlaybackStartedAt/,
  );
  assert.match(
    onlinePage,
    /serverOffsetRef\.current = nextServerOffset/,
  );
  assert.doesNotMatch(onlinePage, /suppressPresentation/);
  assert.match(
    onlineStyles,
    /onlinePublicPassToTable 1\.38s[\s\S]*?both;/,
  );
  assert.match(onlinePage, /const turnPresentationReady =/);
  assert.match(onlinePage, /ROUND_END_MOVE_PRELUDE_MS = 380/);
  assert.match(onlinePage, /ROUND_END_MOVE_SETTLE_MS = 520/);
  assert.match(
    onlinePage,
    /effectiveClock - eventPresentationStartsAt\(event\)/,
  );
  assert.match(onlinePage, /presentationStartsAt: existing\.presentationStartsAt/);
  assert.match(onlinePage, /Promise\.allSettled\(/);
  assert.match(onlinePage, /animation\.finished/);
  assert.match(onlinePage, /stageRankMovement\("round-end"/);
  assert.match(onlinePage, /stageRankMovement\("great-revolution"/);
  assert.match(onlinePage, /player\.id === snapshot\.hostId \|\|/);
  assert.match(onlinePage, /방장 · 준비 완료/);
  assert.match(onlinePage, /달무티에 참가하기/);
  assert.doesNotMatch(onlinePage, /window\.confirm/);
  assert.match(onlinePage, /function RoomExitDialog/);
  assert.match(onlinePage, /setHandRevealElapsedMs/);
  assert.match(onlinePage, /const \[phaseElapsed\] = useState/);
  assert.match(onlinePage, /styles\.greatRevolutionFieldEffect/);
  assert.match(onlinePage, /const openingSequenceActive =/);
  assert.match(onlinePage, /styles\.openingSequenceVeilActive/);
  assert.match(onlineStyles, /\.openingSequenceVeilActive/);
  assert.match(onlinePage, /motionAnchorsEqual\(current, nextAnchors\)/);
  assert.match(onlinePage, /window\.requestAnimationFrame/);
  assert.match(onlinePage, /"--event-elapsed": `\$\{initialElapsed\}ms`/);
  assert.match(onlinePage, /"--phase-elapsed": `\$\{phaseElapsed\}ms`/);
  assert.match(onlinePage, /모든 플레이어가 동시에 자신의 패를 확인합니다/);
  assert.match(onlinePage, /패 공개가 끝나면 세금 교환을 시작합니다/);
  assert.match(quickPage, /제 1막은 세금 교환 없이 진행됩니다/);
  assert.match(onlinePage, /제 1막은 세금 교환 없이 진행됩니다/);
  assert.match(onlinePage, /const skipped = booleanValue\(data\.skipped\)/);
  assert.match(onlineStyles, /\.taxSkippedIntroOverlay/);
  assert.match(onlinePage, /dalmutiCards\.map/);
  assert.match(onlinePage, /"--event-card-mid-offset-x"/);
  assert.match(onlineStyles, /var\(--event-elapsed, 0ms\) \* -1/);
  assert.match(
    onlineStyles,
    /@keyframes onlineOpeningRankCountdown\s*\{[\s\S]*transform: scale\(0\.66\);[\s\S]*25%,[\s\S]*74%[\s\S]*transform: scale\(1\.18\);/,
  );
  assert.match(
    onlineStyles,
    /@keyframes rankCardReveal\s*\{\s*0%\s*\{\s*transform: rotateY\(0deg\);[\s\S]*100%\s*\{\s*transform: rotateY\(180deg\);/,
  );
  assert.match(
    onlineStyles,
    /\.rankConfirmationBody\s*\{[^}]*grid-template-columns: 70px minmax\(0, 1fr\);[^}]*gap: 26px;/s,
  );
  assert.match(
    onlineStyles,
    /\.taxRoutePrivate \.eventCardWrap\s*\{[^}]*--tax-endpoint-scale: 0\.342;[^}]*--tax-mid-scale: 1;/s,
  );
  assert.match(
    onlineStyles,
    /@keyframes seatHandReveal\s*\{[\s\S]*rotate\(var\(--card-angle, 0deg\)\) rotateY\(88deg\) scale\(1\)/,
  );
  assert.doesNotMatch(onlinePage, /event\.durationMs \+ 220/);
});

test("quick and online modes use the official player rank labels", async () => {
  const [quickPage, quickStyles, onlinePage, onlineStyles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
  ]);
  const exposedCopy = `${quickPage}\n${onlinePage}`;

  for (const [hyphenatedKey, underscoredKey, label] of [
    ["great-dalmuti", "great_dalmuti", "달무티"],
    ["lesser-dalmuti", "lesser_dalmuti", "총리대신"],
    ["lesser-peon", "lesser_peon", "소작농"],
    ["great-peon", "great_peon", "농노"],
  ]) {
    assert.match(quickPage, new RegExp(`"${hyphenatedKey}": "${label}"`));
    assert.match(onlinePage, new RegExp(`"${hyphenatedKey}": "${label}"`));
    assert.match(onlinePage, new RegExp(`${underscoredKey}: "${label}"`));
  }
  assert.match(quickPage, /merchant: "상인"/);
  assert.match(onlinePage, /merchant: "상인"/);
  assert.doesNotMatch(
    exposedCopy,
    /대 달무티|소 달무티|대 농노|소 농노|현재 계급/,
  );
  assert.match(quickPage, /<span>서열<\/span>\s*<small>누적 점수<\/small>/);
  assert.match(
    onlinePage,
    /<span>서열<\/span>\s*<small>누적 점수<\/small>/,
  );
  assert.match(quickPage, /className="revolution-joker-pair"/);
  assert.match(quickStyles, /@keyframes revolutionJokerEnterLeft/);
  assert.match(
    quickStyles,
    /@media \(prefers-reduced-motion: reduce\)[\s\S]*\.revolution-announcement\s*\{[^}]*opacity: 1 !important;[^}]*animation: none !important;/,
  );
  assert.match(
    onlinePage,
    /className=\{styles\.revolutionJokers\}[\s\S]{0,100}<span \/>\s*<span \/>/,
  );
  assert.match(onlineStyles, /@keyframes revolutionJokerArrivalLeft/);
  assert.match(
    onlineStyles,
    /\.revolutionJokers > span\s*\{[^}]*width: clamp\(88px, 9vw, 124px\);[^}]*aspect-ratio: 466 \/ 717;[^}]*url\("\/cards\/joker\.webp"\) center \/ cover no-repeat;/s,
  );
});

test("declared revolutions keep the field red for the whole round", async () => {
  const [quickPage, onlinePage] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(
    quickPage,
    /const isRevolutionActive = Boolean\(game\?\.revolutionAnnouncement\)/,
  );
  assert.match(
    quickPage,
    /isRevolutionActive \? "is-revolution" : ""/,
  );
  assert.match(
    quickPage,
    /isRevolutionActive[\s\S]{0,100}kind === "great-revolution"/,
  );
  assert.doesNotMatch(
    quickPage,
    /phase === "revolution-intro" \|\|\s*isGreatRevolutionActive/,
  );

  assert.match(
    onlinePage,
    /const revolutionFieldActive = Boolean\(declaredRevolution\)/,
  );
  assert.match(
    onlinePage,
    /revolutionFieldActive \? styles\.tableRevolution : ""/,
  );
  assert.doesNotMatch(onlinePage, /const revolutionAnnouncementActive/);
});

test("online chat is room-scoped and score rails use compact casino chips", async () => {
  const [
    quickPage,
    quickStyles,
    onlinePage,
    onlineStyles,
    roomStore,
    roomRoute,
    chatRoute,
    emoteRoute,
    leaveRoute,
    schema,
  ] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../lib/online-room-store.ts", import.meta.url), "utf8"),
    readFile(
      new URL("../app/api/online/rooms/[code]/route.ts", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL(
        "../app/api/online/rooms/[code]/chat/route.ts",
        import.meta.url,
      ),
      "utf8",
    ),
    readFile(
      new URL(
        "../app/api/online/rooms/[code]/emote/route.ts",
        import.meta.url,
      ),
      "utf8",
    ),
    readFile(
      new URL(
        "../app/api/online/rooms/[code]/leave/route.ts",
        import.meta.url,
      ),
      "utf8",
    ),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
  ]);

  assert.match(onlinePage, /function OnlineChatPanel/);
  assert.match(quickPage, /빠른 대전 플레이 인원/);
  assert.match(onlinePage, /ONLINE_CHAT_MAX_LENGTH/);
  assert.match(onlinePage, /sinceChatSeq/);
  assert.match(onlinePage, /\/chat`/);
  assert.match(
    onlinePage,
    /ingestChatMessages\(\s*\[message\],[\s\S]{0,120}false,/,
  );
  assert.match(onlinePage, /optimisticMessage: ChatMessageView/);
  assert.match(onlinePage, /seq: Number\.MAX_SAFE_INTEGER/);
  assert.match(onlinePage, /rankChoiceInFlightRef/);
  assert.match(onlinePage, /optimisticSlotIndex=\{optimisticRankSlotIndex\}/);
  assert.doesNotMatch(onlinePage, /dangerouslySetInnerHTML/);
  assert.match(onlineStyles, /\.chatPanel\s*\{/);
  assert.match(
    onlineStyles,
    /\.gameChatPanel\s*\{[\s\S]{0,300}background: transparent/,
  );
  assert.match(
    onlineStyles,
    /\.gameChatPanel\s*\{[\s\S]{0,120}bottom: 61px/,
  );
  assert.match(onlineStyles, /\.emotePicker\s*\{/);
  assert.match(onlineStyles, /\.playerEmote\s*\{/);
  assert.match(onlinePage, /onEmote=\{sendEmote\}/);
  assert.match(onlinePage, /activeEmote=\{activeEmotesByPlayerId/);
  assert.match(chatRoute, /authenticateOnlineRoomRequest/);
  assert.match(chatRoute, /appendOnlineRoomChatMessage/);
  assert.match(chatRoute, /room\.state\.players\.some/);
  assert.match(roomRoute, /readOnlineRoomChatMessages/);
  assert.match(roomRoute, /readOnlineRoomEmotes/);
  assert.match(emoteRoute, /appendOnlineRoomEmote/);
  assert.match(emoteRoute, /room\.state\.players\.some/);
  assert.match(roomStore, /CHAT_RATE_LIMIT/);
  assert.match(
    roomStore,
    /INSERT OR IGNORE INTO online_room_chat_messages[\s\S]*WHERE EXISTS[\s\S]*AND NOT EXISTS/,
  );
  assert.match(roomStore, /online_room_chat_messages/);
  assert.match(roomStore, /online_room_emotes/);
  assert.match(roomStore, /EMOTE_RATE_LIMIT/);
  assert.match(
    roomStore,
    /INNER JOIN online_rooms AS rooms[\s\S]*rooms\.expires_at > \?/,
  );
  assert.doesNotMatch(leaveRoute, /clearOnlineRoomChat/);
  assert.match(schema, /onlineRoomChatMessages/);
  assert.match(schema, /onlineRoomEmotes/);

  assert.match(quickPage, /className="score-display"/);
  assert.match(quickStyles, /\.score-chip-stack/);
  assert.match(onlinePage, /className=\{styles\.scoreDisplay\}/);
  assert.match(onlineStyles, /\.scoreChipStack/);
  assert.match(quickPage, />\s*제출\s*</);
  assert.match(onlinePage, />\s*제출\s*</);
  assert.doesNotMatch(
    `${quickPage}\n${onlinePage}`,
    /패 내기|카드 내기|나의 손패/,
  );
  assert.match(quickPage, /\$\{currentPlayer\.name\}의 차례/);
  assert.match(onlinePage, /snapshot\.currentPlayerId \?\? me\?\.id/);
});

test("online bot seats and quick finished-player acceleration stay wired to the UI", async () => {
  const [quickPage, onlinePage, onlineStyles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/online/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/online/online.module.css", import.meta.url),
      "utf8",
    ),
  ]);

  assert.match(
    onlinePage,
    /sendCommand\("ADD_BOT",\s*\{\s*difficulty\s*\}\)/,
  );
  assert.match(
    onlinePage,
    /sendCommand\("REMOVE_BOT",\s*\{\s*botId: player\.id,/,
  );
  assert.match(onlinePage, /player\.isBot \? styles\.lobbyPlayerBot/);
  assert.match(onlinePage, /player\.isBot[\s\S]{0,900}player\.ready/);
  assert.match(onlineStyles, /\.emptySlotInteractive/);
  assert.match(onlineStyles, /\.lobbyPlayerBot/);
  assert.match(onlineStyles, /\.botRemoveHint/);

  assert.match(
    quickPage,
    /function insufficientCardsPassTurn\([\s\S]*handCount >= state\.table\.count/,
  );
  assert.match(
    quickPage,
    /automaticReason: "insufficient-cards"/,
  );
  assert.match(quickPage, /const FAST_BOT_THINK_MS = 120/);
  assert.match(quickPage, /function skipRemainingBotTurns/);
  assert.match(
    quickPage,
    /const humanFinished = Boolean\(game\?\.finishOrder\.includes\(HUMAN_ID\)\);[\s\S]{0,120}const canSkipRemainingBots =\s*humanFinished && game\?\.phase === "playing"/,
  );
  assert.match(quickPage, /onClick=\{skipRemainingPlayers\}/);
  assert.match(quickPage, /className="skip-round-button"/);
});
