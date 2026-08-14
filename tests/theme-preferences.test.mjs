import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const root = process.cwd();
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");

function themeCssTail(source, marker, label) {
  const start = source.indexOf(marker);
  assert.notEqual(start, -1, `${label} has no Halloween theme block`);
  return source.slice(start);
}

function assertCssRule(
  source,
  selectorFragment,
  declaration,
  message,
  selectorExcludes = [],
) {
  const matches = source.split("}").some((candidate) => {
    const openingBrace = candidate.lastIndexOf("{");
    if (openingBrace < 0) return false;
    const selector = candidate.slice(0, openingBrace);
    const declarations = candidate.slice(openingBrace + 1);
    declaration.lastIndex = 0;
    return (
      selector.includes(selectorFragment) &&
      selectorExcludes.every((fragment) => !selector.includes(fragment)) &&
      declaration.test(declarations)
    );
  });
  assert.equal(matches, true, message ?? `${selectorFragment} is not themed`);
}

const {
  APP_PREFERENCES_STORAGE_KEY,
  APP_THEMES,
  DEFAULT_APP_PREFERENCES,
  HALLOWEEN_CARD_PROFESSION_NAMES,
  ORIGINAL_CARD_PROFESSION_NAMES,
  cardArtPath,
  cardProfessionName,
  clampBgmVolume,
  normalizeAppPreferences,
  parseAppPreferences,
  themeColor,
} = await import(new URL("../lib/app-preferences.ts", import.meta.url));

function webpDimensions(file) {
  const bytes = fs.readFileSync(path.join(root, file));
  assert.equal(bytes.toString("ascii", 0, 4), "RIFF", `${file} is not RIFF`);
  assert.equal(bytes.toString("ascii", 8, 12), "WEBP", `${file} is not WebP`);

  for (let offset = 12; offset + 8 <= bytes.length; ) {
    const type = bytes.toString("ascii", offset, offset + 4);
    const length = bytes.readUInt32LE(offset + 4);
    const data = offset + 8;

    if (type === "VP8X") {
      return {
        width: bytes.readUIntLE(data + 4, 3) + 1,
        height: bytes.readUIntLE(data + 7, 3) + 1,
      };
    }
    if (type === "VP8 ") {
      assert.deepEqual(
        [...bytes.subarray(data + 3, data + 6)],
        [0x9d, 0x01, 0x2a],
        `${file} has an invalid VP8 key-frame header`,
      );
      return {
        width: bytes.readUInt16LE(data + 6) & 0x3fff,
        height: bytes.readUInt16LE(data + 8) & 0x3fff,
      };
    }
    if (type === "VP8L") {
      assert.equal(bytes[data], 0x2f, `${file} has an invalid VP8L header`);
      const b1 = bytes[data + 1];
      const b2 = bytes[data + 2];
      const b3 = bytes[data + 3];
      const b4 = bytes[data + 4];
      return {
        width: 1 + b1 + ((b2 & 0x3f) << 8),
        height: 1 + (b2 >> 6) + (b3 << 2) + ((b4 & 0x0f) << 10),
      };
    }

    offset = data + length + (length % 2);
  }

  assert.fail(`${file} has no supported WebP image chunk`);
}

function pngDimensions(file) {
  const bytes = fs.readFileSync(path.join(root, file));
  assert.deepEqual(
    [...bytes.subarray(0, 8)],
    [137, 80, 78, 71, 13, 10, 26, 10],
    `${file} is not PNG`,
  );
  assert.equal(bytes.toString("ascii", 12, 16), "IHDR", `${file} has no IHDR`);
  return {
    width: bytes.readUInt32BE(16),
    height: bytes.readUInt32BE(20),
  };
}

test("app preferences use safe persistent defaults and normalize stored data", () => {
  assert.equal(APP_PREFERENCES_STORAGE_KEY, "dalmuti.preferences.v1");
  assert.deepEqual(APP_THEMES, ["original", "halloween"]);
  assert.deepEqual(DEFAULT_APP_PREFERENCES, {
    theme: "original",
    bgmEnabled: false,
    bgmVolume: 55,
  });

  assert.deepEqual(parseAppPreferences(null), DEFAULT_APP_PREFERENCES);
  assert.deepEqual(parseAppPreferences("not-json"), DEFAULT_APP_PREFERENCES);
  assert.deepEqual(
    normalizeAppPreferences({
      theme: "halloween",
      bgmEnabled: true,
      bgmVolume: 140,
    }),
    { theme: "halloween", bgmEnabled: true, bgmVolume: 100 },
  );
  assert.deepEqual(normalizeAppPreferences({ theme: "unknown" }), {
    theme: "original",
    bgmEnabled: false,
    bgmVolume: 55,
  });
  assert.equal(clampBgmVolume(-7), 0);
  assert.equal(clampBgmVolume(42.6), 43);
  assert.equal(clampBgmVolume(Number.NaN), 55);
});

test("theme helpers map every legal rank without changing card identity", () => {
  assert.equal(cardArtPath("original", 1), "/cards/01.webp");
  assert.equal(cardArtPath("original", 13), "/cards/joker.webp");
  assert.equal(cardArtPath("halloween", 1), "/cards/halloween/01.webp");
  assert.equal(
    cardArtPath("halloween", 13),
    "/cards/halloween/joker.webp",
  );
  assert.throws(() => cardArtPath("original", 0), /rank/);
  assert.throws(() => cardArtPath("halloween", 13.5), /rank/);
  assert.equal(themeColor("original"), "#18070c");
  assert.equal(themeColor("halloween"), "#09080c");
});

test("card profession names follow the selected artwork without renaming player ranks", () => {
  assert.deepEqual(ORIGINAL_CARD_PROFESSION_NAMES, {
    1: "달무티",
    2: "대주교",
    3: "시종장",
    4: "남작부인",
    5: "수녀원장",
    6: "기사",
    7: "재봉사",
    8: "석공",
    9: "요리사",
    10: "양치기",
    11: "광부",
    12: "농노",
    13: "어릿광대",
  });
  assert.deepEqual(HALLOWEEN_CARD_PROFESSION_NAMES, {
    1: "달무티",
    2: "심령대신",
    3: "법무관",
    4: "뱀 남작부인",
    5: "비술사",
    6: "성전기사",
    7: "거미술사",
    8: "장인",
    9: "약제사",
    10: "사육사",
    11: "박멸자",
    12: "쥐잡이",
    13: "광대",
  });
  for (let rank = 1; rank <= 13; rank += 1) {
    assert.equal(
      cardProfessionName("original", rank),
      ORIGINAL_CARD_PROFESSION_NAMES[rank],
    );
    assert.equal(
      cardProfessionName("halloween", rank),
      HALLOWEEN_CARD_PROFESSION_NAMES[rank],
    );
  }
  assert.throws(() => cardProfessionName("halloween", 0), /rank/);

  const quick = read("app/page.tsx");
  const online = read("app/online/page.tsx");
  const rulebook = read("app/components/RulebookDialog.tsx");
  const settings = read("app/settings/page.tsx");
  for (const source of [quick, online, rulebook, settings]) {
    assert.match(source, /cardProfessionName/);
  }
  assert.doesNotMatch(quick, /const RANK_NAMES/);
  assert.doesNotMatch(online, /const RANK_NAMES/);
  assert.match(rulebook, /themedRulebookCopy/);
});

test("ships a normalized Halloween face, back, and crown asset set", () => {
  const names = [
    ...Array.from({ length: 12 }, (_, index) =>
      String(index + 1).padStart(2, "0"),
    ),
    "joker",
    "back",
  ];

  for (const name of names) {
    const file = `public/cards/halloween/${name}.webp`;
    assert.equal(fs.existsSync(path.join(root, file)), true, `${file} is missing`);
    assert.deepEqual(webpDimensions(file), { width: 1040, height: 1600 });
  }

  assert.ok(
    fs.statSync(path.join(root, "public/cards/halloween/back.webp")).size >
      12_000,
    "Halloween back must fill the exported frame, not embed a corner thumbnail",
  );

  const crown = "public/themes/halloween/crown.webp";
  assert.equal(fs.existsSync(path.join(root, crown)), true, `${crown} is missing`);
  const crownSize = webpDimensions(crown);
  assert.equal(crownSize.width, crownSize.height, "Halloween crown must be square");
  assert.ok(crownSize.width >= 512, "Halloween crown resolution is too small");

  const handSprites =
    "public/themes/halloween/dalmuti-hand-field-atlas-v2.png";
  assert.equal(
    fs.existsSync(path.join(root, handSprites)),
    true,
    `${handSprites} is missing`,
  );
  assert.deepEqual(pngDimensions(handSprites), { width: 4800, height: 1806 });

  const inkBloom =
    "public/themes/halloween/ink-impact-bloom-mask-v1.png";
  assert.equal(
    fs.existsSync(path.join(root, inkBloom)),
    true,
    `${inkBloom} is missing`,
  );
  const inkBloomSize = pngDimensions(inkBloom);
  assert.equal(inkBloomSize.width, inkBloomSize.height);
  assert.ok(inkBloomSize.width >= 1024, "Halloween ink bloom resolution is too small");
  assert.equal(4800 / 4, 1200, "hand atlas must span the field in four columns");
  assert.equal(1806 / 3, 602, "hand atlas must retain three exact rows");
  assert.ok(fs.statSync(path.join(root, handSprites)).size > 10_000);
});

test("settings and theme preferences are wired without starting audio", () => {
  const provider = read("app/components/AppPreferencesProvider.tsx");
  const layout = read("app/layout.tsx");
  const home = read("app/page.tsx");
  const settings = read("app/settings/page.tsx");
  const settingsStyles = read("app/settings/settings.module.css");

  assert.match(provider, /document\.documentElement\.dataset\.theme/);
  assert.match(provider, /querySelectorAll<HTMLMetaElement>/);
  assert.match(provider, /window\.addEventListener\("storage"/);
  assert.match(layout, /THEME_BOOT_SCRIPT/);
  assert.match(layout, /<AppPreferencesProvider>/);
  assert.match(layout, /data-theme="original"/);

  assert.match(home, /className="settings-gear-link"/);
  assert.match(home, /href="\/settings"/);
  assert.match(home, /aria-label="환경설정"/);

  assert.match(settings, /useAppPreferences/);
  assert.match(settings, /APP_THEMES\.map/);
  assert.match(settings, /type="checkbox"/);
  assert.match(settings, /type="range"/);
  assert.match(settings, /role="radiogroup"/);
  assert.match(settings, /cardArtPath\(theme, 1\)/);
  assert.doesNotMatch(settings, /<audio\b|new Audio\b|\.play\s*\(/);
  assert.match(settingsStyles, /data-theme="halloween"/);
});

test("quick, online, and rulebook card faces share the selected theme", () => {
  const quick = read("app/page.tsx");
  const online = read("app/online/page.tsx");
  const rulebook = read("app/components/RulebookDialog.tsx");
  const quickStyles = read("app/globals.css");
  const onlineStyles = read("app/online/online.module.css");
  const rulebookStyles = read("app/components/RulebookDialog.module.css");
  const creditsStyles = read("app/components/CreditsDialog.module.css");

  assert.match(quick, /useAppPreferences/);
  assert.match(quick, /cardArtPath\(preferences\.theme/);
  assert.match(online, /useAppPreferences/);
  assert.match(online, /cardArtPath\(preferences\.theme/);
  assert.match(rulebook, /useAppPreferences/);
  assert.match(rulebook, /cardArtPath\(preferences\.theme/);

  assert.match(quickStyles, /data-theme="halloween"/);
  assert.match(quickStyles, /cards\/halloween\/back\.webp/);
  assert.match(quickStyles, /themes\/halloween\/crown\.webp/);
  assert.match(quickStyles, /--dalmuti-card-back-image/);
  assert.match(quickStyles, /--dalmuti-brand-crown-image/);
  assert.match(onlineStyles, /var\(--dalmuti-card-back-image\)/);
  assert.match(onlineStyles, /var\(--dalmuti-brand-crown-image\)/);
  assert.match(rulebookStyles, /var\(--dalmuti-card-back-image\)/);
  assert.match(rulebookStyles, /var\(--dalmuti-brand-crown-image\)/);
  assert.match(creditsStyles, /var\(--dalmuti-brand-crown-image\)/);
  assert.match(
    creditsStyles,
    /data-theme="halloween"\]\) \.done\s*\{[\s\S]{0,240}background: linear-gradient\(140deg, #543b5b, #29242f 58%, #17151c\);/,
  );
  assert.doesNotMatch(
    creditsStyles,
    /data-theme="halloween"\]\) \.done\s*\{[\s\S]{0,180}#c87849/,
  );

  assert.match(quick, /data-rank=\{card\.rank\}/);
  assert.match(online, /data-rank=\{card\.rank\}/);
  assert.match(
    quickStyles,
    /\.playing-card\[data-rank="1"\][\s\S]{0,180}\.playing-card\[data-rank="13"\]/,
    "quick special face borders must use numeric ranks",
  );
  assert.match(
    onlineStyles,
    /\.card:not\(\.cardBack\)\[data-rank="1"\][\s\S]{0,220}\.card:not\(\.cardBack\)\[data-rank="13"\]/,
    "online hidden card backs must stay outside numeric rank face styling",
  );
});

test("Halloween polish is complete, theme-scoped, and preserves responsive contracts", () => {
  const quickPage = read("app/page.tsx");
  const onlinePage = read("app/online/page.tsx");
  const inkCanvas = read("app/components/HalloweenInkContaminationCanvas.tsx");
  const quickStyles = read("app/globals.css");
  const onlineStyles = read("app/online/online.module.css");
  const rulebookStyles = read("app/components/RulebookDialog.module.css");
  const quickHalloween = themeCssTail(
    quickStyles,
    'html[data-theme="halloween"] .game-shell',
    "Quick Match CSS",
  );
  const onlineHalloween = themeCssTail(
    onlineStyles,
    ':global(html[data-theme="halloween"])',
    "Online CSS",
  );
  const rulebookHalloween = themeCssTail(
    rulebookStyles,
    ':global(html[data-theme="halloween"])',
    "Rulebook CSS",
  );

  assert.ok(
    (quickStyles.match(/var\(--dalmuti-card-back-image\) center \/ cover/g) ?? [])
      .length >= 5,
    "every Quick card-back presentation must use the full-frame shared image",
  );
  assert.ok(
    (onlineStyles.match(/var\(--dalmuti-card-back-image\) center \/ cover/g) ?? [])
      .length >= 6,
    "every Online card-back presentation must use the full-frame shared image",
  );
  assert.match(
    rulebookStyles,
    /var\(--dalmuti-card-back-image\) center \/ cover/,
  );

  const blackFrame =
    /(?:border(?:-color)?|background(?:-color)?):[^;}]*#0[0-9a-f]{5}/i;
  for (const selector of [
    ".opening-rank-card-front",
    ".opening-rank-confirmation-card",
  ]) {
    assertCssRule(
      quickHalloween,
      selector,
      blackFrame,
      `${selector} must use the Halloween black face frame`,
    );
  }
  assertCssRule(
    quickHalloween,
    ".playing-card",
    blackFrame,
    "every Quick face must use the Halloween black frame",
    ["[data-rank"],
  );
  for (const selector of [
    ".rankChoiceCardFront",
    ".rankConfirmationCard",
  ]) {
    assertCssRule(
      onlineHalloween,
      selector,
      blackFrame,
      `${selector} must use the Halloween black face frame`,
    );
  }
  assertCssRule(
    onlineHalloween,
    ".card:not(.cardBack)",
    blackFrame,
    "every Online face must use the Halloween black frame",
    ["[data-rank"],
  );
  for (const selector of [
    ".ruleCard img",
    ".turnExample img",
    ".taxCards img",
    ".rankDrawVisual img",
  ]) {
    assertCssRule(
      rulebookHalloween,
      selector,
      blackFrame,
      `${selector} must use the Halloween black face frame`,
    );
  }

  assert.match(
    onlinePage,
    /className=\{styles\.heroCards\}[\s\S]{0,500}data-rank=\{rank\}/,
    "online decorative cards need rank-addressable pre-hydration masks",
  );
  for (const rank of ["1", "2", "13"]) {
    assert.match(
      onlineStyles,
      new RegExp(`\\.heroCards > span\\[data-rank="${rank}"\\]::before`),
    );
  }
  assert.match(onlineStyles, /\/cards\/halloween\/01\.webp/);
  assert.match(onlineStyles, /\/cards\/halloween\/02\.webp/);
  assert.match(onlineStyles, /\/cards\/halloween\/joker\.webp/);
  assertCssRule(
    onlineHalloween,
    ".heroCards > span::before",
    /z-index:\s*2/,
    "Halloween hero masks must sit above the hydration-time face image",
  );
  assertCssRule(
    onlineHalloween,
    ".heroCards img",
    /opacity:\s*0/,
    "Halloween hero masks must suppress the Original hydration image",
  );

  assert.match(quickHalloween, /themes\/halloween\/dalmuti-hand-field-atlas-v2\.png/);
  assert.match(onlineHalloween, /themes\/halloween\/dalmuti-hand-field-atlas-v2\.png/);
  assert.match(quickHalloween, /halloweenDalmutiHandRevealQuick/);
  assert.match(onlineHalloween, /halloweenDalmutiHandRevealOnline/);
  assert.match(quickHalloween, /halloweenDalmutiCardRevealQuick/);
  assert.match(onlineHalloween, /halloweenDalmutiCardRevealOnline/);
  assert.match(quickHalloween, /-webkit-mask-size:\s*400% 300%/);
  assert.match(onlineHalloween, /-webkit-mask-size:\s*400% 300%/);
  assertCssRule(
    quickHalloween,
    ".public-turn-action-layer.is-dalmuti::before",
    /width:\s*min\(720px, calc\(100% - 32px\)\)/,
    "Quick Halloween Dalmuti vignette must stay centered on the field",
  );
  assertCssRule(
    quickHalloween,
    ".public-turn-action-layer.is-dalmuti::before",
    /transform:\s*translate\(-50%, -50%\)/,
    "Quick Halloween Dalmuti vignette must follow the field midpoint",
  );
  assert.match(
    onlineHalloween,
    /\.dalmutiEffectOverlay\s*\{\s*background:\s*transparent;/,
    "Online Halloween Dalmuti must not darken the full board overlay",
  );
  assert.match(quickHalloween, /--hand-sprite-x/);
  assert.match(onlineHalloween, /--hand-sprite-x/);
  assert.match(quickHalloween, /aspect-ratio:\s*1200 \/ 602/);
  assert.match(onlineHalloween, /aspect-ratio:\s*1200 \/ 602/);
  assert.match(
    quickHalloween,
    /span:nth-of-type\(11\)[^{]*\{[^}]*z-index:\s*2;[^}]*transform:\s*translate\(-7\.67%,\s*20%\)/s,
  );
  assert.match(
    onlineHalloween,
    /span:nth-of-type\(11\)[^{]*\{[^}]*z-index:\s*2;[^}]*transform:\s*translate\(-7\.67%,\s*20%\)/s,
  );
  assert.match(
    quickHalloween,
    /span:nth-of-type\(12\)[^{]*\{[^}]*z-index:\s*2;[^}]*transform:\s*translate\(-7\.67%,\s*23\.82%\)/s,
  );
  assert.match(
    onlineHalloween,
    /span:nth-of-type\(12\)[^{]*\{[^}]*z-index:\s*2;[^}]*transform:\s*translate\(-7\.67%,\s*23\.82%\)/s,
  );
  assert.doesNotMatch(quickPage, /halloween-card-support-grip/);
  assert.doesNotMatch(onlinePage, /halloweenCardSupportGrip/);
  assert.match(quickHalloween, /ink-wash-field-texture-v2\.webp/);
  assert.match(onlineHalloween, /ink-wash-field-texture-v2\.webp/);
  assert.match(quickPage, /"--halloween-pass-delay"/);
  assert.match(onlinePage, /"--halloween-pass-delay"/);
  assert.match(quickHalloween, /animation-duration:\s*1\.22s/);
  assert.match(onlineHalloween, /animation-duration:\s*1\.22s/);

  for (const selector of [
    ".opening-rank-intro",
    ".opening-rank-confirmation",
    ".hand-reveal-intro",
    ".tax-route-caption",
    ".tax-selection-progress",
  ]) {
    assert.match(
      quickHalloween,
      new RegExp(selector.replaceAll(".", "\\.")),
      `${selector} must be present in the neutral Halloween pregame palette`,
    );
  }
  assert.match(
    quickHalloween,
    /\.phase-intro:not\([\s\S]{0,260}\.is-great-revolution-swap/,
    "neutral Quick phase styling must exclude great revolution",
  );
  for (const selector of [
    ".rankChoiceIntro",
    ".rankConfirmation",
    ".taxSkippedIntroOverlay",
    ".taxWaitingField",
    ".taxRoute .transferNames",
  ]) {
    assert.match(
      onlineHalloween,
      new RegExp(selector.replaceAll(".", "\\.")),
      `${selector} must be present in the neutral Halloween pregame palette`,
    );
  }
  assert.match(
    onlineHalloween,
    /\.phaseIntroOverlay \.eventCenterCopy::before/,
    "Online needs a neutral Halloween phase surface",
  );
  assert.match(
    onlineHalloween,
    /\.revolutionOverlay \.eventCenterCopy::before/,
    "Online revolution must explicitly override the neutral phase surface",
  );

  for (const [styles, field, effect, canvas, hand] of [
    [
      quickHalloween,
      ".felt-table.is-revolution",
      ".great-revolution-field-effect",
      ".halloween-revolution-ink-canvas",
      "halloweenGreatRevolutionHandReverseQuick",
    ],
    [
      onlineHalloween,
      ".tableRevolution",
      ".greatRevolutionFieldEffect",
      ".revolutionInkCanvas",
      "halloweenGreatRevolutionHandReverseOnline",
    ],
  ]) {
    assert.match(styles, new RegExp(field.replaceAll(".", "\\.")));
    assert.match(styles, new RegExp(effect.replaceAll(".", "\\.")));
    assert.match(styles, new RegExp(canvas.replaceAll(".", "\\.")));
    assert.match(styles, new RegExp(`@keyframes\\s+${hand}`));
    assert.match(styles, /rotate\(calc\(var\(--clock-start\) - 360deg\)\)/);
  }
  assert.match(quickPage, /halloween-revolution-ink-transition/);
  assert.match(onlinePage, /revolutionInkTransition/);
  assert.match(quickPage, /HalloweenInkContaminationCanvas/);
  assert.match(onlinePage, /HalloweenInkContaminationCanvas/);
  assert.match(quickHalloween, /@keyframes\s+halloweenInkDropQuick/);
  assert.match(onlineHalloween, /@keyframes\s+halloweenInkDropOnline/);
  assert.doesNotMatch(quickPage, /revolution-ink-impact-/);
  assert.doesNotMatch(onlinePage, /revolution-ink-impact-/);
  assert.match(quickHalloween, /clip-path:\s*polygon\(50% 0, 64% 25%/);
  assert.match(onlineHalloween, /clip-path:\s*polygon\(50% 0, 64% 25%/);
  assert.match(inkCanvas, /ink-impact-bloom-mask-v1\.png/);
  assert.match(inkCanvas, /globalCompositeOperation = "source-in"/);
  assert.match(inkCanvas, /DROP_IMPACT_MS = 897/);
  assert.match(inkCanvas, /0\.32 \* progress \+ 4\.44 \* eased \*\* 2\.4/);
  assert.match(inkCanvas, /requestAnimationFrame\(render\)/);
  assert.doesNotMatch(quickPage, /revolution-ink-stain-/);
  assert.doesNotMatch(onlinePage, /revolution-ink-stain-/);
  assert.match(
    quickHalloween,
    /\.great-revolution-field-effect::after[\s\S]{0,900}animation:\s*none/,
    "Quick Match clock face and tick marks must remain fixed",
  );
  assert.match(
    onlineHalloween,
    /\.greatRevolutionFieldEffect::after[\s\S]{0,900}animation:\s*none/,
    "Online clock face and tick marks must remain fixed",
  );
  assert.doesNotMatch(quickPage, /"--tar-bubble-duration"/);
  assert.doesNotMatch(onlinePage, /"--tar-bubble-duration"/);
  assert.match(quickHalloween, /\.great-revolution-field-effect > span[\s\S]{0,220}display:\s*none/);
  assert.match(onlineHalloween, /\.greatRevolutionFieldEffect > span[\s\S]{0,260}display:\s*none/);
  assert.match(quickHalloween, /\.revolution-joker-card[\s\S]{0,160}visibility:\s*visible/);
  assert.match(onlineHalloween, /background-image:\s*var\(--dalmuti-joker-card-image\)/);
  assert.match(quickHalloween, /\.felt-table\.is-great-revolution/);
  assert.match(onlineHalloween, /\.tableGreatRevolution/);

  assert.equal(
    (quickPage.match(/href="\/settings"/g) ?? []).length,
    2,
    "main screen must expose one responsive settings entry per placement",
  );
  assert.match(
    quickPage,
    /className="settings-gear-link main-menu-settings-gear-link"/,
  );
  assertCssRule(
    quickStyles,
    ".main-menu-settings-gear-link",
    /display:\s*none/,
    "desktop panel gear must be hidden by default for touch/mobile layouts",
  );
  const finePointerStart = quickStyles.indexOf(
    "@media (hover: hover) and (pointer: fine)",
  );
  assert.notEqual(finePointerStart, -1, "desktop settings placement needs a fine-pointer query");
  const finePointerStyles = quickStyles.slice(finePointerStart);
  assertCssRule(
    finePointerStyles,
    ".game-shell .topbar > .settings-gear-link",
    /display:\s*none/,
    "fine-pointer PCs must hide the top-bar settings entry",
  );
  for (const property of [
    /display:\s*grid/,
    /position:\s*absolute/,
    /top:\s*\d+px/,
    /right:\s*\d+px/,
  ]) {
    assertCssRule(
      finePointerStyles,
      ".game-shell .main-menu-settings-gear-link",
      property,
      "fine-pointer PCs must place settings in the main panel's upper-right",
    );
  }
});

test("service worker caches theme assets without caching online state", () => {
  const worker = read("public/sw.js");

  assert.match(worker, /2026-08-14-halloween-dalmuti-name-v22/);
  assert.match(worker, /"\/cards\/halloween\/back\.webp"/);
  assert.match(worker, /"\/themes\/halloween\/crown\.webp"/);
  assert.match(worker, /"\/themes\/halloween\/dalmuti-hand-field-atlas-v2\.png"/);
  assert.match(worker, /"\/themes\/halloween\/ink-wash-field-texture-v2\.webp"/);
  assert.match(worker, /"\/themes\/halloween\/ink-impact-bloom-mask-v1\.png"/);
  assert.match(worker, /pathname\.startsWith\("\/themes\/"\)/);
  assert.match(worker, /request\.method !== "GET"/);
  assert.match(worker, /isOnlineApi\(url\.pathname\)/);
  assert.match(worker, /event\.respondWith\(fetch\(request\)\)/);
});

test("offline fallback follows the local theme without changing reconnect", () => {
  const offline = read("public/offline.html");

  assert.match(offline, /localStorage\.getItem\("dalmuti\.preferences\.v1"\)/);
  assert.match(offline, /stored\?\.theme === "halloween"/);
  assert.match(offline, /html\[data-theme="halloween"\]/);
  assert.match(offline, /\/themes\/halloween\/crown\.webp/);
  assert.match(offline, /#09080c/);
  assert.match(offline, /onclick="location\.reload\(\)"/);
});

test("documents the complete visual and no-gameplay-change acceptance pass", () => {
  const checklist = read("docs/halloween-theme-checklist.md");

  for (const required of [
    "Theme architecture and reversibility",
    "Settings screen",
    "Shared card presentation",
    "Main and quick-match palette",
    "Online-only palette",
    "Dialogs and secondary surfaces",
    "Responsive and installed-app verification",
    "Cache, tests, and release gate",
  ]) {
    assert.match(checklist, new RegExp(required));
  }
  assert.match(checklist, /must never change game\s+rules/);
  assert.match(checklist, /Original -> Halloween -> Original/);
});
