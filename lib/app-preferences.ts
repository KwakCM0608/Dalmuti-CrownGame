export const APP_PREFERENCES_STORAGE_KEY = "dalmuti.preferences.v1";

export const APP_THEMES = ["original", "halloween"] as const;

export type AppTheme = (typeof APP_THEMES)[number];

export type AppPreferences = {
  theme: AppTheme;
  bgmEnabled: boolean;
  bgmVolume: number;
};

export const DEFAULT_APP_PREFERENCES: AppPreferences = Object.freeze({
  theme: "original",
  bgmEnabled: false,
  bgmVolume: 55,
});

export const ORIGINAL_CARD_PROFESSION_NAMES: Readonly<Record<number, string>> =
  Object.freeze({
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

/**
 * Korean display names adapted from the professions printed on the Halloween
 * deck. These are card identities, not the players' social ranks.
 */
export const HALLOWEEN_CARD_PROFESSION_NAMES: Readonly<Record<number, string>> =
  Object.freeze({
    1: "달무티",
    2: "심령대신",
    3: "법무관",
    4: "뱀 남작부인",
    5: "비술사",
    6: "성전기사",
    7: "실 잣는 자",
    8: "장인",
    9: "약제사",
    10: "사육사",
    11: "박멸자",
    12: "쥐잡이",
    13: "광대",
  });

export function cardProfessionName(theme: AppTheme, rank: number): string {
  if (!Number.isInteger(rank) || rank < 1 || rank > 13) {
    throw new RangeError("card rank must be an integer from 1 to 13");
  }
  const names =
    theme === "halloween"
      ? HALLOWEEN_CARD_PROFESSION_NAMES
      : ORIGINAL_CARD_PROFESSION_NAMES;
  return names[rank];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function clampBgmVolume(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return DEFAULT_APP_PREFERENCES.bgmVolume;
  }
  return Math.round(Math.max(0, Math.min(100, value)));
}

export function normalizeAppPreferences(value: unknown): AppPreferences {
  if (!isRecord(value)) return { ...DEFAULT_APP_PREFERENCES };
  return {
    theme: APP_THEMES.includes(value.theme as AppTheme)
      ? (value.theme as AppTheme)
      : DEFAULT_APP_PREFERENCES.theme,
    bgmEnabled:
      typeof value.bgmEnabled === "boolean"
        ? value.bgmEnabled
        : DEFAULT_APP_PREFERENCES.bgmEnabled,
    bgmVolume: clampBgmVolume(value.bgmVolume),
  };
}

export function parseAppPreferences(serialized: string | null): AppPreferences {
  if (!serialized) return { ...DEFAULT_APP_PREFERENCES };
  try {
    return normalizeAppPreferences(JSON.parse(serialized));
  } catch {
    return { ...DEFAULT_APP_PREFERENCES };
  }
}

export function cardArtPath(theme: AppTheme, rank: number): string {
  if (!Number.isInteger(rank) || rank < 1 || rank > 13) {
    throw new RangeError("card rank must be an integer from 1 to 13");
  }
  const file = rank === 13 ? "joker" : String(rank).padStart(2, "0");
  return theme === "halloween"
    ? `/cards/halloween/${file}.webp`
    : `/cards/${file}.webp`;
}

export function themeColor(theme: AppTheme): string {
  return theme === "halloween" ? "#09080c" : "#18070c";
}
