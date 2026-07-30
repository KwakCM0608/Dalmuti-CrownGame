export const ROOM_CODE_LENGTH = 6;

const INITIAL_KEYS = [
  "r",
  "R",
  "s",
  "e",
  "E",
  "f",
  "a",
  "q",
  "Q",
  "t",
  "T",
  "d",
  "w",
  "W",
  "c",
  "z",
  "x",
  "v",
  "g",
] as const;

const MEDIAL_KEYS = [
  "k",
  "o",
  "i",
  "O",
  "j",
  "p",
  "u",
  "P",
  "h",
  "hk",
  "ho",
  "hl",
  "y",
  "n",
  "nj",
  "np",
  "nl",
  "b",
  "m",
  "ml",
  "l",
] as const;

const FINAL_KEYS = [
  "",
  "r",
  "R",
  "rt",
  "s",
  "sw",
  "sg",
  "e",
  "f",
  "fr",
  "fa",
  "fq",
  "ft",
  "fx",
  "fv",
  "fg",
  "a",
  "q",
  "qt",
  "t",
  "T",
  "d",
  "w",
  "c",
  "z",
  "x",
  "v",
  "g",
] as const;

const COMPATIBILITY_JAMO_KEYS: Readonly<Record<string, string>> = {
  ㄱ: "r",
  ㄲ: "R",
  ㄳ: "rt",
  ㄴ: "s",
  ㄵ: "sw",
  ㄶ: "sg",
  ㄷ: "e",
  ㄸ: "E",
  ㄹ: "f",
  ㄺ: "fr",
  ㄻ: "fa",
  ㄼ: "fq",
  ㄽ: "ft",
  ㄾ: "fx",
  ㄿ: "fv",
  ㅀ: "fg",
  ㅁ: "a",
  ㅂ: "q",
  ㅃ: "Q",
  ㅄ: "qt",
  ㅅ: "t",
  ㅆ: "T",
  ㅇ: "d",
  ㅈ: "w",
  ㅉ: "W",
  ㅊ: "c",
  ㅋ: "z",
  ㅌ: "x",
  ㅍ: "v",
  ㅎ: "g",
  ㅏ: "k",
  ㅐ: "o",
  ㅑ: "i",
  ㅒ: "O",
  ㅓ: "j",
  ㅔ: "p",
  ㅕ: "u",
  ㅖ: "P",
  ㅗ: "h",
  ㅘ: "hk",
  ㅙ: "ho",
  ㅚ: "hl",
  ㅛ: "y",
  ㅜ: "n",
  ㅝ: "nj",
  ㅞ: "np",
  ㅟ: "nl",
  ㅠ: "b",
  ㅡ: "m",
  ㅢ: "ml",
  ㅣ: "l",
};

function twoSetKeys(character: string): string {
  if (/^[A-Za-z0-9]$/.test(character)) {
    return character.toUpperCase();
  }

  const compatibilityKeys = COMPATIBILITY_JAMO_KEYS[character];
  if (compatibilityKeys) return compatibilityKeys.toUpperCase();

  const codePoint = character.codePointAt(0);
  if (codePoint === undefined) return "";

  if (codePoint >= 0xac00 && codePoint <= 0xd7a3) {
    const syllableIndex = codePoint - 0xac00;
    const initialIndex = Math.floor(syllableIndex / 588);
    const medialIndex = Math.floor((syllableIndex % 588) / 28);
    const finalIndex = syllableIndex % 28;
    return `${INITIAL_KEYS[initialIndex]}${MEDIAL_KEYS[medialIndex]}${
      FINAL_KEYS[finalIndex]
    }`.toUpperCase();
  }

  if (codePoint >= 0x1100 && codePoint <= 0x1112) {
    return INITIAL_KEYS[codePoint - 0x1100].toUpperCase();
  }
  if (codePoint >= 0x1161 && codePoint <= 0x1175) {
    return MEDIAL_KEYS[codePoint - 0x1161].toUpperCase();
  }
  if (codePoint >= 0x11a8 && codePoint <= 0x11c2) {
    return FINAL_KEYS[codePoint - 0x11a7].toUpperCase();
  }

  return "";
}

export function normalizeRoomCodeInput(value: string): string {
  let normalized = "";
  for (const character of value.normalize("NFC")) {
    normalized += twoSetKeys(character);
    if (normalized.length >= ROOM_CODE_LENGTH) break;
  }
  return normalized.replace(/[^A-Z0-9]/g, "").slice(0, ROOM_CODE_LENGTH);
}

export function roomCodeCharacterFromPhysicalKey(
  code: string,
): string | null {
  const letter = /^Key([A-Z])$/.exec(code);
  if (letter) return letter[1];

  const digit = /^(?:Digit|Numpad)([0-9])$/.exec(code);
  return digit?.[1] ?? null;
}

export function applyPhysicalRoomCodeKey(
  value: string,
  selectionStart: number,
  selectionEnd: number,
  code: string,
): { value: string; caret: number } | null {
  const character = roomCodeCharacterFromPhysicalKey(code);
  if (!character) return null;

  const normalized = normalizeRoomCodeInput(value);
  const start = Math.max(0, Math.min(selectionStart, normalized.length));
  const end = Math.max(start, Math.min(selectionEnd, normalized.length));
  const nextValue = normalizeRoomCodeInput(
    `${normalized.slice(0, start)}${character}${normalized.slice(end)}`,
  );

  return {
    value: nextValue,
    caret: Math.min(start + 1, nextValue.length),
  };
}
