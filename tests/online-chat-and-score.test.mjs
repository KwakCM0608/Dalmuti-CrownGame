import assert from "node:assert/strict";
import test from "node:test";

const {
  ONLINE_CHAT_MAX_LENGTH,
  OnlineChatValidationError,
  sanitizeOnlineChatText,
} = await import(new URL("../lib/online-chat.ts", import.meta.url));
const { scoreChipCount } = await import(
  new URL("../lib/score-chips.ts", import.meta.url)
);
const { roundChipAward } = await import(
  new URL("../lib/round-score.ts", import.meta.url)
);
const {
  ONLINE_EMOTES,
  ONLINE_EMOTE_DURATION_MS,
  isOnlineEmoteId,
  onlineEmoteById,
} = await import(new URL("../lib/online-emotes.ts", import.meta.url));
const {
  applyPhysicalRoomCodeKey,
  normalizeRoomCodeInput,
  roomCodeCharacterFromPhysicalKey,
} = await import(new URL("../lib/room-code-input.ts", import.meta.url));

test("online chat normalizes whitespace and strips invisible controls", () => {
  assert.equal(
    sanitizeOnlineChatText("  안녕\n\t\u200b 달무티  "),
    "안녕 달무티",
  );
  assert.equal(sanitizeOnlineChatText("e\u0301"), "é");
});

test("online chat rejects empty and oversized messages by Unicode length", () => {
  assert.throws(
    () => sanitizeOnlineChatText("\n\t\u200b"),
    (error) =>
      error instanceof OnlineChatValidationError &&
      error.code === "INVALID_CHAT_MESSAGE",
  );
  assert.equal(
    sanitizeOnlineChatText("🙂".repeat(ONLINE_CHAT_MAX_LENGTH)),
    "🙂".repeat(ONLINE_CHAT_MAX_LENGTH),
  );
  assert.throws(
    () =>
      sanitizeOnlineChatText(
        "🙂".repeat(ONLINE_CHAT_MAX_LENGTH + 1),
      ),
    (error) =>
      error instanceof OnlineChatValidationError &&
      error.code === "CHAT_MESSAGE_TOO_LONG",
  );
});

test("score chips stay compact while preserving relative score comparison", () => {
  assert.equal(scoreChipCount(0, 12), 0);
  assert.equal(scoreChipCount(3, 12), 2);
  assert.equal(scoreChipCount(6, 12), 3);
  assert.equal(scoreChipCount(12, 12), 5);
  assert.equal(scoreChipCount(100, 100, 4), 4);
});

test("each act awards the same fixed chip curve for 4 to 10 players", () => {
  assert.deepEqual(
    Array.from({ length: 4 }, (_, index) => roundChipAward(index + 1, 4)),
    [4, 3, 1, 0],
  );
  assert.deepEqual(
    Array.from({ length: 5 }, (_, index) => roundChipAward(index + 1, 5)),
    [4, 3, 2, 1, 0],
  );
  assert.deepEqual(
    Array.from({ length: 10 }, (_, index) => roundChipAward(index + 1, 10)),
    [4, 3, 2, 2, 2, 2, 2, 2, 1, 0],
  );
  assert.throws(() => roundChipAward(1, 3), RangeError);
  assert.throws(() => roundChipAward(11, 10), RangeError);
});

test("online emotes use a fixed safe whitelist and a short display window", () => {
  assert.equal(ONLINE_EMOTES.length, 8);
  assert.equal(ONLINE_EMOTE_DURATION_MS, 4_200);
  assert.equal(isOnlineEmoteId("celebrate"), true);
  assert.equal(isOnlineEmoteId("<img src=x onerror=alert(1)>"), false);
  assert.deepEqual(onlineEmoteById("clap"), {
    id: "clap",
    emoji: "👏",
    label: "박수",
  });
  assert.equal(onlineEmoteById("custom"), null);
});

test("room codes accept only six uppercase ASCII letters or numbers", () => {
  assert.equal(normalizeRoomCodeInput("abc-123z"), "ABC123");
  assert.equal(normalizeRoomCodeInput(" a!b@9# "), "AB9");
  assert.equal(normalizeRoomCodeInput("ＡＢＣ１２３"), "");
});

test("Korean room-code input becomes the matching two-set keyboard keys", () => {
  assert.equal(normalizeRoomCodeInput("한글"), "GKSRMF");
  assert.equal(normalizeRoomCodeInput("ㅂㅈㄷㄱㅅㅛ"), "QWERTY");
  assert.equal(normalizeRoomCodeInput("가A1"), "RKA1");
});

test("physical room-code keys replace the current selection without IME text", () => {
  assert.equal(roomCodeCharacterFromPhysicalKey("KeyQ"), "Q");
  assert.equal(roomCodeCharacterFromPhysicalKey("Digit7"), "7");
  assert.equal(roomCodeCharacterFromPhysicalKey("Numpad2"), "2");
  assert.equal(roomCodeCharacterFromPhysicalKey("Minus"), null);
  assert.deepEqual(applyPhysicalRoomCodeKey("ABCD12", 2, 5, "KeyQ"), {
    value: "ABQ2",
    caret: 3,
  });
  assert.deepEqual(applyPhysicalRoomCodeKey("ABC123", 6, 6, "KeyZ"), {
    value: "ABC123",
    caret: 6,
  });
});
