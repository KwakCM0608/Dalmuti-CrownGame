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
