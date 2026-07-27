export const ONLINE_CHAT_MAX_LENGTH = 100;
export const ONLINE_CHAT_HISTORY_LIMIT = 80;
export const ONLINE_CHAT_PAGE_SIZE = 50;
export const ONLINE_CHAT_COOLDOWN_MS = 800;

export type OnlineChatMessage = {
  seq: number;
  id: string;
  roomCode: string;
  playerId: string;
  authorName: string;
  text: string;
  sentAt: number;
};

export class OnlineChatValidationError extends Error {
  readonly code: "INVALID_CHAT_MESSAGE" | "CHAT_MESSAGE_TOO_LONG";

  constructor(
    code: "INVALID_CHAT_MESSAGE" | "CHAT_MESSAGE_TOO_LONG",
    message: string,
  ) {
    super(message);
    this.name = "OnlineChatValidationError";
    this.code = code;
  }
}

export function sanitizeOnlineChatText(input: unknown): string {
  if (typeof input !== "string") {
    throw new OnlineChatValidationError(
      "INVALID_CHAT_MESSAGE",
      "채팅 내용을 입력해 주세요.",
    );
  }

  const normalized = input
    .normalize("NFC")
    .replace(/[\p{Cc}\p{Cf}]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const length = Array.from(normalized).length;

  if (length < 1) {
    throw new OnlineChatValidationError(
      "INVALID_CHAT_MESSAGE",
      "채팅 내용을 입력해 주세요.",
    );
  }
  if (length > ONLINE_CHAT_MAX_LENGTH) {
    throw new OnlineChatValidationError(
      "CHAT_MESSAGE_TOO_LONG",
      `채팅은 ${ONLINE_CHAT_MAX_LENGTH}자까지 입력할 수 있습니다.`,
    );
  }
  return normalized;
}
