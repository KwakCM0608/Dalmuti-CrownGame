export const ONLINE_EMOTES = [
  { id: "thumbs-up", emoji: "👍", label: "좋아요" },
  { id: "laugh", emoji: "😄", label: "웃음" },
  { id: "surprised", emoji: "😮", label: "놀람" },
  { id: "cry", emoji: "😭", label: "슬픔" },
  { id: "angry", emoji: "😡", label: "화남" },
  { id: "clap", emoji: "👏", label: "박수" },
  { id: "heart", emoji: "❤️", label: "하트" },
  { id: "celebrate", emoji: "🎉", label: "축하" },
  { id: "smirk", emoji: "😏", label: "아쉽네요" },
  { id: "sunglasses", emoji: "😎", label: "여유" },
  { id: "yawn", emoji: "🥱", label: "졸리네요" },
  { id: "shush", emoji: "🤫", label: "쉿" },
  { id: "tongue", emoji: "😜", label: "메롱" },
  { id: "wave", emoji: "👋", label: "잘 가요" },
  { id: "eyes", emoji: "👀", label: "다 보여요" },
  { id: "popcorn", emoji: "🍿", label: "구경 중" },
] as const;

export type OnlineEmoteId = (typeof ONLINE_EMOTES)[number]["id"];

export const ONLINE_EMOTE_DURATION_MS = 4_200;
export const ONLINE_EMOTE_COOLDOWN_MS = 1_200;

export type OnlineRoomEmote = {
  seq: number;
  id: string;
  roomCode: string;
  playerId: string;
  emoteId: OnlineEmoteId;
  createdAt: number;
  expiresAt: number;
};

export function isOnlineEmoteId(value: unknown): value is OnlineEmoteId {
  return (
    typeof value === "string" &&
    ONLINE_EMOTES.some((emote) => emote.id === value)
  );
}

export function onlineEmoteById(
  value: unknown,
): (typeof ONLINE_EMOTES)[number] | null {
  if (!isOnlineEmoteId(value)) return null;
  return ONLINE_EMOTES.find((emote) => emote.id === value) ?? null;
}
