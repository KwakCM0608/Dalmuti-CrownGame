import {
  ONLINE_CHAT_COOLDOWN_MS,
  ONLINE_CHAT_HISTORY_LIMIT,
  ONLINE_CHAT_PAGE_SIZE,
  OnlineChatValidationError,
  sanitizeOnlineChatText,
  type OnlineChatMessage,
} from "./online-chat";
import {
  ONLINE_EMOTE_COOLDOWN_MS,
  ONLINE_EMOTE_DURATION_MS,
  isOnlineEmoteId,
  type OnlineRoomEmote,
} from "./online-emotes";

const ROOM_CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ";
const ROOM_CODE_LENGTH = 6;
const TOKEN_BYTES = 32;
const ROOM_TTL_MS = 24 * 60 * 60 * 1_000;
const MAX_WRITE_ATTEMPTS = 5;
const LAST_SEEN_WRITE_INTERVAL_MS = 10_000;

type D1Row = Record<string, unknown>;

export class OnlineStoreError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status = 500,
    public readonly retryable = false,
  ) {
    super(message);
    this.name = "OnlineStoreError";
  }
}

export interface StoredOnlineRoom<State = unknown> {
  code: string;
  state: State;
  revision: number;
  createdAt: number;
  updatedAt: number;
  expiresAt: number;
}

export interface OnlineRoomMember {
  roomCode: string;
  playerId: string;
  nickname: string;
  createdAt: number;
  lastSeenAt: number;
}

export interface IssuedOnlineSession extends OnlineRoomMember {
  token: string;
}

interface PendingIdentity {
  playerId: string;
  nickname: string;
  nicknameKey: string;
  token: string;
  tokenHash: string;
}

interface RoomRow extends D1Row {
  code: string;
  state_json: string;
  revision: number;
  created_at: number;
  updated_at: number;
  expires_at: number;
}

interface MemberRow extends D1Row {
  room_code: string;
  player_id: string;
  nickname: string;
  created_at: number;
  last_seen_at: number;
}

interface ChatMessageRow extends D1Row {
  seq: number;
  room_code: string;
  message_id: string;
  player_id: string;
  author_name: string;
  body: string;
  created_at: number;
}

interface EmoteRow extends D1Row {
  seq: number;
  room_code: string;
  request_id: string;
  player_id: string;
  emote_id: string;
  created_at: number;
  expires_at: number;
}

let schemaReady: Promise<void> | undefined;

async function getD1(): Promise<D1Database> {
  const { env } = await import("cloudflare:workers");
  const db = env.DB;
  if (!db) {
    throw new OnlineStoreError(
      "STORAGE_UNAVAILABLE",
      "온라인 방 저장소를 사용할 수 없습니다.",
      503,
      true,
    );
  }
  return db;
}

export async function ensureOnlineRoomSchema(): Promise<void> {
  if (!schemaReady) {
    schemaReady = initializeSchema().catch((error) => {
      schemaReady = undefined;
      throw error;
    });
  }
  return schemaReady;
}

async function initializeSchema(): Promise<void> {
  const db = await getD1();
  try {
    await db.batch([
      db.prepare(`
        CREATE TABLE IF NOT EXISTS online_rooms (
          code TEXT PRIMARY KEY NOT NULL,
          state_json TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          expires_at INTEGER NOT NULL
        )
      `),
      db.prepare(`
        CREATE TABLE IF NOT EXISTS online_room_members (
          room_code TEXT NOT NULL,
          player_id TEXT NOT NULL,
          nickname TEXT NOT NULL,
          nickname_key TEXT NOT NULL,
          token_hash TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          last_seen_at INTEGER NOT NULL,
          PRIMARY KEY (room_code, player_id),
          FOREIGN KEY (room_code) REFERENCES online_rooms(code) ON DELETE CASCADE
        )
      `),
      db.prepare(`
        CREATE TABLE IF NOT EXISTS online_room_chat_messages (
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          room_code TEXT NOT NULL,
          message_id TEXT NOT NULL,
          player_id TEXT NOT NULL,
          author_name TEXT NOT NULL,
          body TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          FOREIGN KEY (room_code) REFERENCES online_rooms(code) ON DELETE CASCADE
        )
      `),
      db.prepare(`
        CREATE TABLE IF NOT EXISTS online_room_emotes (
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          room_code TEXT NOT NULL,
          request_id TEXT NOT NULL,
          player_id TEXT NOT NULL,
          emote_id TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          expires_at INTEGER NOT NULL,
          FOREIGN KEY (room_code) REFERENCES online_rooms(code) ON DELETE CASCADE
        )
      `),
      db.prepare(
        "CREATE UNIQUE INDEX IF NOT EXISTS online_rooms_code_idx ON online_rooms(code)",
      ),
      db.prepare(
        "CREATE UNIQUE INDEX IF NOT EXISTS online_room_members_token_idx ON online_room_members(token_hash)",
      ),
      db.prepare(
        "CREATE UNIQUE INDEX IF NOT EXISTS online_room_members_nickname_idx ON online_room_members(room_code, nickname_key)",
      ),
      db.prepare(
        "CREATE INDEX IF NOT EXISTS online_rooms_expiry_idx ON online_rooms(expires_at)",
      ),
      db.prepare(
        "CREATE UNIQUE INDEX IF NOT EXISTS online_room_chat_message_id_idx ON online_room_chat_messages(room_code, message_id)",
      ),
      db.prepare(
        "CREATE INDEX IF NOT EXISTS online_room_chat_room_seq_idx ON online_room_chat_messages(room_code, seq)",
      ),
      db.prepare(
        "CREATE INDEX IF NOT EXISTS online_room_chat_player_time_idx ON online_room_chat_messages(room_code, player_id, created_at)",
      ),
      db.prepare(
        "CREATE UNIQUE INDEX IF NOT EXISTS online_room_emote_request_idx ON online_room_emotes(room_code, request_id)",
      ),
      db.prepare(
        "CREATE INDEX IF NOT EXISTS online_room_emote_player_time_idx ON online_room_emotes(room_code, player_id, created_at)",
      ),
      db.prepare(
        "CREATE INDEX IF NOT EXISTS online_room_emote_expiry_idx ON online_room_emotes(room_code, expires_at)",
      ),
    ]);
  } catch (error) {
    throw storageFailure(error);
  }
}

export function sanitizeNickname(input: unknown): {
  nickname: string;
  nicknameKey: string;
} {
  if (typeof input !== "string") {
    throw new OnlineStoreError(
      "INVALID_NICKNAME",
      "닉네임을 입력해 주세요.",
      400,
    );
  }

  const normalized = input
    .normalize("NFC")
    .replace(/[\p{Cc}\p{Cf}]/gu, "")
    .replace(/[^\p{L}\p{N} _-]/gu, "")
    .replace(/\s+/g, " ")
    .trim();
  const length = Array.from(normalized).length;

  if (length < 1 || length > 16) {
    throw new OnlineStoreError(
      "INVALID_NICKNAME",
      "닉네임은 1~16자로 입력해 주세요.",
      400,
    );
  }

  return {
    nickname: normalized,
    nicknameKey: normalized.toLocaleLowerCase("ko-KR"),
  };
}

export function normalizeRoomCode(input: string): string {
  const code = input.trim().toUpperCase();
  if (
    code.length !== ROOM_CODE_LENGTH ||
    Array.from(code).some((character) => !ROOM_CODE_ALPHABET.includes(character))
  ) {
    throw new OnlineStoreError(
      "INVALID_ROOM_CODE",
      "올바른 6자리 방 코드를 입력해 주세요.",
      400,
    );
  }
  return code;
}

export async function createStoredOnlineRoom<State>(
  nicknameInput: unknown,
  createState: (host: {
    roomCode: string;
    playerId: string;
    nickname: string;
  }) => State,
): Promise<{
  room: StoredOnlineRoom<State>;
  session: IssuedOnlineSession;
}> {
  await ensureOnlineRoomSchema();
  const db = await getD1();
  const identity = await issueIdentity(nicknameInput);

  for (let attempt = 0; attempt < 12; attempt += 1) {
    const code = randomRoomCode();
    const now = Date.now();
    const expiresAt = now + ROOM_TTL_MS;
    const state = createState({
      roomCode: code,
      playerId: identity.playerId,
      nickname: identity.nickname,
    });
    const stateJson = stringifyState(state);

    try {
      await db.batch([
        db.prepare(`
          INSERT INTO online_rooms
            (code, state_json, revision, created_at, updated_at, expires_at)
          VALUES (?, ?, 0, ?, ?, ?)
        `).bind(code, stateJson, now, now, expiresAt),
        db.prepare(`
          INSERT INTO online_room_members
            (room_code, player_id, nickname, nickname_key, token_hash,
             created_at, updated_at, last_seen_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `).bind(
          code,
          identity.playerId,
          identity.nickname,
          identity.nicknameKey,
          identity.tokenHash,
          now,
          now,
          now,
        ),
      ]);

      return {
        room: {
          code,
          state,
          revision: 0,
          createdAt: now,
          updatedAt: now,
          expiresAt,
        },
        session: {
          roomCode: code,
          playerId: identity.playerId,
          nickname: identity.nickname,
          token: identity.token,
          createdAt: now,
          lastSeenAt: now,
        },
      };
    } catch (error) {
      if (isRoomCodeCollision(error)) {
        continue;
      }
      throw storageFailure(error);
    }
  }

  throw new OnlineStoreError(
    "ROOM_CODE_EXHAUSTED",
    "방 코드를 만들지 못했습니다. 잠시 후 다시 시도해 주세요.",
    503,
    true,
  );
}

export async function reserveOnlineRoomMember(
  roomCodeInput: string,
  nicknameInput: unknown,
): Promise<IssuedOnlineSession> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  const room = await readStoredOnlineRoom(code);
  if (!room) {
    throw roomNotFound();
  }

  const identity = await issueIdentity(nicknameInput);
  const now = Date.now();

  try {
    await (await getD1()).prepare(`
      INSERT INTO online_room_members
        (room_code, player_id, nickname, nickname_key, token_hash,
         created_at, updated_at, last_seen_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      code,
      identity.playerId,
      identity.nickname,
      identity.nicknameKey,
      identity.tokenHash,
      now,
      now,
      now,
    ).run();
  } catch (error) {
    if (isNicknameCollision(error)) {
      throw new OnlineStoreError(
        "NICKNAME_TAKEN",
        "이미 사용 중인 닉네임입니다.",
        409,
      );
    }
    throw storageFailure(error);
  }

  return {
    roomCode: code,
    playerId: identity.playerId,
    nickname: identity.nickname,
    token: identity.token,
    createdAt: now,
    lastSeenAt: now,
  };
}

export async function removeOnlineRoomMember(
  roomCodeInput: string,
  playerId: string,
): Promise<void> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  try {
    await (await getD1())
      .prepare(
        "DELETE FROM online_room_members WHERE room_code = ? AND player_id = ?",
      )
      .bind(code, playerId)
      .run();
  } catch (error) {
    throw storageFailure(error);
  }
}

export async function deleteStoredOnlineRoom(
  roomCodeInput: string,
): Promise<void> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  try {
    const db = await getD1();
    const results = await db.batch([
      // Delete members explicitly instead of depending on a connection-level
      // SQLite foreign_keys setting. D1 batch statements commit atomically.
      db
        .prepare("DELETE FROM online_room_emotes WHERE room_code = ?")
        .bind(code),
      db
        .prepare("DELETE FROM online_room_chat_messages WHERE room_code = ?")
        .bind(code),
      db
        .prepare("DELETE FROM online_room_members WHERE room_code = ?")
        .bind(code),
      db
        .prepare("DELETE FROM online_rooms WHERE code = ?")
        .bind(code),
    ]);
    if (changes(results[3]) !== 1) {
      throw roomNotFound();
    }
  } catch (error) {
    if (error instanceof OnlineStoreError) {
      throw error;
    }
    throw storageFailure(error);
  }
}

export async function appendOnlineRoomChatMessage(
  roomCodeInput: string,
  member: OnlineRoomMember,
  messageIdInput: unknown,
  textInput: unknown,
  now = Date.now(),
): Promise<OnlineChatMessage> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  if (member.roomCode !== code) {
    throw unauthorized();
  }
  const messageId =
    typeof messageIdInput === "string" ? messageIdInput.trim() : "";
  if (
    !messageId ||
    messageId.length > 128 ||
    !/^[A-Za-z0-9._:-]+$/.test(messageId)
  ) {
    throw new OnlineStoreError(
      "INVALID_CHAT_MESSAGE_ID",
      "채팅 요청을 다시 보내 주세요.",
      400,
    );
  }
  if (!Number.isFinite(now) || now < 0) {
    throw new OnlineStoreError(
      "INVALID_CHAT_TIME",
      "채팅 시간을 확인할 수 없습니다.",
      400,
    );
  }

  let text: string;
  try {
    text = sanitizeOnlineChatText(textInput);
  } catch (error) {
    if (error instanceof OnlineChatValidationError) {
      throw new OnlineStoreError(error.code, error.message, 400);
    }
    throw error;
  }

  const db = await getD1();
  try {
    const existing = await db
      .prepare(`
        SELECT seq, room_code, message_id, player_id, author_name, body, created_at
        FROM online_room_chat_messages
        WHERE room_code = ? AND message_id = ?
        LIMIT 1
      `)
      .bind(code, messageId)
      .first<ChatMessageRow>();
    if (existing) {
      if (existing.player_id !== member.playerId) {
        throw new OnlineStoreError(
          "CHAT_MESSAGE_ID_CONFLICT",
          "채팅 요청이 충돌했습니다. 다시 보내 주세요.",
          409,
        );
      }
      return chatMessageFromRow(existing);
    }

    const insertResult = await db
      .prepare(`
        INSERT OR IGNORE INTO online_room_chat_messages
          (room_code, message_id, player_id, author_name, body, created_at)
        SELECT ?, ?, ?, ?, ?, ?
        WHERE EXISTS (
          SELECT 1
          FROM online_rooms AS rooms
          INNER JOIN online_room_members AS members
            ON members.room_code = rooms.code
          WHERE rooms.code = ?
            AND rooms.expires_at > ?
            AND members.player_id = ?
        )
        AND NOT EXISTS (
          SELECT 1
          FROM online_room_chat_messages
          WHERE room_code = ?
            AND player_id = ?
            AND created_at > ?
        )
      `)
      .bind(
        code,
        messageId,
        member.playerId,
        member.nickname,
        text,
        now,
        code,
        now,
        member.playerId,
        code,
        member.playerId,
        now - ONLINE_CHAT_COOLDOWN_MS,
      )
      .run();

    if (changes(insertResult) !== 1) {
      const duplicate = await db
        .prepare(`
          SELECT seq, room_code, message_id, player_id, author_name, body, created_at
          FROM online_room_chat_messages
          WHERE room_code = ? AND message_id = ?
          LIMIT 1
        `)
        .bind(code, messageId)
        .first<ChatMessageRow>();
      if (duplicate) {
        if (duplicate.player_id === member.playerId) {
          return chatMessageFromRow(duplicate);
        }
        throw new OnlineStoreError(
          "CHAT_MESSAGE_ID_CONFLICT",
          "채팅 요청이 충돌했습니다. 다시 보내 주세요.",
          409,
        );
      }
      const activeMembership = await db
        .prepare(`
          SELECT 1 AS active
          FROM online_rooms AS rooms
          INNER JOIN online_room_members AS members
            ON members.room_code = rooms.code
          WHERE rooms.code = ?
            AND rooms.expires_at > ?
            AND members.player_id = ?
          LIMIT 1
        `)
        .bind(code, now, member.playerId)
        .first<{ active: number }>();
      if (!activeMembership) {
        throw unauthorized();
      }
      throw new OnlineStoreError(
        "CHAT_RATE_LIMIT",
        "채팅은 잠시 후 다시 보낼 수 있습니다.",
        429,
        true,
      );
    }

    await db
      .prepare(`
        DELETE FROM online_room_chat_messages
        WHERE room_code = ?
          AND seq NOT IN (
            SELECT seq
            FROM online_room_chat_messages
            WHERE room_code = ?
            ORDER BY seq DESC
            LIMIT ?
          )
      `)
      .bind(code, code, ONLINE_CHAT_HISTORY_LIMIT)
      .run();

    const inserted = await db
      .prepare(`
        SELECT seq, room_code, message_id, player_id, author_name, body, created_at
        FROM online_room_chat_messages
        WHERE room_code = ? AND message_id = ?
        LIMIT 1
      `)
      .bind(code, messageId)
      .first<ChatMessageRow>();
    if (!inserted) {
      throw new OnlineStoreError(
        "CHAT_WRITE_FAILED",
        "채팅을 보내지 못했습니다. 다시 시도해 주세요.",
        503,
        true,
      );
    }
    return chatMessageFromRow(inserted);
  } catch (error) {
    if (error instanceof OnlineStoreError) throw error;
    throw storageFailure(error);
  }
}

export async function readOnlineRoomChatMessages(
  roomCodeInput: string,
  sinceSequence = 0,
  limit = ONLINE_CHAT_PAGE_SIZE,
): Promise<{ messages: OnlineChatMessage[]; latestSequence: number }> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  if (!Number.isSafeInteger(sinceSequence) || sinceSequence < 0) {
    throw new OnlineStoreError(
      "INVALID_CHAT_SEQUENCE",
      "채팅 기준값을 확인해 주세요.",
      400,
    );
  }
  const safeLimit = Math.max(1, Math.min(ONLINE_CHAT_PAGE_SIZE, limit));
  try {
    const db = await getD1();
    const query =
      sinceSequence === 0
        ? db
            .prepare(`
              SELECT *
              FROM (
                SELECT seq, room_code, message_id, player_id, author_name, body, created_at
                FROM online_room_chat_messages
                WHERE room_code = ?
                ORDER BY seq DESC
                LIMIT ?
              )
              ORDER BY seq ASC
            `)
            .bind(code, safeLimit)
        : db
            .prepare(`
              SELECT seq, room_code, message_id, player_id, author_name, body, created_at
              FROM online_room_chat_messages
              WHERE room_code = ? AND seq > ?
              ORDER BY seq ASC
              LIMIT ?
            `)
            .bind(code, sinceSequence, safeLimit);
    const result = await query.all<ChatMessageRow>();
    const messages = (result.results ?? []).map(chatMessageFromRow);
    return {
      messages,
      latestSequence: messages.at(-1)?.seq ?? sinceSequence,
    };
  } catch (error) {
    throw storageFailure(error);
  }
}

export async function appendOnlineRoomEmote(
  roomCodeInput: string,
  member: OnlineRoomMember,
  requestIdInput: unknown,
  emoteIdInput: unknown,
  now = Date.now(),
): Promise<OnlineRoomEmote> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  if (member.roomCode !== code) {
    throw unauthorized();
  }
  const requestId =
    typeof requestIdInput === "string" ? requestIdInput.trim() : "";
  if (
    !requestId ||
    requestId.length > 128 ||
    !/^[A-Za-z0-9._:-]+$/.test(requestId)
  ) {
    throw new OnlineStoreError(
      "INVALID_EMOTE_REQUEST_ID",
      "감정표현 요청을 다시 보내 주세요.",
      400,
    );
  }
  if (!isOnlineEmoteId(emoteIdInput)) {
    throw new OnlineStoreError(
      "INVALID_EMOTE",
      "지원하지 않는 감정표현입니다.",
      400,
    );
  }
  if (!Number.isFinite(now) || now < 0) {
    throw new OnlineStoreError(
      "INVALID_EMOTE_TIME",
      "감정표현 시간을 확인할 수 없습니다.",
      400,
    );
  }

  const db = await getD1();
  try {
    const existing = await db
      .prepare(`
        SELECT seq, room_code, request_id, player_id, emote_id, created_at, expires_at
        FROM online_room_emotes
        WHERE room_code = ? AND request_id = ?
        LIMIT 1
      `)
      .bind(code, requestId)
      .first<EmoteRow>();
    if (existing) {
      if (existing.player_id !== member.playerId) {
        throw new OnlineStoreError(
          "EMOTE_REQUEST_ID_CONFLICT",
          "감정표현 요청이 충돌했습니다. 다시 선택해 주세요.",
          409,
        );
      }
      return emoteFromRow(existing);
    }

    const expiresAt = now + ONLINE_EMOTE_DURATION_MS;
    const insertResult = await db
      .prepare(`
        INSERT OR IGNORE INTO online_room_emotes
          (room_code, request_id, player_id, emote_id, created_at, expires_at)
        SELECT ?, ?, ?, ?, ?, ?
        WHERE EXISTS (
          SELECT 1
          FROM online_rooms AS rooms
          INNER JOIN online_room_members AS members
            ON members.room_code = rooms.code
          WHERE rooms.code = ?
            AND rooms.expires_at > ?
            AND members.player_id = ?
        )
        AND NOT EXISTS (
          SELECT 1
          FROM online_room_emotes
          WHERE room_code = ?
            AND player_id = ?
            AND created_at > ?
        )
      `)
      .bind(
        code,
        requestId,
        member.playerId,
        emoteIdInput,
        now,
        expiresAt,
        code,
        now,
        member.playerId,
        code,
        member.playerId,
        now - ONLINE_EMOTE_COOLDOWN_MS,
      )
      .run();

    if (changes(insertResult) !== 1) {
      const duplicate = await db
        .prepare(`
          SELECT seq, room_code, request_id, player_id, emote_id, created_at, expires_at
          FROM online_room_emotes
          WHERE room_code = ? AND request_id = ?
          LIMIT 1
        `)
        .bind(code, requestId)
        .first<EmoteRow>();
      if (duplicate) {
        if (duplicate.player_id === member.playerId) {
          return emoteFromRow(duplicate);
        }
        throw new OnlineStoreError(
          "EMOTE_REQUEST_ID_CONFLICT",
          "감정표현 요청이 충돌했습니다. 다시 선택해 주세요.",
          409,
        );
      }
      const activeMembership = await db
        .prepare(`
          SELECT 1 AS active
          FROM online_rooms AS rooms
          INNER JOIN online_room_members AS members
            ON members.room_code = rooms.code
          WHERE rooms.code = ?
            AND rooms.expires_at > ?
            AND members.player_id = ?
          LIMIT 1
        `)
        .bind(code, now, member.playerId)
        .first<{ active: number }>();
      if (!activeMembership) {
        throw unauthorized();
      }
      throw new OnlineStoreError(
        "EMOTE_RATE_LIMIT",
        "감정표현은 잠시 후 다시 사용할 수 있습니다.",
        429,
        true,
      );
    }

    await db
      .prepare(`
        DELETE FROM online_room_emotes
        WHERE room_code = ?
          AND (
            expires_at <= ?
            OR seq NOT IN (
              SELECT seq
              FROM online_room_emotes
              WHERE room_code = ?
              ORDER BY seq DESC
              LIMIT 40
            )
          )
      `)
      .bind(code, now, code)
      .run();

    const inserted = await db
      .prepare(`
        SELECT seq, room_code, request_id, player_id, emote_id, created_at, expires_at
        FROM online_room_emotes
        WHERE room_code = ? AND request_id = ?
        LIMIT 1
      `)
      .bind(code, requestId)
      .first<EmoteRow>();
    if (!inserted) {
      throw new OnlineStoreError(
        "EMOTE_WRITE_FAILED",
        "감정표현을 보내지 못했습니다. 다시 시도해 주세요.",
        503,
        true,
      );
    }
    return emoteFromRow(inserted);
  } catch (error) {
    if (error instanceof OnlineStoreError) throw error;
    throw storageFailure(error);
  }
}

export async function readOnlineRoomEmotes(
  roomCodeInput: string,
  now = Date.now(),
): Promise<OnlineRoomEmote[]> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  if (!Number.isFinite(now) || now < 0) {
    throw new OnlineStoreError(
      "INVALID_EMOTE_TIME",
      "감정표현 시간을 확인할 수 없습니다.",
      400,
    );
  }
  try {
    const result = await (await getD1())
      .prepare(`
        SELECT
          emotes.seq,
          emotes.room_code,
          emotes.request_id,
          emotes.player_id,
          emotes.emote_id,
          emotes.created_at,
          emotes.expires_at
        FROM online_room_emotes AS emotes
        WHERE emotes.room_code = ?
          AND emotes.expires_at > ?
          AND emotes.seq = (
            SELECT MAX(latest.seq)
            FROM online_room_emotes AS latest
            WHERE latest.room_code = emotes.room_code
              AND latest.player_id = emotes.player_id
              AND latest.expires_at > ?
          )
        ORDER BY emotes.seq ASC
      `)
      .bind(code, now, now)
      .all<EmoteRow>();
    return (result.results ?? []).map(emoteFromRow);
  } catch (error) {
    throw storageFailure(error);
  }
}

export async function authenticateOnlineRoomRequest(
  request: Request,
  roomCodeInput: string,
): Promise<OnlineRoomMember> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  const authorization = request.headers.get("authorization") ?? "";
  const match = /^Bearer ([A-Za-z0-9_-]{32,256})$/.exec(authorization);
  if (!match) {
    throw unauthorized();
  }

  const tokenHash = await sha256Hex(match[1]);
  const now = Date.now();

  try {
    const row = await (await getD1()).prepare(`
      SELECT
        members.room_code,
        members.player_id,
        members.nickname,
        members.created_at,
        members.last_seen_at
      FROM online_room_members AS members
      INNER JOIN online_rooms AS rooms
        ON rooms.code = members.room_code
      WHERE members.room_code = ?
        AND members.token_hash = ?
        AND rooms.expires_at > ?
      LIMIT 1
    `).bind(code, tokenHash, now).first<MemberRow>();

    if (!row) {
      throw unauthorized();
    }

    if (now - numberValue(row.last_seen_at) >= LAST_SEEN_WRITE_INTERVAL_MS) {
      await (await getD1())
        .prepare(`
          UPDATE online_room_members
          SET last_seen_at = ?, updated_at = ?
          WHERE room_code = ? AND player_id = ?
        `)
        .bind(now, now, code, row.player_id)
        .run();
    }

    return {
      roomCode: row.room_code,
      playerId: row.player_id,
      nickname: row.nickname,
      createdAt: numberValue(row.created_at),
      lastSeenAt: now,
    };
  } catch (error) {
    if (error instanceof OnlineStoreError) {
      throw error;
    }
    throw storageFailure(error);
  }
}

export async function readStoredOnlineRoom<State = unknown>(
  roomCodeInput: string,
): Promise<StoredOnlineRoom<State> | null> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);
  const now = Date.now();

  try {
    const row = await (await getD1()).prepare(`
      SELECT code, state_json, revision, created_at, updated_at, expires_at
      FROM online_rooms
      WHERE code = ? AND expires_at > ?
      LIMIT 1
    `).bind(code, now).first<RoomRow>();

    return row ? roomFromRow<State>(row) : null;
  } catch (error) {
    throw storageFailure(error);
  }
}

export async function mutateStoredOnlineRoom<State>(
  roomCodeInput: string,
  updater: (
    state: State,
    room: StoredOnlineRoom<State>,
  ) => State | Promise<State>,
  options: { expectedRevision?: number } = {},
): Promise<StoredOnlineRoom<State>> {
  await ensureOnlineRoomSchema();
  const code = normalizeRoomCode(roomCodeInput);

  for (let attempt = 0; attempt < MAX_WRITE_ATTEMPTS; attempt += 1) {
    const current = await readStoredOnlineRoom<State>(code);
    if (!current) {
      throw roomNotFound();
    }
    if (
      options.expectedRevision !== undefined &&
      current.revision !== options.expectedRevision
    ) {
      throw new OnlineStoreError(
        "REVISION_CONFLICT",
        "게임 상태가 갱신되었습니다. 최신 상태를 다시 불러와 주세요.",
        409,
        true,
      );
    }

    const nextState = await updater(current.state, current);
    const nextStateJson = stringifyState(nextState);
    const currentStateJson = stringifyState(current.state);
    if (nextStateJson === currentStateJson) {
      return current;
    }

    const now = Date.now();
    const expiresAt = now + ROOM_TTL_MS;

    try {
      const result = await (await getD1()).prepare(`
        UPDATE online_rooms
        SET state_json = ?,
            revision = revision + 1,
            updated_at = ?,
            expires_at = ?
        WHERE code = ? AND revision = ? AND expires_at > ?
      `).bind(
        nextStateJson,
        now,
        expiresAt,
        code,
        current.revision,
        now,
      ).run();

      if (changes(result) === 1) {
        return {
          code,
          state: nextState,
          revision: current.revision + 1,
          createdAt: current.createdAt,
          updatedAt: now,
          expiresAt,
        };
      }
    } catch (error) {
      throw storageFailure(error);
    }
  }

  throw new OnlineStoreError(
    "WRITE_CONFLICT",
    "다른 플레이어의 요청과 겹쳤습니다. 다시 시도해 주세요.",
    409,
    true,
  );
}

export function onlineStoreErrorResponse(error: unknown): Response {
  const storeError =
    error instanceof OnlineStoreError ? error : storageFailure(error);
  return Response.json(
    {
      error: {
        code: storeError.code,
        message: storeError.message,
        ...(storeError.retryable ? { retryable: true } : {}),
      },
    },
    {
      status: storeError.status,
      headers: {
        "cache-control": "no-store",
      },
    },
  );
}

function roomFromRow<State>(row: RoomRow): StoredOnlineRoom<State> {
  try {
    return {
      code: row.code,
      state: JSON.parse(row.state_json) as State,
      revision: numberValue(row.revision),
      createdAt: numberValue(row.created_at),
      updatedAt: numberValue(row.updated_at),
      expiresAt: numberValue(row.expires_at),
    };
  } catch {
    throw new OnlineStoreError(
      "CORRUPT_ROOM_STATE",
      "저장된 게임 상태를 읽을 수 없습니다.",
      500,
    );
  }
}

function chatMessageFromRow(row: ChatMessageRow): OnlineChatMessage {
  return {
    seq: numberValue(row.seq),
    id: row.message_id,
    roomCode: row.room_code,
    playerId: row.player_id,
    authorName: row.author_name,
    text: row.body,
    sentAt: numberValue(row.created_at),
  };
}

function emoteFromRow(row: EmoteRow): OnlineRoomEmote {
  if (!isOnlineEmoteId(row.emote_id)) {
    throw new OnlineStoreError(
      "CORRUPT_EMOTE",
      "저장된 감정표현을 읽을 수 없습니다.",
      500,
    );
  }
  return {
    seq: numberValue(row.seq),
    id: row.request_id,
    roomCode: row.room_code,
    playerId: row.player_id,
    emoteId: row.emote_id,
    createdAt: numberValue(row.created_at),
    expiresAt: numberValue(row.expires_at),
  };
}

async function issueIdentity(nicknameInput: unknown): Promise<PendingIdentity> {
  const { nickname, nicknameKey } = sanitizeNickname(nicknameInput);
  const token = randomToken();
  return {
    playerId: crypto.randomUUID(),
    nickname,
    nicknameKey,
    token,
    tokenHash: await sha256Hex(token),
  };
}

function randomRoomCode(): string {
  let result = "";
  const limit =
    256 - (256 % ROOM_CODE_ALPHABET.length);

  while (result.length < ROOM_CODE_LENGTH) {
    const bytes = crypto.getRandomValues(new Uint8Array(ROOM_CODE_LENGTH * 2));
    for (const byte of bytes) {
      if (byte >= limit) {
        continue;
      }
      result += ROOM_CODE_ALPHABET[byte % ROOM_CODE_ALPHABET.length];
      if (result.length === ROOM_CODE_LENGTH) {
        break;
      }
    }
  }
  return result;
}

function randomToken(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(TOKEN_BYTES));
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

function stringifyState(value: unknown): string {
  try {
    const json = JSON.stringify(value);
    if (json === undefined) {
      throw new Error("State is not JSON serializable.");
    }
    return json;
  } catch {
    throw new OnlineStoreError(
      "INVALID_ROOM_STATE",
      "게임 상태를 저장할 수 없습니다.",
      500,
    );
  }
}

function numberValue(value: unknown): number {
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(number)) {
    throw new OnlineStoreError(
      "INVALID_STORAGE_VALUE",
      "저장된 값을 읽을 수 없습니다.",
      500,
    );
  }
  return number;
}

function changes(result: D1Result<unknown>): number {
  const value = result.meta?.changes;
  return typeof value === "number" ? value : Number(value ?? 0);
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    const cause =
      error.cause instanceof Error ? ` ${error.cause.message}` : "";
    return `${error.message}${cause}`;
  }
  return String(error);
}

function isRoomCodeCollision(error: unknown): boolean {
  const message = errorMessage(error);
  return (
    message.includes("UNIQUE constraint failed") &&
    message.includes("online_rooms.code")
  );
}

function isNicknameCollision(error: unknown): boolean {
  const message = errorMessage(error);
  return (
    message.includes("UNIQUE constraint failed") &&
    (message.includes("online_room_members.room_code") ||
      message.includes("online_room_members_nickname_idx"))
  );
}

function roomNotFound(): OnlineStoreError {
  return new OnlineStoreError(
    "ROOM_NOT_FOUND",
    "존재하지 않거나 만료된 방입니다.",
    404,
  );
}

function unauthorized(): OnlineStoreError {
  return new OnlineStoreError(
    "UNAUTHORIZED",
    "이 방에 다시 참가해 주세요.",
    401,
  );
}

function storageFailure(error: unknown): OnlineStoreError {
  if (error instanceof OnlineStoreError) {
    return error;
  }
  return new OnlineStoreError(
    "STORAGE_ERROR",
    "온라인 방을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.",
    503,
    true,
  );
}
