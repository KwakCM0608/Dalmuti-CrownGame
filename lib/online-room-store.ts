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
      SELECT room_code, player_id, nickname, created_at, last_seen_at
      FROM online_room_members
      WHERE room_code = ? AND token_hash = ?
      LIMIT 1
    `).bind(code, tokenHash).first<MemberRow>();

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
