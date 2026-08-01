export const V4_PUBLIC_OBSERVATION_SCHEMA_VERSION = 4;
export const V4_MIN_PLAYERS = 4;
export const V4_MAX_PLAYERS = 10;
export const V4_MAX_PUBLIC_HISTORY_EVENTS = 192;

export const V4_PUBLIC_ROLES = [
  "great-dalmuti",
  "lesser-dalmuti",
  "merchant",
  "lesser-peon",
  "great-peon",
] as const;

export const V4_PUBLIC_EVENT_TYPES = [
  "play",
  "pass",
  "clear",
  "finish",
] as const;

export const V4_PUBLIC_PASS_REASONS = [
  "manual",
  "timeout",
  "insufficient-cards",
  "dalmuti",
] as const;

export const V4_PUBLIC_CLEAR_REASONS = [
  "all-passed",
  "dalmuti",
  "act-ended",
] as const;

export const V4_HISTORY_TOKEN_FIELDS = [
  "sequence",
  "type",
  "actorOffset",
  "handCountBefore",
  "handCountAfter",
  "rank",
  "naturalCount",
  "jokerCount",
  "totalCount",
  "passReason",
  "clearReason",
  "nextLeaderOffset",
  "finishPlace",
] as const;

export const V4_MEMORY_TRACE_DECAYS = [0.5, 0.8, 0.95, 0.99] as const;

export const V4_MEMORY_TRACE_FEATURES = [
  "type.play",
  "type.pass",
  "type.clear",
  "type.finish",
  "actor-offset",
  "hand-count-before",
  "hand-count-after",
  "rank",
  "natural-count",
  "joker-count",
  "total-count",
  "pass.manual",
  "pass.timeout",
  "pass.insufficient-cards",
  "pass.dalmuti",
  "clear.all-passed",
  "clear.dalmuti",
  "clear.act-ended",
  "next-leader-offset",
  "finish-place",
] as const;

export type V4PublicRole = (typeof V4_PUBLIC_ROLES)[number];
export type V4PublicPassReason =
  (typeof V4_PUBLIC_PASS_REASONS)[number];
export type V4PublicClearReason =
  (typeof V4_PUBLIC_CLEAR_REASONS)[number];
export type V4RevolutionState =
  | null
  | "revolution"
  | "great-revolution";

export type V4PublicPlayerInput = {
  readonly id: string;
  readonly handCount: number;
  readonly finished: boolean;
  readonly passed: boolean;
  readonly role: V4PublicRole;
  readonly score: number;
};

export type V4PublicCardInput = {
  readonly rank: number;
};

export type V4PublicCardBundleInput = {
  readonly rank: number;
  readonly naturalCount: number;
  readonly jokerCount: number;
  readonly totalCount: number;
};

export type V4PublicTableInput = V4PublicCardBundleInput & {
  readonly playerId: string;
};

type V4PublicEventBase = {
  readonly sequence: number;
  readonly actorId: string;
  readonly handCountBefore: number;
  readonly handCountAfter: number;
};

export type V4PublicPlayEventInput = V4PublicEventBase &
  V4PublicCardBundleInput & {
    readonly type: "play";
  };

export type V4PublicPassEventInput = V4PublicEventBase & {
  readonly type: "pass";
  readonly reason: V4PublicPassReason;
};

export type V4PublicClearEventInput = V4PublicEventBase &
  V4PublicCardBundleInput & {
    readonly type: "clear";
    readonly reason: V4PublicClearReason;
    readonly nextLeaderId: string | null;
  };

export type V4PublicFinishEventInput = V4PublicEventBase & {
  readonly type: "finish";
  readonly place: number;
};

export type V4PublicEventInput =
  | V4PublicPlayEventInput
  | V4PublicPassEventInput
  | V4PublicClearEventInput
  | V4PublicFinishEventInput;

/**
 * The only accepted V4 actor boundary. `players` must be in current social
 * rank order. The actor's own physical cards are the only private cards this
 * object may contain. Unknown keys are rejected recursively instead of being
 * silently copied, so opponent hands and private taxation payloads cannot
 * travel through this interface.
 */
export type V4ActorVisibleObservationInput = {
  readonly actorId: string;
  readonly act: number;
  readonly revolution: V4RevolutionState;
  readonly ownHand: readonly V4PublicCardInput[];
  readonly publicPlayedCounts: readonly number[];
  readonly players: readonly V4PublicPlayerInput[];
  readonly table: V4PublicTableInput | null;
  readonly history: readonly V4PublicEventInput[];
};

export type V4PublicTableToken = {
  readonly actorOffset: number;
  readonly rank: number;
  readonly naturalCount: number;
  readonly jokerCount: number;
  readonly totalCount: number;
};

export type V4PublicPlayerToken = {
  readonly relativeOffset: number;
  readonly handCount: number;
  readonly finished: 0 | 1;
  readonly passed: 0 | 1;
  readonly self: 0 | 1;
  readonly tableLeader: 0 | 1;
  readonly role: number;
  readonly score: number;
};

export type V4PublicHistoryToken = {
  readonly sequence: number;
  readonly type: number;
  readonly actorOffset: number;
  readonly handCountBefore: number;
  readonly handCountAfter: number;
  readonly rank: number;
  readonly naturalCount: number;
  readonly jokerCount: number;
  readonly totalCount: number;
  /** 0 means not a pass; pass reasons are encoded as 1..4. */
  readonly passReason: number;
  /** 0 means not a clear; clear reasons are encoded as 1..3. */
  readonly clearReason: number;
  /** -1 means there is no next leader on this token. */
  readonly nextLeaderOffset: number;
  /** 0 means this is not a finish token. */
  readonly finishPlace: number;
};

export type V4ActorVisibleObservation = {
  readonly schemaVersion: typeof V4_PUBLIC_OBSERVATION_SCHEMA_VERSION;
  readonly playerCount: number;
  readonly act: number;
  readonly actorRole: number;
  readonly revolution: number;
  readonly ownHandCounts: readonly number[];
  readonly publicPlayedCounts: readonly number[];
  readonly table: V4PublicTableToken | null;
  readonly playerTokens: readonly V4PublicPlayerToken[];
  readonly historyTokens: readonly V4PublicHistoryToken[];
  /** Four chronological EMA vectors for the prefix older than 192 events. */
  readonly memoryTraceVectors: readonly (readonly number[])[];
  readonly truncatedHistoryCount: number;
};

type PlainRecord = Record<string, unknown>;

const PLAYER_KEYS = [
  "id",
  "handCount",
  "finished",
  "passed",
  "role",
  "score",
] as const;
const CARD_KEYS = ["rank"] as const;
const TABLE_KEYS = [
  "playerId",
  "rank",
  "naturalCount",
  "jokerCount",
  "totalCount",
] as const;
const TOP_LEVEL_KEYS = [
  "actorId",
  "act",
  "revolution",
  "ownHand",
  "publicPlayedCounts",
  "players",
  "table",
  "history",
] as const;
const EVENT_BASE_KEYS = [
  "type",
  "sequence",
  "actorId",
  "handCountBefore",
  "handCountAfter",
] as const;

function assertPlainRecord(value: unknown, label: string): PlainRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be a plain object`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(`${label} must be a plain object`);
  }
  return value as PlainRecord;
}

function assertExactKeys(
  record: PlainRecord,
  allowedKeys: readonly string[],
  label: string,
): void {
  const allowed = new Set(allowedKeys);
  const unknown = Object.keys(record).filter((key) => !allowed.has(key));
  if (unknown.length > 0 || Object.getOwnPropertySymbols(record).length > 0) {
    throw new TypeError(
      `${label} contains an unknown or private field: ${unknown[0] ?? "symbol"}`,
    );
  }
  const missing = allowedKeys.filter(
    (key) => !Object.prototype.hasOwnProperty.call(record, key),
  );
  if (missing.length > 0) {
    throw new TypeError(`${label} is missing required field ${missing[0]}`);
  }
}

function assertString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string`);
  }
}

function assertBoolean(
  value: unknown,
  label: string,
): asserts value is boolean {
  if (typeof value !== "boolean") {
    throw new TypeError(`${label} must be a boolean`);
  }
}

function assertIntegerInRange(
  value: unknown,
  minimum: number,
  maximum: number,
  label: string,
): asserts value is number {
  if (
    !Number.isSafeInteger(value) ||
    (value as number) < minimum ||
    (value as number) > maximum
  ) {
    throw new RangeError(
      `${label} must be an integer from ${minimum} to ${maximum}`,
    );
  }
}

function roleId(value: unknown, label: string): number {
  const index = V4_PUBLIC_ROLES.indexOf(value as V4PublicRole);
  if (index < 0) {
    throw new TypeError(`${label} must be a supported social role`);
  }
  return index;
}

function expectedRoleAt(index: number, playerCount: number): V4PublicRole {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === playerCount - 2) return "lesser-peon";
  if (index === playerCount - 1) return "great-peon";
  return "merchant";
}

function deckCopies(rank: number): number {
  return rank === 13 ? 2 : rank;
}

function validateCardBundle(
  record: PlainRecord,
  label: string,
): V4PublicCardBundleInput {
  assertIntegerInRange(record.rank, 1, 13, `${label}.rank`);
  assertIntegerInRange(
    record.naturalCount,
    0,
    12,
    `${label}.naturalCount`,
  );
  assertIntegerInRange(record.jokerCount, 0, 2, `${label}.jokerCount`);
  assertIntegerInRange(record.totalCount, 1, 14, `${label}.totalCount`);
  if (record.totalCount !== record.naturalCount + record.jokerCount) {
    throw new RangeError(
      `${label}.totalCount must equal naturalCount plus jokerCount`,
    );
  }
  if (record.rank === 13) {
    if (
      record.naturalCount !== 0 ||
      record.jokerCount !== 1 ||
      record.totalCount !== 1
    ) {
      throw new RangeError(`${label} rank 13 must be one solo joker`);
    }
  } else if (
    record.naturalCount < 1 ||
    record.naturalCount > record.rank
  ) {
    throw new RangeError(
      `${label}.naturalCount must be from 1 through the submitted rank`,
    );
  }
  return {
    rank: record.rank,
    naturalCount: record.naturalCount,
    jokerCount: record.jokerCount,
    totalCount: record.totalCount,
  };
}

function revolutionId(value: unknown): number {
  if (value === null) return 0;
  if (value === "revolution") return 1;
  if (value === "great-revolution") return 2;
  throw new TypeError(
    "observation.revolution must be null, revolution, or great-revolution",
  );
}

function passReasonId(value: unknown, label: string): number {
  const index = V4_PUBLIC_PASS_REASONS.indexOf(
    value as V4PublicPassReason,
  );
  if (index < 0) {
    throw new TypeError(`${label} must be a supported public pass reason`);
  }
  return index + 1;
}

function clearReasonId(value: unknown, label: string): number {
  const index = V4_PUBLIC_CLEAR_REASONS.indexOf(
    value as V4PublicClearReason,
  );
  if (index < 0) {
    throw new TypeError(`${label} must be a supported public clear reason`);
  }
  return index + 1;
}

function flag(value: boolean): 0 | 1 {
  return value ? 1 : 0;
}

function validatePlayers(
  value: unknown,
): readonly V4PublicPlayerInput[] {
  if (!Array.isArray(value)) {
    throw new TypeError("observation.players must be an array");
  }
  if (value.length < V4_MIN_PLAYERS || value.length > V4_MAX_PLAYERS) {
    throw new RangeError("V4 observations support 4 to 10 players");
  }
  const ids = new Set<string>();
  return value.map((entry, index) => {
    const record = assertPlainRecord(entry, `observation.players[${index}]`);
    assertExactKeys(record, PLAYER_KEYS, `observation.players[${index}]`);
    assertString(record.id, `observation.players[${index}].id`);
    if (ids.has(record.id)) {
      throw new TypeError(`observation.players repeats id ${record.id}`);
    }
    ids.add(record.id);
    assertIntegerInRange(
      record.handCount,
      0,
      80,
      `observation.players[${index}].handCount`,
    );
    assertBoolean(record.finished, `observation.players[${index}].finished`);
    assertBoolean(record.passed, `observation.players[${index}].passed`);
    const expectedRole = expectedRoleAt(index, value.length);
    if (record.role !== expectedRole) {
      throw new TypeError(
        `observation rank seat ${index} must be ${expectedRole}`,
      );
    }
    roleId(record.role, `observation.players[${index}].role`);
    assertIntegerInRange(
      record.score,
      0,
      1_000_000_000,
      `observation.players[${index}].score`,
    );
    if (record.finished !== (record.handCount === 0)) {
      throw new TypeError(
        `observation.players[${index}] finished must match a zero hand count`,
      );
    }
    return {
      id: record.id,
      handCount: record.handCount,
      finished: record.finished,
      passed: record.passed,
      role: expectedRole,
      score: record.score,
    };
  });
}

function validateOwnHand(value: unknown): readonly V4PublicCardInput[] {
  if (!Array.isArray(value)) {
    throw new TypeError("observation.ownHand must be an array");
  }
  if (value.length > 20) {
    throw new RangeError("observation.ownHand cannot exceed 20 cards");
  }
  return value.map((entry, index) => {
    const record = assertPlainRecord(entry, `observation.ownHand[${index}]`);
    assertExactKeys(record, CARD_KEYS, `observation.ownHand[${index}]`);
    assertIntegerInRange(
      record.rank,
      1,
      13,
      `observation.ownHand[${index}].rank`,
    );
    return { rank: record.rank };
  });
}

function validatePublicPlayedCounts(value: unknown): readonly number[] {
  if (!Array.isArray(value) || value.length !== 13) {
    throw new TypeError(
      "observation.publicPlayedCounts must contain exactly 13 ranks",
    );
  }
  return value.map((count, index) => {
    const rank = index + 1;
    assertIntegerInRange(
      count,
      0,
      deckCopies(rank),
      `observation.publicPlayedCounts[${index}]`,
    );
    return count;
  });
}

function relativeOffsets(
  players: readonly V4PublicPlayerInput[],
  actorId: string,
): ReadonlyMap<string, number> {
  const actorIndex = players.findIndex((player) => player.id === actorId);
  if (actorIndex < 0) {
    throw new TypeError("observation.players must include actorId");
  }
  return new Map(
    players.map((player, index) => [
      player.id,
      (index - actorIndex + players.length) % players.length,
    ]),
  );
}

function playerOffset(
  id: unknown,
  offsets: ReadonlyMap<string, number>,
  label: string,
): number {
  assertString(id, label);
  const offset = offsets.get(id);
  if (offset === undefined) {
    throw new TypeError(`${label} must identify a public player`);
  }
  return offset;
}

function validateTable(
  value: unknown,
  offsets: ReadonlyMap<string, number>,
): V4PublicTableToken | null {
  if (value === null) return null;
  const record = assertPlainRecord(value, "observation.table");
  assertExactKeys(record, TABLE_KEYS, "observation.table");
  const bundle = validateCardBundle(record, "observation.table");
  return {
    actorOffset: playerOffset(
      record.playerId,
      offsets,
      "observation.table.playerId",
    ),
    ...bundle,
  };
}

function validateEvent(
  value: unknown,
  index: number,
  playerCount: number,
  offsets: ReadonlyMap<string, number>,
): V4PublicHistoryToken {
  const label = `observation.history[${index}]`;
  const record = assertPlainRecord(value, label);
  const type = record.type;
  if (!V4_PUBLIC_EVENT_TYPES.includes(type as never)) {
    throw new TypeError(`${label}.type must be play, pass, clear, or finish`);
  }
  const extraKeys =
    type === "play"
      ? ["rank", "naturalCount", "jokerCount", "totalCount"]
      : type === "pass"
        ? ["reason"]
        : type === "clear"
          ? [
              "rank",
              "naturalCount",
              "jokerCount",
              "totalCount",
              "reason",
              "nextLeaderId",
            ]
          : ["place"];
  assertExactKeys(record, [...EVENT_BASE_KEYS, ...extraKeys], label);
  assertIntegerInRange(
    record.sequence,
    0,
    Number.MAX_SAFE_INTEGER,
    `${label}.sequence`,
  );
  const actorOffset = playerOffset(
    record.actorId,
    offsets,
    `${label}.actorId`,
  );
  assertIntegerInRange(
    record.handCountBefore,
    0,
    80,
    `${label}.handCountBefore`,
  );
  assertIntegerInRange(
    record.handCountAfter,
    0,
    80,
    `${label}.handCountAfter`,
  );

  const token: V4PublicHistoryToken = {
    sequence: record.sequence,
    type: V4_PUBLIC_EVENT_TYPES.indexOf(type as never),
    actorOffset,
    handCountBefore: record.handCountBefore,
    handCountAfter: record.handCountAfter,
    rank: 0,
    naturalCount: 0,
    jokerCount: 0,
    totalCount: 0,
    passReason: 0,
    clearReason: 0,
    nextLeaderOffset: -1,
    finishPlace: 0,
  };

  if (type === "play") {
    const bundle = validateCardBundle(record, label);
    if (record.handCountBefore - record.handCountAfter !== bundle.totalCount) {
      throw new RangeError(
        `${label} hand-count transition must equal totalCount`,
      );
    }
    return { ...token, ...bundle };
  }
  if (type === "pass") {
    if (record.handCountBefore !== record.handCountAfter) {
      throw new RangeError(`${label} pass cannot change hand count`);
    }
    return {
      ...token,
      passReason: passReasonId(record.reason, `${label}.reason`),
    };
  }
  if (type === "clear") {
    if (record.handCountBefore !== record.handCountAfter) {
      throw new RangeError(`${label} clear cannot change hand count`);
    }
    const bundle = validateCardBundle(record, label);
    const nextLeaderOffset =
      record.nextLeaderId === null
        ? -1
        : playerOffset(
            record.nextLeaderId,
            offsets,
            `${label}.nextLeaderId`,
          );
    return {
      ...token,
      ...bundle,
      clearReason: clearReasonId(record.reason, `${label}.reason`),
      nextLeaderOffset,
    };
  }

  if (record.handCountBefore !== 0 || record.handCountAfter !== 0) {
    throw new RangeError(`${label} finish must report a zero hand count`);
  }
  assertIntegerInRange(record.place, 1, playerCount, `${label}.place`);
  return { ...token, finishPlace: record.place };
}

function validateHistory(
  value: unknown,
  playerCount: number,
  offsets: ReadonlyMap<string, number>,
): readonly V4PublicHistoryToken[] {
  if (!Array.isArray(value)) {
    throw new TypeError("observation.history must be an array");
  }
  let previousSequence = -1;
  return value.map((event, index) => {
    const token = validateEvent(event, index, playerCount, offsets);
    if (token.sequence <= previousSequence) {
      throw new RangeError(
        "observation.history sequence values must be strictly increasing",
      );
    }
    previousSequence = token.sequence;
    return token;
  });
}

function eventTraceFeatures(
  playerCount: number,
  event: V4PublicHistoryToken,
): readonly number[] {
  const features = Array.from(
    { length: V4_MEMORY_TRACE_FEATURES.length },
    () => 0,
  );
  features[event.type] = 1;
  features[4] = playerCount <= 1 ? 0 : event.actorOffset / (playerCount - 1);
  features[5] = event.handCountBefore / 20;
  features[6] = event.handCountAfter / 20;
  features[7] = event.rank / 13;
  features[8] = event.naturalCount / 12;
  features[9] = event.jokerCount / 2;
  features[10] = event.totalCount / 14;
  if (event.passReason > 0) features[10 + event.passReason] = 1;
  if (event.clearReason > 0) features[14 + event.clearReason] = 1;
  features[18] =
    event.nextLeaderOffset < 0
      ? 0
      : (event.nextLeaderOffset + 1) / playerCount;
  features[19] = event.finishPlace / playerCount;
  return features;
}

function buildMemoryTraceVectors(
  playerCount: number,
  oldEvents: readonly V4PublicHistoryToken[],
): readonly (readonly number[])[] {
  return V4_MEMORY_TRACE_DECAYS.map((decay) => {
    const trace = Array.from(
      { length: V4_MEMORY_TRACE_FEATURES.length },
      () => 0,
    );
    for (const event of oldEvents) {
      const features = eventTraceFeatures(playerCount, event);
      for (let index = 0; index < trace.length; index += 1) {
        trace[index] = decay * trace[index] + (1 - decay) * features[index];
      }
    }
    return trace;
  });
}

/**
 * Validates and projects an input into an ID-free, integer-only actor record.
 * Player identifiers are used only to derive clockwise offsets and are never
 * retained in the result.
 */
export function buildV4ActorVisibleObservation(
  value: unknown,
): V4ActorVisibleObservation {
  const record = assertPlainRecord(value, "observation");
  assertExactKeys(record, TOP_LEVEL_KEYS, "observation");
  assertString(record.actorId, "observation.actorId");
  assertIntegerInRange(record.act, 1, 1_000_000, "observation.act");
  const revolution = revolutionId(record.revolution);
  const players = validatePlayers(record.players);
  const offsets = relativeOffsets(players, record.actorId);
  const actor = players.find((player) => player.id === record.actorId)!;
  const ownHand = validateOwnHand(record.ownHand);
  if (ownHand.length !== actor.handCount) {
    throw new RangeError(
      "observation.ownHand length must match the actor public hand count",
    );
  }
  const ownHandCounts = Array.from({ length: 13 }, () => 0);
  for (const card of ownHand) ownHandCounts[card.rank - 1] += 1;
  const publicPlayedCounts = validatePublicPlayedCounts(
    record.publicPlayedCounts,
  );
  for (let index = 0; index < 13; index += 1) {
    const rank = index + 1;
    if (ownHandCounts[index] > deckCopies(rank)) {
      throw new RangeError(`observation.ownHand has too many rank ${rank} cards`);
    }
    if (ownHandCounts[index] + publicPlayedCounts[index] > deckCopies(rank)) {
      throw new RangeError(
        `actor and public cards exceed the rank ${rank} deck supply`,
      );
    }
  }
  const publicCardTotal = publicPlayedCounts.reduce(
    (total, count) => total + count,
    0,
  );
  const hiddenAndOwnCardTotal = players.reduce(
    (total, player) => total + player.handCount,
    0,
  );
  if (publicCardTotal + hiddenAndOwnCardTotal !== 80) {
    throw new RangeError(
      "public card counts plus public player hand counts must total 80",
    );
  }

  const table = validateTable(record.table, offsets);
  if (table !== null) {
    const naturalPublicCount = publicPlayedCounts[table.rank - 1];
    if (table.naturalCount > naturalPublicCount) {
      throw new RangeError(
        "observation.table natural cards must be included in publicPlayedCounts",
      );
    }
    if (table.jokerCount > publicPlayedCounts[12]) {
      throw new RangeError(
        "observation.table jokers must be included in publicPlayedCounts",
      );
    }
  }

  const history = validateHistory(record.history, players.length, offsets);
  const truncatedHistoryCount = Math.max(
    0,
    history.length - V4_MAX_PUBLIC_HISTORY_EVENTS,
  );
  const compressedHistory = history.slice(0, truncatedHistoryCount);
  const historyTokens = history.slice(truncatedHistoryCount);
  const playersByOffset = [...players].sort(
    (left, right) => offsets.get(left.id)! - offsets.get(right.id)!,
  );
  const tableOffset = table?.actorOffset ?? -1;

  return {
    schemaVersion: V4_PUBLIC_OBSERVATION_SCHEMA_VERSION,
    playerCount: players.length,
    act: record.act,
    actorRole: roleId(actor.role, "observation actor role"),
    revolution,
    ownHandCounts,
    publicPlayedCounts: [...publicPlayedCounts],
    table,
    playerTokens: playersByOffset.map((player, relativeOffset) => ({
      relativeOffset,
      handCount: player.handCount,
      finished: flag(player.finished),
      passed: flag(player.passed),
      self: relativeOffset === 0 ? 1 : 0,
      tableLeader: relativeOffset === tableOffset ? 1 : 0,
      role: roleId(player.role, `player offset ${relativeOffset} role`),
      score: player.score,
    })),
    historyTokens,
    memoryTraceVectors: buildMemoryTraceVectors(
      players.length,
      compressedHistory,
    ),
    truncatedHistoryCount,
  };
}

/** Stable UTF-8 JSON for Python ingestion, hashing, and golden vectors. */
export function encodeV4ActorVisibleObservationBytes(
  value: unknown,
): Uint8Array {
  return new TextEncoder().encode(
    JSON.stringify(buildV4ActorVisibleObservation(value)),
  );
}
