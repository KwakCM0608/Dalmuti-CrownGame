"use client";

import type { CSSProperties, FormEvent } from "react";
import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { OnlineCommand, OnlineSnapshot } from "@/lib/online-game";
import {
  ONLINE_CHAT_HISTORY_LIMIT,
  ONLINE_CHAT_MAX_LENGTH,
} from "@/lib/online-chat";
import { scoreChipCount } from "@/lib/score-chips";
import {
  BOT_DIFFICULTIES,
  type BotDifficulty,
} from "@/lib/bot-strategy";
import styles from "./online.module.css";

type LooseRecord = Record<string, unknown>;

type CardView = {
  id: string;
  rank: number;
};

type PlayerView = {
  id: string;
  name: string;
  monogram: string;
  isBot: boolean;
  botDifficulty: BotDifficulty | null;
  role: string;
  ready: boolean;
  connected: boolean;
  handCount: number;
  finishedPlace: number | null;
  score: number;
};

type TableView = {
  rank: number;
  count: number;
  playerId: string;
  cards: CardView[];
} | null;

type EventView = {
  id: string;
  seq: number;
  type: string;
  at: number;
  startsAt: number;
  durationMs: number;
  playerIds: string[];
  actorPlayerId: string | null;
  data: LooseRecord;
};

type ChatMessageView = {
  seq: number;
  id: string;
  playerId: string;
  authorName: string;
  text: string;
  sentAt: number;
};

type MotionPoint = {
  x: number;
  y: number;
};

type MotionAnchors = {
  players: Record<string, MotionPoint>;
  center: MotionPoint | null;
};

const MOTION_ANCHOR_EPSILON = 0.5;

function motionPointsEqual(
  previous: MotionPoint | null,
  next: MotionPoint | null,
): boolean {
  if (previous === next) return true;
  if (!previous || !next) return false;
  return (
    Math.abs(previous.x - next.x) <= MOTION_ANCHOR_EPSILON &&
    Math.abs(previous.y - next.y) <= MOTION_ANCHOR_EPSILON
  );
}

function motionAnchorsEqual(
  previous: MotionAnchors,
  next: MotionAnchors,
): boolean {
  if (!motionPointsEqual(previous.center, next.center)) return false;
  const previousIds = Object.keys(previous.players);
  const nextIds = Object.keys(next.players);
  if (previousIds.length !== nextIds.length) return false;
  return nextIds.every((playerId) =>
    motionPointsEqual(
      previous.players[playerId] ?? null,
      next.players[playerId] ?? null,
    ),
  );
}

type RankChoiceCardView = {
  slotIndex: number;
  claimedByPlayerId: string | null;
  revealedRank: number | null;
};

type RankSelectionView = {
  stage: "intro" | "selecting" | "locked" | "revealed" | "confirmed";
  cards: RankChoiceCardView[];
  introStartedAt: number | null;
  countdownStartsAt: number | null;
  countdownEndsAt: number | null;
  revealAt: number | null;
  revealEndsAt: number | null;
  canChoose: boolean;
  selectedSlotIndex: number | null;
};

type DeclaredRevolutionView = {
  round: number;
  playerId: string;
  kind: "great" | "normal";
};

type SnapshotView = {
  code: string;
  revision: number;
  eventSeq: number;
  serverTime: number;
  phase: string;
  phaseEndsAt: number | null;
  turnDeadline: number | null;
  round: number;
  viewerId: string;
  hostId: string;
  players: PlayerView[];
  hand: CardView[] | null;
  table: TableView;
  currentPlayerId: string | null;
  lastPlayedId: string | null;
  passedPlayerIds: string[];
  finishOrder: string[];
  dealSealed: boolean;
  deadline: number | null;
  actionLockUntil: number | null;
  events: EventView[];
  requiredReturnCount: number;
  selectedReturnCount: number;
  waitingTaxPlayerIds: string[];
  revolutionHolderId: string | null;
  canChooseRevolution: boolean;
  rankSelection: RankSelectionView | null;
  declaredRevolution: DeclaredRevolutionView | null;
};

type SnapshotEnvelope = {
  snapshot: SnapshotView | null;
  unchanged: boolean;
  chatMessages: ChatMessageView[];
  latestChatSeq: number;
};

type StoredSession = {
  roomCode: string;
  playerId: string;
  token: string;
  nickname: string;
};

type TaxVisualOverride = {
  phase: "tax-tribute" | "tax-return";
  expiresAt: number;
  hand: CardView[] | null;
  handCounts: Record<string, number>;
};

type ConnectionState =
  | "idle"
  | "connecting"
  | "online"
  | "reconnecting"
  | "offline";

const SESSION_PREFIX = "dalmuti.online.room.";
const LAST_SESSION_KEY = "dalmuti.online.last-session";
const POLL_INTERVAL_MS = 250;
const MAX_EVENT_CATCHUP_MS = 120;
const TURN_DURATION_MS = 30_000;
const RANK_MOVE_DURATION_MS = 2_300;
const ROUND_END_MOVE_PRELUDE_MS = 380;
const ROUND_END_MOVE_SETTLE_MS = 520;
const BOT_DIFFICULTY_LABELS: Record<BotDifficulty, string> = {
  easy: "쉬움",
  normal: "보통",
  hard: "어려움",
};
const BOT_DIFFICULTY_DESCRIPTIONS: Record<BotDifficulty, string> = {
  easy: "기본적인 카드 제출",
  normal: "조커와 묶음을 관리",
  hard: "상대의 완주 위협까지 대응",
};
const ROLE_LABELS: Record<string, string> = {
  "great-dalmuti": "달무티",
  great_dalmuti: "달무티",
  "lesser-dalmuti": "총리대신",
  lesser_dalmuti: "총리대신",
  merchant: "상인",
  "lesser-peon": "소작농",
  lesser_peon: "소작농",
  "great-peon": "농노",
  great_peon: "농노",
};
const ROLE_MARKS: Record<string, string> = {
  "great-dalmuti": "♛",
  great_dalmuti: "♛",
  "lesser-dalmuti": "♕",
  lesser_dalmuti: "♕",
  merchant: "◆",
  "lesser-peon": "♙",
  lesser_peon: "♙",
  "great-peon": "♟",
  great_peon: "♟",
};
const RANK_NAMES: Record<number, string> = {
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
};

function isRecord(value: unknown): value is LooseRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function record(value: unknown): LooseRecord {
  return isRecord(value) ? value : {};
}

function firstRecord(...values: unknown[]): LooseRecord {
  return values.map(record).find((value) => Object.keys(value).length > 0) ?? {};
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function numberValue(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function booleanValue(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizePhase(value: unknown): string {
  return stringValue(value, "lobby").toLowerCase().replaceAll("_", "-");
}

function normalizeCode(value: string): string {
  return value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 6);
}

function cardFrom(value: unknown, index = 0): CardView | null {
  const source = record(value);
  const rank = numberValue(source.rank, numberValue(source.value, 0));
  if (rank < 1 || rank > 13) return null;
  return {
    id: stringValue(
      source.id,
      stringValue(source.cardId, `card-${rank}-${index}`),
    ),
    rank,
  };
}

function cardsFrom(value: unknown): CardView[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry, index) => cardFrom(entry, index))
    .filter((entry): entry is CardView => entry !== null);
}

function playerFrom(value: unknown, index: number): PlayerView {
  const source = record(value);
  const name = stringValue(source.name, `플레이어 ${index + 1}`);
  const finishedValue =
    source.finishedPlace ?? source.finishedRank ?? source.finishPlace;
  return {
    id: stringValue(source.id, stringValue(source.playerId, `player-${index}`)),
    name,
    monogram: stringValue(source.monogram, name.trim().slice(0, 1) || "?"),
    isBot: booleanValue(source.isBot),
    botDifficulty:
      source.botDifficulty === "easy" ||
      source.botDifficulty === "hard" ||
      source.botDifficulty === "normal"
        ? source.botDifficulty
        : booleanValue(source.isBot)
          ? "normal"
          : null,
    role: stringValue(source.role, "merchant"),
    ready: booleanValue(source.ready),
    connected: booleanValue(source.connected, true),
    handCount: numberValue(source.handCount, numberValue(source.cards, 0)),
    finishedPlace:
      typeof finishedValue === "number" && Number.isFinite(finishedValue)
        ? finishedValue
        : null,
    score: numberValue(source.score),
  };
}

function eventFrom(value: unknown, index: number): EventView {
  const source = record(value);
  const payload = firstRecord(source.payload, source.data, source.private);
  const merged = { ...source, ...payload };
  const type = stringValue(source.type, stringValue(source.kind, "EVENT"))
    .toUpperCase()
    .replaceAll("-", "_");
  const seq = numberValue(source.seq, numberValue(source.eventSeq, index));
  const at = numberValue(
    source.at,
    numberValue(source.startsAt, numberValue(payload.at, Date.now())),
  );
  const startsAt = numberValue(
    source.startsAt,
    numberValue(payload.startsAt, at),
  );
  const explicitEndsAt = nullableNumber(source.endsAt ?? payload.endsAt);
  const explicitDuration = nullableNumber(
    source.durationMs ?? payload.durationMs,
  );
  const durationMs =
    explicitEndsAt !== null
      ? Math.max(0, explicitEndsAt - startsAt)
      : explicitDuration !== null && explicitDuration > 0
        ? explicitDuration
        : defaultEventDuration(type);
  return {
    id: stringValue(source.id, `${seq}-${stringValue(source.type, "event")}`),
    seq,
    type,
    at,
    startsAt,
    durationMs,
    playerIds: stringArray(source.playerIds).length
      ? stringArray(source.playerIds)
      : stringArray(payload.playerIds),
    actorPlayerId:
      stringValue(
        source.actorPlayerId,
        stringValue(
          source.playerId,
          stringValue(payload.actorPlayerId, stringValue(payload.playerId)),
        ),
      ) || null,
    data: merged,
  };
}

function chatMessageFrom(value: unknown): ChatMessageView | null {
  const source = record(value);
  const seq = numberValue(source.seq);
  const id = stringValue(source.id, stringValue(source.messageId));
  const playerId = stringValue(source.playerId);
  const authorName = stringValue(source.authorName, "플레이어");
  const text = stringValue(source.text, stringValue(source.body)).trim();
  const sentAt = numberValue(source.sentAt, numberValue(source.createdAt));
  if (!Number.isSafeInteger(seq) || seq < 1 || !id || !playerId || !text) {
    return null;
  }
  return {
    seq,
    id,
    playerId,
    authorName,
    text,
    sentAt,
  };
}

function defaultEventDuration(type: unknown): number {
  const label = stringValue(type).toUpperCase();
  if (label === "DALMUTI_EFFECT") return 3300;
  if (label.includes("REVOLUTION")) return 3300;
  if (label === "MATCH_STARTED") return 2400;
  if (label.includes("HAND_REVEAL")) return 1400;
  if (
    label.includes("TAX_TRIBUTE") ||
    label.includes("TAX_RETURN") ||
    label.includes("TRIBUTE")
  ) {
    return 6000;
  }
  if (label.includes("TAX_INTRO")) return 2400;
  if (label.includes("TAX")) return 2400;
  if (label.includes("RANK_CONFIRM")) return 2600;
  if (label.includes("RANK")) return 3400;
  // PLAYER_PASSED also contains "PLAY", so PASS must be matched first.
  if (label.includes("PASS")) return 1500;
  if (label.includes("PLAY_INTRO") || label.includes("GAME_START")) return 2600;
  if (label.includes("PLAY")) return 2250;
  return 2600;
}

/**
 * The engine owns the canonical OnlineSnapshot. This adapter deliberately
 * accepts both that shape and the public/self envelope used by early API builds.
 */
function snapshotFrom(
  value: OnlineSnapshot | unknown,
  responseMeta: LooseRecord = {},
): SnapshotView {
  const root = record(value);
  const publicView = firstRecord(root.public, root.room, root);
  const selfView = firstRecord(root.self, root.viewer);
  const taxView = firstRecord(root.tax, publicView.tax, selfView.tax);
  const revolutionView = firstRecord(
    root.revolution,
    publicView.revolution,
    selfView.revolution,
  );
  const rankSelectionView = firstRecord(
    root.rankSelection,
    publicView.rankSelection,
    selfView.rankSelection,
  );
  const declaredRevolutionView = firstRecord(
    root.declaredRevolution,
    publicView.declaredRevolution,
  );
  const playersValue = publicView.players ?? root.players;
  const eventsValue = root.events ?? publicView.events ?? responseMeta.events;
  const handValue = root.hand ?? selfView.hand;
  const tableSource = publicView.table ?? root.table;
  const tableRecord = record(tableSource);
  const table: TableView = Object.keys(tableRecord).length
    ? {
        rank: numberValue(tableRecord.rank),
        count: numberValue(
          tableRecord.count,
          cardsFrom(tableRecord.cards).length,
        ),
        playerId: stringValue(tableRecord.playerId),
        cards: cardsFrom(tableRecord.cards),
      }
    : null;
  const eventViews = Array.isArray(eventsValue)
    ? eventsValue.map(eventFrom)
    : [];
  const metaServerTime = numberValue(
    responseMeta.serverTime,
    numberValue(root.serverTime, Date.now()),
  );
  const normalizedPhase = normalizePhase(publicView.phase ?? root.phase);
  const rankCardsValue =
    rankSelectionView.cards ?? rankSelectionView.slots;
  const rankCards = Array.isArray(rankCardsValue)
    ? rankCardsValue.map((value, index): RankChoiceCardView => {
        const source = record(value);
        const revealedRank = nullableNumber(
          source.revealedRank ?? source.rank,
        );
        return {
          slotIndex: numberValue(
            source.slotIndex,
            numberValue(source.id, index),
          ),
          claimedByPlayerId:
            stringValue(
              source.claimedByPlayerId,
              stringValue(source.chosenBy),
            ) || null,
          revealedRank:
            revealedRank !== null && revealedRank >= 1 && revealedRank <= 12
              ? revealedRank
              : null,
        };
      })
    : [];
  const rankStageValue = stringValue(
    rankSelectionView.stage,
    normalizedPhase === "rank-intro"
      ? "intro"
      : normalizedPhase === "rank-confirm"
        ? "confirmed"
      : normalizedPhase === "rank-reveal"
        ? "revealed"
        : "selecting",
  );
  const rankStage: RankSelectionView["stage"] =
    rankStageValue === "intro" ||
    rankStageValue === "locked" ||
    rankStageValue === "revealed" ||
    rankStageValue === "confirmed"
      ? rankStageValue
      : "selecting";
  const declaredKind = stringValue(declaredRevolutionView.kind).toLowerCase();
  const normalizedDeclaredKind: DeclaredRevolutionView["kind"] | null =
    declaredKind === "great" || declaredKind === "great-revolution"
      ? "great"
      : declaredKind === "normal" || declaredKind === "revolution"
        ? "normal"
        : null;
  const declaredRound = numberValue(declaredRevolutionView.round);
  const declaredPlayerId = stringValue(
    declaredRevolutionView.playerId,
    stringValue(declaredRevolutionView.actorPlayerId),
  );

  return {
    code: normalizeCode(
      stringValue(
        root.code,
        stringValue(
          publicView.code,
          stringValue(responseMeta.roomCode, stringValue(responseMeta.code)),
        ),
      ),
    ),
    revision: numberValue(
      root.revision,
      numberValue(publicView.revision, numberValue(responseMeta.revision)),
    ),
    eventSeq: numberValue(
      root.eventSeq,
      numberValue(
        root.latestEventSeq,
        numberValue(
          publicView.eventSeq,
          numberValue(
            publicView.latestEventSeq,
            numberValue(
              responseMeta.eventSeq,
              eventViews.reduce((max, event) => Math.max(max, event.seq), 0),
            ),
          ),
        ),
      ),
    ),
    serverTime: metaServerTime,
    phase: normalizedPhase,
    phaseEndsAt:
      typeof (publicView.phaseEndsAt ?? root.phaseEndsAt) === "number"
        ? numberValue(publicView.phaseEndsAt ?? root.phaseEndsAt)
        : null,
    turnDeadline:
      typeof (
        publicView.turnDeadline ??
        publicView.turnEndsAt ??
        root.turnDeadline ??
        root.turnEndsAt
      ) === "number"
        ? numberValue(
            publicView.turnDeadline ??
              publicView.turnEndsAt ??
              root.turnDeadline ??
              root.turnEndsAt,
          )
        : null,
    round: numberValue(publicView.round, numberValue(root.round, 1)),
    viewerId: stringValue(
      root.viewerId,
      stringValue(
        selfView.playerId,
        stringValue(responseMeta.playerId, stringValue(root.playerId)),
      ),
    ),
    hostId: stringValue(
      publicView.hostId,
      stringValue(publicView.hostPlayerId, stringValue(root.hostId)),
    ),
    players: Array.isArray(playersValue)
      ? playersValue.map(playerFrom)
      : [],
    hand:
      handValue === null || handValue === undefined ? null : cardsFrom(handValue),
    table,
    currentPlayerId:
      stringValue(
        publicView.currentPlayerId,
        stringValue(root.currentPlayerId),
      ) || null,
    lastPlayedId:
      stringValue(publicView.lastPlayedId, stringValue(root.lastPlayedId)) ||
      null,
    passedPlayerIds: stringArray(
      publicView.passedPlayerIds ?? publicView.passed ?? root.passedPlayerIds,
    ),
    finishOrder: stringArray(publicView.finishOrder ?? root.finishOrder),
    dealSealed: booleanValue(
      publicView.dealSealed,
      booleanValue(root.dealSealed),
    ),
    deadline:
      typeof (publicView.deadline ?? root.deadline) === "number"
        ? numberValue(publicView.deadline ?? root.deadline)
        : null,
    actionLockUntil:
      typeof (publicView.actionLockUntil ?? root.actionLockUntil) === "number"
        ? numberValue(publicView.actionLockUntil ?? root.actionLockUntil)
        : null,
    events: eventViews,
    requiredReturnCount: numberValue(
      taxView.requiredReturnCount,
      numberValue(selfView.pendingTaxCount),
    ),
    selectedReturnCount: numberValue(taxView.selectedReturnCount),
    waitingTaxPlayerIds: stringArray(taxView.waitingForPlayerIds),
    revolutionHolderId:
      stringValue(
        revolutionView.holderId,
        stringValue(root.revolutionHolderId),
      ) || null,
    canChooseRevolution: booleanValue(
      revolutionView.canChoose,
      booleanValue(selfView.isRevolutionHolder),
    ),
    rankSelection:
      rankCards.length ||
      ["rank-intro", "rank-selection", "rank-reveal", "rank-confirm"].includes(
        normalizedPhase,
      )
        ? {
            stage: rankStage,
            cards: rankCards,
            introStartedAt: nullableNumber(rankSelectionView.introStartedAt),
            countdownStartsAt: nullableNumber(
              rankSelectionView.countdownStartsAt,
            ),
            countdownEndsAt: nullableNumber(
              rankSelectionView.countdownEndsAt,
            ),
            revealAt: nullableNumber(rankSelectionView.revealAt),
            revealEndsAt: nullableNumber(rankSelectionView.revealEndsAt),
            canChoose: booleanValue(
              rankSelectionView.canChoose,
              booleanValue(selfView.canChooseRankCard),
            ),
            selectedSlotIndex: nullableNumber(
              rankSelectionView.selectedSlotIndex ??
                selfView.selectedRankSlotIndex,
            ),
          }
        : null,
    declaredRevolution:
      declaredPlayerId &&
      declaredRound > 0 &&
      normalizedDeclaredKind
        ? {
            round: declaredRound,
            playerId: declaredPlayerId,
            kind: normalizedDeclaredKind,
          }
        : null,
  };
}

function unwrapSnapshotResponse(value: unknown): SnapshotEnvelope {
  const response = record(value);
  const chatMessages = Array.isArray(response.chatMessages)
    ? response.chatMessages
        .map(chatMessageFrom)
        .filter((message): message is ChatMessageView => message !== null)
    : [];
  const latestChatSeq = numberValue(
    response.latestChatSeq,
    chatMessages.at(-1)?.seq ?? 0,
  );
  if (response.unchanged === true) {
    return {
      snapshot: null,
      unchanged: true,
      chatMessages,
      latestChatSeq,
    };
  }
  const candidate = response.snapshot ?? response.projection ?? value;
  return {
    snapshot: snapshotFrom(candidate, response),
    unchanged: false,
    chatMessages,
    latestChatSeq,
  };
}

function apiErrorMessage(value: unknown, fallback: string): string {
  const source = record(value);
  const error = record(source.error);
  return stringValue(error.message, stringValue(source.message, fallback));
}

function createCommandId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `cmd-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function sessionKey(code: string): string {
  return `${SESSION_PREFIX}${normalizeCode(code)}`;
}

function saveSession(session: StoredSession): void {
  localStorage.setItem(sessionKey(session.roomCode), JSON.stringify(session));
  localStorage.setItem(LAST_SESSION_KEY, JSON.stringify(session));
}

function clearSavedSession(session: StoredSession | null): void {
  if (!session) return;
  localStorage.removeItem(sessionKey(session.roomCode));
  const remembered = readSession();
  if (
    remembered?.roomCode === session.roomCode &&
    remembered.token === session.token
  ) {
    localStorage.removeItem(LAST_SESSION_KEY);
  }
}

function readSession(code?: string): StoredSession | null {
  try {
    const key = code ? sessionKey(code) : LAST_SESSION_KEY;
    const parsed = JSON.parse(localStorage.getItem(key) ?? "null");
    if (!isRecord(parsed)) return null;
    const roomCode = normalizeCode(stringValue(parsed.roomCode));
    const playerId = stringValue(parsed.playerId);
    const token = stringValue(parsed.token);
    const nickname = stringValue(parsed.nickname);
    if (roomCode.length !== 6 || !playerId || !token) return null;
    return { roomCode, playerId, token, nickname };
  } catch {
    return null;
  }
}

function cardImage(rank: number): string {
  return rank === 13
    ? "/cards/joker.webp"
    : `/cards/${String(rank).padStart(2, "0")}.webp`;
}

function roleLabel(role: string): string {
  return ROLE_LABELS[role] ?? ROLE_LABELS[role.toLowerCase()] ?? "상인";
}

function roleMark(role: string): string {
  return ROLE_MARKS[role] ?? ROLE_MARKS[role.toLowerCase()] ?? "◆";
}

function roleLabelForRank(rankIndex: number, total: number): string {
  if (rankIndex === 0) return roleLabel("great-dalmuti");
  if (rankIndex === 1) return roleLabel("lesser-dalmuti");
  if (rankIndex === total - 2) return roleLabel("lesser-peon");
  if (rankIndex === total - 1) return roleLabel("great-peon");
  return roleLabel("merchant");
}

function playerName(players: PlayerView[], id: unknown): string {
  const playerId = stringValue(id);
  return players.find((player) => player.id === playerId)?.name ?? "플레이어";
}

function dalmutiActorIdFromEvent(
  event: EventView | null,
  players: PlayerView[],
): string | null {
  if (!event) return null;

  const table = record(event.data.table);
  const playedCards = cardsFrom(event.data.cards ?? table.cards);
  const playedRank = numberValue(
    event.data.rank,
    numberValue(
      table.rank,
      playedCards.find((card) => card.rank !== 13)?.rank ?? 13,
    ),
  );
  if (
    event.type !== "DALMUTI_EFFECT" &&
    !(event.type === "CARDS_PLAYED" && playedRank === 1)
  ) {
    return null;
  }

  const actorPlayerId =
    event.actorPlayerId ||
    stringValue(
      event.data.actorPlayerId,
      stringValue(event.data.playerId, stringValue(table.playerId)),
    );
  return players.some((player) => player.id === actorPlayerId)
    ? actorPlayerId
    : null;
}

function declaredRevolutionFromEvent(
  event: EventView | undefined,
  round: number,
): DeclaredRevolutionView | null {
  if (!event || event.type !== "REVOLUTION_DECLARED") return null;
  const eventRound = numberValue(event.data.round, Number.NaN);
  if (eventRound !== round) return null;
  const playerId =
    event.actorPlayerId ??
    stringValue(
      event.data.playerId,
      stringValue(event.data.actorPlayerId),
    );
  if (!playerId) return null;
  return {
    round: eventRound,
    playerId,
    kind:
      ["great", "great-revolution"].includes(
        stringValue(event.data.kind).toLowerCase(),
      ) ||
      booleanValue(event.data.isGreatRevolution)
        ? "great"
        : "normal",
  };
}

function seatPosition(rankIndex: number, total: number): CSSProperties {
  const angle =
    total <= 1 ? 270 : 150 + (240 * rankIndex) / Math.max(1, total - 1);
  const radians = (angle * Math.PI) / 180;
  return {
    "--seat-x": `${50 + Math.cos(radians) * 42}%`,
    "--seat-y": `${46 + Math.sin(radians) * 34}%`,
    "--seat-rank": rankIndex,
    "--seat-grid-column":
      total <= 5 ? rankIndex + 1 : (rankIndex % 4) + 1,
    "--seat-grid-row": total <= 5 ? 1 : Math.floor(rankIndex / 4) + 1,
  } as CSSProperties;
}

function formatRank(rank: number): string {
  return rank === 13
    ? "어릿광대"
    : `${RANK_NAMES[rank] ?? `${rank}등급`}(${rank})`;
}

function PlayingCard({
  card,
  selected = false,
  disabled = false,
  concealed = false,
  onClick,
  onDoubleClick,
  displayOnly = false,
  style,
}: {
  card: CardView;
  selected?: boolean;
  disabled?: boolean;
  concealed?: boolean;
  onClick?: () => void;
  onDoubleClick?: () => void;
  displayOnly?: boolean;
  style?: CSSProperties;
}) {
  return (
    <button
      type="button"
      className={`${styles.card} ${selected ? styles.cardSelected : ""} ${
        concealed ? styles.cardBack : ""
      } ${displayOnly ? styles.cardDisplayOnly : ""}`}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      disabled={disabled || displayOnly}
      style={style}
      aria-pressed={displayOnly ? undefined : selected}
      aria-label={
        concealed
          ? "뒤집힌 카드"
          : `${formatRank(card.rank)} 카드${selected ? ", 선택됨" : ""}`
      }
    >
      {!concealed && (
        <img src={cardImage(card.rank)} alt="" draggable={false} />
      )}
    </button>
  );
}

function Brand({
  onActivate,
  disabled = false,
}: {
  onActivate?: () => void;
  disabled?: boolean;
}) {
  const contents = (
    <>
      <span className={styles.brandSeal} aria-hidden="true" />
      <span>
        <strong>DALMUTI</strong>
        <small>DCLab의 계급전</small>
      </span>
    </>
  );
  if (onActivate) {
    return (
      <button
        type="button"
        className={`${styles.brand} ${styles.brandButton}`}
        onClick={onActivate}
        disabled={disabled}
        aria-label="방을 나가고 홈의 모드 선택 화면으로 이동"
      >
        {contents}
      </button>
    );
  }
  return (
    <Link className={styles.brand} href="/" aria-label="홈의 모드 선택 화면으로 이동">
      {contents}
    </Link>
  );
}

function ConnectionPill({
  state,
}: {
  state: ConnectionState;
}) {
  const copy: Record<ConnectionState, string> = {
    idle: "대기",
    connecting: "연결 중",
    online: "실시간 연결",
    reconnecting: "재연결 중",
    offline: "연결 끊김",
  };
  return (
    <span className={`${styles.connectionPill} ${styles[`connection_${state}`]}`}>
      <i />
      {copy[state]}
    </span>
  );
}

function PlayerSeat({
  player,
  isSelf,
  isHost,
  isCurrent,
  rankNumber,
  isRankMoving = false,
  rankMovement = null,
  showHandBacks = false,
  isHandRevealing = false,
  handRevealElapsedMs = 0,
  isDalmutiHighlighted = false,
  roleHidden = false,
  elementRef,
  style,
}: {
  player: PlayerView;
  isSelf?: boolean;
  isHost: boolean;
  isCurrent: boolean;
  rankNumber?: number;
  isRankMoving?: boolean;
  rankMovement?: "up" | "down" | null;
  showHandBacks?: boolean;
  isHandRevealing?: boolean;
  handRevealElapsedMs?: number;
  isDalmutiHighlighted?: boolean;
  roleHidden?: boolean;
  elementRef?: (element: HTMLElement | null) => void;
  style?: CSSProperties;
}) {
  const visibleRoleLabel = roleHidden ? "계급 미정" : roleLabel(player.role);
  const visibleRoleMark = roleHidden ? "?" : roleMark(player.role);
  return (
    <article
      ref={elementRef}
      className={`${styles.playerSeat} ${
        isSelf ? styles.playerSeatSelf : ""
      } ${isCurrent ? styles.playerSeatCurrent : ""} ${
        !player.connected ? styles.playerSeatDisconnected : ""
      } ${player.finishedPlace ? styles.playerSeatFinished : ""} ${
        isHandRevealing ? styles.playerSeatRevealing : ""
      } ${isRankMoving ? styles.playerSeatRankMoving : ""
      } ${
        isRankMoving && rankMovement === "up"
          ? styles.playerSeatRankMovingUp
          : ""
      } ${
        isRankMoving && rankMovement === "down"
          ? styles.playerSeatRankMovingDown
          : ""
      } ${isDalmutiHighlighted ? styles.playerSeatDalmuti : ""}`}
      style={style}
      data-rank-number={rankNumber}
      data-dalmuti-highlighted={isDalmutiHighlighted || undefined}
      aria-label={`${player.name}, ${
        rankNumber ? `현재 서열 ${rankNumber}위, ` : ""
      }${visibleRoleLabel}, 카드 ${player.handCount}장`}
    >
      <span className={styles.avatar}>
        {player.monogram}
        <i>{visibleRoleMark}</i>
      </span>
      <span className={styles.playerCopy}>
        <strong>
          {player.name}
          {isSelf && <small>나</small>}
          {player.isBot && (
            <small>
              BOT ·{" "}
              {BOT_DIFFICULTY_LABELS[player.botDifficulty ?? "normal"]}
            </small>
          )}
        </strong>
        <em>{visibleRoleLabel}</em>
      </span>
      <span className={styles.handCount}>
        <b>
          {player.finishedPlace
            ? `${player.finishedPlace}위`
            : `${player.handCount}장`}
        </b>
        <em>{player.score}점</em>
      </span>
      {isHost && <span className={styles.hostMark}>방장</span>}
      {isCurrent && <span className={styles.turnMark}>차례</span>}
      {!player.connected && <span className={styles.offlineMark}>재접속 대기</span>}
      {(showHandBacks || isHandRevealing) &&
        player.handCount > 0 &&
        !player.finishedPlace && (
        <span
          className={`${styles.seatRevealCards} ${
            isHandRevealing ? styles.seatRevealCardsAnimating : ""
          }`}
          aria-hidden="true"
        >
          {Array.from(
            { length: Math.min(4, Math.max(1, player.handCount)) },
            (_, index) => (
              <i
                key={index}
                style={
                  {
                    "--seat-card-index": index,
                    animationDelay: `${
                      index * 70 - handRevealElapsedMs
                    }ms`,
                  } as CSSProperties
                }
              />
            ),
          )}
        </span>
      )}
    </article>
  );
}

function RankSelectionField({
  rankSelection,
  players,
  viewerId,
  optimisticSlotIndex,
  effectiveClock,
  phaseEndsAt,
  busy,
  onChoose,
}: {
  rankSelection: RankSelectionView;
  players: PlayerView[];
  viewerId: string;
  optimisticSlotIndex: number | null;
  effectiveClock: number;
  phaseEndsAt: number | null;
  busy: boolean;
  onChoose: (slotIndex: number) => void;
}) {
  const isIntro = rankSelection.stage === "intro";
  const isRevealed = rankSelection.stage === "revealed";
  const isConfirmed = rankSelection.stage === "confirmed";
  const isLocked = rankSelection.stage === "locked";
  const countdownStart =
    rankSelection.countdownStartsAt ??
    rankSelection.introStartedAt ??
    (rankSelection.countdownEndsAt !== null
      ? rankSelection.countdownEndsAt - 3_300
      : effectiveClock);
  const countdownElapsed = Math.max(0, effectiveClock - countdownStart);
  const countdownRemaining =
    rankSelection.countdownEndsAt !== null &&
    effectiveClock >= rankSelection.countdownEndsAt
      ? 0
      : Math.max(1, 3 - Math.floor(countdownElapsed / 1_050));
  const cards = [...rankSelection.cards].sort(
    (a, b) => a.slotIndex - b.slotIndex,
  );
  const viewerCardIndex = cards.findIndex(
    (card) => card.claimedByPlayerId === viewerId,
  );
  const viewerRank =
    viewerCardIndex >= 0 ? cards[viewerCardIndex].revealedRank : null;
  const viewerHasChosen =
    rankSelection.selectedSlotIndex !== null ||
    cards.some((card) => card.claimedByPlayerId === viewerId) ||
    optimisticSlotIndex !== null;
  const phaseStartedAt = isIntro
    ? rankSelection.introStartedAt
    : isRevealed
      ? rankSelection.revealAt
      : isConfirmed && phaseEndsAt !== null
        ? phaseEndsAt - 2_600
        : rankSelection.countdownEndsAt;
  const [phaseElapsed] = useState(() =>
    Math.max(
      0,
      Math.min(
        MAX_EVENT_CATCHUP_MS,
        phaseStartedAt === null ? 0 : effectiveClock - phaseStartedAt,
      ),
    ),
  );

  if (isIntro) {
    return (
      <div
        className={styles.rankChoiceIntro}
        style={{ "--phase-elapsed": `${phaseElapsed}ms` } as CSSProperties}
        role="status"
        aria-live="polite"
      >
        <small>ACT I · RANK DRAW</small>
        <strong>계급 정하기</strong>
        <span>첫 게임은 선착순으로 카드를 한 장씩 골라 계급을 정합니다</span>
        {countdownRemaining > 0 && (
          <b
            className={styles.rankCountdown}
            key={countdownRemaining}
            aria-label={`${countdownRemaining}초 후 선택 시작`}
          >
            {countdownRemaining}
          </b>
        )}
        <em className={styles.rankChoiceIntroHint}>
          숫자가 낮을수록 높은 계급입니다
        </em>
      </div>
    );
  }

  if (isConfirmed && viewerRank !== null) {
    return (
      <div
        className={styles.rankConfirmation}
        style={{ "--phase-elapsed": `${phaseElapsed}ms` } as CSSProperties}
        role="status"
        aria-live="assertive"
      >
        <small>YOUR RANK · ACT I</small>
        <div className={styles.rankConfirmationBody}>
          <div className={styles.rankConfirmationCard} aria-hidden="true">
            <img src={cardImage(viewerRank)} alt="" />
          </div>
          <div className={styles.rankConfirmationCopy}>
            <span>나의 서열</span>
            <strong>
              {roleLabelForRank(viewerRank - 1, players.length)}
            </strong>
            <em>
              {RANK_NAMES[viewerRank] ?? "계급"}({viewerRank}) 카드를
              선택했습니다
            </em>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`${styles.rankChoiceField} ${
        isRevealed ? styles.rankChoiceFieldRevealed : ""
      }`}
      style={
        {
          "--phase-elapsed": `${phaseElapsed}ms`,
          "--rank-card-count": Math.max(1, cards.length),
        } as CSSProperties
      }
      role="group"
      aria-label={isRevealed ? "공개된 계급 카드" : "계급 카드 선택"}
    >
      <div className={styles.rankChoiceHeading}>
        <small>ACT I · RANK DRAW</small>
        <strong>
          {isRevealed
            ? "계급 카드 공개"
            : isLocked
              ? "모든 선택 완료"
              : viewerHasChosen
                ? "다른 플레이어를 기다리는 중"
                : "계급 카드를 고르세요"}
        </strong>
        <p>
          {isRevealed
            ? "낮은 숫자의 카드를 뽑은 순서로 서열이 정해집니다"
            : isLocked
              ? "1초 뒤 선택한 카드를 공개합니다"
              : viewerHasChosen
                ? "빛나는 카드는 이미 선택된 카드입니다"
                : rankSelection.canChoose
                  ? "빛나는 카드는 이미 다른 플레이어가 선택했습니다"
                  : "다른 플레이어들이 카드를 선택하는 중입니다"}
        </p>
      </div>
      <div
        className={styles.rankChoiceCards}
      >
        {cards.map((card, index) => {
          const optimisticMine =
            optimisticSlotIndex === card.slotIndex &&
            card.claimedByPlayerId === null;
          const claimant = card.claimedByPlayerId
            ? players.find((player) => player.id === card.claimedByPlayerId)
            : null;
          const claimed = Boolean(card.claimedByPlayerId) || optimisticMine;
          const mine =
            card.claimedByPlayerId === viewerId || optimisticMine;
          const canChoose =
            !isLocked &&
            !isRevealed &&
            rankSelection.canChoose &&
            !viewerHasChosen &&
            !claimed &&
            !busy;
          const rank = card.revealedRank;
          return (
            <div
              className={`${styles.rankChoiceSlot} ${
                claimed ? styles.rankChoiceSlotClaimed : ""
              } ${mine ? styles.rankChoiceSlotMine : ""} ${
                isRevealed ? styles.rankChoiceSlotRevealed : ""
              }`}
              key={card.slotIndex}
              style={{ "--rank-card-index": index } as CSSProperties}
            >
              <button
                type="button"
                className={styles.rankChoiceCard}
                disabled={!canChoose}
                aria-label={
                  isRevealed
                    ? `${claimant?.name ?? "플레이어"}, ${formatRank(rank ?? index + 1)}`
                    : claimed
                      ? mine
                        ? "내가 선택한 카드"
                        : "이미 선택된 카드"
                      : canChoose
                        ? `${index + 1}번째 계급 카드 선택`
                        : "선택 가능한 계급 카드"
                }
                aria-pressed={mine}
                onClick={() => onChoose(card.slotIndex)}
              >
                <span className={styles.rankChoiceCardInner}>
                  <i className={styles.rankChoiceCardBack} />
                  <i className={styles.rankChoiceCardFront}>
                    {rank !== null && <img src={cardImage(rank)} alt="" />}
                  </i>
                </span>
              </button>
              <span className={styles.rankChoiceOwner}>
                {claimant ? `${claimant.name} 선택` : "선택 가능"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

type EventOverlayProps = {
  event: EventView;
  players: PlayerView[];
  anchors: MotionAnchors;
  effectiveClock: number;
};

function EventOverlayView({
  event,
  players,
  anchors,
  effectiveClock,
}: EventOverlayProps) {
  const type = event.type;
  const data = event.data;
  const cards = cardsFrom(
    data.cards ??
      data.cardDetails ??
      data.taxCards ??
      record(data.transfer).cards,
  );
  const route = Array.isArray(data.routes)
    ? record(data.routes[0])
    : record(data.route);
  const actorId =
    event.actorPlayerId ??
    stringValue(data.actorPlayerId, stringValue(data.playerId)) ??
    null;
  const fromId = stringValue(
    data.fromPlayerId,
    stringValue(
      data.sourcePlayerId,
      stringValue(
        data.peonId,
        stringValue(route.fromPlayerId, stringValue(route.peonId)),
      ),
    ),
  );
  const toId = stringValue(
    data.toPlayerId,
    stringValue(
      data.destinationPlayerId,
      stringValue(
        data.nobleId,
        stringValue(route.toPlayerId, stringValue(route.nobleId)),
      ),
    ),
  );
  const autoPassedIds = stringArray(data.autoPassedPlayerIds);
  const center = anchors.center ?? { x: 0, y: 0 };
  const from = anchors.players[fromId || actorId || ""] ?? center;
  const to = anchors.players[toId] ?? center;
  const [initialElapsed] = useState(() =>
    Math.max(
      0,
      Math.min(
        event.durationMs,
        effectiveClock - event.startsAt,
      ),
    ),
  );
  const overlayStyle = useMemo(
    () =>
      ({
        "--event-duration": `${event.durationMs}ms`,
        "--event-elapsed": `${initialElapsed}ms`,
        "--from-x": `${from.x}px`,
        "--from-y": `${from.y}px`,
        "--to-x": `${to.x}px`,
        "--to-y": `${to.y}px`,
        "--center-x": `${center.x}px`,
        "--center-y": `${center.y}px`,
      }) as CSSProperties,
    [
      center.x,
      center.y,
      event.durationMs,
      from.x,
      from.y,
      initialElapsed,
      to.x,
      to.y,
    ],
  );

  if (
    type === "REVOLUTION_DECLARED" ||
    type === "REVOLUTION_INTRO_STARTED"
  ) {
    const isGreatRevolution =
      stringValue(data.kind).toLowerCase() === "great" ||
      booleanValue(data.isGreatRevolution) ||
      booleanValue(data.great);
    return (
      <div
        className={`${styles.eventOverlay} ${styles.introOverlay} ${
          styles.revolutionOverlay
        } ${styles.phaseIntroOverlay} ${
          isGreatRevolution ? styles.greatRevolutionOverlay : ""
        }`}
        style={overlayStyle}
      >
        <div className={styles.revolutionJokers} aria-hidden="true">
          <span />
          <span />
        </div>
        <div className={styles.eventCenterCopy}>
          <small>{isGreatRevolution ? "GREAT REVOLUTION" : "REVOLUTION"}</small>
          <strong>{isGreatRevolution ? "대혁명" : "혁명"}</strong>
          <b>
            {playerName(players, actorId)}이(가){" "}
            {isGreatRevolution ? "대혁명" : "혁명"}을 일으켰습니다
          </b>
          <span>
            {isGreatRevolution
              ? "세금이 사라지고 곧 계급 전복이 시작됩니다"
              : "이번 막의 세금 교환이 취소되었습니다 · 게임을 시작합니다"}
          </span>
        </div>
      </div>
    );
  }

  if (type === "GREAT_REVOLUTION_RANK_SWAP_STARTED") {
    return (
      <div
        className={`${styles.eventOverlay} ${styles.introOverlay} ${styles.phaseIntroOverlay} ${styles.greatRevolutionRankSwapOverlay}`}
        style={overlayStyle}
      >
        <div className={styles.eventCenterCopy}>
          <small>RANKS OVERTURNED</small>
          <strong>모두의 계급이 뒤바뀝니다</strong>
          <b>대혁명으로 인해 모두의 계급이 뒤바뀝니다</b>
          <span>새로운 서열에 맞춰 자리를 이동합니다</span>
        </div>
      </div>
    );
  }

  if (type.includes("TAX") || type.includes("TRIBUTE")) {
    const isIntro = type.includes("INTRO");
    if (isIntro) {
      return (
        <div
          className={`${styles.eventOverlay} ${styles.introOverlay} ${styles.phaseIntroOverlay} ${styles.taxIntroOverlay}`}
          style={overlayStyle}
        >
          <div className={styles.eventCenterCopy}>
            <small>TRIBUTE PHASE</small>
            <strong>세금 교환</strong>
            <span>계급에 따른 카드 교환을 시작합니다</span>
          </div>
        </div>
      );
    }
    const publicRoutes = (
      Array.isArray(data.routes) && data.routes.length
        ? data.routes.map(record)
        : Object.keys(route).length
          ? [route]
          : [
              {
                fromPlayerId: fromId,
                toPlayerId: toId,
                count: numberValue(data.count, Math.max(1, cards.length)),
              },
            ]
    ).filter((candidate) => Object.keys(candidate).length > 0);
    const privateFromId = stringValue(
      data.fromPlayerId,
      stringValue(data.sourcePlayerId),
    );
    const privateToId = stringValue(
      data.toPlayerId,
      stringValue(data.destinationPlayerId),
    );
    return (
      <div
        className={`${styles.eventOverlay} ${styles.taxOverlay}`}
        style={overlayStyle}
      >
        <div className={styles.taxRoutes}>
          {publicRoutes.map((publicRoute, routeIndex) => {
            const routeFromId = stringValue(
              publicRoute.fromPlayerId,
              stringValue(publicRoute.peonId, fromId),
            );
            const routeToId = stringValue(
              publicRoute.toPlayerId,
              stringValue(publicRoute.nobleId, toId),
            );
            const routeCount = Math.max(
              1,
              numberValue(publicRoute.count, numberValue(data.count, 1)),
            );
            const routeIsPrivate =
              cards.length > 0 &&
              (publicRoutes.length === 1 ||
                (privateFromId === routeFromId && privateToId === routeToId));
            const routeCards = routeIsPrivate
              ? cards
              : Array.from({ length: routeCount }, (_, index) => ({
                  id: `hidden-tax-${routeIndex}-${index}`,
                  rank: 13,
                }));
            const routeFrom =
              anchors.players[routeFromId] ?? anchors.center ?? center;
            const routeTo =
              anchors.players[routeToId] ?? anchors.center ?? center;
            const routeMidpoint = routeIsPrivate
              ? center
              : {
                  x: Math.round((routeFrom.x + routeTo.x) / 2),
                  y: Math.round(
                    Math.min(
                      center.y - 78,
                      Math.max(routeFrom.y, routeTo.y) +
                        88 +
                        routeIndex * 22,
                    ),
                  ),
                };
            return (
            <div
              className={`${styles.taxRoute} ${
                routeIsPrivate ? styles.taxRoutePrivate : styles.taxRoutePublic
              }`}
              key={`${event.id}-${routeFromId}-${routeToId}-${routeIndex}`}
              style={
                {
                  "--from-x": `${routeFrom.x}px`,
                  "--from-y": `${routeFrom.y}px`,
                  "--to-x": `${routeTo.x}px`,
                  "--to-y": `${routeTo.y}px`,
                  "--center-x": `${routeMidpoint.x}px`,
                  "--center-y": `${routeMidpoint.y}px`,
                  "--tax-route-index": routeIndex,
                  "--tax-route-count": publicRoutes.length,
                } as CSSProperties
              }
            >
              <div className={styles.transferNames}>
                <span>{playerName(players, routeFromId)}</span>
                <i>→</i>
                <span>{playerName(players, routeToId)}</span>
              </div>
              <div className={styles.eventCards}>
                {routeCards.map((card, index) => (
                  <div
                    className={styles.eventCardWrap}
                    key={`${event.id}-${routeIndex}-${card.id}-${index}`}
                    style={
                      {
                        "--event-card-endpoint-offset-x": `${
                          (index - (routeCards.length - 1) / 2) * 18
                        }px`,
                        "--event-card-mid-offset-x": `${
                          (index - (routeCards.length - 1) / 2) *
                          (routeIsPrivate ? 132 : 42)
                        }px`,
                        "--event-card-index": index,
                        "--event-card-offset":
                          index - (routeCards.length - 1) / 2,
                      } as CSSProperties
                    }
                  >
                    <PlayingCard
                      card={card}
                      concealed={!routeIsPrivate}
                      displayOnly
                    />
                  </div>
                ))}
              </div>
              <strong>
                {routeIsPrivate
                  ? cards.map((card) => formatRank(card.rank)).join(" · ")
                  : `카드 ${routeCount}장 이동`}
              </strong>
              <small>
                {routeIsPrivate
                  ? "이 카드 정보는 교환 당사자에게만 보입니다"
                  : "카드의 정체는 교환 당사자만 확인할 수 있습니다"}
              </small>
            </div>
            );
          })}
        </div>
      </div>
    );
  }

  if (type === "DALMUTI_EFFECT") {
    const dalmutiCards = cards.length
      ? cards
      : [{ id: `${event.id}-dalmuti`, rank: 1 }];
    const expandedStep =
      dalmutiCards.length <= 1
        ? 0
        : Math.min(112, 430 / Math.max(1, dalmutiCards.length - 1));
    const delayStep =
      dalmutiCards.length <= 1
        ? 0
        : Math.min(36, 100 / Math.max(1, dalmutiCards.length - 1));
    return (
      <div
        className={`${styles.eventOverlay} ${styles.playOverlay} ${styles.dalmutiEffectOverlay}`}
        style={overlayStyle}
      >
        <small>DALMUTI</small>
        <div className={styles.eventCards}>
          {dalmutiCards.map((card, index) => (
            <div
              className={styles.eventCardWrap}
              key={`${event.id}-${card.id}-${index}`}
              style={
                {
                  "--event-card-index": index,
                  "--event-card-offset":
                    index - (dalmutiCards.length - 1) / 2,
                  "--event-card-offset-x": `${
                    (index - (dalmutiCards.length - 1) / 2) * 46
                  }px`,
                  "--event-card-expanded-x": `${
                    (index - (dalmutiCards.length - 1) / 2) * expandedStep
                  }px`,
                  "--event-card-from-spread": `${
                    (index - (dalmutiCards.length - 1) / 2) * 9
                  }px`,
                  "--event-card-angle": `${
                    (index - (dalmutiCards.length - 1) / 2) * 3
                  }deg`,
                  "--event-card-delay": `${index * delayStep}ms`,
                } as CSSProperties
              }
            >
              <PlayingCard card={card} displayOnly />
            </div>
          ))}
        </div>
        <strong>달무티</strong>
        <span>
          {playerName(players, actorId)}이(가) 달무티(1) x{" "}
          {dalmutiCards.length}장을 냈습니다
        </span>
        <div className={styles.autoPassPlayers}>
          {autoPassedIds.map((playerId, playerIndex) => (
            <i
              key={playerId}
              style={
                {
                  "--pass-x": `${anchors.players[playerId]?.x ?? center.x}px`,
                  "--pass-y": `${anchors.players[playerId]?.y ?? center.y}px`,
                  "--pass-offset-x": `${
                    (playerIndex - (autoPassedIds.length - 1) / 2) * 104
                  }px`,
                  "--pass-delay": `${260 + playerIndex * 55}ms`,
                } as CSSProperties
              }
            >
              {playerName(players, playerId)} PASS
            </i>
          ))}
        </div>
        <b>나머지 플레이어 자동 PASS</b>
      </div>
    );
  }

  if (type.includes("PASS")) {
    const passReason = stringValue(event.data.reason);
    return (
      <div
        className={`${styles.eventOverlay} ${styles.passOverlay}`}
        style={overlayStyle}
      >
        <div className={styles.passMotion}>
          <span>{playerName(players, actorId)}</span>
          <strong>PASS</strong>
          <small>
            {passReason === "insufficient-cards"
              ? "필요한 장수보다 손패가 적어 자동 패스합니다"
              : passReason === "timeout"
                ? "제한시간이 끝나 자동 패스합니다"
                : "이번 묶음을 넘겼습니다"}
          </small>
        </div>
      </div>
    );
  }

  if (type.includes("PLAY") && !type.includes("INTRO") && cards.length) {
    const playedRank =
      cards.find((card) => card.rank !== 13)?.rank ?? 13;
    const expandedStep =
      cards.length <= 1
        ? 0
        : Math.min(112, 430 / Math.max(1, cards.length - 1));
    const delayStep =
      cards.length <= 1
        ? 0
        : Math.min(36, 100 / Math.max(1, cards.length - 1));
    return (
      <div
        className={`${styles.eventOverlay} ${styles.playOverlay}`}
        style={overlayStyle}
      >
        <div className={styles.eventCards}>
          {cards.map((card, index) => (
            <div
              className={styles.eventCardWrap}
              key={`${event.id}-${card.id}`}
              style={
                {
                  "--event-card-index": index,
                  "--event-card-offset": index - (cards.length - 1) / 2,
                  "--event-card-offset-x": `${
                    (index - (cards.length - 1) / 2) * 46
                  }px`,
                  "--event-card-expanded-x": `${
                    (index - (cards.length - 1) / 2) * expandedStep
                  }px`,
                  "--event-card-from-spread": `${
                    (index - (cards.length - 1) / 2) * 9
                  }px`,
                  "--event-card-angle": `${(index - 1) * 3}deg`,
                  "--event-card-delay": `${index * delayStep}ms`,
                } as CSSProperties
              }
            >
              <PlayingCard card={card} displayOnly />
            </div>
          ))}
        </div>
        <div className={styles.playCaption}>
          <small>공개 플레이</small>
          <strong>{playerName(players, actorId)}</strong>
          <span>
            {RANK_NAMES[playedRank] ?? "어릿광대"}({playedRank}) x{" "}
            {cards.length}장
          </span>
        </div>
      </div>
    );
  }

  if (type === "MATCH_STARTED") {
    return (
      <div
        className={`${styles.eventOverlay} ${styles.revealOverlay} ${styles.phaseIntroOverlay}`}
        style={overlayStyle}
      >
        <div className={styles.eventCenterCopy}>
          <span className={styles.revealCard} aria-hidden="true" />
          <small>HAND REVEAL</small>
          <strong>패를 공개합니다</strong>
          <span>모든 플레이어가 동시에 자신의 패를 확인합니다</span>
        </div>
      </div>
    );
  }

  if (
    type.includes("GAME_START") ||
    type.includes("ROUND_START") ||
    type.includes("PLAY_INTRO") ||
    (type.includes("PLAY") && type.includes("INTRO"))
  ) {
    const starterId = stringValue(
      data.firstPlayerId,
      stringValue(
        data.currentPlayerId,
        stringValue(data.startingPlayerId),
      ),
    );
    return (
      <div
        className={`${styles.eventOverlay} ${styles.introOverlay} ${styles.phaseIntroOverlay} ${styles.playIntroOverlay}`}
        style={overlayStyle}
      >
        <div className={styles.eventCenterCopy}>
          <small>ROUND {numberValue(data.round, 1)}</small>
          <strong>게임 시작</strong>
          <span>{playerName(players, starterId)}이(가) 먼저 시작합니다</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`${styles.eventOverlay} ${styles.introOverlay}`}
      style={overlayStyle}
    >
      <div className={styles.eventCenterCopy}>
        <small>DALMUTI ONLINE</small>
        <strong>{stringValue(data.title, "게임 진행")}</strong>
        <span>
          {stringValue(data.message, "모든 플레이어의 상태를 맞추고 있습니다")}
        </span>
      </div>
    </div>
  );
}

function eventOverlayPropsEqual(
  previous: EventOverlayProps,
  next: EventOverlayProps,
): boolean {
  if (
    previous.event.id !== next.event.id ||
    previous.event.type !== next.event.type ||
    previous.event.startsAt !== next.event.startsAt ||
    previous.event.durationMs !== next.event.durationMs ||
    previous.event.actorPlayerId !== next.event.actorPlayerId ||
    previous.anchors !== next.anchors ||
    previous.players.length !== next.players.length
  ) {
    return false;
  }

  // Events are immutable once issued. Ignore the 120 ms clock tick and fresh
  // polling projections while the same animation is already running.
  return previous.players.every(
    (player, index) =>
      player.id === next.players[index]?.id &&
      player.name === next.players[index]?.name,
  );
}

const EventOverlay = memo(EventOverlayView, eventOverlayPropsEqual);

function formatChatTime(timestamp: number): string {
  if (!timestamp) return "";
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(timestamp);
}

function OnlineChatPanel({
  className = "",
  messages,
  viewerId,
  connected,
  onSend,
}: {
  className?: string;
  messages: ChatMessageView[];
  viewerId: string;
  connected: boolean;
  onSend: (text: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const sendingRef = useRef(false);

  useEffect(() => {
    const messageList = messageListRef.current;
    if (!messageList) return;
    messageList.scrollTop = messageList.scrollHeight;
  }, [messages.length]);

  const submitChat = async (event: FormEvent) => {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sendingRef.current || !connected) return;
    sendingRef.current = true;
    setSending(true);
    setDraft("");
    setChatError(null);
    try {
      await onSend(text);
    } catch (reason) {
      setDraft((current) => current || text);
      setChatError(
        reason instanceof Error
          ? reason.message
          : "채팅을 보내지 못했습니다.",
      );
    } finally {
      sendingRef.current = false;
      setSending(false);
    }
  };

  return (
    <aside
      className={`${styles.chatPanel} ${
        collapsed ? styles.chatPanelCollapsed : ""
      } ${className}`}
      aria-label="플레이어 채팅"
    >
      <div className={styles.chatHeading}>
        <span>
          <i aria-hidden="true" />
          채팅
        </span>
        <button
          type="button"
          onClick={() => setCollapsed((current) => !current)}
          aria-expanded={!collapsed}
          aria-label={collapsed ? "채팅 펼치기" : "채팅 접기"}
        >
          {collapsed ? "+" : "−"}
        </button>
      </div>
      <div
        className={styles.chatMessages}
        ref={messageListRef}
        aria-live="polite"
        aria-relevant="additions"
      >
        {messages.map((message) => (
          <div
            className={`${styles.chatMessage} ${
              message.playerId === viewerId ? styles.chatMessageSelf : ""
            }`}
            key={message.id}
          >
            <span>
              <strong>{message.authorName}</strong>
              <time dateTime={new Date(message.sentAt).toISOString()}>
                {formatChatTime(message.sentAt)}
              </time>
            </span>
            <p>{message.text}</p>
          </div>
        ))}
        {!messages.length && (
          <p className={styles.chatEmpty}>첫 메시지를 남겨보세요.</p>
        )}
      </div>
      <form className={styles.chatComposer} onSubmit={submitChat}>
        <input
          value={draft}
          onChange={(event) =>
            setDraft(
              Array.from(event.target.value)
                .slice(0, ONLINE_CHAT_MAX_LENGTH)
                .join(""),
            )
          }
          placeholder={connected ? "메시지 입력" : "연결 복구 중"}
          aria-label="채팅 메시지"
          disabled={!connected}
        />
        <button
          type="submit"
          disabled={!connected || sending || !draft.trim()}
        >
          {sending ? "…" : "전송"}
        </button>
      </form>
      {chatError && <small className={styles.chatError}>{chatError}</small>}
    </aside>
  );
}

export default function OnlinePage() {
  const router = useRouter();
  const [screen, setScreen] = useState<"entry" | "room">("entry");
  const [entryMode, setEntryMode] = useState<"create" | "join">("create");
  const [nickname, setNickname] = useState("");
  const [roomCodeInput, setRoomCodeInput] = useState("");
  const [session, setSession] = useState<StoredSession | null>(null);
  const [lastSession, setLastSession] = useState<StoredSession | null>(null);
  const [snapshot, setSnapshot] = useState<SnapshotView | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [eventBuffer, setEventBuffer] = useState<EventView[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessageView[]>([]);
  const [clock, setClock] = useState(() => Date.now());
  const [observedRevolution, setObservedRevolution] =
    useState<DeclaredRevolutionView | null>(null);
  const [taxVisualOverride, setTaxVisualOverride] =
    useState<TaxVisualOverride | null>(null);
  const [serverOffset, setServerOffset] = useState(0);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fatalError, setFatalError] = useState(false);
  const [copied, setCopied] = useState(false);
  const [entryBusy, setEntryBusy] = useState(false);
  const [botDifficultyPickerSlot, setBotDifficultyPickerSlot] = useState<
    number | null
  >(null);
  const [rankMovingPlayerIds, setRankMovingPlayerIds] = useState<string[]>([]);
  const [seatRankOverrides, setSeatRankOverrides] = useState<
    Record<string, number> | null
  >(null);
  const [pendingRoundEndMoveIds, setPendingRoundEndMoveIds] = useState<
    string[] | null
  >(null);
  const [
    pendingGreatRevolutionMoveIds,
    setPendingGreatRevolutionMoveIds,
  ] = useState<string[] | null>(null);
  const [roundEndResultReady, setRoundEndResultReady] = useState(true);
  const [optimisticRankSlotIndex, setOptimisticRankSlotIndex] = useState<
    number | null
  >(null);
  const [handRevealElapsedMs, setHandRevealElapsedMs] = useState(0);
  const [motionAnchors, setMotionAnchors] = useState<MotionAnchors>({
    players: {},
    center: null,
  });
  const inFlightRef = useRef(false);
  const failureCountRef = useRef(0);
  const snapshotRef = useRef<SnapshotView | null>(null);
  const sessionRef = useRef<StoredSession | null>(null);
  const latestChatSeqRef = useRef(0);
  const rankChoiceInFlightRef = useRef<number | null>(null);
  const rankMoveTimerRef = useRef<number | null>(null);
  const seatElementsRef = useRef(new Map<string, HTMLElement>());
  const seatRectsRef = useRef(new Map<string, DOMRect>());
  const tableColumnRef = useRef<HTMLDivElement | null>(null);
  const tableCenterRef = useRef<HTMLDivElement | null>(null);

  const bindSeatElement = useCallback(
    (playerId: string, element: HTMLElement | null) => {
      if (element) {
        seatElementsRef.current.set(playerId, element);
      } else {
        seatElementsRef.current.delete(playerId);
      }
    },
    [],
  );

  const ingestSnapshot = useCallback((next: SnapshotView) => {
    const previous = snapshotRef.current;
    if (previous && next.revision < previous.revision) return;
    if (
      next.phase === "hand-reveal" &&
      next.phaseEndsAt !== null &&
      (previous?.phase !== "hand-reveal" ||
        previous.phaseEndsAt !== next.phaseEndsAt)
    ) {
      setHandRevealElapsedMs(
        Math.max(
          0,
          Math.min(
            MAX_EVENT_CATCHUP_MS,
            next.serverTime - (next.phaseEndsAt - 1_400),
          ),
        ),
      );
    } else if (
      previous?.phase === "hand-reveal" &&
      next.phase !== "hand-reveal"
    ) {
      setHandRevealElapsedMs(0);
    }
    if (
      previous &&
      (next.phase === "tax-tribute" || next.phase === "tax-return") &&
      previous.phase !== next.phase &&
      next.phaseEndsAt !== null
    ) {
      setTaxVisualOverride({
        phase: next.phase,
        expiresAt: next.phaseEndsAt,
        hand:
          previous.hand === null
            ? null
            : previous.hand.map((card) => ({ ...card })),
        handCounts: Object.fromEntries(
          previous.players.map((player) => [player.id, player.handCount]),
        ),
      });
    } else if (
      next.phase !== "tax-tribute" &&
      next.phase !== "tax-return"
    ) {
      setTaxVisualOverride(null);
    }
    const enteringGreatRevolutionSwap =
      previous &&
      previous.phase !== "great-revolution-swap" &&
      next.phase === "great-revolution-swap";
    if (enteringGreatRevolutionSwap) {
      const previousRankById = new Map(
        previous.players.map((player, index) => [player.id, index]),
      );
      const nextRankById = new Map(
        next.players.map((player, index) => [player.id, index]),
      );
      const movingPlayerIds = next.players
        .filter(
          (player) =>
            previousRankById.get(player.id) !== nextRankById.get(player.id),
        )
        .map((player) => player.id);
      setSeatRankOverrides(
        Object.fromEntries(
          previous.players.map((player, index) => [player.id, index]),
        ),
      );
      setRankMovingPlayerIds([]);
      setPendingGreatRevolutionMoveIds(movingPlayerIds);
      if (rankMoveTimerRef.current !== null) {
        window.clearTimeout(rankMoveTimerRef.current);
        rankMoveTimerRef.current = null;
      }
    }
    const enteringRoundEnd =
      previous &&
      previous.phase !== "round-end" &&
      next.phase === "round-end" &&
      next.finishOrder.length === next.players.length;
    if (enteringRoundEnd) {
      const previousRankById = new Map(
        previous.players.map((player, index) => [player.id, index]),
      );
      const nextRankById = new Map(
        next.finishOrder.map((playerId, index) => [playerId, index]),
      );
      const movingPlayerIds = previous.players
        .filter(
          (player) =>
            previousRankById.get(player.id) !== nextRankById.get(player.id),
        )
        .map((player) => player.id);
      if (movingPlayerIds.length) {
        setSeatRankOverrides(
          Object.fromEntries(
            previous.players.map((player, index) => [player.id, index]),
          ),
        );
      } else {
        setSeatRankOverrides(null);
      }
      setRankMovingPlayerIds([]);
      setPendingRoundEndMoveIds(movingPlayerIds);
      setRoundEndResultReady(false);
      if (rankMoveTimerRef.current !== null) {
        window.clearTimeout(rankMoveTimerRef.current);
        rankMoveTimerRef.current = null;
      }
    } else if (previous?.phase === "round-end" && next.phase !== "round-end") {
      setSeatRankOverrides(null);
      setPendingRoundEndMoveIds(null);
      setRankMovingPlayerIds([]);
      setRoundEndResultReady(true);
    }
    snapshotRef.current = next;
    setSnapshot(next);
    if (
      !["rank-intro", "rank-selection"].includes(next.phase) ||
      next.rankSelection?.selectedSlotIndex !== null
    ) {
      rankChoiceInFlightRef.current = null;
      setOptimisticRankSlotIndex(null);
    }
    setServerOffset(next.serverTime - Date.now());
    const declarationEvent = [...next.events]
      .reverse()
      .find((event) => event.type === "REVOLUTION_DECLARED");
    const eventDeclaration = declaredRevolutionFromEvent(
      declarationEvent,
      next.round,
    );
    setObservedRevolution((current) => {
      const declaration = next.declaredRevolution ?? eventDeclaration;
      if (declaration?.round === next.round) return declaration;
      if (
        ["lobby", "rank-intro", "rank-selection", "rank-reveal"].includes(
          next.phase,
        )
      ) {
        return null;
      }
      return current?.round === next.round ? current : null;
    });
    setSelectedIds((current) => {
      const handIds = new Set((next.hand ?? []).map((card) => card.id));
      return current.filter((id) => handIds.has(id));
    });
    if (next.events.length) {
      setEventBuffer((current) => {
        const merged = new Map(current.map((event) => [event.id, event]));
        next.events.forEach((event) => merged.set(event.id, event));
        return [...merged.values()]
          .sort((a, b) => a.seq - b.seq || a.startsAt - b.startsAt)
          .slice(-24);
      });
    }
  }, []);

  const ingestChatMessages = useCallback(
    (
      incoming: ChatMessageView[],
      latestSequence: number,
      advanceCursor = true,
    ) => {
      if (incoming.length) {
        setChatMessages((current) => {
          const merged = new Map(
            current.map((message) => [message.id, message]),
          );
          incoming.forEach((message) => merged.set(message.id, message));
          return [...merged.values()]
            .sort((a, b) => a.seq - b.seq)
            .slice(-ONLINE_CHAT_HISTORY_LIMIT);
        });
      }
      if (advanceCursor) {
        latestChatSeqRef.current = Math.max(
          latestChatSeqRef.current,
          latestSequence,
          ...incoming.map((message) => message.seq),
        );
      }
    },
    [],
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const query = new URLSearchParams(window.location.search);
      const invitedCode = normalizeCode(
        query.get("room") ?? query.get("code") ?? "",
      );
      const remembered = readSession(invitedCode || undefined);
      const last = readSession();
      setLastSession(last);
      if (invitedCode) {
        setRoomCodeInput(invitedCode);
        setEntryMode("join");
      }
      if (remembered && (!invitedCode || remembered.roomCode === invitedCode)) {
        sessionRef.current = remembered;
        setSession(remembered);
        setNickname(remembered.nickname);
        setRoomCodeInput(remembered.roomCode);
        setScreen("room");
        setConnection("connecting");
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  const pollRoom = useCallback(async () => {
    const activeSession = sessionRef.current;
    if (!activeSession || inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const sinceEventSeq = snapshotRef.current?.eventSeq ?? 0;
      const sinceChatSeq = latestChatSeqRef.current;
      const response = await fetch(
        `/api/online/rooms/${activeSession.roomCode}?sinceEventSeq=${sinceEventSeq}&sinceChatSeq=${sinceChatSeq}`,
        {
          headers: {
            Authorization: `Bearer ${activeSession.token}`,
            Accept: "application/json",
          },
          cache: "no-store",
        },
      );
      const body: unknown = await response.json().catch(() => ({}));
      if (sessionRef.current?.token !== activeSession.token) return;
      if (!response.ok) {
        const message = apiErrorMessage(body, "방에 다시 연결하지 못했습니다.");
        if (response.status === 401 || response.status === 404) {
          setFatalError(true);
          setConnection("offline");
        }
        throw new Error(message);
      }
      const result = unwrapSnapshotResponse(body);
      if (result.snapshot) ingestSnapshot(result.snapshot);
      ingestChatMessages(result.chatMessages, result.latestChatSeq);
      failureCountRef.current = 0;
      setConnection("online");
      setError(null);
    } catch (reason) {
      if (sessionRef.current?.token !== activeSession.token) return;
      failureCountRef.current += 1;
      setConnection(
        failureCountRef.current >= 4 ? "offline" : "reconnecting",
      );
      setError(
        reason instanceof Error
          ? reason.message
          : "실시간 연결을 복구하고 있습니다.",
      );
    } finally {
      inFlightRef.current = false;
    }
  }, [ingestChatMessages, ingestSnapshot]);

  useEffect(() => {
    if (!session || screen !== "room") return;
    const initialPoll = window.setTimeout(() => void pollRoom(), 0);
    const interval = window.setInterval(() => void pollRoom(), POLL_INTERVAL_MS);
    return () => {
      window.clearTimeout(initialPoll);
      window.clearInterval(interval);
    };
  }, [pollRoom, screen, session]);

  useEffect(() => {
    const interval = window.setInterval(() => setClock(Date.now()), 120);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(
    () => () => {
      if (rankMoveTimerRef.current !== null) {
        window.clearTimeout(rankMoveTimerRef.current);
      }
    },
    [],
  );

  const submitEntry = async (event: FormEvent) => {
    event.preventDefault();
    const cleanNickname = nickname.trim().slice(0, 16);
    const cleanCode = normalizeCode(roomCodeInput);
    if (cleanNickname.length < 1) {
      setError("사용할 닉네임을 입력해 주세요.");
      return;
    }
    if (entryMode === "join" && cleanCode.length !== 6) {
      setError("초대 코드 6자리를 확인해 주세요.");
      return;
    }
    setEntryBusy(true);
    setError(null);
    setConnection("connecting");
    try {
      const endpoint =
        entryMode === "create"
          ? "/api/online/rooms"
          : `/api/online/rooms/${cleanCode}/join`;
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ nickname: cleanNickname }),
      });
      const body: unknown = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(apiErrorMessage(body, "방에 입장하지 못했습니다."));
      }
      const source = record(body);
      const roomCode = normalizeCode(
        stringValue(source.roomCode, stringValue(source.code, cleanCode)),
      );
      const nextSession: StoredSession = {
        roomCode,
        playerId: stringValue(source.playerId),
        token: stringValue(source.token),
        nickname: cleanNickname,
      };
      if (
        nextSession.roomCode.length !== 6 ||
        !nextSession.playerId ||
        !nextSession.token
      ) {
        throw new Error("입장 정보를 확인하지 못했습니다. 다시 시도해 주세요.");
      }
      saveSession(nextSession);
      sessionRef.current = nextSession;
      setSession(nextSession);
      setLastSession(nextSession);
      setRoomCodeInput(nextSession.roomCode);
      setScreen("room");
      setConnection("online");
      const unwrapped = unwrapSnapshotResponse(body);
      if (unwrapped.snapshot) ingestSnapshot(unwrapped.snapshot);
      ingestChatMessages(
        unwrapped.chatMessages,
        unwrapped.latestChatSeq,
      );
      window.history.replaceState(
        null,
        "",
        `/online?room=${nextSession.roomCode}`,
      );
    } catch (reason) {
      setConnection("idle");
      setError(
        reason instanceof Error ? reason.message : "입장 중 오류가 발생했습니다.",
      );
    } finally {
      setEntryBusy(false);
    }
  };

  const resumeSession = (remembered: StoredSession) => {
    sessionRef.current = remembered;
    setSession(remembered);
    setNickname(remembered.nickname);
    setRoomCodeInput(remembered.roomCode);
    setScreen("room");
    setConnection("connecting");
    setError(null);
    setFatalError(false);
    window.history.replaceState(
      null,
      "",
      `/online?room=${remembered.roomCode}`,
    );
  };

  const sendCommand = useCallback(
    async (type: OnlineCommand["type"], payload: LooseRecord = {}) => {
      const activeSession = sessionRef.current;
      const current = snapshotRef.current;
      if (!activeSession || !current || busy) return;
      const commandId = createCommandId();
      const command = {
        id: commandId,
        commandId,
        type,
        expectedRevision: current.revision,
        baseRevision: current.revision,
        ...payload,
      };
      setBusy(true);
      setError(null);
      try {
        const response = await fetch(
          `/api/online/rooms/${activeSession.roomCode}/commands`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${activeSession.token}`,
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify({
              command,
              expectedRevision: current.revision,
            }),
          },
        );
        const body: unknown = await response.json().catch(() => ({}));
        if (sessionRef.current?.token !== activeSession.token) return;
        if (!response.ok) {
          throw new Error(apiErrorMessage(body, "행동을 처리하지 못했습니다."));
        }
        const result = unwrapSnapshotResponse(body);
        if (result.snapshot) ingestSnapshot(result.snapshot);
        ingestChatMessages(result.chatMessages, result.latestChatSeq);
        setSelectedIds([]);
        setConnection("online");
      } catch (reason) {
        setError(
          reason instanceof Error
            ? reason.message
            : "행동을 처리하지 못했습니다.",
        );
        void pollRoom();
      } finally {
        setBusy(false);
      }
    },
    [busy, ingestChatMessages, ingestSnapshot, pollRoom],
  );

  const chooseRankCard = useCallback(
    (slotIndex: number) => {
      const current = snapshotRef.current;
      if (
        rankChoiceInFlightRef.current !== null ||
        busy ||
        current?.phase !== "rank-selection" ||
        !current.rankSelection?.canChoose
      ) {
        return;
      }

      rankChoiceInFlightRef.current = slotIndex;
      setOptimisticRankSlotIndex(slotIndex);
      void sendCommand("CHOOSE_RANK_CARD", { slotIndex }).finally(() => {
        if (rankChoiceInFlightRef.current === slotIndex) {
          rankChoiceInFlightRef.current = null;
          setOptimisticRankSlotIndex(null);
        }
      });
    },
    [busy, sendCommand],
  );

  const sendChatMessage = useCallback(
    async (text: string) => {
      const activeSession = sessionRef.current;
      const current = snapshotRef.current;
      if (!activeSession || connection !== "online") {
        throw new Error("연결이 복구된 뒤 채팅을 보낼 수 있습니다.");
      }
      const messageId = createCommandId();
      const viewer =
        current?.players.find((player) => player.id === current.viewerId) ??
        null;
      const optimisticMessage: ChatMessageView = {
        seq: Number.MAX_SAFE_INTEGER,
        id: messageId,
        playerId: current?.viewerId ?? activeSession.playerId,
        authorName: viewer?.name ?? activeSession.nickname,
        text,
        sentAt: Date.now(),
      };
      setChatMessages((messages) => [
        ...messages.filter((message) => message.id !== messageId),
        optimisticMessage,
      ].slice(-ONLINE_CHAT_HISTORY_LIMIT));

      try {
        const response = await fetch(
          `/api/online/rooms/${activeSession.roomCode}/chat`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${activeSession.token}`,
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify({
              id: messageId,
              text,
            }),
          },
        );
        const body: unknown = await response.json().catch(() => ({}));
        if (sessionRef.current?.token !== activeSession.token) {
          throw new Error("방이 변경되어 메시지를 보내지 않았습니다.");
        }
        if (!response.ok) {
          if (response.status === 401 || response.status === 404) {
            setFatalError(true);
            setConnection("offline");
          }
          throw new Error(apiErrorMessage(body, "채팅을 보내지 못했습니다."));
        }
        const source = record(body);
        const message = chatMessageFrom(source.message);
        if (!message) {
          throw new Error("전송된 채팅을 확인하지 못했습니다.");
        }
        ingestChatMessages(
          [message],
          numberValue(source.latestChatSeq, message.seq),
          false,
        );
      } catch (reason) {
        setChatMessages((messages) =>
          messages.filter(
            (message) =>
              message.id !== messageId ||
              message.seq !== Number.MAX_SAFE_INTEGER,
          ),
        );
        throw reason;
      }
    },
    [connection, ingestChatMessages],
  );

  const me = useMemo(
    () =>
      snapshot?.players.find(
        (player) =>
          player.id === (snapshot.viewerId || session?.playerId),
      ) ?? null,
    [session?.playerId, snapshot],
  );
  const isHost = Boolean(me && snapshot?.hostId === me.id);
  const highestScore = Math.max(
    1,
    ...(snapshot?.players.map((player) => player.score) ?? [0]),
  );
  const isLobby = snapshot?.phase === "lobby";
  const isMyTurn = Boolean(
    snapshot &&
      me &&
      snapshot.phase === "playing" &&
      snapshot.currentPlayerId === me.id &&
      !me.finishedPlace,
  );
  const isTaxSelection = Boolean(
    snapshot &&
      snapshot.phase === "tax-selection" &&
      snapshot.requiredReturnCount > 0 &&
      snapshot.selectedReturnCount < snapshot.requiredReturnCount,
  );
  const isHandRevealing = Boolean(
    snapshot &&
      snapshot.hand !== null &&
      snapshot.phase === "hand-reveal" &&
      (snapshot.phaseEndsAt === null ||
        clock + serverOffset < snapshot.phaseEndsAt),
  );
  const isRankSelectionPhase = Boolean(
    snapshot &&
      ["rank-intro", "rank-selection", "rank-reveal", "rank-confirm"].includes(
        snapshot.phase,
      ) &&
      snapshot.rankSelection,
  );
  const hand = snapshot?.hand ?? [];
  const activeTaxVisualOverride =
    snapshot &&
    taxVisualOverride?.phase === snapshot.phase &&
    clock + serverOffset < taxVisualOverride.expiresAt
      ? taxVisualOverride
      : null;
  const renderedHandValue = activeTaxVisualOverride
    ? activeTaxVisualOverride.hand
    : snapshot?.hand ?? null;
  const renderedHand = renderedHandValue ?? [];
  const displayedMe =
    me && activeTaxVisualOverride
      ? {
          ...me,
          handCount:
            activeTaxVisualOverride.handCounts[me.id] ?? me.handCount,
        }
      : me;
  const isHandConcealed = Boolean(
    snapshot &&
      (snapshot.phase === "reveal-intro" || snapshot.hand === null),
  );
  const selectedCards = hand.filter((card) => selectedIds.includes(card.id));
  const selectedNormalRanks = [
    ...new Set(
      selectedCards.filter((card) => card.rank !== 13).map((card) => card.rank),
    ),
  ];
  const selectedRank =
    selectedNormalRanks[0] ??
    (selectedCards.length && selectedCards.every((card) => card.rank === 13)
      ? 13
      : null);
  const playError = useMemo(() => {
    if (!selectedCards.length) return "낼 카드를 선택하세요.";
    if (selectedNormalRanks.length > 1)
      return "같은 숫자의 카드와 어릿광대만 함께 낼 수 있습니다.";
    if (selectedCards.every((card) => card.rank === 13) && selectedCards.length > 1)
      return "어릿광대만 낼 때는 한 장만 선택하세요.";
    if (snapshot?.table) {
      if (selectedCards.length !== snapshot.table.count)
        return `카드 ${snapshot.table.count}장을 내야 합니다.`;
      if ((selectedRank ?? 13) >= snapshot.table.rank)
        return `${snapshot.table.rank}보다 낮은 숫자가 필요합니다.`;
    }
    return null;
  }, [selectedCards, selectedNormalRanks.length, selectedRank, snapshot?.table]);
  const effectiveClock = clock + serverOffset;
  const turnStartsAt =
    snapshot?.phase === "playing" && snapshot.turnDeadline !== null
      ? snapshot.turnDeadline - TURN_DURATION_MS
      : null;
  const turnRemainingMs =
    snapshot?.phase === "playing" &&
    snapshot.turnDeadline !== null &&
    turnStartsAt !== null &&
    effectiveClock >= turnStartsAt
      ? Math.max(0, snapshot.turnDeadline - effectiveClock)
      : null;
  const turnSecondsRemaining =
    turnRemainingMs === null ? null : Math.ceil(turnRemainingMs / 1000);
  const turnProgress =
    turnRemainingMs === null
      ? 0
      : Math.max(0, Math.min(1, turnRemainingMs / TURN_DURATION_MS));
  const turnUrgency =
    turnRemainingMs === null
      ? 0
      : Math.max(0, Math.min(1, (10_000 - turnRemainingMs) / 10_000));
  const turnAlertHue = 42 * (1 - turnUrgency);
  const currentTurnPlayer =
    snapshot?.players.find(
      (player) => player.id === snapshot.currentPlayerId,
    ) ?? null;
  const taxSecondsRemaining =
    snapshot?.phase === "tax-selection" && snapshot.phaseEndsAt !== null
      ? Math.max(
          0,
          Math.ceil((snapshot.phaseEndsAt - effectiveClock) / 1000),
        )
      : null;
  const actionLocked = Boolean(
    snapshot?.actionLockUntil &&
      effectiveClock < snapshot.actionLockUntil,
  );
  const taxSelectionValid =
    isTaxSelection &&
    selectedIds.length === (snapshot?.requiredReturnCount ?? 0) &&
    !busy;
  const allReady = Boolean(
    snapshot &&
      snapshot.players.length >= 4 &&
      snapshot.players.every((player) => player.ready && player.connected),
  );
  const taxObserverCopy = useMemo(() => {
    if (
      !snapshot ||
      snapshot.phase !== "tax-selection" ||
      isTaxSelection
    ) {
      return null;
    }
    const names = snapshot.waitingTaxPlayerIds.map((id) =>
      playerName(snapshot.players, id),
    );
    const subject =
      names.length > 2
        ? `${names[0]} 외 ${names.length - 1}명`
        : names.join(", ") || "상위 계급 플레이어";
    return `${subject}이(가) 세금 교환 중`;
  }, [isTaxSelection, snapshot]);
  const activeEvent = useMemo(() => {
    if (
      isHandRevealing ||
      snapshot?.phase === "tax-selection" ||
      isRankSelectionPhase
    ) {
      return null;
    }
    const candidates = [...eventBuffer]
      .filter(
        (event) =>
          [
            "MATCH_STARTED",
            "TAX_INTRO_STARTED",
            "TAX_TRIBUTE_STARTED",
            "TAX_TRIBUTE",
            "TAX_RETURN_STARTED",
            "TAX_RETURN",
            "PLAY_INTRO_STARTED",
            "CARDS_PLAYED",
            "DALMUTI_EFFECT",
            "REVOLUTION_INTRO_STARTED",
            "REVOLUTION_DECLARED",
            "GREAT_REVOLUTION_RANK_SWAP_STARTED",
            "PLAYER_PASSED",
          ].includes(event.type) &&
            !(
              event.type === "REVOLUTION_DECLARED" &&
              eventBuffer.some(
                (candidate) =>
                  candidate.type === "REVOLUTION_INTRO_STARTED" &&
                  Math.abs(candidate.startsAt - event.startsAt) < 2_000,
              )
            ) &&
            effectiveClock >= event.startsAt - 120 &&
            effectiveClock <= event.startsAt + event.durationMs,
      );
    return (
      [...candidates].reverse().find(
        (event) =>
          event.type === "GREAT_REVOLUTION_RANK_SWAP_STARTED",
      ) ??
      [...candidates].reverse().find(
        (event) => event.type === "DALMUTI_EFFECT",
      ) ??
      [...candidates].reverse().find(
        (event) => event.type === "REVOLUTION_INTRO_STARTED",
      ) ??
      [...candidates].reverse().find(
        (event) => event.type === "REVOLUTION_DECLARED",
      ) ??
      candidates.at(-1) ??
      null
    );
  }, [
    effectiveClock,
    eventBuffer,
    isHandRevealing,
    isRankSelectionPhase,
    snapshot?.phase,
  ]);
  const visibleTable = useMemo<TableView>(() => {
    if (
      !activeEvent ||
      !["CARDS_PLAYED", "DALMUTI_EFFECT"].includes(activeEvent.type)
    ) {
      return snapshot?.table ?? null;
    }
    const previousTable = record(activeEvent.data.previousTable);
    if (!Object.keys(previousTable).length) return null;
    const previousCards = cardsFrom(previousTable.cards);
    const rank = numberValue(previousTable.rank);
    const count = numberValue(previousTable.count, previousCards.length);
    if (rank < 1 || count < 1) return null;
    return {
      rank,
      count,
      playerId: stringValue(previousTable.playerId),
      cards: previousCards,
    };
  }, [activeEvent, snapshot?.table]);
  const motionLayoutKey = snapshot
    ? `${snapshot.phase}:${snapshot.players.map((player) => player.id).join("|")}`
    : "";
  const rankMotionKey = rankMovingPlayerIds.join("|");

  useLayoutEffect(() => {
    const root = tableColumnRef.current;
    const centerElement = tableCenterRef.current;
    if (!root || !centerElement) return;

    const measure = () => {
      const rootRect = root.getBoundingClientRect();
      const centerRect = centerElement.getBoundingClientRect();
      const players: Record<string, MotionPoint> = {};
      seatElementsRef.current.forEach((element, playerId) => {
        const rect = element.getBoundingClientRect();
        players[playerId] = {
          x: rect.left - rootRect.left + rect.width / 2,
          y: rect.top - rootRect.top + rect.height / 2,
        };
      });
      const nextAnchors: MotionAnchors = {
        players,
        center: {
          x: centerRect.left - rootRect.left + centerRect.width / 2,
          y: centerRect.top - rootRect.top + centerRect.height / 2,
        },
      };
      setMotionAnchors((current) =>
        motionAnchorsEqual(current, nextAnchors) ? current : nextAnchors,
      );
    };

    let pendingFrame: number | null = null;
    const scheduleMeasure = () => {
      if (pendingFrame !== null) return;
      pendingFrame = window.requestAnimationFrame(() => {
        pendingFrame = null;
        measure();
      });
    };

    measure();
    const observer = new ResizeObserver(scheduleMeasure);
    observer.observe(root);
    observer.observe(centerElement);
    seatElementsRef.current.forEach((element) => observer.observe(element));
    window.addEventListener("resize", scheduleMeasure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", scheduleMeasure);
      if (pendingFrame !== null) {
        window.cancelAnimationFrame(pendingFrame);
      }
    };
  }, [
    activeEvent?.id,
    isHandRevealing,
    motionLayoutKey,
    rankMotionKey,
  ]);

  const dalmutiHighlightPlayerId = useMemo(
    () =>
      activeEvent && snapshot
        ? dalmutiActorIdFromEvent(activeEvent, snapshot.players)
        : null,
    [activeEvent, snapshot],
  );
  const turnPresentationReady =
    snapshot?.phase === "playing" &&
    !actionLocked &&
    !activeEvent &&
    turnRemainingMs !== null;
  const showMyTurnHighlight =
    turnPresentationReady &&
    isMyTurn &&
    !busy &&
    connection === "online";
  const showUrgentTurnHighlight =
    showMyTurnHighlight &&
    turnRemainingMs !== null &&
    turnRemainingMs <= 10_000;
  const canPlay =
    isMyTurn &&
    !playError &&
    !busy &&
    !actionLocked &&
    !activeEvent &&
    connection === "online";
  const declaredRevolution =
    snapshot && snapshot.declaredRevolution?.round === snapshot.round
      ? snapshot.declaredRevolution
      : observedRevolution?.round === snapshot?.round
        ? observedRevolution
        : null;
  const greatRevolutionActive = ["great", "great-revolution"].includes(
    declaredRevolution?.kind ?? "",
  );
  // The announcement event is intentionally short-lived, but the revolution
  // changes the visual state of the whole round. Drive the field from the
  // canonical round declaration (or its observed reconnect fallback), not the
  // currently playing event overlay.
  const revolutionFieldActive = Boolean(declaredRevolution);
  const sortedFinishers = useMemo(() => {
    if (!snapshot) return [];
    const ids = snapshot.finishOrder.length
      ? snapshot.finishOrder
      : [...snapshot.players]
          .filter((player) => player.finishedPlace)
          .sort(
            (a, b) =>
              (a.finishedPlace ?? Number.MAX_SAFE_INTEGER) -
              (b.finishedPlace ?? Number.MAX_SAFE_INTEGER),
          )
          .map((player) => player.id);
    return ids
      .map((id) => snapshot.players.find((player) => player.id === id))
      .filter((player): player is PlayerView => Boolean(player));
  }, [snapshot]);
  const tableRankedPlayers = useMemo(
    () =>
      snapshot?.phase === "round-end" &&
      sortedFinishers.length === snapshot.players.length
        ? sortedFinishers
        : (snapshot?.players ?? []),
    [snapshot, sortedFinishers],
  );
  const rankedOpponents = useMemo(
    () =>
      tableRankedPlayers
        .map((player, rankIndex) => ({ player, rankIndex }))
        .filter(({ player }) => player.id !== me?.id),
    [me?.id, tableRankedPlayers],
  );

  useEffect(() => {
    if (
      snapshot?.phase !== "great-revolution-swap" ||
      pendingGreatRevolutionMoveIds === null
    ) {
      return;
    }

    const movingPlayerIds = pendingGreatRevolutionMoveIds;
    const startTimer = window.setTimeout(() => {
      setSeatRankOverrides(null);
      setPendingGreatRevolutionMoveIds(null);
      setRankMovingPlayerIds(movingPlayerIds);
      if (rankMoveTimerRef.current !== null) {
        window.clearTimeout(rankMoveTimerRef.current);
      }
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      const remainingMs =
        snapshot.phaseEndsAt === null
          ? RANK_MOVE_DURATION_MS
          : Math.max(
              120,
              Math.min(
                RANK_MOVE_DURATION_MS,
                snapshot.phaseEndsAt - (Date.now() + serverOffset),
              ),
            );
      rankMoveTimerRef.current = window.setTimeout(() => {
        setRankMovingPlayerIds([]);
        rankMoveTimerRef.current = null;
      }, reduceMotion ? 80 : remainingMs);
    }, 40);

    return () => window.clearTimeout(startTimer);
  }, [
    pendingGreatRevolutionMoveIds,
    serverOffset,
    snapshot?.phase,
    snapshot?.phaseEndsAt,
  ]);

  useEffect(() => {
    if (
      snapshot?.phase !== "round-end" ||
      pendingRoundEndMoveIds === null ||
      activeEvent ||
      actionLocked
    ) {
      return;
    }

    const movingPlayerIds = pendingRoundEndMoveIds;
    const startTimer = window.setTimeout(() => {
      setPendingRoundEndMoveIds(null);
      if (!movingPlayerIds.length) {
        window.setTimeout(
          () => setRoundEndResultReady(true),
          ROUND_END_MOVE_SETTLE_MS,
        );
        return;
      }

      setSeatRankOverrides(null);
      setRankMovingPlayerIds(movingPlayerIds);
      if (rankMoveTimerRef.current !== null) {
        window.clearTimeout(rankMoveTimerRef.current);
      }
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      rankMoveTimerRef.current = window.setTimeout(() => {
        setRankMovingPlayerIds([]);
        setRoundEndResultReady(true);
        rankMoveTimerRef.current = null;
      }, reduceMotion
        ? 120
        : RANK_MOVE_DURATION_MS + ROUND_END_MOVE_SETTLE_MS);
    }, ROUND_END_MOVE_PRELUDE_MS);
    return () => window.clearTimeout(startTimer);
  }, [
    actionLocked,
    activeEvent,
    pendingRoundEndMoveIds,
    snapshot?.phase,
  ]);

  useLayoutEffect(() => {
    const nextRects = new Map<string, DOMRect>();
    seatElementsRef.current.forEach((element, playerId) => {
      nextRects.set(playerId, element.getBoundingClientRect());
    });

    const useGridFlip =
      !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (useGridFlip && rankMovingPlayerIds.length) {
      for (const playerId of rankMovingPlayerIds) {
        const element = seatElementsRef.current.get(playerId);
        const previousRect = seatRectsRef.current.get(playerId);
        const nextRect = nextRects.get(playerId);
        if (!element || !previousRect || !nextRect) continue;
        const deltaX = previousRect.left - nextRect.left;
        const deltaY = previousRect.top - nextRect.top;
        if (Math.abs(deltaX) < 1 && Math.abs(deltaY) < 1) continue;
        element.animate(
          [
            {
              translate: `${deltaX}px ${deltaY}px`,
              scale: "1",
              filter: "brightness(1)",
            },
            {
              offset: 0.3,
              translate: `${deltaX * 0.7 + Math.sign(deltaY || 1) * 34}px ${
                deltaY * 0.7 - 22
              }px`,
              scale: "1.07",
              filter: "brightness(1.32)",
            },
            {
              offset: 0.72,
              translate: `${deltaX * 0.24 - Math.sign(deltaY || 1) * 18}px ${
                deltaY * 0.24 + 8
              }px`,
              scale: "1.035",
              filter: "brightness(1.16)",
            },
            {
              translate: "0 0",
              scale: "1",
              filter: "brightness(1)",
            },
          ],
          {
            duration: RANK_MOVE_DURATION_MS,
            easing: "cubic-bezier(0.16, 0.74, 0.2, 1)",
          },
        );
      }
    }
    seatRectsRef.current = nextRects;
  }, [rankMovingPlayerIds, seatRankOverrides, tableRankedPlayers]);

  const toggleCard = (cardId: string) => {
    if ((!isMyTurn || !turnPresentationReady) && !isTaxSelection) return;
    setSelectedIds((current) =>
      current.includes(cardId)
        ? current.filter((id) => id !== cardId)
        : [...current, cardId],
    );
  };

  const toggleRank = (rank: number) => {
    if ((!isMyTurn || !turnPresentationReady) && !isTaxSelection) return;
    const rankIds = hand
      .filter((card) => card.rank === rank)
      .map((card) => card.id);
    setSelectedIds((current) => {
      const allSelected = rankIds.every((id) => current.includes(id));
      return allSelected
        ? current.filter((id) => !rankIds.includes(id))
        : [...new Set([...current, ...rankIds])];
    });
  };

  const copyInvite = async () => {
    if (!snapshot?.code && !session?.roomCode) return;
    const code = snapshot?.code || session!.roomCode;
    const inviteUrl = `${window.location.origin}/online?room=${code}`;
    try {
      await navigator.clipboard.writeText(inviteUrl);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setError(`초대 링크: ${inviteUrl}`);
    }
  };

  const finishLocalExit = () => {
    const leavingSession = sessionRef.current;
    clearSavedSession(leavingSession);
    setScreen("entry");
    setSession(null);
    setLastSession(null);
    sessionRef.current = null;
    setSnapshot(null);
    snapshotRef.current = null;
    setSelectedIds([]);
    setEventBuffer([]);
    setChatMessages([]);
    latestChatSeqRef.current = 0;
    setTaxVisualOverride(null);
    setRankMovingPlayerIds([]);
    setSeatRankOverrides(null);
    setPendingRoundEndMoveIds(null);
    setPendingGreatRevolutionMoveIds(null);
    setRoundEndResultReady(true);
    rankChoiceInFlightRef.current = null;
    setOptimisticRankSlotIndex(null);
    setBotDifficultyPickerSlot(null);
    if (rankMoveTimerRef.current !== null) {
      window.clearTimeout(rankMoveTimerRef.current);
      rankMoveTimerRef.current = null;
    }
    setObservedRevolution(null);
    setConnection("idle");
    setFatalError(false);
    setError(null);
    setBusy(false);
    failureCountRef.current = 0;
    setEntryMode("create");
    setRoomCodeInput("");
    window.history.replaceState(null, "", "/online");
  };

  const exitRoom = async (goHome = false) => {
    const activeSession = sessionRef.current;
    const current = snapshotRef.current;
    if (!activeSession || !current || fatalError) {
      finishLocalExit();
      if (goHome) router.push("/");
      return;
    }
    const confirmed = window.confirm(
      isHost
        ? "방을 초기화하면 현재 방과 모든 참가자의 접속 정보가 삭제됩니다. 계속할까요?"
        : "현재 방에서 나가면 진행 중인 막은 대기실로 초기화됩니다. 계속할까요?",
    );
    if (!confirmed) return;

    setBusy(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/online/rooms/${activeSession.roomCode}/${
          isHost ? "reset" : "leave"
        }`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${activeSession.token}`,
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            id: createCommandId(),
          }),
        },
      );
      const body: unknown = await response.json().catch(() => ({}));
      if (sessionRef.current?.token !== activeSession.token) return;
      if (!response.ok) {
        if (response.status === 401 || response.status === 404) {
          finishLocalExit();
          if (goHome) router.push("/");
          return;
        }
        throw new Error(
          apiErrorMessage(
            body,
            isHost
              ? "방을 초기화하지 못했습니다."
              : "방에서 나가지 못했습니다.",
          ),
        );
      }
      finishLocalExit();
      if (goHome) router.push("/");
    } catch (reason) {
      if (sessionRef.current?.token !== activeSession.token) return;
      setError(
        reason instanceof Error
          ? reason.message
          : "퇴장 요청을 처리하지 못했습니다.",
      );
    } finally {
      if (sessionRef.current?.token === activeSession.token) {
        setBusy(false);
      }
    }
  };

  if (screen === "entry") {
    return (
      <main className={styles.entryShell}>
        <div className={styles.grain} />
        <header className={styles.entryHeader}>
          <Brand />
          <Link className={styles.soloLink} href="/">
            혼자 하기
          </Link>
        </header>

        <section className={styles.entryHero}>
          <div className={styles.heroCopy}>
            <span className={styles.eyebrow}>REAL-TIME • 4–8 PLAYERS</span>
            <h1>DALMUTI</h1>
            <div className={styles.heroCards} aria-hidden="true">
              {[1, 2, 13].map((rank, index) => (
                <span
                  key={rank}
                  style={{ "--hero-card": index } as CSSProperties}
                >
                  <img src={cardImage(rank)} alt="" />
                </span>
              ))}
            </div>
          </div>

          <form className={styles.entryCard} onSubmit={submitEntry}>
            <span className={styles.entryCrown} aria-hidden="true" />
            <div className={styles.modeTabs}>
              <button
                type="button"
                className={entryMode === "create" ? styles.activeTab : ""}
                onClick={() => setEntryMode("create")}
              >
                방 만들기
              </button>
              <button
                type="button"
                className={entryMode === "join" ? styles.activeTab : ""}
                onClick={() => setEntryMode("join")}
              >
                코드로 참가
              </button>
            </div>
            <div className={styles.formHeading}>
              <small>{entryMode === "create" ? "CREATE A ROOM" : "JOIN A ROOM"}</small>
              <h2>
                {entryMode === "create"
                  ? "새로운 계급전"
                  : "초대받은 계급전"}
              </h2>
              <p>
                {entryMode === "create"
                  ? "당신이 방장이 되어 게임을 시작합니다."
                  : "방장에게 받은 6자리 코드를 입력하세요."}
              </p>
            </div>
            <label className={styles.field}>
              <span>닉네임</span>
              <input
                value={nickname}
                onChange={(event) => setNickname(event.target.value.slice(0, 16))}
                placeholder="랩실에서 부르는 이름"
                autoComplete="nickname"
                maxLength={16}
              />
              <small>{nickname.trim().length}/16</small>
            </label>
            {entryMode === "join" && (
              <label className={`${styles.field} ${styles.codeField}`}>
                <span>초대 코드</span>
                <input
                  value={roomCodeInput}
                  onChange={(event) =>
                    setRoomCodeInput(normalizeCode(event.target.value))
                  }
                  placeholder="ABC123"
                  autoComplete="one-time-code"
                  inputMode="text"
                  maxLength={6}
                />
                <small>{roomCodeInput.length}/6</small>
              </label>
            )}
            {error && <p className={styles.formError}>{error}</p>}
            <button
              type="submit"
              className={styles.primaryButton}
              disabled={entryBusy}
            >
              <span>
                {entryBusy
                  ? "입장 정보를 확인하는 중"
                  : entryMode === "create"
                    ? "온라인 방 만들기"
                    : "계급전에 참가하기"}
              </span>
              <b>{entryBusy ? "···" : "→"}</b>
            </button>
            {lastSession && (
              <button
                type="button"
                className={styles.resumeButton}
                onClick={() => resumeSession(lastSession)}
              >
                <span>
                  <b>{lastSession.roomCode}</b>
                  {lastSession.nickname}으로 이어하기
                </span>
                <i>재접속</i>
              </button>
            )}
            <p className={styles.entryNote}>
              계정 없이 바로 시작할 수 있습니다. 재접속 정보는 이 기기에만
              저장됩니다.
            </p>
          </form>
        </section>
      </main>
    );
  }

  const displayCode = snapshot?.code || session?.roomCode || "------";

  if (!snapshot || isLobby) {
    return (
      <main className={styles.lobbyShell}>
        <div className={styles.grain} />
        <header className={styles.roomHeader}>
          <Brand
            onActivate={() => void exitRoom(true)}
            disabled={busy || !snapshot}
          />
          <div className={styles.headerRoom}>
            <span>ROOM</span>
            <strong>{displayCode}</strong>
            <button type="button" onClick={copyInvite}>
              {copied ? "복사됨" : "초대 링크"}
            </button>
          </div>
          <div className={styles.headerActions}>
            <ConnectionPill state={connection} />
            <button
              type="button"
              onClick={() => void exitRoom()}
              disabled={busy}
              aria-label={isHost ? "방 초기화 후 나가기" : "방 나가기"}
            >
              {busy ? "처리 중" : isHost ? "방 초기화" : "방 나가기"}
            </button>
          </div>
        </header>

        <section className={styles.lobbyLayout}>
          <div className={styles.lobbyIntro}>
            <span className={styles.eyebrow}>THE TABLE IS OPEN</span>
            <h1>
              플레이어를
              <br />
              기다리는 중
            </h1>
            <p>
              참가자와 봇을 합쳐 최소 4명이 준비되면 방장이 PLAY를
              누를 수 있습니다.
            </p>
            <button
              type="button"
              className={styles.inviteCard}
              onClick={copyInvite}
            >
              <span>
                <small>초대 코드</small>
                <strong>{displayCode}</strong>
              </span>
              <i>{copied ? "링크를 복사했습니다" : "링크 복사 →"}</i>
            </button>
            {snapshot && (
              <OnlineChatPanel
                className={styles.lobbyChatPanel}
                messages={chatMessages}
                viewerId={snapshot.viewerId}
                connected={connection === "online"}
                onSend={sendChatMessage}
              />
            )}
          </div>

          <section className={styles.lobbyPlayers}>
            <div className={styles.lobbyHeading}>
              <div>
                <small>PLAYERS</small>
                <h2>계급전 참가자</h2>
              </div>
              <span>{snapshot?.players.length ?? 0} / 8</span>
            </div>
            <ol>
              {Array.from({ length: 8 }, (_, index) => {
                const player = snapshot?.players[index];
                if (!player) {
                  return (
                    <li
                      className={`${styles.emptySlot} ${
                        isHost ? styles.emptySlotInteractive : ""
                      }`}
                      key={`empty-${index}`}
                    >
                      <span>{index + 1}</span>
                      <p>빈 자리</p>
                      <em>
                        {isHost
                          ? "클릭하여 봇 추가"
                          : "초대 링크로 참가할 수 있습니다"}
                      </em>
                      {isHost && (
                        <button
                          type="button"
                          className={styles.slotOverlayButton}
                          disabled={busy || connection !== "online"}
                          onClick={() => setBotDifficultyPickerSlot(index)}
                          aria-label={`${index + 1}번 빈 자리에 봇 추가`}
                        />
                      )}
                    </li>
                  );
                }
                return (
                  <li
                    className={`${styles.lobbyPlayer} ${
                      player.id === me?.id ? styles.lobbyPlayerSelf : ""
                    } ${player.isBot ? styles.lobbyPlayerBot : ""}`}
                    key={player.id}
                  >
                    <span className={styles.avatar}>
                      {player.monogram}
                      <i>?</i>
                    </span>
                    <p>
                      <strong>{player.name}</strong>
                      <small>
                        {player.isBot
                          ? `봇 · ${
                              BOT_DIFFICULTY_LABELS[
                                player.botDifficulty ?? "normal"
                              ]
                            } · 자동 준비`
                          : player.id === snapshot?.hostId
                          ? "방장"
                          : player.id === me?.id
                            ? "나"
                            : "참가자"}
                      </small>
                    </p>
                    {snapshot?.dealSealed && player.handCount > 0 ? (
                      <span
                        className={styles.sealedHand}
                        aria-label={`뒤집힌 패 ${player.handCount}장`}
                      >
                        <i />
                        <i />
                        <i />
                        <b>{player.handCount}장</b>
                      </span>
                    ) : (
                      <em
                        className={
                          player.ready ? styles.readyBadge : styles.waitBadge
                        }
                      >
                        {player.isBot
                          ? `BOT · ${
                              BOT_DIFFICULTY_LABELS[
                                player.botDifficulty ?? "normal"
                              ]
                            } · 준비 완료`
                          : player.ready
                            ? "준비 완료"
                            : "준비 중"}
                      </em>
                    )}
                    {!player.connected && (
                      <i className={styles.disconnectedBadge}>연결 끊김</i>
                    )}
                    {isHost && player.isBot && (
                      <>
                        <i className={styles.botRemoveHint}>클릭해 삭제</i>
                        <button
                          type="button"
                          className={styles.slotOverlayButton}
                          disabled={busy || connection !== "online"}
                          onClick={() =>
                            void sendCommand("REMOVE_BOT", {
                              botId: player.id,
                            })
                          }
                          aria-label={`${player.name} 삭제`}
                        />
                      </>
                    )}
                  </li>
                );
              })}
            </ol>
            <div className={styles.lobbyControls}>
              <button
                type="button"
                className={`${styles.readyButton} ${
                  me?.ready ? styles.readyButtonOn : ""
                }`}
                disabled={!me || busy}
                onClick={() =>
                  void sendCommand("SET_READY", { ready: !me?.ready })
                }
              >
                <span>{me?.ready ? "준비 취소" : "준비하기"}</span>
                <small>
                  {me?.ready ? "방장이 시작하기를 기다립니다" : "게임 참가 준비"}
                </small>
              </button>
              {isHost && (
                <button
                  type="button"
                  className={styles.playButton}
                  disabled={!allReady || busy}
                  onClick={() => void sendCommand("START_MATCH")}
                >
                  <span>PLAY</span>
                  <small>
                    {allReady
                      ? "첫 계급 정하기 시작"
                      : "4명 이상(봇 포함) · 모두 준비 필요"}
                  </small>
                </button>
              )}
            </div>
            {error && <p className={styles.roomError}>{error}</p>}
          </section>
        </section>

        {botDifficultyPickerSlot !== null && isHost && (
          <div
            className={styles.botDifficultyLayer}
            role="presentation"
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) {
                setBotDifficultyPickerSlot(null);
              }
            }}
          >
            <section
              className={styles.botDifficultyDialog}
              role="dialog"
              aria-modal="true"
              aria-labelledby="bot-difficulty-title"
            >
              <small>EMPTY SLOT {botDifficultyPickerSlot + 1}</small>
              <h2 id="bot-difficulty-title">봇 난이도를 선택하세요</h2>
              <p>
                추가된 봇은 즉시 준비 완료 상태가 되며, 선택한 수준으로
                게임을 판단합니다.
              </p>
              <div>
                {BOT_DIFFICULTIES.map((difficulty) => (
                  <button
                    key={difficulty}
                    type="button"
                    disabled={busy || connection !== "online"}
                    onClick={async () => {
                      setBotDifficultyPickerSlot(null);
                      await sendCommand("ADD_BOT", { difficulty });
                    }}
                  >
                    <strong>{BOT_DIFFICULTY_LABELS[difficulty]}</strong>
                    <span>{BOT_DIFFICULTY_DESCRIPTIONS[difficulty]}</span>
                  </button>
                ))}
              </div>
              <button
                type="button"
                className={styles.botDifficultyCancel}
                onClick={() => setBotDifficultyPickerSlot(null)}
              >
                취소
              </button>
            </section>
          </div>
        )}

        {connection !== "online" && (
          <div className={styles.reconnectOverlay}>
            <span className={styles.spinner} />
            <small>REAL-TIME CONNECTION</small>
            <strong>
              {fatalError ? "방에 다시 들어갈 수 없습니다" : "연결을 복구하는 중"}
            </strong>
            <p>{error ?? "잠시만 기다려 주세요."}</p>
            <div>
              {!fatalError && (
                <button type="button" onClick={() => void pollRoom()}>
                  지금 다시 연결
                </button>
              )}
              <button
                type="button"
                onClick={() =>
                  fatalError ? finishLocalExit() : void exitRoom()
                }
              >
                입장 화면으로
              </button>
            </div>
          </div>
        )}
      </main>
    );
  }

  return (
    <main className={styles.gameShell}>
      <div className={styles.grain} />
      <header className={styles.roomHeader}>
        <Brand
          onActivate={() => void exitRoom(true)}
          disabled={busy}
        />
        <div className={styles.roundInfo}>
          <span>제 {snapshot.round}막</span>
          <i />
          <span>{snapshot.players.length}인</span>
        </div>
        <div className={styles.headerActions}>
          <ConnectionPill state={connection} />
          <button type="button" onClick={copyInvite}>
            {displayCode}
          </button>
          <button
            type="button"
            onClick={() => void exitRoom()}
            disabled={busy}
            aria-label={isHost ? "방 초기화 후 나가기" : "방 나가기"}
          >
            {busy ? "처리 중" : isHost ? "방 초기화" : "방 나가기"}
          </button>
        </div>
      </header>

      <section className={styles.gameLayout}>
        <aside className={styles.rankRail}>
          <div className={styles.railHeading}>
            <span>서열</span>
            <small>누적 점수</small>
          </div>
          <ol>
            {snapshot.players.map((player) => {
              const chipCount = scoreChipCount(
                player.score,
                highestScore,
              );
              return (
                <li
                  key={player.id}
                  className={player.id === me?.id ? styles.rankSelf : ""}
                >
                  <span>
                    {isRankSelectionPhase ? "·" : roleMark(player.role)}
                  </span>
                  <p>
                    <strong>{player.name}</strong>
                    <small>
                      {isRankSelectionPhase
                        ? "계급 미정"
                        : roleLabel(player.role)}
                    </small>
                  </p>
                  <em
                    className={styles.scoreDisplay}
                    aria-label={`${player.name} 누적 점수 ${player.score}점`}
                  >
                    <span
                      className={styles.scoreChipStack}
                      aria-hidden="true"
                    >
                      {Array.from(
                        { length: chipCount },
                        (_, chipIndex) => (
                          <i
                            key={chipIndex}
                            style={
                              {
                                "--score-chip-index": chipIndex,
                              } as CSSProperties
                            }
                          />
                        ),
                      )}
                    </span>
                    <span>{player.score}</span>
                  </em>
                </li>
              );
            })}
          </ol>
          <div className={styles.railNote}>
            <span>계급의 법칙</span>
            <p>숫자가 낮을수록 강합니다. 더 강하게 맞서세요.</p>
          </div>
        </aside>

        <section
          ref={tableColumnRef}
          className={`${styles.boardColumn} ${
            isRankSelectionPhase ? styles.boardColumnRankSelection : ""
          } ${showMyTurnHighlight ? styles.boardColumnMyTurn : ""
          } ${showUrgentTurnHighlight ? styles.boardColumnTurnUrgent : ""
          }`}
          style={
            {
              "--turn-alert-hue": turnAlertHue,
              "--turn-alert-strength": turnUrgency,
              "--turn-alert-alpha": 0.24 + turnUrgency * 0.5,
              "--turn-pulse-duration": `${Math.round(
                920 - turnUrgency * 430,
              )}ms`,
            } as CSSProperties
          }
        >
          {turnSecondsRemaining !== null && currentTurnPlayer && (
            <div
              className={`${styles.turnCountdown} ${
                isMyTurn ? styles.turnCountdownSelf : ""
              } ${
                turnSecondsRemaining <= 10
                  ? styles.turnCountdownUrgent
                  : ""
              }`}
              style={
                {
                  "--turn-angle": `${turnProgress * 360}deg`,
                  "--turn-urgency": turnUrgency,
                  "--turn-accent-hue": turnAlertHue,
                } as CSSProperties
              }
              role="timer"
              aria-label={`${currentTurnPlayer.name}의 차례, ${turnSecondsRemaining}초 남음`}
            >
              <div className={styles.turnCountdownRing} aria-hidden="true">
                <span>
                  <b>{turnSecondsRemaining}</b>
                  <small>SEC</small>
                </span>
              </div>
              <p className={styles.turnCountdownCopy}>
                {isMyTurn ? "내 차례" : `${currentTurnPlayer.name}의 차례`}
              </p>
            </div>
          )}
          <div
            className={`${styles.table} ${
              revolutionFieldActive ? styles.tableRevolution : ""
            } ${
              greatRevolutionActive ? styles.tableGreatRevolution : ""
            } ${isRankSelectionPhase ? styles.tableRankSelection : ""} ${
              showMyTurnHighlight ? styles.tableMyTurn : ""
            } ${showUrgentTurnHighlight ? styles.tableMyTurnUrgent : ""} ${
              dalmutiHighlightPlayerId ? styles.tableDalmutiBurst : ""
            }`}
            aria-label={
              greatRevolutionActive
                ? showMyTurnHighlight
                  ? "대혁명 진행 중, 내 차례입니다"
                  : "대혁명 진행 중"
                : showMyTurnHighlight
                  ? "내 차례입니다"
                  : undefined
            }
          >
            <div className={styles.tableLine} aria-hidden="true">
              <span>♜</span>
              <i />
              <span>♞</span>
              <i />
              <span>♝</span>
            </div>
            {greatRevolutionActive && (
              <div
                className={styles.greatRevolutionFieldEffect}
                aria-hidden="true"
              >
                <i />
                <i />
                {Array.from({ length: 14 }, (_, index) => (
                  <span
                    key={`great-revolution-ember-${index}`}
                    style={
                      {
                        "--ember-x": `${8 + ((index * 17) % 84)}%`,
                        "--ember-y": `${12 + ((index * 23) % 74)}%`,
                        "--ember-size": `${3 + (index % 3)}px`,
                        "--ember-delay": `${(index % 7) * -0.52}s`,
                        "--ember-duration": `${
                          3.2 + (index % 5) * 0.42
                        }s`,
                      } as CSSProperties
                    }
                  />
                ))}
              </div>
            )}
            <div className={styles.seatRing}>
              {rankedOpponents.map(({ player, rankIndex }) => {
                const displayedPlayer = activeTaxVisualOverride
                  ? {
                      ...player,
                      handCount:
                        activeTaxVisualOverride.handCounts[player.id] ??
                        player.handCount,
                    }
                  : player;
                const displayedRankIndex =
                  seatRankOverrides?.[player.id] ?? rankIndex;
                const priorRankIndex = snapshot.players.findIndex(
                  (candidate) => candidate.id === player.id,
                );
                const movementOriginRankIndex =
                  seatRankOverrides?.[player.id] ?? priorRankIndex;
                const movementDirection =
                  movementOriginRankIndex > rankIndex
                    ? "up"
                    : movementOriginRankIndex < rankIndex
                      ? "down"
                      : null;
                return (
                  <PlayerSeat
                    key={player.id}
                    player={displayedPlayer}
                    isHost={player.id === snapshot.hostId}
                    isCurrent={
                      turnPresentationReady &&
                      !dalmutiHighlightPlayerId &&
                      player.id === snapshot.currentPlayerId
                    }
                    rankNumber={rankIndex + 1}
                    isRankMoving={rankMovingPlayerIds.includes(player.id)}
                    rankMovement={movementDirection}
                    showHandBacks={player.handCount > 0}
                    isHandRevealing={isHandRevealing}
                    handRevealElapsedMs={handRevealElapsedMs}
                    isDalmutiHighlighted={
                      player.id === dalmutiHighlightPlayerId
                    }
                    roleHidden={isRankSelectionPhase}
                    elementRef={(element) =>
                      bindSeatElement(player.id, element)
                    }
                    style={seatPosition(
                      displayedRankIndex,
                      tableRankedPlayers.length,
                    )}
                  />
                );
              })}
            </div>
            {rankMovingPlayerIds.length > 0 && (
              <div
                className={styles.rankShiftEffect}
                aria-hidden="true"
              >
                <i />
                <i />
                {Array.from({ length: 10 }, (_, index) => (
                  <span
                    key={`rank-transition-spark-${index}`}
                    style={
                      {
                        "--transition-spark-y": `${16 + index * 7}%`,
                        "--transition-spark-delay": `${index * 90}ms`,
                      } as CSSProperties
                    }
                  />
                ))}
              </div>
            )}
            <div ref={tableCenterRef} className={styles.tableCenter}>
              {isRankSelectionPhase && snapshot.rankSelection ? (
                <RankSelectionField
                  key={`${snapshot.rankSelection.stage}:${
                    snapshot.phaseEndsAt ??
                    snapshot.rankSelection.countdownEndsAt ??
                    0
                  }`}
                  rankSelection={snapshot.rankSelection}
                  players={snapshot.players}
                  viewerId={snapshot.viewerId}
                  optimisticSlotIndex={optimisticRankSlotIndex}
                  effectiveClock={effectiveClock}
                  phaseEndsAt={snapshot.phaseEndsAt}
                  busy={busy}
                  onChoose={chooseRankCard}
                />
              ) : snapshot.phase === "revolution" &&
              !snapshot.canChooseRevolution ? (
                <div
                  className={styles.taxWaitingField}
                  role="status"
                  aria-live="polite"
                >
                  <span aria-hidden="true">◇</span>
                  <small>REVOLUTION DECISION</small>
                  <strong>한 플레이어가 혁명 여부를 결정 중</strong>
                  <p>결정이 끝나면 세금 교환 또는 게임 시작으로 이어집니다</p>
                </div>
              ) : isTaxSelection ? (
                <div
                  className={styles.taxDecisionField}
                  role="status"
                  aria-live="polite"
                >
                  <span aria-hidden="true">↕</span>
                  <small>PRIVATE TAX RETURN</small>
                  <strong>
                    돌려줄 카드 {snapshot.requiredReturnCount}장을 선택하세요
                  </strong>
                  {taxSecondsRemaining !== null && (
                    <b className={styles.taxDeadline}>
                      <span>남은 시간</span>
                      {taxSecondsRemaining}초
                    </b>
                  )}
                  <p>
                    내 패에서 원하는 카드를 고른 뒤 반환 확정을 누르세요.
                    시간이 끝나면 가장 낮은 가치의 카드부터 자동 반환됩니다.
                  </p>
                </div>
              ) : taxObserverCopy ? (
                <div
                  className={styles.taxWaitingField}
                  role="status"
                  aria-live="polite"
                >
                  <span aria-hidden="true">◇</span>
                  <small>TAX EXCHANGE</small>
                  <strong>{taxObserverCopy}</strong>
                  {taxSecondsRemaining !== null && (
                    <b className={styles.taxDeadline}>
                      <span>남은 시간</span>
                      {taxSecondsRemaining}초
                    </b>
                  )}
                  <p>
                    카드의 정체는 교환 당사자에게만 공개됩니다. 시간이 끝나면
                    미선택 상위 계급 플레이어의 가장 낮은 가치 카드가 자동
                    반환됩니다.
                  </p>
                </div>
              ) : snapshot.phase === "hand-reveal" ? (
                <div
                  className={styles.taxWaitingField}
                  role="status"
                  aria-live="polite"
                >
                  <small>HAND REVEAL</small>
                  <strong>패를 확인하는 중</strong>
                  <p>패 공개가 끝나면 세금 교환을 시작합니다</p>
                </div>
              ) : visibleTable?.cards.length ? (
                <>
                  <small>마지막으로 놓인 카드</small>
                  <div
                    className={styles.tableCards}
                    style={
                      {
                        "--table-card-step-wide": `${Math.min(
                          66,
                          420 /
                            Math.max(1, visibleTable.cards.length - 1),
                        )}px`,
                        "--table-card-step-medium": `${Math.min(
                          58,
                          330 /
                            Math.max(1, visibleTable.cards.length - 1),
                        )}px`,
                        "--table-card-step-small": `${Math.min(
                          51,
                          170 /
                            Math.max(1, visibleTable.cards.length - 1),
                        )}px`,
                      } as CSSProperties
                    }
                  >
                    {visibleTable.cards.map((card, index) => {
                      const offset =
                        index - (visibleTable.cards.length - 1) / 2;
                      return (
                        <span
                          key={card.id}
                          style={
                            {
                              "--table-card-offset": offset,
                              "--table-card-lift": `${Math.abs(offset) * 1.25}px`,
                            } as CSSProperties
                          }
                        >
                          <PlayingCard card={card} displayOnly />
                        </span>
                      );
                    })}
                  </div>
                  <strong>
                    {formatRank(visibleTable.rank)} × {visibleTable.count}장
                  </strong>
                  <p>
                    {visibleTable.rank}보다 낮은 숫자의 카드{" "}
                    {visibleTable.count}장을 내세요
                  </p>
                </>
              ) : (
                <div className={styles.emptyTable}>
                  <span aria-hidden="true">◇</span>
                  <strong>비어 있는 필드</strong>
                  <small>
                    {turnPresentationReady
                      ? `${playerName(
                          snapshot.players,
                          snapshot.currentPlayerId,
                        )}이(가) 새 묶음을 시작합니다`
                      : "이전 행동을 정리하고 있습니다"}
                  </small>
                </div>
              )}
            </div>
          </div>

          <section
            className={`${styles.ownDock} ${
              isTaxSelection ? styles.ownDockTaxSelection : ""
            } ${
              me?.finishedPlace && hand.length === 0
                ? styles.ownDockFinished
                : ""
            }`}
          >
            {me && (
              <PlayerSeat
                player={displayedMe ?? me}
                isSelf
                isHost={isHost}
                isCurrent={
                  turnPresentationReady &&
                  isMyTurn &&
                  !dalmutiHighlightPlayerId
                }
                rankNumber={
                  tableRankedPlayers.findIndex((player) => player.id === me.id) +
                  1
                }
                isRankMoving={rankMovingPlayerIds.includes(me.id)}
                rankMovement={(() => {
                  const priorRankIndex =
                    seatRankOverrides?.[me.id] ??
                    snapshot.players.findIndex(
                      (player) => player.id === me.id,
                    );
                  const nextRankIndex = tableRankedPlayers.findIndex(
                    (player) => player.id === me.id,
                  );
                  return priorRankIndex > nextRankIndex
                    ? "up"
                    : priorRankIndex < nextRankIndex
                      ? "down"
                      : null;
                })()}
                isHandRevealing={isHandRevealing}
                handRevealElapsedMs={handRevealElapsedMs}
                isDalmutiHighlighted={me.id === dalmutiHighlightPlayerId}
                roleHidden={isRankSelectionPhase}
                elementRef={(element) => bindSeatElement(me.id, element)}
              />
            )}
            <div className={styles.handScroller}>
              <div
                className={`${styles.hand} ${
                  isHandConcealed ? styles.handConcealed : ""
                } ${isHandRevealing ? styles.handRevealing : ""}`}
                style={
                  {
                    "--phase-elapsed": `${handRevealElapsedMs}ms`,
                  } as CSSProperties
                }
              >
                {renderedHandValue === null
                  ? Array.from(
                      { length: Math.max(1, displayedMe?.handCount ?? 14) },
                      (_, index) => (
                        <PlayingCard
                          key={`back-${index}`}
                          card={{ id: `back-${index}`, rank: 13 }}
                          concealed
                          displayOnly
                          style={
                            {
                              "--card-index": index,
                            } as CSSProperties
                          }
                        />
                      ),
                    )
                  : renderedHand.map((card, index) => (
                      <PlayingCard
                        key={card.id}
                        card={card}
                        concealed={isHandConcealed}
                        selected={selectedIds.includes(card.id)}
                        disabled={
                          (!isMyTurn || !turnPresentationReady) &&
                          !isTaxSelection
                        }
                        onClick={() => toggleCard(card.id)}
                        onDoubleClick={() => toggleRank(card.rank)}
                        style={
                          {
                            "--card-index": index,
                          } as CSSProperties
                        }
                      />
                    ))}
                {me?.finishedPlace && hand.length === 0 && (
                  <div
                    className={styles.finishedHand}
                    role="status"
                    aria-live="polite"
                  >
                    <span className={styles.finishedMedal}>
                      <b>{me?.finishedPlace ?? "–"}</b>
                      <i>PLACE</i>
                    </span>
                    <span className={styles.finishedHandCopy}>
                      <small>ROUND COMPLETE</small>
                      <strong>먼저 모든 카드를 냈습니다</strong>
                      <em>
                        {me?.finishedPlace
                          ? `${me.finishedPlace}위 확정 · 남은 경기를 관전하는 중`
                          : "다른 플레이어의 순위를 기다리는 중"}
                      </em>
                    </span>
                  </div>
                )}
              </div>
            </div>
            <OnlineChatPanel
              messages={chatMessages}
              viewerId={snapshot.viewerId}
              connected={connection === "online"}
              onSend={sendChatMessage}
            />
            <div className={styles.actionBar}>
              <div className={styles.selectionCopy}>
                <strong>
                  {isTaxSelection
                    ? `반환 카드 ${snapshot.requiredReturnCount}장 선택`
                    : turnPresentationReady
                      ? `${playerName(
                          snapshot.players,
                          snapshot.currentPlayerId ?? me?.id,
                        )}의 차례`
                      : "행동 처리 중"}
                </strong>
                <small>
                  {isTaxSelection
                    ? `${selectedIds.length} / ${snapshot.requiredReturnCount} · 원하는 카드를 고르세요`
                    : !turnPresentationReady
                      ? "다음 차례를 준비하고 있습니다"
                      : isMyTurn
                      ? selectedIds.length
                        ? `${selectedIds.length}장 선택 · ${
                            playError ??
                            `${formatRank(selectedRank ?? 13)} × ${selectedIds.length}장`
                          }`
                        : playError ?? "카드를 선택하세요"
                      : "상대의 행동을 기다리고 있습니다"}
                </small>
              </div>
              {isTaxSelection ? (
                <button
                  type="button"
                  className={styles.submitButton}
                  disabled={!taxSelectionValid}
                  onClick={() =>
                    void sendCommand("SELECT_TAX_RETURN", {
                      cardIds: selectedIds,
                    })
                  }
                >
                  반환 확정
                  <span>→</span>
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className={styles.passButton}
                    disabled={
                      !isMyTurn ||
                      !snapshot.table ||
                      busy ||
                      actionLocked ||
                      Boolean(activeEvent) ||
                      connection !== "online"
                    }
                    onClick={() => void sendCommand("PASS")}
                  >
                    패스
                  </button>
                  <button
                    type="button"
                    className={styles.submitButton}
                    disabled={!canPlay}
                    onClick={() =>
                      void sendCommand("PLAY_CARDS", { cardIds: selectedIds })
                    }
                  >
                    제출
                    <span>→</span>
                  </button>
                </>
              )}
            </div>
          </section>
          {activeEvent &&
            !(snapshot.phase === "round-end" && roundEndResultReady) && (
              <EventOverlay
                key={activeEvent.id}
                event={activeEvent}
                players={snapshot.players}
                anchors={motionAnchors}
                effectiveClock={effectiveClock}
              />
            )}
        </section>

        <aside className={styles.historyRail}>
          <div className={styles.railHeading}>
            <span>기록</span>
            <small>최근 행동</small>
          </div>
          <ul>
            {[...eventBuffer]
              .reverse()
              .slice(0, 9)
              .map((event) => (
                <li key={event.id}>
                  <span>{String(event.seq).padStart(2, "0")}</span>
                  <p>
                    {event.type === "BOT_ADDED"
                      ? `${stringValue(event.data.name, "봇")}이(가) 추가되었습니다`
                      : event.type === "BOT_REMOVED"
                        ? `${stringValue(event.data.name, "봇")}이(가) 삭제되었습니다`
                    : event.type === "RANK_ORDER_ASSIGNED"
                      ? "첫 막의 서열이 정해졌습니다"
                      : event.type === "RANK_CARD_CHOSEN"
                        ? "계급 카드가 선택되었습니다"
                        : event.type === "REVOLUTION_DECLARED"
                          ? `${playerName(snapshot.players, event.actorPlayerId ?? event.data.playerId)}이(가) ${
                              ["great", "great-revolution"].includes(
                                stringValue(event.data.kind),
                              )
                                ? "대혁명"
                                : "혁명"
                            }을 일으켰습니다`
                    : event.type === "DALMUTI_EFFECT"
                      ? `${playerName(snapshot.players, event.actorPlayerId)}이(가) 달무티를 내 모두 자동 패스했습니다`
                      : event.type.includes("PASS")
                      ? `${playerName(snapshot.players, event.actorPlayerId)}이(가) 패스했습니다`
                      : event.type.includes("PLAY")
                        ? `${playerName(snapshot.players, event.actorPlayerId)}이(가) 카드를 냈습니다`
                        : event.type.includes("TAX")
                          ? "세금 교환이 진행되었습니다"
                          : event.type.includes("REVEAL")
                            ? "각자의 패가 공개되었습니다"
                            : "게임 상태가 변경되었습니다"}
                  </p>
                </li>
              ))}
            {!eventBuffer.length && (
              <li>
                <span>01</span>
                <p>게임 기록을 기다리고 있습니다.</p>
              </li>
            )}
          </ul>
          <div className={styles.quickLegend}>
            <span>
              <i className={styles.legendStrong} />1은 가장 강함
            </span>
            <span>
              <i className={styles.legendWeak} />12는 가장 약함
            </span>
            <span>
              <i className={styles.legendJoker} />조커는 만능 카드
            </span>
          </div>
        </aside>
      </section>

      {snapshot.phase === "revolution" &&
        snapshot.canChooseRevolution &&
        !activeEvent && (
          <div className={styles.modalLayer}>
            <section className={styles.decisionCard}>
              <span className={styles.decisionJokers}>♠ ♣</span>
              <small>두 어릿광대가 당신의 손에 있습니다</small>
              <h2>
                {["great-peon", "great_peon"].includes(me?.role ?? "")
                  ? "대혁명을 선포하시겠습니까?"
                  : "혁명을 선포하시겠습니까?"}
              </h2>
              <p>
                {["great-peon", "great_peon"].includes(me?.role ?? "")
                  ? "대혁명을 선포하면 모든 계급이 뒤집히고 이번 막의 세금이 사라집니다."
                  : "혁명을 선포하면 이번 막의 세금이 사라집니다."}
              </p>
              <div>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  disabled={busy}
                  onClick={() =>
                    void sendCommand("CHOOSE_REVOLUTION", { declare: false })
                  }
                >
                  조용히 지나가기
                </button>
                <button
                  type="button"
                  className={styles.submitButton}
                  disabled={busy}
                  onClick={() =>
                    void sendCommand("CHOOSE_REVOLUTION", { declare: true })
                  }
                >
                  {["great-peon", "great_peon"].includes(me?.role ?? "")
                    ? "대혁명 선포"
                    : "혁명 선포"}
                </button>
              </div>
            </section>
          </div>
        )}

      {snapshot.phase === "round-end" && roundEndResultReady && (
        <div className={styles.modalLayer}>
          <section className={styles.resultCard}>
            <span className={styles.eyebrow}>THE COURT HAS SPOKEN</span>
            <h2>제 {snapshot.round}막의 새로운 계급</h2>
            <ol>
              {sortedFinishers.map((player, index) => {
                const priorRankIndex = snapshot.players.findIndex(
                  (candidate) => candidate.id === player.id,
                );
                const previousRole = roleLabel(player.role);
                const nextRole = roleLabelForRank(
                  index,
                  sortedFinishers.length,
                );
                const rankDirection =
                  priorRankIndex > index
                    ? "up"
                    : priorRankIndex < index
                      ? "down"
                      : "same";
                return (
                  <li
                    key={player.id}
                    className={`${player.id === me?.id ? styles.resultSelf : ""} ${
                      index === 0
                        ? styles.resultFirst
                        : index === 1
                          ? styles.resultSecond
                          : ""
                    }`}
                    aria-label={`${index + 1}위 ${player.name}, ${previousRole}에서 ${nextRole}, ${player.score}점`}
                  >
                    <span>{index + 1}</span>
                    <p>
                      <strong>{player.name}</strong>
                      <small
                        className={`${styles.resultRoleChange} ${
                          rankDirection === "up"
                            ? styles.resultRoleUp
                            : rankDirection === "down"
                              ? styles.resultRoleDown
                              : styles.resultRoleSame
                        }`}
                      >
                        {previousRole}
                        <i>→</i>
                        {nextRole}
                      </small>
                    </p>
                    <em>{player.score}점</em>
                  </li>
                );
              })}
            </ol>
            {isHost ? (
              <button
                type="button"
                className={styles.primaryButton}
                disabled={busy}
                onClick={() => void sendCommand("START_NEXT_ROUND")}
              >
                <span>다음 막으로</span>
                <small>새 계급으로 카드 배분</small>
                <b>→</b>
              </button>
            ) : (
              <p className={styles.resultWait}>
                방장이 다음 막을 시작하기를 기다리고 있습니다.
              </p>
            )}
          </section>
        </div>
      )}

      {connection !== "online" && (
        <div className={styles.reconnectOverlay}>
          <span className={styles.spinner} />
          <small>REAL-TIME CONNECTION</small>
          <strong>
            {fatalError ? "방에 다시 들어갈 수 없습니다" : "게임에 다시 연결 중"}
          </strong>
          <p>
            {error ??
              "좌석과 패를 그대로 유지하고 있습니다. 잠시만 기다려 주세요."}
          </p>
          <div>
            {!fatalError && (
              <button type="button" onClick={() => void pollRoom()}>
                지금 다시 연결
              </button>
            )}
            <button
              type="button"
              onClick={() =>
                fatalError ? finishLocalExit() : void exitRoom()
              }
            >
              입장 화면으로
            </button>
          </div>
        </div>
      )}

      {error && connection === "online" && (
        <button
          type="button"
          className={styles.errorToast}
          onClick={() => setError(null)}
        >
          <span>{error}</span>
          <b>닫기</b>
        </button>
      )}
    </main>
  );
}
