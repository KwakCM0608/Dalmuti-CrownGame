"use client";

import type { CSSProperties, FormEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import type { OnlineCommand, OnlineSnapshot } from "@/lib/online-game";
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

type SnapshotView = {
  code: string;
  revision: number;
  eventSeq: number;
  serverTime: number;
  phase: string;
  phaseEndsAt: number | null;
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
  revolutionHolderId: string | null;
  canChooseRevolution: boolean;
};

type StoredSession = {
  roomCode: string;
  playerId: string;
  token: string;
  nickname: string;
};

type ConnectionState =
  | "idle"
  | "connecting"
  | "online"
  | "reconnecting"
  | "offline";

const SESSION_PREFIX = "dalmuti.online.room.";
const LAST_SESSION_KEY = "dalmuti.online.last-session";
const POLL_INTERVAL_MS = 700;
const ROLE_LABELS: Record<string, string> = {
  "great-dalmuti": "대 달무티",
  great_dalmuti: "대 달무티",
  "lesser-dalmuti": "소 달무티",
  lesser_dalmuti: "소 달무티",
  merchant: "상인",
  "lesser-peon": "소 농노",
  lesser_peon: "소 농노",
  "great-peon": "대 농노",
  great_peon: "대 농노",
};
const ROLE_MARKS: Record<string, string> = {
  "great-dalmuti": "Ⅰ",
  great_dalmuti: "Ⅰ",
  "lesser-dalmuti": "Ⅱ",
  lesser_dalmuti: "Ⅱ",
  merchant: "◆",
  "lesser-peon": "Ⅺ",
  lesser_peon: "Ⅺ",
  "great-peon": "Ⅻ",
  great_peon: "Ⅻ",
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
  const seq = numberValue(source.seq, numberValue(source.eventSeq, index));
  const at = numberValue(
    source.at,
    numberValue(source.startsAt, numberValue(payload.at, Date.now())),
  );
  const startsAt = numberValue(
    source.startsAt,
    numberValue(payload.startsAt, at),
  );
  const endsAt = numberValue(
    source.endsAt,
    numberValue(payload.endsAt, startsAt + defaultEventDuration(source.type)),
  );
  return {
    id: stringValue(source.id, `${seq}-${stringValue(source.type, "event")}`),
    seq,
    type: stringValue(source.type, stringValue(source.kind, "EVENT"))
      .toUpperCase()
      .replaceAll("-", "_"),
    at,
    startsAt,
    durationMs: Math.max(
      700,
      numberValue(
        source.durationMs,
        numberValue(
          payload.durationMs,
          Math.max(0, endsAt - startsAt) || defaultEventDuration(source.type),
        ),
      ),
    ),
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

function defaultEventDuration(type: unknown): number {
  const label = stringValue(type).toUpperCase();
  if (label.includes("HAND_REVEAL")) return 900;
  if (label.includes("TAX") || label.includes("TRIBUTE")) return 3400;
  if (label.includes("PLAY")) return 2200;
  if (label.includes("PASS")) return 1500;
  return 1800;
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
    phase: normalizePhase(publicView.phase ?? root.phase),
    phaseEndsAt:
      typeof (publicView.phaseEndsAt ?? root.phaseEndsAt) === "number"
        ? numberValue(publicView.phaseEndsAt ?? root.phaseEndsAt)
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
    revolutionHolderId:
      stringValue(
        revolutionView.holderId,
        stringValue(root.revolutionHolderId),
      ) || null,
    canChooseRevolution: booleanValue(
      revolutionView.canChoose,
      booleanValue(selfView.isRevolutionHolder),
    ),
  };
}

function unwrapSnapshotResponse(value: unknown): {
  snapshot: SnapshotView | null;
  unchanged: boolean;
} {
  const response = record(value);
  if (response.unchanged === true) return { snapshot: null, unchanged: true };
  const candidate = response.snapshot ?? response.projection ?? value;
  return {
    snapshot: snapshotFrom(candidate, response),
    unchanged: false,
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

function playerName(players: PlayerView[], id: unknown): string {
  const playerId = stringValue(id);
  return players.find((player) => player.id === playerId)?.name ?? "플레이어";
}

function seatPosition(index: number, total: number): CSSProperties {
  const angle =
    total <= 1 ? 270 : 150 + (240 * index) / Math.max(1, total - 1);
  const radians = (angle * Math.PI) / 180;
  return {
    "--seat-x": `${50 + Math.cos(radians) * 42}%`,
    "--seat-y": `${46 + Math.sin(radians) * 34}%`,
    "--seat-order": index,
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
}: {
  card: CardView;
  selected?: boolean;
  disabled?: boolean;
  concealed?: boolean;
  onClick?: () => void;
  onDoubleClick?: () => void;
  displayOnly?: boolean;
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

function Brand() {
  return (
    <Link className={styles.brand} href="/" aria-label="달무티 혼자 하기">
      <span className={styles.brandSeal} aria-hidden="true" />
      <span>
        <strong>DALMUTI</strong>
        <small>DCLab의 온라인 계급전</small>
      </span>
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
  passed,
  style,
}: {
  player: PlayerView;
  isSelf?: boolean;
  isHost: boolean;
  isCurrent: boolean;
  passed: boolean;
  style?: CSSProperties;
}) {
  return (
    <article
      className={`${styles.playerSeat} ${
        isSelf ? styles.playerSeatSelf : ""
      } ${isCurrent ? styles.playerSeatCurrent : ""} ${
        !player.connected ? styles.playerSeatDisconnected : ""
      } ${player.finishedPlace ? styles.playerSeatFinished : ""}`}
      style={style}
      aria-label={`${player.name}, ${roleLabel(player.role)}, 카드 ${player.handCount}장`}
    >
      <span className={styles.avatar}>
        {player.monogram}
        <i>{roleMark(player.role)}</i>
      </span>
      <span className={styles.playerCopy}>
        <strong>
          {player.name}
          {isSelf && <small>나</small>}
        </strong>
        <em>{roleLabel(player.role)}</em>
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
      {passed && <span className={styles.passedMark}>PASS</span>}
      {!player.connected && <span className={styles.offlineMark}>재접속 대기</span>}
    </article>
  );
}

function EventOverlay({
  event,
  players,
}: {
  event: EventView;
  players: PlayerView[];
}) {
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

  if (type.includes("TAX") || type.includes("TRIBUTE")) {
    const isIntro = type.includes("INTRO");
    if (isIntro) {
      return (
        <div className={`${styles.eventOverlay} ${styles.introOverlay}`}>
          <small>TRIBUTE PHASE</small>
          <strong>세금 교환</strong>
          <span>계급에 따른 카드 교환을 시작합니다</span>
        </div>
      );
    }
    return (
      <div className={`${styles.eventOverlay} ${styles.taxOverlay}`}>
        <div className={styles.transferNames}>
          <span>{playerName(players, fromId)}</span>
          <i>→</i>
          <span>{playerName(players, toId)}</span>
        </div>
        <div className={styles.eventCards}>
          {(cards.length
            ? cards
            : Array.from(
                {
                  length: Math.max(
                    1,
                    numberValue(data.count, numberValue(route.count, 1)),
                  ),
                },
                (_, index) => ({ id: `hidden-tax-${index}`, rank: 13 }),
              )
          ).map((card, index) => (
            <div
              className={styles.eventCardWrap}
              key={`${event.id}-${card.id}-${index}`}
              style={{ "--event-card-index": index } as CSSProperties}
            >
              <PlayingCard card={card} concealed={!cards.length} displayOnly />
            </div>
          ))}
        </div>
        <strong>
          {cards.length
            ? cards.map((card) => formatRank(card.rank)).join(" · ")
            : `카드 ${numberValue(data.count, numberValue(route.count, 1))}장 이동`}
        </strong>
        <small>
          {cards.length
            ? "이 카드 정보는 교환 당사자에게만 보입니다"
            : "카드의 정체는 교환 당사자만 확인할 수 있습니다"}
        </small>
      </div>
    );
  }

  if (type.includes("PASS")) {
    return (
      <div className={`${styles.eventOverlay} ${styles.passOverlay}`}>
        <span>{playerName(players, actorId)}</span>
        <strong>PASS</strong>
        <small>이번 묶음을 넘겼습니다</small>
      </div>
    );
  }

  if (type.includes("PLAY") && !type.includes("INTRO") && cards.length) {
    return (
      <div className={`${styles.eventOverlay} ${styles.playOverlay}`}>
        <span>{playerName(players, actorId)}</span>
        <div className={styles.eventCards}>
          {cards.map((card, index) => (
            <div
              className={styles.eventCardWrap}
              key={`${event.id}-${card.id}`}
              style={{ "--event-card-index": index } as CSSProperties}
            >
              <PlayingCard card={card} displayOnly />
            </div>
          ))}
        </div>
        <strong>
          {formatRank(cards.find((card) => card.rank !== 13)?.rank ?? 13)} ×{" "}
          {cards.length}장
        </strong>
      </div>
    );
  }

  if (type.includes("REVEAL") || type === "MATCH_STARTED") {
    return (
      <div className={`${styles.eventOverlay} ${styles.revealOverlay}`}>
        <span className={styles.revealCard} aria-hidden="true" />
        <small>HAND REVEAL</small>
        <strong>패를 공개합니다</strong>
        <span>모든 플레이어가 자신의 패를 확인합니다</span>
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
      data.currentPlayerId,
      stringValue(data.startingPlayerId),
    );
    return (
      <div className={`${styles.eventOverlay} ${styles.introOverlay}`}>
        <small>ROUND START</small>
        <strong>게임 시작</strong>
        <span>{playerName(players, starterId)}이(가) 먼저 시작합니다</span>
      </div>
    );
  }

  return (
    <div className={`${styles.eventOverlay} ${styles.introOverlay}`}>
      <small>DALMUTI ONLINE</small>
      <strong>{stringValue(data.title, "게임 진행")}</strong>
      <span>{stringValue(data.message, "모든 플레이어의 상태를 맞추고 있습니다")}</span>
    </div>
  );
}

export default function OnlinePage() {
  const [screen, setScreen] = useState<"entry" | "room">("entry");
  const [entryMode, setEntryMode] = useState<"create" | "join">("create");
  const [nickname, setNickname] = useState("");
  const [roomCodeInput, setRoomCodeInput] = useState("");
  const [session, setSession] = useState<StoredSession | null>(null);
  const [lastSession, setLastSession] = useState<StoredSession | null>(null);
  const [snapshot, setSnapshot] = useState<SnapshotView | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [eventBuffer, setEventBuffer] = useState<EventView[]>([]);
  const [clock, setClock] = useState(() => Date.now());
  const [serverOffset, setServerOffset] = useState(0);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fatalError, setFatalError] = useState(false);
  const [copied, setCopied] = useState(false);
  const [entryBusy, setEntryBusy] = useState(false);
  const inFlightRef = useRef(false);
  const failureCountRef = useRef(0);
  const snapshotRef = useRef<SnapshotView | null>(null);
  const sessionRef = useRef<StoredSession | null>(null);

  const ingestSnapshot = useCallback((next: SnapshotView) => {
    snapshotRef.current = next;
    setSnapshot(next);
    setServerOffset(next.serverTime - Date.now());
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
      const response = await fetch(
        `/api/online/rooms/${activeSession.roomCode}?sinceEventSeq=${sinceEventSeq}`,
        {
          headers: {
            Authorization: `Bearer ${activeSession.token}`,
            Accept: "application/json",
          },
          cache: "no-store",
        },
      );
      const body: unknown = await response.json().catch(() => ({}));
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
      failureCountRef.current = 0;
      setConnection("online");
      setError(null);
    } catch (reason) {
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
  }, [ingestSnapshot]);

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
        if (!response.ok) {
          throw new Error(apiErrorMessage(body, "행동을 처리하지 못했습니다."));
        }
        const result = unwrapSnapshotResponse(body);
        if (result.snapshot) ingestSnapshot(result.snapshot);
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
    [busy, ingestSnapshot, pollRoom],
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
      snapshot.requiredReturnCount > 0,
  );
  const hand = snapshot?.hand ?? [];
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
  const actionLocked = Boolean(
    snapshot?.actionLockUntil &&
      effectiveClock < snapshot.actionLockUntil,
  );
  const canPlay = isMyTurn && !playError && !busy && !actionLocked;
  const taxSelectionValid =
    isTaxSelection &&
    selectedIds.length === (snapshot?.requiredReturnCount ?? 0) &&
    !busy;
  const allReady = Boolean(
    snapshot &&
      snapshot.players.length >= 4 &&
      snapshot.players.every((player) => player.ready && player.connected),
  );
  const activeEvent = useMemo(
    () =>
      [...eventBuffer]
        .filter((event) =>
          [
            "MATCH_STARTED",
            "HAND_REVEAL_STARTED",
            "HAND_REVEALED",
            "TAX_INTRO_STARTED",
            "TAX_TRIBUTE_STARTED",
            "TAX_TRIBUTE",
            "TAX_RETURN_STARTED",
            "TAX_RETURN",
            "PLAY_INTRO_STARTED",
            "CARDS_PLAYED",
            "PLAYER_PASSED",
          ].includes(event.type),
        )
        .reverse()
        .find(
          (event) =>
            effectiveClock >= event.startsAt - 120 &&
            effectiveClock <= event.startsAt + event.durationMs + 220,
        ) ?? null,
    [effectiveClock, eventBuffer],
  );
  const opponents = useMemo(
    () => snapshot?.players.filter((player) => player.id !== me?.id) ?? [],
    [me?.id, snapshot?.players],
  );
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

  const toggleCard = (cardId: string) => {
    if (!isMyTurn && !isTaxSelection) return;
    setSelectedIds((current) =>
      current.includes(cardId)
        ? current.filter((id) => id !== cardId)
        : [...current, cardId],
    );
  };

  const toggleRank = (rank: number) => {
    if (!isMyTurn && !isTaxSelection) return;
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

  const leaveToEntry = () => {
    setScreen("entry");
    setSession(null);
    sessionRef.current = null;
    setSnapshot(null);
    snapshotRef.current = null;
    setEventBuffer([]);
    setConnection("idle");
    setFatalError(false);
    setError(null);
    window.history.replaceState(null, "", "/online");
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
            <h1>
              한 테이블에서
              <br />
              <em>계급을 뒤집으세요</em>
            </h1>
            <p>
              방을 만들고 초대 링크를 공유하세요. 세금 교환의 비밀은
              당사자에게만, 카드 제출과 패스는 모두에게 실시간으로 전달됩니다.
            </p>
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
          <Brand />
          <div className={styles.headerRoom}>
            <span>ROOM</span>
            <strong>{displayCode}</strong>
            <button type="button" onClick={copyInvite}>
              {copied ? "복사됨" : "초대 링크"}
            </button>
          </div>
          <div className={styles.headerActions}>
            <ConnectionPill state={connection} />
            <button type="button" onClick={leaveToEntry}>
              나가기
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
              최소 4명이 모이고 모두 준비하면 방장이 PLAY를 누를 수
              있습니다.
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
          </div>

          <section className={styles.lobbyPlayers}>
            <div className={styles.lobbyHeading}>
              <div>
                <small>PLAYERS</small>
                <h2>랩실 서열 참가자</h2>
              </div>
              <span>{snapshot?.players.length ?? 0} / 8</span>
            </div>
            <ol>
              {Array.from({ length: 8 }, (_, index) => {
                const player = snapshot?.players[index];
                if (!player) {
                  return (
                    <li className={styles.emptySlot} key={`empty-${index}`}>
                      <span>{index + 1}</span>
                      <p>빈 자리</p>
                      <em>초대 링크로 참가할 수 있습니다</em>
                    </li>
                  );
                }
                return (
                  <li
                    className={`${styles.lobbyPlayer} ${
                      player.id === me?.id ? styles.lobbyPlayerSelf : ""
                    }`}
                    key={player.id}
                  >
                    <span className={styles.avatar}>
                      {player.monogram}
                      <i>{index + 1}</i>
                    </span>
                    <p>
                      <strong>{player.name}</strong>
                      <small>
                        {player.id === snapshot?.hostId
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
                        {player.ready ? "준비 완료" : "준비 중"}
                      </em>
                    )}
                    {!player.connected && (
                      <i className={styles.disconnectedBadge}>연결 끊김</i>
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
                      ? "패 공개와 세금 교환 시작"
                      : "4명 이상 · 모두 준비 필요"}
                  </small>
                </button>
              )}
            </div>
            {error && <p className={styles.roomError}>{error}</p>}
          </section>
        </section>

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
              <button type="button" onClick={leaveToEntry}>
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
        <Brand />
        <div className={styles.roundInfo}>
          <span>ROUND {snapshot.round}</span>
          <i />
          <strong>{roleLabel(me?.role ?? "merchant")}</strong>
          <i />
          <span>{snapshot.players.length} PLAYERS</span>
        </div>
        <div className={styles.headerActions}>
          <ConnectionPill state={connection} />
          <button type="button" onClick={copyInvite}>
            {displayCode}
          </button>
        </div>
      </header>

      <section className={styles.gameLayout}>
        <aside className={styles.rankRail}>
          <div className={styles.railHeading}>
            <span>랩실 서열</span>
            <small>ROUND {snapshot.round}</small>
          </div>
          <ol>
            {snapshot.players.map((player, index) => (
              <li
                key={player.id}
                className={player.id === me?.id ? styles.rankSelf : ""}
              >
                <span>{index + 1}</span>
                <p>
                  <strong>{player.name}</strong>
                  <small>{roleLabel(player.role)}</small>
                </p>
                <em>{player.score}점</em>
              </li>
            ))}
          </ol>
          <div className={styles.railRoom}>
            <small>ROOM CODE</small>
            <strong>{displayCode}</strong>
            <button type="button" onClick={copyInvite}>
              {copied ? "복사됨" : "초대"}
            </button>
          </div>
        </aside>

        <section className={styles.boardColumn}>
          <div className={styles.table}>
            <div className={styles.tableLine} />
            <div className={styles.seatRing}>
              {opponents.map((player, index) => (
                <PlayerSeat
                  key={player.id}
                  player={player}
                  isHost={player.id === snapshot.hostId}
                  isCurrent={player.id === snapshot.currentPlayerId}
                  passed={snapshot.passedPlayerIds.includes(player.id)}
                  style={seatPosition(index, opponents.length)}
                />
              ))}
            </div>
            <div className={styles.tableCenter}>
              {snapshot.table?.cards.length ? (
                <>
                  <small>마지막으로 놓인 카드</small>
                  <div className={styles.tableCards}>
                    {snapshot.table.cards.map((card, index) => (
                      <span
                        key={card.id}
                        style={{ "--table-card": index } as CSSProperties}
                      >
                        <PlayingCard card={card} displayOnly />
                      </span>
                    ))}
                  </div>
                  <strong>
                    {formatRank(snapshot.table.rank)} × {snapshot.table.count}장
                  </strong>
                  <p>
                    {snapshot.table.rank}보다 낮은 숫자의 카드{" "}
                    {snapshot.table.count}장을 내세요
                  </p>
                </>
              ) : (
                <div className={styles.emptyTable}>
                  <span aria-hidden="true">◇</span>
                  <strong>비어 있는 필드</strong>
                  <small>
                    {playerName(snapshot.players, snapshot.currentPlayerId)}이(가)
                    새 묶음을 시작합니다
                  </small>
                </div>
              )}
            </div>
          </div>

          <section className={styles.ownDock}>
            {me && (
              <PlayerSeat
                player={me}
                isSelf
                isHost={isHost}
                isCurrent={isMyTurn}
                passed={snapshot.passedPlayerIds.includes(me.id)}
              />
            )}
            <div className={styles.handScroller}>
              <div
                className={`${styles.hand} ${
                  snapshot.hand === null ? styles.handConcealed : ""
                }`}
              >
                {snapshot.hand === null
                  ? Array.from(
                      { length: Math.min(16, Math.max(1, me?.handCount ?? 14)) },
                      (_, index) => (
                        <PlayingCard
                          key={`back-${index}`}
                          card={{ id: `back-${index}`, rank: 13 }}
                          concealed
                          displayOnly
                        />
                      ),
                    )
                  : hand.map((card) => (
                      <PlayingCard
                        key={card.id}
                        card={card}
                        selected={selectedIds.includes(card.id)}
                        disabled={!isMyTurn && !isTaxSelection}
                        onClick={() => toggleCard(card.id)}
                        onDoubleClick={() => toggleRank(card.rank)}
                      />
                    ))}
                {snapshot.hand !== null && hand.length === 0 && (
                  <div className={styles.finishedHand}>
                    <span>✓</span>
                    <strong>모든 카드를 냈습니다</strong>
                    <small>
                      {me?.finishedPlace
                        ? `${me.finishedPlace}위로 이번 막을 마쳤습니다`
                        : "다른 플레이어의 순위를 기다리는 중입니다"}
                    </small>
                  </div>
                )}
              </div>
            </div>
            <div className={styles.actionBar}>
              <div className={styles.selectionCopy}>
                <strong>
                  {isTaxSelection
                    ? `반환 카드 ${snapshot.requiredReturnCount}장 선택`
                    : isMyTurn
                      ? selectedIds.length
                        ? `${selectedIds.length}장 선택`
                        : "당신의 차례"
                      : `${playerName(snapshot.players, snapshot.currentPlayerId)}의 차례`}
                </strong>
                <small>
                  {isTaxSelection
                    ? `${selectedIds.length} / ${snapshot.requiredReturnCount} · 원하는 카드를 고르세요`
                    : isMyTurn
                      ? playError ??
                        `${formatRank(selectedRank ?? 13)} × ${selectedIds.length}장`
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
                    disabled={!isMyTurn || !snapshot.table || busy || actionLocked}
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
                    카드 내기
                    <span>→</span>
                  </button>
                </>
              )}
            </div>
          </section>
        </section>

        <aside className={styles.historyRail}>
          <div className={styles.railHeading}>
            <span>기록</span>
            <small>LIVE</small>
          </div>
          <ul>
            {[...eventBuffer]
              .reverse()
              .slice(0, 9)
              .map((event) => (
                <li key={event.id}>
                  <span>{String(event.seq).padStart(2, "0")}</span>
                  <p>
                    {event.type.includes("PASS")
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
          <div className={styles.liveNote}>
            <ConnectionPill state={connection} />
            <p>서버가 모든 행동과 카드 소유권을 검증합니다.</p>
          </div>
        </aside>
      </section>

      {activeEvent && (
        <EventOverlay event={activeEvent} players={snapshot.players} />
      )}

      {snapshot.phase === "revolution" &&
        snapshot.canChooseRevolution &&
        !activeEvent && (
          <div className={styles.modalLayer}>
            <section className={styles.decisionCard}>
              <span className={styles.decisionJokers}>♠ ♣</span>
              <small>두 어릿광대가 당신의 손에 있습니다</small>
              <h2>혁명을 선포하시겠습니까?</h2>
              <p>
                혁명을 선포하면 이번 막의 세금이 사라집니다. 대 농노라면
                모든 계급이 뒤집힙니다.
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
                  혁명 선포
                </button>
              </div>
            </section>
          </div>
        )}

      {snapshot.phase === "round-end" && !activeEvent && (
        <div className={styles.modalLayer}>
          <section className={styles.resultCard}>
            <span className={styles.eyebrow}>THE LAB HAS SPOKEN</span>
            <h2>제{snapshot.round}막 랩실 서열</h2>
            <ol>
              {sortedFinishers.map((player, index) => (
                <li
                  key={player.id}
                  className={player.id === me?.id ? styles.resultSelf : ""}
                >
                  <span>{index + 1}</span>
                  <p>
                    <strong>{player.name}</strong>
                    <small>{roleLabel(player.role)}</small>
                  </p>
                  <em>{player.score}점</em>
                </li>
              ))}
            </ol>
            {isHost ? (
              <button
                type="button"
                className={styles.primaryButton}
                disabled={busy}
                onClick={() => void sendCommand("START_NEXT_ROUND")}
              >
                <span>다음 막 시작</span>
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
            <button type="button" onClick={leaveToEntry}>
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
