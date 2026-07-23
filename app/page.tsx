"use client";

import { useEffect, useMemo, useState } from "react";

type Role =
  | "great-dalmuti"
  | "lesser-dalmuti"
  | "merchant"
  | "lesser-peon"
  | "great-peon";

type Phase = "playing" | "revolution" | "round-end";

type Card = {
  id: string;
  rank: number;
};

type Player = {
  id: string;
  name: string;
  monogram: string;
  isHuman: boolean;
  role: Role;
};

type PlayedSet = {
  rank: number;
  count: number;
  playerId: string;
};

type GameState = {
  phase: Phase;
  round: number;
  revision: number;
  players: Player[];
  hands: Record<string, Card[]>;
  scores: Record<string, number>;
  currentIndex: number;
  table: PlayedSet | null;
  lastPlayedId: string | null;
  passed: string[];
  finishOrder: string[];
  log: string[];
  revolutionHolder: string | null;
};

const HUMAN_ID = "you";
const ROOM_CODE = "CROWN";

const BASE_PLAYERS: Omit<Player, "role">[] = [
  { id: "seraphine", name: "세라핀", monogram: "세", isHuman: false },
  { id: "marco", name: "마르코", monogram: "마", isHuman: false },
  { id: HUMAN_ID, name: "나", monogram: "나", isHuman: true },
  { id: "luna", name: "루나", monogram: "루", isHuman: false },
  { id: "tobias", name: "토비아스", monogram: "토", isHuman: false },
];

const ROLE_LABELS: Record<Role, string> = {
  "great-dalmuti": "대 달무티",
  "lesser-dalmuti": "소 달무티",
  merchant: "상인",
  "lesser-peon": "소 농노",
  "great-peon": "대 농노",
};

const ROLE_MARKS: Record<Role, string> = {
  "great-dalmuti": "♛",
  "lesser-dalmuti": "♕",
  merchant: "◆",
  "lesser-peon": "♙",
  "great-peon": "♟",
};

const RANK_NAMES: Record<number, string> = {
  1: "대 달무티",
  2: "대주교",
  3: "원수",
  4: "남작",
  5: "수도원장",
  6: "기사",
  7: "재봉사",
  8: "석공",
  9: "요리사",
  10: "목동",
  11: "채석공",
  12: "농노",
  13: "광대",
};

function roleForIndex(index: number, total: number): Role {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === total - 2) return "lesser-peon";
  if (index === total - 1) return "great-peon";
  return "merchant";
}

function assignRoles(players: Omit<Player, "role">[] | Player[]): Player[] {
  return players.map((player, index) => ({
    ...player,
    role: roleForIndex(index, players.length),
  }));
}

function createDeck(): Card[] {
  const deck: Card[] = [];
  for (let rank = 1; rank <= 12; rank += 1) {
    for (let copy = 0; copy < rank; copy += 1) {
      deck.push({ id: `${rank}-${copy}`, rank });
    }
  }
  deck.push({ id: "joker-1", rank: 13 });
  deck.push({ id: "joker-2", rank: 13 });
  return deck;
}

function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const target = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[target]] = [copy[target], copy[index]];
  }
  return copy;
}

function sortHand(cards: Card[]): Card[] {
  return [...cards].sort((a, b) => b.rank - a.rank || a.id.localeCompare(b.id));
}

function deal(players: Player[]): Record<string, Card[]> {
  const hands = Object.fromEntries(players.map((player) => [player.id, [] as Card[]]));
  shuffle(createDeck()).forEach((card, index) => {
    const player = players[index % players.length];
    hands[player.id].push(card);
  });
  for (const player of players) hands[player.id] = sortHand(hands[player.id]);
  return hands;
}

function removeCards(hand: Card[], ids: string[]): Card[] {
  const selected = new Set(ids);
  return hand.filter((card) => !selected.has(card.id));
}

function normalizedSet(cards: Card[]): { rank: number; count: number } | null {
  if (cards.length === 0) return null;
  const normalCards = cards.filter((card) => card.rank !== 13);
  if (normalCards.length === 0) {
    return cards.length === 1 ? { rank: 13, count: 1 } : null;
  }
  const rank = normalCards[0].rank;
  if (normalCards.some((card) => card.rank !== rank)) return null;
  return { rank, count: cards.length };
}

function validationMessage(cards: Card[], table: PlayedSet | null): string | null {
  const set = normalizedSet(cards);
  if (!set) return "같은 계급의 카드만 함께 낼 수 있어요.";
  if (!table) return null;
  if (set.count !== table.count) return `${table.count}장을 내야 해요.`;
  if (set.rank >= table.rank) return `${table.rank}보다 강한 낮은 숫자가 필요해요.`;
  return null;
}

function findPlayer(players: Player[], role: Role): Player {
  return players.find((player) => player.role === role)!;
}

function applyTax(
  players: Player[],
  sourceHands: Record<string, Card[]>,
): { hands: Record<string, Card[]>; notes: string[] } {
  const hands = Object.fromEntries(
    Object.entries(sourceHands).map(([id, hand]) => [id, [...hand]]),
  );
  const notes: string[] = [];

  const exchange = (nobleRole: Role, peonRole: Role, count: number) => {
    const noble = findPlayer(players, nobleRole);
    const peon = findPlayer(players, peonRole);
    const peonGift = [...sourceHands[peon.id]]
      .sort((a, b) => a.rank - b.rank)
      .slice(0, count);
    const nobleGift = [...sourceHands[noble.id]]
      .sort((a, b) => b.rank - a.rank)
      .slice(0, count);

    hands[peon.id] = sortHand([
      ...removeCards(hands[peon.id], peonGift.map((card) => card.id)),
      ...nobleGift,
    ]);
    hands[noble.id] = sortHand([
      ...removeCards(hands[noble.id], nobleGift.map((card) => card.id)),
      ...peonGift,
    ]);
    notes.push(`${ROLE_LABELS[peonRole]}가 ${ROLE_LABELS[nobleRole]}에게 세금을 바쳤습니다.`);
  };

  exchange("great-dalmuti", "great-peon", 2);
  exchange("lesser-dalmuti", "lesser-peon", 1);
  return { hands, notes };
}

function prepareRound(
  orderedPlayers: Player[],
  round: number,
  scores: Record<string, number>,
): GameState {
  let players = assignRoles(orderedPlayers);
  let hands = deal(players);
  const holder = players.find(
    (player) => hands[player.id].filter((card) => card.rank === 13).length === 2,
  );
  let phase: Phase = "playing";
  let revolutionHolder: string | null = null;
  let log = [`제 ${round}막이 시작되었습니다.`];

  if (holder?.isHuman) {
    phase = "revolution";
    revolutionHolder = holder.id;
    log = ["두 광대가 당신의 손에 모였습니다. 혁명을 선택하세요.", ...log];
  } else if (holder) {
    if (holder.role === "great-peon") {
      players = assignRoles([...players].reverse());
      log = [`${holder.name}의 대혁명! 모든 계급이 뒤집혔습니다.`, ...log];
    } else {
      log = [`${holder.name}이 혁명을 선포해 세금이 취소되었습니다.`, ...log];
    }
  } else {
    const taxed = applyTax(players, hands);
    hands = taxed.hands;
    log = [...taxed.notes, ...log];
  }

  return {
    phase,
    round,
    revision: 0,
    players,
    hands,
    scores,
    currentIndex: 0,
    table: null,
    lastPlayedId: null,
    passed: [],
    finishOrder: [],
    log,
    revolutionHolder,
  };
}

function nextActiveIndex(state: GameState, fromIndex: number): number {
  for (let step = 1; step <= state.players.length; step += 1) {
    const index = (fromIndex + step) % state.players.length;
    if (state.hands[state.players[index].id].length > 0) return index;
  }
  return fromIndex;
}

function playCards(state: GameState, playerId: string, cardIds: string[]): GameState {
  if (state.phase !== "playing") return state;
  const current = state.players[state.currentIndex];
  if (current.id !== playerId) return state;

  const hand = state.hands[playerId];
  const selected = hand.filter((card) => cardIds.includes(card.id));
  const set = normalizedSet(selected);
  if (!set || validationMessage(selected, state.table)) return state;

  const hands = { ...state.hands, [playerId]: removeCards(hand, cardIds) };
  let finishOrder = [...state.finishOrder];
  const scores = { ...state.scores };
  const log = [
    `${current.name}이 ${set.rank === 13 ? "광대" : `${set.rank}등급`} ${set.count}장을 냈습니다.`,
    ...state.log,
  ].slice(0, 12);

  if (hands[playerId].length === 0) {
    finishOrder.push(playerId);
    scores[playerId] += state.players.length - finishOrder.length;
    log.unshift(`${current.name}이 ${finishOrder.length}위로 계급 경쟁을 마쳤습니다.`);
  }

  if (finishOrder.length === state.players.length - 1) {
    const last = state.players.find((player) => !finishOrder.includes(player.id));
    if (last) finishOrder.push(last.id);
    return {
      ...state,
      phase: "round-end",
      revision: state.revision + 1,
      hands,
      scores,
      table: { ...set, playerId },
      lastPlayedId: playerId,
      finishOrder,
      log: ["이번 막의 새로운 계급이 결정되었습니다.", ...log],
    };
  }

  const nextState: GameState = {
    ...state,
    revision: state.revision + 1,
    hands,
    scores,
    table: { ...set, playerId },
    lastPlayedId: playerId,
    passed: [],
    finishOrder,
    log,
  };
  nextState.currentIndex = nextActiveIndex(nextState, state.currentIndex);
  return nextState;
}

function passTurn(state: GameState, playerId: string): GameState {
  if (state.phase !== "playing" || !state.table) return state;
  const current = state.players[state.currentIndex];
  if (current.id !== playerId) return state;

  const passed = [...new Set([...state.passed, playerId])];
  const active = state.players.filter((player) => state.hands[player.id].length > 0);
  const requiredToPass = active.filter((player) => player.id !== state.lastPlayedId);
  const log = [`${current.name}이 패스했습니다.`, ...state.log].slice(0, 12);

  if (requiredToPass.every((player) => passed.includes(player.id))) {
    const lastIndex = state.players.findIndex(
      (player) => player.id === state.lastPlayedId,
    );
    const lastStillActive =
      lastIndex >= 0 && state.hands[state.players[lastIndex].id].length > 0;
    const cleared: GameState = {
      ...state,
      revision: state.revision + 1,
      table: null,
      passed: [],
      log: ["판이 비워졌습니다. 새로운 묶음을 시작합니다.", ...log].slice(0, 12),
    };
    cleared.currentIndex = lastStillActive
      ? lastIndex
      : nextActiveIndex(cleared, lastIndex);
    return cleared;
  }

  const nextState = {
    ...state,
    revision: state.revision + 1,
    passed,
    log,
  };
  nextState.currentIndex = nextActiveIndex(nextState, state.currentIndex);
  return nextState;
}

function chooseBotCards(state: GameState, playerId: string): string[] | null {
  const hand = state.hands[playerId];
  const jokers = hand.filter((card) => card.rank === 13);
  const groups = new Map<number, Card[]>();
  for (const card of hand) {
    if (card.rank === 13) continue;
    groups.set(card.rank, [...(groups.get(card.rank) ?? []), card]);
  }

  if (!state.table) {
    if (jokers.length > 0) return [jokers[0].id];
    const ranks = [...groups.keys()].sort((a, b) => b - a);
    const rank = ranks[0];
    return rank ? groups.get(rank)!.map((card) => card.id) : null;
  }

  const targetCount = state.table.count;
  const ranks = [...groups.keys()]
    .filter((rank) => rank < state.table!.rank)
    .sort((a, b) => b - a);

  for (const rank of ranks) {
    const cards = groups.get(rank)!;
    if (cards.length + jokers.length < targetCount) continue;
    return [
      ...cards.slice(0, targetCount),
      ...jokers.slice(0, Math.max(0, targetCount - cards.length)),
    ].map((card) => card.id);
  }
  return null;
}

function PlayerSeat({
  player,
  handCount,
  score,
  isCurrent,
  isFinished,
}: {
  player: Player;
  handCount: number;
  score: number;
  isCurrent: boolean;
  isFinished: boolean;
}) {
  return (
    <article
      className={`player-seat role-${player.role} ${isCurrent ? "is-current" : ""} ${
        isFinished ? "is-finished" : ""
      }`}
      aria-label={`${player.name}, ${ROLE_LABELS[player.role]}, 카드 ${handCount}장`}
    >
      <div className="player-avatar">
        <span>{player.monogram}</span>
        <i>{ROLE_MARKS[player.role]}</i>
      </div>
      <div className="player-copy">
        <strong>{player.name}</strong>
        <span>{ROLE_LABELS[player.role]}</span>
      </div>
      <div className="player-count">
        <b>{isFinished ? "완료" : handCount}</b>
        <span>{isFinished ? `${score}점` : "장"}</span>
      </div>
      {isCurrent && <em className="turn-flag">차례</em>}
    </article>
  );
}

function PlayingCard({
  card,
  selected,
  disabled,
  onClick,
  displayOnly = false,
}: {
  card: Card;
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  displayOnly?: boolean;
}) {
  const isJoker = card.rank === 13;
  const content = (
    <>
      <span className="card-corner">{isJoker ? "★" : card.rank}</span>
      <span className="card-emblem">{isJoker ? "☾" : ROLE_MARKS[roleForCard(card.rank)]}</span>
      <strong>{isJoker ? "JESTER" : String(card.rank).padStart(2, "0")}</strong>
      <small>{RANK_NAMES[card.rank]}</small>
      <span className="card-corner card-corner-bottom">{isJoker ? "★" : card.rank}</span>
    </>
  );

  if (displayOnly) {
    return <div className={`playing-card ${isJoker ? "is-joker" : ""}`}>{content}</div>;
  }

  return (
    <button
      type="button"
      className={`playing-card ${isJoker ? "is-joker" : ""} ${
        selected ? "is-selected" : ""
      }`}
      disabled={disabled}
      aria-pressed={selected}
      aria-label={`${RANK_NAMES[card.rank]} 카드 ${selected ? "선택됨" : ""}`}
      onClick={onClick}
    >
      {content}
    </button>
  );
}

function roleForCard(rank: number): Role {
  if (rank === 1) return "great-dalmuti";
  if (rank === 2) return "lesser-dalmuti";
  if (rank >= 11) return "great-peon";
  if (rank >= 9) return "lesser-peon";
  return "merchant";
}

export default function Home() {
  const [game, setGame] = useState<GameState | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [showRules, setShowRules] = useState(false);

  const currentPlayer = game?.players[game.currentIndex] ?? null;
  const humanHand = game?.hands[HUMAN_ID] ?? [];
  const isHumanTurn =
    game?.phase === "playing" && currentPlayer?.id === HUMAN_ID;
  const selectedCards = humanHand.filter((card) => selectedIds.includes(card.id));
  const selectedSet = normalizedSet(selectedCards);
  const selectedError = game
    ? validationMessage(selectedCards, game.table)
    : "카드를 선택하세요.";
  const canPlay = Boolean(
    game &&
      isHumanTurn &&
      selectedIds.length > 0 &&
      selectedSet &&
      !selectedError,
  );

  const orderedOpponents = useMemo(
    () => game?.players.filter((player) => !player.isHuman) ?? [],
    [game?.players],
  );

  useEffect(() => {
    if (!game || game.phase !== "playing" || currentPlayer?.isHuman) return;
    const turnRevision = game.revision;
    const timer = window.setTimeout(() => {
      setGame((latest) => {
        if (
          !latest ||
          latest.phase !== "playing" ||
          latest.revision !== turnRevision
        ) {
          return latest;
        }
        const bot = latest.players[latest.currentIndex];
        if (bot.isHuman) return latest;
        const cards = chooseBotCards(latest, bot.id);
        return cards ? playCards(latest, bot.id, cards) : passTurn(latest, bot.id);
      });
    }, 760);
    return () => window.clearTimeout(timer);
  }, [currentPlayer?.id, currentPlayer?.isHuman, game?.phase, game?.revision]);

  useEffect(() => {
    setSelectedIds([]);
  }, [game?.revision, game?.phase]);

  const startGame = () => {
    const players = assignRoles(BASE_PLAYERS);
    const scores = Object.fromEntries(players.map((player) => [player.id, 0]));
    setGame(prepareRound(players, 1, scores));
  };

  const resolveRevolution = (declare: boolean) => {
    setGame((current) => {
      if (!current || current.phase !== "revolution") return current;
      let players = current.players;
      let hands = current.hands;
      let log = current.log;
      const holder = current.players.find(
        (player) => player.id === current.revolutionHolder,
      );

      if (declare && holder?.role === "great-peon") {
        players = assignRoles([...current.players].reverse());
        log = ["당신의 대혁명으로 모든 계급이 뒤집혔습니다.", ...log];
      } else if (declare) {
        log = ["당신이 혁명을 선포했습니다. 이번 막의 세금은 없습니다.", ...log];
      } else {
        const taxed = applyTax(players, hands);
        hands = taxed.hands;
        log = ["당신은 혁명을 숨겼습니다.", ...taxed.notes, ...log];
      }

      return {
        ...current,
        phase: "playing",
        revision: current.revision + 1,
        players,
        hands,
        log,
        revolutionHolder: null,
        currentIndex: 0,
      };
    });
  };

  const toggleCard = (cardId: string) => {
    if (!isHumanTurn) return;
    setSelectedIds((current) =>
      current.includes(cardId)
        ? current.filter((id) => id !== cardId)
        : [...current, cardId],
    );
  };

  const playSelected = () => {
    if (!game || !canPlay) return;
    setGame(playCards(game, HUMAN_ID, selectedIds));
  };

  const pass = () => {
    if (!game || !isHumanTurn || !game.table) return;
    setGame(passTurn(game, HUMAN_ID));
  };

  const nextRound = () => {
    if (!game || game.phase !== "round-end") return;
    const ordered = game.finishOrder.map(
      (id) => game.players.find((player) => player.id === id)!,
    );
    setGame(prepareRound(ordered, game.round + 1, game.scores));
  };

  const turnMessage = !game
    ? "왕실의 자리가 비어 있습니다"
    : game.phase === "round-end"
      ? "새로운 계급이 결정되었습니다"
      : game.phase === "revolution"
        ? "두 광대가 혁명을 기다립니다"
        : isHumanTurn
          ? game.table
            ? `${game.table.count}장 · ${game.table.rank}보다 낮은 숫자를 내세요`
            : "새로운 묶음을 시작하세요"
          : `${currentPlayer?.name}의 선택을 기다리는 중`;

  const tablePreview = game?.table
    ? Array.from({ length: game.table.count }, (_, index) => ({
        id: `table-${index}`,
        rank: game.table!.rank,
      }))
    : [];

  return (
    <main className="game-shell">
      <div className="paper-grain" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <span className="brand-seal">D</span>
          <div>
            <strong>DALMUTI</strong>
            <small>왕관의 계급전</small>
          </div>
        </div>

        <div className="round-chip" aria-label="게임 정보">
          <span>제 {game?.round ?? 1}막</span>
          <i />
          <span>5인 궁정</span>
          <i />
          <span className="room-code">{ROOM_CODE}</span>
        </div>

        <nav className="top-actions" aria-label="게임 메뉴">
          <button type="button" onClick={() => setShowRules(true)}>
            규칙
          </button>
          <button type="button" onClick={startGame}>
            새 게임
          </button>
        </nav>
      </header>

      <section className="game-stage" aria-label="달무티 게임 테이블">
        <aside className="score-rail">
          <div className="rail-heading">
            <span>궁정 서열</span>
            <small>현재 계급</small>
          </div>
          <ol>
            {(game?.players ?? assignRoles(BASE_PLAYERS)).map((player) => (
              <li key={player.id} className={player.id === HUMAN_ID ? "is-you" : ""}>
                <span>{ROLE_MARKS[player.role]}</span>
                <div>
                  <b>{player.name}</b>
                  <small>{ROLE_LABELS[player.role]}</small>
                </div>
                <em>{game?.scores[player.id] ?? 0}</em>
              </li>
            ))}
          </ol>
          <div className="rail-note">
            <span>계급의 법칙</span>
            <p>숫자가 낮을수록 강합니다. 같은 장수로 더 강하게 맞서세요.</p>
          </div>
        </aside>

        <div className="table-column">
          <div className="opponent-row">
            {(orderedOpponents.length
              ? orderedOpponents
              : assignRoles(BASE_PLAYERS).filter((player) => !player.isHuman)
            ).map((player) => (
              <PlayerSeat
                key={player.id}
                player={player}
                handCount={game?.hands[player.id]?.length ?? 16}
                score={game?.scores[player.id] ?? 0}
                isCurrent={currentPlayer?.id === player.id}
                isFinished={Boolean(game?.finishOrder.includes(player.id))}
              />
            ))}
          </div>

          <div className="felt-table">
            <div className="table-ring" aria-hidden="true">
              <span>♜</span>
              <i />
              <span>♞</span>
              <i />
              <span>♝</span>
            </div>

            <section className="play-area" aria-live="polite">
              <span className="play-kicker">
                {game?.table ? "마지막으로 놓인 패" : "비어 있는 판"}
              </span>
              <div className={`table-cards ${tablePreview.length ? "" : "is-empty"}`}>
                {tablePreview.length ? (
                  tablePreview.map((card, index) => (
                    <div
                      key={card.id}
                      className="table-card-wrap"
                      style={{ "--card-index": index } as React.CSSProperties}
                    >
                      <PlayingCard card={card} displayOnly />
                    </div>
                  ))
                ) : (
                  <div className="empty-pile">
                    <span>♛</span>
                    <small>선 플레이어가<br />새 묶음을 냅니다</small>
                  </div>
                )}
              </div>
              {game?.table && (
                <strong className="table-callout">
                  {game.table.rank === 13 ? "광대" : `${game.table.rank}등급`} ·{" "}
                  {game.table.count}장
                </strong>
              )}
              <p>{turnMessage}</p>
            </section>
          </div>

          <section className={`human-zone ${isHumanTurn ? "is-active" : ""}`}>
            <div className="human-status">
              <div className="human-avatar">나</div>
              <div>
                <span>{game ? ROLE_LABELS[game.players.find((p) => p.id === HUMAN_ID)!.role] : "상인"}</span>
                <strong>{isHumanTurn ? "당신의 차례" : "나의 손패"}</strong>
              </div>
              <em>{humanHand.length || 16}장</em>
            </div>

            <div className="hand-wrap">
              <div className="hand" data-testid="player-hand">
                {(humanHand.length
                  ? humanHand
                  : [
                      { id: "demo-12", rank: 12 },
                      { id: "demo-11", rank: 11 },
                      { id: "demo-10", rank: 10 },
                      { id: "demo-9", rank: 9 },
                      { id: "demo-8", rank: 8 },
                      { id: "demo-7", rank: 7 },
                      { id: "demo-6", rank: 6 },
                      { id: "demo-5", rank: 5 },
                    ]
                ).map((card) => (
                  <PlayingCard
                    key={card.id}
                    card={card}
                    selected={selectedIds.includes(card.id)}
                    disabled={!game || !isHumanTurn}
                    onClick={() => toggleCard(card.id)}
                  />
                ))}
              </div>
            </div>

            <div className="turn-controls">
              <div className={`selection-hint ${selectedError ? "has-error" : "is-valid"}`}>
                <span>{selectedIds.length ? `${selectedIds.length}장 선택` : "카드를 선택하세요"}</span>
                <small>
                  {selectedIds.length
                    ? selectedError ?? `${selectedSet?.rank}등급 묶음 · 낼 수 있습니다`
                    : game?.table
                      ? `현재 ${game.table.count}장 묶음`
                      : "같은 숫자는 여러 장 선택 가능"}
                </small>
              </div>
              <button
                type="button"
                className="pass-button"
                disabled={!isHumanTurn || !game?.table}
                onClick={pass}
              >
                패스
              </button>
              <button
                type="button"
                className="play-button"
                disabled={!canPlay}
                onClick={playSelected}
              >
                패 내기
                <span>↗</span>
              </button>
            </div>
          </section>
        </div>

        <aside className="history-rail">
          <div className="rail-heading">
            <span>궁정 기록</span>
            <small>최근 행동</small>
          </div>
          <ul>
            {(game?.log ?? [
              "빠른 대전을 시작해 왕관을 차지하세요.",
              "첫 판의 계급은 이미 정해져 있습니다.",
              "세금과 혁명도 자동으로 진행됩니다.",
            ]).map((entry, index) => (
              <li key={`${entry}-${index}`}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <p>{entry}</p>
              </li>
            ))}
          </ul>
          <div className="legend">
            <div><span className="legend-dot strongest" />1은 가장 강함</div>
            <div><span className="legend-dot weakest" />12는 가장 약함</div>
            <div><span className="legend-dot joker" />광대는 만능 카드</div>
          </div>
        </aside>
      </section>

      {!game && (
        <div className="welcome-layer">
          <section className="welcome-card" role="dialog" aria-labelledby="welcome-title">
            <div className="welcome-crown">♛</div>
            <span className="eyebrow">PLAYABLE PROTOTYPE · 5 PLAYERS</span>
            <h1 id="welcome-title">왕관은<br /><em>공평하지 않다</em></h1>
            <p>
              약한 패부터 영리하게 털어내고, 계급을 뒤집으세요.
              네 명의 궁정 AI와 바로 한 판을 시작합니다.
            </p>
            <div className="welcome-features">
              <span>80장 정식 덱</span>
              <span>세금과 혁명</span>
              <span>연속 라운드</span>
            </div>
            <button type="button" className="start-button" onClick={startGame}>
              <span>5인 빠른 대전</span>
              <i>게임 시작</i>
              <b>→</b>
            </button>
            <small className="welcome-note">
              이 버전은 혼자 테스트하는 플레이어 대 AI 체험판입니다.
            </small>
          </section>
        </div>
      )}

      {game?.phase === "revolution" && (
        <div className="modal-layer">
          <section className="decision-card" role="dialog" aria-labelledby="revolution-title">
            <span className="decision-icon">☾ ☾</span>
            <small>두 광대가 한 손에 모였습니다</small>
            <h2 id="revolution-title">혁명을 선포하시겠습니까?</h2>
            <p>
              혁명을 선포하면 이번 막의 세금이 사라집니다.
              대 농노라면 모든 계급까지 뒤집힙니다.
            </p>
            <div>
              <button type="button" className="secondary-button" onClick={() => resolveRevolution(false)}>
                조용히 지나간다
              </button>
              <button type="button" className="play-button" onClick={() => resolveRevolution(true)}>
                혁명 선포
              </button>
            </div>
          </section>
        </div>
      )}

      {game?.phase === "round-end" && (
        <div className="modal-layer">
          <section className="result-card" role="dialog" aria-labelledby="result-title">
            <span className="eyebrow">THE COURT HAS SPOKEN</span>
            <h2 id="result-title">제 {game.round}막의 새로운 계급</h2>
            <ol>
              {game.finishOrder.map((id, index) => {
                const player = game.players.find((candidate) => candidate.id === id)!;
                const nextRole = roleForIndex(index, game.players.length);
                return (
                  <li key={id} className={id === HUMAN_ID ? "is-you" : ""}>
                    <span>{index + 1}</span>
                    <div>
                      <b>{player.name}</b>
                      <small>{ROLE_LABELS[nextRole]}</small>
                    </div>
                    <em>{game.scores[id]}점</em>
                  </li>
                );
              })}
            </ol>
            <button type="button" className="start-button" onClick={nextRound}>
              <span>다음 막으로</span>
              <i>새 계급으로 카드 배분</i>
              <b>→</b>
            </button>
          </section>
        </div>
      )}

      {showRules && (
        <div className="modal-layer">
          <section className="rules-card" role="dialog" aria-labelledby="rules-title">
            <button
              type="button"
              className="close-button"
              aria-label="규칙 닫기"
              onClick={() => setShowRules(false)}
            >
              ×
            </button>
            <span className="eyebrow">HOW TO PLAY</span>
            <h2 id="rules-title">세 가지만 기억하세요</h2>
            <div className="rules-grid">
              <article>
                <span>01</span>
                <h3>같은 숫자를 묶기</h3>
                <p>한 장 또는 같은 숫자 여러 장을 한 번에 냅니다.</p>
              </article>
              <article>
                <span>02</span>
                <h3>낮은 숫자로 이기기</h3>
                <p>앞사람과 같은 장수이면서 더 낮은 숫자만 낼 수 있습니다.</p>
              </article>
              <article>
                <span>03</span>
                <h3>가장 먼저 털기</h3>
                <p>손패를 먼저 비울수록 다음 막의 계급이 높아집니다.</p>
              </article>
            </div>
            <div className="rule-detail">
              광대는 다른 카드와 함께 내면 그 숫자로 변합니다. 단독으로 내면 가장 약한 13입니다.
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
