export const BOT_DIFFICULTIES = ["easy", "normal", "hard"] as const;

export type BotDifficulty = (typeof BOT_DIFFICULTIES)[number];

export type BotCard = {
  id: string;
  rank: number;
};

/**
 * Public player information only. Deliberately does not contain a hand.
 *
 * `players` in BotPlayObservation must be ordered by the current rank/turn
 * order, so the policy can reason about who receives the next opportunity.
 */
export type BotPublicPlayer = {
  id: string;
  handCount: number;
  finished?: boolean;
};

export type BotPublicCardCount = {
  rank: number;
  count: number;
};

export type BotTable = {
  rank: number;
  count: number;
  playerId?: string;
};

/**
 * This is the complete information boundary for a play decision.
 *
 * Neither quick play nor the authoritative online server should pass
 * opponents' hidden cards to this module. Hard bots may count public cards,
 * but they never receive or inspect another player's hand.
 */
export type BotPlayObservation = {
  actorId: string;
  hand: readonly BotCard[];
  table: BotTable | null;
  players: readonly BotPublicPlayer[];
  passedPlayerIds?: readonly string[];
  publicPlayedCards?: readonly BotPublicCardCount[];
};

export type BotPlayAction = {
  type: "play";
  cardIds: string[];
  rank: number;
  count: number;
  jokerCount: number;
};

export type BotPassAction = {
  type: "pass";
};

export type BotAction = BotPlayAction | BotPassAction;

export type BotPlayDecision = {
  action: BotAction;
  score: number;
  reasons: string[];
};

export type BotTaxDecision = {
  cardIds: string[];
  score: number;
  reasons: string[];
};

export type BotRole =
  | "great-dalmuti"
  | "lesser-dalmuti"
  | "merchant"
  | "lesser-peon"
  | "great-peon";

export type BotRevolutionObservation = {
  hand: readonly BotCard[];
  role: BotRole;
  playerCount: number;
};

export type BotRevolutionDecision = {
  declare: boolean;
  kind: "revolution" | "great-revolution";
  score: number;
  reasons: string[];
};

const JOKER_RANK = 13;
const NORMAL_RANK_MIN = 1;

function assertDifficulty(difficulty: BotDifficulty): void {
  if (!BOT_DIFFICULTIES.includes(difficulty)) {
    throw new RangeError(`unknown bot difficulty: ${String(difficulty)}`);
  }
}

function assertCard(card: BotCard): void {
  if (!card.id) throw new TypeError("a bot card must have a non-empty id");
  if (
    !Number.isInteger(card.rank) ||
    card.rank < NORMAL_RANK_MIN ||
    card.rank > JOKER_RANK
  ) {
    throw new RangeError(`card ${card.id} has an invalid rank`);
  }
}

function assertCards(cards: readonly BotCard[]): void {
  const ids = new Set<string>();
  for (const card of cards) {
    assertCard(card);
    if (ids.has(card.id)) {
      throw new TypeError(`duplicate card id: ${card.id}`);
    }
    ids.add(card.id);
  }
}

function sortedCards(cards: readonly BotCard[]): BotCard[] {
  return [...cards].sort(
    (left, right) =>
      left.rank - right.rank || left.id.localeCompare(right.id),
  );
}

function combinations<T>(items: readonly T[], count: number): T[][] {
  if (count === 0) return [[]];
  if (count < 0 || count > items.length) return [];
  const result: T[][] = [];
  const selected: T[] = [];

  const visit = (start: number): void => {
    if (selected.length === count) {
      result.push([...selected]);
      return;
    }
    const stillNeeded = count - selected.length;
    for (
      let index = start;
      index <= items.length - stillNeeded;
      index += 1
    ) {
      selected.push(items[index]);
      visit(index + 1);
      selected.pop();
    }
  };

  visit(0);
  return result;
}

function groupedNormalCards(
  hand: readonly BotCard[],
): Map<number, BotCard[]> {
  const groups = new Map<number, BotCard[]>();
  for (const card of sortedCards(hand)) {
    if (card.rank === JOKER_RANK) continue;
    const group = groups.get(card.rank) ?? [];
    group.push(card);
    groups.set(card.rank, group);
  }
  return groups;
}

function playAction(cards: readonly BotCard[], rank: number): BotPlayAction {
  return {
    type: "play",
    cardIds: cards.map((card) => card.id).sort((a, b) => a.localeCompare(b)),
    rank,
    count: cards.length,
    jokerCount: cards.filter((card) => card.rank === JOKER_RANK).length,
  };
}

function assertObservation(observation: BotPlayObservation): void {
  if (!observation.actorId) {
    throw new TypeError("actorId must be a non-empty string");
  }
  assertCards(observation.hand);

  const playerIds = new Set<string>();
  for (const player of observation.players) {
    if (!player.id) throw new TypeError("a public player must have an id");
    if (playerIds.has(player.id)) {
      throw new TypeError(`duplicate public player id: ${player.id}`);
    }
    if (!Number.isInteger(player.handCount) || player.handCount < 0) {
      throw new RangeError(`player ${player.id} has an invalid hand count`);
    }
    playerIds.add(player.id);
  }
  if (!playerIds.has(observation.actorId)) {
    throw new TypeError("players must include the acting bot");
  }

  if (observation.table) {
    if (
      !Number.isInteger(observation.table.rank) ||
      observation.table.rank < NORMAL_RANK_MIN ||
      observation.table.rank > JOKER_RANK
    ) {
      throw new RangeError("the table has an invalid rank");
    }
    if (
      !Number.isInteger(observation.table.count) ||
      observation.table.count < 1
    ) {
      throw new RangeError("the table has an invalid card count");
    }
  }

  for (const publicCount of observation.publicPlayedCards ?? []) {
    if (
      !Number.isInteger(publicCount.rank) ||
      publicCount.rank < NORMAL_RANK_MIN ||
      publicCount.rank > JOKER_RANK ||
      !Number.isInteger(publicCount.count) ||
      publicCount.count < 0
    ) {
      throw new RangeError("publicPlayedCards contains an invalid count");
    }
  }
}

/**
 * Enumerates every distinct legal card-id selection.
 *
 * A lead may contain any positive number of one natural rank plus zero, one,
 * or two jokers. One or two jokers may also lead by themselves at rank 13.
 * A response must match the table count and use a lower rank.
 */
export function enumerateLegalBotPlays(
  observation: BotPlayObservation,
): BotPlayAction[] {
  assertObservation(observation);
  const groups = groupedNormalCards(observation.hand);
  const jokers = sortedCards(
    observation.hand.filter((card) => card.rank === JOKER_RANK),
  );
  const result: BotPlayAction[] = [];

  if (!observation.table) {
    for (const joker of jokers) {
      result.push(playAction([joker], JOKER_RANK));
    }
    if (jokers.length === 2) {
      result.push(playAction(jokers, JOKER_RANK));
    }

    for (const [rank, cards] of groups) {
      for (let naturalCount = 1; naturalCount <= cards.length; naturalCount += 1) {
        for (
          let jokerCount = 0;
          jokerCount <= jokers.length;
          jokerCount += 1
        ) {
          for (const naturals of combinations(cards, naturalCount)) {
            for (const wilds of combinations(jokers, jokerCount)) {
              result.push(playAction([...naturals, ...wilds], rank));
            }
          }
        }
      }
    }
  } else {
    const targetCount = observation.table.count;
    for (const [rank, cards] of groups) {
      if (rank >= observation.table.rank) continue;
      const minimumNaturals = Math.max(1, targetCount - jokers.length);
      const maximumNaturals = Math.min(cards.length, targetCount);
      for (
        let naturalCount = minimumNaturals;
        naturalCount <= maximumNaturals;
        naturalCount += 1
      ) {
        const jokerCount = targetCount - naturalCount;
        for (const naturals of combinations(cards, naturalCount)) {
          for (const wilds of combinations(jokers, jokerCount)) {
            result.push(playAction([...naturals, ...wilds], rank));
          }
        }
      }
    }
  }

  return result.sort(
    (left, right) =>
      left.rank - right.rank ||
      left.count - right.count ||
      left.jokerCount - right.jokerCount ||
      left.cardIds.join("\u0000").localeCompare(right.cardIds.join("\u0000")),
  );
}

function remainingHand(
  hand: readonly BotCard[],
  action: BotPlayAction,
): BotCard[] {
  const played = new Set(action.cardIds);
  return hand.filter((card) => !played.has(card.id));
}

function handGroupCounts(hand: readonly BotCard[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const card of hand) {
    counts.set(card.rank, (counts.get(card.rank) ?? 0) + 1);
  }
  return counts;
}

function estimatedTurns(hand: readonly BotCard[]): number {
  const normalRanks = new Set(
    hand
      .filter((card) => card.rank !== JOKER_RANK)
      .map((card) => card.rank),
  );
  const jokerCount = hand.filter((card) => card.rank === JOKER_RANK).length;
  if (normalRanks.size === 0) return Math.ceil(jokerCount / 2);
  return normalRanks.size;
}

function findPublicPlayer(
  observation: BotPlayObservation,
  playerId: string | undefined,
): BotPublicPlayer | undefined {
  return playerId
    ? observation.players.find((player) => player.id === playerId)
    : undefined;
}

function nextActiveOpponent(
  observation: BotPlayObservation,
): BotPublicPlayer | undefined {
  const actorIndex = observation.players.findIndex(
    (player) => player.id === observation.actorId,
  );
  const passed = new Set(observation.passedPlayerIds ?? []);
  for (let step = 1; step < observation.players.length; step += 1) {
    const player =
      observation.players[(actorIndex + step) % observation.players.length];
    if (
      player.id !== observation.actorId &&
      !player.finished &&
      player.handCount > 0 &&
      !passed.has(player.id)
    ) {
      return player;
    }
  }
  return undefined;
}

function structureDamage(
  hand: readonly BotCard[],
  action: BotPlayAction,
): number {
  if (action.rank === JOKER_RANK) return 0;
  const before = hand.filter((card) => card.rank === action.rank).length;
  const naturalPlayed = action.count - action.jokerCount;
  const after = before - naturalPlayed;
  if (after <= 0 || before <= 1) return 0;
  return 28 + (after === 1 ? 18 : 0) + Math.min(12, before * 2);
}

function copiesInDeck(rank: number): number {
  return rank === JOKER_RANK ? 2 : rank;
}

function publicCountByRank(
  publicCounts: readonly BotPublicCardCount[],
): Map<number, number> {
  const result = new Map<number, number>();
  for (const entry of publicCounts) {
    result.set(entry.rank, (result.get(entry.rank) ?? 0) + entry.count);
  }
  return result;
}

function unseenStrongerCards(
  observation: BotPlayObservation,
  action: BotPlayAction,
): number {
  const publicCounts = publicCountByRank(
    observation.publicPlayedCards ?? [],
  );
  const ownCounts = handGroupCounts(observation.hand);
  let unseen = 0;
  for (let rank = NORMAL_RANK_MIN; rank < action.rank; rank += 1) {
    unseen += Math.max(
      0,
      copiesInDeck(rank) -
        (publicCounts.get(rank) ?? 0) -
        (ownCounts.get(rank) ?? 0),
    );
  }
  return unseen;
}

function scorePlay(
  observation: BotPlayObservation,
  action: BotPlayAction,
  difficulty: Exclude<BotDifficulty, "easy">,
): BotPlayDecision {
  const after = remainingHand(observation.hand, action);
  const beforeCounts = handGroupCounts(observation.hand);
  const afterCounts = handGroupCounts(after);
  const reasons: string[] = [];
  const hard = difficulty === "hard";
  let score = 0;

  score += action.count * (hard ? 28 : 24);

  if (after.length === 0) {
    score += 100_000;
    reasons.push("즉시 완주");
  }

  if (action.rank === JOKER_RANK) {
    score -= hard ? 100 : 82;
  } else {
    const naturalCount = action.count - action.jokerCount;
    score -= (JOKER_RANK - action.rank) * naturalCount * (hard ? 2.4 : 2);
  }

  if (action.jokerCount > 0) {
    score -= action.jokerCount * (hard ? 92 : 78);
    reasons.push("조커 사용");
  } else {
    reasons.push("조커 보존");
  }

  const damage = structureDamage(observation.hand, action);
  score -= damage * (hard ? 1.25 : 1);
  if (damage > 0) {
    reasons.push("묶음 분리");
  } else if (
    action.rank !== JOKER_RANK &&
    (beforeCounts.get(action.rank) ?? 0) > 1
  ) {
    reasons.push("묶음 정리");
  }

  if (
    action.rank !== JOKER_RANK &&
    (beforeCounts.get(action.rank) ?? 0) > 0 &&
    !afterCounts.has(action.rank)
  ) {
    score += hard ? 24 : 18;
  }

  const turnsSaved = estimatedTurns(observation.hand) - estimatedTurns(after);
  score += turnsSaved * (hard ? 34 : 26);

  const tableLeader = findPublicPlayer(
    observation,
    observation.table?.playerId,
  );
  if (
    observation.table &&
    tableLeader &&
    !tableLeader.finished &&
    tableLeader.handCount <= 2
  ) {
    const threatReward =
      (3 - tableLeader.handCount) * (hard ? 85 : 58);
    score += threatReward;
    reasons.push("완주 위기 상대 저지");
  }

  const nextOpponent = nextActiveOpponent(observation);
  if (
    nextOpponent &&
    nextOpponent.handCount <= action.count &&
    after.length > 0
  ) {
    const rankRisk =
      action.rank === JOKER_RANK ? 1 : Math.max(0.15, action.rank / 12);
    const nextRisk =
      (hard ? 105 : 72) *
      rankRisk *
      (nextOpponent.handCount === action.count ? 1 : 0.7);
    score -= nextRisk;
    reasons.push("다음 상대 완주 위험");
  }

  if (hard && observation.publicPlayedCards?.length) {
    const unseen = unseenStrongerCards(observation, action);
    const publicControl = Math.max(-20, 22 - unseen * 2);
    score += publicControl;
    reasons.push(
      publicControl >= 0 ? "공개 카드 기반 주도권" : "남은 강한 카드 위험",
    );
  }

  return { action, score, reasons };
}

function scorePass(
  observation: BotPlayObservation,
  difficulty: Exclude<BotDifficulty, "easy">,
  legalPlays: readonly BotPlayAction[],
): BotPlayDecision {
  const reasons = ["패 보존"];
  let score = difficulty === "hard" ? 18 : 14;
  const tableLeader = findPublicPlayer(
    observation,
    observation.table?.playerId,
  );

  if (legalPlays.length === 0) {
    return {
      action: { type: "pass" },
      score: Number.POSITIVE_INFINITY,
      reasons: ["제출 가능한 조합 없음"],
    };
  }

  if (legalPlays.every((play) => play.jokerCount > 0)) {
    score += difficulty === "hard" ? 45 : 34;
    reasons.push("조커 절약");
  }

  if (
    tableLeader &&
    !tableLeader.finished &&
    tableLeader.handCount <= 2
  ) {
    score -=
      (3 - tableLeader.handCount) * (difficulty === "hard" ? 100 : 68);
    reasons.push("완주 위기 상대에게 주도권 허용");
  }

  return { action: { type: "pass" }, score, reasons };
}

function actionTieKey(action: BotAction): string {
  return action.type === "pass"
    ? "\uffff"
    : `${String(action.rank).padStart(2, "0")}:${String(action.count).padStart(
        2,
        "0",
      )}:${action.cardIds.join(",")}`;
}

function legacyEasyDecision(
  observation: BotPlayObservation,
): BotPlayDecision {
  const hand = sortedCards(observation.hand);
  const jokers = hand.filter((card) => card.rank === JOKER_RANK);
  const groups = groupedNormalCards(hand);

  if (!observation.table) {
    if (jokers.length > 0) {
      return {
        action: playAction([jokers[0]], JOKER_RANK),
        score: 0,
        reasons: ["쉬움 난이도 기본 수"],
      };
    }
    const rank = [...groups.keys()].sort((a, b) => b - a)[0];
    if (rank !== undefined) {
      return {
        action: playAction(groups.get(rank) ?? [], rank),
        score: 0,
        reasons: ["쉬움 난이도 기본 수"],
      };
    }
    return {
      action: { type: "pass" },
      score: 0,
      reasons: ["제출 가능한 조합 없음"],
    };
  }

  const targetCount = observation.table.count;
  const ranks = [...groups.keys()]
    .filter((rank) => rank < observation.table!.rank)
    .sort((a, b) => b - a);
  for (const rank of ranks) {
    const cards = groups.get(rank) ?? [];
    if (cards.length + jokers.length < targetCount) continue;
    const selected = [
      ...cards.slice(0, targetCount),
      ...jokers.slice(0, Math.max(0, targetCount - cards.length)),
    ];
    return {
      action: playAction(selected, rank),
      score: 0,
      reasons: ["쉬움 난이도 기본 수"],
    };
  }
  return {
    action: { type: "pass" },
    score: 0,
    reasons: ["제출 가능한 조합 없음"],
  };
}

/**
 * Chooses one public-information-safe play or PASS decision.
 *
 * Easy preserves the former deterministic bot behaviour. Normal and hard
 * score every legal play plus a strategic PASS. Hard adds stronger threat
 * weights and optional public-card counting.
 */
export function chooseBotPlay(
  observation: BotPlayObservation,
  difficulty: BotDifficulty = "normal",
): BotPlayDecision {
  assertDifficulty(difficulty);
  assertObservation(observation);
  if (difficulty === "easy") return legacyEasyDecision(observation);

  const legalPlays = enumerateLegalBotPlays(observation);
  if (!observation.table && legalPlays.length === 0) {
    return {
      action: { type: "pass" },
      score: Number.POSITIVE_INFINITY,
      reasons: ["손에 카드가 없음"],
    };
  }

  const decisions = legalPlays.map((action) =>
    scorePlay(observation, action, difficulty),
  );
  if (observation.table) {
    decisions.push(scorePass(observation, difficulty, legalPlays));
  }

  decisions.sort(
    (left, right) =>
      right.score - left.score ||
      actionTieKey(left.action).localeCompare(actionTieKey(right.action)),
  );
  return decisions[0];
}

/**
 * Compatibility helper for the existing quick/online engines, where null
 * means PASS.
 */
export function chooseBotCardIds(
  observation: BotPlayObservation,
  difficulty: BotDifficulty = "normal",
): string[] | null {
  const decision = chooseBotPlay(observation, difficulty);
  return decision.action.type === "play" ? decision.action.cardIds : null;
}

function taxPriority(card: BotCard): number {
  // Project house rule: jokers are the strongest cards when paying tax.
  return card.rank === JOKER_RANK ? 0 : card.rank;
}

/**
 * Forced peon tribute. This is a rule operation, not an intelligence choice.
 */
export function selectForcedBotTribute(
  hand: readonly BotCard[],
  count: number,
): string[] {
  assertCards(hand);
  if (!Number.isInteger(count) || count < 0 || count > hand.length) {
    throw new RangeError("invalid tribute count");
  }
  return hand
    .filter((card) => card.rank !== JOKER_RANK)
    .sort(
      (left, right) =>
        left.rank - right.rank ||
        left.id.localeCompare(right.id),
    )
    .slice(0, count)
    .map((card) => card.id);
}

function scoreTaxSelection(
  hand: readonly BotCard[],
  cards: readonly BotCard[],
  difficulty: Exclude<BotDifficulty, "easy">,
): number {
  const selectedCounts = handGroupCounts(cards);
  const handCounts = handGroupCounts(hand);
  let score = 0;

  for (const card of cards) {
    if (card.rank === JOKER_RANK) {
      score -= difficulty === "hard" ? 500 : 400;
      continue;
    }
    score += card.rank * (difficulty === "hard" ? 13 : 10);
    if ((handCounts.get(card.rank) ?? 0) === 1) {
      score += difficulty === "hard" ? 55 : 40;
    }
  }

  for (const [rank, selectedCount] of selectedCounts) {
    if (rank === JOKER_RANK) continue;
    const originalCount = handCounts.get(rank) ?? 0;
    const remainingCount = originalCount - selectedCount;
    if (originalCount > 1) {
      score -= (difficulty === "hard" ? 60 : 45) * selectedCount;
      if (remainingCount === 1) {
        score -= difficulty === "hard" ? 45 : 30;
      }
      if (remainingCount === 0) {
        score -= difficulty === "hard" ? 35 : 22;
      }
    }
    if (selectedCount > 1) {
      // Avoid handing the peon an immediately useful pair when alternatives
      // are available.
      score -= (difficulty === "hard" ? 35 : 20) * (selectedCount - 1);
    }
  }

  return score;
}

/**
 * Chooses cards a Dalmuti-side bot returns during taxation.
 *
 * Easy returns the individually weakest cards, matching the old bot. Normal
 * and hard prefer weak isolated singles and preserve useful pairs/triples.
 */
export function chooseBotTaxReturn(
  hand: readonly BotCard[],
  count: number,
  difficulty: BotDifficulty = "normal",
): BotTaxDecision {
  assertDifficulty(difficulty);
  assertCards(hand);
  if (!Number.isInteger(count) || count < 0 || count > hand.length) {
    throw new RangeError("invalid tax return count");
  }
  if (count === 0) {
    return { cardIds: [], score: 0, reasons: ["반환할 카드 없음"] };
  }

  if (difficulty === "easy") {
    const cards = [...hand]
      .sort(
        (left, right) =>
          taxPriority(right) - taxPriority(left) ||
          left.id.localeCompare(right.id),
      )
      .slice(0, count);
    return {
      cardIds: cards.map((card) => card.id),
      score: 0,
      reasons: ["쉬움 난이도: 가장 약한 카드 반환"],
    };
  }

  const choices = combinations(sortedCards(hand), count)
    .map((cards) => ({
      cards,
      score: scoreTaxSelection(hand, cards, difficulty),
    }))
    .sort(
      (left, right) =>
        right.score - left.score ||
        left.cards
          .map((card) => card.id)
          .join("\u0000")
          .localeCompare(right.cards.map((card) => card.id).join("\u0000")),
    );
  const best = choices[0];
  const selectedRanks = best.cards.map((card) => card.rank);
  return {
    cardIds: best.cards.map((card) => card.id),
    score: best.score,
    reasons: [
      selectedRanks.every(
        (rank) => hand.filter((card) => card.rank === rank).length === 1,
      )
        ? "고립된 약한 카드 우선"
        : "패 묶음 손실 최소화",
      "조커 보존",
    ],
  };
}

function handBurden(hand: readonly BotCard[]): number {
  const normalCards = hand.filter((card) => card.rank !== JOKER_RANK);
  if (normalCards.length === 0) return 0;
  return (
    normalCards.reduce((total, card) => total + card.rank, 0) /
    normalCards.length
  );
}

/**
 * Decides whether a two-joker holder declares a revolution.
 *
 * Easy keeps the former always-declare behaviour. Normal/hard compare the
 * public role's tax expectation: peons cancel a loss, Dalmutis retain a gain,
 * and a great peon always takes the rank-reversing great revolution.
 */
export function chooseBotRevolution(
  observation: BotRevolutionObservation,
  difficulty: BotDifficulty = "normal",
): BotRevolutionDecision {
  assertDifficulty(difficulty);
  assertCards(observation.hand);
  if (
    !Number.isInteger(observation.playerCount) ||
    observation.playerCount < 4
  ) {
    throw new RangeError("playerCount must be at least four");
  }
  const jokerCount = observation.hand.filter(
    (card) => card.rank === JOKER_RANK,
  ).length;
  if (jokerCount !== 2) {
    throw new TypeError("a revolution decision requires exactly two jokers");
  }

  const kind =
    observation.role === "great-peon"
      ? "great-revolution"
      : "revolution";
  if (difficulty === "easy") {
    return {
      declare: true,
      kind,
      score: 0,
      reasons: ["쉬움 난이도: 항상 혁명 선언"],
    };
  }

  if (observation.role === "great-peon") {
    return {
      declare: true,
      kind,
      score: 1_000,
      reasons: ["대혁명으로 최상위 계급 획득", "세금 취소"],
    };
  }

  const baseValue: Record<Exclude<BotRole, "great-peon">, number> = {
    "great-dalmuti": -95,
    "lesser-dalmuti": -62,
    merchant: -4,
    "lesser-peon": 72,
  };
  let score = baseValue[observation.role];
  const burden = handBurden(observation.hand);

  if (observation.role === "lesser-peon") {
    score += Math.max(0, burden - 6) * (difficulty === "hard" ? 5 : 3);
  } else if (
    observation.role === "great-dalmuti" ||
    observation.role === "lesser-dalmuti"
  ) {
    // A weak, fragmented noble hand values taxation even more.
    score -= Math.max(0, burden - 6) * (difficulty === "hard" ? 3 : 2);
  } else if (difficulty === "hard" && burden >= 8.5) {
    // A merchant receives no direct tax benefit. On hard, a very weak hand
    // slightly prefers disrupting the nobles instead of preserving status quo.
    score += 10;
  }

  return {
    declare: score > 0,
    kind,
    score,
    reasons:
      score > 0
        ? ["예상 세금 손실 차단", "현재 패 부담 고려"]
        : ["예상 세금 이득 또는 중립 상태 유지"],
  };
}

/**
 * Facedown rank cards contain no actionable information. Keeping this choice
 * random is both fair and stronger than pretending a bot can infer the card.
 */
export function chooseFacedownRankSlot(
  openSlotIndices: readonly number[],
  randomInt: (maxExclusive: number) => number,
): number | null {
  if (openSlotIndices.length === 0) return null;
  const selected = randomInt(openSlotIndices.length);
  if (
    !Number.isInteger(selected) ||
    selected < 0 ||
    selected >= openSlotIndices.length
  ) {
    throw new RangeError(
      `randomInt(${openSlotIndices.length}) returned an invalid value`,
    );
  }
  return openSlotIndices[selected];
}
