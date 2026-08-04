import type {
  BotAction,
  BotCard,
  BotPlayAction,
  BotPlayObservation,
} from "../lib/bot-strategy.ts";

export const PASS_ACTION_INDEX = 0;
export const SOLO_JOKER_ACTION_INDEX = 1;
export const NORMAL_RANK_COUNT = 12;
export const MAX_PLAY_COUNT = 14;
export const JOKER_COUNT_OPTIONS = 3;
export const ACTION_SPACE_SIZE =
  2 + NORMAL_RANK_COUNT * MAX_PLAY_COUNT * JOKER_COUNT_OPTIONS;
// Reuse a physically impossible V1 slot (rank 12, count 2, two jokers and no
// natural card) so deployed 506-logit policies remain binary-compatible.
export const DOUBLE_JOKER_ACTION_INDEX =
  2 +
  ((NORMAL_RANK_COUNT - 1) * MAX_PLAY_COUNT + (2 - 1)) *
    JOKER_COUNT_OPTIONS +
  2;

export type SemanticAction =
  | { type: "pass" }
  | { type: "solo-joker" }
  | { type: "double-joker" }
  | {
      type: "play";
      rank: number;
      count: number;
      jokerCount: number;
    };

function assertIntegerInRange(
  value: number,
  minimum: number,
  maximum: number,
  label: string,
): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(
      `${label} must be an integer from ${minimum} to ${maximum}`,
    );
  }
}

/**
 * Stable V1 action layout:
 *
 * 0 = PASS
 * 1 = a single joker
 * 2..505 = (normal rank 1..12, total count 1..14, joker count 0..2)
 *
 * Some indices are impossible for a particular hand. They remain in the
 * fixed action space and are disabled by the legal-action mask.
 */
export function encodeSemanticAction(action: SemanticAction): number {
  if (action.type === "pass") return PASS_ACTION_INDEX;
  if (action.type === "solo-joker") return SOLO_JOKER_ACTION_INDEX;
  if (action.type === "double-joker") return DOUBLE_JOKER_ACTION_INDEX;

  assertIntegerInRange(action.rank, 1, NORMAL_RANK_COUNT, "rank");
  assertIntegerInRange(action.count, 1, MAX_PLAY_COUNT, "count");
  assertIntegerInRange(action.jokerCount, 0, 2, "jokerCount");

  return (
    2 +
    ((action.rank - 1) * MAX_PLAY_COUNT + (action.count - 1)) *
      JOKER_COUNT_OPTIONS +
    action.jokerCount
  );
}

export function decodeSemanticAction(actionIndex: number): SemanticAction {
  assertIntegerInRange(
    actionIndex,
    0,
    ACTION_SPACE_SIZE - 1,
    "actionIndex",
  );
  if (actionIndex === PASS_ACTION_INDEX) return { type: "pass" };
  if (actionIndex === SOLO_JOKER_ACTION_INDEX) {
    return { type: "solo-joker" };
  }
  if (actionIndex === DOUBLE_JOKER_ACTION_INDEX) {
    return { type: "double-joker" };
  }

  const offset = actionIndex - 2;
  const jokerCount = offset % JOKER_COUNT_OPTIONS;
  const rankAndCount = Math.floor(offset / JOKER_COUNT_OPTIONS);
  const count = (rankAndCount % MAX_PLAY_COUNT) + 1;
  const rank = Math.floor(rankAndCount / MAX_PLAY_COUNT) + 1;
  return { type: "play", rank, count, jokerCount };
}

function handCounts(hand: readonly BotCard[]): number[] {
  const counts = Array.from({ length: 14 }, () => 0);
  for (const card of hand) {
    if (!Number.isInteger(card.rank) || card.rank < 1 || card.rank > 13) {
      throw new RangeError(`card ${card.id} has an invalid rank`);
    }
    counts[card.rank] += 1;
  }
  return counts;
}

export function legalSemanticActionIndices(
  observation: BotPlayObservation,
): number[] {
  const counts = handCounts(observation.hand);
  const jokerCountInHand = counts[13];
  const result: number[] = [];

  if (observation.table) {
    result.push(PASS_ACTION_INDEX);
  } else if (jokerCountInHand > 0) {
    result.push(SOLO_JOKER_ACTION_INDEX);
    if (jokerCountInHand >= 2) result.push(DOUBLE_JOKER_ACTION_INDEX);
  }

  for (let rank = 1; rank <= NORMAL_RANK_COUNT; rank += 1) {
    if (observation.table && rank >= observation.table.rank) continue;

    for (let jokers = 0; jokers <= jokerCountInHand; jokers += 1) {
      const naturalCount = observation.table
        ? observation.table.count - jokers
        : undefined;
      const minimumNatural = naturalCount ?? 1;
      const maximumNatural = naturalCount ?? counts[rank];

      for (
        let naturals = minimumNatural;
        naturals <= maximumNatural;
        naturals += 1
      ) {
        if (naturals < 1 || naturals > counts[rank]) continue;
        const totalCount = naturals + jokers;
        if (
          totalCount < 1 ||
          totalCount > MAX_PLAY_COUNT ||
          (observation.table &&
            totalCount !== observation.table.count)
        ) {
          continue;
        }
        result.push(
          encodeSemanticAction({
            type: "play",
            rank,
            count: totalCount,
            jokerCount: jokers,
          }),
        );
      }
    }
  }

  return [...new Set(result)].sort((left, right) => left - right);
}

export function legalSemanticActionMask(
  observation: BotPlayObservation,
): Uint8Array {
  const mask = new Uint8Array(ACTION_SPACE_SIZE);
  for (const actionIndex of legalSemanticActionIndices(observation)) {
    mask[actionIndex] = 1;
  }
  return mask;
}

export function semanticActionIndexFromBotAction(action: BotAction): number {
  if (action.type === "pass") return PASS_ACTION_INDEX;
  if (
    action.rank === 13 &&
    action.count === 1 &&
    action.jokerCount === 1
  ) {
    return SOLO_JOKER_ACTION_INDEX;
  }
  if (
    action.rank === 13 &&
    action.count === 2 &&
    action.jokerCount === 2
  ) {
    return DOUBLE_JOKER_ACTION_INDEX;
  }
  return encodeSemanticAction({
    type: "play",
    rank: action.rank,
    count: action.count,
    jokerCount: action.jokerCount,
  });
}

function sortedCards(cards: readonly BotCard[]): BotCard[] {
  return [...cards].sort(
    (left, right) =>
      left.rank - right.rank || left.id.localeCompare(right.id),
  );
}

/**
 * Converts a semantic policy choice into concrete card IDs for the production
 * engine. Equivalent physical copies are selected deterministically.
 */
export function resolveSemanticAction(
  observation: BotPlayObservation,
  actionIndex: number,
): BotAction {
  const legal = new Set(legalSemanticActionIndices(observation));
  if (!legal.has(actionIndex)) {
    throw new RangeError(`action ${actionIndex} is illegal in this state`);
  }

  const semantic = decodeSemanticAction(actionIndex);
  if (semantic.type === "pass") return { type: "pass" };

  const jokers = sortedCards(
    observation.hand.filter((card) => card.rank === 13),
  );
  if (semantic.type === "solo-joker") {
    const joker = jokers[0];
    if (!joker) throw new Error("legal solo-joker action has no joker");
    return {
      type: "play",
      cardIds: [joker.id],
      rank: 13,
      count: 1,
      jokerCount: 1,
    };
  }
  if (semantic.type === "double-joker") {
    const selectedJokers = jokers.slice(0, 2);
    if (selectedJokers.length !== 2) {
      throw new Error("legal double-joker action requires two jokers");
    }
    return {
      type: "play",
      cardIds: selectedJokers.map((card) => card.id),
      rank: 13,
      count: 2,
      jokerCount: 2,
    };
  }

  const naturals = sortedCards(
    observation.hand.filter((card) => card.rank === semantic.rank),
  ).slice(0, semantic.count - semantic.jokerCount);
  const selectedJokers = jokers.slice(0, semantic.jokerCount);
  const cards = [...naturals, ...selectedJokers];
  if (cards.length !== semantic.count) {
    throw new Error("legal semantic action could not be resolved to cards");
  }

  const action: BotPlayAction = {
    type: "play",
    cardIds: cards.map((card) => card.id).sort((a, b) => a.localeCompare(b)),
    rank: semantic.rank,
    count: semantic.count,
    jokerCount: semantic.jokerCount,
  };
  return action;
}
