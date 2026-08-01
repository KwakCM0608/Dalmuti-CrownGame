import type { BotCard } from "../lib/bot-strategy.ts";
import {
  validateRevolutionObservation,
  validateTaxReturnObservation,
  type RevolutionObservation,
  type TaxReturnObservation,
} from "./non-card-observation.ts";

export const TAX_RETURN_ACTION_CATALOGUE_VERSION = 1;
export const REVOLUTION_ACTION_CATALOGUE_VERSION = 1;

export type TaxReturnSemanticAction = {
  readonly type: "tax-return";
  /** Sorted physical ranks. Cards sharing a rank are gameplay-equivalent. */
  readonly ranks: readonly number[];
};

export type RevolutionSemanticAction =
  | { readonly type: "decline" }
  | { readonly type: "declare" };

function createTaxReturnCatalogue(): readonly TaxReturnSemanticAction[] {
  const actions: TaxReturnSemanticAction[] = [];
  for (let rank = 1; rank <= 13; rank += 1) {
    actions.push(
      Object.freeze({
        type: "tax-return" as const,
        ranks: Object.freeze([rank]),
      }),
    );
  }
  for (let firstRank = 1; firstRank <= 13; firstRank += 1) {
    for (let secondRank = firstRank; secondRank <= 13; secondRank += 1) {
      // There is only one physical rank-1 card in the deck.
      if (firstRank === 1 && secondRank === 1) continue;
      actions.push(
        Object.freeze({
          type: "tax-return" as const,
          ranks: Object.freeze([firstRank, secondRank]),
        }),
      );
    }
  }
  return Object.freeze(actions);
}

/**
 * Stable semantic catalogue: 13 single-rank actions followed by 90 sorted
 * two-rank multisets. Duplicate physical cards of one rank intentionally map
 * to one action because they have identical game effects.
 */
export const TAX_RETURN_ACTION_CATALOGUE = createTaxReturnCatalogue();
export const TAX_RETURN_ACTION_COUNT = TAX_RETURN_ACTION_CATALOGUE.length;

if (TAX_RETURN_ACTION_COUNT !== 103) {
  throw new Error(
    `tax-return catalogue has ${TAX_RETURN_ACTION_COUNT} actions; expected 103`,
  );
}

export const REVOLUTION_ACTION_CATALOGUE = Object.freeze([
  Object.freeze({ type: "decline" as const }),
  Object.freeze({ type: "declare" as const }),
]);
export const REVOLUTION_ACTION_COUNT = REVOLUTION_ACTION_CATALOGUE.length;
export const REVOLUTION_DECLINE_ACTION_INDEX = 0;
export const REVOLUTION_DECLARE_ACTION_INDEX = 1;

const taxActionIndicesByRanks = new Map(
  TAX_RETURN_ACTION_CATALOGUE.map((action, actionIndex) => [
    action.ranks.join("+"),
    actionIndex,
  ]),
);

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

export function encodeTaxReturnAction(ranks: readonly number[]): number {
  if (!Array.isArray(ranks) || (ranks.length !== 1 && ranks.length !== 2)) {
    throw new RangeError("tax-return action must contain one or two ranks");
  }
  const canonicalRanks = [...ranks].sort((left, right) => left - right);
  for (const rank of canonicalRanks) {
    assertIntegerInRange(rank, 1, 13, "tax-return rank");
  }
  const actionIndex = taxActionIndicesByRanks.get(canonicalRanks.join("+"));
  if (actionIndex === undefined) {
    throw new RangeError(
      `tax-return ranks ${canonicalRanks.join("+")} are structurally impossible`,
    );
  }
  return actionIndex;
}

export function decodeTaxReturnAction(
  actionIndex: number,
): TaxReturnSemanticAction {
  assertIntegerInRange(
    actionIndex,
    0,
    TAX_RETURN_ACTION_COUNT - 1,
    "tax-return actionIndex",
  );
  return TAX_RETURN_ACTION_CATALOGUE[actionIndex];
}

export function decodeRevolutionAction(
  actionIndex: number,
): RevolutionSemanticAction {
  assertIntegerInRange(
    actionIndex,
    0,
    REVOLUTION_ACTION_COUNT - 1,
    "revolution actionIndex",
  );
  return REVOLUTION_ACTION_CATALOGUE[actionIndex];
}

function handRankCounts(hand: readonly BotCard[]): number[] {
  const counts = Array.from({ length: 14 }, () => 0);
  for (const card of hand) counts[card.rank] += 1;
  return counts;
}

function actionFitsHand(
  action: TaxReturnSemanticAction,
  handCounts: readonly number[],
): boolean {
  const required = Array.from({ length: 14 }, () => 0);
  for (const rank of action.ranks) required[rank] += 1;
  return required.every((count, rank) => count <= (handCounts[rank] ?? 0));
}

export function legalTaxReturnActionMask(
  observation: TaxReturnObservation,
): boolean[] {
  validateTaxReturnObservation(observation);
  const counts = handRankCounts(observation.hand);
  const mask = TAX_RETURN_ACTION_CATALOGUE.map(
    (action) =>
      action.ranks.length === observation.returnCount &&
      actionFitsHand(action, counts),
  );
  if (!mask.some(Boolean)) {
    throw new Error("valid tax observation produced no legal action");
  }
  return mask;
}

export function legalTaxReturnActionIndices(
  observation: TaxReturnObservation,
): number[] {
  return legalTaxReturnActionMask(observation).flatMap(
    (legal, actionIndex) => (legal ? [actionIndex] : []),
  );
}

export function legalRevolutionActionMask(
  observation: RevolutionObservation,
): boolean[] {
  validateRevolutionObservation(observation);
  return [true, true];
}

export function legalRevolutionActionIndices(
  observation: RevolutionObservation,
): number[] {
  legalRevolutionActionMask(observation);
  return [REVOLUTION_DECLINE_ACTION_INDEX, REVOLUTION_DECLARE_ACTION_INDEX];
}

export const TAX_RETURN_ACTION_FEATURE_LAYOUT = Object.freeze([
  "returns-one-card",
  "returns-two-cards",
  ...Array.from({ length: 13 }, (_, index) => `rank-${index + 1}-fraction`),
]);
export const TAX_RETURN_ACTION_FEATURE_COUNT =
  TAX_RETURN_ACTION_FEATURE_LAYOUT.length;

/**
 * Encodes the return size and selected rank multiplicities. Rank features are
 * divided by two, so a repeated pair is 1 and a single occurrence is 0.5.
 */
export function encodeTaxReturnActionFeatures(
  actionOrIndex: TaxReturnSemanticAction | number,
): number[] {
  const action =
    typeof actionOrIndex === "number"
      ? decodeTaxReturnAction(actionOrIndex)
      : decodeTaxReturnAction(encodeTaxReturnAction(actionOrIndex.ranks));
  const features = [
    action.ranks.length === 1 ? 1 : 0,
    action.ranks.length === 2 ? 1 : 0,
    ...Array.from({ length: 13 }, () => 0),
  ];
  for (const rank of action.ranks) features[rank + 1] += 0.5;
  return features;
}

export const TAX_RETURN_ACTION_FEATURES = Object.freeze(
  TAX_RETURN_ACTION_CATALOGUE.map((action) =>
    Object.freeze(encodeTaxReturnActionFeatures(action)),
  ),
);

export const REVOLUTION_ACTION_FEATURE_LAYOUT = Object.freeze([
  "decline",
  "declare-normal-revolution",
  "declare-great-revolution",
]);
export const REVOLUTION_ACTION_FEATURE_COUNT =
  REVOLUTION_ACTION_FEATURE_LAYOUT.length;

/** Declaration kind is an exact public consequence of the holder's role. */
export function encodeRevolutionActionFeatures(
  observation: RevolutionObservation,
  actionIndex: number,
): number[] {
  validateRevolutionObservation(observation);
  const action = decodeRevolutionAction(actionIndex);
  const actor = observation.players.find(
    (player) => player.id === observation.actorId,
  );
  if (action.type === "decline") return [1, 0, 0];
  return actor?.role === "great-peon" ? [0, 0, 1] : [0, 1, 0];
}

/** Resolves a semantic rank action to stable physical card IDs. */
export function resolveTaxReturnAction(
  observation: TaxReturnObservation,
  actionIndex: number,
): string[] {
  const legalMask = legalTaxReturnActionMask(observation);
  const action = decodeTaxReturnAction(actionIndex);
  if (!legalMask[actionIndex]) {
    throw new RangeError(`tax-return action ${actionIndex} is illegal`);
  }
  const cardsByRank = new Map<number, BotCard[]>();
  for (const card of observation.hand) {
    const cards = cardsByRank.get(card.rank) ?? [];
    cards.push(card);
    cardsByRank.set(card.rank, cards);
  }
  for (const cards of cardsByRank.values()) {
    cards.sort((left, right) => left.id.localeCompare(right.id));
  }
  return action.ranks.map((rank) => {
    const card = cardsByRank.get(rank)?.shift();
    if (!card) {
      throw new Error(`legal rank ${rank} could not resolve to a physical card`);
    }
    return card.id;
  });
}

export type TaxReturnActionCandidate = {
  readonly actionIndex: number;
  readonly action: TaxReturnSemanticAction;
  readonly actionFeatures: readonly number[];
  readonly cardIds: readonly string[];
};

export function enumerateLegalTaxReturnActions(
  observation: TaxReturnObservation,
): TaxReturnActionCandidate[] {
  return legalTaxReturnActionIndices(observation).map((actionIndex) => ({
    actionIndex,
    action: decodeTaxReturnAction(actionIndex),
    actionFeatures: TAX_RETURN_ACTION_FEATURES[actionIndex],
    cardIds: resolveTaxReturnAction(observation, actionIndex),
  }));
}

export type RevolutionActionCandidate = {
  readonly actionIndex: number;
  readonly action: RevolutionSemanticAction;
  readonly actionFeatures: readonly number[];
  readonly declarationKind: "revolution" | "great-revolution" | null;
};

export function enumerateLegalRevolutionActions(
  observation: RevolutionObservation,
): RevolutionActionCandidate[] {
  validateRevolutionObservation(observation);
  const actor = observation.players.find(
    (player) => player.id === observation.actorId,
  );
  return legalRevolutionActionIndices(observation).map((actionIndex) => {
    const action = decodeRevolutionAction(actionIndex);
    return {
      actionIndex,
      action,
      actionFeatures: encodeRevolutionActionFeatures(observation, actionIndex),
      declarationKind:
        action.type === "decline"
          ? null
          : actor?.role === "great-peon"
            ? "great-revolution"
            : "revolution",
    };
  });
}
