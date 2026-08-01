import type { BotCard, BotRole } from "../lib/bot-strategy.ts";

export const NON_CARD_OBSERVATION_SCHEMA_VERSION = 1;
export const NON_CARD_MIN_PLAYERS = 4;
export const NON_CARD_MAX_PLAYERS = 10;
export const NON_CARD_MAX_DEALT_HAND_SIZE = 20;

export const NON_CARD_ROLES = [
  "great-dalmuti",
  "lesser-dalmuti",
  "merchant",
  "lesser-peon",
  "great-peon",
] as const satisfies readonly BotRole[];

export type NonCardPublicPlayer = {
  readonly id: string;
  readonly role: BotRole;
  readonly handCount: number;
  readonly score: number;
};

/**
 * The shared information boundary for pre-play decisions.
 *
 * `hand` is the acting player's hand. Opponent hands and tax-card identities
 * deliberately have no field in this type. `players` must be in current
 * social-rank order and contains public values only.
 */
export type NonCardPublicContext = {
  readonly actorId: string;
  readonly hand: readonly BotCard[];
  readonly players: readonly NonCardPublicPlayer[];
  readonly round: number;
};

export type TaxReturnObservation = NonCardPublicContext & {
  readonly returnCount: 1 | 2;
};

export type RevolutionObservation = NonCardPublicContext;

export type NonCardObservationFeatureGroup = {
  readonly name: string;
  readonly offset: number;
  readonly length: number;
  readonly description: string;
};

const COMMON_FEATURE_GROUP_DEFINITIONS = [
  ["global", 3, "player count, act number, own hand size"],
  ["actorRole", 5, "acting player's public social role one-hot"],
  ["ownHandCounts", 13, "acting player's private counts by physical rank"],
  [
    "relativePublicPlayers",
    NON_CARD_MAX_PLAYERS * 8,
    "actor-relative public slots: occupied, hand count, score, role one-hot",
  ],
] as const;

function featureGroups(
  suffix: readonly (readonly [string, number, string])[],
): readonly NonCardObservationFeatureGroup[] {
  let offset = 0;
  return Object.freeze(
    [...COMMON_FEATURE_GROUP_DEFINITIONS, ...suffix].map(
      ([name, length, description]) => {
        const group = Object.freeze({ name, offset, length, description });
        offset += length;
        return group;
      },
    ),
  );
}

export const TAX_RETURN_OBSERVATION_FEATURE_GROUPS = featureGroups([
  ["returnCount", 2, "one-card or two-card return one-hot"],
]);

export const REVOLUTION_OBSERVATION_FEATURE_GROUPS = featureGroups([
  ["taxApplies", 1, "whether declining preserves taxation in this act"],
]);

function totalFeatureCount(
  groups: readonly NonCardObservationFeatureGroup[],
): number {
  const last = groups.at(-1);
  return last ? last.offset + last.length : 0;
}

export const TAX_RETURN_OBSERVATION_FEATURE_COUNT = totalFeatureCount(
  TAX_RETURN_OBSERVATION_FEATURE_GROUPS,
);
export const REVOLUTION_OBSERVATION_FEATURE_COUNT = totalFeatureCount(
  REVOLUTION_OBSERVATION_FEATURE_GROUPS,
);

const HIDDEN_PLAYER_FIELDS = [
  "hand",
  "hands",
  "cards",
  "cardIds",
  "hiddenCards",
  "hiddenHand",
] as const;

function expectedRoleAt(index: number, playerCount: number): BotRole {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === playerCount - 2) return "lesser-peon";
  if (index === playerCount - 1) return "great-peon";
  return "merchant";
}

function deckCopies(rank: number): number {
  return rank === 13 ? 2 : rank;
}

function assertCards(hand: readonly BotCard[]): void {
  if (!Array.isArray(hand)) {
    throw new TypeError("hand must be an array");
  }
  if (hand.length > NON_CARD_MAX_DEALT_HAND_SIZE) {
    throw new RangeError(
      `pre-play hand cannot exceed ${NON_CARD_MAX_DEALT_HAND_SIZE} cards`,
    );
  }
  const ids = new Set<string>();
  const rankCounts = Array.from({ length: 14 }, () => 0);
  for (const card of hand) {
    if (!card || typeof card.id !== "string" || card.id.length === 0) {
      throw new TypeError("every card must have a non-empty id");
    }
    if (ids.has(card.id)) {
      throw new TypeError(`duplicate card id: ${card.id}`);
    }
    ids.add(card.id);
    if (!Number.isInteger(card.rank) || card.rank < 1 || card.rank > 13) {
      throw new RangeError(`card ${card.id} has an invalid rank`);
    }
    rankCounts[card.rank] += 1;
    if (rankCounts[card.rank] > deckCopies(card.rank)) {
      throw new RangeError(`hand contains too many rank ${card.rank} cards`);
    }
  }
}

/**
 * Validates the public/private boundary shared by both non-card decisions.
 * In particular, an opponent entry carrying card-level fields is rejected
 * instead of being silently ignored.
 */
export function validateNonCardPublicContext(
  context: NonCardPublicContext,
): void {
  if (!context || typeof context !== "object") {
    throw new TypeError("non-card observation must be an object");
  }
  if (typeof context.actorId !== "string" || context.actorId.length === 0) {
    throw new TypeError("actorId must be a non-empty string");
  }
  if (!Number.isInteger(context.round) || context.round < 1) {
    throw new RangeError("round must be a positive integer");
  }
  if (
    !Array.isArray(context.players) ||
    context.players.length < NON_CARD_MIN_PLAYERS ||
    context.players.length > NON_CARD_MAX_PLAYERS
  ) {
    throw new RangeError("non-card observations support 4 to 10 players");
  }
  assertCards(context.hand);

  const playerIds = new Set<string>();
  let actor: NonCardPublicPlayer | undefined;
  for (const [index, player] of context.players.entries()) {
    if (!player || typeof player !== "object") {
      throw new TypeError("every public player must be an object");
    }
    for (const field of HIDDEN_PLAYER_FIELDS) {
      if (Object.hasOwn(player, field)) {
        throw new TypeError(
          `public player ${String(player.id)} must not contain ${field}`,
        );
      }
    }
    if (typeof player.id !== "string" || player.id.length === 0) {
      throw new TypeError("every public player must have a non-empty id");
    }
    if (playerIds.has(player.id)) {
      throw new TypeError(`duplicate public player id: ${player.id}`);
    }
    playerIds.add(player.id);
    const expectedRole = expectedRoleAt(index, context.players.length);
    if (player.role !== expectedRole) {
      throw new TypeError(
        `player ${player.id} at rank seat ${index} must be ${expectedRole}`,
      );
    }
    if (
      !Number.isInteger(player.handCount) ||
      player.handCount < 0 ||
      player.handCount > NON_CARD_MAX_DEALT_HAND_SIZE
    ) {
      throw new RangeError(
        `public hand count for ${player.id} must be from 0 to ${NON_CARD_MAX_DEALT_HAND_SIZE}`,
      );
    }
    if (!Number.isFinite(player.score)) {
      throw new RangeError(`public score for ${player.id} must be finite`);
    }
    if (player.id === context.actorId) actor = player;
  }
  if (!actor) {
    throw new TypeError("players must include actorId");
  }
  if (actor.handCount !== context.hand.length) {
    throw new TypeError("actor public hand count must match the private hand");
  }
}

export function validateTaxReturnObservation(
  observation: TaxReturnObservation,
): void {
  validateNonCardPublicContext(observation);
  const actor = observation.players.find(
    (player) => player.id === observation.actorId,
  );
  const expectedReturnCount =
    actor?.role === "great-dalmuti"
      ? 2
      : actor?.role === "lesser-dalmuti"
        ? 1
        : null;
  if (expectedReturnCount === null) {
    throw new TypeError("only a Dalmuti-side player chooses a tax return");
  }
  if (observation.returnCount !== expectedReturnCount) {
    throw new TypeError(
      `${actor?.role} must return exactly ${expectedReturnCount} card(s)`,
    );
  }
  if (observation.hand.length < observation.returnCount) {
    throw new RangeError("tax return count exceeds the actor hand");
  }
}

export function validateRevolutionObservation(
  observation: RevolutionObservation,
): void {
  validateNonCardPublicContext(observation);
  const jokerCount = observation.hand.filter((card) => card.rank === 13).length;
  if (jokerCount !== 2) {
    throw new TypeError("a revolution decision requires exactly two jokers");
  }
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function roleOneHot(role: BotRole): number[] {
  return NON_CARD_ROLES.map((candidate) => (candidate === role ? 1 : 0));
}

function encodeCommonObservation(context: NonCardPublicContext): number[] {
  const actorIndex = context.players.findIndex(
    (player) => player.id === context.actorId,
  );
  const actor = context.players[actorIndex];
  const features: number[] = [
    (context.players.length - NON_CARD_MIN_PLAYERS) /
      (NON_CARD_MAX_PLAYERS - NON_CARD_MIN_PLAYERS),
    clamp01((context.round - 1) / 19),
    context.hand.length / NON_CARD_MAX_DEALT_HAND_SIZE,
    ...roleOneHot(actor.role),
  ];

  const handCounts = Array.from({ length: 14 }, () => 0);
  for (const card of context.hand) handCounts[card.rank] += 1;
  for (let rank = 1; rank <= 13; rank += 1) {
    features.push(handCounts[rank] / deckCopies(rank));
  }

  const relativePlayers = Array.from(
    { length: context.players.length },
    (_, relativeIndex) =>
      context.players[(actorIndex + relativeIndex) % context.players.length],
  );
  for (let slot = 0; slot < NON_CARD_MAX_PLAYERS; slot += 1) {
    const player = relativePlayers[slot];
    if (!player) {
      features.push(0, 0, 0, 0, 0, 0, 0, 0);
      continue;
    }
    features.push(
      1,
      player.handCount / NON_CARD_MAX_DEALT_HAND_SIZE,
      Math.tanh(player.score / 10),
      ...roleOneHot(player.role),
    );
  }
  return features;
}

function assertFeatureVector(features: readonly number[], expected: number): void {
  if (features.length !== expected) {
    throw new Error(
      `non-card encoder produced ${features.length} features; expected ${expected}`,
    );
  }
  if (features.some((feature) => !Number.isFinite(feature))) {
    throw new Error("non-card encoder produced a non-finite feature");
  }
}

/**
 * Encodes a noble's pre-tribute hand and public state. The incoming tribute's
 * card identities are intentionally absent because current production locks
 * the return before tribute cards move.
 */
export function encodeTaxReturnObservation(
  observation: TaxReturnObservation,
): number[] {
  validateTaxReturnObservation(observation);
  const features = encodeCommonObservation(observation);
  features.push(
    observation.returnCount === 1 ? 1 : 0,
    observation.returnCount === 2 ? 1 : 0,
  );
  assertFeatureVector(features, TAX_RETURN_OBSERVATION_FEATURE_COUNT);
  return features;
}

/** Encodes the two-joker holder's hand plus public, pre-play information. */
export function encodeRevolutionObservation(
  observation: RevolutionObservation,
): number[] {
  validateRevolutionObservation(observation);
  const features = encodeCommonObservation(observation);
  features.push(observation.round > 1 ? 1 : 0);
  assertFeatureVector(features, REVOLUTION_OBSERVATION_FEATURE_COUNT);
  return features;
}
