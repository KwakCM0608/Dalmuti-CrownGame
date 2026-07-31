import type {
  BotPlayObservation,
  BotRole,
} from "../lib/bot-strategy.ts";

export const OBSERVATION_SCHEMA_VERSION = 2;
export const MAX_TRAINING_PLAYERS = 10;
export const TRAINING_ROLES = [
  "great-dalmuti",
  "lesser-dalmuti",
  "merchant",
  "lesser-peon",
  "great-peon",
] as const satisfies readonly BotRole[];

export type RevolutionState =
  | null
  | "revolution"
  | "great-revolution";

export type ObservationContext = {
  observation: BotPlayObservation;
  round: number;
  rolesByPlayerId: Readonly<Record<string, BotRole>>;
  scoresByPlayerId: Readonly<Record<string, number>>;
  revolution: RevolutionState;
};

export type ObservationFeatureGroup = {
  name: string;
  offset: number;
  length: number;
  description: string;
};

const FEATURE_GROUP_DEFINITIONS = [
  ["global", 3, "player count, act number, actor rank seat"],
  ["actorRole", 5, "actor social role one-hot"],
  ["tablePresent", 1, "whether a submission is on the table"],
  ["tableRank", 13, "current table rank one-hot"],
  ["tableCount", 1, "current table card count"],
  ["ownHandCounts", 13, "actor hand counts by physical rank"],
  ["publicPlayedCounts", 13, "publicly submitted counts by physical rank"],
  [
    "relativePlayers",
    MAX_TRAINING_PLAYERS * 12,
    "clockwise public player slots: occupied, hand count, finished, passed, self, table leader, score, role one-hot",
  ],
  ["revolution", 3, "none, normal revolution, great revolution one-hot"],
] as const;

let featureOffset = 0;
export const OBSERVATION_FEATURE_GROUPS: readonly ObservationFeatureGroup[] =
  FEATURE_GROUP_DEFINITIONS.map(([name, length, description]) => {
    const group = {
      name,
      offset: featureOffset,
      length,
      description,
    };
    featureOffset += length;
    return group;
  });

export const OBSERVATION_FEATURE_COUNT = featureOffset;

function deckCopies(rank: number): number {
  return rank === 13 ? 2 : rank;
}

function roleOneHot(role: BotRole | undefined): number[] {
  return TRAINING_ROLES.map((candidate) => (candidate === role ? 1 : 0));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/**
 * Encodes only the acting player's private hand plus public information.
 * Opponents' hidden cards and private tax identities cannot enter this API.
 */
export function encodeTrainingObservation(
  context: ObservationContext,
): number[] {
  const {
    observation,
    round,
    rolesByPlayerId,
    scoresByPlayerId,
    revolution,
  } = context;
  const actorIndex = observation.players.findIndex(
    (player) => player.id === observation.actorId,
  );
  if (actorIndex < 0) {
    throw new TypeError("observation players must include the actor");
  }
  if (
    observation.players.length < 4 ||
    observation.players.length > MAX_TRAINING_PLAYERS
  ) {
    throw new RangeError("training observations support 4 to 10 players");
  }

  const features: number[] = [];
  features.push(
    (observation.players.length - 4) / 6,
    clamp01(round / 20),
    observation.players.length === 1
      ? 0
      : actorIndex / (observation.players.length - 1),
  );
  features.push(...roleOneHot(rolesByPlayerId[observation.actorId]));

  features.push(observation.table ? 1 : 0);
  for (let rank = 1; rank <= 13; rank += 1) {
    features.push(observation.table?.rank === rank ? 1 : 0);
  }
  features.push(
    observation.table ? clamp01(observation.table.count / 14) : 0,
  );

  for (let rank = 1; rank <= 13; rank += 1) {
    const count = observation.hand.filter((card) => card.rank === rank).length;
    features.push(clamp01(count / deckCopies(rank)));
  }

  const publicCounts = new Map<number, number>();
  for (const entry of observation.publicPlayedCards ?? []) {
    publicCounts.set(
      entry.rank,
      (publicCounts.get(entry.rank) ?? 0) + entry.count,
    );
  }
  for (let rank = 1; rank <= 13; rank += 1) {
    features.push(
      clamp01((publicCounts.get(rank) ?? 0) / deckCopies(rank)),
    );
  }

  const passed = new Set(observation.passedPlayerIds ?? []);
  const relativePlayers = Array.from(
    { length: observation.players.length },
    (_, relativeIndex) =>
      observation.players[
        (actorIndex + relativeIndex) % observation.players.length
      ],
  );
  for (let slot = 0; slot < MAX_TRAINING_PLAYERS; slot += 1) {
    const player = relativePlayers[slot];
    if (!player) {
      features.push(...Array.from({ length: 12 }, () => 0));
      continue;
    }
    features.push(
      1,
      clamp01(player.handCount / 20),
      player.finished ? 1 : 0,
      passed.has(player.id) ? 1 : 0,
      player.id === observation.actorId ? 1 : 0,
      player.id === observation.table?.playerId ? 1 : 0,
      Math.tanh((scoresByPlayerId[player.id] ?? 0) / 10),
      ...roleOneHot(rolesByPlayerId[player.id]),
    );
  }

  features.push(
    revolution === null ? 1 : 0,
    revolution === "revolution" ? 1 : 0,
    revolution === "great-revolution" ? 1 : 0,
  );

  if (features.length !== OBSERVATION_FEATURE_COUNT) {
    throw new Error(
      `observation encoder produced ${features.length} features; expected ${OBSERVATION_FEATURE_COUNT}`,
    );
  }
  if (features.some((feature) => !Number.isFinite(feature))) {
    throw new Error("observation encoder produced a non-finite feature");
  }
  return features;
}
