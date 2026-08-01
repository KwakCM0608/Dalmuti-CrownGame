import {
  chooseBotPlay,
  chooseBotRevolution,
  chooseBotTaxReturn,
  selectForcedBotTribute,
  type BotAction,
  type BotCard,
  type BotDifficulty,
  type BotPlayObservation,
  type BotRole,
} from "../lib/bot-strategy.ts";
import { rankedDealCounts } from "../lib/dealing.ts";
import { roundChipAward } from "../lib/round-score.ts";
import {
  legalSemanticActionIndices,
  resolveSemanticAction,
  semanticActionIndexFromBotAction,
} from "./action-space.ts";
import {
  REVOLUTION_ACTION_CATALOGUE_VERSION,
  REVOLUTION_DECLARE_ACTION_INDEX,
  REVOLUTION_DECLINE_ACTION_INDEX,
  TAX_RETURN_ACTION_CATALOGUE_VERSION,
  encodeTaxReturnAction,
  legalRevolutionActionIndices,
  legalTaxReturnActionIndices,
  resolveTaxReturnAction,
} from "./non-card-action-space.ts";
import {
  NON_CARD_OBSERVATION_SCHEMA_VERSION,
  encodeRevolutionObservation,
  encodeTaxReturnObservation,
  type RevolutionObservation,
  type TaxReturnObservation,
} from "./non-card-observation.ts";
import {
  encodeTrainingObservation,
  type RevolutionState,
} from "./observation.ts";
import { SeededRandom } from "./random.ts";

const MAX_TRANSITIONS_PER_ACT = 20_000;

type SimulationPlayer = {
  id: string;
  difficulty: BotDifficulty;
  role: BotRole;
  score: number;
};

type SimulationTable = {
  rank: number;
  count: number;
  playerId: string;
};

export type TrainingPolicyContext = {
  observation: BotPlayObservation;
  encodedObservation: readonly number[];
  legalActionIndices: readonly number[];
  actorRole: BotRole;
  actorScore: number;
  round: number;
  random: () => number;
};

export type TrainingPolicyDecision = {
  actionIndex: number;
  logProbability?: number;
  valueEstimate?: number;
  policyVersion?: string;
};

export type TrainingPolicy = (
  context: TrainingPolicyContext,
) => number | TrainingPolicyDecision;

export type NonCardDecisionKind = "tax-return" | "revolution";

export type TrainingNonCardPolicyDecision = {
  actionIndex: number;
  logProbability?: number;
  valueEstimate?: number;
  policyVersion?: string;
};

type TrainingNonCardPolicyContextBase = {
  episodeId: string;
  round: number;
  actorId: string;
  actorSeat: number;
  actorRole: BotRole;
  actorScore: number;
  decisionKey: string;
  encodedObservation: readonly number[];
  legalActionIndices: readonly number[];
  /**
   * A decision-local deterministic stream. Consuming it cannot perturb deals
   * or card-play policy randomness in the simulator's environment stream.
   */
  random: () => number;
};

export type TrainingTaxReturnPolicyContext =
  TrainingNonCardPolicyContextBase & {
    decision: "tax-return";
    observation: TaxReturnObservation;
  };

export type TrainingRevolutionPolicyContext =
  TrainingNonCardPolicyContextBase & {
    decision: "revolution";
    observation: RevolutionObservation;
  };

export type TrainingTaxReturnPolicy = (
  context: TrainingTaxReturnPolicyContext,
) => number | TrainingNonCardPolicyDecision;

export type TrainingRevolutionPolicy = (
  context: TrainingRevolutionPolicyContext,
) => number | TrainingNonCardPolicyDecision;

/**
 * Overrides use separate namespaces because one noble can make both a
 * revolution and a tax-return decision in the same episode/round.
 */
export type TrainingNonCardForcedOverrides = {
  taxReturn?: Readonly<Record<string, number>>;
  revolution?: Readonly<Record<string, number>>;
};

export type TrainingNonCardPublicPlayerSnapshot = {
  readonly id: string;
  readonly role: BotRole;
  readonly handCount: number;
  readonly score: number;
};

export type TrainingNonCardDeterminizationExpectedState = {
  readonly decision: NonCardDecisionKind;
  readonly decisionKey: string;
  readonly round: number;
  readonly decisionStep: number;
  readonly actorId: string;
  readonly encodedObservation: readonly number[];
  readonly legalActionIndices: readonly number[];
  readonly baselineActionIndex: number;
  readonly returnCount: 1 | 2 | null;
  readonly publicPlayers: readonly TrainingNonCardPublicPlayerSnapshot[];
  readonly publicHistory: readonly SimulatedAct[];
};

/**
 * Training-only hidden-world override. It never changes the environment RNG:
 * only non-actor cards in the target act are shuffled by `hiddenWorldSeed`.
 */
export type TrainingNonCardDeterminizationOverride = {
  readonly algorithmVersion: 1;
  readonly hiddenWorldSeed: number;
  readonly resampleOpponents?: boolean;
  readonly continuationSeed?: number | null;
  readonly expected: TrainingNonCardDeterminizationExpectedState;
};

/**
 * Presence enables privacy-safe non-card decision recording. An empty object
 * records the existing heuristic decisions without changing their behavior.
 */
export type TrainingNonCardHooks = {
  taxReturnPolicy?: TrainingTaxReturnPolicy;
  revolutionPolicy?: TrainingRevolutionPolicy;
  forcedOverrides?: TrainingNonCardForcedOverrides;
  determinization?: TrainingNonCardDeterminizationOverride;
};

export type SimulationConfig = {
  playerCount: number;
  acts?: number;
  seed?: number;
  difficulties?: readonly BotDifficulty[];
  episodeId?: string;
  policy?: TrainingPolicy;
  policyByPlayerId?: Readonly<Record<string, TrainingPolicy | undefined>>;
  supervisionPolicy?: TrainingPolicy;
  nonCard?: TrainingNonCardHooks;
};

export type TrainingStep = {
  episodeId: string;
  round: number;
  step: number;
  actorId: string;
  actorSeat: number;
  actorRole: BotRole;
  behaviorPolicy: BotDifficulty | "custom";
  observation: number[];
  legalActionIndices: number[];
  actionIndex: number;
  supervisedActionIndex: number | null;
  behaviorLogProbability: number | null;
  behaviorValueEstimate: number | null;
  behaviorPolicyVersion: string | null;
  forced: boolean;
  reward: number;
  actorTerminal: boolean;
  environmentTerminal: boolean;
  finishPlace: number;
};

type TrainingNonCardStepBase = {
  episodeId: string;
  round: number;
  step: number;
  actorId: string;
  actorSeat: number;
  actorRole: BotRole;
  decisionKey: string;
  observationSchemaVersion: number;
  actionCatalogueVersion: number;
  behaviorPolicy: BotDifficulty | "custom" | "forced-override";
  observation: number[];
  legalActionIndices: number[];
  actionIndex: number;
  behaviorLogProbability: number | null;
  behaviorValueEstimate: number | null;
  behaviorPolicyVersion: string | null;
  forced: boolean;
  forcedOverride: boolean;
  publicPlayers: TrainingNonCardPublicPlayerSnapshot[];
  reward: number;
  finishPlace: number;
};

export type TrainingTaxReturnStep = TrainingNonCardStepBase & {
  decision: "tax-return";
  metadata: {
    playerCount: number;
    actorHandCount: number;
    returnCount: 1 | 2;
  };
};

export type TrainingRevolutionStep = TrainingNonCardStepBase & {
  decision: "revolution";
  metadata: {
    playerCount: number;
    actorHandCount: number;
    declarationKind: RevolutionState;
  };
};

export type TrainingNonCardStep =
  | TrainingTaxReturnStep
  | TrainingRevolutionStep;

export type SimulatedAct = {
  round: number;
  revolution: RevolutionState;
  playerOrder: string[];
  finishOrder: string[];
  chipAwards: Record<string, number>;
  transitions: number;
};

export type SimulatedMatch = {
  episodeId: string;
  seed: number;
  playerCount: number;
  acts: SimulatedAct[];
  steps: TrainingStep[];
  /** Omitted exactly when `SimulationConfig.nonCard` is omitted. */
  nonCardSteps?: TrainingNonCardStep[];
  /** Present only for an explicit training-only determinization override. */
  nonCardDeterminizationAudit?: TrainingNonCardDeterminizationAudit;
  finalScores: Record<string, number>;
};

export type TrainingNonCardDeterminizationAudit = {
  algorithmVersion: 1;
  hiddenWorldSeed: number;
  targetDecision: NonCardDecisionKind;
  targetRound: number;
  targetActorId: string;
  physicalCardCount: 80;
  uniquePhysicalCardCount: 80;
  actorHandPreserved: true;
  publicHandCountsPreserved: true;
  publicHistoryPreserved: true;
  environmentRandomDrawsConsumed: 0;
  hiddenWorldKind: "original-replay" | "resampled";
  opponentCardsResampled: number;
  changedOpponentOwnershipCards: number;
  continuationRngReplaced: boolean;
  targetEncodedObservationPreserved: true;
  targetLegalActionsPreserved: true;
  targetBaselineActionPreserved: true;
  targetReturnCountPreserved: true;
  targetPublicPlayersPreserved: true;
  taxTransfer:
    | {
        exchangeCount: 2;
        tributeCardCount: 3;
        returnCardCount: 3;
        ownershipCompleteAfterTransfer: true;
        publicHandCountsRestoredAfterTransfer: true;
      }
    | null;
};

export class NonCardDeterminizationRejectedError extends Error {
  readonly reasonCode: string;

  constructor(reasonCode: string, message: string) {
    super(message);
    this.name = "NonCardDeterminizationRejectedError";
    this.reasonCode = reasonCode;
  }
}

export type TrainingNonCardDecisionIdentity = {
  episodeId: string;
  round: number;
  actorId: string;
};

/** A collision-free, JSON-serializable key for paired action overrides. */
export function createTrainingNonCardDecisionKey(
  identity: TrainingNonCardDecisionIdentity,
): string {
  if (
    typeof identity.episodeId !== "string" ||
    identity.episodeId.length === 0
  ) {
    throw new TypeError("non-card episodeId must be a non-empty string");
  }
  if (!Number.isInteger(identity.round) || identity.round < 1) {
    throw new RangeError("non-card round must be a positive integer");
  }
  if (typeof identity.actorId !== "string" || identity.actorId.length === 0) {
    throw new TypeError("non-card actorId must be a non-empty string");
  }
  return JSON.stringify([
    identity.episodeId,
    identity.round,
    identity.actorId,
  ]);
}

type PreparedNonCardOverrides = {
  taxReturn: Map<string, number>;
  revolution: Map<string, number>;
  consumedTaxReturn: Set<string>;
  consumedRevolution: Set<string>;
};

type NonCardRuntime = {
  hooks: TrainingNonCardHooks;
  matchSeed: number;
  overrides: PreparedNonCardOverrides;
  determinization?: {
    request: TrainingNonCardDeterminizationOverride;
    publicHistoryValidated: boolean;
    applied: boolean;
    targetValidated: boolean;
    audit: TrainingNonCardDeterminizationAudit | null;
  };
};

function rejectDeterminization(reasonCode: string, message: string): never {
  throw new NonCardDeterminizationRejectedError(reasonCode, message);
}

function assertFiniteNumberArray(
  value: readonly number[],
  label: string,
): void {
  if (!Array.isArray(value) || value.some((item) => !Number.isFinite(item))) {
    throw new TypeError(`${label} must be a finite number array`);
  }
}

function prepareDeterminization(
  request: TrainingNonCardDeterminizationOverride | undefined,
): NonCardRuntime["determinization"] {
  if (request === undefined) return undefined;
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    throw new TypeError("determinization must be an object");
  }
  if (request.algorithmVersion !== 1) {
    throw new RangeError("unsupported non-card determinization algorithm");
  }
  if (
    !Number.isSafeInteger(request.hiddenWorldSeed) ||
    request.hiddenWorldSeed < 0 ||
    request.hiddenWorldSeed > 0xffff_ffff
  ) {
    throw new RangeError(
      "determinization hiddenWorldSeed must be an unsigned 32-bit integer",
    );
  }
  if (
    request.resampleOpponents !== undefined &&
    typeof request.resampleOpponents !== "boolean"
  ) {
    throw new TypeError("determinization resampleOpponents must be boolean");
  }
  if (
    request.continuationSeed !== undefined &&
    request.continuationSeed !== null &&
    (!Number.isSafeInteger(request.continuationSeed) ||
      request.continuationSeed < 0 ||
      request.continuationSeed > 0xffff_ffff)
  ) {
    throw new RangeError(
      "determinization continuationSeed must be an unsigned 32-bit integer or null",
    );
  }
  const expected = request.expected;
  if (!expected || typeof expected !== "object" || Array.isArray(expected)) {
    throw new TypeError("determinization expected state must be an object");
  }
  if (expected.decision !== "tax-return" && expected.decision !== "revolution") {
    throw new TypeError("determinization expected decision is unsupported");
  }
  if (
    typeof expected.decisionKey !== "string" ||
    expected.decisionKey.length === 0 ||
    typeof expected.actorId !== "string" ||
    expected.actorId.length === 0
  ) {
    throw new TypeError("determinization target identifiers must be non-empty");
  }
  if (!Number.isInteger(expected.round) || expected.round < 1) {
    throw new RangeError("determinization target round must be positive");
  }
  if (!Number.isInteger(expected.decisionStep) || expected.decisionStep < 0) {
    throw new RangeError("determinization decisionStep must be non-negative");
  }
  assertFiniteNumberArray(
    expected.encodedObservation,
    "determinization encodedObservation",
  );
  if (
    !Array.isArray(expected.legalActionIndices) ||
    expected.legalActionIndices.length === 0 ||
    expected.legalActionIndices.some(
      (index, position) =>
        !Number.isInteger(index) ||
        index < 0 ||
        (position > 0 && index <= expected.legalActionIndices[position - 1]),
    )
  ) {
    throw new TypeError(
      "determinization legalActionIndices must be strictly increasing",
    );
  }
  if (!expected.legalActionIndices.includes(expected.baselineActionIndex)) {
    throw new RangeError("determinization baseline action must be legal");
  }
  if (
    expected.returnCount !== null &&
    expected.returnCount !== 1 &&
    expected.returnCount !== 2
  ) {
    throw new RangeError("determinization returnCount must be 1, 2, or null");
  }
  if (
    (expected.decision === "tax-return" && expected.returnCount === null) ||
    (expected.decision === "revolution" && expected.returnCount !== null)
  ) {
    throw new TypeError("determinization returnCount does not match decision");
  }
  if (!Array.isArray(expected.publicPlayers) || expected.publicPlayers.length < 4) {
    throw new TypeError("determinization publicPlayers is invalid");
  }
  if (!Array.isArray(expected.publicHistory)) {
    throw new TypeError("determinization publicHistory must be an array");
  }
  return {
    request,
    publicHistoryValidated: false,
    applied: false,
    targetValidated: false,
    audit: null,
  };
}

function prepareOverrideMap(
  record: Readonly<Record<string, number>> | undefined,
  label: string,
): Map<string, number> {
  if (record === undefined) return new Map();
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    throw new TypeError(`${label} overrides must be a record`);
  }
  const result = new Map<string, number>();
  for (const [key, actionIndex] of Object.entries(record)) {
    if (key.length === 0) {
      throw new TypeError(`${label} override keys must be non-empty`);
    }
    if (!Number.isInteger(actionIndex) || actionIndex < 0) {
      throw new RangeError(
        `${label} override action indices must be non-negative integers`,
      );
    }
    result.set(key, actionIndex);
  }
  return result;
}

function prepareNonCardRuntime(
  hooks: TrainingNonCardHooks,
  matchSeed: number,
): NonCardRuntime {
  if (!hooks || typeof hooks !== "object" || Array.isArray(hooks)) {
    throw new TypeError("nonCard must be an object");
  }
  if (
    hooks.taxReturnPolicy !== undefined &&
    typeof hooks.taxReturnPolicy !== "function"
  ) {
    throw new TypeError("taxReturnPolicy must be a function");
  }
  if (
    hooks.revolutionPolicy !== undefined &&
    typeof hooks.revolutionPolicy !== "function"
  ) {
    throw new TypeError("revolutionPolicy must be a function");
  }
  return {
    hooks,
    matchSeed,
    overrides: {
      taxReturn: prepareOverrideMap(
        hooks.forcedOverrides?.taxReturn,
        "tax-return",
      ),
      revolution: prepareOverrideMap(
        hooks.forcedOverrides?.revolution,
        "revolution",
      ),
      consumedTaxReturn: new Set(),
      consumedRevolution: new Set(),
    },
    determinization: prepareDeterminization(hooks.determinization),
  };
}

function assertAllNonCardOverridesConsumed(runtime: NonCardRuntime): void {
  const groups = [
    {
      label: "tax-return",
      overrides: runtime.overrides.taxReturn,
      consumed: runtime.overrides.consumedTaxReturn,
    },
    {
      label: "revolution",
      overrides: runtime.overrides.revolution,
      consumed: runtime.overrides.consumedRevolution,
    },
  ];
  for (const group of groups) {
    const unused = [...group.overrides.keys()].filter(
      (key) => !group.consumed.has(key),
    );
    if (unused.length > 0) {
      throw new RangeError(
        `unused ${group.label} forced override(s): ${unused.join(", ")}`,
      );
    }
  }
}

function forcedNonCardAction(
  runtime: NonCardRuntime,
  decision: NonCardDecisionKind,
  decisionKey: string,
): number | undefined {
  const overrides =
    decision === "tax-return"
      ? runtime.overrides.taxReturn
      : runtime.overrides.revolution;
  if (!overrides.has(decisionKey)) return undefined;
  const consumed =
    decision === "tax-return"
      ? runtime.overrides.consumedTaxReturn
      : runtime.overrides.consumedRevolution;
  if (consumed.has(decisionKey)) {
    throw new Error(
      `${decision} forced override was consumed more than once: ${decisionKey}`,
    );
  }
  consumed.add(decisionKey);
  return overrides.get(decisionKey);
}

function nonCardDecisionRandom(
  matchSeed: number,
  decision: NonCardDecisionKind,
  decisionKey: string,
): SeededRandom {
  let hash = (matchSeed >>> 0) ^ 0x811c9dc5;
  const material = `${decision}\u0000${decisionKey}`;
  for (let index = 0; index < material.length; index += 1) {
    hash ^= material.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return new SeededRandom(hash);
}

function validateNonCardPolicyDecision(
  selected: number | TrainingNonCardPolicyDecision,
  legalActionIndices: readonly number[],
  actorId: string,
  decision: NonCardDecisionKind,
  source: "policy" | "forced override" = "policy",
): TrainingNonCardPolicyDecision {
  const result =
    typeof selected === "number" ? { actionIndex: selected } : selected;
  if (!result || typeof result !== "object") {
    throw new TypeError(`${decision} ${source} must return an action decision`);
  }
  if (!legalActionIndices.includes(result.actionIndex)) {
    throw new RangeError(
      `${decision} ${source} selected illegal action ${String(result.actionIndex)} for ${actorId}`,
    );
  }
  if (
    result.logProbability !== undefined &&
    (!Number.isFinite(result.logProbability) || result.logProbability > 1e-9)
  ) {
    throw new RangeError(
      `${decision} policy logProbability must be finite and <= 0`,
    );
  }
  if (
    result.valueEstimate !== undefined &&
    !Number.isFinite(result.valueEstimate)
  ) {
    throw new RangeError(`${decision} policy valueEstimate must be finite`);
  }
  if (
    result.policyVersion !== undefined &&
    (typeof result.policyVersion !== "string" ||
      result.policyVersion.length < 1)
  ) {
    throw new TypeError(`${decision} policyVersion must be a non-empty string`);
  }
  return result;
}

function roleForIndex(index: number, total: number): BotRole {
  if (index === 0) return "great-dalmuti";
  if (index === 1) return "lesser-dalmuti";
  if (index === total - 2) return "lesser-peon";
  if (index === total - 1) return "great-peon";
  return "merchant";
}

function assignRoles(players: readonly SimulationPlayer[]): SimulationPlayer[] {
  return players.map((player, index) => ({
    ...player,
    role: roleForIndex(index, players.length),
  }));
}

function createDeck(): BotCard[] {
  const deck: BotCard[] = [];
  for (let rank = 1; rank <= 12; rank += 1) {
    for (let copy = 0; copy < rank; copy += 1) {
      deck.push({ id: `${rank}-${copy}`, rank });
    }
  }
  deck.push({ id: "joker-1", rank: 13 });
  deck.push({ id: "joker-2", rank: 13 });
  return deck;
}

function sortHand(cards: readonly BotCard[]): BotCard[] {
  return [...cards].sort(
    (left, right) =>
      right.rank - left.rank || left.id.localeCompare(right.id),
  );
}

function deal(
  players: readonly SimulationPlayer[],
  random: SeededRandom,
): Record<string, BotCard[]> {
  const deck = random.shuffle(createDeck());
  const counts = rankedDealCounts(deck.length, players.length);
  const hands: Record<string, BotCard[]> = {};
  let cursor = 0;
  players.forEach((player, index) => {
    hands[player.id] = sortHand(
      deck.slice(cursor, cursor + counts[index]),
    );
    cursor += counts[index];
  });
  return hands;
}

function sameNumberArray(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => Object.is(value, right[index]))
  );
}

function sameIntegerArray(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function publicPlayerSnapshots(
  players: readonly SimulationPlayer[],
  hands: Readonly<Record<string, readonly BotCard[]>>,
): TrainingNonCardPublicPlayerSnapshot[] {
  return players.map((player) => ({
    id: player.id,
    role: player.role,
    handCount: hands[player.id].length,
    score: player.score,
  }));
}

function samePublicPlayers(
  left: readonly TrainingNonCardPublicPlayerSnapshot[],
  right: readonly TrainingNonCardPublicPlayerSnapshot[],
): boolean {
  return (
    left.length === right.length &&
    left.every((player, index) => {
      const candidate = right[index];
      return (
        candidate !== undefined &&
        player.id === candidate.id &&
        player.role === candidate.role &&
        player.handCount === candidate.handCount &&
        Object.is(player.score, candidate.score)
      );
    })
  );
}

function assertCompletePhysicalDeck(
  players: readonly SimulationPlayer[],
  hands: Readonly<Record<string, readonly BotCard[]>>,
  label: string,
): void {
  const expected = new Map(createDeck().map((card) => [card.id, card.rank]));
  const seen = new Set<string>();
  let total = 0;
  for (const player of players) {
    const hand = hands[player.id];
    if (!Array.isArray(hand)) {
      throw new Error(`${label}: missing hand for ${player.id}`);
    }
    for (const card of hand) {
      total += 1;
      if (seen.has(card.id)) {
        throw new Error(`${label}: duplicate physical card ownership`);
      }
      seen.add(card.id);
      if (expected.get(card.id) !== card.rank) {
        throw new Error(`${label}: unknown or rank-drifted physical card`);
      }
    }
  }
  if (total !== 80 || seen.size !== 80 || seen.size !== expected.size) {
    throw new Error(`${label}: physical deck is incomplete`);
  }
  for (const id of expected.keys()) {
    if (!seen.has(id)) throw new Error(`${label}: physical card is missing`);
  }
}

function applyTrainingDeterminization(
  players: readonly SimulationPlayer[],
  hands: Record<string, BotCard[]>,
  round: number,
  runtime: NonCardRuntime,
): void {
  const state = runtime.determinization;
  if (!state || state.request.expected.round !== round) return;
  if (state.applied) {
    throw new Error("non-card determinization was applied more than once");
  }
  const { expected } = state.request;
  const actor = players.find((player) => player.id === expected.actorId);
  if (!actor) {
    rejectDeterminization(
      "target-actor-missing",
      "determinization target actor is absent in the target act",
    );
  }
  assertCompletePhysicalDeck(players, hands, "before determinization");
  const beforePublicPlayers = publicPlayerSnapshots(players, hands);
  if (!samePublicPlayers(beforePublicPlayers, expected.publicPlayers)) {
    rejectDeterminization(
      "pre-resample-public-state-drift",
      "public player state drifted before determinization",
    );
  }
  const actorHandBefore = hands[actor.id].map((card) => ({ ...card }));
  const opponentPlayers = players.filter((player) => player.id !== actor.id);
  const opponentCounts = opponentPlayers.map(
    (player) => hands[player.id].length,
  );
  const resampleOpponents = state.request.resampleOpponents !== false;
  const originalOwnerByCardId = new Map<string, string>();
  const opponentCards: BotCard[] = [];
  for (const player of opponentPlayers) {
    for (const card of hands[player.id]) {
      originalOwnerByCardId.set(card.id, player.id);
      opponentCards.push(card);
    }
  }
  const shuffled = resampleOpponents
    ? new SeededRandom(state.request.hiddenWorldSeed).shuffle(opponentCards)
    : [...opponentCards];
  let cursor = 0;
  let changedOpponentOwnershipCards = 0;
  opponentPlayers.forEach((player, index) => {
    const nextHand = shuffled.slice(cursor, cursor + opponentCounts[index]);
    cursor += opponentCounts[index];
    for (const card of nextHand) {
      if (originalOwnerByCardId.get(card.id) !== player.id) {
        changedOpponentOwnershipCards += 1;
      }
    }
    hands[player.id] = sortHand(nextHand);
  });
  if (cursor !== opponentCards.length) {
    throw new Error("determinization did not assign every opponent card");
  }
  if (resampleOpponents && changedOpponentOwnershipCards === 0) {
    rejectDeterminization(
      "unchanged-hidden-world",
      "determinization produced the original opponent ownership",
    );
  }
  if (
    actorHandBefore.length !== hands[actor.id].length ||
    actorHandBefore.some(
      (card, index) =>
        card.id !== hands[actor.id][index].id ||
        card.rank !== hands[actor.id][index].rank,
    )
  ) {
    throw new Error("determinization changed the target actor hand");
  }
  const afterPublicPlayers = publicPlayerSnapshots(players, hands);
  if (!samePublicPlayers(afterPublicPlayers, expected.publicPlayers)) {
    throw new Error("determinization changed public player state or hand counts");
  }
  assertCompletePhysicalDeck(players, hands, "after determinization");
  state.applied = true;
  state.audit = {
    algorithmVersion: 1,
    hiddenWorldSeed: state.request.hiddenWorldSeed,
    targetDecision: expected.decision,
    targetRound: expected.round,
    targetActorId: expected.actorId,
    physicalCardCount: 80,
    uniquePhysicalCardCount: 80,
    actorHandPreserved: true,
    publicHandCountsPreserved: true,
    publicHistoryPreserved: true,
    environmentRandomDrawsConsumed: 0,
    hiddenWorldKind: resampleOpponents ? "resampled" : "original-replay",
    opponentCardsResampled: resampleOpponents ? opponentCards.length : 0,
    changedOpponentOwnershipCards,
    continuationRngReplaced:
      state.request.continuationSeed !== undefined &&
      state.request.continuationSeed !== null,
    targetEncodedObservationPreserved: true,
    targetLegalActionsPreserved: true,
    targetBaselineActionPreserved: true,
    targetReturnCountPreserved: true,
    targetPublicPlayersPreserved: true,
    taxTransfer: null,
  };
}

function cardsByIds(
  hand: readonly BotCard[],
  cardIds: readonly string[],
): BotCard[] {
  const byId = new Map(hand.map((card) => [card.id, card]));
  return cardIds.map((id) => {
    const card = byId.get(id);
    if (!card) throw new Error(`card ${id} is not in the player's hand`);
    return card;
  });
}

function removeCardIds(
  hand: readonly BotCard[],
  cardIds: readonly string[],
): BotCard[] {
  const removed = new Set(cardIds);
  return hand.filter((card) => !removed.has(card.id));
}

function transferCards(
  hands: Record<string, BotCard[]>,
  fromId: string,
  toId: string,
  cardIds: readonly string[],
): void {
  const cards = cardsByIds(hands[fromId], cardIds);
  hands[fromId] = sortHand(removeCardIds(hands[fromId], cardIds));
  hands[toId] = sortHand([...hands[toId], ...cards]);
}

function applyTaxation(
  players: readonly SimulationPlayer[],
  hands: Record<string, BotCard[]>,
): void {
  const pairs = [
    {
      nobleRole: "great-dalmuti" as const,
      peonRole: "great-peon" as const,
      count: 2,
    },
    {
      nobleRole: "lesser-dalmuti" as const,
      peonRole: "lesser-peon" as const,
      count: 1,
    },
  ];

  const exchanges = pairs.map((pair) => {
    const noble = players.find((player) => player.role === pair.nobleRole);
    const peon = players.find((player) => player.role === pair.peonRole);
    if (!noble || !peon) {
      throw new Error("tax roles are missing");
    }
    const peonCardIds = selectForcedBotTribute(
      hands[peon.id],
      pair.count,
    );
    const nobleCardIds = chooseBotTaxReturn(
      hands[noble.id],
      pair.count,
      noble.difficulty,
    ).cardIds;
    if (
      peonCardIds.length !== pair.count ||
      nobleCardIds.length !== pair.count
    ) {
      throw new Error("tax exchange selected the wrong number of cards");
    }
    return {
      nobleId: noble.id,
      peonId: peon.id,
      peonCardIds,
      nobleCardIds,
    };
  });

  // Production locks both sides' choices before either direction moves.
  for (const exchange of exchanges) {
    transferCards(
      hands,
      exchange.peonId,
      exchange.nobleId,
      exchange.peonCardIds,
    );
  }
  for (const exchange of exchanges) {
    transferCards(
      hands,
      exchange.nobleId,
      exchange.peonId,
      exchange.nobleCardIds,
    );
  }
}

type PendingTrainingTaxReturnStep = Omit<
  TrainingTaxReturnStep,
  "reward" | "finishPlace"
>;
type PendingTrainingRevolutionStep = Omit<
  TrainingRevolutionStep,
  "reward" | "finishPlace"
>;
type PendingTrainingNonCardStep =
  | PendingTrainingTaxReturnStep
  | PendingTrainingRevolutionStep;

function createNonCardPublicPlayers(
  players: readonly SimulationPlayer[],
  hands: Readonly<Record<string, readonly BotCard[]>>,
) {
  return publicPlayerSnapshots(players, hands);
}

function assertDeterminizedTargetState({
  runtime,
  decision,
  decisionKey,
  round,
  decisionStep,
  actorId,
  encodedObservation,
  legalActionIndices,
  baselineActionIndex,
  returnCount,
  publicPlayers,
}: {
  runtime: NonCardRuntime;
  decision: NonCardDecisionKind;
  decisionKey: string;
  round: number;
  decisionStep: number;
  actorId: string;
  encodedObservation: readonly number[];
  legalActionIndices: readonly number[];
  baselineActionIndex: number;
  returnCount: 1 | 2 | null;
  publicPlayers: readonly TrainingNonCardPublicPlayerSnapshot[];
}): void {
  const state = runtime.determinization;
  if (!state) return;
  const { expected } = state.request;
  if (
    decision !== expected.decision ||
    decisionKey !== expected.decisionKey ||
    round !== expected.round ||
    actorId !== expected.actorId
  ) {
    return;
  }
  if (!state.applied || !state.publicHistoryValidated || !state.audit) {
    throw new Error("determinization target was reached before world validation");
  }
  if (state.targetValidated) {
    throw new Error("determinization target was validated more than once");
  }
  if (decisionStep !== expected.decisionStep) {
    rejectDeterminization(
      "decision-order-drift",
      "non-card decision order changed before the target",
    );
  }
  if (!sameNumberArray(encodedObservation, expected.encodedObservation)) {
    rejectDeterminization(
      "encoded-observation-drift",
      "target encoded observation changed after hidden-card resampling",
    );
  }
  if (!sameIntegerArray(legalActionIndices, expected.legalActionIndices)) {
    rejectDeterminization(
      "legal-actions-drift",
      "target legal action mask changed after hidden-card resampling",
    );
  }
  if (baselineActionIndex !== expected.baselineActionIndex) {
    rejectDeterminization(
      "baseline-action-drift",
      "normal baseline action changed after hidden-card resampling",
    );
  }
  if (returnCount !== expected.returnCount) {
    rejectDeterminization(
      "return-count-drift",
      "target tax return count changed after hidden-card resampling",
    );
  }
  if (!samePublicPlayers(publicPlayers, expected.publicPlayers)) {
    rejectDeterminization(
      "public-player-state-drift",
      "target public player state changed after hidden-card resampling",
    );
  }
  state.targetValidated = true;
}

function baselineTaxReturnActionIndex(
  player: SimulationPlayer,
  hand: readonly BotCard[],
  returnCount: 1 | 2,
): number {
  const selectedCardIds = chooseBotTaxReturn(
    hand,
    returnCount,
    player.difficulty,
  ).cardIds;
  const selectedRanks = cardsByIds(hand, selectedCardIds).map(
    (card) => card.rank,
  );
  return encodeTaxReturnAction(selectedRanks);
}

function chooseTrainingTaxReturn(
  episodeId: string,
  round: number,
  players: readonly SimulationPlayer[],
  hands: Readonly<Record<string, readonly BotCard[]>>,
  noble: SimulationPlayer,
  returnCount: 1 | 2,
  runtime: NonCardRuntime,
  pendingSteps: PendingTrainingNonCardStep[],
): string[] {
  const actorSeat = players.findIndex((player) => player.id === noble.id);
  if (actorSeat < 0) throw new Error(`unknown tax-return actor ${noble.id}`);
  const publicPlayers = createNonCardPublicPlayers(players, hands);
  const observation: TaxReturnObservation = {
    actorId: noble.id,
    hand: hands[noble.id],
    players: publicPlayers,
    round,
    returnCount,
  };
  const encodedObservation = encodeTaxReturnObservation(observation);
  const legalActionIndices = legalTaxReturnActionIndices(observation);
  const decisionKey = createTrainingNonCardDecisionKey({
    episodeId,
    round,
    actorId: noble.id,
  });
  const forcedActionIndex = forcedNonCardAction(
    runtime,
    "tax-return",
    decisionKey,
  );
  const normalBaselineActionIndex = baselineTaxReturnActionIndex(
    noble,
    hands[noble.id],
    returnCount,
  );
  assertDeterminizedTargetState({
    runtime,
    decision: "tax-return",
    decisionKey,
    round,
    decisionStep: pendingSteps.length,
    actorId: noble.id,
    encodedObservation,
    legalActionIndices,
    baselineActionIndex: normalBaselineActionIndex,
    returnCount,
    publicPlayers,
  });
  let decision: TrainingNonCardPolicyDecision;
  let behaviorPolicy: TrainingNonCardStep["behaviorPolicy"];
  if (forcedActionIndex !== undefined) {
    decision = validateNonCardPolicyDecision(
      forcedActionIndex,
      legalActionIndices,
      noble.id,
      "tax-return",
      "forced override",
    );
    behaviorPolicy = "forced-override";
  } else if (runtime.hooks.taxReturnPolicy) {
    const decisionRandom = nonCardDecisionRandom(
      runtime.matchSeed,
      "tax-return",
      decisionKey,
    );
    decision = validateNonCardPolicyDecision(
      runtime.hooks.taxReturnPolicy({
        decision: "tax-return",
        episodeId,
        round,
        actorId: noble.id,
        actorSeat,
        actorRole: noble.role,
        actorScore: noble.score,
        decisionKey,
        observation,
        encodedObservation,
        legalActionIndices,
        random: () => decisionRandom.next(),
      }),
      legalActionIndices,
      noble.id,
      "tax-return",
    );
    behaviorPolicy = "custom";
  } else {
    decision = {
      actionIndex: normalBaselineActionIndex,
    };
    behaviorPolicy = noble.difficulty;
  }

  pendingSteps.push({
    decision: "tax-return",
    episodeId,
    round,
    step: pendingSteps.length,
    actorId: noble.id,
    actorSeat,
    actorRole: noble.role,
    decisionKey,
    observationSchemaVersion: NON_CARD_OBSERVATION_SCHEMA_VERSION,
    actionCatalogueVersion: TAX_RETURN_ACTION_CATALOGUE_VERSION,
    behaviorPolicy,
    observation: [...encodedObservation],
    legalActionIndices: [...legalActionIndices],
    actionIndex: decision.actionIndex,
    behaviorLogProbability: decision.logProbability ?? null,
    behaviorValueEstimate: decision.valueEstimate ?? null,
    behaviorPolicyVersion: decision.policyVersion ?? null,
    forced: legalActionIndices.length === 1,
    forcedOverride: forcedActionIndex !== undefined,
    publicPlayers,
    metadata: {
      playerCount: players.length,
      actorHandCount: hands[noble.id].length,
      returnCount,
    },
  });
  return resolveTaxReturnAction(observation, decision.actionIndex);
}

function applyTrainingTaxation(
  episodeId: string,
  round: number,
  players: readonly SimulationPlayer[],
  hands: Record<string, BotCard[]>,
  runtime: NonCardRuntime,
  pendingSteps: PendingTrainingNonCardStep[],
): void {
  assertCompletePhysicalDeck(players, hands, "before training taxation");
  const handCountsBefore = new Map(
    players.map((player) => [player.id, hands[player.id].length]),
  );
  const pairs = [
    {
      nobleRole: "great-dalmuti" as const,
      peonRole: "great-peon" as const,
      count: 2 as const,
    },
    {
      nobleRole: "lesser-dalmuti" as const,
      peonRole: "lesser-peon" as const,
      count: 1 as const,
    },
  ];
  const exchanges = pairs.map((pair) => {
    const noble = players.find((player) => player.role === pair.nobleRole);
    const peon = players.find((player) => player.role === pair.peonRole);
    if (!noble || !peon) throw new Error("tax roles are missing");
    const peonCardIds = selectForcedBotTribute(hands[peon.id], pair.count);
    const nobleCardIds = chooseTrainingTaxReturn(
      episodeId,
      round,
      players,
      hands,
      noble,
      pair.count,
      runtime,
      pendingSteps,
    );
    if (
      peonCardIds.length !== pair.count ||
      nobleCardIds.length !== pair.count
    ) {
      throw new Error("tax exchange selected the wrong number of cards");
    }
    if (
      new Set(peonCardIds).size !== pair.count ||
      new Set(nobleCardIds).size !== pair.count
    ) {
      throw new Error("tax exchange selected duplicate physical cards");
    }
    cardsByIds(hands[peon.id], peonCardIds);
    cardsByIds(hands[noble.id], nobleCardIds);
    return {
      nobleId: noble.id,
      peonId: peon.id,
      peonCardIds,
      nobleCardIds,
    };
  });

  // As in production, lock both return actions before moving either tribute.
  for (const exchange of exchanges) {
    transferCards(
      hands,
      exchange.peonId,
      exchange.nobleId,
      exchange.peonCardIds,
    );
  }
  for (const exchange of exchanges) {
    if (
      hands[exchange.peonId].length !==
        (handCountsBefore.get(exchange.peonId) ?? -1) -
          exchange.peonCardIds.length ||
      hands[exchange.nobleId].length !==
        (handCountsBefore.get(exchange.nobleId) ?? -1) +
          exchange.peonCardIds.length
    ) {
      throw new Error("tax tribute transfer produced inconsistent hand counts");
    }
  }
  assertCompletePhysicalDeck(players, hands, "after training tribute transfer");
  for (const exchange of exchanges) {
    transferCards(
      hands,
      exchange.nobleId,
      exchange.peonId,
      exchange.nobleCardIds,
    );
  }
  for (const player of players) {
    if (hands[player.id].length !== handCountsBefore.get(player.id)) {
      throw new Error("tax return transfer did not restore public hand counts");
    }
  }
  assertCompletePhysicalDeck(players, hands, "after training tax exchange");
  const determinization = runtime.determinization;
  if (
    determinization?.audit &&
    determinization.request.expected.round === round
  ) {
    const tributeCardCount = exchanges.reduce<number>(
      (total, exchange) => total + exchange.peonCardIds.length,
      0,
    );
    const returnCardCount = exchanges.reduce<number>(
      (total, exchange) => total + exchange.nobleCardIds.length,
      0,
    );
    if (tributeCardCount !== 3 || returnCardCount !== 3) {
      throw new Error("tax audit expected exactly three cards each way");
    }
    determinization.audit.taxTransfer = {
      exchangeCount: 2,
      tributeCardCount,
      returnCardCount,
      ownershipCompleteAfterTransfer: true,
      publicHandCountsRestoredAfterTransfer: true,
    };
  }
}

function chooseTrainingRevolution(
  episodeId: string,
  round: number,
  players: readonly SimulationPlayer[],
  hands: Readonly<Record<string, readonly BotCard[]>>,
  holder: SimulationPlayer,
  runtime: NonCardRuntime,
  pendingSteps: PendingTrainingNonCardStep[],
): RevolutionState {
  const actorSeat = players.findIndex((player) => player.id === holder.id);
  if (actorSeat < 0) throw new Error(`unknown revolution actor ${holder.id}`);
  const publicPlayers = createNonCardPublicPlayers(players, hands);
  const observation: RevolutionObservation = {
    actorId: holder.id,
    hand: hands[holder.id],
    players: publicPlayers,
    round,
  };
  const encodedObservation = encodeRevolutionObservation(observation);
  const legalActionIndices = legalRevolutionActionIndices(observation);
  const decisionKey = createTrainingNonCardDecisionKey({
    episodeId,
    round,
    actorId: holder.id,
  });
  const forcedActionIndex = forcedNonCardAction(
    runtime,
    "revolution",
    decisionKey,
  );
  const normalBaseline = chooseBotRevolution(
    {
      hand: hands[holder.id],
      role: holder.role,
      playerCount: players.length,
    },
    holder.difficulty,
  );
  const normalBaselineActionIndex = normalBaseline.declare
    ? REVOLUTION_DECLARE_ACTION_INDEX
    : REVOLUTION_DECLINE_ACTION_INDEX;
  assertDeterminizedTargetState({
    runtime,
    decision: "revolution",
    decisionKey,
    round,
    decisionStep: pendingSteps.length,
    actorId: holder.id,
    encodedObservation,
    legalActionIndices,
    baselineActionIndex: normalBaselineActionIndex,
    returnCount: null,
    publicPlayers,
  });
  let decision: TrainingNonCardPolicyDecision;
  let behaviorPolicy: TrainingNonCardStep["behaviorPolicy"];
  if (forcedActionIndex !== undefined) {
    decision = validateNonCardPolicyDecision(
      forcedActionIndex,
      legalActionIndices,
      holder.id,
      "revolution",
      "forced override",
    );
    behaviorPolicy = "forced-override";
  } else if (runtime.hooks.revolutionPolicy) {
    const decisionRandom = nonCardDecisionRandom(
      runtime.matchSeed,
      "revolution",
      decisionKey,
    );
    decision = validateNonCardPolicyDecision(
      runtime.hooks.revolutionPolicy({
        decision: "revolution",
        episodeId,
        round,
        actorId: holder.id,
        actorSeat,
        actorRole: holder.role,
        actorScore: holder.score,
        decisionKey,
        observation,
        encodedObservation,
        legalActionIndices,
        random: () => decisionRandom.next(),
      }),
      legalActionIndices,
      holder.id,
      "revolution",
    );
    behaviorPolicy = "custom";
  } else {
    decision = {
      actionIndex: normalBaselineActionIndex,
    };
    behaviorPolicy = holder.difficulty;
  }
  const declarationKind: RevolutionState =
    decision.actionIndex === REVOLUTION_DECLINE_ACTION_INDEX
      ? null
      : holder.role === "great-peon"
        ? "great-revolution"
        : "revolution";
  pendingSteps.push({
    decision: "revolution",
    episodeId,
    round,
    step: pendingSteps.length,
    actorId: holder.id,
    actorSeat,
    actorRole: holder.role,
    decisionKey,
    observationSchemaVersion: NON_CARD_OBSERVATION_SCHEMA_VERSION,
    actionCatalogueVersion: REVOLUTION_ACTION_CATALOGUE_VERSION,
    behaviorPolicy,
    observation: [...encodedObservation],
    legalActionIndices: [...legalActionIndices],
    actionIndex: decision.actionIndex,
    behaviorLogProbability: decision.logProbability ?? null,
    behaviorValueEstimate: decision.valueEstimate ?? null,
    behaviorPolicyVersion: decision.policyVersion ?? null,
    forced: legalActionIndices.length === 1,
    forcedOverride: forcedActionIndex !== undefined,
    publicPlayers,
    metadata: {
      playerCount: players.length,
      actorHandCount: hands[holder.id].length,
      declarationKind,
    },
  });
  return declarationKind;
}

function nextActiveIndex(
  players: readonly SimulationPlayer[],
  hands: Readonly<Record<string, readonly BotCard[]>>,
  fromIndex: number,
): number {
  for (let step = 1; step <= players.length; step += 1) {
    const index = (fromIndex + step + players.length) % players.length;
    if (hands[players[index].id].length > 0) return index;
  }
  return fromIndex;
}

function publicPlayedCards(counts: readonly number[]) {
  return counts
    .map((count, rank) => ({ rank, count }))
    .filter((entry) => entry.rank >= 1 && entry.count > 0);
}

function createPlayObservation(
  actorId: string,
  players: readonly SimulationPlayer[],
  hands: Readonly<Record<string, readonly BotCard[]>>,
  table: SimulationTable | null,
  passedPlayerIds: readonly string[],
  finishOrder: readonly string[],
  playedCounts: readonly number[],
): BotPlayObservation {
  return {
    actorId,
    hand: hands[actorId],
    table,
    players: players.map((player) => ({
      id: player.id,
      handCount: hands[player.id].length,
      finished: finishOrder.includes(player.id),
    })),
    passedPlayerIds,
    publicPlayedCards: publicPlayedCards(playedCounts),
  };
}

function choosePolicyDecision(
  player: SimulationPlayer,
  observation: BotPlayObservation,
  encodedObservation: readonly number[],
  legalActionIndices: readonly number[],
  round: number,
  random: SeededRandom,
  policy?: TrainingPolicy,
): Required<Pick<TrainingPolicyDecision, "actionIndex">> &
  Omit<TrainingPolicyDecision, "actionIndex"> {
  const selected = policy
    ? policy({
        observation,
        encodedObservation,
        legalActionIndices,
        actorRole: player.role,
        actorScore: player.score,
        round,
        random: () => random.next(),
      })
    : semanticActionIndexFromBotAction(
        chooseBotPlay(observation, player.difficulty).action,
      );
  const decision =
    typeof selected === "number"
      ? { actionIndex: selected }
      : selected;
  const { actionIndex } = decision;
  if (!legalActionIndices.includes(actionIndex)) {
    throw new RangeError(
      `policy selected illegal action ${actionIndex} for ${player.id}`,
    );
  }
  if (
    decision.logProbability !== undefined &&
    (!Number.isFinite(decision.logProbability) ||
      decision.logProbability > 1e-9)
  ) {
    throw new RangeError("policy logProbability must be finite and <= 0");
  }
  if (
    decision.valueEstimate !== undefined &&
    !Number.isFinite(decision.valueEstimate)
  ) {
    throw new RangeError("policy valueEstimate must be finite");
  }
  if (
    decision.policyVersion !== undefined &&
    (typeof decision.policyVersion !== "string" ||
      decision.policyVersion.length < 1)
  ) {
    throw new TypeError("policyVersion must be a non-empty string");
  }
  return decision;
}

function applyPlay(
  action: BotAction,
  actorId: string,
  hands: Record<string, BotCard[]>,
  playedCounts: number[],
): void {
  if (action.type !== "play") {
    throw new TypeError("applyPlay requires a play action");
  }
  const cards = cardsByIds(hands[actorId], action.cardIds);
  hands[actorId] = sortHand(
    removeCardIds(hands[actorId], action.cardIds),
  );
  for (const card of cards) {
    playedCounts[card.rank] += 1;
  }
}

function simulateAct(
  episodeId: string,
  round: number,
  initialPlayers: readonly SimulationPlayer[],
  random: SeededRandom,
  policy?: TrainingPolicy,
  policyByPlayerId?: Readonly<Record<string, TrainingPolicy | undefined>>,
  supervisionPolicy?: TrainingPolicy,
  nonCardRuntime?: NonCardRuntime,
): {
  act: SimulatedAct;
  players: SimulationPlayer[];
  steps: TrainingStep[];
  nonCardSteps?: TrainingNonCardStep[];
  nextRandom: SeededRandom;
} {
  let players = assignRoles(initialPlayers);
  const hands = deal(players, random);
  if (nonCardRuntime) {
    applyTrainingDeterminization(players, hands, round, nonCardRuntime);
  }
  let revolution: RevolutionState = null;
  const pendingNonCardSteps = nonCardRuntime
    ? ([] as PendingTrainingNonCardStep[])
    : undefined;

  const revolutionHolder = players.find(
    (player) =>
      hands[player.id].filter((card) => card.rank === 13).length === 2,
  );
  if (revolutionHolder) {
    if (nonCardRuntime && pendingNonCardSteps) {
      revolution = chooseTrainingRevolution(
        episodeId,
        round,
        players,
        hands,
        revolutionHolder,
        nonCardRuntime,
        pendingNonCardSteps,
      );
    } else {
      const decision = chooseBotRevolution(
        {
          hand: hands[revolutionHolder.id],
          role: revolutionHolder.role,
          playerCount: players.length,
        },
        revolutionHolder.difficulty,
      );
      if (decision.declare) revolution = decision.kind;
    }
    if (revolution === "great-revolution") {
      players = assignRoles([...players].reverse());
    }
  }

  if (round > 1 && revolution === null) {
    if (nonCardRuntime && pendingNonCardSteps) {
      applyTrainingTaxation(
        episodeId,
        round,
        players,
        hands,
        nonCardRuntime,
        pendingNonCardSteps,
      );
    } else {
      applyTaxation(players, hands);
    }
  }

  const continuationSeed =
    nonCardRuntime?.determinization?.request.expected.round === round
      ? nonCardRuntime.determinization.request.continuationSeed
      : null;
  const continuationRandom =
    continuationSeed === undefined || continuationSeed === null
      ? random
      : new SeededRandom(continuationSeed);

  const playerOrder = players.map((player) => player.id);
  const rolesByPlayerId = Object.fromEntries(
    players.map((player) => [player.id, player.role]),
  ) as Record<string, BotRole>;
  const scoresByPlayerId = Object.fromEntries(
    players.map((player) => [player.id, player.score]),
  );
  const finishOrder: string[] = [];
  const passedPlayerIds: string[] = [];
  const playedCounts = Array.from({ length: 14 }, () => 0);
  const rawSteps: Omit<
    TrainingStep,
    "reward" | "actorTerminal" | "environmentTerminal" | "finishPlace"
  >[] = [];
  let table: SimulationTable | null = null;
  let lastPlayedId: string | null = null;
  let currentIndex = 0;
  let transitions = 0;

  while (finishOrder.length < players.length) {
    transitions += 1;
    if (transitions > MAX_TRANSITIONS_PER_ACT) {
      throw new Error(
        `act exceeded ${MAX_TRANSITIONS_PER_ACT} transitions`,
      );
    }

    const actor = players[currentIndex];
    if (hands[actor.id].length === 0) {
      currentIndex = nextActiveIndex(players, hands, currentIndex);
      continue;
    }
    const observation = createPlayObservation(
      actor.id,
      players,
      hands,
      table,
      passedPlayerIds,
      finishOrder,
      playedCounts,
    );
    const legalActionIndices = legalSemanticActionIndices(observation);
    if (legalActionIndices.length === 0) {
      throw new Error("an active player has no legal action");
    }
    const encodedObservation = encodeTrainingObservation({
      observation,
      round,
      rolesByPlayerId,
      scoresByPlayerId,
      revolution,
    });
    const behaviorDecision = choosePolicyDecision(
      actor,
      observation,
      encodedObservation,
      legalActionIndices,
      round,
      continuationRandom,
      policyByPlayerId?.[actor.id] ?? policy,
    );
    const actionIndex = behaviorDecision.actionIndex;
    const supervisedActionIndex = supervisionPolicy
      ? choosePolicyDecision(
          actor,
          observation,
          encodedObservation,
          legalActionIndices,
          round,
          continuationRandom,
          supervisionPolicy,
        ).actionIndex
      : null;
    rawSteps.push({
      episodeId,
      round,
      step: rawSteps.length,
      actorId: actor.id,
      actorSeat: currentIndex,
      actorRole: actor.role,
      behaviorPolicy:
        policyByPlayerId?.[actor.id] || policy
          ? "custom"
          : actor.difficulty,
      observation: encodedObservation,
      legalActionIndices: [...legalActionIndices],
      actionIndex,
      supervisedActionIndex,
      behaviorLogProbability:
        behaviorDecision.logProbability ?? null,
      behaviorValueEstimate:
        behaviorDecision.valueEstimate ?? null,
      behaviorPolicyVersion:
        behaviorDecision.policyVersion ?? null,
      forced: legalActionIndices.length === 1,
    });

    const action = resolveSemanticAction(observation, actionIndex);
    if (action.type === "pass") {
      if (!table) throw new Error("a leading player cannot pass");
      if (!passedPlayerIds.includes(actor.id)) {
        passedPlayerIds.push(actor.id);
      }
      const active = players.filter(
        (player) => hands[player.id].length > 0,
      );
      const requiredToPass = active.filter(
        (player) => player.id !== lastPlayedId,
      );
      const trickIsOver = requiredToPass.every((player) =>
        passedPlayerIds.includes(player.id),
      );
      if (trickIsOver) {
        const previousLeaderIndex = players.findIndex(
          (player) => player.id === lastPlayedId,
        );
        const leaderStillActive =
          previousLeaderIndex >= 0 &&
          hands[players[previousLeaderIndex].id].length > 0;
        table = null;
        passedPlayerIds.length = 0;
        currentIndex = leaderStillActive
          ? previousLeaderIndex
          : nextActiveIndex(players, hands, previousLeaderIndex);
      } else {
        currentIndex = nextActiveIndex(players, hands, currentIndex);
      }
      continue;
    }

    applyPlay(action, actor.id, hands, playedCounts);
    table = {
      rank: action.rank,
      count: action.count,
      playerId: actor.id,
    };
    lastPlayedId = actor.id;
    passedPlayerIds.length = 0;

    if (hands[actor.id].length === 0) {
      finishOrder.push(actor.id);
    }
    if (finishOrder.length === players.length - 1) {
      const last = players.find(
        (player) => !finishOrder.includes(player.id),
      );
      if (!last) throw new Error("could not find the last-place player");
      finishOrder.push(last.id);
      break;
    }

    if (action.rank === 1) {
      const actorStillActive = hands[actor.id].length > 0;
      table = null;
      passedPlayerIds.length = 0;
      currentIndex = actorStillActive
        ? currentIndex
        : nextActiveIndex(players, hands, currentIndex);
    } else {
      currentIndex = nextActiveIndex(players, hands, currentIndex);
    }
  }

  const chipAwards = Object.fromEntries(
    finishOrder.map((playerId, index) => [
      playerId,
      roundChipAward(index + 1, players.length),
    ]),
  );
  const places = new Map(
    finishOrder.map((playerId, index) => [playerId, index + 1]),
  );
  players = players.map((player) => ({
    ...player,
    score: player.score + chipAwards[player.id],
  }));

  const lastStepByActor = new Map<string, number>();
  rawSteps.forEach((step, index) => lastStepByActor.set(step.actorId, index));
  const steps: TrainingStep[] = rawSteps.map((step, index) => {
    const award = chipAwards[step.actorId];
    const actorTerminal = lastStepByActor.get(step.actorId) === index;
    return {
      ...step,
      reward: actorTerminal ? (award - 2) / 2 : 0,
      actorTerminal,
      environmentTerminal: index === rawSteps.length - 1,
      finishPlace: places.get(step.actorId) ?? players.length,
    };
  });
  const nonCardSteps = pendingNonCardSteps?.map((step) => ({
    ...step,
    reward: (chipAwards[step.actorId] - 2) / 2,
    finishPlace: places.get(step.actorId) ?? players.length,
  })) as TrainingNonCardStep[] | undefined;

  const byId = new Map(players.map((player) => [player.id, player]));
  const nextPlayers = finishOrder.map((playerId) => {
    const player = byId.get(playerId);
    if (!player) throw new Error(`unknown finisher ${playerId}`);
    return player;
  });

  return {
    act: {
      round,
      revolution,
      playerOrder,
      finishOrder,
      chipAwards,
      transitions,
    },
    players: nextPlayers,
    steps,
    ...(nonCardSteps ? { nonCardSteps } : {}),
    nextRandom: continuationRandom,
  };
}

function normalizedDifficulties(
  playerCount: number,
  difficulties: readonly BotDifficulty[] | undefined,
): BotDifficulty[] {
  if (!difficulties || difficulties.length === 0) {
    return Array.from({ length: playerCount }, () => "hard");
  }
  if (difficulties.length === 1) {
    return Array.from({ length: playerCount }, () => difficulties[0]);
  }
  if (difficulties.length !== playerCount) {
    throw new RangeError(
      "difficulties must contain one value or one value per player",
    );
  }
  return [...difficulties];
}

function validateDeterminizationPublicHistory(
  runtime: NonCardRuntime,
  round: number,
  simulatedActs: readonly SimulatedAct[],
): void {
  const state = runtime.determinization;
  if (!state || state.request.expected.round !== round) return;
  if (state.publicHistoryValidated) {
    throw new Error("determinization public history was validated twice");
  }
  const expectedHistory = state.request.expected.publicHistory;
  if (
    expectedHistory.length !== round - 1 ||
    JSON.stringify(simulatedActs) !== JSON.stringify(expectedHistory)
  ) {
    rejectDeterminization(
      "public-history-drift",
      "public history before the target act does not match the baseline",
    );
  }
  state.publicHistoryValidated = true;
}

function assertDeterminizationCompleted(runtime: NonCardRuntime): void {
  const state = runtime.determinization;
  if (!state) return;
  if (!state.publicHistoryValidated) {
    rejectDeterminization(
      "target-round-not-reached",
      "determinization target round was not reached",
    );
  }
  if (!state.applied || !state.audit) {
    rejectDeterminization(
      "hidden-world-not-applied",
      "opponent hidden cards were not resampled in the target act",
    );
  }
  if (!state.targetValidated) {
    rejectDeterminization(
      "target-decision-not-reached",
      "resampled world did not reach the exact target decision",
    );
  }
  if (
    state.request.expected.decision === "tax-return" &&
    state.audit.taxTransfer === null
  ) {
    rejectDeterminization(
      "tax-transfer-not-completed",
      "target tax decision did not complete the audited exchange",
    );
  }
}

export function simulateMatch(config: SimulationConfig): SimulatedMatch {
  if (
    !Number.isInteger(config.playerCount) ||
    config.playerCount < 4 ||
    config.playerCount > 10
  ) {
    throw new RangeError("playerCount must be an integer from 4 to 10");
  }
  const acts = config.acts ?? 1;
  if (!Number.isInteger(acts) || acts < 1) {
    throw new RangeError("acts must be a positive integer");
  }
  const seed = config.seed ?? 1;
  const episodeId = config.episodeId ?? `seed-${seed}`;
  let random = new SeededRandom(seed);
  const nonCardRuntime =
    config.nonCard === undefined
      ? undefined
      : prepareNonCardRuntime(config.nonCard, seed);
  const difficulties = normalizedDifficulties(
    config.playerCount,
    config.difficulties,
  );
  let players = random.shuffle(
    difficulties.map<SimulationPlayer>((difficulty, index) => ({
      id: `player-${index + 1}`,
      difficulty,
      role: "merchant",
      score: 0,
    })),
  );
  const simulatedActs: SimulatedAct[] = [];
  const steps: TrainingStep[] = [];
  const nonCardSteps = nonCardRuntime
    ? ([] as TrainingNonCardStep[])
    : undefined;

  for (let round = 1; round <= acts; round += 1) {
    if (nonCardRuntime) {
      validateDeterminizationPublicHistory(
        nonCardRuntime,
        round,
        simulatedActs,
      );
    }
    const result = simulateAct(
      episodeId,
      round,
      players,
      random,
      config.policy,
      config.policyByPlayerId,
      config.supervisionPolicy,
      nonCardRuntime,
    );
    simulatedActs.push(result.act);
    steps.push(...result.steps);
    if (nonCardSteps && result.nonCardSteps) {
      nonCardSteps.push(...result.nonCardSteps);
    }
    players = result.players;
    random = result.nextRandom;
  }
  if (nonCardRuntime) {
    assertDeterminizationCompleted(nonCardRuntime);
    assertAllNonCardOverridesConsumed(nonCardRuntime);
  }

  return {
    episodeId,
    seed,
    playerCount: config.playerCount,
    acts: simulatedActs,
    steps,
    ...(nonCardSteps ? { nonCardSteps } : {}),
    ...(nonCardRuntime?.determinization?.audit
      ? {
          nonCardDeterminizationAudit:
            nonCardRuntime.determinization.audit,
        }
      : {}),
    finalScores: Object.fromEntries(
      players.map((player) => [player.id, player.score]),
    ),
  };
}

export function createBaselineTrainingPolicy(
  difficulty: BotDifficulty,
): TrainingPolicy {
  return ({ observation }) =>
    semanticActionIndexFromBotAction(
      chooseBotPlay(observation, difficulty).action,
    );
}
