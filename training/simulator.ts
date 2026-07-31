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

export type TrainingPolicy = (context: TrainingPolicyContext) => number;

export type SimulationConfig = {
  playerCount: number;
  acts?: number;
  seed?: number;
  difficulties?: readonly BotDifficulty[];
  episodeId?: string;
  policy?: TrainingPolicy;
  policyByPlayerId?: Readonly<Record<string, TrainingPolicy | undefined>>;
  supervisionPolicy?: TrainingPolicy;
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
  forced: boolean;
  reward: number;
  actorTerminal: boolean;
  environmentTerminal: boolean;
  finishPlace: number;
};

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
  finalScores: Record<string, number>;
};

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

function chooseActionIndex(
  player: SimulationPlayer,
  observation: BotPlayObservation,
  encodedObservation: readonly number[],
  legalActionIndices: readonly number[],
  round: number,
  random: SeededRandom,
  policy?: TrainingPolicy,
): number {
  const actionIndex = policy
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
  if (!legalActionIndices.includes(actionIndex)) {
    throw new RangeError(
      `policy selected illegal action ${actionIndex} for ${player.id}`,
    );
  }
  return actionIndex;
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
): {
  act: SimulatedAct;
  players: SimulationPlayer[];
  steps: TrainingStep[];
} {
  let players = assignRoles(initialPlayers);
  const hands = deal(players, random);
  let revolution: RevolutionState = null;

  const revolutionHolder = players.find(
    (player) =>
      hands[player.id].filter((card) => card.rank === 13).length === 2,
  );
  if (revolutionHolder) {
    const decision = chooseBotRevolution(
      {
        hand: hands[revolutionHolder.id],
        role: revolutionHolder.role,
        playerCount: players.length,
      },
      revolutionHolder.difficulty,
    );
    if (decision.declare) {
      revolution = decision.kind;
      if (decision.kind === "great-revolution") {
        players = assignRoles([...players].reverse());
      }
    }
  }

  if (round > 1 && revolution === null) {
    applyTaxation(players, hands);
  }

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
    const actionIndex = chooseActionIndex(
      actor,
      observation,
      encodedObservation,
      legalActionIndices,
      round,
      random,
      policyByPlayerId?.[actor.id] ?? policy,
    );
    const supervisedActionIndex = supervisionPolicy
      ? chooseActionIndex(
          actor,
          observation,
          encodedObservation,
          legalActionIndices,
          round,
          random,
          supervisionPolicy,
        )
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
  const random = new SeededRandom(seed);
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

  for (let round = 1; round <= acts; round += 1) {
    const result = simulateAct(
      episodeId,
      round,
      players,
      random,
      config.policy,
      config.policyByPlayerId,
      config.supervisionPolicy,
    );
    simulatedActs.push(result.act);
    steps.push(...result.steps);
    players = result.players;
  }

  return {
    episodeId,
    seed,
    playerCount: config.playerCount,
    acts: simulatedActs,
    steps,
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
