import hardBotModelJson from "./bot-models/hard-ppo5-epoch11.json" with {
  type: "json",
};
import {
  chooseBotCardIds,
  type BotPlayObservation,
  type BotRole,
} from "./bot-strategy.ts";
import {
  evaluateActorCritic,
  parseActorCriticModel,
  type ActorCriticModel,
} from "../training/actor-critic.ts";
import {
  legalSemanticActionIndices,
  resolveSemanticAction,
} from "../training/action-space.ts";
import {
  encodeTrainingObservation,
  type RevolutionState,
} from "../training/observation.ts";

export const DEPLOYED_HARD_BOT_MODEL_SHA256 =
  "3599a32b6a1a96eb2e353fabf6c1b47db93db92a980e26ab247af996d2b00395";
export const DEPLOYED_HARD_BOT_MODEL_ID = "ppo5-aux0-epoch11";

export type DeployedHardBotContext = {
  round: number;
  rolesByPlayerId: Readonly<Record<string, BotRole>>;
  scoresByPlayerId: Readonly<Record<string, number>>;
  revolution: RevolutionState;
};

let parsedModel: ActorCriticModel | null = null;

function getModel(): ActorCriticModel {
  parsedModel ??= parseActorCriticModel(hardBotModelJson as unknown);
  return parsedModel;
}

export function prepareDeployedHardBotPolicy(): void {
  getModel();
}

function normalFallback(observation: BotPlayObservation): string[] | null {
  return chooseBotCardIds(observation, "normal");
}

/**
 * Production adapter for the validated PPO5 play policy.
 *
 * It receives only the actor's private hand and public game information. Any
 * parsing, encoding, numerical, mask, or action-resolution failure falls back
 * to the exact Normal play policy instead of interrupting the game.
 */
export function chooseDeployedHardBotCardIds(
  observation: BotPlayObservation,
  context: DeployedHardBotContext,
): string[] | null {
  try {
    const model = getModel();
    const encodedObservation = encodeTrainingObservation({
      observation,
      round: context.round,
      rolesByPlayerId: context.rolesByPlayerId,
      scoresByPlayerId: context.scoresByPlayerId,
      revolution: context.revolution,
    });
    const legalActionIndices = legalSemanticActionIndices(observation);
    if (legalActionIndices.length < 1) return normalFallback(observation);

    const { logits } = evaluateActorCritic(model, encodedObservation);
    let selectedActionIndex = legalActionIndices[0];
    let selectedLogit = Number.NEGATIVE_INFINITY;
    for (const actionIndex of legalActionIndices) {
      const logit = logits[actionIndex];
      if (!Number.isFinite(logit)) return normalFallback(observation);
      if (logit > selectedLogit) {
        selectedActionIndex = actionIndex;
        selectedLogit = logit;
      }
    }

    const action = resolveSemanticAction(observation, selectedActionIndex);
    return action.type === "play" ? action.cardIds : null;
  } catch {
    return normalFallback(observation);
  }
}
