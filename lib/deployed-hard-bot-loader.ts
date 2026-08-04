import {
  chooseBotCardIds,
  type BotPlayObservation,
} from "@/lib/bot-strategy";
import type { DeployedHardBotContext } from "@/lib/deployed-hard-bot-policy";

type HardBotModule = typeof import("@/lib/deployed-hard-bot-policy");

let loadedModule: HardBotModule | null = null;
let loadingModule: Promise<HardBotModule> | null = null;

export async function loadDeployedHardBot(): Promise<void> {
  if (loadedModule) return;
  loadingModule ??= import("@/lib/deployed-hard-bot-policy");
  try {
    loadedModule = await loadingModule;
    loadedModule.prepareDeployedHardBotPolicy();
  } catch (error) {
    loadingModule = null;
    throw error;
  }
}

export function deployedHardBotIsLoaded(): boolean {
  return loadedModule !== null;
}

export function chooseLoadedHardBotCardIds(
  observation: BotPlayObservation,
  context: DeployedHardBotContext,
): string[] | null {
  return loadedModule
    ? loadedModule.chooseDeployedHardBotCardIds(observation, context)
    : chooseBotCardIds(observation, "normal");
}
