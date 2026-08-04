import type { BotDifficulty } from "./bot-strategy.ts";

/**
 * Maps the user-facing difficulty to the legacy heuristic that backs it.
 *
 * Easy deliberately reuses the previous Hard policy. Normal is unchanged.
 * The learned Hard policy was validated with Normal for tax, revolution, and
 * fail-safe card decisions, so Hard also maps to Normal at those boundaries.
 */
export function deploymentHeuristicDifficulty(
  difficulty: BotDifficulty | null | undefined,
): Exclude<BotDifficulty, "easy"> {
  return difficulty === "easy" ? "hard" : "normal";
}
