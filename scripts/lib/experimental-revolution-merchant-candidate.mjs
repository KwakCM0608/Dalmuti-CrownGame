import { chooseBotRevolution } from "../../lib/bot-strategy.ts";
import {
  REVOLUTION_DECLARE_ACTION_INDEX,
  REVOLUTION_DECLINE_ACTION_INDEX,
} from "../../training/non-card-action-space.ts";
import { validateRevolutionObservation } from "../../training/non-card-observation.ts";

/**
 * Training-only candidate selected from the completed v2 determinization
 * study. Only merchant observations at p6 had a positive exploratory
 * Student-t 95% lower bound for declaring instead of the exact current normal
 * action. The p9 normal-approximation lower bound was barely positive, but its
 * small-sample Student-t lower bound was negative, so p9 is excluded.
 */
export const EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS = Object.freeze([
  6,
]);

export const EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION =
  "experimental-merchant-revolution-p6-v1";

export const REVOLUTION_MERCHANT_SOURCE_DATA = Object.freeze({
  format: "dalmuti-non-card-counterfactual-ndjson",
  version: 2,
  determinizationSchema: "world-clustered-paired-baseline-advantages-v2",
  sha256: "b861bc857e4e9d845f224ab06b3f9b2c503f9e3721e8e6979a99ef694dc96a05",
  normalizedRewardToActualChipMultiplier: 2,
});

function normalActionIndex(observation) {
  const actor = observation.players.find(
    (player) => player.id === observation.actorId,
  );
  if (!actor) {
    throw new TypeError("revolution observation does not contain its actor");
  }
  const decision = chooseBotRevolution(
    {
      hand: observation.hand,
      role: actor.role,
      playerCount: observation.players.length,
    },
    "normal",
  );
  return decision.declare
    ? REVOLUTION_DECLARE_ACTION_INDEX
    : REVOLUTION_DECLINE_ACTION_INDEX;
}

function validateEnabledPlayerCounts(enabledPlayerCounts) {
  if (!Array.isArray(enabledPlayerCounts) || enabledPlayerCounts.length < 1) {
    throw new TypeError("enabledPlayerCounts must be a non-empty array");
  }
  const unique = new Set();
  for (const playerCount of enabledPlayerCounts) {
    if (
      !Number.isInteger(playerCount) ||
      playerCount < 4 ||
      playerCount > 10
    ) {
      throw new RangeError(
        "enabledPlayerCounts entries must be integers from 4 to 10",
      );
    }
    if (unique.has(playerCount)) {
      throw new RangeError(`duplicate enabled player count ${playerCount}`);
    }
    unique.add(playerCount);
  }
  return unique;
}

/**
 * Deterministic candidate with an explicit exact-normal fallback.
 *
 * The candidate never changes a noble or peon decision and never changes a
 * merchant decision outside the evidence-supported player counts. The caller
 * can inject a count list in tests, but experiments use the frozen p6 list.
 */
export function selectExperimentalMerchantRevolution(
  observation,
  {
    enabledPlayerCounts = EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
  } = {},
) {
  validateRevolutionObservation(observation);
  const enabled = validateEnabledPlayerCounts(enabledPlayerCounts);
  const actor = observation.players.find(
    (player) => player.id === observation.actorId,
  );
  if (!actor) {
    throw new TypeError("revolution observation does not contain its actor");
  }
  const baselineActionIndex = normalActionIndex(observation);
  const eligible =
    actor.role === "merchant" && enabled.has(observation.players.length);
  const actionIndex = eligible
    ? REVOLUTION_DECLARE_ACTION_INDEX
    : baselineActionIndex;
  return Object.freeze({
    actionIndex,
    baselineActionIndex,
    changedFromBaseline: actionIndex !== baselineActionIndex,
    routing: eligible ? "merchant-declare" : "exact-normal-fallback",
    reason: eligible
      ? "merchant-role-and-player-count-supported-by-v2-determinization"
      : actor.role !== "merchant"
        ? "non-merchant-role"
        : "player-count-not-enabled",
    actorRole: actor.role,
    playerCount: observation.players.length,
    policyVersion: EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION,
  });
}

/** Training-simulator adapter that records every routing decision. */
export function createExperimentalMerchantRevolutionHook({
  candidateIds,
  telemetry = [],
  enabledPlayerCounts = EXPERIMENTAL_MERCHANT_REVOLUTION_PLAYER_COUNTS,
}) {
  if (!(candidateIds instanceof Set)) {
    throw new TypeError("candidateIds must be a Set");
  }
  if (!Array.isArray(telemetry)) {
    throw new TypeError("telemetry must be an array");
  }
  // Validate once here and again in the pure selector so a mutated caller
  // array cannot silently alter the experiment contract between decisions.
  const enabled = Object.freeze([
    ...validateEnabledPlayerCounts(enabledPlayerCounts),
  ].sort((left, right) => left - right));
  return Object.freeze({
    revolutionPolicy(context) {
      const selected = selectExperimentalMerchantRevolution(
        context.observation,
        { enabledPlayerCounts: enabled },
      );
      const candidateActor = candidateIds.has(context.actorId);
      const actionIndex = candidateActor
        ? selected.actionIndex
        : selected.baselineActionIndex;
      const routing = candidateActor
        ? selected.routing
        : "non-candidate-exact-normal";
      telemetry.push(
        Object.freeze({
          decisionKey: context.decisionKey,
          actorId: context.actorId,
          actorRole: context.actorRole,
          playerCount: context.observation.players.length,
          candidateActor,
          actionIndex,
          baselineActionIndex: selected.baselineActionIndex,
          changedFromBaseline: actionIndex !== selected.baselineActionIndex,
          routing,
          reason: candidateActor ? selected.reason : "non-candidate-actor",
          policyVersion: EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION,
        }),
      );
      return {
        actionIndex,
        logProbability: 0,
        policyVersion: EXPERIMENTAL_MERCHANT_REVOLUTION_POLICY_VERSION,
      };
    },
  });
}
