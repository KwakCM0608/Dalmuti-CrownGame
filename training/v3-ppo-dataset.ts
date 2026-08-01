import {
  OBSERVATION_FEATURE_COUNT,
  OBSERVATION_SCHEMA_VERSION,
} from "./observation.ts";
import {
  V3_ACTION_CATALOGUE,
  V3_ACTION_CATALOGUE_VERSION,
  V3_ACTION_COUNT,
  V3_ACTION_FEATURE_COUNT,
  V3_ACTION_FEATURE_LAYOUT,
  V3_ACTION_FEATURES,
} from "./v3-action-catalogue.ts";
import { V3_LEGAL_MASK_HEX_LENGTH } from "./v3-action-bridge.ts";

export const V3_PPO_ROLLOUT_FORMAT = "dalmuti-v3-ppo-ndjson";
export const V3_PPO_ROLLOUT_FORMAT_VERSION = 1;

export function createV3PpoRolloutManifest(config: {
  createdAt: string;
  episodes: number;
  playerCount: number;
  acts: number;
  seed: number;
  behaviorModelSha256: string;
  temperature: number;
  mode?: "self-play" | "league";
  opponentPolicies?: readonly string[];
}) {
  return {
    type: "manifest",
    format: V3_PPO_ROLLOUT_FORMAT,
    formatVersion: V3_PPO_ROLLOUT_FORMAT_VERSION,
    createdAt: config.createdAt,
    environment: {
      game: "DALMUTI",
      rules: "project-house-rules-v1",
      playerCount: config.playerCount,
      actsPerEpisode: config.acts,
      episodes: config.episodes,
      initialSeed: config.seed,
      rolloutMode: config.mode ?? "self-play",
      opponentPolicies: config.opponentPolicies ?? [],
      nonCardDecisions: "normal bot policy",
      reward: "actorTerminal ? (roundChipAward - 2) / 2 : 0",
    },
    behaviorModel: {
      sha256: config.behaviorModelSha256,
      format: "dalmuti-action-conditioned-actor-critic",
      observationSchemaVersion: OBSERVATION_SCHEMA_VERSION,
      observationFeatures: OBSERVATION_FEATURE_COUNT,
      actionCatalogueVersion: V3_ACTION_CATALOGUE_VERSION,
    },
    behaviorPolicy: {
      sampling: "softmax",
      temperature: config.temperature,
      logProbabilityBinding:
        "recomputed from behavior model over exactly legalMaskHex at this temperature",
    },
    observation: {
      version: OBSERVATION_SCHEMA_VERSION,
      featureCount: OBSERVATION_FEATURE_COUNT,
      privacy: "own private hand plus public state only; opponent hands excluded",
    },
    actionSpace: {
      catalogueVersion: V3_ACTION_CATALOGUE_VERSION,
      size: V3_ACTION_COUNT,
      catalogue: V3_ACTION_CATALOGUE,
      actionFeatures: V3_ACTION_FEATURE_COUNT,
      actionFeatureLayout: V3_ACTION_FEATURE_LAYOUT,
      encodedActionFeatures: V3_ACTION_FEATURES,
      legalMaskEncoding: {
        field: "legalMaskHex",
        lowercaseHexDigits: V3_LEGAL_MASK_HEX_LENGTH,
        bitOrder: "action index i = bit (i % 4) of hex digit floor(i / 4)",
      },
    },
    sampleBindings: {
      observationSchemaVersion: OBSERVATION_SCHEMA_VERSION,
      actionCatalogueVersion: V3_ACTION_CATALOGUE_VERSION,
      policyVersion: `sha256:${config.behaviorModelSha256}`,
      legalActionIndices: "unique ascending indices exactly equal to legalMaskHex",
      forced: "true exactly when legalActionIndices has length one",
    },
  };
}
