import {
  ACTION_SPACE_SIZE,
  MAX_PLAY_COUNT,
} from "./action-space.ts";
import {
  OBSERVATION_FEATURE_COUNT,
  OBSERVATION_FEATURE_GROUPS,
  OBSERVATION_SCHEMA_VERSION,
} from "./observation.ts";

export const ROLLOUT_FORMAT = "dalmuti-rl-ndjson";
export const ROLLOUT_FORMAT_VERSION = 1;

export function createRolloutManifest(config: {
  createdAt: string;
  episodes: number;
  playerCount: number;
  acts: number;
  seed: number;
  difficulties: readonly string[];
}) {
  return {
    type: "manifest",
    format: ROLLOUT_FORMAT,
    formatVersion: ROLLOUT_FORMAT_VERSION,
    createdAt: config.createdAt,
    environment: {
      game: "DALMUTI",
      rules: "project-house-rules-v1",
      playerCount: config.playerCount,
      actsPerEpisode: config.acts,
      episodes: config.episodes,
      initialSeed: config.seed,
      behaviorPolicies: config.difficulties,
      reward:
        "actorTerminal ? (roundChipAward - 2) / 2 : 0",
      syntheticSelfPlayOnly: true,
    },
    actionSpace: {
      size: ACTION_SPACE_SIZE,
      passIndex: 0,
      soloJokerIndex: 1,
      normalActionLayout:
        "2 + ((rank - 1) * 14 + (count - 1)) * 3 + jokerCount",
      normalRanks: 12,
      maximumCount: MAX_PLAY_COUNT,
      jokerCountOptions: 3,
      invalidActions: "masked by legalActionIndices",
    },
    observation: {
      version: OBSERVATION_SCHEMA_VERSION,
      featureCount: OBSERVATION_FEATURE_COUNT,
      groups: OBSERVATION_FEATURE_GROUPS,
      privacy:
        "own private hand plus public state only; opponent hands and private tax cards are excluded",
    },
    sampleFields: {
      episodeId: "string",
      round: "integer",
      step: "integer",
      actorId: "string",
      actorSeat: "integer",
      actorRole: "string",
      behaviorPolicy: "easy | normal | hard | custom",
      observation: `float[${OBSERVATION_FEATURE_COUNT}]`,
      legalActionIndices: "integer[]",
      actionIndex: "integer",
      supervisedActionIndex:
        "integer | null; when present this is the training target",
      forced: "boolean",
      reward: "float",
      actorTerminal: "boolean",
      environmentTerminal: "boolean",
      finishPlace: "integer",
    },
  };
}
