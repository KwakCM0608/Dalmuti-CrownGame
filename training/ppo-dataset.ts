import {
  ACTION_SPACE_SIZE,
} from "./action-space.ts";
import {
  OBSERVATION_FEATURE_COUNT,
  OBSERVATION_SCHEMA_VERSION,
} from "./observation.ts";

export const PPO_ROLLOUT_FORMAT = "dalmuti-ppo-ndjson";
export const PPO_ROLLOUT_FORMAT_VERSION = 1;

export function createPpoRolloutManifest(config: {
  createdAt: string;
  episodes: number;
  playerCount: number;
  acts: number;
  seed: number;
  behaviorModelSha256: string;
  behaviorModelFormat: string;
  mode?: "self-play" | "league";
  opponentPolicies?: readonly string[];
}) {
  return {
    type: "manifest",
    format: PPO_ROLLOUT_FORMAT,
    formatVersion: PPO_ROLLOUT_FORMAT_VERSION,
    createdAt: config.createdAt,
    environment: {
      game: "DALMUTI",
      rules: "project-house-rules-v1",
      playerCount: config.playerCount,
      actsPerEpisode: config.acts,
      episodes: config.episodes,
      initialSeed: config.seed,
      rolloutMode: config.mode ?? "self-play",
      learnerSeats:
        config.mode === "league"
          ? "approximately half of seats; only these decisions are samples"
          : "all card-play seats use the same behavior policy",
      opponentPolicies: config.opponentPolicies ?? [],
      nonCardDecisions: "normal bot policy",
      reward:
        "actorTerminal ? (roundChipAward - 2) / 2 : 0",
    },
    behaviorModel: {
      sha256: config.behaviorModelSha256,
      format: config.behaviorModelFormat,
    },
    observation: {
      version: OBSERVATION_SCHEMA_VERSION,
      featureCount: OBSERVATION_FEATURE_COUNT,
      privacy:
        "own private hand plus public state only; opponent hands excluded",
    },
    actionSpace: {
      size: ACTION_SPACE_SIZE,
      invalidActions: "masked by legalActionIndices",
    },
    sampleFields: {
      trajectoryId: "episodeId:round:actorId",
      episodeId: "string",
      round: "integer",
      step: "integer; global environment step within the act",
      actorId: "string",
      actorSeat: "integer",
      actorRole: "string",
      observation: `float[${OBSERVATION_FEATURE_COUNT}]`,
      legalActionIndices: "integer[]",
      actionIndex: "integer",
      oldLogProbability: "float <= 0",
      oldValue: "float",
      reward: "float",
      terminal: "boolean; final decision for this actor in this act",
      forced: "boolean",
      finishPlace: "integer",
      policyVersion: "sha256:<behavior model hash>",
    },
  };
}
