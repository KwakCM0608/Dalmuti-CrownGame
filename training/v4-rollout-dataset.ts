import {
  V3_ACTION_CATALOGUE,
  V3_ACTION_CATALOGUE_VERSION,
  V3_ACTION_COUNT,
  V3_ACTION_FEATURES,
} from "./v3-action-catalogue.ts";
import { V3_LEGAL_MASK_HEX_LENGTH } from "./v3-action-bridge.ts";
import {
  V4_MAX_PUBLIC_HISTORY_EVENTS,
  V4_MEMORY_TRACE_DECAYS,
  V4_MEMORY_TRACE_FEATURES,
  V4_PUBLIC_OBSERVATION_SCHEMA_VERSION,
} from "./v4-public-history.ts";
import {
  V4_PRIVILEGED_CRITIC_FEATURE_COUNT,
  V4_PRIVILEGED_CRITIC_LAYOUT,
  V4_PRIVILEGED_CRITIC_SCHEMA_VERSION,
} from "./simulator.ts";

export const V4_NORMAL_WARMSTART_FORMAT =
  "dalmuti-v4-normal-warmstart-ndjson";
export const V4_NORMAL_WARMSTART_FORMAT_VERSION = 1;

const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export type V4RolloutSourceHashes = {
  readonly actorObservationContract: string;
  readonly privilegedCriticContract: string;
  readonly actionCatalogue: string;
  readonly normalPolicy: string;
  readonly generator: string;
  readonly datasetManifest: string;
};

export type V4NormalWarmstartManifestConfig = {
  readonly playerCount: number;
  readonly acts: number;
  readonly initialSeed: number;
  readonly targetNonForcedDecisions: number;
  readonly maxEpisodes: number;
  readonly sourceHashes: V4RolloutSourceHashes;
};

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
}

function assertSha256(value: string, label: string): void {
  if (!SHA256_PATTERN.test(value)) {
    throw new TypeError(`${label} must be a lowercase SHA-256 digest`);
  }
}

/** A deterministic manifest: no path, wall clock, hostname, or process ID. */
export function createV4NormalWarmstartManifest(
  config: V4NormalWarmstartManifestConfig,
) {
  assertPositiveInteger(config.playerCount, "playerCount");
  if (config.playerCount < 4 || config.playerCount > 10) {
    throw new RangeError("playerCount must be from 4 to 10");
  }
  assertPositiveInteger(config.acts, "acts");
  assertPositiveInteger(config.initialSeed, "initialSeed");
  assertPositiveInteger(
    config.targetNonForcedDecisions,
    "targetNonForcedDecisions",
  );
  assertPositiveInteger(config.maxEpisodes, "maxEpisodes");
  for (const [label, digest] of Object.entries(config.sourceHashes)) {
    assertSha256(digest, `sourceHashes.${label}`);
  }

  return {
    type: "manifest",
    format: V4_NORMAL_WARMSTART_FORMAT,
    formatVersion: V4_NORMAL_WARMSTART_FORMAT_VERSION,
    environment: {
      game: "DALMUTI",
      rules: "project-house-rules-v1",
      playerCount: config.playerCount,
      actsPerEpisode: config.acts,
      initialSeed: config.initialSeed,
      behaviorPolicy: "normal",
      reward: "actorTerminal ? (roundChipAward - 2) / 2 : 0",
      collection: {
        mode: "target-non-forced-decisions",
        targetNonForcedDecisions: config.targetNonForcedDecisions,
        maxEpisodes: config.maxEpisodes,
        completeEpisodesOnly: true,
      },
    },
    actorObservation: {
      schemaVersion: V4_PUBLIC_OBSERVATION_SCHEMA_VERSION,
      sourceSha256: config.sourceHashes.actorObservationContract,
      canonicalBuilder: "buildV4ActorVisibleObservation",
      maxRecentHistoryEvents: V4_MAX_PUBLIC_HISTORY_EVENTS,
      memoryTraceDecays: V4_MEMORY_TRACE_DECAYS,
      memoryTraceFeatures: V4_MEMORY_TRACE_FEATURES,
      privacy:
        "own physical hand plus public state/history only; IDs and opponent hands excluded",
    },
    privilegedCritic: {
      schemaVersion: V4_PRIVILEGED_CRITIC_SCHEMA_VERSION,
      sourceSha256: config.sourceHashes.privilegedCriticContract,
      featureCount: V4_PRIVILEGED_CRITIC_FEATURE_COUNT,
      layout: V4_PRIVILEGED_CRITIC_LAYOUT,
      actorExportAllowed: false,
      privacyClass: "restricted-training-only-full-state",
    },
    actionSpace: {
      catalogueVersion: V3_ACTION_CATALOGUE_VERSION,
      size: V3_ACTION_COUNT,
      catalogueSha256: config.sourceHashes.actionCatalogue,
      catalogue: V3_ACTION_CATALOGUE,
      encodedActionFeatures: V3_ACTION_FEATURES,
      legalMaskEncoding: {
        field: "legalMaskHex",
        lowercaseHexDigits: V3_LEGAL_MASK_HEX_LENGTH,
        bitOrder:
          "action index i = bit (i % 4) of hex digit floor(i / 4)",
      },
    },
    sampleBindings: {
      actionIndex: "236-action catalogue index selected by exact Normal",
      legalActionIndices:
        "unique ascending indices exactly equal to legalMaskHex",
      actorObservation:
        "canonical sanitized state immediately before the selected action",
      privilegedCriticState:
        "separate full-information state immediately before the action",
      eventsAfterAction:
        "ordered public play/pass/clear/finish events emitted by the action",
      forced: "true exactly when legalActionIndices has length one",
    },
    sourceHashes: {
      ...config.sourceHashes,
    },
  };
}
