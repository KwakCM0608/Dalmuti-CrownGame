"""Versioned dataset adapter for conservative tax-return advantage training.

Version 1 records contain one replay world and expose per-action
``decisionActUtility``.  Version 2 records aggregate multiple privacy-safe
determinizations and expose paired action-vs-normal-baseline statistics.  This
adapter produces the same actual-chip advantage tensor from either source:

* v1: ``(decisionActUtility[a] - decisionActUtility[baseline]) * 2``
* v2: ``pairedDecisionActBaselineAdvantage.mean * 2``

The factor two is mandatory because simulator reward is
``(roundChipAward - 2) / 2`` while the exported scorer is denominated in
actual chip units.  Every information state has weight one regardless of a v2
record's hidden-world count.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from non_card_action_conditioned import (
    NON_CARD_OBSERVATION_SCHEMA_VERSION,
    TAX_RETURN_ACTION_CATALOGUE_VERSION,
    TAX_RETURN_ACTION_COUNT,
    TAX_RETURN_ACTION_FEATURE_COUNT,
    TAX_RETURN_ACTION_FEATURES,
    TAX_RETURN_OBSERVATION_FEATURE_COUNT,
    legal_tax_return_masks_from_observations,
)
from non_card_counterfactual_dataset import (
    COUNTERFACTUAL_FORMAT,
    DecisionArrays,
    ROLE_NAMES,
    UINT32_MAX,
    _roles_for_player_count,
    _validate_common_observation_semantics,
    deterministic_validation_membership,
    expand_input_paths,
    file_sha256,
    load_non_card_counterfactuals,
)


DETERMINIZATION_FORMAT_VERSION = 2
DETERMINIZATION_SCHEMA = "world-clustered-paired-baseline-advantages-v2"
CANONICAL_INFORMATION_STATE_KEY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PLAIN_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DETERMINIZATION_ALGORITHM = (
    "target-act-opponent-physical-card-fisher-yates-v1"
)
DETERMINIZATION_ALGORITHM_VERSION = 1
DETERMINIZATION_CANDIDATE_SEED_DERIVATION = (
    "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,"
    "worldIndex,attempt)))"
)
DETERMINIZATION_CONTINUATION_SEED_DERIVATION = (
    "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,"
    "worldIndex,continuationIndex,continuation)))"
)
DETERMINIZATION_CONTINUATION_RNG_PAIRING = (
    "same-environment-stream-and-hidden-world-seed-for-every-root-action"
)
DETERMINIZATION_CONTRACT = {
    "algorithm": DETERMINIZATION_ALGORITHM,
    "version": DETERMINIZATION_ALGORITHM_VERSION,
    "actorHand": "original-replay-hand-fixed",
    "opponents": (
        "all-non-actor-physical-cards-shuffled-then-dealt-to-original-"
        "public-hand-counts-in-rank-order"
    ),
    "environmentRng": "not-consumed",
    "candidateSeed": DETERMINIZATION_CANDIDATE_SEED_DERIVATION,
    "continuationSeed": DETERMINIZATION_CONTINUATION_SEED_DERIVATION,
}


def _compact_json_bytes(value: object) -> bytes:
    """Match the collector's insertion-order ``JSON.stringify`` material."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


DETERMINIZATION_CONTRACT_SHA256 = hashlib.sha256(
    _compact_json_bytes(DETERMINIZATION_CONTRACT)
).hexdigest()
if DETERMINIZATION_CONTRACT_SHA256 != (
    "368240f14f2e5d84bb3085610a176ad4519bc6e5ae288b70de549f63212905c4"
):
    raise RuntimeError("tax-return determinization contract hash drifted")

SIMULATOR_REWARD_TO_CHIP_MULTIPLIER = 2.0
TARGET_TRANSFORM = {
    "scoreUnit": "chip-units",
    "sourceUnit": "(roundChipAward-2)/2",
    "operation": "multiply-source-baseline-advantage-by-2",
    "multiplier": SIMULATOR_REWARD_TO_CHIP_MULTIPLIER,
}


def simulator_reward_advantage_to_chips(value: float) -> float:
    """Convert simulator reward advantage to actual chip-award advantage."""

    result = _number(value, "simulator reward advantage") * SIMULATOR_REWARD_TO_CHIP_MULTIPLIER
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class TaxAdvantageArrays:
    observations: np.ndarray
    legal_masks: np.ndarray
    baseline_actions: np.ndarray
    target_advantages: np.ndarray
    sample_ids: tuple[str, ...]
    group_keys: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class TaxAdvantageDataset:
    train: TaxAdvantageArrays
    validation: TaxAdvantageArrays
    exclusion_counts: Mapping[str, int]
    source_files: tuple[Mapping[str, object], ...]
    source_contract: Mapping[str, object]
    group_split_key: str
    validation_fraction: float
    split_seed: int


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_line(raw: bytes, path: Path, line_number: int) -> dict[str, object]:
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path}:{line_number}: NDJSON line lacks a newline")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _integer(
    value: object,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        maximum_suffix = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(
            f"{label} must be an integer of at least {minimum}{maximum_suffix}"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _plain_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not PLAIN_SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_list(value: object, length: int, label: str) -> list[float]:
    items = _list(value, label)
    if len(items) != length:
        raise ValueError(f"{label} must contain {length} items")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(items)]


def _stats(
    value: object,
    *,
    label: str,
    independent_world_count: int,
) -> tuple[float, float, float]:
    stats = _object(value, label)
    if set(stats) != {
        "mean",
        "sampleStandardDeviation",
        "standardError",
        "count",
        "standardErrorEstimable",
    }:
        raise ValueError(f"{label} fields mismatch")
    if stats["count"] != independent_world_count:
        raise ValueError(
            f"{label}.count does not match effectiveIndependentWorlds"
        )
    if stats["standardErrorEstimable"] is not (independent_world_count > 1):
        raise ValueError(f"{label}.standardErrorEstimable is inconsistent")
    mean = _number(stats["mean"], f"{label}.mean")
    standard_deviation = _number(
        stats["sampleStandardDeviation"],
        f"{label}.sampleStandardDeviation",
    )
    standard_error = _number(stats["standardError"], f"{label}.standardError")
    if standard_deviation < 0 or standard_error < 0:
        raise ValueError(f"{label} uncertainty must be non-negative")
    expected_error = standard_deviation / math.sqrt(independent_world_count)
    if not math.isclose(standard_error, expected_error, rel_tol=1.0e-7, abs_tol=1.0e-10):
        raise ValueError(f"{label} uncertainty is inconsistent with worldCount")
    return mean, standard_deviation, standard_error


def _uncertainty(
    value: object,
    *,
    label: str,
    independent_world_count: int,
) -> tuple[float, float]:
    uncertainty = _object(value, label)
    if set(uncertainty) != {
        "sampleStandardDeviation",
        "standardError",
        "count",
        "standardErrorEstimable",
    }:
        raise ValueError(f"{label} fields mismatch")
    wrapped = {
        "mean": 0.0,
        **uncertainty,
    }
    _, standard_deviation, standard_error = _stats(
        wrapped,
        label=label,
        independent_world_count=independent_world_count,
    )
    return standard_deviation, standard_error


def _validate_v2_manifest(value: dict[str, object], path: Path) -> dict[str, object]:
    if set(value) != {
        "type",
        "format",
        "version",
        "createdAt",
        "observationSchemaVersion",
        "actionCatalogueVersions",
        "featureDimensions",
        "collection",
        "privacy",
        "groupSplitKey",
        "determinizationSchema",
    }:
        raise ValueError(f"{path}: v2 manifest fields mismatch")
    if (
        value.get("type") != "manifest"
        or value.get("format") != COUNTERFACTUAL_FORMAT
        or value.get("version") != DETERMINIZATION_FORMAT_VERSION
    ):
        raise ValueError(f"{path}: unsupported determinization dataset manifest")
    _string(value.get("createdAt"), f"{path}: createdAt")
    if (
        value.get("observationSchemaVersion") != NON_CARD_OBSERVATION_SCHEMA_VERSION
        or value.get("groupSplitKey") != "canonicalInformationStateKey"
        or value.get("determinizationSchema") != DETERMINIZATION_SCHEMA
    ):
        raise ValueError(f"{path}: determinization schema binding mismatch")
    catalogue = _object(value.get("actionCatalogueVersions"), "actionCatalogueVersions")
    if catalogue != {"taxReturn": TAX_RETURN_ACTION_CATALOGUE_VERSION, "revolution": 1}:
        raise ValueError(f"{path}: tax-return catalogue version mismatch")
    dimensions = _object(value.get("featureDimensions"), "featureDimensions")
    if set(dimensions) != {"taxReturn", "revolution"}:
        raise ValueError(f"{path}: feature-dimension fields mismatch")
    tax_dimensions = _object(dimensions.get("taxReturn"), "featureDimensions.taxReturn")
    if tax_dimensions != {
        "observation": TAX_RETURN_OBSERVATION_FEATURE_COUNT,
        "action": TAX_RETURN_ACTION_FEATURE_COUNT,
        "catalogue": TAX_RETURN_ACTION_COUNT,
    }:
        raise ValueError(f"{path}: tax-return feature dimensions mismatch")
    revolution_dimensions = _object(
        dimensions.get("revolution"), "featureDimensions.revolution"
    )
    if revolution_dimensions != {"observation": 102, "action": 3, "catalogue": 2}:
        raise ValueError(f"{path}: revolution feature dimensions mismatch")
    collection = _object(value.get("collection"), "collection")
    if set(collection) != {
        "playerCounts",
        "episodesPerPlayerCount",
        "acts",
        "initialSeed",
        "matchSeedDerivation",
        "decisionKinds",
        "policyTemperature",
        "maxDecisions",
        "baselineNonCardHooks",
        "continuationPolicy",
        "resumeAllowed",
        "taxReturnCounts",
        "determinization",
    }:
        raise ValueError(f"{path}: v2 collection fields mismatch")
    player_counts_raw = _list(collection.get("playerCounts"), "collection.playerCounts")
    player_counts = [
        _integer(item, "collection.playerCounts", 4, 10)
        for item in player_counts_raw
    ]
    if player_counts != sorted(set(player_counts)) or not player_counts:
        raise ValueError(f"{path}: playerCounts must be unique and ascending")
    episodes = _integer(
        collection.get("episodesPerPlayerCount"),
        "collection.episodesPerPlayerCount",
        1,
    )
    acts = _integer(collection.get("acts"), "collection.acts", 1)
    initial_seed = _integer(
        collection.get("initialSeed"), "collection.initialSeed", 0, UINT32_MAX
    )
    if collection.get("matchSeedDerivation") != (
        "initialSeed + zero-based index over ascending playerCount then episode"
    ):
        raise ValueError(f"{path}: match-seed derivation mismatch")
    decision_kinds = _list(collection.get("decisionKinds"), "collection.decisionKinds")
    if (
        "tax-return" not in decision_kinds
        or decision_kinds != list(dict.fromkeys(decision_kinds))
        or any(item not in ("tax-return", "revolution") for item in decision_kinds)
    ):
        raise ValueError(f"{path}: dataset does not include tax-return decisions")
    policy_temperature = _number(
        collection.get("policyTemperature"), "collection.policyTemperature"
    )
    if policy_temperature <= 0:
        raise ValueError(f"{path}: policyTemperature must be positive")
    max_decisions = collection.get("maxDecisions")
    if max_decisions is not None:
        _integer(max_decisions, "collection.maxDecisions", 1)
    if collection.get("baselineNonCardHooks") != {}:
        raise ValueError(f"{path}: v2 baseline must use the exact normal hooks")
    if collection.get("continuationPolicy") != "normal-deterministic":
        raise ValueError(f"{path}: unsupported continuation policy")
    if collection.get("resumeAllowed") is not False:
        raise ValueError(f"{path}: v2 collection must forbid resume")
    tax_return_counts_raw = _list(
        collection.get("taxReturnCounts"), "collection.taxReturnCounts"
    )
    tax_return_counts = [
        _integer(item, "collection.taxReturnCounts", 1, 2)
        for item in tax_return_counts_raw
    ]
    if (
        2 not in tax_return_counts
        or tax_return_counts != sorted(set(tax_return_counts))
    ):
        raise ValueError(f"{path}: taxReturnCounts must uniquely include two")
    determinization = _object(collection.get("determinization"), "collection.determinization")
    if set(determinization) != {
        "worldCountPerInformationState",
        "continuationCountPerHiddenWorld",
        "rawContinuationEvaluationsPerInformationState",
        "effectiveIndependentWorldsPerInformationState",
        "standardErrorEstimable",
        "originalReplayWorldIncluded",
        "rootSeed",
        "maxAttemptsPerResampledWorld",
        "algorithm",
        "algorithmVersion",
        "algorithmContractSha256",
        "candidateSeedDerivation",
        "continuationSeedDerivation",
    }:
        raise ValueError(f"{path}: v2 determinization fields mismatch")
    world_count = _integer(
        determinization.get("worldCountPerInformationState"),
        "worldCountPerInformationState",
        1,
    )
    continuation_count = _integer(
        determinization.get("continuationCountPerHiddenWorld"),
        "continuationCountPerHiddenWorld",
        1,
    )
    effective_worlds = _integer(
        determinization.get("effectiveIndependentWorldsPerInformationState"),
        "effectiveIndependentWorldsPerInformationState",
        1,
    )
    raw_continuation_evaluations = _integer(
        determinization.get("rawContinuationEvaluationsPerInformationState"),
        "rawContinuationEvaluationsPerInformationState",
        1,
    )
    if (
        effective_worlds != world_count
        or raw_continuation_evaluations != world_count * continuation_count
    ):
        raise ValueError(
            f"{path}: effective/raw continuation count binding mismatch"
        )
    if determinization.get("standardErrorEstimable") is not (world_count > 1):
        raise ValueError(f"{path}: standard-error estimability mismatch")
    if determinization.get("originalReplayWorldIncluded") is not True:
        raise ValueError(f"{path}: original replay world must be included")
    root_seed = _integer(
        determinization.get("rootSeed"), "determinization.rootSeed", 0, UINT32_MAX
    )
    max_attempts = _integer(
        determinization.get("maxAttemptsPerResampledWorld"),
        "determinization.maxAttemptsPerResampledWorld",
        1,
    )
    if (
        determinization.get("algorithm") != DETERMINIZATION_ALGORITHM
        or determinization.get("algorithmVersion")
        != DETERMINIZATION_ALGORITHM_VERSION
        or determinization.get("algorithmContractSha256")
        != DETERMINIZATION_CONTRACT_SHA256
        or determinization.get("candidateSeedDerivation")
        != DETERMINIZATION_CANDIDATE_SEED_DERIVATION
        or determinization.get("continuationSeedDerivation")
        != DETERMINIZATION_CONTINUATION_SEED_DERIVATION
    ):
        raise ValueError(f"{path}: unknown determinization algorithm contract")
    privacy = _object(value.get("privacy"), "privacy")
    required_privacy = {
        "observation": "encoded-actor-hand-and-public-state-only",
        "opponentCardIdentitiesIncluded": False,
        "physicalCardIdsIncluded": False,
        "individualReplaySeedsIncluded": False,
        "explicitIndividualSeedsIncluded": False,
        "individualSeedsDerivableFromRestrictedRootProvenance": True,
        "individualWorldUtilitiesIncluded": False,
        "aggregateTargetsOnly": True,
        "distribution": "restricted-training-only",
    }
    if privacy != required_privacy:
        raise ValueError(f"{path}: determinization privacy contract mismatch")
    return {
        "worldCount": world_count,
        "continuationCount": continuation_count,
        "effectiveIndependentWorlds": effective_worlds,
        "rawContinuationEvaluations": raw_continuation_evaluations,
        "standardErrorEstimable": world_count > 1,
        "determinizationAlgorithm": DETERMINIZATION_ALGORITHM,
        "determinizationAlgorithmVersion": DETERMINIZATION_ALGORITHM_VERSION,
        "determinizationAlgorithmContractSha256": DETERMINIZATION_CONTRACT_SHA256,
        "candidateSeedDerivation": DETERMINIZATION_CANDIDATE_SEED_DERIVATION,
        "continuationSeedDerivation": DETERMINIZATION_CONTINUATION_SEED_DERIVATION,
        "rootSeed": root_seed,
        "maxAttemptsPerResampledWorld": max_attempts,
        "playerCounts": tuple(player_counts),
        "acts": acts,
        "episodesPerPlayerCount": episodes,
        "initialSeed": initial_seed,
        "policyTemperature": policy_temperature,
        "taxReturnCounts": tuple(tax_return_counts),
    }


def canonical_information_state_key(value: Mapping[str, object]) -> str:
    """Recompute the collector's public-state grouping key exactly.

    The key deliberately excludes opponent identities, replay seeds, and every
    hidden-world result.  It includes the normal baseline action because that
    action defines the residual target used by this pipeline.
    """

    raw_metadata = value.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("canonical information-state metadata must be an object")
    metadata = {
        "playerCount": raw_metadata.get("playerCount"),
        "actorHandCount": raw_metadata.get("actorHandCount"),
    }
    if value.get("decision") == "tax-return":
        metadata["returnCount"] = raw_metadata.get("returnCount")
    material = {
        "decision": value.get("decision"),
        "observationSchemaVersion": value.get("observationSchemaVersion"),
        "actionCatalogueVersion": value.get("actionCatalogueVersion"),
        "observation": value.get("observation"),
        "legalActionIndices": value.get("legalActionIndices"),
        "baselineActionIndex": value.get("baselineActionIndex"),
        "metadata": metadata,
    }
    return "sha256:" + hashlib.sha256(_compact_json_bytes(material)).hexdigest()


def _validate_v2_tax_record(
    value: dict[str, object],
    *,
    path: Path,
    line_number: int,
    manifest_contract: Mapping[str, object],
) -> dict[str, object] | None:
    label = f"{path}:{line_number}"
    if value.get("type") != "counterfactual-decision":
        raise ValueError(f"{label}: unsupported record type")
    expected_record_keys = {
        "type",
        "sampleId",
        "canonicalInformationStateKey",
        "decision",
        "playerCount",
        "acts",
        "round",
        "actorId",
        "actorSeat",
        "actorRole",
        "observationSchemaVersion",
        "actionCatalogueVersion",
        "observation",
        "legalMask",
        "legalActionIndices",
        "baselineActionIndex",
        "metadata",
        "pairing",
        "determinization",
        "utility",
        "targetBuilder",
        "targetSampleCount",
        "bestActionIndex",
        "bestDecisionActActionIndex",
        "forcedActionEvaluations",
        "actions",
    }
    if set(value) != expected_record_keys:
        raise ValueError(f"{label}: v2 decision fields mismatch")
    if value.get("decision") != "tax-return":
        return None
    sample_id = _string(value.get("sampleId"), f"{label}.sampleId")
    player_count = _integer(
        value.get("playerCount"), f"{label}.playerCount", 4, 10
    )
    if player_count not in manifest_contract["playerCounts"]:
        raise ValueError(f"{label}: playerCount is absent from the manifest")
    acts = _integer(value.get("acts"), f"{label}.acts", 1)
    if acts != manifest_contract["acts"]:
        raise ValueError(f"{label}: acts does not match the manifest")
    round_number = _integer(value.get("round"), f"{label}.round", 1, acts)
    _string(value.get("actorId"), f"{label}.actorId")
    actor_seat = _integer(
        value.get("actorSeat"), f"{label}.actorSeat", 0, player_count - 1
    )
    actor_role = _string(value.get("actorRole"), f"{label}.actorRole")
    if actor_role not in ROLE_NAMES:
        raise ValueError(f"{label}: actorRole is invalid")
    if actor_role != _roles_for_player_count(player_count)[actor_seat]:
        raise ValueError(f"{label}: actorRole does not match actorSeat")
    key = value.get("canonicalInformationStateKey")
    if not isinstance(key, str) or not CANONICAL_INFORMATION_STATE_KEY_RE.fullmatch(key):
        raise ValueError(f"{label}: canonicalInformationStateKey is invalid")
    pairing = _object(value.get("pairing"), f"{label}.pairing")
    if set(pairing) != {
        "canonicalInformationStateKey",
        "preDecisionSha256",
        "continuationPolicy",
        "forcedOverrideNamespace",
        "rootActionCoverage",
        "continuationRngPairing",
    }:
        raise ValueError(f"{label}: v2 pairing fields mismatch")
    if pairing.get("canonicalInformationStateKey") != key:
        raise ValueError(f"{label}: pairing information-state key mismatch")
    predecision_sha = _plain_sha256(
        pairing.get("preDecisionSha256"), f"{label}.pairing.preDecisionSha256"
    )
    if sample_id != f"tax-return:{predecision_sha}":
        raise ValueError(f"{label}: sampleId does not bind preDecisionSha256")
    if pairing.get("continuationPolicy") != "normal-deterministic":
        raise ValueError(f"{label}: continuation policy mismatch")
    if pairing.get("forcedOverrideNamespace") != "taxReturn":
        raise ValueError(f"{label}: forced override namespace mismatch")
    if pairing.get("rootActionCoverage") != "all-legal-actions-in-every-accepted-hidden-world":
        raise ValueError(f"{label}: hidden-world action coverage mismatch")
    if (
        pairing.get("continuationRngPairing")
        != DETERMINIZATION_CONTINUATION_RNG_PAIRING
    ):
        raise ValueError(f"{label}: continuation RNG pairing mismatch")
    determinization = _object(value.get("determinization"), f"{label}.determinization")
    if set(determinization) != {
        "worldCount",
        "continuationCount",
        "rawContinuationEvaluations",
        "effectiveIndependentWorlds",
        "standardErrorEstimable",
        "originalReplayWorldIncluded",
        "resampledWorldCount",
        "rootSeed",
        "maxAttemptsPerResampledWorld",
        "algorithm",
        "algorithmVersion",
        "algorithmContractSha256",
        "candidateSeedDerivation",
        "continuationSeedDerivation",
        "acceptedWorldAttempts",
        "individualReplaySeedsIncluded",
        "explicitIndividualSeedsIncluded",
        "individualSeedsDerivableFromRestrictedRootProvenance",
        "individualWorldUtilitiesIncluded",
        "distribution",
    }:
        raise ValueError(f"{label}: v2 determinization fields mismatch")
    world_count = _integer(determinization.get("worldCount"), f"{label}.worldCount", 1)
    if world_count != manifest_contract["worldCount"]:
        raise ValueError(f"{label}: decision/manifest worldCount mismatch")
    continuation_count = _integer(
        determinization.get("continuationCount"),
        f"{label}.continuationCount",
        1,
    )
    effective_worlds = _integer(
        determinization.get("effectiveIndependentWorlds"),
        f"{label}.effectiveIndependentWorlds",
        1,
    )
    raw_continuation_evaluations = _integer(
        determinization.get("rawContinuationEvaluations"),
        f"{label}.rawContinuationEvaluations",
        1,
    )
    if (
        continuation_count != manifest_contract["continuationCount"]
        or effective_worlds != manifest_contract["effectiveIndependentWorlds"]
        or effective_worlds != world_count
        or raw_continuation_evaluations
        != manifest_contract["rawContinuationEvaluations"]
        or raw_continuation_evaluations != world_count * continuation_count
        or determinization.get("standardErrorEstimable")
        is not manifest_contract["standardErrorEstimable"]
    ):
        raise ValueError(f"{label}: continuation/sample-count binding mismatch")
    if (
        determinization.get("originalReplayWorldIncluded") is not True
        or determinization.get("resampledWorldCount") != world_count - 1
        or determinization.get("rootSeed") != manifest_contract["rootSeed"]
        or determinization.get("maxAttemptsPerResampledWorld")
        != manifest_contract["maxAttemptsPerResampledWorld"]
        or determinization.get("algorithm")
        != manifest_contract["determinizationAlgorithm"]
        or determinization.get("algorithmVersion")
        != manifest_contract["determinizationAlgorithmVersion"]
        or determinization.get("algorithmContractSha256")
        != manifest_contract["determinizationAlgorithmContractSha256"]
        or determinization.get("candidateSeedDerivation")
        != manifest_contract["candidateSeedDerivation"]
        or determinization.get("continuationSeedDerivation")
        != manifest_contract["continuationSeedDerivation"]
    ):
        raise ValueError(f"{label}: determinization algorithm binding mismatch")
    accepted_attempts = _list(
        determinization.get("acceptedWorldAttempts"),
        f"{label}.determinization.acceptedWorldAttempts",
    )
    if len(accepted_attempts) != world_count - 1:
        raise ValueError(f"{label}: accepted-world attempt count mismatch")
    for world_index, raw_attempt in enumerate(accepted_attempts, 1):
        attempt = _object(
            raw_attempt,
            f"{label}.determinization.acceptedWorldAttempts[{world_index - 1}]",
        )
        if set(attempt) != {
            "worldIndex",
            "attemptCount",
            "rejectedAttemptCount",
            "rejectedReasonCounts",
        }:
            raise ValueError(f"{label}: accepted-world attempt fields mismatch")
        if attempt.get("worldIndex") != world_index:
            raise ValueError(f"{label}: accepted-world indices are not canonical")
        attempt_count = _integer(
            attempt.get("attemptCount"),
            f"{label}.acceptedWorldAttempts.attemptCount",
            1,
            int(manifest_contract["maxAttemptsPerResampledWorld"]),
        )
        rejected_count = _integer(
            attempt.get("rejectedAttemptCount"),
            f"{label}.acceptedWorldAttempts.rejectedAttemptCount",
            0,
        )
        if rejected_count != attempt_count - 1:
            raise ValueError(f"{label}: rejected-attempt count mismatch")
        reason_counts = _object(
            attempt.get("rejectedReasonCounts"),
            f"{label}.acceptedWorldAttempts.rejectedReasonCounts",
        )
        if any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for reason, count in reason_counts.items()
        ) or sum(int(count) for count in reason_counts.values()) != rejected_count:
            raise ValueError(f"{label}: rejected-reason counts mismatch")
    if (
        determinization.get("individualReplaySeedsIncluded") is not False
        or determinization.get("explicitIndividualSeedsIncluded") is not False
        or determinization.get(
            "individualSeedsDerivableFromRestrictedRootProvenance"
        )
        is not True
        or determinization.get("individualWorldUtilitiesIncluded") is not False
        or determinization.get("distribution") != "restricted-training-only"
    ):
        raise ValueError(f"{label}: individual hidden-world data must remain private")
    if value.get("observationSchemaVersion") != NON_CARD_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(f"{label}: observation schema mismatch")
    if value.get("actionCatalogueVersion") != TAX_RETURN_ACTION_CATALOGUE_VERSION:
        raise ValueError(f"{label}: action catalogue mismatch")
    observation = _finite_list(
        value.get("observation"),
        TAX_RETURN_OBSERVATION_FEATURE_COUNT,
        f"{label}.observation",
    )
    legal_mask_raw = _list(value.get("legalMask"), f"{label}.legalMask")
    if len(legal_mask_raw) != TAX_RETURN_ACTION_COUNT or any(
        not isinstance(item, bool) for item in legal_mask_raw
    ):
        raise ValueError(f"{label}: legalMask is invalid")
    legal_mask = [bool(item) for item in legal_mask_raw]
    derived = legal_tax_return_masks_from_observations(
        torch.tensor([observation], dtype=torch.float32)
    )[0].tolist()
    if legal_mask != derived:
        raise ValueError(f"{label}: legalMask does not match observation")
    legal_indices = [index for index, legal in enumerate(legal_mask) if legal]
    if value.get("legalActionIndices") != legal_indices:
        raise ValueError(f"{label}: legalActionIndices mismatch")
    baseline_action = _integer(value.get("baselineActionIndex"), f"{label}.baselineActionIndex")
    if baseline_action not in legal_indices:
        raise ValueError(f"{label}: baseline action is illegal")
    metadata = _object(value.get("metadata"), f"{label}.metadata")
    if set(metadata) != {"playerCount", "actorHandCount", "returnCount"}:
        raise ValueError(f"{label}: tax-return metadata fields mismatch")
    if metadata.get("playerCount") != player_count:
        raise ValueError(f"{label}: metadata playerCount mismatch")
    actor_hand_count = _integer(
        metadata.get("actorHandCount"), f"{label}.actorHandCount", 1, 20
    )
    return_count = _integer(
        metadata.get("returnCount"), f"{label}.returnCount", 1, 2
    )
    if return_count not in manifest_contract["taxReturnCounts"]:
        raise ValueError(f"{label}: returnCount was not requested by the manifest")
    encoded_return_count = 1 if observation[101] > 0.5 else 2
    if return_count != encoded_return_count:
        raise ValueError(f"{label}: returnCount metadata mismatch")
    _validate_common_observation_semantics(
        observation,
        decision="tax-return",
        player_count=player_count,
        round_number=round_number,
        actor_hand_count=actor_hand_count,
        actor_role=actor_role,
        actor_seat=actor_seat,
    )
    expected_key = canonical_information_state_key(value)
    if key != expected_key:
        raise ValueError(
            f"{label}: canonicalInformationStateKey does not match public state"
        )
    utility = _object(value.get("utility"), f"{label}.utility")
    if utility != {
        "terminalDefinition": "terminal-cumulative-chip-score",
        "decisionActDefinition": "centered-round-chip-award",
        "centeredAcrossLegalActions": True,
        "pairedBaselineAdvantagesBeforeAggregation": True,
    }:
        raise ValueError(f"{label}: v2 utility contract mismatch")
    if value.get("targetBuilder") != (
        "training/non-card-search-targets.ts#buildPairedCounterfactualTargets"
    ):
        raise ValueError(f"{label}: target builder mismatch")
    target_sample_count = _integer(
        value.get("targetSampleCount"), f"{label}.targetSampleCount", 1
    )
    if target_sample_count != effective_worlds:
        raise ValueError(
            f"{label}: targetSampleCount must equal effectiveIndependentWorlds"
        )
    actions = _list(value.get("actions"), f"{label}.actions")
    if len(actions) != len(legal_indices):
        raise ValueError(f"{label}: actions must cover every legal action")
    forced_action_evaluations = _integer(
        value.get("forcedActionEvaluations"),
        f"{label}.forcedActionEvaluations",
        1,
    )
    if forced_action_evaluations != (
        world_count * continuation_count * len(legal_indices)
    ):
        raise ValueError(f"{label}: forcedActionEvaluations mismatch")
    best_action = _integer(
        value.get("bestActionIndex"), f"{label}.bestActionIndex", 0,
        TAX_RETURN_ACTION_COUNT - 1,
    )
    best_decision_act_action = _integer(
        value.get("bestDecisionActActionIndex"),
        f"{label}.bestDecisionActActionIndex",
        0,
        TAX_RETURN_ACTION_COUNT - 1,
    )
    if best_action not in legal_indices or best_decision_act_action not in legal_indices:
        raise ValueError(f"{label}: best action is illegal")
    target_advantages = np.zeros(TAX_RETURN_ACTION_COUNT, dtype=np.float32)
    terminal_means: list[float] = []
    terminal_centers: list[float] = []
    terminal_probabilities: list[float] = []
    terminal_paired_means: list[float] = []
    decision_act_means: list[float] = []
    decision_act_centers: list[float] = []
    decision_act_probabilities: list[float] = []
    decision_act_paired_means: list[float] = []
    for position, (raw_action, action_index) in enumerate(zip(actions, legal_indices)):
        action = _object(raw_action, f"{label}.actions[{position}]")
        if set(action) != {
            "actionIndex",
            "actionFeatures",
            "meanUtility",
            "centeredUtility",
            "uncertainty",
            "softTargetProbability",
            "pairedBaselineAdvantage",
            "decisionActUtilityAggregate",
            "pairedDecisionActBaselineAdvantage",
        }:
            raise ValueError(f"{label}: v2 action fields mismatch")
        if action.get("actionIndex") != action_index:
            raise ValueError(f"{label}: action order mismatch")
        features = _finite_list(
            action.get("actionFeatures"),
            TAX_RETURN_ACTION_FEATURE_COUNT,
            f"{label}.actions[{position}].actionFeatures",
        )
        if not np.allclose(
            features,
            TAX_RETURN_ACTION_FEATURES[action_index],
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(f"{label}: action feature mismatch")
        terminal_mean = _number(
            action.get("meanUtility"),
            f"{label}.actions[{position}].meanUtility",
        )
        terminal_center = _number(
            action.get("centeredUtility"),
            f"{label}.actions[{position}].centeredUtility",
        )
        terminal_probability = _number(
            action.get("softTargetProbability"),
            f"{label}.actions[{position}].softTargetProbability",
        )
        if not 0 <= terminal_probability <= 1:
            raise ValueError(f"{label}: terminal softTargetProbability is invalid")
        _uncertainty(
            action.get("uncertainty"),
            label=f"{label}.actions[{position}].uncertainty",
            independent_world_count=effective_worlds,
        )
        terminal_advantage_stats = _stats(
            action.get("pairedBaselineAdvantage"),
            label=f"{label}.actions[{position}].pairedBaselineAdvantage",
            independent_world_count=effective_worlds,
        )
        terminal_paired_mean = terminal_advantage_stats[0]
        decision_act_aggregate = _object(
            action.get("decisionActUtilityAggregate"),
            f"{label}.actions[{position}].decisionActUtilityAggregate",
        )
        if set(decision_act_aggregate) != {
            "meanUtility",
            "centeredUtility",
            "uncertainty",
            "softTargetProbability",
        }:
            raise ValueError(
                f"{label}: decision-act utility aggregate fields mismatch"
            )
        decision_act_mean = _number(
            decision_act_aggregate["meanUtility"],
            f"{label}.actions[{position}].decisionActUtilityAggregate.meanUtility",
        )
        decision_act_center = _number(
            decision_act_aggregate["centeredUtility"],
            f"{label}.actions[{position}].decisionActUtilityAggregate.centeredUtility",
        )
        decision_act_probability = _number(
            decision_act_aggregate["softTargetProbability"],
            f"{label}.actions[{position}].decisionActUtilityAggregate.softTargetProbability",
        )
        if not -1 <= decision_act_mean <= 1:
            raise ValueError(f"{label}: decision-act mean utility is out of range")
        if not 0 <= decision_act_probability <= 1:
            raise ValueError(f"{label}: decision-act softTargetProbability is invalid")
        _uncertainty(
            decision_act_aggregate["uncertainty"],
            label=f"{label}.actions[{position}].decisionActUtilityAggregate.uncertainty",
            independent_world_count=effective_worlds,
        )
        mean, standard_deviation, standard_error = _stats(
            action.get("pairedDecisionActBaselineAdvantage"),
            label=f"{label}.actions[{position}].pairedDecisionActBaselineAdvantage",
            independent_world_count=effective_worlds,
        )
        if action_index == baseline_action and (
            mean != 0.0 or standard_deviation != 0.0 or standard_error != 0.0
        ):
            raise ValueError(f"{label}: baseline paired advantage must be exactly zero")
        if action_index == baseline_action and terminal_advantage_stats != (
            0.0,
            0.0,
            0.0,
        ):
            raise ValueError(
                f"{label}: baseline terminal paired advantage must be exactly zero"
            )
        target_advantages[action_index] = simulator_reward_advantage_to_chips(
            mean
        )
        terminal_means.append(terminal_mean)
        terminal_centers.append(terminal_center)
        terminal_probabilities.append(terminal_probability)
        terminal_paired_means.append(terminal_paired_mean)
        decision_act_means.append(decision_act_mean)
        decision_act_centers.append(decision_act_center)
        decision_act_probabilities.append(decision_act_probability)
        decision_act_paired_means.append(mean)
    baseline_position = legal_indices.index(baseline_action)
    terminal_average = sum(terminal_means) / len(terminal_means)
    decision_act_average = sum(decision_act_means) / len(decision_act_means)
    for position in range(len(legal_indices)):
        if not math.isclose(
            terminal_centers[position],
            terminal_means[position] - terminal_average,
            rel_tol=1.0e-7,
            abs_tol=1.0e-10,
        ) or not math.isclose(
            terminal_paired_means[position],
            terminal_means[position] - terminal_means[baseline_position],
            rel_tol=1.0e-7,
            abs_tol=1.0e-10,
        ):
            raise ValueError(f"{label}: terminal aggregate semantics mismatch")
        if not math.isclose(
            decision_act_centers[position],
            decision_act_means[position] - decision_act_average,
            rel_tol=1.0e-7,
            abs_tol=1.0e-10,
        ) or not math.isclose(
            decision_act_paired_means[position],
            decision_act_means[position] - decision_act_means[baseline_position],
            rel_tol=1.0e-7,
            abs_tol=1.0e-10,
        ):
            raise ValueError(f"{label}: decision-act aggregate semantics mismatch")
    for probabilities, centers, name in (
        (terminal_probabilities, terminal_centers, "terminal"),
        (decision_act_probabilities, decision_act_centers, "decision-act"),
    ):
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1.0e-7, abs_tol=1.0e-10):
            raise ValueError(f"{label}: {name} soft targets do not sum to one")
        scaled = [
            center / float(manifest_contract["policyTemperature"])
            for center in centers
        ]
        maximum = max(scaled)
        exponentials = [math.exp(item - maximum) for item in scaled]
        denominator = sum(exponentials)
        expected_probabilities = [item / denominator for item in exponentials]
        if not np.allclose(
            probabilities,
            expected_probabilities,
            rtol=1.0e-7,
            atol=1.0e-10,
        ):
            raise ValueError(f"{label}: {name} soft-target probabilities mismatch")
    expected_best = legal_indices[max(
        range(len(legal_indices)), key=lambda position: terminal_means[position]
    )]
    expected_decision_act_best = legal_indices[max(
        range(len(legal_indices)), key=lambda position: decision_act_means[position]
    )]
    if best_action != expected_best or best_decision_act_action != expected_decision_act_best:
        raise ValueError(f"{label}: best-action aggregate mismatch")
    if target_advantages[baseline_action] != 0.0:
        raise ValueError(f"{label}: transformed baseline advantage must be exactly zero")
    return {
        "sampleId": value.get("sampleId"),
        "groupKey": key,
        "observation": observation,
        "legalMask": legal_mask,
        "baselineActionIndex": baseline_action,
        "targetAdvantages": target_advantages,
        "returnCount": return_count,
        "worldCount": world_count,
        "continuationCount": continuation_count,
        "effectiveIndependentWorlds": effective_worlds,
        "rawContinuationEvaluations": raw_continuation_evaluations,
    }


def _empty_arrays() -> dict[str, list[object]]:
    return {
        "observations": [],
        "legalMasks": [],
        "baselineActions": [],
        "targetAdvantages": [],
        "sampleIds": [],
        "groupKeys": [],
    }


def _arrays(buffer: Mapping[str, list[object]]) -> TaxAdvantageArrays:
    count = len(buffer["sampleIds"])
    return TaxAdvantageArrays(
        observations=(
            np.stack(buffer["observations"]).astype(np.float32, copy=False)
            if count
            else np.empty((0, TAX_RETURN_OBSERVATION_FEATURE_COUNT), dtype=np.float32)
        ),
        legal_masks=(
            np.stack(buffer["legalMasks"]).astype(np.bool_, copy=False)
            if count
            else np.empty((0, TAX_RETURN_ACTION_COUNT), dtype=np.bool_)
        ),
        baseline_actions=np.asarray(buffer["baselineActions"], dtype=np.int64),
        target_advantages=(
            np.stack(buffer["targetAdvantages"]).astype(np.float32, copy=False)
            if count
            else np.empty((0, TAX_RETURN_ACTION_COUNT), dtype=np.float32)
        ),
        sample_ids=tuple(str(value) for value in buffer["sampleIds"]),
        group_keys=tuple(str(value) for value in buffer["groupKeys"]),
    )


def _append_v2(buffer: dict[str, list[object]], record: Mapping[str, object]) -> None:
    buffer["observations"].append(np.asarray(record["observation"], dtype=np.float32))
    buffer["legalMasks"].append(np.asarray(record["legalMask"], dtype=np.bool_))
    buffer["baselineActions"].append(int(record["baselineActionIndex"]))
    buffer["targetAdvantages"].append(record["targetAdvantages"])
    buffer["sampleIds"].append(str(record["sampleId"]))
    buffer["groupKeys"].append(str(record["groupKey"]))


def _load_v2(
    paths: Sequence[Path],
    *,
    validation_fraction: float,
    split_seed: int,
) -> TaxAdvantageDataset:
    buffers = {"train": _empty_arrays(), "validation": _empty_arrays()}
    exclusions = {
        "trainReturnCountTwo": 0,
        "validationReturnCountTwo": 0,
        "trainReturnCountOneExcluded": 0,
        "validationReturnCountOneExcluded": 0,
    }
    reports: list[Mapping[str, object]] = []
    seen_sample_ids: set[str] = set()
    seen_group_keys: set[str] = set()
    source_contract: dict[str, object] | None = None
    for path in paths:
        content_digest = hashlib.sha256()
        content_bytes = 0
        decisions = 0
        action_evaluations = 0
        summary: dict[str, object] | None = None
        manifest_contract: Mapping[str, object] | None = None
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                value = _parse_line(raw, path, line_number)
                if line_number == 1:
                    manifest_contract = _validate_v2_manifest(value, path)
                    candidate_contract = {
                        "sourceFormatVersions": [DETERMINIZATION_FORMAT_VERSION],
                        "groupSplitKey": "canonicalInformationStateKey",
                        "determinizationSchema": DETERMINIZATION_SCHEMA,
                        "worldCountPerInformationState": manifest_contract["worldCount"],
                        "continuationCountPerHiddenWorld": manifest_contract[
                            "continuationCount"
                        ],
                        "effectiveIndependentWorldsPerInformationState": (
                            manifest_contract["effectiveIndependentWorlds"]
                        ),
                        "rawContinuationEvaluationsPerInformationState": (
                            manifest_contract["rawContinuationEvaluations"]
                        ),
                        "standardErrorEstimable": manifest_contract[
                            "standardErrorEstimable"
                        ],
                        "determinizationAlgorithm": manifest_contract[
                            "determinizationAlgorithm"
                        ],
                        "determinizationAlgorithmVersion": manifest_contract[
                            "determinizationAlgorithmVersion"
                        ],
                        "determinizationAlgorithmContractSha256": (
                            manifest_contract[
                                "determinizationAlgorithmContractSha256"
                            ]
                        ),
                        "candidateSeedDerivation": manifest_contract[
                            "candidateSeedDerivation"
                        ],
                        "continuationSeedDerivation": manifest_contract[
                            "continuationSeedDerivation"
                        ],
                        "targetField": "actions[].pairedDecisionActBaselineAdvantage.mean",
                        "targetTransform": TARGET_TRANSFORM,
                        "stateWeighting": "one-per-information-state-independent-of-worldCount",
                    }
                    if source_contract is None:
                        source_contract = candidate_contract
                    elif source_contract != candidate_contract:
                        raise ValueError("v2 input files use different determinization contracts")
                    content_digest.update(raw)
                    content_bytes += len(raw)
                    continue
                if value.get("type") == "summary":
                    if summary is not None:
                        raise ValueError(f"{path}: duplicate summary")
                    summary = value
                    if stream.read(1):
                        raise ValueError(f"{path}: summary must be the final record")
                    break
                if summary is not None or manifest_contract is None:
                    raise ValueError(f"{path}: invalid record order")
                content_digest.update(raw)
                content_bytes += len(raw)
                decisions += 1
                action_evaluations += int(value.get("forcedActionEvaluations", 0))
                record = _validate_v2_tax_record(
                    value,
                    path=path,
                    line_number=line_number,
                    manifest_contract=manifest_contract,
                )
                if record is None:
                    continue
                sample_id = record["sampleId"]
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"{path}:{line_number}: sampleId is invalid")
                if sample_id in seen_sample_ids:
                    raise ValueError(f"duplicate v2 sampleId: {sample_id}")
                seen_sample_ids.add(sample_id)
                group_key = str(record["groupKey"])
                if group_key in seen_group_keys:
                    raise ValueError(
                        "v2 requires exactly one aggregate record per canonical information state"
                    )
                seen_group_keys.add(group_key)
                partition = (
                    "validation"
                    if deterministic_validation_membership(
                        group_key,
                        split_seed=split_seed,
                        validation_fraction=validation_fraction,
                    )
                    else "train"
                )
                if record["returnCount"] == 1:
                    exclusions[f"{partition}ReturnCountOneExcluded"] += 1
                else:
                    exclusions[f"{partition}ReturnCountTwo"] += 1
                    _append_v2(buffers[partition], record)
        if summary is None:
            raise ValueError(f"{path}: missing summary")
        hashes = _object(summary.get("hashes"), f"{path}: summary.hashes")
        if (
            hashes.get("algorithm") != "sha256"
            or hashes.get("contentBeforeSummary") != content_digest.hexdigest()
            or hashes.get("contentBeforeSummaryBytes") != content_bytes
        ):
            raise ValueError(f"{path}: content-before-summary binding mismatch")
        if summary.get("decisionsWritten") != decisions:
            raise ValueError(f"{path}: decision summary count mismatch")
        if summary.get("actionEvaluations") != action_evaluations:
            raise ValueError(f"{path}: action-evaluation summary mismatch")
        reports.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "formatVersion": DETERMINIZATION_FORMAT_VERSION,
            }
        )
        checksum_path = path.with_suffix(f"{path.suffix}.sha256")
        if not checksum_path.is_file():
            raise FileNotFoundError(f"{path}: v2 dataset checksum sidecar is missing")
        expected_sidecar = f"{file_sha256(path)}  {path.name}\n"
        if checksum_path.read_text(encoding="ascii") != expected_sidecar:
            raise ValueError(f"{path}: v2 dataset checksum sidecar mismatch")
    train = _arrays(buffers["train"])
    validation = _arrays(buffers["validation"])
    if len(train) < 1 or len(validation) < 1:
        raise ValueError("v2 tax advantage data requires train and validation states")
    if set(train.group_keys) & set(validation.group_keys):
        raise RuntimeError("v2 canonical-information-state leakage")
    assert source_contract is not None
    return TaxAdvantageDataset(
        train=train,
        validation=validation,
        exclusion_counts=exclusions,
        source_files=tuple(reports),
        source_contract=source_contract,
        group_split_key="canonicalInformationStateKey",
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )


def _subset_v1(arrays: DecisionArrays) -> tuple[TaxAdvantageArrays, int]:
    return_one = arrays.observations[:, 101] > 0.5
    return_two = arrays.observations[:, 102] > 0.5
    if not np.logical_xor(return_one, return_two).all():
        raise ValueError("v1 tax return-count features must be one-hot")
    included = np.flatnonzero(return_two)
    if included.size == 0:
        raise ValueError("v1 tax-return split contains no returnCount=2 states")
    observations = arrays.observations[included].astype(np.float32, copy=True)
    legal_masks = arrays.legal_masks[included].astype(np.bool_, copy=True)
    baseline_actions = arrays.baseline_actions[included].astype(np.int64, copy=True)
    utilities = arrays.decision_act_utilities[included].astype(np.float32, copy=True)
    baseline_utilities = utilities[np.arange(included.size), baseline_actions][:, None]
    targets = np.vectorize(
        simulator_reward_advantage_to_chips,
        otypes=[np.float32],
    )(utilities - baseline_utilities)
    targets[~legal_masks] = 0.0
    targets[np.arange(included.size), baseline_actions] = 0.0
    return (
        TaxAdvantageArrays(
            observations=observations,
            legal_masks=legal_masks,
            baseline_actions=baseline_actions,
            target_advantages=targets,
            sample_ids=tuple(arrays.sample_ids[index] for index in included),
            group_keys=tuple(arrays.world_keys[index] for index in included),
        ),
        int(return_one.sum()),
    )


def _load_v1(
    patterns: Sequence[str],
    *,
    validation_fraction: float,
    split_seed: int,
) -> TaxAdvantageDataset:
    datasets = load_non_card_counterfactuals(
        patterns,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    if datasets.tax_return is None:
        raise ValueError("v1 dataset contains no tax-return decisions")
    train, train_excluded = _subset_v1(datasets.tax_return.train)
    validation, validation_excluded = _subset_v1(datasets.tax_return.validation)
    return TaxAdvantageDataset(
        train=train,
        validation=validation,
        exclusion_counts={
            "trainReturnCountTwo": len(train),
            "validationReturnCountTwo": len(validation),
            "trainReturnCountOneExcluded": train_excluded,
            "validationReturnCountOneExcluded": validation_excluded,
        },
        source_files=tuple(
            {
                "path": report.path,
                "sha256": report.sha256,
                "bytes": report.bytes,
                "formatVersion": 1,
            }
            for report in datasets.files
        ),
        source_contract={
            "sourceFormatVersions": [1],
            "groupSplitKey": "canonicalWorldKey",
            "determinizationSchema": None,
            "worldCountPerInformationState": 1,
            "continuationCountPerHiddenWorld": 1,
            "effectiveIndependentWorldsPerInformationState": 1,
            "rawContinuationEvaluationsPerInformationState": 1,
            "standardErrorEstimable": False,
            "determinizationAlgorithm": None,
            "determinizationAlgorithmVersion": None,
            "determinizationAlgorithmContractSha256": None,
            "candidateSeedDerivation": None,
            "continuationSeedDerivation": None,
            "targetField": (
                "actions[].decisionActUtility-minus-baseline.decisionActUtility"
            ),
            "targetTransform": TARGET_TRANSFORM,
            "stateWeighting": "one-per-information-state-independent-of-worldCount",
        },
        group_split_key="canonicalWorldKey",
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )


def load_tax_return_advantage_dataset(
    patterns: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 20260801,
) -> TaxAdvantageDataset:
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    paths = expand_input_paths(patterns)
    versions: set[int] = set()
    for path in paths:
        with path.open("rb") as stream:
            raw = stream.readline()
        manifest = _parse_line(raw, path, 1)
        if manifest.get("format") != COUNTERFACTUAL_FORMAT:
            raise ValueError(f"unsupported counterfactual format: {path}")
        version = manifest.get("version")
        if version not in (1, DETERMINIZATION_FORMAT_VERSION):
            raise ValueError(f"unsupported counterfactual version: {version}")
        versions.add(int(version))
    if len(versions) != 1:
        raise ValueError("v1 and v2 tax advantage inputs cannot be mixed in one run")
    if versions == {1}:
        return _load_v1(
            patterns,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
    return _load_v2(
        paths,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
