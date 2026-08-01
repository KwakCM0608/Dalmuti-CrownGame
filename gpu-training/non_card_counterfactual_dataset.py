"""Strict streaming loader for DALMUTI non-card counterfactual data.

The collector writes one exhaustive legal-action decision per NDJSON record.
This module validates the complete wire contract before materializing compact
NumPy arrays. Train/validation membership is derived from a canonical hidden
world key ``(playerCount, acts, matchSeed, continuationPolicy)`` and a caller
supplied split seed. Every decision from one world therefore stays in one
partition even when separate collection files use different episode labels.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from non_card_action_conditioned import (
    NON_CARD_OBSERVATION_SCHEMA_VERSION,
    REVOLUTION_ACTION_CATALOGUE_VERSION,
    REVOLUTION_ACTION_COUNT,
    REVOLUTION_ACTION_FEATURE_COUNT,
    REVOLUTION_OBSERVATION_FEATURE_COUNT,
    TAX_RETURN_ACTION_CATALOGUE_VERSION,
    TAX_RETURN_ACTION_COUNT,
    TAX_RETURN_ACTION_FEATURE_COUNT,
    TAX_RETURN_ACTION_FEATURES,
    TAX_RETURN_ACTION_REQUIRED_COUNTS,
    TAX_RETURN_ACTION_SIZES,
    TAX_RETURN_OBSERVATION_FEATURE_COUNT,
)


COUNTERFACTUAL_FORMAT = "dalmuti-non-card-counterfactual-ndjson"
COUNTERFACTUAL_FORMAT_VERSION = 1
DECISION_KINDS = ("tax-return", "revolution")
ROLE_NAMES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
UINT32_MAX = 0xFFFF_FFFF


@dataclass(frozen=True)
class DatasetFileReport:
    path: str
    sha256: str
    bytes: int
    manifest: dict[str, object]
    decisions: int
    action_evaluations: int


@dataclass(frozen=True)
class DecisionArrays:
    decision: str
    observations: np.ndarray
    legal_masks: np.ndarray
    policy_targets: np.ndarray
    action_value_targets: np.ndarray
    decision_act_utilities: np.ndarray
    action_weights: np.ndarray
    value_targets: np.ndarray
    best_actions: np.ndarray
    baseline_actions: np.ndarray
    source_policy_temperatures: np.ndarray
    sample_weights: np.ndarray
    sample_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    world_keys: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class DecisionSplit:
    train: DecisionArrays
    validation: DecisionArrays


@dataclass(frozen=True)
class NonCardCounterfactualDatasets:
    tax_return: DecisionSplit | None
    revolution: DecisionSplit | None
    files: tuple[DatasetFileReport, ...]
    validation_fraction: float
    split_seed: int
    group_split_key: str = "canonicalWorldKey"


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


def _parse_json_line(raw: bytes, path: Path, line_number: int) -> dict[str, object]:
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path}:{line_number}: NDJSON line lacks a newline")
    try:
        text = raw[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}:{line_number}: invalid UTF-8") from error
    if not text or text.isspace():
        raise ValueError(f"{path}:{line_number}: blank NDJSON line")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _DuplicateKeyError, ValueError) as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    return value


def _keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} fields mismatch; missing={missing}, unexpected={unexpected}"
        )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: object, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return value


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if strictly_positive and result <= 0:
        raise ValueError(f"{label} must be greater than zero")
    return result


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _sha(value: object, label: str, *, prefixed: bool | None = None) -> str:
    result = _string(value, label)
    if not SHA256_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    if prefixed is True and not result.startswith("sha256:"):
        raise ValueError(f"{label} must use the sha256: prefix")
    if prefixed is False and result.startswith("sha256:"):
        raise ValueError(f"{label} must not use a prefix")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expand_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.update(path.resolve() for path in matches if path.is_file())
    result = sorted(paths, key=lambda path: str(path).lower())
    if not result:
        raise FileNotFoundError("no non-card counterfactual files matched")
    return result


def canonical_world_key(
    *,
    player_count: int,
    acts: int,
    match_seed: int,
    continuation_policy: str,
) -> str:
    """Return the stable, unambiguous split/dedup identity for one world."""

    _integer(player_count, "playerCount", minimum=4, maximum=10)
    _integer(acts, "acts", minimum=1)
    _integer(match_seed, "matchSeed", minimum=0, maximum=UINT32_MAX)
    _string(continuation_policy, "continuationPolicy")
    return json.dumps(
        [player_count, acts, match_seed, continuation_policy],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def deterministic_validation_membership(
    group_id: str,
    *,
    split_seed: int,
    validation_fraction: float,
) -> bool:
    """Map one baseline-match group ID to a stable partition."""

    _string(group_id, "canonicalWorldKey")
    _integer(split_seed, "split_seed", minimum=0)
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    digest = hashlib.sha256(f"{split_seed}\0{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < validation_fraction


def _validate_manifest(value: dict[str, object], path: Path) -> dict[str, object]:
    label = f"{path}: manifest"
    _keys(
        value,
        {
            "type",
            "format",
            "version",
            "createdAt",
            "observationSchemaVersion",
            "actionCatalogueVersions",
            "featureDimensions",
            "collection",
            "privacy",
        },
        label,
    )
    if value["type"] != "manifest" or value["format"] != COUNTERFACTUAL_FORMAT:
        raise ValueError(f"{label}: unsupported format")
    if value["version"] != COUNTERFACTUAL_FORMAT_VERSION:
        raise ValueError(f"{label}: unsupported format version")
    _string(value["createdAt"], f"{label}.createdAt")
    if value["observationSchemaVersion"] != NON_CARD_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(f"{label}: observation schema mismatch")

    catalogue = _object(value["actionCatalogueVersions"], f"{label}.actionCatalogueVersions")
    _keys(catalogue, {"taxReturn", "revolution"}, f"{label}.actionCatalogueVersions")
    if (
        catalogue["taxReturn"] != TAX_RETURN_ACTION_CATALOGUE_VERSION
        or catalogue["revolution"] != REVOLUTION_ACTION_CATALOGUE_VERSION
    ):
        raise ValueError(f"{label}: action catalogue mismatch")

    dimensions = _object(value["featureDimensions"], f"{label}.featureDimensions")
    _keys(dimensions, {"taxReturn", "revolution"}, f"{label}.featureDimensions")
    expected_dimensions = {
        "taxReturn": {
            "observation": TAX_RETURN_OBSERVATION_FEATURE_COUNT,
            "action": TAX_RETURN_ACTION_FEATURE_COUNT,
            "catalogue": TAX_RETURN_ACTION_COUNT,
        },
        "revolution": {
            "observation": REVOLUTION_OBSERVATION_FEATURE_COUNT,
            "action": REVOLUTION_ACTION_FEATURE_COUNT,
            "catalogue": REVOLUTION_ACTION_COUNT,
        },
    }
    if dimensions != expected_dimensions:
        raise ValueError(f"{label}: feature dimensions mismatch")

    collection = _object(value["collection"], f"{label}.collection")
    _keys(
        collection,
        {
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
        },
        f"{label}.collection",
    )
    player_counts = _list(collection["playerCounts"], f"{label}.collection.playerCounts")
    parsed_counts = [
        _integer(item, f"{label}.collection.playerCounts", minimum=4, maximum=10)
        for item in player_counts
    ]
    if not parsed_counts or parsed_counts != sorted(set(parsed_counts)):
        raise ValueError(f"{label}: playerCounts must be non-empty, unique, and ascending")
    _integer(collection["episodesPerPlayerCount"], f"{label}.collection.episodes", minimum=1)
    _integer(collection["acts"], f"{label}.collection.acts", minimum=1)
    _integer(collection["initialSeed"], f"{label}.collection.initialSeed", minimum=0, maximum=UINT32_MAX)
    if collection["matchSeedDerivation"] != (
        "initialSeed + zero-based index over ascending playerCount then episode"
    ):
        raise ValueError(f"{label}: unsupported match seed derivation")
    decision_kinds = _list(collection["decisionKinds"], f"{label}.collection.decisionKinds")
    if (
        not decision_kinds
        or len(set(decision_kinds)) != len(decision_kinds)
        or any(kind not in DECISION_KINDS for kind in decision_kinds)
    ):
        raise ValueError(f"{label}: invalid decisionKinds")
    _number(collection["policyTemperature"], f"{label}.collection.policyTemperature", strictly_positive=True)
    max_decisions = collection["maxDecisions"]
    if max_decisions is not None:
        _integer(max_decisions, f"{label}.collection.maxDecisions", minimum=1)
    if collection["baselineNonCardHooks"] != {}:
        raise ValueError(f"{label}: baseline non-card hooks must be empty")
    if collection["continuationPolicy"] != "normal-deterministic":
        raise ValueError(f"{label}: unsupported continuation policy")
    if collection["resumeAllowed"] is not False:
        raise ValueError(f"{label}: resumable data is unsupported")

    privacy = _object(value["privacy"], f"{label}.privacy")
    _keys(
        privacy,
        {
            "observation",
            "opponentCardIdentitiesIncluded",
            "physicalCardIdsIncluded",
        },
        f"{label}.privacy",
    )
    if privacy != {
        "observation": "encoded-actor-hand-and-public-state-only",
        "opponentCardIdentitiesIncluded": False,
        "physicalCardIdsIncluded": False,
    }:
        raise ValueError(f"{label}: privacy contract mismatch")
    return value


def _float_list(value: object, length: int, label: str) -> list[float]:
    items = _list(value, label)
    if len(items) != length:
        raise ValueError(f"{label} must contain {length} values")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(items)]


def _derived_legal_mask(decision: str, observation: list[float]) -> list[bool]:
    role_values = observation[3:8]
    rounded_roles = [round(value) for value in role_values]
    if (
        any(abs(value - rounded) > 1.0e-4 for value, rounded in zip(role_values, rounded_roles))
        or sum(rounded_roles) != 1
        or any(value not in (0, 1) for value in rounded_roles)
    ):
        raise ValueError("actor-role observation features must be one-hot")
    role_index = rounded_roles.index(1)
    if decision == "revolution":
        return [True, True]

    return_values = observation[101:103]
    if all(abs(value - expected) <= 1.0e-4 for value, expected in zip(return_values, (1.0, 0.0))):
        return_count = 1
    elif all(abs(value - expected) <= 1.0e-4 for value, expected in zip(return_values, (0.0, 1.0))):
        return_count = 2
    else:
        raise ValueError("tax return-count observation features must be one-hot")
    expected_return_count = 2 if role_index == 0 else 1 if role_index == 1 else 0
    if return_count != expected_return_count:
        raise ValueError("tax return count does not match actor role")

    deck_counts = (*range(1, 13), 2)
    hand_counts: list[int] = []
    for rank_index, (normalized, copies) in enumerate(zip(observation[8:21], deck_counts)):
        scaled = normalized * copies
        rounded = round(scaled)
        if abs(scaled - rounded) > 1.0e-4 or rounded < 0 or rounded > copies:
            raise ValueError(f"tax hand count for rank {rank_index + 1} is invalid")
        hand_counts.append(rounded)
    mask: list[bool] = []
    for size, required in zip(TAX_RETURN_ACTION_SIZES, TAX_RETURN_ACTION_REQUIRED_COUNTS):
        mask.append(
            size == return_count
            and all(needed <= available for needed, available in zip(required, hand_counts))
        )
    if not any(mask):
        raise ValueError("tax observation produces no legal action")
    return mask


def _roles_for_player_count(player_count: int) -> list[str]:
    if player_count == 4:
        return [
            "great-dalmuti",
            "lesser-dalmuti",
            "lesser-peon",
            "great-peon",
        ]
    return [
        "great-dalmuti",
        "lesser-dalmuti",
        *(["merchant"] * (player_count - 4)),
        "lesser-peon",
        "great-peon",
    ]


def _validate_common_observation_semantics(
    observation: list[float],
    *,
    decision: str,
    player_count: int,
    round_number: int,
    actor_hand_count: int,
    actor_role: str,
    actor_seat: int,
) -> None:
    expected_global = (
        (player_count - 4) / 6,
        (round_number - 1) / 19,
        actor_hand_count / 20,
    )
    if not np.allclose(observation[:3], expected_global, rtol=0.0, atol=1.0e-7):
        raise ValueError("global observation features do not match record metadata")
    role_values = observation[3:8]
    expected_actor_role = [1.0 if role == actor_role else 0.0 for role in ROLE_NAMES]
    if not np.allclose(role_values, expected_actor_role, rtol=0.0, atol=1.0e-7):
        raise ValueError("actorRole does not match encoded observation")

    deck_counts = (*range(1, 13), 2)
    physical_counts: list[int] = []
    for normalized, copies in zip(observation[8:21], deck_counts):
        scaled = normalized * copies
        rounded = round(scaled)
        if abs(scaled - rounded) > 1.0e-4 or rounded < 0 or rounded > copies:
            raise ValueError("own-hand observation contains a nonphysical rank count")
        physical_counts.append(rounded)
    if sum(physical_counts) != actor_hand_count:
        raise ValueError("own-hand rank counts do not match actorHandCount")
    if decision == "revolution" and physical_counts[12] != 2:
        raise ValueError("revolution observation must contain both jokers")

    rank_roles = _roles_for_player_count(player_count)
    relative_roles = [
        rank_roles[(actor_seat + offset) % player_count]
        for offset in range(player_count)
    ]
    for slot in range(10):
        offset = 21 + slot * 8
        features = observation[offset : offset + 8]
        if slot >= player_count:
            if any(abs(value) > 1.0e-7 for value in features):
                raise ValueError("unused public-player observation slot is nonzero")
            continue
        if abs(features[0] - 1.0) > 1.0e-7:
            raise ValueError("occupied public-player slot lacks its marker")
        if not 0.0 <= features[1] <= 1.0 or not -1.0 < features[2] < 1.0:
            raise ValueError("public hand-count or score feature is out of range")
        expected_role = [
            1.0 if role == relative_roles[slot] else 0.0
            for role in ROLE_NAMES
        ]
        if not np.allclose(features[3:8], expected_role, rtol=0.0, atol=1.0e-7):
            raise ValueError("public-player role features do not match rank order")
        if slot == 0 and abs(features[1] - actor_hand_count / 20) > 1.0e-7:
            raise ValueError("actor public hand count does not match private hand")
    if decision == "revolution":
        expected_taxation_flag = 1.0 if round_number > 1 else 0.0
        if abs(observation[101] - expected_taxation_flag) > 1.0e-7:
            raise ValueError("revolution taxation flag does not match the act")


def _validate_record(
    value: dict[str, object],
    manifest: Mapping[str, object],
    path: Path,
    line_number: int,
) -> dict[str, object]:
    label = f"{path}:{line_number}"
    _keys(
        value,
        {
            "type",
            "sampleId",
            "decision",
            "episodeId",
            "matchSeed",
            "playerCount",
            "acts",
            "round",
            "actorId",
            "actorSeat",
            "actorRole",
            "decisionKey",
            "observationSchemaVersion",
            "actionCatalogueVersion",
            "observation",
            "legalMask",
            "legalActionIndices",
            "baselineActionIndex",
            "metadata",
            "pairing",
            "utility",
            "targetBuilder",
            "targetSampleCount",
            "bestActionIndex",
            "actions",
        },
        label,
    )
    if value["type"] != "counterfactual-decision":
        raise ValueError(f"{label}: unsupported record type")
    decision = _string(value["decision"], f"{label}.decision")
    if decision not in DECISION_KINDS:
        raise ValueError(f"{label}: unsupported decision")
    collection = _object(manifest["collection"], f"{label}.manifest.collection")
    if decision not in collection["decisionKinds"]:
        raise ValueError(f"{label}: decision was not requested by manifest")
    sample_id = _string(value["sampleId"], f"{label}.sampleId")
    episode_id = _string(value["episodeId"], f"{label}.episodeId")
    match_seed = _integer(value["matchSeed"], f"{label}.matchSeed", minimum=0, maximum=UINT32_MAX)
    player_count = _integer(value["playerCount"], f"{label}.playerCount", minimum=4, maximum=10)
    if player_count not in collection["playerCounts"]:
        raise ValueError(f"{label}: player count is absent from manifest")
    acts = _integer(value["acts"], f"{label}.acts", minimum=1)
    if acts != collection["acts"]:
        raise ValueError(f"{label}: acts mismatch")
    round_number = _integer(value["round"], f"{label}.round", minimum=1, maximum=acts)
    actor_id = _string(value["actorId"], f"{label}.actorId")
    actor_seat = _integer(value["actorSeat"], f"{label}.actorSeat", minimum=0, maximum=player_count - 1)
    actor_role = _string(value["actorRole"], f"{label}.actorRole")
    if actor_role not in ROLE_NAMES:
        raise ValueError(f"{label}: invalid actor role")
    _string(value["decisionKey"], f"{label}.decisionKey")
    if value["observationSchemaVersion"] != NON_CARD_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(f"{label}: observation schema mismatch")
    expected_catalogue_version = (
        TAX_RETURN_ACTION_CATALOGUE_VERSION
        if decision == "tax-return"
        else REVOLUTION_ACTION_CATALOGUE_VERSION
    )
    if value["actionCatalogueVersion"] != expected_catalogue_version:
        raise ValueError(f"{label}: action catalogue mismatch")

    observation_count = (
        TAX_RETURN_OBSERVATION_FEATURE_COUNT
        if decision == "tax-return"
        else REVOLUTION_OBSERVATION_FEATURE_COUNT
    )
    action_count = TAX_RETURN_ACTION_COUNT if decision == "tax-return" else REVOLUTION_ACTION_COUNT
    action_feature_count = (
        TAX_RETURN_ACTION_FEATURE_COUNT
        if decision == "tax-return"
        else REVOLUTION_ACTION_FEATURE_COUNT
    )
    observation = _float_list(value["observation"], observation_count, f"{label}.observation")
    legal_mask_raw = _list(value["legalMask"], f"{label}.legalMask")
    if len(legal_mask_raw) != action_count or any(not isinstance(item, bool) for item in legal_mask_raw):
        raise ValueError(f"{label}: legalMask is invalid")
    legal_mask = [bool(item) for item in legal_mask_raw]
    derived_mask = _derived_legal_mask(decision, observation)
    if legal_mask != derived_mask:
        raise ValueError(f"{label}: legalMask does not match encoded observation")
    legal_indices_raw = _list(value["legalActionIndices"], f"{label}.legalActionIndices")
    legal_indices = [
        _integer(item, f"{label}.legalActionIndices", minimum=0, maximum=action_count - 1)
        for item in legal_indices_raw
    ]
    expected_indices = [index for index, legal in enumerate(legal_mask) if legal]
    if legal_indices != expected_indices:
        raise ValueError(f"{label}: legalActionIndices do not exactly match legalMask")
    baseline_action = _integer(value["baselineActionIndex"], f"{label}.baselineActionIndex")
    best_action = _integer(value["bestActionIndex"], f"{label}.bestActionIndex")
    if baseline_action not in legal_indices or best_action not in legal_indices:
        raise ValueError(f"{label}: selected action is illegal")

    metadata = _object(value["metadata"], f"{label}.metadata")
    expected_metadata_keys = {"playerCount", "actorHandCount"}
    if decision == "tax-return":
        expected_metadata_keys.add("returnCount")
    _keys(metadata, expected_metadata_keys, f"{label}.metadata")
    if metadata["playerCount"] != player_count:
        raise ValueError(f"{label}: metadata player count mismatch")
    actor_hand_count = _integer(metadata["actorHandCount"], f"{label}.metadata.actorHandCount", minimum=1, maximum=20)
    if decision == "tax-return":
        return_count = _integer(metadata["returnCount"], f"{label}.metadata.returnCount", minimum=1, maximum=2)
        encoded_return_count = 1 if observation[101] > 0.5 else 2
        if return_count != encoded_return_count:
            raise ValueError(f"{label}: metadata return count mismatch")
    _validate_common_observation_semantics(
        observation,
        decision=decision,
        player_count=player_count,
        round_number=round_number,
        actor_hand_count=actor_hand_count,
        actor_role=actor_role,
        actor_seat=actor_seat,
    )

    pairing = _object(value["pairing"], f"{label}.pairing")
    _keys(
        pairing,
        {
            "pairedWorldId",
            "preDecisionSha256",
            "continuationPolicy",
            "forcedOverrideNamespace",
            "rootActionCoverage",
        },
        f"{label}.pairing",
    )
    paired_world_id = _sha(pairing["pairedWorldId"], f"{label}.pairing.pairedWorldId", prefixed=True)
    predecision_sha = _sha(pairing["preDecisionSha256"], f"{label}.pairing.preDecisionSha256", prefixed=False)
    if sample_id != f"{decision}:{predecision_sha}":
        raise ValueError(f"{label}: sampleId does not bind the pre-decision hash")
    if pairing["continuationPolicy"] != "normal-deterministic":
        raise ValueError(f"{label}: continuation policy mismatch")
    world_key = canonical_world_key(
        player_count=player_count,
        acts=acts,
        match_seed=match_seed,
        continuation_policy=str(pairing["continuationPolicy"]),
    )
    expected_namespace = "taxReturn" if decision == "tax-return" else "revolution"
    if pairing["forcedOverrideNamespace"] != expected_namespace:
        raise ValueError(f"{label}: forced override namespace mismatch")
    if pairing["rootActionCoverage"] != "all-legal-actions-exactly-once":
        raise ValueError(f"{label}: root action coverage mismatch")

    utility = _object(value["utility"], f"{label}.utility")
    _keys(utility, {"definition", "centeredAcrossLegalActions"}, f"{label}.utility")
    if utility != {
        "definition": "terminal-cumulative-chip-score",
        "centeredAcrossLegalActions": True,
    }:
        raise ValueError(f"{label}: utility contract mismatch")
    if value["targetBuilder"] != "training/non-card-search-targets.ts#buildPairedCounterfactualTargets":
        raise ValueError(f"{label}: target builder mismatch")
    sample_count = _integer(value["targetSampleCount"], f"{label}.targetSampleCount", minimum=1)

    actions_raw = _list(value["actions"], f"{label}.actions")
    if len(actions_raw) != len(legal_indices):
        raise ValueError(f"{label}: actions must cover each legal action once")
    parsed_actions: list[dict[str, object]] = []
    for position, (action_raw, expected_index) in enumerate(zip(actions_raw, legal_indices)):
        action = _object(action_raw, f"{label}.actions[{position}]")
        _keys(
            action,
            {
                "actionIndex",
                "actionFeatures",
                "pairedWorldId",
                "terminalActorUtility",
                "decisionActUtility",
                "terminalFinishPlaceInDecisionAct",
                "meanUtility",
                "centeredUtility",
                "uncertainty",
                "softTargetProbability",
            },
            f"{label}.actions[{position}]",
        )
        if action["actionIndex"] != expected_index:
            raise ValueError(f"{label}: actions are not in legal catalogue order")
        features = _float_list(
            action["actionFeatures"],
            action_feature_count,
            f"{label}.actions[{position}].actionFeatures",
        )
        if decision == "tax-return":
            expected_features = list(TAX_RETURN_ACTION_FEATURES[expected_index])
        else:
            expected_features = [1.0, 0.0, 0.0] if expected_index == 0 else (
                [0.0, 0.0, 1.0] if actor_role == "great-peon" else [0.0, 1.0, 0.0]
            )
        if not np.allclose(features, expected_features, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{label}: action features mismatch for action {expected_index}")
        if action["pairedWorldId"] != paired_world_id:
            raise ValueError(f"{label}: action paired world mismatch")
        terminal_utility = _number(action["terminalActorUtility"], f"{label}.actions[{position}].terminalActorUtility")
        decision_act_utility = _number(
            action["decisionActUtility"],
            f"{label}.actions[{position}].decisionActUtility",
        )
        finish_place = _integer(
            action["terminalFinishPlaceInDecisionAct"],
            f"{label}.actions[{position}].terminalFinishPlaceInDecisionAct",
            minimum=1,
            maximum=player_count,
        )
        if finish_place == 1:
            round_chip_award = 4
        elif finish_place == 2:
            round_chip_award = 3
        elif finish_place == player_count - 1:
            round_chip_award = 1
        elif finish_place == player_count:
            round_chip_award = 0
        else:
            round_chip_award = 2
        expected_decision_act_utility = (round_chip_award - 2) / 2
        if not math.isclose(
            decision_act_utility,
            expected_decision_act_utility,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{label}: decisionActUtility does not match "
                "roundChipAward(finishPlace, playerCount)"
            )
        mean_utility = _number(action["meanUtility"], f"{label}.actions[{position}].meanUtility")
        centered_utility = _number(action["centeredUtility"], f"{label}.actions[{position}].centeredUtility")
        if sample_count == 1 and not math.isclose(mean_utility, terminal_utility, abs_tol=1.0e-9):
            raise ValueError(f"{label}: one-world mean utility mismatch")
        uncertainty = _object(action["uncertainty"], f"{label}.actions[{position}].uncertainty")
        _keys(uncertainty, {"sampleStandardDeviation", "standardError"}, f"{label}.actions[{position}].uncertainty")
        standard_deviation = _number(
            uncertainty["sampleStandardDeviation"],
            f"{label}.actions[{position}].uncertainty.sampleStandardDeviation",
            minimum=0.0,
        )
        standard_error = _number(
            uncertainty["standardError"],
            f"{label}.actions[{position}].uncertainty.standardError",
            minimum=0.0,
        )
        if not math.isclose(standard_error, standard_deviation / math.sqrt(sample_count), rel_tol=1.0e-7, abs_tol=1.0e-9):
            raise ValueError(f"{label}: uncertainty is internally inconsistent")
        probability = _number(
            action["softTargetProbability"],
            f"{label}.actions[{position}].softTargetProbability",
            minimum=0.0,
        )
        parsed_actions.append(
            {
                "actionIndex": expected_index,
                "meanUtility": mean_utility,
                "centeredUtility": centered_utility,
                "decisionActUtility": decision_act_utility,
                "standardError": standard_error,
                "softTargetProbability": probability,
            }
        )

    means = [float(action["meanUtility"]) for action in parsed_actions]
    center = sum(means) / len(means)
    for action in parsed_actions:
        if not math.isclose(float(action["centeredUtility"]), float(action["meanUtility"]) - center, rel_tol=1.0e-7, abs_tol=1.0e-8):
            raise ValueError(f"{label}: centered utility mismatch")
    probabilities = [float(action["softTargetProbability"]) for action in parsed_actions]
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1.0e-7, abs_tol=1.0e-8):
        raise ValueError(f"{label}: soft policy target does not sum to one")
    temperature = float(collection["policyTemperature"])
    maximum = max(means)
    exponentials = [math.exp((mean - maximum) / temperature) for mean in means]
    total = sum(exponentials)
    expected_probabilities = [value / total for value in exponentials]
    if not np.allclose(probabilities, expected_probabilities, rtol=1.0e-7, atol=1.0e-9):
        raise ValueError(f"{label}: soft policy target does not match utilities and temperature")
    best_position = max(range(len(means)), key=lambda index: means[index])
    if best_action != legal_indices[best_position]:
        raise ValueError(f"{label}: best action is inconsistent with mean utilities")

    # Attach validated, derived training values without changing the source object.
    return {
        "sampleId": sample_id,
        "decision": decision,
        "observation": observation,
        "legalMask": legal_mask,
        "bestActionIndex": best_action,
        "baselineActionIndex": baseline_action,
        "targetSampleCount": sample_count,
        "actions": parsed_actions,
        "episodeId": episode_id,
        "matchSeed": match_seed,
        "playerCount": player_count,
        "acts": acts,
        "actorId": actor_id,
        "worldKey": world_key,
        "sourcePolicyTemperature": float(
            collection["policyTemperature"]
        ),
    }


def _validate_summary(
    value: dict[str, object],
    *,
    path: Path,
    content_digest: str,
    content_bytes: int,
    decisions: int,
    action_evaluations: int,
    by_decision: Mapping[str, tuple[int, int]],
    by_player: Mapping[int, tuple[int, int]],
    expected_player_counts: Sequence[int],
    maximum_baseline_matches: int,
) -> None:
    label = f"{path}: summary"
    _keys(
        value,
        {
            "type",
            "baselineMatches",
            "decisionsDiscovered",
            "decisionsWritten",
            "actionEvaluations",
            "stoppedAtMaxDecisions",
            "counts",
            "hashes",
        },
        label,
    )
    if value["type"] != "summary":
        raise ValueError(f"{label}: final record is not a summary")
    baseline_matches = _integer(value["baselineMatches"], f"{label}.baselineMatches", minimum=0)
    if baseline_matches > maximum_baseline_matches:
        raise ValueError(f"{label}: baseline match count exceeds the manifest plan")
    discovered = _integer(value["decisionsDiscovered"], f"{label}.decisionsDiscovered", minimum=0)
    if discovered < decisions:
        raise ValueError(f"{label}: discovered decisions are fewer than written decisions")
    if value["decisionsWritten"] != decisions or value["actionEvaluations"] != action_evaluations:
        raise ValueError(f"{label}: top-level record counts mismatch")
    _bool(value["stoppedAtMaxDecisions"], f"{label}.stoppedAtMaxDecisions")
    hashes = _object(value["hashes"], f"{label}.hashes")
    _keys(hashes, {"algorithm", "contentBeforeSummary", "contentBeforeSummaryBytes", "scope"}, f"{label}.hashes")
    if hashes["algorithm"] != "sha256" or hashes["scope"] != (
        "UTF-8 NDJSON bytes for manifest and decision records, including newlines"
    ):
        raise ValueError(f"{label}: unsupported hash contract")
    if _sha(hashes["contentBeforeSummary"], f"{label}.hashes.contentBeforeSummary", prefixed=False) != content_digest:
        raise ValueError(f"{label}: content SHA-256 mismatch")
    if hashes["contentBeforeSummaryBytes"] != content_bytes:
        raise ValueError(f"{label}: content byte count mismatch")

    counts = _object(value["counts"], f"{label}.counts")
    _keys(counts, {"byDecision", "byPlayerCount"}, f"{label}.counts")
    by_decision_value = _object(counts["byDecision"], f"{label}.counts.byDecision")
    _keys(by_decision_value, set(DECISION_KINDS), f"{label}.counts.byDecision")
    discovered_sum = 0
    for decision in DECISION_KINDS:
        entry = _object(by_decision_value[decision], f"{label}.counts.byDecision.{decision}")
        _keys(entry, {"discovered", "written", "actionEvaluations"}, f"{label}.counts.byDecision.{decision}")
        discovered_sum += _integer(entry["discovered"], f"{label}.{decision}.discovered", minimum=0)
        expected_written, expected_actions = by_decision.get(decision, (0, 0))
        if entry["written"] != expected_written or entry["actionEvaluations"] != expected_actions:
            raise ValueError(f"{label}: {decision} counts mismatch")
    if discovered_sum != discovered:
        raise ValueError(f"{label}: discovered decision counts mismatch")
    by_player_value = _object(counts["byPlayerCount"], f"{label}.counts.byPlayerCount")
    if any(not isinstance(key, str) for key in by_player_value):
        raise ValueError(f"{label}: player count keys must be strings")
    if set(by_player_value) != {str(value) for value in expected_player_counts}:
        raise ValueError(f"{label}: player count entries do not match the manifest")
    baseline_sum = 0
    written_sum = 0
    action_sum = 0
    for key, entry_raw in by_player_value.items():
        try:
            player_count = int(key)
        except ValueError as error:
            raise ValueError(f"{label}: invalid player count key {key}") from error
        if str(player_count) != key or player_count < 4 or player_count > 10:
            raise ValueError(f"{label}: invalid player count key {key}")
        entry = _object(entry_raw, f"{label}.counts.byPlayerCount.{key}")
        _keys(entry, {"baselineMatches", "decisionsWritten", "actionEvaluations"}, f"{label}.counts.byPlayerCount.{key}")
        baseline = _integer(entry["baselineMatches"], f"{label}.{key}.baselineMatches", minimum=0)
        expected_written, expected_actions = by_player.get(player_count, (0, 0))
        if entry["decisionsWritten"] != expected_written or entry["actionEvaluations"] != expected_actions:
            raise ValueError(f"{label}: player {key} counts mismatch")
        baseline_sum += baseline
        written_sum += expected_written
        action_sum += expected_actions
    if baseline_sum != baseline_matches or written_sum != decisions or action_sum != action_evaluations:
        raise ValueError(f"{label}: aggregate counts mismatch")


def _walk_file(
    path: Path,
    visitor: Callable[[dict[str, object], Path], None] | None,
) -> DatasetFileReport:
    content_hash = hashlib.sha256()
    content_bytes = 0
    full_hash = hashlib.sha256()
    manifest: dict[str, object] | None = None
    summary_seen = False
    decisions = 0
    action_evaluations = 0
    by_decision: dict[str, list[int]] = {kind: [0, 0] for kind in DECISION_KINDS}
    by_player: dict[int, list[int]] = {}
    file_bytes = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            full_hash.update(raw)
            file_bytes += len(raw)
            record = _parse_json_line(raw, path, line_number)
            if line_number == 1:
                manifest = _validate_manifest(record, path)
                content_hash.update(raw)
                content_bytes += len(raw)
                continue
            if manifest is None:
                raise RuntimeError("manifest state was not initialized")
            if summary_seen:
                raise ValueError(f"{path}:{line_number}: record appears after summary")
            if record.get("type") == "summary":
                _validate_summary(
                    record,
                    path=path,
                    content_digest=content_hash.hexdigest(),
                    content_bytes=content_bytes,
                    decisions=decisions,
                    action_evaluations=action_evaluations,
                    by_decision={key: tuple(value) for key, value in by_decision.items()},
                    by_player={key: tuple(value) for key, value in by_player.items()},
                    expected_player_counts=manifest["collection"]["playerCounts"],
                    maximum_baseline_matches=(
                        len(manifest["collection"]["playerCounts"])
                        * manifest["collection"]["episodesPerPlayerCount"]
                    ),
                )
                summary_seen = True
                continue
            validated = _validate_record(record, manifest, path, line_number)
            content_hash.update(raw)
            content_bytes += len(raw)
            decisions += 1
            evaluated = len(validated["actions"])
            action_evaluations += evaluated
            decision = str(validated["decision"])
            by_decision[decision][0] += 1
            by_decision[decision][1] += evaluated
            player_count = int(validated["playerCount"])
            player_entry = by_player.setdefault(player_count, [0, 0])
            player_entry[0] += 1
            player_entry[1] += evaluated
            if visitor is not None:
                visitor(validated, path)
    if manifest is None:
        raise ValueError(f"{path}: empty dataset")
    if not summary_seen:
        raise ValueError(f"{path}: dataset is incomplete; final summary is missing")
    return DatasetFileReport(
        path=str(path),
        sha256=full_hash.hexdigest(),
        bytes=file_bytes,
        manifest=manifest,
        decisions=decisions,
        action_evaluations=action_evaluations,
    )


def validate_non_card_counterfactual_files(patterns: Sequence[str]) -> tuple[DatasetFileReport, ...]:
    return tuple(_walk_file(path, None) for path in expand_input_paths(patterns))


def _peek_manifest(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        raw = stream.readline()
    if not raw:
        raise ValueError(f"{path}: empty dataset")
    return _validate_manifest(_parse_json_line(raw, path, 1), path)


def _empty_buffers() -> dict[str, list[object]]:
    return {
        "observations": [],
        "legal_masks": [],
        "policy_targets": [],
        "action_value_targets": [],
        "decision_act_utilities": [],
        "action_weights": [],
        "value_targets": [],
        "best_actions": [],
        "baseline_actions": [],
        "source_policy_temperatures": [],
        "sample_weights": [],
        "sample_ids": [],
        "episode_ids": [],
        "world_keys": [],
    }


def _append_record(buffer: dict[str, list[object]], record: Mapping[str, object]) -> None:
    decision = str(record["decision"])
    action_count = TAX_RETURN_ACTION_COUNT if decision == "tax-return" else REVOLUTION_ACTION_COUNT
    policy = np.zeros(action_count, dtype=np.float32)
    action_values = np.zeros(action_count, dtype=np.float32)
    decision_act_utilities = np.zeros(action_count, dtype=np.float32)
    action_weights = np.zeros(action_count, dtype=np.float32)
    mean_utilities = np.zeros(action_count, dtype=np.float64)
    for action in record["actions"]:
        index = int(action["actionIndex"])
        policy[index] = float(action["softTargetProbability"])
        action_values[index] = float(action["centeredUtility"])
        decision_act_utilities[index] = float(
            action["decisionActUtility"]
        )
        standard_error = float(action["standardError"])
        action_weights[index] = 1.0 / (1.0 + standard_error * standard_error)
        mean_utilities[index] = float(action["meanUtility"])
    legal_mask = np.asarray(record["legalMask"], dtype=np.bool_)
    value_target = float(np.sum(policy.astype(np.float64) * mean_utilities))
    sample_count = int(record["targetSampleCount"])
    # Repeated paired worlds deserve more weight, but sqrt prevents a large
    # aggregate from completely dominating distinct states.
    sample_weight = math.sqrt(sample_count)
    buffer["observations"].append(np.asarray(record["observation"], dtype=np.float32))
    buffer["legal_masks"].append(legal_mask)
    buffer["policy_targets"].append(policy)
    buffer["action_value_targets"].append(action_values)
    buffer["decision_act_utilities"].append(decision_act_utilities)
    buffer["action_weights"].append(action_weights)
    buffer["value_targets"].append(value_target)
    buffer["best_actions"].append(int(record["bestActionIndex"]))
    buffer["baseline_actions"].append(int(record["baselineActionIndex"]))
    buffer["source_policy_temperatures"].append(
        float(record["sourcePolicyTemperature"])
    )
    buffer["sample_weights"].append(sample_weight)
    buffer["sample_ids"].append(str(record["sampleId"]))
    buffer["episode_ids"].append(str(record["episodeId"]))
    buffer["world_keys"].append(str(record["worldKey"]))


def _arrays(decision: str, buffer: dict[str, list[object]]) -> DecisionArrays:
    observation_count = TAX_RETURN_OBSERVATION_FEATURE_COUNT if decision == "tax-return" else REVOLUTION_OBSERVATION_FEATURE_COUNT
    action_count = TAX_RETURN_ACTION_COUNT if decision == "tax-return" else REVOLUTION_ACTION_COUNT
    count = len(buffer["sample_ids"])
    if count:
        observations = np.stack(buffer["observations"]).astype(np.float32, copy=False)
        legal_masks = np.stack(buffer["legal_masks"]).astype(np.bool_, copy=False)
        policy_targets = np.stack(buffer["policy_targets"]).astype(np.float32, copy=False)
        action_value_targets = np.stack(buffer["action_value_targets"]).astype(np.float32, copy=False)
        decision_act_utilities = np.stack(
            buffer["decision_act_utilities"]
        ).astype(np.float32, copy=False)
        action_weights = np.stack(buffer["action_weights"]).astype(np.float32, copy=False)
    else:
        observations = np.empty((0, observation_count), dtype=np.float32)
        legal_masks = np.empty((0, action_count), dtype=np.bool_)
        policy_targets = np.empty((0, action_count), dtype=np.float32)
        action_value_targets = np.empty((0, action_count), dtype=np.float32)
        decision_act_utilities = np.empty(
            (0, action_count), dtype=np.float32
        )
        action_weights = np.empty((0, action_count), dtype=np.float32)
    return DecisionArrays(
        decision=decision,
        observations=observations,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        action_value_targets=action_value_targets,
        decision_act_utilities=decision_act_utilities,
        action_weights=action_weights,
        value_targets=np.asarray(buffer["value_targets"], dtype=np.float32),
        best_actions=np.asarray(buffer["best_actions"], dtype=np.int64),
        baseline_actions=np.asarray(
            buffer["baseline_actions"], dtype=np.int64
        ),
        source_policy_temperatures=np.asarray(
            buffer["source_policy_temperatures"], dtype=np.float64
        ),
        sample_weights=np.asarray(buffer["sample_weights"], dtype=np.float32),
        sample_ids=tuple(str(value) for value in buffer["sample_ids"]),
        episode_ids=tuple(str(value) for value in buffer["episode_ids"]),
        world_keys=tuple(str(value) for value in buffer["world_keys"]),
    )


def load_non_card_counterfactuals(
    patterns: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 20260801,
    allow_mixed_policy_temperatures: bool = False,
) -> NonCardCounterfactualDatasets:
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    _integer(split_seed, "split_seed", minimum=0)
    if not isinstance(allow_mixed_policy_temperatures, bool):
        raise TypeError("allow_mixed_policy_temperatures must be boolean")
    paths = expand_input_paths(patterns)
    input_manifests = tuple(_peek_manifest(path) for path in paths)
    source_acts = {
        int(manifest["collection"]["acts"])
        for manifest in input_manifests
    }
    if len(source_acts) != 1:
        raise ValueError("all non-card input files must use the same acts horizon")
    source_temperatures = {
        float(manifest["collection"]["policyTemperature"])
        for manifest in input_manifests
    }
    if len(source_temperatures) != 1 and not allow_mixed_policy_temperatures:
        raise ValueError(
            "input files use mixed policyTemperature values; supply a training "
            "policy-temperature override to recompute all targets"
        )
    buffers = {
        decision: {"train": _empty_buffers(), "validation": _empty_buffers()}
        for decision in DECISION_KINDS
    }
    seen_sample_ids: dict[str, str] = {}
    seen_episode_worlds: dict[str, str] = {}
    seen_world_sources: dict[str, tuple[Path, str]] = {}

    def visitor(record: dict[str, object], source_path: Path) -> None:
        world_key = str(record["worldKey"])
        episode_id = str(record["episodeId"])
        previous_source = seen_world_sources.get(world_key)
        if previous_source is not None:
            previous_path, previous_episode_id = previous_source
            if previous_path != source_path:
                raise ValueError(
                    "canonical hidden world overlaps input files: "
                    f"{world_key} ({previous_path} and {source_path})"
                )
            if previous_episode_id != episode_id:
                raise ValueError(
                    f"canonical hidden world {world_key} has multiple episodeId values"
                )
        else:
            seen_world_sources[world_key] = (source_path, episode_id)
        previous_world = seen_episode_worlds.get(episode_id)
        if previous_world is not None and previous_world != world_key:
            raise ValueError(
                f"episodeId {episode_id} is reused for a different hidden world"
            )
        seen_episode_worlds[episode_id] = world_key

        sample_id = str(record["sampleId"])
        if sample_id in seen_sample_ids:
            raise ValueError(
                f"duplicate sampleId across inputs: {sample_id} "
                f"(first decision {seen_sample_ids[sample_id]})"
            )
        decision = str(record["decision"])
        seen_sample_ids[sample_id] = decision
        partition = "validation" if deterministic_validation_membership(
            world_key,
            split_seed=split_seed,
            validation_fraction=validation_fraction,
        ) else "train"
        _append_record(buffers[decision][partition], record)

    reports = tuple(_walk_file(path, visitor) for path in paths)
    train_world_keys = {
        str(value)
        for decision in DECISION_KINDS
        for value in buffers[decision]["train"]["world_keys"]
    }
    validation_world_keys = {
        str(value)
        for decision in DECISION_KINDS
        for value in buffers[decision]["validation"]["world_keys"]
    }
    leaked_worlds = train_world_keys & validation_world_keys
    if leaked_worlds:
        raise RuntimeError(
            "canonicalWorldKey train/validation leakage: "
            + ", ".join(sorted(leaked_worlds)[:3])
        )

    def split(decision: str) -> DecisionSplit | None:
        train = _arrays(decision, buffers[decision]["train"])
        validation = _arrays(decision, buffers[decision]["validation"])
        if len(train) + len(validation) == 0:
            return None
        if set(train.sample_ids) & set(validation.sample_ids):
            raise RuntimeError(f"{decision} train/validation sample leakage")
        if set(train.world_keys) & set(validation.world_keys):
            raise RuntimeError(f"{decision} train/validation world leakage")
        return DecisionSplit(train=train, validation=validation)

    return NonCardCounterfactualDatasets(
        tax_return=split("tax-return"),
        revolution=split("revolution"),
        files=reports,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        group_split_key="canonicalWorldKey",
    )
