from __future__ import annotations

"""Strict deterministic merger for prepared V4 trajectory datasets.

The merger accepts only the two currently documented producers (Normal
warm-start conversion and direct DAgger collection), plus its own output.  It
does not reinterpret samples: valid tensor values are copied byte-for-byte,
and only invalid suffix dimensions are padded.  The privileged critic tensor
remains a separate training-only array throughout.
"""

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_COUNT,
)
from v4_dataset import (
    V4_DATASET_FORMAT,
    V4_DATASET_VERSION,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
)
from v4_env import (
    PRIVILEGED_STATE_LAYOUT,
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
    PRIVILEGED_STATE_SIZE,
)
from v4_model import V4ActorConfig, V4CriticConfig


NORMAL_PREPARATION_FORMAT = "dalmuti-v4-prepared-dataset-metadata"
DAGGER_PREPARATION_FORMAT = "dalmuti-v4-dagger-direct-npz"
PPO_PREPARATION_FORMAT = "dalmuti-v4-ppo-league-direct-npz"
MERGED_PREPARATION_FORMAT = "dalmuti-v4-merged-prepared-dataset-metadata"
MERGED_PREPARATION_VERSION = 1

NORMAL_INPUT_FORMAT = "dalmuti-v4-normal-warmstart-ndjson"
ROLE_NAMES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Explicit resource limits make malformed archives fail before unbounded work.
MAX_INPUTS = 256
MAX_ARCHIVE_MEMBERS = 64
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024**3
MAX_TRAJECTORIES = 10_000_000
MAX_TIME_STEPS = 1_000_000
MAX_TOTAL_SAMPLES = 100_000_000

STANDARD_FIELDS = tuple(field.name for field in fields(V4TrajectoryTensors))
BOOLEAN_STANDARD_FIELDS = {
    "player_mask", "history_mask", "legal_masks", "dones", "valid_masks",
}
INTEGER_STANDARD_FIELDS = {"actions", "expert_actions"}
FLOAT_STANDARD_FIELDS = set(STANDARD_FIELDS) - BOOLEAN_STANDARD_FIELDS - INTEGER_STANDARD_FIELDS

SAMPLE_AUXILIARY_DTYPES: dict[str, np.dtype] = {
    "bc_eligible_masks": np.dtype(np.bool_),
    "ppo_eligible_masks": np.dtype(np.bool_),
    "critic_eligible_masks": np.dtype(np.bool_),
    "candidate_actions": np.dtype(np.int64),
    "behavior_sources": np.dtype(np.int8),
    "forced_masks": np.dtype(np.bool_),
    "finish_places": np.dtype(np.int16),
    "environment_terminals": np.dtype(np.bool_),
    "source_steps": np.dtype(np.int64),
    "source_decision_indices": np.dtype(np.int64),
    "raw_returns": np.dtype(np.float32),
    "baseline_values": np.dtype(np.float32),
    "raw_advantages": np.dtype(np.float32),
    "advantage_scales": np.dtype(np.float32),
    "baseline_tiers": np.dtype(np.int8),
    "baseline_reference_counts": np.dtype(np.int32),
    "selected_action_probabilities": np.dtype(np.float64),
    "policy_entropies": np.dtype(np.float32),
    "terminal_chip_awards": np.dtype(np.int8),
}
TRAJECTORY_AUXILIARY_DTYPES: dict[str, np.dtype | None] = {
    "trajectory_ids": None,
    "trajectory_input_sha256s": None,
    "trajectory_player_counts": np.dtype(np.int16),
    "trajectory_roles": np.dtype(np.int8),
    "trajectory_acts": np.dtype(np.int16),
    "trajectory_actor_ids": np.dtype(np.int16),
    "trajectory_match_indices": np.dtype(np.int32),
    "trajectory_match_seeds": np.dtype(np.uint32),
    "trajectory_match_clusters": None,
    "trajectory_finish_places": np.dtype(np.int16),
    "trajectory_source_npz_sha256s": None,
}
KNOWN_ARRAYS = (
    set(STANDARD_FIELDS)
    | set(SAMPLE_AUXILIARY_DTYPES)
    | set(TRAJECTORY_AUXILIARY_DTYPES)
    | {"metadata_json"}
)

NORMAL_REQUIRED_AUX = {
    "finish_places", "environment_terminals", "source_steps",
    "trajectory_ids", "trajectory_input_sha256s",
}
DAGGER_REQUIRED_AUX = {
    "candidate_actions", "behavior_sources", "forced_masks", "finish_places",
    "environment_terminals", "source_decision_indices", "trajectory_ids",
    "trajectory_player_counts", "trajectory_roles", "trajectory_acts",
    "trajectory_actor_ids", "trajectory_match_indices", "trajectory_match_seeds",
}
PPO_REQUIRED_AUX = {
    "raw_returns", "baseline_values", "raw_advantages", "advantage_scales",
    "baseline_tiers", "baseline_reference_counts",
    "selected_action_probabilities", "policy_entropies",
    "terminal_chip_awards", "forced_masks", "source_decision_indices",
    "trajectory_ids", "trajectory_player_counts", "trajectory_roles",
    "trajectory_acts", "trajectory_actor_ids", "trajectory_match_indices",
    "trajectory_match_seeds", "trajectory_match_clusters",
    "trajectory_finish_places",
}
MERGED_REQUIRED_AUX = set(SAMPLE_AUXILIARY_DTYPES) | set(TRAJECTORY_AUXILIARY_DTYPES)


@dataclass(frozen=True)
class MergeResult:
    output_path: Path
    metadata_path: Path
    checksum_path: Path
    metadata_checksum_path: Path
    npz_sha256: str
    metadata_sha256: str
    fingerprint: str
    trajectories: int
    samples: int


@dataclass
class _PreparedInput:
    path: Path
    sha256: str
    checksum_sha256: str
    metadata: dict[str, object]
    actor_config: V4ActorConfig
    critic_config: V4CriticConfig
    arrays: dict[str, np.ndarray]
    trajectory_ids: list[str]
    lengths: np.ndarray
    player_counts: np.ndarray
    roles: np.ndarray
    acts: np.ndarray
    source_hashes: dict[str, object]
    privileged_layout: dict[str, object]

    @property
    def trajectory_count(self) -> int:
        return int(self.arrays["actions"].shape[0])

    @property
    def time_steps(self) -> int:
        return int(self.arrays["actions"].shape[1])

    @property
    def sample_count(self) -> int:
        return int(self.lengths.sum(dtype=np.int64))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _checksum_digest(sidecar: Path, source_name: str, label: str) -> str:
    try:
        text = sidecar.read_text(encoding="ascii")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required {label} is missing: {sidecar}") from error
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError(f"{label} must contain exactly one checksum line")
    parts = lines[0].split()
    if len(parts) not in (1, 2) or not SHA256_RE.fullmatch(parts[0]):
        raise ValueError(f"{label} must contain one lowercase SHA-256")
    if len(parts) == 2:
        named = parts[1].lstrip("*")
        if Path(named).name != source_name:
            raise ValueError(f"{label} names a different source file")
    return parts[0]


def _verify_checksum(source: Path, sidecar: Path, label: str) -> tuple[str, str]:
    expected = _checksum_digest(sidecar, source.name, label)
    actual = _sha256_file(source)
    if actual != expected:
        raise ValueError(f"{label} does not match {source.name}")
    return actual, _sha256_file(sidecar)


def _preflight_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not 1 <= len(members) <= MAX_ARCHIVE_MEMBERS:
                raise ValueError("V4 NPZ has an invalid number of members")
            names: set[str] = set()
            total = 0
            for member in members:
                if member.is_dir() or Path(member.filename).name != member.filename:
                    raise ValueError("V4 NPZ contains an unsafe member name")
                if not member.filename.endswith(".npy"):
                    raise ValueError("V4 NPZ contains a non-NPY member")
                if member.filename in names:
                    raise ValueError("V4 NPZ contains a duplicate member")
                names.add(member.filename)
                total += member.file_size
                if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("V4 NPZ exceeds the merger size limit")
    except zipfile.BadZipFile as error:
        raise ValueError("input is not a valid NPZ archive") from error


def _metadata_from_archive(archive: np.lib.npyio.NpzFile, label: str) -> dict[str, object]:
    if "metadata_json" not in archive.files:
        raise ValueError(f"{label} lacks metadata_json")
    value = archive["metadata_json"]
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{label} metadata_json must be a scalar string")
    raw_value = value.item()
    raw = raw_value if isinstance(raw_value, bytes) else str(raw_value).encode("utf-8")
    return _json_object(raw, f"{label} metadata_json")


def _validate_external_metadata(path: Path, embedded: Mapping[str, object], npz_sha: str) -> None:
    metadata_path = Path(f"{path}.metadata.json")
    if not metadata_path.exists():
        return
    external = _json_object(metadata_path.read_bytes(), f"{path.name} external metadata")
    if external.get("npzSha256") != npz_sha:
        raise ValueError("external metadata NPZ checksum does not match")
    internal_view = dict(external)
    del internal_view["npzSha256"]
    if internal_view != embedded:
        raise ValueError("external and embedded V4 metadata disagree")
    metadata_checksum = Path(f"{metadata_path}.sha256")
    if metadata_checksum.exists():
        _verify_checksum(metadata_path, metadata_checksum, "metadata checksum sidecar")


def _config_from_metadata(metadata: Mapping[str, object]) -> tuple[V4ActorConfig, V4CriticConfig]:
    actor_value = metadata.get("actorConfig")
    critic_value = metadata.get("criticConfig")
    if not isinstance(actor_value, dict) or not isinstance(critic_value, dict):
        raise ValueError("V4 metadata requires actorConfig and criticConfig objects")
    actor_fields = {field.name for field in fields(V4ActorConfig)}
    critic_fields = {field.name for field in fields(V4CriticConfig)}
    if set(actor_value) != actor_fields or set(critic_value) != critic_fields:
        raise ValueError("V4 actorConfig or criticConfig fields drifted")
    try:
        actor = V4ActorConfig(**actor_value)
        critic = V4CriticConfig(**critic_value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid V4 actorConfig or criticConfig") from error
    if (
        actor.observation_schema_version != 4
        or actor.action_catalogue_version != V3_ACTION_CATALOGUE_VERSION
        or V3_ACTION_COUNT != 236
    ):
        raise ValueError("unsupported V4 observation or action semantics")
    if critic.privileged_features != 512:
        raise ValueError("V4 critic must use the separate 512-feature privileged state")
    return actor, critic


def _validate_sha_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty source-hash object")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"{label} contains an invalid source SHA-256")
        result[key] = digest
    return result


def _catalogue_sha256() -> str:
    payload = json.dumps(
        {
            "version": V3_ACTION_CATALOGUE_VERSION,
            "catalogue": [dict(item) for item in V3_ACTION_CATALOGUE],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _canonical_privileged_layout() -> dict[str, object]:
    layout = json.loads(_canonical_json(PRIVILEGED_STATE_LAYOUT).decode("utf-8"))
    if (
        PRIVILEGED_STATE_SIZE != 512
        or PRIVILEGED_STATE_LAYOUT_ID != "dalmuti-v4-ts-privileged-critic-raw-v1"
        or _sha256_bytes(_canonical_json(layout)) != PRIVILEGED_STATE_LAYOUT_SHA256
    ):
        raise RuntimeError("the imported canonical privileged critic layout is inconsistent")
    return {
        "id": PRIVILEGED_STATE_LAYOUT_ID,
        "sha256": PRIVILEGED_STATE_LAYOUT_SHA256,
        "layout": layout,
        "featureCount": PRIVILEGED_STATE_SIZE,
        "matchesTypescriptNormalContract": True,
    }


def _validate_dagger_layout(metadata: Mapping[str, object]) -> dict[str, object]:
    expected = _canonical_privileged_layout()
    value = metadata.get("privilegedCriticLayout")
    if not isinstance(value, dict) or any(
        value.get(name) != expected[name]
        for name in (
            "id", "sha256", "layout", "featureCount",
            "matchesTypescriptNormalContract",
        )
    ):
        raise ValueError("DAgger privileged critic layout is missing or non-canonical")
    return expected


def _validate_ppo_layout(
    metadata: Mapping[str, object], source_hashes: Mapping[str, str]
) -> dict[str, object]:
    expected = _canonical_privileged_layout()
    value = metadata.get("privilegedCriticBinding")
    if (
        not isinstance(value, dict)
        or value.get("layoutId") != expected["id"]
        or value.get("layoutSha256") != expected["sha256"]
        or value.get("layout") != expected["layout"]
        or value.get("featureCount") != expected["featureCount"]
        or value.get("actorExportAllowed") is not False
        or value.get("environmentSourceSha256")
        != source_hashes.get("gpu-training/v4_env.py")
    ):
        raise ValueError("PPO privileged critic layout is missing or non-canonical")
    return expected


def _validate_preparation(
    metadata: Mapping[str, object]
) -> tuple[str, dict[str, object], dict[str, object]]:
    if metadata.get("format") != V4_DATASET_FORMAT or metadata.get("version") != V4_DATASET_VERSION:
        raise ValueError("unsupported V4 dataset format or version")
    preparation = metadata.get("preparationFormat")
    if preparation not in {
        NORMAL_PREPARATION_FORMAT,
        DAGGER_PREPARATION_FORMAT,
        PPO_PREPARATION_FORMAT,
        MERGED_PREPARATION_FORMAT,
    } or metadata.get("preparationVersion") != 1:
        raise ValueError("unsupported V4 preparation semantics")

    if preparation == NORMAL_PREPARATION_FORMAT:
        if metadata.get("privilegedCriticExportAllowed") is not False:
            raise ValueError("Normal dataset does not prohibit privileged critic export")
        inputs = metadata.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("Normal prepared metadata requires source inputs")
        bound: list[dict[str, object]] = []
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("Normal source input metadata must be an object")
            if item.get("format") != NORMAL_INPUT_FORMAT or item.get("formatVersion") != 1:
                raise ValueError("Normal source input format drifted")
            digest = item.get("sha256")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError("Normal source input lacks a valid SHA-256")
            sources = _validate_sha_mapping(item.get("sourceHashes"), "Normal sourceHashes")
            if sources.get("actionCatalogue") != _catalogue_sha256():
                raise ValueError("Normal source action catalogue semantics drifted")
            if "privilegedCriticContract" not in sources:
                raise ValueError("Normal source lacks its strict TS privileged critic binding")
            bound.append({
                "inputSha256": digest,
                "sourceHashes": sources,
                "playerCount": item.get("playerCount"),
                "actsPerEpisode": item.get("actsPerEpisode"),
            })
        # v4_prepare_dataset accepts a Normal input only after exact comparison
        # with the TS manifest's raw 512-feature layout.  Bind that validated
        # preparation format to the shared canonical layout here.
        return str(preparation), {"normalInputs": bound}, _canonical_privileged_layout()

    if preparation == DAGGER_PREPARATION_FORMAT:
        privacy = metadata.get("privacy")
        if not isinstance(privacy, dict) or any(
            privacy.get(name) is not expected
            for name, expected in {
                "actorPublicOnly": True,
                "opponentPhysicalHandsExcluded": True,
                "taxCardIdentitiesExcluded": True,
                "privilegedCriticStateSeparate": True,
                "privilegedCriticExportAllowed": False,
            }.items()
        ):
            raise ValueError("DAgger public-only privacy contract is missing")
        collection = metadata.get("collection")
        environment = metadata.get("environmentBinding")
        model = metadata.get("modelBinding")
        if (
            not isinstance(collection, dict)
            or collection.get("algorithm") != "DAgger"
            or collection.get("expert") != "exact-v4-env-Normal"
            or collection.get("expertLabelForEveryDecision") is not True
            or not isinstance(environment, dict)
            or environment.get("normalExpertCallback") != "DalmutiScalarEnv.normal_action"
            or not isinstance(model, dict)
            or model.get("criticExcluded") is not True
        ):
            raise ValueError("DAgger action/expert semantics drifted")
        sources = _validate_sha_mapping(metadata.get("sourceHashes"), "DAgger sourceHashes")
        if environment.get("v4EnvSha256") != sources.get("gpu-training/v4_env.py"):
            raise ValueError("DAgger environment source binding is inconsistent")
        for name in ("bundleManifestSha256", "actorCheckpointSha256"):
            if not isinstance(model.get(name), str) or not SHA256_RE.fullmatch(str(model[name])):
                raise ValueError("DAgger model binding contains an invalid SHA-256")
        layout = _validate_dagger_layout(metadata)
        return str(preparation), {
            "sourceHashes": sources,
            "modelBinding": dict(model),
            "environmentBinding": dict(environment),
        }, layout

    if preparation == PPO_PREPARATION_FORMAT:
        privacy = metadata.get("privacy")
        collection = metadata.get("collection")
        environment = metadata.get("environmentBinding")
        model = metadata.get("modelBinding")
        returns = metadata.get("returnsAndAdvantages")
        if not isinstance(privacy, dict) or any(
            privacy.get(name) is not expected
            for name, expected in {
                "actorPublicOnly": True,
                "opponentPhysicalHandsExcluded": True,
                "taxCardIdentitiesExcluded": True,
                "privilegedCriticStateSeparate": True,
                "privilegedCriticExportAllowed": False,
            }.items()
        ):
            raise ValueError("PPO public-only privacy contract is missing")
        if (
            not isinstance(collection, dict)
            or collection.get("algorithm") != "on-policy PPO league rollout"
            or collection.get("exactOldLogProbabilityForEveryLearnerDecision") is not True
            or collection.get("exactNormalExpertLabelForEveryLearnerDecision") is not True
            or not isinstance(returns, dict)
            or not isinstance(returns.get("standardized"), bool)
            or not isinstance(environment, dict)
            or environment.get("normalExpertCallback") != "DalmutiScalarEnv.normal_action"
            or not isinstance(model, dict)
            or model.get("criticExcluded") is not True
        ):
            raise ValueError("PPO action/return semantics drifted")
        sources = _validate_sha_mapping(metadata.get("sourceHashes"), "PPO sourceHashes")
        if environment.get("v4EnvSha256") != sources.get("gpu-training/v4_env.py"):
            raise ValueError("PPO environment source binding is inconsistent")
        for name in ("bundleManifestSha256", "actorCheckpointSha256"):
            if not isinstance(model.get(name), str) or not SHA256_RE.fullmatch(str(model[name])):
                raise ValueError("PPO model binding contains an invalid SHA-256")
        layout = _validate_ppo_layout(metadata, sources)
        return str(preparation), {
            "sourceHashes": sources,
            "modelBinding": dict(model),
            "environmentBinding": dict(environment),
        }, layout

    privacy = metadata.get("privacy")
    if not isinstance(privacy, dict) or privacy.get("actorPublicOnly") is not True or privacy.get("privilegedCriticExportAllowed") is not False:
        raise ValueError("merged dataset privacy semantics drifted")
    action_semantics = metadata.get("actionSemantics")
    if (
        not isinstance(action_semantics, dict)
        or action_semantics.get("catalogueVersion") != V3_ACTION_CATALOGUE_VERSION
        or action_semantics.get("actionCount") != V3_ACTION_COUNT
        or action_semantics.get("catalogueSha256") != _catalogue_sha256()
    ):
        raise ValueError("merged dataset action semantics drifted")
    sources = metadata.get("sourceHashesByInput")
    if not isinstance(sources, list):
        raise ValueError("merged dataset source hash bindings are missing")
    for item in sources:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("inputSha256"), str)
            or not SHA256_RE.fullmatch(str(item["inputSha256"]))
        ):
            raise ValueError("merged dataset source hash binding is invalid")
    expected_layout = _canonical_privileged_layout()
    if metadata.get("privilegedCriticLayout") != expected_layout:
        raise ValueError("merged dataset privileged critic layout is missing or non-canonical")
    eligibility = metadata.get("lossEligibility")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("version") != 1
        or eligibility.get("masks") != {
            "behaviorCloning": "bc_eligible_masks",
            "ppo": "ppo_eligible_masks",
            "critic": "critic_eligible_masks",
        }
        or eligibility.get("semantics") != {
            "behaviorCloning": "exact Normal expert label",
            "ppo": "on-policy PPO collector samples only",
            "critic": "on-policy PPO collector samples only",
        }
        or not isinstance(eligibility.get("eligibleSampleCounts"), dict)
    ):
        raise ValueError("merged dataset loss eligibility contract is missing or incompatible")
    behavior_hashes = eligibility.get("ppoBehaviorActorSha256s")
    if (
        not isinstance(behavior_hashes, list)
        or behavior_hashes != sorted(set(behavior_hashes))
        or any(not isinstance(value, str) or not SHA256_RE.fullmatch(value) for value in behavior_hashes)
    ):
        raise ValueError("merged PPO behavior actor bindings are invalid")
    return str(preparation), {"sourceHashesByInput": sources}, expected_layout


def _expected_standard_shapes(
    count: int, time_steps: int, actor: V4ActorConfig, critic: V4CriticConfig
) -> dict[str, tuple[int, ...]]:
    prefix = (count, time_steps)
    return {
        "global_features": (*prefix, actor.global_features),
        "rank_features": (*prefix, actor.rank_tokens, actor.rank_features),
        "player_features": (*prefix, actor.max_players, actor.player_features),
        "player_mask": (*prefix, actor.max_players),
        "memory_trace_features": (*prefix, actor.memory_tokens, actor.memory_features),
        "history_features": (*prefix, actor.max_history, actor.history_features),
        "history_mask": (*prefix, actor.max_history),
        "legal_masks": (*prefix, V3_ACTION_COUNT),
        "actions": prefix,
        "expert_actions": prefix,
        "old_action_log_probs": prefix,
        "advantages": prefix,
        "rewards": prefix,
        "dones": prefix,
        "valid_masks": prefix,
        "privileged_states": (*prefix, critic.privileged_features),
    }


def _require_dtype(array: np.ndarray, expected: np.dtype, name: str) -> None:
    if array.dtype != expected:
        raise ValueError(f"{name} has dtype {array.dtype}, expected {expected}")


def _all_default(array: np.ndarray, value: object) -> bool:
    if array.size == 0:
        return True
    return bool(np.all(array == value))


def _contiguous_prefix(mask: np.ndarray, label: str) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError(f"{label} must be a bool [trajectory,time] array")
    lengths = mask.sum(axis=1, dtype=np.int64)
    positions = np.arange(mask.shape[1], dtype=np.int64)[None, :]
    if not np.array_equal(mask, positions < lengths[:, None]):
        raise ValueError(f"{label} must contain contiguous non-empty prefixes")
    if np.any(lengths < 1):
        raise ValueError("every V4 trajectory requires at least one sample")
    return lengths


def _validate_invalid_suffix(arrays: Mapping[str, np.ndarray], valid: np.ndarray) -> None:
    invalid = ~valid
    for name in STANDARD_FIELDS:
        array = arrays[name]
        expanded = invalid[(...,) + (None,) * (array.ndim - 2)]
        if np.any(np.where(expanded, array, 0) != 0):
            raise ValueError(f"{name} contains data in an invalid trajectory suffix")


def _decode_identity(
    arrays: Mapping[str, np.ndarray], lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(lengths)
    player_counts = np.empty(count, np.int16)
    roles = np.empty(count, np.int8)
    acts = np.empty(count, np.int16)
    for trajectory in range(count):
        length = int(lengths[trajectory])
        player_masks = arrays["player_mask"][trajectory, :length]
        player_lengths = player_masks.sum(axis=1, dtype=np.int64)
        positions = np.arange(player_masks.shape[1])[None, :]
        if not np.array_equal(player_masks, positions < player_lengths[:, None]):
            raise ValueError("player masks must be contiguous public prefixes")
        if np.any(player_lengths != player_lengths[0]) or not 4 <= player_lengths[0] <= 10:
            raise ValueError("player count must be stable and from 4 through 10")
        p = int(player_lengths[0])
        player_counts[trajectory] = p
        expected_player_scalar = (p - 4) / 6.0
        globals_ = arrays["global_features"][trajectory, :length]
        if not np.allclose(globals_[:, 0], expected_player_scalar, rtol=0.0, atol=2e-6):
            raise ValueError("public player-count feature disagrees with player masks")

        role_vectors = globals_[:, 2:7]
        role = int(np.argmax(role_vectors[0]))
        expected_role = np.zeros(5, np.float32)
        expected_role[role] = 1.0
        if not np.allclose(role_vectors, expected_role, rtol=0.0, atol=2e-6):
            raise ValueError("actor role must be a stable public one-hot feature")
        roles[trajectory] = role

        encoded = float(globals_[0, 1])
        if not -1.0 < encoded < 1.0:
            raise ValueError("public act feature is outside tanh encoding")
        act = int(round(1.0 + 10.0 * math.atanh(encoded)))
        if not 1 <= act <= 32767:
            raise ValueError("decoded act is outside supported metadata range")
        expected_act = math.tanh((act - 1) / 10.0)
        if not np.allclose(globals_[:, 1], expected_act, rtol=0.0, atol=2e-6):
            raise ValueError("act must be stable and use the canonical public encoding")
        acts[trajectory] = act

        history_masks = arrays["history_mask"][trajectory, :length]
        history_lengths = history_masks.sum(axis=1, dtype=np.int64)
        history_positions = np.arange(history_masks.shape[1])[None, :]
        if not np.array_equal(history_masks, history_positions < history_lengths[:, None]):
            raise ValueError("history masks must be contiguous public prefixes")
        player_features = arrays["player_features"][trajectory, :length]
        if np.any(player_features[~player_masks] != 0):
            raise ValueError("masked player slots must be zero and cannot carry private data")
        history_features = arrays["history_features"][trajectory, :length]
        if np.any(history_features[~history_masks] != 0):
            raise ValueError("masked history slots must be zero")
    return player_counts, roles, acts


def _tensor_dataset(
    arrays: Mapping[str, np.ndarray], actor: V4ActorConfig, critic: V4CriticConfig
) -> V4TrajectoryDataset:
    tensors: dict[str, torch.Tensor] = {}
    for name in STANDARD_FIELDS:
        tensor = torch.from_numpy(arrays[name])
        if name in BOOLEAN_STANDARD_FIELDS:
            tensor = tensor.to(dtype=torch.bool)
        elif name in INTEGER_STANDARD_FIELDS:
            tensor = tensor.to(dtype=torch.long)
        else:
            tensor = tensor.to(dtype=torch.float32)
        tensors[name] = tensor
    return V4TrajectoryDataset(V4TrajectoryTensors(**tensors), actor, critic)


def _validate_auxiliary(
    arrays: Mapping[str, np.ndarray], preparation: str, valid: np.ndarray,
    player_counts: np.ndarray, roles: np.ndarray, acts: np.ndarray,
    *, ppo_standardized: bool | None = None,
) -> None:
    required = (
        NORMAL_REQUIRED_AUX if preparation == NORMAL_PREPARATION_FORMAT
        else DAGGER_REQUIRED_AUX if preparation == DAGGER_PREPARATION_FORMAT
        else PPO_REQUIRED_AUX if preparation == PPO_PREPARATION_FORMAT
        else MERGED_REQUIRED_AUX
    )
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{preparation} lacks required auxiliary array {missing[0]}")
    count, time_steps = valid.shape
    for name, dtype in SAMPLE_AUXILIARY_DTYPES.items():
        if name not in arrays:
            continue
        array = arrays[name]
        if array.shape != (count, time_steps):
            raise ValueError(f"{name} must have [trajectory,time] shape")
        _require_dtype(array, dtype, name)
    for name, dtype in TRAJECTORY_AUXILIARY_DTYPES.items():
        if name not in arrays:
            continue
        array = arrays[name]
        if array.shape != (count,):
            raise ValueError(f"{name} must have [trajectory] shape")
        if dtype is None:
            if array.dtype.kind != "U":
                raise ValueError(f"{name} must use a fixed-width Unicode dtype")
        else:
            _require_dtype(array, dtype, name)

    invalid = ~valid
    invalid_defaults = {
        "candidate_actions": -1,
        "behavior_sources": -1,
        "forced_masks": False,
        "finish_places": 0,
        "environment_terminals": False,
        "source_steps": -1,
        "source_decision_indices": -1,
        "raw_returns": 0.0,
        "baseline_values": 0.0,
        "raw_advantages": 0.0,
        "advantage_scales": 1.0,
        "baseline_tiers": -1,
        "baseline_reference_counts": 0,
        "selected_action_probabilities": 0.0,
        "policy_entropies": 0.0,
        "terminal_chip_awards": 0,
        "bc_eligible_masks": False,
        "ppo_eligible_masks": False,
        "critic_eligible_masks": False,
    }
    for name, default in invalid_defaults.items():
        if name in arrays and not _all_default(arrays[name][invalid], default):
            raise ValueError(f"{name} uses a non-canonical invalid-suffix sentinel")

    if "candidate_actions" in arrays:
        candidate = arrays["candidate_actions"]
        legal = arrays["legal_masks"]
        safe = np.clip(candidate, 0, V3_ACTION_COUNT - 1)
        selected = np.take_along_axis(legal, safe[..., None], axis=-1)[..., 0]
        invalid_candidate = (
            ((candidate < 0) | (candidate >= V3_ACTION_COUNT) | ~selected)
            if preparation == DAGGER_PREPARATION_FORMAT
            else ((candidate != -1) & ((candidate < 0) | (candidate >= V3_ACTION_COUNT) | ~selected))
        )
        if np.any(valid & invalid_candidate):
            raise ValueError("candidate_actions contains an illegal valid action")
    if "behavior_sources" in arrays:
        allowed_behavior = [-1, 0, 1] if preparation == MERGED_PREPARATION_FORMAT else [0, 1]
        if np.any(valid & ~np.isin(arrays["behavior_sources"], allowed_behavior)):
            raise ValueError("behavior_sources contains an invalid behavior-policy code")
    if "forced_masks" in arrays:
        derived = arrays["legal_masks"].sum(axis=-1) == 1
        if np.any(valid & (arrays["forced_masks"] != derived)):
            raise ValueError("forced_masks disagrees with the legal action mask")
    if "source_steps" in arrays and preparation == NORMAL_PREPARATION_FORMAT:
        if np.any(arrays["source_steps"][valid] < 0):
            raise ValueError("Normal source_steps must be non-negative")
    if "source_decision_indices" in arrays and preparation in {
        DAGGER_PREPARATION_FORMAT, PPO_PREPARATION_FORMAT,
    }:
        if np.any(arrays["source_decision_indices"][valid] < 0):
            raise ValueError("direct-rollout source_decision_indices must be non-negative")

    if preparation == PPO_PREPARATION_FORMAT:
        selected = arrays["selected_action_probabilities"]
        if np.any(valid & ((selected <= 0.0) | (selected > 1.0) | ~np.isfinite(selected))):
            raise ValueError("PPO selected action probabilities must be finite in (0, 1]")
        if not np.allclose(
            arrays["old_action_log_probs"][valid],
            np.log(selected[valid]),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError("PPO old action log probabilities do not match behavior probabilities")
        scales = arrays["advantage_scales"]
        if np.any(valid & ((scales <= 0.0) | ~np.isfinite(scales))):
            raise ValueError("PPO advantage scales must be positive and finite")
        raw_expected = arrays["raw_returns"] - arrays["baseline_values"]
        if not np.allclose(
            arrays["raw_advantages"][valid], raw_expected[valid],
            rtol=0.0, atol=2.0e-6,
        ):
            raise ValueError("PPO raw advantages do not match returns minus baselines")
        if ppo_standardized is None:
            raise RuntimeError("PPO standardization binding was not supplied")
        expected_advantage = (
            arrays["raw_advantages"] / scales
            if ppo_standardized else arrays["raw_advantages"]
        )
        if not np.allclose(
            arrays["advantages"][valid], expected_advantage[valid],
            rtol=0.0, atol=2.0e-6,
        ):
            raise ValueError("PPO training advantages do not match their bound derivation")
        terminal = arrays["dones"] & valid
        expected_reward = (arrays["terminal_chip_awards"].astype(np.float32) - 2.0) / 2.0
        if not np.allclose(
            arrays["rewards"][terminal], expected_reward[terminal],
            rtol=0.0, atol=1.0e-7,
        ):
            raise ValueError("PPO terminal reward does not match its chip award")
        if np.any(valid & ~arrays["dones"] & (arrays["terminal_chip_awards"] != 0)):
            raise ValueError("PPO non-terminal samples cannot carry a chip award")
        if np.any(valid & (arrays["baseline_tiers"] < 0)) or np.any(
            valid & (arrays["baseline_reference_counts"] < 0)
        ):
            raise ValueError("PPO baseline metadata arrays contain invalid valid values")

    for name, expected in {
        "trajectory_player_counts": player_counts,
        "trajectory_roles": roles,
        "trajectory_acts": acts,
    }.items():
        if name in arrays and not np.array_equal(arrays[name].astype(expected.dtype), expected):
            raise ValueError(f"{name} disagrees with public actor tensors")
    eligibility_names = (
        "bc_eligible_masks", "ppo_eligible_masks", "critic_eligible_masks",
    )
    if preparation == MERGED_PREPARATION_FORMAT:
        bc, ppo, critic = (arrays[name] for name in eligibility_names)
        if not np.array_equal(bc, valid):
            raise ValueError("merged BC eligibility must equal every valid exact-Normal label")
        if np.any(ppo & ~valid) or np.any(critic & ~valid) or not np.array_equal(ppo, critic):
            raise ValueError("merged PPO and critic eligibility masks must match within valid samples")
    if preparation == PPO_PREPARATION_FORMAT:
        if any(not str(value) for value in arrays["trajectory_match_clusters"]):
            raise ValueError("PPO trajectory match clusters must be non-empty")
        if np.any(arrays["trajectory_actor_ids"] < 0) or np.any(
            arrays["trajectory_actor_ids"] >= player_counts
        ):
            raise ValueError("PPO trajectory actor IDs are outside their player counts")
        if np.any(arrays["trajectory_match_indices"] < 0):
            raise ValueError("PPO trajectory match indices must be non-negative")
        if np.any(arrays["trajectory_finish_places"] < 1) or np.any(
            arrays["trajectory_finish_places"] > player_counts
        ):
            raise ValueError("PPO trajectory finish places are invalid")


def _load_input(path: Path, checksum: Path) -> _PreparedInput:
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise ValueError(f"V4 merger input must be an existing .npz: {path}")
    npz_sha, checksum_sha = _verify_checksum(path, checksum, "input checksum sidecar")
    _preflight_zip(path)
    with np.load(path, allow_pickle=False) as archive:
        unknown = sorted(set(archive.files) - KNOWN_ARRAYS)
        if unknown:
            raise ValueError(f"input contains unknown incompatible array {unknown[0]}")
        metadata = _metadata_from_archive(archive, path.name)
        actor, critic = _config_from_metadata(metadata)
        preparation, source_hashes, privileged_layout = _validate_preparation(metadata)
        documented_aux = (
            NORMAL_REQUIRED_AUX if preparation == NORMAL_PREPARATION_FORMAT
            else DAGGER_REQUIRED_AUX if preparation == DAGGER_PREPARATION_FORMAT
            else PPO_REQUIRED_AUX if preparation == PPO_PREPARATION_FORMAT
            else MERGED_REQUIRED_AUX
        )
        incompatible_aux = sorted(
            set(archive.files) - set(STANDARD_FIELDS) - documented_aux - {"metadata_json"}
        )
        if incompatible_aux:
            raise ValueError(
                f"{preparation} contains incompatible auxiliary semantics: "
                f"{incompatible_aux[0]}"
            )
        missing_standard = sorted(set(STANDARD_FIELDS) - set(archive.files))
        if missing_standard:
            raise ValueError(f"input lacks standard V4 array {missing_standard[0]}")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files if name != "metadata_json"}

    actions = arrays["actions"]
    if actions.ndim != 2:
        raise ValueError("actions must have [trajectory,time] shape")
    count, time_steps = map(int, actions.shape)
    if not 1 <= count <= MAX_TRAJECTORIES or not 1 <= time_steps <= MAX_TIME_STEPS:
        raise ValueError("V4 dataset dimensions exceed merger limits")
    expected_shapes = _expected_standard_shapes(count, time_steps, actor, critic)
    for name in STANDARD_FIELDS:
        array = arrays[name]
        if array.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape does not match actor/critic configuration")
        expected_dtype = (
            np.dtype(np.bool_) if name in BOOLEAN_STANDARD_FIELDS
            else np.dtype(np.int64) if name in INTEGER_STANDARD_FIELDS
            else np.dtype(np.float32)
        )
        _require_dtype(array, expected_dtype, name)
        if name in FLOAT_STANDARD_FIELDS and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")

    lengths = _contiguous_prefix(arrays["valid_masks"], "valid_masks")
    _validate_invalid_suffix(arrays, arrays["valid_masks"])
    player_counts, roles, acts = _decode_identity(arrays, lengths)
    returns_metadata = metadata.get("returnsAndAdvantages")
    ppo_standardized = (
        bool(returns_metadata["standardized"])
        if preparation == PPO_PREPARATION_FORMAT and isinstance(returns_metadata, dict)
        else None
    )
    _validate_auxiliary(
        arrays, preparation, arrays["valid_masks"], player_counts, roles, acts,
        ppo_standardized=ppo_standardized,
    )
    if preparation == MERGED_PREPARATION_FORMAT:
        eligibility = metadata["lossEligibility"]
        assert isinstance(eligibility, dict)
        counts = eligibility["eligibleSampleCounts"]
        assert isinstance(counts, dict)
        expected_counts = {
            "behaviorCloning": int(arrays["bc_eligible_masks"].sum()),
            "ppo": int(arrays["ppo_eligible_masks"].sum()),
            "critic": int(arrays["critic_eligible_masks"].sum()),
        }
        if counts != expected_counts:
            raise ValueError("merged loss eligibility counts do not match their masks")
    ids_array = arrays.get("trajectory_ids")
    assert ids_array is not None
    trajectory_ids = [str(value) for value in ids_array.tolist()]
    if any(not value or len(value.encode("utf-8")) > 512 for value in trajectory_ids):
        raise ValueError("trajectory IDs must be non-empty and at most 512 UTF-8 bytes")
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("input contains duplicate trajectory IDs")
    if preparation in {NORMAL_PREPARATION_FORMAT, MERGED_PREPARATION_FORMAT}:
        metadata_ids = metadata.get("trajectoryIds")
        if metadata_ids != trajectory_ids:
            raise ValueError("embedded metadata trajectory IDs disagree with the NPZ array")

    if preparation == NORMAL_PREPARATION_FORMAT:
        allowed_hashes = {
            str(item["inputSha256"])
            for item in source_hashes["normalInputs"]  # type: ignore[index]
        }
        if any(str(value) not in allowed_hashes for value in arrays["trajectory_input_sha256s"]):
            raise ValueError("trajectory_input_sha256s is not bound by Normal metadata")

    dataset = _tensor_dataset(arrays, actor, critic)
    if metadata.get("fingerprint") != dataset.fingerprint:
        raise ValueError("input V4 dataset fingerprint does not match its tensors")
    if metadata.get("trajectoryCount") != count:
        raise ValueError("input trajectoryCount metadata does not match")
    if "sampleCount" in metadata and metadata.get("sampleCount") != int(lengths.sum()):
        raise ValueError("input sampleCount metadata does not match")
    if metadata.get("maxTimeSteps") != time_steps:
        raise ValueError("input maxTimeSteps metadata does not match")
    listed_aux = metadata.get("auxiliaryArrays")
    if not isinstance(listed_aux, list) or any(not isinstance(name, str) or name not in arrays for name in listed_aux):
        raise ValueError("input auxiliaryArrays metadata is invalid")
    _validate_external_metadata(path, metadata, npz_sha)
    return _PreparedInput(
        path=path,
        sha256=npz_sha,
        checksum_sha256=checksum_sha,
        metadata=metadata,
        actor_config=actor,
        critic_config=critic,
        arrays=arrays,
        trajectory_ids=trajectory_ids,
        lengths=lengths,
        player_counts=player_counts,
        roles=roles,
        acts=acts,
        source_hashes=source_hashes,
        privileged_layout=privileged_layout,
    )


def _compatible_actor_base(config: V4ActorConfig) -> dict[str, object]:
    value = config.to_dict()
    del value["max_players"]
    del value["max_history"]
    return value


def _allocate_output(
    trajectories: int, time_steps: int, actor: V4ActorConfig, critic: V4CriticConfig
) -> dict[str, np.ndarray]:
    prefix = (trajectories, time_steps)
    arrays: dict[str, np.ndarray] = {
        "global_features": np.zeros((*prefix, actor.global_features), np.float32),
        "rank_features": np.zeros(
            (*prefix, actor.rank_tokens, actor.rank_features), np.float32
        ),
        "player_features": np.zeros((*prefix, actor.max_players, actor.player_features), np.float32),
        "player_mask": np.zeros((*prefix, actor.max_players), np.bool_),
        "memory_trace_features": np.zeros((*prefix, actor.memory_tokens, actor.memory_features), np.float32),
        "history_features": np.zeros((*prefix, actor.max_history, actor.history_features), np.float32),
        "history_mask": np.zeros((*prefix, actor.max_history), np.bool_),
        "legal_masks": np.zeros((*prefix, V3_ACTION_COUNT), np.bool_),
        "actions": np.zeros(prefix, np.int64),
        "expert_actions": np.zeros(prefix, np.int64),
        "old_action_log_probs": np.zeros(prefix, np.float32),
        "advantages": np.zeros(prefix, np.float32),
        "rewards": np.zeros(prefix, np.float32),
        "dones": np.zeros(prefix, np.bool_),
        "valid_masks": np.zeros(prefix, np.bool_),
        "privileged_states": np.zeros((*prefix, critic.privileged_features), np.float32),
        "bc_eligible_masks": np.zeros(prefix, np.bool_),
        "ppo_eligible_masks": np.zeros(prefix, np.bool_),
        "critic_eligible_masks": np.zeros(prefix, np.bool_),
        "candidate_actions": np.full(prefix, -1, np.int64),
        "behavior_sources": np.full(prefix, -1, np.int8),
        "forced_masks": np.zeros(prefix, np.bool_),
        "finish_places": np.zeros(prefix, np.int16),
        "environment_terminals": np.zeros(prefix, np.bool_),
        "source_steps": np.full(prefix, -1, np.int64),
        "source_decision_indices": np.full(prefix, -1, np.int64),
        "raw_returns": np.zeros(prefix, np.float32),
        "baseline_values": np.zeros(prefix, np.float32),
        "raw_advantages": np.zeros(prefix, np.float32),
        "advantage_scales": np.ones(prefix, np.float32),
        "baseline_tiers": np.full(prefix, -1, np.int8),
        "baseline_reference_counts": np.zeros(prefix, np.int32),
        "selected_action_probabilities": np.zeros(prefix, np.float64),
        "policy_entropies": np.zeros(prefix, np.float32),
        "terminal_chip_awards": np.zeros(prefix, np.int8),
        "trajectory_ids": np.empty(trajectories, dtype="<U512"),
        "trajectory_input_sha256s": np.empty(trajectories, dtype="<U64"),
        "trajectory_player_counts": np.empty(trajectories, np.int16),
        "trajectory_roles": np.empty(trajectories, np.int8),
        "trajectory_acts": np.empty(trajectories, np.int16),
        "trajectory_actor_ids": np.full(trajectories, -1, np.int16),
        "trajectory_match_indices": np.full(trajectories, -1, np.int32),
        "trajectory_match_seeds": np.full(trajectories, np.iinfo(np.uint32).max, np.uint32),
        "trajectory_match_clusters": np.empty(trajectories, dtype="<U512"),
        "trajectory_finish_places": np.full(trajectories, -1, np.int16),
        "trajectory_source_npz_sha256s": np.empty(trajectories, dtype="<U64"),
    }
    arrays["trajectory_input_sha256s"].fill("")
    arrays["trajectory_match_clusters"].fill("")
    return arrays


def _copy_input(output: dict[str, np.ndarray], source: _PreparedInput, start: int) -> None:
    stop = start + source.trajectory_count
    time_steps = source.time_steps
    player_slots = source.actor_config.max_players
    history_slots = source.actor_config.max_history
    direct_standard = {
        "global_features", "rank_features", "memory_trace_features", "legal_masks",
        "actions", "expert_actions", "old_action_log_probs", "advantages", "rewards",
        "dones", "valid_masks", "privileged_states",
    }
    for name in direct_standard:
        output[name][start:stop, :time_steps] = source.arrays[name]
    output["player_features"][start:stop, :time_steps, :player_slots] = source.arrays["player_features"]
    output["player_mask"][start:stop, :time_steps, :player_slots] = source.arrays["player_mask"]
    output["history_features"][start:stop, :time_steps, :history_slots] = source.arrays["history_features"]
    output["history_mask"][start:stop, :time_steps, :history_slots] = source.arrays["history_mask"]

    for name in SAMPLE_AUXILIARY_DTYPES:
        if name in source.arrays:
            output[name][start:stop, :time_steps] = source.arrays[name]
    if "bc_eligible_masks" not in source.arrays:
        output["bc_eligible_masks"][start:stop, :time_steps] = source.arrays["valid_masks"]
    if (
        "ppo_eligible_masks" not in source.arrays
        and source.metadata["preparationFormat"] == PPO_PREPARATION_FORMAT
    ):
        output["ppo_eligible_masks"][start:stop, :time_steps] = source.arrays["valid_masks"]
    if (
        "critic_eligible_masks" not in source.arrays
        and source.metadata["preparationFormat"] == PPO_PREPARATION_FORMAT
    ):
        output["critic_eligible_masks"][start:stop, :time_steps] = source.arrays["valid_masks"]
    # Safe, explicit defaults for a Normal input's documented missing DAgger fields.
    if (
        "behavior_sources" not in source.arrays
        and source.metadata["preparationFormat"] == NORMAL_PREPARATION_FORMAT
    ):
        valid = source.arrays["valid_masks"]
        view = output["behavior_sources"][start:stop, :time_steps]
        view[valid] = 0
    if "forced_masks" not in source.arrays:
        output["forced_masks"][start:stop, :time_steps] = (
            source.arrays["legal_masks"].sum(axis=-1) == 1
        ) & source.arrays["valid_masks"]

    output["trajectory_ids"][start:stop] = source.arrays["trajectory_ids"]
    output["trajectory_player_counts"][start:stop] = source.player_counts
    output["trajectory_roles"][start:stop] = source.roles
    output["trajectory_acts"][start:stop] = source.acts
    for name in (
        "trajectory_input_sha256s", "trajectory_actor_ids", "trajectory_match_indices",
        "trajectory_match_seeds", "trajectory_match_clusters",
        "trajectory_finish_places",
    ):
        if name in source.arrays:
            output[name][start:stop] = source.arrays[name]
    output["trajectory_source_npz_sha256s"][start:stop] = source.sha256


def _balance(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    lengths = arrays["valid_masks"].sum(axis=1, dtype=np.int64)
    scopes: dict[str, dict[str, dict[str, int]]] = {
        "byPlayerCount": {}, "byRole": {}, "byAct": {}, "byPlayerRoleAct": {},
    }
    for index, samples in enumerate(lengths.tolist()):
        player = int(arrays["trajectory_player_counts"][index])
        role = ROLE_NAMES[int(arrays["trajectory_roles"][index])]
        act = int(arrays["trajectory_acts"][index])
        keys = {
            "byPlayerCount": str(player),
            "byRole": role,
            "byAct": str(act),
            "byPlayerRoleAct": f"{player}|{role}|{act}",
        }
        for scope, key in keys.items():
            record = scopes[scope].setdefault(key, {"trajectories": 0, "samples": 0})
            record["trajectories"] += 1
            record["samples"] += int(samples)
    return scopes


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("w+b") as output:
        with zipfile.ZipFile(
            output, mode="w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9, strict_timestamps=True,
        ) as archive:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                with archive.open(info, mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asanyarray(arrays[name]), allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())


def _temporary_path(directory: Path, name: str) -> Path:
    descriptor, value = tempfile.mkstemp(prefix=f".{name}.", suffix=".partial", dir=directory)
    os.close(descriptor)
    return Path(value)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _promote_all(temporaries: Mapping[Path, Path]) -> None:
    promoted: list[Path] = []
    try:
        for target, temporary in temporaries.items():
            os.link(temporary, target)
            promoted.append(target)
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
    except Exception:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        for target in promoted:
            target.unlink(missing_ok=True)
        raise


def _path_list(value: str | Path | Sequence[str | Path], label: str) -> list[Path]:
    values: Sequence[str | Path] = [value] if isinstance(value, (str, Path)) else value
    if not values or len(values) > MAX_INPUTS:
        raise ValueError(f"{label} requires from 1 through {MAX_INPUTS} paths")
    return [Path(item).resolve() for item in values]


def merge_v4_datasets(
    input_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    checksum_path: str | Path | Sequence[str | Path] | None = None,
) -> MergeResult:
    inputs = _path_list(input_path, "input")
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("merged V4 output must end in .npz")
    if checksum_path is None:
        checksums = [Path(f"{path}.sha256") for path in inputs]
    else:
        checksums = _path_list(checksum_path, "input checksum")
        if len(checksums) != len(inputs):
            raise ValueError("input checksum paths must match input paths")
    if len(set(inputs)) != len(inputs):
        raise ValueError("the same V4 input path was supplied more than once")

    loaded = [_load_input(path, checksum) for path, checksum in zip(inputs, checksums, strict=True)]
    loaded.sort(key=lambda item: item.sha256)
    if len({item.sha256 for item in loaded}) != len(loaded):
        raise ValueError("duplicate V4 input content is not allowed")
    actor_base = _compatible_actor_base(loaded[0].actor_config)
    critic_config = loaded[0].critic_config
    for item in loaded[1:]:
        if _compatible_actor_base(item.actor_config) != actor_base:
            raise ValueError("V4 actor configuration drift exceeds pad-only dimensions")
        if item.critic_config != critic_config:
            raise ValueError("V4 critic configuration drift is not mergeable")
        if item.privileged_layout != loaded[0].privileged_layout:
            raise ValueError("V4 privileged critic layout drift is not mergeable")
    if loaded[0].privileged_layout != _canonical_privileged_layout():
        raise ValueError("V4 inputs are not bound to the canonical TS critic layout")
    actor_dict = loaded[0].actor_config.to_dict()
    actor_dict["max_players"] = max(item.actor_config.max_players for item in loaded)
    actor_dict["max_history"] = max(item.actor_config.max_history for item in loaded)
    actor_config = V4ActorConfig(**actor_dict)

    all_ids = [identifier for item in loaded for identifier in item.trajectory_ids]
    seen_ids: set[str] = set()
    duplicate_id: str | None = None
    for identifier in all_ids:
        if identifier in seen_ids:
            duplicate_id = identifier if duplicate_id is None else min(duplicate_id, identifier)
        seen_ids.add(identifier)
    if duplicate_id is not None:
        raise ValueError(f"V4 input datasets duplicate trajectory ID: {duplicate_id}")
    trajectory_count = sum(item.trajectory_count for item in loaded)
    sample_count = sum(item.sample_count for item in loaded)
    max_time = max(item.time_steps for item in loaded)
    if trajectory_count > MAX_TRAJECTORIES or sample_count > MAX_TOTAL_SAMPLES:
        raise ValueError("merged V4 dataset exceeds merger trajectory/sample limits")
    arrays = _allocate_output(trajectory_count, max_time, actor_config, critic_config)
    offset = 0
    for item in loaded:
        _copy_input(arrays, item, offset)
        offset += item.trajectory_count

    dataset = _tensor_dataset(arrays, actor_config, critic_config)
    ordered_inputs: list[dict[str, object]] = []
    source_hashes_by_input: list[dict[str, object]] = []
    for item in loaded:
        item_preparation = str(item.metadata["preparationFormat"])
        item_bc = (
            int(item.arrays["bc_eligible_masks"].sum())
            if "bc_eligible_masks" in item.arrays else item.sample_count
        )
        item_ppo = (
            int(item.arrays["ppo_eligible_masks"].sum())
            if "ppo_eligible_masks" in item.arrays
            else item.sample_count if item_preparation == PPO_PREPARATION_FORMAT else 0
        )
        item_critic = (
            int(item.arrays["critic_eligible_masks"].sum())
            if "critic_eligible_masks" in item.arrays
            else item.sample_count if item_preparation == PPO_PREPARATION_FORMAT else 0
        )
        ordered_inputs.append({
            "sha256": item.sha256,
            "checksumSidecarSha256": item.checksum_sha256,
            "fingerprint": item.metadata["fingerprint"],
            "format": item.metadata["format"],
            "version": item.metadata["version"],
            "preparationFormat": item.metadata["preparationFormat"],
            "preparationVersion": item.metadata["preparationVersion"],
            "trajectoryCount": item.trajectory_count,
            "sampleCount": item.sample_count,
            "maxTimeSteps": item.time_steps,
            "actorConfig": item.actor_config.to_dict(),
            "criticConfig": item.critic_config.to_dict(),
            "privilegedCriticLayoutSha256": item.privileged_layout["sha256"],
            "eligibility": {
                "bcSamples": item_bc,
                "ppoSamples": item_ppo,
                "criticSamples": item_critic,
            },
        })
        source_hashes_by_input.append({"inputSha256": item.sha256, **item.source_hashes})

    defaulted: dict[str, list[str]] = {}
    ppo_behavior_actor_sha256s: set[str] = set()
    for item in loaded:
        item_preparation = str(item.metadata["preparationFormat"])
        if item_preparation == PPO_PREPARATION_FORMAT:
            model_binding = item.metadata["modelBinding"]
            assert isinstance(model_binding, dict)
            ppo_behavior_actor_sha256s.add(str(model_binding["actorCheckpointSha256"]))
        elif item_preparation == MERGED_PREPARATION_FORMAT:
            loss_eligibility = item.metadata["lossEligibility"]
            assert isinstance(loss_eligibility, dict)
            ppo_behavior_actor_sha256s.update(
                str(value) for value in loss_eligibility["ppoBehaviorActorSha256s"]
            )
        missing = sorted(
            (set(SAMPLE_AUXILIARY_DTYPES) | set(TRAJECTORY_AUXILIARY_DTYPES))
            - set(item.arrays)
            - {"trajectory_source_npz_sha256s"}
        )
        defaulted[item.sha256] = missing
    metadata: dict[str, object] = {
        "format": V4_DATASET_FORMAT,
        "version": V4_DATASET_VERSION,
        "preparationFormat": MERGED_PREPARATION_FORMAT,
        "preparationVersion": MERGED_PREPARATION_VERSION,
        "actorConfig": actor_config.to_dict(),
        "criticConfig": critic_config.to_dict(),
        "privilegedCriticLayout": loaded[0].privileged_layout,
        "actionSemantics": {
            "catalogueVersion": V3_ACTION_CATALOGUE_VERSION,
            "actionCount": V3_ACTION_COUNT,
            "catalogueSha256": _catalogue_sha256(),
            "actions": "behavior action selected from the 236-action catalogue",
            "expertActions": "exact Normal label selected from the same legal mask",
        },
        "fingerprint": dataset.fingerprint,
        "inputs": ordered_inputs,
        "sourceHashesByInput": source_hashes_by_input,
        "trajectoryCount": trajectory_count,
        "sampleCount": sample_count,
        "maxTimeSteps": max_time,
        "trajectoryIds": all_ids,
        "balance": _balance(arrays),
        "arrays": {
            "standard": list(STANDARD_FIELDS),
            "auxiliary": sorted(set(SAMPLE_AUXILIARY_DTYPES) | set(TRAJECTORY_AUXILIARY_DTYPES)),
            "defaultedByInputSha256": defaulted,
            "defaults": {
                "candidate_actions": -1,
                "bc_eligible_masks": "true exactly on every valid exact-Normal-labelled sample",
                "ppo_eligible_masks": "true only on valid direct PPO collector samples",
                "critic_eligible_masks": "true only on valid direct PPO collector samples",
                "behavior_sources": "0 on valid non-DAgger Normal samples; -1 on invalid suffix",
                "forced_masks": "derived from exactly one legal action when absent",
                "finish_places": 0,
                "environment_terminals": False,
                "source_steps": -1,
                "source_decision_indices": -1,
                "raw_returns": 0.0,
                "baseline_values": 0.0,
                "raw_advantages": 0.0,
                "advantage_scales": 1.0,
                "baseline_tiers": -1,
                "baseline_reference_counts": 0,
                "selected_action_probabilities": 0.0,
                "policy_entropies": 0.0,
                "terminal_chip_awards": 0,
                "trajectory_input_sha256s": "empty string",
                "trajectory_actor_ids": -1,
                "trajectory_match_indices": -1,
                "trajectory_match_seeds": int(np.iinfo(np.uint32).max),
                "trajectory_match_clusters": "empty string",
                "trajectory_finish_places": -1,
            },
        },
        "auxiliaryArrays": sorted(set(SAMPLE_AUXILIARY_DTYPES) | set(TRAJECTORY_AUXILIARY_DTYPES)),
        "eligibility": {
            "contract": {
                "bc_eligible_masks": "all valid Normal, DAgger, and PPO samples have exact Normal expert labels",
                "ppo_eligible_masks": "only direct PPO collector samples have real behavior log probabilities and advantages",
                "critic_eligible_masks": "only direct PPO collector samples are admitted to the mixed-data critic objective",
                "invalidSuffixAlwaysFalse": True,
            },
            "counts": {
                "validSamples": sample_count,
                "bcSamples": int(arrays["bc_eligible_masks"].sum()),
                "ppoSamples": int(arrays["ppo_eligible_masks"].sum()),
                "criticSamples": int(arrays["critic_eligible_masks"].sum()),
            },
        },
        "lossEligibility": {
            "version": 1,
            "masks": {
                "behaviorCloning": "bc_eligible_masks",
                "ppo": "ppo_eligible_masks",
                "critic": "critic_eligible_masks",
            },
            "semantics": {
                "behaviorCloning": "exact Normal expert label",
                "ppo": "on-policy PPO collector samples only",
                "critic": "on-policy PPO collector samples only",
            },
            "eligibleSampleCounts": {
                "behaviorCloning": int(arrays["bc_eligible_masks"].sum()),
                "ppo": int(arrays["ppo_eligible_masks"].sum()),
                "critic": int(arrays["critic_eligible_masks"].sum()),
            },
            "ppoBehaviorActorSha256s": sorted(ppo_behavior_actor_sha256s),
        },
        "padding": {
            "rule": "copy every valid sample unchanged; pad only invalid time/player/history suffix dimensions",
            "maxPlayers": actor_config.max_players,
            "maxHistory": actor_config.max_history,
            "maxTimeSteps": max_time,
        },
        "privacy": {
            "actorPublicOnly": True,
            "publicArrays": [
                "global_features", "rank_features", "player_features", "player_mask",
                "memory_trace_features", "history_features", "history_mask", "legal_masks",
            ],
            "maskedPublicSlotsZero": True,
            "privilegedCriticStateSeparate": True,
            "privilegedCriticArray": "privileged_states",
            "privilegedCriticExportAllowed": False,
        },
    }
    arrays["metadata_json"] = np.asarray(_canonical_json(metadata).decode("utf-8"))

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(f"{output}.metadata.json")
    checksum_output = Path(f"{output}.sha256")
    metadata_checksum = Path(f"{metadata_path}.sha256")
    targets = (output, metadata_path, checksum_output, metadata_checksum)
    for target in targets:
        if target.exists():
            raise FileExistsError(f"output already exists: {target}")
    temporaries = {target: _temporary_path(output.parent, target.name) for target in targets}
    try:
        _write_deterministic_npz(temporaries[output], arrays)
        npz_sha = _sha256_file(temporaries[output])
        external = dict(metadata)
        external["npzSha256"] = npz_sha
        metadata_bytes = _canonical_json(external) + b"\n"
        metadata_sha = _sha256_bytes(metadata_bytes)
        _write_bytes(temporaries[metadata_path], metadata_bytes)
        _write_bytes(temporaries[checksum_output], f"{npz_sha}  {output.name}\n".encode("ascii"))
        _write_bytes(
            temporaries[metadata_checksum],
            f"{metadata_sha}  {metadata_path.name}\n".encode("ascii"),
        )
        _promote_all(temporaries)
    except Exception:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        raise
    return MergeResult(
        output_path=output,
        metadata_path=metadata_path,
        checksum_path=checksum_output,
        metadata_checksum_path=metadata_checksum,
        npz_sha256=npz_sha,
        metadata_sha256=metadata_sha,
        fingerprint=dataset.fingerprint,
        trajectories=trajectory_count,
        samples=sample_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly merge checksum-bound Normal and DAgger V4 trajectory NPZ datasets."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--input-checksum", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = merge_v4_datasets(
        args.input, args.output, checksum_path=args.input_checksum,
    )
    print(json.dumps({
        "output": str(result.output_path),
        "npzSha256": result.npz_sha256,
        "metadataSha256": result.metadata_sha256,
        "fingerprint": result.fingerprint,
        "trajectories": result.trajectories,
        "samples": result.samples,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DAGGER_PREPARATION_FORMAT",
    "MERGED_PREPARATION_FORMAT",
    "MERGED_PREPARATION_VERSION",
    "MergeResult",
    "NORMAL_PREPARATION_FORMAT",
    "PPO_PREPARATION_FORMAT",
    "main",
    "merge_v4_datasets",
]
