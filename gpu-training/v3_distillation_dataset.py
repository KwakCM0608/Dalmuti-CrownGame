from __future__ import annotations

import glob
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from actor_critic import ACTION_COUNT as LEGACY_ACTION_COUNT
from actor_critic import OBSERVATION_FEATURES, load_behavior_model
from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_COUNT,
    V3_ACTION_FEATURE_COUNT,
    V3_ACTION_FEATURE_LAYOUT,
    V3_ACTION_FEATURES,
)
from v3_ppo_dataset import legal_actions_from_observation


DISTILLATION_FORMAT = "dalmuti-v3-distillation-ndjson"
DISTILLATION_FORMAT_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 2
SHA256_HEX_LENGTH = 64


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.update(path.resolve() for path in matches if path.is_file())
    result = sorted(paths, key=lambda path: str(path).lower())
    if not result:
        raise FileNotFoundError("no legacy rollout files matched")
    return result


def _legacy_semantic_action(action_index: int) -> dict[str, object]:
    if isinstance(action_index, bool) or not isinstance(action_index, int):
        raise ValueError("legacy action index must be an integer")
    if action_index < 0 or action_index >= LEGACY_ACTION_COUNT:
        raise ValueError("legacy action index is out of range")
    if action_index == 0:
        return {"type": "pass"}
    if action_index == 1:
        return {"type": "solo-joker"}
    offset = action_index - 2
    joker_count = offset % 3
    rank_and_count = offset // 3
    count = rank_and_count % 14 + 1
    rank = rank_and_count // 14 + 1
    natural_count = count - joker_count
    if natural_count < 1 or natural_count > rank:
        raise ValueError(
            f"legacy action {action_index} is not structurally possible"
        )
    return {
        "type": "play",
        "rank": rank,
        "count": count,
        "jokerCount": joker_count,
    }


_V3_ACTION_LOOKUP = {
    json.dumps(dict(action), sort_keys=True, separators=(",", ":")): index
    for index, action in enumerate(V3_ACTION_CATALOGUE)
}


def legacy_action_index_to_v3(action_index: int) -> int:
    semantic = _legacy_semantic_action(action_index)
    key = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
    try:
        return _V3_ACTION_LOOKUP[key]
    except KeyError as error:
        raise ValueError(
            f"legacy action {action_index} has no V3 semantic mapping"
        ) from error


def v3_action_index_to_legacy(action_index: int) -> int:
    if (
        isinstance(action_index, bool)
        or not isinstance(action_index, int)
        or action_index < 0
        or action_index >= V3_ACTION_COUNT
    ):
        raise ValueError("V3 action index is out of range")
    action = V3_ACTION_CATALOGUE[action_index]
    action_type = action["type"]
    if action_type == "pass":
        return 0
    if action_type == "solo-joker":
        return 1
    rank = int(action["rank"])
    count = int(action["count"])
    joker_count = int(action["jokerCount"])
    return 2 + ((rank - 1) * 14 + (count - 1)) * 3 + joker_count


def legacy_legal_action_indices_to_v3(
    legacy_indices: Sequence[int],
) -> list[int]:
    if not legacy_indices or len(set(legacy_indices)) != len(legacy_indices):
        raise ValueError("legacy legal actions must be non-empty and unique")
    result = sorted(legacy_action_index_to_v3(index) for index in legacy_indices)
    if len(set(result)) != len(result):
        raise ValueError("legacy legal actions do not map one-to-one into V3")
    round_trip = sorted(v3_action_index_to_legacy(index) for index in result)
    if round_trip != sorted(legacy_indices):
        raise ValueError("legacy/V3 legal-action set round trip failed")
    return result


_STRUCTURAL_LEGACY_INDICES = tuple(
    v3_action_index_to_legacy(index) for index in range(V3_ACTION_COUNT)
)
if (
    len(set(_STRUCTURAL_LEGACY_INDICES)) != V3_ACTION_COUNT
    or any(
        legacy_action_index_to_v3(legacy_index) != v3_index
        for v3_index, legacy_index in enumerate(_STRUCTURAL_LEGACY_INDICES)
    )
):
    raise RuntimeError("legacy/V3 action bridge is not bijective")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def validate_legacy_manifest(value: object, path: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: legacy rollout manifest must be an object")
    if (
        value.get("type") != "manifest"
        or value.get("format") != "dalmuti-ppo-ndjson"
        or value.get("formatVersion") != 1
        or value.get("observation", {}).get("version")
        != OBSERVATION_SCHEMA_VERSION
        or value.get("observation", {}).get("featureCount")
        != OBSERVATION_FEATURES
        or value.get("actionSpace", {}).get("size") != LEGACY_ACTION_COUNT
    ):
        raise ValueError(f"{path}: unsupported legacy PPO rollout contract")
    behavior = value.get("behaviorModel")
    if not isinstance(behavior, dict) or not _is_sha256(behavior.get("sha256")):
        raise ValueError(f"{path}: source behavior-model SHA-256 is missing")
    environment = value.get("environment")
    player_count = (
        environment.get("playerCount") if isinstance(environment, dict) else None
    )
    if (
        isinstance(player_count, bool)
        or not isinstance(player_count, int)
        or player_count < 4
        or player_count > 10
    ):
        raise ValueError(f"{path}: invalid source player count")
    return {
        "playerCount": player_count,
        "behaviorModelSha256": behavior["sha256"],
        "behaviorModelFormat": behavior.get("format"),
    }


def validate_legacy_sample(
    value: object,
    path: Path,
    line_number: int,
    *,
    expected_policy_version: str | None = None,
) -> tuple[list[float], list[int], list[int], str, str, bool]:
    label = f"{path}:{line_number}"
    if not isinstance(value, dict) or value.get("type") != "sample":
        raise ValueError(f"{label}: record is not a legacy PPO sample")
    observation = value.get("observation")
    if (
        not isinstance(observation, list)
        or len(observation) != OBSERVATION_FEATURES
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in observation
        )
    ):
        raise ValueError(f"{label}: invalid observation")
    legacy_legal = value.get("legalActionIndices")
    if (
        not isinstance(legacy_legal, list)
        or not legacy_legal
        or legacy_legal != sorted(set(legacy_legal))
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= LEGACY_ACTION_COUNT
            for index in legacy_legal
        )
    ):
        raise ValueError(f"{label}: invalid legacy legal actions")
    v3_legal = legacy_legal_action_indices_to_v3(legacy_legal)
    if legal_actions_from_observation(observation, label) != v3_legal:
        raise ValueError(
            f"{label}: legacy legal actions do not match encoded observation"
        )
    episode_id = value.get("episodeId")
    trajectory_id = value.get("trajectoryId")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(f"{label}: episodeId is missing")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"{label}: trajectoryId is missing")
    forced = value.get("forced")
    if not isinstance(forced, bool) or forced is not (len(legacy_legal) == 1):
        raise ValueError(f"{label}: forced flag does not match legal actions")
    action = value.get("actionIndex")
    if (
        isinstance(action, bool)
        or not isinstance(action, int)
        or action not in legacy_legal
    ):
        raise ValueError(f"{label}: selected legacy action is illegal")
    old_log_probability = _finite_number(
        value.get("oldLogProbability"), "oldLogProbability"
    )
    if old_log_probability > 1.0e-6:
        raise ValueError(f"{label}: old log probability must not be positive")
    _finite_number(value.get("oldValue"), "oldValue")
    _finite_number(value.get("reward"), "reward")
    if not isinstance(value.get("terminal"), bool):
        raise ValueError(f"{label}: terminal flag is invalid")
    if (
        expected_policy_version is not None
        and value.get("policyVersion") != expected_policy_version
    ):
        raise ValueError(f"{label}: source policy-version binding mismatch")
    return (
        [float(item) for item in observation],
        list(legacy_legal),
        v3_legal,
        episode_id,
        trajectory_id,
        forced,
    )


@dataclass(frozen=True)
class V3DistillationData:
    observations: np.ndarray
    legal_masks: np.ndarray
    teacher_probabilities: np.ndarray
    teacher_values: np.ndarray
    teacher_argmax_actions: np.ndarray
    group_keys: np.ndarray
    sample_ids: tuple[str, ...]
    teacher_sha256: str
    temperature: float
    manifest: Mapping[str, object]
    path: str

    def __len__(self) -> int:
        return int(self.observations.shape[0])


def _read_checksum(path: Path) -> str:
    checksum_path = path.with_suffix(f"{path.suffix}.sha256")
    parts = checksum_path.read_text(encoding="ascii").split()
    if len(parts) != 2 or parts[1] != path.name or not _is_sha256(parts[0]):
        raise ValueError(f"malformed dataset checksum: {checksum_path}")
    actual = file_sha256(path)
    if parts[0] != actual:
        raise ValueError(f"dataset SHA-256 mismatch: {path}")
    return actual


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def load_v3_distillation_data(
    path: str | Path,
    *,
    teacher_model_path: str | Path,
    binding_tolerance: float = 2.0e-5,
    verify_teacher_bindings: bool = True,
) -> V3DistillationData:
    dataset_path = Path(path).resolve()
    _read_checksum(dataset_path)
    teacher_path = Path(teacher_model_path).resolve()
    teacher_sha256 = file_sha256(teacher_path)
    teacher_model, teacher_payload = load_behavior_model(teacher_path)
    if (
        teacher_payload.get("format") != "dalmuti-actor-critic"
        or teacher_payload.get("version") != 1
        or teacher_payload.get("observationFeatures") != OBSERVATION_FEATURES
        or teacher_payload.get("actionCount") != LEGACY_ACTION_COUNT
    ):
        raise ValueError("teacher model is not a legacy 506-action actor-critic")
    observations: list[list[float]] = []
    legal_lists: list[list[int]] = []
    legacy_legal_lists: list[list[int]] = []
    probability_lists: list[list[float]] = []
    teacher_logits: list[list[float]] = []
    values: list[float] = []
    argmax_actions: list[int] = []
    group_keys: list[str] = []
    sample_ids: list[str] = []
    seen_sample_ids: set[str] = set()
    manifest: dict[str, object] | None = None
    summary: dict[str, object] | None = None
    declared_source_hashes: set[str] = set()
    observed_source_counts: dict[str, int] = {}
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{dataset_path}:{line_number}: invalid JSON"
                ) from error
            if line_number == 1:
                if not isinstance(record, dict):
                    raise ValueError("distillation manifest must be an object")
                if (
                    record.get("type") != "manifest"
                    or record.get("format") != DISTILLATION_FORMAT
                    or record.get("formatVersion") != DISTILLATION_FORMAT_VERSION
                ):
                    raise ValueError("unsupported V3 distillation dataset")
                teacher = record.get("teacher")
                action_space = record.get("actionSpace")
                observation_contract = record.get("observation")
                if (
                    not isinstance(teacher, dict)
                    or teacher.get("sha256") != teacher_sha256
                    or teacher.get("format") != "dalmuti-actor-critic"
                    or teacher.get("observationFeatures")
                    != OBSERVATION_FEATURES
                    or teacher.get("actionCount") != LEGACY_ACTION_COUNT
                    or not isinstance(action_space, dict)
                    or action_space.get("catalogueVersion")
                    != V3_ACTION_CATALOGUE_VERSION
                    or action_space.get("size") != V3_ACTION_COUNT
                    or action_space.get("catalogue")
                    != [dict(action) for action in V3_ACTION_CATALOGUE]
                    or action_space.get("actionFeatureLayout")
                    != list(V3_ACTION_FEATURE_LAYOUT)
                    or action_space.get("encodedActionFeatures")
                    != [list(features) for features in V3_ACTION_FEATURES]
                    or observation_contract
                    != {
                        "schemaVersion": OBSERVATION_SCHEMA_VERSION,
                        "featureCount": OBSERVATION_FEATURES,
                        "privacy": (
                            "own private hand plus public state only; "
                            "opponent hands excluded"
                        ),
                    }
                ):
                    raise ValueError("V3 distillation manifest contract mismatch")
                sources = record.get("sources")
                if not isinstance(sources, list) or not sources:
                    raise ValueError("distillation source provenance is missing")
                for source in sources:
                    if (
                        not isinstance(source, dict)
                        or not _is_sha256(source.get("sha256"))
                        or source["sha256"] in declared_source_hashes
                    ):
                        raise ValueError("invalid or duplicate source provenance")
                    declared_source_hashes.add(source["sha256"])
                manifest = record
                continue
            if not isinstance(record, dict):
                raise ValueError(
                    f"{dataset_path}:{line_number}: record must be an object"
                )
            if record.get("type") == "summary":
                if summary is not None:
                    raise ValueError("distillation dataset has duplicate summary")
                summary = record
                continue
            if summary is not None:
                raise ValueError("distillation sample appears after summary")
            _require_exact_keys(
                record,
                {
                    "type",
                    "sampleId",
                    "sourceSha256",
                    "episodeId",
                    "trajectoryId",
                    "groupKey",
                    "observation",
                    "legacyLegalActionIndices",
                    "legalActionIndices",
                    "teacherLogits",
                    "teacherLogProbabilities",
                    "teacherArgmaxActionIndex",
                    "teacherValue",
                },
                f"{dataset_path}:{line_number}",
            )
            if record["type"] != "sample":
                raise ValueError(f"{dataset_path}:{line_number}: unknown record")
            observation = record["observation"]
            if (
                not isinstance(observation, list)
                or len(observation) != OBSERVATION_FEATURES
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in observation
                )
            ):
                raise ValueError(f"{dataset_path}:{line_number}: invalid observation")
            legacy_legal = record["legacyLegalActionIndices"]
            legal = record["legalActionIndices"]
            if (
                not isinstance(legacy_legal, list)
                or legacy_legal != sorted(set(legacy_legal))
                or legacy_legal_action_indices_to_v3(legacy_legal) != legal
                or legal_actions_from_observation(observation, str(line_number))
                != legal
            ):
                raise ValueError(
                    f"{dataset_path}:{line_number}: legal mapping mismatch"
                )
            logits = record["teacherLogits"]
            log_probabilities = record["teacherLogProbabilities"]
            if (
                not isinstance(logits, list)
                or not isinstance(log_probabilities, list)
                or len(logits) != len(legal)
                or len(log_probabilities) != len(legal)
            ):
                raise ValueError(
                    f"{dataset_path}:{line_number}: teacher distribution shape mismatch"
                )
            logit_array = np.asarray(logits, dtype=np.float64)
            log_probability_array = np.asarray(log_probabilities, dtype=np.float64)
            if not np.isfinite(logit_array).all() or not np.isfinite(
                log_probability_array
            ).all():
                raise ValueError(
                    f"{dataset_path}:{line_number}: teacher distribution is non-finite"
                )
            temperature = _finite_number(
                manifest["teacher"]["temperature"], "teacher temperature"
            )
            normalized = logit_array / temperature
            expected_log_probabilities = normalized - np.logaddexp.reduce(normalized)
            if not np.allclose(
                log_probability_array,
                expected_log_probabilities,
                rtol=0.0,
                atol=binding_tolerance,
            ):
                raise ValueError(
                    f"{dataset_path}:{line_number}: teacher log-probability binding mismatch"
                )
            argmax = record["teacherArgmaxActionIndex"]
            maximum = float(logit_array.max())
            expected_argmax = min(
                action
                for action, logit in zip(legal, logit_array.tolist())
                if logit == maximum
            )
            if argmax != expected_argmax:
                raise ValueError(
                    f"{dataset_path}:{line_number}: teacher argmax binding mismatch"
                )
            value = _finite_number(record["teacherValue"], "teacher value")
            sample_id = record["sampleId"]
            source_sha = record["sourceSha256"]
            episode_id = record["episodeId"]
            trajectory_id = record["trajectoryId"]
            group_key = record["groupKey"]
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in seen_sample_ids
                or not _is_sha256(source_sha)
                or source_sha not in declared_source_hashes
                or not isinstance(episode_id, str)
                or not episode_id
                or not isinstance(trajectory_id, str)
                or not trajectory_id
                or group_key != f"{source_sha}:{episode_id}"
            ):
                raise ValueError(
                    f"{dataset_path}:{line_number}: sample provenance mismatch"
                )
            observations.append([float(item) for item in observation])
            legal_lists.append(list(legal))
            legacy_legal_lists.append(list(legacy_legal))
            probability_lists.append(np.exp(log_probability_array).tolist())
            teacher_logits.append(logit_array.tolist())
            values.append(value)
            argmax_actions.append(argmax)
            group_keys.append(group_key)
            sample_ids.append(sample_id)
            seen_sample_ids.add(sample_id)
            observed_source_counts[source_sha] = (
                observed_source_counts.get(source_sha, 0) + 1
            )
    if manifest is None or summary is None or not observations:
        raise ValueError("distillation dataset is incomplete")
    if (
        summary.get("samples") != len(observations)
        or summary.get("uniqueGroups") != len(set(group_keys))
        or summary.get("teacherSha256") != teacher_sha256
        or summary.get("temperature") != manifest["teacher"]["temperature"]
        or summary.get("sourceSampleCounts") != observed_source_counts
    ):
        raise ValueError("distillation summary counters or binding mismatch")
    temperature = float(manifest["teacher"]["temperature"])
    legal_masks = np.zeros((len(observations), V3_ACTION_COUNT), dtype=np.bool_)
    probabilities = np.zeros((len(observations), V3_ACTION_COUNT), dtype=np.float32)
    for index, (legal, probability) in enumerate(
        zip(legal_lists, probability_lists)
    ):
        legal_masks[index, legal] = True
        probabilities[index, legal] = probability
    observations_array = np.asarray(observations, dtype=np.float32)
    teacher_values = np.asarray(values, dtype=np.float32)
    if verify_teacher_bindings:
        teacher_model.eval()
        with torch.no_grad():
            for start in range(0, len(observations), 1024):
                end = min(start + 1024, len(observations))
                batch_observations = torch.from_numpy(observations_array[start:end])
                legacy_masks = torch.zeros(
                    (end - start, LEGACY_ACTION_COUNT), dtype=torch.bool
                )
                for offset, legacy_legal in enumerate(
                    legacy_legal_lists[start:end]
                ):
                    legacy_masks[offset, legacy_legal] = True
                logits, recomputed_values = teacher_model(
                    batch_observations, legacy_masks
                )
                for offset, (legacy_legal, legal) in enumerate(
                    zip(
                        legacy_legal_lists[start:end],
                        legal_lists[start:end],
                    )
                ):
                    by_v3 = sorted(
                        zip(
                            (
                                legacy_action_index_to_v3(value)
                                for value in legacy_legal
                            ),
                            legacy_legal,
                        )
                    )
                    if [entry[0] for entry in by_v3] != legal:
                        raise RuntimeError("legal mapping changed during verification")
                    recomputed = np.asarray(
                        [
                            float(logits[offset, legacy_index])
                            for _, legacy_index in by_v3
                        ],
                        dtype=np.float64,
                    )
                    if not np.allclose(
                        recomputed,
                        teacher_logits[start + offset],
                        rtol=0.0,
                        atol=binding_tolerance,
                    ):
                        raise ValueError("teacher logit binding mismatch")
                if not np.allclose(
                    recomputed_values.detach().cpu().numpy(),
                    teacher_values[start:end],
                    rtol=0.0,
                    atol=binding_tolerance,
                ):
                    raise ValueError("teacher value binding mismatch")
    return V3DistillationData(
        observations=observations_array,
        legal_masks=legal_masks,
        teacher_probabilities=probabilities,
        teacher_values=teacher_values,
        teacher_argmax_actions=np.asarray(argmax_actions, dtype=np.int64),
        group_keys=np.asarray(group_keys, dtype=object),
        sample_ids=tuple(sample_ids),
        teacher_sha256=teacher_sha256,
        temperature=temperature,
        manifest=manifest,
        path=str(dataset_path),
    )


def group_split_mask(
    group_keys: Sequence[str], *, validation_fraction: float, seed: int
) -> np.ndarray:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a non-negative integer")
    assignments: dict[str, bool] = {}
    for group_key in group_keys:
        if group_key not in assignments:
            digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
            unit = int.from_bytes(digest[:8], "big") / 2**64
            assignments[group_key] = unit < validation_fraction
    result = np.asarray([assignments[key] for key in group_keys], dtype=np.bool_)
    if not result.any() or result.all():
        raise ValueError("group split must produce non-empty train and validation sets")
    train_groups = {key for key, validation in assignments.items() if not validation}
    validation_groups = {key for key, validation in assignments.items() if validation}
    if train_groups & validation_groups:
        raise RuntimeError("group split leaked an episode between partitions")
    return result
