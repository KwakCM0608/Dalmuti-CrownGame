from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


OBSERVATION_FEATURES = 172
OBSERVATION_VERSION = 2
ACTION_COUNT = 506
ROLLOUT_FORMAT = "dalmuti-rl-ndjson"
ROLLOUT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RolloutSplit:
    observations: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class LoadedRollouts:
    train: RolloutSplit
    validation: RolloutSplit
    files: tuple[str, ...]
    total_samples: int
    forced_samples_skipped: int


def expand_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.update(path.resolve() for path in matches if path.is_file())
    result = sorted(paths, key=lambda path: str(path).lower())
    if not result:
        raise FileNotFoundError("no rollout files matched the supplied paths")
    return result


def _is_validation_episode(key: str, validation_fraction: float) -> bool:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    return value < validation_fraction


def _allocate_split(sample_count: int) -> RolloutSplit:
    return RolloutSplit(
        observations=np.empty(
            (sample_count, OBSERVATION_FEATURES),
            dtype=np.float32,
        ),
        legal_masks=np.zeros(
            (sample_count, ACTION_COUNT),
            dtype=np.bool_,
        ),
        actions=np.empty(sample_count, dtype=np.int64),
        weights=np.empty(sample_count, dtype=np.float32),
    )


def _validate_manifest(manifest: dict, path: Path) -> None:
    if manifest.get("type") != "manifest":
        raise ValueError(f"{path}: first record is not a manifest")
    if manifest.get("format") != ROLLOUT_FORMAT:
        raise ValueError(f"{path}: unsupported rollout format")
    if manifest.get("formatVersion") != ROLLOUT_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported rollout version")
    if manifest.get("observation", {}).get("featureCount") != OBSERVATION_FEATURES:
        raise ValueError(f"{path}: observation feature count mismatch")
    if manifest.get("observation", {}).get("version") != OBSERVATION_VERSION:
        raise ValueError(f"{path}: observation version mismatch")
    if manifest.get("actionSpace", {}).get("size") != ACTION_COUNT:
        raise ValueError(f"{path}: action count mismatch")


def _validate_sample(sample: dict, path: Path, line_number: int) -> None:
    observation = sample.get("observation")
    legal_indices = sample.get("legalActionIndices")
    action = sample.get("actionIndex")
    if not isinstance(observation, list) or len(observation) != OBSERVATION_FEATURES:
        raise ValueError(f"{path}:{line_number}: invalid observation")
    if not isinstance(legal_indices, list) or not legal_indices:
        raise ValueError(f"{path}:{line_number}: missing legal actions")
    if any(
        not isinstance(index, int) or index < 0 or index >= ACTION_COUNT
        for index in legal_indices
    ):
        raise ValueError(f"{path}:{line_number}: invalid legal action index")
    if not isinstance(action, int) or action not in legal_indices:
        raise ValueError(f"{path}:{line_number}: selected action is not legal")
    supervised_action = sample.get("supervisedActionIndex")
    if (
        supervised_action is not None
        and (
            not isinstance(supervised_action, int)
            or supervised_action not in legal_indices
        )
    ):
        raise ValueError(
            f"{path}:{line_number}: supervised action is not legal"
        )


def _walk_selected_samples(
    paths: Sequence[Path],
    *,
    validation_fraction: float,
    include_forced: bool,
    max_samples: int | None,
    visitor: Callable[[bool, dict], None] | None,
) -> tuple[int, int, int, int]:
    total_samples = 0
    forced_samples_skipped = 0
    train_samples = 0
    validation_samples = 0
    selected_samples = 0

    for path in paths:
        if max_samples is not None and selected_samples >= max_samples:
            break
        with path.open("r", encoding="utf-8") as stream:
            first_line = stream.readline()
            if not first_line:
                raise ValueError(f"{path}: empty rollout file")
            _validate_manifest(json.loads(first_line), path)

            for line_number, line in enumerate(stream, start=2):
                record = json.loads(line)
                if record.get("type") != "sample":
                    continue
                total_samples += 1
                if record.get("forced") and not include_forced:
                    forced_samples_skipped += 1
                    continue
                if max_samples is not None and selected_samples >= max_samples:
                    break

                _validate_sample(record, path, line_number)
                episode_key = f"{path.name}:{record['episodeId']}"
                is_validation = _is_validation_episode(
                    episode_key,
                    validation_fraction,
                )
                if is_validation:
                    validation_samples += 1
                else:
                    train_samples += 1
                selected_samples += 1
                if visitor is not None:
                    visitor(is_validation, record)

    return (
        total_samples,
        forced_samples_skipped,
        train_samples,
        validation_samples,
    )


def load_rollouts(
    patterns: Sequence[str],
    *,
    validation_fraction: float = 0.1,
    include_forced: bool = False,
    max_samples: int | None = None,
    supervised_weight: float = 5.0,
) -> LoadedRollouts:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    if not np.isfinite(supervised_weight) or supervised_weight < 1.0:
        raise ValueError("supervised_weight must be at least one")

    paths = expand_input_paths(patterns)
    (
        total_samples,
        forced_samples_skipped,
        train_count,
        validation_count,
    ) = _walk_selected_samples(
        paths,
        validation_fraction=validation_fraction,
        include_forced=include_forced,
        max_samples=max_samples,
        visitor=None,
    )
    if train_count == 0 or validation_count == 0:
        raise ValueError(
            "episode split produced an empty train or validation set; "
            "use more episodes or a different validation fraction"
        )
    train = _allocate_split(train_count)
    validation = _allocate_split(validation_count)
    positions = {False: 0, True: 0}

    def fill(is_validation: bool, record: dict) -> None:
        destination = validation if is_validation else train
        row = positions[is_validation]
        destination.observations[row] = record["observation"]
        destination.legal_masks[row, record["legalActionIndices"]] = True
        destination.actions[row] = (
            record.get("supervisedActionIndex")
            if record.get("supervisedActionIndex") is not None
            else record["actionIndex"]
        )
        destination.weights[row] = (
            supervised_weight
            if record.get("supervisedActionIndex") is not None
            else 1.0
        )
        positions[is_validation] += 1

    _walk_selected_samples(
        paths,
        validation_fraction=validation_fraction,
        include_forced=include_forced,
        max_samples=max_samples,
        visitor=fill,
    )
    return LoadedRollouts(
        train=train,
        validation=validation,
        files=tuple(str(path) for path in paths),
        total_samples=total_samples,
        forced_samples_skipped=forced_samples_skipped,
    )


def iter_action_histogram(split: RolloutSplit) -> Iterable[tuple[int, int]]:
    actions, counts = np.unique(split.actions, return_counts=True)
    return zip(actions.tolist(), counts.tolist(), strict=True)
