from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


OBSERVATION_FEATURES = 172
OBSERVATION_VERSION = 2
ACTION_COUNT = 506
PPO_ROLLOUT_FORMAT = "dalmuti-ppo-ndjson"
PPO_ROLLOUT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PpoRollouts:
    observations: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    rank_auxiliary_rewards: np.ndarray
    effective_rewards: np.ndarray
    terminals: np.ndarray
    forced: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    trajectory_ids: np.ndarray
    files: tuple[str, ...]
    behavior_model_sha256: str
    behavior_temperature: float | None
    trajectory_count: int
    terminal_rank_auxiliary_coefficient: float
    skip_forced_policy_time: bool

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def expand_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.update(path.resolve() for path in matches if path.is_file())
    result = sorted(paths, key=lambda path: str(path).lower())
    if not result:
        raise FileNotFoundError("no PPO rollout files matched")
    return result


def _validate_manifest(
    manifest: dict,
    path: Path,
) -> tuple[str, int | None, float]:
    if manifest.get("type") != "manifest":
        raise ValueError(f"{path}: first record is not a manifest")
    if manifest.get("format") != PPO_ROLLOUT_FORMAT:
        raise ValueError(f"{path}: unsupported PPO rollout format")
    if manifest.get("formatVersion") != PPO_ROLLOUT_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported PPO rollout version")
    if manifest.get("observation", {}).get("featureCount") != OBSERVATION_FEATURES:
        raise ValueError(f"{path}: observation feature count mismatch")
    if manifest.get("observation", {}).get("version") != OBSERVATION_VERSION:
        raise ValueError(f"{path}: observation version mismatch")
    if manifest.get("actionSpace", {}).get("size") != ACTION_COUNT:
        raise ValueError(f"{path}: action count mismatch")
    sha256 = manifest.get("behaviorModel", {}).get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"{path}: behavior model SHA-256 is missing")
    environment = manifest.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError(f"{path}: invalid environment metadata")
    player_count = environment.get("playerCount")
    if player_count is not None and (
        not isinstance(player_count, int) or player_count < 2
    ):
        raise ValueError(f"{path}: invalid environment player count")
    behavior_policy = manifest.get("behaviorPolicy", {})
    if not isinstance(behavior_policy, dict):
        raise ValueError(f"{path}: invalid behavior policy metadata")
    behavior_temperature = behavior_policy.get("temperature")
    if behavior_policy and behavior_policy.get("sampling") != "softmax":
        raise ValueError(f"{path}: unsupported behavior policy sampling")
    if behavior_policy and behavior_temperature is None:
        raise ValueError(f"{path}: behavior policy temperature is missing")
    if not behavior_policy:
        behavior_temperature = 1.0
    if behavior_temperature is not None and (
        not isinstance(behavior_temperature, (float, int))
        or not np.isfinite(behavior_temperature)
        or behavior_temperature <= 0
    ):
        raise ValueError(f"{path}: invalid behavior policy temperature")
    return sha256, player_count, float(behavior_temperature)


def _validate_sample(sample: dict, path: Path, line_number: int) -> None:
    observation = sample.get("observation")
    legal_indices = sample.get("legalActionIndices")
    action = sample.get("actionIndex")
    if (
        not isinstance(observation, list)
        or len(observation) != OBSERVATION_FEATURES
        or not all(np.isfinite(value) for value in observation)
    ):
        raise ValueError(f"{path}:{line_number}: invalid observation")
    if not isinstance(legal_indices, list) or not legal_indices:
        raise ValueError(f"{path}:{line_number}: missing legal actions")
    if (
        len(set(legal_indices)) != len(legal_indices)
        or any(
            not isinstance(index, int) or index < 0 or index >= ACTION_COUNT
            for index in legal_indices
        )
    ):
        raise ValueError(f"{path}:{line_number}: invalid legal actions")
    if not isinstance(action, int) or action not in legal_indices:
        raise ValueError(f"{path}:{line_number}: selected action is illegal")
    old_log_probability = sample.get("oldLogProbability")
    if (
        not isinstance(old_log_probability, (float, int))
        or not np.isfinite(old_log_probability)
        or old_log_probability > 1.0e-6
    ):
        raise ValueError(f"{path}:{line_number}: invalid old log probability")
    for field in ("oldValue", "reward"):
        value = sample.get(field)
        if not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError(f"{path}:{line_number}: invalid {field}")
    if not isinstance(sample.get("terminal"), bool):
        raise ValueError(f"{path}:{line_number}: terminal must be boolean")
    if not isinstance(sample.get("forced"), bool):
        raise ValueError(f"{path}:{line_number}: forced must be boolean")
    if not isinstance(sample.get("trajectoryId"), str):
        raise ValueError(f"{path}:{line_number}: trajectoryId is missing")
    finish_place = sample.get("finishPlace")
    if finish_place is not None and (
        not isinstance(finish_place, int) or finish_place < 1
    ):
        raise ValueError(f"{path}:{line_number}: invalid finishPlace")


def _walk_samples(
    paths: Sequence[Path],
    visitor: Callable[[dict, int | None], None] | None,
) -> tuple[int, str, float | None]:
    sample_count = 0
    behavior_sha256: str | None = None
    behavior_temperature: float | None = None
    temperature_initialized = False
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            first_line = stream.readline()
            if not first_line:
                raise ValueError(f"{path}: empty rollout file")
            file_sha256, player_count, file_temperature = _validate_manifest(
                json.loads(first_line),
                path,
            )
            if behavior_sha256 is None:
                behavior_sha256 = file_sha256
            elif behavior_sha256 != file_sha256:
                raise ValueError("PPO files use different behavior models")
            if not temperature_initialized:
                behavior_temperature = file_temperature
                temperature_initialized = True
            elif behavior_temperature != file_temperature:
                raise ValueError(
                    "PPO files use different behavior policy temperatures"
                )
            expected_policy_version = f"sha256:{file_sha256}"
            for line_number, line in enumerate(stream, start=2):
                record = json.loads(line)
                if record.get("type") != "sample":
                    continue
                _validate_sample(record, path, line_number)
                if record.get("policyVersion") != expected_policy_version:
                    raise ValueError(
                        f"{path}:{line_number}: policy version mismatch"
                    )
                sample_count += 1
                if visitor is not None:
                    visitor(record, player_count)
    if sample_count < 1 or behavior_sha256 is None:
        raise ValueError("PPO rollout contains no samples")
    return sample_count, behavior_sha256, behavior_temperature


def load_ppo_rollouts(
    patterns: Sequence[str],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    skip_forced_policy_time: bool = False,
    terminal_rank_auxiliary_coefficient: float = 0.0,
) -> PpoRollouts:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be between zero and one")
    if not isinstance(skip_forced_policy_time, bool):
        raise TypeError("skip_forced_policy_time must be boolean")
    if not np.isfinite(terminal_rank_auxiliary_coefficient):
        raise ValueError(
            "terminal_rank_auxiliary_coefficient must be finite"
        )
    paths = expand_input_paths(patterns)
    sample_count, behavior_sha256, behavior_temperature = _walk_samples(
        paths,
        None,
    )
    observations = np.empty(
        (sample_count, OBSERVATION_FEATURES),
        dtype=np.float32,
    )
    legal_masks = np.zeros((sample_count, ACTION_COUNT), dtype=np.bool_)
    actions = np.empty(sample_count, dtype=np.int64)
    old_log_probabilities = np.empty(sample_count, dtype=np.float32)
    old_values = np.empty(sample_count, dtype=np.float32)
    rewards = np.empty(sample_count, dtype=np.float32)
    rank_auxiliary_rewards = np.zeros(sample_count, dtype=np.float32)
    terminals = np.empty(sample_count, dtype=np.bool_)
    forced = np.empty(sample_count, dtype=np.bool_)
    trajectory_ids = np.empty(sample_count, dtype=np.int32)
    trajectory_indices: dict[str, int] = {}
    terminal_counts: list[int] = []
    position = 0

    def fill(record: dict, player_count: int | None) -> None:
        nonlocal position
        observations[position] = record["observation"]
        legal_masks[position, record["legalActionIndices"]] = True
        actions[position] = record["actionIndex"]
        old_log_probabilities[position] = record["oldLogProbability"]
        old_values[position] = record["oldValue"]
        rewards[position] = record["reward"]
        terminals[position] = record["terminal"]
        forced[position] = record["forced"]
        if terminal_rank_auxiliary_coefficient != 0.0:
            finish_place = record.get("finishPlace")
            if player_count is None:
                raise ValueError(
                    "terminal rank auxiliary reward requires manifest "
                    "environment.playerCount"
                )
            if (
                not isinstance(finish_place, int)
                or finish_place < 1
                or finish_place > player_count
            ):
                raise ValueError(
                    "terminal rank auxiliary reward requires a valid "
                    "finishPlace on every sample"
                )
            if terminals[position]:
                normalized_rank = (
                    1.0
                    - 2.0
                    * (finish_place - 1)
                    / (player_count - 1)
                )
                rank_auxiliary_rewards[position] = (
                    terminal_rank_auxiliary_coefficient
                    * normalized_rank
                )
        trajectory_key = record["trajectoryId"]
        if trajectory_key not in trajectory_indices:
            trajectory_indices[trajectory_key] = len(trajectory_indices)
            terminal_counts.append(0)
        trajectory_id = trajectory_indices[trajectory_key]
        trajectory_ids[position] = trajectory_id
        if terminals[position]:
            terminal_counts[trajectory_id] += 1
        position += 1

    second_count, second_sha256, second_temperature = _walk_samples(
        paths,
        fill,
    )
    if (
        second_count != sample_count
        or second_sha256 != behavior_sha256
        or second_temperature != behavior_temperature
    ):
        raise RuntimeError("PPO rollout changed while it was being loaded")
    if any(count != 1 for count in terminal_counts):
        raise ValueError("every trajectory must contain exactly one terminal")

    trajectory_count = len(trajectory_indices)
    effective_rewards = rewards + rank_auxiliary_rewards
    value_advantages = np.empty(sample_count, dtype=np.float32)
    next_advantages = np.zeros(trajectory_count, dtype=np.float64)
    next_values = np.zeros(trajectory_count, dtype=np.float64)
    terminal_seen = np.zeros(trajectory_count, dtype=np.bool_)
    for index in range(sample_count - 1, -1, -1):
        trajectory_id = trajectory_ids[index]
        if terminals[index]:
            next_advantage = 0.0
            next_value = 0.0
            terminal_seen[trajectory_id] = True
            nonterminal = 0.0
        else:
            if not terminal_seen[trajectory_id]:
                raise ValueError("trajectory contains samples after its terminal")
            next_advantage = next_advantages[trajectory_id]
            next_value = next_values[trajectory_id]
            nonterminal = 1.0
        delta = (
            float(effective_rewards[index])
            + gamma * next_value * nonterminal
            - float(old_values[index])
        )
        advantage = (
            delta
            + gamma * gae_lambda * next_advantage * nonterminal
        )
        value_advantages[index] = advantage
        next_advantages[trajectory_id] = advantage
        next_values[trajectory_id] = old_values[index]
    returns = value_advantages + old_values
    advantages = value_advantages.copy()
    if skip_forced_policy_time:
        next_policy_advantages = np.zeros(
            trajectory_count,
            dtype=np.float64,
        )
        next_policy_values = np.zeros(
            trajectory_count,
            dtype=np.float64,
        )
        has_next_policy_step = np.zeros(
            trajectory_count,
            dtype=np.bool_,
        )
        pending_rewards = np.zeros(
            trajectory_count,
            dtype=np.float64,
        )
        for index in range(sample_count - 1, -1, -1):
            trajectory_id = trajectory_ids[index]
            if terminals[index]:
                next_policy_advantages[trajectory_id] = 0.0
                next_policy_values[trajectory_id] = 0.0
                has_next_policy_step[trajectory_id] = False
                pending_rewards[trajectory_id] = 0.0
            if forced[index]:
                pending_rewards[trajectory_id] += float(
                    effective_rewards[index]
                )
                continue
            reward = (
                float(effective_rewards[index])
                + pending_rewards[trajectory_id]
            )
            pending_rewards[trajectory_id] = 0.0
            nonterminal = float(has_next_policy_step[trajectory_id])
            delta = (
                reward
                + gamma
                * next_policy_values[trajectory_id]
                * nonterminal
                - float(old_values[index])
            )
            advantage = (
                delta
                + gamma
                * gae_lambda
                * next_policy_advantages[trajectory_id]
                * nonterminal
            )
            advantages[index] = advantage
            next_policy_advantages[trajectory_id] = advantage
            next_policy_values[trajectory_id] = old_values[index]
            has_next_policy_step[trajectory_id] = True
    return PpoRollouts(
        observations=observations,
        legal_masks=legal_masks,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        old_values=old_values,
        rewards=rewards,
        rank_auxiliary_rewards=rank_auxiliary_rewards,
        effective_rewards=effective_rewards,
        terminals=terminals,
        forced=forced,
        advantages=advantages,
        returns=returns,
        trajectory_ids=trajectory_ids,
        files=tuple(str(path) for path in paths),
        behavior_model_sha256=behavior_sha256,
        behavior_temperature=behavior_temperature,
        trajectory_count=trajectory_count,
        terminal_rank_auxiliary_coefficient=(
            terminal_rank_auxiliary_coefficient
        ),
        skip_forced_policy_time=skip_forced_policy_time,
    )
