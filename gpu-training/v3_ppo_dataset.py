from __future__ import annotations

import glob
import hashlib
import json
import math
import multiprocessing
import re
import stat
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from ppo_dataset import PpoRollouts
from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_COUNT,
    V3_ACTION_FEATURE_COUNT,
    V3_ACTION_FEATURE_LAYOUT,
    V3_ACTION_FEATURES,
    load_v3_action_conditioned_json,
)


OBSERVATION_FEATURES = 172
OBSERVATION_VERSION = 2
V3_PPO_ROLLOUT_FORMAT = "dalmuti-v3-ppo-ndjson"
V3_PPO_ROLLOUT_FORMAT_VERSION = 1
V3_LEGAL_MASK_HEX_LENGTH = V3_ACTION_COUNT // 4
MAX_JS_SAFE_INTEGER = 9_007_199_254_740_991
MAX_TRANSITIONS_PER_ACT = 20_000
FLOAT32_MAX = float(np.finfo(np.float32).max)
TRAINING_ROLES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)

V3_PPO_SEMANTICS_CONTRACT = {
    "format": V3_PPO_ROLLOUT_FORMAT,
    "formatVersion": V3_PPO_ROLLOUT_FORMAT_VERSION,
    "sourceProvenance": {
        "createdAt": "UTC ISO-8601 milliseconds generated at collection",
        "initialSeed": (
            "positive JavaScript-safe integer; one-based episode n uses "
            "initialSeed + n - 1"
        ),
        "episodeId": (
            "v3-league-p{playerCount}-episode-{one-based episode number}"
        ),
        "environmentDecisions": (
            "exact full simulator card-play decision count including all seats; "
            "learner-only samples provide a checked lower bound"
        ),
    },
    "environment": {
        "game": "DALMUTI",
        "rules": "project-house-rules-v1",
        "rolloutMode": "league",
        "learnerSeats": (
            "approximately half; only behavior-model decisions are samples"
        ),
        "nonCardDecisions": "normal bot policy",
        "maximumTransitionsPerAct": MAX_TRANSITIONS_PER_ACT,
    },
    "observation": {
        "version": OBSERVATION_VERSION,
        "featureCount": OBSERVATION_FEATURES,
        "privacy": (
            "own private hand plus public state only; opponent hands excluded"
        ),
        "encoder": "training/observation.ts encodeTrainingObservation V2",
    },
    "sample": {
        "trajectoryId": "{episodeId}:round-{round}:{actorId}",
        "terminal": "last card-play decision by that actor in that act",
        "finishPlace": "final one-based finish order in that act",
        "forced": "true exactly when legalActionIndices has length one",
        "policyVersion": "sha256:{behavior model file SHA-256}",
    },
    "reward": {
        "manifestExpression": (
            "actorTerminal ? (roundChipAward - 2) / 2 : 0"
        ),
        "nonTerminal": 0,
        "terminalFormula": "(roundChipAward(finishPlace, playerCount) - 2) / 2",
        "roundChipAward": {
            "first": 4,
            "second": 3,
            "penultimate": 1,
            "last": 0,
            "otherwise": 2,
            "precedence": [
                "first",
                "second",
                "penultimate",
                "last",
                "otherwise",
            ],
        },
    },
    "summary": {
        "learnerSamples": "exact sample-record count",
        "forcedSamples": "exact forced=true sample count",
        "nonForcedSamples": "exact forced=false sample count",
        "episodes": "exact distinct sequential episode count",
        "environmentDecisions": (
            "source-declared full-decision count, bounded by sampled step indices "
            "and maximumTransitionsPerAct"
        ),
        "opponentSeatAssignments": (
            "exact configured non-learner seat total across episodes"
        ),
    },
}


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# This literal also appears in v3-ppo-schema.json. Changing any semantic above
# requires an intentional contract-version/hash update in both places.
V3_PPO_SEMANTICS_CONTRACT_SHA256 = (
    "d7d249a24153ecc204add53f3d3ab352fabfd5ab175001b51f8ed2ba1296e275"
)
if (
    _canonical_json_sha256(V3_PPO_SEMANTICS_CONTRACT)
    != V3_PPO_SEMANTICS_CONTRACT_SHA256
):
    raise RuntimeError("V3 PPO semantics contract SHA-256 is stale")


@dataclass(frozen=True)
class _ManifestContract:
    behavior_sha256: str
    player_count: int
    temperature: float
    episodes: int
    acts_per_episode: int
    initial_seed: int
    opponent_model_sha256: tuple[str, ...]
    collection_mode: str
    target_non_forced_decisions: int | None
    max_episodes: int


@dataclass(frozen=True)
class _WalkResult:
    samples: int
    behavior_sha256: str
    temperature: float
    source_content_sha256: tuple[str, ...]
    file_samples: tuple[int, ...]
    file_trajectory_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _ParallelFillTask:
    path: str
    offset: int
    samples: int
    trajectory_offset: int
    expected_trajectory_ids: tuple[str, ...]
    array_paths: tuple[tuple[str, str], ...]
    terminal_rank_auxiliary_coefficient: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_strict_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        literal = Path(pattern)
        if not matches and literal.is_file():
            matches = [literal]
        for candidate in matches:
            if candidate.is_symlink():
                raise ValueError(f"V3 PPO source must not be a symlink: {candidate}")
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            metadata = resolved.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"V3 PPO source must be a regular file: {candidate}"
                )
            paths.add(resolved)
    result = sorted(paths, key=lambda path: str(path).lower())
    if not result:
        raise FileNotFoundError("no V3 PPO rollout files matched")
    return result


def _source_file_binding(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"V3 PPO source must not be a symlink: {path}")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"V3 PPO source must be a regular file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    after = path.lstat()
    if (
        not stat.S_ISREG(after.st_mode)
        or before.st_size != byte_count
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise RuntimeError(f"V3 PPO source changed while hashing: {path}")
    return {
        "path": str(path),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _exact_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{label}: exact fields do not match; expected {sorted(expected)}, "
            f"got {sorted(actual) if isinstance(actual, set) else actual}"
        )
    return value


def _strict_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_JS_SAFE_INTEGER,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(
            f"{label}: must be an integer from {minimum} to {maximum}"
        )
    return value


def _strict_finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{label}: must be a finite number")
    return float(value)


def _lowercase_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{label}: must be a lowercase SHA-256")
    return value


def _binary_feature(value: object, label: str) -> int:
    if isinstance(value, bool) or value not in (0, 0.0, 1, 1.0):
        raise ValueError(f"{label}: feature must be canonical zero or one")
    return int(value)


def _role_for_index(index: int, player_count: int) -> str:
    if index == 0:
        return "great-dalmuti"
    if index == 1:
        return "lesser-dalmuti"
    if index == player_count - 2:
        return "lesser-peon"
    if index == player_count - 1:
        return "great-peon"
    return "merchant"


def _round_chip_award(place: int, player_count: int) -> int:
    if place == 1:
        return 4
    if place == 2:
        return 3
    if place == player_count - 1:
        return 1
    if place == player_count:
        return 0
    return 2


def _expected_reward(terminal: bool, finish_place: int, player_count: int) -> float:
    if not terminal:
        return 0.0
    return (_round_chip_award(finish_place, player_count) - 2) / 2


def _validate_created_at(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value)
        is None
    ):
        raise ValueError(f"{label}: createdAt must be UTC ISO-8601 milliseconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise ValueError(f"{label}: invalid createdAt timestamp") from error


def decode_legal_mask_hex(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, str)
        or len(value) != V3_LEGAL_MASK_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{label}: legalMaskHex must be {V3_LEGAL_MASK_HEX_LENGTH} "
            "lowercase hex digits"
        )
    result: list[int] = []
    for nibble_index, character in enumerate(value):
        nibble = int(character, 16)
        for bit in range(4):
            if nibble & (1 << bit):
                result.append(nibble_index * 4 + bit)
    if not result:
        raise ValueError(f"{label}: legal mask is empty")
    return result


def _integer_feature(
    value: object, scale: int, maximum: int, label: str
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: feature must be numeric")
    decoded = int(round(float(value) * scale))
    if (
        decoded < 0
        or decoded > maximum
        or not np.isclose(value, decoded / scale, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError(f"{label}: feature is not a canonical encoded count")
    return decoded


def legal_actions_from_observation(
    observation: Sequence[float], label: str
) -> list[int]:
    """Reconstruct exact card-play legality from observation schema V2."""
    table_present = _integer_feature(observation[8], 1, 1, label)
    table_rank_bits = observation[9:22]
    if any(value not in (0, 0.0, 1, 1.0) for value in table_rank_bits):
        raise ValueError(f"{label}: table rank one-hot is non-canonical")
    active_ranks = [
        index + 1 for index, value in enumerate(table_rank_bits) if value == 1
    ]
    if (table_present == 1 and len(active_ranks) != 1) or (
        table_present == 0 and active_ranks
    ):
        raise ValueError(f"{label}: table rank one-hot contradicts table presence")
    table_rank = active_ranks[0] if active_ranks else None
    table_count = _integer_feature(observation[22], 14, 14, label)
    if (table_present == 0 and table_count != 0) or (
        table_present == 1 and table_count < 1
    ):
        raise ValueError(f"{label}: table count contradicts table presence")
    hand_counts = []
    for rank, value in enumerate(observation[23:36], start=1):
        copies = 2 if rank == 13 else rank
        hand_counts.append(_integer_feature(value, copies, copies, label))
    joker_count = hand_counts[12]
    legal: list[int] = []
    if table_present:
        legal.append(0)
    elif joker_count > 0:
        legal.append(1)
    for rank in range(1, 13):
        if table_rank is not None and rank >= table_rank:
            continue
        natural_in_hand = hand_counts[rank - 1]
        for jokers in range(joker_count + 1):
            minimum_natural = table_count - jokers if table_present else 1
            maximum_natural = (
                table_count - jokers if table_present else natural_in_hand
            )
            for naturals in range(minimum_natural, maximum_natural + 1):
                if naturals < 1 or naturals > natural_in_hand:
                    continue
                total = naturals + jokers
                if table_present and total != table_count:
                    continue
                first_rank_index = 2 + (3 * (rank - 1) * rank) // 2
                legal.append(first_rank_index + (naturals - 1) * 3 + jokers)
    result = sorted(set(legal))
    if not result:
        raise ValueError(f"{label}: observation yields no legal actions")
    return result


def _validate_manifest(manifest: dict, path: Path) -> _ManifestContract:
    label = str(path)
    _exact_keys(
        manifest,
        {
            "type",
            "format",
            "formatVersion",
            "createdAt",
            "environment",
            "behaviorModel",
            "behaviorPolicy",
            "observation",
            "actionSpace",
            "sampleBindings",
        },
        f"{label}: manifest",
    )
    if (
        manifest["type"] != "manifest"
        or manifest["format"] != V3_PPO_ROLLOUT_FORMAT
        or manifest["formatVersion"] != V3_PPO_ROLLOUT_FORMAT_VERSION
    ):
        raise ValueError(f"{path}: unsupported V3 PPO rollout manifest")
    _validate_created_at(manifest["createdAt"], label)

    observation = manifest["observation"]
    if observation != {
        "version": OBSERVATION_VERSION,
        "featureCount": OBSERVATION_FEATURES,
        "privacy": "own private hand plus public state only; opponent hands excluded",
    }:
        raise ValueError(f"{path}: observation contract mismatch")

    expected_action_space = {
        "catalogueVersion": V3_ACTION_CATALOGUE_VERSION,
        "size": V3_ACTION_COUNT,
        "catalogue": [dict(action) for action in V3_ACTION_CATALOGUE],
        "actionFeatures": V3_ACTION_FEATURE_COUNT,
        "actionFeatureLayout": list(V3_ACTION_FEATURE_LAYOUT),
        "encodedActionFeatures": [list(features) for features in V3_ACTION_FEATURES],
        "legalMaskEncoding": {
            "field": "legalMaskHex",
            "lowercaseHexDigits": V3_LEGAL_MASK_HEX_LENGTH,
            "bitOrder": (
                "action index i = bit (i % 4) of hex digit floor(i / 4)"
            ),
        },
    }
    if manifest["actionSpace"] != expected_action_space:
        raise ValueError(f"{path}: V3 action catalogue contract mismatch")

    behavior = _exact_keys(
        manifest["behaviorModel"],
        {
            "sha256",
            "format",
            "observationSchemaVersion",
            "observationFeatures",
            "actionCatalogueVersion",
        },
        f"{label}: behaviorModel",
    )
    sha256 = _lowercase_sha256(
        behavior["sha256"], f"{label}: behaviorModel.sha256"
    )
    if behavior != {
        "sha256": sha256,
        "format": "dalmuti-action-conditioned-actor-critic",
        "observationSchemaVersion": OBSERVATION_VERSION,
        "observationFeatures": OBSERVATION_FEATURES,
        "actionCatalogueVersion": V3_ACTION_CATALOGUE_VERSION,
    }:
        raise ValueError(f"{path}: V3 behavior model contract mismatch")

    expected_bindings = {
        "observationSchemaVersion": OBSERVATION_VERSION,
        "actionCatalogueVersion": V3_ACTION_CATALOGUE_VERSION,
        "policyVersion": f"sha256:{sha256}",
        "legalActionIndices": (
            "unique ascending indices exactly equal to legalMaskHex"
        ),
        "forced": "true exactly when legalActionIndices has length one",
    }
    if manifest["sampleBindings"] != expected_bindings:
        raise ValueError(f"{path}: sample binding contract mismatch")

    policy = _exact_keys(
        manifest["behaviorPolicy"],
        {"sampling", "temperature", "logProbabilityBinding"},
        f"{label}: behaviorPolicy",
    )
    temperature = _strict_finite_number(
        policy["temperature"], f"{label}: behaviorPolicy.temperature"
    )
    if (
        policy["sampling"] != "softmax"
        or temperature < 0.05
        or temperature > 10.0
        or policy["logProbabilityBinding"]
        != (
            "recomputed from behavior model over exactly legalMaskHex at this "
            "temperature"
        )
    ):
        raise ValueError(f"{path}: invalid behavior policy contract")

    environment = _exact_keys(
        manifest["environment"],
        {
            "game",
            "rules",
            "playerCount",
            "actsPerEpisode",
            "episodes",
            "initialSeed",
            "rolloutMode",
            "opponentPolicies",
            "nonCardDecisions",
            "reward",
            "learnerSeats",
            "opponentMix",
            "collection",
        },
        f"{label}: environment",
    )
    if (
        environment["game"] != "DALMUTI"
        or environment["rules"] != "project-house-rules-v1"
        or environment["rolloutMode"] != "league"
        or environment["nonCardDecisions"] != "normal bot policy"
        or environment["reward"]
        != "actorTerminal ? (roundChipAward - 2) / 2 : 0"
        or environment["learnerSeats"]
        != "approximately half; only behavior-model decisions are samples"
    ):
        raise ValueError(f"{path}: environment semantics contract mismatch")
    player_count = _strict_integer(
        environment["playerCount"],
        f"{label}: environment.playerCount",
        minimum=4,
        maximum=10,
    )
    acts_per_episode = _strict_integer(
        environment["actsPerEpisode"],
        f"{label}: environment.actsPerEpisode",
        minimum=1,
    )
    episodes = _strict_integer(
        environment["episodes"],
        f"{label}: environment.episodes",
        minimum=1,
    )
    initial_seed = _strict_integer(
        environment["initialSeed"],
        f"{label}: environment.initialSeed",
        minimum=1,
    )
    if initial_seed + episodes - 1 > MAX_JS_SAFE_INTEGER:
        raise ValueError(f"{path}: episode seed range exceeds JavaScript safety")

    opponent_policies = environment["opponentPolicies"]
    if (
        not isinstance(opponent_policies, list)
        or not opponent_policies
        or opponent_policies[0] != "normal"
    ):
        raise ValueError(f"{path}: invalid opponentPolicies provenance")
    opponent_hashes: list[str] = []
    for index, policy_name in enumerate(opponent_policies[1:], start=1):
        if not isinstance(policy_name, str) or not policy_name.startswith("sha256:"):
            raise ValueError(
                f"{path}: opponentPolicies[{index}] must be a SHA-bound model"
            )
        opponent_hashes.append(
            _lowercase_sha256(
                policy_name.removeprefix("sha256:"),
                f"{label}: opponentPolicies[{index}]",
            )
        )
    if (
        len(set(opponent_hashes)) != len(opponent_hashes)
        or sha256 in opponent_hashes
    ):
        raise ValueError(f"{path}: opponent model hashes are not unique")

    opponent_mix = _exact_keys(
        environment["opponentMix"],
        {
            "normalFraction",
            "trainedModelFraction",
            "trainedModelSelection",
            "trainedModels",
        },
        f"{label}: opponentMix",
    )
    normal_fraction = _strict_finite_number(
        opponent_mix["normalFraction"], f"{label}: opponentMix.normalFraction"
    )
    trained_fraction = _strict_finite_number(
        opponent_mix["trainedModelFraction"],
        f"{label}: opponentMix.trainedModelFraction",
    )
    expected_trained_models = [{"sha256": value} for value in opponent_hashes]
    if (
        not 0.0 <= normal_fraction <= 1.0
        or not 0.0 <= trained_fraction <= 1.0
        or not np.isclose(
            normal_fraction + trained_fraction, 1.0, rtol=0.0, atol=1.0e-12
        )
        or opponent_mix["trainedModelSelection"] != "uniform"
        or opponent_mix["trainedModels"] != expected_trained_models
        or (not opponent_hashes and (normal_fraction != 1.0 or trained_fraction != 0.0))
    ):
        raise ValueError(f"{path}: opponentMix contract mismatch")

    collection = environment["collection"]
    if not isinstance(collection, dict):
        raise ValueError(f"{path}: invalid collection provenance")
    collection_mode = collection.get("mode")
    target_non_forced_decisions: int | None
    max_episodes: int
    if collection_mode == "fixed-episodes":
        _exact_keys(
            collection,
            {"mode", "requestedEpisodes"},
            f"{label}: collection",
        )
        requested = _strict_integer(
            collection["requestedEpisodes"],
            f"{label}: collection.requestedEpisodes",
            minimum=1,
        )
        if requested != episodes:
            raise ValueError(f"{path}: fixed episode count does not match manifest")
        target_non_forced_decisions = None
        max_episodes = episodes
    elif collection_mode == "target-non-forced-decisions":
        _exact_keys(
            collection,
            {"mode", "targetNonForcedDecisions", "maxEpisodes"},
            f"{label}: collection",
        )
        target_non_forced_decisions = _strict_integer(
            collection["targetNonForcedDecisions"],
            f"{label}: collection.targetNonForcedDecisions",
            minimum=1,
        )
        max_episodes = _strict_integer(
            collection["maxEpisodes"],
            f"{label}: collection.maxEpisodes",
            minimum=1,
        )
        if episodes > max_episodes:
            raise ValueError(f"{path}: actual episodes exceed maxEpisodes")
    else:
        raise ValueError(f"{path}: unsupported collection mode")

    return _ManifestContract(
        behavior_sha256=sha256,
        player_count=player_count,
        temperature=temperature,
        episodes=episodes,
        acts_per_episode=acts_per_episode,
        initial_seed=initial_seed,
        opponent_model_sha256=tuple(opponent_hashes),
        collection_mode=collection_mode,
        target_non_forced_decisions=target_non_forced_decisions,
        max_episodes=max_episodes,
    )


def _validate_observation(
    observation: Sequence[float],
    record: dict,
    contract: _ManifestContract,
    label: str,
) -> None:
    player_count = contract.player_count
    round_number = record["round"]
    actor_seat = record["actorSeat"]
    actor_role = record["actorRole"]

    if _integer_feature(observation[0], 6, 6, label) != player_count - 4:
        raise ValueError(f"{label}: observation player count mismatch")
    expected_round = min(round_number / 20, 1.0)
    if not np.isclose(observation[1], expected_round, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{label}: observation act number mismatch")
    if (
        _integer_feature(observation[2], player_count - 1, player_count - 1, label)
        != actor_seat
    ):
        raise ValueError(f"{label}: observation actor seat mismatch")

    actor_role_bits = [
        _binary_feature(value, f"{label}: actor role")
        for value in observation[3:8]
    ]
    expected_actor_role = TRAINING_ROLES.index(actor_role)
    if actor_role_bits != [
        int(index == expected_actor_role) for index in range(len(TRAINING_ROLES))
    ]:
        raise ValueError(f"{label}: observation actor role mismatch")

    table_present = _binary_feature(observation[8], f"{label}: table presence")
    table_rank_bits = [
        _binary_feature(value, f"{label}: table rank")
        for value in observation[9:22]
    ]
    active_table_ranks = [
        index + 1 for index, value in enumerate(table_rank_bits) if value
    ]
    if (table_present == 1 and len(active_table_ranks) != 1) or (
        table_present == 0 and active_table_ranks
    ):
        raise ValueError(f"{label}: table rank contradicts table presence")
    table_count = _integer_feature(observation[22], 14, 14, label)
    if (table_present == 0 and table_count != 0) or (
        table_present == 1 and table_count < 1
    ):
        raise ValueError(f"{label}: table count contradicts table presence")

    own_hand_counts: list[int] = []
    public_played_counts: list[int] = []
    for rank, value in enumerate(observation[23:36], start=1):
        copies = 2 if rank == 13 else rank
        own_hand_counts.append(_integer_feature(value, copies, copies, label))
    for rank, value in enumerate(observation[36:49], start=1):
        copies = 2 if rank == 13 else rank
        public_played_counts.append(_integer_feature(value, copies, copies, label))
    if any(
        hand + played > (2 if rank == 13 else rank)
        for rank, (hand, played) in enumerate(
            zip(own_hand_counts, public_played_counts), start=1
        )
    ):
        raise ValueError(f"{label}: own and publicly played cards exceed the deck")

    relative_hand_counts: list[int] = []
    table_leader_count = 0
    for slot in range(10):
        offset = 49 + slot * 12
        values = observation[offset : offset + 12]
        occupied = _binary_feature(values[0], f"{label}: relative slot {slot}")
        if slot >= player_count:
            if occupied != 0 or any(value != 0 and value != 0.0 for value in values):
                raise ValueError(f"{label}: unused relative player slot is not zero")
            continue
        if occupied != 1:
            raise ValueError(f"{label}: occupied relative player slot is missing")
        hand_count = _integer_feature(values[1], 20, 20, label)
        finished = _binary_feature(values[2], f"{label}: finished flag")
        passed = _binary_feature(values[3], f"{label}: passed flag")
        self_flag = _binary_feature(values[4], f"{label}: self flag")
        table_leader = _binary_feature(values[5], f"{label}: table leader flag")
        if self_flag != int(slot == 0):
            raise ValueError(f"{label}: relative self slot is non-canonical")
        if finished != int(hand_count == 0):
            raise ValueError(f"{label}: finished flag contradicts public hand count")
        if (passed and (finished or self_flag or table_leader)):
            raise ValueError(f"{label}: impossible passed-player flags")
        table_leader_count += table_leader
        score_feature = _strict_finite_number(
            values[6], f"{label}: relative player score"
        )
        if score_feature < 0.0 or score_feature >= 1.0:
            raise ValueError(f"{label}: relative player score is out of range")
        decoded_score = int(round(math.atanh(score_feature) * 10))
        if (
            decoded_score < 0
            or decoded_score > 4 * (round_number - 1)
            or not np.isclose(
                score_feature,
                math.tanh(decoded_score / 10),
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            raise ValueError(f"{label}: relative player score is non-canonical")
        role_bits = [
            _binary_feature(value, f"{label}: relative role")
            for value in values[7:12]
        ]
        absolute_seat = (actor_seat + slot) % player_count
        expected_role = _role_for_index(absolute_seat, player_count)
        expected_role_index = TRAINING_ROLES.index(expected_role)
        if role_bits != [
            int(index == expected_role_index)
            for index in range(len(TRAINING_ROLES))
        ]:
            raise ValueError(f"{label}: relative player role order mismatch")
        relative_hand_counts.append(hand_count)
    if table_leader_count != table_present:
        raise ValueError(f"{label}: table leader count contradicts table presence")
    if relative_hand_counts[0] != sum(own_hand_counts):
        raise ValueError(f"{label}: private and public actor hand counts disagree")
    if relative_hand_counts[0] < 1:
        raise ValueError(f"{label}: acting player has no cards")

    revolution = [
        _binary_feature(value, f"{label}: revolution state")
        for value in observation[169:172]
    ]
    if sum(revolution) != 1:
        raise ValueError(f"{label}: revolution state is not one-hot")


def _validate_sample(
    record: dict,
    path: Path,
    line_number: int,
    contract: _ManifestContract,
) -> tuple[int, int, int, str]:
    label = f"{path}:{line_number}"
    _exact_keys(
        record,
        {
            "type",
            "trajectoryId",
            "episodeId",
            "round",
            "step",
            "actorId",
            "actorSeat",
            "actorRole",
            "observationSchemaVersion",
            "actionCatalogueVersion",
            "observation",
            "legalActionIndices",
            "legalMaskHex",
            "actionIndex",
            "oldLogProbability",
            "oldValue",
            "reward",
            "terminal",
            "forced",
            "finishPlace",
            "policyVersion",
        },
        label,
    )
    if record["type"] != "sample":
        raise ValueError(f"{label}: unsupported record type")
    expected_policy_version = f"sha256:{contract.behavior_sha256}"
    if (
        record.get("observationSchemaVersion") != OBSERVATION_VERSION
        or record.get("actionCatalogueVersion")
        != V3_ACTION_CATALOGUE_VERSION
        or record.get("policyVersion") != expected_policy_version
    ):
        raise ValueError(f"{label}: sample contract binding mismatch")

    episode_id = record["episodeId"]
    episode_pattern = re.fullmatch(
        rf"v3-league-p{contract.player_count}-episode-([1-9]\d*)",
        episode_id if isinstance(episode_id, str) else "",
    )
    if episode_pattern is None:
        raise ValueError(f"{label}: episodeId provenance mismatch")
    episode_number = int(episode_pattern.group(1))
    if episode_number > contract.episodes:
        raise ValueError(f"{label}: episode number exceeds manifest")
    round_number = _strict_integer(
        record["round"], label, minimum=1, maximum=contract.acts_per_episode
    )
    step = _strict_integer(record["step"], label)
    actor_id = record["actorId"]
    actor_match = re.fullmatch(
        r"player-([1-9]\d*)", actor_id if isinstance(actor_id, str) else ""
    )
    if actor_match is None or int(actor_match.group(1)) > contract.player_count:
        raise ValueError(f"{label}: invalid actorId")
    actor_seat = _strict_integer(
        record["actorSeat"], label, maximum=contract.player_count - 1
    )
    actor_role = record["actorRole"]
    if actor_role != _role_for_index(actor_seat, contract.player_count):
        raise ValueError(f"{label}: actorRole does not match actorSeat")
    expected_trajectory_id = (
        f"{episode_id}:round-{round_number}:{actor_id}"
    )
    if record["trajectoryId"] != expected_trajectory_id:
        raise ValueError(f"{label}: trajectoryId provenance mismatch")

    observation = record.get("observation")
    if (
        not isinstance(observation, list)
        or len(observation) != OBSERVATION_FEATURES
        or not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and np.isfinite(value)
            for value in observation
        )
    ):
        raise ValueError(f"{label}: invalid observation")
    _validate_observation(observation, record, contract, label)
    legal = record.get("legalActionIndices")
    if (
        not isinstance(legal, list)
        or not legal
        or legal != sorted(set(legal))
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= V3_ACTION_COUNT
            for index in legal
        )
    ):
        raise ValueError(f"{label}: invalid legal action indices")
    if decode_legal_mask_hex(record.get("legalMaskHex"), label) != legal:
        raise ValueError(f"{label}: legalMaskHex does not match legal actions")
    if legal_actions_from_observation(observation, label) != legal:
        raise ValueError(
            f"{label}: legal actions do not match the encoded observation"
        )
    action = record.get("actionIndex")
    if isinstance(action, bool) or not isinstance(action, int) or action not in legal:
        raise ValueError(f"{label}: selected action is illegal")
    if record.get("forced") is not (len(legal) == 1):
        raise ValueError(f"{label}: forced flag does not match legal mask")
    for field in ("oldLogProbability", "oldValue", "reward"):
        value = record.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or abs(float(value)) > FLOAT32_MAX
        ):
            raise ValueError(f"{label}: invalid {field} for float32 storage")
    if record["oldLogProbability"] > 1.0e-8:
        raise ValueError(f"{label}: old log probability must be <= 0")
    if not isinstance(record.get("terminal"), bool):
        raise ValueError(f"{label}: terminal must be boolean")
    finish_place = record.get("finishPlace")
    finish_place = _strict_integer(
        finish_place, label, minimum=1, maximum=contract.player_count
    )
    expected_reward = _expected_reward(
        record["terminal"], finish_place, contract.player_count
    )
    if not np.isclose(
        record["reward"], expected_reward, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            f"{label}: reward does not match terminal, finishPlace, and playerCount"
        )
    return episode_number, round_number, step, actor_id


def _walk_samples(
    paths: Sequence[Path],
    visitor: Callable[[dict, int], None] | None,
) -> _WalkResult:
    total = 0
    shared_sha256: str | None = None
    shared_temperature: float | None = None
    trajectory_source_paths: dict[str, Path] = {}
    source_content_sha256: list[str] = []
    file_sample_counts: list[int] = []
    file_trajectory_ids: list[tuple[str, ...]] = []
    for path in paths:
        file_trajectory_order: list[str] = []
        with path.open("r", encoding="utf-8") as stream:
            first = stream.readline()
            if not first:
                raise ValueError(f"{path}: empty V3 rollout")
            manifest = json.loads(first)
            contract = _validate_manifest(manifest, path)
            if shared_sha256 is None:
                shared_sha256 = contract.behavior_sha256
                shared_temperature = contract.temperature
            elif (
                shared_sha256 != contract.behavior_sha256
                or shared_temperature != contract.temperature
            ):
                raise ValueError("V3 PPO files use different behavior bindings")
            file_samples = 0
            forced = 0
            non_forced = 0
            summary: dict | None = None
            episode_numbers: set[int] = set()
            group_actors: dict[tuple[int, int], set[str]] = {}
            group_steps: dict[tuple[int, int], set[int]] = {}
            group_finish_places: dict[tuple[int, int], dict[str, int]] = {}
            group_revolution: dict[tuple[int, int], tuple[int, int, int]] = {}
            group_max_steps: dict[tuple[int, int], int] = {}
            trajectory_states: dict[str, dict[str, object]] = {}
            last_stream_key: tuple[int, int, int] | None = None
            for line_number, line in enumerate(stream, start=2):
                record = json.loads(line)
                record_type = record.get("type")
                if record_type == "sample":
                    if summary is not None:
                        raise ValueError(
                            f"{path}:{line_number}: sample follows summary"
                        )
                    episode_number, round_number, step, actor_id = _validate_sample(
                        record, path, line_number, contract
                    )
                    stream_key = (episode_number, round_number, step)
                    if last_stream_key is not None and stream_key <= last_stream_key:
                        raise ValueError(
                            f"{path}:{line_number}: samples are not in canonical "
                            "episode/round/step order"
                        )
                    last_stream_key = stream_key
                    group = (episode_number, round_number)
                    steps = group_steps.setdefault(group, set())
                    if step in steps:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate environment step"
                        )
                    steps.add(step)
                    group_max_steps[group] = max(group_max_steps.get(group, -1), step)
                    episode_numbers.add(episode_number)
                    group_actors.setdefault(group, set()).add(actor_id)
                    finishes = group_finish_places.setdefault(group, {})
                    previous_finish = finishes.setdefault(
                        actor_id, record["finishPlace"]
                    )
                    if previous_finish != record["finishPlace"]:
                        raise ValueError(
                            f"{path}:{line_number}: finishPlace changes within "
                            "trajectory"
                        )
                    revolution = tuple(
                        int(value) for value in record["observation"][169:172]
                    )
                    previous_revolution = group_revolution.setdefault(
                        group, revolution
                    )
                    if previous_revolution != revolution:
                        raise ValueError(
                            f"{path}:{line_number}: revolution state changes within act"
                        )

                    trajectory_id = record["trajectoryId"]
                    if trajectory_id not in trajectory_source_paths:
                        file_trajectory_order.append(trajectory_id)
                    source_path = trajectory_source_paths.setdefault(
                        trajectory_id, path
                    )
                    if source_path != path:
                        raise ValueError(
                            "trajectoryId is duplicated across V3 PPO source files"
                        )
                    state = trajectory_states.setdefault(
                        trajectory_id,
                        {
                            "actorId": actor_id,
                            "actorSeat": record["actorSeat"],
                            "actorRole": record["actorRole"],
                            "finishPlace": record["finishPlace"],
                            "lastStep": -1,
                            "terminalCount": 0,
                            "terminalSeen": False,
                        },
                    )
                    if any(
                        state[field] != record[field]
                        for field in (
                            "actorId",
                            "actorSeat",
                            "actorRole",
                            "finishPlace",
                        )
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: trajectory metadata changes"
                        )
                    if state["terminalSeen"]:
                        raise ValueError(
                            f"{path}:{line_number}: sample follows trajectory terminal"
                        )
                    if step <= state["lastStep"]:
                        raise ValueError(
                            f"{path}:{line_number}: trajectory steps are not increasing"
                        )
                    state["lastStep"] = step
                    if record["terminal"]:
                        state["terminalSeen"] = True
                        state["terminalCount"] = int(state["terminalCount"]) + 1

                    file_samples += 1
                    forced += int(record["forced"])
                    non_forced += int(not record["forced"])
                    total += 1
                    if visitor is not None:
                        visitor(record, contract.player_count)
                elif record_type == "summary":
                    if summary is not None:
                        raise ValueError(f"{path}:{line_number}: duplicate summary")
                    summary = record
                else:
                    raise ValueError(f"{path}:{line_number}: unsupported record type")
            if summary is None:
                raise ValueError(f"{path}: rollout summary is missing")
            _exact_keys(
                summary,
                {
                    "type",
                    "episodes",
                    "learnerSamples",
                    "environmentDecisions",
                    "forcedSamples",
                    "nonForcedSamples",
                    "behaviorModelSha256",
                    "samplingTemperature",
                    "targetNonForcedDecisions",
                    "opponentModelSha256",
                    "opponentSeatAssignments",
                },
                f"{path}: summary",
            )
            if summary["type"] != "summary":
                raise ValueError(f"{path}: invalid summary record")

            expected_episode_numbers = set(range(1, contract.episodes + 1))
            if episode_numbers != expected_episode_numbers:
                raise ValueError(f"{path}: sample episodes do not match environment")
            expected_groups = {
                (episode_number, round_number)
                for episode_number in expected_episode_numbers
                for round_number in range(1, contract.acts_per_episode + 1)
            }
            if set(group_actors) != expected_groups:
                raise ValueError(f"{path}: sample acts do not match environment")
            for episode_number in expected_episode_numbers:
                actor_sets = [
                    group_actors[(episode_number, round_number)]
                    for round_number in range(1, contract.acts_per_episode + 1)
                ]
                expected_learners = (
                    contract.player_count // 2
                    if contract.player_count % 2 == 0 or episode_number % 2 == 1
                    else contract.player_count // 2 + 1
                )
                if (
                    any(len(actors) != expected_learners for actors in actor_sets)
                    or any(actors != actor_sets[0] for actors in actor_sets[1:])
                ):
                    raise ValueError(
                        f"{path}: learner-seat assignments do not match environment"
                    )

            for trajectory_id, state in trajectory_states.items():
                if state["terminalCount"] != 1 or not state["terminalSeen"]:
                    raise ValueError(
                        f"{path}: trajectory {trajectory_id} must end once"
                    )
            for group, finishes in group_finish_places.items():
                if len(set(finishes.values())) != len(finishes):
                    raise ValueError(f"{path}: learner finish places are duplicated")
                episode_number, round_number = group
                if round_number <= 1:
                    continue
                previous_finishes = group_finish_places[
                    (episode_number, round_number - 1)
                ]
                great_revolution = group_revolution[group] == (0, 0, 1)
                for actor_id in group_actors[group]:
                    previous_place = previous_finishes[actor_id]
                    expected_seat = (
                        contract.player_count - previous_place
                        if great_revolution
                        else previous_place - 1
                    )
                    trajectory_id = (
                        f"v3-league-p{contract.player_count}-episode-"
                        f"{episode_number}:round-{round_number}:{actor_id}"
                    )
                    if trajectory_states[trajectory_id]["actorSeat"] != expected_seat:
                        raise ValueError(
                            f"{path}: next-act seat does not match prior finish order"
                        )

            environment_decisions = _strict_integer(
                summary["environmentDecisions"],
                f"{path}: summary.environmentDecisions",
                minimum=1,
            )
            summary_episodes = _strict_integer(
                summary["episodes"], f"{path}: summary.episodes", minimum=1
            )
            summary_samples = _strict_integer(
                summary["learnerSamples"],
                f"{path}: summary.learnerSamples",
                minimum=1,
            )
            summary_forced = _strict_integer(
                summary["forcedSamples"], f"{path}: summary.forcedSamples"
            )
            summary_non_forced = _strict_integer(
                summary["nonForcedSamples"],
                f"{path}: summary.nonForcedSamples",
            )
            sampled_step_lower_bound = sum(
                maximum + 1 for maximum in group_max_steps.values()
            )
            maximum_environment_decisions = (
                contract.episodes
                * contract.acts_per_episode
                * MAX_TRANSITIONS_PER_ACT
            )
            assignments = _exact_keys(
                summary["opponentSeatAssignments"],
                {"normal", "byModelSha256"},
                f"{path}: summary.opponentSeatAssignments",
            )
            normal_assignments = _strict_integer(
                assignments["normal"],
                f"{path}: summary.opponentSeatAssignments.normal",
            )
            by_model = _exact_keys(
                assignments["byModelSha256"],
                set(contract.opponent_model_sha256),
                f"{path}: summary.opponentSeatAssignments.byModelSha256",
            )
            model_assignments = sum(
                _strict_integer(
                    by_model[sha256],
                    f"{path}: opponent assignment {sha256}",
                )
                for sha256 in contract.opponent_model_sha256
            )
            expected_opponent_seats = sum(
                contract.player_count
                - (
                    contract.player_count // 2
                    if contract.player_count % 2 == 0 or episode % 2 == 1
                    else contract.player_count // 2 + 1
                )
                for episode in expected_episode_numbers
            )
            summary_temperature = _strict_finite_number(
                summary["samplingTemperature"],
                f"{path}: summary.samplingTemperature",
            )
            if contract.target_non_forced_decisions is None:
                if summary["targetNonForcedDecisions"] is not None:
                    raise ValueError(f"{path}: fixed collection has a target count")
            else:
                summary_target = _strict_integer(
                    summary["targetNonForcedDecisions"],
                    f"{path}: summary.targetNonForcedDecisions",
                    minimum=1,
                )
                if summary_target != contract.target_non_forced_decisions:
                    raise ValueError(f"{path}: summary target does not match manifest")
            if (
                summary_episodes != contract.episodes
                or summary_samples != file_samples
                or summary_forced != forced
                or summary_non_forced != non_forced
                or file_samples != forced + non_forced
                or summary["behaviorModelSha256"] != contract.behavior_sha256
                or not np.isclose(
                    summary_temperature,
                    contract.temperature,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or summary["opponentModelSha256"]
                != list(contract.opponent_model_sha256)
                or normal_assignments + model_assignments
                != expected_opponent_seats
                or environment_decisions < file_samples
                or environment_decisions < sampled_step_lower_bound
                or environment_decisions > maximum_environment_decisions
                or (
                    contract.target_non_forced_decisions is not None
                    and non_forced < contract.target_non_forced_decisions
                )
            ):
                raise ValueError(f"{path}: rollout summary does not match samples")
            source_content_sha256.append(
                str(_source_file_binding(path)["sha256"])
            )
            file_sample_counts.append(file_samples)
            file_trajectory_ids.append(tuple(file_trajectory_order))
    if total < 1 or shared_sha256 is None or shared_temperature is None:
        raise ValueError("V3 PPO rollout contains no samples")
    return _WalkResult(
        total,
        shared_sha256,
        shared_temperature,
        tuple(source_content_sha256),
        tuple(file_sample_counts),
        tuple(file_trajectory_ids),
    )


def _walk_one_source(path: str) -> _WalkResult:
    """Process-pool entry point for one independently validated source file."""
    return _walk_samples((Path(path),), None)


def _combine_independent_walks(
    results: Sequence[_WalkResult],
) -> _WalkResult:
    if not results:
        raise ValueError("V3 PPO rollout contains no samples")
    behavior_sha256 = results[0].behavior_sha256
    temperature = results[0].temperature
    total = 0
    source_content_sha256: list[str] = []
    file_samples: list[int] = []
    file_trajectory_ids: list[tuple[str, ...]] = []
    trajectory_sources: dict[str, int] = {}
    for file_index, result in enumerate(results):
        if len(result.file_samples) != 1 or len(result.file_trajectory_ids) != 1:
            raise RuntimeError("parallel V3 walk returned a non-file result")
        if (
            result.behavior_sha256 != behavior_sha256
            or result.temperature != temperature
        ):
            raise ValueError("V3 PPO files use different behavior bindings")
        trajectories = result.file_trajectory_ids[0]
        for trajectory_id in trajectories:
            previous = trajectory_sources.setdefault(trajectory_id, file_index)
            if previous != file_index:
                raise ValueError(
                    "trajectoryId is duplicated across V3 PPO source files"
                )
        total += result.samples
        source_content_sha256.extend(result.source_content_sha256)
        file_samples.extend(result.file_samples)
        file_trajectory_ids.append(trajectories)
    return _WalkResult(
        total,
        behavior_sha256,
        temperature,
        tuple(source_content_sha256),
        tuple(file_samples),
        tuple(file_trajectory_ids),
    )


def _parallel_context() -> multiprocessing.context.BaseContext:
    # Spawn never inherits a partially initialized PyTorch/CUDA runtime and is
    # available on both the local Windows verifier and the Linux GPU host.
    return multiprocessing.get_context("spawn")


def _walk_samples_parallel(paths: Sequence[Path], workers: int) -> _WalkResult:
    effective_workers = min(workers, len(paths))
    if effective_workers <= 1:
        return _walk_samples(paths, None)
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=_parallel_context(),
    ) as executor:
        results = list(executor.map(_walk_one_source, map(str, paths)))
    return _combine_independent_walks(results)


def _source_bindings_parallel(
    paths: Sequence[Path], workers: int
) -> list[dict[str, object]]:
    effective_workers = min(workers, len(paths))
    if effective_workers <= 1:
        return [_source_file_binding(path) for path in paths]
    # hashlib releases the GIL for these 1 MiB blocks. Threads avoid importing
    # PyTorch in a third process pool while retaining deterministic path order.
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        return list(executor.map(_source_file_binding, paths))


_PARALLEL_ARRAY_LAYOUT: tuple[
    tuple[str, np.dtype, Callable[[int], tuple[int, ...]]], ...
] = (
    (
        "observations",
        np.dtype(np.float32),
        lambda count: (count, OBSERVATION_FEATURES),
    ),
    ("legal_masks", np.dtype(np.bool_), lambda count: (count, V3_ACTION_COUNT)),
    ("actions", np.dtype(np.int64), lambda count: (count,)),
    ("old_logs", np.dtype(np.float32), lambda count: (count,)),
    ("old_values", np.dtype(np.float32), lambda count: (count,)),
    ("rewards", np.dtype(np.float32), lambda count: (count,)),
    ("rank_rewards", np.dtype(np.float32), lambda count: (count,)),
    ("terminals", np.dtype(np.bool_), lambda count: (count,)),
    ("forced", np.dtype(np.bool_), lambda count: (count,)),
    ("trajectory_ids", np.dtype(np.int32), lambda count: (count,)),
)


def _parallel_array_paths(root: Path) -> dict[str, Path]:
    return {
        name: root / f"{name}.npy"
        for name, _dtype, _shape in _PARALLEL_ARRAY_LAYOUT
    }


def _create_parallel_arrays(root: Path, count: int) -> dict[str, Path]:
    paths = _parallel_array_paths(root)
    for name, dtype, shape_for_count in _PARALLEL_ARRAY_LAYOUT:
        array = np.lib.format.open_memmap(
            paths[name],
            mode="w+",
            dtype=dtype,
            shape=shape_for_count(count),
        )
        array.flush()
        del array
    return paths


def _fill_one_source(
    task: _ParallelFillTask,
) -> tuple[_WalkResult, tuple[int, ...]]:
    arrays = {
        name: np.lib.format.open_memmap(Path(path), mode="r+")
        for name, path in task.array_paths
    }
    position = task.offset
    end = task.offset + task.samples
    trajectory_indices: dict[str, int] = {}
    terminal_counts = [0] * len(task.expected_trajectory_ids)

    def fill(record: dict, player_count: int) -> None:
        nonlocal position
        if position >= end:
            raise RuntimeError("V3 PPO source grew between parallel walks")
        arrays["observations"][position] = record["observation"]
        arrays["legal_masks"][position] = False
        arrays["legal_masks"][position, record["legalActionIndices"]] = True
        arrays["actions"][position] = record["actionIndex"]
        arrays["old_logs"][position] = record["oldLogProbability"]
        arrays["old_values"][position] = record["oldValue"]
        arrays["rewards"][position] = record["reward"]
        arrays["terminals"][position] = record["terminal"]
        arrays["forced"][position] = record["forced"]
        finish_place = record["finishPlace"]
        if finish_place > player_count:
            raise ValueError("finishPlace exceeds manifest player count")
        rank_reward = 0.0
        if (
            record["terminal"]
            and task.terminal_rank_auxiliary_coefficient != 0.0
        ):
            rank_reward = task.terminal_rank_auxiliary_coefficient * (
                1.0 - 2.0 * (finish_place - 1) / (player_count - 1)
            )
        arrays["rank_rewards"][position] = rank_reward
        key = record["trajectoryId"]
        if key not in trajectory_indices:
            local_id = len(trajectory_indices)
            if (
                local_id >= len(task.expected_trajectory_ids)
                or task.expected_trajectory_ids[local_id] != key
            ):
                raise RuntimeError(
                    "V3 PPO trajectory order changed between parallel walks"
                )
            trajectory_indices[key] = local_id
        local_id = trajectory_indices[key]
        arrays["trajectory_ids"][position] = task.trajectory_offset + local_id
        terminal_counts[local_id] += int(record["terminal"])
        position += 1

    try:
        result = _walk_samples((Path(task.path),), fill)
        if position != end or result.samples != task.samples:
            raise RuntimeError("V3 PPO source changed while filling parallel arrays")
        if result.file_trajectory_ids != (task.expected_trajectory_ids,):
            raise RuntimeError(
                "V3 PPO trajectory order changed while filling parallel arrays"
            )
        for array in arrays.values():
            array.flush()
        return result, tuple(terminal_counts)
    finally:
        arrays.clear()


def _load_parallel_arrays(
    paths: dict[str, Path], count: int
) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    for name, dtype, shape_for_count in _PARALLEL_ARRAY_LAYOUT:
        mapped = np.lib.format.open_memmap(paths[name], mode="r")
        expected_shape = shape_for_count(count)
        if mapped.dtype != dtype or mapped.shape != expected_shape:
            raise RuntimeError(f"parallel V3 array contract mismatch: {name}")
        loaded[name] = np.array(mapped, dtype=dtype, copy=True, order="C")
        del mapped
    return loaded


def _verify_behavior_bindings(
    rollouts: PpoRollouts,
    behavior_model_path: Path,
    tolerance: float,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 2048,
) -> None:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("behavior binding batch size must be a positive integer")
    try:
        resolved_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("behavior binding device must be cpu or cuda") from error
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("behavior binding device must be cpu or cuda")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA behavior binding verification was requested but CUDA is unavailable"
        )
    if file_sha256(behavior_model_path) != rollouts.behavior_model_sha256:
        raise ValueError("behavior model SHA-256 does not match V3 rollouts")
    model, payload = load_v3_action_conditioned_json(behavior_model_path)
    if (
        payload.get("observationSchemaVersion") != OBSERVATION_VERSION
        or payload.get("observationFeatures") != OBSERVATION_FEATURES
        or payload.get("actionCatalogueVersion")
        != V3_ACTION_CATALOGUE_VERSION
    ):
        raise ValueError("V3 behavior model metadata does not match rollouts")
    model = model.to(resolved_device).eval()
    maximum_log_error = 0.0
    maximum_value_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(rollouts), batch_size):
            end = min(start + batch_size, len(rollouts))
            observations = torch.from_numpy(
                rollouts.observations[start:end]
            ).to(resolved_device)
            masks = torch.from_numpy(rollouts.legal_masks[start:end]).to(
                resolved_device
            )
            logits, values = model(observations, masks)
            log_probabilities = torch.log_softmax(
                logits / rollouts.behavior_temperature, dim=1
            )
            actions = torch.from_numpy(rollouts.actions[start:end]).to(
                resolved_device
            )
            selected = log_probabilities.gather(1, actions[:, None]).squeeze(1)
            if not (
                torch.isfinite(logits).all()
                and torch.isfinite(values).all()
                and torch.isfinite(log_probabilities).all()
                and torch.isfinite(selected).all()
            ):
                raise ValueError("behavior model recomputation is non-finite")
            log_error = torch.max(
                torch.abs(
                    selected
                    - torch.from_numpy(
                        rollouts.old_log_probabilities[start:end]
                    ).to(resolved_device)
                )
            ).item()
            value_error = torch.max(
                torch.abs(
                    values
                    - torch.from_numpy(rollouts.old_values[start:end]).to(
                        resolved_device
                    )
                )
            ).item()
            maximum_log_error = max(maximum_log_error, log_error)
            maximum_value_error = max(maximum_value_error, value_error)
    if maximum_log_error > tolerance:
        raise ValueError(
            "behavior log-probability binding mismatch: "
            f"maximum error {maximum_log_error:.8g}"
        )
    if maximum_value_error > tolerance:
        raise ValueError(
            "behavior value binding mismatch: "
            f"maximum error {maximum_value_error:.8g}"
        )


def build_v3_ppo_data_verification(
    rollouts: PpoRollouts,
    *,
    source_files: Sequence[dict[str, object]],
    gamma: float,
    gae_lambda: float,
    rollout_temperature: float,
    binding_tolerance: float,
) -> dict[str, object]:
    """Build the shared, versioned V3 data attestation without filesystem I/O."""
    if not np.isclose(
        rollouts.behavior_temperature,
        rollout_temperature,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("rollout-temperature does not match V3 rollout manifest")
    finite_arrays = {
        "observations": rollouts.observations,
        "oldLogProbabilities": rollouts.old_log_probabilities,
        "oldValues": rollouts.old_values,
        "rewards": rollouts.rewards,
        "rankAuxiliaryRewards": rollouts.rank_auxiliary_rewards,
        "effectiveRewards": rollouts.effective_rewards,
        "advantages": rollouts.advantages,
        "returns": rollouts.returns,
    }
    non_finite = [
        name
        for name, values in finite_arrays.items()
        if not np.isfinite(values).all()
    ]
    if non_finite:
        raise ValueError(
            "V3 PPO verification found non-finite arrays: "
            + ", ".join(non_finite)
        )
    return {
        "format": "dalmuti-v3-ppo-data-verification",
        "version": 2,
        "files": list(rollouts.files),
        "sourceFiles": list(source_files),
        "samples": len(rollouts),
        "trajectories": rollouts.trajectory_count,
        "behaviorModelSha256": rollouts.behavior_model_sha256,
        "observationShape": list(rollouts.observations.shape),
        "legalMaskShape": list(rollouts.legal_masks.shape),
        "actionCatalogueVersion": 1,
        "actionCount": 236,
        "forcedSamples": int(rollouts.forced.sum()),
        "policySamples": int((~rollouts.forced).sum()),
        "terminalSamples": int(rollouts.terminals.sum()),
        "gamma": gamma,
        "gaeLambda": gae_lambda,
        "skipForcedPolicyTime": rollouts.skip_forced_policy_time,
        "terminalRankAuxiliaryCoefficient": (
            rollouts.terminal_rank_auxiliary_coefficient
        ),
        "rolloutTemperature": rollout_temperature,
        "rolloutSemanticsContract": {
            "sha256": V3_PPO_SEMANTICS_CONTRACT_SHA256,
            "environment": "exact-game-rules-policy-and-provenance-verified",
            "reward": "recomputed-from-terminal-finishPlace-and-playerCount",
            "summaryCounts": "recomputed-or-strictly-bounded-and-verified",
        },
        "behaviorBinding": {
            "observation": "verified",
            "actionCatalogue": "verified",
            "legalMask": "verified",
            "logProbability": "recomputed-and-verified",
            "value": "recomputed-and-verified",
            "absoluteTolerance": binding_tolerance,
        },
        "finite": True,
    }


def load_v3_ppo_rollouts(
    patterns: Sequence[str],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    skip_forced_policy_time: bool = True,
    terminal_rank_auxiliary_coefficient: float = 0.0,
    behavior_model_path: str | Path | None = None,
    binding_tolerance: float = 2.0e-5,
    behavior_binding_device: str | torch.device = "cpu",
    behavior_binding_batch_size: int = 2048,
    loader_workers: int = 1,
    source_files_out: list[dict[str, object]] | None = None,
) -> PpoRollouts:
    """Strictly load V3 rollouts with an optional deterministic file pool.

    A single worker retains the original in-memory path. Multiple workers
    validate each sorted source independently and fill disjoint offsets in
    temporary NPY memmaps, which avoids pickling multi-gigabyte arrays between
    processes. The memmaps are copied into ordinary C-contiguous arrays and
    removed before this function returns.
    """
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between zero and one")
    if not isinstance(skip_forced_policy_time, bool):
        raise TypeError("skip_forced_policy_time must be boolean")
    if not np.isfinite(terminal_rank_auxiliary_coefficient):
        raise ValueError("terminal rank auxiliary coefficient must be finite")
    if not np.isfinite(binding_tolerance) or binding_tolerance < 0:
        raise ValueError("binding tolerance must be finite and non-negative")
    if (
        isinstance(behavior_binding_batch_size, bool)
        or not isinstance(behavior_binding_batch_size, int)
        or behavior_binding_batch_size < 1
    ):
        raise ValueError("behavior binding batch size must be a positive integer")
    if (
        isinstance(loader_workers, bool)
        or not isinstance(loader_workers, int)
        or loader_workers < 1
    ):
        raise ValueError("loader workers must be a positive integer")
    try:
        resolved_binding_device = torch.device(behavior_binding_device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("behavior binding device must be cpu or cuda") from error
    if resolved_binding_device.type not in {"cpu", "cuda"}:
        raise ValueError("behavior binding device must be cpu or cuda")
    if source_files_out is not None and (
        not isinstance(source_files_out, list) or source_files_out
    ):
        raise ValueError("source_files_out must be an empty list when provided")
    paths = _expand_strict_input_paths(patterns)
    first = _walk_samples_parallel(paths, loader_workers)
    count = first.samples
    behavior_sha256 = first.behavior_sha256
    temperature = first.temperature
    if loader_workers > 1 and len(paths) > 1:
        with tempfile.TemporaryDirectory(
            prefix="dalmuti-v3-ppo-parallel-loader-"
        ) as temporary:
            array_paths = _create_parallel_arrays(Path(temporary), count)
            serialized_paths = tuple(
                (name, str(path)) for name, path in array_paths.items()
            )
            tasks: list[_ParallelFillTask] = []
            sample_offset = 0
            trajectory_offset = 0
            for path, file_samples, expected_trajectories in zip(
                paths,
                first.file_samples,
                first.file_trajectory_ids,
                strict=True,
            ):
                tasks.append(
                    _ParallelFillTask(
                        path=str(path),
                        offset=sample_offset,
                        samples=file_samples,
                        trajectory_offset=trajectory_offset,
                        expected_trajectory_ids=expected_trajectories,
                        array_paths=serialized_paths,
                        terminal_rank_auxiliary_coefficient=(
                            terminal_rank_auxiliary_coefficient
                        ),
                    )
                )
                sample_offset += file_samples
                trajectory_offset += len(expected_trajectories)
            effective_workers = min(loader_workers, len(tasks))
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=_parallel_context(),
            ) as executor:
                filled = list(executor.map(_fill_one_source, tasks))
            second = _combine_independent_walks(
                [walk for walk, _terminal_counts in filled]
            )
            if second != first:
                raise RuntimeError("V3 PPO rollout changed while loading")
            terminal_counts = [
                value
                for _walk, counts in filled
                for value in counts
            ]
            if any(value != 1 for value in terminal_counts):
                raise ValueError(
                    "every V3 trajectory must contain exactly one terminal"
                )
            arrays = _load_parallel_arrays(array_paths, count)
        observations = arrays["observations"]
        legal_masks = arrays["legal_masks"]
        actions = arrays["actions"]
        old_logs = arrays["old_logs"]
        old_values = arrays["old_values"]
        rewards = arrays["rewards"]
        rank_rewards = arrays["rank_rewards"]
        terminals = arrays["terminals"]
        forced = arrays["forced"]
        trajectory_ids = arrays["trajectory_ids"]
        trajectory_count = len(terminal_counts)
    else:
        observations = np.empty(
            (count, OBSERVATION_FEATURES), dtype=np.float32
        )
        legal_masks = np.zeros((count, V3_ACTION_COUNT), dtype=np.bool_)
        actions = np.empty(count, dtype=np.int64)
        old_logs = np.empty(count, dtype=np.float32)
        old_values = np.empty(count, dtype=np.float32)
        rewards = np.empty(count, dtype=np.float32)
        rank_rewards = np.zeros(count, dtype=np.float32)
        terminals = np.empty(count, dtype=np.bool_)
        forced = np.empty(count, dtype=np.bool_)
        trajectory_ids = np.empty(count, dtype=np.int32)
        trajectory_indices: dict[str, int] = {}
        terminal_counts: list[int] = []
        position = 0

        def fill(record: dict, player_count: int) -> None:
            nonlocal position
            observations[position] = record["observation"]
            legal_masks[position, record["legalActionIndices"]] = True
            actions[position] = record["actionIndex"]
            old_logs[position] = record["oldLogProbability"]
            old_values[position] = record["oldValue"]
            rewards[position] = record["reward"]
            terminals[position] = record["terminal"]
            forced[position] = record["forced"]
            finish_place = record["finishPlace"]
            if finish_place > player_count:
                raise ValueError("finishPlace exceeds manifest player count")
            if (
                terminals[position]
                and terminal_rank_auxiliary_coefficient != 0.0
            ):
                rank_rewards[position] = (
                    terminal_rank_auxiliary_coefficient
                    * (1.0 - 2.0 * (finish_place - 1) / (player_count - 1))
                )
            key = record["trajectoryId"]
            if key not in trajectory_indices:
                trajectory_indices[key] = len(trajectory_indices)
                terminal_counts.append(0)
            trajectory_id = trajectory_indices[key]
            trajectory_ids[position] = trajectory_id
            terminal_counts[trajectory_id] += int(terminals[position])
            position += 1

        second = _walk_samples(paths, fill)
        if second != first:
            raise RuntimeError("V3 PPO rollout changed while loading")
        if any(value != 1 for value in terminal_counts):
            raise ValueError(
                "every V3 trajectory must contain exactly one terminal"
            )
        trajectory_count = len(trajectory_indices)
    effective_rewards = rewards + rank_rewards
    value_advantages = np.empty(count, dtype=np.float32)
    next_advantages = np.zeros(trajectory_count, dtype=np.float64)
    next_values = np.zeros(trajectory_count, dtype=np.float64)
    terminal_seen = np.zeros(trajectory_count, dtype=np.bool_)
    for index in range(count - 1, -1, -1):
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
        advantage = delta + gamma * gae_lambda * next_advantage * nonterminal
        value_advantages[index] = advantage
        next_advantages[trajectory_id] = advantage
        next_values[trajectory_id] = old_values[index]
    returns = value_advantages + old_values
    advantages = value_advantages.copy()
    if skip_forced_policy_time:
        next_advantages = np.zeros(trajectory_count, dtype=np.float64)
        next_values = np.zeros(trajectory_count, dtype=np.float64)
        has_next = np.zeros(trajectory_count, dtype=np.bool_)
        pending = np.zeros(trajectory_count, dtype=np.float64)
        for index in range(count - 1, -1, -1):
            trajectory_id = trajectory_ids[index]
            if terminals[index]:
                next_advantages[trajectory_id] = 0.0
                next_values[trajectory_id] = 0.0
                has_next[trajectory_id] = False
                pending[trajectory_id] = 0.0
            if forced[index]:
                pending[trajectory_id] += float(effective_rewards[index])
                continue
            reward = float(effective_rewards[index]) + pending[trajectory_id]
            pending[trajectory_id] = 0.0
            nonterminal = float(has_next[trajectory_id])
            delta = (
                reward
                + gamma * next_values[trajectory_id] * nonterminal
                - float(old_values[index])
            )
            advantage = (
                delta
                + gamma
                * gae_lambda
                * next_advantages[trajectory_id]
                * nonterminal
            )
            advantages[index] = advantage
            next_advantages[trajectory_id] = advantage
            next_values[trajectory_id] = old_values[index]
            has_next[trajectory_id] = True
    computed_arrays = {
        "observations": observations,
        "old_log_probabilities": old_logs,
        "old_values": old_values,
        "rewards": rewards,
        "rank_auxiliary_rewards": rank_rewards,
        "effective_rewards": effective_rewards,
        "value_advantages": value_advantages,
        "advantages": advantages,
        "returns": returns,
    }
    non_finite = [
        name
        for name, values in computed_arrays.items()
        if not np.isfinite(values).all()
    ]
    if non_finite:
        raise ValueError(
            "V3 PPO rollout produced non-finite arrays: " + ", ".join(non_finite)
        )
    result = PpoRollouts(
        observations=observations,
        legal_masks=legal_masks,
        actions=actions,
        old_log_probabilities=old_logs,
        old_values=old_values,
        rewards=rewards,
        rank_auxiliary_rewards=rank_rewards,
        effective_rewards=effective_rewards,
        terminals=terminals,
        forced=forced,
        advantages=advantages,
        returns=returns,
        trajectory_ids=trajectory_ids,
        files=tuple(str(path) for path in paths),
        behavior_model_sha256=behavior_sha256,
        behavior_temperature=temperature,
        trajectory_count=trajectory_count,
        terminal_rank_auxiliary_coefficient=terminal_rank_auxiliary_coefficient,
        skip_forced_policy_time=skip_forced_policy_time,
    )
    if behavior_model_path is not None:
        _verify_behavior_bindings(
            result,
            Path(behavior_model_path).resolve(),
            binding_tolerance,
            device=resolved_binding_device,
            batch_size=behavior_binding_batch_size,
        )
    final_source_bindings = _source_bindings_parallel(paths, loader_workers)
    if tuple(
        str(binding["sha256"]) for binding in final_source_bindings
    ) != first.source_content_sha256:
        raise RuntimeError("V3 PPO rollout changed after validated loading")
    if source_files_out is not None:
        source_files_out.extend(final_source_bindings)
    return result
