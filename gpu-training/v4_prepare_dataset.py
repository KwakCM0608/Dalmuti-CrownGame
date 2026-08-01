from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_COUNT,
    V3_ACTION_FEATURES,
)
from v4_dataset import (
    V4_DATASET_FORMAT,
    V4_DATASET_VERSION,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
    fingerprint_v4_tensors,
    tensorize_v4_public_observation,
)
from v4_model import V4ActorConfig, V4CriticConfig


INPUT_FORMAT = "dalmuti-v4-normal-warmstart-ndjson"
INPUT_VERSION = 1
OUTPUT_METADATA_FORMAT = "dalmuti-v4-prepared-dataset-metadata"
OUTPUT_METADATA_VERSION = 1
ACTOR_SCHEMA_VERSION = 4
CRITIC_SCHEMA_VERSION = 1
PRIVILEGED_FEATURES = 512
LEGAL_MASK_HEX_LENGTH = 59
SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
ROLE_INDEX = {
    "great-dalmuti": 0,
    "lesser-dalmuti": 1,
    "merchant": 2,
    "lesser-peon": 3,
    "great-peon": 4,
}

BASE_SAMPLE_KEYS = {
    "type", "trajectoryId", "episodeId", "act", "step", "actorId",
    "actorSeat", "actorRole", "actorObservation", "privilegedCriticState",
    "legalActionIndices", "legalMaskHex", "actionIndex", "reward",
    "actorTerminal", "environmentTerminal", "finishPlace", "forced",
    "eventsAfterAction",
}
OPTIONAL_SAMPLE_KEYS = {
    "expertActionIndex", "oldLogProbability", "advantage",
}
SOURCE_FILES = {
    "actorObservationContract": "training/v4-public-history.ts",
    "privilegedCriticContract": "training/simulator.ts",
    "normalPolicy": "lib/bot-strategy.ts",
    "generator": "scripts/rl-generate-v4-rollouts.mjs",
    "datasetManifest": "training/v4-rollout-dataset.ts",
}


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        unknown = sorted(actual - keys)
        missing = sorted(keys - actual)
        name = unknown[0] if unknown else missing[0]
        raise ValueError(f"{label} has an unknown or missing field: {name}")
    return value


def _integer(value: object, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be from {minimum} to {maximum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(raw: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key}")
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
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_sidecar(source: Path, checksum_path: Path) -> str:
    expected_text = checksum_path.read_text(encoding="ascii")
    if expected_text.endswith("\n"):
        expected_text = expected_text[:-1]
    if not SHA256_RE.fullmatch(expected_text):
        raise ValueError("input checksum sidecar must contain one lowercase SHA-256")
    actual = _sha256_file(source)
    if actual != expected_text:
        raise ValueError("input NDJSON checksum does not match its sidecar")
    return actual


def _current_source_hashes(repository_root: Path) -> dict[str, str]:
    return {
        key: _sha256_file(repository_root / relative)
        for key, relative in SOURCE_FILES.items()
    }


def _validate_manifest(
    value: Mapping[str, object], repository_root: Path
) -> dict[str, object]:
    manifest = _exact(value, {
        "type", "format", "formatVersion", "environment", "actorObservation",
        "privilegedCritic", "actionSpace", "sampleBindings", "sourceHashes",
    }, "manifest")
    if manifest["type"] != "manifest" or manifest["format"] != INPUT_FORMAT:
        raise ValueError("unsupported V4 rollout manifest")
    if manifest["formatVersion"] != INPUT_VERSION:
        raise ValueError("V4 rollout manifest version mismatch")

    source_hashes = _exact(manifest["sourceHashes"], {
        "actorObservationContract", "privilegedCriticContract", "actionCatalogue",
        "normalPolicy", "generator", "datasetManifest",
    }, "manifest.sourceHashes")
    for name, digest in source_hashes.items():
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"manifest.sourceHashes.{name} is not a SHA-256")
    current = _current_source_hashes(repository_root)
    for name, digest in current.items():
        if source_hashes[name] != digest:
            raise ValueError(f"V4 source hash drift: {name}")

    environment = _exact(manifest["environment"], {
        "game", "rules", "playerCount", "actsPerEpisode", "initialSeed",
        "behaviorPolicy", "reward", "collection",
    }, "manifest.environment")
    player_count = _integer(environment["playerCount"], 4, 10, "playerCount")
    acts = _integer(environment["actsPerEpisode"], 1, 1_000_000, "actsPerEpisode")
    initial_seed = _integer(environment["initialSeed"], 1, 2**53 - 1, "initialSeed")
    if (
        environment["game"] != "DALMUTI"
        or environment["rules"] != "project-house-rules-v1"
        or environment["behaviorPolicy"] != "normal"
        or environment["reward"]
        != "actorTerminal ? (roundChipAward - 2) / 2 : 0"
    ):
        raise ValueError("V4 environment contract mismatch")
    collection = _exact(environment["collection"], {
        "mode", "targetNonForcedDecisions", "maxEpisodes", "completeEpisodesOnly",
    }, "manifest.environment.collection")
    target = _integer(
        collection["targetNonForcedDecisions"], 1, 2**53 - 1,
        "targetNonForcedDecisions",
    )
    max_episodes = _integer(collection["maxEpisodes"], 1, 2**53 - 1, "maxEpisodes")
    if (
        collection["mode"] != "target-non-forced-decisions"
        or collection["completeEpisodesOnly"] is not True
    ):
        raise ValueError("V4 collection contract mismatch")

    actor = _exact(manifest["actorObservation"], {
        "schemaVersion", "sourceSha256", "canonicalBuilder",
        "maxRecentHistoryEvents", "memoryTraceDecays", "memoryTraceFeatures",
        "privacy",
    }, "manifest.actorObservation")
    expected_memory_features = [
        "type.play", "type.pass", "type.clear", "type.finish", "actor-offset",
        "hand-count-before", "hand-count-after", "rank", "natural-count",
        "joker-count", "total-count", "pass.manual", "pass.timeout",
        "pass.insufficient-cards", "pass.dalmuti", "clear.all-passed",
        "clear.dalmuti", "clear.act-ended", "next-leader-offset", "finish-place",
    ]
    if (
        actor["schemaVersion"] != ACTOR_SCHEMA_VERSION
        or actor["sourceSha256"] != source_hashes["actorObservationContract"]
        or actor["canonicalBuilder"] != "buildV4ActorVisibleObservation"
        or actor["maxRecentHistoryEvents"] != 192
        or actor["memoryTraceDecays"] != [0.5, 0.8, 0.95, 0.99]
        or actor["memoryTraceFeatures"] != expected_memory_features
        or actor["privacy"]
        != "own physical hand plus public state/history only; IDs and opponent hands excluded"
    ):
        raise ValueError("V4 public actor contract drift")

    critic = _exact(manifest["privilegedCritic"], {
        "schemaVersion", "sourceSha256", "featureCount", "layout",
        "actorExportAllowed", "privacyClass",
    }, "manifest.privilegedCritic")
    if (
        critic["schemaVersion"] != CRITIC_SCHEMA_VERSION
        or critic["sourceSha256"] != source_hashes["privilegedCriticContract"]
        or critic["featureCount"] != PRIVILEGED_FEATURES
        or critic["actorExportAllowed"] is not False
        or critic["privacyClass"] != "restricted-training-only-full-state"
    ):
        raise ValueError("V4 privileged critic contract drift")
    expected_layout = {
        "version": 1,
        "featureCount": 512,
        "global": {
            "offset": 0,
            "fields": [
                "playerCount", "act", "revolution", "table.present",
                "table.rank", "table.naturalCount", "table.jokerCount",
                "table.totalCount", "table.actorOffsetOrMinusOne",
                "publicPlayedTotal", "activePlayerCount", "finishedPlayerCount",
                "actorRole", "actorScore", "actorHandCount",
                "publicHistoryEventCount",
            ],
        },
        "publicPlayedRankCounts": {"offset": 16, "length": 13, "ranks": "1..13"},
        "players": {
            "offset": 29, "seats": 10, "stride": 25,
            "fields": [
                "present", "relativeOffset", "role.oneHot[5]", "score",
                "handCount", "passed", "finished", "finishPlace",
                "handRankCounts[13]",
            ],
        },
        "reservedZeroTail": {"offset": 279, "length": 233},
    }
    if critic["layout"] != expected_layout:
        raise ValueError("V4 privileged critic layout drift")

    action = _exact(manifest["actionSpace"], {
        "catalogueVersion", "size", "catalogueSha256", "catalogue",
        "encodedActionFeatures", "legalMaskEncoding",
    }, "manifest.actionSpace")
    expected_catalogue = [dict(item) for item in V3_ACTION_CATALOGUE]
    expected_features = [list(row) for row in V3_ACTION_FEATURES]
    catalogue_bytes = json.dumps(
        {"version": V3_ACTION_CATALOGUE_VERSION, "catalogue": expected_catalogue},
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    catalogue_hash = hashlib.sha256(catalogue_bytes).hexdigest()
    if (
        action["catalogueVersion"] != V3_ACTION_CATALOGUE_VERSION
        or action["size"] != V3_ACTION_COUNT
        or action["catalogue"] != expected_catalogue
        or action["encodedActionFeatures"] != expected_features
        or action["catalogueSha256"] != catalogue_hash
        or source_hashes["actionCatalogue"] != catalogue_hash
    ):
        raise ValueError("V4 action catalogue or feature contract drift")
    mask_encoding = _exact(action["legalMaskEncoding"], {
        "field", "lowercaseHexDigits", "bitOrder",
    }, "manifest.actionSpace.legalMaskEncoding")
    if mask_encoding != {
        "field": "legalMaskHex",
        "lowercaseHexDigits": LEGAL_MASK_HEX_LENGTH,
        "bitOrder": "action index i = bit (i % 4) of hex digit floor(i / 4)",
    }:
        raise ValueError("V4 legal-mask contract drift")
    bindings = _exact(manifest["sampleBindings"], {
        "actionIndex", "legalActionIndices", "actorObservation",
        "privilegedCriticState", "eventsAfterAction", "forced",
    }, "manifest.sampleBindings")
    if bindings != {
        "actionIndex": "236-action catalogue index selected by exact Normal",
        "legalActionIndices": "unique ascending indices exactly equal to legalMaskHex",
        "actorObservation": "canonical sanitized state immediately before the selected action",
        "privilegedCriticState": "separate full-information state immediately before the action",
        "eventsAfterAction": "ordered public play/pass/clear/finish events emitted by the action",
        "forced": "true exactly when legalActionIndices has length one",
    }:
        raise ValueError("V4 sample binding contract drift")
    return {
        "playerCount": player_count,
        "acts": acts,
        "initialSeed": initial_seed,
        "target": target,
        "maxEpisodes": max_episodes,
        "sourceHashes": dict(source_hashes),
    }


def _decode_mask(value: object) -> list[int]:
    if not isinstance(value, str) or len(value) != LEGAL_MASK_HEX_LENGTH:
        raise ValueError("legalMaskHex must contain exactly 59 lowercase hex digits")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("legalMaskHex must contain exactly 59 lowercase hex digits")
    result: list[int] = []
    for nibble_index, character in enumerate(value):
        nibble = int(character, 16)
        for bit in range(4):
            if nibble & (1 << bit):
                result.append(nibble_index * 4 + bit)
    if not result:
        raise ValueError("legalMaskHex must contain a legal action")
    return result


def _validate_events(value: object, player_count: int, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty public event list")
    previous = -1
    base = {"type", "sequence", "actorId", "handCountBefore", "handCountAfter"}
    extras = {
        "play": {"rank", "naturalCount", "jokerCount", "totalCount"},
        "pass": {"reason"},
        "clear": {"rank", "naturalCount", "jokerCount", "totalCount", "reason", "nextLeaderId"},
        "finish": {"place"},
    }
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or raw.get("type") not in extras:
            raise ValueError(f"{label}[{index}] has an invalid public event type")
        event_type = str(raw["type"])
        event = _exact(raw, base | extras[event_type], f"{label}[{index}]")
        sequence = _integer(event["sequence"], 0, 2**53 - 1, "event.sequence")
        if sequence <= previous:
            raise ValueError(f"{label} sequences must be strictly increasing")
        previous = sequence
        _string(event["actorId"], "event.actorId")
        before = _integer(event["handCountBefore"], 0, 80, "event.handCountBefore")
        after = _integer(event["handCountAfter"], 0, 80, "event.handCountAfter")
        if event_type in {"play", "clear"}:
            rank = _integer(event["rank"], 1, 13, "event.rank")
            natural = _integer(event["naturalCount"], 0, 14, "event.naturalCount")
            jokers = _integer(event["jokerCount"], 0, 2, "event.jokerCount")
            total = _integer(event["totalCount"], 1, 14, "event.totalCount")
            if natural + jokers != total or (rank == 13 and natural != 0):
                raise ValueError(f"{label}[{index}] has inconsistent card counts")
        if event_type == "play" and before - after != event["totalCount"]:
            raise ValueError(f"{label}[{index}] play hand-count transition is invalid")
        if event_type == "pass":
            if event["reason"] not in {"manual", "timeout", "insufficient-cards", "dalmuti"} or before != after:
                raise ValueError(f"{label}[{index}] pass event is invalid")
        if event_type == "clear":
            if event["reason"] not in {"all-passed", "dalmuti", "act-ended"}:
                raise ValueError(f"{label}[{index}] clear reason is invalid")
            if event["nextLeaderId"] is not None:
                _string(event["nextLeaderId"], "event.nextLeaderId")
        if event_type == "finish":
            _integer(event["place"], 1, player_count, "event.place")
            if before != 0 or after != 0:
                raise ValueError(f"{label}[{index}] finish hand counts must be zero")


def _validate_privileged(
    value: object,
    player_count: int,
    act: int,
    actor_role: str,
    actor_observation: Mapping[str, object],
    label: str,
) -> np.ndarray:
    state = _exact(value, {
        "schemaVersion", "playerCount", "act", "actorRole", "revolution",
        "publicPlayedCounts", "table", "players", "features",
    }, label)
    if (
        state["schemaVersion"] != CRITIC_SCHEMA_VERSION
        or state["playerCount"] != player_count
        or state["act"] != act
        or state["actorRole"] != actor_role
    ):
        raise ValueError(f"{label} binding mismatch")
    if state["revolution"] not in {None, "revolution", "great-revolution"}:
        raise ValueError(f"{label}.revolution is invalid")
    public_counts = state["publicPlayedCounts"]
    if not isinstance(public_counts, list) or len(public_counts) != 13:
        raise ValueError(f"{label}.publicPlayedCounts must contain 13 values")
    public_counts = [
        _integer(item, 0, 12 if index < 12 else 2, f"{label}.publicPlayedCounts[{index}]")
        for index, item in enumerate(public_counts)
    ]
    if public_counts != actor_observation["publicPlayedCounts"]:
        raise ValueError(f"{label} public counts differ from actor observation")
    table = state["table"]
    if table is not None:
        table = _exact(table, {"actorOffset", "rank", "naturalCount", "jokerCount", "totalCount"}, f"{label}.table")
        _integer(table["actorOffset"], 0, player_count - 1, "critic.table.actorOffset")
        _integer(table["rank"], 1, 13, "critic.table.rank")
        natural = _integer(table["naturalCount"], 0, 14, "critic.table.naturalCount")
        jokers = _integer(table["jokerCount"], 0, 2, "critic.table.jokerCount")
        total = _integer(table["totalCount"], 1, 14, "critic.table.totalCount")
        if natural + jokers != total:
            raise ValueError(f"{label}.table card counts are inconsistent")
    if table != actor_observation["table"]:
        raise ValueError(f"{label} table differs from actor observation")
    players = state["players"]
    if not isinstance(players, list) or len(players) != player_count:
        raise ValueError(f"{label}.players must match playerCount")
    normalized_players: list[dict[str, object]] = []
    offsets: set[int] = set()
    for index, raw in enumerate(players):
        player = _exact(raw, {
            "relativeOffset", "role", "score", "passed", "finishPlace", "handRankCounts",
        }, f"{label}.players[{index}]")
        offset = _integer(player["relativeOffset"], 0, player_count - 1, "critic.player.relativeOffset")
        if offset in offsets:
            raise ValueError(f"{label} player offsets are duplicated")
        offsets.add(offset)
        role = player["role"]
        if role not in ROLE_INDEX:
            raise ValueError(f"{label}.players[{index}].role is invalid")
        score = _number(player["score"], "critic.player.score")
        passed = _boolean(player["passed"], "critic.player.passed")
        place = _integer(player["finishPlace"], 0, player_count, "critic.player.finishPlace")
        counts = player["handRankCounts"]
        if not isinstance(counts, list) or len(counts) != 13:
            raise ValueError(f"{label}.players[{index}].handRankCounts must contain 13 values")
        counts = [_integer(item, 0, rank + 1 if rank < 12 else 2, "critic hand count") for rank, item in enumerate(counts)]
        normalized_players.append({
            "relativeOffset": offset, "role": role, "score": score,
            "passed": passed, "finishPlace": place, "handRankCounts": counts,
        })
    if offsets != set(range(player_count)):
        raise ValueError(f"{label} player offsets are incomplete")
    normalized_players.sort(key=lambda player: int(player["relativeOffset"]))

    features_raw = state["features"]
    if not isinstance(features_raw, list) or len(features_raw) != PRIVILEGED_FEATURES:
        raise ValueError(f"{label}.features must contain exactly 512 values")
    features = np.asarray([
        _number(item, f"{label}.features[{index}]")
        for index, item in enumerate(features_raw)
    ], dtype=np.float32)
    expected = np.zeros(PRIVILEGED_FEATURES, dtype=np.float64)
    revolution = 0 if state["revolution"] is None else 1 if state["revolution"] == "revolution" else 2
    table_values = table or {}
    expected[:16] = [
        player_count, act, revolution, int(table is not None),
        table_values.get("rank", 0), table_values.get("naturalCount", 0),
        table_values.get("jokerCount", 0), table_values.get("totalCount", 0),
        table_values.get("actorOffset", -1), sum(public_counts),
        sum(sum(player["handRankCounts"]) > 0 for player in normalized_players),
        sum(int(player["finishPlace"]) > 0 for player in normalized_players),
        ROLE_INDEX[actor_role], normalized_players[0]["score"],
        sum(normalized_players[0]["handRankCounts"]),
        len(actor_observation["historyTokens"]) + int(actor_observation["truncatedHistoryCount"]),
    ]
    expected[16:29] = public_counts
    for player in normalized_players:
        offset = 29 + int(player["relativeOffset"]) * 25
        counts = player["handRankCounts"]
        hand_count = sum(counts)
        row = [
            1, player["relativeOffset"],
            *[int(index == ROLE_INDEX[str(player["role"])]) for index in range(5)],
            player["score"], hand_count, int(bool(player["passed"])),
            int(hand_count == 0), player["finishPlace"], *counts,
        ]
        expected[offset:offset + 25] = row
    if not np.array_equal(np.asarray(features_raw, dtype=np.float64), expected):
        raise ValueError(f"{label}.features do not match the declared full state")
    return features


def _round_reward(place: int, player_count: int) -> float:
    award = 4 if place == 1 else 3 if place == 2 else 1 if place == player_count - 1 else 0 if place == player_count else 2
    return (award - 2) / 2


def _validate_sample(
    value: Mapping[str, object],
    manifest: Mapping[str, object],
    line_number: int,
    optional_mode: frozenset[str] | None,
) -> tuple[dict[str, object], frozenset[str]]:
    label = f"sample line {line_number}"
    actual = set(value)
    optional = frozenset(actual - BASE_SAMPLE_KEYS)
    if not optional.issubset(OPTIONAL_SAMPLE_KEYS) or actual - optional != BASE_SAMPLE_KEYS:
        _exact(value, BASE_SAMPLE_KEYS | optional, label)
        raise ValueError(f"{label} has unknown or missing fields")
    if ("oldLogProbability" in optional) != ("advantage" in optional):
        raise ValueError(f"{label} PPO fields must include oldLogProbability and advantage together")
    if optional_mode is not None and optional != optional_mode:
        raise ValueError("V4 sample optional-field schema drift within one dataset")
    if value["type"] != "sample":
        raise ValueError(f"{label}.type must be sample")
    player_count = int(manifest["playerCount"])
    trajectory_id = _string(value["trajectoryId"], f"{label}.trajectoryId")
    episode_id = _string(value["episodeId"], f"{label}.episodeId")
    actor_id = _string(value["actorId"], f"{label}.actorId")
    act = _integer(value["act"], 1, int(manifest["acts"]), f"{label}.act")
    step = _integer(value["step"], 0, 2**53 - 1, f"{label}.step")
    seat = _integer(value["actorSeat"], 0, player_count - 1, f"{label}.actorSeat")
    role = value["actorRole"]
    if role not in ROLE_INDEX:
        raise ValueError(f"{label}.actorRole is invalid")
    expected_trajectory_id = f"{episode_id}:act-{act}:{actor_id}"
    if trajectory_id != expected_trajectory_id:
        raise ValueError(f"{label}.trajectoryId does not bind episode, act, and actor")
    observation = value["actorObservation"]
    if not isinstance(observation, Mapping):
        raise ValueError(f"{label}.actorObservation must be an object")
    public_tensors = tensorize_v4_public_observation(observation)
    if (
        observation["playerCount"] != player_count
        or observation["act"] != act
        or observation["actorRole"] != ROLE_INDEX[str(role)]
    ):
        raise ValueError(f"{label}.actorObservation binding mismatch")
    self_tokens = [token for token in observation["playerTokens"] if token.get("self") == 1]
    if len(self_tokens) != 1 or self_tokens[0].get("relativeOffset") != 0:
        raise ValueError(f"{label}.actorObservation self token is invalid")
    privileged = _validate_privileged(
        value["privilegedCriticState"], player_count, act, str(role), observation,
        f"{label}.privilegedCriticState",
    )
    legal = value["legalActionIndices"]
    if not isinstance(legal, list):
        raise ValueError(f"{label}.legalActionIndices must be a list")
    legal = [_integer(item, 0, V3_ACTION_COUNT - 1, f"{label}.legalActionIndices") for item in legal]
    if not legal or legal != sorted(set(legal)) or legal != _decode_mask(value["legalMaskHex"]):
        raise ValueError(f"{label} legal indices and exact 59-hex mask differ")
    action = _integer(value["actionIndex"], 0, V3_ACTION_COUNT - 1, f"{label}.actionIndex")
    expert = action if "expertActionIndex" not in optional else _integer(value["expertActionIndex"], 0, V3_ACTION_COUNT - 1, f"{label}.expertActionIndex")
    if action not in legal or expert not in legal:
        raise ValueError(f"{label} selected an illegal action")
    reward = _number(value["reward"], f"{label}.reward")
    terminal = _boolean(value["actorTerminal"], f"{label}.actorTerminal")
    environment_terminal = _boolean(value["environmentTerminal"], f"{label}.environmentTerminal")
    finish_place = _integer(value["finishPlace"], 1, player_count, f"{label}.finishPlace")
    forced = _boolean(value["forced"], f"{label}.forced")
    if forced != (len(legal) == 1):
        raise ValueError(f"{label}.forced does not match legal action count")
    expected_reward = _round_reward(finish_place, player_count) if terminal else 0.0
    if reward != expected_reward:
        raise ValueError(f"{label}.reward does not match terminal finish place")
    if environment_terminal and not terminal:
        raise ValueError(f"{label} environment terminal must also be actor terminal")
    old_log_prob = 0.0 if "oldLogProbability" not in optional else _number(value["oldLogProbability"], f"{label}.oldLogProbability")
    advantage = 0.0 if "advantage" not in optional else _number(value["advantage"], f"{label}.advantage")
    if old_log_prob > 1e-9:
        raise ValueError(f"{label}.oldLogProbability must be <= 0")
    _validate_events(value["eventsAfterAction"], player_count, f"{label}.eventsAfterAction")
    return ({
        "trajectoryId": trajectory_id, "episodeId": episode_id, "act": act,
        "step": step, "actorId": actor_id, "actorSeat": seat, "actorRole": role,
        "public": public_tensors, "privileged": privileged, "legal": legal,
        "action": action, "expert": expert, "oldLogProbability": old_log_prob,
        "advantage": advantage, "reward": reward, "done": terminal,
        "environmentTerminal": environment_terminal, "finishPlace": finish_place,
        "forced": forced,
    }, optional)


def _read_and_validate(
    source: Path, repository_root: Path
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], bytes]:
    data = source.read_bytes()
    if not data or not data.endswith(b"\n") or b"\r\n" in data:
        raise ValueError("V4 NDJSON must be non-empty LF-terminated UTF-8")
    lines = data.splitlines(keepends=True)
    if len(lines) < 3 or any(line == b"\n" for line in lines):
        raise ValueError("V4 NDJSON requires manifest, samples, and summary without blanks")
    manifest_record = _json_load(lines[0][:-1], "manifest line 1")
    manifest = _validate_manifest(manifest_record, repository_root)
    optional_mode: frozenset[str] | None = None
    samples: list[dict[str, object]] = []
    forced = 0
    episodes: set[str] = set()
    seen_steps: set[tuple[str, int, int]] = set()
    for index, raw in enumerate(lines[1:-1], start=2):
        record = _json_load(raw[:-1], f"line {index}")
        sample, optional = _validate_sample(record, manifest, index, optional_mode)
        optional_mode = optional
        key = (str(sample["episodeId"]), int(sample["act"]), int(sample["step"]))
        if key in seen_steps:
            raise ValueError(f"line {index} duplicates an episode/act step")
        seen_steps.add(key)
        episodes.add(str(sample["episodeId"]))
        forced += int(bool(sample["forced"]))
        samples.append(sample)
    summary_record = _json_load(lines[-1][:-1], f"summary line {len(lines)}")
    summary = _exact(summary_record, {
        "type", "episodes", "samples", "forcedSamples", "nonForcedSamples",
        "targetNonForcedDecisions", "recordsBeforeSummarySha256",
    }, "summary")
    digest = hashlib.sha256(b"".join(lines[:-1])).hexdigest()
    expected_summary = {
        "type": "summary", "episodes": len(episodes), "samples": len(samples),
        "forcedSamples": forced, "nonForcedSamples": len(samples) - forced,
        "targetNonForcedDecisions": manifest["target"],
        "recordsBeforeSummarySha256": digest,
    }
    if summary != expected_summary:
        raise ValueError("V4 summary counts or records-before-summary hash mismatch")
    if len(episodes) < 1 or len(episodes) > int(manifest["maxEpisodes"]):
        raise ValueError("V4 summary episode count is outside the manifest bounds")
    if len(samples) - forced < int(manifest["target"]):
        raise ValueError("V4 rollout did not reach its non-forced target")
    manifest["optionalFields"] = sorted(optional_mode or ())
    return manifest, samples, dict(summary), data


def _build_dataset(
    samples: list[dict[str, object]], actor: V4ActorConfig, critic: V4CriticConfig
) -> tuple[
    V4TrajectoryDataset,
    dict[str, np.ndarray],
    list[str],
    list[str],
]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for sample in samples:
        key = (str(sample["inputSha256"]), str(sample["trajectoryId"]))
        grouped.setdefault(key, []).append(sample)
    trajectory_keys = sorted(grouped)
    trajectory_ids = [key[1] for key in trajectory_keys]
    trajectory_input_hashes = [key[0] for key in trajectory_keys]
    max_steps = max(len(grouped[key]) for key in trajectory_keys)
    count = len(trajectory_keys)
    shape = (count, max_steps)
    arrays: dict[str, np.ndarray] = {
        "global_features": np.zeros((*shape, actor.global_features), np.float32),
        "rank_features": np.zeros((*shape, actor.rank_tokens, actor.rank_features), np.float32),
        "player_features": np.zeros((*shape, actor.max_players, actor.player_features), np.float32),
        "player_mask": np.zeros((*shape, actor.max_players), np.bool_),
        "memory_trace_features": np.zeros((*shape, actor.memory_tokens, actor.memory_features), np.float32),
        "history_features": np.zeros((*shape, actor.max_history, actor.history_features), np.float32),
        "history_mask": np.zeros((*shape, actor.max_history), np.bool_),
        "legal_masks": np.zeros((*shape, V3_ACTION_COUNT), np.bool_),
        "actions": np.zeros(shape, np.int64), "expert_actions": np.zeros(shape, np.int64),
        "old_action_log_probs": np.zeros(shape, np.float32), "advantages": np.zeros(shape, np.float32),
        "rewards": np.zeros(shape, np.float32), "dones": np.zeros(shape, np.bool_),
        "valid_masks": np.zeros(shape, np.bool_),
        "privileged_states": np.zeros((*shape, critic.privileged_features), np.float32),
    }
    auxiliary = {
        "finish_places": np.zeros(shape, np.int16),
        "environment_terminals": np.zeros(shape, np.bool_),
        "source_steps": np.full(shape, -1, np.int64),
    }
    for trajectory_index, trajectory_key in enumerate(trajectory_keys):
        trajectory_id = trajectory_key[1]
        trajectory = sorted(grouped[trajectory_key], key=lambda item: int(item["step"]))
        first = trajectory[0]
        previous_step = -1
        terminals = 0
        for time_index, sample in enumerate(trajectory):
            if any(sample[name] != first[name] for name in ("episodeId", "act", "actorId", "actorSeat", "actorRole", "finishPlace")):
                raise ValueError(f"trajectory {trajectory_id} changes its bound identity or finish place")
            step = int(sample["step"])
            if step <= previous_step:
                raise ValueError(f"trajectory {trajectory_id} steps are not strictly ordered")
            previous_step = step
            if terminals:
                raise ValueError(f"trajectory {trajectory_id} contains data after terminal")
            terminals += int(bool(sample["done"]))
            public = sample["public"]
            arrays["global_features"][trajectory_index, time_index] = public.global_features.numpy()
            arrays["rank_features"][trajectory_index, time_index] = public.rank_features.numpy()
            arrays["player_features"][trajectory_index, time_index] = public.player_features.numpy()
            arrays["player_mask"][trajectory_index, time_index] = public.player_mask.numpy()
            arrays["memory_trace_features"][trajectory_index, time_index] = public.memory_trace_features.numpy()
            arrays["history_features"][trajectory_index, time_index] = public.history_features.numpy()
            arrays["history_mask"][trajectory_index, time_index] = public.history_mask.numpy()
            arrays["legal_masks"][trajectory_index, time_index, sample["legal"]] = True
            arrays["actions"][trajectory_index, time_index] = sample["action"]
            arrays["expert_actions"][trajectory_index, time_index] = sample["expert"]
            arrays["old_action_log_probs"][trajectory_index, time_index] = sample["oldLogProbability"]
            arrays["advantages"][trajectory_index, time_index] = sample["advantage"]
            arrays["rewards"][trajectory_index, time_index] = sample["reward"]
            arrays["dones"][trajectory_index, time_index] = sample["done"]
            arrays["valid_masks"][trajectory_index, time_index] = True
            arrays["privileged_states"][trajectory_index, time_index] = sample["privileged"]
            auxiliary["finish_places"][trajectory_index, time_index] = sample["finishPlace"]
            auxiliary["environment_terminals"][trajectory_index, time_index] = sample["environmentTerminal"]
            auxiliary["source_steps"][trajectory_index, time_index] = step
        if terminals != 1 or not bool(trajectory[-1]["done"]):
            raise ValueError(f"trajectory {trajectory_id} requires exactly one final terminal")
    boolean_names = {"player_mask", "history_mask", "legal_masks", "dones", "valid_masks"}
    integer_names = {"actions", "expert_actions"}
    tensors: dict[str, torch.Tensor] = {}
    for field in fields(V4TrajectoryTensors):
        array = arrays[field.name]
        if field.name in boolean_names:
            tensors[field.name] = torch.from_numpy(array)
        elif field.name in integer_names:
            tensors[field.name] = torch.from_numpy(array)
        else:
            tensors[field.name] = torch.from_numpy(array)
    dataset = V4TrajectoryDataset(V4TrajectoryTensors(**tensors), actor, critic)
    return dataset, auxiliary, trajectory_ids, trajectory_input_hashes


def _atomic_temp(directory: Path, name: str, payload: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, prefix=f".{name}.", suffix=".partial", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _promote_exclusive(temp: Path, target: Path) -> None:
    try:
        os.link(temp, target)
    except FileExistsError as error:
        raise FileExistsError(f"output already exists: {target}") from error
    finally:
        temp.unlink(missing_ok=True)


def _path_list(value: str | Path | Sequence[str | Path], label: str) -> list[Path]:
    if isinstance(value, (str, Path)):
        values: Sequence[str | Path] = [value]
    else:
        values = value
    if not values:
        raise ValueError(f"{label} requires at least one path")
    return [Path(item).resolve() for item in values]


def convert_v4_rollouts(
    input_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    checksum_path: str | Path | Sequence[str | Path] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    sources = _path_list(input_path, "input")
    output = Path(output_path).resolve()
    if output.suffix != ".npz":
        raise ValueError("V4 prepared dataset output must end in .npz")
    if checksum_path is None:
        checksums = [Path(f"{source}.sha256") for source in sources]
    else:
        checksums = _path_list(checksum_path, "input checksum")
        if len(checksums) != len(sources):
            raise ValueError("input checksum paths must match the number of inputs")
    root = Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parent.parent
    shard_results: list[dict[str, object]] = []
    all_samples: list[dict[str, object]] = []
    seen_episodes: set[str] = set()
    seen_trajectories: set[str] = set()
    common_acts: int | None = None
    common_source_hashes: dict[str, str] | None = None
    for source, checksum in zip(sources, checksums, strict=True):
        input_sha = _verify_sidecar(source, checksum)
        manifest, samples, summary, _ = _read_and_validate(source, root)
        if common_acts is None:
            common_acts = int(manifest["acts"])
            common_source_hashes = dict(manifest["sourceHashes"])
        elif (
            int(manifest["acts"]) != common_acts
            or manifest["sourceHashes"] != common_source_hashes
        ):
            raise ValueError(
                "V4 input shards must share acts, actor/action/critic contracts, "
                "and source hashes"
            )
        shard_episode_ids = {str(sample["episodeId"]) for sample in samples}
        shard_trajectory_ids = {str(sample["trajectoryId"]) for sample in samples}
        episode_collision = seen_episodes.intersection(shard_episode_ids)
        trajectory_collision = seen_trajectories.intersection(shard_trajectory_ids)
        if episode_collision:
            raise ValueError(
                f"V4 input shards duplicate episode ID: {min(episode_collision)}"
            )
        if trajectory_collision:
            raise ValueError(
                "V4 input shards duplicate trajectory ID: "
                f"{min(trajectory_collision)}"
            )
        seen_episodes.update(shard_episode_ids)
        seen_trajectories.update(shard_trajectory_ids)
        for sample in samples:
            sample["inputSha256"] = input_sha
        all_samples.extend(samples)
        optional = set(manifest["optionalFields"])
        shard_results.append({
            "sha256": input_sha,
            "format": INPUT_FORMAT,
            "formatVersion": INPUT_VERSION,
            "playerCount": manifest["playerCount"],
            "actsPerEpisode": manifest["acts"],
            "initialSeed": manifest["initialSeed"],
            "episodeCount": len(shard_episode_ids),
            "trajectoryCount": len(shard_trajectory_ids),
            "sampleCount": len(samples),
            "summary": summary,
            "sourceHashes": manifest["sourceHashes"],
            "sampleFieldsPresent": sorted(optional),
            "syntheticDefaults": {
                "expertActionIndex": "expertActionIndex" not in optional,
                "oldLogProbability": "oldLogProbability" not in optional,
                "advantage": "advantage" not in optional,
            },
        })
    ordered_inputs = sorted(shard_results, key=lambda item: str(item["sha256"]))
    actor = V4ActorConfig()
    critic = V4CriticConfig(privileged_features=PRIVILEGED_FEATURES)
    dataset, auxiliary, trajectory_ids, trajectory_input_hashes = _build_dataset(
        all_samples, actor, critic
    )
    present_fields = {
        name
        for shard in shard_results
        for name in shard["sampleFieldsPresent"]
    }
    synthetic_defaults = {
        name: any(bool(shard["syntheticDefaults"][name]) for shard in shard_results)
        for name in ("expertActionIndex", "oldLogProbability", "advantage")
    }
    metadata: dict[str, object] = {
        "format": V4_DATASET_FORMAT,
        "version": V4_DATASET_VERSION,
        "preparationFormat": OUTPUT_METADATA_FORMAT,
        "preparationVersion": OUTPUT_METADATA_VERSION,
        "actorConfig": actor.to_dict(), "criticConfig": critic.to_dict(),
        "fingerprint": dataset.fingerprint,
        "inputs": ordered_inputs,
        "trajectoryCount": len(trajectory_ids), "maxTimeSteps": int(dataset.tensors.actions.shape[1]),
        "trajectoryIds": trajectory_ids,
        "trajectoryInputSha256s": trajectory_input_hashes,
        "sampleFieldsPresent": sorted(present_fields),
        "syntheticDefaults": synthetic_defaults,
        "padding": "zero-valued invalid suffix strictly after the sole actor terminal",
        "auxiliaryArrays": ["finish_places", "environment_terminals", "source_steps"],
        "privilegedCriticExportAllowed": False,
    }
    # Preserve the original single-input metadata access while exposing the
    # deterministic ordered list used by merged datasets.
    if len(ordered_inputs) == 1:
        metadata["input"] = ordered_inputs[0]
    arrays = {
        field.name: getattr(dataset.tensors, field.name).cpu().numpy()
        for field in fields(dataset.tensors)
    }
    arrays.update(auxiliary)
    arrays["trajectory_ids"] = np.asarray(trajectory_ids, dtype=np.str_)
    arrays["trajectory_input_sha256s"] = np.asarray(
        trajectory_input_hashes, dtype=np.str_
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(f"{output}.metadata.json")
    checksum_output = Path(f"{output}.sha256")
    for path in (output, metadata_path, checksum_output):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=output.parent, prefix=f".{output.name}.", suffix=".partial", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        npz_temp = Path(handle.name)
    npz_sha = _sha256_file(npz_temp)
    external_metadata = dict(metadata)
    external_metadata["npzSha256"] = npz_sha
    metadata_bytes = (json.dumps(external_metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    metadata_temp = _atomic_temp(output.parent, metadata_path.name, metadata_bytes)
    checksum_temp = _atomic_temp(output.parent, checksum_output.name, f"{npz_sha}\n".encode("ascii"))
    promoted: list[Path] = []
    try:
        _promote_exclusive(metadata_temp, metadata_path); promoted.append(metadata_path)
        _promote_exclusive(checksum_temp, checksum_output); promoted.append(checksum_output)
        _promote_exclusive(npz_temp, output); promoted.append(output)
    except Exception:
        npz_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
        checksum_temp.unlink(missing_ok=True)
        for path in promoted:
            path.unlink(missing_ok=True)
        raise
    return external_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly prepare a V4 rollout NDJSON as a trajectory NPZ.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--input-checksum", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = convert_v4_rollouts(
        args.input, args.output, checksum_path=args.input_checksum,
        repository_root=args.repository_root,
    )
    print(json.dumps({
        "output": str(args.output.resolve()), "fingerprint": metadata["fingerprint"],
        "trajectories": metadata["trajectoryCount"], "npzSha256": metadata["npzSha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["convert_v4_rollouts", "main"]
