from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_FEATURES,
)
from v4_dataset import load_v4_dataset_npz
from v4_prepare_dataset import convert_v4_rollouts


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ROLES = [
    "great-dalmuti", "lesser-dalmuti", "merchant", "lesser-peon",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legal_mask(indices: list[int]) -> str:
    nibbles = [0] * 59
    for index in indices:
        nibbles[index // 4] |= 1 << (index % 4)
    return "".join(format(value, "x") for value in nibbles)


def source_hashes() -> dict[str, str]:
    catalogue = [dict(item) for item in V3_ACTION_CATALOGUE]
    catalogue_bytes = json.dumps(
        {"version": V3_ACTION_CATALOGUE_VERSION, "catalogue": catalogue},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return {
        "actorObservationContract": sha256(REPOSITORY_ROOT / "training/v4-public-history.ts"),
        "privilegedCriticContract": sha256(REPOSITORY_ROOT / "training/simulator.ts"),
        "actionCatalogue": hashlib.sha256(catalogue_bytes).hexdigest(),
        "normalPolicy": sha256(REPOSITORY_ROOT / "lib/bot-strategy.ts"),
        "generator": sha256(REPOSITORY_ROOT / "scripts/rl-generate-v4-rollouts.mjs"),
        "datasetManifest": sha256(REPOSITORY_ROOT / "training/v4-rollout-dataset.ts"),
    }


def manifest(initial_seed: int = 20260801, acts: int = 1) -> dict[str, object]:
    hashes = source_hashes()
    memory_features = [
        "type.play", "type.pass", "type.clear", "type.finish", "actor-offset",
        "hand-count-before", "hand-count-after", "rank", "natural-count",
        "joker-count", "total-count", "pass.manual", "pass.timeout",
        "pass.insufficient-cards", "pass.dalmuti", "clear.all-passed",
        "clear.dalmuti", "clear.act-ended", "next-leader-offset", "finish-place",
    ]
    return {
        "type": "manifest",
        "format": "dalmuti-v4-normal-warmstart-ndjson",
        "formatVersion": 1,
        "environment": {
            "game": "DALMUTI", "rules": "project-house-rules-v1",
            "playerCount": 4, "actsPerEpisode": acts, "initialSeed": initial_seed,
            "behaviorPolicy": "normal",
            "reward": "actorTerminal ? (roundChipAward - 2) / 2 : 0",
            "collection": {
                "mode": "target-non-forced-decisions",
                "targetNonForcedDecisions": 1, "maxEpisodes": 10,
                "completeEpisodesOnly": True,
            },
        },
        "actorObservation": {
            "schemaVersion": 4,
            "sourceSha256": hashes["actorObservationContract"],
            "canonicalBuilder": "buildV4ActorVisibleObservation",
            "maxRecentHistoryEvents": 192,
            "memoryTraceDecays": [0.5, 0.8, 0.95, 0.99],
            "memoryTraceFeatures": memory_features,
            "privacy": "own physical hand plus public state/history only; IDs and opponent hands excluded",
        },
        "privilegedCritic": {
            "schemaVersion": 1,
            "sourceSha256": hashes["privilegedCriticContract"],
            "featureCount": 512,
            "layout": {
                "version": 1,
                "featureCount": 512,
                "global": {
                    "offset": 0,
                    "fields": [
                        "playerCount", "act", "revolution", "table.present",
                        "table.rank", "table.naturalCount", "table.jokerCount",
                        "table.totalCount", "table.actorOffsetOrMinusOne",
                        "publicPlayedTotal", "activePlayerCount",
                        "finishedPlayerCount", "actorRole", "actorScore",
                        "actorHandCount", "publicHistoryEventCount",
                    ],
                },
                "publicPlayedRankCounts": {
                    "offset": 16, "length": 13, "ranks": "1..13",
                },
                "players": {
                    "offset": 29, "seats": 10, "stride": 25,
                    "fields": [
                        "present", "relativeOffset", "role.oneHot[5]", "score",
                        "handCount", "passed", "finished", "finishPlace",
                        "handRankCounts[13]",
                    ],
                },
                "reservedZeroTail": {"offset": 279, "length": 233},
            },
            "actorExportAllowed": False,
            "privacyClass": "restricted-training-only-full-state",
        },
        "actionSpace": {
            "catalogueVersion": 1, "size": 236,
            "catalogueSha256": hashes["actionCatalogue"],
            "catalogue": [dict(item) for item in V3_ACTION_CATALOGUE],
            "encodedActionFeatures": [list(row) for row in V3_ACTION_FEATURES],
            "legalMaskEncoding": {
                "field": "legalMaskHex", "lowercaseHexDigits": 59,
                "bitOrder": "action index i = bit (i % 4) of hex digit floor(i / 4)",
            },
        },
        "sampleBindings": {
            "actionIndex": "236-action catalogue index selected by exact Normal",
            "legalActionIndices": "unique ascending indices exactly equal to legalMaskHex",
            "actorObservation": "canonical sanitized state immediately before the selected action",
            "privilegedCriticState": "separate full-information state immediately before the action",
            "eventsAfterAction": "ordered public play/pass/clear/finish events emitted by the action",
            "forced": "true exactly when legalActionIndices has length one",
        },
        "sourceHashes": hashes,
    }


def actor_observation(actor_index: int) -> dict[str, object]:
    players = []
    for offset in range(4):
        role_index = (actor_index + offset) % 4
        players.append({
            "relativeOffset": offset, "handCount": 1, "finished": 0,
            "passed": 0, "self": int(offset == 0), "tableLeader": 0,
            "role": role_index, "score": role_index,
        })
    own = [0] * 13
    own[0] = 1
    return {
        "schemaVersion": 4, "playerCount": 4, "act": 1,
        "actorRole": actor_index, "revolution": 0,
        "ownHandCounts": own, "publicPlayedCounts": [0] * 13,
        "table": None, "playerTokens": players, "historyTokens": [],
        "memoryTraceVectors": [[0.0] * 20 for _ in range(4)],
        "truncatedHistoryCount": 0,
    }


def privileged_state(actor_index: int) -> dict[str, object]:
    players = []
    for offset in range(4):
        role_index = (actor_index + offset) % 4
        counts = [0] * 13
        counts[0] = 1
        players.append({
            "relativeOffset": offset, "role": ROLES[role_index],
            "score": role_index, "passed": False, "finishPlace": 0,
            "handRankCounts": counts,
        })
    features = [0.0] * 512
    features[:16] = [4, 1, 0, 0, 0, 0, 0, 0, -1, 0, 4, 0, actor_index, actor_index, 1, 0]
    for player in players:
        offset = 29 + player["relativeOffset"] * 25
        role_index = ROLES.index(player["role"])
        features[offset:offset + 25] = [
            1, player["relativeOffset"],
            *[int(index == role_index) for index in range(5)],
            player["score"], 1, 0, 0, 0, *player["handRankCounts"],
        ]
    return {
        "schemaVersion": 1, "playerCount": 4, "act": 1,
        "actorRole": ROLES[actor_index], "revolution": None,
        "publicPlayedCounts": [0] * 13, "table": None,
        "players": players, "features": features,
    }


def sample(
    actor_index: int,
    step: int,
    terminal: bool,
    place: int,
    episode_number: int = 1,
) -> dict[str, object]:
    actor_id = f"player-{actor_index + 1}"
    events: list[dict[str, object]] = [{
        "type": "play", "sequence": step * 2, "actorId": actor_id,
        "handCountBefore": 1, "handCountAfter": 0,
        "rank": 1, "naturalCount": 1, "jokerCount": 0, "totalCount": 1,
    }]
    if terminal:
        events.append({
            "type": "finish", "sequence": step * 2 + 1, "actorId": actor_id,
            "handCountBefore": 0, "handCountAfter": 0, "place": place,
        })
    reward_by_place = {1: 1.0, 2: 0.5, 3: -0.5, 4: -1.0}
    episode_id = f"v4-normal-p4-episode-{episode_number}"
    return {
        "type": "sample", "trajectoryId": f"{episode_id}:act-1:{actor_id}",
        "episodeId": episode_id, "act": 1, "step": step,
        "actorId": actor_id, "actorSeat": actor_index,
        "actorRole": ROLES[actor_index],
        "actorObservation": actor_observation(actor_index),
        "privilegedCriticState": privileged_state(actor_index),
        "legalActionIndices": [2, 5], "legalMaskHex": legal_mask([2, 5]),
        "actionIndex": 2, "reward": reward_by_place[place] if terminal else 0,
        "actorTerminal": terminal, "environmentTerminal": False,
        "finishPlace": place, "forced": False, "eventsAfterAction": events,
    }


def records(
    *, episode_number: int = 1, initial_seed: int = 20260801, acts: int = 1
) -> list[dict[str, object]]:
    # Globally ordered actions form four actor trajectories of lengths 2, 1, 3, 2.
    schedule = [
        (0, False, 2), (1, True, 1), (2, False, 4), (3, False, 3),
        (0, True, 2), (2, False, 4), (3, True, 3), (2, True, 4),
    ]
    result = [manifest(initial_seed, acts)]
    result.extend(
        sample(actor, step, terminal, place, episode_number)
        for step, (actor, terminal, place) in enumerate(schedule)
    )
    result[-1]["environmentTerminal"] = True
    return result


def write_contract(
    directory: Path, values: list[dict[str, object]], name: str = "rollouts.ndjson"
) -> Path:
    before_summary = b"".join(
        (json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        for value in values
    )
    forced = sum(int(value["forced"]) for value in values[1:])
    summary = {
        "type": "summary", "episodes": 1, "samples": len(values) - 1,
        "forcedSamples": forced, "nonForcedSamples": len(values) - 1 - forced,
        "targetNonForcedDecisions": 1,
        "recordsBeforeSummarySha256": hashlib.sha256(before_summary).hexdigest(),
    }
    payload = before_summary + (json.dumps(summary, separators=(",", ":")) + "\n").encode("utf-8")
    path = directory / name
    path.write_bytes(payload)
    Path(f"{path}.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
    return path


class V4PrepareDatasetTest(unittest.TestCase):
    def test_strict_normal_bc_conversion_with_variable_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(directory, records())
            output = directory / "prepared.npz"
            metadata = convert_v4_rollouts(source, output, repository_root=REPOSITORY_ROOT)
            self.assertEqual(metadata["trajectoryCount"], 4)
            self.assertEqual(metadata["maxTimeSteps"], 3)
            self.assertEqual(metadata["syntheticDefaults"], {
                "expertActionIndex": True,
                "oldLogProbability": True,
                "advantage": True,
            })
            self.assertEqual(sha256(output), metadata["npzSha256"])
            self.assertEqual(Path(f"{output}.sha256").read_text().strip(), metadata["npzSha256"])
            dataset = load_v4_dataset_npz(output)
            np.testing.assert_array_equal(
                dataset.tensors.valid_masks.numpy().sum(axis=1), [2, 1, 3, 2]
            )
            np.testing.assert_array_equal(
                dataset.tensors.expert_actions.numpy(), dataset.tensors.actions.numpy()
            )
            self.assertTrue(np.all(dataset.tensors.old_action_log_probs.numpy() == 0))
            self.assertTrue(np.all(dataset.tensors.advantages.numpy() == 0))
            with np.load(output, allow_pickle=False) as archive:
                self.assertIn("finish_places", archive.files)
                self.assertEqual(archive["source_steps"][2].tolist(), [2, 5, 7])
            with self.assertRaises(FileExistsError):
                convert_v4_rollouts(source, output, repository_root=REPOSITORY_ROOT)

    def test_merges_shards_in_input_hash_then_trajectory_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            first = write_contract(
                directory,
                records(episode_number=1, initial_seed=20260801),
                "p4-a.ndjson",
            )
            second = write_contract(
                directory,
                records(episode_number=2, initial_seed=20260802),
                "p4-b.ndjson",
            )
            output = directory / "merged.npz"
            # Reverse CLI order deliberately; output ordering is content-derived.
            metadata = convert_v4_rollouts(
                [second, first], output, repository_root=REPOSITORY_ROOT
            )
            ordered_hashes = sorted([sha256(first), sha256(second)])
            self.assertEqual(
                [item["sha256"] for item in metadata["inputs"]], ordered_hashes
            )
            self.assertEqual(metadata["trajectoryCount"], 8)
            self.assertEqual(metadata["maxTimeSteps"], 3)
            expected_pairs = sorted(
                [
                    (sha256(first), f"v4-normal-p4-episode-1:act-1:player-{index}")
                    for index in range(1, 5)
                ]
                + [
                    (sha256(second), f"v4-normal-p4-episode-2:act-1:player-{index}")
                    for index in range(1, 5)
                ]
            )
            self.assertEqual(
                list(zip(metadata["trajectoryInputSha256s"], metadata["trajectoryIds"])),
                expected_pairs,
            )
            dataset = load_v4_dataset_npz(output)
            np.testing.assert_array_equal(
                sorted(dataset.tensors.valid_masks.numpy().sum(axis=1).tolist()),
                [1, 1, 2, 2, 2, 2, 3, 3],
            )
            self.assertEqual(metadata["syntheticDefaults"], {
                "expertActionIndex": True,
                "oldLogProbability": True,
                "advantage": True,
            })

    def test_rejects_cross_shard_contract_and_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            first = write_contract(directory, records(), "first.ndjson")
            different_acts = write_contract(
                directory,
                records(episode_number=2, initial_seed=20260802, acts=2),
                "different-acts.ndjson",
            )
            with self.assertRaisesRegex(ValueError, "must share acts"):
                convert_v4_rollouts(
                    [first, different_acts], directory / "drift.npz",
                    repository_root=REPOSITORY_ROOT,
                )
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            first = write_contract(directory, records(), "first.ndjson")
            duplicate = write_contract(
                directory, copy.deepcopy(records()), "duplicate.ndjson"
            )
            with self.assertRaisesRegex(ValueError, "duplicate episode ID"):
                convert_v4_rollouts(
                    [first, duplicate], directory / "duplicate.npz",
                    repository_root=REPOSITORY_ROOT,
                )

    def test_rejects_private_actor_field_even_with_recomputed_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            values = records()
            values[1]["actorObservation"]["opponentHands"] = [[1]]
            source = write_contract(Path(directory_value), values)
            with self.assertRaisesRegex(ValueError, "unknown or missing field"):
                convert_v4_rollouts(source, Path(directory_value) / "bad.npz", repository_root=REPOSITORY_ROOT)

    def test_rejects_legal_mask_and_source_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            values = records()
            values[1]["legalMaskHex"] = legal_mask([2])
            source = write_contract(directory, values)
            with self.assertRaisesRegex(ValueError, "exact 59-hex mask"):
                convert_v4_rollouts(source, directory / "mask.npz", repository_root=REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            values = records()
            values[0]["sourceHashes"]["normalPolicy"] = "0" * 64
            source = write_contract(directory, values)
            with self.assertRaisesRegex(ValueError, "source hash drift"):
                convert_v4_rollouts(source, directory / "hash.npz", repository_root=REPOSITORY_ROOT)

    def test_rejects_bad_input_sidecar_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(directory, records())
            Path(f"{source}.sha256").write_text("0" * 64 + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "checksum does not match"):
                convert_v4_rollouts(source, directory / "bad.npz", repository_root=REPOSITORY_ROOT)


if __name__ == "__main__":
    unittest.main()
