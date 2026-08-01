from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_v4_prepare_dataset import legal_mask, manifest
from v4_label_search_teacher import (
    _episode_shard_index,
    _output_paths,
    label_v4_search_teacher,
)
from v4_prepare_dataset import _read_and_validate


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DECK_COUNTS = list(range(1, 13)) + [2]
SOCIAL_ROLE_IDS = [0, 1, 3, 4]
ROLE_NAMES = [
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
]
PLAY_ELEVEN = 167
PLAY_TWELVE = 200


def _rank_counts(*ranks: int) -> list[int]:
    result = [0] * 13
    for rank in ranks:
        result[rank - 1] += 1
    return result


def actor_observation(actor_seat: int, stage: int) -> dict[str, object]:
    own = _rank_counts(11, 12) if stage == 0 else _rank_counts(11)
    hidden = [own, _rank_counts(4, 4), _rank_counts(5, 5), _rank_counts(6, 6)]
    played = [
        copies - sum(hand[index] for hand in hidden)
        for index, copies in enumerate(DECK_COUNTS)
    ]
    roles = [
        SOCIAL_ROLE_IDS[(actor_seat + offset) % 4]
        for offset in range(4)
    ]
    return {
        "schemaVersion": 4,
        "playerCount": 4,
        "act": 1,
        "actorRole": SOCIAL_ROLE_IDS[actor_seat],
        "revolution": 0,
        "ownHandCounts": own,
        "publicPlayedCounts": played,
        "table": None,
        "playerTokens": [
            {
                "relativeOffset": offset,
                "handCount": sum(hidden[offset]),
                "finished": 0,
                "passed": 0,
                "self": int(offset == 0),
                "tableLeader": 0,
                "role": roles[offset],
                "score": 0,
            }
            for offset in range(4)
        ],
        "historyTokens": [],
        "memoryTraceVectors": [[0.0] * 20 for _ in range(4)],
        "truncatedHistoryCount": 0,
    }


def privileged_state(actor_seat: int, stage: int) -> dict[str, object]:
    observation = actor_observation(actor_seat, stage)
    roles = [
        SOCIAL_ROLE_IDS[(actor_seat + offset) % 4]
        for offset in range(4)
    ]
    hands = [
        list(observation["ownHandCounts"]),
        _rank_counts(4, 4),
        _rank_counts(5, 5),
        _rank_counts(6, 6),
    ]
    players = [
        {
            "relativeOffset": offset,
            "role": ROLE_NAMES[roles[offset]],
            "score": 0,
            "passed": False,
            "finishPlace": 0,
            "handRankCounts": hands[offset],
        }
        for offset in range(4)
    ]
    features = [0.0] * 512
    public_counts = list(observation["publicPlayedCounts"])
    features[:16] = [
        4,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        sum(public_counts),
        4,
        0,
        SOCIAL_ROLE_IDS[actor_seat],
        0,
        sum(hands[0]),
        0,
    ]
    features[16:29] = public_counts
    for player in players:
        offset = 29 + int(player["relativeOffset"]) * 25
        role_id = ROLE_NAMES.index(str(player["role"]))
        hand = list(player["handRankCounts"])
        features[offset : offset + 25] = [
            1,
            player["relativeOffset"],
            *[int(index == role_id) for index in range(5)],
            0,
            sum(hand),
            0,
            0,
            0,
            *hand,
        ]
    return {
        "schemaVersion": 1,
        "playerCount": 4,
        "act": 1,
        "actorRole": ROLE_NAMES[SOCIAL_ROLE_IDS[actor_seat]],
        "revolution": None,
        "publicPlayedCounts": public_counts,
        "table": None,
        "players": players,
        "features": features,
    }


def sample(
    actor_seat: int,
    stage: int,
    *,
    episode_id: str | None = None,
    step: int | None = None,
) -> dict[str, object]:
    episode_id = episode_id or f"v4-teacher-fixture-role-{actor_seat}"
    actor_id = f"actor-{actor_seat}"
    terminal = stage == 1
    action = PLAY_TWELVE if stage == 0 else PLAY_ELEVEN
    legal = [PLAY_ELEVEN, PLAY_TWELVE] if stage == 0 else [PLAY_ELEVEN]
    before = 2 if stage == 0 else 1
    rank = 12 if stage == 0 else 11
    events: list[dict[str, object]] = [
        {
            "type": "play",
            "sequence": stage * 2,
            "actorId": actor_id,
            "handCountBefore": before,
            "handCountAfter": before - 1,
            "rank": rank,
            "naturalCount": 1,
            "jokerCount": 0,
            "totalCount": 1,
        }
    ]
    if terminal:
        events.append({
            "type": "finish",
            "sequence": stage * 2 + 1,
            "actorId": actor_id,
            "handCountBefore": 0,
            "handCountAfter": 0,
            "place": 1,
        })
    return {
        "type": "sample",
        "trajectoryId": f"{episode_id}:act-1:{actor_id}",
        "episodeId": episode_id,
        "act": 1,
        "step": stage if step is None else step,
        "actorId": actor_id,
        "actorSeat": actor_seat,
        "actorRole": ROLE_NAMES[SOCIAL_ROLE_IDS[actor_seat]],
        "actorObservation": actor_observation(actor_seat, stage),
        "privilegedCriticState": privileged_state(actor_seat, stage),
        "legalActionIndices": legal,
        "legalMaskHex": legal_mask(legal),
        "actionIndex": action,
        "reward": 1.0 if terminal else 0.0,
        "actorTerminal": terminal,
        "environmentTerminal": False,
        "finishPlace": 1,
        "forced": len(legal) == 1,
        "eventsAfterAction": events,
    }


def fixture_records() -> list[dict[str, object]]:
    contract = manifest(initial_seed=20260801, acts=1)
    contract["environment"]["collection"]["targetNonForcedDecisions"] = 4
    records: list[dict[str, object]] = [contract]
    for actor_seat in range(4):
        records.extend((sample(actor_seat, 0), sample(actor_seat, 1)))
    return records


def partition_fixture_records(episode_ids: list[str]) -> list[dict[str, object]]:
    contract = manifest(initial_seed=20260801, acts=1)
    contract["environment"]["collection"]["targetNonForcedDecisions"] = (
        len(episode_ids) * 4
    )
    contract["environment"]["collection"]["maxEpisodes"] = len(episode_ids)
    records: list[dict[str, object]] = [contract]
    for episode_id in episode_ids:
        for actor_seat in range(4):
            records.extend((
                sample(
                    actor_seat,
                    0,
                    episode_id=episode_id,
                    step=actor_seat * 2,
                ),
                sample(
                    actor_seat,
                    1,
                    episode_id=episode_id,
                    step=actor_seat * 2 + 1,
                ),
            ))
    return records


def write_contract(
    directory: Path,
    records: list[dict[str, object]],
    name: str = "normal.ndjson",
) -> Path:
    body = b"".join(
        (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for record in records
    )
    samples = records[1:]
    forced = sum(int(bool(record["forced"])) for record in samples)
    episodes = {str(record["episodeId"]) for record in samples}
    nonforced = len(samples) - forced
    summary = {
        "type": "summary",
        "episodes": len(episodes),
        "samples": len(samples),
        "forcedSamples": forced,
        "nonForcedSamples": nonforced,
        "targetNonForcedDecisions": records[0]["environment"]["collection"][
            "targetNonForcedDecisions"
        ],
        "recordsBeforeSummarySha256": hashlib.sha256(body).hexdigest(),
    }
    payload = body + (
        json.dumps(summary, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path = directory / name
    path.write_bytes(payload)
    Path(f"{path}.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )
    return path


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_label(
    source: Path,
    output: Path,
    *,
    target: int = 2,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, object]:
    return label_v4_search_teacher(
        source,
        output,
        repository_root=REPOSITORY_ROOT,
        seed=99117,
        target_trajectories=target,
        hypotheses=2,
        rollouts_per_action=1,
        max_evaluations=8,
        max_rollout_steps=100,
        selection="mean",
        lcb_z=0.0,
        shard_count=shard_count,
        shard_index=shard_index,
    )


def ids_hash(values: set[str]) -> str:
    payload = json.dumps(
        sorted(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def episode_ids_for_all_shards(shard_count: int, seed: int) -> list[str]:
    by_shard: dict[int, str] = {}
    candidate = 0
    while len(by_shard) < shard_count:
        episode_id = f"v4-teacher-partition-{candidate}"
        assigned = _episode_shard_index(
            episode_id, seed=seed, shard_count=shard_count
        )
        by_shard.setdefault(assigned, episode_id)
        candidate += 1
    return [by_shard[index] for index in range(shard_count)]


def fake_search_teacher(_observation, legal, _adapter, *, config, **_kwargs):
    legal_actions = [index for index, allowed in enumerate(legal) if allowed]
    diagnostics = SimpleNamespace(
        incomplete_legal_actions=(),
        batched_leaf_evaluations=0,
        terminal_evaluations=1,
        evaluations=1,
        legal_action_count=len(legal_actions),
        hypotheses_generated=1,
        unique_determinizations=1,
        stopped_reason="completed",
    )
    return SimpleNamespace(
        teacher_action=legal_actions[0],
        diagnostics=diagnostics,
    )


class V4SearchTeacherLabelTests(unittest.TestCase):
    def test_deterministic_balanced_full_trajectory_labels_and_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(directory, fixture_records())
            first = directory / "first.ndjson"
            second = directory / "second.ndjson"
            first_metadata = run_label(source, first)
            second_metadata = run_label(source, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first.with_suffix(".teacher-metadata.json").read_bytes(),
                second.with_suffix(".teacher-metadata.json").read_bytes(),
            )
            self.assertEqual(first_metadata, second_metadata)
            self.assertEqual(first_metadata["selection"]["selectedTrajectories"], 2)
            self.assertEqual(first_metadata["samples"], {
                "total": 4, "forced": 2, "nonForced": 2,
            })
            self.assertEqual(sum(
                count > 0
                for count in first_metadata["selection"]["balancedStrata"].values()
            ), 2)
            self.assertGreaterEqual(first_metadata["changedVsNormal"]["rate"], 0.0)
            self.assertLessEqual(first_metadata["changedVsNormal"]["rate"], 1.0)

            output_records = read_records(first)
            manifest_record, *middle, summary = output_records
            self.assertEqual(
                manifest_record["environment"]["collection"]["targetNonForcedDecisions"],
                2,
            )
            self.assertEqual(summary["samples"], 4)
            self.assertEqual(summary["nonForcedSamples"], 2)
            by_trajectory: dict[str, list[dict[str, object]]] = {}
            for record in middle:
                by_trajectory.setdefault(str(record["trajectoryId"]), []).append(record)
                self.assertIn(record["expertActionIndex"], record["legalActionIndices"])
                if record["forced"]:
                    self.assertEqual(record["expertActionIndex"], record["actionIndex"])
            self.assertEqual(len(by_trajectory), 2)
            for trajectory in by_trajectory.values():
                trajectory.sort(key=lambda record: int(record["step"]))
                self.assertEqual(len(trajectory), 2)
                self.assertFalse(trajectory[0]["actorTerminal"])
                self.assertTrue(trajectory[-1]["actorTerminal"])

            input_by_key = {
                (record["trajectoryId"], record["step"]): record
                for record in read_records(source)[1:-1]
            }
            for record in middle:
                original = input_by_key[(record["trajectoryId"], record["step"])]
                for field in (
                    "actionIndex", "reward", "privilegedCriticState",
                ):
                    self.assertEqual(record[field], original[field])

            output_sha = hashlib.sha256(first.read_bytes()).hexdigest()
            self.assertEqual(Path(f"{first}.sha256").read_text().strip(), output_sha)
            metadata_path = first.with_suffix(".teacher-metadata.json")
            metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            self.assertEqual(
                Path(f"{metadata_path}.sha256").read_text().strip(), metadata_sha
            )
            validated_manifest, validated_samples, _, _ = _read_and_validate(
                first, REPOSITORY_ROOT
            )
            self.assertEqual(validated_manifest["optionalFields"], ["expertActionIndex"])
            self.assertEqual(len(validated_samples), 4)

    def test_default_shard_is_byte_compatible_with_explicit_single_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(directory, fixture_records())
            implicit = directory / "implicit.ndjson"
            explicit = directory / "explicit.ndjson"
            implicit_metadata = label_v4_search_teacher(
                source,
                implicit,
                repository_root=REPOSITORY_ROOT,
                seed=99117,
                target_trajectories=2,
                hypotheses=2,
                rollouts_per_action=1,
                max_evaluations=8,
                max_rollout_steps=100,
                selection="mean",
                lcb_z=0.0,
            )
            explicit_metadata = run_label(
                source,
                explicit,
                target=2,
                shard_count=1,
                shard_index=0,
            )

            self.assertEqual(implicit.read_bytes(), explicit.read_bytes())
            self.assertEqual(implicit_metadata, explicit_metadata)
            self.assertEqual(
                implicit.with_suffix(".teacher-metadata.json").read_bytes(),
                explicit.with_suffix(".teacher-metadata.json").read_bytes(),
            )
            sharding = implicit_metadata["episodeSharding"]
            self.assertEqual(sharding["shardCount"], 1)
            self.assertEqual(sharding["shardIndex"], 0)
            self.assertEqual(sharding["eligibleEpisodes"], 4)
            self.assertEqual(sharding["eligibleTrajectories"], 4)

    def test_episode_partitions_cover_input_are_disjoint_and_merge_safe(self) -> None:
        seed = 99117
        shard_count = 3
        episode_ids = episode_ids_for_all_shards(shard_count, seed)
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(
                directory,
                partition_fixture_records(episode_ids),
            )
            episode_sets: list[set[str]] = []
            trajectory_sets: list[set[str]] = []
            namespaces: set[str] = set()
            with mock.patch(
                "v4_label_search_teacher.run_v4_search_teacher",
                side_effect=fake_search_teacher,
            ):
                for shard_index in range(shard_count):
                    output = directory / f"shard-{shard_index}.ndjson"
                    metadata = run_label(
                        source,
                        output,
                        target=4,
                        shard_count=shard_count,
                        shard_index=shard_index,
                    )
                    _, validated, _, _ = _read_and_validate(
                        output, REPOSITORY_ROOT
                    )
                    self.assertEqual(len(validated), 8)
                    middle = read_records(output)[1:-1]
                    shard_episodes = {
                        str(record["episodeId"]) for record in middle
                    }
                    shard_trajectories = {
                        str(record["trajectoryId"]) for record in middle
                    }
                    self.assertEqual(len(shard_episodes), 1)
                    self.assertEqual(len(shard_trajectories), 4)
                    sharding = metadata["episodeSharding"]
                    self.assertEqual(sharding["eligibleEpisodes"], 1)
                    self.assertEqual(sharding["eligibleTrajectories"], 4)
                    self.assertEqual(
                        sharding["eligibleEpisodeIdsSha256"],
                        ids_hash(shard_episodes),
                    )
                    self.assertEqual(
                        sharding["eligibleTrajectoryIdsSha256"],
                        ids_hash(shard_trajectories),
                    )
                    self.assertEqual(
                        sharding["selectedEpisodeIdsSha256"],
                        ids_hash(shard_episodes),
                    )
                    self.assertEqual(
                        sharding["selectedTrajectoryIdsSha256"],
                        ids_hash(shard_trajectories),
                    )
                    namespaces.add(str(sharding["outputNamespace"]))
                    episode_sets.append(shard_episodes)
                    trajectory_sets.append(shard_trajectories)

            self.assertEqual(len(namespaces), shard_count)
            self.assertEqual(set().union(*episode_sets), set(episode_ids))
            self.assertEqual(
                sum(len(values) for values in episode_sets),
                len(set().union(*episode_sets)),
            )
            self.assertEqual(
                sum(len(values) for values in trajectory_sets),
                len(set().union(*trajectory_sets)),
            )

    def test_shard_selection_is_deterministic(self) -> None:
        seed = 99117
        episode_ids = episode_ids_for_all_shards(2, seed)
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(
                directory,
                partition_fixture_records(episode_ids),
            )
            first = directory / "first-shard.ndjson"
            second = directory / "second-shard.ndjson"
            with mock.patch(
                "v4_label_search_teacher.run_v4_search_teacher",
                side_effect=fake_search_teacher,
            ):
                first_metadata = run_label(
                    source, first, target=3, shard_count=2, shard_index=0
                )
                second_metadata = run_label(
                    source, second, target=3, shard_count=2, shard_index=0
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_metadata, second_metadata)
            self.assertEqual(
                first.with_suffix(".teacher-metadata.json").read_bytes(),
                second.with_suffix(".teacher-metadata.json").read_bytes(),
            )

    def test_invalid_empty_and_oversubscribed_shards_are_rejected(self) -> None:
        seed = 99117
        episode_ids = episode_ids_for_all_shards(2, seed)
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(
                directory,
                partition_fixture_records([episode_ids[0]]),
            )
            invalid = (
                (0, 0, "shard-count"),
                (2, -1, "shard-index"),
                (2, 2, "shard-index"),
            )
            for case, (shard_count, shard_index, message) in enumerate(invalid):
                with self.subTest(shard_count=shard_count, shard_index=shard_index):
                    with self.assertRaisesRegex(ValueError, message):
                        run_label(
                            source,
                            directory / f"invalid-{case}.ndjson",
                            target=1,
                            shard_count=shard_count,
                            shard_index=shard_index,
                        )
            with self.assertRaisesRegex(ValueError, "is empty"):
                run_label(
                    source,
                    directory / "empty.ndjson",
                    target=1,
                    shard_count=2,
                    shard_index=1,
                )
            with self.assertRaisesRegex(
                ValueError, "exceeds available trajectories 4"
            ):
                run_label(
                    source,
                    directory / "oversubscribed.ndjson",
                    target=5,
                    shard_count=2,
                    shard_index=0,
                )

    def test_private_payload_and_checksum_corruption_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(directory, fixture_records())
            source.write_bytes(source.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_label(source, directory / "bad-sha.ndjson", target=1)

        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            records = fixture_records()
            records[1]["actorObservation"]["opponentHiddenHands"] = [[1, 2, 3]]
            source = write_contract(directory, records)
            with self.assertRaisesRegex(ValueError, "unknown|boundary|field"):
                run_label(source, directory / "private.ndjson", target=1)

    def test_overwrite_is_refused_and_publish_failure_cleans_partials(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            source = write_contract(directory, fixture_records())
            output = directory / "labeled.ndjson"
            run_label(source, output, target=1)
            preserved = {
                path: path.read_bytes() for path in _output_paths(output)
            }
            with self.assertRaises(FileExistsError):
                run_label(source, output, target=1)
            self.assertEqual(
                preserved, {path: path.read_bytes() for path in _output_paths(output)}
            )

            failed = directory / "failed.ndjson"
            real_link = os.link
            calls = 0

            def fail_second_link(source_path, destination_path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected promotion failure")
                return real_link(source_path, destination_path)

            with mock.patch(
                "v4_label_search_teacher.os.link", side_effect=fail_second_link
            ):
                with self.assertRaisesRegex(OSError, "promotion failure"):
                    run_label(source, failed, target=1)
            self.assertTrue(all(not path.exists() for path in _output_paths(failed)))
            self.assertEqual(list(directory.glob("*.partial")), [])
            self.assertEqual(list(directory.glob(".*.partial")), [])


if __name__ == "__main__":
    unittest.main()
