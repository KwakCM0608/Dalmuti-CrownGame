from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import v5_dataset
from v4_env import DalmutiScalarEnv
from v5_dataset import (
    V5_ACTION_COUNT,
    load_v5_actor_index,
    load_v5_actor_shard,
    load_v5_index_manifest,
    load_v5_training_shard,
    publish_v5_index_manifest,
    publish_v5_shard,
)
from v5_public import (
    actor_batch_from_packed_arrays,
    pack_v5_public_observations,
    v5_public_from_v4_actor_observation,
)


def _fixture_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Seven tiny complete matches make the fixture cover every p4..p10 stratum.
    decisions = 7
    player_counts = np.arange(4, 11, dtype=np.uint8)
    environments = [
        DalmutiScalarEnv(
            int(player_count),
            acts=1,
            seed=970_000_000 + int(player_count),
        )
        for player_count in player_counts
    ]
    observations = [
        v5_public_from_v4_actor_observation(environment.public_observation())
        for environment in environments
    ]
    public_arrays, history_events, history_end = pack_v5_public_observations(
        observations
    )
    normal_actions = np.asarray(
        [environment.normal_action() for environment in environments],
        dtype=np.uint16,
    )
    actor_ids = np.asarray(
        [environment.current_player_id for environment in environments],
        dtype=np.uint8,
    )
    actor = {
        **public_arrays,
        "match_offsets": np.arange(decisions + 1, dtype=np.uint32),
        "candidate_bitsets": np.asarray(
            [1 << int(actor_id) for actor_id in actor_ids], dtype=np.uint16
        ),
        "player_counts": player_counts,
        "decision_actor_ids": actor_ids,
        "decision_acts": np.ones(decisions, np.uint8),
        "normal_actions": normal_actions,
        "actions": normal_actions.copy(),
        "old_log_probs": np.zeros(decisions, np.float32),
        "old_values": np.linspace(-0.3, 0.3, decisions, dtype=np.float32),
        "reward_to_next": np.linspace(-1.0, 1.0, decisions, dtype=np.float32),
        "done": np.ones(decisions, np.bool_),
        "forced": np.asarray(
            [int(observation.legal_mask.sum()) == 1 for observation in observations],
            dtype=np.bool_,
        ),
        "next_decision": np.full(decisions, -1, np.int32),
        "history_events": history_events,
        "history_end": history_end,
    }
    privileged = {
        "privileged_states": np.arange(decisions * 5, dtype=np.float16).reshape(
            decisions, 5
        )
    }
    return actor, privileged


class V5MmapDatasetTests(unittest.TestCase):
    def test_exact_public_semantics_and_verified_mapping_fail_closed(self) -> None:
        actor, privileged = _fixture_arrays()
        wrong_response = dict(actor)
        wrong_response["belief_response_feasibility"] = actor[
            "belief_response_feasibility"
        ].copy()
        wrong_response["belief_response_feasibility"][0, 0] = np.float32(0.5)
        wrong_finished = dict(actor)
        wrong_finished["player_codes"] = actor["player_codes"].copy()
        wrong_finished["player_codes"][0, 0, 3] ^= np.uint8(1)
        wrong_legal = dict(actor)
        wrong_legal["legal_action_bits"] = actor["legal_action_bits"].copy()
        wrong_legal["legal_action_bits"][0, 0] ^= np.uint8(1)
        with tempfile.TemporaryDirectory() as temporary:
            for index, malformed in enumerate(
                (wrong_response, wrong_finished, wrong_legal)
            ):
                with self.assertRaises(ValueError):
                    publish_v5_shard(
                        Path(temporary) / f"semantic-invalid-{index}",
                        malformed,
                        privileged,
                    )

            target = Path(temporary) / "verified"
            publish_v5_shard(target, actor, privileged)
            loaded = load_v5_actor_shard(target)
            marker = getattr(
                loaded.arrays, "__v5_exact_public_semantics__", None
            )
            self.assertIsInstance(marker, tuple)
            actor_batch_from_packed_arrays(loaded.arrays, [0], "cpu")

            copied = dict(loaded.arrays)
            self.assertIsNone(
                getattr(copied, "__v5_exact_public_semantics__", None)
            )
            copied["belief_response_feasibility"] = np.asarray(
                copied["belief_response_feasibility"]
            ).copy()
            copied["belief_response_feasibility"][0, 0] = np.float32(0.5)
            with self.assertRaisesRegex(ValueError, "exact public value"):
                actor_batch_from_packed_arrays(copied, [0], "cpu")

            class SpoofedMapping(dict[str, np.ndarray]):
                @property
                def __v5_exact_public_semantics__(self):  # type: ignore[no-untyped-def]
                    return marker

            spoofed = SpoofedMapping(copied)
            with self.assertRaisesRegex(ValueError, "exact public value"):
                actor_batch_from_packed_arrays(spoofed, [0], "cpu")
            loaded.close()

            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["semanticValidation"]["validatorVersion"] = 2
            raw = (
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            manifest_path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            (target / "manifest.json.sha256").write_text(
                f"{digest}  manifest.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "semantic-validation"):
                load_v5_actor_shard(target)

    def test_end_to_end_publish_load_and_ragged_history(self) -> None:
        actor, privileged = _fixture_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "shard-a"
            digest = publish_v5_shard(
                target, actor, privileged, metadata={"seed": 7, "split": "train"}
            )
            self.assertEqual(len(digest), 64)
            manifest_bytes = (target / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            expected = (
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            self.assertEqual(manifest_bytes, expected)

            actor_shard = load_v5_actor_shard(target)
            self.assertEqual(actor_shard.decision_count, 7)
            self.assertEqual(actor_shard.match_count, 7)
            self.assertTrue(
                all(isinstance(value, np.memmap) for value in actor_shard.arrays.values())
            )
            np.testing.assert_array_equal(
                actor_shard.history(3), np.zeros((0, 12), dtype=np.uint8)
            )
            expected_mask = v5_public_from_v4_actor_observation(
                DalmutiScalarEnv(4, acts=1, seed=970_000_004).public_observation()
            ).legal_mask
            np.testing.assert_array_equal(actor_shard.legal_mask(0), expected_mask)

            training = load_v5_training_shard(target)
            np.testing.assert_array_equal(
                training.actor.arrays["actions"], actor_shard.arrays["actions"]
            )
            self.assertIsInstance(training.privileged_arrays["privileged_states"], np.memmap)
            np.testing.assert_array_equal(
                training.privileged_arrays["privileged_states"],
                privileged["privileged_states"],
            )
            training.close()
            actor_shard.close()

    def test_match_provenance_is_private_and_malformed_pairs_fail_closed(self) -> None:
        actor, privileged = _fixture_arrays()
        with_provenance = {
            **privileged,
            "match_indices": np.arange(7, dtype=np.uint32),
            "match_seeds": np.arange(4_000, 4_007, dtype=np.uint32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "provenance"
            publish_v5_shard(target, actor, with_provenance)

            actor_shard = load_v5_actor_shard(target)
            self.assertNotIn("match_indices", actor_shard.arrays)
            self.assertNotIn("match_seeds", actor_shard.arrays)
            actor_shard.close()

            training = load_v5_training_shard(target)
            np.testing.assert_array_equal(
                training.privileged_arrays["match_indices"],
                with_provenance["match_indices"],
            )
            np.testing.assert_array_equal(
                training.privileged_arrays["match_seeds"],
                with_provenance["match_seeds"],
            )
            training.close()

            partial = dict(with_provenance)
            del partial["match_seeds"]
            with self.assertRaisesRegex(ValueError, "one provenance pair"):
                publish_v5_shard(root / "partial", actor, partial)

            wrong_dtype = dict(with_provenance)
            wrong_dtype["match_indices"] = with_provenance[
                "match_indices"
            ].astype(np.int64)
            with self.assertRaisesRegex(ValueError, "uint32"):
                publish_v5_shard(root / "wrong-dtype", actor, wrong_dtype)

            duplicate_seed = dict(with_provenance)
            duplicate_seed["match_seeds"] = with_provenance["match_seeds"].copy()
            duplicate_seed["match_seeds"][1] = duplicate_seed["match_seeds"][0]
            with self.assertRaisesRegex(ValueError, "repeat"):
                publish_v5_shard(root / "duplicate-seed", actor, duplicate_seed)

    def test_actor_loader_never_opens_privileged_partition(self) -> None:
        actor, privileged = _fixture_arrays()
        privileged = {
            **privileged,
            "match_indices": np.arange(7, dtype=np.uint32),
            "match_seeds": np.arange(5_000, 5_007, dtype=np.uint32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "privacy"
            publish_v5_shard(target, actor, privileged)
            original_hash = v5_dataset._sha256_file
            original_load = v5_dataset.np.load

            def guarded_hash(path: Path) -> str:
                if "privileged" in Path(path).parts:
                    raise AssertionError("actor loader opened privileged checksum input")
                return original_hash(path)

            def guarded_load(path: object, *args: object, **kwargs: object) -> object:
                if "privileged" in Path(path).parts:
                    raise AssertionError("actor loader opened privileged NPY")
                return original_load(path, *args, **kwargs)

            with mock.patch.object(
                v5_dataset, "_sha256_file", side_effect=guarded_hash
            ), mock.patch.object(v5_dataset.np, "load", side_effect=guarded_load):
                loaded = load_v5_actor_shard(target)
            self.assertEqual(loaded.decision_count, 7)
            loaded.close()

    def test_privileged_tamper_does_not_enter_actor_loader_but_training_fails(self) -> None:
        actor, privileged = _fixture_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "privacy-tamper"
            publish_v5_shard(target, actor, privileged)
            path = target / "privileged" / "privileged_states.npy"
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                previous = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([previous[0] ^ 0x01]))
            # Actor inference remains independent of critic-only bytes.
            loaded = load_v5_actor_shard(target)
            self.assertEqual(loaded.decision_count, 7)
            loaded.close()
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_v5_training_shard(target)

    def test_actor_tamper_is_rejected_before_numpy_load(self) -> None:
        actor, privileged = _fixture_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "tamper"
            publish_v5_shard(target, actor, privileged)
            path = target / "actor" / "actions.npy"
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                previous = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([previous[0] ^ 0x01]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_v5_actor_shard(target)

    def test_shape_dtype_and_bitpacking_contracts_fail_closed(self) -> None:
        actor, privileged = _fixture_arrays()
        mutations = []
        wrong_dtype = dict(actor)
        wrong_dtype["next_decision"] = actor["next_decision"].astype(np.int64)
        mutations.append(wrong_dtype)
        wrong_history = dict(actor)
        wrong_history["history_end"] = np.ones(7, np.uint32)
        mutations.append(wrong_history)
        wrong_shape = dict(actor)
        wrong_shape["old_values"] = actor["old_values"][:-1]
        mutations.append(wrong_shape)
        illegal_trailing = dict(actor)
        illegal_trailing["legal_action_bits"] = actor["legal_action_bits"].copy()
        illegal_trailing["legal_action_bits"][0, -1] = 0x80
        mutations.append(illegal_trailing)
        wrong_forced = dict(actor)
        wrong_forced["forced"] = np.ones(7, np.bool_)
        mutations.append(wrong_forced)
        with tempfile.TemporaryDirectory() as temporary:
            for index, mutation in enumerate(mutations):
                target = Path(temporary) / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    publish_v5_shard(target, mutation, privileged)
                self.assertFalse(target.exists())

    def test_publish_is_exclusive_and_never_overwrites(self) -> None:
        actor, privileged = _fixture_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "immutable"
            first = publish_v5_shard(target, actor, privileged)
            before = (target / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                publish_v5_shard(target, actor, privileged)
            self.assertEqual((target / "manifest.json").read_bytes(), before)
            self.assertEqual(first, (target / "manifest.json.sha256").read_text().split()[0])

    def test_directory_publish_is_race_safe_crash_tolerant_and_durable(self) -> None:
        def fsynced_builder(payload: bytes, barrier: threading.Barrier | None = None):
            def build(staging: Path) -> None:
                with (staging / "payload.bin").open("xb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                if barrier is not None:
                    barrier.wait(timeout=10)

            return build

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            # A process crash in the retired lock implementation must no longer
            # brick a fresh immutable publication.  Foreign staging is ignored.
            recovered = root / "recovered"
            (root / ".recovered.publish.lock").write_text("pid=dead\n", encoding="ascii")
            orphan = root / ".recovered.staging-crash"
            orphan.mkdir()
            v5_dataset._exclusive_publish_directory(
                recovered, fsynced_builder(b"recovered")
            )
            self.assertEqual((recovered / "payload.bin").read_bytes(), b"recovered")
            self.assertTrue(orphan.is_dir())

            empty = root / "empty-existing"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                v5_dataset._exclusive_publish_directory(
                    empty, fsynced_builder(b"must-not-replace")
                )
            self.assertEqual(list(empty.iterdir()), [])

            target = root / "concurrent"
            barrier = threading.Barrier(2)

            def publish(payload: bytes) -> str:
                try:
                    v5_dataset._exclusive_publish_directory(
                        target, fsynced_builder(payload, barrier)
                    )
                    return "published"
                except FileExistsError:
                    return "exists"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(publish, (b"first", b"second")))
            self.assertCountEqual(outcomes, ["published", "exists"])
            self.assertIn((target / "payload.bin").read_bytes(), (b"first", b"second"))
            self.assertEqual(list(root.glob(".concurrent.staging-*")), [])

            durable = root / "durable"
            fsynced: list[Path] = []
            original_fsync_directory = v5_dataset._fsync_directory

            def observe_fsync(path: Path) -> None:
                fsynced.append(Path(path))
                original_fsync_directory(path)

            with mock.patch.object(
                v5_dataset, "_fsync_directory", side_effect=observe_fsync
            ):
                v5_dataset._exclusive_publish_directory(
                    durable, fsynced_builder(b"durable")
                )
            self.assertEqual(fsynced[-1], root)
            self.assertTrue(any(path.name.startswith(".durable.staging-") for path in fsynced))

    def test_duplicate_actor_loaders_have_independent_close_lifetimes(self) -> None:
        actor, privileged = _fixture_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "duplicate-load"
            publish_v5_shard(target, actor, privileged)
            first = load_v5_actor_shard(target)
            second = load_v5_actor_shard(target)
            first.close()
            # Closing the first loader must not invalidate a separately opened
            # mmap of the same immutable file.
            self.assertEqual(
                int(second.arrays["actions"][0]), int(actor["actions"][0])
            )
            second.close()

    def test_zero_copy_index_references_mmaps_without_copying_npy(self) -> None:
        actor, privileged = _fixture_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "shard-1"
            second = root / "shard-2"
            publish_v5_shard(first, actor, privileged, metadata={"shard": 1})
            publish_v5_shard(second, actor, privileged, metadata={"shard": 2})
            index_path = root / "merged-index"
            digest = publish_v5_index_manifest(index_path, [second, first])
            self.assertEqual(len(digest), 64)
            self.assertEqual(list(index_path.rglob("*.npy")), [])
            index = load_v5_index_manifest(index_path)
            self.assertEqual(index.decision_count, 14)
            self.assertEqual(index.match_count, 14)
            shards = load_v5_actor_index(index_path)
            self.assertEqual(len(shards), 2)
            self.assertTrue(
                all(isinstance(shard.arrays["actions"], np.memmap) for shard in shards)
            )
            for shard in shards:
                shard.close()
            index.close()


if __name__ == "__main__":
    unittest.main()
