from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from test_v5_model import public_batch, tiny_actor_config
from v5_model import V5PublicActor
from v6_pretrain import (
    V6MatchRecord,
    V6MatchView,
    V6PretrainConfig,
    _actor_training_weights,
    _pilot_matches,
    _strict_dataset_index,
    _strict_split_manifest,
    freeze_and_zero_residual,
    load_v6_split_dataset,
    monte_carlo_targets_by_shard,
    residual_zero_receipt,
)


def _write_json_with_sidecar(path: Path, value: object) -> str:
    raw = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


class _FakeTrainingShard:
    def __init__(self) -> None:
        players = np.asarray(list(range(4, 11)) * 3, dtype=np.uint8)
        matches = len(players)
        global_codes = np.zeros((matches, 6), dtype=np.int32)
        global_codes[:, 1] = players
        self.actor = SimpleNamespace(
            match_count=matches,
            arrays={
                "match_offsets": np.arange(matches + 1, dtype=np.uint32),
                "player_counts": players,
                "forced": np.zeros(matches, dtype=np.bool_),
                "global_codes": global_codes,
            },
        )
        self.privileged_arrays = {
            "match_indices": np.asarray(
                [split for split in range(3) for _ in range(7)], dtype=np.uint32
            ),
            "match_seeds": np.arange(1000, 1000 + matches, dtype=np.uint32),
        }
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _split_manifest(shard_sha: str) -> dict[str, object]:
    splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split_ordinal, split in enumerate(splits):
        for p_offset, player_count in enumerate(range(4, 11)):
            local = split_ordinal * 7 + p_offset
            splits[split].append(
                {
                    "decisionEnd": local + 1,
                    "decisionStart": local,
                    "decisionCount": 1,
                    "localMatchIndex": local,
                    "matchIndex": split_ordinal,
                    "matchSeed": 1000 + local,
                    "nonforcedDecisionCount": 1,
                    "playerCount": player_count,
                    "shardManifestSha256": shard_sha,
                    # These intentionally do not describe the live index.
                    "shardOrdinal": 999,
                    "shardRelativePath": "stale/moved/shard",
                    "splitHash": hashlib.sha256(
                        f"{split}-{player_count}".encode("ascii")
                    ).hexdigest(),
                    "split": split,
                }
            )
    return {
        "assignment": {},
        "corpusIdentitySha256": "a" * 64,
        "format": "dalmuti-v6-match-disjoint-split",
        "privacy": {},
        "sourceCounts": {},
        "sourceIndex": "C:/obsolete/location",
        "splits": splits,
        "summary": {},
        "version": 1,
    }


class V6PretrainTests(unittest.TestCase):
    def test_manifest_readers_accept_only_exact_producer_canonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_split = root / "split.json"
            valid_split.write_bytes(b'{"a":1}\n')
            self.assertEqual(_strict_split_manifest(valid_split), {"a": 1})

            valid_index = root / "index.json"
            valid_index.write_bytes(b'{"metadata":{}}\n')
            self.assertEqual(_strict_dataset_index(valid_index), {"metadata": {}})

            malformed = {
                "missing trailing newline": b'{"a":1}',
                "extra JSON whitespace": b'{"a": 1}\n',
                "extra trailing whitespace": b'{"a":1}\n\n',
                "duplicate key": b'{"a":1,"a":2}\n',
                "non-finite": b'{"a":NaN}\n',
                "raw non-ASCII": '{"a":"한"}\n'.encode("utf-8"),
            }
            for label, raw in malformed.items():
                path = root / f"bad-{len(label)}-{hashlib.sha256(raw).hexdigest()}.json"
                path.write_bytes(raw)
                for reader in (_strict_split_manifest, _strict_dataset_index):
                    with self.subTest(label=label, reader=reader.__name__):
                        with self.assertRaises(ValueError):
                            reader(path)

    def test_config_supports_small_pilot_and_rejects_empty_fraction(self) -> None:
        self.assertEqual(V6PretrainConfig(train_fraction=0.1).train_fraction, 0.1)
        with self.assertRaisesRegex(ValueError, "train_fraction"):
            V6PretrainConfig(train_fraction=0.0)

    def test_pilot_selects_only_whole_matches_per_player_count(self) -> None:
        records = tuple(
            V6MatchRecord(
                split="train",
                player_count=player,
                split_hash=hashlib.sha256(f"{player}-{match}".encode()).hexdigest(),
                shard_manifest_sha256="b" * 64,
                local_match_index=match,
                decision_start=match * 7,
                decision_end=match * 7 + 7,
                match_index=match,
                match_seed=player * 100 + match,
                nonforced_decision_count=5,
            )
            for player in range(4, 11)
            for match in range(10)
        )
        selected = _pilot_matches(records, 0.1)
        self.assertEqual(len(selected), 7)
        self.assertEqual({record.player_count for record in selected}, set(range(4, 11)))
        self.assertTrue(all(record.decision_count == 7 for record in selected))

    def test_portable_split_remaps_by_manifest_sha_not_saved_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_root = root / "moved-index"
            shard_root = root / "actually-here" / "shard"
            shard_root.mkdir(parents=True)
            shard_sha = _write_json_with_sidecar(shard_root / "manifest.json", {})
            _write_json_with_sidecar(
                index_root / "manifest.json",
                {"metadata": {"sourceIndexManifestSha256": "a" * 64}},
            )
            split_path = root / "portable-split.json"
            _write_json_with_sidecar(split_path, _split_manifest(shard_sha))
            fake = _FakeTrainingShard()
            with (
                mock.patch(
                    "v6_pretrain.load_v5_index_manifest",
                    return_value=SimpleNamespace(shard_paths=(shard_root,)),
                ),
                mock.patch(
                    "v6_pretrain.load_v5_training_shard", return_value=fake
                ),
            ):
                dataset = load_v6_split_dataset(index_root, split_path)
            try:
                self.assertEqual(dataset.views["train"].match_count, 7)
                self.assertEqual(set(dataset.shards), {shard_sha})
            finally:
                dataset.close()
            self.assertTrue(fake.closed)

    def test_actor_weights_have_equal_p_mass_and_bounded_local_class_ratio(self) -> None:
        labels: list[int] = []
        players: list[int] = []
        for player in range(4, 11):
            labels.extend([0] * (player * 2) + [1, 2])
            players.extend([player] * (player * 2 + 2))
        label_array = np.asarray(labels, dtype=np.int64)
        player_array = np.asarray(players, dtype=np.int64)
        weights, report = _actor_training_weights(
            label_array,
            player_array,
            exponent=0.5,
            maximum_ratio=3.0,
        )
        masses = [
            float(weights[player_array == player].sum(dtype=np.float64))
            for player in range(4, 11)
        ]
        self.assertLess(max(masses) - min(masses), 1.0e-5)
        for record in report["perPlayerCount"].values():
            self.assertLessEqual(record["realizedClassWeightRatio"], 3.0 + 1e-6)

    def test_residual_is_frozen_zero_and_preserves_exact_normal_greedy(self) -> None:
        torch.manual_seed(9101)
        actor = V5PublicActor(tiny_actor_config())
        with torch.no_grad():
            actor.residual_output.weight.normal_()
            actor.residual_output.bias.fill_(2.0)
        receipt = freeze_and_zero_residual(actor)
        self.assertTrue(receipt["exactZero"])
        self.assertTrue(receipt["frozen"])
        self.assertEqual(receipt, residual_zero_receipt(actor))
        batch, normal = public_batch()
        output = actor.forward_packed_batch(batch, normal)
        self.assertTrue(torch.equal(output.greedy_actions(), output.normal_actions))

    def test_match_mc_targets_preserve_complete_actor_chains(self) -> None:
        arrays = {
            "reward_to_next": np.asarray([0.0, 0.2, 1.0, -1.0], np.float32),
            "next_decision": np.asarray([2, 3, -1, -1], np.int32),
            "done": np.asarray([False, False, True, True], np.bool_),
            "decision_actor_ids": np.asarray([0, 1, 0, 1], np.uint8),
            "candidate_bitsets": np.asarray([0b0011], np.uint16),
        }
        shard = SimpleNamespace(actor=SimpleNamespace(arrays=arrays))
        record = V6MatchRecord(
            split="train",
            player_count=4,
            split_hash="c" * 64,
            shard_manifest_sha256="d" * 64,
            local_match_index=0,
            decision_start=0,
            decision_end=4,
            match_index=0,
            match_seed=1,
            nonforced_decision_count=4,
        )
        result = monte_carlo_targets_by_shard(
            V6MatchView("train", (record,)), {"d" * 64: shard}
        )
        rows, returns = result["d" * 64]
        np.testing.assert_array_equal(rows, [0, 1, 2, 3])
        np.testing.assert_allclose(returns, [1.0, -0.8, 1.0, -1.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
