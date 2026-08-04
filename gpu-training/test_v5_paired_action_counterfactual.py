from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import v5_paired_action_counterfactual as diagnostic
from v4_env import ACTION_COUNT
from v5_collect_mappo import (
    V5MAPPOCollectionConfig,
    collect_v5_mappo,
    publish_v5_mappo_collection,
)
from v5_dataset import load_v5_training_shard, publish_v5_shard
from v5_export import sha256_file
from v5_model import V5PackedActorOutput, pack_legal_actions
from v5_public import V5PublicObservation


class _AlwaysDeviateActor(torch.nn.Module):
    """Tiny test Actor that greedily takes a legal non-Normal action."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(dropout=0.0)

    def forward_packed_batch(  # type: ignore[no-untyped-def]
        self, batch, normal_actions
    ) -> V5PackedActorOutput:
        action_indices, action_mask = pack_legal_actions(batch.legal_mask)
        logits = torch.full(
            action_indices.shape,
            -1.0e9,
            dtype=torch.float32,
            device=action_indices.device,
        )
        logits = logits.masked_fill(action_mask, 0.0)
        for row in range(action_indices.shape[0]):
            legal = action_indices[row][action_mask[row]]
            alternatives = legal[legal != normal_actions[row]]
            if alternatives.numel():
                chosen = alternatives[-1]
                position = torch.nonzero(
                    (action_indices[row] == chosen) & action_mask[row],
                    as_tuple=False,
                )[0, 0]
                logits[row, position] = 2.0
        zeros = torch.zeros_like(logits)
        return V5PackedActorOutput(
            logits,
            zeros,
            zeros,
            action_indices,
            action_mask,
            normal_actions,
        )


def _behaviour_actor(
    observations: tuple[object, ...], normal_actions: tuple[int, ...]
) -> np.ndarray:
    result = np.full((len(observations), ACTION_COUNT), -1.0e9, dtype=np.float64)
    for row, observation in enumerate(observations):
        if type(observation) is not V5PublicObservation:
            raise TypeError("fixture received a non-public Actor observation")
        legal = np.asarray(observation.legal_mask, dtype=np.bool_)
        result[row, legal] = np.linspace(-0.25, 0.25, int(legal.sum()))
        if not legal[normal_actions[row]]:
            raise AssertionError("fixture Normal action is illegal")
    return result


def _critic(states: tuple[torch.Tensor, ...]) -> np.ndarray:
    return np.zeros(len(states), dtype=np.float64)


def _published_fixture(root: Path) -> Path:
    collection = collect_v5_mappo(
        _behaviour_actor,
        _critic,
        V5MAPPOCollectionConfig(
            run_namespace="v5-paired-counterfactual-test",
            seed_base=850_007_001,
            match_counts=((4, 1),),
            lane_count=1,
        ),
    )
    target = root / "source-shard"
    publish_v5_mappo_collection(
        target,
        collection,
        behavior_actor_sha256="1" * 64,
        behavior_actor_manifest_sha256="2" * 64,
        behavior_critic_sha256="3" * 64,
    )
    return target


def _actor_identity() -> dict[str, str]:
    return {
        "actorSha256": "4" * 64,
        "manifestSha256": "5" * 64,
        "tensorStateSha256": "6" * 64,
        "publicContractSha256": "7" * 64,
        "policyNumericsSha256": "8" * 64,
    }


class V5PairedActionCounterfactualTests(unittest.TestCase):
    def test_two_stage_boundary_and_exact_replay(self) -> None:
        caps = diagnostic.CounterfactualCaps(((4, 1),), ((4, 2),))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = _published_fixture(root)
            # Phase one must never cross the Actor-only loader boundary.
            with mock.patch.object(
                diagnostic,
                "load_v5_training_shard",
                side_effect=AssertionError("private loader reached during scan"),
            ):
                records = diagnostic.scan_actor_deviations(
                    _AlwaysDeviateActor(),
                    [shard],
                    caps,
                    device="cpu",
                    batch_size=8,
                )
            self.assertGreaterEqual(len(records), 1)
            self.assertLessEqual(len(records), 2)
            self.assertTrue(
                all(set(record) == diagnostic.SELECTION_KEYS for record in records)
            )
            selection_path = root / "selection.json"
            selection_sha = diagnostic.write_selection_file(selection_path, records)
            loaded, loaded_sha = diagnostic.load_selection_file(selection_path)
            self.assertEqual(records, loaded)
            self.assertEqual(selection_sha, loaded_sha)
            self.assertEqual(
                json.loads(selection_path.read_text(encoding="ascii")), records
            )
            shard_sha = sha256_file(shard / "manifest.json")
            scan_receipt = diagnostic.build_scan_receipt(
                _actor_identity(), loaded_sha, loaded, [shard_sha], caps
            )
            scan_receipt_path = root / "scan-receipt.json"
            scan_receipt_sha = diagnostic.write_scan_receipt(
                scan_receipt_path, scan_receipt
            )
            loaded_receipt, loaded_receipt_sha = diagnostic.load_scan_receipt(
                scan_receipt_path
            )
            self.assertEqual(scan_receipt_sha, loaded_receipt_sha)
            self.assertTrue(loaded_receipt["publicActorPartitionOnly"])
            self.assertFalse(loaded_receipt["privatePartitionOpened"])

            # Phase two has no Actor argument and must not tensorize a model
            # batch; it uses only saved actions after exact row comparison.
            with mock.patch.object(
                diagnostic,
                "actor_batch_from_packed_arrays",
                side_effect=AssertionError("Actor inference reached during replay"),
            ):
                report = diagnostic.replay_paired_counterfactuals(
                    loaded,
                    loaded_sha,
                    loaded_receipt,
                    loaded_receipt_sha,
                    [shard],
                    caps,
                    max_rollout_steps=2048,
                )
            self.assertTrue(report["diagnosticOnly"])
            self.assertFalse(report["promotionEligible"])
            self.assertFalse(report["actorInferenceDuringReplay"])
            self.assertEqual(report["finalActorIdentity"], _actor_identity())
            summary = report["summary"]
            self.assertEqual(summary["failedMatchClusters"], 0)
            self.assertEqual(summary["completedRoots"], len(records))
            clusters = report["matchClusters"]
            self.assertEqual(len(clusters), 1)
            self.assertEqual(clusters[0]["status"], "complete")
            for root_record in clusters[0]["roots"]:
                self.assertTrue(root_record["branchStateVerified"])
                self.assertTrue(root_record["rngStateVerified"])
                self.assertTrue(root_record["currentActBoundVerified"])
                self.assertEqual(len(root_record["stateFingerprintSha256"]), 64)
                self.assertEqual(
                    root_record["finalMinusNormalChip"],
                    root_record["finalActorOutcome"]["chipAward"]
                    - root_record["normalOutcome"]["chipAward"],
                )
            serialized = json.dumps(report, sort_keys=True).lower()
            for forbidden in (
                "privileged_states",
                "rawhand",
                "opponenthand",
                "matchseeds",
                "initialorder",
            ):
                self.assertNotIn(forbidden, serialized)

    def test_private_row_mismatch_discards_the_complete_match(self) -> None:
        caps = diagnostic.CounterfactualCaps(((4, 1),), ((4, 1),))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = _published_fixture(root)
            records = diagnostic.scan_actor_deviations(
                _AlwaysDeviateActor(),
                [original],
                caps,
                device="cpu",
                batch_size=16,
            )
            self.assertEqual(len(records), 1)
            training = load_v5_training_shard(original)
            try:
                actor_arrays = {
                    name: np.asarray(value).copy()
                    for name, value in training.actor.arrays.items()
                }
                privileged = {
                    name: np.asarray(value).copy()
                    for name, value in training.privileged_arrays.items()
                }
            finally:
                training.close()
            original_privileged = {
                name: value.copy() for name, value in privileged.items()
            }
            # Corrupt the final row, after the selected early root.  The root
            # may be computed internally, but publication must discard it when
            # the later exact private comparison fails.
            corrupt_row = len(actor_arrays["actions"]) - 1
            self.assertLessEqual(int(records[0]["localRow"]), corrupt_row)
            privileged["privileged_states"][corrupt_row, 0] += np.float16(1.0)
            corrupted = root / "private-mismatch-shard"
            publish_v5_shard(corrupted, actor_arrays, privileged)
            corrupted_sha = sha256_file(corrupted / "manifest.json")
            corrupted_records = [
                {**records[0], "shardManifestSha256": corrupted_sha}
            ]
            selection_sha = "a" * 64
            scan_receipt = diagnostic.build_scan_receipt(
                _actor_identity(),
                selection_sha,
                corrupted_records,
                [corrupted_sha],
                caps,
            )
            report = diagnostic.replay_paired_counterfactuals(
                corrupted_records,
                selection_sha,
                scan_receipt,
                "b" * 64,
                [corrupted],
                caps,
            )
            self.assertEqual(report["summary"]["completeMatchClusters"], 0)
            self.assertEqual(report["summary"]["failedMatchClusters"], 1)
            self.assertEqual(report["summary"]["completedRoots"], 0)
            self.assertEqual(report["summary"]["discardedRoots"], 1)
            cluster = report["matchClusters"][0]
            self.assertEqual(cluster["status"], "failed")
            self.assertEqual(cluster["failureCode"], "private-row-mismatch")
            self.assertEqual(cluster["roots"], [])

            # A different saved seed leaves the shard internally valid but
            # cannot reproduce the packed public row.  It must likewise
            # discard the whole selected match rather than retain early roots.
            seed_mismatch_privileged = {
                name: value.copy() for name, value in original_privileged.items()
            }
            seed_mismatch_privileged["match_seeds"][0] += np.uint32(1)
            public_mismatch = root / "public-mismatch-shard"
            publish_v5_shard(
                public_mismatch, actor_arrays, seed_mismatch_privileged
            )
            public_mismatch_sha = sha256_file(public_mismatch / "manifest.json")
            public_records = [
                {**records[0], "shardManifestSha256": public_mismatch_sha}
            ]
            public_selection_sha = "c" * 64
            public_receipt = diagnostic.build_scan_receipt(
                _actor_identity(),
                public_selection_sha,
                public_records,
                [public_mismatch_sha],
                caps,
            )
            public_report = diagnostic.replay_paired_counterfactuals(
                public_records,
                public_selection_sha,
                public_receipt,
                "d" * 64,
                [public_mismatch],
                caps,
            )
            public_cluster = public_report["matchClusters"][0]
            self.assertEqual(public_cluster["status"], "failed")
            self.assertEqual(public_cluster["failureCode"], "public-row-mismatch")
            self.assertEqual(public_cluster["roots"], [])

    def test_caps_and_selection_schema_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            diagnostic.CounterfactualCaps(((4, 1),), ((5, 1),))
        malformed = [
            {
                "shardManifestSha256": "0" * 64,
                "localRow": 0,
                "finalAction": 1,
                "normalAction": 0,
                "margin": 1.0,
                "seed": 7,
            }
        ]
        with self.assertRaisesRegex(ValueError, "fields drifted"):
            diagnostic._validate_selection_records(malformed)


if __name__ == "__main__":
    unittest.main()
