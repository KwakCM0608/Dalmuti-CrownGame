from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from v4_accuracy import (
    evaluate_v4_accuracy,
    load_public_accuracy_dataset,
    write_accuracy_report_exclusive,
)
from v4_dataset import (
    V4_DATASET_FORMAT,
    V4_DATASET_VERSION,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
)
from v4_export import export_v4_actor_bundle
from v4_model import V4ActorConfig, V4CriticConfig, V4PublicActor


def _configs() -> tuple[V4ActorConfig, V4CriticConfig]:
    return (
        V4ActorConfig(
            max_players=4,
            max_history=2,
            d_model=16,
            layers=1,
            heads=4,
            feedforward=32,
            action_hidden=12,
        ),
        V4CriticConfig(
            privileged_features=12,
            d_model=16,
            hidden_layers=1,
            action_hidden=12,
        ),
    )


def _dataset() -> V4TrajectoryDataset:
    actor, critic = _configs()
    prefix = (2, 2)
    global_features = torch.zeros(*prefix, actor.global_features)
    # p4, act 1. Trajectory zero is Dalmuti and trajectory one Lesser Dalmuti.
    global_features[0, :, 2] = 1.0
    global_features[1, :, 3] = 1.0
    rank_features = torch.zeros(*prefix, actor.rank_tokens, actor.rank_features)
    player_features = torch.zeros(
        *prefix, actor.max_players, actor.player_features
    )
    player_mask = torch.ones(*prefix, actor.max_players, dtype=torch.bool)
    memory = torch.zeros(*prefix, actor.memory_tokens, actor.memory_features)
    history = torch.zeros(*prefix, actor.max_history, actor.history_features)
    history_mask = torch.zeros(*prefix, actor.max_history, dtype=torch.bool)
    history_mask[0, 1, 0] = True
    history_mask[1, 0, :] = True
    legal = torch.zeros(*prefix, 236, dtype=torch.bool)
    legal[0, 0, [0, 1]] = True
    legal[0, 1, [0, 1]] = True
    legal[1, 0, [5]] = True
    legal[1, 1, [2, 3, 4]] = True
    actions = torch.tensor([[0, 1], [5, 2]], dtype=torch.long)
    dones = torch.tensor([[False, True], [False, True]], dtype=torch.bool)
    valid = torch.ones(*prefix, dtype=torch.bool)
    return V4TrajectoryDataset(
        V4TrajectoryTensors(
            global_features=global_features,
            rank_features=rank_features,
            player_features=player_features,
            player_mask=player_mask,
            memory_trace_features=memory,
            history_features=history,
            history_mask=history_mask,
            legal_masks=legal,
            actions=actions,
            expert_actions=actions.clone(),
            old_action_log_probs=torch.zeros(*prefix),
            advantages=torch.zeros(*prefix),
            rewards=torch.zeros(*prefix),
            dones=dones,
            valid_masks=valid,
            privileged_states=torch.full(
                (*prefix, critic.privileged_features), 987654.0
            ),
        ),
        actor,
        critic,
    )


def _write_prepared_dataset(directory: Path) -> Path:
    dataset = _dataset()
    output = directory / "prepared.npz"
    arrays = {
        field.name: getattr(dataset.tensors, field.name).numpy()
        for field in fields(dataset.tensors)
    }
    arrays["trajectory_ids"] = np.asarray(
        [
            "v4-normal-p4-episode-1:act-1:player-1",
            "v4-normal-p4-episode-1:act-1:player-2",
        ],
        dtype=np.str_,
    )
    metadata = {
        "format": V4_DATASET_FORMAT,
        "version": V4_DATASET_VERSION,
        "preparationFormat": "dalmuti-v4-prepared-dataset-metadata",
        "preparationVersion": 1,
        "actorConfig": dataset.actor_config.to_dict(),
        "criticConfig": dataset.critic_config.to_dict(),
        "fingerprint": dataset.fingerprint,
        "inputs": [
            {
                "format": "dalmuti-v4-normal-warmstart-ndjson",
                "sha256": "1" * 64,
            }
        ],
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(output, **arrays)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    Path(f"{output}.sha256").write_text(digest + "\n", encoding="ascii")
    return output


def _zero_actor(config: V4ActorConfig, marker: float = 0.0) -> V4PublicActor:
    actor = V4PublicActor(config).eval()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.cls_token.fill_(marker)
    return actor


class V4AccuracyTests(unittest.TestCase):
    def test_public_only_accuracy_and_immutable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            dataset_path = _write_prepared_dataset(directory)
            actor_config, _ = _configs()
            bundle = directory / "actor"
            export_v4_actor_bundle(
                _zero_actor(actor_config), bundle, metadata={"seed": 41}
            )

            public = load_public_accuracy_dataset(dataset_path)
            self.assertFalse(hasattr(public, "privileged_states"))
            report = evaluate_v4_accuracy(
                dataset_path,
                [bundle],
                device="cpu",
                batch_size=2,
                bootstrap_seed=101,
                bootstrap_resamples=50,
            )
            all_valid = report["metrics"]["allValid"]
            non_forced = report["metrics"]["nonForced"]
            self.assertEqual(all_valid["greedyAgreement"]["correct"], 3)
            self.assertEqual(all_valid["greedyAgreement"]["rate"], 0.75)
            self.assertEqual(non_forced["greedyAgreement"]["correct"], 2)
            self.assertAlmostEqual(
                non_forced["greedyAgreement"]["rate"], 2.0 / 3.0
            )
            self.assertEqual(all_valid["topKAccuracy"]["3"]["rate"], 1.0)
            expected_nll = (2.0 * np.log(2.0) + np.log(3.0)) / 4.0
            self.assertAlmostEqual(
                all_valid["crossEntropyNll"], expected_nll, places=6
            )
            self.assertEqual(
                all_valid["illegalProbabilityMass"]["maximum"], 0.0
            )
            self.assertTrue(report["privacyAudit"]["passed"])
            self.assertFalse(report["privacyAudit"]["privilegedStateLoaded"])
            self.assertIn("p4", report["metrics"]["perPlayerCount"])
            self.assertIn("great-dalmuti", report["metrics"]["perRole"])
            self.assertIn("1", report["metrics"]["perAct"])
            self.assertIn(
                "3-4", report["metrics"]["legalActionCountBuckets"]["metrics"]
            )
            bootstrap = all_valid["trajectoryClusterBootstrap95"]
            self.assertEqual(bootstrap["unit"], "actor-trajectory")
            self.assertEqual(bootstrap["clusters"], 2)

            output = directory / "accuracy.json"
            artifact = write_accuracy_report_exclusive(output, report)
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(), artifact["sha256"]
            )
            self.assertTrue(Path(f"{output}.sha256").is_file())
            with self.assertRaises(FileExistsError):
                write_accuracy_report_exclusive(output, report)

    def test_three_distinct_bundles_use_centered_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            dataset_path = _write_prepared_dataset(directory)
            actor_config, _ = _configs()
            bundles = []
            for index, seed in enumerate((41, 43, 47), start=1):
                bundle = directory / f"actor-{index}"
                export_v4_actor_bundle(
                    _zero_actor(actor_config, marker=index / 100.0),
                    bundle,
                    metadata={"seed": seed},
                )
                bundles.append(bundle)
            report = evaluate_v4_accuracy(
                dataset_path,
                bundles,
                device="cpu",
                batch_size=4,
                bootstrap_seed=103,
                bootstrap_resamples=20,
            )
            model = report["bindings"]["model"]
            self.assertEqual(model["bundleCount"], 3)
            self.assertEqual(
                model["ensembleRule"],
                "mean-of-per-actor-logits-centered-over-legal-actions",
            )
            self.assertEqual(
                report["metrics"]["allValid"]["greedyAgreement"]["rate"],
                0.75,
            )

    def test_rejects_dataset_checksum_drift_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            dataset_path = _write_prepared_dataset(directory)
            Path(f"{dataset_path}.sha256").write_text(
                "0" * 64 + "\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "checksum does not match"):
                load_public_accuracy_dataset(dataset_path)


if __name__ == "__main__":
    unittest.main()
