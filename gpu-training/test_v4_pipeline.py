import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from v4_dataset import (
    V4_LOSS_MASK_NAMES,
    V4TrajectoryDataset,
    create_v4_smoke_dataset,
    load_v4_dataset_npz,
    save_v4_dataset_npz,
    tensorize_v4_public_observation,
)
from v4_export import (
    export_v4_actor_bundle,
    load_v4_actor_checkpoint,
    make_v4_export_inputs,
    try_export_v4_onnx,
    verify_v4_actor_bundle,
)
from v4_model import V4ActorConfig, V4CriticConfig, V4PublicActor
from v4_train import V4TrainingConfig, _trim_public_padding, train_v4


def tiny_configs() -> tuple[V4ActorConfig, V4CriticConfig]:
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


def public_record() -> dict[str, object]:
    players = []
    for index in range(4):
        players.append({
            "relativeOffset": index,
            "handCount": 8 + index,
            "finished": 0,
            "passed": 1 if index == 2 else 0,
            "self": 1 if index == 0 else 0,
            "tableLeader": 1 if index == 1 else 0,
            "role": min(index, 4),
            "score": index * 2,
        })
    history = [
        {
            "sequence": 8,
            "type": 0,
            "actorOffset": 1,
            "handCountBefore": 10,
            "handCountAfter": 8,
            "rank": 3,
            "naturalCount": 1,
            "jokerCount": 1,
            "totalCount": 2,
            "passReason": 0,
            "clearReason": 0,
            "nextLeaderOffset": -1,
            "finishPlace": 0,
        },
        {
            "sequence": 9,
            "type": 1,
            "actorOffset": 2,
            "handCountBefore": 9,
            "handCountAfter": 9,
            "rank": 0,
            "naturalCount": 0,
            "jokerCount": 0,
            "totalCount": 0,
            "passReason": 4,
            "clearReason": 0,
            "nextLeaderOffset": -1,
            "finishPlace": 0,
        },
    ]
    public_counts = [0] * 13
    public_counts[0] = 1
    public_counts[2] = 1
    own_counts = [0] * 13
    own_counts[1] = 1
    return {
        "schemaVersion": 4,
        "playerCount": 4,
        "act": 3,
        "actorRole": 0,
        "revolution": 1,
        "ownHandCounts": own_counts,
        "publicPlayedCounts": public_counts,
        "table": {
            "actorOffset": 1,
            "rank": 3,
            "naturalCount": 1,
            "jokerCount": 0,
            "totalCount": 1,
        },
        "playerTokens": players,
        "historyTokens": history,
        "memoryTraceVectors": [
            [0.01 * (row + column) for column in range(20)]
            for row in range(4)
        ],
        "truncatedHistoryCount": 12,
    }


class V4PipelineTests(unittest.TestCase):
    def test_training_trims_only_public_padding_columns(self) -> None:
        batch = {
            "player_features": torch.randn(2, 3, 10, 12),
            "player_mask": torch.zeros(2, 3, 10, dtype=torch.bool),
            "history_features": torch.randn(2, 3, 192, 20),
            "history_mask": torch.zeros(2, 3, 192, dtype=torch.bool),
            "actions": torch.zeros(2, 3, dtype=torch.long),
        }
        batch["player_mask"][..., :6] = True
        batch["history_mask"][0, 0, :17] = True
        batch["history_mask"][1, 2, :41] = True

        trimmed = _trim_public_padding(batch)

        self.assertEqual(tuple(trimmed["player_features"].shape), (2, 3, 6, 12))
        self.assertEqual(tuple(trimmed["history_features"].shape), (2, 3, 41, 20))
        self.assertEqual(tuple(trimmed["actions"].shape), (2, 3))
        self.assertIs(trimmed["actions"], batch["actions"])

    def test_canonical_public_record_tensorizes_without_private_fields(self) -> None:
        config, _ = tiny_configs()
        tensors = tensorize_v4_public_observation(public_record(), config)
        self.assertEqual(tuple(tensors.global_features.shape), (12,))
        self.assertEqual(tuple(tensors.rank_features.shape), (13, 6))
        self.assertEqual(tuple(tensors.player_features.shape), (4, 12))
        self.assertEqual(tuple(tensors.memory_trace_features.shape), (4, 20))
        self.assertEqual(tuple(tensors.history_features.shape), (2, 20))
        self.assertTrue(tensors.player_mask.all())
        self.assertTrue(tensors.history_mask.all())
        # passReason=4 maps to the fourth pass-reason feature.
        self.assertEqual(float(tensors.history_features[1, 14]), 1.0)
        leaked = public_record()
        leaked["opponentHands"] = [[1, 2, 3]]
        with self.assertRaisesRegex(ValueError, "unknown"):
            tensorize_v4_public_observation(leaked, config)

    def test_npz_dataset_round_trip_is_fingerprint_bound(self) -> None:
        actor_config, critic_config = tiny_configs()
        dataset = create_v4_smoke_dataset(
            actor_config, critic_config, trajectories=2, time_steps=2, seed=31
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.npz"
            save_v4_dataset_npz(dataset, path)
            loaded = load_v4_dataset_npz(path)
            self.assertEqual(loaded.fingerprint, dataset.fingerprint)
            self.assertTrue(
                torch.equal(loaded.tensors.legal_masks, dataset.tensors.legal_masks)
            )
            self.assertEqual(
                loaded.loss_contract_fingerprint,
                dataset.loss_contract_fingerprint,
            )
            self.assertTrue(
                torch.equal(
                    loaded.loss_eligibility.behavior_cloning,
                    dataset.tensors.valid_masks,
                )
            )
            self.assertFalse(loaded.loss_eligibility.ppo.any())
            self.assertIn(V4_LOSS_MASK_NAMES["behaviorCloning"], loaded[0])

    def test_training_loss_configuration_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Q boost requires"):
            V4TrainingConfig(ppo_weight=1.0, critic_weight=0.0, q_boost_coefficient=0.1)
        with self.assertRaisesRegex(ValueError, "entropy regularization"):
            V4TrainingConfig(entropy_coefficient=0.01)
        with self.assertRaisesRegex(ValueError, "BC or PPO"):
            V4TrainingConfig(bc_weight=0.0, ppo_weight=0.0, critic_weight=1.0)

    def test_smoke_and_unbound_data_reject_ppo_before_creating_output(self) -> None:
        actor_config, critic_config = tiny_configs()
        smoke = create_v4_smoke_dataset(
            actor_config, critic_config, trajectories=2, time_steps=2, seed=35
        )
        unbound = V4TrajectoryDataset(
            smoke.tensors, actor_config, critic_config
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ppo_output = root / "ppo-rejected"
            with self.assertRaisesRegex(ValueError, "no eligible samples"):
                train_v4(
                    smoke,
                    ppo_output,
                    V4TrainingConfig(
                        bc_weight=0.0,
                        ppo_weight=1.0,
                        critic_weight=0.0,
                        amp=False,
                    ),
                )
            self.assertFalse(ppo_output.exists())
            unbound_output = root / "unbound-rejected"
            with self.assertRaisesRegex(ValueError, "no bound loss"):
                train_v4(
                    unbound,
                    unbound_output,
                    V4TrainingConfig(amp=False),
                )
            self.assertFalse(unbound_output.exists())

    def test_actor_bundle_round_trip_excludes_critic_and_verifies_hashes(self) -> None:
        actor_config, _ = tiny_configs()
        torch.manual_seed(37)
        actor = V4PublicActor(actor_config).eval()
        inputs = make_v4_export_inputs(actor_config, batch_size=2)
        before = actor(*inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            manifest = export_v4_actor_bundle(
                actor, output, metadata={"seed": 37}, include_onnx=False
            )
            verified = verify_v4_actor_bundle(output)
            loaded, payload = load_v4_actor_checkpoint(output / "actor.pt")
            after = loaded(*inputs)

            self.assertTrue(torch.allclose(before, after, atol=1e-6))
            self.assertTrue(manifest["model"]["criticExcluded"])
            self.assertEqual(verified["files"], manifest["files"])
            self.assertFalse(
                any("critic" in key.lower() for key in payload["stateDict"])
            )

    def test_missing_onnx_dependency_is_a_nonfatal_result(self) -> None:
        actor_config, _ = tiny_configs()
        actor = V4PublicActor(actor_config).eval()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "actor.onnx"
            with mock.patch("v4_export._onnx_is_available", return_value=False):
                result = try_export_v4_onnx(actor, output)
            self.assertFalse(result.exported)
            self.assertIsNone(result.path)
            self.assertIn("optional dependency", result.reason)
            self.assertFalse(output.exists())

    def test_fresh_training_records_verified_actor_initialization(self) -> None:
        actor_config, critic_config = tiny_configs()
        dataset = create_v4_smoke_dataset(
            actor_config, critic_config, trajectories=2, time_steps=2, seed=39
        )
        actor = V4PublicActor(actor_config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "initial-actor"
            bundle_manifest = export_v4_actor_bundle(actor, bundle)
            output = root / "continued"
            train_v4(
                dataset,
                output,
                V4TrainingConfig(
                    epochs=1,
                    batch_size=1,
                    seed=40,
                    amp=False,
                ),
                initialize_actor_bundle=bundle,
            )
            run_manifest = json.loads(
                (output / "run-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                run_manifest["initialActor"]["actorSha256"],
                bundle_manifest["files"]["actor.pt"]["sha256"],
            )

    def test_smoke_training_checkpoints_and_resumes_without_exporting_critic(self) -> None:
        actor_config, critic_config = tiny_configs()
        dataset = create_v4_smoke_dataset(
            actor_config, critic_config, trajectories=2, time_steps=2, seed=41
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = train_v4(
                dataset,
                output,
                V4TrainingConfig(
                    epochs=1,
                    batch_size=1,
                    gradient_accumulation=2,
                    seed=43,
                    amp=False,
                ),
            )
            resumed = train_v4(
                dataset,
                output,
                V4TrainingConfig(
                    epochs=2,
                    batch_size=1,
                    gradient_accumulation=2,
                    seed=43,
                    amp=False,
                ),
                resume="latest",
            )
            latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))

            self.assertEqual(first["completedEpochs"], 1)
            self.assertEqual(resumed["completedEpochs"], 2)
            self.assertEqual(latest["completedEpoch"], 2)
            self.assertEqual(len(result["metrics"]), 2)
            self.assertFalse(result["privilegedCriticExported"])
            self.assertTrue((output / "candidate" / "actor.pt").is_file())


if __name__ == "__main__":
    unittest.main()
