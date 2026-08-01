from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:  # Local syntax-only workstations may omit torch.
    raise unittest.SkipTest("torch and numpy are required for V4 DAgger tests") from error

from v4_collect_dagger import (
    DAGGER_PREPARATION_FORMAT,
    DaggerCollectionConfig,
    _batch_candidate_actions,
    audit_hidden_state_privacy,
    collect_v4_dagger,
)
from v4_dataset import load_v4_dataset_npz
from v4_env import (
    ACTION_COUNT,
    PRIVILEGED_STATE_LAYOUT,
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
    DalmutiScalarEnv,
    V4ActorObservation,
)
from v4_export import (
    export_v4_actor_bundle,
    load_v4_actor_checkpoint,
    sha256_file,
)
from v4_model import V4ActorConfig, V4PublicActor


class V4DaggerCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        torch.manual_seed(20260801)
        actor = V4PublicActor(
            V4ActorConfig(
                max_players=10,
                max_history=8,
                d_model=16,
                layers=1,
                heads=4,
                feedforward=32,
                action_hidden=16,
            )
        ).eval()
        cls.bundle = cls.root / "candidate"
        export_v4_actor_bundle(
            actor,
            cls.bundle,
            metadata={"purpose": "v4-dagger-test"},
            include_onnx=False,
        )

        def alternate_candidate(
            model: object,
            observations: list[V4ActorObservation] | tuple[V4ActorObservation, ...],
            device: torch.device,
        ) -> list[int]:
            del model, device
            actions = []
            for observation in observations:
                # This is the privacy boundary under test: model inference gets
                # the actor dataclass, never V4EnvironmentObservation.
                if not isinstance(observation, V4ActorObservation):
                    raise AssertionError("candidate received a non-public observation")
                if hasattr(observation, "privileged_state"):
                    raise AssertionError("candidate received privileged critic state")
                legal = torch.nonzero(
                    observation.legal_mask, as_tuple=False
                ).flatten()
                actions.append(int(legal[-1].item()))
            return actions

        cls.alternate_candidate = staticmethod(alternate_candidate)
        base = dict(
            seed_base=771_900_001,
            player_counts=(4,),
            matches_per_player_count=1,
            acts=1,
            lane_count=2,
            device="cpu",
        )
        requests = (
            ("repeat-a", "repeat", 0.5),
            ("repeat-b", "repeat", 0.5),
            ("other-shard", "remote-shard", 0.5),
            ("normal-state", "state-dist", 0.0),
            ("candidate-state", "state-dist", 1.0),
        )
        cls.outputs: dict[str, Path] = {}
        with mock.patch(
            "v4_collect_dagger._batch_candidate_actions",
            side_effect=alternate_candidate,
        ):
            for directory_name, namespace, beta in requests:
                directory = cls.root / directory_name
                output = directory / "dagger.npz"
                collect_v4_dagger(
                    cls.bundle,
                    output,
                    DaggerCollectionConfig(
                        run_namespace=namespace,
                        candidate_beta=beta,
                        **base,
                    ),
                )
                cls.outputs[directory_name] = output

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _metadata(self, label: str) -> dict[str, object]:
        return json.loads(
            Path(f"{self.outputs[label]}.metadata.json").read_text(encoding="utf-8")
        )

    def test_collection_is_byte_deterministic_and_namespaces_are_disjoint(self) -> None:
        first = self.outputs["repeat-a"]
        second = self.outputs["repeat-b"]
        other = self.outputs["other-shard"]
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(sha256_file(first), sha256_file(second))
        self.assertEqual(
            Path(f"{first}.metadata.json").read_bytes(),
            Path(f"{second}.metadata.json").read_bytes(),
        )
        sidecar = Path(f"{first}.sha256").read_text(encoding="ascii").split()
        self.assertEqual(sidecar, [sha256_file(first), first.name])

        with np.load(first, allow_pickle=False) as left, np.load(
            other, allow_pickle=False
        ) as right:
            left_ids = set(left["trajectory_ids"].tolist())
            right_ids = set(right["trajectory_ids"].tolist())
            self.assertTrue(left_ids)
            self.assertTrue(left_ids.isdisjoint(right_ids))
            self.assertFalse(
                np.array_equal(
                    left["trajectory_match_seeds"], right["trajectory_match_seeds"]
                )
            )

    def test_real_actor_forward_batches_public_observations_and_stays_legal(self) -> None:
        actor, _ = load_v4_actor_checkpoint(self.bundle / "actor.pt")
        environments = (
            DalmutiScalarEnv(4, acts=1, seed=400_101),
            DalmutiScalarEnv(4, acts=1, seed=400_102),
        )
        observations = [environment.public_observation() for environment in environments]
        actions = _batch_candidate_actions(actor, observations, torch.device("cpu"))
        self.assertEqual(len(actions), len(observations))
        for observation, action in zip(observations, actions, strict=True):
            self.assertTrue(bool(observation.legal_mask[action].item()))

    def test_actor_trajectories_have_one_legal_terminal_and_terminal_reward(self) -> None:
        source = self.outputs["repeat-a"]
        dataset = load_v4_dataset_npz(source)
        tensors = dataset.tensors
        valid = tensors.valid_masks
        lengths = valid.sum(dim=-1)
        terminals = tensors.dones & valid
        self.assertTrue(torch.equal(terminals.sum(dim=-1), torch.ones(len(dataset), dtype=torch.long)))
        last = lengths - 1
        self.assertTrue(tensors.dones.gather(1, last[:, None]).all())
        nonterminal = valid & ~tensors.dones
        self.assertTrue(torch.equal(tensors.rewards[nonterminal], torch.zeros_like(tensors.rewards[nonterminal])))
        for name in ("actions", "expert_actions"):
            actions = getattr(tensors, name)
            selected = tensors.legal_masks.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            self.assertTrue((selected | ~valid).all())

        terminal_rewards = tensors.rewards.gather(1, last[:, None]).squeeze(1)
        # One p4 act has four actor trajectories and the exact 4/3/1/0 chip curve.
        self.assertEqual(
            sorted(round(float(value), 5) for value in terminal_rewards.tolist()),
            [-1.0, -0.5, 0.5, 1.0],
        )
        with np.load(source, allow_pickle=False) as archive:
            candidates = archive["candidate_actions"]
            behavior = archive["actions"]
            experts = archive["expert_actions"]
            behavior_sources = archive["behavior_sources"]
            valid_np = archive["valid_masks"]
            legal = archive["legal_masks"]
            rows, columns = np.nonzero(valid_np)
            self.assertTrue(np.all(legal[rows, columns, candidates[rows, columns]]))
            selected_candidate = valid_np & (behavior_sources == 1)
            selected_normal = valid_np & (behavior_sources == 0)
            self.assertTrue(np.array_equal(behavior[selected_candidate], candidates[selected_candidate]))
            self.assertTrue(np.array_equal(behavior[selected_normal], experts[selected_normal]))
            disagreement = valid_np & (candidates != experts)
            expected = np.zeros_like(archive["old_action_log_probs"])
            expected[disagreement] = math.log(0.5)
            self.assertTrue(
                np.allclose(archive["old_action_log_probs"][valid_np], expected[valid_np])
            )

    def test_privacy_audit_and_serialized_actor_boundary_exclude_private_data(self) -> None:
        audit = audit_hidden_state_privacy(
            DalmutiScalarEnv(4, acts=1, seed=919_331), 551_002
        )
        self.assertTrue(audit["publicInvariantAcrossEightOpponentHandResamples"])
        self.assertTrue(audit["privilegedStateChanged"])
        self.assertTrue(audit["opponentPhysicalHandsExcluded"])
        self.assertTrue(audit["taxCardIdentitiesExcluded"])

        metadata = self._metadata("repeat-a")
        privacy = metadata["privacy"]
        self.assertTrue(privacy["actorPublicOnly"])
        self.assertTrue(privacy["privilegedCriticStateSeparate"])
        critic_layout = metadata["privilegedCriticLayout"]
        self.assertEqual(critic_layout["id"], PRIVILEGED_STATE_LAYOUT_ID)
        self.assertEqual(critic_layout["sha256"], PRIVILEGED_STATE_LAYOUT_SHA256)
        self.assertEqual(critic_layout["layout"], PRIVILEGED_STATE_LAYOUT)
        self.assertTrue(critic_layout["matchesTypescriptNormalContract"])
        self.assertNotIn("privileged_states", privacy["actorInputFields"])
        self.assertNotIn("opponentHands", privacy["actorInputFields"])
        self.assertNotIn("taxation", privacy["actorInputFields"])
        with np.load(self.outputs["repeat-a"], allow_pickle=False) as archive:
            self.assertEqual(archive["privileged_states"].shape[-1], 512)
            self.assertEqual(archive["global_features"].shape[-1], 12)
            self.assertEqual(archive["legal_masks"].shape[-1], ACTION_COUNT)
            self.assertFalse(
                any("card_id" in name.lower() or "opponent_hand" in name.lower() for name in archive.files)
            )

    def test_candidate_beta_reaches_candidate_induced_states_with_normal_labels(self) -> None:
        normal_path = self.outputs["normal-state"]
        candidate_path = self.outputs["candidate-state"]
        normal_metadata = self._metadata("normal-state")
        candidate_metadata = self._metadata("candidate-state")
        self.assertEqual(normal_metadata["collection"]["candidateBeta"], 0.0)
        self.assertEqual(candidate_metadata["collection"]["candidateBeta"], 1.0)
        self.assertEqual(
            normal_metadata["shard"]["environmentSeeds"],
            candidate_metadata["shard"]["environmentSeeds"],
        )
        self.assertEqual(
            normal_metadata["shard"]["runNamespace"],
            candidate_metadata["shard"]["runNamespace"],
        )
        normal_rates = normal_metadata["changedActionRates"]["overall"]
        candidate_rates = candidate_metadata["changedActionRates"]["overall"]
        self.assertEqual(normal_rates["behaviorExpertChanges"], 0)
        self.assertGreater(candidate_rates["behaviorExpertChanges"], 0)
        self.assertGreater(candidate_rates["candidateExpertDisagreements"], 0)
        self.assertEqual(candidate_rates["candidateSelectionRate"], 1.0)

        normal_dataset = load_v4_dataset_npz(normal_path)
        candidate_dataset = load_v4_dataset_npz(candidate_path)
        # Same namespace/seed gives the same initial deal and first public state.
        self.assertTrue(
            torch.equal(
                normal_dataset.tensors.global_features[0, 0],
                candidate_dataset.tensors.global_features[0, 0],
            )
        )
        # Candidate behavior then drives a different legal state distribution.
        self.assertNotEqual(normal_dataset.fingerprint, candidate_dataset.fingerprint)
        self.assertEqual(
            candidate_metadata["preparationFormat"], DAGGER_PREPARATION_FORMAT
        )
        self.assertTrue(candidate_metadata["collection"]["expertLabelForEveryDecision"])

    def test_immutable_outputs_reject_overwrite(self) -> None:
        source = self.outputs["repeat-a"]
        config = DaggerCollectionConfig(
            run_namespace="repeat",
            seed_base=771_900_001,
            player_counts=(4,),
            matches_per_player_count=1,
            acts=1,
            candidate_beta=0.5,
            lane_count=2,
            device="cpu",
        )
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            collect_v4_dagger(self.bundle, source, config)


if __name__ == "__main__":
    unittest.main()
