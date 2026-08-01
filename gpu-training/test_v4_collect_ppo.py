from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - syntax-only workstation
    raise unittest.SkipTest("torch and numpy are required for V4 PPO tests") from error

from v4_collect_ppo import (
    BASELINE_FALLBACK_HIERARCHY,
    CANONICAL_PRIVILEGED_LAYOUT_SHA256,
    PPO_PREPARATION_FORMAT,
    BaselineRecord,
    PPOCollectionConfig,
    _keyed_uniform,
    assert_canonical_privileged_layout,
    candidate_opponent_ids,
    collect_v4_ppo,
    learner_actor_ids,
    leave_one_match_out_baselines,
    masked_categorical_probabilities,
    sample_masked_categorical,
)
from v4_dataset import load_v4_dataset_npz
from v4_env import ACTION_COUNT, DalmutiScalarEnv, V4ActorObservation
from v4_export import export_v4_actor_bundle
from v4_model import V4ActorConfig, V4PublicActor
from v4_train import V4TrainingConfig, train_v4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V4PPOCollectorTests(unittest.TestCase):
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
            metadata={"purpose": "v4-ppo-collector-test"},
            include_onnx=False,
        )

        def deterministic_logits(
            model: object,
            observations: list[V4ActorObservation] | tuple[V4ActorObservation, ...],
            device: torch.device,
        ) -> list[torch.Tensor]:
            del model, device
            output: list[torch.Tensor] = []
            for observation in observations:
                if not isinstance(observation, V4ActorObservation):
                    raise AssertionError("candidate inference crossed the public boundary")
                if hasattr(observation, "privileged_state"):
                    raise AssertionError("candidate received privileged critic state")
                logits = torch.full((ACTION_COUNT,), float("-inf"), dtype=torch.float64)
                legal = torch.nonzero(observation.legal_mask, as_tuple=False).flatten()
                logits[legal] = torch.linspace(-0.5, 0.5, len(legal), dtype=torch.float64)
                output.append(logits)
            return output

        cls.deterministic_logits = staticmethod(deterministic_logits)
        base = dict(
            seed_base=830_400_001,
            player_counts=(4,),
            matches_per_player_count=2,
            acts=1,
            candidate_seats_per_act=2,
            opponent_candidate_fraction=0.5,
            temperature=0.9,
            epsilon_floor=1.0e-5,
            gamma=1.0,
            standardize_advantages=False,
            lane_count=2,
            device="cpu",
        )
        requests = (
            ("repeat-a", "repeat"),
            ("repeat-b", "repeat"),
            ("other", "other-namespace"),
        )
        cls.outputs: dict[str, Path] = {}
        with mock.patch(
            "v4_collect_ppo._batch_candidate_logits",
            side_effect=deterministic_logits,
        ):
            for directory, namespace in requests:
                output = cls.root / directory / "ppo.npz"
                collect_v4_ppo(
                    cls.bundle,
                    output,
                    PPOCollectionConfig(run_namespace=namespace, **base),
                )
                cls.outputs[directory] = output

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _metadata(self, label: str) -> dict[str, object]:
        return json.loads(
            Path(f"{self.outputs[label]}.metadata.json").read_text(encoding="utf-8")
        )

    def test_masked_categorical_sampling_is_reproducible_legal_and_exact(self) -> None:
        logits = torch.linspace(-3.0, 3.0, ACTION_COUNT)
        legal = torch.zeros(ACTION_COUNT, dtype=torch.bool)
        legal[[0, 7, 21, 235]] = True
        probabilities = masked_categorical_probabilities(
            logits, legal, temperature=0.7, epsilon_floor=1.0e-4
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        self.assertTrue(np.all(probabilities[legal.numpy()] >= 1.0e-4))
        self.assertTrue(np.all(probabilities[~legal.numpy()] == 0.0))
        uniform = _keyed_uniform("sampling", 44, 4, 2, 9)
        first = sample_masked_categorical(probabilities, uniform)
        second = sample_masked_categorical(probabilities, uniform)
        self.assertEqual(first, second)
        action, old_log_probability, entropy = first
        self.assertTrue(bool(legal[action].item()))
        self.assertEqual(old_log_probability, math.log(float(probabilities[action])))
        self.assertTrue(math.isfinite(entropy))

    def test_leave_one_match_baseline_never_uses_own_cluster(self) -> None:
        records = (
            BaselineRecord(4, "great-dalmuti", 1, "match-a", 10.0),
            BaselineRecord(4, "great-dalmuti", 1, "match-a", 20.0),
            BaselineRecord(4, "great-dalmuti", 1, "match-b", 1.0),
            BaselineRecord(4, "great-dalmuti", 1, "match-c", 3.0),
        )
        baseline = leave_one_match_out_baselines(records)
        self.assertEqual(baseline[0].baseline, 2.0)
        self.assertEqual(baseline[1].baseline, 2.0)
        self.assertEqual(baseline[0].reference_count, 2)
        self.assertEqual(baseline[0].tier, 0)
        perturbed = (
            BaselineRecord(4, "great-dalmuti", 1, "match-a", -999.0),
            BaselineRecord(4, "great-dalmuti", 1, "match-a", 999.0),
            *records[2:],
        )
        changed = leave_one_match_out_baselines(perturbed)
        self.assertEqual(changed[0], baseline[0])
        self.assertEqual(changed[1], baseline[1])
        symmetric_records = (records[2], records[3])
        symmetric_baselines = leave_one_match_out_baselines(symmetric_records)
        centered = [
            record.value - result.baseline
            for record, result in zip(symmetric_records, symmetric_baselines, strict=True)
        ]
        self.assertAlmostEqual(sum(centered), 0.0, places=12)
        singleton = leave_one_match_out_baselines((records[0],))[0]
        self.assertEqual(singleton.tier, len(BASELINE_FALLBACK_HIERARCHY) - 1)
        self.assertEqual(singleton.baseline, 0.0)

    def test_rotating_learners_and_opponent_mix_are_deterministic_and_balanced(self) -> None:
        assignments = [
            learner_actor_ids(4, match_index, 1, 1, 2)
            for match_index in range(2)
        ]
        self.assertEqual(assignments, [(0, 1), (2, 3)])
        counts = {actor: 0 for actor in range(4)}
        for values in assignments:
            for actor in values:
                counts[actor] += 1
        self.assertEqual(set(counts.values()), {1})
        first = candidate_opponent_ids(6, 3, 2, 5, (0, 1), 0.5)
        second = candidate_opponent_ids(6, 3, 2, 5, (0, 1), 0.5)
        self.assertEqual(first, second)
        self.assertTrue(set(first).isdisjoint({0, 1}))
        self.assertEqual(candidate_opponent_ids(6, 3, 2, 5, (0, 1), 0.0), ())
        self.assertEqual(set(candidate_opponent_ids(6, 3, 2, 5, (0, 1), 1.0)), {2, 3, 4, 5})

    def test_collection_is_byte_reproducible_and_namespaces_are_disjoint(self) -> None:
        first = self.outputs["repeat-a"]
        second = self.outputs["repeat-b"]
        other = self.outputs["other"]
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(Path(f"{first}.metadata.json").read_bytes(), Path(f"{second}.metadata.json").read_bytes())
        self.assertEqual(Path(f"{first}.sha256").read_text(encoding="ascii").split(), [_sha256(first), first.name])
        metadata_path = Path(f"{first}.metadata.json")
        self.assertEqual(
            Path(f"{metadata_path}.sha256").read_text(encoding="ascii").split(),
            [_sha256(metadata_path), metadata_path.name],
        )
        with np.load(first, allow_pickle=False) as left, np.load(other, allow_pickle=False) as right:
            self.assertTrue(set(left["trajectory_ids"].tolist()).isdisjoint(set(right["trajectory_ids"].tolist())))
            self.assertFalse(np.array_equal(left["trajectory_match_seeds"], right["trajectory_match_seeds"]))

    def test_standard_contract_logprob_trajectory_reward_and_advantage_arrays(self) -> None:
        source = self.outputs["repeat-a"]
        dataset = load_v4_dataset_npz(source)
        tensors = dataset.tensors
        valid = tensors.valid_masks
        self.assertTrue(torch.equal(dataset.loss_eligibility.behavior_cloning, valid))
        self.assertTrue(torch.equal(dataset.loss_eligibility.ppo, valid))
        self.assertTrue(torch.equal(dataset.loss_eligibility.critic, valid))
        self.assertEqual(
            dataset.loss_eligibility.behavior_actor_sha256s,
            (_sha256(self.bundle / "actor.pt"),),
        )
        lengths = valid.sum(dim=1)
        terminals = tensors.dones & valid
        self.assertTrue(torch.equal(terminals.sum(dim=1), torch.ones(len(dataset), dtype=torch.long)))
        self.assertTrue(tensors.dones.gather(1, (lengths - 1)[:, None]).all())
        for name in ("actions", "expert_actions"):
            selected = tensors.legal_masks.gather(-1, getattr(tensors, name).unsqueeze(-1)).squeeze(-1)
            self.assertTrue((selected | ~valid).all())
        self.assertTrue(torch.isfinite(tensors.old_action_log_probs[valid]).all())
        self.assertTrue(torch.isfinite(tensors.advantages[valid]).all())

        with np.load(source, allow_pickle=False) as archive:
            valid_np = archive["valid_masks"]
            selected_probability = archive["selected_action_probabilities"][valid_np]
            self.assertTrue(np.all(selected_probability > 0.0))
            self.assertTrue(
                np.allclose(
                    archive["old_action_log_probs"][valid_np],
                    np.log(selected_probability),
                    atol=2.0e-7,
                )
            )
            self.assertTrue(np.allclose(
                archive["advantages"][valid_np], archive["raw_advantages"][valid_np], atol=1.0e-7
            ))
            self.assertTrue(np.allclose(
                archive["raw_advantages"][valid_np],
                archive["raw_returns"][valid_np] - archive["baseline_values"][valid_np],
                atol=1.0e-7,
            ))
            rows, columns = np.nonzero(archive["dones"] & valid_np)
            rewards = archive["rewards"][rows, columns]
            chips = archive["terminal_chip_awards"][rows, columns]
            self.assertTrue(np.allclose(rewards, (chips.astype(np.float32) - 2.0) / 2.0))
            self.assertTrue(np.all(archive["baseline_reference_counts"][valid_np] >= 0))

    def test_metadata_audits_privacy_entropy_sources_and_seat_balance(self) -> None:
        metadata = self._metadata("repeat-a")
        self.assertEqual(metadata["preparationFormat"], PPO_PREPARATION_FORMAT)
        self.assertTrue(metadata["collection"]["exactOldLogProbabilityForEveryLearnerDecision"])
        self.assertTrue(metadata["collection"]["exactNormalExpertLabelForEveryLearnerDecision"])
        self.assertFalse(metadata["returnsAndAdvantages"]["futureHoldoutUsed"])
        self.assertFalse(metadata["returnsAndAdvantages"]["opponentHiddenHandsUsed"])
        self.assertGreater(metadata["policyEntropy"]["count"], 0)
        privacy = metadata["privacy"]
        self.assertTrue(privacy["actorPublicOnly"])
        self.assertTrue(privacy["privilegedCriticStateSeparate"])
        self.assertNotIn("privileged_states", privacy["actorInputFields"])
        self.assertNotIn("opponentHands", privacy["actorInputFields"])
        self.assertEqual(
            metadata["opponentAndSeatBalance"]["learnerIdentityMaxMinusMin"]["4"], 0
        )
        self.assertTrue(
            metadata["opponentAndSeatBalance"][
                "everyIdentityReceivesLearnerExperience"
            ]
        )
        critic_binding = metadata["privilegedCriticBinding"]
        self.assertEqual(
            critic_binding["layoutSha256"], CANONICAL_PRIVILEGED_LAYOUT_SHA256
        )
        self.assertTrue(
            critic_binding["perPlayerCountLiveLayoutAudits"]["4"][
                "liveVectorMatchedCanonicalLayout"
            ]
        )
        for relative, checksum in metadata["sourceHashes"].items():
            self.assertEqual(len(checksum), 64, relative)

    def test_privileged_critic_layout_drift_is_rejected(self) -> None:
        env = DalmutiScalarEnv(4, acts=1, seed=830_488_003)
        audit = assert_canonical_privileged_layout(env)
        self.assertEqual(audit["layoutSha256"], CANONICAL_PRIVILEGED_LAYOUT_SHA256)
        tampered = env.privileged_state().clone()
        tampered[29] += 1.0
        with mock.patch.object(env, "privileged_state", return_value=tampered):
            with self.assertRaisesRegex(RuntimeError, "layout drifted"):
                assert_canonical_privileged_layout(env)

    def test_match_shards_and_namespace_configuration_are_disjoint(self) -> None:
        base = dict(
            run_namespace="sharded",
            seed_base=777,
            player_counts=(4,),
            matches_per_player_count=4,
            acts=1,
            candidate_seats_per_act=1,
            opponent_candidate_fraction=0.0,
            lane_count=1,
            device="cpu",
        )
        left_config = PPOCollectionConfig(match_shard_count=2, match_shard_index=0, **base)
        right_config = PPOCollectionConfig(match_shard_count=2, match_shard_index=1, **base)
        left_indexes = {
            index for index in range(4)
            if index % left_config.match_shard_count == left_config.match_shard_index
        }
        right_indexes = {
            index for index in range(4)
            if index % right_config.match_shard_count == right_config.match_shard_index
        }
        self.assertTrue(left_indexes.isdisjoint(right_indexes))
        self.assertEqual(left_indexes | right_indexes, set(range(4)))
        self.assertNotEqual(left_config.match_shard_index, right_config.match_shard_index)

    def test_output_is_immutable(self) -> None:
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            collect_v4_ppo(
                self.bundle,
                self.outputs["repeat-a"],
                PPOCollectionConfig(
                    run_namespace="repeat",
                    seed_base=830_400_001,
                    player_counts=(4,),
                    matches_per_player_count=2,
                    acts=1,
                    candidate_seats_per_act=2,
                    device="cpu",
                ),
            )

    def test_collected_npz_runs_one_real_v4_train_ppo_smoke_epoch(self) -> None:
        dataset = load_v4_dataset_npz(self.outputs["repeat-a"])
        output = self.root / "train-smoke"
        result = train_v4(
            dataset,
            output,
            V4TrainingConfig(
                epochs=1,
                batch_size=2,
                gradient_accumulation=1,
                bc_weight=0.1,
                ppo_weight=1.0,
                critic_weight=0.1,
                q_boost_coefficient=0.0,
                entropy_coefficient=0.0,
                amp=False,
                seed=830_499_001,
            ),
            device="cpu",
            initialize_actor_bundle=self.bundle,
            include_onnx=False,
        )
        self.assertEqual(result["completedEpochs"], 1)
        self.assertTrue((output / "candidate" / "manifest.json").is_file())
        metrics = result["metrics"][0]
        for name in ("loss", "policyLoss", "criticLoss", "approxKl"):
            self.assertTrue(math.isfinite(float(metrics[name])), name)
        raw_counts = metrics["eligibleSamplesSeen"]
        effective_counts = metrics["effectiveNonforcedActorSamplesSeen"]
        excluded_counts = metrics["forcedActorSamplesExcluded"]
        for name in ("behaviorCloning", "ppo"):
            self.assertEqual(
                raw_counts[name],
                effective_counts[name] + excluded_counts[name],
            )
            self.assertGreater(effective_counts[name], 0)
        self.assertEqual(
            raw_counts["critic"],
            int(dataset.loss_eligibility.critic.sum()),
        )
        run_manifest = json.loads(
            (output / "run-manifest.json").read_text(encoding="utf-8")
        )
        contract = run_manifest["trainingContract"]
        self.assertEqual(
            contract["effectiveNonforcedActorSampleCounts"]["ppo"],
            effective_counts["ppo"],
        )
        self.assertEqual(
            contract["actorPolicyMask"],
            "loss eligibility AND legal-action count greater than one",
        )

    def test_fresh_ppo_training_requires_the_collector_behavior_actor(self) -> None:
        dataset = load_v4_dataset_npz(self.outputs["repeat-a"])
        output = self.root / "missing-ppo-initialization"
        with self.assertRaisesRegex(ValueError, "initialize-actor-bundle"):
            train_v4(
                dataset,
                output,
                V4TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    bc_weight=0.0,
                    ppo_weight=1.0,
                    critic_weight=0.0,
                    q_boost_coefficient=0.0,
                    entropy_coefficient=0.0,
                    amp=False,
                ),
                device="cpu",
            )
        self.assertFalse(output.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA host-only collector smoke")
    def test_real_actor_collection_executes_masked_batches_on_cuda(self) -> None:
        output = self.root / "cuda-real" / "ppo.npz"
        result = collect_v4_ppo(
            self.bundle,
            output,
            PPOCollectionConfig(
                run_namespace="cuda-real",
                seed_base=830_499_771,
                player_counts=(4,),
                matches_per_player_count=1,
                acts=1,
                candidate_seats_per_act=4,
                opponent_candidate_fraction=0.0,
                temperature=1.0,
                epsilon_floor=1.0e-6,
                standardize_advantages=True,
                lane_count=1,
                device="cuda",
            ),
        )
        self.assertGreater(result.samples, 0)
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["execution"]["device"], "cuda")
        self.assertTrue(metadata["collection"]["batchedGpuMaskedLogitInference"])
        self.assertTrue(metadata["execution"]["cudaAvailable"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA host-only p4-p10 smoke")
    def test_real_cuda_rolling_lanes_cover_p4_through_p10_and_both_opponents(self) -> None:
        output = self.root / "cuda-p4-p10" / "ppo.npz"
        result = collect_v4_ppo(
            self.bundle,
            output,
            PPOCollectionConfig(
                run_namespace="cuda-p4-p10",
                seed_base=830_499_881,
                player_counts=tuple(range(4, 11)),
                matches_per_player_count=3,
                acts=1,
                candidate_seats_per_act=4,
                opponent_candidate_fraction=0.5,
                temperature=1.1,
                epsilon_floor=1.0e-6,
                standardize_advantages=True,
                lane_count=21,
                device="cuda",
            ),
        )
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        with np.load(output, allow_pickle=False) as archive:
            self.assertEqual(
                set(int(value) for value in archive["trajectory_player_counts"]),
                set(range(4, 11)),
            )
        complete = metadata["opponentAndSeatBalance"][
            "completeMatchRangeAcrossAllShards"
        ]
        self.assertTrue(complete["everyIdentityReceivesLearnerExperience"])
        self.assertTrue(
            all(value <= 1 for value in complete["learnerIdentityMaxMinusMin"].values())
        )
        self.assertGreater(metadata["actionRates"]["candidateOpponent"]["decisions"], 0)
        self.assertGreater(metadata["actionRates"]["normalOpponent"]["decisions"], 0)


if __name__ == "__main__":
    unittest.main()
