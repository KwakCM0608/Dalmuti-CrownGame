from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:  # pragma: no cover
    raise unittest.SkipTest("torch and numpy are required") from error

from v4_collect_fixed_match_ppo import evaluation_candidate_initial_seats
from v4_env import ACTION_COUNT, DalmutiScalarEnv, V4ActorObservation, round_chip_award
from v5_collect_mappo import (
    V5MAPPOCollectionConfig,
    V5TorchInferenceRuntime,
    canonicalize_v5_privileged_state,
    collect_v5_mappo,
    derive_v5_collection_match_seed,
    evaluator_team_act_reward,
    publish_v5_mappo_collection,
    v5_collection_match_ordinal,
    v5_collection_seed_permutation_parameters,
)
from v5_collection_plan import build_collection_plan, expected_planned_shard_metadata
from v5_dataset import load_v5_actor_shard, load_v5_training_shard
from v5_public import V5PublicObservation
from v5_public import v5_public_from_v4_actor_observation
from v5_model import (
    V5ActorConfig,
    V5CentralStateValueCritic,
    V5CriticConfig,
    V5PublicActor,
)


class V5MAPPOCollectorTests(unittest.TestCase):
    @staticmethod
    def _actor(
        observations: tuple[object, ...], normal_actions: tuple[int, ...]
    ) -> np.ndarray:
        assert len(observations) == len(normal_actions)
        result = np.full((len(observations), ACTION_COUNT), -1000.0, dtype=np.float64)
        for row, (observation, normal_action) in enumerate(
            zip(observations, normal_actions, strict=True)
        ):
            assert not hasattr(observation, "privileged_state")
            assert isinstance(observation, dict)
            legal = np.asarray(observation["legal_mask"], dtype=np.bool_)
            # Non-uniform public-only logits make exact log-probability replay
            # more discriminating than a uniform policy.
            result[row, legal] = np.linspace(-0.75, 0.75, int(legal.sum()))
            assert legal[normal_action]
        return result

    @staticmethod
    def _critic(states: tuple[torch.Tensor, ...]) -> np.ndarray:
        values = []
        for state in states:
            assert state.shape == (512,)
            values.append(float(state[:16].sum().item()) / 100.0)
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _public_encoder(observation: V4ActorObservation) -> dict[str, object]:
        assert isinstance(observation, V4ActorObservation)
        assert not hasattr(observation, "privileged_state")
        return {
            "actor_id": observation.actor_id,
            "legal_mask": observation.legal_mask.detach().cpu().numpy().copy(),
            "history_count": int(observation.history_mask.sum().item()),
        }

    def test_privileged_state_uses_one_canonical_float16_roundtrip(self) -> None:
        raw = torch.linspace(-1.001, 1.001, 512, dtype=torch.float32)
        canonical = canonicalize_v5_privileged_state(raw)
        expected = torch.from_numpy(
            raw.numpy().astype(np.float16).astype(np.float32)
        )
        self.assertEqual(canonical.dtype, torch.float32)
        self.assertTrue(torch.equal(canonical, expected))
        self.assertTrue(
            torch.equal(canonicalize_v5_privileged_state(canonical), canonical)
        )

    @staticmethod
    def _config(namespace: str = "v5-mappo-test") -> V5MAPPOCollectionConfig:
        return V5MAPPOCollectionConfig(
            run_namespace=namespace,
            seed_base=810_000_001,
            match_counts=((4, 1),),
        )

    def test_team_reward_is_exact_evaluator_metric_divided_by_five(self) -> None:
        finish = (0, 1, 2, 3)
        chips = {
            actor: round_chip_award(place, 4)
            for place, actor in enumerate(finish, start=1)
        }
        outcome = evaluator_team_act_reward(finish, chips, (0, 2))
        self.assertEqual(outcome.candidate_mean_chip, 2.5)
        self.assertEqual(outcome.normal_mean_chip, 1.5)
        self.assertEqual(outcome.chip_difference, 1.0)
        self.assertEqual(outcome.pairwise_before, 3)
        self.assertEqual(outcome.pairwise_comparisons, 4)
        self.assertEqual(outcome.pairwise_rate, 0.75)
        self.assertEqual(outcome.pairwise_centered, 0.25)
        self.assertEqual(outcome.team_reward, (1.0 + 0.25 * 0.25) / 5.0)

    def test_two_million_global_match_seeds_are_exactly_unique(self) -> None:
        namespace = "v5-two-million-seed-proof"
        seed_base = 810_123_456
        multiplier, offset = v5_collection_seed_permutation_parameters(
            namespace,
            seed_base,
        )
        self.assertEqual(multiplier & 1, 1)
        ordinals = np.arange(2_000_000, dtype=np.uint64)
        seeds = (
            multiplier * ordinals + np.uint64(offset)
        ) & np.uint64(0xFFFF_FFFF)
        self.assertEqual(np.unique(seeds).size, ordinals.size)
        for ordinal in (0, 1, 6, 7, 999_999, 1_999_999):
            player_count = 4 + ordinal % 7
            match_index = ordinal // 7
            self.assertEqual(
                v5_collection_match_ordinal(player_count, match_index),
                ordinal,
            )
            self.assertEqual(
                derive_v5_collection_match_seed(
                    namespace,
                    seed_base,
                    player_count,
                    match_index,
                ),
                int(seeds[ordinal]),
            )

    def test_every_fixed_candidate_decision_is_recorded_and_normal_is_not(self) -> None:
        collection = collect_v5_mappo(
            self._actor,
            self._critic,
            self._config(),
            public_encoder=self._public_encoder,
        )
        self.assertEqual(len(collection.matches), 1)
        match = collection.matches[0]
        expected_seats = evaluation_candidate_initial_seats(4, 0)
        expected_ids = tuple(sorted(match.initial_order[seat] for seat in expected_seats))
        self.assertEqual(match.candidate_initial_seats, expected_seats)
        self.assertEqual(match.candidate_ids, expected_ids)
        self.assertEqual(set(collection.decision_actor_ids.tolist()), set(expected_ids))
        self.assertFalse(set(range(4)) - set(expected_ids) & set(collection.decision_actor_ids.tolist()))
        self.assertEqual(tuple(outcome.act for outcome in match.act_outcomes), (1, 2, 3, 4, 5))
        for actor_id in expected_ids:
            actor_rows = np.flatnonzero(collection.decision_actor_ids == actor_id)
            self.assertGreater(actor_rows.size, 0)
            self.assertTrue(collection.done[actor_rows[-1]])
            self.assertEqual(int(collection.next_decision[actor_rows[-1]]), -1)
            self.assertEqual(int(collection.done[actor_rows].sum()), 1)
            for left, right in zip(actor_rows[:-1], actor_rows[1:]):
                self.assertEqual(int(collection.next_decision[left]), int(right))
        self.assertEqual(int(collection.done.sum()), len(expected_ids))
        self.assertTrue(np.array_equal(collection.gae.policy_mask, ~collection.forced))
        self.assertTrue(collection.gae.value_mask.all())
        self.assertTrue(np.isfinite(collection.gae.advantages).all())
        self.assertTrue(np.isfinite(collection.gae.returns).all())

    def test_log_probabilities_replay_and_collection_is_seed_deterministic(self) -> None:
        first = collect_v5_mappo(
            self._actor,
            self._critic,
            self._config("v5-replay-test"),
            public_encoder=self._public_encoder,
        )
        second = collect_v5_mappo(
            self._actor,
            self._critic,
            self._config("v5-replay-test"),
            public_encoder=self._public_encoder,
        )
        for name in (
            "match_offsets",
            "candidate_bitsets",
            "player_counts",
            "decision_actor_ids",
            "decision_acts",
            "normal_actions",
            "actions",
            "old_log_probs",
            "old_values",
            "rewards_to_next",
            "done",
            "forced",
            "next_decision",
        ):
            self.assertTrue(np.array_equal(getattr(first, name), getattr(second, name)), name)
        for match in first.matches:
            for item in match.decisions:
                self.assertAlmostEqual(
                    item.old_log_probability,
                    float(np.log(item.selected_probability)),
                    places=12,
                )
                self.assertTrue(0 <= item.action < ACTION_COUNT)
                if item.forced:
                    self.assertEqual(item.selected_probability, 1.0)
                    self.assertEqual(item.old_log_probability, 0.0)

    def test_config_rejects_noncanonical_behavior_and_empty_shard(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature=1.0"):
            V5MAPPOCollectionConfig("bad-temperature", 1, ((4, 1),), temperature=0.9)
        with self.assertRaisesRegex(ValueError, "epsilon_floor=0.0"):
            V5MAPPOCollectionConfig("bad-floor", 1, ((4, 1),), epsilon_floor=1.0e-6)
        with self.assertRaisesRegex(ValueError, "uint32"):
            V5MAPPOCollectionConfig(
                "overflow-match-index",
                1,
                ((4, 2),),
                match_start=0xFFFF_FFFF,
            )
        empty = V5MAPPOCollectionConfig(
            "empty-shard", 1, ((4, 1),), match_shard_count=2, match_shard_index=1
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            collect_v5_mappo(
                self._actor,
                self._critic,
                empty,
                public_encoder=self._public_encoder,
            )

    def test_real_public_contract_publishes_privacy_separated_mmap_shard(self) -> None:
        def public_actor(
            observations: tuple[object, ...], normal_actions: tuple[int, ...]
        ) -> np.ndarray:
            rows = []
            for observation, normal_action in zip(
                observations, normal_actions, strict=True
            ):
                self.assertIsInstance(observation, V5PublicObservation)
                assert isinstance(observation, V5PublicObservation)
                self.assertTrue(observation.legal_mask[normal_action])
                rows.append(np.where(observation.legal_mask, 0.0, -1000.0))
            return np.asarray(rows, dtype=np.float64)

        collection = collect_v5_mappo(
            public_actor,
            self._critic,
            self._config("v5-publish-test"),
        )
        plan = build_collection_plan(
            run_namespace="v5-planned-publish-boundary-s810000001",
            seed_base=810_000_001,
            behavior_actor_sha256="a" * 64,
            behavior_actor_manifest_sha256="b" * 64,
            behavior_critic_sha256="c" * 64,
            behavior_pair_id="1" * 64,
            behavior_pair_manifest_sha256="2" * 64,
            calibration_report_sha256="f" * 64,
            source_inventory={"gpu-training/mock.py": "d" * 64},
            total_matches=140,
            diagnostic_unbalanced=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "shard-000"
            published = publish_v5_mappo_collection(
                target,
                collection,
                behavior_actor_sha256="a" * 64,
                behavior_actor_manifest_sha256="b" * 64,
                behavior_critic_sha256="c" * 64,
                metadata=expected_planned_shard_metadata(plan, plan.shards[0]),
            )
            self.assertEqual(published.decisions, collection.decision_count)
            self.assertEqual(len(published.manifest_sha256), 64)
            actor = load_v5_actor_shard(target)
            self.assertEqual(
                actor.manifest["metadata"]["behaviorModelPairId"], "1" * 64
            )
            self.assertEqual(
                actor.manifest["metadata"]["behaviorModelPairManifestSha256"],
                "2" * 64,
            )
            self.assertNotIn("privileged_states", actor.arrays)
            self.assertNotIn("match_indices", actor.arrays)
            self.assertNotIn("match_seeds", actor.arrays)
            self.assertEqual(actor.decision_count, collection.decision_count)
            self.assertTrue(np.array_equal(actor.legal_mask(0), collection.matches[0].decisions[0].public.legal_mask))  # type: ignore[union-attr]
            training = load_v5_training_shard(target)
            self.assertTrue(np.array_equal(
                training.privileged_arrays["match_indices"],
                np.asarray(
                    [match.match_index for match in collection.matches],
                    dtype=np.uint32,
                ),
            ))
            self.assertTrue(np.array_equal(
                training.privileged_arrays["match_seeds"],
                np.asarray(
                    [
                        derive_v5_collection_match_seed(
                            collection.config.run_namespace,
                            collection.config.seed_base,
                            match.player_count,
                            match.match_index,
                        )
                        for match in collection.matches
                    ],
                    dtype=np.uint32,
                ),
            ))
            self.assertEqual(
                training.privileged_arrays["privileged_states"].shape,
                (collection.decision_count, 512),
            )
            self.assertEqual(
                training.privileged_arrays["privileged_states"].dtype,
                np.dtype(np.float16),
            )
            self.assertIsInstance(
                training.privileged_arrays["privileged_states"], np.memmap
            )
            stored_states = training.privileged_arrays["privileged_states"]
            replay_values = self._critic(
                tuple(
                    torch.from_numpy(
                        np.asarray(row, dtype=np.float32).copy()
                    )
                    for row in stored_states
                )
            ).astype(np.float32)
            self.assertTrue(
                np.array_equal(replay_values, collection.old_values)
            )
            stored_index = 0
            for match in collection.matches:
                for decision in match.decisions:
                    self.assertTrue(
                        np.array_equal(
                            decision.privileged_state,
                            stored_states[stored_index],
                        )
                    )
                    stored_index += 1
            for array in (
                *actor.arrays.values(),
                *training.actor.arrays.values(),
                *training.privileged_arrays.values(),
            ):
                if isinstance(array, np.memmap) and array._mmap is not None:
                    array._mmap.close()

    def test_lane_batching_changes_throughput_not_seeded_results(self) -> None:
        maximum_batch = 0

        def tracked_actor(
            observations: tuple[object, ...], normal_actions: tuple[int, ...]
        ) -> np.ndarray:
            nonlocal maximum_batch
            maximum_batch = max(maximum_batch, len(observations))
            return self._actor(observations, normal_actions)

        base = {
            "run_namespace": "v5-lane-parity",
            "seed_base": 810_000_002,
            "match_counts": ((4, 4),),
        }
        serial = collect_v5_mappo(
            self._actor,
            self._critic,
            V5MAPPOCollectionConfig(**base, lane_count=1),
            public_encoder=self._public_encoder,
        )
        batched = collect_v5_mappo(
            tracked_actor,
            self._critic,
            V5MAPPOCollectionConfig(**base, lane_count=4),
            public_encoder=self._public_encoder,
        )
        self.assertGreater(maximum_batch, 1)
        for name in (
            "match_offsets",
            "candidate_bitsets",
            "player_counts",
            "decision_actor_ids",
            "decision_acts",
            "normal_actions",
            "actions",
            "old_log_probs",
            "old_values",
            "rewards_to_next",
            "done",
            "forced",
            "next_decision",
        ):
            self.assertTrue(np.array_equal(getattr(serial, name), getattr(batched, name)), name)

    def test_torch_runtime_keeps_actor_public_and_critic_privileged(self) -> None:
        torch.manual_seed(20260802)
        actor = V5PublicActor(V5ActorConfig(
            history_latents=2,
            d_model=32,
            core_layers=1,
            action_layers=2,
            heads=4,
            feedforward=64,
        ))
        critic = V5CentralStateValueCritic(V5CriticConfig(
            d_model=32,
            hidden_layers=1,
            player_count_embedding=8,
        ))
        runtime = V5TorchInferenceRuntime(actor, critic, device="cpu")
        env = DalmutiScalarEnv(4, acts=5, seed=930_000_001, device="cpu")
        public = v5_public_from_v4_actor_observation(env.public_observation())
        normal = env.normal_action()
        logits = runtime.actor_batch((public,), (normal,))
        self.assertEqual(logits.shape, (1, ACTION_COUNT))
        self.assertEqual(int(np.argmax(logits[0])), normal)
        values = runtime.critic_batch((env.privileged_state(),))
        self.assertEqual(values.shape, (1,))
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
