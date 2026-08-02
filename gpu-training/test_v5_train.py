from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from v4_env import ACTION_COUNT
from v5_collect_mappo import (
    V5MAPPOCollectionConfig,
    collect_v5_mappo,
    publish_v5_mappo_collection,
    v5_collection_array_partitions,
)
from v5_dataset import load_v5_training_shard, publish_v5_shard
from v5_export import (
    load_v5_actor_bundle,
    sha256_file,
    tensor_state_sha256,
    v5_actor_bundle_digests,
    verify_v5_actor_bundle,
)
from v5_model import V5_POLICY_NUMERICS_SHA256, V5ActorConfig, V5CriticConfig
from v5_public import V5PublicObservation
from v5_train import (
    V5_BEHAVIOR_LOG_PROBABILITY_ABSOLUTE_TOLERANCE,
    V5TrainingConfig,
    enforce_v5_critic_hard_gates,
    enforce_v5_training_hard_gates,
    load_v5_critic_checkpoint,
    load_verified_v5_behavior_pair,
    publish_seeded_v5_initialization,
    publish_v5_model_pair_manifest,
    train_v5_mappo,
    _Source,
    _accumulation_loss_scale,
    _deterministic_optimizer_groups,
    _deterministic_shard_local_microbatches,
    _global_weight_contract,
    _step_accumulated_optimizers,
    _optimizer_group_loss_scale,
    _validate_gpu_memory_preflight_binding,
)


_ACTOR_CONFIG = V5ActorConfig(
    history_latents=2,
    d_model=32,
    core_layers=1,
    action_layers=2,
    heads=4,
    feedforward=64,
)
_CRITIC_CONFIG = V5CriticConfig(
    privileged_features=512,
    d_model=32,
    hidden_layers=1,
    player_count_embedding=4,
)


def _normal_prior_actor(
    observations: tuple[object, ...], normal_actions: tuple[int, ...]
) -> np.ndarray:
    logits = np.full((len(observations), ACTION_COUNT), -1.0e9, dtype=np.float64)
    for row, (observation, normal_action) in enumerate(
        zip(observations, normal_actions, strict=True)
    ):
        if type(observation) is not V5PublicObservation:
            raise TypeError("fixture Actor received a non-public observation")
        legal = observation.legal_mask
        legal_count = int(legal.sum())
        logits[row, legal] = 0.0
        if legal_count > 1:
            logits[row, normal_action] = np.log(9.0 * (legal_count - 1))
        if not bool(legal[normal_action]):
            raise AssertionError("Normal fixture action is illegal")
    return logits


def _zero_critic(states: tuple[torch.Tensor, ...]) -> np.ndarray:
    if any(state.shape != (512,) for state in states):
        raise AssertionError("fixture critic did not receive private 512-vectors")
    return np.zeros(len(states), dtype=np.float64)


def _fixture(root: Path) -> tuple[dict[str, object], Path, object]:
    initialization = publish_seeded_v5_initialization(
        root / "initial",
        seed=751_000_001,
        actor_config=_ACTOR_CONFIG,
        critic_config=_CRITIC_CONFIG,
        metadata={"fixture": True},
    )
    collection = collect_v5_mappo(
        _normal_prior_actor,
        _zero_critic,
        V5MAPPOCollectionConfig(
            run_namespace="v5-trainer-unit-fixture",
            seed_base=752_000_001,
            match_counts=((4, 1),),
            lane_count=1,
        ),
    )
    actor_digests = v5_actor_bundle_digests(initialization["actorBundle"])
    pair = load_verified_v5_behavior_pair(initialization["outputDirectory"])
    shard_path = root / "training-shard"
    publish_v5_mappo_collection(
        shard_path,
        collection,
        behavior_actor_sha256=actor_digests["actorSha256"],
        behavior_actor_manifest_sha256=actor_digests["manifestSha256"],
        behavior_critic_sha256=sha256_file(initialization["criticCheckpoint"]),
        metadata={
            "behaviorModelPairId": pair["pairId"],
            "behaviorModelPairManifestSha256": pair["pairManifestSha256"],
        },
    )
    return initialization, shard_path, collection


def _training_config() -> V5TrainingConfig:
    return V5TrainingConfig(
        seed=753_000_001,
        microbatch_size=32,
        gradient_accumulation=1,
        critic_batch_size=256,
        audit_batch_size=64,
        actor_learning_rate=1.0e-8,
        critic_learning_rate=3.0e-5,
        entropy_coefficient=0.0,
        normal_auxiliary_coefficient=0.0,
        use_amp=False,
        require_all_player_counts=False,
    )


class V5TrainerTests(unittest.TestCase):
    def test_shard_local_batches_are_complete_deterministic_and_seeded(self) -> None:
        shards = tuple(
            SimpleNamespace(actor=SimpleNamespace(decision_count=count))
            for count in (5, 7, 2)
        )
        source = _Source("0" * 64, shards, (5, 7, 2), (0, 5, 12, 14))

        def normalized(seed: int) -> list[tuple[int, tuple[int, ...]]]:
            return [
                (shard, tuple(int(value) for value in indexes))
                for shard, indexes in _deterministic_shard_local_microbatches(
                    source, microbatch_size=4, seed=seed
                )
            ]

        first = normalized(7001)
        self.assertEqual(first, normalized(7001))
        self.assertNotEqual(first, normalized(7002))
        self.assertEqual(sorted(len(indexes) for _, indexes in first), [1, 2, 3, 4, 4])
        for shard_id, count in enumerate(source.counts):
            observed = [
                value
                for shard, indexes in first
                if shard == shard_id
                for value in indexes
            ]
            self.assertEqual(sorted(observed), list(range(count)))
            self.assertEqual(len(observed), len(set(observed)))

    def test_actor_batches_cover_only_policy_rows_once(self) -> None:
        masks = (
            np.asarray([True, False, True, False, True], dtype=np.bool_),
            np.asarray([False, True, True], dtype=np.bool_),
        )
        shards = tuple(
            SimpleNamespace(
                actor=SimpleNamespace(
                    decision_count=len(mask), arrays={"policy_mask": mask}
                )
            )
            for mask in masks
        )
        source = _Source("0" * 64, shards, (5, 3), (0, 5, 8))
        batches = _deterministic_shard_local_microbatches(
            source,
            microbatch_size=2,
            seed=7003,
            population="policy",
        )
        for shard_id, mask in enumerate(masks):
            observed = [
                int(value)
                for batch_shard, indexes in batches
                if batch_shard == shard_id
                for value in indexes
            ]
            expected = np.flatnonzero(mask).tolist()
            self.assertEqual(sorted(observed), expected)
            self.assertEqual(len(observed), len(set(observed)))

    def test_training_config_fixes_actor_effective_batch_and_critic_candidates(self) -> None:
        for microbatch, accumulation in ((8, 4), (16, 2), (32, 1)):
            config = V5TrainingConfig(
                microbatch_size=microbatch,
                gradient_accumulation=accumulation,
            )
            self.assertEqual(
                config.microbatch_size * config.gradient_accumulation, 32
            )
        with self.assertRaisesRegex(ValueError, "effective batch 32"):
            V5TrainingConfig(microbatch_size=8, gradient_accumulation=2)
        with self.assertRaisesRegex(ValueError, "critic_batch_size"):
            V5TrainingConfig(critic_batch_size=128)

    def test_gpu_preflight_binding_is_exact_and_input_bound(self) -> None:
        config = _training_config()
        source = _Source("a" * 64, (), (), (0,))
        pair = {"pairId": "b" * 64}
        with self.assertRaisesRegex(ValueError, "requires a bound GPU"):
            train_v5_mappo(
                "missing-dataset",
                "missing-pair/actor-bundle",
                "missing-pair/critic.pt",
                "must-not-publish",
                device="cuda:0",
            )
        record = {
            "config": {
                "audit_batch_size": config.audit_batch_size,
                "critic_batch_size": config.critic_batch_size,
                "gradient_accumulation": config.gradient_accumulation,
                "microbatch_size": config.microbatch_size,
            },
            "datasetIdentitySha256": "a" * 64,
            "device": "cuda:0",
            "format": "dalmuti-v5-gpu-memory-admission-binding",
            "modelPairId": "b" * 64,
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "reportSha256": "c" * 64,
            "version": 1,
        }
        for target_device, allow_cpu in (
            (torch.device("cuda:0"), False),
            (torch.device("cpu"), False),
        ):
            with self.assertRaisesRegex(ValueError, "requires a bound GPU"):
                _validate_gpu_memory_preflight_binding(
                    None,
                    source=source,
                    model_pair=pair,
                    config=config,
                    device=target_device,
                    allow_unadmitted_cpu=allow_cpu,
                )
        self.assertIsNone(
            _validate_gpu_memory_preflight_binding(
                None,
                source=source,
                model_pair=pair,
                config=config,
                device=torch.device("cpu"),
                allow_unadmitted_cpu=True,
            )
        )
        self.assertEqual(
            _validate_gpu_memory_preflight_binding(
                record,
                source=source,
                model_pair=pair,
                config=config,
                device=torch.device("cuda:0"),
                allow_unadmitted_cpu=False,
            ),
            record,
        )
        attacked = json.loads(json.dumps(record))
        attacked["config"]["critic_batch_size"] = 512
        with self.assertRaisesRegex(ValueError, "critic_batch_size"):
            _validate_gpu_memory_preflight_binding(
                attacked,
                source=source,
                model_pair=pair,
                config=config,
                device=torch.device("cuda:0"),
                allow_unadmitted_cpu=False,
            )
        attacked = json.loads(json.dumps(record))
        attacked["extra"] = True
        with self.assertRaisesRegex(ValueError, "fields drifted"):
            _validate_gpu_memory_preflight_binding(
                attacked,
                source=source,
                model_pair=pair,
                config=config,
                device=torch.device("cuda:0"),
                allow_unadmitted_cpu=False,
            )

    def test_partial_microbatch_gradient_is_row_weighted(self) -> None:
        microbatches = [
            (0, np.arange(4, dtype=np.int64)),
            (1, np.arange(1, dtype=np.int64)),
            (0, np.arange(3, dtype=np.int64)),
        ]
        rows = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([9.0]),
        )
        parameter = torch.tensor(2.0, requires_grad=True)
        for index in range(2):
            loss = parameter * rows[index].mean()
            (loss * _accumulation_loss_scale(microbatches, index, 2)).backward()
        self.assertAlmostEqual(
            float(parameter.grad),
            float(torch.cat(rows).mean()),
            places=6,
        )

    def test_remainders_share_one_row_weighted_adamw_step(self) -> None:
        batches = [
            (0, np.arange(4, dtype=np.int64)),
            (1, np.arange(1, dtype=np.int64)),
        ]
        groups = _deterministic_optimizer_groups(
            batches, target_rows=4, seed=7004, seed_domain="actor"
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(sum(len(indexes) for _, indexes in groups[0]), 5)
        values = {
            0: torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
            1: torch.tensor([25.0], dtype=torch.float64),
        }

        grouped = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        grouped_optimizer = torch.optim.AdamW(
            [grouped], lr=0.1, weight_decay=0.5
        )
        for index, (shard_id, indexes) in enumerate(groups[0]):
            loss = grouped * values[shard_id][indexes].mean()
            (loss * _optimizer_group_loss_scale(groups[0], index)).backward()
        grouped_optimizer.step()

        direct = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        direct_optimizer = torch.optim.AdamW(
            [direct], lr=0.1, weight_decay=0.5
        )
        (direct * torch.cat(tuple(values.values())).mean()).backward()
        direct_optimizer.step()
        self.assertAlmostEqual(
            float(grouped.detach()), float(direct.detach()), places=12
        )

        legacy = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        legacy_optimizer = torch.optim.AdamW(
            [legacy], lr=0.1, weight_decay=0.5
        )
        for rows in values.values():
            legacy_optimizer.zero_grad(set_to_none=True)
            (legacy * rows.mean()).backward()
            legacy_optimizer.step()
        self.assertNotAlmostEqual(
            float(legacy.detach()), float(direct.detach()), places=6
        )

    def test_actor_physical_choices_preserve_effective_group_population(self) -> None:
        expected_rows: list[int] | None = None
        expected_steps: int | None = None
        for physical in (8, 16, 32):
            batches: list[tuple[int, np.ndarray]] = []
            for shard_id, count in enumerate((37, 44)):
                order = np.arange(count, dtype=np.int64)
                batches.extend(
                    (shard_id, order[start : start + physical])
                    for start in range(0, count, physical)
                )
            groups = _deterministic_optimizer_groups(
                batches,
                target_rows=32,
                seed=7005,
                seed_domain="actor",
            )
            rows = sorted(
                sum(len(indexes) for _, indexes in group) for group in groups
            )
            self.assertEqual(rows, [32, 49])
            self.assertEqual(sum(rows), 81)
            if expected_rows is None:
                expected_rows = rows
                expected_steps = len(groups)
            else:
                self.assertEqual(rows, expected_rows)
                self.assertEqual(len(groups), expected_steps)

    def test_global_equal_p_weights_replace_shard_local_weights(self) -> None:
        def shard(player: int, forced: list[bool]) -> object:
            forced_array = np.asarray(forced, dtype=np.bool_)
            count = len(forced)
            legal_bits = np.zeros((count, 30), dtype=np.uint8)
            legal_bits[:, 0] = np.where(forced_array, 0b0000_0001, 0b0000_0011)
            arrays = MappingProxyType({
                "global_codes": np.column_stack((
                    np.ones(count, dtype=np.int32),
                    np.full(count, player, dtype=np.int32),
                    np.ones((count, 4), dtype=np.int32),
                )),
                "policy_mask": ~forced_array,
                "value_mask": np.ones(count, dtype=np.bool_),
                "forced": forced_array,
                "legal_action_bits": legal_bits,
                "actions": np.zeros(count, dtype=np.uint16),
                "normal_actions": np.zeros(count, dtype=np.uint16),
                "old_log_probs": np.where(
                    forced_array, np.float32(0.0), np.float32(-0.1)
                ).astype(np.float32),
                # These intentionally mimic individually balanced single-p shards.
                "policy_loss_weights": (~forced_array).astype(np.float32),
                "value_loss_weights": np.ones(count, dtype=np.float32),
            })
            return SimpleNamespace(
                actor=SimpleNamespace(arrays=arrays, decision_count=count),
                close=lambda: None,
            )

        shards = (shard(4, [False, True, False, True]), shard(5, [False] * 6))
        source = _Source("0" * 64, shards, (4, 6), (0, 4, 10))
        contract = _global_weight_contract(source, require_all_player_counts=False)
        self.assertNotEqual(contract.policy_weights[4], 1.0)
        self.assertAlmostEqual(
            contract.policy_counts[4] * contract.policy_weights[4],
            contract.policy_counts[5] * contract.policy_weights[5],
            places=6,
        )
        self.assertAlmostEqual(
            contract.value_counts[4] * contract.value_weights[4],
            contract.value_counts[5] * contract.value_weights[5],
            places=6,
        )
        with self.assertRaisesRegex(ValueError, "p4..p10"):
            _global_weight_contract(source, require_all_player_counts=True)

    def test_value_only_optimizer_group_does_not_decay_actor(self) -> None:
        actor = torch.nn.Linear(3, 2)
        critic = torch.nn.Linear(3, 1)
        actor_optimizer = torch.optim.AdamW(
            actor.parameters(), lr=0.1, weight_decay=0.5
        )
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before = [value.detach().clone() for value in actor.parameters()]
        critic(torch.ones(1, 3)).sum().backward()
        stepped = _step_accumulated_optimizers(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            scaler=scaler,
            max_gradient_norm=1.0,
            device=torch.device("cpu"),
        )
        self.assertFalse(stepped)
        for expected, actual in zip(before, actor.parameters(), strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_seeded_initialization_reproduces_tensor_states_and_records_both_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = publish_seeded_v5_initialization(
                root / "first",
                seed=754_000_001,
                actor_config=_ACTOR_CONFIG,
                critic_config=_CRITIC_CONFIG,
            )
            # Disturb ambient RNG to prove the factory does not consume it.
            torch.manual_seed(99)
            _ = torch.randn(17)
            second = publish_seeded_v5_initialization(
                root / "second",
                seed=754_000_001,
                actor_config=_ACTOR_CONFIG,
                critic_config=_CRITIC_CONFIG,
            )
            first_actor, first_manifest = load_v5_actor_bundle(first["actorBundle"])
            second_actor, second_manifest = load_v5_actor_bundle(second["actorBundle"])
            first_critic, first_payload = load_v5_critic_checkpoint(
                first["criticCheckpoint"]
            )
            second_critic, second_payload = load_v5_critic_checkpoint(
                second["criticCheckpoint"]
            )
            self.assertEqual(first["seeds"], second["seeds"])
            self.assertNotEqual(
                first["seeds"]["actorInitializationSeed"],
                first["seeds"]["criticInitializationSeed"],
            )
            self.assertEqual(
                tensor_state_sha256(first_actor.state_dict()),
                tensor_state_sha256(second_actor.state_dict()),
            )
            self.assertEqual(
                tensor_state_sha256(first_critic.state_dict()),
                tensor_state_sha256(second_critic.state_dict()),
            )
            for name, value in first["seeds"].items():
                self.assertEqual(first_manifest["metadata"][name], value)
                self.assertEqual(first_payload["metadata"][name], value)
                self.assertEqual(second_manifest["metadata"][name], value)
                self.assertEqual(second_payload["metadata"][name], value)
            first_pair = load_verified_v5_behavior_pair(first["outputDirectory"])
            self.assertEqual(first_pair["pairId"], first["modelPair"]["pairId"])
            self.assertEqual(
                first_pair["actorTensorStateSha256"],
                tensor_state_sha256(first_actor.state_dict()),
            )
            mixed = root / "mixed"
            mixed.mkdir()
            other = publish_seeded_v5_initialization(
                root / "other",
                seed=754_000_002,
                actor_config=_ACTOR_CONFIG,
                critic_config=_CRITIC_CONFIG,
            )
            shutil.copytree(Path(first["actorBundle"]), mixed / "actor-bundle")
            shutil.copy2(other["criticCheckpoint"], mixed / "critic.pt")
            with self.assertRaisesRegex(ValueError, "initialization seed mismatch"):
                publish_v5_model_pair_manifest(mixed)

    def test_real_shard_training_is_deterministic_private_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialization, shard_path, collection = _fixture(root)
            first = train_v5_mappo(
                shard_path,
                initialization["actorBundle"],
                initialization["criticCheckpoint"],
                root / "first-output",
                config=_training_config(),
                device="cpu",
                allow_unadmitted_cpu=True,
            )
            second = train_v5_mappo(
                shard_path,
                initialization["actorBundle"],
                initialization["criticCheckpoint"],
                root / "second-output",
                config=_training_config(),
                device="cpu",
                allow_unadmitted_cpu=True,
            )
            first_result = first["result"]
            second_result = second["result"]
            self.assertTrue(first_result["hardGates"]["passed"])
            self.assertLessEqual(
                first_result["initialPolicyReplay"][
                    "allRowMaximumAbsoluteOldLogProbabilityError"
                ],
                V5_BEHAVIOR_LOG_PROBABILITY_ABSOLUTE_TOLERANCE,
            )
            self.assertEqual(
                first_result["initialPolicyReplay"]["allRowsReplayed"],
                collection.decision_count,
            )
            self.assertEqual(
                first_result["initialCriticAudit"]["allRowsAudited"],
                collection.decision_count,
            )
            self.assertEqual(
                first_result["postEpochCriticAudit"]["allRowsAudited"],
                collection.decision_count,
            )
            self.assertLess(
                first_result["postEpochCriticAudit"]["weightedHuberLoss"],
                first_result["initialCriticAudit"]["weightedHuberLoss"],
            )
            self.assertGreater(int(collection.forced.sum()), 0)
            self.assertEqual(
                first_result["initialPolicyReplay"]["forcedRows"],
                int(collection.forced.sum()),
            )
            self.assertEqual(
                first_result["initialPolicyReplay"]["actorForwardRows"],
                int((~collection.forced).sum()),
            )
            self.assertEqual(
                first_result["initialPolicyReplay"][
                    "forcedActorForwardRowsSkipped"
                ],
                int(collection.forced.sum()),
            )
            self.assertEqual(
                first_result["initialPolicyReplay"][
                    "allRowsSemanticallyVisited"
                ],
                collection.decision_count,
            )
            self.assertEqual(
                first_result["epoch"]["actorDecisionRowsSeen"],
                int((~collection.forced).sum()),
            )
            self.assertEqual(
                first_result["epoch"]["criticDecisionRowsSeen"],
                collection.decision_count,
            )
            self.assertEqual(
                first_result["outputActor"]["tensorStateSha256"],
                second_result["outputActor"]["tensorStateSha256"],
            )
            self.assertEqual(
                first_result["outputCritic"]["tensorStateSha256"],
                second_result["outputCritic"]["tensorStateSha256"],
            )
            self.assertEqual(
                first_result["epoch"], second_result["epoch"]
            )
            self.assertEqual(
                first_result["postEpochPolicyAudit"],
                second_result["postEpochPolicyAudit"],
            )
            output = Path(first["outputDirectory"])
            actor_bundle = output / "actor-bundle"
            verify_v5_actor_bundle(actor_bundle)
            self.assertNotIn(
                "critic.pt", {path.name for path in actor_bundle.iterdir()}
            )
            self.assertTrue((output / "critic.pt").is_file())
            self.assertTrue((output / "optimizer.pt").is_file())
            self.assertTrue((output / "training-checkpoint.pt").is_file())
            manifest_raw = (output / "manifest.json").read_bytes()
            manifest = json.loads(manifest_raw.decode("ascii"))
            self.assertEqual(
                manifest_raw,
                json.dumps(
                    manifest,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii"),
            )
            for relative, record in manifest["files"].items():
                self.assertEqual(sha256_file(output / relative), record["sha256"])
            checkpoint = torch.load(
                output / "training-checkpoint.pt", weights_only=False
            )
            optimizer = torch.load(output / "optimizer.pt", weights_only=True)
            self.assertEqual(
                checkpoint["optimizerSha256"], sha256_file(output / "optimizer.pt")
            )
            self.assertEqual(
                checkpoint["actorStateSha256"],
                tensor_state_sha256(checkpoint["actorStateDict"]),
            )
            self.assertEqual(
                checkpoint["criticStateSha256"],
                tensor_state_sha256(checkpoint["criticStateDict"]),
            )
            self.assertEqual(
                checkpoint["modelPairId"], first_result["outputModelPair"]["pairId"]
            )
            self.assertEqual(optimizer["modelPairId"], checkpoint["modelPairId"])

    def test_all_row_replay_rejects_stale_old_log_probability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialization, _, collection = _fixture(root)
            actor_arrays, privileged_arrays = v5_collection_array_partitions(collection)
            actor_arrays["old_log_probs"] = actor_arrays["old_log_probs"].copy()
            nonforced = np.flatnonzero(~actor_arrays["forced"])
            self.assertGreater(nonforced.size, 0)
            actor_arrays["old_log_probs"][nonforced[0]] -= np.float32(0.01)
            actor_arrays["selected_action_probabilities"] = actor_arrays[
                "selected_action_probabilities"
            ].copy()
            actor_arrays["selected_action_probabilities"][nonforced[0]] = np.float32(
                np.exp(float(actor_arrays["old_log_probs"][nonforced[0]]))
            )
            digests = v5_actor_bundle_digests(initialization["actorBundle"])
            pair = load_verified_v5_behavior_pair(initialization["outputDirectory"])
            stale = root / "stale-shard"
            publish_v5_shard(
                stale,
                actor_arrays,
                privileged_arrays,
                metadata={
                    "behaviorActorSha256": digests["actorSha256"],
                    "behaviorActorManifestSha256": digests["manifestSha256"],
                    "behaviorCriticSha256": sha256_file(
                        initialization["criticCheckpoint"]
                    ),
                    "behaviorModelPairId": pair["pairId"],
                    "behaviorModelPairManifestSha256": pair[
                        "pairManifestSha256"
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "old-log-probability"):
                train_v5_mappo(
                    stale,
                    initialization["actorBundle"],
                    initialization["criticCheckpoint"],
                    root / "must-not-publish",
                    config=_training_config(),
                    device="cpu",
                    allow_unadmitted_cpu=True,
                )
            self.assertFalse((root / "must-not-publish").exists())

    def test_forced_zero_log_probability_contract_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialization, _, collection = _fixture(root)
            actor_arrays, privileged_arrays = v5_collection_array_partitions(collection)
            actor_arrays.pop("selected_action_probabilities", None)
            actor_arrays.pop("policy_entropies", None)
            actor_arrays["old_log_probs"] = actor_arrays["old_log_probs"].copy()
            forced = np.flatnonzero(actor_arrays["forced"])
            self.assertGreater(forced.size, 0)
            actor_arrays["old_log_probs"][forced[0]] = np.float32(-0.25)
            digests = v5_actor_bundle_digests(initialization["actorBundle"])
            pair = load_verified_v5_behavior_pair(initialization["outputDirectory"])
            attacked = root / "forced-attack-shard"
            publish_v5_shard(
                attacked,
                actor_arrays,
                privileged_arrays,
                metadata={
                    "behaviorActorSha256": digests["actorSha256"],
                    "behaviorActorManifestSha256": digests["manifestSha256"],
                    "behaviorCriticSha256": sha256_file(
                        initialization["criticCheckpoint"]
                    ),
                    "behaviorModelPairId": pair["pairId"],
                    "behaviorModelPairManifestSha256": pair[
                        "pairManifestSha256"
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "exact zero old log probability"):
                train_v5_mappo(
                    attacked,
                    initialization["actorBundle"],
                    initialization["criticCheckpoint"],
                    root / "must-not-publish-forced-attack",
                    config=_training_config(),
                    device="cpu",
                    allow_unadmitted_cpu=True,
                )
            self.assertFalse((root / "must-not-publish-forced-attack").exists())

    def test_behavior_hash_binding_and_hard_gate_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "approxKl"):
            enforce_v5_training_hard_gates(
                {
                    "approxKl": 0.02,
                    "clipFraction": 0.0,
                    "entropyRetentionRatio": 1.0,
                }
            )

        initial_critic = {
            "weightedHuberLoss": 1.0,
            "explainedVariance": 0.10,
            "perPlayerCount": {"4": {"weightedHuberLoss": 1.0}},
        }
        enforce_v5_critic_hard_gates(
            initial_critic,
            {
                "weightedHuberLoss": 0.9,
                "explainedVariance": 0.08,
                "perPlayerCount": {"4": {"weightedHuberLoss": 1.1}},
            },
        )
        with self.assertRaisesRegex(RuntimeError, "strictly below"):
            enforce_v5_critic_hard_gates(initial_critic, initial_critic)
        with self.assertRaisesRegex(RuntimeError, "p4"):
            enforce_v5_critic_hard_gates(
                initial_critic,
                {
                    "weightedHuberLoss": 0.9,
                    "explainedVariance": 0.10,
                    "perPlayerCount": {"4": {"weightedHuberLoss": 1.1001}},
                },
            )
        with self.assertRaisesRegex(RuntimeError, "explainedVariance"):
            enforce_v5_critic_hard_gates(
                initial_critic,
                {
                    "weightedHuberLoss": 0.9,
                    "explainedVariance": 0.079,
                    "perPlayerCount": {"4": {"weightedHuberLoss": 1.0}},
                },
            )
        with self.assertRaisesRegex(RuntimeError, "clipFraction"):
            enforce_v5_training_hard_gates(
                {
                    "approxKl": 0.0,
                    "clipFraction": 0.25,
                    "entropyRetentionRatio": 1.0,
                }
            )
        with self.assertRaisesRegex(RuntimeError, "entropyRetention"):
            enforce_v5_training_hard_gates(
                {
                    "approxKl": 0.0,
                    "clipFraction": 0.0,
                    "entropyRetentionRatio": 0.70,
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialization, shard_path, _ = _fixture(root)
            second = publish_seeded_v5_initialization(
                root / "wrong-initial",
                seed=755_000_001,
                actor_config=_ACTOR_CONFIG,
                critic_config=_CRITIC_CONFIG,
            )
            with self.assertRaisesRegex(ValueError, "behavior binding mismatch"):
                train_v5_mappo(
                    shard_path,
                    second["actorBundle"],
                    second["criticCheckpoint"],
                    root / "must-not-publish",
                    config=_training_config(),
                    device="cpu",
                    allow_unadmitted_cpu=True,
                )

    def test_actor_loader_never_receives_privileged_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, shard_path, _ = _fixture(root)
            with load_v5_training_shard(shard_path) as shard:
                self.assertNotIn("privileged_states", shard.actor.arrays)
                self.assertIn("privileged_states", shard.privileged_arrays)


if __name__ == "__main__":
    unittest.main()
