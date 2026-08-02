from __future__ import annotations

from dataclasses import fields
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from v4_dataset import (
    V4_FIXED_COLLECTION_PLAN_ID,
    V4_FIXED_PPO_SOURCE_CONTRACT,
    V4_LEGACY_PPO_SOURCE_CONTRACT,
    V4LossEligibility,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
    canonical_fixed_ppo_behavior_policy_contract,
    canonical_fixed_ppo_reward_contract,
    create_v4_smoke_dataset,
)
from v4_collect_ppo import masked_categorical_probabilities
from v4_export import (
    canonical_json_bytes,
    export_v4_actor_bundle,
    load_v4_actor_checkpoint,
    sha256_file,
)
from v4_model import (
    V4ActorConfig,
    V4CriticConfig,
    V4PublicActor,
    canonical_v4_policy_numerics_contract,
)
from v4_objectives import masked_behavior_cloning_loss as real_bc_loss
from v4_train import (
    V4TrainingConfig,
    _actor_state_sha256,
    _audit_initial_policy_reproduction,
    _balanced_batch_estimator_multiplier,
    _parser,
    _player_count_balance_contract,
    _policy_audit_batch_size,
    _resolve_training_contract,
    _resolve_resume,
    _resume_training,
    _validated_checkpoint_cuda_rng_states,
    train_v4,
)


FIXED_SOURCE = V4_FIXED_PPO_SOURCE_CONTRACT
LEGACY_SOURCE = V4_LEGACY_PPO_SOURCE_CONTRACT
FIXED_PLAN_SHA256 = "b" * 64
FIXED_PLAN_ID = (
    f"{V4_FIXED_COLLECTION_PLAN_ID}:sha256={FIXED_PLAN_SHA256}"
)


FIXED_REWARD_ID = canonical_fixed_ppo_reward_contract(
    {
        "version": 1,
        "chipComponent": (
            "mean exact chip award(fixed candidate IDs) - "
            "mean exact chip award(fixed Normal IDs)"
        ),
        "pairwiseRate": (
            "candidate-before-Normal finish pairs / "
            "(candidate identity count * Normal identity count)"
        ),
        "pairwiseCenteredComponent": "pairwiseRate - 0.5",
        "pairwiseCoefficient": 0.25,
        "actTotal": (
            "(chipComponent + pairwiseCoefficient * pairwiseCenteredComponent) / 5"
        ),
        "trajectoryReturn": (
            "sum of actTotal from trajectory act through act five; "
            "never divide by remaining horizon"
        ),
        "rawComponentsSeparatelyBoundForAblation": True,
    }
)[0]
FIXED_BEHAVIOR_ID = canonical_fixed_ppo_behavior_policy_contract(
    {
        "behaviorPolicyContract": "raw-masked-softmax-v1",
        "behaviorPolicyContractVersion": 1,
        "rawMaskedSoftmaxExactBinding": True,
        "initialOldCurrentRatioMathematicallyOneForFrozenActor": True,
        "initialOldCurrentLogProbabilityAbsoluteTolerance": 2.0e-5,
        "fixedPpoActorAutocastDisabled": True,
        "requiresFullDatasetInitialPolicyReproductionAudit": True,
        "dropoutDisabled": True,
        "temperature": 1.0,
        "epsilonFloorPerLegalAction": 0.0,
    },
    require_exact_fields=True,
)[0]


def tiny_configs(*, actor_dropout: float = 0.0) -> tuple[V4ActorConfig, V4CriticConfig]:
    return (
        V4ActorConfig(
            max_players=10,
            max_history=2,
            d_model=16,
            layers=1,
            heads=4,
            feedforward=32,
            action_hidden=12,
            dropout=actor_dropout,
        ),
        V4CriticConfig(
            privileged_features=12,
            d_model=16,
            hidden_layers=1,
            action_hidden=12,
        ),
    )


def fixed_dataset(
    player_counts: list[int],
    *,
    lengths: list[int] | None = None,
    force_first_row: bool = False,
    sources: tuple[str, ...] = (FIXED_SOURCE,),
    requires_balance: bool | None = None,
    behavior_actor_sha256: str = "a" * 64,
    behavior_actor: V4PublicActor | None = None,
    actor_dropout: float = 0.0,
    fixed_collection_plan_ids: tuple[str, ...] | None = None,
) -> V4TrajectoryDataset:
    actor_config, critic_config = tiny_configs(actor_dropout=actor_dropout)
    lengths = lengths or [2] * len(player_counts)
    if len(lengths) != len(player_counts):
        raise ValueError("lengths must match player_counts")
    time_steps = max(lengths)
    smoke = create_v4_smoke_dataset(
        actor_config,
        critic_config,
        trajectories=len(player_counts),
        time_steps=time_steps,
        seed=811,
    )
    values = {
        field.name: getattr(smoke.tensors, field.name).clone()
        for field in fields(V4TrajectoryTensors)
    }
    values["valid_masks"].zero_()
    values["dones"].zero_()
    values["player_mask"].zero_()
    values["legal_masks"].zero_()
    values["actions"].zero_()
    values["expert_actions"].zero_()
    values["old_action_log_probs"].zero_()
    for trajectory, (player_count, length) in enumerate(
        zip(player_counts, lengths, strict=True)
    ):
        values["valid_masks"][trajectory, :length] = True
        values["dones"][trajectory, length - 1] = True
        values["player_mask"][trajectory, :, :player_count] = True
        values["global_features"][trajectory, :, 0] = (player_count - 4) / 6.0
        values["legal_masks"][trajectory, :length, :2] = True
        if force_first_row:
            values["legal_masks"][trajectory, 0, 1] = False
    tensors = V4TrajectoryTensors(**values)
    if behavior_actor is not None:
        if behavior_actor.config.to_dict() != actor_config.to_dict():
            raise ValueError("behavior actor config must match the fixture")
        safe_legal = tensors.legal_masks.clone()
        safe_legal[~tensors.valid_masks, 0] = True
        with torch.no_grad():
            logits = behavior_actor.eval()(
                tensors.global_features.reshape(-1, actor_config.global_features),
                tensors.rank_features.reshape(
                    -1, actor_config.rank_tokens, actor_config.rank_features
                ),
                tensors.player_features.reshape(
                    -1, actor_config.max_players, actor_config.player_features
                ),
                tensors.player_mask.reshape(-1, actor_config.max_players),
                tensors.memory_trace_features.reshape(
                    -1, actor_config.memory_tokens, actor_config.memory_features
                ),
                tensors.history_features.reshape(
                    -1, actor_config.max_history, actor_config.history_features
                ),
                tensors.history_mask.reshape(-1, actor_config.max_history),
                safe_legal.reshape(-1, safe_legal.shape[-1]),
            ).reshape(*tensors.actions.shape, -1)
        for trajectory, time_index in tensors.valid_masks.nonzero().tolist():
            probabilities = masked_categorical_probabilities(
                logits[trajectory, time_index],
                tensors.legal_masks[trajectory, time_index],
                temperature=1.0,
                epsilon_floor=0.0,
            )
            action = int(tensors.actions[trajectory, time_index])
            tensors.old_action_log_probs[trajectory, time_index] = math.log(
                float(probabilities[action])
            )
    valid = tensors.valid_masks.clone()
    if requires_balance is None:
        requires_balance = sources == (FIXED_SOURCE,)
    if fixed_collection_plan_ids is None:
        fixed_collection_plan_ids = (
            (FIXED_PLAN_ID,) if FIXED_SOURCE in sources else ()
        )
    eligibility = V4LossEligibility(
        behavior_cloning=valid.clone(),
        ppo=valid.clone(),
        critic=valid.clone(),
        preparation_format="dalmuti-v4-merged-prepared-dataset-metadata",
        preparation_version=1,
        behavior_actor_sha256s=(behavior_actor_sha256,),
        ppo_source_contracts=sources,
        requires_player_count_balanced_loss=requires_balance,
        requires_qboost_coefficient_zero=FIXED_SOURCE in sources,
        ppo_reward_contracts=(FIXED_REWARD_ID,) if FIXED_SOURCE in sources else (),
        ppo_behavior_policy_contracts=(FIXED_BEHAVIOR_ID,)
        if FIXED_SOURCE in sources
        else (),
        fixed_collection_plan_ids=fixed_collection_plan_ids,
    )
    return V4TrajectoryDataset(
        tensors,
        actor_config,
        critic_config,
        loss_eligibility=eligibility,
        metadata={
            "preparationFormat": "dalmuti-v4-merged-prepared-dataset-metadata",
            "preparationVersion": 1,
        },
    )


def fixed_training_config(**kwargs: object) -> V4TrainingConfig:
    return V4TrainingConfig(
        expected_fixed_collection_plan_sha256=FIXED_PLAN_SHA256,
        **kwargs,
    )


def export_bound_actor(
    root: Path,
    *,
    seed: int = 20260802,
    actor_dropout: float = 0.0,
) -> tuple[V4PublicActor, Path, str]:
    torch.manual_seed(seed)
    actor_config, _ = tiny_configs(actor_dropout=actor_dropout)
    actor = V4PublicActor(actor_config).eval()
    bundle = root / f"actor-{seed}-{actor_dropout}"
    manifest = export_v4_actor_bundle(actor, bundle)
    return actor, bundle, str(manifest["files"]["actor.pt"]["sha256"])


def save_checkpoint_with_sidecar(
    path: Path,
    payload: object,
    *,
    sidecar_checksum: str | None = None,
) -> None:
    torch.save(payload, path)
    checksum = sidecar_checksum or sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{checksum}  {path.name}\n",
        encoding="ascii",
    )


class V4BalancedTrainingTests(unittest.TestCase):
    def test_balance_counts_exclude_forced_actor_rows_but_not_critic(self) -> None:
        dataset = fixed_dataset(
            [player_count for player_count in range(4, 11) for _ in range(2)],
            force_first_row=True,
        )
        contract = _player_count_balance_contract(dataset, FIXED_PLAN_SHA256)
        assert contract is not None
        counts = contract["eligibleRowCountsByLossAndPlayerCount"]
        for player_count in range(4, 11):
            key = str(player_count)
            self.assertEqual(counts["behaviorCloning"][key], 2)
            self.assertEqual(counts["ppo"][key], 2)
            self.assertEqual(counts["critic"][key], 4)
        for loss_name in ("behaviorCloning", "ppo", "critic"):
            total = contract["totalEligibleRowsByLoss"][loss_name]
            weights = contract[
                "runtimeFloat32WeightsByLossAndPlayerCount"
            ][loss_name]
            for player_count in range(4, 11):
                count = counts[loss_name][str(player_count)]
                self.assertAlmostEqual(
                    weights[str(player_count)],
                    total / (7 * count),
                    places=7,
                )

    def test_trajectory_batch_estimator_is_unbiased_for_variable_lengths(self) -> None:
        # Each tuple is (player count, row losses).  Player-count row totals are
        # deliberately unequal, as are trajectory lengths.
        trajectories = (
            (4, (1.0, 3.0, 5.0)),
            (4, (2.0,)),
            (5, (7.0, 11.0)),
            (5, (13.0, 17.0, 19.0, 23.0)),
            (6, (29.0,)),
            (7, (31.0, 37.0, 41.0)),
            (8, (43.0, 47.0)),
            (9, (53.0,)),
            (10, (59.0, 61.0, 67.0, 71.0)),
        )
        counts = {
            player_count: sum(
                len(rows) for value, rows in trajectories if value == player_count
            )
            for player_count in range(4, 11)
        }
        total = sum(counts.values())
        weights = {
            player_count: total / (len(counts) * count)
            for player_count, count in counts.items()
        }
        target = sum(
            weights[player_count] * sum(rows)
            for player_count, rows in trajectories
        ) / total
        estimates = []
        batch_size = 2
        multiplier = _balanced_batch_estimator_multiplier(
            trajectory_count=len(trajectories),
            batch_trajectory_count=batch_size,
            total_eligible_rows=total,
        )
        for selected in combinations(range(len(trajectories)), batch_size):
            weighted_sum = sum(
                weights[trajectories[index][0]] * sum(trajectories[index][1])
                for index in selected
            )
            estimates.append(multiplier * weighted_sum)
        self.assertAlmostEqual(sum(estimates) / len(estimates), target, places=12)

        # A homogeneous player-count batch retains its global inverse-frequency
        # scale; normalizing by its own weight sum would
        # incorrectly reduce this to the unweighted row mean.
        homogeneous = (0, 1)
        weighted_sum = sum(
            weights[trajectories[index][0]] * sum(trajectories[index][1])
            for index in homogeneous
        )
        estimator = multiplier * weighted_sum
        unweighted_batch_mean = sum(
            sum(trajectories[index][1]) for index in homogeneous
        ) / sum(len(trajectories[index][1]) for index in homogeneous)
        self.assertNotAlmostEqual(estimator, unweighted_batch_mean)

    def test_estimator_uses_actual_last_partial_batch_size(self) -> None:
        full = _balanced_batch_estimator_multiplier(
            trajectory_count=5,
            batch_trajectory_count=2,
            total_eligible_rows=20,
        )
        partial = _balanced_batch_estimator_multiplier(
            trajectory_count=5,
            batch_trajectory_count=1,
            total_eligible_rows=20,
        )
        self.assertEqual(partial, 2.0 * full)

    def test_fixed_plan_sha_and_dropout_admission_fail_before_output(self) -> None:
        player_counts = [
            player_count for player_count in range(4, 11) for _ in range(2)
        ]
        second_plan_sha = "c" * 64
        second_plan_id = (
            f"{V4_FIXED_COLLECTION_PLAN_ID}:sha256={second_plan_sha}"
        )
        actor_config, critic_config = tiny_configs()
        smoke = create_v4_smoke_dataset(
            actor_config,
            critic_config,
            trajectories=2,
            time_steps=2,
            seed=817,
        )
        cases = (
            (
                fixed_dataset(player_counts),
                V4TrainingConfig(amp=False),
                "requires expected_fixed_collection_plan_sha256",
            ),
            (
                fixed_dataset(player_counts),
                V4TrainingConfig(
                    amp=False,
                    expected_fixed_collection_plan_sha256=second_plan_sha,
                ),
                "does not match the corpus",
            ),
            (
                fixed_dataset(
                    player_counts,
                    fixed_collection_plan_ids=(FIXED_PLAN_ID, second_plan_id),
                ),
                fixed_training_config(amp=False),
                "exactly one collection plan",
            ),
            (
                smoke,
                fixed_training_config(amp=False),
                "valid only for fixed-only PPO training",
            ),
            (
                fixed_dataset(player_counts, actor_dropout=0.1),
                fixed_training_config(amp=False),
                "actorConfig.dropout=0.0",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (dataset, config, pattern) in enumerate(cases):
                output = root / f"plan-rejected-{index}"
                with self.assertRaisesRegex(ValueError, pattern):
                    train_v4(dataset, output, config)
                self.assertFalse(output.exists())

        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            V4TrainingConfig(
                expected_fixed_collection_plan_sha256="B" * 64
            )
        parsed = _parser().parse_args(
            [
                "--smoke",
                "--output",
                "unused",
                "--expected-fixed-collection-plan-sha256",
                FIXED_PLAN_SHA256,
            ]
        )
        self.assertEqual(
            parsed.expected_fixed_collection_plan_sha256,
            FIXED_PLAN_SHA256,
        )

    def test_full_dataset_fp32_initial_policy_reproduction_audit(self) -> None:
        player_counts = list(range(4, 11))
        lengths = [2, 4, 3, 5, 2, 4, 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, bundle, actor_sha = export_bound_actor(root, seed=819)
            dataset = fixed_dataset(
                player_counts,
                lengths=lengths,
                force_first_row=True,
                behavior_actor=actor,
                behavior_actor_sha256=actor_sha,
            )
            torch.manual_seed(123456)
            np.random.seed(234567)
            random.seed(345678)
            torch_rng_before = torch.get_rng_state().clone()
            numpy_rng_before = np.random.get_state()
            python_rng_before = random.getstate()
            actor_state_before = {
                name: tensor.detach().clone()
                for name, tensor in actor.state_dict().items()
            }
            audit = _audit_initial_policy_reproduction(
                actor,
                dataset,
                device=torch.device("cpu"),
                batch_size=3,
                num_workers=0,
            )
            self.assertTrue(torch.equal(torch_rng_before, torch.get_rng_state()))
            numpy_rng_after = np.random.get_state()
            self.assertEqual(numpy_rng_before[0], numpy_rng_after[0])
            self.assertTrue(
                np.array_equal(numpy_rng_before[1], numpy_rng_after[1])
            )
            self.assertEqual(numpy_rng_before[2:], numpy_rng_after[2:])
            self.assertEqual(python_rng_before, random.getstate())
            for name, tensor in actor.state_dict().items():
                self.assertTrue(torch.equal(actor_state_before[name], tensor))
            self.assertFalse(actor.training)
            actor.train()
            training_mode_audit = _audit_initial_policy_reproduction(
                actor,
                dataset,
                device=torch.device("cpu"),
                batch_size=3,
                num_workers=0,
            )
            self.assertTrue(actor.training)
            self.assertEqual(
                training_mode_audit["maximumAbsoluteLogProbabilityError"],
                audit["maximumAbsoluteLogProbabilityError"],
            )
            self.assertEqual(audit["ppoEligibleRowCount"], sum(lengths))
            self.assertLessEqual(
                audit["maximumAbsoluteLogProbabilityError"], 2.0e-5
            )
            self.assertLessEqual(
                audit["meanAbsoluteLogProbabilityError"],
                audit["maximumAbsoluteLogProbabilityError"],
            )
            self.assertFalse(audit["actorAutocastEnabled"])
            self.assertEqual(audit["actorForwardDtype"], "torch.float32")
            self.assertEqual(audit["auditBatchSize"], 3)

            dataset.tensors.old_action_log_probs[0, 0] += 1.0e-3
            output = root / "reproduction-rejected"
            with self.assertRaisesRegex(ValueError, "exceeded absolute"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(amp=False, batch_size=3),
                    initialize_actor_bundle=bundle,
                )
            self.assertFalse(output.exists())

    def test_policy_audit_batch_selection_is_device_bound(self) -> None:
        config = V4TrainingConfig(batch_size=3, amp=False)
        self.assertEqual(
            _policy_audit_batch_size(config, torch.device("cpu")), 3
        )
        self.assertEqual(
            _policy_audit_batch_size(config, torch.device("cuda")), 64
        )

    def test_actor_state_sha_is_order_stable_and_x86_byte_compatible(self) -> None:
        torch.manual_seed(820)
        actor_config, _ = tiny_configs()
        actor = V4PublicActor(actor_config)
        state = actor.state_dict()
        actual = _actor_state_sha256(state)
        self.assertEqual(
            actual,
            _actor_state_sha256(dict(reversed(list(state.items())))),
        )
        if sys.byteorder == "little":
            digest = hashlib.sha256()
            digest.update(b"dalmuti-v4-actor-state-v1\0")
            for name in sorted(state):
                tensor = state[name].detach().cpu().contiguous()
                metadata = canonical_json_bytes(
                    {
                        "name": name,
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                    }
                )
                raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                digest.update(len(metadata).to_bytes(8, "big"))
                digest.update(metadata)
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
            self.assertEqual(actual, digest.hexdigest())

    def test_cuda_rng_state_count_is_validated_before_restore(self) -> None:
        states = [torch.arange(8, dtype=torch.uint8) for _ in range(2)]
        with mock.patch("torch.cuda.device_count", return_value=2):
            restored = _validated_checkpoint_cuda_rng_states(
                states, torch.device("cuda")
            )
            assert restored is not None
            self.assertEqual(len(restored), 2)
            with self.assertRaisesRegex(ValueError, "CUDA RNG states"):
                _validated_checkpoint_cuda_rng_states(
                    states[:1], torch.device("cuda")
                )
        self.assertIsNone(
            _validated_checkpoint_cuda_rng_states(None, torch.device("cpu"))
        )
        with self.assertRaisesRegex(ValueError, "CPU fixed"):
            _validated_checkpoint_cuda_rng_states(states, torch.device("cpu"))

    def test_resume_rng_transaction_rolls_back_every_rng_on_exception(self) -> None:
        torch.manual_seed(821)
        np.random.seed(822)
        random.seed(823)
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        cuda_holder = [torch.arange(16, dtype=torch.uint8)]
        cuda_before = [state.clone() for state in cuda_holder]

        def get_cuda_states():
            return [state.clone() for state in cuda_holder]

        def set_cuda_states(states):
            cuda_holder[:] = [state.clone() for state in states]

        def fail_after_mutating_rng(*args, **kwargs):
            torch.manual_seed(999)
            np.random.seed(998)
            random.seed(997)
            set_cuda_states([torch.full((16,), 7, dtype=torch.uint8)])
            raise ValueError("synthetic resume failure")

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torch.cuda.get_rng_state_all", side_effect=get_cuda_states
            ),
            mock.patch(
                "torch.cuda.set_rng_state_all", side_effect=set_cuda_states
            ),
            mock.patch(
                "v4_train._resume_training_impl",
                side_effect=fail_after_mutating_rng,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "synthetic resume"):
                _resume_training(
                    Path("unused.pt"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    V4TrainingConfig(amp=False),
                    torch.device("cpu"),
                )
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(torch.equal(cuda_before[0], cuda_holder[0]))

    def test_latest_checkpoint_path_and_record_are_canonical_before_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, bundle, actor_sha = export_bound_actor(root, seed=824)
            dataset = fixed_dataset(
                list(range(4, 11)),
                behavior_actor=actor,
                behavior_actor_sha256=actor_sha,
            )
            output = root / "latest-path"
            train_v4(
                dataset,
                output,
                fixed_training_config(
                    epochs=1,
                    batch_size=7,
                    amp=False,
                    seed=825,
                ),
                initialize_actor_bundle=bundle,
            )
            latest_path = output / "latest.json"
            original = json.loads(latest_path.read_text(encoding="utf-8"))
            checkpoint = output / original["checkpoint"]
            legacy_shaped = {
                key: value
                for key, value in original.items()
                if key
                in {
                    "format",
                    "version",
                    "completedEpoch",
                    "globalStep",
                    "datasetFingerprint",
                    "lossContractFingerprint",
                    "checkpoint",
                    "sha256",
                }
            }
            cases = (
                (
                    "parent",
                    {**original, "checkpoint": "checkpoints/../checkpoints/epoch-0001.pt"},
                    "path is not canonical",
                    True,
                ),
                (
                    "absolute",
                    {**original, "checkpoint": checkpoint.resolve().as_posix()},
                    "path is not canonical",
                    True,
                ),
                (
                    "backslash",
                    {**original, "checkpoint": "checkpoints\\epoch-0001.pt"},
                    "path is not canonical",
                    True,
                ),
                (
                    "alternate",
                    {**original, "checkpoint": "checkpoints/alternate.pt"},
                    "path is not canonical",
                    True,
                ),
                (
                    "extra-key",
                    {**original, "unexpected": True},
                    "key set",
                    True,
                ),
                (
                    "missing-key",
                    {key: value for key, value in original.items() if key != "globalStep"},
                    "key set",
                    True,
                ),
                (
                    "bool-epoch",
                    {**original, "completedEpoch": True},
                    "fields are non-canonical",
                    True,
                ),
                (
                    "bool-step",
                    {**original, "globalStep": True},
                    "fields are non-canonical",
                    True,
                ),
                (
                    "fixed-as-legacy",
                    original,
                    "variant does not match",
                    False,
                ),
                (
                    "legacy-as-fixed",
                    legacy_shaped,
                    "variant does not match",
                    True,
                ),
                (
                    "balance-sha-type",
                    {**original, "balanceContractFingerprint": 1},
                    "fingerprints are non-canonical",
                    True,
                ),
                (
                    "fixed-plan-uppercase",
                    {
                        **original,
                        "fixedCollectionPlanSha256": original[
                            "fixedCollectionPlanSha256"
                        ].upper(),
                    },
                    "fingerprints are non-canonical",
                    True,
                ),
                (
                    "execution-sha-short",
                    {**original, "fixedPpoExecutionContractFingerprint": "0" * 63},
                    "fingerprints are non-canonical",
                    True,
                ),
                (
                    "initial-audit-not-object",
                    {**original, "initialPolicyReproductionAudit": []},
                    "initial policy reproduction audit must be an object",
                    True,
                ),
                (
                    "initial-audit-fingerprint-type",
                    {**original, "initialPolicyReproductionAuditFingerprint": False},
                    "fingerprints are non-canonical",
                    True,
                ),
                (
                    "rng-contract-bool",
                    {**original, "fixedCheckpointRngContractVersion": True},
                    "RNG contract version is non-canonical",
                    True,
                ),
                (
                    "rng-contract-version",
                    {**original, "fixedCheckpointRngContractVersion": 2},
                    "RNG contract version is non-canonical",
                    True,
                ),
                (
                    "cuda-count-bool",
                    {**original, "cudaRngStateCount": False},
                    "CUDA RNG state count is non-canonical",
                    True,
                ),
                (
                    "cuda-count-negative",
                    {**original, "cudaRngStateCount": -1},
                    "CUDA RNG state count is non-canonical",
                    True,
                ),
                (
                    "post-contract-uppercase",
                    {
                        **original,
                        "postEpochPolicyDriftAuditContractFingerprint": (
                            original[
                                "postEpochPolicyDriftAuditContractFingerprint"
                            ].upper()
                        ),
                    },
                    "fingerprints are non-canonical",
                    True,
                ),
                (
                    "post-audit-not-object",
                    {**original, "postEpochPolicyDriftAudit": []},
                    "post-epoch policy drift audit must be an object",
                    True,
                ),
                (
                    "post-audit-fingerprint-type",
                    {**original, "postEpochPolicyDriftAuditFingerprint": None},
                    "fingerprints are non-canonical",
                    True,
                ),
            )
            for label, tampered, pattern, require_sidecar in cases:
                with self.subTest(label=label):
                    latest_path.write_text(
                        json.dumps(
                            tampered, sort_keys=True, separators=(",", ":")
                        ),
                        encoding="utf-8",
                    )
                    with (
                        mock.patch("v4_train.sha256_file") as checksum,
                        mock.patch(
                            "v4_train._verify_checkpoint_sha256_sidecar"
                        ) as sidecar,
                        mock.patch("v4_train._torch_load") as torch_load,
                    ):
                        with self.assertRaisesRegex(ValueError, pattern):
                            _resolve_resume(
                                output,
                                "latest",
                                require_checkpoint_sidecar=require_sidecar,
                            )
                        checksum.assert_not_called()
                        sidecar.assert_not_called()
                        torch_load.assert_not_called()
            latest_path.write_text(
                json.dumps(original, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            alias_cases = (
                ("checkpoints-junction", (output / "checkpoints").resolve()),
                ("epoch-symlink", checkpoint.resolve()),
            )
            for label, aliased_path in alias_cases:
                with self.subTest(label=label):
                    with (
                        mock.patch(
                            "v4_train._path_is_link_or_reparse_alias",
                            side_effect=lambda path, target=aliased_path: (
                                path == target
                            ),
                        ),
                        mock.patch("v4_train.sha256_file") as checksum,
                        mock.patch(
                            "v4_train._verify_checkpoint_sha256_sidecar"
                        ) as sidecar,
                        mock.patch("v4_train._torch_load") as torch_load,
                    ):
                        with self.assertRaisesRegex(ValueError, "path alias"):
                            _resolve_resume(
                                output,
                                "latest",
                                require_checkpoint_sidecar=True,
                            )
                        checksum.assert_not_called()
                        sidecar.assert_not_called()
                        torch_load.assert_not_called()
            latest_path.write_text(
                json.dumps(original, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            self.assertEqual(
                _resolve_resume(
                    output,
                    "latest",
                    require_checkpoint_sidecar=True,
                ),
                checkpoint.resolve(),
            )

    def test_missing_player_count_qboost_and_mixed_sources_fail_before_output(self) -> None:
        valid_counts = [player_count for player_count in range(4, 11) for _ in range(2)]
        cases = (
            (
                fixed_dataset([player_count for player_count in range(4, 10) for _ in range(2)]),
                fixed_training_config(amp=False),
                "missing eligible rows.*p10",
            ),
            (
                fixed_dataset(valid_counts),
                fixed_training_config(
                    bc_weight=1.0,
                    ppo_weight=1.0,
                    critic_weight=1.0,
                    q_boost_coefficient=0.1,
                    amp=False,
                ),
                "q_boost_coefficient=0",
            ),
            (
                fixed_dataset(
                    valid_counts,
                    sources=(FIXED_SOURCE, LEGACY_SOURCE),
                    requires_balance=False,
                ),
                fixed_training_config(amp=False),
                "mixed fixed-match and legacy",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (dataset, config, pattern) in enumerate(cases):
                output = root / f"rejected-{index}"
                with self.assertRaisesRegex(ValueError, pattern):
                    train_v4(dataset, output, config)
                self.assertFalse(output.exists())

    def test_fixed_requirement_tamper_fails_before_output(self) -> None:
        player_counts = [
            player_count for player_count in range(4, 11) for _ in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = fixed_dataset(player_counts)
            eligibility = dataset.loss_eligibility
            assert eligibility is not None
            object.__setattr__(
                eligibility, "requires_player_count_balanced_loss", False
            )
            output = Path(directory) / "rejected-balance"
            with self.assertRaisesRegex(ValueError, "balanced-loss requirement"):
                train_v4(dataset, output, fixed_training_config(amp=False))
            self.assertFalse(output.exists())

            qboost_dataset = fixed_dataset(player_counts)
            qboost_eligibility = qboost_dataset.loss_eligibility
            assert qboost_eligibility is not None
            object.__setattr__(
                qboost_eligibility, "requires_qboost_coefficient_zero", False
            )
            qboost_output = Path(directory) / "rejected-qboost-binding"
            with self.assertRaisesRegex(ValueError, "q-boost prohibition binding"):
                train_v4(
                    qboost_dataset,
                    qboost_output,
                    fixed_training_config(amp=False),
                )
            self.assertFalse(qboost_output.exists())

            reward_dataset = fixed_dataset(player_counts)
            reward_eligibility = reward_dataset.loss_eligibility
            assert reward_eligibility is not None
            object.__setattr__(reward_eligibility, "ppo_reward_contracts", ())
            reward_output = Path(directory) / "rejected-reward-binding"
            with self.assertRaisesRegex(ValueError, "canonical reward and behavior"):
                train_v4(
                    reward_dataset,
                    reward_output,
                    fixed_training_config(amp=False),
                )
            self.assertFalse(reward_output.exists())

    def test_legacy_training_keeps_none_weight_path_and_contract_shape(self) -> None:
        actor_config, critic_config = tiny_configs()
        dataset = create_v4_smoke_dataset(
            actor_config,
            critic_config,
            trajectories=2,
            time_steps=2,
            seed=823,
        )
        contract = _resolve_training_contract(
            dataset,
            V4TrainingConfig(amp=False),
            resume=None,
            initial_actor_sha256=None,
        )
        self.assertEqual(contract["version"], 1)
        self.assertNotIn("playerCountBalancedLoss", contract)
        self.assertNotIn("balanceContractFingerprint", contract)
        observed_weights: list[torch.Tensor | None] = []

        def recording_bc(*args, weights=None, **kwargs):
            observed_weights.append(weights)
            return real_bc_loss(*args, weights=weights, **kwargs)

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "v4_train.masked_behavior_cloning_loss",
            side_effect=recording_bc,
        ):
            output = Path(directory) / "legacy"
            train_v4(
                dataset,
                output,
                V4TrainingConfig(
                    epochs=1,
                    batch_size=1,
                    amp=False,
                    seed=827,
                ),
            )
            manifest = json.loads(
                (output / "run-manifest.json").read_text(encoding="utf-8")
            )
            latest = json.loads(
                (output / "latest.json").read_text(encoding="utf-8")
            )
            try:
                checkpoint = torch.load(
                    output / "checkpoints" / "epoch-0001.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    output / "checkpoints" / "epoch-0001.pt",
                    map_location="cpu",
                )
            self.assertFalse(
                (output / "checkpoints" / "epoch-0001.pt.sha256").exists()
            )
            self.assertEqual(
                _resolve_resume(
                    output,
                    "latest",
                    require_checkpoint_sidecar=False,
                ),
                (output / "checkpoints" / "epoch-0001.pt").resolve(),
            )
            legacy_resumed = train_v4(
                dataset,
                output,
                V4TrainingConfig(
                    epochs=2,
                    batch_size=1,
                    amp=False,
                    seed=827,
                ),
                resume=output / "checkpoints" / "epoch-0001.pt",
            )
            self.assertEqual(legacy_resumed["completedEpochs"], 2)
            self.assertFalse(
                (output / "checkpoints" / "epoch-0002.pt.sha256").exists()
            )
        self.assertTrue(observed_weights)
        self.assertTrue(all(value is None for value in observed_weights))
        self.assertNotIn("playerCountBalancedLoss", manifest["trainingContract"])
        self.assertNotIn("balanceContractFingerprint", manifest["trainingContract"])
        self.assertNotIn("balanceContractFingerprint", latest)
        self.assertNotIn("balanceContractFingerprint", checkpoint)
        self.assertNotIn("fixedCollectionPlanSha256", manifest["trainingContract"])
        self.assertNotIn("fixedCollectionPlanSha256", latest)
        self.assertNotIn("fixedCollectionPlanSha256", checkpoint)
        self.assertNotIn(
            "initialPolicyReproductionAudit", manifest["trainingContract"]
        )
        self.assertNotIn("initialPolicyReproductionAudit", latest)
        self.assertNotIn("initialPolicyReproductionAudit", checkpoint)
        self.assertNotIn("postEpochPolicyDriftAudit", manifest["trainingContract"])
        self.assertNotIn("postEpochPolicyDriftAudit", latest)
        self.assertNotIn("postEpochPolicyDriftAudit", checkpoint)

    def test_epoch_diagnostic_is_one_global_weighted_reduction(self) -> None:
        player_counts = [
            player_count for player_count in range(4, 11) for _ in range(2)
        ] + [4]
        lengths = [1 + index % 3 for index in range(len(player_counts))]

        def weight_value_bc(logits, legal, actions, *, weights=None):
            assert weights is not None
            # Row diagnostic x_i=w_i gives a hand-checkable global result
            # sum(C_p*w_p^2)/sum(C_p*w_p).  Keep a zero Actor dependency so
            # backward and optimizer behavior remain exercised.
            return logits.sum() * 0.0 + (weights * weights).sum() / weights.sum()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, bundle, actor_sha = export_bound_actor(root, seed=828)
            dataset = fixed_dataset(
                player_counts,
                lengths=lengths,
                behavior_actor=actor,
                behavior_actor_sha256=actor_sha,
            )
            output = Path(directory) / "balanced"
            with mock.patch(
                "v4_train.masked_behavior_cloning_loss",
                side_effect=weight_value_bc,
            ):
                result = train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=1,
                        batch_size=4,
                        amp=False,
                        seed=829,
                    ),
                    initialize_actor_bundle=bundle,
                )
            latest = json.loads(
                (output / "latest.json").read_text(encoding="utf-8")
            )
            try:
                checkpoint = torch.load(
                    output / "checkpoints" / "epoch-0001.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    output / "checkpoints" / "epoch-0001.pt",
                    map_location="cpu",
                )
            candidate_actor, _ = load_v4_actor_checkpoint(
                output / "candidate" / "actor.pt"
            )
            candidate_actor_state_sha256 = _actor_state_sha256(
                candidate_actor.state_dict()
            )
            checkpoint_sidecar = (
                output / "checkpoints" / "epoch-0001.pt.sha256"
            ).read_text(encoding="ascii")
        training_contract = result["trainingContract"]
        balance = training_contract["playerCountBalancedLoss"]
        self.assertEqual(training_contract["version"], 3)
        self.assertEqual(
            training_contract["fixedCollectionPlanIds"], [FIXED_PLAN_ID]
        )
        self.assertEqual(
            training_contract["fixedCollectionPlanSha256"], FIXED_PLAN_SHA256
        )
        self.assertEqual(balance["fixedCollectionPlanId"], FIXED_PLAN_ID)
        self.assertEqual(
            balance["fixedCollectionPlanSha256"], FIXED_PLAN_SHA256
        )
        self.assertEqual(balance["actorDropout"], 0.0)
        self.assertFalse(
            training_contract["fixedPpoExecutionContract"][
                "actorAutocastEnabled"
            ]
        )
        self.assertEqual(
            training_contract["fixedPpoExecutionContract"]["policyNumerics"],
            canonical_v4_policy_numerics_contract(),
        )
        self.assertEqual(
            balance["fixedPpoPolicyNumerics"],
            canonical_v4_policy_numerics_contract(),
        )
        post_contract = training_contract["postEpochPolicyDriftAuditContract"]
        self.assertEqual(post_contract["auditBatchSize"], 4)
        self.assertEqual(
            post_contract["auditBatchSelectionRule"],
            "cuda uses fixed 64; cpu uses trainingConfig.batch_size",
        )
        self.assertTrue(post_contract["auditMustNotMutateOptimizationRngOrWeights"])
        self.assertEqual(
            post_contract["actorModeRestoration"],
            "restore exact pre-audit actor.training boolean in finally",
        )
        self.assertEqual(post_contract["checkpointRngContractVersion"], 1)
        self.assertEqual(post_contract["actorStateSha256ContractVersion"], 1)
        audit = training_contract["initialPolicyReproductionAudit"]
        self.assertTrue(audit["passed"])
        self.assertGreater(audit["ppoEligibleRowCount"], 0)
        self.assertGreater(audit["effectiveNonforcedPpoRowCount"], 0)
        self.assertGreater(audit["nonforcedBalancedEntropy"], 0.0)
        self.assertIn("maximumAbsoluteLogProbabilityError", audit)
        self.assertIn("meanAbsoluteLogProbabilityError", audit)
        for record in (latest, checkpoint):
            self.assertEqual(
                record["fixedCollectionPlanSha256"], FIXED_PLAN_SHA256
            )
            self.assertEqual(
                record["initialPolicyReproductionAudit"], audit
            )
            self.assertEqual(
                record["initialPolicyReproductionAuditFingerprint"],
                training_contract[
                    "initialPolicyReproductionAuditFingerprint"
                ],
            )
        self.assertIsNone(checkpoint["cudaRngStates"])
        self.assertEqual(checkpoint["fixedCheckpointRngContractVersion"], 1)
        self.assertEqual(latest["cudaRngStateCount"], 0)
        self.assertEqual(latest["fixedCheckpointRngContractVersion"], 1)
        self.assertEqual(
            checkpoint_sidecar,
            f'{latest["sha256"]}  epoch-0001.pt\n',
        )
        self.assertEqual(training_contract["ppoRewardContracts"], [FIXED_REWARD_ID])
        self.assertEqual(
            training_contract["ppoBehaviorPolicyContracts"],
            [FIXED_BEHAVIOR_ID],
        )
        self.assertEqual(balance["ppoRewardContract"], FIXED_REWARD_ID)
        self.assertEqual(
            balance["ppoBehaviorPolicyContract"], FIXED_BEHAVIOR_ID
        )
        counts = balance["eligibleRowCountsByLossAndPlayerCount"][
            "behaviorCloning"
        ]
        weights = balance["runtimeFloat32WeightsByLossAndPlayerCount"][
            "behaviorCloning"
        ]
        expected_numerator = sum(
            counts[player_count] * weights[player_count] ** 2
            for player_count in counts
        )
        expected_denominator = sum(
            counts[player_count] * weights[player_count]
            for player_count in counts
        )
        metric = result["metrics"][0]
        post_audit = metric["postEpochPolicyDriftAudit"]
        post_fingerprint = metric["postEpochPolicyDriftAuditFingerprint"]
        self.assertEqual(metric["approxKl"], post_audit["approxKl"])
        self.assertEqual(metric["clipFraction"], post_audit["clipFraction"])
        self.assertEqual(metric["entropy"], post_audit["entropy"])
        self.assertEqual(post_audit["auditBatchSize"], 4)
        self.assertIn("approxKl", metric["optimizationPassDiagnostics"])
        self.assertIn("clipFraction", metric["optimizationPassDiagnostics"])
        self.assertIn("entropy", metric["optimizationPassDiagnostics"])
        self.assertEqual(
            post_audit["initialBehaviorEntropy"],
            audit["nonforcedBalancedEntropy"],
        )
        self.assertAlmostEqual(
            post_audit["entropyRetentionRatio"],
            post_audit["entropy"] / post_audit["initialBehaviorEntropy"],
            places=12,
        )
        self.assertEqual(
            post_audit["entropyCollapseExceeds30Percent"],
            post_audit["entropyCollapseFraction"] > 0.30,
        )
        ppo_masses = balance["runtimeWeightMassByLossAndPlayerCount"]["ppo"]
        total_ppo_mass = sum(ppo_masses.values())
        for name in (
            "approxKl",
            "clipFraction",
            "entropy",
            "meanLogRatio",
            "meanAbsoluteLogRatio",
        ):
            reconstructed = sum(
                post_audit["perPlayerCount"][key][name] * ppo_masses[key]
                for key in ppo_masses
            ) / total_ppo_mass
            self.assertAlmostEqual(post_audit[name], reconstructed, places=12)
        for record in (latest, checkpoint):
            self.assertEqual(record["postEpochPolicyDriftAudit"], post_audit)
            self.assertEqual(
                record["postEpochPolicyDriftAuditFingerprint"],
                post_fingerprint,
            )
            self.assertEqual(
                record["postEpochPolicyDriftAuditContractFingerprint"],
                training_contract[
                    "postEpochPolicyDriftAuditContractFingerprint"
                ],
            )
        self.assertEqual(result["finalPostEpochPolicyDriftAudit"], post_audit)
        self.assertEqual(
            result["finalPostEpochPolicyDriftAuditFingerprint"],
            post_fingerprint,
        )
        candidate_metadata = result["candidate"]["metadata"]
        self.assertEqual(
            candidate_metadata["finalPostEpochPolicyDriftAudit"], post_audit
        )
        self.assertEqual(
            candidate_metadata["finalPostEpochPolicyDriftAuditFingerprint"],
            post_fingerprint,
        )
        self.assertEqual(
            post_audit["actorStateSha256"],
            candidate_actor_state_sha256,
        )
        self.assertAlmostEqual(
            metric["behaviorCloningLoss"],
            expected_numerator / expected_denominator,
            places=6,
        )
        self.assertAlmostEqual(metric["loss"], metric["behaviorCloningLoss"])
        self.assertEqual(
            metric["balanceContractFingerprint"],
            training_contract["balanceContractFingerprint"],
        )
        self.assertEqual(
            metric["balancedEligibleRowsSeenByLossAndPlayerCount"],
            balance["eligibleRowCountsByLossAndPlayerCount"],
        )
        self.assertEqual(
            metric["balancedWeightMassSeenByLossAndPlayerCount"],
            balance["runtimeWeightMassByLossAndPlayerCount"],
        )

    def test_post_epoch_audit_excludes_forced_rows_and_keeps_p_equal_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, bundle, actor_sha = export_bound_actor(root, seed=834)
            dataset = fixed_dataset(
                list(range(4, 11)),
                lengths=[2] * 7,
                force_first_row=True,
                behavior_actor=actor,
                behavior_actor_sha256=actor_sha,
            )
            result = train_v4(
                dataset,
                root / "forced-evidence",
                fixed_training_config(
                    epochs=1,
                    batch_size=7,
                    amp=False,
                    seed=835,
                ),
                initialize_actor_bundle=bundle,
            )
        initial = result["trainingContract"]["initialPolicyReproductionAudit"]
        post = result["metrics"][0]["postEpochPolicyDriftAudit"]
        self.assertEqual(initial["ppoEligibleRowCount"], 14)
        self.assertEqual(initial["effectiveNonforcedPpoRowCount"], 7)
        self.assertEqual(initial["forcedSingletonPpoRowCount"], 7)
        self.assertEqual(post["ppoEligibleRowCount"], 14)
        self.assertEqual(post["effectiveNonforcedPpoRowCount"], 7)
        self.assertEqual(post["forcedSingletonPpoRowCount"], 7)
        self.assertLessEqual(
            post["forcedMaximumAbsoluteLogRatio"], 2.0e-5
        )
        for player_count in range(4, 11):
            key = str(player_count)
            self.assertEqual(post["nonforcedRowsByPlayerCount"][key], 1)
            self.assertEqual(post["forcedSingletonRowsByPlayerCount"][key], 1)
            self.assertEqual(post["perPlayerCount"][key]["count"], 1)
        self.assertAlmostEqual(
            post["maximumAbsoluteLogRatio"],
            max(
                record["maximumAbsoluteLogRatio"]
                for record in post["perPlayerCount"].values()
            ),
            places=12,
        )

    def test_resume_is_bound_to_balance_contract_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, bundle, actor_sha = export_bound_actor(root, seed=838)
            dataset = fixed_dataset(
                list(range(4, 11)),
                force_first_row=True,
                behavior_actor=actor,
                behavior_actor_sha256=actor_sha,
            )
            output = root / "balanced-resume"
            result = train_v4(
                dataset,
                output,
                fixed_training_config(
                    epochs=1,
                    batch_size=7,
                    amp=False,
                    seed=839,
                ),
                initialize_actor_bundle=bundle,
            )
            checkpoint_path = output / "checkpoints" / "epoch-0001.pt"
            epoch_one_metric_bytes = (
                output / "metrics" / "epoch-0001.json"
            ).read_bytes()
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(
                checkpoint["balanceContractFingerprint"],
                result["trainingContract"]["balanceContractFingerprint"],
            )
            original_checkpoint_sha256 = sha256_file(checkpoint_path)
            stale_checksum_mutations = {
                "critic": ("criticState", {}),
                "optimizer": ("actorOptimizerState", {}),
                "scaler": ("scalerState", {"tampered": True}),
                "rng": (
                    "torchRngState",
                    checkpoint["torchRngState"].clone(),
                ),
            }
            stale_checksum_mutations["rng"][1][0] ^= 1
            for label, (field, replacement) in stale_checksum_mutations.items():
                checksum_tampered = dict(checkpoint)
                checksum_tampered[field] = replacement
                tampered_checksum_path = (
                    output / "checkpoints" / f"tampered-checksum-{label}.pt"
                )
                save_checkpoint_with_sidecar(
                    tampered_checksum_path,
                    checksum_tampered,
                    sidecar_checksum=original_checkpoint_sha256,
                )
                with self.assertRaisesRegex(ValueError, "sidecar checksum"):
                    train_v4(
                        dataset,
                        output,
                        fixed_training_config(
                            epochs=2,
                            batch_size=7,
                            amp=False,
                            seed=839,
                        ),
                        resume=tampered_checksum_path,
                    )
            original_balance_fingerprint = checkpoint[
                "balanceContractFingerprint"
            ]
            checkpoint["balanceContractFingerprint"] = "0" * 64
            tampered = output / "checkpoints" / "tampered.pt"
            save_checkpoint_with_sidecar(tampered, checkpoint)
            with self.assertRaisesRegex(ValueError, "balance contract"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered,
                )

            checkpoint["balanceContractFingerprint"] = (
                original_balance_fingerprint
            )
            checkpoint["fixedCollectionPlanSha256"] = "0" * 64
            tampered_plan = output / "checkpoints" / "tampered-plan.pt"
            save_checkpoint_with_sidecar(tampered_plan, checkpoint)
            with self.assertRaisesRegex(ValueError, "fixed collection plan"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_plan,
                )

            checkpoint["fixedCollectionPlanSha256"] = FIXED_PLAN_SHA256
            original_actor_state = checkpoint["actorState"]
            tampered_actor_state = dict(original_actor_state)
            first_actor_state_name = next(
                name
                for name, tensor in tampered_actor_state.items()
                if tensor.is_floating_point()
            )
            tampered_actor_tensor = tampered_actor_state[
                first_actor_state_name
            ].clone()
            tampered_actor_tensor.reshape(-1)[0] += 1.0e-3
            tampered_actor_state[first_actor_state_name] = tampered_actor_tensor
            checkpoint["actorState"] = tampered_actor_state
            tampered_actor_checkpoint = (
                output / "checkpoints" / "tampered-actor-state.pt"
            )
            save_checkpoint_with_sidecar(tampered_actor_checkpoint, checkpoint)
            with self.assertRaisesRegex(ValueError, "Actor state SHA-256"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_actor_checkpoint,
                )
            checkpoint["actorState"] = original_actor_state

            checkpoint.pop("cudaRngStates")
            tampered_rng_contract = (
                output / "checkpoints" / "tampered-rng-contract.pt"
            )
            save_checkpoint_with_sidecar(tampered_rng_contract, checkpoint)
            with self.assertRaisesRegex(ValueError, "RNG contract"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_rng_contract,
                )
            checkpoint["cudaRngStates"] = None

            original_training_config = checkpoint["trainingConfig"]
            deleted_key_training_config = dict(original_training_config)
            deleted_key_training_config.pop("actor_learning_rate")
            checkpoint["trainingConfig"] = deleted_key_training_config
            tampered_training_keys = (
                output / "checkpoints" / "tampered-training-keys.pt"
            )
            save_checkpoint_with_sidecar(tampered_training_keys, checkpoint)
            with self.assertRaisesRegex(ValueError, "key set"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_training_keys,
                )
            checkpoint["trainingConfig"] = original_training_config
            for label, field, replacement, pattern in (
                ("float-int", "bc_weight", 1, "training setting"),
                ("int-bool", "num_workers", False, "training setting"),
                ("epochs-float", "epochs", 1.0, "epochs type"),
            ):
                type_tampered_training_config = dict(original_training_config)
                type_tampered_training_config[field] = replacement
                checkpoint["trainingConfig"] = type_tampered_training_config
                tampered_training_type = (
                    output
                    / "checkpoints"
                    / f"tampered-training-type-{label}.pt"
                )
                save_checkpoint_with_sidecar(
                    tampered_training_type, checkpoint
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    train_v4(
                        dataset,
                        output,
                        fixed_training_config(
                            epochs=2,
                            batch_size=7,
                            amp=False,
                            seed=839,
                        ),
                        resume=tampered_training_type,
                    )
            checkpoint["trainingConfig"] = original_training_config

            original_initial_audit = checkpoint[
                "initialPolicyReproductionAudit"
            ]
            redistributed_initial_forced = dict(
                original_initial_audit["forcedSingletonRowsByPlayerCount"]
            )
            redistributed_initial_forced["4"] -= 1
            redistributed_initial_forced["5"] += 1
            checkpoint["initialPolicyReproductionAudit"] = {
                **original_initial_audit,
                "forcedSingletonRowsByPlayerCount": (
                    redistributed_initial_forced
                ),
            }
            tampered_initial_forced = (
                output / "checkpoints" / "tampered-initial-forced.pt"
            )
            save_checkpoint_with_sidecar(tampered_initial_forced, checkpoint)
            with self.assertRaisesRegex(ValueError, "initial policy.*failed"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_initial_forced,
                )
            checkpoint["initialPolicyReproductionAudit"] = original_initial_audit

            original_post_forced_audit = checkpoint[
                "postEpochPolicyDriftAudit"
            ]
            redistributed_post_forced = dict(
                original_post_forced_audit[
                    "forcedSingletonRowsByPlayerCount"
                ]
            )
            redistributed_post_forced["4"] -= 1
            redistributed_post_forced["5"] += 1
            checkpoint["postEpochPolicyDriftAudit"] = {
                **original_post_forced_audit,
                "forcedSingletonRowsByPlayerCount": redistributed_post_forced,
            }
            tampered_post_forced = (
                output / "checkpoints" / "tampered-post-forced.pt"
            )
            save_checkpoint_with_sidecar(tampered_post_forced, checkpoint)
            with self.assertRaisesRegex(ValueError, "forced-row evidence"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_post_forced,
                )
            checkpoint["postEpochPolicyDriftAudit"] = original_post_forced_audit

            original_post_contract_fingerprint = checkpoint[
                "postEpochPolicyDriftAuditContractFingerprint"
            ]
            checkpoint["postEpochPolicyDriftAuditContractFingerprint"] = "0" * 64
            tampered_post_contract = (
                output / "checkpoints" / "tampered-post-contract.pt"
            )
            save_checkpoint_with_sidecar(tampered_post_contract, checkpoint)
            with self.assertRaisesRegex(ValueError, "post-epoch.*contract"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_post_contract,
                )
            checkpoint["postEpochPolicyDriftAuditContractFingerprint"] = (
                original_post_contract_fingerprint
            )
            original_post_fingerprint = checkpoint[
                "postEpochPolicyDriftAuditFingerprint"
            ]
            checkpoint["postEpochPolicyDriftAuditFingerprint"] = "0" * 64
            tampered_post_fingerprint = (
                output / "checkpoints" / "tampered-post-fingerprint.pt"
            )
            save_checkpoint_with_sidecar(tampered_post_fingerprint, checkpoint)
            with self.assertRaisesRegex(ValueError, "post-epoch.*fingerprint"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_post_fingerprint,
                )

            checkpoint["postEpochPolicyDriftAuditFingerprint"] = (
                original_post_fingerprint
            )
            original_post_audit = checkpoint["postEpochPolicyDriftAudit"]
            checkpoint["postEpochPolicyDriftAudit"] = {
                **original_post_audit,
                "approxKl": original_post_audit["approxKl"] + 0.1,
            }
            tampered_post_value = (
                output / "checkpoints" / "tampered-post-value.pt"
            )
            save_checkpoint_with_sidecar(tampered_post_value, checkpoint)
            with self.assertRaisesRegex(ValueError, "p-balanced approxKl"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_post_value,
                )

            checkpoint["postEpochPolicyDriftAudit"] = original_post_audit
            checkpoint["initialPolicyReproductionAuditFingerprint"] = "0" * 64
            tampered_audit = output / "checkpoints" / "tampered-audit.pt"
            save_checkpoint_with_sidecar(tampered_audit, checkpoint)
            with self.assertRaisesRegex(ValueError, "audit fingerprint"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=2,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume=tampered_audit,
                )

            resumed = train_v4(
                dataset,
                output,
                fixed_training_config(
                    epochs=2,
                    batch_size=7,
                    amp=False,
                    seed=839,
                ),
                resume="latest",
            )
            self.assertEqual(resumed["completedEpochs"], 2)
            self.assertEqual(
                (output / "metrics" / "epoch-0001.json").read_bytes(),
                epoch_one_metric_bytes,
            )
            latest_path = output / "latest.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            latest["cudaRngStateCount"] = 1
            latest_path.write_text(
                json.dumps(latest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "latest fixed checkpoint RNG"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=3,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume="latest",
                )
            latest["cudaRngStateCount"] = 0
            original_latest_post_fingerprint = latest[
                "postEpochPolicyDriftAuditFingerprint"
            ]
            latest["postEpochPolicyDriftAuditFingerprint"] = "0" * 64
            latest_path.write_text(
                json.dumps(latest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "latest post-epoch.*fingerprint"
            ):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=3,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume="latest",
                )
            latest["postEpochPolicyDriftAuditFingerprint"] = (
                original_latest_post_fingerprint
            )
            latest["initialPolicyReproductionAuditFingerprint"] = "0" * 64
            latest_path.write_text(
                json.dumps(latest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "latest initial policy"):
                train_v4(
                    dataset,
                    output,
                    fixed_training_config(
                        epochs=3,
                        batch_size=7,
                        amp=False,
                        seed=839,
                    ),
                    resume="latest",
                )

    def test_ppo_and_critic_paths_use_their_bound_global_weight_masses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, bundle, actor_sha = export_bound_actor(root, seed=852)
            dataset = fixed_dataset(
                list(range(4, 11)),
                behavior_actor_sha256=actor_sha,
                behavior_actor=actor,
            )
            result = train_v4(
                dataset,
                root / "ppo-critic",
                fixed_training_config(
                    epochs=1,
                    batch_size=3,
                    bc_weight=0.0,
                    ppo_weight=1.0,
                    critic_weight=1.0,
                    q_boost_coefficient=0.0,
                    amp=False,
                    seed=853,
                ),
                initialize_actor_bundle=bundle,
            )
        balance = result["trainingContract"]["playerCountBalancedLoss"]
        metric = result["metrics"][0]
        self.assertAlmostEqual(
            metric["balancedDiagnosticWeightMassByLoss"]["ppo"],
            balance["runtimeTotalWeightMassByLoss"]["ppo"],
            places=5,
        )
        self.assertAlmostEqual(
            metric["balancedDiagnosticWeightMassByLoss"]["critic"],
            balance["runtimeTotalWeightMassByLoss"]["critic"],
            places=5,
        )
        self.assertTrue(torch.isfinite(torch.tensor(metric["policyLoss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metric["criticLoss"])))


if __name__ == "__main__":
    unittest.main()
