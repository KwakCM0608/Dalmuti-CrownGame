from __future__ import annotations

from dataclasses import fields
from itertools import combinations
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
from v4_export import export_v4_actor_bundle
from v4_model import V4ActorConfig, V4CriticConfig, V4PublicActor
from v4_objectives import masked_behavior_cloning_loss as real_bc_loss
from v4_train import (
    V4TrainingConfig,
    _audit_initial_policy_reproduction,
    _balanced_batch_estimator_multiplier,
    _parser,
    _player_count_balance_contract,
    _resolve_training_contract,
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
            audit = _audit_initial_policy_reproduction(
                actor,
                dataset,
                device=torch.device("cpu"),
                batch_size=3,
                num_workers=0,
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
        audit = training_contract["initialPolicyReproductionAudit"]
        self.assertTrue(audit["passed"])
        self.assertGreater(audit["ppoEligibleRowCount"], 0)
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

    def test_resume_is_bound_to_balance_contract_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, bundle, actor_sha = export_bound_actor(root, seed=838)
            dataset = fixed_dataset(
                list(range(4, 11)),
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
            original_balance_fingerprint = checkpoint[
                "balanceContractFingerprint"
            ]
            checkpoint["balanceContractFingerprint"] = "0" * 64
            tampered = output / "checkpoints" / "tampered.pt"
            torch.save(checkpoint, tampered)
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
            torch.save(checkpoint, tampered_plan)
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
            checkpoint["initialPolicyReproductionAuditFingerprint"] = "0" * 64
            tampered_audit = output / "checkpoints" / "tampered-audit.pt"
            torch.save(checkpoint, tampered_audit)
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
            latest_path = output / "latest.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
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
