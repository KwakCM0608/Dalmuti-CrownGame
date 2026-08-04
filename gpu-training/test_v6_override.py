from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import unittest

import numpy as np


GPU_TRAINING = Path(__file__).resolve().parent
if str(GPU_TRAINING) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINING))

import v6_override as override


class SafeOverrideTests(unittest.TestCase):
    def test_normal_relative_delta_is_bit_exact_zero(self) -> None:
        decision = override.choose_safe_override(
            action_ids=np.array([7, 2, 90], dtype=np.int64),
            legal_mask=np.array([True, True, True], dtype=np.bool_),
            head_scores=np.array(
                [[1.1, -2.0, 4.0], [0.3, 0.4, 0.5], [0.1, 0.1, 0.1]],
                dtype=np.float64,
            ),
            normal_action=7,
            beta=1.0,
            threshold=10.0,
        )
        normal_position = int(np.flatnonzero(decision.action_ids == 7)[0])
        self.assertEqual(decision.delta_mean[normal_position], 0.0)
        self.assertEqual(decision.delta_std[normal_position], 0.0)
        self.assertEqual(decision.delta_lcb[normal_position], 0.0)
        self.assertEqual(decision.action_id, 7)

    def test_illegal_action_is_excluded_even_with_huge_score(self) -> None:
        decision = override.choose_safe_override(
            action_ids=np.array([1, 2, 3], dtype=np.int64),
            legal_mask=np.array([True, False, True], dtype=np.bool_),
            head_scores=np.array(
                [[0.0, 0.0, 0.0], [1.0e6, 1.0e6, 1.0e6], [2.0, 2.0, 2.0]],
                dtype=np.float32,
            ),
            normal_action=1,
            threshold=0.1,
        )
        self.assertEqual(decision.action_id, 3)
        self.assertTrue(decision.overridden)

    def test_uncertain_and_negative_alternatives_fall_back_to_normal(self) -> None:
        uncertain = override.choose_safe_override(
            action_ids=np.array([1, 2], dtype=np.int64),
            legal_mask=np.ones(2, dtype=np.bool_),
            head_scores=np.array([[0.0, 0.0, 0.0], [3.0, -3.0, 3.0]]),
            normal_action=1,
            beta=2.0,
        )
        negative = override.choose_safe_override(
            action_ids=np.array([1, 2], dtype=np.int64),
            legal_mask=np.ones(2, dtype=np.bool_),
            head_scores=np.array([[0.0, 0.0, 0.0], [-0.1, -0.2, -0.3]]),
            normal_action=1,
        )
        self.assertEqual(uncertain.action_id, 1)
        self.assertEqual(negative.action_id, 1)
        self.assertFalse(uncertain.overridden)
        self.assertFalse(negative.overridden)

    def test_only_strong_legal_alternative_overrides(self) -> None:
        decision = override.choose_safe_override(
            action_ids=np.array([11, 22, 33], dtype=np.int64),
            legal_mask=np.array([True, True, True], dtype=np.bool_),
            head_scores=np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.1, 0.9], [0.2, -0.2, 0.2]],
                dtype=np.float64,
            ),
            normal_action=11,
            beta=1.0,
            threshold=0.5,
        )
        self.assertEqual(decision.action_id, 22)
        self.assertEqual(decision.best_alternative_id, 22)
        self.assertTrue(decision.overridden)

    def test_equal_lcb_tie_uses_lowest_action_id(self) -> None:
        decision = override.choose_safe_override(
            action_ids=np.array([50, 40, 7], dtype=np.int64),
            legal_mask=np.ones(3, dtype=np.bool_),
            head_scores=np.array(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
            ),
            normal_action=50,
            threshold=0.1,
        )
        self.assertEqual(decision.action_id, 7)

    def test_positive_infinity_threshold_is_exact_normal_control(self) -> None:
        decision = override.choose_safe_override(
            action_ids=np.array([1, 2], dtype=np.int64),
            legal_mask=np.ones(2, dtype=np.bool_),
            head_scores=np.array([[0.0, 0.0, 0.0], [1.0e9, 1.0e9, 1.0e9]]),
            normal_action=1,
            threshold=math.inf,
        )
        self.assertEqual(decision.action_id, 1)
        self.assertFalse(decision.overridden)

    def test_nan_and_negative_infinity_thresholds_are_rejected(self) -> None:
        arguments = {
            "action_ids": np.array([1, 2], dtype=np.int64),
            "legal_mask": np.ones(2, dtype=np.bool_),
            "head_scores": np.zeros((2, 3), dtype=np.float32),
            "normal_action": 1,
        }
        for threshold in (math.nan, -math.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "positive infinity"):
                    override.choose_safe_override(
                        **arguments,
                        threshold=threshold,
                    )

    def test_forced_minus_one_resolves_to_sole_legal_action(self) -> None:
        decision = override.choose_safe_override(
            action_ids=np.array([4, 8], dtype=np.int64),
            legal_mask=np.array([False, True], dtype=np.bool_),
            head_scores=np.zeros((2, 3), dtype=np.float32),
            normal_action=-1,
        )
        self.assertEqual(decision.action_id, 8)
        self.assertEqual(decision.normal_action_id, 8)
        self.assertFalse(decision.overridden)

    def test_duplicate_padding_ids_do_not_invalidate_packed_actions(self) -> None:
        decision = override.choose_safe_override(
            action_ids=np.array([0, 7, 0, 0], dtype=np.int64),
            legal_mask=np.array([True, True, False, False], dtype=np.bool_),
            head_scores=np.zeros((4, 3), dtype=np.float32),
            normal_action=7,
        )
        self.assertEqual(decision.action_id, 7)


class BootstrapAndParityTests(unittest.TestCase):
    def test_bootstrap_membership_is_deterministic_at_match_unit(self) -> None:
        keys = ["match-a", "match-b", "match-a", 77, b"match-c"]
        first = override.deterministic_bootstrap_membership(keys, seed=840060001)
        second = override.deterministic_bootstrap_membership(keys, seed=840060001)
        another_seed = override.deterministic_bootstrap_membership(
            keys, seed=840060002
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], first[2])
        self.assertEqual(first.shape, (len(keys), 3))
        self.assertEqual(first.dtype, np.dtype(np.bool_))
        self.assertFalse(np.array_equal(first, another_seed))

    def test_zero_heads_have_full_exact_normal_parity_and_stable_digest(self) -> None:
        action_ids = np.array([0, 3, 8, 20], dtype=np.int64)
        legal_masks = np.array(
            [[True, True, False, False], [False, False, True, False]],
            dtype=np.bool_,
        )
        scores = np.zeros((2, 4, 3), dtype=np.float32)
        normals = np.array([3, -1], dtype=np.int64)
        first = override.assert_zero_head_exact_normal_parity(
            action_ids=action_ids,
            legal_masks=legal_masks,
            head_scores=scores,
            normal_actions=normals,
        )
        second = override.assert_zero_head_exact_normal_parity(
            action_ids=action_ids,
            legal_masks=legal_masks,
            head_scores=scores,
            normal_actions=normals,
        )
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

    def test_zero_head_helper_rejects_nonzero_scores_even_without_override(self) -> None:
        with self.assertRaisesRegex(AssertionError, "not exactly zero"):
            override.assert_zero_head_exact_normal_parity(
                action_ids=np.array([1, 2], dtype=np.int64),
                legal_masks=np.array([[True, True]], dtype=np.bool_),
                head_scores=np.array(
                    [[[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]],
                    dtype=np.float32,
                ),
                normal_actions=np.array([1], dtype=np.int64),
            )

    def test_parity_helper_rejects_actual_override(self) -> None:
        with self.assertRaisesRegex(AssertionError, "diverged from Normal"):
            override.assert_exact_normal_parity(
                action_ids=np.array([1, 2], dtype=np.int64),
                legal_masks=np.array([[True, True]], dtype=np.bool_),
                head_scores=np.array([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]]),
                normal_actions=np.array([1], dtype=np.int64),
                threshold=0.1,
            )


class PrivacyBoundaryTests(unittest.TestCase):
    def test_public_forward_signature_has_no_private_input(self) -> None:
        parameters = tuple(
            inspect.signature(override.V6PublicDeltaScorer.forward).parameters
        )
        self.assertEqual(
            parameters,
            (
                "self",
                "public_batch",
                "legal_action_indices",
                "legal_action_mask",
            ),
        )
        self.assertTrue(override.public_delta_api_has_no_privileged_input())

    def test_central_critic_is_explicitly_privileged_training_api(self) -> None:
        if not override.TORCH_AVAILABLE:
            self.skipTest("PyTorch is available only on the GPU host")
        parameters = tuple(
            inspect.signature(
                override.V6CentralBootstrapActionQCritic.forward
            ).parameters
        )
        self.assertIn("privileged_states", parameters)
        self.assertNotIn("privileged_states", inspect.signature(
            override.V6PublicDeltaScorer.forward
        ).parameters)

    def test_torch_critic_masks_padding_and_backpropagates(self) -> None:
        if not override.TORCH_AVAILABLE:
            self.skipTest("PyTorch is available only on the GPU host")
        torch = override.torch
        critic = override.V6CentralBootstrapActionQCritic()
        states = torch.randn(2, 512, dtype=torch.float32)
        features = torch.randn(2, 4, 22, dtype=torch.float32)
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]],
            dtype=torch.bool,
        )
        counts = torch.tensor([4, 10], dtype=torch.long)
        output = critic(states, features, mask, counts)
        self.assertEqual(tuple(output.values.shape), (2,))
        self.assertEqual(tuple(output.q_values.shape), (2, 4, 3))
        self.assertTrue((output.q_values[~mask] <= -1.0e8).all().item())
        membership = torch.tensor(
            [[True, False, True], [False, True, True]], dtype=torch.bool
        )
        loss = override.bootstrap_action_q_huber_loss(
            output,
            torch.tensor([1, 2], dtype=torch.long),
            torch.zeros(2, dtype=torch.float32),
            membership,
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

    def test_zero_advantage_heads_start_at_exact_value_baseline(self) -> None:
        if not override.TORCH_AVAILABLE:
            self.skipTest("PyTorch is available only on the GPU host")
        torch = override.torch
        critic = override.V6CentralBootstrapActionQCritic()
        # Use an exactly representable, nonzero V so this proves Q is the
        # residual identity V + 0, rather than merely observing that two
        # independently zero outputs happen to agree.
        with torch.no_grad():
            critic.value_output.weight.zero_()
            critic.value_output.bias.fill_(0.625)
        states = torch.randn(3, 512, dtype=torch.float32)
        features = torch.randn(3, 5, 22, dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, True],
            ],
            dtype=torch.bool,
        )
        output = critic(states, features, mask, torch.tensor([4, 7, 10]))
        self.assertTrue(
            torch.equal(output.values, torch.full_like(output.values, 0.625))
        )
        expected = output.values[:, None, None].expand_as(output.q_values)
        self.assertTrue(torch.equal(output.q_values[mask], expected[mask]))

    def test_q_only_gradient_cannot_modify_value_or_state_trunk(self) -> None:
        if not override.TORCH_AVAILABLE:
            self.skipTest("PyTorch is available only on the GPU host")
        torch = override.torch
        torch.manual_seed(860_600_001)
        critic = override.V6CentralBootstrapActionQCritic()
        # Activate the residual branch before checking isolation.  With the
        # exact-zero initialization left in place, a missing state.detach()
        # could be hidden by the zero output weights blocking all upstream
        # residual gradients.
        with torch.no_grad():
            for head in critic.q_heads:
                head[-1].weight.fill_(0.25)
        states = torch.randn(2, 512, dtype=torch.float32)
        features = torch.randn(2, 3, 22, dtype=torch.float32)
        mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        output = critic(states, features, mask, torch.tensor([4, 8]))
        loss = override.bootstrap_action_q_huber_loss(
            output,
            torch.tensor([1, 2], dtype=torch.long),
            torch.full((2,), 3.0),
            torch.ones(2, 3, dtype=torch.bool),
        )
        loss.backward()

        protected = (
            list(critic.player_count_embedding.parameters())
            + list(critic.state_encoder.parameters())
            + list(critic.value_output.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is None
                or int(torch.count_nonzero(parameter.grad)) == 0
                for parameter in protected
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and int(torch.count_nonzero(parameter.grad)) > 0
                for parameter in critic.q_heads.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and int(torch.count_nonzero(parameter.grad)) > 0
                for parameter in critic.state_to_action.parameters()
            )
        )

    def test_zero_public_scorer_runs_full_v5_batch_with_normal_parity(self) -> None:
        if not override.TORCH_AVAILABLE:
            self.skipTest("PyTorch is available only on the GPU host")
        from test_v5_model import public_batch, tiny_actor_config
        from v5_model import V5PublicActor

        actor = V5PublicActor(tiny_actor_config())
        scorer = override.V6PublicDeltaScorer(actor)
        batch, normals = public_batch(batch_size=3)
        output = scorer(batch)
        self.assertTrue((output.head_scores == 0.0).all().item())
        digest = override.assert_zero_head_exact_normal_parity(
            action_ids=output.action_indices.detach().numpy(),
            legal_masks=output.action_mask.detach().numpy(),
            head_scores=output.head_scores.detach().numpy(),
            normal_actions=normals.detach().numpy(),
        )
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_q_loss_supervises_only_logged_action_and_member_heads(self) -> None:
        if not override.TORCH_AVAILABLE:
            self.skipTest("PyTorch is available only on the GPU host")
        torch = override.torch
        mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        q_values = torch.zeros((2, 3, 3), dtype=torch.float32, requires_grad=True)
        output = override.V6CentralBootstrapQOutput(
            torch.zeros(2), q_values, mask
        )
        positions = torch.tensor([1, 2], dtype=torch.long)
        membership = torch.tensor(
            [[True, False, True], [False, True, True]], dtype=torch.bool
        )
        loss = override.bootstrap_action_q_huber_loss(
            output,
            positions,
            torch.tensor([1.0, -2.0]),
            membership,
        )
        loss.backward()
        expected_nonzero = torch.zeros_like(q_values, dtype=torch.bool)
        expected_nonzero[0, 1, 0] = True
        expected_nonzero[0, 1, 2] = True
        expected_nonzero[1, 2, 1] = True
        expected_nonzero[1, 2, 2] = True
        self.assertTrue(torch.equal(q_values.grad != 0.0, expected_nonzero))


if __name__ == "__main__":
    unittest.main()
