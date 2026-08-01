import unittest

import torch

from v4_model import V4_ACTION_COUNT
from v4_objectives import (
    action_q_regression_loss,
    expected_action_q,
    expected_sarsa_lambda_targets,
    masked_behavior_cloning_loss,
    masked_probabilities,
    nonforced_policy_eligibility,
    vrpo_clipped_policy_loss,
)


class V4ObjectiveTests(unittest.TestCase):
    def test_expected_sarsa_lambda_endpoints(self) -> None:
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        dones = torch.tensor([[False], [False], [True]])
        legal = torch.zeros(3, 1, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :, :2] = True
        logits = torch.zeros(3, 1, V4_ACTION_COUNT)
        q_values = torch.zeros(3, 1, V4_ACTION_COUNT)
        q_values[0, 0, :2] = torch.tensor([1.0, 3.0])
        q_values[1, 0, :2] = torch.tensor([3.0, 5.0])
        q_values[2, 0, :2] = torch.tensor([5.0, 7.0])

        one_step = expected_sarsa_lambda_targets(
            rewards,
            dones,
            q_values,
            logits,
            legal,
            gamma=1.0,
            lambda_=0.0,
        )
        monte_carlo = expected_sarsa_lambda_targets(
            rewards,
            dones,
            q_values,
            logits,
            legal,
            gamma=1.0,
            lambda_=1.0,
        )
        self.assertTrue(torch.allclose(one_step[:, 0], torch.tensor([5.0, 8.0, 3.0])))
        self.assertTrue(torch.allclose(monte_carlo[:, 0], torch.tensor([6.0, 5.0, 3.0])))

    def test_expected_sarsa_honors_padded_suffix(self) -> None:
        rewards = torch.tensor([[1.0], [2.0], [999.0]])
        dones = torch.tensor([[False], [True], [False]])
        valid = torch.tensor([[True], [True], [False]])
        legal = torch.zeros(3, 1, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :, 0] = True
        logits = torch.zeros_like(legal, dtype=torch.float32)
        q_values = torch.zeros_like(logits)
        targets = expected_sarsa_lambda_targets(
            rewards,
            dones,
            q_values,
            logits,
            legal,
            valid_masks=valid,
            lambda_=1.0,
        )
        self.assertTrue(torch.equal(targets[:, 0], torch.tensor([3.0, 2.0, 0.0])))

    def test_masked_expectation_and_probabilities_exclude_illegal_values(self) -> None:
        legal = torch.zeros(1, V4_ACTION_COUNT, dtype=torch.bool)
        legal[0, [4, 9]] = True
        logits = torch.zeros(1, V4_ACTION_COUNT)
        q_values = torch.full((1, V4_ACTION_COUNT), 1.0e20)
        q_values[0, 4] = 2.0
        q_values[0, 9] = 4.0
        probabilities = masked_probabilities(logits, legal)
        expectation = expected_action_q(q_values, logits, legal)
        self.assertEqual(float(probabilities[~legal].sum()), 0.0)
        self.assertAlmostEqual(float(probabilities[0, 4]), 0.5)
        self.assertAlmostEqual(float(expectation[0]), 3.0)

    def test_vrpo_policy_detaches_privileged_q_and_is_finite(self) -> None:
        torch.manual_seed(19)
        logits = torch.randn(4, V4_ACTION_COUNT, requires_grad=True)
        q_values = torch.randn(4, V4_ACTION_COUNT, requires_grad=True)
        legal = torch.zeros(4, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :8] = True
        actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        old_log_probabilities = torch.log_softmax(
            logits.detach().masked_fill(~legal, -1.0e9), dim=-1
        ).gather(1, actions[:, None]).squeeze(1)
        result = vrpo_clipped_policy_loss(
            logits,
            legal,
            actions,
            old_log_probabilities,
            torch.tensor([1.0, -0.5, 0.25, 0.75]),
            q_values=q_values,
            q_boost_coefficient=0.5,
        )
        result.loss.backward()

        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(torch.isfinite(result.entropy))
        self.assertIsNotNone(logits.grad)
        self.assertIsNone(q_values.grad)

    def test_nonforced_policy_objective_is_invariant_to_forced_rows_and_partition(self) -> None:
        torch.manual_seed(23)
        logits = torch.randn(7, V4_ACTION_COUNT)
        legal = torch.zeros(7, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:4, :3] = True
        legal[4:, 0] = True
        eligible = torch.ones(7, dtype=torch.bool)
        actions = torch.tensor([0, 1, 2, 0, 0, 0, 0], dtype=torch.long)
        advantages = torch.tensor([1.25, -0.5, 0.75, -1.0, 50.0, -80.0, 30.0])
        old_log_probs = torch.log_softmax(
            logits.masked_fill(~legal, -1.0e9), dim=-1
        ).gather(1, actions[:, None]).squeeze(1)
        policy_mask = nonforced_policy_eligibility(legal, eligible)

        combined = vrpo_clipped_policy_loss(
            logits[policy_mask],
            legal[policy_mask],
            actions[policy_mask],
            old_log_probs[policy_mask],
            advantages[policy_mask],
            entropy_coefficient=0.003,
            normalize_advantages=False,
        )
        first = vrpo_clipped_policy_loss(
            logits[:1], legal[:1], actions[:1], old_log_probs[:1], advantages[:1],
            entropy_coefficient=0.003,
            normalize_advantages=False,
        )
        second = vrpo_clipped_policy_loss(
            logits[1:4], legal[1:4], actions[1:4], old_log_probs[1:4], advantages[1:4],
            entropy_coefficient=0.003,
            normalize_advantages=False,
        )

        self.assertTrue(torch.equal(policy_mask, torch.tensor([True] * 4 + [False] * 3)))
        for name in ("loss", "policy_loss", "entropy", "approx_kl", "clip_fraction"):
            partitioned = (
                getattr(first, name) + 3.0 * getattr(second, name)
            ) / 4.0
            self.assertTrue(torch.allclose(getattr(combined, name), partitioned, atol=1e-7), name)

    def test_bc_and_action_q_losses_reject_illegal_targets(self) -> None:
        logits = torch.zeros(2, V4_ACTION_COUNT, requires_grad=True)
        q_values = torch.zeros(2, V4_ACTION_COUNT, requires_grad=True)
        legal = torch.zeros(2, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :2] = True
        actions = torch.tensor([0, 1])
        bc = masked_behavior_cloning_loss(logits, legal, actions)
        q_loss = action_q_regression_loss(
            q_values, legal, actions, torch.tensor([1.0, -1.0])
        )
        (bc + q_loss).backward()
        self.assertTrue(torch.isfinite(bc))
        self.assertTrue(torch.isfinite(q_loss))
        with self.assertRaisesRegex(ValueError, "legal"):
            masked_behavior_cloning_loss(logits.detach(), legal, torch.tensor([0, 2]))


if __name__ == "__main__":
    unittest.main()
