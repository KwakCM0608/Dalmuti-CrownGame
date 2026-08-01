import unittest

import torch
from torch.nn import functional as F

from v4_model import V4_ACTION_COUNT
from v4_objectives import (
    action_q_regression_loss,
    expected_action_q,
    expected_sarsa_lambda_targets,
    masked_behavior_cloning_loss,
    masked_log_probabilities,
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

    def test_weighted_bc_and_q_losses_use_exact_row_reduction(self) -> None:
        logits = torch.zeros(3, V4_ACTION_COUNT)
        logits[:, :3] = torch.tensor(
            [[2.0, -1.0, 0.5], [-0.5, 1.5, 0.25], [0.1, -0.2, 0.8]]
        )
        q_values = torch.zeros(3, V4_ACTION_COUNT)
        q_values[:, :3] = torch.tensor(
            [[2.0, -1.0, 0.5], [0.0, 1.5, -0.5], [3.0, 0.25, -2.0]]
        )
        legal = torch.zeros(3, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :3] = True
        actions = torch.tensor([0, 1, 2], dtype=torch.long)
        targets = torch.tensor([0.5, 2.0, -0.5])
        weights = torch.tensor([1.0, 3.0, 2.0])

        bc = masked_behavior_cloning_loss(
            logits, legal, actions, weights=weights
        )
        bc_rows = F.nll_loss(
            masked_log_probabilities(logits, legal),
            actions,
            reduction="none",
        )
        expected_bc = (weights * bc_rows).sum() / weights.sum()
        self.assertTrue(torch.equal(bc, expected_bc))

        q_loss = action_q_regression_loss(
            q_values,
            legal,
            actions,
            targets,
            huber_delta=0.75,
            weights=weights,
        )
        predictions = q_values.gather(1, actions[:, None]).squeeze(1)
        q_rows = F.huber_loss(
            predictions,
            targets,
            delta=0.75,
            reduction="none",
        )
        expected_q = (weights * q_rows).sum() / weights.sum()
        self.assertTrue(torch.equal(q_loss, expected_q))

    def test_weighted_vrpo_reduces_loss_and_diagnostics_with_same_weights(self) -> None:
        logits = torch.zeros(3, V4_ACTION_COUNT)
        logits[:, :3] = torch.tensor(
            [[0.5, -0.25, 1.0], [1.25, -0.75, 0.0], [-0.5, 0.25, 0.75]]
        )
        behavior_logits = torch.zeros_like(logits)
        behavior_logits[:, :3] = torch.tensor(
            [[0.1, 0.2, -0.1], [-0.5, 0.5, 0.25], [0.75, -0.25, 0.0]]
        )
        q_values = torch.zeros_like(logits)
        q_values[:, :3] = torch.tensor(
            [[2.0, -1.0, 0.5], [0.0, 1.5, -0.5], [3.0, 0.25, -2.0]]
        )
        legal = torch.zeros(3, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :3] = True
        actions = torch.tensor([0, 1, 2], dtype=torch.long)
        advantages = torch.tensor([1.0, -2.0, 0.5])
        desired_ratios = torch.tensor([1.4, 0.7, 1.05])
        log_probabilities = masked_log_probabilities(logits, legal)
        action_log_probs = log_probabilities.gather(
            1, actions[:, None]
        ).squeeze(1)
        old_action_log_probs = action_log_probs - desired_ratios.log()
        weights = torch.tensor([1.0, 2.0, 5.0])
        clip_ratio = 0.15
        entropy_coefficient = 0.07
        q_boost_coefficient = 0.4

        result = vrpo_clipped_policy_loss(
            logits,
            legal,
            actions,
            old_action_log_probs,
            advantages,
            q_values=q_values,
            behavior_policy_logits=behavior_logits,
            q_boost_coefficient=q_boost_coefficient,
            clip_ratio=clip_ratio,
            entropy_coefficient=entropy_coefficient,
            weights=weights,
        )

        ratio = (action_log_probs - old_action_log_probs).exp()
        q_baseline = expected_action_q(q_values, behavior_logits, legal)
        q_taken = q_values.gather(1, actions[:, None]).squeeze(1)
        q_boost = q_taken - q_baseline
        combined_advantages = advantages + q_boost_coefficient * q_boost
        unclipped = ratio * combined_advantages
        clipped = ratio.clamp(
            1.0 - clip_ratio, 1.0 + clip_ratio
        ) * combined_advantages
        policy_rows = -torch.minimum(unclipped, clipped)
        probabilities = log_probabilities.exp().masked_fill(~legal, 0.0)
        entropy_rows = -(
            probabilities * log_probabilities.masked_fill(~legal, 0.0)
        ).sum(dim=-1)
        log_ratio = action_log_probs - old_action_log_probs
        kl_rows = (ratio - 1.0) - log_ratio
        clip_rows = ((ratio - 1.0).abs() > clip_ratio).to(logits.dtype)

        def weighted(values: torch.Tensor) -> torch.Tensor:
            return (weights * values).sum() / weights.sum()

        expected_policy = weighted(policy_rows)
        expected_entropy = weighted(entropy_rows)
        self.assertTrue(torch.equal(result.policy_loss, expected_policy))
        self.assertTrue(torch.equal(result.entropy, expected_entropy))
        self.assertTrue(torch.equal(result.approx_kl, weighted(kl_rows)))
        self.assertTrue(torch.equal(result.clip_fraction, weighted(clip_rows)))
        self.assertTrue(torch.equal(result.mean_q_boost, weighted(q_boost)))
        self.assertTrue(
            torch.equal(
                result.loss,
                expected_policy - entropy_coefficient * expected_entropy,
            )
        )

    def test_none_weights_preserve_unweighted_results_exactly(self) -> None:
        torch.manual_seed(29)
        logits = torch.randn(4, V4_ACTION_COUNT)
        q_values = torch.randn(4, V4_ACTION_COUNT)
        legal = torch.zeros(4, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :5] = True
        actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        targets = torch.tensor([1.0, -0.5, 2.0, 0.25])
        advantages = torch.tensor([0.5, -1.0, 0.25, 1.5])
        old_action_log_probs = masked_log_probabilities(
            logits, legal
        ).gather(1, actions[:, None]).squeeze(1)

        legacy_bc = masked_behavior_cloning_loss(logits, legal, actions)
        explicit_bc = masked_behavior_cloning_loss(
            logits, legal, actions, weights=None
        )
        expected_bc = F.nll_loss(
            masked_log_probabilities(logits, legal), actions
        )
        self.assertTrue(torch.equal(legacy_bc, explicit_bc))
        self.assertTrue(torch.equal(legacy_bc, expected_bc))

        legacy_q = action_q_regression_loss(
            q_values, legal, actions, targets
        )
        explicit_q = action_q_regression_loss(
            q_values, legal, actions, targets, weights=None
        )
        expected_q = F.huber_loss(
            q_values.gather(1, actions[:, None]).squeeze(1),
            targets,
            delta=1.0,
        )
        self.assertTrue(torch.equal(legacy_q, explicit_q))
        self.assertTrue(torch.equal(legacy_q, expected_q))

        legacy_vrpo = vrpo_clipped_policy_loss(
            logits,
            legal,
            actions,
            old_action_log_probs,
            advantages,
            q_values=q_values,
        )
        explicit_vrpo = vrpo_clipped_policy_loss(
            logits,
            legal,
            actions,
            old_action_log_probs,
            advantages,
            q_values=q_values,
            weights=None,
        )
        log_probabilities = masked_log_probabilities(logits, legal)
        action_log_probs = log_probabilities.gather(
            1, actions[:, None]
        ).squeeze(1)
        detached_q = q_values.detach()
        q_baseline = expected_action_q(detached_q, logits.detach(), legal)
        q_taken = detached_q.gather(1, actions[:, None]).squeeze(1)
        q_boost = q_taken - q_baseline
        combined_advantages = (advantages + q_boost).detach()
        log_ratio = action_log_probs - old_action_log_probs
        ratio = log_ratio.exp()
        unclipped = ratio * combined_advantages
        clipped = ratio.clamp(0.85, 1.15) * combined_advantages
        expected_policy = -torch.minimum(unclipped, clipped).mean()
        probabilities = log_probabilities.exp().masked_fill(~legal, 0.0)
        expected_entropy = -(
            probabilities * log_probabilities.masked_fill(~legal, 0.0)
        ).sum(dim=-1).mean()
        expected_vrpo = {
            "loss": expected_policy - 0.01 * expected_entropy,
            "policy_loss": expected_policy,
            "entropy": expected_entropy,
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
            "clip_fraction": ((ratio - 1.0).abs() > 0.15)
            .to(logits.dtype)
            .mean(),
            "mean_q_boost": q_boost.mean(),
        }
        for field in (
            "loss",
            "policy_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "mean_q_boost",
        ):
            self.assertTrue(
                torch.equal(getattr(legacy_vrpo, field), getattr(explicit_vrpo, field)),
                field,
            )
            self.assertTrue(
                torch.equal(getattr(legacy_vrpo, field), expected_vrpo[field]),
                f"legacy {field}",
            )

    def test_all_weighted_losses_fail_closed_on_invalid_weights(self) -> None:
        logits = torch.zeros(3, V4_ACTION_COUNT)
        q_values = torch.zeros(3, V4_ACTION_COUNT)
        legal = torch.zeros(3, V4_ACTION_COUNT, dtype=torch.bool)
        legal[:, :2] = True
        actions = torch.tensor([0, 1, 0], dtype=torch.long)
        targets = torch.tensor([1.0, -1.0, 0.5])
        old_action_log_probs = masked_log_probabilities(
            logits, legal
        ).gather(1, actions[:, None]).squeeze(1)
        advantages = torch.tensor([0.25, -0.5, 1.0])
        calls = {
            "bc": lambda row_weights: masked_behavior_cloning_loss(
                logits, legal, actions, weights=row_weights
            ),
            "q": lambda row_weights: action_q_regression_loss(
                q_values, legal, actions, targets, weights=row_weights
            ),
            "vrpo": lambda row_weights: vrpo_clipped_policy_loss(
                logits,
                legal,
                actions,
                old_action_log_probs,
                advantages,
                weights=row_weights,
            ),
        }
        invalid_weights = {
            "wrong shape": torch.ones(3, 1),
            "non-floating": torch.ones(3, dtype=torch.long),
            "nan": torch.tensor([1.0, float("nan"), 1.0]),
            "infinity": torch.tensor([1.0, float("inf"), 1.0]),
            "zero entry": torch.tensor([1.0, 0.0, 1.0]),
            "negative entry": torch.tensor([1.0, -0.25, 1.0]),
            "zero sum": torch.zeros(3),
            "non-finite sum": torch.full(
                (3,), torch.finfo(torch.float32).max
            ),
        }
        for call_name, call in calls.items():
            for weight_name, row_weights in invalid_weights.items():
                with self.subTest(call=call_name, weights=weight_name):
                    with self.assertRaises(ValueError):
                        call(row_weights)


if __name__ == "__main__":
    unittest.main()
