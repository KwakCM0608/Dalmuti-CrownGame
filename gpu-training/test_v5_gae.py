from __future__ import annotations

import unittest

import numpy as np

from v5_gae import (
    compute_smdp_gae,
    decision_match_ids,
    equal_player_count_loss_weights,
)


class V5DecisionTimeGAETests(unittest.TestCase):
    def test_interleaved_candidate_links_use_reward_to_next_and_next_value(self) -> None:
        result = compute_smdp_gae(
            reward_to_next=np.asarray([0.5, 1.0, 2.0, -1.0], np.float32),
            next_decision=np.asarray([2, 3, -1, -1], np.int32),
            done=np.asarray([False, False, True, True], np.bool_),
            old_values=np.asarray([0.2, -0.1, 0.4, 0.3], np.float32),
            match_offsets=np.asarray([0, 4], np.uint32),
            decision_actor_ids=np.asarray([0, 1, 0, 1], np.uint8),
            player_counts=np.asarray([4], np.uint8),
            forced=np.asarray([False, False, True, False], np.bool_),
            candidate_bitsets=np.asarray([0b0011], np.uint16),
        )
        np.testing.assert_allclose(
            result.deltas, np.asarray([0.7, 1.4, 1.6, -1.3]), rtol=0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            result.advantages,
            np.asarray([2.22, 0.165, 1.6, -1.3]),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.returns,
            np.asarray([2.42, 0.065, 2.0, -1.0]),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            result.policy_mask, np.asarray([True, True, False, True])
        )
        np.testing.assert_array_equal(result.value_mask, np.ones(4, np.bool_))
        np.testing.assert_allclose(result.policy_loss_weights[result.policy_mask], 1.0)
        np.testing.assert_allclose(result.value_loss_weights, 1.0)

    def test_cross_match_next_decision_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "match boundary"):
            compute_smdp_gae(
                reward_to_next=np.zeros(4, np.float32),
                next_decision=np.asarray([2, -1, -1, -1], np.int32),
                done=np.asarray([False, True, True, True], np.bool_),
                old_values=np.zeros(4, np.float32),
                match_offsets=np.asarray([0, 2, 4], np.uint32),
                decision_actor_ids=np.asarray([0, 1, 0, 1], np.uint8),
                player_counts=np.asarray([4, 4], np.uint8),
                forced=np.zeros(4, np.bool_),
            )

    def test_chain_requires_same_actor_and_exact_terminal(self) -> None:
        arguments = dict(
            reward_to_next=np.zeros(2, np.float32),
            next_decision=np.asarray([1, -1], np.int32),
            done=np.asarray([False, True], np.bool_),
            old_values=np.zeros(2, np.float32),
            match_offsets=np.asarray([0, 2], np.uint32),
            decision_actor_ids=np.asarray([0, 1], np.uint8),
            player_counts=np.asarray([4], np.uint8),
            forced=np.zeros(2, np.bool_),
        )
        with self.assertRaisesRegex(ValueError, "same candidate"):
            compute_smdp_gae(**arguments)

        arguments["decision_actor_ids"] = np.asarray([0, 0], np.uint8)
        arguments["done"] = np.asarray([False, False], np.bool_)
        with self.assertRaisesRegex(ValueError, "strictly forward"):
            compute_smdp_gae(**arguments)

    def test_player_count_weights_equalize_stratum_total(self) -> None:
        counts = np.asarray([4, 5, 5, 5], np.uint8)
        weights = equal_player_count_loss_weights(counts)
        self.assertAlmostEqual(float(weights[counts == 4].sum()), 2.0, places=6)
        self.assertAlmostEqual(float(weights[counts == 5].sum()), 2.0, places=6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        masked = equal_player_count_loss_weights(
            counts, np.asarray([True, True, False, False], np.bool_)
        )
        np.testing.assert_allclose(masked, np.asarray([1.0, 1.0, 0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "missing player counts"):
            equal_player_count_loss_weights(
                counts, require_all_player_counts=True
            )

    def test_require_all_p4_to_p10_balances_each_total(self) -> None:
        counts = np.repeat(np.arange(4, 11, dtype=np.uint8), np.arange(1, 8))
        weights = equal_player_count_loss_weights(
            counts, require_all_player_counts=True
        )
        totals = [float(weights[counts == player].sum()) for player in range(4, 11)]
        np.testing.assert_allclose(totals, np.repeat(totals[0], 7), atol=2e-6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_match_offsets_are_strict_complete_prefixes(self) -> None:
        np.testing.assert_array_equal(
            decision_match_ids(np.asarray([0, 2, 5], np.uint32), 5),
            np.asarray([0, 0, 1, 1, 1], np.int32),
        )
        for offsets in (
            np.asarray([1, 2], np.uint32),
            np.asarray([0, 0, 2], np.uint32),
            np.asarray([0, 3], np.uint32),
        ):
            with self.assertRaises(ValueError):
                decision_match_ids(offsets, 2)

    def test_noncanonical_gamma_or_lambda_is_rejected(self) -> None:
        base = dict(
            reward_to_next=np.asarray([1.0], np.float32),
            next_decision=np.asarray([-1], np.int32),
            done=np.asarray([True], np.bool_),
            old_values=np.asarray([0.0], np.float32),
            match_offsets=np.asarray([0, 1], np.uint32),
            decision_actor_ids=np.asarray([0], np.uint8),
            player_counts=np.asarray([4], np.uint8),
            forced=np.asarray([True], np.bool_),
        )
        with self.assertRaisesRegex(ValueError, "gamma=1.0"):
            compute_smdp_gae(**base, gamma=0.99)
        with self.assertRaisesRegex(ValueError, "gae_lambda=0.95"):
            compute_smdp_gae(**base, gae_lambda=1.0)


if __name__ == "__main__":
    unittest.main()
