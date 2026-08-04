from __future__ import annotations

import unittest

import numpy as np

from v6_targets import (
    balanced_normal_action_weights,
    compute_v6_monte_carlo_returns,
    extract_v6_opponent_hand_targets,
)


class V6TargetTests(unittest.TestCase):
    def test_monte_carlo_returns_follow_interleaved_actor_chains(self) -> None:
        values = compute_v6_monte_carlo_returns(
            reward_to_next=np.asarray([0.0, 0.2, 1.0, -1.0], np.float32),
            next_decision=np.asarray([2, 3, -1, -1], np.int32),
            done=np.asarray([False, False, True, True], np.bool_),
            match_offsets=np.asarray([0, 4], np.uint32),
            decision_actor_ids=np.asarray([0, 1, 0, 1], np.uint8),
            player_counts=np.asarray([4], np.uint8),
            candidate_bitsets=np.asarray([0b0011], np.uint16),
        )
        np.testing.assert_allclose(values, [1.0, -0.8, 1.0, -1.0], rtol=0, atol=1e-7)

    def test_monte_carlo_rejects_cross_match_successor(self) -> None:
        with self.assertRaisesRegex(ValueError, "match boundary"):
            compute_v6_monte_carlo_returns(
                reward_to_next=np.zeros(2, np.float32),
                next_decision=np.asarray([1, -1], np.int32),
                done=np.asarray([False, True], np.bool_),
                match_offsets=np.asarray([0, 1, 2], np.uint32),
                decision_actor_ids=np.asarray([0, 0], np.uint8),
                player_counts=np.asarray([4, 4], np.uint8),
            )

    def test_balanced_normal_weights_are_bounded_and_mean_one(self) -> None:
        actions = np.asarray([0] * 100 + [1] * 4 + [2], np.uint16)
        weights, report = balanced_normal_action_weights(
            actions, np.ones(actions.shape, np.bool_), maximum_ratio=5.0
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertEqual(report["observedActions"], 3)
        self.assertLessEqual(report["realizedClassWeightRatio"], 5.0 + 1e-6)

    def test_extracts_only_opponent_hand_rank_targets(self) -> None:
        states = np.zeros((1, 512), np.float16)
        # Relative blocks 0..3 are present in a p4 state.  Only 1..3 are targets.
        for relative in range(4):
            start = 29 + relative * 25
            states[0, start] = 1
            states[0, start + 1] = relative
            states[0, start + 8] = relative + 1
            states[0, start + 12] = relative + 1
        targets, mask = extract_v6_opponent_hand_targets(
            states, np.asarray([4], np.uint8)
        )
        self.assertEqual(targets.shape, (1, 9, 13))
        self.assertEqual(mask[0].tolist(), [True, True, True] + [False] * 6)
        self.assertEqual(targets[0, 0, 0], 2)
        self.assertEqual(targets[0, 2, 0], 4)
        self.assertFalse(bool(targets[0, 3:].any()))


if __name__ == "__main__":
    unittest.main()
