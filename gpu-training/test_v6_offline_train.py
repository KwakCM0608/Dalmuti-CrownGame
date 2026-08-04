from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


GPU_TRAINING = Path(__file__).resolve().parent
if str(GPU_TRAINING) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINING))

import v6_offline_train as offline
from v5_model import V5CentralStateValueCritic
from v6_override import (
    V6CentralBootstrapActionQCritic,
    V6CentralBootstrapQConfig,
)


class BalanceContractTests(unittest.TestCase):
    def test_equal_p_and_behavior_kind_mass_is_exact(self) -> None:
        players: list[int] = []
        behavior: list[int] = []
        normal: list[int] = []
        for player in range(4, 11):
            # Deliberately vary both cell sizes with p.
            for _ in range(player - 2):
                players.append(player)
                behavior.append(7)
                normal.append(7)
            for _ in range(12 - player):
                players.append(player)
                behavior.append(8)
                normal.append(7)
        players_array = np.asarray(players, dtype=np.uint8)
        behavior_array = np.asarray(behavior, dtype=np.uint16)
        normal_array = np.asarray(normal, dtype=np.uint16)
        weights, report = offline.balanced_behavior_player_weights(
            players_array, behavior_array, normal_array
        )
        expected = len(players) / 14.0
        alternative = behavior_array != normal_array
        for player in range(4, 11):
            for kind in (False, True):
                selected = (players_array == player) & (alternative == kind)
                self.assertAlmostEqual(
                    float(weights[selected].sum(dtype=np.float64)), expected, places=5
                )
        self.assertAlmostEqual(float(weights.mean(dtype=np.float64)), 1.0, places=6)
        self.assertEqual(report["contract"], offline.V6_BEHAVIOR_BALANCE_CONTRACT)

    def test_missing_alternative_cell_fails_closed(self) -> None:
        players = np.repeat(np.arange(4, 11, dtype=np.uint8), 2)
        behavior = np.tile(np.asarray([1, 2], np.uint16), 7)
        normal = np.tile(np.asarray([1, 1], np.uint16), 7)
        behavior[players == 9] = normal[players == 9]
        with self.assertRaisesRegex(ValueError, "alternative rows for p9"):
            offline.balanced_behavior_player_weights(players, behavior, normal)

    def test_pilot_fraction_is_restricted_to_declared_vertical_slices(self) -> None:
        self.assertEqual(offline.V6OfflineConfig(pilot_fraction=0.05).pilot_fraction, 0.05)
        self.assertEqual(offline.V6OfflineConfig(pilot_fraction=0.10).pilot_fraction, 0.10)
        for value in (0.0, 0.01, 0.25, 1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "0.05 or 0.10"):
                    offline.V6OfflineConfig(pilot_fraction=value)

    def test_value_weights_ignore_behavior_type_and_equalize_only_p(self) -> None:
        players = np.asarray([4, 4, 4, 5, 5, 6, 7, 8, 9, 10], dtype=np.uint8)
        weights, report = offline.equal_player_value_weights(players)
        expected = len(players) / 7.0
        for player in range(4, 11):
            self.assertAlmostEqual(
                float(weights[players == player].sum(dtype=np.float64)),
                expected,
                places=5,
            )
        self.assertEqual(report["contract"], "dalmuti-v6-equal-p-state-value-mass-v1")

    def test_alternative_mass_is_explicit_and_adjustable(self) -> None:
        players = np.repeat(np.arange(4, 11, dtype=np.uint8), 4)
        normal = np.ones(players.shape, dtype=np.uint16)
        behavior = normal.copy()
        behavior.reshape(7, 4)[:, -1] = 2
        weights, report = offline.balanced_behavior_player_weights(
            players,
            behavior,
            normal,
            alternative_mass_fraction=0.25,
        )
        alternative = behavior != normal
        for player in range(4, 11):
            local = players == player
            total = float(weights[local].sum(dtype=np.float64))
            alt = float(weights[local & alternative].sum(dtype=np.float64))
            self.assertAlmostEqual(alt / total, 0.25, places=6)
        self.assertEqual(report["alternativeMassFraction"], 0.25)
        self.assertGreater(report["maximumToMinimumRowWeightRatio"], 0.0)

    def test_q_and_distillation_alternative_mass_are_independent(self) -> None:
        config = offline.V6OfflineConfig(
            alternative_mass_fraction=0.25,
            distill_alternative_mass_fraction=0.50,
        )
        self.assertEqual(config.alternative_mass_fraction, 0.25)
        self.assertEqual(config.distill_alternative_mass_fraction, 0.50)
        self.assertEqual(config.to_dict()["alternative_mass_fraction"], 0.25)
        self.assertEqual(config.to_dict()["distill_alternative_mass_fraction"], 0.50)

        for name in (
            "alternative_mass_fraction",
            "distill_alternative_mass_fraction",
        ):
            for value in (0.0, 1.0, float("nan")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        offline.V6OfflineConfig(**{name: value})


class PackedActionContractTests(unittest.TestCase):
    def test_batches_are_globally_interleaved_and_visit_each_row_once(self) -> None:
        rows = {
            "p4-a": np.arange(0, 12, dtype=np.int64),
            "p7-b": np.arange(100, 112, dtype=np.int64),
            "p10-c": np.arange(200, 212, dtype=np.int64),
        }
        batches = list(offline._shuffled_batches(rows, batch_size=2, seed=77))
        keys = [key for key, _ in batches]
        # No shard is drained as one contiguous six-batch block.
        self.assertTrue(any(keys[index] == keys[index + 2] != keys[index + 1] for index in range(len(keys) - 2)))
        for key, expected in rows.items():
            actual = np.concatenate([batch for batch_key, batch in batches if batch_key == key])
            np.testing.assert_array_equal(np.sort(actual), expected)

    def test_logged_behavior_must_appear_exactly_once(self) -> None:
        ids = torch.tensor([[1, 3, 0], [4, 9, 12]], dtype=torch.long)
        mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        selected = torch.tensor([3, 12], dtype=torch.long)
        positions = offline.selected_packed_positions(ids, mask, selected)
        torch.testing.assert_close(positions, torch.tensor([1, 2]))

        with self.assertRaisesRegex(ValueError, "exactly once"):
            offline.selected_packed_positions(
                ids, mask, torch.tensor([2, 12], dtype=torch.long)
            )
        duplicate = torch.tensor([[1, 1, 3]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "2 occurrences"):
            offline.selected_packed_positions(
                duplicate,
                torch.ones_like(duplicate, dtype=torch.bool),
                torch.tensor([1], dtype=torch.long),
            )


class CentralQLossTests(unittest.TestCase):
    def test_q_gradient_touches_selected_logged_position_only(self) -> None:
        values = torch.zeros(2, requires_grad=True)
        q = torch.zeros(2, 3, 3, requires_grad=True)
        positions = torch.tensor([1, 2], dtype=torch.long)
        targets = torch.tensor([1.0, -1.0])
        weights = torch.tensor([2.0, 0.5])
        membership = torch.tensor(
            [[True, False, True], [False, True, True]], dtype=torch.bool
        )
        loss, parts = offline.weighted_central_v_q_loss(
            values=values,
            q_values=q,
            selected_positions=positions,
            targets=targets,
            value_row_weights=torch.ones(2),
            q_row_weights=weights,
            bootstrap_membership=membership,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(parts["qHeads"].shape), (3,))
        loss.backward()
        self.assertIsNotNone(q.grad)
        assert q.grad is not None
        unselected = torch.ones((2, 3), dtype=torch.bool)
        unselected[torch.arange(2), positions] = False
        self.assertEqual(int(torch.count_nonzero(q.grad[unselected])), 0)
        self.assertGreater(int(torch.count_nonzero(q.grad[~unselected])), 0)

    def test_batch_local_inactive_bootstrap_head_is_skipped(self) -> None:
        loss, parts = offline.weighted_central_v_q_loss(
            values=torch.zeros(2),
            q_values=torch.zeros(2, 2, 3),
            selected_positions=torch.tensor([0, 1]),
            targets=torch.ones(2),
            value_row_weights=torch.ones(2),
            q_row_weights=torch.ones(2),
            bootstrap_membership=torch.tensor(
                [[True, True, False], [True, True, False]], dtype=torch.bool
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(
            parts["qActiveHeads"], torch.tensor([True, True, False])
        )

    def test_forced_rows_have_exactly_zero_q_gradient_but_train_value(self) -> None:
        values = torch.zeros(2, requires_grad=True)
        q = torch.zeros(2, 2, 3, requires_grad=True)
        loss, _ = offline.weighted_central_v_q_loss(
            values=values,
            q_values=q,
            selected_positions=torch.tensor([0, 1]),
            targets=torch.ones(2),
            value_row_weights=torch.ones(2),
            q_row_weights=torch.zeros(2),
            bootstrap_membership=torch.ones(2, 3, dtype=torch.bool),
        )
        loss.backward()
        assert q.grad is not None and values.grad is not None
        self.assertEqual(int(torch.count_nonzero(q.grad)), 0)
        self.assertGreater(int(torch.count_nonzero(values.grad)), 0)

    def test_global_row_weights_are_not_cancelled_by_local_normalization(self) -> None:
        def value_loss(weight: float) -> float:
            loss, _ = offline.weighted_central_v_q_loss(
                values=torch.zeros(1),
                q_values=torch.zeros(1, 1, 3),
                selected_positions=torch.tensor([0]),
                targets=torch.ones(1),
                value_row_weights=torch.tensor([weight]),
                q_row_weights=torch.zeros(1),
                bootstrap_membership=torch.zeros(1, 3, dtype=torch.bool),
            )
            return float(loss)

        self.assertAlmostEqual(value_loss(2.0), 4.0 * value_loss(0.5), places=6)


class PublicDistillationTests(unittest.TestCase):
    def _inputs(self):  # type: ignore[no-untyped-def]
        scores = torch.zeros(2, 4, 3, requires_grad=True)
        ids = torch.tensor([[1, 2, 3, 0], [5, 6, 0, 0]], dtype=torch.long)
        mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]], dtype=torch.bool
        )
        behavior_positions = torch.tensor([1, 0], dtype=torch.long)
        normal_positions = torch.tensor([0, 0], dtype=torch.long)
        behavior = torch.tensor([2, 5], dtype=torch.long)
        normal = torch.tensor([1, 5], dtype=torch.long)
        teacher = torch.tensor([[1.0, 0.8, 1.2], [0.0, 0.0, 0.0]])
        return scores, ids, mask, behavior_positions, normal_positions, behavior, normal, teacher

    def test_only_observed_actions_get_two_sided_teacher_targets(self) -> None:
        scores, ids, mask, bpos, npos, behavior, normal, teacher = self._inputs()
        loss, parts = offline.observed_delta_distillation_loss(
            student_scores=scores,
            action_ids=ids,
            action_mask=mask,
            behavior_positions=bpos,
            normal_positions=npos,
            behavior_actions=behavior,
            normal_actions=normal,
            teacher_behavior_deltas=teacher,
            row_weights=torch.ones(2),
            retention_hinge_weight=0.0,
        )
        loss.backward()
        assert scores.grad is not None
        # Row 0 action 3 is legal but unobserved; with retention disabled it has
        # no target and therefore no gradient.  Illegal padding also has none.
        self.assertEqual(int(torch.count_nonzero(scores.grad[0, 2])), 0)
        self.assertEqual(int(torch.count_nonzero(scores.grad[0, 3])), 0)
        self.assertGreater(int(torch.count_nonzero(scores.grad[0, 1])), 0)
        self.assertEqual(int(parts["alternativeRows"]), 1)
        self.assertEqual(int(parts["unobservedLegalItems"]), 2)

    def test_retention_is_one_sided_not_a_copied_teacher_label(self) -> None:
        scores, ids, mask, bpos, npos, behavior, normal, teacher = self._inputs()
        with torch.no_grad():
            scores[0, 2] = torch.tensor([0.5, -0.5, 0.2])
        loss, parts = offline.observed_delta_distillation_loss(
            student_scores=scores,
            action_ids=ids,
            action_mask=mask,
            behavior_positions=bpos,
            normal_positions=npos,
            behavior_actions=behavior,
            normal_actions=normal,
            teacher_behavior_deltas=teacher,
            row_weights=torch.ones(2),
            retention_hinge_weight=1.0,
        )
        loss.backward()
        assert scores.grad is not None
        self.assertGreater(float(parts["retention"]), 0.0)
        self.assertGreater(float(scores.grad[0, 2, 0]), 0.0)
        self.assertEqual(float(scores.grad[0, 2, 1]), 0.0)
        self.assertGreater(float(scores.grad[0, 2, 2]), 0.0)


class ValidationMetricTests(unittest.TestCase):
    def test_numpy_huber_metrics_match_known_values(self) -> None:
        metrics = offline._metrics(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.5, 2.0], dtype=np.float32),
            1.0,
        )
        self.assertEqual(metrics["count"], 3)
        self.assertAlmostEqual(metrics["huber"], (0.0 + 0.125 + 1.5) / 3.0)
        self.assertAlmostEqual(metrics["mae"], (0.0 + 0.5 + 2.0) / 3.0)


class TorchInitializationTests(unittest.TestCase):
    def test_pretrained_v_trunk_is_copied_exactly_while_q_heads_stay_separate(self) -> None:
        torch.manual_seed(123)
        source = V5CentralStateValueCritic()
        with torch.no_grad():
            source.value_output.weight.zero_()
            source.value_output.bias.fill_(-0.375)
        target = V6CentralBootstrapActionQCritic(V6CentralBootstrapQConfig())
        q_before = {
            name: value.detach().clone()
            for name, value in target.q_heads.state_dict().items()
        }
        receipt = offline._initialize_central_from_pretrained_value(target, source)
        self.assertRegex(receipt["sourceTensorStateSha256"], r"^[0-9a-f]{64}$")
        for name, value in source.player_count_embedding.state_dict().items():
            torch.testing.assert_close(target.player_count_embedding.state_dict()[name], value)
        for name, value in source.value_network[:-1].state_dict().items():
            torch.testing.assert_close(target.state_encoder.state_dict()[name], value)
        for name, value in source.value_output.state_dict().items():
            torch.testing.assert_close(target.value_output.state_dict()[name], value)
        for name, value in q_before.items():
            torch.testing.assert_close(target.q_heads.state_dict()[name], value)

        target.eval()
        mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        with torch.no_grad():
            output = target(
                torch.randn(2, 512),
                torch.randn(2, 3, 22),
                mask,
                torch.tensor([4, 10], dtype=torch.long),
            )
        torch.testing.assert_close(
            output.values, torch.full_like(output.values, -0.375), rtol=0.0, atol=0.0
        )
        expected = output.values[:, None, None].expand_as(output.q_values)
        self.assertTrue(torch.equal(output.q_values[mask], expected[mask]))


if __name__ == "__main__":
    unittest.main()
