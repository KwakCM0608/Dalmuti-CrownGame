from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np
import torch

from v4_env import DalmutiScalarEnv
from v5_contract import (
    V5_DECK_COUNTS,
    V5_MAX_HISTORY,
    V5_PUBLIC_CONTRACT_SHA256,
    canonical_v5_public_contract,
    validate_v5_public_contract,
)
from v5_public import (
    V5PublicObservation,
    actor_batch_from_packed_arrays,
    compute_v5_public_beliefs,
    encode_v5_public_observation_bytes,
    pack_v5_public_from_v4,
    pack_v5_public_observations,
    stack_v5_actor_public_features,
    tensorize_v5_public_observation,
    v5_legal_mask_from_public_cards,
    v5_public_from_v4_actor_observation,
    v5_public_observation_from_mapping,
    validate_v5_public_observation,
)


def _role_at(index: int, player_count: int) -> int:
    if index == 0:
        return 0
    if index == 1:
        return 1
    if index == player_count - 2:
        return 3
    if index == player_count - 1:
        return 4
    return 2


def _rank_counts(cards: list[int]) -> np.ndarray:
    result = np.zeros(13, dtype=np.uint8)
    for rank in cards:
        result[rank - 1] += 1
    return result


def _initial_observation(player_count: int) -> V5PublicObservation:
    base, remainder = divmod(80, player_count)
    bonus_start = player_count - remainder
    remaining = [
        base + int(index >= bonus_start) for index in range(player_count)
    ]
    deck = [
        rank
        for rank, copies in enumerate(V5_DECK_COUNTS, start=1)
        for _ in range(copies)
    ]
    own = _rank_counts(deck[: remaining[0]])
    players = np.zeros((10, 6), dtype=np.uint8)
    for index, hand_count in enumerate(remaining):
        players[index] = (
            index,
            hand_count,
            _role_at(index, player_count),
            0,
            int(index % 3 == 1),
            0,
        )
    table = np.zeros(6, dtype=np.uint8)
    return validate_v5_public_observation(
        V5PublicObservation(
            global_codes=np.asarray(
                [5, player_count, 1, 0, 0, 0], dtype=np.int32
            ),
            own_rank_counts=own,
            public_played_counts=np.zeros(13, dtype=np.uint8),
            player_codes=players,
            player_mask=np.arange(10) < player_count,
            table_codes=table,
            history_codes=np.zeros((192, 12), dtype=np.uint8),
            history_mask=np.zeros(192, dtype=np.bool_),
            legal_mask=v5_legal_mask_from_public_cards(own, table),
        )
    )


def _tiny_exact_observation() -> V5PublicObservation:
    # Only rank 1, rank 2, and one joker are unseen.  Opponent offset 1 has two
    # of those three cards.  Against a pair of rank 3, exactly two of the three
    # equally likely hands ({1,J}, {2,J}) can respond.
    own = np.zeros(13, dtype=np.uint8)
    own[11] = 1
    unknown = np.zeros(13, dtype=np.uint8)
    unknown[0] = 1
    unknown[1] = 1
    unknown[12] = 1
    played = (
        np.asarray(V5_DECK_COUNTS, dtype=np.int16)
        - own.astype(np.int16)
        - unknown.astype(np.int16)
    ).astype(np.uint8)
    players = np.zeros((10, 6), dtype=np.uint8)
    remaining = (1, 2, 1, 0)
    for index, count in enumerate(remaining):
        players[index] = (
            index,
            count,
            _role_at(index, 4),
            int(count == 0),
            0,
            int(index == 3),
        )
    table = np.asarray([1, 3, 2, 2, 0, 3], dtype=np.uint8)
    return validate_v5_public_observation(
        V5PublicObservation(
            global_codes=np.asarray([5, 4, 7, 0, 0, 0], dtype=np.int32),
            own_rank_counts=own,
            public_played_counts=played,
            player_codes=players,
            player_mask=np.asarray(
                [True, True, True, True, False, False, False, False, False, False]
            ),
            table_codes=table,
            history_codes=np.zeros((192, 12), dtype=np.uint8),
            history_mask=np.zeros(192, dtype=np.bool_),
            legal_mask=v5_legal_mask_from_public_cards(own, table),
        )
    )


class V5PublicContractTests(unittest.TestCase):
    def test_contract_fingerprint_is_canonical_and_seals_privacy(self) -> None:
        first = canonical_v5_public_contract()
        second = canonical_v5_public_contract()
        self.assertEqual(first, second)
        self.assertEqual(first["contractSha256"], V5_PUBLIC_CONTRACT_SHA256)
        self.assertEqual(len(V5_PUBLIC_CONTRACT_SHA256), 64)
        self.assertEqual(validate_v5_public_contract(first), first)
        self.assertIn("opponentHands", first["privacy"]["forbiddenFields"])
        mutated = {**first, "version": int(first["version"]) + 1}
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            validate_v5_public_contract(mutated)

    def test_all_player_counts_four_through_ten_validate_and_tensorize(self) -> None:
        tensors = []
        for player_count in range(4, 11):
            observation = _initial_observation(player_count)
            belief = compute_v5_public_beliefs(observation)
            self.assertEqual(observation.player_count, player_count)
            self.assertEqual(int(belief.opponent_mask.sum()), player_count - 1)
            self.assertEqual(
                int(belief.unknown_rank_counts.sum()),
                sum(observation.opponent_remaining_counts),
            )
            self.assertTrue(np.isfinite(belief.expected_counts).all())
            tensors.append(tensorize_v5_public_observation(observation))

        # Same fixed shapes make a heterogeneous p4..p10 batch safe.
        batch = stack_v5_actor_public_features(tensors)
        self.assertEqual(batch.global_codes.shape, (7, 6))
        self.assertEqual(batch.player_codes.shape, (7, 10, 6))
        self.assertEqual(batch.history_codes.shape, (7, 192, 12))
        self.assertEqual(batch.belief_expected_counts.shape, (7, 9, 13))
        self.assertEqual(batch.global_codes.dtype, torch.int64)
        self.assertEqual(batch.belief_expected_counts.dtype, torch.float32)

    def test_exact_hypergeometric_features_and_joker_response_probability(self) -> None:
        observation = _tiny_exact_observation()
        belief = compute_v5_public_beliefs(observation)
        self.assertTrue(
            np.array_equal(
                belief.unknown_rank_counts,
                np.asarray([1, 1] + [0] * 10 + [1], dtype=np.uint8),
            )
        )
        for rank_index in (0, 1, 12):
            self.assertAlmostEqual(
                float(belief.expected_counts[0, rank_index]), 2.0 / 3.0, places=7
            )
            self.assertAlmostEqual(
                float(belief.probability_at_least_one[0, rank_index]),
                2.0 / 3.0,
                places=7,
            )
            self.assertEqual(
                float(belief.probability_at_least_required[0, rank_index]), 0.0
            )
        self.assertAlmostEqual(
            float(belief.response_feasibility[0]), 2.0 / 3.0, places=7
        )
        self.assertEqual(float(belief.response_feasibility[1]), 0.0)
        self.assertTrue(np.all(belief.expected_counts[3:] == 0.0))

    def test_hidden_hand_resampling_cannot_change_actor_bytes_or_beliefs(self) -> None:
        # Two distinct, physically consistent hidden worlds.  They remain
        # deliberately outside the returned actor record; only public hand
        # lengths survive projection.
        hidden_world_a = ((1, 13), (2,), ())
        hidden_world_b = ((1, 2), (13,), ())
        self.assertNotEqual(hidden_world_a, hidden_world_b)

        def public_projection(
            hidden_world: tuple[tuple[int, ...], ...]
        ) -> V5PublicObservation:
            base = _tiny_exact_observation()
            hidden_counts = _rank_counts(
                [rank for hand in hidden_world for rank in hand]
            )
            self.assertTrue(
                np.array_equal(
                    hidden_counts,
                    np.asarray([1, 1] + [0] * 10 + [1], dtype=np.uint8),
                )
            )
            players = base.player_codes.copy()
            for offset, hand in enumerate(hidden_world, start=1):
                players[offset, 1] = len(hand)
                players[offset, 3] = int(len(hand) == 0)
            return validate_v5_public_observation(
                replace(base, player_codes=players)
            )

        observation_a = public_projection(hidden_world_a)
        observation_b = public_projection(hidden_world_b)
        self.assertEqual(
            encode_v5_public_observation_bytes(observation_a),
            encode_v5_public_observation_bytes(observation_b),
        )
        belief_a = compute_v5_public_beliefs(observation_a)
        belief_b = compute_v5_public_beliefs(observation_b)
        for field_name in (
            "unknown_rank_counts",
            "expected_counts",
            "probability_at_least_one",
            "probability_at_least_required",
            "response_feasibility",
            "opponent_mask",
        ):
            self.assertTrue(
                np.array_equal(
                    getattr(belief_a, field_name), getattr(belief_b, field_name)
                )
            )

        leaked = {
            field_name: getattr(observation_a, field_name)
            for field_name in observation_a.__dataclass_fields__
        }
        leaked["opponent_hands"] = hidden_world_a
        with self.assertRaisesRegex(ValueError, "opponent_hands"):
            v5_public_observation_from_mapping(leaked)

    def test_dtype_shape_range_and_semantics_fail_closed(self) -> None:
        observation = _initial_observation(4)
        with self.assertRaisesRegex(TypeError, "dtype uint8"):
            validate_v5_public_observation(
                replace(
                    observation,
                    own_rank_counts=observation.own_rank_counts.astype(np.int16),
                )
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_v5_public_observation(
                replace(observation, player_mask=np.ones(9, dtype=np.bool_))
            )
        bad_padding = observation.player_codes.copy()
        bad_padding[9, 1] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_v5_public_observation(
                replace(observation, player_codes=bad_padding)
            )
        bad_supply = observation.public_played_counts.copy()
        bad_supply[0] = 1
        with self.assertRaisesRegex(ValueError, "physical deck"):
            validate_v5_public_observation(
                replace(observation, public_played_counts=bad_supply)
            )
        bad_legal = observation.legal_mask.copy()
        bad_legal[0] = True
        with self.assertRaisesRegex(ValueError, "legal_mask disagrees"):
            validate_v5_public_observation(
                replace(observation, legal_mask=bad_legal)
            )

    def test_compact_history_is_categorical_and_rejects_cross_type_payloads(self) -> None:
        observation = _tiny_exact_observation()
        history = observation.history_codes.copy()
        mask = observation.history_mask.copy()
        # A clear event can retain the public bundle and a next-leader offset.
        history[0] = (3, 3, 0, 0, 3, 2, 0, 2, 0, 1, 1, 0)
        mask[0] = True
        valid = validate_v5_public_observation(
            replace(observation, history_codes=history, history_mask=mask)
        )
        self.assertEqual(valid.history_codes.dtype, np.uint8)
        self.assertEqual(valid.history_codes.shape, (V5_MAX_HISTORY, 12))

        malformed = history.copy()
        malformed[0, 8] = 1  # pass reason on a clear token
        with self.assertRaisesRegex(ValueError, "non-clear"):
            validate_v5_public_observation(
                replace(observation, history_codes=malformed, history_mask=mask)
            )

    def test_v4_public_adapter_and_ragged_pack_use_dataset_ready_layout(self) -> None:
        converted = []
        for player_count in range(4, 11):
            public = DalmutiScalarEnv(
                player_count, acts=1, seed=8000 + player_count
            ).public_observation()
            converted.append(v5_public_from_v4_actor_observation(public))
        actor_arrays, history_events, history_end = pack_v5_public_observations(
            converted
        )
        direct_arrays, direct_events, direct_end = pack_v5_public_from_v4(
            [
                DalmutiScalarEnv(p, acts=1, seed=9000 + p).public_observation()
                for p in range(4, 11)
            ]
        )
        self.assertEqual(actor_arrays["global_codes"].shape, (7, 6))
        self.assertEqual(actor_arrays["legal_action_bits"].shape, (7, 30))
        self.assertEqual(actor_arrays["legal_action_bits"].dtype, np.uint8)
        self.assertEqual(history_events.shape, (0, 12))
        self.assertEqual(history_events.dtype, np.uint8)
        self.assertEqual(history_end.dtype, np.uint32)
        self.assertTrue(np.array_equal(history_end, np.zeros(7, dtype=np.uint32)))
        self.assertEqual(direct_arrays["global_codes"].shape, (7, 6))
        self.assertEqual(direct_events.dtype, np.uint8)
        self.assertEqual(direct_end.dtype, np.uint32)
        self.assertTrue(
            np.all(actor_arrays["legal_action_bits"][:, -1] & 0xF0 == 0)
        )

    def test_v4_clear_offset_zero_is_recovered_from_public_reason(self) -> None:
        env = DalmutiScalarEnv(4, acts=1, seed=20260802)
        public = env.public_observation()
        # Build a structurally valid V4 public view with a clear token whose
        # next leader is relative offset zero.  V4's scalar feature is also
        # zero for None, so the clear reason supplies the disambiguation.
        history = public.history_features.clone()
        history_mask = public.history_mask.clone()
        history[0, 2] = 1.0  # clear
        history[0, 4] = 0.0
        history[0, 5] = 1.0
        history[0, 6] = 1.0
        history[0, 7] = 3.0 / 13.0
        history[0, 8] = 2.0 / 14.0
        history[0, 10] = 2.0 / 14.0
        history[0, 15] = 1.0  # all-passed
        history[0, 18] = 0.0  # relative offset zero
        history_mask[0] = True
        adapted = v5_public_from_v4_actor_observation(
            replace(public, history_features=history, history_mask=history_mask)
        )
        self.assertEqual(int(adapted.history_codes[0, 10]), 1)

    def test_packed_actor_batch_roundtrip_and_private_keys_fail_closed(self) -> None:
        first = _initial_observation(7)
        second = _tiny_exact_observation()
        second_history = second.history_codes.copy()
        second_history_mask = second.history_mask.copy()
        second_history[0] = (3, 3, 0, 0, 3, 2, 0, 2, 0, 1, 1, 0)
        second_history_mask[0] = True
        second = validate_v5_public_observation(
            replace(
                second,
                history_codes=second_history,
                history_mask=second_history_mask,
            )
        )
        actor_arrays, history_events, history_end = pack_v5_public_observations(
            [first, second]
        )
        packed = {
            **actor_arrays,
            "history_events": history_events,
            "history_end": history_end,
            # Known actor-side labels may accompany the public features.
            "actions": np.asarray([1, 0], dtype=np.uint16),
        }
        batch = actor_batch_from_packed_arrays(
            packed, np.asarray([1, 0, 1], dtype=np.uint32), "cpu"
        )
        expected = stack_v5_actor_public_features(
            [
                tensorize_v5_public_observation(second),
                tensorize_v5_public_observation(first),
                tensorize_v5_public_observation(second),
            ]
        )
        for field_name in batch.__dataclass_fields__:
            self.assertTrue(
                torch.equal(
                    getattr(batch, field_name), getattr(expected, field_name)
                ),
                field_name,
            )
        self.assertEqual(batch.history_mask[:, 0].tolist(), [True, False, True])

        with self.assertRaisesRegex(ValueError, "privileged_states"):
            actor_batch_from_packed_arrays(
                {
                    **packed,
                    "privileged_states": np.zeros((2, 512), np.float32),
                },
                [0],
                "cpu",
            )
        with self.assertRaisesRegex(ValueError, "opponent_hands"):
            actor_batch_from_packed_arrays(
                {**packed, "opponent_hands": np.zeros((2, 3), np.uint8)},
                [0],
                "cpu",
            )
        malformed = dict(packed)
        malformed["history_end"] = history_end.astype(np.int64)
        with self.assertRaisesRegex(TypeError, "history_end.*uint32"):
            actor_batch_from_packed_arrays(malformed, [0], "cpu")


if __name__ == "__main__":
    unittest.main()
