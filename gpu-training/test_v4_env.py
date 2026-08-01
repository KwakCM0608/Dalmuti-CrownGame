from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as error:  # Local syntax-only workstations may omit torch.
    raise unittest.SkipTest("torch is required for V4 environment tests") from error

from v4_env import (
    ACTION_CATALOGUE,
    ACTION_COUNT,
    Card,
    DalmutiBatchEnv,
    DalmutiScalarEnv,
    Mulberry32,
    NormalObservation,
    NormalPublicPlayer,
    PASS_ACTION_INDEX,
    PRIVILEGED_LAYOUT,
    PRIVILEGED_GLOBAL_FIELDS,
    PRIVILEGED_PLAYER_OFFSET,
    PRIVILEGED_PLAYER_STRIDE,
    PRIVILEGED_PUBLIC_RANK_OFFSET,
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
    PRIVILEGED_STATE_SIZE,
    SOLO_JOKER_ACTION_INDEX,
    TablePlay,
    choose_normal_action,
    create_deck,
    encode_action,
    legal_action_masks,
    normal_revolution_decision,
    normal_tax_return_card_ids,
    ranked_deal_counts,
    round_chip_award,
)


def _assert_public_equal(
    testcase: unittest.TestCase, left: object, right: object
) -> None:
    testcase.assertEqual(left.actor_id, right.actor_id)
    for name in (
        "valid",
        "global_features",
        "rank_features",
        "player_features",
        "player_mask",
        "memory_trace_features",
        "history_features",
        "history_mask",
        "legal_mask",
    ):
        testcase.assertTrue(
            torch.equal(getattr(left, name), getattr(right, name)),
            f"public tensor changed: {name}",
        )


class DeckAndRandomTests(unittest.TestCase):
    def test_deck_and_ranked_deals_for_every_supported_table(self) -> None:
        deck = create_deck()
        self.assertEqual(len(deck), 80)
        self.assertEqual(len({card.id for card in deck}), 80)
        for rank in range(1, 13):
            self.assertEqual(sum(card.rank == rank for card in deck), rank)
        self.assertEqual(sum(card.rank == 13 for card in deck), 2)

        for player_count in range(4, 11):
            expected = ranked_deal_counts(80, player_count)
            env = DalmutiScalarEnv(player_count, acts=1, seed=900 + player_count)
            actual = env.hand_counts.sum(dim=-1).cpu().tolist()
            self.assertEqual(actual, expected)
            self.assertEqual(sum(actual), 80)
            self.assertEqual(env.physical_card_count, 80)

    def test_mulberry32_matches_typescript_golden_vector(self) -> None:
        random = Mulberry32(123_456_789)
        self.assertEqual(
            [random.next_uint32() for _ in range(5)],
            [1107202814, 4169434471, 3372958138, 885470128, 1301683845],
        )

    def test_same_seed_replays_identically(self) -> None:
        left = DalmutiScalarEnv(7, acts=2, seed=551_020)
        right = DalmutiScalarEnv(7, acts=2, seed=551_020)
        self.assertEqual(left.state_fingerprint(), right.state_fingerprint())
        for _ in range(40):
            action = left.normal_action()
            self.assertEqual(action, right.normal_action())
            left_result = left.step(action)
            right_result = right.step(action)
            self.assertEqual(left.state_fingerprint(), right.state_fingerprint())
            self.assertTrue(torch.equal(left_result.rewards, right_result.rewards))
            if left_result.terminated:
                break


class ActionRuleTests(unittest.TestCase):
    def test_catalogue_and_known_legal_masks(self) -> None:
        self.assertEqual(len(ACTION_CATALOGUE), ACTION_COUNT)
        self.assertEqual(ACTION_COUNT, 236)

        lead_hand = torch.zeros(13, dtype=torch.int16)
        lead_hand[2] = 3  # rank 3 has three natural cards
        lead_hand[12] = 2
        lead = legal_action_masks(lead_hand)
        self.assertFalse(bool(lead[PASS_ACTION_INDEX]))
        self.assertTrue(bool(lead[SOLO_JOKER_ACTION_INDEX]))
        expected = {SOLO_JOKER_ACTION_INDEX}
        expected.update(
            encode_action(3, naturals, jokers)
            for naturals in range(1, 4)
            for jokers in range(3)
        )
        self.assertEqual(set(lead.nonzero().flatten().tolist()), expected)

        response_hand = torch.zeros(13, dtype=torch.int16)
        response_hand[2] = 2
        response_hand[12] = 1
        response = legal_action_masks(
            response_hand, torch.tensor(5), torch.tensor(2)
        )
        self.assertEqual(
            set(response.nonzero().flatten().tolist()),
            {
                PASS_ACTION_INDEX,
                encode_action(3, 2, 0),
                encode_action(3, 1, 1),
            },
        )

    def test_joker_only_pair_is_never_legal(self) -> None:
        hand = torch.zeros(13, dtype=torch.int16)
        hand[12] = 2
        lead = legal_action_masks(hand)
        self.assertEqual(lead.nonzero().flatten().tolist(), [SOLO_JOKER_ACTION_INDEX])
        response = legal_action_masks(hand, torch.tensor(12), torch.tensor(2))
        self.assertEqual(response.nonzero().flatten().tolist(), [PASS_ACTION_INDEX])
        self.assertFalse(
            any(
                action.rank == 13 and action.count == 2
                for action in ACTION_CATALOGUE
            )
        )

    def test_normal_policy_matches_typescript_golden_decisions(self) -> None:
        hand = (
            Card("1-0", 1),
            Card("5-0", 5),
            Card("5-1", 5),
            Card("12-0", 12),
            Card("joker-1", 13),
        )
        players = (
            NormalPublicPlayer(0, 5),
            NormalPublicPlayer(1, 4),
            NormalPublicPlayer(2, 2),
            NormalPublicPlayer(3, 7),
        )
        lead = choose_normal_action(NormalObservation(0, hand, None, players))
        self.assertEqual(lead.action_index, encode_action(12, 1, 0))
        self.assertEqual(lead.score, 66.0)
        response = choose_normal_action(
            NormalObservation(0, hand, TablePlay(10, 2, 2), players)
        )
        self.assertEqual(response.action_index, encode_action(5, 2, 0))
        self.assertEqual(response.score, 118.0)

    def test_normal_tax_and_revolution_helpers_match_house_rules(self) -> None:
        hand = (
            Card("1-0", 1),
            Card("5-0", 5),
            Card("5-1", 5),
            Card("9-0", 9),
            Card("12-0", 12),
            Card("joker-1", 13),
        )
        self.assertEqual(normal_tax_return_card_ids(hand, 2), ("9-0", "12-0"))
        revolution_hand = (*hand, Card("joker-2", 13))
        self.assertEqual(normal_revolution_decision(revolution_hand, "great-peon"), 2)
        self.assertEqual(normal_revolution_decision(revolution_hand, "lesser-peon"), 1)
        self.assertEqual(normal_revolution_decision(revolution_hand, "merchant"), 0)
        self.assertEqual(normal_revolution_decision(revolution_hand, "great-dalmuti"), 0)

    def test_dalmuti_play_clears_and_returns_lead(self) -> None:
        env = None
        for seed in range(1, 100):
            candidate = DalmutiScalarEnv(4, acts=1, seed=seed)
            actor_hand = candidate._hands[candidate.current_player_id]
            if any(card.rank == 1 for card in actor_hand) and len(actor_hand) > 1:
                env = candidate
                break
        self.assertIsNotNone(env)
        assert env is not None
        acting_id = env.current_player_id
        result = env.step(encode_action(1, 1, 0))
        self.assertFalse(result.act_ended)
        self.assertEqual(env.current_player_id, acting_id)
        self.assertIsNone(env._table)
        self.assertEqual(env.physical_card_count, 80)
        self.assertTrue(
            any(
                event["type"] == "clear" and event["clear_reason"] == "dalmuti"
                for event in env._history
            )
        )
        dalmuti_passes = [
            event
            for event in env._history
            if event["type"] == "pass" and event["pass_reason"] == "dalmuti"
        ]
        self.assertEqual(len(dalmuti_passes), 3)


class MatchAndBoundaryTests(unittest.TestCase):
    def test_batch_terminal_lane_auto_resets_deterministically(self) -> None:
        batch = DalmutiBatchEnv(4, batch_size=1, acts=1, seeds=[31337])
        terminal = None
        for _ in range(20_000):
            terminal = batch.step(batch.normal_actions())
            if bool(terminal.terminated[0]):
                break
        assert terminal is not None
        self.assertTrue(bool(terminal.terminated[0]))
        self.assertTrue(bool(terminal.observation.public.valid[0]))
        expected_seed = (31337 + 0x9E37_79B9) & 0xFFFF_FFFF
        self.assertEqual(terminal.infos[0]["auto_reset_seed"], expected_seed)
        self.assertFalse(batch.envs[0].terminated)

    def test_three_act_tax_and_great_revolution_match_typescript(self) -> None:
        env = DalmutiScalarEnv(4, acts=3, seed=24680)
        results = []
        while not env.terminated:
            step = env.step(env.normal_action())
            if step.act_ended:
                results.append(step.info["act_result"])
        self.assertEqual(
            [result["player_order"] for result in results],
            [
                (3, 0, 2, 1),
                (0, 1, 2, 3),
                (2, 1, 3, 0),
            ],
        )
        self.assertEqual(
            [result["finish_order"] for result in results],
            [
                (3, 2, 1, 0),
                (2, 1, 3, 0),
                (1, 2, 0, 3),
            ],
        )
        self.assertEqual([result["revolution"] for result in results], [0, 2, 0])
        self.assertEqual([result["transitions"] for result in results], [73, 73, 83])
        self.assertEqual(dict(sorted(env._scores.items())), {0: 1, 1: 8, 2: 10, 3: 5})
        self.assertEqual(len(results[0]["taxation"]), 0)
        self.assertEqual(len(results[1]["taxation"]), 0)
        self.assertEqual(len(results[2]["taxation"]), 2)

    def test_complete_matches_match_typescript_reference_goldens(self) -> None:
        # Generated directly from training/simulator.ts with all difficulties
        # set to Normal. Player ids are converted from player-1..N to 0..N-1.
        goldens = (
            (4, 12345, (1, 2, 0, 3), (0, 1, 2, 3), 69),
            (7, 54321, (5, 4, 1, 6, 0, 2, 3), (3, 0, 4, 1, 5, 6, 2), 150),
            (
                10,
                998877,
                (7, 3, 0, 9, 1, 6, 2, 8, 4, 5),
                (2, 5, 6, 8, 4, 7, 0, 9, 1, 3),
                204,
            ),
        )
        for player_count, seed, player_order, finish_order, transitions in goldens:
            env = DalmutiScalarEnv(player_count, acts=1, seed=seed)
            self.assertEqual(tuple(env._order), player_order)
            result = None
            while not env.terminated:
                result = env.step(env.normal_action())
            assert result is not None
            act_result = result.info["act_result"]
            self.assertEqual(act_result["finish_order"], finish_order)
            self.assertEqual(act_result["transitions"], transitions)
            self.assertEqual(
                [act_result["chip_awards"][player_id] for player_id in finish_order],
                [
                    round_chip_award(place, player_count)
                    for place in range(1, player_count + 1)
                ],
            )

    def test_normal_match_finishes_with_exact_chip_curve(self) -> None:
        for player_count in (4, 7, 10):
            env = DalmutiScalarEnv(player_count, acts=1, seed=70_000 + player_count)
            result = None
            for _ in range(20_000):
                result = env.step(env.normal_action())
                if result.terminated:
                    break
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.terminated)
            act_result = result.info["act_result"]
            finish_order = act_result["finish_order"]
            chips = act_result["chip_awards"]
            self.assertEqual(len(finish_order), player_count)
            self.assertEqual(len(set(finish_order)), player_count)
            self.assertEqual(
                [chips[player_id] for player_id in finish_order],
                [round_chip_award(place, player_count) for place in range(1, player_count + 1)],
            )
            self.assertEqual(sum(chips.values()), player_count * 2)
            self.assertEqual(env.physical_card_count, 80)

    def test_hidden_hand_resample_cannot_change_actor_tensors(self) -> None:
        env = DalmutiScalarEnv(4, acts=1, seed=123_987)
        before = env.observe()
        self.assertEqual(before.public.global_features.shape, (12,))
        self.assertEqual(before.public.rank_features.shape, (13, 6))
        self.assertEqual(before.public.player_features.shape, (10, 12))
        self.assertEqual(before.public.memory_trace_features.shape, (4, 20))
        self.assertEqual(before.public.history_features.shape, (192, 20))
        self.assertEqual(before.public.legal_mask.shape, (236,))
        self.assertEqual(before.public.global_features.dtype, torch.float32)
        after = env.resample_hidden_hands(908_172)
        _assert_public_equal(self, before.public, after.public)
        self.assertFalse(
            torch.equal(before.privileged_state, after.privileged_state),
            "privileged critic state should observe changed opponent ownership",
        )
        self.assertEqual(before.privileged_state.shape, (PRIVILEGED_STATE_SIZE,))
        self.assertEqual(len(PRIVILEGED_GLOBAL_FIELDS), 16)
        reserved_start, reserved_end = PRIVILEGED_LAYOUT["reserved_zero_tail"]
        self.assertTrue(
            torch.equal(
                after.privileged_state[reserved_start:reserved_end],
                torch.zeros(reserved_end - reserved_start),
            )
        )

    def test_privileged_state_matches_typescript_raw_layout(self) -> None:
        env = DalmutiScalarEnv(4, acts=1, seed=123_987)
        state = env.privileged_state()
        self.assertEqual(
            PRIVILEGED_STATE_LAYOUT_ID,
            "dalmuti-v4-ts-privileged-critic-raw-v1",
        )
        self.assertEqual(
            PRIVILEGED_STATE_LAYOUT_SHA256,
            "be332c07e1753b6e87082917bbf5528faef8fed3cda794c853f655d3ade0110f",
        )
        self.assertEqual(
            state[:16].tolist(),
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
             0.0, 4.0, 0.0, 0.0, 0.0, 20.0, 0.0],
        )
        self.assertEqual(
            state[PRIVILEGED_PUBLIC_RANK_OFFSET:PRIVILEGED_PLAYER_OFFSET].tolist(),
            [0.0] * 13,
        )
        role_ids = [0, 1, 3, 4]
        total_hidden_cards = 0
        for offset, role_id in enumerate(role_ids):
            start = PRIVILEGED_PLAYER_OFFSET + offset * PRIVILEGED_PLAYER_STRIDE
            row = state[start:start + PRIVILEGED_PLAYER_STRIDE].tolist()
            self.assertEqual(row[0:2], [1.0, float(offset)])
            self.assertEqual(
                row[2:7], [float(index == role_id) for index in range(5)]
            )
            self.assertEqual(row[7:12], [0.0, 20.0, 0.0, 0.0, 0.0])
            self.assertEqual(sum(row[12:25]), 20.0)
            total_hidden_cards += int(sum(row[12:25]))
        self.assertEqual(total_hidden_cards, 80)

    def test_scalar_and_batched_lanes_have_transition_parity(self) -> None:
        counts = [4, 6]
        seeds = [81_001, 81_002]
        scalars = [
            DalmutiScalarEnv(count, acts=2, seed=seed)
            for count, seed in zip(counts, seeds)
        ]
        batch = DalmutiBatchEnv(counts, acts=2, seeds=seeds)
        initial = batch.reset(seeds)
        self.assertEqual(initial.public.legal_masks.shape, (2, ACTION_COUNT))
        self.assertEqual(initial.privileged_states.shape, (2, PRIVILEGED_STATE_SIZE))

        for _ in range(50):
            actions = [env.normal_action() for env in scalars]
            scalar_results = [env.step(action) for env, action in zip(scalars, actions)]
            batch_result = batch.step(torch.tensor(actions, dtype=torch.long))
            for lane, (scalar, lane_env) in enumerate(zip(scalar_results, batch.envs)):
                self.assertEqual(
                    scalars[lane].state_fingerprint(), lane_env.state_fingerprint()
                )
                self.assertTrue(torch.equal(scalar.rewards, batch_result.rewards[lane]))
                _assert_public_equal(
                    self, scalar.observation.public, lane_env.observe().public
                )
            if any(result.terminated for result in scalar_results):
                break


if __name__ == "__main__":
    unittest.main()
