from __future__ import annotations

import copy
import unittest
from typing import Mapping, Sequence

from v4_env import encode_action, ranked_deal_counts, role_for_index, ROLES
from v4_search import (
    V4SearchConfig,
    determinize_v4_unseen_hands,
    run_v4_search_teacher,
    validate_v4_search_public_observation,
)
from v4_search_env_adapter import (
    DalmutiV4SearchEnvAdapter,
    V4SearchAdapterUnsupportedError,
)


def _role_id(position: int, player_count: int) -> int:
    return ROLES.index(role_for_index(position, player_count))


def initial_observation(player_count: int) -> dict[str, object]:
    hand_counts = ranked_deal_counts(80, player_count)
    own_counts = [0] * 13
    remaining = hand_counts[0]
    for rank_index in range(11, -1, -1):
        take = min(rank_index + 1, remaining)
        own_counts[rank_index] = take
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        own_counts[12] = remaining
        remaining = 0
    if remaining:
        raise AssertionError("could not construct the actor hand")
    return {
        "schemaVersion": 4,
        "playerCount": player_count,
        "act": 1,
        "actorRole": 0,
        "revolution": 0,
        "ownHandCounts": own_counts,
        "publicPlayedCounts": [0] * 13,
        "table": None,
        "playerTokens": [
            {
                "relativeOffset": offset,
                "handCount": hand_counts[offset],
                "finished": 0,
                "passed": 0,
                "self": int(offset == 0),
                "tableLeader": 0,
                "role": _role_id(offset, player_count),
                "score": offset,
            }
            for offset in range(player_count)
        ],
        "historyTokens": [],
        "memoryTraceVectors": [[0.0] * 20 for _ in range(4)],
        "truncatedHistoryCount": 0,
    }


def duel_observation(*, truncated: bool = False) -> dict[str, object]:
    own = [0] * 13
    own[0] = 1
    own[11] = 1
    public_played = [0, 1, *range(3, 12), 11, 2]
    if len(public_played) != 13 or sum(public_played) != 77:
        raise AssertionError("duel deck fixture is malformed")
    history = [
        {
            "sequence": 0,
            "type": 3,
            "actorOffset": 2,
            "handCountBefore": 0,
            "handCountAfter": 0,
            "rank": 0,
            "naturalCount": 0,
            "jokerCount": 0,
            "totalCount": 0,
            "passReason": 0,
            "clearReason": 0,
            "nextLeaderOffset": -1,
            "finishPlace": 1,
        },
        {
            "sequence": 1,
            "type": 3,
            "actorOffset": 3,
            "handCountBefore": 0,
            "handCountAfter": 0,
            "rank": 0,
            "naturalCount": 0,
            "jokerCount": 0,
            "totalCount": 0,
            "passReason": 0,
            "clearReason": 0,
            "nextLeaderOffset": -1,
            "finishPlace": 2,
        },
    ]
    return {
        "schemaVersion": 4,
        "playerCount": 4,
        "act": 3,
        "actorRole": 0,
        "revolution": 0,
        "ownHandCounts": own,
        "publicPlayedCounts": public_played,
        "table": None,
        "playerTokens": [
            {
                "relativeOffset": 0,
                "handCount": 2,
                "finished": 0,
                "passed": 0,
                "self": 1,
                "tableLeader": 0,
                "role": 0,
                "score": 3,
            },
            {
                "relativeOffset": 1,
                "handCount": 1,
                "finished": 0,
                "passed": 0,
                "self": 0,
                "tableLeader": 0,
                "role": 1,
                "score": 4,
            },
            {
                "relativeOffset": 2,
                "handCount": 0,
                "finished": 1,
                "passed": 0,
                "self": 0,
                "tableLeader": 0,
                "role": 3,
                "score": 5,
            },
            {
                "relativeOffset": 3,
                "handCount": 0,
                "finished": 1,
                "passed": 0,
                "self": 0,
                "tableLeader": 0,
                "role": 4,
                "score": 6,
            },
        ],
        "historyTokens": history,
        "memoryTraceVectors": (
            [[0.125] * 20 for _ in range(4)]
            if truncated
            else [[0.0] * 20 for _ in range(4)]
        ),
        "truncatedHistoryCount": int(truncated),
    }


def search_advantage_observation() -> dict[str, object]:
    observation = initial_observation(4)
    observation["ownHandCounts"] = [1, 0, 2, 0, 1, 1, 3, 0, 2, 4, 2, 4, 0]
    for token in observation["playerTokens"]:
        token["score"] = 0
    return observation


class V4SearchEnvironmentAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = DalmutiV4SearchEnvAdapter()

    def _root(self, observation: dict[str, object], *, seed: int = 71):
        determinization = determinize_v4_unseen_hands(observation, seed=seed)
        return self.adapter.build_root(observation, determinization, seed)

    def test_root_reconstruction_and_legal_masks_are_exact_for_p4_and_p10(self) -> None:
        for player_count in (4, 10):
            with self.subTest(player_count=player_count):
                observation = initial_observation(player_count)
                root = self._root(observation, seed=100 + player_count)
                self.assertEqual(self.adapter.public_observation(root), observation)
                mask = self.adapter.legal_action_mask(root)
                self.assertEqual(len(mask), 236)
                self.assertEqual(mask, tuple(bool(value) for value in root.env.legal_mask()))
                self.assertGreater(sum(mask), 1)
                self.assertEqual(root.env.current_player_id, 0)
                self.assertEqual(root.env.physical_card_count, 80)

    def test_roots_are_isolated_and_seeded_rollouts_are_deterministic(self) -> None:
        observation = initial_observation(6)
        determinization = determinize_v4_unseen_hands(observation, seed=991)
        first = self.adapter.build_root(observation, determinization, 444)
        second = self.adapter.build_root(observation, determinization, 444)
        untouched = second.env.state_fingerprint()
        action = next(
            index
            for index, legal in enumerate(self.adapter.legal_action_mask(first))
            if legal
        )
        first_leaf = self.adapter.simulate_root_action(first, action, None, 1, 444)
        self.assertIsNone(first_leaf.terminal_value)
        self.assertEqual(second.env.state_fingerprint(), untouched)
        self.assertNotEqual(first.env.state_fingerprint(), untouched)

        replay = self.adapter.build_root(observation, determinization, 444)
        replay_leaf = self.adapter.simulate_root_action(replay, action, None, 1, 444)
        self.assertEqual(first_leaf, replay_leaf)
        self.assertEqual(first.env.state_fingerprint(), replay.env.state_fingerprint())

    def test_injected_policy_receives_only_exact_rebased_public_state(self) -> None:
        observation = initial_observation(5)
        root = self._root(observation, seed=803)
        initial_action = next(
            index
            for index, legal in enumerate(self.adapter.legal_action_mask(root))
            if legal
        )
        calls: list[dict[str, object]] = []

        def public_policy(
            public: Mapping[str, object], legal: tuple[bool, ...]
        ) -> Sequence[float]:
            validate_v4_search_public_observation(public)
            self.assertNotIn("privilegedCriticState", public)
            self.assertNotIn("opponentHiddenHands", public)
            self.assertEqual(public["playerTokens"][0]["self"], 1)
            calls.append(copy.deepcopy(dict(public)))
            return [float(index) if legal[index] else -1.0e9 for index in range(236)]

        leaf = self.adapter.simulate_root_action(
            root, initial_action, public_policy, 2, 803
        )
        self.assertEqual(leaf.depth, 2)
        self.assertTrue(calls)
        self.assertIsNotNone(leaf.public_observation)
        validate_v4_search_public_observation(leaf.public_observation)

    def test_illegal_actions_and_private_payloads_are_rejected(self) -> None:
        observation = initial_observation(4)
        root = self._root(observation)
        illegal = next(
            index
            for index, legal in enumerate(self.adapter.legal_action_mask(root))
            if not legal
        )
        with self.assertRaisesRegex(ValueError, "illegal root action"):
            self.adapter.simulate_root_action(root, illegal, None, 10, 1)

        private = copy.deepcopy(observation)
        private["opponentHiddenHands"] = [[1, 2, 3]]
        determinization = determinize_v4_unseen_hands(observation, seed=1)
        with self.assertRaisesRegex(ValueError, "public-information boundary"):
            self.adapter.build_root(private, determinization, 1)

        privileged = copy.deepcopy(observation)
        privileged["privilegedCriticState"] = [0.0] * 512
        with self.assertRaisesRegex(ValueError, "public-information boundary"):
            self.adapter.build_root(privileged, determinization, 1)

    def test_compressed_history_is_strictly_terminal_only(self) -> None:
        observation = duel_observation(truncated=True)
        play_one = encode_action(1, 1)

        policy_root = self._root(observation, seed=10)
        with self.assertRaisesRegex(
            V4SearchAdapterUnsupportedError, "injected public rollout"
        ):
            self.adapter.simulate_root_action(
                policy_root,
                play_one,
                lambda public, legal: [0.0] * 236,
                10,
                10,
            )

        capped_root = self._root(observation, seed=10)
        with self.assertRaisesRegex(
            V4SearchAdapterUnsupportedError, "did not reach the act terminal"
        ):
            self.adapter.simulate_root_action(capped_root, play_one, None, 1, 10)

        terminal_root = self._root(observation, seed=10)
        terminal = self.adapter.simulate_root_action(
            terminal_root, play_one, None, 10, 10
        )
        self.assertEqual(terminal.terminal_value, -0.5)
        self.assertIsNone(terminal.public_observation)

    def test_real_teacher_selects_a_higher_normal_rollout_value_than_normal(self) -> None:
        observation = search_advantage_observation()
        root = self._root(observation, seed=701000)
        normal_action = root.env.normal_action()
        self.assertEqual(normal_action, 209)

        legal = self.adapter.legal_action_mask(root)
        result = run_v4_search_teacher(
            observation,
            legal,
            self.adapter,
            config=V4SearchConfig(
                seed=701000,
                hypotheses=4,
                rollouts_per_action=1,
                max_evaluations=200,
                max_seconds=None,
                max_rollout_steps=256,
                selection="mean",
                distribution_temperature=0.1,
            ),
            clock=lambda: 0.0,
        )
        self.assertEqual(result.teacher_action, 113)
        self.assertGreater(
            result.action_scores[result.teacher_action],
            result.action_scores[normal_action],
        )
        self.assertEqual(result.action_scores[normal_action], 0.5)
        self.assertEqual(result.action_scores[result.teacher_action], 0.75)


if __name__ == "__main__":
    unittest.main()
