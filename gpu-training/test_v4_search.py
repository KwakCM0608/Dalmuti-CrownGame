import copy
import json
import unittest
from typing import Mapping, Sequence

from v4_search import (
    V4_SEARCH_ACTION_COUNT,
    V4Determinization,
    V4LeafRequest,
    V4RolloutPolicy,
    V4SearchConfig,
    V4SearchLeaf,
    determinize_v4_unseen_hands,
    run_v4_search_teacher,
)


def observation_for_players(player_count: int) -> dict[str, object]:
    base = 80 // player_count
    remainder = 80 % player_count
    hand_counts = [
        base + (1 if offset >= player_count - remainder else 0)
        for offset in range(player_count)
    ]
    own_target = hand_counts[0]
    own_counts = [0] * 13
    remaining = own_target
    for rank_index in range(11, -1, -1):
        take = min(rank_index + 1, remaining)
        own_counts[rank_index] = take
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        take = min(2, remaining)
        own_counts[12] = take
        remaining -= take
    if remaining:
        raise AssertionError("test hand construction failed")
    players = [
        {
            "relativeOffset": offset,
            "handCount": hand_counts[offset],
            "finished": 0,
            "passed": 0,
            "self": 1 if offset == 0 else 0,
            "tableLeader": 0,
            "role": min(offset, 4),
            "score": offset,
        }
        for offset in range(player_count)
    ]
    return {
        "schemaVersion": 4,
        "playerCount": player_count,
        "act": 2,
        "actorRole": 0,
        "revolution": 0,
        "ownHandCounts": own_counts,
        "publicPlayedCounts": [0] * 13,
        "table": None,
        "playerTokens": players,
        "historyTokens": [],
        "memoryTraceVectors": [[0.0] * 20 for _ in range(4)],
        "truncatedHistoryCount": 0,
    }


def legal_mask(*actions: int) -> tuple[bool, ...]:
    mask = [False] * V4_SEARCH_ACTION_COUNT
    for action in actions:
        mask[action] = True
    return tuple(mask)


class FakeAdapter:
    def __init__(
        self,
        root_legal: Sequence[bool],
        *,
        leaf_mode: bool = False,
        malicious_policy_payload: bool = False,
    ) -> None:
        self.root_legal = tuple(root_legal)
        self.leaf_mode = leaf_mode
        self.malicious_policy_payload = malicious_policy_payload
        self.actions: list[int] = []
        self.policy_calls = 0

    def build_root(
        self,
        public_observation: Mapping[str, object],
        determinization: V4Determinization,
        seed: int,
    ) -> dict[str, object]:
        return {
            "public": copy.deepcopy(dict(public_observation)),
            "determinization": determinization,
            "seed": seed,
        }

    def legal_action_mask(self, root: dict[str, object]) -> Sequence[bool]:
        return self.root_legal

    def simulate_root_action(
        self,
        root: dict[str, object],
        action_index: int,
        rollout_policy: V4RolloutPolicy | None,
        max_rollout_steps: int,
        seed: int,
    ) -> V4SearchLeaf:
        if not self.root_legal[action_index]:
            raise AssertionError("search attempted an illegal action")
        self.actions.append(action_index)
        public = copy.deepcopy(root["public"])
        if rollout_policy is not None:
            self.policy_calls += 1
            if self.malicious_policy_payload:
                public["opponentHiddenHands"] = [[1, 2, 3]]
            scores = rollout_policy(public, self.root_legal)
            if len(scores) != 236:
                raise AssertionError("wrapped policy returned wrong shape")
        if self.leaf_mode:
            return V4SearchLeaf.evaluate(public, self.root_legal, depth=3)
        determinization = root["determinization"]
        signal = (int(determinization.sha256[:8], 16) % 2001) / 1000.0 - 1.0
        if action_index == 0:
            value = 2.0 + 3.0 * signal
        elif action_index == 1:
            value = 1.5
        elif action_index == 5:
            value = 4.0
        else:
            value = -float(action_index) / 100.0
        return V4SearchLeaf.terminal(value, depth=min(4, max_rollout_steps))


class AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 1.0
        return current


class V4SearchTests(unittest.TestCase):
    def test_determinization_is_seeded_and_satisfies_public_constraints_p4_to_p10(self) -> None:
        for player_count in range(4, 11):
            with self.subTest(player_count=player_count):
                observation = observation_for_players(player_count)
                first = determinize_v4_unseen_hands(observation, seed=991)
                second = determinize_v4_unseen_hands(observation, seed=991)
                self.assertEqual(first, second)
                self.assertEqual(
                    len(first.rank_counts_by_relative_offset), player_count
                )
                expected_hands = [
                    token["handCount"] for token in observation["playerTokens"]
                ]
                for offset, counts in enumerate(first.rank_counts_by_relative_offset):
                    self.assertEqual(sum(counts), expected_hands[offset])
                for rank_index, deck_count in enumerate(tuple(range(1, 13)) + (2,)):
                    dealt = sum(
                        hand[rank_index]
                        for hand in first.rank_counts_by_relative_offset
                    )
                    self.assertEqual(dealt, deck_count)

    def test_seeded_teacher_obeys_236_mask_and_emits_diagnostics(self) -> None:
        observation = observation_for_players(6)
        legal = legal_mask(0, 1, 5, 235)
        config = V4SearchConfig(
            seed=77,
            hypotheses=8,
            rollouts_per_action=1,
            max_evaluations=100,
            max_seconds=None,
            selection="mean",
            distribution_temperature=0.25,
        )
        adapter_a = FakeAdapter(legal)
        adapter_b = FakeAdapter(legal)
        first = run_v4_search_teacher(
            observation, legal, adapter_a, config=config, clock=lambda: 0.0
        )
        second = run_v4_search_teacher(
            observation, legal, adapter_b, config=config, clock=lambda: 0.0
        )

        self.assertEqual(first, second)
        self.assertEqual(first.teacher_action, 5)
        self.assertEqual(len(first.teacher_distribution), 236)
        self.assertAlmostEqual(sum(first.teacher_distribution), 1.0, places=12)
        self.assertTrue(all(legal[action] for action in adapter_a.actions))
        self.assertTrue(
            all(
                probability == 0.0
                for action, probability in enumerate(first.teacher_distribution)
                if not legal[action]
            )
        )
        self.assertEqual(first.diagnostics.player_count, 6)
        self.assertEqual(first.diagnostics.hypotheses_generated, 8)
        self.assertEqual(first.diagnostics.evaluations, 32)
        json.dumps(first.to_dict(), allow_nan=False)

    def test_mean_and_lcb_modes_use_cross_hypothesis_risk(self) -> None:
        observation = observation_for_players(5)
        legal = legal_mask(0, 1)
        common = dict(
            seed=101,
            hypotheses=24,
            rollouts_per_action=1,
            max_evaluations=100,
            max_seconds=None,
            distribution_temperature=0.1,
        )
        mean_result = run_v4_search_teacher(
            observation,
            legal,
            FakeAdapter(legal),
            config=V4SearchConfig(selection="mean", lcb_z=4.0, **common),
            clock=lambda: 0.0,
        )
        lcb_result = run_v4_search_teacher(
            observation,
            legal,
            FakeAdapter(legal),
            config=V4SearchConfig(selection="lcb", lcb_z=4.0, **common),
            clock=lambda: 0.0,
        )
        noisy_stats = lcb_result.action_stats[0]
        self.assertGreater(noisy_stats.standard_error, 0.0)
        self.assertLess(noisy_stats.lcb, noisy_stats.mean)
        self.assertAlmostEqual(mean_result.action_scores[0], noisy_stats.mean)
        self.assertAlmostEqual(lcb_result.action_scores[0], noisy_stats.lcb)
        self.assertEqual(mean_result.teacher_action, 0)
        self.assertEqual(lcb_result.teacher_action, 1)

    def test_batched_leaf_evaluator_and_rollout_policy_are_injectable(self) -> None:
        observation = observation_for_players(4)
        legal = legal_mask(0, 1, 5)
        adapter = FakeAdapter(legal, leaf_mode=True)
        evaluated_batches: list[int] = []

        def policy(
            public: Mapping[str, object], mask: tuple[bool, ...]
        ) -> Sequence[float]:
            self.assertEqual(set(public), set(observation))
            return [float(index) if mask[index] else -1.0e9 for index in range(236)]

        def evaluate(requests: Sequence[V4LeafRequest]) -> Sequence[float]:
            evaluated_batches.append(len(requests))
            return [10.0 if request.root_action == 5 else 0.0 for request in requests]

        result = run_v4_search_teacher(
            observation,
            legal,
            adapter,
            config=V4SearchConfig(
                seed=33,
                hypotheses=3,
                rollouts_per_action=1,
                max_evaluations=20,
                max_seconds=None,
                leaf_batch_size=4,
                selection="mean",
            ),
            rollout_policy=policy,
            batched_leaf_evaluator=evaluate,
            clock=lambda: 0.0,
        )
        self.assertEqual(result.teacher_action, 5)
        self.assertEqual(adapter.policy_calls, 9)
        self.assertEqual(sum(evaluated_batches), 9)
        self.assertEqual(result.diagnostics.batched_leaf_evaluations, 9)
        self.assertEqual(result.diagnostics.terminal_evaluations, 0)

    def test_private_or_privileged_payloads_are_rejected_at_every_boundary(self) -> None:
        observation = observation_for_players(4)
        legal = legal_mask(0, 1)
        hidden = copy.deepcopy(observation)
        hidden["opponentHiddenHands"] = [[1, 2, 3]]
        with self.assertRaisesRegex(ValueError, "public-information boundary"):
            run_v4_search_teacher(hidden, legal, FakeAdapter(legal))

        privileged = copy.deepcopy(observation)
        privileged["privilegedCriticState"] = [0.25, 0.75]
        with self.assertRaisesRegex(ValueError, "public-information boundary"):
            run_v4_search_teacher(privileged, legal, FakeAdapter(legal))

        nested = copy.deepcopy(observation)
        nested["playerTokens"][1]["privateCards"] = [1]
        with self.assertRaisesRegex(ValueError, "public-information boundary"):
            determinize_v4_unseen_hands(nested, seed=1)

        malicious = FakeAdapter(legal, malicious_policy_payload=True)
        with self.assertRaisesRegex(ValueError, "public-information boundary"):
            run_v4_search_teacher(
                observation,
                legal,
                malicious,
                config=V4SearchConfig(
                    hypotheses=1,
                    max_evaluations=2,
                    max_seconds=None,
                ),
                rollout_policy=lambda public, mask: [0.0] * 236,
            )

    def test_budget_and_time_limits_stop_without_crossing_the_hard_cap(self) -> None:
        observation = observation_for_players(7)
        legal = legal_mask(0, 1, 5)
        budget_result = run_v4_search_teacher(
            observation,
            legal,
            FakeAdapter(legal),
            config=V4SearchConfig(
                hypotheses=20,
                max_evaluations=5,
                max_seconds=None,
            ),
            clock=lambda: 0.0,
        )
        self.assertEqual(budget_result.diagnostics.evaluations, 5)
        self.assertEqual(budget_result.diagnostics.stopped_reason, "evaluation-budget")

        timed_result = run_v4_search_teacher(
            observation,
            legal,
            FakeAdapter(legal),
            config=V4SearchConfig(
                hypotheses=20,
                max_evaluations=100,
                max_seconds=1.5,
            ),
            clock=AdvancingClock(),
        )
        self.assertLess(timed_result.diagnostics.evaluations, 100)
        self.assertEqual(timed_result.diagnostics.stopped_reason, "time-budget")


if __name__ == "__main__":
    unittest.main()
