from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError as error:  # Local syntax-only workstations may omit torch.
    raise unittest.SkipTest("torch is required for exact V4 search tests") from error

from v4_env import Card, DalmutiScalarEnv
from v4_exact_search_screen import (
    REPORT_FORMAT,
    REPORT_VERSION,
    SearchConfig,
    _rollout_root_action,
    evaluate_player_count_search,
    evaluate_root_actions,
    public_observation_sha256,
    validate_diagnostic_report,
    write_diagnostic_report_exclusive,
)


DECK_COUNTS = tuple(range(1, 13)) + (2,)


def _constant_clock() -> float:
    return 17.0


def _small_public_state(
    root_ranks: tuple[int, ...] = (11, 12),
    opponent_ranks: tuple[int, int, int] = (8, 9, 10),
) -> DalmutiScalarEnv:
    env = DalmutiScalarEnv(4, acts=1, seed=771_001, device="cpu")
    env._order = [0, 1, 2, 3]
    env._act = 1
    env._terminated = False
    env._scores = {player_id: 0 for player_id in range(4)}
    env._hands = {
        0: [Card(f"root-{index}", rank) for index, rank in enumerate(root_ranks)],
        **{
            player_id: [Card(f"opponent-{player_id}", rank)]
            for player_id, rank in enumerate(opponent_ranks, start=1)
        },
    }
    remaining = [0] * 13
    for hand in env._hands.values():
        for card in hand:
            remaining[card.rank - 1] += 1
    env._public_played = [
        copies - remaining[index] for index, copies in enumerate(DECK_COUNTS)
    ]
    env._revolution = 0
    env._tax_audit = ()
    env._finish_order = []
    env._passed = set()
    env._history = []
    env._event_sequence = 0
    env._table = None
    env._last_played_id = None
    env._current_index = 0
    env._transitions = 0
    env._last_act_result = None
    return env


def _semantic_result(result: object) -> dict[str, object]:
    value = result.report_value()
    value.pop("elapsedSeconds")
    return value


def _minimal_report() -> dict[str, object]:
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "certification": {
            "status": "research-only-not-certification",
            "promotionEligible": False,
        },
        "configuration": {},
        "privacyAudit": {
            "strictPublicInformationOnly": True,
            "liveOpponentOwnershipReadOrBranchedOn": False,
            "privilegedCriticStateConsumed": False,
            "canonicalOpponentPoolBuiltFromActorPublicCounts": True,
            "determinizationsUseDalmutiScalarEnvCopies": True,
            "resampleHiddenHandsCalledForEveryPublicHypothesis": True,
            "statement": "fixture",
        },
        "sourceHashes": {"fixture": "0" * 64},
        "results": [],
        "aggregate": {},
        "reproducibility": {},
    }


class ExactPublicRootSearchTests(unittest.TestCase):
    def test_seed_matched_screening_loop_reports_required_statistics(self) -> None:
        def factory(_: int, __: int, ___: int) -> DalmutiScalarEnv:
            return _small_public_state()

        result = evaluate_player_count_search(
            player_count=4,
            matches=2,
            acts=1,
            base_seed=380_000_001,
            search_config=SearchConfig(
                seed=771,
                hypotheses=1,
                selection="mean",
                max_rollout_steps=128,
            ),
            bootstrap_resamples=100,
            env_factory=factory,
            clock=_constant_clock,
        )
        self.assertEqual(result["playerCount"], 4)
        self.assertEqual(result["matches"], 2)
        self.assertEqual(result["actCount"], 2)
        self.assertEqual(result["meanChipDifferenceInference"]["clusters"], 2)
        self.assertGreater(
            result["pairwiseCandidateBeforeNormal"]["comparisons"], 0
        )
        audit = result["decisionAudit"]
        self.assertEqual(
            audit["candidateDecisions"],
            audit["forcedCandidateDecisions"] + audit["searchDecisions"],
        )
        work = result["searchWork"]
        self.assertGreater(work["rootActionEvaluations"], 0)
        self.assertGreater(work["simulatedSteps"], work["rootActionEvaluations"])

    def test_real_seed_initial_root_smoke(self) -> None:
        env = DalmutiScalarEnv(4, acts=5, seed=360_004_001, device="cpu")
        legal_count = int(env.legal_mask().sum().item())
        result = evaluate_root_actions(
            env,
            SearchConfig(
                seed=991_117,
                hypotheses=1,
                selection="mean",
                max_rollout_steps=2048,
            ),
        )
        self.assertFalse(result.forced)
        self.assertTrue(bool(env.legal_mask()[result.action].item()))
        self.assertEqual(result.determinizations, 1)
        self.assertEqual(result.root_action_evaluations, legal_count)
        self.assertGreater(result.simulated_steps, legal_count)

    def test_common_random_numbers_are_action_order_invariant_deterministic_and_legal(
        self,
    ) -> None:
        env = _small_public_state()
        config = SearchConfig(
            seed=991_117,
            hypotheses=3,
            selection="lcb",
            lcb_z=1.0,
            max_rollout_steps=128,
        )
        legal = tuple(int(value) for value in env.legal_mask().nonzero().flatten())
        first = evaluate_root_actions(env, config, clock=_constant_clock)
        reverse = evaluate_root_actions(
            env, config, action_order=tuple(reversed(legal)), clock=_constant_clock
        )
        replay = evaluate_root_actions(env, config, clock=_constant_clock)
        self.assertEqual(_semantic_result(first), _semantic_result(reverse))
        self.assertEqual(_semantic_result(first), _semantic_result(replay))
        self.assertIn(first.action, legal)
        self.assertEqual(first.determinizations, config.hypotheses)
        self.assertEqual(
            first.root_action_evaluations, config.hypotheses * len(legal)
        )
        self.assertTrue(all(stat.samples == config.hypotheses for stat in first.action_stats))

    def test_private_hidden_ownership_cannot_change_search_at_fixed_public_state(
        self,
    ) -> None:
        first_env = _small_public_state()
        second_env = copy.deepcopy(first_env)
        second_env._hands[1], second_env._hands[2] = (
            second_env._hands[2],
            second_env._hands[1],
        )
        self.assertEqual(
            public_observation_sha256(first_env),
            public_observation_sha256(second_env),
        )
        config = SearchConfig(seed=812_337, hypotheses=4, max_rollout_steps=128)
        first = evaluate_root_actions(first_env, config, clock=_constant_clock)
        second = evaluate_root_actions(second_env, config, clock=_constant_clock)
        self.assertEqual(_semantic_result(first), _semantic_result(second))

    def test_rollout_scores_root_actors_actual_terminal_act_outcome(self) -> None:
        # The root actor empties its hand immediately, but the rollout must wait
        # for the other three players so it can read the actual act result.
        env = _small_public_state(root_ranks=(12,), opponent_ranks=(9, 10, 11))
        root_actor = env.current_player_id
        action = env.normal_action()
        outcome = _rollout_root_action(
            env,
            action,
            root_actor_id=root_actor,
            max_rollout_steps=128,
        )
        self.assertEqual(outcome.finish_place, 1)
        self.assertEqual(outcome.chip_award, 4)
        self.assertEqual(outcome.environment_reward, 1.0)
        self.assertGreater(outcome.exact_normal_continuation_steps, 0)

    def test_forced_decision_uses_exact_normal_without_search(self) -> None:
        env = _small_public_state(root_ranks=(12,), opponent_ranks=(9, 10, 11))
        exact_normal = env.normal_action()
        with mock.patch.object(
            DalmutiScalarEnv,
            "resample_hidden_hands",
            side_effect=AssertionError("forced path must not determinize"),
        ):
            result = evaluate_root_actions(
                env,
                SearchConfig(seed=71, hypotheses=8, max_rollout_steps=128),
                clock=_constant_clock,
            )
        self.assertTrue(result.forced)
        self.assertEqual(result.action, exact_normal)
        self.assertFalse(result.deviated_from_normal)
        self.assertEqual(result.determinizations, 0)
        self.assertEqual(result.root_action_evaluations, 0)
        self.assertEqual(result.simulated_steps, 0)
        self.assertEqual(result.action_stats, ())

    def test_invalid_or_incomplete_action_order_is_rejected(self) -> None:
        env = _small_public_state()
        legal = tuple(int(value) for value in env.legal_mask().nonzero().flatten())
        with self.assertRaisesRegex(ValueError, "permutation"):
            evaluate_root_actions(
                env,
                SearchConfig(seed=19, hypotheses=1, max_rollout_steps=128),
                action_order=legal[:-1],
                clock=_constant_clock,
            )


class ExactSearchReportTests(unittest.TestCase):
    def test_report_and_checksum_are_exclusive_and_immutable(self) -> None:
        report = _minimal_report()
        validate_diagnostic_report(report)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic.json"
            published = write_diagnostic_report_exclusive(output, report)
            original = output.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            self.assertEqual(published["sha256"], digest)
            self.assertEqual(json.loads(original), report)
            self.assertEqual(
                Path(f"{output}.sha256").read_text(encoding="ascii"),
                f"{digest}  {output.name}\n",
            )
            with self.assertRaises(FileExistsError):
                write_diagnostic_report_exclusive(output, report)
            self.assertEqual(output.read_bytes(), original)

    def test_report_cannot_claim_certification_or_private_access(self) -> None:
        certified = _minimal_report()
        certified["certification"]["promotionEligible"] = True
        with self.assertRaisesRegex(ValueError, "must not claim"):
            validate_diagnostic_report(certified)
        private = _minimal_report()
        private["privacyAudit"]["privilegedCriticStateConsumed"] = True
        with self.assertRaisesRegex(ValueError, "privacy audit failed"):
            validate_diagnostic_report(private)


if __name__ == "__main__":
    unittest.main()
