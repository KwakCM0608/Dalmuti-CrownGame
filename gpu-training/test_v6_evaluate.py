from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


GPU_TRAINING = Path(__file__).resolve().parent
if str(GPU_TRAINING) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINING))

import v6_evaluate as evaluate


ACTS_PER_MATCH = 5


@dataclass(frozen=True)
class _FakePublic:
    legal_mask: np.ndarray


def _candidate_seats(player_count: int, match_index: int) -> tuple[int, ...]:
    lower = player_count // 2
    count = lower if player_count % 2 == 0 or match_index % 2 == 1 else lower + 1
    extras_before = (match_index + 1) // 2 if player_count % 2 else 0
    start = (match_index * lower + extras_before) % player_count
    return tuple((start + offset) % player_count for offset in range(count))


def _components(
    finish: list[int], awards: dict[int, int], candidate: list[int]
) -> tuple[float, float, float, int, int, float]:
    candidate_set = set(candidate)
    normal = set(finish) - candidate_set
    candidate_mean = sum(awards[value] for value in candidate_set) / len(candidate_set)
    normal_mean = sum(awards[value] for value in normal) / len(normal)
    positions = {value: index for index, value in enumerate(finish)}
    before = sum(
        int(positions[left] < positions[right])
        for left in candidate_set for right in normal
    )
    comparisons = len(candidate_set) * len(normal)
    return (
        candidate_mean, normal_mean, candidate_mean - normal_mean,
        before, comparisons, before / comparisons,
    )


def _identity() -> dict[str, object]:
    return {
        "format": evaluate.V6_CANDIDATE_FORMAT,
        "version": 1,
        "baseActor": {
            "actorSha256": "a" * 64,
            "manifestSha256": "b" * 64,
            "tensorStateSha256": "c" * 64,
            "publicContractSha256": "d" * 64,
            "policyNumericsSha256": "e" * 64,
        },
        "publicDelta": {
            "checkpointSha256": "f" * 64,
            "format": "unit-public-delta",
            "version": 1,
            "kind": "public-delta-heads-only",
            "tensorStateSha256": "1" * 64,
        },
        "scorerConfigSha256": "2" * 64,
        "safeOverrideContract": evaluate.SAFE_OVERRIDE_CONTRACT,
    }


class _DeterministicPublicScorer:
    public_only_contract = evaluate.V6_PUBLIC_SCORER_ADAPTER_CONTRACT

    def __init__(self, alternative_score: float = 3.0) -> None:
        self.alternative_score = alternative_score
        self.calls: list[int] = []

    def score_public(
        self, observations: list[object]
    ) -> evaluate.V6PublicScoreBatch:
        self.calls.append(len(observations))
        legal = [np.flatnonzero(value.legal_mask) for value in observations]
        width = max(len(value) for value in legal)
        indices = np.zeros((len(legal), width), dtype=np.int64)
        mask = np.zeros((len(legal), width), dtype=np.bool_)
        scores = np.zeros((len(legal), width, 3), dtype=np.float64)
        for row, actions in enumerate(legal):
            indices[row, : len(actions)] = actions
            mask[row, : len(actions)] = True
            # A deterministic alternative has a high score; the Normal-relative
            # gate still decides whether it may replace production Normal.
            if len(actions) > 1:
                scores[row, 0] = self.alternative_score
        return evaluate.V6PublicScoreBatch(indices, mask, scores)


def _public(seed: int = 9191) -> tuple[_FakePublic, int]:
    # Vary legal IDs while keeping the fixture independent from PyTorch.
    start = seed % 20
    legal = np.zeros(236, dtype=np.bool_)
    legal[[start, start + 3, start + 9]] = True
    return _FakePublic(legal), start + 3


def _real_public(seed: int = 9191):  # type: ignore[no-untyped-def]
    from v4_env import DalmutiScalarEnv
    from v5_public import v5_public_from_v4_actor_observation

    env = DalmutiScalarEnv(4, acts=1, seed=seed, device="cpu")
    return (
        v5_public_from_v4_actor_observation(env.public_observation()),
        int(env.normal_action()),
    )


def _synthetic_cluster(
    family_id: str,
    seed_base: int,
    player_count: int,
    match_index: int,
) -> dict[str, object]:
    initial = list(range(player_count))
    seats = list(_candidate_seats(player_count, match_index))
    candidate = sorted(initial[seat] for seat in seats)
    normal = [value for value in initial if value not in candidate]
    # Alternate which group is first; the exact score is reconstructed by V5.
    finish = candidate + normal if match_index % 2 == 0 else normal + candidate
    awards = {
        actor: (
            4 if place == 1 else 3 if place == 2 else
            1 if place == player_count - 1 else 0 if place == player_count else 2
        )
        for place, actor in enumerate(finish, start=1)
    }
    components = _components(finish, awards, candidate)
    acts = [{
        "act": act,
        "finishOrder": finish,
        "candidatePhysicalIds": candidate,
        "candidateMeanChip": components[0],
        "normalMeanChip": components[1],
        "meanChipDifference": components[2],
        "candidateBefore": components[3],
        "comparisons": components[4],
        "pairwiseRate": components[5],
    } for act in range(1, ACTS_PER_MATCH + 1)]
    return {
        "playerCount": player_count,
        "matchIndex": match_index,
        "seed": evaluate.derive_v5_evaluation_seed(
            family_id, seed_base, player_count, match_index
        ),
        "initialOrder": initial,
        "candidateInitialSeats": seats,
        "candidatePhysicalIds": candidate,
        "acts": acts,
        "decisions": 100,
        "meanChipDifference": components[2],
        "candidateBefore": components[3] * ACTS_PER_MATCH,
        "comparisons": components[4] * ACTS_PER_MATCH,
    }


def _calibration_report() -> dict[str, object]:
    plan = evaluate.V6CalibrationPlan(
        "unit-calibration", 12345, evaluate.V6PolicyParameters(1.25, 0.4),
        lane_count=1, bootstrap_resamples=100,
    )
    clusters = tuple(
        _synthetic_cluster(
            plan.family_id, plan.seed_base, player_count, match_index
        )
        for player_count, count in evaluate.CALIBRATION_MATCH_COUNTS.items()
        for match_index in range(count)
    )
    return evaluate.summarize_v6_evaluation(
        evaluate.V6EvaluationCollection(clusters, ()), plan, _identity()
    )


class PurePolicyTests(unittest.TestCase):
    def test_positive_infinity_is_exact_normal_and_forced_rows_bypass_scorer(self) -> None:
        observation, normal = _public()
        scorer = _DeterministicPublicScorer(alternative_score=1.0e12)
        policy = evaluate.V6SafeOverridePolicy(
            scorer, evaluate.V6PolicyParameters(beta=0.0, threshold=math.inf)
        )
        result = policy.actions([observation], [normal])
        self.assertEqual(result.actions, (normal,))
        self.assertFalse(result.audits[0]["deviated"])

        forced_mask = np.zeros_like(observation.legal_mask)
        forced_mask[normal] = True
        forced = replace(observation, legal_mask=forced_mask)
        before = list(scorer.calls)
        forced_result = policy.actions([forced], [normal])
        self.assertEqual(forced_result.actions, (normal,))
        self.assertTrue(forced_result.audits[0]["forced"])
        self.assertEqual(scorer.calls, before)

    def test_same_public_input_is_deterministic_and_audit_is_public_only(self) -> None:
        observation, normal = _public(8282)
        policy = evaluate.V6SafeOverridePolicy(
            _DeterministicPublicScorer(),
            evaluate.V6PolicyParameters(beta=1.0, threshold=-100.0),
        )
        first = policy.actions([observation], [normal])
        second = policy.actions([observation], [normal])
        self.assertEqual(first, second)
        audit = first.audits[0]
        self.assertEqual(set(audit), evaluate._DECISION_AUDIT_FIELDS)
        self.assertNotIn("hand", " ".join(audit).lower())
        self.assertEqual(audit["deviated"], audit["selectedAction"] != normal)

    def test_packed_actions_must_equal_original_public_legal_mask(self) -> None:
        observation, normal = _public(7373)

        class BadScorer(_DeterministicPublicScorer):
            def score_public(self, observations):  # type: ignore[no-untyped-def]
                result = super().score_public(observations)
                mask = result.action_mask.copy()
                mask[0, -1] = False
                return evaluate.V6PublicScoreBatch(
                    result.action_indices.copy(), mask, result.head_scores.copy()
                )

        policy = evaluate.V6SafeOverridePolicy(
            BadScorer(), evaluate.V6PolicyParameters()
        )
        with self.assertRaisesRegex(ValueError, "changed.*legal mask"):
            policy.actions([observation], [normal])


class CollectionCompatibilityTests(unittest.TestCase):
    def test_real_cluster_is_directly_accepted_by_v5_summarizer(self) -> None:
        if not evaluate.TORCH_AVAILABLE:
            self.skipTest("full environment collection requires PyTorch")
        from v5_evaluate import V5EvaluationConfig, summarize_v5_evaluation_clusters

        config = V5EvaluationConfig(
            "screening", "unit-v6-compatibility", 8877,
            match_counts=((4, 1),), lane_count=1, bootstrap_resamples=20,
        )
        policy = evaluate.V6SafeOverridePolicy(
            _DeterministicPublicScorer(),
            evaluate.V6PolicyParameters(threshold=math.inf),
        )
        first = evaluate.collect_v6_evaluation_clusters(policy, config)
        second = evaluate.collect_v6_evaluation_clusters(policy, config)
        self.assertEqual(first, second)
        self.assertEqual(len(first.match_clusters), 1)
        report = summarize_v5_evaluation_clusters(first.match_clusters, config)
        self.assertTrue(report["completeEvaluation"])
        self.assertEqual(len(report["matchClusters"][0]["acts"]), ACTS_PER_MATCH)


class CalibrationIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not evaluate.TORCH_AVAILABLE:
            raise unittest.SkipTest("V5 statistical report validation requires PyTorch")
        cls.calibration = _calibration_report()

    def test_calibration_is_exact_70_matches_and_has_no_screening_coordinate(self) -> None:
        report = evaluate.validate_v6_evaluation_report(self.calibration)
        self.assertEqual(report["stage"], "calibration")
        self.assertEqual(sum(report["matchPlan"].values()), 70)
        self.assertNotIn("calibrationBinding", report)
        text = evaluate.canonical_json_bytes(report).decode("ascii").lower()
        self.assertNotIn("screeningfamily", text)
        self.assertNotIn("screeningseed", text)

    def test_screening_is_bound_afterward_to_digest_and_disjoint_420_family(self) -> None:
        plan = evaluate.bind_v6_screening_plan(
            self.calibration,
            family_id="unit-screening",
            seed_base=67890,
            bootstrap_resamples=100,
        )
        self.assertEqual(sum(plan.v5_config.resolved_match_counts.values()), 420)
        receipt = evaluate.validate_v6_calibration_receipt(
            plan.calibration_receipt
        )
        self.assertEqual(
            receipt["calibrationReportSha256"],
            evaluate._canonical_sha(self.calibration),
        )
        self.assertNotIn("screening", " ".join(receipt).lower())
        self.assertEqual(receipt["policy"], plan.policy.to_dict())

    def test_same_coordinate_or_policy_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "coordinates must differ"):
            evaluate.bind_v6_screening_plan(
                self.calibration,
                family_id="unit-calibration",
                seed_base=12345,
                bootstrap_resamples=100,
            )

    def test_external_calibration_bytes_are_required_for_binding_validation(self) -> None:
        plan = evaluate.bind_v6_screening_plan(
            self.calibration,
            family_id="unit-screening-binding",
            seed_base=45678,
            bootstrap_resamples=100,
        )
        clusters = tuple(
            _synthetic_cluster(
                plan.family_id, plan.seed_base, player_count, match_index
            )
            for player_count, count in evaluate.SCREENING_MATCH_COUNTS.items()
            for match_index in range(count)
        )
        screening = evaluate.summarize_v6_evaluation(
            evaluate.V6EvaluationCollection(clusters, ()), plan, _identity()
        )
        self.assertEqual(
            evaluate.validate_v6_screening_calibration_binding(
                screening, self.calibration
            ),
            screening,
        )
        foreign = dict(self.calibration)
        foreign["coordinate"] = {"familyId": "foreign-calibration", "seedBase": 9876}
        foreign_stats = dict(foreign["normalComparison"])
        foreign_stats["familyId"] = "foreign-calibration"
        foreign["normalComparison"] = foreign_stats
        with self.assertRaises(ValueError):
            evaluate.validate_v6_screening_calibration_binding(screening, foreign)
        receipt = evaluate.build_v6_calibration_receipt(self.calibration)
        with self.assertRaisesRegex(ValueError, "differs from calibrated"):
            evaluate.V6ScreeningPlan(
                "unit-screening-two",
                23456,
                evaluate.V6PolicyParameters(1.25, 0.5),
                receipt,
                bootstrap_resamples=100,
            )

    def test_report_publication_boundary_rejects_private_named_identity(self) -> None:
        identity = _identity()
        identity["privateRows"] = []
        with self.assertRaisesRegex(ValueError, "fields drifted"):
            evaluate.validate_v6_candidate_identity(identity)


class TorchAdapterTests(unittest.TestCase):
    def test_zero_head_torch_scorer_runs_public_batch_and_stays_normal(self) -> None:
        if not evaluate.TORCH_AVAILABLE:
            self.skipTest("PyTorch unavailable")
        from test_v5_model import tiny_actor_config
        from v5_model import V5PublicActor
        from v6_override import V6PublicDeltaScorer

        evaluate.torch.manual_seed(55)
        scorer = V6PublicDeltaScorer(V5PublicActor(tiny_actor_config()))
        adapter = evaluate.TorchV6PublicBatchScorer(scorer)
        policy = evaluate.V6SafeOverridePolicy(
            adapter, evaluate.V6PolicyParameters(threshold=math.inf)
        )
        observation, normal = _real_public(6262)
        result = policy.actions([observation], [normal])
        self.assertEqual(result.actions, (normal,))
        self.assertFalse(result.audits[0]["forced"])

    def test_thin_checkpoint_loader_verifies_public_only_state_and_digest(self) -> None:
        if not evaluate.TORCH_AVAILABLE:
            self.skipTest("PyTorch unavailable")
        from test_v5_model import tiny_actor_config
        from v5_export import tensor_state_sha256
        from v5_model import V5PublicActor
        from v6_override import V6PublicDeltaScorer

        scorer = V6PublicDeltaScorer(V5PublicActor(tiny_actor_config()))
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in scorer.delta_heads.state_dict().items()
        }
        payload = {
            "format": "unit-public-delta",
            "version": 1,
            "kind": "public-delta-heads-only",
            "privilegedInputAllowed": False,
            "containsRawPrivateRows": False,
            "config": asdict(scorer.config),
            "baseActor": dict(_identity()["baseActor"]),
            "publicActorDModel": scorer.public_actor.config.d_model,
            "tensorStateSha256": tensor_state_sha256(state),
            "stateDict": state,
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "public-delta.pt"
            evaluate.torch.save(payload, checkpoint)
            loaded = evaluate._load_public_delta_payload(checkpoint)
            self.assertEqual(loaded["tensorStateSha256"], payload["tensorStateSha256"])

            corrupt = dict(payload)
            corrupt["tensorStateSha256"] = "0" * 64
            evaluate.torch.save(corrupt, checkpoint)
            with self.assertRaisesRegex(ValueError, "public-only contract"):
                evaluate._load_public_delta_payload(checkpoint)


if __name__ == "__main__":
    unittest.main()
