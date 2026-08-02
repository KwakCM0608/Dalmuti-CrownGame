from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

import v5_evaluate
from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v4_evaluate import rotating_candidate_seats
from v4_env import DalmutiScalarEnv
from v5_evaluate import (
    ACTS_PER_MATCH,
    EXACT_GATES,
    V5EvaluationConfig,
    V5GreedyActorPolicy,
    approve_v5_final_evaluation_report,
    collect_v5_evaluation_clusters,
    derive_v5_evaluation_seed,
    evaluate_v5_actor,
    evaluation_candidate_initial_seats,
    merge_v5_evaluation_reports,
    summarize_v5_evaluation_clusters,
    validate_v5_evaluation_report,
)
from v5_model import V5ActorConfig, V5PublicActor, V5_POLICY_NUMERICS_SHA256
from v5_public import v5_public_from_v4_actor_observation
from v5_test_provenance_fixture import synthetic_v5_evaluation_provenance


def _small_actor() -> V5PublicActor:
    torch.manual_seed(77)
    return V5PublicActor(V5ActorConfig(
        history_latents=2,
        d_model=32,
        core_layers=1,
        heads=4,
        feedforward=64,
    ))


def _final_identity() -> dict[str, str]:
    return {
        "actorSha256": "a" * 64,
        "manifestSha256": "b" * 64,
        "tensorStateSha256": "c" * 64,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
    }


_PROVENANCE = synthetic_v5_evaluation_provenance()


def _provenance_record(count: int, index: int) -> dict[str, object]:
    return {
        "provenance": copy.deepcopy(_PROVENANCE),
        "shard": {"count": count, "index": index},
    }


def _claim_binding(
    reservation_id: str, claim_id: str, count: int, index: int
) -> dict[str, object]:
    return {
        "claimId": claim_id,
        "evaluationProvenanceSha256": _PROVENANCE["provenanceSha256"],
        "outputPath": (
            f"final-results/{reservation_id}/shard-{index:03d}.json"
        ),
        "reservationId": reservation_id,
        "shard": {"count": count, "index": index},
    }


def _synthetic_cluster(player_count: int, match_index: int, winning: bool) -> dict[str, object]:
    initial = list(range(player_count))
    seats = list(evaluation_candidate_initial_seats(player_count, match_index))
    candidate = sorted(initial[seat] for seat in seats)
    normal = [value for value in initial if value not in candidate]
    finish = (candidate + normal) if winning else (normal + candidate)
    awards = {
        actor: 4 if place == 1 else 3 if place == 2 else
        1 if place == player_count - 1 else 0 if place == player_count else 2
        for place, actor in enumerate(finish, start=1)
    }
    candidate_mean = sum(awards[value] for value in candidate) / len(candidate)
    normal_mean = sum(awards[value] for value in normal) / len(normal)
    before = sum(
        int(finish.index(left) < finish.index(right))
        for left in candidate for right in normal
    )
    comparisons = len(candidate) * len(normal)
    acts = [{
        "act": act,
        "finishOrder": finish,
        "candidatePhysicalIds": candidate,
        "candidateMeanChip": candidate_mean,
        "normalMeanChip": normal_mean,
        "meanChipDifference": candidate_mean - normal_mean,
        "candidateBefore": before,
        "comparisons": comparisons,
        "pairwiseRate": before / comparisons,
    } for act in range(1, ACTS_PER_MATCH + 1)]
    return {
        "playerCount": player_count,
        "matchIndex": match_index,
        "seed": derive_v5_evaluation_seed("unit-family", 123, player_count, match_index),
        "initialOrder": initial,
        "candidateInitialSeats": seats,
        "candidatePhysicalIds": candidate,
        "acts": acts,
        "decisions": 100,
        "meanChipDifference": candidate_mean - normal_mean,
        "candidateBefore": before * ACTS_PER_MATCH,
        "comparisons": comparisons * ACTS_PER_MATCH,
    }


class V5EvaluationTests(unittest.TestCase):
    def test_rotation_is_exact_evaluator_schedule_and_balanced(self) -> None:
        for players in range(4, 11):
            counts = [0] * players
            for match_index in range(players * 2):
                actual = evaluation_candidate_initial_seats(players, match_index)
                self.assertEqual(actual, rotating_candidate_seats(players, match_index))
                for seat in actual:
                    counts[seat] += 1
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_initial_zero_residual_actor_matches_normal_during_actual_gameplay(self) -> None:
        actor = _small_actor()
        policy = V5GreedyActorPolicy(actor)
        env = DalmutiScalarEnv(4, acts=ACTS_PER_MATCH, seed=98123, device="cpu")
        compared = 0
        while not env.terminated:
            normal = int(env.normal_action())
            public = v5_public_from_v4_actor_observation(env.public_observation())
            action = policy.actions([public], [normal])[0]
            self.assertEqual(action, normal)
            env.step(action)
            compared += 1
        self.assertGreater(compared, 20)

    def test_real_collection_keeps_candidate_physical_ids_for_five_acts(self) -> None:
        config = V5EvaluationConfig(
            "screening", "unit-real", 400,
            match_counts=((4, 1),), lane_count=1, bootstrap_resamples=20,
        )
        records = collect_v5_evaluation_clusters(_small_actor(), config)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(len(record["acts"]), ACTS_PER_MATCH)
        expected = record["candidatePhysicalIds"]
        self.assertTrue(all(
            act["candidatePhysicalIds"] == expected for act in record["acts"]
        ))

    def test_exact_metrics_ci_and_all_three_gates(self) -> None:
        config = V5EvaluationConfig(
            "screening", "unit-family", 123,
            match_counts=((4, 2),), bootstrap_resamples=100,
        )
        winning = [_synthetic_cluster(4, index, True) for index in range(2)]
        report = summarize_v5_evaluation_clusters(winning, config)
        self.assertEqual(validate_v5_evaluation_report(report), report)
        result = report["results"][0]
        self.assertGreaterEqual(
            result["meanCandidateMinusNormalChipPerAct"],
            EXACT_GATES["minMeanChipDifference"],
        )
        self.assertEqual(result["candidateBeforeNormalPairwise"]["rate"], 1.0)
        self.assertTrue(result["gate"]["passed"])
        self.assertTrue(report["allPlayerCountsPassed"])

        losing = [_synthetic_cluster(4, index, False) for index in range(2)]
        failed = summarize_v5_evaluation_clusters(losing, config)
        self.assertFalse(failed["results"][0]["gate"]["passed"])
        self.assertFalse(failed["allPlayerCountsPassed"])
        tampered = copy.deepcopy(report)
        tampered["results"][0]["meanCandidateMinusNormalChipPerAct"] += 0.01
        with self.assertRaises(ValueError):
            validate_v5_evaluation_report(tampered)

    def test_final_mode_rejects_reduced_plan_and_unbound_approval(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact canonical"):
            V5EvaluationConfig(
                "final", "unit-family", 123,
                match_counts=((4, 1),), bootstrap_resamples=20,
            )

        screening = V5EvaluationConfig(
            "screening", "unit-family", 123,
            match_counts=((4, 1),), bootstrap_resamples=20,
        )
        report = summarize_v5_evaluation_clusters(
            [_synthetic_cluster(4, 0, True)], screening
        )
        with self.assertRaisesRegex(ValueError, "only a final"):
            approve_v5_final_evaluation_report(report, "missing-actor-bundle")

    def test_seed_schedule_is_deterministic_and_domain_separated(self) -> None:
        first = derive_v5_evaluation_seed("unit-family", 123, 4, 0)
        self.assertEqual(first, derive_v5_evaluation_seed("unit-family", 123, 4, 0))
        self.assertNotEqual(first, derive_v5_evaluation_seed("unit-family", 123, 5, 0))
        self.assertNotEqual(first, derive_v5_evaluation_seed("unit-family", 123, 4, 1))

    def test_match_shards_merge_only_when_complete_and_without_duplicates(self) -> None:
        plan = ((4, 4),)
        reports = []
        for shard_index in range(2):
            config = V5EvaluationConfig(
                "screening", "unit-family", 123,
                match_counts=plan, match_shard_count=2,
                match_shard_index=shard_index, bootstrap_resamples=50,
            )
            records = [
                _synthetic_cluster(4, index, True)
                for index in range(4) if index % 2 == shard_index
            ]
            report = summarize_v5_evaluation_clusters(records, config)
            self.assertFalse(report["completeEvaluation"])
            reports.append(report)
        merged = merge_v5_evaluation_reports(reports)
        self.assertTrue(merged["completeEvaluation"])
        self.assertTrue(merged["allPlayerCountsPassed"])
        duplicate = copy.deepcopy(reports)
        duplicate[1]["matchClusters"][0] = duplicate[0]["matchClusters"][0]
        with self.assertRaises(ValueError):
            merge_v5_evaluation_reports(duplicate)

    def test_final_serial_report_claim_round_trip(self) -> None:
        claim = _claim_binding("e" * 64, "d" * 64, 1, 0)
        with mock.patch.dict(v5_evaluate.FINAL_MATCH_COUNTS, {4: 2}, clear=True):
            config = V5EvaluationConfig(
                "final",
                "unit-family",
                123,
                match_counts=((4, 2),),
                bootstrap_resamples=10_000,
            )
            report = summarize_v5_evaluation_clusters(
                [_synthetic_cluster(4, index, True) for index in range(2)],
                config,
                model_identity=_final_identity(),
                final_claims=[claim],
                evaluation_provenance=[_provenance_record(1, 0)],
            )
            self.assertEqual(validate_v5_evaluation_report(report), report)
            self.assertEqual(report["finalClaims"], [claim])
            missing = copy.deepcopy(report)
            missing.pop("evaluationProvenance")
            with self.assertRaisesRegex(ValueError, "requires evaluation"):
                validate_v5_evaluation_report(missing)
            tampered = copy.deepcopy(report)
            tampered["evaluationProvenance"][0]["provenance"]["source"][
                "files"
            ]["gpu-training/v5_evaluate.py"] = "0" * 64
            with self.assertRaises(ValueError):
                validate_v5_evaluation_report(tampered)

    def test_certification_report_requires_preregistered_execution_binding(self) -> None:
        reservation = "e" * 64
        binding = {
            "coordinate": {"familyId": "unit-family", "seedBase": 123},
            "evaluationProvenanceSha256": _PROVENANCE["provenanceSha256"],
            "outputPath": f"certification-results/{reservation}/a.json",
            "reservationId": reservation,
            "reservationSha256": "f" * 64,
        }
        with mock.patch.dict(
            v5_evaluate.SCREENING_MATCH_COUNTS, {4: 2}, clear=True
        ):
            config = V5EvaluationConfig(
                "certification",
                "unit-family",
                123,
                match_counts=((4, 2),),
                bootstrap_resamples=10_000,
            )
            clusters = [
                _synthetic_cluster(4, index, True) for index in range(2)
            ]
            report = summarize_v5_evaluation_clusters(
                clusters,
                config,
                model_identity=_final_identity(),
                certification_reservation=binding,
                evaluation_provenance=[_provenance_record(1, 0)],
            )
            self.assertEqual(validate_v5_evaluation_report(report), report)
            missing = copy.deepcopy(report)
            missing.pop("certificationReservation")
            with self.assertRaisesRegex(ValueError, "fields drifted|binding"):
                validate_v5_evaluation_report(missing)
            with self.assertRaisesRegex(ValueError, "prior execution reservation"):
                collect_v5_evaluation_clusters(_small_actor(), config)

        with self.assertRaisesRegex(ValueError, "unsharded"):
            V5EvaluationConfig(
                "certification",
                "unit-family",
                123,
                match_shard_count=2,
                match_shard_index=0,
                bootstrap_resamples=10_000,
            )

    def test_evaluate_final_threads_plan_and_claim_into_report(self) -> None:
        claim = _claim_binding("e" * 64, "d" * 64, 1, 0)
        actor = _small_actor()
        records = [_synthetic_cluster(4, index, True) for index in range(2)]
        with mock.patch.dict(
            v5_evaluate.FINAL_MATCH_COUNTS, {4: 2}, clear=True
        ), mock.patch(
            "v5_evaluate._authorize_final_actor_evaluation", return_value=claim
        ) as authorize, mock.patch(
            "v5_evaluate.collect_v5_evaluation_clusters", return_value=records
        ) as collect:
            config = V5EvaluationConfig(
                "final",
                "unit-family",
                123,
                match_counts=((4, 2),),
                bootstrap_resamples=10_000,
            )
            report = evaluate_v5_actor(
                actor,
                config,
                model_identity=_final_identity(),
                evaluation_provenance=_PROVENANCE,
                promotion_plan="plan.json",
                final_claim="claim.json",
            )
            authorize.assert_called_once()
            self.assertEqual(
                collect.call_args.kwargs["final_authorization"], claim
            )
            self.assertEqual(report["finalClaims"], [claim])

        with mock.patch.dict(v5_evaluate.FINAL_MATCH_COUNTS, {4: 1}, clear=True):
            config = V5EvaluationConfig(
                "final",
                "unit-family",
                123,
                match_counts=((4, 1),),
                bootstrap_resamples=10_000,
            )
            with self.assertRaisesRegex(ValueError, "prior claim authorization"):
                collect_v5_evaluation_clusters(actor, config)

    def test_final_multi_shard_reports_merge_complete_claim_inventory(self) -> None:
        reservation = "e" * 64
        reports = []
        with mock.patch.dict(v5_evaluate.FINAL_MATCH_COUNTS, {4: 4}, clear=True):
            for shard_index, claim_id in enumerate(("d" * 64, "f" * 64)):
                config = V5EvaluationConfig(
                    "final",
                    "unit-family",
                    123,
                    match_counts=((4, 4),),
                    match_shard_count=2,
                    match_shard_index=shard_index,
                    bootstrap_resamples=10_000,
                )
                records = [
                    _synthetic_cluster(4, index, True)
                    for index in range(4)
                    if index % 2 == shard_index
                ]
                report = summarize_v5_evaluation_clusters(
                    records,
                    config,
                    model_identity=_final_identity(),
                    final_claims=[_claim_binding(
                        reservation, claim_id, 2, shard_index
                    )],
                    evaluation_provenance=[
                        _provenance_record(2, shard_index)
                    ],
                )
                self.assertEqual(validate_v5_evaluation_report(report), report)
                reports.append(report)
            merged = merge_v5_evaluation_reports(reports)
            self.assertEqual(validate_v5_evaluation_report(merged), merged)
            self.assertTrue(merged["completeEvaluation"])
            self.assertEqual(
                [claim["shard"]["index"] for claim in merged["finalClaims"]],
                [0, 1],
            )

    def test_direct_cli_rejects_production_final_marker_bypass(self) -> None:
        arguments = [
            "--bundle", "actor-bundle",
            "--mode", "final",
            "--family-id", "unit-family",
            "--seed-base", "123",
            "--output", "final.json",
            "--promotion-plan", "plan.json",
            "--final-claim", "claim.json",
            "--source-commit", "1" * 40,
            "--source-snapshot", "source.tar.zst",
            "--git-bundle", "source.bundle",
        ]
        with self.assertRaisesRegex(RuntimeError, "only through v5_workflow"):
            v5_evaluate.main(arguments)


if __name__ == "__main__":
    unittest.main()
