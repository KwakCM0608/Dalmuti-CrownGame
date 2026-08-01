from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from v4_evaluate import (
    ACTS_PER_MATCH,
    ActorActionDiagnostics,
    CandidatePolicyRouting,
    DEVELOPMENT_GATES,
    DEVELOPMENT_MATCH_COUNTS,
    EvaluationBindings,
    EvaluationSeedSchedule,
    FINAL_GATES,
    FINAL_MATCH_COUNTS,
    SCREENING_MATCH_COUNTS,
    CenteredLogitActorPolicy,
    benchmark_actor_policy_batching,
    certify_development_families,
    deterministic_cluster_bootstrap95,
    evaluate_benchmark,
    evaluate_player_count,
    history_inference_bucket,
    rotating_candidate_seats,
    resolve_cli_evaluation_bindings,
    validate_benchmark_report,
    validate_evaluation_plan,
    write_report_exclusive,
)


@dataclass(frozen=True)
class _FakeStep:
    info: dict[str, object]


class _FakeEnv:
    def __init__(self, player_count: int, acts: int, seed: int):
        self.player_count = player_count
        self.acts = acts
        rotation = seed % player_count
        self.order = list(range(rotation, player_count)) + list(range(rotation))
        self.act = 0
        self.cursor = 0
        self.actions: dict[int, int] = {}
        self.done = False


class _FakeExactAdapter:
    """One decision per seat and act; action 1 deterministically finishes first."""

    def make_env(self, player_count: int, acts: int, seed: int) -> _FakeEnv:
        return _FakeEnv(player_count, acts, seed)

    def current_player_id(self, env: _FakeEnv) -> int:
        return env.order[env.cursor]

    def player_order(self, env: _FakeEnv) -> tuple[int, ...]:
        return tuple(env.order)

    def observe(self, env: _FakeEnv) -> dict[str, int]:
        return {"actor": self.current_player_id(env)}

    def normal_action(self, env: _FakeEnv) -> int:
        return 0

    def legal_mask(self, env: _FakeEnv) -> tuple[bool, bool]:
        return (True, True)

    def step(self, env: _FakeEnv, action: int) -> _FakeStep:
        actor = self.current_player_id(env)
        env.actions[actor] = action
        env.cursor += 1
        if env.cursor < env.player_count:
            return _FakeStep({"acting_player_id": actor})
        original_position = {player_id: index for index, player_id in enumerate(env.order)}
        finish_order = tuple(
            sorted(
                env.order,
                key=lambda player_id: (-env.actions[player_id], original_position[player_id]),
            )
        )
        chips: dict[int, int] = {}
        for place, player_id in enumerate(finish_order, start=1):
            if place == 1:
                award = 4
            elif place == 2:
                award = 3
            elif place == env.player_count - 1:
                award = 1
            elif place == env.player_count:
                award = 0
            else:
                award = 2
            chips[player_id] = award
        env.order = list(finish_order)
        env.cursor = 0
        env.actions = {}
        env.act += 1
        env.done = env.act == env.acts
        return _FakeStep(
            {
                "acting_player_id": actor,
                "act_result": {
                    "finish_order": finish_order,
                    "chip_awards": chips,
                },
            }
        )

    def terminated(self, env: _FakeEnv) -> bool:
        return env.done


def _candidate_policy(_: object) -> int:
    return 1


class _FakeBatchPolicy:
    audit_metadata = {
        "kind": "single-actor",
        "actorCount": 1,
        "seeds": [7],
        "ensembleRule": None,
        "inferenceExecution": "deterministic-eager",
        "compileAutomaticFallback": False,
        "historyInferenceBuckets": [0, 16, 32, 64, 96, 128, 160, 192],
        "playerWidthBucketing": "test",
    }

    def __init__(self) -> None:
        self.maximum_batch = 0

    def __call__(self, _: object) -> int:
        return 1

    def actions(self, observations: object) -> list[int]:
        values = list(observations)
        self.maximum_batch = max(self.maximum_batch, len(values))
        return [1] * len(values)


class _FakeDiagnosticPolicy(_FakeBatchPolicy):
    def __init__(
        self,
        *,
        action: int = 1,
        margin: float = 0.75,
        probability: float = 0.8,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self.action = action
        self.margin = margin
        self.probability = probability
        self.fail = fail
        self.diagnostic_calls = 0

    def __call__(self, _: object) -> int:
        return self.action

    def actions(self, observations: object) -> list[int]:
        values = list(observations)
        return [self.action] * len(values)

    def action_diagnostics(
        self, observations: object
    ) -> list[ActorActionDiagnostics]:
        if self.fail:
            raise RuntimeError("actor inference failed")
        values = list(observations)
        self.diagnostic_calls += 1
        return [
            ActorActionDiagnostics(
                action=self.action,
                legal_logit_margin=self.margin,
                top_probability=self.probability,
                legal_action_count=2,
            )
            for _ in values
        ]


def _normal_candidate_policy(_: object) -> int:
    return 0


_BINDINGS = EvaluationBindings(
    artifact_sha256="f" * 64,
    actor_sha256="a" * 64,
    observation_contract_sha256="b" * 64,
    normal_baseline_sha256="c" * 64,
    normal_baseline_source_commit="d" * 40,
)


def _reservation(schedule: EvaluationSeedSchedule) -> dict[str, object]:
    return {
        "format": "dalmuti-v4-final-seed-reservation",
        "version": 1,
        "baseSeed": schedule.base_seed,
        "matchSeedRanges": schedule.ranges(FINAL_MATCH_COUNTS),
        "reuseForbidden": True,
        "finalFeedbackPolicy": "sealed-holdout-not-a-training-input",
    }


class V4EvaluationTests(unittest.TestCase):
    @unittest.skipIf(shutil.which("git") is None, "Git unavailable")
    def test_cli_bindings_hash_actual_files_and_reject_one_byte_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal = root / "normal.py"
            observation = root / "observation.json"
            normal.write_bytes(b"def normal_action():\n    return 1\n")
            observation.write_bytes(b'{"format":"v4-observation"}\n')
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "normal.py"], check=True)
            subprocess.run(
                [
                    "git", "-C", str(root), "-c", "user.name=V4 Test",
                    "-c", "user.email=v4@example.invalid", "commit", "-qm", "freeze",
                ],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            observation_sha = __import__("hashlib").sha256(
                observation.read_bytes()
            ).hexdigest()
            normal_sha = __import__("hashlib").sha256(normal.read_bytes()).hexdigest()
            resolved = resolve_cli_evaluation_bindings(
                artifact_sha256="1" * 64,
                actor_sha256="2" * 64,
                observation_contract_path=observation,
                frozen_normal_source_path=normal,
                repository_root=root,
                frozen_normal_source_commit=commit,
                expected_observation_sha256=observation_sha,
                expected_normal_sha256=normal_sha,
            )
            self.assertTrue(resolved.actual_files_verified)
            self.assertEqual(resolved.observation_contract_sha256, observation_sha)

            observation.write_bytes(b'{"format":"v4-observatioN"}\n')
            with self.assertRaisesRegex(ValueError, "actual file"):
                resolve_cli_evaluation_bindings(
                    artifact_sha256="1" * 64,
                    actor_sha256="2" * 64,
                    observation_contract_path=observation,
                    frozen_normal_source_path=normal,
                    repository_root=root,
                    frozen_normal_source_commit=commit,
                    expected_observation_sha256=observation_sha,
                    expected_normal_sha256=normal_sha,
                )

            observation.write_bytes(b'{"format":"v4-observation"}\n')
            normal.write_bytes(b"def normal_action():\n    return 2\n")
            with self.assertRaisesRegex(ValueError, "Git commit"):
                resolve_cli_evaluation_bindings(
                    artifact_sha256="1" * 64,
                    actor_sha256="2" * 64,
                    observation_contract_path=observation,
                    frozen_normal_source_path=normal,
                    repository_root=root,
                    frozen_normal_source_commit=commit,
                )

    def test_history_bucket_boundaries_are_exact(self) -> None:
        self.assertEqual(
            [history_inference_bucket(value) for value in (0, 1, 16, 17, 32, 33, 96, 97, 192)],
            [0, 16, 16, 32, 32, 64, 96, 128, 192],
        )
        self.assertEqual(history_inference_bucket(31, max_history=31), 31)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            history_inference_bucket(193)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch unavailable")
    def test_compile_actor_failure_is_explicit_and_never_falls_back(self) -> None:
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch

        class Actor(torch.nn.Module):
            config = SimpleNamespace(max_players=10, max_history=192)

        with patch.object(torch, "compile", side_effect=RuntimeError("unsupported")):
            with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
                CenteredLogitActorPolicy(
                    [Actor()], seeds=[1], compile_actor=True
                )

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch unavailable")
    def test_random_real_like_transformer_actions_match_unbucketed_reference(self) -> None:
        import torch
        from types import SimpleNamespace
        from v4_model import V4ActorConfig, V4PublicActor

        torch.manual_seed(20260801)
        config = V4ActorConfig(
            d_model=32,
            layers=1,
            heads=4,
            feedforward=64,
            action_hidden=32,
            dropout=0.0,
        )
        actor = V4PublicActor(config).eval()
        policy = CenteredLogitActorPolicy([actor], seeds=[20260801])
        observations = []
        for history_length in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 127, 159, 191, 192):
            player_mask = torch.zeros(10, dtype=torch.bool)
            player_mask[:7] = True
            history_mask = torch.zeros(192, dtype=torch.bool)
            history_mask[:history_length] = True
            history = torch.zeros(192, 20)
            if history_length:
                history[:history_length] = torch.randn(history_length, 20)
            legal = torch.rand(236) < 0.12
            legal[0] = True
            observations.append(
                SimpleNamespace(
                    global_features=torch.randn(12),
                    rank_features=torch.randn(13, 6),
                    player_features=torch.randn(10, 12),
                    player_mask=player_mask,
                    memory_trace_features=torch.randn(4, 20),
                    history_features=history,
                    history_mask=history_mask,
                    legal_mask=legal,
                )
            )
        reference = policy.actions_unbucketed(observations)
        bucketed = policy.actions(observations)
        self.assertEqual(bucketed, reference)
        diagnostics = policy.action_diagnostics(observations)
        self.assertEqual([value.action for value in diagnostics], reference)
        for observation, diagnostic in zip(observations, diagnostics):
            legal_count = int(observation.legal_mask.sum().item())
            self.assertEqual(diagnostic.legal_action_count, legal_count)
            self.assertGreaterEqual(diagnostic.top_probability, 0.0)
            self.assertLessEqual(diagnostic.top_probability, 1.0)
            if legal_count == 1:
                self.assertIsNone(diagnostic.legal_logit_margin)
            else:
                self.assertIsNotNone(diagnostic.legal_logit_margin)
                self.assertGreaterEqual(diagnostic.legal_logit_margin, 0.0)
        self.assertEqual(
            sorted(
                group["historyBucket"]
                for group in policy.last_batch_audit["groups"]
            ),
            [0, 16, 32, 64, 96, 128, 160, 192],
        )

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch unavailable")
    def test_actor_callback_dynamically_trims_and_pads_a_batch(self) -> None:
        import torch
        from types import SimpleNamespace

        class RecordingActor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(max_players=10, max_history=192)
                self.shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

            def forward(
                self,
                global_features: object,
                rank_features: object,
                player_features: object,
                player_mask: object,
                memory_trace_features: object,
                history_features: object,
                history_mask: object,
                legal_masks: object,
            ) -> object:
                self.shapes.append(
                    (tuple(player_features.shape), tuple(history_features.shape))
                )
                return torch.arange(236, dtype=torch.float32).expand(
                    global_features.shape[0], -1
                ).masked_fill(~legal_masks, -1.0e9)

        def observation(players: int, history: int, target: int) -> object:
            from types import SimpleNamespace

            player_mask = torch.zeros(10, dtype=torch.bool)
            player_mask[:players] = True
            history_mask = torch.zeros(192, dtype=torch.bool)
            history_mask[:history] = True
            legal = torch.zeros(236, dtype=torch.bool)
            legal[0] = True
            legal[target] = True
            return SimpleNamespace(
                global_features=torch.zeros(12),
                rank_features=torch.zeros(13, 6),
                player_features=torch.zeros(10, 12),
                player_mask=player_mask,
                memory_trace_features=torch.zeros(4, 20),
                history_features=torch.zeros(192, 20),
                history_mask=history_mask,
                legal_mask=legal,
            )

        actor = RecordingActor()
        policy = CenteredLogitActorPolicy([actor], seeds=[17])
        observations = [
            observation(4, 0, 3),
            observation(7, 5, 9),
            observation(9, 37, 21),
        ]
        self.assertEqual(policy.actions_unbucketed(observations), [3, 9, 21])
        self.assertEqual(actor.shapes[-1], ((3, 9, 12), (3, 37, 20)))
        before_bucketed = len(actor.shapes)
        self.assertEqual(policy.actions(observations), [3, 9, 21])
        self.assertEqual(
            actor.shapes[before_bucketed:],
            [
                ((1, 4, 12), (1, 0, 20)),
                ((1, 7, 12), (1, 16, 20)),
                ((1, 9, 12), (1, 64, 20)),
            ],
        )
        self.assertLess(
            policy.last_batch_audit["historyTokenRatioVsUnbucketed"], 1.0
        )
        self.assertEqual(policy(observations[0]), 3)
        self.assertEqual(actor.shapes[-1], ((1, 4, 12), (1, 0, 20)))

        benchmark = benchmark_actor_policy_batching(
            policy,
            observations,
            warmup_iterations=0,
            measured_iterations=1,
        )
        self.assertTrue(benchmark["actionsIdentical"])
        self.assertGreater(benchmark["bucketedDecisionsPerSecond"], 0)

    def test_rotating_assignments_and_role_audit_are_balanced(self) -> None:
        counts = [0] * 5
        for match_index in range(35):
            for seat in rotating_candidate_seats(5, match_index):
                counts[seat] += 1
        self.assertLessEqual(max(counts) - min(counts), 1)

        result = evaluate_player_count(
            player_count=5,
            matches=35,
            acts=ACTS_PER_MATCH,
            seed_schedule=EvaluationSeedSchedule(
                "screening", "seat-audit", 100_001
            ),
            candidate_policy=_candidate_policy,
            adapter=_FakeExactAdapter(),
            gates=DEVELOPMENT_GATES,
            bootstrap_resamples=100,
        )
        audit = result["roleAudit"]
        self.assertEqual(audit["initialCandidateSeats"], counts)
        self.assertTrue(audit["initialSeatBalance"]["passed"])
        self.assertEqual(audit["totalSeatActs"], 35 * 5 * 5)
        self.assertEqual(
            sum(
                role["candidate"]["seatActs"] + role["normal"]["seatActs"]
                for role in audit["allActRoles"].values()
            ),
            35 * 5 * 5,
        )

    def test_cluster_bootstrap_is_deterministic_and_match_clustered(self) -> None:
        samples = [-1.0, 0.0, 1.0, 2.0]
        first = deterministic_cluster_bootstrap95(samples, seed=8123, resamples=1000)
        second = deterministic_cluster_bootstrap95(samples, seed=8123, resamples=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["unit"], "seed-matched-match")
        self.assertEqual(first["clusters"], 4)
        self.assertEqual(first["mean"], 0.5)
        self.assertLess(first["low"], first["mean"])
        self.assertGreater(first["high"], first["mean"])
        constant = deterministic_cluster_bootstrap95(
            [0.75] * 8, seed=1, resamples=50
        )
        self.assertEqual(constant["low"], 0.75)
        self.assertEqual(constant["high"], 0.75)

    def test_batch_sizes_preserve_match_order_and_result_bytes(self) -> None:
        outputs = []
        policies = []
        for batch_size in (1, 7, 128):
            policy = _FakeBatchPolicy()
            policies.append(policy)
            outputs.append(
                evaluate_player_count(
                    player_count=7,
                    matches=60,
                    acts=5,
                    seed_schedule=EvaluationSeedSchedule(
                        "screening", "batch-parity", 150_001
                    ),
                    candidate_policy=policy,
                    adapter=_FakeExactAdapter(),
                    gates=DEVELOPMENT_GATES,
                    bootstrap_resamples=100,
                    batch_size=batch_size,
                )
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(policies[0].maximum_batch, 1)
        self.assertGreater(policies[1].maximum_batch, 1)
        self.assertGreater(policies[2].maximum_batch, policies[1].maximum_batch)

    def test_default_routing_is_backwards_compatible_pure_actor(self) -> None:
        arguments = dict(
            player_count=4,
            matches=8,
            acts=5,
            seed_schedule=EvaluationSeedSchedule(
                "screening", "routing-default", 175_001
            ),
            candidate_policy=_candidate_policy,
            adapter=_FakeExactAdapter(),
            gates=DEVELOPMENT_GATES,
            bootstrap_resamples=20,
            batch_size=4,
        )
        implicit = evaluate_player_count(**arguments)
        explicit = evaluate_player_count(
            **arguments,
            candidate_policy_routing=CandidatePolicyRouting("pure-actor"),
        )
        self.assertEqual(implicit, explicit)
        audit = implicit["candidateDecisionAudit"]["overall"]
        self.assertEqual(audit["actorDecisions"], audit["candidateDecisions"])
        self.assertEqual(audit["fallbackDecisions"], 0)
        self.assertGreater(audit["deviationsFromNormal"], 0)

    def test_confidence_thresholds_route_low_to_actor_and_high_to_normal(self) -> None:
        def run(margin: float, probability: float) -> dict[str, object]:
            return evaluate_player_count(
                player_count=4,
                matches=4,
                acts=5,
                seed_schedule=EvaluationSeedSchedule(
                    "screening", "routing-threshold", 180_001
                ),
                candidate_policy=_FakeDiagnosticPolicy(),
                adapter=_FakeExactAdapter(),
                gates=DEVELOPMENT_GATES,
                bootstrap_resamples=20,
                batch_size=4,
                candidate_policy_routing=CandidatePolicyRouting(
                    "confidence-fallback",
                    minimum_legal_logit_margin=margin,
                    minimum_top_probability=probability,
                ),
            )

        actor = run(0.75, 0.8)
        fallback = run(0.750_001, 0.800_001)
        actor_audit = actor["candidateDecisionAudit"]["overall"]
        fallback_audit = fallback["candidateDecisionAudit"]["overall"]
        self.assertEqual(
            actor_audit["actorDecisions"], actor_audit["candidateDecisions"]
        )
        self.assertEqual(actor_audit["fallbackDecisions"], 0)
        self.assertEqual(fallback_audit["actorDecisions"], 0)
        self.assertEqual(
            fallback_audit["fallbackDecisions"],
            fallback_audit["candidateDecisions"],
        )
        self.assertEqual(fallback_audit["deviationsFromNormal"], 0)

    def test_exact_normal_mode_is_behaviorally_identical_and_never_calls_actor(self) -> None:
        schedule = EvaluationSeedSchedule(
            "screening", "routing-exact-normal", 190_001
        )
        policy = _FakeDiagnosticPolicy(fail=True)
        exact = evaluate_player_count(
            player_count=5,
            matches=10,
            acts=5,
            seed_schedule=schedule,
            candidate_policy=policy,
            adapter=_FakeExactAdapter(),
            gates=DEVELOPMENT_GATES,
            bootstrap_resamples=20,
            batch_size=5,
            candidate_policy_routing=CandidatePolicyRouting("exact-normal"),
        )
        normal = evaluate_player_count(
            player_count=5,
            matches=10,
            acts=5,
            seed_schedule=schedule,
            candidate_policy=_normal_candidate_policy,
            adapter=_FakeExactAdapter(),
            gates=DEVELOPMENT_GATES,
            bootstrap_resamples=20,
            batch_size=5,
        )
        self.assertEqual(policy.diagnostic_calls, 0)
        exact_behavior = dict(exact)
        normal_behavior = dict(normal)
        exact_behavior.pop("candidateDecisionAudit")
        normal_behavior.pop("candidateDecisionAudit")
        self.assertEqual(exact_behavior, normal_behavior)
        audit = exact["candidateDecisionAudit"]["overall"]
        self.assertEqual(audit["fallbackRate"], 1.0)
        self.assertEqual(audit["deviationsFromNormal"], 0)

    def test_confidence_routing_never_hides_actor_errors_or_illegal_actions(self) -> None:
        common = dict(
            player_count=4,
            matches=1,
            acts=5,
            seed_schedule=EvaluationSeedSchedule(
                "screening", "routing-errors", 195_001
            ),
            adapter=_FakeExactAdapter(),
            gates=DEVELOPMENT_GATES,
            bootstrap_resamples=20,
            candidate_policy_routing=CandidatePolicyRouting(
                "confidence-fallback", minimum_legal_logit_margin=99.0
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "actor inference failed"):
            evaluate_player_count(
                **common, candidate_policy=_FakeDiagnosticPolicy(fail=True)
            )
        with self.assertRaisesRegex(ValueError, "illegal action"):
            evaluate_player_count(
                **common,
                candidate_policy=_FakeDiagnosticPolicy(action=7),
            )

    def test_routing_threshold_ranges_and_audit_counts_are_deterministic(self) -> None:
        for constructor in (
            lambda: CandidatePolicyRouting(
                "confidence-fallback", minimum_legal_logit_margin=-0.01
            ),
            lambda: CandidatePolicyRouting(
                "confidence-fallback", minimum_top_probability=1.01
            ),
            lambda: CandidatePolicyRouting("confidence-fallback"),
            lambda: CandidatePolicyRouting(
                "pure-actor", minimum_legal_logit_margin=0.0
            ),
        ):
            with self.assertRaises(ValueError):
                constructor()

        routing = CandidatePolicyRouting(
            "confidence-fallback", minimum_top_probability=0.9
        )
        reports = [
            evaluate_benchmark(
                mode="screening",
                seed_schedule=EvaluationSeedSchedule(
                    "screening", "routing-audit", 196_001
                ),
                candidate_policy=_FakeDiagnosticPolicy(),
                bindings=_BINDINGS,
                adapter=_FakeExactAdapter(),
                bootstrap_resamples=20,
                batch_size=16,
                candidate_policy_routing=routing,
            )
            for _ in range(2)
        ]
        self.assertEqual(reports[0], reports[1])
        validate_benchmark_report(reports[0], expected_mode="screening")
        root = reports[0]["candidateDecisionAudit"]
        self.assertEqual(
            root["overall"]["candidateDecisions"],
            sum(
                value["candidateDecisions"]
                for value in root["byPlayerCount"]
            ),
        )
        self.assertEqual(
            root["overall"]["candidateDecisions"],
            sum(value["candidateDecisions"] for value in root["byAct"]),
        )

    def test_exact_mode_maps_gates_and_reserved_seed_contract(self) -> None:
        screening = EvaluationSeedSchedule("screening", "coarse-a", 100_001)
        validate_evaluation_plan(
            mode="screening",
            match_counts=SCREENING_MATCH_COUNTS,
            acts=5,
            gates=DEVELOPMENT_GATES,
            seed_schedule=screening,
        )
        with self.assertRaisesRegex(ValueError, "p4 match count"):
            validate_evaluation_plan(
                mode="screening",
                match_counts={**SCREENING_MATCH_COUNTS, 4: 59},
                acts=5,
                gates=DEVELOPMENT_GATES,
                seed_schedule=screening,
            )
        with self.assertRaisesRegex(ValueError, "minPointDifference"):
            validate_evaluation_plan(
                mode="development",
                match_counts=DEVELOPMENT_MATCH_COUNTS,
                acts=5,
                gates={**DEVELOPMENT_GATES, "minPointDifference": 0.299},
                seed_schedule=EvaluationSeedSchedule(
                    "development", "cert-a", 200_001
                ),
            )
        final_schedule = EvaluationSeedSchedule(
            "final", "sealed-final-a", 900_000_001
        )
        validate_evaluation_plan(
            mode="final",
            match_counts=FINAL_MATCH_COUNTS,
            acts=5,
            gates=FINAL_GATES,
            seed_schedule=final_schedule,
            final_seed_reservation=_reservation(final_schedule),
        )
        with self.assertRaisesRegex(ValueError, "reservation"):
            validate_evaluation_plan(
                mode="final",
                match_counts=FINAL_MATCH_COUNTS,
                acts=5,
                gates=FINAL_GATES,
                seed_schedule=final_schedule,
            )
        with self.assertRaisesRegex(ValueError, "final seed namespace"):
            EvaluationSeedSchedule("development", "bad-family", 900_000_001)

    def test_report_is_reproducible_and_gates_are_independent_per_player_count(self) -> None:
        schedule = EvaluationSeedSchedule("screening", "repro-a", 300_001)
        first = evaluate_benchmark(
            mode="screening",
            seed_schedule=schedule,
            candidate_policy=_candidate_policy,
            bindings=_BINDINGS,
            adapter=_FakeExactAdapter(),
            bootstrap_resamples=100,
        )
        second = evaluate_benchmark(
            mode="screening",
            seed_schedule=schedule,
            candidate_policy=_candidate_policy,
            bindings=_BINDINGS,
            adapter=_FakeExactAdapter(),
            bootstrap_resamples=100,
        )
        self.assertEqual(first, second)
        validate_benchmark_report(first, expected_mode="screening")
        legacy = json.loads(json.dumps(first))
        legacy["candidatePolicy"].pop("routing")
        legacy.pop("candidateDecisionAudit")
        for legacy_result in legacy["results"]:
            legacy_result.pop("candidateDecisionAudit")
        validate_benchmark_report(legacy, expected_mode="screening")
        self.assertEqual(
            [result["playerCount"] for result in first["results"]],
            list(range(4, 11)),
        )
        self.assertTrue(first["promotionPassed"])
        self.assertTrue(
            all(result["effectSizeGate"]["passed"] for result in first["results"])
        )
        self.assertTrue(
            all(
                result["pairwiseCandidateBeforeNormal"]["sampleCount"] > 0
                for result in first["results"]
            )
        )

        corrupted = json.loads(json.dumps(first))
        corrupted["results"][0]["meanChipDifference"] = 0.299
        with self.assertRaisesRegex(ValueError, "bootstrap|effect gate"):
            validate_benchmark_report(corrupted)

    def test_two_disjoint_development_families_are_required(self) -> None:
        reports = [
            evaluate_benchmark(
                mode="development",
                seed_schedule=EvaluationSeedSchedule(
                    "development", family_id, base_seed
                ),
                candidate_policy=_candidate_policy,
                bindings=_BINDINGS,
                adapter=_FakeExactAdapter(),
                bootstrap_resamples=20,
            )
            for family_id, base_seed in (
                ("cert-a", 400_001),
                ("cert-b", 40_000_001),
            )
        ]
        certification = certify_development_families(reports)
        self.assertTrue(certification["promotionEligibleForFinal"])
        self.assertEqual(certification["requiredFamilies"], 2)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            certify_development_families(reports[:1])
        duplicate = json.loads(json.dumps(reports[0]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            certify_development_families([reports[0], duplicate])

    def test_report_and_checksum_are_exclusive(self) -> None:
        report = evaluate_benchmark(
            mode="screening",
            seed_schedule=EvaluationSeedSchedule(
                "screening", "publish-a", 500_001
            ),
            candidate_policy=_candidate_policy,
            bindings=_BINDINGS,
            adapter=_FakeExactAdapter(),
            bootstrap_resamples=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            published = write_report_exclusive(output, report)
            original = output.read_bytes()
            self.assertEqual(len(original), published["bytes"])
            self.assertEqual(
                (Path(str(output) + ".sha256")).read_text("ascii"),
                f"{published['sha256']}  report.json\n",
            )
            with self.assertRaises(FileExistsError):
                write_report_exclusive(output, report)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
