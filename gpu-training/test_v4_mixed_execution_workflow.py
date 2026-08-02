from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from v4_mixed_workflow import (
    BACKEND_MAP,
    BEHAVIOR_ACTOR_SHA256,
    BEHAVIOR_MANIFEST_SHA256,
    CALIBRATION_NAMESPACE,
    CALIBRATION_SEED,
    ENVIRONMENT_SEED,
    MATCH_COUNTS,
    POLICY_NUMERICS_CONTRACT,
    RUN_NAMESPACE,
    SCREEN_FAMILY,
    SCREEN_SEED,
    TRAINING_SEED,
    build_mixed_phase_plan,
    canonical_json_bytes,
    canonical_sha256,
    load_recipe,
    load_fixed_collection_plan_sha256,
    materialize_argv,
    plan_document,
)
from v4_model import canonical_v4_policy_numerics_contract


RECIPE_PATH = Path(__file__).with_name("v4_mixed_execution_recipe.json")


def _option(argv: tuple[str, ...], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


class MixedExecutionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recipe_bytes = RECIPE_PATH.read_bytes()
        cls.recipe = load_recipe(RECIPE_PATH)
        cls.phases = build_mixed_phase_plan(cls.recipe)
        cls.by_phase = {phase.phase_id: phase for phase in cls.phases}
        cls.commands = {
            command.command_id: command
            for phase in cls.phases
            for command in phase.commands
        }

    def test_recipe_is_canonical_and_binds_every_fixed_identity(self) -> None:
        self.assertEqual(self.recipe_bytes, canonical_json_bytes(self.recipe))
        contract = self.recipe["runContract"]
        identity = contract["identity"]
        layout = contract["artifactLayout"]
        self.assertEqual(self.recipe["packageId"], RUN_NAMESPACE)
        self.assertEqual(identity["runNamespace"], RUN_NAMESPACE)
        self.assertEqual(identity["calibrationNamespace"], CALIBRATION_NAMESPACE)
        self.assertEqual(identity["environmentSeed"], ENVIRONMENT_SEED)
        self.assertEqual(identity["trainingSeed"], TRAINING_SEED)
        self.assertEqual(identity["calibrationSeed"], CALIBRATION_SEED)
        self.assertEqual(
            contract["policyNumerics"],
            canonical_v4_policy_numerics_contract(),
        )
        self.assertEqual(contract["policyNumerics"], POLICY_NUMERICS_CONTRACT)
        self.assertEqual(
            layout["localRunDirectory"],
            "v4-fixedid-ppo-i001-mixedmath-s600000001-local-run-001",
        )
        self.assertEqual(
            layout["remoteRunDirectory"],
            "/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmath-s600000001-run-001",
        )
        self.assertEqual(
            layout["trainingCandidate"],
            "training/train-seed-610000001-run-001/candidate",
        )
        self.assertEqual(
            self.recipe["screening"]["candidateDirectory"],
            layout["trainingCandidate"],
        )
        self.assertEqual(contract["behaviorActor"]["actorSha256"], BEHAVIOR_ACTOR_SHA256)
        self.assertEqual(contract["behaviorActor"]["manifestSha256"], BEHAVIOR_MANIFEST_SHA256)
        self.assertEqual(contract["baseline"]["normalSource"], "frozen-git-bundle")
        self.assertEqual(
            contract["baseline"]["observationSource"], "sealed-current-source"
        )
        self.assertEqual(contract["productionCollection"]["backendMap"], list(BACKEND_MAP))
        self.assertEqual(
            contract["productionCollection"]["matchCounts"],
            {str(key): value for key, value in MATCH_COUNTS.items()},
        )
        self.assertFalse(contract["training"]["automaticSecondEpoch"])
        self.assertEqual(contract["training"]["actorPrecision"], "fp32-no-autocast")
        self.assertTrue(contract["training"]["criticAmpAllowed"])
        self.assertEqual(
            contract["calibration"]["companionFilesRequired"],
            {
                "cpuNpz": ["sha256", "metadata", "metadataSha256"],
                "cudaNpz": ["sha256", "metadata", "metadataSha256"],
                "report": ["sha256"],
            },
        )

    def test_dag_has_required_parallelism_and_barriers(self) -> None:
        expected = [
            "preflight",
            "calibration-local-cpu",
            "calibration-remote-cuda",
            "calibration-admission",
            *(f"production-local-shard-{index:02d}" for index in (0, 1)),
            *(f"production-remote-wave-one-shard-{index:02d}" for index in range(2, 8)),
            *(f"production-remote-wave-two-shard-{index:02d}" for index in range(8, 14)),
            "retrieve-merge-upload",
            "pretraining-cuda-replay",
            "train-epoch-one",
            "post-training-hard-gates",
            "screen-epoch-one",
            "verify-all-player-promotion-gates",
            "verify-complete-remote-screening",
            "retrieve-verify-seal",
        ]
        self.assertEqual([phase.phase_id for phase in self.phases], expected)
        self.assertEqual(
            self.by_phase["calibration-local-cpu"].concurrency_group,
            "calibration-cpu-cuda",
        )
        self.assertEqual(
            self.by_phase["calibration-remote-cuda"].concurrency_group,
            "calibration-cpu-cuda",
        )
        self.assertEqual(
            self.by_phase["calibration-admission"].dependencies,
            ("calibration-local-cpu", "calibration-remote-cuda"),
        )
        production = [
            phase for phase in self.phases if phase.phase_id.startswith("production-")
        ]
        self.assertEqual(len(production), 14)
        self.assertTrue(
            all(
                phase.concurrency_group == "production-local-and-remote"
                and len(phase.commands) == 1
                for phase in production
            )
        )
        wave_one_ids = tuple(
            f"production-remote-wave-one-shard-{index:02d}"
            for index in range(2, 8)
        )
        for index in range(8, 14):
            self.assertEqual(
                self.by_phase[
                    f"production-remote-wave-two-shard-{index:02d}"
                ].dependencies,
                wave_one_ids,
            )
        merge_dependencies = self.by_phase["retrieve-merge-upload"].dependencies
        self.assertEqual(
            set(merge_dependencies),
            {
                "production-local-shard-00",
                "production-local-shard-01",
                *(
                    f"production-remote-wave-two-shard-{index:02d}"
                    for index in range(8, 14)
                ),
            },
        )
        self.assertEqual(
            [command.command_id for command in self.by_phase["calibration-admission"].commands],
            [
                "retrieve-calibration-cuda",
                "compare-calibration-backends",
                "upload-calibration-triple",
            ],
        )
        self.assertEqual(
            [command.command_id for command in self.by_phase["retrieve-merge-upload"].commands],
            [
                "retrieve-remote-production-shards",
                "merge-production-shards",
                "upload-merged-production",
            ],
        )
        self.assertEqual(
            [command.command_id for command in self.by_phase["retrieve-verify-seal"].commands],
            [
                "finalize-remote-run",
                "retrieve-checksummed-results",
                "verify-and-seal-local-copy",
            ],
        )
        self.assertEqual(
            self.by_phase["retrieve-verify-seal"].dependencies,
            ("verify-complete-remote-screening",),
        )
        self.assertEqual(
            [
                command.command_id
                for command in self.by_phase["verify-complete-remote-screening"].commands
            ],
            ["verify-complete-remote-screening"],
        )

    def test_calibration_uses_identical_schedule_and_preserves_ten_files(self) -> None:
        cpu = self.commands["collect-calibration-cpu"]
        cuda = self.commands["collect-calibration-cuda"]
        self.assertEqual(cpu.host, "local")
        self.assertEqual(cuda.host, "remote")
        for command, device in ((cpu, "cpu"), (cuda, "cuda")):
            self.assertEqual(_option(command.argv, "--run-namespace"), CALIBRATION_NAMESPACE)
            self.assertEqual(_option(command.argv, "--seed-base"), str(CALIBRATION_SEED))
            self.assertEqual(_option(command.argv, "--match-counts"), "4:1,5:1,6:1,7:1,8:1,9:1,10:1")
            self.assertEqual(_option(command.argv, "--match-shard-count"), "1")
            self.assertEqual(_option(command.argv, "--match-shard-index"), "0")
            self.assertEqual(_option(command.argv, "--lanes"), "7")
            self.assertEqual(_option(command.argv, "--device"), device)
            self.assertNotIn("--shard-backend-map", command.argv)
            self.assertEqual(len(command.outputs), 4)
        compare = self.commands["compare-calibration-backends"]
        self.assertTrue(compare.argv[1].endswith("v4_compare_fixed_match_backends.py"))
        upload = self.commands["upload-calibration-triple"]
        self.assertEqual(len(upload.outputs), 10)
        self.assertEqual(len(set(upload.outputs)), 10)

    def test_production_has_exact_fourteen_shards_and_backend_map(self) -> None:
        shard_commands = [
            self.commands[f"collect-production-shard-{index:02d}"]
            for index in range(14)
        ]
        self.assertEqual([command.host for command in shard_commands[:2]], ["local", "local"])
        self.assertTrue(all(command.host == "remote" for command in shard_commands[2:]))
        backend_map = ",".join(BACKEND_MAP)
        for index, command in enumerate(shard_commands):
            self.assertEqual(_option(command.argv, "--run-namespace"), RUN_NAMESPACE)
            self.assertEqual(_option(command.argv, "--seed-base"), str(ENVIRONMENT_SEED))
            self.assertEqual(_option(command.argv, "--match-shard-count"), "14")
            self.assertEqual(_option(command.argv, "--match-shard-index"), str(index))
            self.assertEqual(_option(command.argv, "--device"), BACKEND_MAP[index])
            self.assertEqual(_option(command.argv, "--shard-backend-map"), backend_map)
            self.assertEqual(_option(command.argv, "--lanes"), "16")
            self.assertEqual(_option(command.argv, "--temperature"), "1.0")
            self.assertEqual(_option(command.argv, "--epsilon-floor"), "0.0")
            self.assertEqual(_option(command.argv, "--pairwise-coefficient"), "0.25")
            self.assertNotIn("--no-standardize-advantages", command.argv)
            self.assertIn("--cross-backend-calibration-report", command.argv)
            self.assertIn("--cross-backend-calibration-cpu-npz", command.argv)
            self.assertIn("--cross-backend-calibration-cuda-npz", command.argv)
            self.assertEqual(len(command.outputs), 4)

    def test_merge_replay_training_gate_and_screen_are_mandatory(self) -> None:
        merge = self.commands["merge-production-shards"].argv
        self.assertEqual(merge.count("--input"), 14)
        self.assertEqual(merge.count("--input-checksum"), 14)
        replay = self.commands["replay-full-ppo-dataset"].argv
        self.assertEqual(_option(replay, "--device"), "cuda")
        self.assertEqual(
            _option(replay, "--maximum-absolute-log-probability-error"), "2e-5"
        )
        training = self.commands["train-epoch-one-cuda"].argv
        expected_options = {
            "--device": "cuda",
            "--epochs": "1",
            "--batch-size": "2",
            "--gradient-accumulation": "1",
            "--seed": str(TRAINING_SEED),
            "--actor-learning-rate": "2e-5",
            "--critic-learning-rate": "2e-4",
            "--weight-decay": "1e-4",
            "--bc-weight": "0.05",
            "--ppo-weight": "1.0",
            "--critic-weight": "0.2",
            "--q-boost-coefficient": "0.0",
            "--gamma": "1.0",
            "--lambda": "0.95",
            "--clip-ratio": "0.12",
            "--entropy-coefficient": "0.0005",
            "--max-gradient-norm": "1.0",
            "--num-workers": "0",
            "--checkpoint-every": "1",
            "--expected-fixed-collection-plan-sha256": "{merged_collection_plan_sha256}",
        }
        for option, value in expected_options.items():
            self.assertEqual(_option(training, option), value)
        self.assertNotIn("--no-amp", training)
        gate = self.commands["verify-epoch-one-hard-gates"].argv
        self.assertTrue(_option(gate, "--candidate").endswith("/candidate"))
        self.assertEqual(_option(gate, "--maximum-approx-kl"), "0.020")
        self.assertEqual(_option(gate, "--maximum-clip-fraction"), "0.25")
        self.assertEqual(_option(gate, "--minimum-entropy-retention"), "0.70")
        screen = self.commands["screen-epoch-one-p4-p10"].argv
        self.assertEqual(_option(screen, "--mode"), "screening")
        self.assertEqual(_option(screen, "--family-id"), SCREEN_FAMILY)
        self.assertEqual(_option(screen, "--base-seed"), str(SCREEN_SEED))
        self.assertEqual(_option(screen, "--candidate-policy-mode"), "pure-actor")
        self.assertEqual(_option(screen, "--batch-size"), "64")
        self.assertEqual(_option(screen, "--bootstrap-resamples"), "10000")
        self.assertEqual(_option(screen, "--workers"), "4")
        self.assertEqual(
            _option(screen, "--observation-contract"),
            "{remote_source_root}/training/v4-public-history.ts",
        )
        self.assertEqual(
            _option(screen, "--frozen-normal-source"),
            "{remote_frozen_baseline_repository}/lib/bot-strategy.ts",
        )
        self.assertNotIn("--final-reservation", screen)
        promotion = self.commands["verify-screening-promotion-gates"].argv
        self.assertEqual(
            _option(promotion, "--minimum-mean-chip-difference-per-act"), "0.25"
        )
        self.assertEqual(
            _option(promotion, "--minimum-clustered-95-lower-bound"), "0.15"
        )
        self.assertEqual(
            _option(promotion, "--minimum-pairwise-before-normal"), "0.55"
        )
        self.assertEqual(
            self.commands["verify-complete-remote-screening"].host, "remote"
        )
        self.assertEqual(
            self.commands["finalize-remote-run"].host, "coordinator-finalize"
        )
        self.assertEqual(len(self.commands["upload-merged-production"].outputs), 12)

    def test_plan_sha_placeholder_is_loaded_from_strict_metadata(self) -> None:
        plan_sha = "c" * 64
        metadata = {
            "lossEligibility": {
                "fixedCollectionPlans": [
                    {
                        "canonicalFields": {
                            "matchShardCount": 14,
                            "shardBackendMap": {
                                str(index): backend
                                for index, backend in enumerate(BACKEND_MAP)
                            },
                            "version": 2,
                        },
                        "canonicalSha256": plan_sha,
                        "opaqueId": (
                            "fixed-complete-mixed-backend-shard-plan-v2:sha256="
                            + plan_sha
                        ),
                    }
                ]
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "merged.npz.metadata.json"
            payload = canonical_json_bytes(metadata)
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            Path(f"{path}.sha256").write_bytes(
                f"{digest}  {path.name}\n".encode("ascii")
            )
            self.assertEqual(load_fixed_collection_plan_sha256(path), plan_sha)
        materialized = materialize_argv(
            self.commands["train-epoch-one-cuda"].argv,
            {
                "{remote_python}": "python3",
                "{remote_source_root}": "/run/source",
                "{remote_run_directory}": "/run",
                "{remote_behavior_actor_bundle}": "/run/actor",
                "{merged_collection_plan_sha256}": plan_sha,
            },
        )
        self.assertEqual(
            _option(materialized, "--expected-fixed-collection-plan-sha256"),
            plan_sha,
        )

    def test_dry_run_plan_is_canonical_and_contains_no_forbidden_command(self) -> None:
        document = plan_document(self.phases, self.recipe)
        self.assertEqual(document["recipeSha256"], canonical_sha256(self.recipe))
        unsigned = dict(document)
        digest = unsigned.pop("canonicalSha256")
        self.assertEqual(digest, canonical_sha256(unsigned))
        command_text = json.dumps(
            [list(command.argv) for command in self.commands.values()],
            sort_keys=True,
        ).lower()
        self.assertNotIn("v3-ppo-i2", command_text)
        self.assertNotIn("deploy", command_text)
        self.assertNotIn("final-reservation", command_text)


if __name__ == "__main__":
    unittest.main()
