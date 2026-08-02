from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import v5_workflow
import v5_gpu_memory_preflight
from v5_evaluate import SCREENING_MATCH_COUNTS
from v5_workflow import (
    _remove_tree_force_writable,
    bootstrap_v5_run,
    evaluate_v5_run_stage,
    materialize_v5_source_checkout,
    train_v5_run,
    v5_run_directory_name,
    v5_seed_schedule,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _repository(root: Path) -> str:
    _git(root, "init")
    _git(root, "config", "user.name", "V5 Workflow Test")
    _git(root, "config", "user.email", "v5-workflow@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    (root / "tracked.txt").write_bytes(b"sealed\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "sealed source")
    return _git(root, "rev-parse", "HEAD")


class V5WorkflowTests(unittest.TestCase):
    def test_first_schedule_and_new_directory_name_are_exact(self) -> None:
        self.assertEqual(
            v5_seed_schedule(1, 1),
            {
                "initialization": 830_000_001,
                "calibration": 835_000_001,
                "collection": 840_000_001,
                "training": 850_000_001,
                "screening": 860_000_001,
            },
        )
        self.assertEqual(
            v5_run_directory_name(1, 1),
            "v5-mappo-normalresidual-i001-s840000001-run-001",
        )
        second = v5_seed_schedule(1, 2)
        self.assertEqual(len(set(second.values())), len(second))
        self.assertNotEqual(second, v5_seed_schedule(2, 1))
        with self.assertRaises(ValueError):
            v5_seed_schedule(0, 1)

    def test_bootstrap_seals_clean_head_and_publishes_one_atomic_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repository = parent / "repository"
            repository.mkdir()
            commit = _repository(repository)
            run = parent / v5_run_directory_name(1, 1)

            def initialize(output: Path, **_: object) -> dict[str, object]:
                output.mkdir()
                (output / "actor-bundle").mkdir()
                (output / "critic.pt").write_bytes(b"critic")
                return {"initializationSha256": "a" * 64}

            pair = {"pairId": "b" * 64, "pairManifestSha256": "c" * 64}
            evaluation_binding = {
                "sourceCommit": commit,
                "sourceBindingSha256": "d" * 64,
            }
            inventory = {"tracked.txt": "e" * 64}
            with mock.patch.object(
                v5_workflow, "V5_EVALUATION_SOURCE_FILES", ("tracked.txt",)
            ), mock.patch.object(
                v5_workflow,
                "resolve_v5_evaluation_source_binding",
                return_value=evaluation_binding,
            ), mock.patch.object(
                v5_workflow, "build_source_inventory", return_value=inventory
            ), mock.patch.object(
                v5_workflow,
                "source_inventory_sha256",
                return_value="f" * 64,
            ), mock.patch.object(
                v5_workflow,
                "publish_seeded_v5_initialization",
                side_effect=initialize,
            ) as publish, mock.patch.object(
                v5_workflow, "verify_v5_model_pair", return_value=pair
            ):
                result = bootstrap_v5_run(
                    run,
                    repository_root=repository,
                    source_commit=commit,
                    iteration=1,
                    run_number=1,
                )
            self.assertTrue((run / "source-seal" / "source.bundle").is_file())
            self.assertTrue((run / "source-seal" / "source.tar").is_file())
            self.assertTrue((run / "workflow.json.sha256").is_file())
            self.assertEqual(result["workflow"]["sourceCommit"], commit)
            self.assertEqual(publish.call_args.kwargs["seed"], 830_000_001)
            global_config = parent / "forced-global.gitconfig"
            global_config.write_bytes(b"[core]\n\tautocrlf = true\n")
            with mock.patch.object(
                v5_workflow,
                "resolve_v5_evaluation_source_binding",
                return_value=evaluation_binding,
            ), mock.patch.object(
                v5_workflow, "build_source_inventory", return_value=inventory
            ), mock.patch.object(
                v5_workflow,
                "source_inventory_sha256",
                return_value="f" * 64,
            ), mock.patch.object(
                v5_workflow, "verify_v5_model_pair", return_value=pair
            ), mock.patch.dict(
                os.environ,
                {
                    "GIT_CONFIG_GLOBAL": str(global_config),
                    "GIT_CONFIG_NOSYSTEM": "1",
                },
            ):
                materialized = materialize_v5_source_checkout(run)
            checkout = Path(materialized["repositoryRoot"])
            self.assertEqual(_git(checkout, "rev-parse", "HEAD"), commit)
            self.assertEqual(_git(checkout, "status", "--porcelain=v1"), "")
            self.assertEqual(
                _git(checkout, "config", "--local", "--get", "core.autocrlf"),
                "false",
            )
            self.assertEqual(
                _git(checkout, "config", "--local", "--get", "core.eol"),
                "lf",
            )
            self.assertEqual((checkout / "tracked.txt").read_bytes(), b"sealed\n")
            with self.assertRaises(FileExistsError):
                bootstrap_v5_run(
                    run,
                    repository_root=repository,
                    source_commit=commit,
                    iteration=1,
                    run_number=1,
                )

    def test_private_staging_cleanup_handles_read_only_git_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / ".source-checkout.staging-test"
            objects = staging / ".git" / "objects" / "pack"
            objects.mkdir(parents=True)
            packed = objects / "pack-test.idx"
            packed.write_bytes(b"sealed pack")
            packed.chmod(0o444)

            _remove_tree_force_writable(staging)

            self.assertFalse(staging.exists())

    def test_materialize_failure_preserves_read_only_checkout_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sealed-run"
            seal_root = root / "source-seal"
            seal_root.mkdir(parents=True)
            bundle = seal_root / "source.bundle"
            bundle.write_bytes(b"bundle")
            marker = RuntimeError("checkout verification failed")

            def clone(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
                staging = Path(command[-1])
                pack_root = staging / ".git" / "objects" / "pack"
                pack_root.mkdir(parents=True)
                packed = pack_root / "pack-failure.idx"
                packed.write_bytes(b"failed checkout evidence")
                packed.chmod(0o444)
                return subprocess.CompletedProcess(command, 0, b"", b"")

            seal = {
                "artifacts": {"gitBundle": bundle.name},
                "sealId": "b" * 64,
            }
            try:
                with mock.patch.object(
                    v5_workflow,
                    "load_v5_run",
                    return_value={"sourceCommit": "a" * 40},
                ), mock.patch.object(
                    v5_workflow,
                    "_load_canonical_with_sidecar",
                    return_value=(seal, "c" * 64),
                ), mock.patch.object(
                    v5_workflow, "_validate_source_seal", return_value=seal
                ), mock.patch.object(
                    v5_workflow.subprocess, "run", side_effect=clone
                ), mock.patch.object(v5_workflow, "_run_git", side_effect=marker):
                    with self.assertRaises(RuntimeError) as raised:
                        materialize_v5_source_checkout(root)

                self.assertIs(raised.exception, marker)
                self.assertFalse((root / "source-checkout").exists())
                staging = list(root.glob(".source-checkout.staging-*"))
                self.assertEqual(len(staging), 1)
                self.assertEqual(
                    (staging[0] / ".git" / "objects" / "pack" / "pack-failure.idx").read_bytes(),
                    b"failed checkout evidence",
                )
            finally:
                for staging in root.glob(".source-checkout.staging-*"):
                    _remove_tree_force_writable(staging)

    def test_atomic_run_publish_refuses_racing_empty_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repository = parent / "repository"
            repository.mkdir()
            commit = _repository(repository)
            target = parent / v5_run_directory_name(1, 1)
            original_rename = v5_workflow._rename_directory_noreplace

            def initialize(output: Path, **_: object) -> dict[str, object]:
                output.mkdir()
                return {"initializationSha256": "a" * 64}

            def race(source: Path, destination: Path) -> None:
                destination.mkdir()
                original_rename(source, destination)

            with mock.patch.object(
                v5_workflow, "V5_EVALUATION_SOURCE_FILES", ("tracked.txt",)
            ), mock.patch.object(
                v5_workflow,
                "resolve_v5_evaluation_source_binding",
                return_value={
                    "sourceCommit": commit,
                    "sourceBindingSha256": "d" * 64,
                },
            ), mock.patch.object(
                v5_workflow,
                "build_source_inventory",
                return_value={"tracked.txt": "e" * 64},
            ), mock.patch.object(
                v5_workflow, "source_inventory_sha256", return_value="f" * 64
            ), mock.patch.object(
                v5_workflow,
                "publish_seeded_v5_initialization",
                side_effect=initialize,
            ), mock.patch.object(
                v5_workflow,
                "verify_v5_model_pair",
                return_value={"pairId": "b" * 64, "pairManifestSha256": "c" * 64},
            ), mock.patch.object(
                v5_workflow, "_rename_directory_noreplace", side_effect=race
            ):
                with self.assertRaises(FileExistsError):
                    bootstrap_v5_run(
                        target,
                        repository_root=repository,
                        source_commit=commit,
                        iteration=1,
                        run_number=1,
                    )
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])

    def test_bootstrap_rejects_wrong_directory_and_dirty_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            root.mkdir()
            commit = _repository(root)
            with self.assertRaisesRegex(ValueError, "named exactly"):
                bootstrap_v5_run(
                    Path(temporary) / "wrong",
                    repository_root=root,
                    source_commit=commit,
                    iteration=1,
                    run_number=1,
                )
            (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")
            target = Path(temporary) / v5_run_directory_name(1, 1)
            with self.assertRaisesRegex(ValueError, "clean worktree"):
                bootstrap_v5_run(
                    target,
                    repository_root=root,
                    source_commit=commit,
                    iteration=1,
                    run_number=1,
                )

    def test_train_binds_scheduled_seed_and_one_model_pair(self) -> None:
        workflow = {"seeds": {"training": 850_000_001}}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_workflow, "load_v5_run", return_value=workflow
        ), mock.patch.object(
            v5_workflow, "verify_v5_model_pair", return_value={}
        ), mock.patch.object(
            v5_workflow, "_verify_v5_training_execution_source"
        ), mock.patch.object(
            v5_workflow,
            "_verify_v5_training_dataset",
            return_value=Path(temporary).resolve() / "run" / "collection" / "index",
        ), mock.patch.object(
            v5_workflow, "train_v5_mappo", return_value={"trained": True}
        ) as train, mock.patch.object(
            v5_gpu_memory_preflight,
            "load_v5_gpu_memory_preflight_report",
            return_value=(
                {
                    "config": {
                        "audit_batch_size": 64,
                        "critic_batch_size": 256,
                        "gradient_accumulation": 1,
                        "maximum_reserved_fraction": 0.9,
                        "microbatch_size": 32,
                        "minimum_free_bytes": 1024**3,
                        "timing_iterations": 7,
                        "warmup_iterations": 2,
                    }
                },
                "d" * 64,
            ),
        ), mock.patch.object(
            v5_gpu_memory_preflight,
            "verify_v5_gpu_memory_admission",
            return_value={"reportSha256": "d" * 64},
        ):
            root = Path(temporary) / "run"
            result = train_v5_run(
                root,
                "index",
                device="cuda",
                repository_root=temporary,
                gpu_memory_preflight="preflight.json",
                initial_model_pair=root / "initialization",
            )
            self.assertTrue(result["trained"])
            config = train.call_args.kwargs["config"]
            self.assertEqual(config.seed, 850_000_001)
            self.assertEqual(config.microbatch_size, 32)
            self.assertEqual(config.gradient_accumulation, 1)
            self.assertEqual(config.critic_batch_size, 256)
            self.assertEqual(
                train.call_args.kwargs["gpu_memory_preflight"],
                {"reportSha256": "d" * 64},
            )
            self.assertEqual(
                train.call_args.args[0],
                root.resolve() / "collection" / "index",
            )
            self.assertEqual(train.call_args.args[3], root.resolve() / "training")
            with self.assertRaisesRegex(ValueError, "RTX 3080 admission calibration"):
                train_v5_run(
                    root,
                    "index",
                    device="cuda",
                    repository_root=temporary,
                    gpu_memory_preflight="preflight.json",
                    config_overrides={
                        "microbatch_size": 8,
                        "gradient_accumulation": 4,
                    },
                )
            with self.assertRaisesRegex(ValueError, "immutable run schedule"):
                train_v5_run(
                    root,
                    "index",
                    device="cuda",
                    repository_root=temporary,
                    gpu_memory_preflight="preflight.json",
                    config_overrides={"seed": 1},
                )
            with self.assertRaisesRegex(ValueError, "external V5 initial"):
                train_v5_run(
                    root,
                    "index",
                    device="cuda",
                    repository_root=temporary,
                    gpu_memory_preflight="preflight.json",
                    initial_model_pair=Path(temporary) / "foreign-pair",
                )

    def test_training_dataset_reopens_and_cross_checks_exact_hybrid_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / (
                "v5-mappo-normalresidual-i001-s840000001-run-001"
            )
            collection = run_root / "collection"
            index_root = collection / "index"
            index_root.mkdir(parents=True)
            (index_root / "manifest.json").write_bytes(b"{}\n")
            (index_root / "manifest.json.sha256").write_bytes(b"fixture\n")
            for name in (
                "low-disk-stage-plan",
                "source-index-record",
            ):
                (collection / name).mkdir()
            for name in ("persistent-shards", "low-disk-promotion-receipts"):
                (run_root / name).mkdir()
            volatile_root = (
                Path(temporary)
                / "volatile-independent"
                / run_root.name
                / "volatile-shards"
            )
            volatile_root.mkdir(parents=True)

            pair = {
                "actorManifestSha256": "1" * 64,
                "actorSha256": "2" * 64,
                "criticSha256": "3" * 64,
                "pairId": "4" * 64,
                "pairManifestSha256": "5" * 64,
            }
            workflow = {
                "initialModelPair": {
                    "pairId": pair["pairId"],
                    "pairManifestSha256": pair["pairManifestSha256"],
                },
                "runNamespace": run_root.name,
                "seeds": {"collection": 840_000_001},
                "sourceCommit": "a" * 40,
            }
            nonforced = {str(player): 230_000 for player in range(4, 11)}
            decisions = {str(player): 240_000 for player in range(4, 11)}
            matches = {str(player): 1 for player in range(4, 11)}
            gate = {
                "passed": True,
                "relativeTolerance": 0.08,
                "total": sum(nonforced.values()),
            }
            plan_document = {
                "calibration": {"reportSha256": "6" * 64},
                "matchCounts": matches,
                "policyNumericsSha256": "7" * 64,
                "sourceInventorySha256": "8" * 64,
                "targets": {
                    "actualStratumRelativeTolerance": 0.08,
                    "maximumNonforcedDecisions": 2_000_000,
                    "minimumNonforcedDecisions": 1_500_000,
                    "targetNonforcedDecisions": 1_600_000,
                },
                "totalMatches": 7,
            }
            plan = SimpleNamespace(
                behavior=pair,
                document=plan_document,
                manifest_sha256="9" * 64,
                purpose="production",
                run_namespace=run_root.name,
                seed_base=840_000_001,
                source_inventory=[{"path": "fixture"}],
            )
            corpus = {
                "actualDecisionCountsByPlayerCount": decisions,
                "actualMatchCountsByPlayerCount": matches,
                "actualNonforcedDecisionCountsByPlayerCount": nonforced,
                "completeShardIndices": [0],
                "matchCoordinatesSha256": "b" * 64,
                "matchProvenanceContract": "collision-free-fixture-v2",
                "shardManifestSha256s": {"0": "c" * 64},
                "totalUniqueMatches": 7,
            }
            metadata = {
                "actualCorpusGate": gate,
                "actualDecisionCountsByPlayerCount": decisions,
                "actualMatchCountsByPlayerCount": matches,
                "actualNonforcedDecisionCountsByPlayerCount": nonforced,
                "behavior": pair,
                "calibrationReportSha256": "6" * 64,
                "collectionPlanManifestSha256": "9" * 64,
                "completeShardIndices": [0],
                "matchCoordinatesSha256": "b" * 64,
                "matchProvenanceContract": "collision-free-fixture-v2",
                "plannedMatchCountsByPlayerCount": matches,
                "policyNumericsSha256": "7" * 64,
                "shardManifestSha256s": {"0": "c" * 64},
                "sourceInventorySha256": "8" * 64,
                "totalUniqueMatches": 7,
            }
            shard = (run_root / "persistent-shards" / "shard-fixture").resolve()
            index = SimpleNamespace(
                close=mock.Mock(),
                manifest={"metadata": {"lowDiskStageContract": "fixture"}, "playerCounts": list(range(4, 11))},
                match_count=7,
                shard_paths=(shard,),
            )
            hybrid = SimpleNamespace(
                actual_corpus_gate=gate,
                actual_decision_counts_by_player_count=decisions,
                actual_match_counts_by_player_count=matches,
                actual_nonforced_decision_counts_by_player_count=nonforced,
                shard_paths=(shard,),
                source_index=SimpleNamespace(document={"metadata": metadata}),
            )
            seal = {
                "collectionSourceInventory": plan.source_inventory,
                "collectionSourceInventorySha256": "8" * 64,
                "sourceCommit": "a" * 40,
            }
            with mock.patch(
                "v5_collection_plan.load_collection_plan", return_value=plan
            ), mock.patch(
                "v5_collection_plan.verify_planned_collection_corpus",
                return_value=corpus,
            ) as corpus_verify, mock.patch(
                "v5_collection_plan.validate_actual_nonforced_corpus",
                return_value=gate,
            ), mock.patch(
                "v5_dataset.load_v5_index_manifest", return_value=index
            ), mock.patch(
                "v5_low_disk_stage.verify_v5_hybrid_stage", return_value=hybrid
            ) as hybrid_verify, mock.patch.object(
                v5_workflow,
                "_load_canonical_with_sidecar",
                return_value=(seal, "d" * 64),
            ), mock.patch.object(
                v5_workflow, "_validate_source_seal", return_value=seal
            ):
                with self.assertRaisesRegex(ValueError, "explicit persistent"):
                    v5_workflow._verify_v5_training_dataset(
                        run_root, workflow, index_root, pair
                    )
                hybrid_verify.assert_not_called()
                admitted = v5_workflow._verify_v5_training_dataset(
                    run_root,
                    workflow,
                    index_root,
                    pair,
                    low_disk_persistent_root=run_root / "persistent-shards",
                    low_disk_volatile_root=volatile_root,
                    low_disk_promotion_receipt_root=(
                        run_root / "low-disk-promotion-receipts"
                    ),
                )
                plan_document["targets"]["targetNonforcedDecisions"] = 1_500_000  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, "target=1.6M"):
                    v5_workflow._verify_v5_training_dataset(
                        run_root,
                        workflow,
                        index_root,
                        pair,
                        low_disk_persistent_root=run_root / "persistent-shards",
                        low_disk_volatile_root=volatile_root,
                        low_disk_promotion_receipt_root=(
                            run_root / "low-disk-promotion-receipts"
                        ),
                    )

            self.assertEqual(admitted, index_root.resolve())
            hybrid_verify.assert_called_once_with(
                collection / "low-disk-stage-plan",
                persistent_root=run_root / "persistent-shards",
                volatile_root=volatile_root,
                source_index_record=collection / "source-index-record",
                promotion_receipt_root=run_root / "low-disk-promotion-receipts",
                hybrid_index=index_root.resolve(),
                collection_plan=(collection / "plan").resolve(),
            )
            self.assertEqual(
                corpus_verify.call_args.kwargs["index_shard_paths"],
                (shard,),
            )
            index.close.assert_called_once_with()

    def test_training_dataset_rejects_noncanonical_index_root_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            foreign = Path(temporary) / "foreign-index"
            foreign.mkdir()
            with self.assertRaisesRegex(ValueError, "canonical"):
                v5_workflow._verify_v5_training_dataset(
                    run_root,
                    {},
                    foreign,
                    {},
                )

    def test_certification_requires_passed_same_actor_screening(self) -> None:
        model = {
            "actorSha256": "a" * 64,
            "manifestSha256": "b" * 64,
            "tensorStateSha256": "c" * 64,
            "publicContractSha256": "d" * 64,
            "policyNumericsSha256": "e" * 64,
        }
        workflow = {
            "runNamespace": "v5-mappo-normalresidual-i001-s840000001-run-001",
            "seeds": {"screening": 860_000_001},
        }
        screening = {
            "mode": "screening",
            "completeEvaluation": True,
            "allPlayerCountsPassed": True,
            "model": model,
        }
        screening_id = "4" * 64
        coordinate = {
            "familyId": "v5-certification-c-a",
            "seedBase": 123,
            "matchPlan": {str(player): 60 for player in range(4, 11)},
            "label": "a",
            "outputPath": "certification-results/" + "1" * 64 + "/a.json",
        }
        actor = object()
        evaluation = {
            "allPlayerCountsPassed": True,
            "completeEvaluation": False,
        }
        binding = {
            "coordinate": {"familyId": coordinate["familyId"], "seedBase": 123},
            "evaluationProvenanceSha256": "2" * 64,
            "outputPath": coordinate["outputPath"],
            "reservationId": "1" * 64,
            "reservationSha256": "3" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_workflow, "load_v5_run", return_value=workflow
        ), mock.patch.object(
            v5_workflow, "load_v5_actor_bundle", return_value=(actor, {})
        ), mock.patch.object(
            v5_workflow, "v5_actor_bundle_digests", return_value=model
        ), mock.patch.object(
            v5_workflow,
            "load_v5_certification_execution_reservation",
            return_value={
                "coordinates": [
                    coordinate,
                    {
                        **coordinate,
                        "familyId": "v5-certification-c-b",
                        "seedBase": 124,
                        "label": "b",
                        "outputPath": "certification-results/" + "1" * 64 + "/b.json",
                    },
                ],
                "screening": {
                    "evaluationProvenanceSha256": "2" * 64,
                    "reportSha256": "9" * 64,
                    "reservationId": screening_id,
                    "reservationSha256": "8" * 64,
                },
            },
        ), mock.patch.object(
            v5_workflow, "_source_provenance_for_run", return_value={"sealed": True}
        ), mock.patch.object(
            v5_workflow,
            "_passed_screening_for_actor",
            return_value=screening,
        ), mock.patch.object(
            v5_workflow,
            "authorize_v5_certification_evaluation",
            return_value=binding,
        ), mock.patch.object(
            v5_workflow, "evaluate_v5_actor", return_value=evaluation
        ) as evaluate, mock.patch.object(
            v5_workflow,
            "_write_workflow_evaluation_report",
            return_value="f" * 64,
        ), mock.patch.object(
            v5_workflow, "sha256_file", return_value="9" * 64
        ), mock.patch.object(
            v5_workflow,
            "_verify_exact_evaluation_execution",
            return_value=evaluation,
        ):
            reservation = (
                Path(temporary)
                / "registry"
                / "certification-reservations"
                / ("1" * 64 + ".json")
            )
            screening_path = (
                Path(temporary)
                / "registry"
                / "screening-results"
                / screening_id
                / "report.json"
            )
            result = evaluate_v5_run_stage(
                Path(temporary) / "run",
                Path(temporary) / "actor-bundle",
                None,
                stage="certification-a",
                device="cuda",
                repository_root=temporary,
                screening_report=screening_path,
                certification_reservation=reservation,
            )
            self.assertEqual(result["familyId"], coordinate["familyId"])
            config = evaluate.call_args.args[1]
            self.assertEqual(config.mode, "certification")
            self.assertEqual(config.resolved_match_counts, SCREENING_MATCH_COUNTS)
            self.assertEqual(
                evaluate.call_args.kwargs["evaluation_provenance"], {"sealed": True}
            )
            self.assertEqual(
                evaluate.call_args.kwargs["certification_reservation"], reservation
            )
            self.assertEqual(
                Path(result["output"]),
                Path(temporary)
                / "registry"
                / "certification-results"
                / ("1" * 64)
                / "a.json",
            )

            with mock.patch.object(
                v5_workflow,
                "_passed_screening_for_actor",
                side_effect=ValueError(
                    "certification prerequisite screening failed"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "prerequisite screening"):
                    evaluate_v5_run_stage(
                        Path(temporary) / "run",
                        Path(temporary) / "actor-bundle",
                        Path(temporary) / "never.json",
                        stage="certification-a",
                        device="cuda",
                        repository_root=temporary,
                        screening_report=screening_path,
                        certification_reservation=reservation,
                    )

    def test_reserve_cli_requires_exactly_two_certificates(self) -> None:
        arguments = [
            "reserve-final",
            "--registry", "registry",
            "--bundle", "actor-bundle",
            "--certification-report", "only-one.json",
            "--final-shards", "2",
        ]
        with self.assertRaisesRegex(ValueError, "exactly two"):
            v5_workflow.main(arguments)
        with mock.patch.object(
            v5_workflow,
            "reserve_v5_final_holdout",
            return_value={"planPath": "plan.json"},
        ) as reserve, mock.patch("builtins.print"):
            self.assertEqual(
                v5_workflow.main(
                    arguments[:-2]
                    + ["--certification-report", "second.json", "--final-shards", "2"]
                ),
                0,
            )
            self.assertEqual(
                reserve.call_args.args[2], ["only-one.json", "second.json"]
            )

    def test_recover_promotion_lock_cli_forwards_explicit_reason(self) -> None:
        with mock.patch.object(
            v5_workflow,
            "recover_v5_promotion_lock",
            return_value={"resumedRecovery": True},
        ) as recover, mock.patch("builtins.print"):
            self.assertEqual(
                v5_workflow.main([
                    "recover-promotion-lock",
                    "--registry", "registry",
                    "--reason", "audited stale process",
                ]),
                0,
            )
        recover.assert_called_once_with(
            "registry", recovery_reason="audited stale process"
        )

    def test_existing_exact_screening_report_is_verified_without_gameplay(self) -> None:
        model = {
            "actorSha256": "a" * 64,
            "manifestSha256": "b" * 64,
            "tensorStateSha256": "c" * 64,
            "publicContractSha256": "d" * 64,
            "policyNumericsSha256": "e" * 64,
        }
        workflow = {"runNamespace": "run-one", "seeds": {"screening": 1}}
        verified = {
            "allPlayerCountsPassed": True,
            "completeEvaluation": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            reservation_id = "1" * 64
            reservation_path = (
                Path(temporary)
                / "registry"
                / "screening-reservations"
                / f"{reservation_id}.json"
            )
            output = (
                Path(temporary)
                / "registry"
                / "screening-results"
                / reservation_id
                / "report.json"
            )
            reservation = {
                "coordinate": {
                    "familyId": "v5-screening-c",
                    "seedBase": 8123,
                    "matchPlan": {str(player): 60 for player in range(4, 11)},
                },
                "model": model,
                "outputPath": f"screening-results/{reservation_id}/report.json",
            }
            provenance = {"sealed": True}
            binding = {
                "coordinate": {"familyId": "v5-screening-c", "seedBase": 8123},
                "evaluationProvenanceSha256": "2" * 64,
                "outputPath": reservation["outputPath"],
                "reservationId": reservation_id,
                "reservationSha256": "3" * 64,
            }
            config = v5_workflow.V5EvaluationConfig(
                mode="screening",
                family_id="v5-screening-c",
                seed_base=8123,
                match_counts=tuple((player, 60) for player in range(4, 11)),
                lane_count=32,
                bootstrap_resamples=10_000,
            )
            v5_workflow._claim_evaluation_execution_once(
                output,
                stage="screening",
                device="cpu",
                config=config,
                model=model,
                provenance=provenance,
                binding=binding,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"preserved-report")
            with mock.patch.object(
                v5_workflow, "load_v5_run", return_value=workflow
            ), mock.patch.object(
                v5_workflow, "load_v5_actor_bundle", return_value=(object(), {})
            ), mock.patch.object(
                v5_workflow, "v5_actor_bundle_digests", return_value=model
            ), mock.patch.object(
                v5_workflow,
                "load_v5_screening_execution_reservation",
                return_value=reservation,
            ), mock.patch.object(
                v5_workflow, "_source_provenance_for_run", return_value=provenance
            ), mock.patch.object(
                v5_workflow,
                "authorize_v5_screening_evaluation",
                return_value=binding,
            ), mock.patch.object(
                v5_workflow,
                "_load_workflow_evaluation_report",
                return_value={"report": True},
            ), mock.patch.object(
                v5_workflow,
                "_verify_exact_evaluation_execution",
                return_value=verified,
            ), mock.patch.object(
                v5_workflow, "evaluate_v5_actor"
            ) as gameplay:
                result = evaluate_v5_run_stage(
                    Path(temporary) / "run",
                    Path(temporary) / "actor-bundle",
                    output,
                    stage="screening",
                    device="cpu",
                    repository_root=temporary,
                    screening_reservation=reservation_path,
                )
            self.assertTrue(result["reusedExistingReport"])
            gameplay.assert_not_called()
            self.assertTrue(v5_workflow._execution_marker_path(output).exists())

    def test_execution_marker_blocks_ambiguous_replay(self) -> None:
        model = {
            "actorSha256": "a" * 64,
            "manifestSha256": "b" * 64,
            "tensorStateSha256": "c" * 64,
            "publicContractSha256": "d" * 64,
            "policyNumericsSha256": "e" * 64,
        }
        workflow = {"runNamespace": "run-one", "seeds": {"screening": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            reservation_id = "1" * 64
            reservation_path = Path(temporary) / "registry" / "screening-reservations" / f"{reservation_id}.json"
            output = Path(temporary) / "registry" / "screening-results" / reservation_id / "report.json"
            reservation = {
                "coordinate": {
                    "familyId": "v5-screening-c",
                    "seedBase": 8123,
                    "matchPlan": {str(player): 60 for player in range(4, 11)},
                },
                "model": model,
                "outputPath": f"screening-results/{reservation_id}/report.json",
            }
            provenance = {"sealed": True}
            binding = {
                "coordinate": {"familyId": "v5-screening-c", "seedBase": 8123},
                "evaluationProvenanceSha256": "2" * 64,
                "outputPath": reservation["outputPath"],
                "reservationId": reservation_id,
                "reservationSha256": "3" * 64,
            }
            config = v5_workflow.V5EvaluationConfig(
                mode="screening", family_id="v5-screening-c", seed_base=8123,
                match_counts=tuple((player, 60) for player in range(4, 11)),
                lane_count=32, bootstrap_resamples=10_000,
            )
            v5_workflow._claim_evaluation_execution_once(
                output, stage="screening", device="cpu", config=config,
                model=model, provenance=provenance, binding=binding,
            )
            with mock.patch.object(
                v5_workflow, "load_v5_run", return_value=workflow
            ), mock.patch.object(
                v5_workflow, "load_v5_actor_bundle", return_value=(object(), {})
            ), mock.patch.object(
                v5_workflow, "v5_actor_bundle_digests", return_value=model
            ), mock.patch.object(
                v5_workflow, "load_v5_screening_execution_reservation", return_value=reservation
            ), mock.patch.object(
                v5_workflow, "_source_provenance_for_run", return_value=provenance
            ), mock.patch.object(
                v5_workflow, "authorize_v5_screening_evaluation", return_value=binding
            ), mock.patch.object(
                v5_workflow, "evaluate_v5_actor"
            ) as gameplay:
                with self.assertRaisesRegex(RuntimeError, "explicit crash-recovery"):
                    evaluate_v5_run_stage(
                        Path(temporary) / "run",
                        Path(temporary) / "actor-bundle",
                        output,
                        stage="screening",
                        device="cpu",
                        repository_root=temporary,
                        screening_reservation=reservation_path,
                    )
            gameplay.assert_not_called()

    def test_evaluation_crash_recovery_is_evidence_bound_and_fail_closed(self) -> None:
        model = {
            "actorSha256": "a" * 64,
            "manifestSha256": "b" * 64,
            "tensorStateSha256": "c" * 64,
            "publicContractSha256": "d" * 64,
            "policyNumericsSha256": "e" * 64,
        }
        provenance = {"provenanceSha256": "f" * 64}
        binding = {
            "coordinate": {"familyId": "screen", "seedBase": 8123},
            "evaluationProvenanceSha256": "f" * 64,
            "outputPath": "screening-results/one/report.json",
            "reservationId": "1" * 64,
            "reservationSha256": "2" * 64,
        }
        config = v5_workflow.V5EvaluationConfig(
            mode="screening",
            family_id="screen",
            seed_base=8123,
            match_counts=tuple((player, 60) for player in range(4, 11)),
            lane_count=32,
            bootstrap_resamples=10_000,
        )

        def setup(root: Path, name: str) -> tuple[Path, dict[str, object]]:
            output = root / name / "report.json"
            identity = v5_workflow._evaluation_execution_identity(
                output,
                stage="screening",
                device="cpu",
                config=config,
                model=model,
                provenance=provenance,
                binding=binding,
            )
            v5_workflow._claim_evaluation_execution_once(
                output,
                stage="screening",
                device="cpu",
                config=config,
                model=model,
                provenance=provenance,
                binding=binding,
            )
            return output, identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_output, active_identity = setup(root, "active")
            with self.assertRaisesRegex(RuntimeError, "still active"):
                v5_workflow._recover_evaluation_execution(
                    active_output,
                    reason="must reject an active attempt",
                    identity=active_identity,
                    config=config,
                    model=model,
                    provenance=provenance,
                    screening_binding=binding,
                    certification_binding=None,
                    final_binding=None,
                )

            with self.assertRaisesRegex(ValueError, "another Actor/provenance"):
                v5_workflow._execution_markers(
                    active_output,
                    {**active_identity, "provenance": {"provenanceSha256": "0" * 64}},
                )

            retry_output, retry_identity = setup(root, "retry")
            retry_output.write_bytes(b"partial-canonical-output")
            with mock.patch.object(
                v5_workflow,
                "_prove_execution_process_inactive",
                return_value=("process-missing", {
                    "bootId": "boot", "hostname": "host", "pid": 99,
                    "processStartTicks": 100, "identitySha256": "3" * 64,
                }),
            ), mock.patch.object(
                v5_workflow,
                "_load_evaluation_report",
                side_effect=ValueError("partial report"),
            ):
                restored = v5_workflow._recover_evaluation_execution(
                    retry_output,
                    reason="process died while publishing report",
                    identity=retry_identity,
                    config=config,
                    model=model,
                    provenance=provenance,
                    screening_binding=binding,
                    certification_binding=None,
                    final_binding=None,
                )
            self.assertFalse(restored)
            self.assertFalse(retry_output.exists())
            self.assertTrue(
                v5_workflow._execution_marker_path(retry_output, 2).exists()
            )
            retired = list(
                v5_workflow._execution_recovery_directory(retry_output)
                .joinpath("retired")
                .glob("*.bin")
            )
            self.assertIn(b"partial-canonical-output", [path.read_bytes() for path in retired])

            orphan_output, orphan_identity = setup(root, "orphan")
            orphan = orphan_output.parent / f".{orphan_output.name}.one.tmp"
            orphan.write_bytes(b"one-complete-report")
            with mock.patch.object(
                v5_workflow,
                "_prove_execution_process_inactive",
                return_value=("process-missing", {
                    "bootId": "boot", "hostname": "host", "pid": 99,
                    "processStartTicks": 100, "identitySha256": "3" * 64,
                }),
            ), mock.patch.object(
                v5_workflow, "_load_evaluation_report", return_value={"exact": True}
            ), mock.patch.object(
                v5_workflow,
                "_verify_exact_evaluation_execution",
                return_value={"exact": True},
            ):
                restored = v5_workflow._recover_evaluation_execution(
                    orphan_output,
                    reason="recover fully fsynced orphan",
                    identity=orphan_identity,
                    config=config,
                    model=model,
                    provenance=provenance,
                    screening_binding=binding,
                    certification_binding=None,
                    final_binding=None,
                )
            self.assertTrue(restored)
            self.assertEqual(orphan_output.read_bytes(), b"one-complete-report")
            self.assertTrue(
                orphan_output.with_name(orphan_output.name + ".sha256").exists()
            )

            conflict_output, conflict_identity = setup(root, "conflict")
            (conflict_output.parent / f".{conflict_output.name}.one.tmp").write_bytes(b"a")
            (conflict_output.parent / f".{conflict_output.name}.two.tmp").write_bytes(b"b")
            with mock.patch.object(
                v5_workflow,
                "_prove_execution_process_inactive",
                return_value=("process-missing", {
                    "bootId": "boot", "hostname": "host", "pid": 99,
                    "processStartTicks": 100, "identitySha256": "3" * 64,
                }),
            ), mock.patch.object(
                v5_workflow, "_load_evaluation_report", return_value={"exact": True}
            ), mock.patch.object(
                v5_workflow,
                "_verify_exact_evaluation_execution",
                return_value={"exact": True},
            ), self.assertRaisesRegex(RuntimeError, "conflicting valid outputs"):
                v5_workflow._recover_evaluation_execution(
                    conflict_output,
                    reason="conflicting outputs must fail",
                    identity=conflict_identity,
                    config=config,
                    model=model,
                    provenance=provenance,
                    screening_binding=binding,
                    certification_binding=None,
                    final_binding=None,
                )

    def test_final_claim_wrapper_binds_run_source_provenance(self) -> None:
        workflow = {"sourceCommit": "a" * 40}
        provenance = {"provenanceSha256": "b" * 64}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_workflow, "load_v5_run", return_value=workflow
        ), mock.patch.object(
            v5_workflow, "_source_provenance_for_run", return_value=provenance
        ), mock.patch.object(
            v5_workflow,
            "claim_v5_final_evaluation_shard",
            return_value={"claimPath": "claim.json"},
        ) as claim:
            result = v5_workflow.claim_v5_final_run_shard(
                Path(temporary) / "run",
                "plan.json",
                "actor-bundle",
                repository_root=temporary,
                device="cuda:0",
                match_shard_count=2,
                match_shard_index=1,
            )
        self.assertEqual(result["claimPath"], "claim.json")
        self.assertEqual(claim.call_args.kwargs["evaluation_provenance"], provenance)
        self.assertEqual(claim.call_args.kwargs["match_shard_index"], 1)


if __name__ == "__main__":
    unittest.main()
