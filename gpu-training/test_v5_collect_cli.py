from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import v5_collect_cli
from v5_collect_cli import V5LoadedBehavior, create_collection_plan
from v5_collect_mappo import V5PublishedCollection
from v5_collection_plan import V5CollectionPlan
from v5_model import V5_POLICY_NUMERICS_SHA256


def _behavior(actor: str = "a", manifest: str = "b", critic: str = "c") -> V5LoadedBehavior:
    return V5LoadedBehavior(
        actor=torch.nn.Linear(1, 1),
        critic=torch.nn.Linear(1, 1),
        actor_sha256=actor * 64,
        actor_manifest_sha256=manifest * 64,
        critic_sha256=critic * 64,
        policy_numerics_sha256=V5_POLICY_NUMERICS_SHA256,
        pair_id="1" * 64,
        pair_manifest_sha256="2" * 64,
    )


def _calibration() -> tuple[dict[str, object], str]:
    return (
        {
            "measurements": {
                str(player): {
                    "matches": 32,
                    "nonforcedDecisions": 3_200 + (player - 4) * 160,
                }
                for player in range(4, 11)
            }
        },
        "e" * 64,
    )


class V5CollectionCLITests(unittest.TestCase):
    def test_worker_slot_enforces_planned_concurrency_and_torch_threads(self) -> None:
        plan = V5CollectionPlan(
            {
                "backendPolicy": {
                    "cpuWorkerCount": 2,
                    "cpuTorchThreadsPerWorker": 1,
                    "cudaWorkerCount": 1,
                    "cudaTorchThreadsPerWorker": 1,
                }
            },
            "a" * 64,
            (),
        )
        with tempfile.TemporaryDirectory() as temporary:
            original = torch.get_num_threads()
            with v5_collect_cli._planned_worker_slot(plan, temporary, "cpu"):
                self.assertEqual(torch.get_num_threads(), 1)
                with v5_collect_cli._planned_worker_slot(plan, temporary, "cpu"):
                    with self.assertRaisesRegex(RuntimeError, "saturated"):
                        with v5_collect_cli._planned_worker_slot(plan, temporary, "cpu"):
                            pass
            self.assertEqual(torch.get_num_threads(), original)

    def test_worker_slot_recovery_refuses_active_and_evidence_retires_stale(self) -> None:
        plan = V5CollectionPlan(
            {
                "backendPolicy": {
                    "cpuWorkerCount": 1,
                    "cpuTorchThreadsPerWorker": 1,
                    "cudaWorkerCount": 1,
                    "cudaTorchThreadsPerWorker": 1,
                }
            },
            "a" * 64,
            (),
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_collect_cli, "load_collection_plan", return_value=plan
        ), mock.patch.object(
            v5_collect_cli, "_host_boot_id", return_value="boot-a"
        ), mock.patch.object(
            v5_collect_cli.socket, "gethostname", return_value="host-a"
        ):
            lock = (
                Path(temporary)
                / ".v5-worker-slots"
                / "cpu"
                / "slot-000.lock"
            )
            lock.parent.mkdir(parents=True)
            value = {
                "backend": "cpu",
                "bootId": "boot-a",
                "format": v5_collect_cli.V5_WORKER_SLOT_LOCK_FORMAT,
                "hostname": "host-a",
                "pid": 42,
                "planSha256": "a" * 64,
                "processStartTicks": 123,
                "slot": 0,
                "version": 1,
            }
            lock.write_bytes(v5_collect_cli.canonical_json_bytes(value))
            with mock.patch.object(
                v5_collect_cli, "_process_start_ticks", return_value=123
            ):
                with self.assertRaisesRegex(RuntimeError, "active process"):
                    v5_collect_cli.recover_v5_worker_slot(
                        "plan",
                        temporary,
                        backend="cpu",
                        slot_index=0,
                        recovery_reason="operator verified collector status",
                    )
            self.assertTrue(lock.is_file())
            self.assertFalse(
                (Path(temporary) / ".v5-worker-slot-recoveries").exists()
            )
            with mock.patch.object(
                v5_collect_cli,
                "_process_start_ticks",
                side_effect=lambda pid: 999 if pid != 42 else None,
            ), mock.patch.object(
                v5_collect_cli, "_process_may_exist", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "may still be active"):
                    v5_collect_cli.recover_v5_worker_slot(
                        "plan",
                        temporary,
                        backend="cpu",
                        slot_index=0,
                        recovery_reason="identity lookup was denied",
                    )
            self.assertTrue(lock.is_file())
            with mock.patch.object(
                v5_collect_cli,
                "_process_start_ticks",
                side_effect=lambda pid: 999 if pid != 42 else None,
            ), mock.patch.object(
                v5_collect_cli, "_process_may_exist", return_value=False
            ):
                result = v5_collect_cli.recover_v5_worker_slot(
                    "plan",
                    temporary,
                    backend="cpu",
                    slot_index=0,
                    recovery_reason="collector crashed after host audit",
                )
            self.assertFalse(lock.exists())
            self.assertTrue(Path(result["receipt"]).is_file())
            self.assertTrue(Path(result["retiredLock"]).is_file())
            receipt = __import__("json").loads(Path(result["receipt"]).read_bytes())
            self.assertEqual(receipt["oldLockSha256"], result["oldLockSha256"])
            self.assertEqual(receipt["observedStaleEvidence"], "process-missing")

    def test_worker_slot_setup_failure_removes_only_its_new_lock(self) -> None:
        plan = V5CollectionPlan(
            {
                "backendPolicy": {
                    "cpuWorkerCount": 1,
                    "cpuTorchThreadsPerWorker": 1,
                    "cudaWorkerCount": 1,
                    "cudaTorchThreadsPerWorker": 1,
                }
            },
            "a" * 64,
            (),
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_collect_cli,
            "_worker_lock_document",
            side_effect=RuntimeError("identity unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "identity unavailable"):
                with v5_collect_cli._planned_worker_slot(plan, temporary, "cpu"):
                    pass
            lock = (
                Path(temporary)
                / ".v5-worker-slots"
                / "cpu"
                / "slot-000.lock"
            )
            self.assertFalse(lock.exists())

    def test_worker_slot_recovery_refuses_another_hostname(self) -> None:
        plan = V5CollectionPlan(
            {
                "backendPolicy": {
                    "cpuWorkerCount": 1,
                    "cpuTorchThreadsPerWorker": 1,
                    "cudaWorkerCount": 1,
                    "cudaTorchThreadsPerWorker": 1,
                }
            },
            "a" * 64,
            (),
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_collect_cli, "load_collection_plan", return_value=plan
        ), mock.patch.object(
            v5_collect_cli, "_host_boot_id", return_value="new-boot"
        ), mock.patch.object(
            v5_collect_cli.socket, "gethostname", return_value="new-host"
        ):
            lock = (
                Path(temporary)
                / ".v5-worker-slots"
                / "cpu"
                / "slot-000.lock"
            )
            lock.parent.mkdir(parents=True)
            lock.write_bytes(
                v5_collect_cli.canonical_json_bytes(
                    {
                        "backend": "cpu",
                        "bootId": "old-boot",
                        "format": v5_collect_cli.V5_WORKER_SLOT_LOCK_FORMAT,
                        "hostname": "old-host",
                        "pid": 42,
                        "planSha256": "a" * 64,
                        "processStartTicks": 123,
                        "slot": 0,
                        "version": 1,
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "another host"):
                v5_collect_cli.recover_v5_worker_slot(
                    "plan",
                    temporary,
                    backend="cpu",
                    slot_index=0,
                    recovery_reason="must not trust remote PID absence",
                )
            self.assertTrue(lock.exists())

    def test_worker_slot_recovery_never_replaces_retired_target(self) -> None:
        plan = V5CollectionPlan(
            {
                "backendPolicy": {
                    "cpuWorkerCount": 1,
                    "cpuTorchThreadsPerWorker": 1,
                    "cudaWorkerCount": 1,
                    "cudaTorchThreadsPerWorker": 1,
                }
            },
            "a" * 64,
            (),
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_collect_cli, "load_collection_plan", return_value=plan
        ), mock.patch.object(
            v5_collect_cli, "_host_boot_id", return_value="boot-a"
        ), mock.patch.object(
            v5_collect_cli.socket, "gethostname", return_value="host-a"
        ), mock.patch.object(
            v5_collect_cli,
            "_process_start_ticks",
            side_effect=lambda pid: 999 if pid != 42 else None,
        ), mock.patch.object(
            v5_collect_cli, "_process_may_exist", return_value=False
        ):
            lock = (
                Path(temporary)
                / ".v5-worker-slots"
                / "cpu"
                / "slot-000.lock"
            )
            lock.parent.mkdir(parents=True)
            lock.write_bytes(
                v5_collect_cli.canonical_json_bytes(
                    {
                        "backend": "cpu",
                        "bootId": "boot-a",
                        "format": v5_collect_cli.V5_WORKER_SLOT_LOCK_FORMAT,
                        "hostname": "host-a",
                        "pid": 42,
                        "planSha256": "a" * 64,
                        "processStartTicks": 123,
                        "slot": 0,
                        "version": 1,
                    }
                )
            )
            digest = v5_collect_cli.hashlib.sha256(lock.read_bytes()).hexdigest()
            retired = (
                Path(temporary)
                / ".v5-worker-slot-recoveries"
                / "cpu"
                / "retired-locks"
                / f"slot-000-{digest}.lock"
            )
            retired.parent.mkdir(parents=True)
            retired.write_bytes(b"attacker-target")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                v5_collect_cli.recover_v5_worker_slot(
                    "plan",
                    temporary,
                    backend="cpu",
                    slot_index=0,
                    recovery_reason="collision regression",
                )
            self.assertEqual(retired.read_bytes(), b"attacker-target")
            self.assertTrue(lock.exists())

    def test_throughput_preflight_measures_every_p_and_publishes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            v5_collect_cli, "load_verified_behavior", return_value=_behavior()
        ), mock.patch.object(
            v5_collect_cli,
            "build_source_inventory",
            return_value={"gpu-training/mock.py": "d" * 64},
        ), mock.patch.object(
            v5_collect_cli, "V5TorchInferenceRuntime"
        ) as runtime, mock.patch.object(
            v5_collect_cli, "collect_v5_mappo", return_value=object()
        ) as collect, mock.patch.object(
            v5_collect_cli,
            "publish_v5_mappo_collection",
            return_value=V5PublishedCollection(
                Path(temporary) / "shard", "e" * 64, 20, 2_000, 1_000
            ),
        ), mock.patch.object(
            v5_collect_cli.time, "perf_counter", side_effect=range(14)
        ):
            runtime.return_value.actor_batch = object()
            runtime.return_value.critic_batch = object()
            output = Path(temporary) / "throughput.json"
            result = v5_collect_cli.collect_throughput_preflight(
                actor_bundle="actor",
                critic_checkpoint="critic",
                behavior_pair="pair",
                source_root=".",
                output=output,
                backend="cpu",
                device="cpu",
                run_namespace="v5-throughput-unit",
                seed_base=835_000_001,
                scratch_root=temporary,
                matches_per_player_count=20,
                lane_count=4,
            )
            self.assertEqual(collect.call_count, 7)
            self.assertEqual(set(result["secondsPerMatch"]), set(range(4, 11)))
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name(output.name + ".sha256").is_file())
            report = __import__("json").loads(output.read_bytes())
            self.assertEqual(set(report["measurements"]), {str(p) for p in range(4, 11)})

    def test_production_rate_preflight_rejects_four_match_parity_sample(self) -> None:
        report, _ = _calibration()
        measurements = report["measurements"]
        assert isinstance(measurements, dict)
        measurements["10"] = {"matches": 4, "nonforcedDecisions": 400}
        with self.assertRaisesRegex(ValueError, "at least 20"):
            v5_collect_cli._preflight_from_calibration_report(report)

    def test_dry_run_is_deterministic_and_writes_nothing(self) -> None:
        sources = {"gpu-training/mock.py": "d" * 64}
        arguments = {
            "actor_bundle": "actor",
            "critic_checkpoint": "critic.pt",
            "behavior_pair": "pair",
            "source_root": ".",
            "calibration_report": "report",
            "calibration_cpu_snapshot": "cpu",
            "calibration_cuda_snapshot": "cuda",
            "run_namespace": "v5-dry-s910000001",
            "seed_base": 910_000_001,
            "output": None,
            "dry_run": True,
            "total_matches": 12_000,
            "diagnostic_unbalanced": True,
            "source_files": ("gpu-training/mock.py",),
        }
        with mock.patch.object(
            v5_collect_cli, "load_verified_behavior", return_value=_behavior()
        ), mock.patch.object(
            v5_collect_cli, "build_source_inventory", return_value=sources
        ), mock.patch.object(
            v5_collect_cli, "_verify_calibration_for_plan", return_value=_calibration()
        ):
            first = create_collection_plan(**arguments)
            second = create_collection_plan(**arguments)
        self.assertEqual(first.document, second.document)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.document["totalMatches"], 12_000)
        self.assertEqual(first.purpose, "diagnostic-unbalanced")

    def test_explicit_or_aggregate_sizing_requires_diagnostic_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "diagnostic-unbalanced"):
            create_collection_plan(
                actor_bundle="must-not-load",
                critic_checkpoint="must-not-load",
                behavior_pair="must-not-load",
                source_root=".",
                calibration_report="must-not-load",
                calibration_cpu_snapshot="must-not-load",
                calibration_cuda_snapshot="must-not-load",
                run_namespace="v5-fail-fast-s910000004",
                seed_base=910_000_004,
                output=None,
                dry_run=True,
                total_matches=12_000,
            )

    def test_non_dry_plan_requires_output(self) -> None:
        with mock.patch.object(
            v5_collect_cli, "load_verified_behavior", return_value=_behavior()
        ), mock.patch.object(
            v5_collect_cli,
            "build_source_inventory",
            return_value={"gpu-training/mock.py": "d" * 64},
        ), mock.patch.object(
            v5_collect_cli, "_verify_calibration_for_plan", return_value=_calibration()
        ):
            with self.assertRaisesRegex(ValueError, "output"):
                create_collection_plan(
                    actor_bundle="actor",
                    critic_checkpoint="critic.pt",
                    behavior_pair="pair",
                    source_root=".",
                    calibration_report="report",
                    calibration_cpu_snapshot="cpu",
                    calibration_cuda_snapshot="cuda",
                    run_namespace="v5-no-output-s910000002",
                    seed_base=910_000_002,
                    output=None,
                )

    def test_behavior_and_source_hash_mismatch_fail_closed(self) -> None:
        expected = _behavior()
        with self.assertRaisesRegex(ValueError, "actorSha256"):
            v5_collect_cli._require_behavior_matches(
                expected.hashes, _behavior(actor="f")
            )

    def test_verified_pair_rejects_actor_from_another_iteration(self) -> None:
        actor = torch.nn.Linear(1, 1)
        critic = torch.nn.Linear(1, 1)
        actor_digests = {
            "actorSha256": "a" * 64,
            "manifestSha256": "b" * 64,
            "tensorStateSha256": "3" * 64,
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "publicContractSha256": __import__("v5_contract").V5_PUBLIC_CONTRACT_SHA256,
        }
        critic_payload = {"metadata": {}, "tensorStateSha256": "4" * 64}
        mixed_pair = {
            "actorSha256": "f" * 64,  # different training iteration
            "actorManifestSha256": "b" * 64,
            "actorTensorStateSha256": "3" * 64,
            "criticSha256": "c" * 64,
            "criticTensorStateSha256": "4" * 64,
            "pairId": "1" * 64,
            "pairManifestSha256": "2" * 64,
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "publicContractSha256": __import__("v5_contract").V5_PUBLIC_CONTRACT_SHA256,
        }
        import v5_export
        import v5_train

        with mock.patch.object(
            v5_export, "v5_actor_bundle_digests", return_value=actor_digests
        ), mock.patch.object(
            v5_export, "load_v5_actor_bundle", return_value=(actor, {})
        ), mock.patch.object(
            v5_train, "load_v5_critic_checkpoint", return_value=(critic, critic_payload)
        ), mock.patch.object(
            v5_train, "load_verified_v5_behavior_pair", return_value=mixed_pair,
            create=True,
        ), mock.patch.object(
            v5_collect_cli, "sha256_file", return_value="c" * 64
        ):
            with self.assertRaisesRegex(ValueError, "pair binding mismatch: actorSha256"):
                v5_collect_cli.load_verified_behavior("actor", "critic", "pair")
        with self.assertRaisesRegex(ValueError, "mock.py"):
            v5_collect_cli._require_source_matches(
                {"gpu-training/mock.py": "a" * 64},
                {"gpu-training/mock.py": "b" * 64},
            )

    def test_parser_exposes_all_distributed_non_ssh_commands(self) -> None:
        parser = v5_collect_cli.argument_parser()
        for command in (
            "calibrate-collect",
            "benchmark-throughput",
            "calibrate-compare",
            "plan",
            "collect-shard",
            "recover-worker-slot",
            "publish-index",
        ):
            help_text = parser.format_help()
            self.assertIn(command, help_text)

    def test_dry_run_rejects_output_path_without_touching_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "must-not-exist"
            with mock.patch.object(
                v5_collect_cli, "load_verified_behavior", return_value=_behavior()
            ), mock.patch.object(
                v5_collect_cli,
                "build_source_inventory",
                return_value={"gpu-training/mock.py": "d" * 64},
            ), mock.patch.object(
                v5_collect_cli, "_verify_calibration_for_plan", return_value=_calibration()
            ):
                with self.assertRaisesRegex(ValueError, "dry-run"):
                    create_collection_plan(
                        actor_bundle="actor",
                        critic_checkpoint="critic.pt",
                        behavior_pair="pair",
                        source_root=".",
                        calibration_report="report",
                        calibration_cpu_snapshot="cpu",
                        calibration_cuda_snapshot="cuda",
                        run_namespace="v5-dry-output-s910000003",
                        seed_base=910_000_003,
                        output=target,
                        dry_run=True,
                    )
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
