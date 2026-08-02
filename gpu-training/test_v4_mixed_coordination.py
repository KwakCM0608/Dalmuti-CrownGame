from __future__ import annotations

import json
import hashlib
import io
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from v4_mixed_local_coordinator import _parser as coordinator_parser
from v4_mixed_local_coordinator import execute_phase_dag
from v4_build_mixed_package import build_package
import v4_mixed_package_runtime as package_runtime
import v4_mixed_remote_worker as worker
from v4_mixed_package_runtime import extract_source
from v4_mixed_remote_worker import _parser as worker_parser
from v4_mixed_workflow import (
    BACKEND_MAP,
    CommandSpec,
    PhaseSpec,
    RUN_NAMESPACE,
    build_mixed_phase_plan,
    canonical_json_bytes,
    canonical_sha256,
    load_recipe,
)


RECIPE_PATH = Path(__file__).with_name("v4_mixed_execution_recipe.json")


def _publish_test_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    Path(f"{path}.sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


def _publish_test_file(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    Path(f"{path}.sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


def _replace_test_json(path: Path, value: object) -> str:
    sidecar = Path(f"{path}.sha256")
    for target in (path, sidecar):
        if target.exists():
            target.chmod(0o600)
    return _publish_test_json(path, value)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def _make_writable_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            os.chmod(
                path,
                stat.S_IRUSR
                | stat.S_IWUSR
                | (stat.S_IXUSR if path.is_dir() else 0),
            )
        except OSError:
            pass
    try:
        os.chmod(root, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except OSError:
        pass


class MixedCoordinationTests(unittest.TestCase):
    def test_executor_parallelizes_only_grouped_phases_and_keeps_command_order(self) -> None:
        b_started = threading.Event()
        c_started = threading.Event()
        log: list[str] = []
        lock = threading.Lock()

        def command(name: str) -> CommandSpec:
            return CommandSpec(name, "test", (name,), ())

        phases = (
            PhaseSpec("a", (), (command("a1"), command("a2")), "pipeline"),
            PhaseSpec("b", (), (command("b1"), command("b2")), "pipeline"),
            PhaseSpec("c", ("a",), (command("c1"),), "pipeline"),
            PhaseSpec("d", ("b", "c"), (command("d1"),), None),
            PhaseSpec("e", ("d",), (command("e1"),), "other"),
            PhaseSpec("f", ("d",), (command("f1"),), None),
        )

        def runner(spec: CommandSpec) -> None:
            with lock:
                log.append(spec.command_id + "-start")
            if spec.command_id == "a1":
                self.assertTrue(b_started.wait(2.0))
            elif spec.command_id == "b1":
                b_started.set()
            elif spec.command_id == "b2":
                self.assertTrue(c_started.wait(2.0))
            elif spec.command_id == "c1":
                c_started.set()
            with lock:
                log.append(spec.command_id + "-end")

        completed = execute_phase_dag(phases, runner, max_parallel_phases=2)
        self.assertEqual(set(completed), {phase.phase_id for phase in phases})
        for first, second in (("a1", "a2"), ("b1", "b2")):
            self.assertLess(log.index(first + "-end"), log.index(second + "-start"))
        self.assertLess(log.index("a2-end"), log.index("c1-start"))
        self.assertLess(log.index("b2-end"), log.index("d1-start"))
        self.assertLess(log.index("c1-end"), log.index("d1-start"))
        # e and f have different concurrency permissions, so the earlier e
        # phase must fully finish before the ungrouped f phase can start.
        self.assertLess(log.index("e1-end"), log.index("f1-start"))

    def test_executor_failure_stops_dependent_phases(self) -> None:
        seen: list[str] = []
        phases = (
            PhaseSpec(
                "first",
                (),
                (
                    CommandSpec("good", "test", ("good",), ()),
                    CommandSpec("bad", "test", ("bad",), ()),
                    CommandSpec("never", "test", ("never",), ()),
                ),
            ),
            PhaseSpec(
                "dependent",
                ("first",),
                (CommandSpec("also-never", "test", ("also-never",), ()),),
            ),
        )

        def runner(spec: CommandSpec) -> None:
            seen.append(spec.command_id)
            if spec.command_id == "bad":
                raise RuntimeError("expected")

        with self.assertRaisesRegex(RuntimeError, "expected"):
            execute_phase_dag(phases, runner)
        self.assertEqual(seen, ["good", "bad"])

    def _worker_fixture(self, root: Path) -> dict[str, object]:
        recipe = load_recipe(RECIPE_PATH)
        run = root / "run"
        source = run / "source"
        package = run / "package"
        actor = run / "behavior-actor"
        baseline = run / "frozen-baseline"
        for directory in (
            source / "gpu-training",
            package,
            actor,
            baseline,
            run / "control" / "completions",
            run / "status",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        manifest_sha = "a" * 64
        bindings = {
            "behaviorActorBundle": str(actor.resolve()),
            "format": worker.RUNTIME_BINDING_FORMAT,
            "frozenBaselineRepository": str(baseline.resolve()),
            "packageDirectory": str(package.resolve()),
            "packageManifestSha256": manifest_sha,
            "pythonExecutable": str(Path(os.path.abspath(sys.executable))),
            "recipeSha256": canonical_sha256(recipe),
            "runDirectory": str(run.resolve()),
            "runNamespace": RUN_NAMESPACE,
            "sourceRoot": str(source.resolve()),
            "version": 1,
        }
        binding_path = run / "control" / "runtime-bindings.json"
        binding_sha = _publish_test_json(binding_path, bindings)
        loaded_bindings = worker.load_runtime_bindings(
            binding_path,
            source_root=source,
            run_directory=run,
            package_directory=package,
            package_manifest_sha256=manifest_sha,
            recipe_sha256=canonical_sha256(recipe),
        )
        return {
            "actor": actor,
            "baseline": baseline,
            "binding_path": binding_path,
            "binding_sha": binding_sha,
            "bindings": bindings,
            "loaded_bindings": loaded_bindings,
            "manifest_sha": manifest_sha,
            "package": package,
            "recipe": recipe,
            "run": run,
            "source": source,
        }

    def _publish_prerequisites(
        self, fixture: dict[str, object], command_id: str
    ) -> None:
        recipe = fixture["recipe"]
        self.assertIsInstance(recipe, dict)
        plan = worker._index_plan(build_mixed_phase_plan(recipe))
        phase, command_index, _ = plan.command_by_id[command_id]
        bindings = fixture["bindings"]
        run = fixture["run"]
        self.assertIsInstance(bindings, dict)
        self.assertIsInstance(run, Path)
        for dependency_phase, dependency in worker._required_prerequisites(
            plan, phase, command_index
        ):
            expected_values = worker._receipt_output_values(
                dependency, fixture["loaded_bindings"]
            )
            outputs: list[dict[str, object]] = []
            for template, expected in zip(
                dependency.outputs, expected_values, strict=True
            ):
                current = worker.inventory_output(Path(expected))
                local_suffix = worker._local_output_suffix(template)
                if local_suffix is not None:
                    current["path"] = f"C:/local-run/{local_suffix}"
                outputs.append(current)
            receipt = worker.build_completion_receipt(
                phase=dependency_phase,
                command=dependency,
                materialized_argv=("coordinator-verified", dependency.command_id),
                runtime_bindings=bindings,
                runtime_bindings_sha256=str(fixture["binding_sha"]),
                outputs=outputs,
            )
            worker.publish_completion_receipt(
                worker.completion_path(run, dependency.command_id), receipt
            )

    @staticmethod
    def _publish_npz_outputs(argv: list[str]) -> None:
        output = Path(argv[argv.index("--output") + 1])
        _publish_test_file(output, b"strict-test-npz")
        _publish_test_json(Path(f"{output}.metadata.json"), {"test": True})

    def _prepare_finalize_fixture(
        self, root: Path
    ) -> tuple[dict[str, object], object]:
        fixture = self._worker_fixture(root)
        source = fixture["source"]
        run = fixture["run"]
        recipe = fixture["recipe"]
        self.assertIsInstance(source, Path)
        self.assertIsInstance(run, Path)
        self.assertIsInstance(recipe, dict)

        dataset = run / "merged" / "production.npz"
        _publish_test_file(dataset, b"merged-production")
        plan_sha = "b" * 64
        _publish_test_json(
            Path(f"{dataset}.metadata.json"),
            {
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
                                "fixed-complete-mixed-backend-shard-plan-v2:"
                                f"sha256={plan_sha}"
                            ),
                        }
                    ]
                }
            },
        )

        index = worker._index_plan(build_mixed_phase_plan(recipe))
        for phase in index.phases:
            for command in phase.commands:
                if command.host != "remote":
                    continue
                script_token = command.argv[1]
                prefix = "{remote_source_root}/"
                self.assertTrue(script_token.startswith(prefix))
                script = source.joinpath(*Path(script_token[len(prefix) :]).parts)
                script.parent.mkdir(parents=True, exist_ok=True)
                if not script.exists():
                    script.write_text("# sealed worker fixture\n", encoding="utf-8")

        remote_materialized: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        output_values_by_command: dict[str, tuple[str, ...]] = {}
        for phase in index.phases:
            for command in phase.commands:
                if command.command_id not in package_runtime.FINALIZATION_COMMAND_IDS:
                    continue
                output_values = worker._receipt_output_values(
                    command, fixture["loaded_bindings"]
                )
                output_values_by_command[command.command_id] = output_values
                for value in output_values:
                    output = Path(value)
                    if output.exists():
                        continue
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"finalize-fixture")
                if command.host == "remote":
                    argv, remote_outputs = worker._materialize_command(
                        command, fixture["loaded_bindings"]
                    )
                    self.assertEqual(remote_outputs, output_values)
                    remote_materialized[command.command_id] = (
                        argv,
                        remote_outputs,
                    )

        screen_path = run / "screening" / "epoch-0001.json"
        screen_sha = _publish_test_json(screen_path, {"passed": True})
        _publish_test_json(
            run / "screening" / "epoch-0001-promotion-gates.json",
            {
                "allPlayerCountsPassed": True,
                "format": "dalmuti-v4-mixed-promotion-gates",
                "passed": True,
                "screeningReportSha256": screen_sha,
            },
        )

        for phase in index.phases:
            for command in phase.commands:
                if command.command_id not in package_runtime.FINALIZATION_COMMAND_IDS:
                    continue
                output_values = output_values_by_command[command.command_id]
                if command.host == "remote":
                    argv, _ = remote_materialized[command.command_id]
                else:
                    argv = ("coordinator-verified", command.command_id)
                outputs = []
                for template, value in zip(
                    command.outputs, output_values, strict=True
                ):
                    current = worker.inventory_output(Path(value))
                    local_suffix = worker._local_output_suffix(template)
                    if local_suffix is not None:
                        current["path"] = f"C:/local-run/{local_suffix}"
                    outputs.append(current)
                receipt = worker.build_completion_receipt(
                    phase=phase,
                    command=command,
                    materialized_argv=argv,
                    runtime_bindings=fixture["bindings"],
                    runtime_bindings_sha256=str(fixture["binding_sha"]),
                    outputs=outputs,
                )
                worker.publish_completion_receipt(
                    worker.completion_path(run, command.command_id), receipt
                )
        return fixture, index

    def _finalize_fixture(self, fixture: dict[str, object]) -> object:
        with mock.patch.object(
            worker, "_verify_frozen_baseline_inputs", return_value=()
        ):
            return worker.finalize_run(
                source_root=fixture["source"],
                run_directory=fixture["run"],
                package_directory=fixture["package"],
                package_manifest_sha256=str(fixture["manifest_sha"]),
                runtime_bindings_path=fixture["binding_path"],
            )

    def _prepare_frozen_baseline_fixture(
        self, root: Path
    ) -> dict[str, object]:
        fixture = self._worker_fixture(root)
        run = fixture["run"]
        source = fixture["source"]
        baseline = fixture["baseline"]
        self.assertIsInstance(run, Path)
        self.assertIsInstance(source, Path)
        self.assertIsInstance(baseline, Path)

        artifact_root = Path(__file__).resolve().parent.parent / "artifacts" / "rl" / "v4-frozen-baseline-git-bundle-run-001"
        source_bundle = artifact_root / worker.FROZEN_BASELINE_BUNDLE_NAME
        source_sidecar = Path(f"{source_bundle}.sha256")
        bundle_root = run / "baseline-bundle"
        bundle_root.mkdir(parents=True)
        shutil.copyfile(source_bundle, bundle_root / source_bundle.name)
        shutil.copyfile(source_sidecar, bundle_root / source_sidecar.name)

        baseline.rmdir()
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-checkout",
                str(bundle_root / source_bundle.name),
                str(baseline),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _git(baseline, "config", "core.autocrlf", "false")
        _git(baseline, "checkout", "--quiet", "--detach", worker.FROZEN_BASELINE_COMMIT)
        observation = source / "training" / "v4-public-history.ts"
        observation.parent.mkdir(parents=True, exist_ok=True)
        observation.write_bytes(
            (Path(__file__).resolve().parent.parent / "training" / "v4-public-history.ts").read_bytes()
        )
        return fixture

    def test_worker_rebuilds_exact_recipe_command_and_publishes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            self.assertIsInstance(source, Path)
            script = source / "gpu-training" / "v4_collect_fixed_match_ppo.py"
            script.write_text("# exact sealed test script\n", encoding="utf-8")
            self._publish_prerequisites(fixture, "collect-calibration-cuda")
            observed: list[list[str]] = []

            def run_exact(
                argv: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[bytes]:
                self.assertIsInstance(argv, list)
                self.assertFalse(bool(kwargs.get("shell")))
                self.assertFalse(bool(kwargs.get("check")))
                self.assertIsNotNone(kwargs.get("stdout"))
                self.assertIsNotNone(kwargs.get("stderr"))
                observed.append(argv)
                self._publish_npz_outputs(argv)
                return subprocess.CompletedProcess(argv, 0)

            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker.subprocess, "run", side_effect=run_exact),
            ):
                result = worker.execute_recipe_command(
                    source_root=source,
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )
            self.assertEqual(len(observed), 1)
            argv = observed[0]
            self.assertEqual(argv[0], str(Path(sys.executable).resolve()))
            self.assertEqual(Path(argv[1]).resolve(), script.resolve())
            self.assertEqual(argv[argv.index("--device") + 1], "cuda")
            self.assertEqual(argv[argv.index("--match-counts") + 1], "4:1,5:1,6:1,7:1,8:1,9:1,10:1")
            self.assertEqual(result["commandId"], "collect-calibration-cuda")
            receipt = worker.completion_path(
                fixture["run"], "collect-calibration-cuda"
            )
            self.assertTrue(receipt.is_file())
            self.assertTrue(Path(f"{receipt}.sha256").is_file())
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker.subprocess, "run") as duplicate_run,
                self.assertRaisesRegex(ValueError, "already exists"),
            ):
                worker.execute_recipe_command(
                    source_root=source,
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )
            duplicate_run.assert_not_called()

    def test_worker_cli_keeps_printing_child_stdout_out_of_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            run = fixture["run"]
            self.assertIsInstance(source, Path)
            self.assertIsInstance(run, Path)
            script = source / "gpu-training" / "v4_collect_fixed_match_ppo.py"
            script.write_text(
                "import hashlib, json, pathlib, sys\n"
                "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "output.parent.mkdir(parents=True, exist_ok=True)\n"
                "payload = b'npz-from-printing-child'\n"
                "output.write_bytes(payload)\n"
                "digest = hashlib.sha256(payload).hexdigest()\n"
                "pathlib.Path(str(output) + '.sha256').write_bytes(f'{digest}  {output.name}\\n'.encode('ascii'))\n"
                "metadata = pathlib.Path(str(output) + '.metadata.json')\n"
                "metadata_payload = (json.dumps({'test': True}, sort_keys=True, separators=(',', ':')) + '\\n').encode()\n"
                "metadata.write_bytes(metadata_payload)\n"
                "metadata_digest = hashlib.sha256(metadata_payload).hexdigest()\n"
                "pathlib.Path(str(metadata) + '.sha256').write_bytes(f'{metadata_digest}  {metadata.name}\\n'.encode('ascii'))\n"
                "print('child-noise-that-must-not-reach-worker-stdout')\n",
                encoding="utf-8",
            )
            self._publish_prerequisites(fixture, "collect-calibration-cuda")

            class BinaryCapture:
                def __init__(self) -> None:
                    self.buffer = io.BytesIO()

                def write(self, value: str) -> int:
                    return len(value)

                def flush(self) -> None:
                    return None

            capture = BinaryCapture()
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker.sys, "stdout", capture),
            ):
                returncode = worker.main(
                    [
                        "run-command",
                        "--source-root",
                        str(source),
                        "--run-directory",
                        str(run),
                        "--package-directory",
                        str(fixture["package"]),
                        "--package-manifest-sha256",
                        str(fixture["manifest_sha"]),
                        "--runtime-bindings",
                        str(fixture["binding_path"]),
                        "--command-id",
                        "collect-calibration-cuda",
                    ]
                )
            self.assertEqual(returncode, 0)
            result = json.loads(capture.buffer.getvalue().decode("utf-8"))
            self.assertEqual(result["commandId"], "collect-calibration-cuda")
            self.assertEqual(
                (run / "logs" / "collect-calibration-cuda.stdout").read_text(
                    encoding="utf-8"
                ),
                "child-noise-that-must-not-reach-worker-stdout\n",
            )
            self.assertEqual(
                (run / "logs" / "collect-calibration-cuda.stderr").read_bytes(),
                b"",
            )

    def test_worker_failure_status_preserves_sanitized_preflight_cause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            run = fixture["run"]
            binding_path = fixture["binding_path"]
            self.assertIsInstance(run, Path)
            self.assertIsInstance(binding_path, Path)
            Path(f"{binding_path}.sha256").write_bytes(b"stale-sidecar\n")
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                self.assertRaisesRegex(RuntimeError, "sidecar") as raised,
            ):
                worker.main(
                    [
                        "run-command",
                        "--source-root",
                        str(fixture["source"]),
                        "--run-directory",
                        str(run),
                        "--package-directory",
                        str(fixture["package"]),
                        "--package-manifest-sha256",
                        str(fixture["manifest_sha"]),
                        "--runtime-bindings",
                        str(binding_path),
                        "--command-id",
                        "collect-calibration-cuda",
                    ]
                )
            self.assertIsInstance(raised.exception.__cause__, ValueError)
            failed = run / "status" / "998-failed.json"
            status = json.loads(failed.read_bytes())
            self.assertEqual(status["stage"], "collect-calibration-cuda")
            self.assertEqual(status["state"], "failed")
            self.assertIn("errorType=ValueError", status["detail"])
            self.assertIn("message=runtime bindings sidecar", status["detail"])
            self.assertIn("stdout=unavailable", status["detail"])
            self.assertIn("stderr=unavailable", status["detail"])
            self.assertTrue(Path(f"{failed}.sha256").is_file())
            self.assertFalse((run / "status" / "999-succeeded.json").exists())

    def test_worker_child_failure_exposes_only_bound_log_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            run = fixture["run"]
            self.assertIsInstance(source, Path)
            self.assertIsInstance(run, Path)
            script = source / "gpu-training" / "v4_collect_fixed_match_ppo.py"
            script.write_bytes(
                b"import sys\n"
                b"print('child-stdout-private')\n"
                b"print('password=SHOULD_NOT_ESCAPE', file=sys.stderr)\n"
                b"raise SystemExit(7)\n"
            )
            self._publish_prerequisites(fixture, "collect-calibration-cuda")
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                self.assertRaisesRegex(RuntimeError, "exited 7") as raised,
            ):
                worker.main(
                    [
                        "run-command",
                        "--source-root",
                        str(source),
                        "--run-directory",
                        str(run),
                        "--package-directory",
                        str(fixture["package"]),
                        "--package-manifest-sha256",
                        str(fixture["manifest_sha"]),
                        "--runtime-bindings",
                        str(fixture["binding_path"]),
                        "--command-id",
                        "collect-calibration-cuda",
                    ]
                )
            failed = run / "status" / "998-failed.json"
            status = json.loads(failed.read_bytes())
            detail = status["detail"]
            self.assertIn("remote command collect-calibration-cuda exited 7", detail)
            self.assertRegex(
                detail,
                r"stdout=logs/collect-calibration-cuda\.stdout:[0-9a-f]{64}",
            )
            self.assertRegex(
                detail,
                r"stderr=logs/collect-calibration-cuda\.stderr:[0-9a-f]{64}",
            )
            for exposed in (detail, str(raised.exception)):
                self.assertNotIn("SHOULD_NOT_ESCAPE", exposed)
                self.assertNotIn("child-stdout-private", exposed)
            self.assertIn(
                b"password=SHOULD_NOT_ESCAPE",
                (run / "logs" / "collect-calibration-cuda.stderr").read_bytes(),
            )

    def test_worker_failure_status_never_coexists_with_partial_or_complete_success(self) -> None:
        for successful_parts in (("json",), ("json", "sidecar")):
            with self.subTest(successful_parts=successful_parts), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary) / "run"
                status = run / "status"
                status.mkdir(parents=True)
                succeeded = status / "999-succeeded.json"
                succeeded.write_bytes(b"partial-or-complete-success")
                if "sidecar" in successful_parts:
                    Path(f"{succeeded}.sha256").write_bytes(b"immutable")
                worker._publish_failure_status_best_effort(
                    run, "finalize-remote-run", ValueError("must not coexist")
                )
                self.assertFalse((status / "998-failed.json").exists())
                self.assertFalse(Path(f"{status / '998-failed.json'}.sha256").exists())

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            (run / "status").mkdir(parents=True)
            worker._publish_failure_status_best_effort(
                run, "finalize-remote-run", ValueError("bounded failure")
            )
            self.assertTrue((run / "status" / "998-failed.json").is_file())
            self.assertTrue(
                Path(f"{run / 'status' / '998-failed.json'}.sha256").is_file()
            )
            self.assertFalse((run / "status" / "999-succeeded.json").exists())

    def test_worker_rejects_tampered_prerequisite_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            actor = fixture["actor"]
            self.assertIsInstance(source, Path)
            self.assertIsInstance(actor, Path)
            (source / "gpu-training" / "v4_collect_fixed_match_ppo.py").write_text(
                "# test\n", encoding="utf-8"
            )
            self._publish_prerequisites(fixture, "collect-calibration-cuda")
            (actor / "injected-after-receipt.bin").write_bytes(b"tampered")
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker.subprocess, "run") as child,
                self.assertRaisesRegex(ValueError, "completion output bytes"),
            ):
                worker.execute_recipe_command(
                    source_root=source,
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )
            child.assert_not_called()

    def test_worker_rejects_skipped_dependency_and_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            self.assertIsInstance(source, Path)
            (source / "gpu-training" / "v4_collect_fixed_match_ppo.py").write_text(
                "# test\n", encoding="utf-8"
            )
            common = {
                "source_root": source,
                "run_directory": fixture["run"],
                "package_directory": fixture["package"],
                "package_manifest_sha256": fixture["manifest_sha"],
                "runtime_bindings_path": fixture["binding_path"],
                "command_id": "collect-calibration-cuda",
            }
            with mock.patch.object(
                worker,
                "_load_sealed_recipe",
                return_value=(fixture["recipe"], ()),
            ):
                with self.assertRaisesRegex(ValueError, "completion|regular file"):
                    worker.execute_recipe_command(**common)
                self._publish_prerequisites(fixture, "collect-calibration-cuda")
                with (
                    mock.patch.object(
                        worker.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ),
                    self.assertRaisesRegex(ValueError, "did not publish|incomplete"),
                ):
                    worker.execute_recipe_command(**common)

    def test_worker_rejects_stale_binding_and_dependency_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            binding_path = fixture["binding_path"]
            self.assertIsInstance(binding_path, Path)
            binding_path.write_bytes(binding_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "sidecar|canonical"):
                worker.load_runtime_bindings(
                    binding_path,
                    source_root=fixture["source"],
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    recipe_sha256=canonical_sha256(fixture["recipe"]),
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            self.assertIsInstance(source, Path)
            (source / "gpu-training" / "v4_collect_fixed_match_ppo.py").write_text(
                "# test\n", encoding="utf-8"
            )
            self._publish_prerequisites(fixture, "collect-calibration-cuda")
            dependency = worker.completion_path(fixture["run"], "verify-local-actor")
            dependency_sidecar = Path(f"{dependency}.sha256")
            dependency_sidecar.chmod(0o600)
            dependency_sidecar.write_bytes(b"0" * 64)
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                self.assertRaisesRegex(ValueError, "sidecar"),
            ):
                worker.execute_recipe_command(
                    source_root=source,
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )

    def test_worker_rejects_recipe_mutation_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._worker_fixture(root)
            source = fixture["source"]
            self.assertIsInstance(source, Path)
            (source / "gpu-training" / "v4_collect_fixed_match_ppo.py").write_text(
                "# test\n", encoding="utf-8"
            )
            changed = json.loads(json.dumps(fixture["recipe"]))
            changed["packageId"] = "mutated-package"
            with (
                mock.patch.object(
                    worker, "_load_sealed_recipe", return_value=(changed, ())
                ),
                self.assertRaisesRegex(ValueError, "recipe binding"),
            ):
                worker.execute_recipe_command(
                    source_root=source,
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )

            bindings = dict(fixture["bindings"])
            outside = root / "outside-actor"
            outside.mkdir()
            bindings["behaviorActorBundle"] = str(outside.resolve())
            binding_path = fixture["binding_path"]
            self.assertIsInstance(binding_path, Path)
            binding_path.unlink()
            Path(f"{binding_path}.sha256").unlink()
            _publish_test_json(binding_path, bindings)
            with self.assertRaisesRegex(ValueError, "escapes"):
                worker.load_runtime_bindings(
                    binding_path,
                    source_root=source,
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    recipe_sha256=canonical_sha256(fixture["recipe"]),
                )

    def test_worker_accepts_outside_venv_python_but_rejects_other_outside_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            run = fixture["run"]
            loaded = fixture["loaded_bindings"]
            self.assertIsInstance(run, Path)
            self.assertIsInstance(loaded, worker.RuntimeBindings)
            self.assertNotIn(run.resolve(), loaded.python_executable.parents)
            self.assertEqual(
                loaded.python_executable,
                Path(os.path.abspath(sys.executable)),
            )

        for field, argument in (
            ("behaviorActorBundle", None),
            ("frozenBaselineRepository", None),
            ("sourceRoot", "source_root"),
            ("packageDirectory", "package_directory"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._worker_fixture(root)
                outside = root / f"outside-{field}"
                outside.mkdir()
                bindings = dict(fixture["bindings"])
                bindings[field] = str(outside.resolve())
                binding_path = fixture["binding_path"]
                self.assertIsInstance(binding_path, Path)
                binding_path.unlink()
                Path(f"{binding_path}.sha256").unlink()
                _publish_test_json(binding_path, bindings)
                parameters = {
                    "source_root": fixture["source"],
                    "run_directory": fixture["run"],
                    "package_directory": fixture["package"],
                    "package_manifest_sha256": str(fixture["manifest_sha"]),
                    "recipe_sha256": canonical_sha256(fixture["recipe"]),
                }
                if argument is not None:
                    parameters[argument] = outside
                with self.assertRaisesRegex(ValueError, "escapes|non-canonical"):
                    worker.load_runtime_bindings(binding_path, **parameters)

    def test_worker_preserves_bound_venv_symlink_entrypoint_in_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entrypoint = root / "venv" / "bin" / "python"
            entrypoint.parent.mkdir(parents=True)
            try:
                entrypoint.symlink_to(Path(sys.executable).resolve())
            except OSError as error:
                self.skipTest(f"Python symlinks are unavailable: {error}")
            with mock.patch.object(worker.sys, "executable", str(entrypoint)):
                fixture = self._worker_fixture(root / "fixture")
                bindings = fixture["loaded_bindings"]
                self.assertIsInstance(bindings, worker.RuntimeBindings)
                self.assertEqual(bindings.python_executable, entrypoint)
                self.assertTrue(bindings.python_executable.is_symlink())
                source = fixture["source"]
                self.assertIsInstance(source, Path)
                (source / "gpu-training" / "v4_collect_fixed_match_ppo.py").write_bytes(
                    b"# sealed venv entrypoint fixture\n"
                )
                plan = worker._index_plan(
                    build_mixed_phase_plan(fixture["recipe"])
                )
                _, _, command = plan.command_by_id["collect-calibration-cuda"]
                argv, _ = worker._materialize_command(command, bindings)
                self.assertEqual(argv[0], str(entrypoint))
                self.assertNotEqual(argv[0], str(entrypoint.resolve()))

    def test_worker_rechecks_the_initial_package_source_handle_without_recapture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            source = run / "source"
            package = run / "package"
            source_training = source / "gpu-training"
            source_training.mkdir(parents=True)
            package.mkdir()
            recipe = load_recipe(RECIPE_PATH)
            recipe_path = source_training / "v4_mixed_execution_recipe.json"
            recipe_path.write_bytes(canonical_json_bytes(recipe))
            manifest_path = package / "package-manifest.json"
            manifest_path.write_bytes(b"sealed-package")
            recipe_snapshot = package_runtime.stable_snapshot(
                recipe_path, "recipe"
            )
            package_snapshot = package_runtime.stable_snapshot(
                manifest_path, "package manifest"
            )
            source_snapshots = {
                "gpu-training/v4_mixed_execution_recipe.json": recipe_snapshot
            }
            source_directories = {
                ".": worker._identity(source.lstat()),
                "gpu-training": worker._identity(source_training.lstat()),
            }
            package_identity = worker._identity(package.lstat())
            calls = {"package": 0, "source": 0}

            def load_package(*_: object, **__: object) -> object:
                calls["package"] += 1
                return (
                    {"test": True},
                    {"test": True},
                    {manifest_path.name: package_snapshot},
                    package_identity,
                    {manifest_path.name},
                )

            def verify_source(*_: object, **__: object) -> object:
                calls["source"] += 1
                return (
                    source_snapshots,
                    source_directories["."],
                    source_directories,
                )

            def mutate_after_verification(*_: object, **__: object) -> object:
                (source / "sitecustomize.py").write_bytes(b"injected")
                return recipe, recipe_snapshot

            with (
                mock.patch.object(worker, "_load_package", side_effect=load_package),
                mock.patch.object(
                    worker, "_verify_extracted_source", side_effect=verify_source
                ),
                mock.patch.object(
                    worker,
                    "_load_package_recipe",
                    side_effect=mutate_after_verification,
                ),
                self.assertRaisesRegex(
                    ValueError, "unbound file|inventory|identity"
                ),
            ):
                worker._load_sealed_recipe(source, package, "a" * 64)
            self.assertEqual(calls, {"package": 1, "source": 1})

    def test_worker_loads_a_real_built_and_extracted_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            run = root / "run"
            repository.mkdir()
            run.mkdir()
            try:
                _git(repository, "init", "--quiet")
                _git(repository, "config", "user.email", "test@example.invalid")
                _git(repository, "config", "user.name", "Worker Package Test")
                _git(repository, "config", "core.autocrlf", "false")
                recipe = load_recipe(RECIPE_PATH)
                recipe_path = "gpu-training/v4_mixed_execution_recipe.json"
                for relative in recipe["sourcePaths"]:
                    target = repository.joinpath(*Path(relative).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if relative == recipe_path:
                        target.write_bytes(canonical_json_bytes(recipe))
                    elif relative == recipe["packagingBuilderPath"]:
                        target.write_bytes(
                            Path(__file__)
                            .with_name("v4_build_mixed_package.py")
                            .read_bytes()
                        )
                    elif relative == recipe["runtimeVerifierPath"]:
                        target.write_bytes(
                            Path(__file__)
                            .with_name("v4_mixed_package_runtime.py")
                            .read_bytes()
                        )
                    else:
                        target.write_bytes(b"# sealed package fixture\n")
                _git(repository, "add", ".")
                _git(repository, "commit", "--quiet", "-m", "sealed worker fixture")
                package = run / "package"
                build = build_package(
                    repository, "HEAD", recipe_path, package
                )
                source = run / "source"
                extract_source(
                    package,
                    str(build["packageManifestSha256"]),
                    source,
                )
                loaded, snapshots = worker._load_sealed_recipe(
                    source, package, str(build["packageManifestSha256"])
                )
                self.assertEqual(loaded, recipe)
                self.assertEqual(len(snapshots), 1)
                verification = snapshots[0]
                self.assertIsInstance(
                    verification, package_runtime.RemoteSourceVerification
                )
                self.assertIn(
                    source / recipe_path,
                    {
                        snapshot.path
                        for snapshot in verification.source_snapshots.values()
                    },
                )
                self.assertIn(
                    package / "package-manifest.json",
                    {
                        snapshot.path
                        for snapshot in verification.package_snapshots.values()
                    },
                )
            finally:
                _make_writable_tree(root)

    def test_worker_rejects_duplicate_reordered_and_forbidden_plan(self) -> None:
        command = CommandSpec(
            "test-command", "remote", (sys.executable, "script.py"), ()
        )
        first = PhaseSpec("first", (), (command,))
        duplicate = PhaseSpec("first", (), ())
        with self.assertRaisesRegex(ValueError, "duplicate workflow phase"):
            worker._index_plan((first, duplicate))
        dependent = PhaseSpec("dependent", ("first",), ())
        with self.assertRaisesRegex(ValueError, "reordered"):
            worker._index_plan((dependent, first))
        duplicate_command = PhaseSpec("second", (), (command,))
        with self.assertRaisesRegex(ValueError, "duplicate workflow command"):
            worker._index_plan((first, duplicate_command))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._worker_fixture(root)
            source = fixture["source"]
            self.assertIsInstance(source, Path)
            script = source / "gpu-training" / "v4_train.py"
            script.write_text("# test\n", encoding="utf-8")
            forbidden = CommandSpec(
                "forbidden-command",
                "remote",
                (
                    "{remote_python}",
                    "{remote_source_root}/gpu-training/v4_train.py",
                    "--deploy",
                ),
                (),
            )
            phase = PhaseSpec("forbidden-phase", (), (forbidden,))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                worker._materialize_command(
                    forbidden, fixture["loaded_bindings"]
                )
            escaped = CommandSpec(
                "escaped-output",
                "remote",
                (
                    "{remote_python}",
                    "{remote_source_root}/gpu-training/v4_train.py",
                ),
                (str((root / "outside-output.json").resolve()),),
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                worker._materialize_command(
                    escaped, fixture["loaded_bindings"]
                )

    def test_worker_rejects_mutated_exact_argv_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            self.assertIsInstance(source, Path)
            script = source / "gpu-training" / "v4_collect_fixed_match_ppo.py"
            script.write_text("# sealed command\n", encoding="utf-8")
            plan = worker._index_plan(
                build_mixed_phase_plan(fixture["recipe"])
            )
            phase, _, command = plan.command_by_id["collect-calibration-cuda"]
            argv, outputs = worker._materialize_command(
                command, fixture["loaded_bindings"]
            )
            output_records = [
                {
                    "kind": "file",
                    "path": str(Path(path).resolve(strict=False)),
                    "sha256": "0" * 64,
                    "size": 0,
                }
                for path in outputs
            ]
            receipt = worker.build_completion_receipt(
                phase=phase,
                command=command,
                materialized_argv=(*argv, "--mutated-argv"),
                runtime_bindings=fixture["bindings"],
                runtime_bindings_sha256=str(fixture["binding_sha"]),
                outputs=output_records,
            )
            worker.publish_completion_receipt(
                worker.completion_path(
                    fixture["run"], "collect-calibration-cuda"
                ),
                receipt,
            )
            with self.assertRaisesRegex(ValueError, "exact argv drifted"):
                worker._load_completion(
                    fixture["run"],
                    phase,
                    command,
                    fixture["loaded_bindings"],
                )

    def test_worker_rechecks_dependency_receipts_after_child_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            self.assertIsInstance(source, Path)
            script = source / "gpu-training" / "v4_collect_fixed_match_ppo.py"
            script.write_text("# exact sealed test script\n", encoding="utf-8")
            self._publish_prerequisites(fixture, "collect-calibration-cuda")
            dependency = worker.completion_path(
                fixture["run"], "verify-local-actor"
            )

            def mutate_after_launch(
                argv: list[str], **_: object
            ) -> subprocess.CompletedProcess[bytes]:
                self._publish_npz_outputs(argv)
                dependency.chmod(0o600)
                dependency.write_bytes(dependency.read_bytes() + b" ")
                return subprocess.CompletedProcess(argv, 0)

            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(
                    worker.subprocess, "run", side_effect=mutate_after_launch
                ),
                self.assertRaisesRegex(ValueError, "changed after validation"),
            ):
                worker.execute_recipe_command(
                    source_root=source,
                    run_directory=fixture["run"],
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )
            self.assertFalse(
                worker.completion_path(
                    fixture["run"], "collect-calibration-cuda"
                ).exists()
            )

    def test_worker_rolls_back_receipt_on_output_replacement_during_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            run = fixture["run"]
            self.assertIsInstance(source, Path)
            self.assertIsInstance(run, Path)
            (source / "gpu-training" / "v4_collect_fixed_match_ppo.py").write_text(
                "# test\n", encoding="utf-8"
            )
            self._publish_prerequisites(fixture, "collect-calibration-cuda")
            real_publish = worker.publish_completion_receipt

            def run_exact(
                argv: list[str], **_: object
            ) -> subprocess.CompletedProcess[bytes]:
                self._publish_npz_outputs(argv)
                return subprocess.CompletedProcess(argv, 0)

            def publish_then_replace(path: Path, receipt: object) -> str:
                digest = real_publish(path, receipt)
                output = run / "calibration" / "cuda.npz"
                replacement = run / "calibration" / "replacement.npz"
                replacement.write_bytes(output.read_bytes())
                os.replace(replacement, output)
                return digest

            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker.subprocess, "run", side_effect=run_exact),
                mock.patch.object(
                    worker,
                    "publish_completion_receipt",
                    side_effect=publish_then_replace,
                ),
                self.assertRaisesRegex(ValueError, "changed after validation"),
            ):
                worker.execute_recipe_command(
                    source_root=source,
                    run_directory=run,
                    package_directory=fixture["package"],
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )
            receipt_path = worker.completion_path(
                run, "collect-calibration-cuda"
            )
            self.assertFalse(receipt_path.exists())
            self.assertFalse(Path(f"{receipt_path}.sha256").exists())

    def test_worker_rolls_back_receipt_on_postpublish_source_injection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            run = fixture["run"]
            package = fixture["package"]
            self.assertIsInstance(source, Path)
            self.assertIsInstance(run, Path)
            self.assertIsInstance(package, Path)
            (source / "gpu-training" / "v4_collect_fixed_match_ppo.py").write_text(
                "# test\n", encoding="utf-8"
            )
            self._publish_prerequisites(fixture, "collect-calibration-cuda")
            sealed = (
                worker._capture_artifact(source, run),
                worker._capture_artifact(package, run),
            )
            real_publish = worker.publish_completion_receipt

            def run_exact(
                argv: list[str], **_: object
            ) -> subprocess.CompletedProcess[bytes]:
                self._publish_npz_outputs(argv)
                return subprocess.CompletedProcess(argv, 0)

            def publish_then_inject(path: Path, receipt: object) -> str:
                digest = real_publish(path, receipt)
                (source / "sitecustomize.py").write_bytes(b"# injected\n")
                return digest

            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], sealed),
                ),
                mock.patch.object(worker.subprocess, "run", side_effect=run_exact),
                mock.patch.object(
                    worker,
                    "publish_completion_receipt",
                    side_effect=publish_then_inject,
                ),
                self.assertRaisesRegex(ValueError, "inventory|hash|size"),
            ):
                worker.execute_recipe_command(
                    source_root=source,
                    run_directory=run,
                    package_directory=package,
                    package_manifest_sha256=str(fixture["manifest_sha"]),
                    runtime_bindings_path=fixture["binding_path"],
                    command_id="collect-calibration-cuda",
                )
            receipt_path = worker.completion_path(
                run, "collect-calibration-cuda"
            )
            self.assertFalse(receipt_path.exists())
            self.assertFalse(Path(f"{receipt_path}.sha256").exists())

    def test_worker_resolves_plan_sha_from_strict_merged_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._worker_fixture(Path(temporary))
            source = fixture["source"]
            run = fixture["run"]
            self.assertIsInstance(source, Path)
            self.assertIsInstance(run, Path)
            (source / "gpu-training" / "v4_train.py").write_text(
                "# test\n", encoding="utf-8"
            )
            dataset = run / "merged" / "production.npz"
            _publish_test_file(dataset, b"merged-npz")
            plan_sha = "b" * 64
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
                            "opaqueId": f"fixed-complete-mixed-backend-shard-plan-v2:sha256={plan_sha}",
                        }
                    ]
                }
            }
            metadata_path = Path(f"{dataset}.metadata.json")
            _publish_test_json(metadata_path, metadata)
            index = worker._index_plan(build_mixed_phase_plan(fixture["recipe"]))
            _, _, training = index.command_by_id["train-epoch-one-cuda"]
            argv, _ = worker._materialize_command(
                training, fixture["loaded_bindings"]
            )
            self.assertEqual(
                argv[argv.index("--expected-fixed-collection-plan-sha256") + 1],
                plan_sha,
            )
            Path(f"{metadata_path}.sha256").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "sidecar"):
                worker._materialize_command(training, fixture["loaded_bindings"])

    def test_parallel_remote_shards_do_not_require_each_other(self) -> None:
        recipe = load_recipe(RECIPE_PATH)
        index = worker._index_plan(build_mixed_phase_plan(recipe))
        phase_two, offset_two, _ = index.command_by_id[
            "collect-production-shard-02"
        ]
        phase_three, offset_three, _ = index.command_by_id[
            "collect-production-shard-03"
        ]
        required_two = {
            command.command_id
            for _, command in worker._required_prerequisites(
                index, phase_two, offset_two
            )
        }
        required_three = {
            command.command_id
            for _, command in worker._required_prerequisites(
                index, phase_three, offset_three
            )
        }
        self.assertNotIn("collect-production-shard-03", required_two)
        self.assertNotIn("collect-production-shard-02", required_three)

        hard_gate_phase, hard_gate_offset, _ = index.command_by_id[
            "verify-epoch-one-hard-gates"
        ]
        hard_gate_required = {
            command.command_id
            for _, command in worker._required_prerequisites(
                index, hard_gate_phase, hard_gate_offset
            )
        }
        self.assertIn("publish-candidate-actor-sidecar", hard_gate_required)

    def test_finalize_run_publishes_exact_audit_and_semantic_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture, _ = self._prepare_finalize_fixture(Path(temporary))
            run = fixture["run"]
            self.assertIsInstance(run, Path)
            before = set((run / "control" / "completions").iterdir())
            seal_digest = "d" * 64
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker, "verify_screening") as verify_screen,
                mock.patch.object(
                    worker,
                    "seal_run",
                    return_value={"sealSha256": seal_digest},
                ) as seal,
                mock.patch.object(
                    worker,
                    "write_status",
                    return_value={"runSealSha256": seal_digest},
                ) as status,
                mock.patch.object(
                    worker, "verify_run_seal", return_value=seal_digest
                ) as verify_seal,
            ):
                result = self._finalize_fixture(fixture)
            self.assertTrue(result["passed"])
            audit_path = run / "provenance" / "finalization-audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [row["commandId"] for row in audit["requiredCommands"]],
                list(package_runtime.FINALIZATION_COMMAND_IDS),
            )
            self.assertEqual(
                audit["packageManifestSha256"], fixture["manifest_sha"]
            )
            self.assertEqual(audit["recipeSha256"], canonical_sha256(fixture["recipe"]))
            self.assertEqual(audit["runtimeBindingsSha256"], fixture["binding_sha"])
            audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            seal.assert_called_once_with(
                run,
                run / "provenance" / "final-files.json",
                run / "status",
                run / "screening" / "epoch-0001.json",
                run / "screening" / "epoch-0001-promotion-gates.json",
                fixture["manifest_sha"],
                canonical_sha256(fixture["recipe"]),
                canonical_sha256(fixture["recipe"]["runContract"]),
                fixture["binding_sha"],
                audit_path,
                audit_sha,
                profile="remote-semantic",
            )
            verify_screen.assert_called_once()
            verify_seal.assert_called_once_with(
                run / "provenance" / "final-files.json"
            )
            self.assertEqual(
                status.call_args.args[4], run / "provenance" / "final-files.json"
            )
            self.assertEqual(
                before, set((run / "control" / "completions").iterdir())
            )
            self.assertFalse(
                worker.completion_path(run, "finalize-remote-run").exists()
            )

    def test_finalize_run_rejects_missing_and_tampered_receipts(self) -> None:
        for mode in ("missing", "tampered"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture, _ = self._prepare_finalize_fixture(Path(temporary))
                run = fixture["run"]
                self.assertIsInstance(run, Path)
                receipt = worker.completion_path(run, "verify-local-actor")
                sidecar = Path(f"{receipt}.sha256")
                if mode == "missing":
                    sidecar.chmod(0o600)
                    sidecar.unlink()
                else:
                    sidecar.chmod(0o600)
                    sidecar.write_bytes(b"0" * 64)
                with (
                    mock.patch.object(
                        worker,
                        "_load_sealed_recipe",
                        return_value=(fixture["recipe"], ()),
                    ),
                    mock.patch.object(worker, "verify_screening") as screen,
                    mock.patch.object(worker, "seal_run") as seal,
                    self.assertRaisesRegex(ValueError, "sidecar|missing"),
                ):
                    self._finalize_fixture(fixture)
                screen.assert_not_called()
                seal.assert_not_called()
                self.assertFalse(
                    (run / "provenance" / "finalization-audit.json").exists()
                )

    def test_finalize_run_rejects_tampered_current_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture, _ = self._prepare_finalize_fixture(Path(temporary))
            run = fixture["run"]
            self.assertIsInstance(run, Path)
            (run / "rollouts" / "shard-02.npz").write_bytes(
                b"tampered-after-completion"
            )
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker, "verify_screening") as screen,
                mock.patch.object(worker, "seal_run") as seal,
                self.assertRaisesRegex(ValueError, "completion output bytes"),
            ):
                self._finalize_fixture(fixture)
            screen.assert_not_called()
            seal.assert_not_called()

    def test_finalize_run_rejects_local_suffix_and_counterpart_drift(self) -> None:
        for mode in ("suffix", "counterpart"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture, _ = self._prepare_finalize_fixture(Path(temporary))
                run = fixture["run"]
                self.assertIsInstance(run, Path)
                receipt_path = worker.completion_path(
                    run, "collect-calibration-cpu"
                )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if mode == "suffix":
                    receipt["outputs"][0]["path"] = (
                        "C:/local-run/calibration/wrong-cpu.npz"
                    )
                    message = "wrong canonical suffix"
                else:
                    receipt["outputs"][0]["sha256"] = "f" * 64
                    message = "counterpart bytes drifted"
                _replace_test_json(receipt_path, receipt)
                with (
                    mock.patch.object(
                        worker,
                        "_load_sealed_recipe",
                        return_value=(fixture["recipe"], ()),
                    ),
                    mock.patch.object(worker, "verify_screening") as screen,
                    mock.patch.object(worker, "seal_run") as seal,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    self._finalize_fixture(fixture)
                screen.assert_not_called()
                seal.assert_not_called()

    def test_finalize_run_rejects_package_inventory_injection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture, _ = self._prepare_finalize_fixture(Path(temporary))
            source = fixture["source"]
            run = fixture["run"]
            package = fixture["package"]
            self.assertIsInstance(source, Path)
            self.assertIsInstance(run, Path)
            self.assertIsInstance(package, Path)
            sealed = (
                worker._capture_artifact(source, run),
                worker._capture_artifact(package, run),
            )

            def inject_package(*_: object, **__: object) -> None:
                (package / "torch.py").write_bytes(b"# injected\n")

            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], sealed),
                ),
                mock.patch.object(
                    worker, "verify_screening", side_effect=inject_package
                ),
                mock.patch.object(worker, "seal_run") as seal,
                self.assertRaisesRegex(ValueError, "inventory|hash|size"),
            ):
                self._finalize_fixture(fixture)
            seal.assert_not_called()
            self.assertFalse(
                (run / "provenance" / "finalization-audit.json").exists()
            )

    def test_frozen_baseline_protection_rejects_source_tamper_before_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                fixture = self._prepare_frozen_baseline_fixture(root)
                protections = worker._verify_frozen_baseline_inputs(
                    fixture["loaded_bindings"]
                )
                baseline = fixture["baseline"]
                self.assertIsInstance(baseline, Path)
                normal = baseline / "lib" / "bot-strategy.ts"
                normal.write_bytes(normal.read_bytes() + b"\n// tampered\n")
                with self.assertRaisesRegex(
                    ValueError, "bytes|hash|size|changed|inventory"
                ):
                    worker._recheck_protections(protections)
            finally:
                _make_writable_tree(root)

    def test_frozen_baseline_rejects_commit_drift_before_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                fixture = self._prepare_frozen_baseline_fixture(root)
                baseline = fixture["baseline"]
                self.assertIsInstance(baseline, Path)
                _git(baseline, "config", "user.email", "test@example.invalid")
                _git(baseline, "config", "user.name", "Baseline Drift Test")
                _git(baseline, "commit", "--quiet", "--allow-empty", "-m", "drift")
                with self.assertRaisesRegex(ValueError, "commit drifted"):
                    worker._verify_frozen_baseline_inputs(
                        fixture["loaded_bindings"]
                    )
            finally:
                _make_writable_tree(root)

    def test_frozen_baseline_rejects_untracked_file_before_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                fixture = self._prepare_frozen_baseline_fixture(root)
                baseline = fixture["baseline"]
                self.assertIsInstance(baseline, Path)
                (baseline / "untracked-injection.bin").write_bytes(b"injected")
                with self.assertRaisesRegex(ValueError, "exact clean checkout"):
                    worker._verify_frozen_baseline_inputs(
                        fixture["loaded_bindings"]
                    )
            finally:
                _make_writable_tree(root)

    def test_finalize_run_rejects_candidate_and_hard_gate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture, _ = self._prepare_finalize_fixture(Path(temporary))
            run = fixture["run"]
            self.assertIsInstance(run, Path)
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(
                    worker,
                    "verify_screening",
                    side_effect=ValueError("screening Actor binding mismatch"),
                ),
                mock.patch.object(worker, "seal_run") as seal,
                self.assertRaisesRegex(ValueError, "Actor binding mismatch"),
            ):
                self._finalize_fixture(fixture)
            seal.assert_not_called()
            self.assertFalse(
                (run / "provenance" / "finalization-audit.json").exists()
            )

        with tempfile.TemporaryDirectory() as temporary:
            fixture, _ = self._prepare_finalize_fixture(Path(temporary))
            run = fixture["run"]
            self.assertIsInstance(run, Path)
            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker, "verify_screening"),
                mock.patch.object(
                    worker,
                    "seal_run",
                    side_effect=ValueError("hard-gate candidate binding mismatch"),
                ),
                mock.patch.object(worker, "write_status") as status,
                self.assertRaisesRegex(ValueError, "hard-gate candidate binding mismatch"),
            ):
                self._finalize_fixture(fixture)
            status.assert_not_called()

    def test_finalize_run_has_no_fallible_post_success_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture, _ = self._prepare_finalize_fixture(Path(temporary))
            seal_digest = "d" * 64
            events: list[str] = []

            def verify_seal(_: Path) -> str:
                events.append("verify-seal")
                return seal_digest

            def fail_success(*_: object) -> object:
                events.append("publish-success")
                raise ValueError("success postcheck failed")

            with (
                mock.patch.object(
                    worker,
                    "_load_sealed_recipe",
                    return_value=(fixture["recipe"], ()),
                ),
                mock.patch.object(worker, "verify_screening"),
                mock.patch.object(
                    worker,
                    "seal_run",
                    return_value={"sealSha256": seal_digest},
                ),
                mock.patch.object(
                    worker,
                    "write_status",
                    side_effect=fail_success,
                ),
                mock.patch.object(
                    worker, "verify_run_seal", side_effect=verify_seal
                ),
                self.assertRaisesRegex(ValueError, "success postcheck failed"),
            ):
                self._finalize_fixture(fixture)
            self.assertEqual(events, ["verify-seal", "publish-success"])
            success = fixture["run"] / "status" / "999-succeeded.json"
            self.assertFalse(success.exists())
            self.assertFalse(Path(f"{success}.sha256").exists())

    def test_coordinator_and_worker_recipe_entrypoints_parse(self) -> None:
        self.assertEqual(
            coordinator_parser().parse_args(["dry-run"]).command, "dry-run"
        )
        self.assertEqual(
            coordinator_parser()
            .parse_args(
                [
                    "verify-actor",
                    "--actor-bundle",
                    "actor",
                    "--expected-actor-sha256",
                    "a" * 64,
                    "--expected-manifest-sha256",
                    "b" * 64,
                ]
            )
            .command,
            "verify-actor",
        )
        parsed = worker_parser().parse_args(
            [
                "run-command",
                "--source-root",
                "source",
                "--package-directory",
                "package",
                "--package-manifest-sha256",
                "c" * 64,
                "--run-directory",
                "run",
                "--runtime-bindings",
                "run/control/runtime-bindings.json",
                "--command-id",
                "collect-calibration-cuda",
            ]
        )
        self.assertEqual(parsed.command, "run-command")
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                worker_parser().parse_args(
                    [
                        "run-command",
                        "--source-root",
                        "source",
                        "--package-directory",
                        "package",
                        "--package-manifest-sha256",
                        "c" * 64,
                        "--run-directory",
                        "run",
                        "--runtime-bindings",
                        "run/control/runtime-bindings.json",
                        "--command-id",
                        "collect-calibration-cuda",
                        "--argv",
                        "python mutated.py",
                    ]
                )
        self.assertEqual(
            worker_parser()
            .parse_args(
                [
                    "finalize-run",
                    "--source-root",
                    "source",
                    "--package-directory",
                    "package",
                    "--package-manifest-sha256",
                    "c" * 64,
                    "--run-directory",
                    "run",
                    "--runtime-bindings",
                    "run/control/runtime-bindings.json",
                ]
            )
            .command,
            "finalize-run",
        )


if __name__ == "__main__":
    unittest.main()
