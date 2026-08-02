from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import v4_mixed_local_coordinator as coordinator
import v4_mixed_package_runtime as runtime

from v4_mixed_local_coordinator import (
    MERGED_PLAN_SHA_SENTINEL,
    MixedCommandRunner,
    SshTransport,
    _materialized_replacements,
    _parser,
    _remote_python_probe_argv,
    _retrieve_file_family,
    _upload_file_family,
    execute_mixed_workflow,
    materialize_phase_plan,
    verify_artifact_family,
    verify_and_seal,
    validate_run_layout,
)
from v4_mixed_package_runtime import canonical_json_bytes, seal_run
from v4_mixed_workflow import build_mixed_phase_plan, load_recipe
from v4_mixed_workflow import CommandSpec


def _sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )


def _npz_family(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"npz")
    _sidecar(path)
    metadata = Path(f"{path}.metadata.json")
    metadata.write_bytes(canonical_json_bytes({"fixture": path.name}))
    _sidecar(metadata)


class _DownloadTransport:
    def __init__(self, files: dict[str, Path]) -> None:
        self.files = files

    def download(self, remote_paths: tuple[str, ...], destination: Path) -> None:
        for remote in remote_paths:
            shutil.copy2(self.files[remote], destination / Path(remote).name)


class MixedLocalCoordinatorTests(unittest.TestCase):
    def test_remote_python_probe_preserves_virtualenv_entrypoint(self) -> None:
        argv = _remote_python_probe_argv("/home/pangmin/dalmuti/.venv/bin/python")
        self.assertEqual(argv[0], "/home/pangmin/dalmuti/.venv/bin/python")
        self.assertIn("abspath(sys.executable)", argv[2])
        self.assertNotIn("realpath", argv[2])

    def test_ssh_and_scp_are_noninteractive_bounded_and_accept_only_new_hosts(self) -> None:
        transport = SshTransport(
            endpoint="pangmin@220.70.2.226",
            port=2222,
            identity_file=None,
            ssh_executable="ssh",
            scp_executable="scp",
        )
        ssh = transport.ssh_arguments()
        scp = transport.scp_arguments()
        for argv in (ssh, scp):
            joined = " ".join(argv)
            self.assertIn("BatchMode=yes", joined)
            self.assertIn("IdentitiesOnly=yes", joined)
            self.assertIn("PasswordAuthentication=no", joined)
            self.assertIn("StrictHostKeyChecking=accept-new", joined)
            self.assertIn("ConnectTimeout=10", joined)
            self.assertIn("ConnectionAttempts=1", joined)
            self.assertIn("ServerAliveInterval=15", joined)
            self.assertIn("ServerAliveCountMax=3", joined)

    def test_execute_cli_requires_full_local_and_remote_contract(self) -> None:
        parsed = _parser().parse_args(
            [
                "execute",
                "--source-root",
                "source",
                "--package-directory",
                "package",
                "--package-manifest-sha256",
                "a" * 64,
                "--local-run-directory",
                "local-run",
                "--remote-endpoint",
                "pangmin@220.70.2.226",
                "--remote-run-directory",
                "/home/pangmin/dalmuti/run-001",
                "--behavior-actor-bundle",
                "actor",
                "--frozen-baseline-bundle",
                "baseline.bundle",
            ]
        )
        self.assertEqual(parsed.command, "execute")
        self.assertEqual(parsed.remote_endpoint, "pangmin@220.70.2.226")
        self.assertEqual(parsed.frozen_baseline_bundle, Path("baseline.bundle"))

    def test_materialized_plan_uses_exact_paths_and_only_deferred_plan_sentinel(self) -> None:
        recipe_path = Path(__file__).with_name("v4_mixed_execution_recipe.json")
        recipe = load_recipe(recipe_path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            actor = root / "actor"
            run = root / "v4-fixedid-ppo-i001-mixedmathfp32-s620000001-local-run-001"
            for path in (source, actor, run):
                path.mkdir()
            replacements = _materialized_replacements(
                source_root=source,
                local_run=run,
                actor_bundle=actor,
                local_python=str(Path(os.__file__).resolve()),
                remote_run="/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-run-001",
                remote_python="/usr/bin/python3",
                package_manifest_sha256="b" * 64,
            )
            phases = materialize_phase_plan(
                build_mixed_phase_plan(recipe), replacements
            )
            commands = {
                command.command_id: command
                for phase in phases
                for command in phase.commands
            }
            for command in commands.values():
                self.assertFalse(any("{" in token or "}" in token for token in command.argv))
            self.assertIn(
                MERGED_PLAN_SHA_SENTINEL,
                commands["train-epoch-one-cuda"].argv,
            )
            self.assertEqual(len(commands["upload-merged-production"].outputs), 12)

    def test_recipe_enforces_exact_local_and_remote_run_paths(self) -> None:
        recipe = load_recipe(Path(__file__).with_name("v4_mixed_execution_recipe.json"))
        local = Path(
            "C:/runs/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-local-run-001"
        )
        remote = "/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-run-001"
        self.assertEqual(validate_run_layout(recipe, local, remote), remote)
        with self.assertRaisesRegex(ValueError, "local run directory"):
            validate_run_layout(recipe, local.with_name("wrong-local"), remote)
        with self.assertRaisesRegex(ValueError, "remote run directory"):
            validate_run_layout(recipe, local, remote + "-wrong")

    def test_npz_and_calibration_families_are_exact_and_sidecar_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz = root / "single" / "value.npz"
            _npz_family(npz)
            self.assertEqual(len(verify_artifact_family("npz", (npz,))), 4)
            calibration = root / "calibration"
            report = calibration / "backend-comparison.json"
            report.parent.mkdir()
            report.write_bytes(canonical_json_bytes({"passed": True}))
            _sidecar(report)
            cpu = calibration / "cpu.npz"
            cuda = calibration / "cuda.npz"
            _npz_family(cpu)
            _npz_family(cuda)
            self.assertEqual(
                len(
                    verify_artifact_family(
                        "calibration-triple", (report, cpu, cuda)
                    )
                ),
                10,
            )
            (calibration / "unexpected.bin").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "exact ten-file"):
                verify_artifact_family(
                    "calibration-triple", (report, cpu, cuda)
                )

    def test_retrieve_npz_is_atomic_and_missing_companion_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote_root = root / "remote"
            remote = remote_root / "cuda.npz"
            _npz_family(remote)
            remote_names = (
                "/remote/cuda.npz",
                "/remote/cuda.npz.sha256",
                "/remote/cuda.npz.metadata.json",
                "/remote/cuda.npz.metadata.json.sha256",
            )
            files = {
                name: remote_root / Path(name).name for name in remote_names
            }
            local_run = root / "local"
            (local_run / "control").mkdir(parents=True)
            local = local_run / "calibration" / "cuda.npz"
            context = SimpleNamespace(
                local_run_directory=local_run,
                transport=_DownloadTransport(files),
            )
            records = _retrieve_file_family(
                context,
                remote_roots=(remote_names[0],),
                local_roots=(local,),
                kind="npz",
            )
            self.assertEqual(len(records), 4)
            broken_run = root / "broken-local"
            (broken_run / "control").mkdir(parents=True)
            broken = broken_run / "calibration" / "cuda.npz"
            incomplete = dict(files)
            incomplete.pop(remote_names[-1])
            broken_context = SimpleNamespace(
                local_run_directory=broken_run,
                transport=_DownloadTransport(incomplete),
            )
            with self.assertRaises((KeyError, FileNotFoundError)):
                _retrieve_file_family(
                    broken_context,
                    remote_roots=(remote_names[0],),
                    local_roots=(broken,),
                    kind="npz",
                )
            self.assertFalse(broken.exists())

    def test_calibration_upload_never_overwrites_remote_cuda_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "calibration" / "backend-comparison.json"
            report.parent.mkdir()
            report.write_bytes(canonical_json_bytes({"passed": True}))
            _sidecar(report)
            cpu = report.parent / "cpu.npz"
            cuda = report.parent / "cuda.npz"
            _npz_family(cpu)
            _npz_family(cuda)
            uploaded: list[str] = []

            class Transport:
                def run(self, argv: list[str], *, capture: bool = False) -> str:
                    return ""

                def upload(self, paths: tuple[Path, ...], remote: str) -> None:
                    del remote
                    uploaded.extend(path.name for path in paths)

            context = SimpleNamespace(transport=Transport())
            remote_roots = (
                "/run/calibration/backend-comparison.json",
                "/run/calibration/cpu.npz",
                "/run/calibration/cuda.npz",
            )
            remote_paths = coordinator._remote_family_paths(
                "calibration-triple", remote_roots
            )

            def record(path: str) -> dict[str, object]:
                return {
                    "kind": "file",
                    "path": path,
                    "sha256": "a" * 64,
                    "size": 1,
                }

            full = [record(path) for path in remote_paths]
            with mock.patch.object(
                coordinator,
                "_remote_inventory",
                return_value=full[6:],
            ), mock.patch.object(
                coordinator,
                "_remote_verify_family",
                return_value=full,
            ):
                records = _upload_file_family(
                    context,
                    local_roots=(report, cpu, cuda),
                    remote_roots=remote_roots,
                    kind="calibration-triple",
                )
            self.assertEqual(records, full)
            self.assertEqual(
                uploaded,
                [
                    "backend-comparison.json",
                    "backend-comparison.json.sha256",
                    "cpu.npz",
                    "cpu.npz.sha256",
                    "cpu.npz.metadata.json",
                    "cpu.npz.metadata.json.sha256",
                ],
            )
            self.assertNotIn("cuda.npz", uploaded)

    def test_top_level_preflight_failure_always_leaves_998_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "v4-fixedid-ppo-i001-mixedmathfp32-s620000001-local-run-001"
            (run / "status").mkdir(parents=True)
            (run / "logs").mkdir()
            (run / "source").mkdir()
            with self.assertRaises(FileNotFoundError):
                execute_mixed_workflow(
                    source_root=run / "source",
                    package_directory=root / "missing-package",
                    package_manifest_sha256="a" * 64,
                    local_run_directory=run,
                    remote_endpoint="pangmin@220.70.2.226",
                    remote_run_directory="/home/pangmin/dalmuti/v4-fixedid-ppo-i001-mixedmathfp32-s620000001-run-001",
                    behavior_actor_bundle=root / "actor",
                    frozen_baseline_bundle=root / "baseline.bundle",
                    port=2222,
                    identity_file=None,
                    local_python="python",
                    remote_python="python3",
                    ssh_executable="ssh",
                    scp_executable="scp",
                )
            failure = run / "status" / "998-failed.json"
            self.assertTrue(failure.is_file())
            value = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(value["state"], "failed")
            _sidecar_value = Path(f"{failure}.sha256").read_bytes()
            self.assertIn(hashlib.sha256(failure.read_bytes()).hexdigest().encode(), _sidecar_value)

    def test_local_commands_recheck_sealed_source_before_and_after_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            local_run = root / "run"
            (local_run / "control").mkdir(parents=True)
            runner = object.__new__(MixedCommandRunner)
            runner.context = SimpleNamespace(
                source_root=source,
                local_run_directory=local_run,
                source_snapshots={"fixture": object()},
                source_root_identity=object(),
                source_directory_identities={".": object()},
            )
            command = CommandSpec(
                "fixture-local",
                "local",
                (sys.executable, "-c", "pass"),
                (),
            )
            with mock.patch.object(
                coordinator, "_recheck_extracted_source"
            ) as recheck:
                self.assertEqual(runner._run_local(command), [])
            self.assertEqual(recheck.call_count, 2)

    def test_result_archive_rejects_links_and_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                info = tarfile.TarInfo("run/link")
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                output.addfile(info)
            with self.assertRaisesRegex(ValueError, "link or special"):
                MixedCommandRunner._extract_remote_result_archive(
                    archive, root / "copy", "run"
                )
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "fresh"):
                MixedCommandRunner._extract_remote_result_archive(
                    archive, existing, "run"
                )

    def test_local_aggregate_requires_one_canonical_nested_remote_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            (missing / "status").mkdir(parents=True)
            (missing / "local.bin").write_bytes(b"local")
            with self.assertRaisesRegex(ValueError, "exactly one canonical"):
                verify_and_seal(missing)

            aggregate = root / "aggregate"
            (aggregate / "status").mkdir(parents=True)
            remote = aggregate / "remote-sealed-run"
            (remote / "status").mkdir(parents=True)
            (remote / "artifact.bin").write_bytes(b"remote")
            seal_run(
                remote,
                remote / "provenance" / "final-files.json",
                remote / "status",
            )
            (aggregate / "local.bin").write_bytes(b"local")
            with mock.patch.object(
                runtime,
                "_verify_local_aggregate_remote_copy",
                return_value=({}, "a" * 64, "b" * 64),
            ):
                sealed = verify_and_seal(aggregate)
            self.assertTrue(sealed["passed"])

        with tempfile.TemporaryDirectory() as temporary:
            aggregate = Path(temporary) / "extra"
            (aggregate / "status").mkdir(parents=True)
            for name in ("remote-sealed-run", "misplaced"):
                nested = aggregate / name
                (nested / "status").mkdir(parents=True)
                (nested / "artifact.bin").write_bytes(name.encode())
                seal_run(
                    nested,
                    nested / "provenance" / "final-files.json",
                    nested / "status",
                )
            with self.assertRaisesRegex(ValueError, "exactly one canonical"):
                verify_and_seal(aggregate)


if __name__ == "__main__":
    unittest.main()
