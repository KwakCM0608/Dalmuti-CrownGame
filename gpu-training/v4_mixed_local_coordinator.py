#!/usr/bin/env python3
"""Local coordinator primitives and deterministic dry-run for V4 mixed PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence

sys.dont_write_bytecode = True

from v4_mixed_package_runtime import (
    _load_package,
    _recheck_extracted_source,
    _verify_extracted_source,
    load_canonical_json,
    recheck_snapshot,
    seal_run,
    sha256_bytes,
    snapshot_with_sidecar,
    stable_snapshot,
    verify_package,
    verify_run_seal,
    write_status,
)
from v4_mixed_workflow import (
    BEHAVIOR_ACTOR_SHA256,
    BEHAVIOR_MANIFEST_SHA256,
    CommandSpec,
    FROZEN_BASELINE_COMMIT,
    FROZEN_BASELINE_SHA256,
    PhaseSpec,
    RUN_NAMESPACE,
    build_mixed_phase_plan,
    canonical_json_bytes,
    canonical_sha256,
    load_fixed_collection_plan_sha256,
    load_recipe,
    materialize_argv,
    plan_document,
)


CommandRunner = Callable[[CommandSpec], None]

FROZEN_BASELINE_BUNDLE_NAME = "dalmuti-e0c52b0.bundle"
FROZEN_BASELINE_BUNDLE_SHA256 = (
    "9ea0b9eb4200ac369fbc3ffb1493efe59625b34f5f994359f8a01d4b5610db4d"
)
MERGED_PLAN_SHA_SENTINEL = "__LOAD_FIXED_COLLECTION_PLAN_SHA256__"
REMOTE_ENDPOINT_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.:-]*$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _publish(path: Path, value: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{path}.sha256")
    _require(not path.exists() and not sidecar.exists(), "immutable coordinator output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for target, data in (
        (path, payload),
        (sidecar, f"{digest}  {path.name}\n".encode("ascii")),
    ):
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
    return digest


def _resolve_executable(value: str) -> str:
    candidate = shutil.which(value)
    _require(candidate is not None, f"executable is unavailable: {value}")
    return str(Path(candidate).resolve(strict=True))


def _remote_python_probe_argv(value: str) -> tuple[str, ...]:
    _require(isinstance(value, str) and value, "remote Python command is invalid")
    return (
        value,
        "-c",
        "import os,sys;print(os.path.abspath(sys.executable))",
    )


def _remote_absolute(value: str, label: str) -> str:
    _require(
        isinstance(value, str)
        and value.startswith("/")
        and "\\" not in value
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value,
        f"invalid {label}",
    )
    path = PurePosixPath(value)
    _require(all(part not in ("", ".", "..") for part in path.parts[1:]), f"unsafe {label}")
    return str(path)


def _join_remote(root: str, *parts: str) -> str:
    value = PurePosixPath(_remote_absolute(root, "remote root")).joinpath(*parts)
    return _remote_absolute(str(value), "remote path")


@dataclass(frozen=True)
class SshTransport:
    endpoint: str
    port: int
    identity_file: Path | None
    ssh_executable: str
    scp_executable: str

    def __post_init__(self) -> None:
        _require(
            REMOTE_ENDPOINT_RE.fullmatch(self.endpoint) is not None,
            "remote endpoint must be a literal user@host",
        )
        _require(1 <= self.port <= 65535, "remote SSH port is invalid")
        if self.identity_file is not None:
            path = self.identity_file.resolve(strict=True)
            _require(path.is_file() and not path.is_symlink(), "SSH identity is not a regular file")

    def _identity_arguments(self) -> list[str]:
        if self.identity_file is None:
            return []
        return ["-i", str(self.identity_file.resolve(strict=True))]

    def ssh_arguments(self) -> list[str]:
        return [
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            *self._identity_arguments(),
            self.endpoint,
        ]

    def scp_arguments(self) -> list[str]:
        return [
            "-P",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            *self._identity_arguments(),
        ]

    def run(self, argv: Sequence[str], *, capture: bool = False) -> str:
        _require(argv and all(isinstance(item, str) and item for item in argv), "invalid remote argv")
        effective = tuple(argv)
        if "python" in PurePosixPath(effective[0]).name.lower():
            effective = ("env", "PYTHONDONTWRITEBYTECODE=1", *effective)
        command = shlex.join(effective)
        completed = subprocess.run(
            [self.ssh_executable, *self.ssh_arguments(), command],
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()
            raise RuntimeError(f"remote command failed ({argv[0]}): {detail}")
        return (completed.stdout or "").strip()

    def upload(self, paths: Sequence[Path], remote_directory: str) -> None:
        destination = _remote_absolute(remote_directory, "remote upload directory")
        _require(paths, "upload inventory is empty")
        resolved: list[str] = []
        for path in paths:
            candidate = path.resolve(strict=True)
            _require(candidate.is_file() and not candidate.is_symlink(), f"upload is not a regular file: {path}")
            resolved.append(str(candidate))
        completed = subprocess.run(
            [
                self.scp_executable,
                *self.scp_arguments(),
                "--",
                *resolved,
                f"{self.endpoint}:{destination}/",
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("SCP upload failed")

    def download(self, remote_paths: Sequence[str], local_directory: Path) -> None:
        _require(remote_paths, "download inventory is empty")
        target = local_directory.resolve(strict=True)
        _require(target.is_dir() and not target.is_symlink(), "download destination is invalid")
        for raw in remote_paths:
            remote = _remote_absolute(raw, "remote download path")
            completed = subprocess.run(
                [
                    self.scp_executable,
                    *self.scp_arguments(),
                    "--",
                    f"{self.endpoint}:{remote}",
                    str(target),
                ],
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"SCP download failed: {remote}")


def _sha_record(path: Path) -> dict[str, object]:
    snapshot = stable_snapshot(path, f"command output {path.name}")
    return {
        "kind": "file",
        "path": str(path),
        "sha256": snapshot.sha256,
        "size": len(snapshot.payload),
    }


def verify_output_files(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Verify an exact declared regular-file inventory and sidecar contents."""

    _require(paths, "declared output inventory is empty")
    normalized = [path.resolve(strict=True) for path in paths]
    _require(len(normalized) == len(set(normalized)), "declared output inventory is duplicated")
    records: list[dict[str, object]] = []
    snapshots = []
    for path in normalized:
        snapshot = stable_snapshot(path, f"command output {path.name}")
        snapshots.append(snapshot)
        if path.name.endswith(".sha256"):
            payload_path = Path(str(path)[: -len(".sha256")])
            payload_snapshot = stable_snapshot(payload_path, f"sidecar payload {payload_path.name}")
            _require(
                snapshot.payload
                == f"{payload_snapshot.sha256}  {payload_path.name}\n".encode("ascii"),
                f"output sidecar is stale or malformed: {path.name}",
            )
        records.append(
            {
                "kind": "file",
                "path": str(path),
                "sha256": snapshot.sha256,
                "size": len(snapshot.payload),
            }
        )
    for snapshot in snapshots:
        recheck_snapshot(snapshot, f"command output {snapshot.path.name}")
    return records


def materialize_phase_plan(
    phases: Sequence[PhaseSpec], replacements: Mapping[str, str]
) -> tuple[PhaseSpec, ...]:
    materialized: list[PhaseSpec] = []
    for phase in phases:
        commands: list[CommandSpec] = []
        for command in phase.commands:
            commands.append(
                CommandSpec(
                    command.command_id,
                    command.host,
                    materialize_argv(command.argv, replacements),
                    materialize_argv(command.outputs, replacements)
                    if command.outputs
                    else (),
                )
            )
        materialized.append(
            PhaseSpec(
                phase.phase_id,
                phase.dependencies,
                tuple(commands),
                phase.concurrency_group,
            )
        )
    return tuple(materialized)


def verify_frozen_baseline(
    bundle: Path, repository: Path
) -> Mapping[str, object]:
    bundle_snapshot, bundle_sidecar = snapshot_with_sidecar(
        bundle, FROZEN_BASELINE_BUNDLE_SHA256
    )
    _require(bundle.name == FROZEN_BASELINE_BUNDLE_NAME, "baseline bundle name drifted")
    root = repository.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "frozen baseline repository is invalid")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    _require(head == FROZEN_BASELINE_COMMIT, "frozen baseline commit drifted")
    subprocess.run(
        ["git", "-C", str(root), "bundle", "verify", str(bundle.resolve(strict=True))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    normal = stable_snapshot(root / "lib" / "bot-strategy.ts", "frozen Normal source")
    _require(normal.sha256 == FROZEN_BASELINE_SHA256, "frozen Normal source drifted")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    _require(status == "", "frozen baseline worktree is dirty")
    for snapshot, label in (
        (bundle_snapshot, "baseline bundle"),
        (bundle_sidecar, "baseline bundle sidecar"),
        (normal, "frozen Normal source"),
    ):
        recheck_snapshot(snapshot, label)
    return {
        "bundleSha256": bundle_snapshot.sha256,
        "commit": head,
        "format": "dalmuti-v4-frozen-baseline-verification",
        "normalSha256": normal.sha256,
        "passed": True,
        "version": 1,
    }


@dataclass
class ExecutionContext:
    source_root: Path
    package_directory: Path
    package_manifest_sha256: str
    local_run_directory: Path
    remote_run_directory: str
    behavior_actor_bundle: Path
    frozen_baseline_bundle: Path
    local_python: str
    remote_python: str
    transport: SshTransport
    recipe: Mapping[str, Any]
    recipe_phases: tuple[PhaseSpec, ...]
    phases: tuple[PhaseSpec, ...]
    runtime_bindings: Mapping[str, object]
    runtime_bindings_sha256: str
    resolved_plan_sha256: str | None = None
    source_snapshots: Mapping[str, object] | None = None
    source_root_identity: object | None = None
    source_directory_identities: Mapping[str, object] | None = None

    @property
    def remote_source_root(self) -> str:
        return _join_remote(self.remote_run_directory, "source")

    @property
    def remote_package_directory(self) -> str:
        return _join_remote(self.remote_run_directory, "package")

    @property
    def remote_actor_bundle(self) -> str:
        return _join_remote(self.remote_run_directory, "behavior-actor")

    @property
    def remote_baseline_repository(self) -> str:
        return _join_remote(self.remote_run_directory, "frozen-baseline")

    @property
    def remote_runtime_bindings(self) -> str:
        return _join_remote(
            self.remote_run_directory, "control", "runtime-bindings.json"
        )


def _write_sidecar_only(path: Path, digest: str, payload_leaf: str) -> None:
    _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "invalid sidecar digest")
    _require(not path.exists() and not path.is_symlink(), "immutable sidecar exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(f"{digest}  {payload_leaf}\n".encode("ascii"))
        output.flush()
        os.fsync(output.fileno())


def _actor_stage_files(actor_bundle: Path) -> tuple[Path, ...]:
    root = actor_bundle.resolve(strict=True)
    expected = (root / "actor.pt", root / "manifest.json", root / "manifest.json.sha256")
    _require(
        {path.name for path in root.iterdir()} == {path.name for path in expected},
        "behavior Actor bundle inventory drifted",
    )
    verify_output_files(expected)
    return expected


def _remote_payload_files(package_directory: Path) -> tuple[Path, ...]:
    manifest_path = package_directory / "package-manifest.json"
    manifest = load_canonical_json(manifest_path, "package manifest")
    files = manifest.get("files")
    _require(isinstance(files, list), "package payload inventory is missing")
    names = ["package-manifest.json", "package-manifest.json.sha256"]
    for record in files:
        _require(isinstance(record, Mapping), "package payload record is invalid")
        if record.get("remotePayload") is True:
            name = record.get("name")
            _require(isinstance(name, str), "package payload name is invalid")
            names.extend((name, f"{name}.sha256"))
    result = tuple(package_directory / name for name in names)
    verify_output_files(result)
    return result


def _materialized_replacements(
    *,
    source_root: Path,
    local_run: Path,
    actor_bundle: Path,
    local_python: str,
    remote_run: str,
    remote_python: str,
    package_manifest_sha256: str,
) -> dict[str, str]:
    remote_source = _join_remote(remote_run, "source")
    return {
        "{local_source_root}": source_root.resolve(strict=True).as_posix(),
        "{remote_source_root}": remote_source,
        "{local_run_directory}": local_run.resolve(strict=True).as_posix(),
        "{remote_run_directory}": remote_run,
        "{local_behavior_actor_bundle}": actor_bundle.resolve(strict=True).as_posix(),
        "{remote_behavior_actor_bundle}": _join_remote(remote_run, "behavior-actor"),
        "{local_python}": local_python,
        "{remote_python}": remote_python,
        "{remote_frozen_baseline_repository}": _join_remote(
            remote_run, "frozen-baseline"
        ),
        "{remote_package_directory}": _join_remote(remote_run, "package"),
        "{package_manifest_sha256}": package_manifest_sha256,
        "{merged_collection_plan_sha256}": MERGED_PLAN_SHA_SENTINEL,
    }


def _runtime_binding_value(
    *,
    recipe: Mapping[str, Any],
    package_manifest_sha256: str,
    remote_source_root: str,
    remote_run_directory: str,
    remote_actor_bundle: str,
    remote_baseline_repository: str,
    remote_python: str,
    remote_package_directory: str,
) -> dict[str, object]:
    return {
        "behaviorActorBundle": remote_actor_bundle,
        "format": "dalmuti-v4-mixed-remote-runtime-bindings",
        "frozenBaselineRepository": remote_baseline_repository,
        "packageDirectory": remote_package_directory,
        "packageManifestSha256": package_manifest_sha256,
        "pythonExecutable": remote_python,
        "recipeSha256": canonical_sha256(recipe),
        "runDirectory": remote_run_directory,
        "runNamespace": RUN_NAMESPACE,
        "sourceRoot": remote_source_root,
        "version": 1,
    }


def validate_run_layout(
    recipe: Mapping[str, Any], local_run: Path, remote_run: str
) -> str:
    contract = recipe.get("runContract")
    _require(isinstance(contract, Mapping), "recipe run contract is missing")
    artifact_layout = contract.get("artifactLayout")
    _require(isinstance(artifact_layout, Mapping), "recipe artifact layout is missing")
    _require(
        artifact_layout.get("localRunDirectory") == local_run.name,
        "local run directory name drifted from recipe",
    )
    checked_remote = _remote_absolute(remote_run, "remote run directory")
    _require(
        artifact_layout.get("remoteRunDirectory") == checked_remote,
        "remote run directory drifted from recipe",
    )
    return checked_remote


def _prepare_remote_stage(
    *,
    transport: SshTransport,
    source_root: Path,
    package_directory: Path,
    package_manifest_sha256: str,
    local_run: Path,
    remote_run: str,
    actor_bundle: Path,
    baseline_bundle: Path,
    remote_python: str,
    recipe: Mapping[str, Any],
) -> tuple[Mapping[str, object], str]:
    remote_run = _remote_absolute(remote_run, "remote run directory")
    remote_package = _join_remote(remote_run, "package")
    remote_source = _join_remote(remote_run, "source")
    remote_actor = _join_remote(remote_run, "behavior-actor")
    remote_bundle_root = _join_remote(remote_run, "baseline-bundle")
    remote_baseline = _join_remote(remote_run, "frozen-baseline")
    remote_control = _join_remote(remote_run, "control")
    remote_completions = _join_remote(remote_control, "completions")
    remote_status = _join_remote(remote_run, "status")
    remote_logs = _join_remote(remote_run, "logs")
    transport.run(["mkdir", "-m", "700", "--", remote_run])
    transport.run(
        [
            "mkdir",
            "-m",
            "700",
            "--",
            remote_package,
            remote_actor,
            remote_bundle_root,
            remote_control,
            remote_completions,
            remote_status,
            remote_logs,
        ]
    )
    package_files = _remote_payload_files(package_directory)
    transport.upload(package_files, remote_package)
    manifest = load_canonical_json(
        package_directory / "package-manifest.json", "package manifest"
    )
    verifier_records = [
        record
        for record in manifest["files"]
        if isinstance(record, Mapping) and record.get("role") == "verifier"
    ]
    _require(len(verifier_records) == 1, "package verifier record is not unique")
    remote_verifier = _join_remote(
        remote_package, str(verifier_records[0]["name"])
    )
    transport.run(
        [
            remote_python,
            remote_verifier,
            "verify-package",
            "--package-dir",
            remote_package,
            "--expected-manifest-sha256",
            package_manifest_sha256,
            "--remote-only",
        ]
    )
    transport.run(
        [
            remote_python,
            remote_verifier,
            "extract-source",
            "--package-dir",
            remote_package,
            "--expected-manifest-sha256",
            package_manifest_sha256,
            "--destination",
            remote_source,
        ]
    )
    actor_files = _actor_stage_files(actor_bundle)
    transport.upload(actor_files, remote_actor)
    remote_coordinator = _join_remote(
        remote_source, "gpu-training", "v4_mixed_local_coordinator.py"
    )
    transport.run(
        [
            remote_python,
            remote_coordinator,
            "verify-actor",
            "--actor-bundle",
            remote_actor,
            "--expected-actor-sha256",
            BEHAVIOR_ACTOR_SHA256,
            "--expected-manifest-sha256",
            BEHAVIOR_MANIFEST_SHA256,
        ]
    )
    baseline_snapshot, baseline_sidecar = snapshot_with_sidecar(
        baseline_bundle, FROZEN_BASELINE_BUNDLE_SHA256
    )
    _require(
        baseline_bundle.name == FROZEN_BASELINE_BUNDLE_NAME,
        "frozen baseline bundle filename drifted",
    )
    heads = subprocess.run(
        ["git", "bundle", "list-heads", str(baseline_bundle.resolve(strict=True))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip().splitlines()
    _require(
        heads == [f"{FROZEN_BASELINE_COMMIT} HEAD"],
        "frozen baseline bundle HEAD drifted",
    )
    transport.upload((baseline_bundle, Path(f"{baseline_bundle}.sha256")), remote_bundle_root)
    remote_bundle = _join_remote(remote_bundle_root, FROZEN_BASELINE_BUNDLE_NAME)
    transport.run(["git", "clone", "--no-checkout", "--", remote_bundle, remote_baseline])
    transport.run(
        ["git", "-C", remote_baseline, "config", "core.autocrlf", "false"]
    )
    transport.run(["git", "-C", remote_baseline, "config", "core.eol", "lf"])
    transport.run(
        [
            "git",
            "-C",
            remote_baseline,
            "checkout",
            "--detach",
            FROZEN_BASELINE_COMMIT,
        ]
    )
    transport.run(
        [
            remote_python,
            remote_coordinator,
            "verify-baseline",
            "--bundle",
            remote_bundle,
            "--repository",
            remote_baseline,
        ]
    )
    for snapshot, label in (
        (baseline_snapshot, "frozen baseline bundle"),
        (baseline_sidecar, "frozen baseline bundle sidecar"),
    ):
        recheck_snapshot(snapshot, label)
    bindings = _runtime_binding_value(
        recipe=recipe,
        package_manifest_sha256=package_manifest_sha256,
        remote_source_root=remote_source,
        remote_run_directory=remote_run,
        remote_actor_bundle=remote_actor,
        remote_baseline_repository=remote_baseline,
        remote_python=remote_python,
        remote_package_directory=remote_package,
    )
    local_bindings = local_run / "control" / "runtime-bindings.json"
    bindings_sha = _publish(local_bindings, bindings)
    transport.upload((local_bindings, Path(f"{local_bindings}.sha256")), remote_control)
    return bindings, bindings_sha


def inventory_paths(paths: Sequence[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in paths:
        path = raw.resolve(strict=True)
        _require(not path.is_symlink(), f"output path is a symlink: {path}")
        if path.is_file():
            records.append(_sha_record(path))
            continue
        _require(path.is_dir(), f"output path is neither file nor directory: {path}")
        children: list[dict[str, object]] = []
        total = 0
        for child in sorted(
            path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()
        ):
            _require(not child.is_symlink(), f"output directory contains a symlink: {child}")
            if child.is_dir():
                continue
            snapshot = stable_snapshot(child, f"directory output {child.name}")
            children.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "sha256": snapshot.sha256,
                    "size": len(snapshot.payload),
                }
            )
            total += len(snapshot.payload)
        records.append(
            {
                "kind": "directory",
                "path": str(path),
                "sha256": sha256_bytes(canonical_json_bytes(children)),
                "size": total,
            }
        )
    return records


def _family_paths(kind: str, roots: Sequence[Path]) -> tuple[Path, ...]:
    if kind == "report":
        _require(len(roots) == 1, "report family requires one root")
        return (roots[0], Path(f"{roots[0]}.sha256"))
    if kind in {"npz", "merged"}:
        _require(len(roots) == 1, "NPZ family requires one root")
    elif kind == "calibration-triple":
        _require(len(roots) == 3, "calibration triple requires report, CPU, CUDA")
        report, cpu, cuda = roots
        return (
            report,
            Path(f"{report}.sha256"),
            cpu,
            Path(f"{cpu}.sha256"),
            Path(f"{cpu}.metadata.json"),
            Path(f"{cpu}.metadata.json.sha256"),
            cuda,
            Path(f"{cuda}.sha256"),
            Path(f"{cuda}.metadata.json"),
            Path(f"{cuda}.metadata.json.sha256"),
        )
    elif kind == "shards":
        _require(len(roots) == 12, "remote shard family requires twelve NPZ roots")
        return tuple(
            artifact
            for root in roots
            for artifact in (
                root,
                Path(f"{root}.sha256"),
                Path(f"{root}.metadata.json"),
                Path(f"{root}.metadata.json.sha256"),
            )
        )
    else:
        _require(kind in {"npz", "merged"}, f"unsupported artifact family: {kind}")
    root = roots[0]
    return (
        root,
        Path(f"{root}.sha256"),
        Path(f"{root}.metadata.json"),
        Path(f"{root}.metadata.json.sha256"),
    )


def verify_artifact_family(kind: str, roots: Sequence[Path]) -> list[dict[str, object]]:
    paths = _family_paths(kind, roots)
    records = verify_output_files(paths)
    if kind == "calibration-triple":
        parent = paths[0].parent.resolve(strict=True)
        _require(
            all(path.parent.resolve(strict=True) == parent for path in paths),
            "calibration triple is not co-located",
        )
        _require(
            {path.name for path in parent.iterdir()} == {path.name for path in paths},
            "calibration directory is not the exact ten-file inventory",
        )
    return records


def _remote_family_paths(kind: str, roots: Sequence[str]) -> tuple[str, ...]:
    checked = tuple(_remote_absolute(value, "remote artifact") for value in roots)
    if kind == "report":
        _require(len(checked) == 1, "report family requires one remote root")
        return (checked[0], f"{checked[0]}.sha256")
    if kind == "calibration-triple":
        _require(len(checked) == 3, "calibration triple requires three remote roots")
        report, cpu, cuda = checked
        return (
            report,
            f"{report}.sha256",
            cpu,
            f"{cpu}.sha256",
            f"{cpu}.metadata.json",
            f"{cpu}.metadata.json.sha256",
            cuda,
            f"{cuda}.sha256",
            f"{cuda}.metadata.json",
            f"{cuda}.metadata.json.sha256",
        )
    if kind == "shards":
        _require(len(checked) == 12, "remote shard family requires twelve roots")
    else:
        _require(
            kind in {"npz", "merged"} and len(checked) == 1,
            "invalid remote NPZ family",
        )
    return tuple(
        artifact
        for root in checked
        for artifact in (
            root,
            f"{root}.sha256",
            f"{root}.metadata.json",
            f"{root}.metadata.json.sha256",
        )
    )


def _parse_remote_inventory(payload: str) -> list[dict[str, object]]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("remote inventory output is invalid JSON") from error
    _require(
        isinstance(value, Mapping)
        and value.get("format") == "dalmuti-v4-mixed-output-inventory"
        and value.get("version") == 1
        and value.get("passed") is True
        and isinstance(value.get("outputs"), list),
        "remote inventory output is invalid",
    )
    return [dict(item) for item in value["outputs"] if isinstance(item, Mapping)]


def _remote_inventory(
    context: ExecutionContext, paths: Sequence[str]
) -> list[dict[str, object]]:
    payload = context.transport.run(
        [
            context.remote_python,
            _join_remote(
                context.remote_source_root,
                "gpu-training",
                "v4_mixed_local_coordinator.py",
            ),
            "inventory-paths",
            *(
                token
                for path in paths
                for token in ("--path", _remote_absolute(path, "remote output"))
            ),
        ],
        capture=True,
    )
    result = _parse_remote_inventory(payload)
    _require(len(result) == len(paths), "remote output inventory is incomplete")
    return result


def _remote_verify_family(
    context: ExecutionContext, kind: str, roots: Sequence[str]
) -> list[dict[str, object]]:
    payload = context.transport.run(
        [
            context.remote_python,
            _join_remote(
                context.remote_source_root,
                "gpu-training",
                "v4_mixed_local_coordinator.py",
            ),
            "verify-artifacts",
            "--kind",
            kind,
            *(token for root in roots for token in ("--root", root)),
        ],
        capture=True,
    )
    return _parse_remote_inventory(payload)


def _retrieve_file_family(
    context: ExecutionContext,
    *,
    remote_roots: Sequence[str],
    local_roots: Sequence[Path],
    kind: str,
) -> list[dict[str, object]]:
    _require(len(remote_roots) == len(local_roots), "transfer root mapping drifted")
    remote_paths = _remote_family_paths(kind, remote_roots)
    local_paths = _family_paths(kind, tuple(local_roots))
    _require(len(remote_paths) == len(local_paths), "transfer inventory size drifted")
    parents = {path.parent for path in local_paths}
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v4-retrieve-", dir=context.local_run_directory / "control"
    ) as temporary:
        staging = Path(temporary)
        context.transport.download(remote_paths, staging)
        staged_paths = tuple(staging / PurePosixPath(path).name for path in remote_paths)
        staged_roots = tuple(
            staging / PurePosixPath(value).name for value in remote_roots
        )
        verify_artifact_family(kind, staged_roots)
        _require(
            {path.name for path in staging.iterdir()}
            == {path.name for path in staged_paths},
            "downloaded artifact inventory contains an unexpected file",
        )
        for staged, destination in zip(staged_paths, local_paths):
            _require(
                not destination.exists() and not destination.is_symlink(),
                f"immutable transfer destination exists: {destination}",
            )
            os.replace(staged, destination)
    return verify_artifact_family(kind, local_roots)


def _upload_file_family(
    context: ExecutionContext,
    *,
    local_roots: Sequence[Path],
    remote_roots: Sequence[str],
    kind: str,
) -> list[dict[str, object]]:
    _require(len(local_roots) == len(remote_roots), "upload root mapping drifted")
    local_paths = _family_paths(kind, tuple(local_roots))
    remote_paths = _remote_family_paths(kind, remote_roots)
    verify_artifact_family(kind, local_roots)
    remote_parents = sorted({str(PurePosixPath(path).parent) for path in remote_paths})
    context.transport.run(["mkdir", "-p", "-m", "700", "--", *remote_parents])
    _require(len(remote_parents) == 1, "artifact upload must target one directory")
    if kind == "calibration-triple":
        # CUDA's four files are the actual remote calibration output.  Never
        # overwrite their inode/identity with the downloaded local copy.
        cuda_before = _remote_inventory(context, remote_paths[6:])
        context.transport.upload(local_paths[:6], remote_parents[0])
        records = _remote_verify_family(context, kind, remote_roots)
        _require(
            records[6:] == cuda_before,
            "remote CUDA calibration changed while publishing the triple",
        )
        return records
    context.transport.upload(local_paths, remote_parents[0])
    return _remote_inventory(context, remote_paths)


def _store_worker_result(
    context: ExecutionContext, command_id: str, payload: str
) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"worker {command_id} returned invalid JSON") from error
    _require(isinstance(value, Mapping) and value.get("passed") is True, f"worker {command_id} did not pass")
    output = (
        context.local_run_directory
        / "control"
        / "remote-results"
        / f"{command_id}.json"
    )
    _publish(output, value)
    return value


class MixedCommandRunner:
    def __init__(self, context: ExecutionContext) -> None:
        self.context = context
        self.raw_by_id: dict[str, tuple[PhaseSpec, CommandSpec]] = {}
        for phase in context.recipe_phases:
            for command in phase.commands:
                _require(command.command_id not in self.raw_by_id, "duplicate raw command ID")
                self.raw_by_id[command.command_id] = (phase, command)
        self.pending_receipts: list[
            tuple[PhaseSpec, CommandSpec, tuple[str, ...], list[dict[str, object]]]
        ] = []
        self.remote_is_staged = False
        self.receipt_lock = threading.Lock()

    def _recheck_local_source(self) -> None:
        if self.context.source_snapshots is None:
            return
        _recheck_extracted_source(
            self.context.source_root,
            self.context.source_snapshots,
            self.context.source_root_identity,
            self.context.source_directory_identities,
        )

    def _receipt(
        self, command: CommandSpec, outputs: list[dict[str, object]]
    ) -> None:
        from v4_mixed_remote_worker import (
            build_completion_receipt,
            publish_completion_receipt,
        )

        with self.receipt_lock:
            raw_phase, raw_command = self.raw_by_id[command.command_id]
            pending = (raw_phase, raw_command, command.argv, outputs)
            if not self.context.runtime_bindings_sha256:
                self.pending_receipts.append(pending)
                return
            self.pending_receipts.append(pending)
            while self.pending_receipts:
                phase, spec, materialized_argv, inventory = self.pending_receipts.pop(0)
                receipt = build_completion_receipt(
                    phase=phase,
                    command=spec,
                    materialized_argv=materialized_argv,
                    runtime_bindings=self.context.runtime_bindings,
                    runtime_bindings_sha256=self.context.runtime_bindings_sha256,
                    outputs=inventory,
                )
                local_receipt = (
                    self.context.local_run_directory
                    / "control"
                    / "outbound-completions"
                    / f"{spec.command_id}.json"
                )
                publish_completion_receipt(local_receipt, receipt)
                self.context.transport.upload(
                    (local_receipt, Path(f"{local_receipt}.sha256")),
                    _join_remote(
                        self.context.remote_run_directory,
                        "control",
                        "completions",
                    ),
                )

    def _run_local(self, command: CommandSpec) -> list[dict[str, object]]:
        self._recheck_local_source()
        outputs = tuple(Path(value) for value in command.outputs)
        for output in outputs:
            _require(
                not output.exists() and not output.is_symlink(),
                f"local output is not fresh: {output}",
            )
        completed = subprocess.run(
            list(command.argv),
            cwd=self.context.source_root,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"local command {command.command_id} failed: {detail}")
        self._recheck_local_source()
        if completed.stdout and command.command_id != "verify-and-seal-local-copy":
            result_path = (
                self.context.local_run_directory
                / "control"
                / "local-results"
                / f"{command.command_id}.stdout"
            )
            _require(not result_path.exists(), "local command stdout is not fresh")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_bytes(completed.stdout)
        return verify_output_files(outputs) if outputs else []

    def _run_remote(self, command: CommandSpec) -> None:
        if command.command_id == "train-epoch-one-cuda":
            metadata = (
                self.context.local_run_directory
                / "merged"
                / "production.npz.metadata.json"
            )
            plan_sha = load_fixed_collection_plan_sha256(metadata)
            self.context.resolved_plan_sha256 = plan_sha
            resolution = self.context.local_run_directory / "control" / "merged-plan-resolution.json"
            _publish(
                resolution,
                {
                    "fixedCollectionPlanSha256": plan_sha,
                    "format": "dalmuti-v4-mixed-plan-sha-resolution",
                    "passed": True,
                    "version": 1,
                },
            )
            _require(
                MERGED_PLAN_SHA_SENTINEL in command.argv,
                "training command lacks deferred plan binding",
            )
        worker = _join_remote(
            self.context.remote_source_root,
            "gpu-training",
            "v4_mixed_remote_worker.py",
        )
        payload = self.context.transport.run(
            [
                self.context.remote_python,
                worker,
                "run-command",
                "--source-root",
                self.context.remote_source_root,
                "--run-directory",
                self.context.remote_run_directory,
                "--package-directory",
                self.context.remote_package_directory,
                "--package-manifest-sha256",
                self.context.package_manifest_sha256,
                "--runtime-bindings",
                self.context.remote_runtime_bindings,
                "--command-id",
                command.command_id,
            ],
            capture=True,
        )
        _store_worker_result(self.context, command.command_id, payload)

    def _finalize_remote(self, command: CommandSpec) -> list[dict[str, object]]:
        worker = _join_remote(
            self.context.remote_source_root,
            "gpu-training",
            "v4_mixed_remote_worker.py",
        )
        payload = self.context.transport.run(
            [
                self.context.remote_python,
                worker,
                "finalize-run",
                "--source-root",
                self.context.remote_source_root,
                "--run-directory",
                self.context.remote_run_directory,
                "--package-directory",
                self.context.remote_package_directory,
                "--package-manifest-sha256",
                self.context.package_manifest_sha256,
                "--runtime-bindings",
                self.context.remote_runtime_bindings,
            ],
            capture=True,
        )
        result = _store_worker_result(self.context, command.command_id, payload)
        records = _remote_inventory(self.context, command.outputs)
        seal_record = records[0]
        _require(
            result.get("runSealSha256") == seal_record.get("sha256"),
            "remote finalization result disagrees with sealed file",
        )
        return records

    @staticmethod
    def _extract_remote_result_archive(
        archive_path: Path, destination: Path, expected_top: str
    ) -> None:
        _require(
            not destination.exists() and not destination.is_symlink(),
            "remote-copy destination must be fresh",
        )
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.extracting-", dir=destination.parent
            )
        )
        try:
            seen: set[str] = set()
            directory_modes: list[tuple[Path, int]] = []
            with tarfile.open(archive_path, "r:gz") as archive:
                members = archive.getmembers()
                _require(members, "remote result archive is empty")
                for member in members:
                    path = PurePosixPath(member.name)
                    _require(
                        not path.is_absolute()
                        and path.parts
                        and path.parts[0] == expected_top
                        and all(part not in ("", ".", "..") for part in path.parts),
                        "remote result archive contains an unsafe path",
                    )
                    _require(member.name not in seen, "remote result archive has duplicate members")
                    seen.add(member.name)
                    _require(
                        member.isdir() or member.isfile(),
                        "remote result archive contains a link or special file",
                    )
                for member in members:
                    relative = PurePosixPath(member.name)
                    target = temporary.joinpath(*relative.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        directory_modes.append((target, stat.S_IMODE(member.mode)))
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    _require(source is not None, "cannot read remote result member")
                    descriptor = os.open(
                        target,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        stat.S_IMODE(member.mode),
                    )
                    with os.fdopen(descriptor, "wb") as output:
                        shutil.copyfileobj(source, output)
                        output.flush()
                        os.fsync(output.fileno())
                    os.chmod(target, stat.S_IMODE(member.mode))
                for directory, mode in sorted(
                    directory_modes, key=lambda item: len(item[0].parts), reverse=True
                ):
                    os.chmod(directory, mode)
            extracted = temporary / expected_top
            _require(extracted.is_dir(), "remote result archive lacks its run root")
            os.replace(extracted, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def _retrieve_remote_results(
        self, command: CommandSpec
    ) -> list[dict[str, object]]:
        remote_run = _remote_absolute(command.argv[1], "remote sealed run")
        destination = Path(command.argv[2])
        _require(
            not destination.exists() and not destination.is_symlink(),
            "remote-copy destination must be fresh",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        remote_parent = str(PurePosixPath(remote_run).parent)
        remote_leaf = PurePosixPath(remote_run).name
        remote_archive = f"{remote_run}.sealed-results.tar.gz"
        self.context.transport.run(["test", "!", "-e", remote_archive])
        self.context.transport.run(
            [
                "tar",
                "--format=ustar",
                "-C",
                remote_parent,
                "-czf",
                remote_archive,
                remote_leaf,
            ]
        )
        digest_output = self.context.transport.run(
            ["sha256sum", "--", remote_archive], capture=True
        )
        parts = digest_output.split()
        _require(
            len(parts) >= 2
            and re.fullmatch(r"[0-9a-f]{64}", parts[0]) is not None,
            "remote result archive digest output is invalid",
        )
        with tempfile.TemporaryDirectory(
            prefix=".v4-remote-result-", dir=self.context.local_run_directory / "control"
        ) as temporary:
            staging = Path(temporary)
            self.context.transport.download((remote_archive,), staging)
            local_archive = staging / PurePosixPath(remote_archive).name
            snapshot = stable_snapshot(local_archive, "remote sealed result archive")
            _require(snapshot.sha256 == parts[0], "remote result archive digest drifted")
            self._extract_remote_result_archive(local_archive, destination, remote_leaf)
            recheck_snapshot(snapshot, "remote sealed result archive")
        verify_run_seal(destination / "provenance" / "final-files.json")
        return inventory_paths(tuple(Path(value) for value in command.outputs))

    def _stage(self, command: CommandSpec) -> list[dict[str, object]]:
        bindings, bindings_sha = _prepare_remote_stage(
            transport=self.context.transport,
            source_root=self.context.source_root,
            package_directory=self.context.package_directory,
            package_manifest_sha256=self.context.package_manifest_sha256,
            local_run=self.context.local_run_directory,
            remote_run=self.context.remote_run_directory,
            actor_bundle=self.context.behavior_actor_bundle,
            baseline_bundle=self.context.frozen_baseline_bundle,
            remote_python=self.context.remote_python,
            recipe=self.context.recipe,
        )
        self.context.runtime_bindings = bindings
        self.context.runtime_bindings_sha256 = bindings_sha
        self.remote_is_staged = True
        return _remote_inventory(self.context, command.outputs)

    def _run_transfer(self, command: CommandSpec) -> list[dict[str, object]]:
        argv = command.argv
        if command.command_id == "stage-remote-source-and-actor":
            return self._stage(command)
        if command.command_id == "retrieve-calibration-cuda":
            return _retrieve_file_family(
                self.context,
                remote_roots=(argv[1],),
                local_roots=(Path(argv[2]),),
                kind="npz",
            )
        if command.command_id == "upload-calibration-triple":
            return _upload_file_family(
                self.context,
                local_roots=(Path(argv[1]), Path(argv[2]), Path(argv[3])),
                remote_roots=(argv[4], argv[5], argv[6]),
                kind="calibration-triple",
            )
        if command.command_id == "retrieve-remote-production-shards":
            remote_roots = tuple(
                _join_remote(argv[2], f"shard-{index:02d}.npz")
                for index in range(2, 14)
            )
            local_roots = tuple(
                Path(argv[3]) / f"shard-{index:02d}.npz"
                for index in range(2, 14)
            )
            return _retrieve_file_family(
                self.context,
                remote_roots=remote_roots,
                local_roots=local_roots,
                kind="shards",
            )
        if command.command_id == "upload-merged-production":
            merged_records = _upload_file_family(
                self.context,
                local_roots=(Path(argv[1]),),
                remote_roots=(argv[2],),
                kind="merged",
            )
            shard_records: list[dict[str, object]] = []
            for index in (0, 1):
                shard_records.extend(
                    _upload_file_family(
                        self.context,
                        local_roots=(Path(argv[3]) / f"shard-{index:02d}.npz",),
                        remote_roots=(
                            _join_remote(argv[4], f"shard-{index:02d}.npz"),
                        ),
                        kind="npz",
                    )
                )
            records = [*merged_records, *shard_records]
            _require(
                [record["path"] for record in records] == list(command.outputs),
                "merged upload output ordering drifted",
            )
            return records
        if command.command_id == "retrieve-checksummed-results":
            return self._retrieve_remote_results(command)
        raise ValueError(f"unsupported coordinator transfer: {command.command_id}")

    def __call__(self, command: CommandSpec) -> None:
        if command.host == "remote":
            self._run_remote(command)
            return
        if command.host in {"coordinator-transfer", "local-transfer"}:
            outputs = self._run_transfer(command)
        elif command.host == "local":
            outputs = self._run_local(command)
        elif command.host == "coordinator-finalize":
            outputs = self._finalize_remote(command)
            return
        else:
            raise ValueError(f"unsupported workflow host: {command.host}")
        if command.command_id in {
            "retrieve-checksummed-results",
            "verify-and-seal-local-copy",
        }:
            return
        self._receipt(command, outputs)


def _run_phase_sequentially(phase: PhaseSpec, runner: CommandRunner) -> None:
    """A phase is a strict command barrier: never overlap its own commands."""

    for command in phase.commands:
        runner(command)


def execute_phase_dag(
    phases: Sequence[PhaseSpec], runner: CommandRunner, *, max_parallel_phases: int = 8
) -> tuple[str, ...]:
    """Execute only explicitly co-grouped phases in parallel.

    Dependencies are completion barriers. Commands inside one phase always run
    in declared order. A non-null concurrency group is the sole permission for
    phase overlap; an ungrouped phase executes alone.
    """

    _require(max_parallel_phases >= 1, "max_parallel_phases must be positive")
    by_id = {phase.phase_id: phase for phase in phases}
    _require(len(by_id) == len(phases), "duplicate workflow phase")
    for phase in phases:
        _require(
            all(dependency in by_id for dependency in phase.dependencies),
            f"unknown dependency for phase {phase.phase_id}",
        )
    pending = {phase.phase_id for phase in phases}
    completed: list[str] = []
    completed_set: set[str] = set()
    running: dict[Future[None], PhaseSpec] = {}
    active_group: str | None = None
    order = {phase.phase_id: index for index, phase in enumerate(phases)}

    def ready() -> list[PhaseSpec]:
        return sorted(
            (
                phase
                for phase in phases
                if phase.phase_id in pending
                and set(phase.dependencies).issubset(completed_set)
            ),
            key=lambda phase: order[phase.phase_id],
        )

    with ThreadPoolExecutor(max_workers=max_parallel_phases) as pool:
        while pending or running:
            candidates = ready()
            if not running and candidates:
                first = candidates[0]
                active_group = first.concurrency_group
                allowed = (
                    [first]
                    if active_group is None
                    else [
                        phase
                        for phase in candidates
                        if phase.concurrency_group == active_group
                    ]
                )
                for phase in allowed[:max_parallel_phases]:
                    pending.remove(phase.phase_id)
                    running[pool.submit(_run_phase_sequentially, phase, runner)] = phase
            elif running and active_group is not None:
                for phase in candidates:
                    if (
                        len(running) >= max_parallel_phases
                        or phase.concurrency_group != active_group
                    ):
                        continue
                    pending.remove(phase.phase_id)
                    running[pool.submit(_run_phase_sequentially, phase, runner)] = phase
            if not running:
                raise ValueError("workflow DAG is cyclic or cannot make progress")
            finished, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in finished:
                phase = running.pop(future)
                try:
                    future.result()
                except BaseException:
                    for other in running:
                        other.cancel()
                    if running:
                        wait(tuple(running))
                    raise
                completed.append(phase.phase_id)
                completed_set.add(phase.phase_id)
            if not running:
                active_group = None
    _require(completed_set == set(by_id), "workflow did not complete every phase")
    return tuple(completed)


def verify_actor(
    actor_bundle: Path, expected_actor_sha256: str, expected_manifest_sha256: str
) -> Mapping[str, object]:
    from v4_export import sha256_file, verify_v4_actor_bundle

    manifest = verify_v4_actor_bundle(actor_bundle)
    files = manifest.get("files")
    actor = files.get("actor.pt") if isinstance(files, Mapping) else None
    _require(isinstance(actor, Mapping), "Actor bundle lacks actor.pt")
    _require(
        actor.get("sha256") == expected_actor_sha256,
        "Actor bundle checkpoint SHA-256 drifted",
    )
    _require(
        sha256_file(actor_bundle / "manifest.json") == expected_manifest_sha256,
        "Actor bundle manifest SHA-256 drifted",
    )
    return {
        "actorSha256": expected_actor_sha256,
        "format": "dalmuti-v4-mixed-actor-verification",
        "manifestSha256": expected_manifest_sha256,
        "passed": True,
        "version": 1,
    }


def verify_and_seal(run_directory: Path) -> Mapping[str, object]:
    root = run_directory.resolve(strict=True)
    status = root / "status"
    _require(status.is_dir() and not status.is_symlink(), "local status directory is missing")
    expected_remote_seal = (
        root / "remote-sealed-run" / "provenance" / "final-files.json"
    )
    nested_seals = sorted(
        path
        for path in root.rglob("provenance/final-files.json")
        if path != root / "provenance" / "final-files.json"
    )
    _require(
        nested_seals == [expected_remote_seal]
        and Path(f"{expected_remote_seal}.sha256").is_file(),
        "local aggregate requires exactly one canonical remote-sealed-run seal",
    )
    seal_path = root / "provenance" / "final-files.json"
    value = seal_run(root, seal_path, status, profile="local-aggregate")
    _require(verify_run_seal(seal_path) == value["sealSha256"], "local seal recheck failed")
    return value


def _execute_mixed_workflow_inner(
    *,
    source_root: Path,
    package_directory: Path,
    package_manifest_sha256: str,
    local_run_directory: Path,
    remote_endpoint: str,
    remote_run_directory: str,
    behavior_actor_bundle: Path,
    frozen_baseline_bundle: Path,
    port: int,
    identity_file: Path | None,
    local_python: str,
    remote_python: str,
    ssh_executable: str,
    scp_executable: str,
) -> Mapping[str, object]:
    package = package_directory.resolve(strict=True)
    local_run = local_run_directory.resolve(strict=True)
    source = source_root.resolve(strict=True)
    actor = behavior_actor_bundle.resolve(strict=True)
    baseline = frozen_baseline_bundle.resolve(strict=True)
    _require(source == local_run / "source", "local source root is not canonical")
    _require(
        (local_run / "status").is_dir()
        and not (local_run / "status").is_symlink()
        and (local_run / "logs").is_dir()
        and not (local_run / "logs").is_symlink(),
        "controller did not create the canonical fresh local run",
    )
    _require(
        not (local_run / "control").exists()
        and not (local_run / "provenance").exists(),
        "local coordinator outputs are not fresh",
    )
    (local_run / "control").mkdir(mode=0o700)
    (
        package_manifest,
        source_binding,
        _,
        _,
        _,
    ) = _load_package(package, package_manifest_sha256, remote_only=False)
    _require(
        package_manifest.get("packageId") == RUN_NAMESPACE,
        "package namespace drifted",
    )
    (
        source_snapshots,
        source_root_identity,
        source_directory_identities,
    ) = _verify_extracted_source(source, source_binding)
    recipe_path = source / "gpu-training" / "v4_mixed_execution_recipe.json"
    recipe = load_recipe(recipe_path)
    remote_run = validate_run_layout(recipe, local_run, remote_run_directory)
    local_python_resolved = _resolve_executable(local_python)
    _require(
        baseline.name == FROZEN_BASELINE_BUNDLE_NAME
        and baseline.is_file()
        and not baseline.is_symlink(),
        "frozen baseline input is not the exact bundle",
    )
    baseline_snapshot, baseline_sidecar = snapshot_with_sidecar(
        baseline, FROZEN_BASELINE_BUNDLE_SHA256
    )
    actor_manifest = verify_actor(
        actor, BEHAVIOR_ACTOR_SHA256, BEHAVIOR_MANIFEST_SHA256
    )
    _require(actor_manifest.get("passed") is True, "behavior Actor verification failed")
    transport = SshTransport(
        endpoint=remote_endpoint,
        port=port,
        identity_file=identity_file,
        ssh_executable=_resolve_executable(ssh_executable),
        scp_executable=_resolve_executable(scp_executable),
    )
    remote_python_resolved = transport.run(
        _remote_python_probe_argv(remote_python),
        capture=True,
    )
    _remote_absolute(remote_python_resolved, "remote Python executable")
    raw_phases = build_mixed_phase_plan(recipe)
    replacements = _materialized_replacements(
        source_root=source,
        local_run=local_run,
        actor_bundle=actor,
        local_python=local_python_resolved,
        remote_run=remote_run,
        remote_python=remote_python_resolved,
        package_manifest_sha256=package_manifest_sha256,
    )
    phases = materialize_phase_plan(raw_phases, replacements)
    _publish(
        local_run / "control" / "materialized-plan.json",
        plan_document(phases, recipe),
    )
    context = ExecutionContext(
        source_root=source,
        package_directory=package,
        package_manifest_sha256=package_manifest_sha256,
        local_run_directory=local_run,
        remote_run_directory=remote_run,
        behavior_actor_bundle=actor,
        frozen_baseline_bundle=baseline,
        local_python=local_python_resolved,
        remote_python=remote_python_resolved,
        transport=transport,
        recipe=recipe,
        recipe_phases=raw_phases,
        phases=phases,
        runtime_bindings={},
        runtime_bindings_sha256="",
        source_snapshots=source_snapshots,
        source_root_identity=source_root_identity,
        source_directory_identities=source_directory_identities,
    )
    runner = MixedCommandRunner(context)
    try:
        completed = execute_phase_dag(phases, runner, max_parallel_phases=8)
    except BaseException as error:
        failure = local_run / "status" / "998-failed.json"
        if not failure.exists() and not Path(f"{failure}.sha256").exists():
            try:
                write_status(
                    failure,
                    "workflow",
                    "failed",
                    f"mixed coordinator failed: {type(error).__name__}: {error}"[:4096],
                    None,
                )
            except BaseException:
                pass
        raise
    _require(
        context.resolved_plan_sha256 is not None,
        "training never resolved the merged fixed collection plan",
    )
    local_seal = local_run / "provenance" / "final-files.json"
    seal_sha = verify_run_seal(local_seal)
    recheck_snapshot(baseline_snapshot, "frozen baseline bundle")
    recheck_snapshot(baseline_sidecar, "frozen baseline bundle sidecar")
    _recheck_extracted_source(
        source,
        source_snapshots,
        source_root_identity,
        source_directory_identities,
    )
    success = local_run / "status" / "999-succeeded.json"
    write_status(
        success,
        "complete",
        "succeeded",
        "local CPU and remote CPU/GPU workflow completed and both sealed copies verified",
        local_seal,
    )
    return {
        "completedPhases": list(completed),
        "fixedCollectionPlanSha256": context.resolved_plan_sha256,
        "format": "dalmuti-v4-mixed-coordinator-result",
        "localRunDirectory": str(local_run),
        "localRunSealSha256": seal_sha,
        "passed": True,
        "remoteRunDirectory": remote_run,
        "runtimeBindingsSha256": context.runtime_bindings_sha256,
        "version": 1,
    }


def execute_mixed_workflow(**arguments: object) -> Mapping[str, object]:
    """Run the coordinator and leave immutable failure evidence on every abort."""

    local_value = arguments.get("local_run_directory")
    _require(isinstance(local_value, Path), "local run directory argument is invalid")
    local_run = local_value.resolve(strict=True)
    status = local_run / "status"
    try:
        return _execute_mixed_workflow_inner(**arguments)  # type: ignore[arg-type]
    except BaseException as error:
        if status.is_dir() and not status.is_symlink():
            failure = status / "998-failed.json"
            if not failure.exists() and not Path(f"{failure}.sha256").exists():
                try:
                    write_status(
                        failure,
                        "workflow",
                        "failed",
                        f"mixed coordinator failed: {type(error).__name__}: {error}"[
                            :4096
                        ],
                        None,
                    )
                except BaseException:
                    pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).with_name("v4_mixed_execution_recipe.json"),
    )
    dry_run.add_argument("--output", type=Path)
    execute = commands.add_parser("execute")
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--package-directory", type=Path, required=True)
    execute.add_argument("--package-manifest-sha256", required=True)
    execute.add_argument("--local-run-directory", type=Path, required=True)
    execute.add_argument("--remote-endpoint", required=True)
    execute.add_argument("--remote-run-directory", required=True)
    execute.add_argument("--behavior-actor-bundle", type=Path, required=True)
    execute.add_argument("--frozen-baseline-bundle", type=Path, required=True)
    execute.add_argument("--port", type=int, default=22)
    execute.add_argument("--identity-file", type=Path)
    execute.add_argument("--local-python", default=sys.executable)
    execute.add_argument("--remote-python", default="python3")
    execute.add_argument("--ssh-executable", default="ssh")
    execute.add_argument("--scp-executable", default="scp")
    actor = commands.add_parser("verify-actor")
    actor.add_argument("--actor-bundle", type=Path, required=True)
    actor.add_argument("--expected-actor-sha256", required=True)
    actor.add_argument("--expected-manifest-sha256", required=True)
    plan_sha = commands.add_parser("resolve-plan-sha")
    plan_sha.add_argument("--metadata", type=Path, required=True)
    seal = commands.add_parser("verify-and-seal")
    seal.add_argument("--run-directory", type=Path, required=True)
    baseline = commands.add_parser("verify-baseline")
    baseline.add_argument("--bundle", type=Path, required=True)
    baseline.add_argument("--repository", type=Path, required=True)
    inventory = commands.add_parser("inventory-paths")
    inventory.add_argument("--path", type=Path, action="append", required=True)
    artifact = commands.add_parser("verify-artifacts")
    artifact.add_argument(
        "--kind",
        choices=("report", "npz", "merged", "calibration-triple", "shards"),
        required=True,
    )
    artifact.add_argument("--root", type=Path, action="append", required=True)
    verify_seal = commands.add_parser("verify-run-seal")
    verify_seal.add_argument("--seal", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "dry-run":
        recipe = load_recipe(arguments.recipe)
        value = plan_document(build_mixed_phase_plan(recipe), recipe)
        if arguments.output is not None:
            digest = _publish(arguments.output, value)
            value = {**value, "planFileSha256": digest}
    elif arguments.command == "execute":
        value = execute_mixed_workflow(
            source_root=arguments.source_root,
            package_directory=arguments.package_directory,
            package_manifest_sha256=arguments.package_manifest_sha256,
            local_run_directory=arguments.local_run_directory,
            remote_endpoint=arguments.remote_endpoint,
            remote_run_directory=arguments.remote_run_directory,
            behavior_actor_bundle=arguments.behavior_actor_bundle,
            frozen_baseline_bundle=arguments.frozen_baseline_bundle,
            port=arguments.port,
            identity_file=arguments.identity_file,
            local_python=arguments.local_python,
            remote_python=arguments.remote_python,
            ssh_executable=arguments.ssh_executable,
            scp_executable=arguments.scp_executable,
        )
    elif arguments.command == "verify-actor":
        value = verify_actor(
            arguments.actor_bundle,
            arguments.expected_actor_sha256,
            arguments.expected_manifest_sha256,
        )
    elif arguments.command == "resolve-plan-sha":
        value = {
            "fixedCollectionPlanSha256": load_fixed_collection_plan_sha256(
                arguments.metadata
            ),
            "format": "dalmuti-v4-mixed-plan-sha-resolution",
            "passed": True,
            "version": 1,
        }
    elif arguments.command == "verify-and-seal":
        value = verify_and_seal(arguments.run_directory)
    elif arguments.command == "verify-baseline":
        value = verify_frozen_baseline(arguments.bundle, arguments.repository)
    elif arguments.command == "inventory-paths":
        value = {
            "format": "dalmuti-v4-mixed-output-inventory",
            "outputs": inventory_paths(arguments.path),
            "passed": True,
            "version": 1,
        }
    elif arguments.command == "verify-artifacts":
        value = {
            "format": "dalmuti-v4-mixed-output-inventory",
            "outputs": verify_artifact_family(arguments.kind, arguments.root),
            "passed": True,
            "version": 1,
        }
    else:
        value = {
            "format": "dalmuti-v4-mixed-run-seal-verification",
            "passed": True,
            "sealSha256": verify_run_seal(arguments.seal),
            "version": 1,
        }
    sys.stdout.buffer.write(canonical_json_bytes(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
