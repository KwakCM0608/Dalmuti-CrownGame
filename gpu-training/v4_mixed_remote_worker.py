#!/usr/bin/env python3
"""Recipe-bound remote command worker for the sealed V4 mixed run.

The worker never accepts an argv list from the coordinator.  It loads the
recipe from the verified extracted source, rebuilds the immutable phase DAG,
selects one exact remote CommandSpec by ID, resolves canonical runtime
bindings, proves every prerequisite receipt, and only then executes that one
command without a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from v4_mixed_package_runtime import (
    RemoteSourceVerification,
    _load_package,
    _load_recipe as _load_package_recipe,
    _verify_extracted_source,
    seal_run,
    verify_run_seal,
    verify_screening,
    write_status,
)
from v4_mixed_workflow import (
    CommandSpec,
    FROZEN_BASELINE_COMMIT,
    FROZEN_BASELINE_SHA256,
    OBSERVATION_SCHEMA_SHA256,
    PhaseSpec,
    RUN_NAMESPACE,
    build_mixed_phase_plan,
    canonical_json_bytes,
    canonical_sha256,
    load_fixed_collection_plan_sha256,
    materialize_argv,
    validate_recipe,
)


RUNTIME_BINDING_FORMAT = "dalmuti-v4-mixed-remote-runtime-bindings"
COMPLETION_FORMAT = "dalmuti-v4-mixed-command-completion"
COMMAND_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_COMMAND_TEXT = ("v3-ppo-i2", "deploy", "final-reservation")
FINAL_SCREEN_COMMAND_ID = "verify-complete-remote-screening"
SCREEN_COMMAND_ID = "screen-epoch-one-p4-p10"
FROZEN_BASELINE_BUNDLE_NAME = "dalmuti-e0c52b0.bundle"
FROZEN_BASELINE_BUNDLE_SHA256 = (
    "9ea0b9eb4200ac369fbc3ffb1493efe59625b34f5f994359f8a01d4b5610db4d"
)


def _local_counterparts() -> dict[str, tuple[tuple[str, int], ...]]:
    return {
        "collect-calibration-cpu": tuple(
            ("upload-calibration-triple", index) for index in range(2, 6)
        ),
        "retrieve-calibration-cuda": tuple(
            ("collect-calibration-cuda", index) for index in range(4)
        ),
        "compare-calibration-backends": tuple(
            ("upload-calibration-triple", index) for index in range(2)
        ),
        "collect-production-shard-00": tuple(
            ("upload-merged-production", index) for index in range(4, 8)
        ),
        "collect-production-shard-01": tuple(
            ("upload-merged-production", index) for index in range(8, 12)
        ),
        "retrieve-remote-production-shards": tuple(
            (f"collect-production-shard-{2 + index // 4:02d}", index % 4)
            for index in range(48)
        ),
        "merge-production-shards": tuple(
            ("upload-merged-production", index) for index in range(4)
        ),
    }


LOCAL_OUTPUT_COUNTERPARTS = _local_counterparts()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(stat.S_IMODE(value.st_mode)),
    )


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _ArtifactSnapshot:
    path: Path
    run_directory: Path
    record: Mapping[str, object]
    files: tuple[_Snapshot, ...]
    directories: tuple[tuple[Path, tuple[int, int, int, int, int]], ...]


@dataclass(frozen=True)
class RuntimeBindings:
    raw: Mapping[str, object]
    sha256: str
    source_root: Path
    run_directory: Path
    actor_bundle: Path
    frozen_baseline_repository: Path
    python_executable: Path
    package_directory: Path
    snapshots: tuple[_Snapshot, ...]


@dataclass(frozen=True)
class _PlanIndex:
    phases: tuple[PhaseSpec, ...]
    phase_by_id: Mapping[str, PhaseSpec]
    command_by_id: Mapping[str, tuple[PhaseSpec, int, CommandSpec]]


def _stable_file_snapshot(path: Path, label: str) -> _Snapshot:
    try:
        before_path = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or inaccessible") from error
    _require(
        stat.S_ISREG(before_path.st_mode) and not path.is_symlink(),
        f"{label} is not a regular file",
    )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before_fd = os.fstat(descriptor)
        _require(
            int(before_path.st_dev) == int(before_fd.st_dev)
            and (
                int(before_path.st_ino) == 0
                or int(before_fd.st_ino) == 0
                or int(before_path.st_ino) == int(before_fd.st_ino)
            ),
            f"{label} changed while opening",
        )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    _require(
        _identity(before_path) == _identity(after_path)
        and _identity(before_fd) == _identity(after_fd),
        f"{label} changed while reading",
    )
    _require(
        int(after_path.st_dev) == int(after_fd.st_dev)
        and (
            int(after_path.st_ino) == 0
            or int(after_fd.st_ino) == 0
            or int(after_path.st_ino) == int(after_fd.st_ino)
        ),
        f"{label} was replaced while reading",
    )
    payload = b"".join(chunks)
    return _Snapshot(path, payload, digest.hexdigest(), _identity(after_path))


def _assert_snapshot_unchanged(snapshot: _Snapshot, label: str) -> None:
    current = _stable_file_snapshot(snapshot.path, label)
    _require(
        _identity_matches_allowing_readonly(snapshot.identity, current.identity)
        and current.sha256 == snapshot.sha256
        and current.payload == snapshot.payload,
        f"{label} changed after validation",
    )


def _identity_matches_allowing_readonly(
    expected: tuple[int, int, int, int, int],
    current: tuple[int, int, int, int, int],
) -> bool:
    return (
        current[:4] == expected[:4]
        and ((current[4] & 0o222) & ~(expected[4] & 0o222)) == 0
    )


def _directory_identity(path: Path, label: str) -> tuple[int, int, int, int, int]:
    try:
        value = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or inaccessible") from error
    _require(
        stat.S_ISDIR(value.st_mode) and not path.is_symlink(),
        f"{label} is not a regular directory",
    )
    return _identity(value)


def _capture_artifact(path: Path, run_directory: Path) -> _ArtifactSnapshot:
    resolved = _inside(run_directory, path, "receipt output").resolve(strict=True)
    _require(not resolved.is_symlink(), f"output is a symlink: {resolved}")
    if resolved.is_file():
        snapshot = _stable_file_snapshot(resolved, f"output {resolved.name}")
        return _ArtifactSnapshot(
            path=resolved,
            run_directory=run_directory.resolve(strict=True),
            record={
                "kind": "file",
                "path": str(resolved),
                "sha256": snapshot.sha256,
                "size": len(snapshot.payload),
            },
            files=(snapshot,),
            directories=(),
        )
    _require(resolved.is_dir(), f"output is not a regular file or directory: {resolved}")
    root_identity = _directory_identity(resolved, f"output directory {resolved.name}")
    entries_before = tuple(
        sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix())
    )
    signatures_before: list[tuple[str, str]] = []
    files: list[_Snapshot] = []
    directories: list[tuple[Path, tuple[int, int, int, int, int]]] = [
        (resolved, root_identity)
    ]
    child_records: list[dict[str, object]] = []
    total_size = 0
    for child in entries_before:
        relative = child.relative_to(resolved).as_posix()
        _require(not child.is_symlink(), f"output directory contains a symlink: {relative}")
        if child.is_dir():
            signatures_before.append((relative, "directory"))
            directories.append(
                (child, _directory_identity(child, f"output directory {relative}"))
            )
            continue
        _require(child.is_file(), f"output directory contains a non-regular file: {relative}")
        signatures_before.append((relative, "file"))
        snapshot = _stable_file_snapshot(child, f"output file {relative}")
        files.append(snapshot)
        child_records.append(
            {
                "path": relative,
                "sha256": snapshot.sha256,
                "size": len(snapshot.payload),
            }
        )
        total_size += len(snapshot.payload)
    entries_after = tuple(
        sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix())
    )
    signatures_after = tuple(
        (
            child.relative_to(resolved).as_posix(),
            "directory" if child.is_dir() and not child.is_symlink() else "file",
        )
        for child in entries_after
    )
    _require(
        tuple(signatures_before) == signatures_after,
        f"output directory inventory changed while reading: {resolved}",
    )
    for snapshot in files:
        _assert_snapshot_unchanged(snapshot, f"output file {snapshot.path.name}")
    for directory, identity in directories:
        _require(
            _directory_identity(directory, f"output directory {directory.name}")
            == identity,
            f"output directory changed while reading: {directory}",
        )
    return _ArtifactSnapshot(
        path=resolved,
        run_directory=run_directory.resolve(strict=True),
        record={
            "kind": "directory",
            "path": str(resolved),
            "sha256": canonical_sha256(child_records),
            "size": total_size,
        },
        files=tuple(files),
        directories=tuple(directories),
    )


def _assert_artifact_unchanged(snapshot: _ArtifactSnapshot, label: str) -> None:
    current = _capture_artifact(snapshot.path, snapshot.run_directory)
    _require(current.record == snapshot.record, f"{label} bytes, hash, or size changed")
    _require(
        [path for path, _ in current.directories]
        == [path for path, _ in snapshot.directories]
        and len(current.files) == len(snapshot.files),
        f"{label} inventory changed",
    )
    for expected_file, current_file in zip(snapshot.files, current.files, strict=True):
        _require(expected_file.path == current_file.path, f"{label} file order changed")
        _assert_snapshot_unchanged(expected_file, f"{label} file {expected_file.path.name}")
    for (expected_path, expected_identity), (current_path, current_identity) in zip(
        snapshot.directories, current.directories, strict=True
    ):
        _require(
            expected_path == current_path
            and _identity_matches_allowing_readonly(expected_identity, current_identity),
            f"{label} directory identity changed: {expected_path}",
        )


def _assert_protection_unchanged(
    protection: _Snapshot | _ArtifactSnapshot | RemoteSourceVerification,
    label: str,
) -> None:
    if isinstance(protection, RemoteSourceVerification):
        _assert_remote_source_unchanged(protection)
    elif isinstance(protection, _ArtifactSnapshot):
        _assert_artifact_unchanged(protection, label)
    else:
        _assert_snapshot_unchanged(protection, label)


def _recheck_protections(
    protections: Sequence[
        _Snapshot | _ArtifactSnapshot | RemoteSourceVerification
    ],
) -> None:
    for protection in protections:
        label = (
            "sealed package/source"
            if isinstance(protection, RemoteSourceVerification)
            else protection.path.name
        )
        _assert_protection_unchanged(protection, label)


def _assert_remote_source_unchanged(
    verification: RemoteSourceVerification,
) -> None:
    package_identity = _directory_identity(
        verification.package_root, "remote package directory"
    )
    _require(
        _identity_matches_allowing_readonly(
            verification.package_root_identity, package_identity
        )
        and {path.name for path in verification.package_root.iterdir()}
        == verification.package_expected_names,
        "remote package identity or inventory changed",
    )
    for name, snapshot in verification.package_snapshots.items():
        _require(
            snapshot.path == verification.package_root / name,
            "remote package snapshot path drifted",
        )
        _assert_snapshot_unchanged(snapshot, f"remote package file {name}")

    source_identity = _directory_identity(
        verification.source_root, "extracted source root"
    )
    _require(
        _identity_matches_allowing_readonly(
            verification.source_root_identity, source_identity
        ),
        "extracted source root identity changed",
    )
    observed_files: set[str] = set()
    observed_directories = {"."}
    for path in verification.source_root.rglob("*"):
        relative = path.relative_to(verification.source_root).as_posix()
        _require(
            not path.is_symlink(),
            f"extracted source gained a symlink: {relative}",
        )
        if path.is_dir():
            observed_directories.add(relative)
            expected = verification.source_directory_identities.get(relative)
            current = _directory_identity(path, f"source directory {relative}")
            _require(
                expected is not None
                and _identity_matches_allowing_readonly(expected, current),
                f"extracted source directory changed: {relative}",
            )
            continue
        _require(
            relative in verification.source_snapshots,
            f"extracted source gained an unbound file: {relative}",
        )
        _assert_snapshot_unchanged(
            verification.source_snapshots[relative],
            f"extracted source {relative}",
        )
        observed_files.add(relative)
    _require(
        observed_files == set(verification.source_snapshots),
        "extracted source file inventory changed",
    )
    _require(
        observed_directories == set(verification.source_directory_identities),
        "extracted source directory inventory changed",
    )
    _require(
        _identity_matches_allowing_readonly(
            verification.package_root_identity,
            _directory_identity(
                verification.package_root, "remote package directory"
            ),
        )
        and _identity_matches_allowing_readonly(
            verification.source_root_identity,
            _directory_identity(
                verification.source_root, "extracted source root"
            ),
        ),
        "remote package or extracted source root changed while rechecking",
    )


def _canonical_object(snapshot: _Snapshot, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    _require(
        isinstance(value, Mapping) and snapshot.payload == canonical_json_bytes(value),
        f"{label} is not one canonical JSON object",
    )
    return value


def _canonical_with_sidecar(
    path: Path, label: str, *, expected_sha256: str | None = None
) -> tuple[Mapping[str, object], str, tuple[_Snapshot, _Snapshot]]:
    payload = _stable_file_snapshot(path, label)
    if expected_sha256 is not None:
        _require(payload.sha256 == expected_sha256, f"{label} digest mismatch")
    sidecar_path = Path(f"{path}.sha256")
    sidecar = _stable_file_snapshot(sidecar_path, f"{label} sidecar")
    _require(
        sidecar.payload
        == f"{payload.sha256}  {path.name}\n".encode("ascii"),
        f"{label} sidecar is stale or malformed",
    )
    return _canonical_object(payload, label), payload.sha256, (payload, sidecar)


def _absolute(value: object, label: str, *, strict: bool = True) -> Path:
    _require(isinstance(value, str) and value != "" and "\x00" not in value, f"missing {label}")
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    return path.resolve(strict=strict)


def _python_entrypoint(value: object, label: str) -> Path:
    """Validate, but deliberately do not resolve, the bound venv Python path.

    A POSIX virtual environment commonly exposes ``bin/python`` as a symlink to
    the base interpreter.  Executing the resolved target bypasses the virtual
    environment and can therefore lose its installed packages.  The lexical
    entrypoint is security-bound to this worker's own ``sys.executable`` while
    its resolved target is checked only for regular/executable-file validity.
    """

    _require(
        isinstance(value, str) and value != "" and "\x00" not in value,
        f"missing {label}",
    )
    candidate = Path(value)
    _require(candidate.is_absolute(), f"{label} must be absolute")
    _require(
        value == os.path.abspath(value)
        and os.path.normpath(value) == value
        and ".." not in candidate.parts,
        f"{label} must be a canonical absolute path",
    )
    current = Path(os.path.abspath(sys.executable))
    _require(candidate == current, "runtime Python executable drifted")
    try:
        entrypoint_mode = candidate.lstat().st_mode
        target = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("bound Python executable is invalid") from error
    _require(
        (stat.S_ISREG(entrypoint_mode) or stat.S_ISLNK(entrypoint_mode))
        and target.is_file()
        and not target.is_symlink()
        and os.access(candidate, os.X_OK),
        "bound Python executable is invalid",
    )
    return candidate


def _inside(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    _require(
        lexical == root or root in lexical.parents,
        f"{label} escapes the run directory",
    )
    current = root
    for part in lexical.relative_to(root).parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as error:
            raise ValueError(f"{label} is inaccessible") from error
        _require(not stat.S_ISLNK(mode), f"{label} traverses a symlink")
    resolved = path.resolve(strict=False)
    _require(resolved == root or root in resolved.parents, f"{label} escapes the run directory")
    return resolved


def _safe_command_id(value: object) -> str:
    _require(
        isinstance(value, str) and COMMAND_ID_RE.fullmatch(value) is not None,
        "unsafe command ID",
    )
    return value


def completion_path(run_directory: Path, command_id: str) -> Path:
    command_id = _safe_command_id(command_id)
    root = run_directory.resolve(strict=True)
    candidate = root / "control" / "completions" / f"{command_id}.json"
    _inside(root, candidate, "completion receipt")
    return candidate


def _publish_canonical_with_sidecar(path: Path, value: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(value)
    digest = _sha256_bytes(payload)
    sidecar = Path(f"{path}.sha256")
    _require(
        not path.exists()
        and not path.is_symlink()
        and not sidecar.exists()
        and not sidecar.is_symlink(),
        f"immutable output already exists: {path}",
    )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for target, data in (
        (path, payload),
        (sidecar, f"{digest}  {path.name}\n".encode("ascii")),
    ):
        descriptor = os.open(target, flags, 0o400)
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
        os.chmod(target, 0o444)
    _canonical_with_sidecar(path, "published completion", expected_sha256=digest)
    return digest


def _validate_output_record(record: object) -> Mapping[str, object]:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"kind", "path", "sha256", "size"},
        "completion output record fields drifted",
    )
    _require(record.get("kind") in {"file", "directory"}, "invalid completion output kind")
    _require(isinstance(record.get("path"), str) and record.get("path") != "", "invalid completion output path")
    _require(isinstance(record.get("sha256"), str) and SHA256_RE.fullmatch(str(record["sha256"])) is not None, "invalid completion output digest")
    size = record.get("size")
    _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, "invalid completion output size")
    return record


def build_completion_receipt(
    *,
    phase: PhaseSpec,
    command: CommandSpec,
    materialized_argv: Sequence[str],
    runtime_bindings: Mapping[str, object],
    runtime_bindings_sha256: str,
    outputs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    _safe_command_id(command.command_id)
    _require(command in phase.commands, "command is not in the supplied phase")
    _require(
        isinstance(runtime_bindings_sha256, str)
        and SHA256_RE.fullmatch(runtime_bindings_sha256) is not None,
        "invalid runtime binding digest",
    )
    normalized_outputs = [dict(_validate_output_record(item)) for item in outputs]
    _require(
        len(normalized_outputs) == len(command.outputs),
        "completion output inventory count drifted",
    )
    return {
        "commandId": command.command_id,
        "commandSpecSha256": canonical_sha256(command.to_dict()),
        "format": COMPLETION_FORMAT,
        "host": command.host,
        "materializedArgvSha256": canonical_sha256(list(materialized_argv)),
        "outputs": normalized_outputs,
        "packageManifestSha256": runtime_bindings.get("packageManifestSha256"),
        "passed": True,
        "phaseId": phase.phase_id,
        "recipeSha256": runtime_bindings.get("recipeSha256"),
        "runNamespace": runtime_bindings.get("runNamespace"),
        "runtimeBindingsSha256": runtime_bindings_sha256,
        "version": 1,
    }


def publish_completion_receipt(path: Path, receipt: Mapping[str, object]) -> str:
    _require(
        set(receipt)
        == {
            "commandId",
            "commandSpecSha256",
            "format",
            "host",
            "materializedArgvSha256",
            "outputs",
            "packageManifestSha256",
            "passed",
            "phaseId",
            "recipeSha256",
            "runNamespace",
            "runtimeBindingsSha256",
            "version",
        }
        and receipt.get("format") == COMPLETION_FORMAT
        and receipt.get("version") == 1
        and receipt.get("passed") is True,
        "invalid completion receipt",
    )
    command_id = _safe_command_id(receipt.get("commandId"))
    _require(path.name == f"{command_id}.json", "completion receipt path drifted")
    outputs = receipt.get("outputs")
    _require(isinstance(outputs, list), "completion output inventory is invalid")
    for item in outputs:
        _validate_output_record(item)
    for field in (
        "commandSpecSha256",
        "materializedArgvSha256",
        "packageManifestSha256",
        "recipeSha256",
        "runtimeBindingsSha256",
    ):
        _require(isinstance(receipt.get(field), str) and SHA256_RE.fullmatch(str(receipt[field])) is not None, f"invalid completion {field}")
    _require(receipt.get("runNamespace") == RUN_NAMESPACE, "completion namespace drifted")
    return _publish_canonical_with_sidecar(path, receipt)


def _rollback_completion_receipt(path: Path) -> None:
    failures: list[str] = []
    for target in (Path(f"{path}.sha256"), path):
        try:
            if not target.exists() and not target.is_symlink():
                continue
            if target.is_dir() and not target.is_symlink():
                failures.append(str(target))
                continue
            if not target.is_symlink():
                os.chmod(target, 0o600)
            target.unlink()
        except OSError:
            failures.append(str(target))
    _require(not failures, f"could not roll back completion receipt: {failures}")


def _open_fresh_binary_log(path: Path, run_directory: Path):
    _inside(run_directory, path, "command log")
    _require(
        not path.exists() and not path.is_symlink(),
        f"command log is not fresh: {path}",
    )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    return os.fdopen(descriptor, "wb")


def _failed_command_summary(
    command_id: str,
    returncode: int,
    stdout_snapshot: _Snapshot,
    stderr_snapshot: _Snapshot,
) -> str:
    return (
        f"remote command {command_id} exited {returncode}; "
        f"stdoutBytes={len(stdout_snapshot.payload)}; "
        f"stdoutSha256={stdout_snapshot.sha256}; "
        f"stdoutLog=logs/{stdout_snapshot.path.name}; "
        f"stderrBytes={len(stderr_snapshot.payload)}; "
        f"stderrSha256={stderr_snapshot.sha256}; "
        f"stderrLog=logs/{stderr_snapshot.path.name}"
    )


def inventory_output(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    _require(not path.is_symlink(), f"output is a symlink: {path}")
    if resolved.is_file():
        return {
            "kind": "file",
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
            "size": resolved.stat().st_size,
        }
    _require(resolved.is_dir(), f"output is not a regular file or directory: {path}")
    records: list[dict[str, object]] = []
    total_size = 0
    for child in sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()):
        relative = child.relative_to(resolved).as_posix()
        _require(not child.is_symlink(), f"output directory contains a symlink: {relative}")
        if child.is_dir():
            continue
        _require(child.is_file(), f"output directory contains a non-regular file: {relative}")
        size = child.stat().st_size
        records.append({"path": relative, "sha256": _sha256_file(child), "size": size})
        total_size += size
    return {
        "kind": "directory",
        "path": str(resolved),
        "sha256": canonical_sha256(records),
        "size": total_size,
    }


def _load_sealed_recipe(
    source_root: Path, package_directory: Path, package_manifest_sha256: str
) -> tuple[
    Mapping[str, Any],
    tuple[_Snapshot | _ArtifactSnapshot | RemoteSourceVerification, ...],
]:
    _require(SHA256_RE.fullmatch(package_manifest_sha256) is not None, "invalid package manifest digest")
    (
        manifest,
        binding,
        package_snapshots,
        package_root_identity,
        package_expected_names,
    ) = _load_package(
        package_directory, package_manifest_sha256, remote_only=True
    )
    (
        source_snapshots,
        source_root_identity,
        source_directory_identities,
    ) = _verify_extracted_source(source_root, binding)
    recipe, recipe_snapshot = _load_package_recipe(
        source_root, manifest, binding, source_snapshots
    )
    validate_recipe(recipe)
    source = source_root.resolve(strict=True)
    package = package_directory.resolve(strict=True)
    run = source.parent
    _require(package.parent == run, "sealed package and source roots disagree")
    verification = RemoteSourceVerification(
        package_root=package,
        package_snapshots=package_snapshots,
        package_root_identity=package_root_identity,
        package_expected_names=package_expected_names,
        source_root=source,
        source_snapshots=source_snapshots,
        source_root_identity=source_root_identity,
        source_directory_identities=source_directory_identities,
    )
    _require(
        recipe_snapshot.path in {
            snapshot.path for snapshot in verification.source_snapshots.values()
        },
        "sealed recipe snapshot is missing",
    )
    _assert_remote_source_unchanged(verification)
    return recipe, (verification,)


def load_runtime_bindings(
    path: Path,
    *,
    source_root: Path,
    run_directory: Path,
    package_directory: Path,
    package_manifest_sha256: str,
    recipe_sha256: str,
) -> RuntimeBindings:
    run = run_directory.resolve(strict=True)
    expected_path = run / "control" / "runtime-bindings.json"
    _require(path.resolve(strict=True) == expected_path, "runtime binding path is non-canonical")
    raw, digest, snapshots = _canonical_with_sidecar(path, "runtime bindings")
    expected_fields = {
        "behaviorActorBundle",
        "format",
        "frozenBaselineRepository",
        "packageDirectory",
        "packageManifestSha256",
        "pythonExecutable",
        "recipeSha256",
        "runDirectory",
        "runNamespace",
        "sourceRoot",
        "version",
    }
    _require(
        set(raw) == expected_fields
        and raw.get("format") == RUNTIME_BINDING_FORMAT
        and raw.get("version") == 1,
        "runtime binding fields drifted",
    )
    _require(raw.get("packageManifestSha256") == package_manifest_sha256, "runtime package binding drifted")
    _require(raw.get("recipeSha256") == recipe_sha256, "runtime recipe binding drifted")
    _require(raw.get("runNamespace") == RUN_NAMESPACE, "runtime namespace drifted")
    bound_run = _absolute(raw.get("runDirectory"), "bound run directory")
    _require(bound_run == run, "runtime run directory drifted")
    bound_source = _absolute(raw.get("sourceRoot"), "bound source root")
    bound_package = _absolute(raw.get("packageDirectory"), "bound package directory")
    bound_actor = _absolute(raw.get("behaviorActorBundle"), "bound behavior Actor")
    bound_baseline = _absolute(raw.get("frozenBaselineRepository"), "bound frozen baseline")
    bound_python = _python_entrypoint(
        raw.get("pythonExecutable"), "bound Python executable"
    )
    _require(bound_source == source_root.resolve(strict=True), "runtime source root drifted")
    _require(bound_package == package_directory.resolve(strict=True), "runtime package directory drifted")
    _require(bound_source == run / "source", "runtime source path is non-canonical")
    _require(bound_package == run / "package", "runtime package path is non-canonical")
    for value, label in (
        (bound_source, "source root"),
        (bound_package, "package directory"),
        (bound_actor, "behavior Actor"),
        (bound_baseline, "frozen baseline"),
    ):
        _inside(run, value, label)
    _require(bound_source.is_dir() and not bound_source.is_symlink(), "bound source root is invalid")
    _require(bound_package.is_dir() and not bound_package.is_symlink(), "bound package directory is invalid")
    _require(bound_actor.is_dir() and not bound_actor.is_symlink(), "bound Actor directory is invalid")
    _require(bound_baseline.is_dir() and not bound_baseline.is_symlink(), "bound baseline directory is invalid")
    return RuntimeBindings(
        raw=raw,
        sha256=digest,
        source_root=bound_source,
        run_directory=bound_run,
        actor_bundle=bound_actor,
        frozen_baseline_repository=bound_baseline,
        python_executable=bound_python,
        package_directory=bound_package,
        snapshots=snapshots,
    )


def _git_output(repository: Path, arguments: Sequence[str], label: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env=environment,
    )
    _require(
        completed.returncode == 0,
        f"{label} failed with exit code {completed.returncode}",
    )
    return completed.stdout


def _verify_frozen_baseline_inputs(
    bindings: RuntimeBindings,
) -> tuple[_ArtifactSnapshot, _ArtifactSnapshot]:
    run = bindings.run_directory
    repository = bindings.frozen_baseline_repository
    bundle_root = run / "baseline-bundle"
    bundle = bundle_root / FROZEN_BASELINE_BUNDLE_NAME
    sidecar = Path(f"{bundle}.sha256")
    _require(
        bundle_root.is_dir()
        and not bundle_root.is_symlink()
        and {path.name for path in bundle_root.iterdir()}
        == {bundle.name, sidecar.name},
        "frozen baseline bundle inventory drifted",
    )
    bundle_snapshot = _stable_file_snapshot(bundle, "frozen baseline bundle")
    sidecar_snapshot = _stable_file_snapshot(
        sidecar, "frozen baseline bundle sidecar"
    )
    _require(
        bundle_snapshot.sha256 == FROZEN_BASELINE_BUNDLE_SHA256
        and sidecar_snapshot.payload
        == f"{FROZEN_BASELINE_BUNDLE_SHA256}  {bundle.name}\n".encode("ascii"),
        "frozen baseline bundle digest drifted",
    )
    head = _git_output(
        repository,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        "frozen baseline commit verification",
    ).decode("ascii", errors="strict").strip()
    _require(head == FROZEN_BASELINE_COMMIT, "frozen baseline commit drifted")
    bundle_check = subprocess.run(
        ["git", "-C", str(repository), "bundle", "verify", str(bundle)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    _require(
        bundle_check.returncode == 0,
        f"frozen baseline bundle verification failed with exit code {bundle_check.returncode}",
    )
    normal = _stable_file_snapshot(
        repository / "lib" / "bot-strategy.ts", "frozen Normal source"
    )
    observation = _stable_file_snapshot(
        bindings.source_root / "training" / "v4-public-history.ts",
        "sealed observation source",
    )
    _require(normal.sha256 == FROZEN_BASELINE_SHA256, "frozen Normal source drifted")
    _require(
        observation.sha256 == OBSERVATION_SCHEMA_SHA256,
        "sealed observation source drifted",
    )
    tree_status = _git_output(
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        "frozen baseline cleanliness verification",
    )
    _require(tree_status == b"", "frozen baseline tree is not an exact clean checkout")
    for snapshot, label in (
        (bundle_snapshot, "frozen baseline bundle"),
        (sidecar_snapshot, "frozen baseline bundle sidecar"),
        (normal, "frozen Normal source"),
        (observation, "sealed observation source"),
    ):
        _assert_snapshot_unchanged(snapshot, label)
    repository_artifact = _capture_artifact(repository, run)
    bundle_artifact = _capture_artifact(bundle_root, run)
    _recheck_protections((repository_artifact, bundle_artifact))
    return repository_artifact, bundle_artifact


def _index_plan(phases: Sequence[PhaseSpec]) -> _PlanIndex:
    seen_phases: dict[str, PhaseSpec] = {}
    commands: dict[str, tuple[PhaseSpec, int, CommandSpec]] = {}
    ordered = tuple(phases)
    for phase in ordered:
        _require(phase.phase_id not in seen_phases, "duplicate workflow phase")
        _require(
            all(dependency in seen_phases for dependency in phase.dependencies),
            f"workflow phase is reordered or has an unknown dependency: {phase.phase_id}",
        )
        seen_phases[phase.phase_id] = phase
        for index, command in enumerate(phase.commands):
            _safe_command_id(command.command_id)
            _require(command.command_id not in commands, "duplicate workflow command")
            commands[command.command_id] = (phase, index, command)
    return _PlanIndex(ordered, seen_phases, commands)


def _plan_sha(bindings: RuntimeBindings) -> str:
    dataset = bindings.run_directory / "merged" / "production.npz"
    _verify_npz_family(dataset)
    return load_fixed_collection_plan_sha256(Path(f"{dataset}.metadata.json"))


def _replacements(
    bindings: RuntimeBindings, *, include_plan_sha: bool
) -> dict[str, str]:
    values = {
        "{remote_source_root}": str(bindings.source_root),
        "{remote_run_directory}": str(bindings.run_directory),
        "{remote_behavior_actor_bundle}": str(bindings.actor_bundle),
        "{remote_frozen_baseline_repository}": str(bindings.frozen_baseline_repository),
        "{remote_python}": str(bindings.python_executable),
        "{remote_package_directory}": str(bindings.package_directory),
        "{package_manifest_sha256}": str(bindings.raw["packageManifestSha256"]),
    }
    if include_plan_sha:
        values["{merged_collection_plan_sha256}"] = _plan_sha(bindings)
    return values


def _receipt_output_values(
    command: CommandSpec, bindings: RuntimeBindings
) -> tuple[str, ...]:
    include_plan_sha = any(
        "{merged_collection_plan_sha256}" in value for value in command.outputs
    )
    replacements = _replacements(bindings, include_plan_sha=include_plan_sha)
    replacements.update(
        {
            "{local_run_directory}": str(bindings.run_directory),
            "{local_source_root}": str(bindings.source_root),
            "{local_behavior_actor_bundle}": str(bindings.actor_bundle),
            "{local_python}": str(bindings.python_executable),
        }
    )
    values = materialize_argv(command.outputs, replacements)
    for value in values:
        _inside(bindings.run_directory, Path(value), "receipt output counterpart")
    return values


def _local_output_suffix(template: str) -> str | None:
    prefix = "{local_run_directory}/"
    if not template.startswith(prefix):
        return None
    relative = PurePosixPath(template[len(prefix) :])
    _require(
        not relative.is_absolute()
        and relative.parts
        and all(part not in ("", ".", "..") for part in relative.parts),
        "local receipt output template is unsafe",
    )
    return relative.as_posix()


def _verify_local_receipt_path(value: object, expected_suffix: str) -> None:
    _require(isinstance(value, str) and value != "" and "\x00" not in value, "local receipt output path is invalid")
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    _require(
        all(part not in ("", ".", "..") for part in parts)
        and normalized.endswith(f"/{expected_suffix}"),
        "local receipt output path has the wrong canonical suffix",
    )


def _materialize_command(
    command: CommandSpec, bindings: RuntimeBindings
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    contains_plan_sha = any(
        "{merged_collection_plan_sha256}" in value
        for value in (*command.argv, *command.outputs)
    )
    replacements = _replacements(bindings, include_plan_sha=contains_plan_sha)
    argv = materialize_argv(command.argv, replacements)
    outputs = materialize_argv(command.outputs, replacements)
    _require(
        argv
        and argv[0] == str(bindings.python_executable)
        and Path(os.path.abspath(argv[0])) == bindings.python_executable,
        "remote command Python drifted",
    )
    script = Path(argv[1]).resolve(strict=True) if len(argv) > 1 else None
    _require(
        script is not None
        and script.parent == bindings.source_root / "gpu-training"
        and script.is_file()
        and not script.is_symlink(),
        "remote command script escaped sealed source",
    )
    serialized = " ".join(argv).lower()
    _require(
        not any(value in serialized for value in FORBIDDEN_COMMAND_TEXT),
        "remote command contains a forbidden deployment, final, or V3 token",
    )
    for output in outputs:
        _inside(bindings.run_directory, Path(output), "remote output")
    return argv, outputs


def _ancestor_phase_ids(index: _PlanIndex, phase: PhaseSpec) -> set[str]:
    result: set[str] = set()

    def visit(phase_id: str) -> None:
        if phase_id in result:
            return
        dependency = index.phase_by_id[phase_id]
        for parent in dependency.dependencies:
            visit(parent)
        result.add(phase_id)

    for dependency in phase.dependencies:
        visit(dependency)
    return result


def _required_prerequisites(
    index: _PlanIndex, phase: PhaseSpec, command_index: int
) -> tuple[tuple[PhaseSpec, CommandSpec], ...]:
    ancestor_ids = _ancestor_phase_ids(index, phase)
    required: list[tuple[PhaseSpec, CommandSpec]] = []
    for candidate in index.phases:
        if candidate.phase_id in ancestor_ids:
            required.extend((candidate, command) for command in candidate.commands)
    required.extend((phase, command) for command in phase.commands[:command_index])
    return tuple(required)


def _verify_sidecar_for_file(path: Path) -> None:
    _require(path.is_file() and not path.is_symlink(), f"missing regular output: {path}")
    digest = _sha256_file(path)
    sidecar = Path(f"{path}.sha256")
    _require(sidecar.is_file() and not sidecar.is_symlink(), f"missing output sidecar: {sidecar}")
    _require(
        sidecar.read_bytes() == f"{digest}  {path.name}\n".encode("ascii"),
        f"stale output sidecar: {sidecar}",
    )


def _verify_npz_family(path: Path) -> None:
    expected = {
        path,
        Path(f"{path}.sha256"),
        Path(f"{path}.metadata.json"),
        Path(f"{path}.metadata.json.sha256"),
    }
    _require(all(item.is_file() and not item.is_symlink() for item in expected), "NPZ four-file family is incomplete")
    _verify_sidecar_for_file(path)
    metadata = Path(f"{path}.metadata.json")
    _canonical_with_sidecar(metadata, "NPZ metadata")


def _verify_output_families(paths: Sequence[Path]) -> None:
    path_set = set(paths)
    for path in paths:
        _require(path.exists() and not path.is_symlink(), f"remote command did not publish {path}")
        _require(path.is_file() or path.is_dir(), f"remote output is not regular: {path}")
    for path in paths:
        if path.name.endswith(".npz"):
            expected = {
                path,
                Path(f"{path}.sha256"),
                Path(f"{path}.metadata.json"),
                Path(f"{path}.metadata.json.sha256"),
            }
            _require(expected.issubset(path_set), "NPZ CommandSpec lacks its exact four-file family")
            _verify_npz_family(path)
    for path in paths:
        if path.name.endswith(".sha256"):
            payload = Path(str(path)[: -len(".sha256")])
            _require(payload in path_set or payload.exists(), "sidecar has no bound payload")
            _verify_sidecar_for_file(payload)
        elif path.suffix == ".json" and path.is_file():
            snapshot = _stable_file_snapshot(path, f"JSON output {path.name}")
            _canonical_object(snapshot, f"JSON output {path.name}")


def _verify_receipt_outputs(
    command: CommandSpec,
    records: Sequence[Mapping[str, object]],
    bindings: RuntimeBindings,
) -> tuple[_ArtifactSnapshot, ...]:
    expected_values = _receipt_output_values(command, bindings)
    _require(len(records) == len(expected_values), "completion output count drifted")
    snapshots: list[_ArtifactSnapshot] = []
    for template, value, record in zip(
        command.outputs, expected_values, records, strict=True
    ):
        local_suffix = _local_output_suffix(template)
        if local_suffix is not None:
            _verify_local_receipt_path(record.get("path"), local_suffix)
            continue
        snapshot = _capture_artifact(Path(value), bindings.run_directory)
        _require(
            dict(record) == dict(snapshot.record),
            f"completion output bytes, hash, size, or path drifted: {command.command_id}",
        )
        snapshots.append(snapshot)
    return tuple(snapshots)


def _load_completion(
    run_directory: Path,
    expected_phase: PhaseSpec,
    expected_command: CommandSpec,
    bindings: RuntimeBindings,
) -> tuple[
    Mapping[str, object],
    str,
    tuple[_Snapshot | _ArtifactSnapshot, ...],
]:
    path = completion_path(run_directory, expected_command.command_id)
    receipt, receipt_sha, snapshots = _canonical_with_sidecar(
        path, f"completion {expected_command.command_id}"
    )
    _require(
        set(receipt)
        == {
            "commandId",
            "commandSpecSha256",
            "format",
            "host",
            "materializedArgvSha256",
            "outputs",
            "packageManifestSha256",
            "passed",
            "phaseId",
            "recipeSha256",
            "runNamespace",
            "runtimeBindingsSha256",
            "version",
        }
        and receipt.get("format") == COMPLETION_FORMAT
        and receipt.get("version") == 1
        and receipt.get("passed") is True,
        f"completion receipt is invalid: {expected_command.command_id}",
    )
    _require(receipt.get("phaseId") == expected_phase.phase_id, "completion phase drifted")
    _require(receipt.get("commandId") == expected_command.command_id, "completion command drifted")
    _require(receipt.get("host") == expected_command.host, "completion host drifted")
    _require(receipt.get("commandSpecSha256") == canonical_sha256(expected_command.to_dict()), "completion command spec drifted")
    _require(receipt.get("packageManifestSha256") == bindings.raw["packageManifestSha256"], "completion package binding drifted")
    _require(receipt.get("recipeSha256") == bindings.raw["recipeSha256"], "completion recipe binding drifted")
    _require(receipt.get("runNamespace") == RUN_NAMESPACE, "completion namespace drifted")
    _require(receipt.get("runtimeBindingsSha256") == bindings.sha256, "completion runtime binding drifted")
    argv_sha = receipt.get("materializedArgvSha256")
    _require(isinstance(argv_sha, str) and SHA256_RE.fullmatch(argv_sha) is not None, "completion argv digest is invalid")
    outputs = receipt.get("outputs")
    _require(isinstance(outputs, list) and len(outputs) == len(expected_command.outputs), "completion output count drifted")
    normalized = [_validate_output_record(item) for item in outputs]
    if expected_command.host == "remote":
        expected_argv, expected_outputs = _materialize_command(expected_command, bindings)
        _require(argv_sha == canonical_sha256(list(expected_argv)), "completion exact argv drifted")
        _require([item["path"] for item in normalized] == [str(Path(value).resolve(strict=True)) for value in expected_outputs], "completion output paths drifted")
    output_snapshots = _verify_receipt_outputs(
        expected_command, normalized, bindings
    )
    return receipt, receipt_sha, (*snapshots, *output_snapshots)


def _verify_finalization_counterparts(
    receipts: Sequence[
        tuple[PhaseSpec, CommandSpec, Mapping[str, object], str]
    ],
) -> None:
    by_id = {
        command.command_id: (command, receipt)
        for _, command, receipt, _ in receipts
    }
    _require(len(by_id) == len(receipts), "duplicate finalization completion receipt")
    local_commands = {
        command.command_id
        for _, command, _, _ in receipts
        if any(_local_output_suffix(template) is not None for template in command.outputs)
    }
    _require(
        local_commands == set(LOCAL_OUTPUT_COUNTERPARTS),
        "local finalization counterpart coverage drifted",
    )
    for local_id, counterparts in LOCAL_OUTPUT_COUNTERPARTS.items():
        local_command, local_receipt = by_id[local_id]
        local_outputs = local_receipt.get("outputs")
        _require(
            isinstance(local_outputs, list)
            and len(local_outputs) == len(local_command.outputs)
            and len(local_outputs) == len(counterparts),
            f"local counterpart output count drifted: {local_id}",
        )
        for index, (counterpart_id, counterpart_index) in enumerate(counterparts):
            counterpart = by_id.get(counterpart_id)
            _require(
                counterpart is not None,
                f"missing remote counterpart receipt: {counterpart_id}",
            )
            _, counterpart_receipt = counterpart
            counterpart_outputs = counterpart_receipt.get("outputs")
            _require(
                isinstance(counterpart_outputs, list)
                and counterpart_index < len(counterpart_outputs),
                f"remote counterpart output index drifted: {counterpart_id}",
            )
            local_record = _validate_output_record(local_outputs[index])
            remote_record = _validate_output_record(
                counterpart_outputs[counterpart_index]
            )
            _require(
                local_record.get("kind") == remote_record.get("kind")
                and local_record.get("sha256") == remote_record.get("sha256")
                and local_record.get("size") == remote_record.get("size"),
                f"local/remote counterpart bytes drifted: {local_id}[{index}]",
            )


def execute_recipe_command(
    *,
    source_root: Path,
    run_directory: Path,
    package_directory: Path,
    package_manifest_sha256: str,
    runtime_bindings_path: Path,
    command_id: str,
) -> Mapping[str, object]:
    command_id = _safe_command_id(command_id)
    source = source_root.resolve(strict=True)
    run = run_directory.resolve(strict=True)
    package = package_directory.resolve(strict=True)
    recipe, sealed_snapshots = _load_sealed_recipe(source, package, package_manifest_sha256)
    recipe_sha = canonical_sha256(recipe)
    bindings = load_runtime_bindings(
        runtime_bindings_path,
        source_root=source,
        run_directory=run,
        package_directory=package,
        package_manifest_sha256=package_manifest_sha256,
        recipe_sha256=recipe_sha,
    )
    index = _index_plan(build_mixed_phase_plan(recipe))
    selected = index.command_by_id.get(command_id)
    _require(selected is not None, "command ID is not in the sealed workflow")
    phase, command_index, command = selected
    _require(command.host == "remote", "worker may execute only exact remote commands")
    baseline_snapshots: tuple[_ArtifactSnapshot, ...] = ()
    if command.command_id == SCREEN_COMMAND_ID:
        baseline_snapshots = _verify_frozen_baseline_inputs(bindings)
    argv, output_values = _materialize_command(command, bindings)
    dependency_snapshots: list[_Snapshot | _ArtifactSnapshot] = []
    for dependency_phase, dependency_command in _required_prerequisites(
        index, phase, command_index
    ):
        _, _, snapshots = _load_completion(
            run, dependency_phase, dependency_command, bindings
        )
        dependency_snapshots.extend(snapshots)
    receipt_path = completion_path(run, command_id)
    _require(not receipt_path.exists() and not Path(f"{receipt_path}.sha256").exists(), "command completion receipt already exists")
    output_paths = tuple(Path(value) for value in output_values)
    stdout_path = run / "logs" / f"{command_id}.stdout"
    stderr_path = run / "logs" / f"{command_id}.stderr"
    for output in output_paths:
        _require(not output.exists() and not output.is_symlink(), f"remote output is not fresh: {output}")
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for log_path in (stdout_path, stderr_path):
        _require(
            not log_path.exists() and not log_path.is_symlink(),
            f"command log is not fresh: {log_path}",
        )
    protected_snapshots = (
        *sealed_snapshots,
        *bindings.snapshots,
        *baseline_snapshots,
        *dependency_snapshots,
    )
    _recheck_protections(protected_snapshots)
    with (
        _open_fresh_binary_log(stdout_path, run) as stdout_log,
        _open_fresh_binary_log(stderr_path, run) as stderr_log,
    ):
        completed = subprocess.run(
            list(argv),
            cwd=source,
            stdout=stdout_log,
            stderr=stderr_log,
            check=False,
            shell=False,
        )
        for log in (stdout_log, stderr_log):
            log.flush()
            os.fsync(log.fileno())
    stdout_snapshot = _stable_file_snapshot(stdout_path, "command stdout log")
    stderr_snapshot = _stable_file_snapshot(stderr_path, "command stderr log")
    _require(
        completed.returncode == 0,
        _failed_command_summary(
            command_id, completed.returncode, stdout_snapshot, stderr_snapshot
        ),
    )
    _verify_output_families(output_paths)
    output_snapshots = tuple(
        _capture_artifact(path, run) for path in output_paths
    )
    output_inventory = [dict(snapshot.record) for snapshot in output_snapshots]
    _recheck_protections(protected_snapshots)
    command_artifacts = (
        stdout_snapshot,
        stderr_snapshot,
        *output_snapshots,
    )
    _recheck_protections(command_artifacts)
    receipt = build_completion_receipt(
        phase=phase,
        command=command,
        materialized_argv=argv,
        runtime_bindings=bindings.raw,
        runtime_bindings_sha256=bindings.sha256,
        outputs=output_inventory,
    )
    published = False
    try:
        digest = publish_completion_receipt(receipt_path, receipt)
        published = True
        _recheck_protections(protected_snapshots)
        _recheck_protections(command_artifacts)
    except BaseException:
        if published:
            _rollback_completion_receipt(receipt_path)
        raise
    return {**receipt, "receiptSha256": digest}


def _verify_promotion_screen_binding(run: Path) -> tuple[_Snapshot, ...]:
    screen_path = run / "screening" / "epoch-0001.json"
    promotion_path = run / "screening" / "epoch-0001-promotion-gates.json"
    _, screen_sha, screen_snapshots = _canonical_with_sidecar(
        screen_path, "screening report"
    )
    promotion, _, promotion_snapshots = _canonical_with_sidecar(
        promotion_path, "promotion report"
    )
    _require(
        promotion.get("format") == "dalmuti-v4-mixed-promotion-gates"
        and promotion.get("passed") is True
        and promotion.get("allPlayerCountsPassed") is True
        and promotion.get("screeningReportSha256") == screen_sha,
        "promotion report is not bound to the current passing screening report",
    )
    return (*screen_snapshots, *promotion_snapshots)


def finalize_run(
    *,
    source_root: Path,
    run_directory: Path,
    package_directory: Path,
    package_manifest_sha256: str,
    runtime_bindings_path: Path,
) -> Mapping[str, object]:
    source = source_root.resolve(strict=True)
    run = run_directory.resolve(strict=True)
    package = package_directory.resolve(strict=True)
    recipe, sealed_snapshots = _load_sealed_recipe(source, package, package_manifest_sha256)
    bindings = load_runtime_bindings(
        runtime_bindings_path,
        source_root=source,
        run_directory=run,
        package_directory=package,
        package_manifest_sha256=package_manifest_sha256,
        recipe_sha256=canonical_sha256(recipe),
    )
    baseline_snapshots = _verify_frozen_baseline_inputs(bindings)
    index = _index_plan(build_mixed_phase_plan(recipe))
    selected = index.command_by_id.get(FINAL_SCREEN_COMMAND_ID)
    _require(selected is not None, "sealed workflow lacks final structural screening")
    phase, command_index, command = selected
    target_receipt, target_receipt_sha, target_snapshots = _load_completion(
        run, phase, command, bindings
    )
    dependency_snapshots: list[_Snapshot | _ArtifactSnapshot] = []
    required_receipts: list[tuple[PhaseSpec, CommandSpec, Mapping[str, object], str]] = []
    for dependency_phase, dependency_command in _required_prerequisites(index, phase, command_index):
        receipt, receipt_sha, snapshots = _load_completion(
            run, dependency_phase, dependency_command, bindings
        )
        dependency_snapshots.extend(snapshots)
        required_receipts.append(
            (dependency_phase, dependency_command, receipt, receipt_sha)
        )
    required_receipts.append((phase, command, target_receipt, target_receipt_sha))
    _verify_finalization_counterparts(required_receipts)
    receipt_protections = (
        *sealed_snapshots,
        *bindings.snapshots,
        *baseline_snapshots,
        *target_snapshots,
        *dependency_snapshots,
    )
    _recheck_protections(receipt_protections)
    semantic_snapshots = _verify_promotion_screen_binding(run)
    _recheck_protections((*receipt_protections, *semantic_snapshots))
    verify_screening(
        package,
        package_manifest_sha256,
        source,
        run / "screening" / "epoch-0001.json",
        run / "training" / "train-seed-650000001-run-001" / "candidate",
    )
    _recheck_protections((*receipt_protections, *semantic_snapshots))
    seal_path = run / "provenance" / "final-files.json"
    success_path = run / "status" / "999-succeeded.json"
    _require(
        not seal_path.exists()
        and not Path(f"{seal_path}.sha256").exists()
        and not success_path.exists()
        and not Path(f"{success_path}.sha256").exists(),
        "final run outputs are not fresh",
    )
    finalization_audit_path = run / "provenance" / "finalization-audit.json"
    _require(
        not finalization_audit_path.exists()
        and not Path(f"{finalization_audit_path}.sha256").exists(),
        "finalization audit is not fresh",
    )
    finalization_audit = {
        "fixedCollectionPlanSha256": _plan_sha(bindings),
        "format": "dalmuti-v4-mixed-finalization-audit",
        "packageManifestSha256": package_manifest_sha256,
        "passed": True,
        "recipeSha256": canonical_sha256(recipe),
        "requiredCommands": [
            {
                "commandId": required_command.command_id,
                "commandSpecSha256": canonical_sha256(required_command.to_dict()),
                "completionReceiptSha256": receipt_sha,
                "outputs": receipt["outputs"],
                "phaseId": required_phase.phase_id,
            }
            for required_phase, required_command, receipt, receipt_sha in required_receipts
        ],
        "runNamespace": RUN_NAMESPACE,
        "runtimeBindingsSha256": bindings.sha256,
        "version": 1,
    }
    finalization_audit_sha = _publish_canonical_with_sidecar(
        finalization_audit_path, finalization_audit
    )
    _, _, finalization_audit_snapshots = _canonical_with_sidecar(
        finalization_audit_path,
        "finalization audit",
        expected_sha256=finalization_audit_sha,
    )
    protected_snapshots = (
        *receipt_protections,
        *semantic_snapshots,
        *finalization_audit_snapshots,
    )
    _recheck_protections(protected_snapshots)
    recipe_sha = canonical_sha256(recipe)
    run_contract_sha = canonical_sha256(recipe["runContract"])
    seal = seal_run(
        run,
        seal_path,
        run / "status",
        run / "screening" / "epoch-0001.json",
        run / "screening" / "epoch-0001-promotion-gates.json",
        package_manifest_sha256,
        recipe_sha,
        run_contract_sha,
        bindings.sha256,
        finalization_audit_path,
        finalization_audit_sha,
        profile="remote-semantic",
    )
    seal_digest = verify_run_seal(seal_path)
    _require(
        seal.get("sealSha256") == seal_digest,
        "semantic seal result binding drifted",
    )
    _recheck_protections(protected_snapshots)
    result = {
        "format": "dalmuti-v4-mixed-finalization",
        "packageManifestSha256": package_manifest_sha256,
        "passed": True,
        "recipeSha256": recipe_sha,
        "runNamespace": RUN_NAMESPACE,
        "runSealSha256": seal_digest,
        "version": 1,
    }
    # Terminal success publication is deliberately the final fallible action.
    # The runtime validates the seal before and after its all-or-nothing write;
    # nothing may inspect mutable state after it returns.
    write_status(
        success_path,
        "complete",
        "succeeded",
        "replay, training, hard gates, p4-p10 screening and promotion gates passed",
        seal_path,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run-command", "finalize-run"):
        command = commands.add_parser(name)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--run-directory", type=Path, required=True)
        command.add_argument("--package-directory", type=Path, required=True)
        command.add_argument("--package-manifest-sha256", required=True)
        command.add_argument("--runtime-bindings", type=Path, required=True)
        if name == "run-command":
            command.add_argument("--command-id", required=True)
    return parser


def _failure_log_binding(run: Path, command_id: str, stream: str) -> str:
    path = run / "logs" / f"{command_id}.{stream}"
    try:
        snapshot = _stable_file_snapshot(path, f"failure {stream} log")
    except (OSError, ValueError):
        return "unavailable"
    return f"logs/{path.name}:{snapshot.sha256}"


def _sanitize_failure_message(error: BaseException) -> str:
    message = str(error).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    message = message.replace(";", ",")
    message = re.sub(
        r"(?i)\b(password|passwd|token|secret|authorization|private[-_ ]?key)\s*[:=]\s*[^\s,]+",
        lambda match: f"{match.group(1)}=[redacted]",
        message,
    )
    message = " ".join(message.split())
    return (message or "unavailable")[:2048]


def _publish_failure_status_best_effort(
    run_directory: Path, stage: str, error: BaseException
) -> None:
    try:
        stage = _safe_command_id(stage)
        run = run_directory.resolve(strict=True)
        status = run / "status"
        _require(
            status.is_dir() and not status.is_symlink(),
            "remote status directory is invalid",
        )
        failed = status / "998-failed.json"
        succeeded = status / "999-succeeded.json"
        failed_paths = (failed, Path(f"{failed}.sha256"))
        succeeded_paths = (succeeded, Path(f"{succeeded}.sha256"))
        # A complete or partial terminal publication is immutable evidence.
        # Never add the opposite terminal state beside it.
        if any(path.exists() or path.is_symlink() for path in succeeded_paths):
            return
        if any(path.exists() or path.is_symlink() for path in failed_paths):
            return
        detail = (
            f"errorType={type(error).__name__}; "
            f"message={_sanitize_failure_message(error)}; "
            f"stdout={_failure_log_binding(run, stage, 'stdout')}; "
            f"stderr={_failure_log_binding(run, stage, 'stderr')}"
        )
        write_status(failed, stage, "failed", detail, None)
        _require(
            all(path.is_file() and not path.is_symlink() for path in failed_paths)
            and not any(
                path.exists() or path.is_symlink() for path in succeeded_paths
            ),
            "remote failure status publication is incomplete",
        )
    except BaseException:
        # This path is deliberately best-effort. The original exception remains
        # authoritative, and a partial/terminal status is never overwritten.
        return


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "source_root": arguments.source_root,
        "run_directory": arguments.run_directory,
        "package_directory": arguments.package_directory,
        "package_manifest_sha256": arguments.package_manifest_sha256,
        "runtime_bindings_path": arguments.runtime_bindings,
    }
    stage = (
        arguments.command_id
        if arguments.command == "run-command"
        else "finalize-remote-run"
    )
    try:
        if arguments.command == "run-command":
            result = execute_recipe_command(command_id=arguments.command_id, **common)
        else:
            result = finalize_run(**common)
    except BaseException as error:
        _publish_failure_status_best_effort(
            arguments.run_directory, stage, error
        )
        raise RuntimeError(
            f"remote worker stage {stage} failed: {_sanitize_failure_message(error)}; "
            "see immutable status/log evidence"
        ) from error
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
