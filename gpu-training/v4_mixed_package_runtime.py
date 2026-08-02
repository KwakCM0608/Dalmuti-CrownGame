#!/usr/bin/env python3
"""Standard-library trust boundary for sealed V4 mixed-host run packages.

This file is copied byte-for-byte from a committed Git blob into each package.
It deliberately has no project imports for package/archive/status operations.
The screening command additionally loads the *sealed extracted* evaluator and
then applies the run recipe's stricter, complete screening contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_LEAF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PLAYER_COUNTS = tuple(range(4, 11))
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
PACKAGE_FORMAT = "dalmuti-v4-mixed-run-package"
BINDING_FORMAT = "dalmuti-v4-mixed-source-binding"
RECIPE_FORMAT = "dalmuti-v4-mixed-package-recipe"
RUN_NAMESPACE = "v4-fixedid-ppo-i001-mixedmath-s600000001"
V4_POLICY_NUMERICS_FIELDS = frozenset(
    {
        "actorForwardDtype",
        "contract",
        "contractSha256",
        "cudaMatmulTf32Allowed",
        "cudnnBenchmark",
        "cudnnDeterministic",
        "cudnnSdpEnabled",
        "cudnnTf32Allowed",
        "deterministicAlgorithms",
        "flashSdpEnabled",
        "mathSdpEnabled",
        "memoryEfficientSdpEnabled",
        "mhaFastpathEnabled",
        "requiredCudaCublasWorkspaceConfig",
        "version",
    }
)
V4_REPLAY_AUDIT_FIELDS = frozenset(
    {
        "absoluteTolerance",
        "actorAutocastEnabled",
        "actorForwardDtype",
        "actorMode",
        "auditBatchSize",
        "effectiveNonforcedPpoRowCount",
        "forcedMaximumAbsoluteLogProbabilityError",
        "forcedSingletonPpoRowCount",
        "forcedSingletonRowsByPlayerCount",
        "maximumAbsoluteLogProbabilityError",
        "meanAbsoluteLogProbabilityError",
        "nonforcedBalancedEntropy",
        "nonforcedEntropyByPlayerCount",
        "nonforcedRowsByPlayerCount",
        "nonforcedTotalWeightMass",
        "nonforcedWeightMassByPlayerCount",
        "passed",
        "ppoEligibleRowCount",
        "storedOldActionLogProbabilityDtype",
        "version",
    }
)
V4_REPLAY_REPORT_FIELDS = frozenset(
    {
        "actorSha256",
        "audit",
        "datasetFingerprint",
        "datasetSha256",
        "device",
        "fixedCollectionPlanSha256",
        "format",
        "manifestSha256",
        "passed",
        "policyNumerics",
        "strata",
        "version",
    }
)
V4_REPLAY_STRATUM_FIELDS = frozenset(
    {
        "backend",
        "count",
        "maximumAbsoluteLogProbabilityError",
        "meanAbsoluteLogProbabilityError",
        "playerCount",
        "shardIndex",
    }
)
V4_REPLAY_TOLERANCE = 2.0e-5
V4_MIXED_BACKEND_MAP = ("cpu", "cpu", *("cuda" for _ in range(12)))
FROZEN_BASELINE_BUNDLE_NAME = "dalmuti-e0c52b0.bundle"
FROZEN_BASELINE_BUNDLE_SHA256 = (
    "9ea0b9eb4200ac369fbc3ffb1493efe59625b34f5f994359f8a01d4b5610db4d"
)
FROZEN_BASELINE_COMMIT = "e0c52b0462d86756cf40b90f19d35a3e26b0f674"
FROZEN_NORMAL_SHA256 = (
    "aa44743c64a23ac002d7faf09867bdb3e06232320f8efeb1df0e42724037bb61"
)
FROZEN_OBSERVATION_SHA256 = (
    "13dc7e4846669a4130dd69dd8b450c4ca3a443c2d1f64cfa08583c6a1108e99f"
)
FINALIZATION_COMMAND_IDS = (
    "verify-local-actor",
    "stage-remote-source-and-actor",
    "collect-calibration-cpu",
    "collect-calibration-cuda",
    "retrieve-calibration-cuda",
    "compare-calibration-backends",
    "upload-calibration-triple",
    *(f"collect-production-shard-{index:02d}" for index in range(14)),
    "retrieve-remote-production-shards",
    "merge-production-shards",
    "upload-merged-production",
    "replay-full-ppo-dataset",
    "train-epoch-one-cuda",
    "publish-candidate-actor-sidecar",
    "verify-epoch-one-hard-gates",
    "screen-epoch-one-p4-p10",
    "verify-screening-promotion-gates",
    "verify-complete-remote-screening",
)


def _local_output_counterparts() -> dict[str, tuple[tuple[str, int], ...]]:
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


LOCAL_OUTPUT_COUNTERPARTS = _local_output_counterparts()


@dataclass(frozen=True)
class StableSnapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class RemoteSourceVerification:
    package_root: Path
    package_snapshots: Mapping[str, StableSnapshot]
    package_root_identity: tuple[int, int, int, int, int]
    package_expected_names: set[str]
    source_root: Path
    source_snapshots: Mapping[str, StableSnapshot]
    source_root_identity: tuple[int, int, int, int, int]
    source_directory_identities: Mapping[str, tuple[int, int, int, int, int]]


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    # ``lstat`` and ``fstat`` can report different creation/change times for
    # the same file on Windows.  The path identity intentionally excludes
    # ctime, but retains the inode, size, mtime, and permission bits so a
    # path replacement or post-verification chmod is still detected.
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(stat.S_IMODE(value.st_mode)),
    )


def _path_and_descriptor_match(
    path_stat: os.stat_result, descriptor_stat: os.stat_result
) -> bool:
    """Return whether a path and its open descriptor refer to one file.

    Windows does not guarantee that every metadata field returned by
    ``lstat`` is byte-for-byte identical to the field returned by ``fstat``.
    Size must always agree.  When both APIs provide a real inode, device and
    inode additionally bind the opened descriptor to the verified path.
    """

    if int(path_stat.st_size) != int(descriptor_stat.st_size):
        return False
    path_inode = int(path_stat.st_ino)
    descriptor_inode = int(descriptor_stat.st_ino)
    if path_inode and descriptor_inode:
        return (
            int(path_stat.st_dev) == int(descriptor_stat.st_dev)
            and path_inode == descriptor_inode
        )
    return True


def stable_snapshot(path: Path, label: str) -> StableSnapshot:
    try:
        before_path = path.lstat()
    except OSError as error:
        raise ValueError(f"missing {label}: {path}") from error
    require(stat.S_ISREG(before_path.st_mode) and not path.is_symlink(), f"{label} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before_fd = os.fstat(descriptor)
        require(
            _path_and_descriptor_match(before_path, before_fd),
            f"{label} changed while opening",
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    require(
        _stat_identity(before_fd) == _stat_identity(after_fd),
        f"{label} changed while reading",
    )
    require(
        _stat_identity(before_path) == _stat_identity(after_path),
        f"{label} changed while reading",
    )
    require(
        _path_and_descriptor_match(after_path, after_fd),
        f"{label} changed while reading",
    )
    identity = _stat_identity(after_path)
    payload = b"".join(chunks)
    require(len(payload) == before_fd.st_size, f"{label} was read incompletely")
    return StableSnapshot(path=path, payload=payload, sha256=sha256_bytes(payload), identity=identity)


def recheck_snapshot(snapshot: StableSnapshot, label: str) -> None:
    current = stable_snapshot(snapshot.path, label)
    require(
        current.identity == snapshot.identity
        and current.sha256 == snapshot.sha256
        and current.payload == snapshot.payload,
        f"{label} changed after verification",
    )


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_sha256(value: object, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"invalid {label}")
    return value


def safe_relative_path(value: object, label: str) -> PurePosixPath:
    require(isinstance(value, str) and value != "", f"missing {label}")
    require("\\" not in value and "\x00" not in value, f"unsafe {label}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts), f"unsafe {label}")
    return path


def safe_leaf(value: object, label: str) -> str:
    require(isinstance(value, str) and SAFE_LEAF_RE.fullmatch(value) is not None, f"unsafe {label}")
    return value


def resolve_inside(root: Path, relative: PurePosixPath) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = root_resolved.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=False)
    require(resolved == root_resolved or root_resolved in resolved.parents, "path escaped its root")
    return candidate


def load_canonical_json(path: Path, label: str) -> Mapping[str, Any]:
    return load_canonical_json_snapshot(stable_snapshot(path, label), label)


def load_canonical_json_snapshot(snapshot: StableSnapshot, label: str) -> Mapping[str, Any]:
    payload = snapshot.payload
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    require(isinstance(value, Mapping), f"{label} is not an object")
    require(payload == canonical_json_bytes(value), f"{label} is not canonical JSON")
    return value


def sidecar_bytes(digest: str, leaf: str) -> bytes:
    require_sha256(digest, "sidecar digest")
    safe_leaf(leaf, "sidecar filename")
    return f"{digest}  {leaf}\n".encode("ascii")


def verify_sidecar(path: Path, expected_digest: str | None = None) -> str:
    payload, sidecar = snapshot_with_sidecar(path, expected_digest)
    recheck_snapshot(payload, f"payload {path.name}")
    recheck_snapshot(sidecar, f"sidecar {sidecar.path.name}")
    return payload.sha256


def snapshot_with_sidecar(
    path: Path, expected_digest: str | None = None
) -> tuple[StableSnapshot, StableSnapshot]:
    payload = stable_snapshot(path, f"payload {path.name}")
    digest = payload.sha256
    if expected_digest is not None:
        require(digest == expected_digest, f"payload digest mismatch: {path.name}")
    sidecar_path = Path(f"{path}.sha256")
    sidecar = stable_snapshot(sidecar_path, f"sidecar {sidecar_path.name}")
    require(sidecar.payload == sidecar_bytes(digest, path.name), f"stale or malformed sidecar: {sidecar_path.name}")
    return payload, sidecar


def _package_inventory(manifest: Mapping[str, Any], *, remote_only: bool) -> list[Mapping[str, Any]]:
    files = manifest.get("files")
    require(isinstance(files, list) and files, "package file inventory is missing")
    roles: set[str] = set()
    names: set[str] = set()
    selected: list[Mapping[str, Any]] = []
    for raw in files:
        require(isinstance(raw, Mapping), "package file entry is invalid")
        require(set(raw) == {"name", "remotePayload", "role", "sha256", "size"}, "package file entry fields are non-canonical")
        name = safe_leaf(raw.get("name"), "package filename")
        role = safe_leaf(raw.get("role"), "package role")
        require(role not in roles and name not in names, "duplicate package role or filename")
        roles.add(role)
        names.add(name)
        require(isinstance(raw.get("remotePayload"), bool), "remotePayload is not boolean")
        require_sha256(raw.get("sha256"), "package file digest")
        size = raw.get("size")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, "invalid package file size")
        if not remote_only or raw["remotePayload"]:
            selected.append(raw)
    require(
        roles == {"source-archive", "source-binding", "verifier", "controller"},
        "package roles are incomplete",
    )
    return selected


def _directory_identity(path: Path, label: str) -> tuple[int, int, int, int, int]:
    value = path.lstat()
    require(stat.S_ISDIR(value.st_mode) and not path.is_symlink(), f"{label} is invalid")
    return _stat_identity(value)


def _recheck_snapshot_set(
    root: Path,
    root_identity: tuple[int, int, int, int, int],
    snapshots: Mapping[str, StableSnapshot],
    expected_names: set[str],
    label: str,
) -> None:
    require(_directory_identity(root, label) == root_identity, f"{label} was replaced")
    observed_names = {path.name for path in root.iterdir()}
    require(observed_names == expected_names, f"{label} inventory changed after verification")
    for name, snapshot in snapshots.items():
        require(snapshot.path == root / name, f"{label} snapshot path drifted")
        recheck_snapshot(snapshot, f"{label} file {name}")
    require(_directory_identity(root, label) == root_identity, f"{label} changed during final recheck")


def _load_package(
    package_dir: Path, expected_manifest_sha256: str, *, remote_only: bool
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, StableSnapshot],
    tuple[int, int, int, int, int],
    set[str],
]:
    require_sha256(expected_manifest_sha256, "expected package manifest digest")
    root = package_dir.resolve(strict=True)
    root_identity = _directory_identity(root, "package directory")
    manifest_path = root / "package-manifest.json"
    manifest_snapshot, manifest_sidecar = snapshot_with_sidecar(
        manifest_path, expected_manifest_sha256
    )
    manifest = load_canonical_json_snapshot(manifest_snapshot, "package manifest")
    require(manifest.get("format") == PACKAGE_FORMAT and manifest.get("version") == 1, "unsupported package manifest")
    # canonicalSha256 is defined over the manifest with that field absent;
    # the adjacent sidecar and caller-provided trust root bind the final bytes.
    unsigned = dict(manifest)
    unsigned.pop("canonicalSha256", None)
    require(
        manifest.get("canonicalSha256") == sha256_bytes(canonical_json_bytes(unsigned)),
        "manifest canonical digest mismatch",
    )
    all_records = _package_inventory(manifest, remote_only=False)
    selected: list[Mapping[str, Any]] = []
    for record in all_records:
        path = root / str(record["name"])
        sidecar_path = Path(f"{path}.sha256")
        present = path.exists() or sidecar_path.exists()
        if not remote_only or record["remotePayload"] or present:
            require(path.exists() and sidecar_path.exists(), "package contains a partial optional payload")
            selected.append(record)
    snapshots = {
        "package-manifest.json": manifest_snapshot,
        "package-manifest.json.sha256": manifest_sidecar,
    }
    for record in selected:
        path = root / str(record["name"])
        payload_snapshot, sidecar_snapshot = snapshot_with_sidecar(
            path, str(record["sha256"])
        )
        require(len(payload_snapshot.payload) == record["size"], f"package payload size mismatch: {path.name}")
        snapshots[path.name] = payload_snapshot
        snapshots[f"{path.name}.sha256"] = sidecar_snapshot
    binding_record = next(item for item in manifest["files"] if item["role"] == "source-binding")
    archive_record = next(item for item in manifest["files"] if item["role"] == "source-archive")
    binding_snapshot = snapshots[str(binding_record["name"])]
    archive_snapshot = snapshots[str(archive_record["name"])]
    binding = load_canonical_json_snapshot(binding_snapshot, "source binding")
    require(binding.get("format") == BINDING_FORMAT and binding.get("version") == 1, "unsupported source binding")
    require(binding.get("sourceArchive") == {"name": archive_record["name"], "sha256": archive_record["sha256"], "size": archive_record["size"]}, "archive/binding mismatch")
    require(binding.get("sourceCommit") == manifest.get("sourceCommit"), "commit/binding mismatch")
    require(binding.get("ledger") == manifest.get("ledger"), "ledger/archive binding mismatch")
    require(binding.get("recipe") == manifest.get("recipe"), "recipe/archive binding mismatch")
    _verify_archive(archive_snapshot.payload, binding)
    sources_by_path = {str(item["path"]): item for item in _source_records(binding)}
    for label in ("ledger", "recipe"):
        commitment = binding.get(label)
        require(
            isinstance(commitment, Mapping)
            and set(commitment) == {"gitBlobOid", "path", "sha256"},
            f"{label} commitment is non-canonical",
        )
        source = sources_by_path.get(str(commitment.get("path")))
        require(
            source is not None
            and source["gitBlobOid"] == commitment.get("gitBlobOid")
            and source["sha256"] == commitment.get("sha256"),
            f"{label} commitment is not in the archive inventory",
        )
    expected_names = set(snapshots)
    _recheck_snapshot_set(
        root, root_identity, snapshots, expected_names, "package directory"
    )
    return manifest, binding, snapshots, root_identity, expected_names


def _source_records(binding: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = binding.get("sourceFiles")
    require(isinstance(raw, list) and raw, "source inventory is missing")
    records: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for item in raw:
        require(isinstance(item, Mapping), "source inventory entry is invalid")
        require(set(item) == {"gitBlobOid", "mode", "path", "sha256", "size"}, "source inventory fields are non-canonical")
        path = str(safe_relative_path(item.get("path"), "source path"))
        mode = item.get("mode")
        require(mode in (0o644, 0o755), "source file mode is invalid")
        require_sha256(item.get("sha256"), "source file digest")
        oid = item.get("gitBlobOid")
        require(isinstance(oid, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid) is not None, "source Git blob id is invalid")
        size = item.get("size")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, "source file size is invalid")
        paths.append(path)
        records.append(item)
    require(paths == sorted(paths) and len(paths) == len(set(paths)), "source inventory order or uniqueness is invalid")
    inventory_sha = sha256_bytes(canonical_json_bytes(records))
    require(binding.get("sourceInventorySha256") == inventory_sha, "source inventory digest mismatch")
    return records


def _verify_archive(archive_payload: bytes, binding: Mapping[str, Any]) -> None:
    prefix = str(safe_relative_path(binding.get("archivePrefix"), "archive prefix"))
    records = _source_records(binding)
    expected = {f"{prefix}/{item['path']}": item for item in records}
    seen: dict[str, tarfile.TarInfo] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                require(member.name not in seen, "duplicate archive member")
                require(member.isfile() and not member.issym() and not member.islnk(), "archive contains a non-regular member")
                require(member.name in expected, "archive contains an unbound member")
                record = expected[member.name]
                require(member.uid == 0 and member.gid == 0 and member.uname == "" and member.gname == "" and member.mtime == 0, "archive metadata is not deterministic")
                require(stat.S_IMODE(member.mode) == record["mode"], "archive mode mismatch")
                require(member.size == record["size"], "archive size mismatch")
                handle = archive.extractfile(member)
                require(handle is not None, "cannot read archive member")
                require(sha256_bytes(handle.read()) == record["sha256"], "archive member digest mismatch")
                seen[member.name] = member
    except (tarfile.TarError, OSError) as error:
        raise ValueError("invalid source archive") from error
    require(set(seen) == set(expected), "archive source inventory is incomplete")


def verify_package(package_dir: Path, expected_manifest_sha256: str, *, remote_only: bool = False) -> Mapping[str, Any]:
    manifest, binding, snapshots, root_identity, expected_names = _load_package(
        package_dir, expected_manifest_sha256, remote_only=remote_only
    )
    root = package_dir.resolve(strict=True)
    _recheck_snapshot_set(root, root_identity, snapshots, expected_names, "package directory")
    return {
        "format": "dalmuti-v4-mixed-package-verification",
        "version": 1,
        "packageId": manifest["packageId"],
        "packageManifestSha256": expected_manifest_sha256,
        "sourceCommit": manifest["sourceCommit"],
        "sourceInventorySha256": binding["sourceInventorySha256"],
        "passed": True,
    }


def extract_source(package_dir: Path, expected_manifest_sha256: str, destination: Path) -> Mapping[str, Any]:
    manifest, binding, snapshots, root_identity, expected_names = _load_package(
        package_dir, expected_manifest_sha256, remote_only=True
    )
    require(not destination.exists() and not destination.is_symlink(), "source destination is immutable and must be fresh")
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    archive_record = next(item for item in manifest["files"] if item["role"] == "source-archive")
    archive_snapshot = snapshots[str(archive_record["name"])]
    records = _source_records(binding)
    prefix = str(binding["archivePrefix"])
    expected = {f"{prefix}/{item['path']}": item for item in records}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_snapshot.payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                record = expected[member.name]
                relative = PurePosixPath(str(record["path"]))
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                require(not target.exists() and not target.is_symlink(), "source extraction would overwrite a path")
                handle = archive.extractfile(member)
                require(handle is not None, "cannot extract source member")
                payload = handle.read()
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                descriptor = os.open(target, flags, int(record["mode"]))
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(payload)
                        output.flush()
                        os.fsync(output.fileno())
                except BaseException:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    raise
    except BaseException:
        # Preserve the failed fresh directory as evidence; never silently reuse it.
        raise
    for record in records:
        target = destination.joinpath(*PurePosixPath(str(record["path"])).parts)
        require(target.is_file() and not target.is_symlink(), "extracted source file is invalid")
        require(sha256_file(target) == record["sha256"], "extracted source digest mismatch")
        os.chmod(target, int(record["mode"]) & ~0o222)
    for directory in sorted((item for item in destination.rglob("*") if item.is_dir()), reverse=True):
        os.chmod(directory, 0o555)
    os.chmod(destination, 0o555)
    _verify_extracted_source(destination, binding)
    _recheck_snapshot_set(
        package_dir.resolve(strict=True),
        root_identity,
        snapshots,
        expected_names,
        "package directory",
    )
    return {
        "format": "dalmuti-v4-mixed-source-extraction",
        "version": 1,
        "packageId": manifest["packageId"],
        "sourceCommit": manifest["sourceCommit"],
        "sourceInventorySha256": binding["sourceInventorySha256"],
        "fileCount": len(records),
        "passed": True,
    }


def _verify_extracted_source(
    source_root: Path, binding: Mapping[str, Any]
) -> tuple[
    dict[str, StableSnapshot],
    tuple[int, int, int, int, int],
    dict[str, tuple[int, int, int, int, int]],
]:
    root = source_root.resolve(strict=True)
    root_identity = _directory_identity(root, "extracted source root")
    records = _source_records(binding)
    expected = {str(item["path"]): item for item in records}
    observed: set[str] = set()
    snapshots: dict[str, StableSnapshot] = {}
    directory_identities: dict[str, tuple[int, int, int, int, int]] = {
        ".": root_identity
    }
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"extracted source contains a symlink: {relative}")
        if path.is_dir():
            directory_identities[relative] = _directory_identity(
                path, f"source directory {relative}"
            )
            require(
                stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
                f"extracted source directory remains writable: {relative}",
            )
            continue
        require(relative in expected, f"extracted source contains an unbound file: {relative}")
        record = expected[relative]
        snapshot = stable_snapshot(path, f"extracted source {relative}")
        require(len(snapshot.payload) == record["size"], f"extracted source size mismatch: {relative}")
        require(snapshot.sha256 == record["sha256"], f"extracted source digest mismatch: {relative}")
        require(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0, f"extracted source file remains writable: {relative}")
        snapshots[relative] = snapshot
        observed.add(relative)
    require(observed == set(expected), "extracted source inventory is incomplete")
    require(stat.S_IMODE(root.stat().st_mode) & 0o222 == 0, "extracted source root remains writable")
    _recheck_extracted_source(
        root, snapshots, root_identity, directory_identities
    )
    return snapshots, root_identity, directory_identities


def _recheck_extracted_source(
    root: Path,
    snapshots: Mapping[str, StableSnapshot],
    root_identity: tuple[int, int, int, int, int],
    directory_identities: Mapping[str, tuple[int, int, int, int, int]],
) -> None:
    require(_directory_identity(root, "extracted source root") == root_identity, "extracted source root was replaced")
    observed_files: set[str] = set()
    observed_directories = {"."}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"extracted source gained a symlink: {relative}")
        if path.is_dir():
            observed_directories.add(relative)
            require(
                directory_identities.get(relative)
                == _directory_identity(path, f"source directory {relative}"),
                f"extracted source directory changed: {relative}",
            )
            require(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0, f"extracted source directory became writable: {relative}")
        else:
            require(relative in snapshots, f"extracted source gained an unbound file: {relative}")
            recheck_snapshot(snapshots[relative], f"extracted source {relative}")
            require(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0, f"extracted source file became writable: {relative}")
            observed_files.add(relative)
    require(observed_files == set(snapshots), "extracted source file inventory changed")
    require(observed_directories == set(directory_identities), "extracted source directory inventory changed")
    require(_directory_identity(root, "extracted source root") == root_identity, "extracted source changed during final recheck")


def verify_remote_package_source(
    run_root: Path, expected_manifest_sha256: str
) -> RemoteSourceVerification:
    root = run_root.resolve(strict=True)
    package_root = root / "package"
    source_root = root / "source"
    (
        _,
        binding,
        package_snapshots,
        package_root_identity,
        package_expected_names,
    ) = _load_package(package_root, expected_manifest_sha256, remote_only=True)
    (
        source_snapshots,
        source_root_identity,
        source_directory_identities,
    ) = _verify_extracted_source(source_root, binding)
    return RemoteSourceVerification(
        package_root=package_root,
        package_snapshots=package_snapshots,
        package_root_identity=package_root_identity,
        package_expected_names=package_expected_names,
        source_root=source_root,
        source_snapshots=source_snapshots,
        source_root_identity=source_root_identity,
        source_directory_identities=source_directory_identities,
    )


def recheck_remote_package_source(value: RemoteSourceVerification) -> None:
    _recheck_snapshot_set(
        value.package_root,
        value.package_root_identity,
        value.package_snapshots,
        value.package_expected_names,
        "remote package directory",
    )
    _recheck_extracted_source(
        value.source_root,
        value.source_snapshots,
        value.source_root_identity,
        value.source_directory_identities,
    )


def _load_recipe(
    source_root: Path,
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    source_snapshots: Mapping[str, StableSnapshot],
) -> tuple[Mapping[str, Any], StableSnapshot]:
    recipe_record = binding.get("recipe")
    require(isinstance(recipe_record, Mapping), "recipe binding is missing")
    recipe_relative = safe_relative_path(recipe_record.get("path"), "recipe path")
    recipe_path = resolve_inside(source_root, recipe_relative)
    recipe_snapshot = source_snapshots.get(str(recipe_relative))
    require(recipe_snapshot is not None and recipe_snapshot.path == recipe_path, "sealed recipe snapshot is missing")
    require(recipe_snapshot.sha256 == recipe_record.get("sha256"), "sealed recipe digest mismatch")
    recipe = load_canonical_json_snapshot(recipe_snapshot, "sealed recipe")
    require(recipe.get("format") == RECIPE_FORMAT and recipe.get("version") == 1, "unsupported sealed recipe")
    require(recipe.get("packageId") == manifest.get("packageId"), "recipe/package identity mismatch")
    require(
        sha256_bytes(canonical_json_bytes(recipe.get("runContract")))
        == binding.get("runContractSha256"),
        "sealed run contract digest mismatch",
    )
    require(
        sha256_bytes(canonical_json_bytes(recipe.get("screening")))
        == binding.get("screeningContractSha256"),
        "sealed screening contract digest mismatch",
    )
    return recipe, recipe_snapshot


def _load_sealed_evaluator(
    source_root: Path,
    recipe: Mapping[str, Any],
    source_snapshots: Mapping[str, StableSnapshot],
):
    evaluator_relative = safe_relative_path(recipe.get("screening", {}).get("evaluatorPath") if isinstance(recipe.get("screening"), Mapping) else None, "screening evaluator path")
    evaluator_path = resolve_inside(source_root, evaluator_relative)
    evaluator_snapshot = source_snapshots.get(str(evaluator_relative))
    require(evaluator_snapshot is not None and evaluator_snapshot.path == evaluator_path, "sealed evaluator snapshot is missing")
    module_name = f"dalmuti_v4_sealed_evaluator_{evaluator_snapshot.sha256[:16]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(evaluator_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(evaluator_path.parent))
    try:
        code = compile(evaluator_snapshot.payload, str(evaluator_path), "exec")
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        try:
            sys.path.remove(str(evaluator_path.parent))
        except ValueError:  # pragma: no cover - defensive against imported code
            pass
    return module


def _decision_audit_is_pure_actor(value: object, *, include_player_counts: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    overall = value.get("overall")
    by_act = value.get("byAct")
    if not isinstance(overall, Mapping) or not isinstance(by_act, list) or len(by_act) != 5:
        return False
    groups: list[Mapping[str, Any]] = [overall]
    groups.extend(item for item in by_act if isinstance(item, Mapping))
    if len(groups) != 6:
        return False
    if include_player_counts:
        by_player = value.get("byPlayerCount")
        if not isinstance(by_player, list) or [item.get("playerCount") for item in by_player if isinstance(item, Mapping)] != list(PLAYER_COUNTS):
            return False
        groups.extend(item for item in by_player if isinstance(item, Mapping))
    return all(
        isinstance(item.get("candidateDecisions"), int)
        and not isinstance(item.get("candidateDecisions"), bool)
        and item.get("candidateDecisions", 0) > 0
        and item.get("actorDecisions") == item.get("candidateDecisions")
        and item.get("fallbackDecisions") == 0
        and item.get("actorRate") == 1.0
        and item.get("fallbackRate") == 0.0
        for item in groups
    )


def _canonical_policy_numerics_contract() -> dict[str, object]:
    return {
        "actorForwardDtype": "torch.float32",
        "contract": "fp32-mha-slowpath-math-sdp-v1",
        "contractSha256": (
            "a08de79f95df089fb5c525bb12a14f0fa28985d294f9fa3b2942e5db46df1ca3"
        ),
        "cudaMatmulTf32Allowed": False,
        "cudnnBenchmark": False,
        "cudnnDeterministic": True,
        "cudnnSdpEnabled": False,
        "cudnnTf32Allowed": False,
        "deterministicAlgorithms": True,
        "flashSdpEnabled": False,
        "mathSdpEnabled": True,
        "memoryEfficientSdpEnabled": False,
        "mhaFastpathEnabled": False,
        "requiredCudaCublasWorkspaceConfig": ":4096:8",
        "version": 1,
    }


def _validate_policy_numerics(value: object, label: str) -> Mapping[str, Any]:
    expected = _canonical_policy_numerics_contract()
    require(
        isinstance(value, Mapping)
        and set(value) == V4_POLICY_NUMERICS_FIELDS
        and canonical_json_bytes(dict(value)) == canonical_json_bytes(expected),
        f"{label} is missing, incomplete, or non-canonical",
    )
    return dict(value)


def _recipe_policy_numerics(recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    run_contract = recipe.get("runContract")
    require(isinstance(run_contract, Mapping), "sealed run contract is missing")
    return _validate_policy_numerics(
        run_contract.get("policyNumerics"), "sealed policy numerics contract"
    )


def verify_screening(
    package_dir: Path,
    expected_manifest_sha256: str,
    source_root: Path,
    report_path: Path,
    candidate_dir: Path,
) -> Mapping[str, Any]:
    (
        manifest,
        binding,
        package_snapshots,
        package_root_identity,
        package_expected_names,
    ) = _load_package(package_dir, expected_manifest_sha256, remote_only=True)
    source_root = source_root.resolve(strict=True)
    (
        source_snapshots,
        source_root_identity,
        source_directory_identities,
    ) = _verify_extracted_source(source_root, binding)
    recipe, recipe_snapshot = _load_recipe(
        source_root, manifest, binding, source_snapshots
    )
    screening = recipe.get("screening")
    require(isinstance(screening, Mapping), "screening recipe is missing")
    report_snapshot, report_sidecar_snapshot = snapshot_with_sidecar(report_path)
    report_digest = report_snapshot.sha256
    report = load_canonical_json_snapshot(report_snapshot, "screening report")
    evaluator = _load_sealed_evaluator(source_root, recipe, source_snapshots)
    validator = getattr(evaluator, "validate_benchmark_report", None)
    require(callable(validator), "sealed evaluator has no benchmark validator")
    validator(report, expected_mode="screening")

    candidate_root = candidate_dir.resolve(strict=True)
    actor_path = candidate_root / "actor.pt"
    manifest_path = candidate_root / "manifest.json"
    candidate_root_identity = _directory_identity(candidate_root, "candidate directory")
    actor_snapshot, actor_sidecar_snapshot = snapshot_with_sidecar(actor_path)
    manifest_snapshot, manifest_sidecar_snapshot = snapshot_with_sidecar(manifest_path)
    candidate_snapshots = {
        "actor.pt": actor_snapshot,
        "actor.pt.sha256": actor_sidecar_snapshot,
        "manifest.json": manifest_snapshot,
        "manifest.json.sha256": manifest_sidecar_snapshot,
    }
    require({path.name for path in candidate_root.iterdir()} == set(candidate_snapshots), "candidate inventory is non-canonical")
    actor_digest = actor_snapshot.sha256
    candidate_manifest_digest = manifest_snapshot.sha256
    candidate_manifest = load_canonical_json_snapshot(manifest_snapshot, "candidate manifest")
    require(candidate_manifest.get("files", {}).get("actor.pt", {}).get("sha256") == actor_digest, "candidate manifest does not bind Actor")

    player_counts = screening.get("playerCounts")
    matches = screening.get("matchesPerPlayerCount")
    acts = screening.get("actsPerMatch")
    resamples = screening.get("bootstrapResamples")
    require(player_counts == list(PLAYER_COUNTS), "screening recipe must cover p4-p10 exactly")
    require(isinstance(matches, int) and not isinstance(matches, bool) and matches > 0, "invalid screening match count")
    require(acts == 5, "screening must use five acts")
    require(isinstance(resamples, int) and not isinstance(resamples, bool) and resamples >= 10_000, "screening bootstrap is incomplete")
    require(report.get("playerCounts") == list(PLAYER_COUNTS), "screening p4-p10 coverage is incomplete")
    require(report.get("actsPerMatch") == acts, "screening act count mismatch")
    require(report.get("seed") == screening.get("baseSeed"), "screening base seed mismatch")
    seed_family = report.get("seedFamily")
    require(isinstance(seed_family, Mapping) and seed_family.get("id") == screening.get("familyId") and seed_family.get("mode") == "screening", "screening family mismatch")
    expected_counts = {str(player_count): matches for player_count in PLAYER_COUNTS}
    require(report.get("matchCountsByPlayerCount") == expected_counts, "screening match counts are incomplete")
    bindings = report.get("bindings")
    require(isinstance(bindings, Mapping), "screening bindings are missing")
    require(bindings.get("modelSha256") == actor_digest and report.get("modelSha256") == actor_digest, "screening Actor binding mismatch")
    require(bindings.get("artifactSha256") == candidate_manifest_digest, "screening candidate manifest binding mismatch")
    require(bindings.get("normalBaselineSha256") == screening.get("normalBaselineSha256"), "screening Normal binding mismatch")
    require(bindings.get("observationSchemaSha256") == screening.get("observationSchemaSha256"), "screening observation binding mismatch")
    evidence = report.get("bindingEvidence")
    require(isinstance(evidence, Mapping) and evidence.get("actualFilesVerified") is True and evidence.get("actorModelSha256") == actor_digest and evidence.get("actorBundleArtifactSha256") == candidate_manifest_digest, "screening did not verify actual candidate files")
    policy = report.get("candidatePolicy")
    require(isinstance(policy, Mapping) and policy.get("actorCount") == 1, "screening must use one Actor")
    expected_policy_numerics = _recipe_policy_numerics(recipe)
    report_policy_numerics = _validate_policy_numerics(
        policy.get("policyNumerics"), "screening policy numerics"
    )
    require(
        canonical_json_bytes(dict(report_policy_numerics))
        == canonical_json_bytes(dict(expected_policy_numerics)),
        "screening policy numerics do not match the sealed run contract",
    )
    require(policy.get("bundleActorSha256s") == [actor_digest] and policy.get("bundleManifestSha256s") == [candidate_manifest_digest] and policy.get("bundleArtifactSha256") == candidate_manifest_digest, "screening bundle inventory mismatch")
    routing = policy.get("routing")
    require(isinstance(routing, Mapping) and routing.get("mode") == "pure-actor" and routing.get("runtimeErrorFallback") is False, "screening is not pure Actor")
    require(policy.get("compileAutomaticFallback") is False, "screening permits automatic fallback")
    require(_decision_audit_is_pure_actor(report.get("candidateDecisionAudit"), include_player_counts=True), "screening root decision audit contains fallback")

    results = report.get("results")
    require(isinstance(results, list) and len(results) == len(PLAYER_COUNTS), "screening results are incomplete")
    for player_count, result in zip(PLAYER_COUNTS, results):
        require(isinstance(result, Mapping) and result.get("playerCount") == player_count, f"p{player_count} result is missing")
        require(result.get("matches") == matches and result.get("actsPerMatch") == acts, f"p{player_count} match/act coverage is incomplete")
        inference = result.get("meanChipDifferenceInference")
        interval = result.get("meanChipDifference95")
        require(isinstance(inference, Mapping) and isinstance(interval, Mapping), f"p{player_count} clustered inference is missing")
        require(inference.get("unit") == "seed-matched-match" and inference.get("method") == "deterministic-percentile-bootstrap" and inference.get("clusters") == matches and inference.get("resamples") == resamples, f"p{player_count} bootstrap contract mismatch")
        require(inference.get("mean") == result.get("meanChipDifference") and inference.get("low") == interval.get("low") and inference.get("high") == interval.get("high"), f"p{player_count} bootstrap statistics disagree")
        clusters = result.get("matchClusters")
        require(isinstance(clusters, Mapping) and clusters.get("unit") == "seed-matched-match" and clusters.get("count") == matches and SHA256_RE.fullmatch(str(clusters.get("sha256"))) is not None, f"p{player_count} cluster evidence is incomplete")
        require(_decision_audit_is_pure_actor(result.get("candidateDecisionAudit"), include_player_counts=False), f"p{player_count} decision audit contains fallback")
    require(report.get("deploymentTriggered") is False, "screening attempted deployment")
    recheck_snapshot(report_snapshot, "screening report")
    recheck_snapshot(report_sidecar_snapshot, "screening report sidecar")
    _recheck_snapshot_set(
        candidate_root,
        candidate_root_identity,
        candidate_snapshots,
        set(candidate_snapshots),
        "candidate directory",
    )
    recheck_snapshot(recipe_snapshot, "sealed recipe")
    _recheck_extracted_source(
        source_root,
        source_snapshots,
        source_root_identity,
        source_directory_identities,
    )
    _recheck_snapshot_set(
        package_dir.resolve(strict=True),
        package_root_identity,
        package_snapshots,
        package_expected_names,
        "package directory",
    )
    return {
        "format": "dalmuti-v4-mixed-screening-verification",
        "version": 1,
        "packageId": manifest["packageId"],
        "screeningSha256": report_digest,
        "candidateActorSha256": actor_digest,
        "candidateManifestSha256": candidate_manifest_digest,
        "playerCounts": list(PLAYER_COUNTS),
        "matchesPerPlayerCount": matches,
        "bootstrapResamples": resamples,
        "pureActor": True,
        "passed": True,
    }


def _relative_run_file(run_root: Path, path: Path) -> str:
    root = run_root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    require(root in resolved.parents, "run artifact is outside the run directory")
    return resolved.relative_to(root).as_posix()


def _screening_promotion_snapshots(
    screening_report: Path, promotion_report: Path
) -> tuple[dict[str, StableSnapshot], str, str]:
    """Bind the semantic promotion decision to the exact screened bytes."""

    screening_snapshot, screening_sidecar = snapshot_with_sidecar(screening_report)
    promotion_snapshot, promotion_sidecar = snapshot_with_sidecar(promotion_report)
    promotion = load_canonical_json_snapshot(
        promotion_snapshot, "promotion gate report"
    )
    screening = load_canonical_json_snapshot(screening_snapshot, "screening report")
    require(
        set(promotion)
        == {
            "allPlayerCountsPassed",
            "format",
            "gates",
            "passed",
            "perPlayerCount",
            "screeningReportSha256",
            "version",
        }
        and promotion.get("format") == "dalmuti-v4-mixed-promotion-gates"
        and promotion.get("version") == 1
        and promotion.get("passed") is True
        and promotion.get("allPlayerCountsPassed") is True,
        "promotion gate report is not a passing canonical decision",
    )
    require(
        promotion.get("screeningReportSha256") == screening_snapshot.sha256,
        "promotion gate report is bound to different screening bytes",
    )
    require(
        promotion.get("gates")
        == {
            "minimumClustered95LowerBound": 0.15,
            "minimumMeanChipDifferencePerAct": 0.25,
            "minimumPairwiseBeforeNormal": 0.55,
        },
        "promotion gate thresholds drifted",
    )
    per_player = promotion.get("perPlayerCount")
    require(
        isinstance(per_player, Mapping)
        and set(per_player) == {str(value) for value in PLAYER_COUNTS}
        and all(
            isinstance(per_player[str(value)], Mapping)
            and per_player[str(value)].get("passed") is True
            for value in PLAYER_COUNTS
        ),
        "promotion gate report does not pass every player count",
    )
    require(
        screening.get("format") == "dalmuti-model-benchmark"
        and screening.get("evaluationMode") == "screening"
        and screening.get("playerCounts") == list(PLAYER_COUNTS)
        and screening.get("actsPerMatch") == 5
        and screening.get("matchCountsByPlayerCount")
        == {str(value): 60 for value in PLAYER_COUNTS},
        "sealed screening design drifted",
    )
    require(
        _decision_audit_is_pure_actor(
            screening.get("candidateDecisionAudit"), include_player_counts=True
        ),
        "sealed screening decision audit is not pure Actor",
    )
    results = screening.get("results")
    require(
        isinstance(results, list) and len(results) == len(PLAYER_COUNTS),
        "sealed screening lacks p4-p10 results",
    )
    for expected_player, result in zip(PLAYER_COUNTS, results):
        require(isinstance(result, Mapping), "sealed screening result is invalid")
        inference = result.get("meanChipDifferenceInference")
        pairwise = result.get("pairwiseCandidateBeforeNormal")
        clusters = result.get("matchClusters")
        mean = result.get("meanChipDifference")
        lower = inference.get("low") if isinstance(inference, Mapping) else None
        rate = pairwise.get("rate") if isinstance(pairwise, Mapping) else None
        require(
            result.get("playerCount") == expected_player
            and result.get("matches") == 60
            and result.get("actsPerMatch") == 5
            and isinstance(clusters, Mapping)
            and clusters.get("count") == 60
            and isinstance(inference, Mapping)
            and inference.get("clusters") == 60
            and inference.get("resamples") == 10_000
            and inference.get("unit") == "seed-matched-match"
            and _decision_audit_is_pure_actor(
                result.get("candidateDecisionAudit"), include_player_counts=False
            ),
            f"sealed screening p{expected_player} design drifted",
        )
        require(
            all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (mean, lower, rate)
            ),
            f"sealed screening p{expected_player} metric is invalid",
        )
        promotion_row = per_player[str(expected_player)]
        require(
            promotion_row
            == {
                "clustered95LowerBound": float(lower),
                "meanChipDifferencePerAct": float(mean),
                "pairwiseBeforeNormal": float(rate),
                "passed": True,
            },
            f"promotion p{expected_player} metrics do not match screening",
        )
        require(
            float(mean) >= 0.25
            and float(lower) >= 0.15
            and float(rate) >= 0.55,
            f"sealed screening p{expected_player} does not pass promotion gates",
        )
    run_root = screening_report.resolve(strict=True).parent.parent
    candidate_root = (
        run_root
        / "training"
        / "train-seed-610000001-run-001"
        / "candidate"
    )
    candidate_actor, candidate_actor_sidecar = snapshot_with_sidecar(
        candidate_root / "actor.pt"
    )
    candidate_manifest, candidate_manifest_sidecar = snapshot_with_sidecar(
        candidate_root / "manifest.json"
    )
    candidate_manifest_value = load_canonical_json_snapshot(
        candidate_manifest, "sealed candidate manifest"
    )
    require(
        candidate_manifest_value.get("files", {}).get("actor.pt", {}).get("sha256")
        == candidate_actor.sha256,
        "sealed candidate manifest does not bind Actor",
    )
    bindings = screening.get("bindings")
    require(
        isinstance(bindings, Mapping)
        and bindings.get("modelSha256") == candidate_actor.sha256
        and bindings.get("artifactSha256") == candidate_manifest.sha256,
        "sealed screening does not bind the current candidate",
    )
    hard_gate_path = run_root / "training" / "epoch-0001-hard-gates.json"
    hard_gate_snapshot, hard_gate_sidecar = snapshot_with_sidecar(hard_gate_path)
    hard_gate = load_canonical_json_snapshot(
        hard_gate_snapshot, "training hard-gate report"
    )
    require(
        hard_gate.get("format") == "dalmuti-v4-mixed-training-hard-gates"
        and hard_gate.get("version") == 1
        and hard_gate.get("passed") is True
        and hard_gate.get("candidateActorSha256") == candidate_actor.sha256
        and hard_gate.get("candidateManifestSha256") == candidate_manifest.sha256,
        "training hard gates do not bind the screened candidate",
    )
    snapshots = {
        "screening report": screening_snapshot,
        "screening report sidecar": screening_sidecar,
        "promotion gate report": promotion_snapshot,
        "promotion gate report sidecar": promotion_sidecar,
        "candidate actor": candidate_actor,
        "candidate actor sidecar": candidate_actor_sidecar,
        "candidate manifest": candidate_manifest,
        "candidate manifest sidecar": candidate_manifest_sidecar,
        "training hard-gate report": hard_gate_snapshot,
        "training hard-gate report sidecar": hard_gate_sidecar,
    }
    for label, snapshot in snapshots.items():
        recheck_snapshot(snapshot, label)
    return snapshots, screening_snapshot.sha256, promotion_snapshot.sha256


def _required_npz_family(path: Path, label: str) -> dict[str, StableSnapshot]:
    snapshots: dict[str, StableSnapshot] = {}
    for member in (
        path,
        Path(f"{path}.sha256"),
        Path(f"{path}.metadata.json"),
        Path(f"{path}.metadata.json.sha256"),
    ):
        snapshot = stable_snapshot(member, f"{label} {member.name}")
        snapshots[str(member)] = snapshot
    payload = snapshots[str(path)]
    payload_sidecar = snapshots[f"{path}.sha256"]
    metadata = snapshots[f"{path}.metadata.json"]
    metadata_sidecar = snapshots[f"{path}.metadata.json.sha256"]
    require(
        payload_sidecar.payload == sidecar_bytes(payload.sha256, path.name),
        f"{label} payload sidecar drifted",
    )
    require(
        metadata_sidecar.payload
        == sidecar_bytes(metadata.sha256, Path(f"{path}.metadata.json").name),
        f"{label} metadata sidecar drifted",
    )
    load_canonical_json_snapshot(metadata, f"{label} metadata")
    return snapshots


def _git_read(repository: Path, arguments: Sequence[str], label: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(
        completed.returncode == 0,
        f"{label} failed: {completed.stderr.strip()[:1024]}",
    )
    return completed.stdout


def _verify_remote_frozen_baseline(root: Path) -> dict[str, StableSnapshot]:
    bundle_root = root / "baseline-bundle"
    require(
        bundle_root.is_dir()
        and not bundle_root.is_symlink()
        and {path.name for path in bundle_root.iterdir()}
        == {
            FROZEN_BASELINE_BUNDLE_NAME,
            f"{FROZEN_BASELINE_BUNDLE_NAME}.sha256",
        },
        "frozen baseline bundle inventory drifted",
    )
    bundle = bundle_root / FROZEN_BASELINE_BUNDLE_NAME
    bundle_snapshot, bundle_sidecar_snapshot = snapshot_with_sidecar(
        bundle, FROZEN_BASELINE_BUNDLE_SHA256
    )
    repository = root / "frozen-baseline"
    require(
        repository.is_dir() and not repository.is_symlink(),
        "frozen baseline repository is invalid",
    )
    head = _git_read(
        repository, ("rev-parse", "--verify", "HEAD^{commit}"), "baseline HEAD"
    ).strip()
    require(head == FROZEN_BASELINE_COMMIT, "frozen baseline commit drifted")
    _git_read(
        repository,
        ("bundle", "verify", str(bundle.resolve(strict=True))),
        "baseline bundle verification",
    )
    normal = stable_snapshot(
        repository / "lib" / "bot-strategy.ts", "frozen Normal source"
    )
    observation = stable_snapshot(
        root / "source" / "training" / "v4-public-history.ts",
        "sealed observation source",
    )
    require(normal.sha256 == FROZEN_NORMAL_SHA256, "frozen Normal source drifted")
    require(
        observation.sha256 == FROZEN_OBSERVATION_SHA256,
        "sealed observation source drifted",
    )
    status = _git_read(
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        "baseline worktree status",
    )
    require(status == "", "frozen baseline worktree is dirty")
    snapshots = {
        "frozen baseline bundle": bundle_snapshot,
        "frozen baseline bundle sidecar": bundle_sidecar_snapshot,
        "frozen Normal source": normal,
        "sealed observation source": observation,
    }
    for label, snapshot in snapshots.items():
        recheck_snapshot(snapshot, label)
    require(
        _git_read(
            repository,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            "baseline HEAD recheck",
        ).strip()
        == FROZEN_BASELINE_COMMIT
        and _git_read(
            repository,
            ("status", "--porcelain=v1", "--untracked-files=all"),
            "baseline worktree recheck",
        )
        == "",
        "frozen baseline changed during verification",
    )
    return snapshots


def _finite_nonnegative(value: object, label: str) -> float:
    require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0,
        f"{label} must be a finite nonnegative number",
    )
    return float(value)


def _replay_player_count_integers(
    value: object, label: str, *, positive: bool
) -> dict[str, int]:
    expected_keys = {str(player_count) for player_count in PLAYER_COUNTS}
    require(
        isinstance(value, Mapping) and set(value) == expected_keys,
        f"{label} must cover p4-p10 exactly",
    )
    result: dict[str, int] = {}
    for key in sorted(expected_keys, key=int):
        item = value[key]
        require(
            isinstance(item, int)
            and not isinstance(item, bool)
            and (item > 0 if positive else item >= 0),
            f"{label} contains an invalid p{key} count",
        )
        result[key] = item
    return result


def _replay_player_count_numbers(
    value: object, label: str, *, positive: bool
) -> dict[str, float]:
    expected_keys = {str(player_count) for player_count in PLAYER_COUNTS}
    require(
        isinstance(value, Mapping) and set(value) == expected_keys,
        f"{label} must cover p4-p10 exactly",
    )
    result: dict[str, float] = {}
    for key in sorted(expected_keys, key=int):
        item = _finite_nonnegative(value[key], f"{label} p{key}")
        require(not positive or item > 0.0, f"{label} p{key} must be positive")
        result[key] = item
    return result


def _validate_pretraining_replay(
    report: Mapping[str, Any],
    recipe: Mapping[str, Any],
    merged_payload: StableSnapshot,
    merged_metadata: Mapping[str, Any],
) -> None:
    require(
        set(report) == V4_REPLAY_REPORT_FIELDS
        and report.get("format") == "dalmuti-v4-mixed-pretraining-replay"
        and report.get("version") == 1
        and report.get("passed") is True
        and report.get("device") == "cuda",
        "pretraining replay report header is non-canonical or failed",
    )
    expected_policy_numerics = _recipe_policy_numerics(recipe)
    replay_policy_numerics = _validate_policy_numerics(
        report.get("policyNumerics"), "pretraining replay policy numerics"
    )
    require(
        canonical_json_bytes(dict(replay_policy_numerics))
        == canonical_json_bytes(dict(expected_policy_numerics)),
        "pretraining replay policy numerics do not match the sealed run contract",
    )

    run_contract = recipe.get("runContract")
    assert isinstance(run_contract, Mapping)
    behavior = run_contract.get("behaviorActor")
    require(isinstance(behavior, Mapping), "sealed behavior Actor contract is missing")
    expected_actor_sha = require_sha256(
        behavior.get("actorSha256"), "sealed behavior Actor SHA-256"
    )
    expected_manifest_sha = require_sha256(
        behavior.get("manifestSha256"), "sealed behavior manifest SHA-256"
    )
    require(
        report.get("actorSha256") == expected_actor_sha
        and report.get("manifestSha256") == expected_manifest_sha,
        "pretraining replay does not bind the sealed behavior Actor bundle",
    )

    dataset_fingerprint = require_sha256(
        merged_metadata.get("fingerprint"), "merged dataset fingerprint"
    )
    require(
        report.get("datasetSha256") == merged_payload.sha256
        and report.get("datasetFingerprint") == dataset_fingerprint,
        "pretraining replay does not bind the exact merged dataset",
    )
    loss = merged_metadata.get("lossEligibility")
    require(isinstance(loss, Mapping), "merged dataset lacks loss eligibility")
    plans = loss.get("fixedCollectionPlans")
    require(
        isinstance(plans, list) and len(plans) == 1 and isinstance(plans[0], Mapping),
        "merged dataset must bind one fixed collection plan",
    )
    plan_sha = require_sha256(
        plans[0].get("canonicalSha256"), "merged fixed collection plan SHA-256"
    )
    require(
        report.get("fixedCollectionPlanSha256") == plan_sha,
        "pretraining replay fixed collection plan binding drifted",
    )

    audit = report.get("audit")
    require(
        isinstance(audit, Mapping)
        and set(audit) == V4_REPLAY_AUDIT_FIELDS
        and audit.get("version") == 2
        and audit.get("passed") is True
        and audit.get("auditBatchSize") == 64
        and audit.get("actorMode") == "eval"
        and audit.get("actorForwardDtype") == "torch.float32"
        and audit.get("actorAutocastEnabled") is False
        and audit.get("storedOldActionLogProbabilityDtype") == "torch.float32",
        "pretraining replay audit is missing, incomplete, or non-canonical",
    )
    absolute_tolerance = _finite_nonnegative(
        audit.get("absoluteTolerance"), "pretraining replay absolute tolerance"
    )
    maximum_error = _finite_nonnegative(
        audit.get("maximumAbsoluteLogProbabilityError"),
        "pretraining replay maximum error",
    )
    mean_error = _finite_nonnegative(
        audit.get("meanAbsoluteLogProbabilityError"),
        "pretraining replay mean error",
    )
    forced_maximum = _finite_nonnegative(
        audit.get("forcedMaximumAbsoluteLogProbabilityError"),
        "pretraining replay forced maximum error",
    )
    require(
        absolute_tolerance == V4_REPLAY_TOLERANCE
        and maximum_error <= V4_REPLAY_TOLERANCE
        and mean_error <= maximum_error + 1.0e-15
        and forced_maximum <= maximum_error + 1.0e-15,
        "pretraining replay exceeded the immutable 2e-5 error contract",
    )
    forced_counts = _replay_player_count_integers(
        audit.get("forcedSingletonRowsByPlayerCount"),
        "pretraining replay forced rows",
        positive=False,
    )
    nonforced_counts = _replay_player_count_integers(
        audit.get("nonforcedRowsByPlayerCount"),
        "pretraining replay nonforced rows",
        positive=True,
    )
    weight_masses = _replay_player_count_numbers(
        audit.get("nonforcedWeightMassByPlayerCount"),
        "pretraining replay nonforced weight mass",
        positive=True,
    )
    entropies = _replay_player_count_numbers(
        audit.get("nonforcedEntropyByPlayerCount"),
        "pretraining replay nonforced entropy",
        positive=False,
    )
    ppo_rows = audit.get("ppoEligibleRowCount")
    nonforced_rows = audit.get("effectiveNonforcedPpoRowCount")
    forced_rows = audit.get("forcedSingletonPpoRowCount")
    require(
        isinstance(ppo_rows, int)
        and not isinstance(ppo_rows, bool)
        and ppo_rows > 0
        and isinstance(nonforced_rows, int)
        and not isinstance(nonforced_rows, bool)
        and nonforced_rows == sum(nonforced_counts.values())
        and isinstance(forced_rows, int)
        and not isinstance(forced_rows, bool)
        and forced_rows == sum(forced_counts.values())
        and ppo_rows == nonforced_rows + forced_rows,
        "pretraining replay audit row counts are inconsistent",
    )
    total_mass = _finite_nonnegative(
        audit.get("nonforcedTotalWeightMass"),
        "pretraining replay total nonforced weight mass",
    )
    balanced_entropy = _finite_nonnegative(
        audit.get("nonforcedBalancedEntropy"),
        "pretraining replay balanced entropy",
    )
    require(
        total_mass > 0.0
        and math.isclose(
            total_mass,
            sum(weight_masses.values()),
            rel_tol=2.0e-12,
            abs_tol=2.0e-10,
        ),
        "pretraining replay weight-mass totals drifted",
    )
    reconstructed_entropy = sum(
        entropies[key] * weight_masses[key] for key in weight_masses
    ) / total_mass
    require(
        math.isclose(
            balanced_entropy,
            reconstructed_entropy,
            rel_tol=2.0e-10,
            abs_tol=2.0e-12,
        ),
        "pretraining replay balanced entropy drifted",
    )

    strata = report.get("strata")
    require(
        isinstance(strata, Mapping)
        and set(strata) == {"byPlayerCountShardAndBackend"},
        "pretraining replay strata are missing or non-canonical",
    )
    rows = strata.get("byPlayerCountShardAndBackend")
    expected_keys = {
        (player_count, shard_index, V4_MIXED_BACKEND_MAP[shard_index])
        for player_count in PLAYER_COUNTS
        for shard_index in range(len(V4_MIXED_BACKEND_MAP))
    }
    require(
        isinstance(rows, list) and len(rows) == len(expected_keys),
        "pretraining replay lacks complete p4-p10 CPU/CUDA strata",
    )
    observed_keys: set[tuple[int, int, str]] = set()
    counts_by_player = {str(player_count): 0 for player_count in PLAYER_COUNTS}
    stratum_count = 0
    stratum_maximum = 0.0
    for row in rows:
        require(
            isinstance(row, Mapping) and set(row) == V4_REPLAY_STRATUM_FIELDS,
            "pretraining replay contains a non-canonical stratum",
        )
        player_count = row.get("playerCount")
        shard_index = row.get("shardIndex")
        backend = row.get("backend")
        count = row.get("count")
        require(
            isinstance(player_count, int)
            and not isinstance(player_count, bool)
            and isinstance(shard_index, int)
            and not isinstance(shard_index, bool)
            and isinstance(backend, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0,
            "pretraining replay contains an invalid stratum identity or count",
        )
        key = (player_count, shard_index, backend)
        require(
            key in expected_keys and key not in observed_keys,
            "pretraining replay stratum identity/backend coverage drifted",
        )
        observed_keys.add(key)
        row_maximum = _finite_nonnegative(
            row.get("maximumAbsoluteLogProbabilityError"),
            "pretraining replay stratum maximum error",
        )
        row_mean = _finite_nonnegative(
            row.get("meanAbsoluteLogProbabilityError"),
            "pretraining replay stratum mean error",
        )
        require(
            row_maximum <= V4_REPLAY_TOLERANCE
            and row_mean <= row_maximum + 1.0e-15,
            "a pretraining replay stratum exceeded the immutable 2e-5 contract",
        )
        counts_by_player[str(player_count)] += count
        stratum_count += count
        stratum_maximum = max(stratum_maximum, row_maximum)
    require(
        observed_keys == expected_keys
        and stratum_count == ppo_rows
        and counts_by_player
        == {
            key: forced_counts[key] + nonforced_counts[key]
            for key in counts_by_player
        }
        and math.isclose(
            stratum_maximum, maximum_error, rel_tol=0.0, abs_tol=1.0e-15
        ),
        "pretraining replay strata disagree with the full replay audit",
    )


def _verify_remote_semantic_inventory(root: Path) -> dict[str, StableSnapshot]:
    snapshots: dict[str, StableSnapshot] = {}
    recipe_snapshot = stable_snapshot(
        root / "source" / "gpu-training" / "v4_mixed_execution_recipe.json",
        "remote semantic recipe",
    )
    recipe = load_canonical_json_snapshot(
        recipe_snapshot, "remote semantic recipe"
    )
    _recipe_policy_numerics(recipe)
    snapshots["remote semantic recipe"] = recipe_snapshot
    snapshots.update(_verify_remote_frozen_baseline(root))
    calibration = root / "calibration"
    expected_calibration_names = {
        "backend-comparison.json",
        "backend-comparison.json.sha256",
        "cpu.npz",
        "cpu.npz.sha256",
        "cpu.npz.metadata.json",
        "cpu.npz.metadata.json.sha256",
        "cuda.npz",
        "cuda.npz.sha256",
        "cuda.npz.metadata.json",
        "cuda.npz.metadata.json.sha256",
    }
    require(
        calibration.is_dir()
        and not calibration.is_symlink()
        and {path.name for path in calibration.iterdir()} == expected_calibration_names,
        "remote calibration is not the exact ten-file inventory",
    )
    report, report_sidecar = snapshot_with_sidecar(
        calibration / "backend-comparison.json"
    )
    snapshots["calibration report"] = report
    snapshots["calibration report sidecar"] = report_sidecar
    snapshots.update(_required_npz_family(calibration / "cpu.npz", "calibration CPU"))
    snapshots.update(_required_npz_family(calibration / "cuda.npz", "calibration CUDA"))

    rollouts = root / "rollouts"
    expected_rollout_names = {
        name
        for index in range(14)
        for name in (
            f"shard-{index:02d}.npz",
            f"shard-{index:02d}.npz.sha256",
            f"shard-{index:02d}.npz.metadata.json",
            f"shard-{index:02d}.npz.metadata.json.sha256",
        )
    }
    require(
        rollouts.is_dir()
        and not rollouts.is_symlink()
        and {path.name for path in rollouts.iterdir()} == expected_rollout_names,
        "remote rollout inventory is not fourteen exact four-file shards",
    )
    for index in range(14):
        snapshots.update(
            _required_npz_family(
                rollouts / f"shard-{index:02d}.npz", f"production shard {index:02d}"
            )
        )
    merged = root / "merged"
    require(
        merged.is_dir()
        and not merged.is_symlink()
        and {path.name for path in merged.iterdir()}
        == {
            "production.npz",
            "production.npz.sha256",
            "production.npz.metadata.json",
            "production.npz.metadata.json.sha256",
        },
        "merged production inventory is not the exact four-file family",
    )
    merged_path = merged / "production.npz"
    merged_family = _required_npz_family(merged_path, "merged production")
    snapshots.update(merged_family)
    merged_payload = merged_family[str(merged_path)]
    merged_metadata_snapshot = merged_family[f"{merged_path}.metadata.json"]
    merged_metadata = load_canonical_json_snapshot(
        merged_metadata_snapshot, "merged production metadata"
    )
    replay, replay_sidecar = snapshot_with_sidecar(
        root / "replay" / "pretraining.json"
    )
    replay_report = load_canonical_json_snapshot(replay, "pretraining replay")
    _validate_pretraining_replay(
        replay_report, recipe, merged_payload, merged_metadata
    )
    snapshots["pretraining replay"] = replay
    snapshots["pretraining replay sidecar"] = replay_sidecar
    training_root = root / "training" / "train-seed-610000001-run-001"
    for relative, label in (
        ("result.json", "training result"),
        ("run-manifest.json", "training run manifest"),
    ):
        snapshot = stable_snapshot(training_root / relative, label)
        load_canonical_json_snapshot(snapshot, label)
        snapshots[label] = snapshot
    candidate = training_root / "candidate"
    require(
        candidate.is_dir()
        and not candidate.is_symlink()
        and {path.name for path in candidate.iterdir()}
        == {"actor.pt", "actor.pt.sha256", "manifest.json", "manifest.json.sha256"},
        "trained candidate inventory is not the exact four-file bundle",
    )
    for name in ("actor.pt", "manifest.json"):
        payload, sidecar = snapshot_with_sidecar(candidate / name)
        snapshots[f"candidate {name}"] = payload
        snapshots[f"candidate {name} sidecar"] = sidecar
    hard_gate, hard_gate_sidecar = snapshot_with_sidecar(
        root / "training" / "epoch-0001-hard-gates.json"
    )
    snapshots["training hard gates"] = hard_gate
    snapshots["training hard gates sidecar"] = hard_gate_sidecar
    for name in ("epoch-0001.json", "epoch-0001-promotion-gates.json"):
        payload, sidecar = snapshot_with_sidecar(root / "screening" / name)
        snapshots[f"screening {name}"] = payload
        snapshots[f"screening {name} sidecar"] = sidecar
    for label, snapshot in snapshots.items():
        recheck_snapshot(snapshot, label)
    return snapshots


def _load_sealed_workflow_plan(
    root: Path,
    source_snapshots: Mapping[str, StableSnapshot],
    expected_recipe_sha256: str,
) -> tuple[object, Mapping[str, Any], tuple[object, ...]]:
    recipe_relative = "gpu-training/v4_mixed_execution_recipe.json"
    workflow_relative = "gpu-training/v4_mixed_workflow.py"
    recipe_snapshot = source_snapshots.get(recipe_relative)
    workflow_snapshot = source_snapshots.get(workflow_relative)
    require(
        recipe_snapshot is not None
        and recipe_snapshot.path == root / "source" / recipe_relative
        and recipe_snapshot.sha256 == expected_recipe_sha256,
        "sealed finalization recipe snapshot drifted",
    )
    require(
        workflow_snapshot is not None
        and workflow_snapshot.path == root / "source" / workflow_relative,
        "sealed workflow snapshot is missing",
    )
    recipe = load_canonical_json_snapshot(
        recipe_snapshot, "sealed finalization recipe"
    )
    module_name = f"dalmuti_v4_sealed_workflow_{workflow_snapshot.sha256[:16]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(workflow_snapshot.path)
    module.__package__ = ""
    sys.modules[module_name] = module
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        code = compile(
            workflow_snapshot.payload, str(workflow_snapshot.path), "exec"
        )
        exec(code, module.__dict__)
        module.validate_recipe(recipe)
        phases = tuple(module.build_mixed_phase_plan(recipe))
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    recheck_snapshot(recipe_snapshot, "sealed finalization recipe")
    recheck_snapshot(workflow_snapshot, "sealed workflow")
    return module, recipe, phases


def _local_output_suffix(template: str) -> str | None:
    prefix = "{local_run_directory}/"
    if not template.startswith(prefix):
        return None
    relative = safe_relative_path(
        template[len(prefix) :], "local receipt output suffix"
    )
    return relative.as_posix()


def _verify_local_output_path(value: object, expected_suffix: str) -> None:
    require(
        isinstance(value, str) and value and "\x00" not in value,
        "local receipt output path is invalid",
    )
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    require(
        all(part not in ("", ".", "..") for part in parts)
        and normalized.endswith(f"/{expected_suffix}"),
        "local receipt output path has the wrong canonical suffix",
    )


def _current_remote_output_record(
    root: Path, remote_run: PurePosixPath, expected_path: str
) -> tuple[dict[str, object], list[StableSnapshot]]:
    path = PurePosixPath(expected_path)
    require(
        path.is_absolute()
        and path != remote_run
        and remote_run in path.parents,
        "remote receipt output path escaped its bound run",
    )
    relative = path.relative_to(remote_run)
    current = root.joinpath(*relative.parts)
    require(not current.is_symlink(), "current remote output is a symlink")
    if current.is_file():
        snapshot = stable_snapshot(current, f"current remote output {relative}")
        return (
            {
                "kind": "file",
                "path": expected_path,
                "sha256": snapshot.sha256,
                "size": len(snapshot.payload),
            },
            [snapshot],
        )
    require(current.is_dir(), f"current remote output is missing: {relative}")
    records: list[dict[str, object]] = []
    snapshots: list[StableSnapshot] = []
    total_size = 0
    for child in sorted(
        current.rglob("*"), key=lambda item: item.relative_to(current).as_posix()
    ):
        child_relative = child.relative_to(current).as_posix()
        require(
            not child.is_symlink(),
            f"current remote output directory contains a symlink: {child_relative}",
        )
        if child.is_dir():
            continue
        require(
            child.is_file(),
            f"current remote output directory contains a special file: {child_relative}",
        )
        snapshot = stable_snapshot(
            child, f"current remote output {relative}/{child_relative}"
        )
        records.append(
            {
                "path": child_relative,
                "sha256": snapshot.sha256,
                "size": len(snapshot.payload),
            }
        )
        snapshots.append(snapshot)
        total_size += len(snapshot.payload)
    return (
        {
            "kind": "directory",
            "path": expected_path,
            "sha256": sha256_bytes(canonical_json_bytes(records)),
            "size": total_size,
        },
        snapshots,
    )


def _verify_finalization_counterparts(
    commands: Mapping[str, object], receipts: Mapping[str, Mapping[str, Any]]
) -> None:
    local_commands = {
        command_id
        for command_id, command in commands.items()
        if any(_local_output_suffix(value) is not None for value in command.outputs)
    }
    require(
        local_commands == set(LOCAL_OUTPUT_COUNTERPARTS),
        "local finalization counterpart coverage drifted",
    )
    for local_id, counterparts in LOCAL_OUTPUT_COUNTERPARTS.items():
        local_outputs = receipts[local_id].get("outputs")
        require(
            isinstance(local_outputs, list)
            and len(local_outputs) == len(commands[local_id].outputs)
            and len(local_outputs) == len(counterparts),
            f"local counterpart output count drifted: {local_id}",
        )
        for index, (remote_id, remote_index) in enumerate(counterparts):
            remote_outputs = receipts[remote_id].get("outputs")
            require(
                isinstance(remote_outputs, list) and remote_index < len(remote_outputs),
                f"remote counterpart output index drifted: {remote_id}",
            )
            local_record = local_outputs[index]
            remote_record = remote_outputs[remote_index]
            require(
                isinstance(local_record, Mapping)
                and isinstance(remote_record, Mapping)
                and local_record.get("kind") == remote_record.get("kind")
                and local_record.get("sha256") == remote_record.get("sha256")
                and local_record.get("size") == remote_record.get("size"),
                f"local/remote counterpart bytes drifted: {local_id}[{index}]",
            )


def _verify_finalization_audit(
    root: Path,
    audit_path: Path,
    expected_digest: str,
    *,
    package_manifest_sha256: str,
    recipe_sha256: str,
    runtime_bindings_sha256: str,
    source_snapshots: Mapping[str, StableSnapshot],
) -> tuple[dict[str, StableSnapshot], Mapping[str, Any]]:
    require(
        audit_path.resolve(strict=True)
        == root / "provenance" / "finalization-audit.json",
        "finalization audit path is non-canonical",
    )
    audit_snapshot, audit_sidecar = snapshot_with_sidecar(
        audit_path, expected_digest
    )
    audit = load_canonical_json_snapshot(audit_snapshot, "finalization audit")
    require(
        set(audit)
        == {
            "fixedCollectionPlanSha256",
            "format",
            "packageManifestSha256",
            "passed",
            "recipeSha256",
            "requiredCommands",
            "runNamespace",
            "runtimeBindingsSha256",
            "version",
        }
        and audit.get("format") == "dalmuti-v4-mixed-finalization-audit"
        and audit.get("version") == 1
        and audit.get("passed") is True
        and audit.get("packageManifestSha256") == package_manifest_sha256
        and audit.get("recipeSha256") == recipe_sha256
        and audit.get("runtimeBindingsSha256") == runtime_bindings_sha256
        and audit.get("runNamespace") == RUN_NAMESPACE,
        "finalization audit binding drifted",
    )
    require_sha256(audit.get("fixedCollectionPlanSha256"), "finalization plan SHA-256")
    workflow, _, phases = _load_sealed_workflow_plan(
        root, source_snapshots, recipe_sha256
    )
    ordered_plan = [
        (phase, command)
        for phase in phases
        for command in phase.commands
        if command.command_id in FINALIZATION_COMMAND_IDS
    ]
    require(
        [command.command_id for _, command in ordered_plan]
        == list(FINALIZATION_COMMAND_IDS),
        "sealed workflow finalization command order drifted",
    )
    commands = {command.command_id: command for _, command in ordered_plan}
    command_phases = {
        command.command_id: phase for phase, command in ordered_plan
    }
    runtime_snapshot, runtime_sidecar = snapshot_with_sidecar(
        root / "control" / "runtime-bindings.json", runtime_bindings_sha256
    )
    runtime_bindings = load_canonical_json_snapshot(
        runtime_snapshot, "finalization runtime bindings"
    )
    require(
        set(runtime_bindings)
        == {
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
        and runtime_bindings.get("format")
        == "dalmuti-v4-mixed-remote-runtime-bindings"
        and runtime_bindings.get("version") == 1
        and runtime_bindings.get("runNamespace") == RUN_NAMESPACE
        and runtime_bindings.get("packageManifestSha256")
        == package_manifest_sha256
        and runtime_bindings.get("recipeSha256") == recipe_sha256,
        "finalization runtime binding fields drifted",
    )
    remote_run_value = runtime_bindings.get("runDirectory")
    require(
        isinstance(remote_run_value, str)
        and PurePosixPath(remote_run_value).is_absolute(),
        "bound remote run path is invalid",
    )
    remote_run = PurePosixPath(str(remote_run_value))
    require(
        runtime_bindings.get("sourceRoot") == f"{remote_run}/source"
        and runtime_bindings.get("packageDirectory") == f"{remote_run}/package"
        and runtime_bindings.get("behaviorActorBundle")
        == f"{remote_run}/behavior-actor"
        and runtime_bindings.get("frozenBaselineRepository")
        == f"{remote_run}/frozen-baseline"
        and isinstance(runtime_bindings.get("pythonExecutable"), str)
        and PurePosixPath(str(runtime_bindings["pythonExecutable"])).is_absolute(),
        "finalization runtime paths drifted",
    )
    replacements = {
        "{remote_source_root}": str(runtime_bindings["sourceRoot"]),
        "{remote_run_directory}": str(runtime_bindings["runDirectory"]),
        "{remote_behavior_actor_bundle}": str(
            runtime_bindings["behaviorActorBundle"]
        ),
        "{remote_frozen_baseline_repository}": str(
            runtime_bindings["frozenBaselineRepository"]
        ),
        "{remote_python}": str(runtime_bindings["pythonExecutable"]),
        "{remote_package_directory}": str(runtime_bindings["packageDirectory"]),
        "{package_manifest_sha256}": package_manifest_sha256,
        "{merged_collection_plan_sha256}": str(
            audit["fixedCollectionPlanSha256"]
        ),
    }
    required = audit.get("requiredCommands")
    require(
        isinstance(required, list)
        and [row.get("commandId") for row in required if isinstance(row, Mapping)]
        == list(FINALIZATION_COMMAND_IDS),
        "finalization audit command coverage drifted",
    )
    snapshots = {
        "finalization audit": audit_snapshot,
        "finalization audit sidecar": audit_sidecar,
        "finalization runtime bindings": runtime_snapshot,
        "finalization runtime bindings sidecar": runtime_sidecar,
    }
    receipts: dict[str, Mapping[str, Any]] = {}
    for row in required:
        require(
            isinstance(row, Mapping)
            and set(row)
            == {
                "commandId",
                "commandSpecSha256",
                "completionReceiptSha256",
                "outputs",
                "phaseId",
            },
            "finalization command record fields drifted",
        )
        command_id = safe_leaf(row.get("commandId"), "finalization command ID")
        command = commands.get(command_id)
        phase = command_phases.get(command_id)
        require(command is not None and phase is not None, "audit command is not in sealed workflow")
        expected_spec_sha256 = sha256_bytes(
            canonical_json_bytes(command.to_dict())
        )
        require(
            row.get("phaseId") == phase.phase_id
            and row.get("commandSpecSha256") == expected_spec_sha256,
            f"finalization command spec disagrees with sealed workflow: {command_id}",
        )
        receipt_digest = require_sha256(
            row.get("completionReceiptSha256"), "completion receipt SHA-256"
        )
        receipt_path = root / "control" / "completions" / f"{command_id}.json"
        receipt_snapshot, receipt_sidecar = snapshot_with_sidecar(
            receipt_path, receipt_digest
        )
        receipt = load_canonical_json_snapshot(
            receipt_snapshot, f"completion receipt {command_id}"
        )
        require(
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
            and receipt.get("format") == "dalmuti-v4-mixed-command-completion"
            and receipt.get("version") == 1
            and receipt.get("commandId") == command_id
            and receipt.get("phaseId") == row.get("phaseId")
            and receipt.get("commandSpecSha256") == row.get("commandSpecSha256")
            and receipt.get("host") == command.host
            and receipt.get("outputs") == row.get("outputs")
            and receipt.get("packageManifestSha256") == package_manifest_sha256
            and receipt.get("recipeSha256") == recipe_sha256
            and receipt.get("runtimeBindingsSha256") == runtime_bindings_sha256
            and receipt.get("runNamespace") == RUN_NAMESPACE
            and receipt.get("passed") is True,
            f"completion receipt {command_id} disagrees with finalization audit",
        )
        require_sha256(
            receipt.get("materializedArgvSha256"),
            "completion materialized argv SHA-256",
        )
        if command.host == "remote":
            expected_argv = workflow.materialize_argv(command.argv, replacements)
            require(
                receipt.get("materializedArgvSha256")
                == sha256_bytes(canonical_json_bytes(list(expected_argv))),
                f"remote materialized argv drifted: {command_id}",
            )
        outputs = row.get("outputs")
        require(
            isinstance(outputs, list) and len(outputs) == len(command.outputs),
            "finalization output inventory is invalid",
        )
        for template, output in zip(command.outputs, outputs, strict=True):
            require(
                isinstance(output, Mapping)
                and set(output) == {"kind", "path", "sha256", "size"}
                and output.get("kind") in {"file", "directory"},
                "finalization output record fields drifted",
            )
            require_sha256(output.get("sha256"), "finalization output SHA-256")
            size = output.get("size")
            require(
                isinstance(size, int) and not isinstance(size, bool) and size >= 0,
                "finalization output size is invalid",
            )
            local_suffix = _local_output_suffix(template)
            if local_suffix is not None:
                _verify_local_output_path(output.get("path"), local_suffix)
                continue
            expected_path = workflow.materialize_argv((template,), replacements)[0]
            require(
                output.get("path") == expected_path,
                f"remote finalization output path drifted: {command_id}",
            )
            current_record, current_snapshots = _current_remote_output_record(
                root, remote_run, expected_path
            )
            require(
                dict(output) == current_record,
                f"remote finalization output bytes drifted: {command_id}",
            )
            for index, snapshot in enumerate(current_snapshots):
                snapshots[
                    f"current output {command_id} {len(snapshots)}-{index}"
                ] = snapshot
        snapshots[f"completion receipt {command_id}"] = receipt_snapshot
        snapshots[f"completion receipt {command_id} sidecar"] = receipt_sidecar
        receipts[command_id] = receipt
    _verify_finalization_counterparts(commands, receipts)
    metadata, metadata_sidecar = snapshot_with_sidecar(
        root / "merged" / "production.npz.metadata.json"
    )
    metadata_value = load_canonical_json_snapshot(metadata, "merged production metadata")
    loss_eligibility = metadata_value.get("lossEligibility")
    plans = (
        loss_eligibility.get("fixedCollectionPlans")
        if isinstance(loss_eligibility, Mapping)
        else None
    )
    require(
        isinstance(plans, list)
        and len(plans) == 1
        and isinstance(plans[0], Mapping)
        and plans[0].get("canonicalSha256")
        == audit.get("fixedCollectionPlanSha256"),
        "finalization fixed collection plan binding drifted",
    )
    snapshots["merged production metadata"] = metadata
    snapshots["merged production metadata sidecar"] = metadata_sidecar
    for label, snapshot in snapshots.items():
        recheck_snapshot(snapshot, label)
    return snapshots, audit


def _verify_local_aggregate_remote_copy(
    root: Path,
) -> tuple[dict[str, StableSnapshot], str, str]:
    expected_seal = (
        root / "remote-sealed-run" / "provenance" / "final-files.json"
    )
    expected_seal_sidecar = Path(f"{expected_seal}.sha256")
    nested_seals = sorted(
        path
        for path in root.rglob("final-files.json")
        if path != root / "provenance" / "final-files.json"
    )
    nested_sidecars = sorted(
        path
        for path in root.rglob("final-files.json.sha256")
        if path != root / "provenance" / "final-files.json.sha256"
    )
    require(
        nested_seals == [expected_seal]
        and nested_sidecars == [expected_seal_sidecar],
        "local aggregate requires exactly one canonical remote-sealed-run seal pair",
    )
    nested_seal_snapshot, nested_seal_sidecar_snapshot = snapshot_with_sidecar(
        expected_seal
    )
    nested_seal = load_canonical_json_snapshot(
        nested_seal_snapshot, "nested remote run seal"
    )
    require(
        nested_seal.get("profile") == "remote-semantic",
        "local aggregate nested seal is not remote-semantic",
    )
    nested_seal_digest = verify_run_seal(expected_seal)

    status_root = root / "remote-sealed-run" / "status"
    require(
        status_root.is_dir()
        and not status_root.is_symlink()
        and {path.name for path in status_root.iterdir()}
        == {"999-succeeded.json", "999-succeeded.json.sha256"},
        "local aggregate requires the exact nested success status pair",
    )
    success_path = status_root / "999-succeeded.json"
    success_snapshot, success_sidecar_snapshot = snapshot_with_sidecar(success_path)
    success = load_canonical_json_snapshot(
        success_snapshot, "nested remote success status"
    )
    require(
        set(success)
        == {
            "detail",
            "format",
            "runSealSha256",
            "stage",
            "state",
            "version",
        }
        and success.get("format") == "dalmuti-v4-mixed-stage-status"
        and success.get("version") == 1
        and success.get("stage") == "complete"
        and success.get("state") == "succeeded"
        and isinstance(success.get("detail"), str)
        and len(str(success["detail"])) <= 4096
        and success.get("runSealSha256") == nested_seal_digest,
        "nested remote success status does not bind the canonical remote seal",
    )
    snapshots = {
        "nested remote run seal": nested_seal_snapshot,
        "nested remote run seal sidecar": nested_seal_sidecar_snapshot,
        "nested remote success status": success_snapshot,
        "nested remote success status sidecar": success_sidecar_snapshot,
    }
    for label, snapshot in snapshots.items():
        recheck_snapshot(snapshot, label)
    require(
        verify_run_seal(expected_seal) == nested_seal_digest,
        "nested remote run changed during aggregate verification",
    )
    return snapshots, nested_seal_digest, success_snapshot.sha256


def seal_run(
    run_directory: Path,
    output: Path,
    status_directory: Path,
    screening_report: Path | None = None,
    promotion_report: Path | None = None,
    package_manifest_sha256: str | None = None,
    recipe_sha256: str | None = None,
    run_contract_sha256: str | None = None,
    runtime_bindings_sha256: str | None = None,
    finalization_audit: Path | None = None,
    finalization_audit_sha256: str | None = None,
    profile: str = "structural",
) -> Mapping[str, Any]:
    root = run_directory.resolve(strict=True)
    _directory_identity(root, "run directory")
    output_relative = _relative_run_file(root, output)
    status_relative = _relative_run_file(root, status_directory)
    require(output_relative == "provenance/final-files.json", "run seal output path is non-canonical")
    require(status_relative == "status", "run status directory is non-canonical")
    status_directory = root / "status"
    require(status_directory.is_dir() and not status_directory.is_symlink(), "run status directory is invalid")
    require(not output.exists() and not Path(f"{output}.sha256").exists(), "run seal is immutable")
    require(
        profile in {"structural", "local-aggregate", "remote-semantic"},
        "run seal profile is invalid",
    )
    semantic_arguments = (
        package_manifest_sha256,
        recipe_sha256,
        run_contract_sha256,
        runtime_bindings_sha256,
        finalization_audit,
        finalization_audit_sha256,
    )
    if profile == "remote-semantic":
        require(
            screening_report is not None and promotion_report is not None,
            "remote-semantic seal requires screening and promotion reports",
        )
    else:
        require(
            screening_report is None
            and promotion_report is None
            and all(value is None for value in semantic_arguments),
            f"{profile} seal cannot carry remote-semantic arguments",
        )
    if profile == "structural":
        require(
            not (root / "remote-sealed-run").exists(),
            "structural seal cannot contain a remote aggregate copy",
        )
    semantic_snapshots: dict[str, StableSnapshot] = {}
    remote_source_verification: RemoteSourceVerification | None = None
    screening_digest: str | None = None
    promotion_digest: str | None = None
    local_aggregate_snapshots: dict[str, StableSnapshot] = {}
    nested_remote_seal_sha256: str | None = None
    nested_success_status_sha256: str | None = None
    if profile == "local-aggregate":
        (
            local_aggregate_snapshots,
            nested_remote_seal_sha256,
            nested_success_status_sha256,
        ) = _verify_local_aggregate_remote_copy(root)
        semantic_snapshots.update(local_aggregate_snapshots)
    if profile == "remote-semantic":
        for value, label in (
            (package_manifest_sha256, "package manifest semantic binding"),
            (recipe_sha256, "recipe semantic binding"),
            (run_contract_sha256, "run-contract semantic binding"),
            (runtime_bindings_sha256, "runtime-bindings semantic binding"),
            (finalization_audit_sha256, "finalization-audit semantic binding"),
        ):
            require_sha256(value, label)
        require(
            _relative_run_file(root, screening_report)
            == "screening/epoch-0001.json",
            "screening report path is non-canonical",
        )
        require(
            _relative_run_file(root, promotion_report)
            == "screening/epoch-0001-promotion-gates.json",
            "promotion report path is non-canonical",
        )
        (
            semantic_snapshots,
            screening_digest,
            promotion_digest,
        ) = _screening_promotion_snapshots(screening_report, promotion_report)
        remote_source_verification = verify_remote_package_source(
            root, str(package_manifest_sha256)
        )
        semantic_snapshots.update(remote_source_verification.package_snapshots)
        semantic_snapshots.update(remote_source_verification.source_snapshots)
        require(finalization_audit is not None, "finalization audit is required")
        audit_snapshots, _ = _verify_finalization_audit(
            root,
            finalization_audit,
            str(finalization_audit_sha256),
            package_manifest_sha256=str(package_manifest_sha256),
            recipe_sha256=str(recipe_sha256),
            runtime_bindings_sha256=str(runtime_bindings_sha256),
            source_snapshots=remote_source_verification.source_snapshots,
        )
        semantic_snapshots.update(audit_snapshots)
        semantic_snapshots.update(_verify_remote_semantic_inventory(root))
    records: list[dict[str, object]] = []
    file_snapshots: dict[str, StableSnapshot] = {}
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == output_relative or relative == f"{output_relative}.sha256" or relative == status_relative or relative.startswith(f"{status_relative}/"):
            continue
        require(not path.is_symlink(), f"symlink cannot be sealed: {relative}")
        if path.is_dir():
            continue
        snapshot = stable_snapshot(path, f"run artifact {relative}")
        records.append({"path": relative, "sha256": snapshot.sha256, "size": len(snapshot.payload)})
        file_snapshots[relative] = snapshot
    require(records, "cannot seal an empty run")
    semantic_bindings: dict[str, object]
    if profile == "remote-semantic":
        semantic_bindings = {
            "packageManifestSha256": package_manifest_sha256,
            "finalizationAuditSha256": finalization_audit_sha256,
            "promotionReportSha256": promotion_digest,
            "recipeSha256": recipe_sha256,
            "runContractSha256": run_contract_sha256,
            "runtimeBindingsSha256": runtime_bindings_sha256,
            "screeningReportSha256": screening_digest,
        }
    elif profile == "local-aggregate":
        semantic_bindings = {
            "remoteRunSealSha256": nested_remote_seal_sha256,
            "remoteSuccessStatusSha256": nested_success_status_sha256,
        }
    else:
        semantic_bindings = {}
    seal = {
        "format": "dalmuti-v4-mixed-run-seal",
        "version": 1,
        "profile": profile,
        "semanticBindings": semantic_bindings,
        "files": records,
        "filesSha256": sha256_bytes(canonical_json_bytes(records)),
    }
    if remote_source_verification is not None:
        recheck_remote_package_source(remote_source_verification)
    _write_exclusive_with_sidecar(output, canonical_json_bytes(seal))
    for label, snapshot in semantic_snapshots.items():
        recheck_snapshot(snapshot, label)
    for relative, snapshot in file_snapshots.items():
        recheck_snapshot(snapshot, f"run artifact {relative}")
        target = snapshot.path
        os.chmod(target, stat.S_IMODE(target.stat().st_mode) & ~0o222)
    os.chmod(output, 0o444)
    os.chmod(Path(f"{output}.sha256"), 0o444)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir() and path != status_directory and status_directory not in path.parents),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        require(not directory.is_symlink(), "run directory tree contains a symlink")
        os.chmod(directory, 0o555)
    os.chmod(root, 0o555)
    seal_digest = verify_run_seal(output)
    result = {**seal, "sealSha256": seal_digest, "passed": True}
    if screening_digest is not None and promotion_digest is not None:
        result = {
            **result,
            "promotionReportSha256": promotion_digest,
            "screeningReportSha256": screening_digest,
        }
    return result


def verify_run_seal(seal_path: Path) -> str:
    seal_path = seal_path.resolve(strict=True)
    seal_snapshot, seal_sidecar_snapshot = snapshot_with_sidecar(seal_path)
    seal_digest = seal_snapshot.sha256
    require(
        stat.S_IMODE(seal_path.stat().st_mode) & 0o222 == 0
        and stat.S_IMODE(Path(f"{seal_path}.sha256").stat().st_mode) & 0o222 == 0,
        "run seal or sidecar remains writable",
    )
    seal = load_canonical_json_snapshot(seal_snapshot, "run seal")
    require(
        set(seal)
        == {
            "files",
            "filesSha256",
            "format",
            "profile",
            "semanticBindings",
            "version",
        }
        and seal.get("format") == "dalmuti-v4-mixed-run-seal"
        and seal.get("version") == 1,
        "invalid run seal",
    )
    profile = seal.get("profile")
    semantic_bindings = seal.get("semanticBindings")
    require(
        profile in {"structural", "local-aggregate", "remote-semantic"}
        and isinstance(semantic_bindings, Mapping),
        "run seal profile is invalid",
    )
    if profile == "structural":
        require(semantic_bindings == {}, "structural seal has semantic bindings")
    elif profile == "local-aggregate":
        require(
            set(semantic_bindings)
            == {"remoteRunSealSha256", "remoteSuccessStatusSha256"}
            and all(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                for value in semantic_bindings.values()
            ),
            "local aggregate seal bindings are invalid",
        )
    else:
        require(
            set(semantic_bindings)
            == {
                "packageManifestSha256",
                "finalizationAuditSha256",
                "promotionReportSha256",
                "recipeSha256",
                "runContractSha256",
                "runtimeBindingsSha256",
                "screeningReportSha256",
            }
            and all(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                for value in semantic_bindings.values()
            ),
            "remote semantic seal bindings are invalid",
        )
    root = seal_path.parent.parent.resolve(strict=True)
    root_identity = _directory_identity(root, "sealed run root")
    require(
        seal_path.resolve(strict=True) == root / "provenance" / "final-files.json",
        "run seal is not at the canonical run path",
    )
    records = seal.get("files")
    require(isinstance(records, list) and records, "run seal file inventory is empty")
    require(seal.get("filesSha256") == sha256_bytes(canonical_json_bytes(records)), "run seal inventory digest mismatch")
    expected: dict[str, Mapping[str, Any]] = {}
    for item in records:
        require(isinstance(item, Mapping) and set(item) == {"path", "sha256", "size"}, "run seal entry is invalid")
        relative = str(safe_relative_path(item.get("path"), "sealed run path"))
        require(not relative.startswith("status/"), "run seal includes mutable status")
        require(relative not in expected, "run seal contains a duplicate path")
        require_sha256(item.get("sha256"), "sealed run digest")
        size = item.get("size")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, "sealed run size is invalid")
        expected[relative] = item
    require(list(expected) == sorted(expected), "run seal inventory is not sorted")
    observed: set[str] = set()
    file_snapshots: dict[str, StableSnapshot] = {}
    directory_identities: dict[str, tuple[int, int, int, int, int]] = {
        ".": root_identity
    }
    require(stat.S_IMODE(root.stat().st_mode) == 0o555, "sealed run root mode is not 0555")
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == "provenance/final-files.json" or relative == "provenance/final-files.json.sha256" or relative == "status" or relative.startswith("status/"):
            continue
        require(not path.is_symlink(), f"sealed run now contains a symlink: {relative}")
        if path.is_dir():
            directory_identities[relative] = _directory_identity(path, f"sealed directory {relative}")
            require(stat.S_IMODE(path.stat().st_mode) == 0o555, f"sealed directory mode is not 0555: {relative}")
            continue
        require(path.is_file() and relative in expected, f"sealed run contains an unbound file: {relative}")
        record = expected[relative]
        snapshot = stable_snapshot(path, f"sealed run file {relative}")
        require(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0, f"sealed run file remains writable: {relative}")
        require(len(snapshot.payload) == record["size"], f"sealed run size mismatch: {relative}")
        require(snapshot.sha256 == record["sha256"], f"sealed run digest mismatch: {relative}")
        file_snapshots[relative] = snapshot
        observed.add(relative)
    require(observed == set(expected), "sealed run inventory is incomplete")
    if profile == "structural":
        require(
            not (root / "remote-sealed-run").exists(),
            "structural seal contains a remote aggregate copy",
        )
    if profile == "local-aggregate":
        (
            local_snapshots,
            nested_remote_seal_sha256,
            nested_success_status_sha256,
        ) = _verify_local_aggregate_remote_copy(root)
        require(
            semantic_bindings
            == {
                "remoteRunSealSha256": nested_remote_seal_sha256,
                "remoteSuccessStatusSha256": nested_success_status_sha256,
            },
            "local aggregate semantic bindings drifted",
        )
        for label, snapshot in local_snapshots.items():
            recheck_snapshot(snapshot, label)
    if profile == "remote-semantic":
        remote_source_verification = verify_remote_package_source(
            root, str(semantic_bindings["packageManifestSha256"])
        )
        semantic_snapshots, screening_digest, promotion_digest = (
            _screening_promotion_snapshots(
                root / "screening" / "epoch-0001.json",
                root / "screening" / "epoch-0001-promotion-gates.json",
            )
        )
        require(
            semantic_bindings.get("screeningReportSha256") == screening_digest
            and semantic_bindings.get("promotionReportSha256") == promotion_digest,
            "remote semantic seal report bindings drifted",
        )
        audit_snapshots, _ = _verify_finalization_audit(
            root,
            root / "provenance" / "finalization-audit.json",
            str(semantic_bindings["finalizationAuditSha256"]),
            package_manifest_sha256=str(
                semantic_bindings["packageManifestSha256"]
            ),
            recipe_sha256=str(semantic_bindings["recipeSha256"]),
            runtime_bindings_sha256=str(
                semantic_bindings["runtimeBindingsSha256"]
            ),
            source_snapshots=remote_source_verification.source_snapshots,
        )
        inventory_snapshots = _verify_remote_semantic_inventory(root)
        runtime_snapshot, runtime_sidecar = snapshot_with_sidecar(
            root / "control" / "runtime-bindings.json",
            str(semantic_bindings["runtimeBindingsSha256"]),
        )
        runtime_bindings = load_canonical_json_snapshot(
            runtime_snapshot, "sealed runtime bindings"
        )
        require(
            runtime_bindings.get("packageManifestSha256")
            == semantic_bindings.get("packageManifestSha256")
            and runtime_bindings.get("recipeSha256")
            == semantic_bindings.get("recipeSha256"),
            "sealed runtime bindings disagree with semantic seal",
        )
        recipe_path = root / "source" / "gpu-training" / "v4_mixed_execution_recipe.json"
        recipe_snapshot = stable_snapshot(recipe_path, "sealed mixed recipe")
        recipe = load_canonical_json_snapshot(recipe_snapshot, "sealed mixed recipe")
        require(
            recipe_snapshot.sha256 == semantic_bindings.get("recipeSha256")
            and sha256_bytes(canonical_json_bytes(recipe.get("runContract")))
            == semantic_bindings.get("runContractSha256"),
            "sealed recipe or run contract drifted",
        )
        package_snapshot = stable_snapshot(
            root / "package" / "package-manifest.json",
            "sealed package manifest",
        )
        require(
            package_snapshot.sha256
            == semantic_bindings.get("packageManifestSha256"),
            "sealed package manifest drifted",
        )
        for label, snapshot in {
            **semantic_snapshots,
            **audit_snapshots,
            **inventory_snapshots,
            "runtime bindings": runtime_snapshot,
            "runtime bindings sidecar": runtime_sidecar,
            "sealed mixed recipe": recipe_snapshot,
            "sealed package manifest": package_snapshot,
        }.items():
            recheck_snapshot(snapshot, label)
        recheck_remote_package_source(remote_source_verification)
    recheck_snapshot(seal_snapshot, "run seal")
    recheck_snapshot(seal_sidecar_snapshot, "run seal sidecar")
    for relative, snapshot in file_snapshots.items():
        recheck_snapshot(snapshot, f"sealed run file {relative}")
    observed_directories = {"."}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == "status" or relative.startswith("status/") or not path.is_dir():
            continue
        observed_directories.add(relative)
        require(
            directory_identities.get(relative)
            == _directory_identity(path, f"sealed directory {relative}"),
            f"sealed directory changed: {relative}",
        )
        require(stat.S_IMODE(path.stat().st_mode) == 0o555, f"sealed directory became writable: {relative}")
    require(observed_directories == set(directory_identities), "sealed directory inventory changed")
    require(_directory_identity(root, "sealed run root") == root_identity, "sealed run root changed during verification")
    return seal_digest


def _write_exclusive_with_sidecar(path: Path, payload: bytes) -> str:
    require(not path.exists() and not path.is_symlink(), f"immutable output exists: {path}")
    sidecar = Path(f"{path}.sha256")
    require(not sidecar.exists() and not sidecar.is_symlink(), f"immutable sidecar exists: {sidecar}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    created_path = False
    created_sidecar = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created_path = True
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        digest = sha256_bytes(payload)
        descriptor = os.open(sidecar, flags, 0o600)
        created_sidecar = True
        with os.fdopen(descriptor, "wb") as output:
            output.write(sidecar_bytes(digest, path.name))
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        for target, created in ((sidecar, created_sidecar), (path, created_path)):
            if not created:
                continue
            try:
                os.chmod(target, 0o600)
                target.unlink()
            except OSError:
                pass
        raise
    return digest


def _terminal_status_values(status_root: Path) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for path in sorted(status_root.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid status entry: {path.name}")
        value = load_canonical_json(path, f"status {path.name}")
        if value.get("state") in TERMINAL_STATES:
            values.append(value)
    return values


def write_status(output: Path, stage: str, state: str, detail: str, seal: Path | None) -> Mapping[str, Any]:
    safe_leaf(stage, "status stage")
    require(state in {"started", "completed", *TERMINAL_STATES}, "invalid status state")
    require(isinstance(detail, str) and len(detail) <= 4096, "invalid status detail")
    if state == "succeeded":
        require(seal is not None, "success status requires a completed run seal")
        seal_digest = verify_run_seal(seal)
        run_root = seal.resolve(strict=True).parent.parent
        output_parent = output.parent.resolve(strict=True)
        require(
            output_parent == run_root / "status"
            or run_root / "status" in output_parent.parents,
            "success status must be inside the mutable status directory",
        )
    else:
        require(seal is None, "only success status may bind a seal")
        seal_digest = None
    value = {
        "format": "dalmuti-v4-mixed-stage-status",
        "version": 1,
        "stage": stage,
        "state": state,
        "detail": detail,
        "runSealSha256": seal_digest,
    }
    terminal = state in TERMINAL_STATES
    lock: Path | None = None
    published = False
    if terminal:
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = output.parent / ".terminal-status.lock"
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ValueError("another terminal status publication is in progress") from error
    try:
        if terminal:
            require(
                not _terminal_status_values(output.parent),
                "a terminal status is already published",
            )
        _write_exclusive_with_sidecar(output, canonical_json_bytes(value))
        published = True
        if terminal:
            os.chmod(output, 0o444)
            os.chmod(Path(f"{output}.sha256"), 0o444)
        if state == "succeeded":
            require(verify_run_seal(seal) == seal_digest, "run seal changed while publishing success")
            status_snapshot, status_sidecar_snapshot = snapshot_with_sidecar(output)
            recheck_snapshot(status_snapshot, "success status")
            recheck_snapshot(status_sidecar_snapshot, "success status sidecar")
            require(verify_run_seal(seal) == seal_digest, "sealed tree changed after success publication")
    except BaseException:
        if published:
            for failed_path in (output, Path(f"{output}.sha256")):
                try:
                    os.chmod(failed_path, 0o600)
                    failed_path.unlink()
                except OSError:
                    pass
        raise
    finally:
        if lock is not None:
            try:
                lock.rmdir()
            except OSError:
                if published:
                    for failed_path in (output, Path(f"{output}.sha256")):
                        try:
                            os.chmod(failed_path, 0o600)
                            failed_path.unlink()
                        except OSError:
                            pass
                raise
    return value


def _print(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-package")
    verify.add_argument("--package-dir", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    verify.add_argument("--remote-only", action="store_true")
    extract = subparsers.add_parser("extract-source")
    extract.add_argument("--package-dir", type=Path, required=True)
    extract.add_argument("--expected-manifest-sha256", required=True)
    extract.add_argument("--destination", type=Path, required=True)
    screening = subparsers.add_parser("verify-screening")
    screening.add_argument("--package-dir", type=Path, required=True)
    screening.add_argument("--expected-manifest-sha256", required=True)
    screening.add_argument("--source-root", type=Path, required=True)
    screening.add_argument("--report", type=Path, required=True)
    screening.add_argument("--candidate", type=Path, required=True)
    seal = subparsers.add_parser("seal-run")
    seal.add_argument("--run-directory", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--status-directory", type=Path, required=True)
    seal.add_argument("--screening-report", type=Path)
    seal.add_argument("--promotion-report", type=Path)
    seal.add_argument("--package-manifest-sha256")
    seal.add_argument("--recipe-sha256")
    seal.add_argument("--run-contract-sha256")
    seal.add_argument("--runtime-bindings-sha256")
    seal.add_argument("--finalization-audit", type=Path)
    seal.add_argument("--finalization-audit-sha256")
    seal.add_argument(
        "--profile",
        choices=("structural", "local-aggregate", "remote-semantic"),
        default="structural",
    )
    status_parser = subparsers.add_parser("write-status")
    status_parser.add_argument("--output", type=Path, required=True)
    status_parser.add_argument("--stage", required=True)
    status_parser.add_argument("--state", required=True)
    status_parser.add_argument("--detail", default="")
    status_parser.add_argument("--seal", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.command == "verify-package":
        result = verify_package(args.package_dir, args.expected_manifest_sha256, remote_only=args.remote_only)
    elif args.command == "extract-source":
        result = extract_source(args.package_dir, args.expected_manifest_sha256, args.destination)
    elif args.command == "verify-screening":
        result = verify_screening(args.package_dir, args.expected_manifest_sha256, args.source_root, args.report, args.candidate)
    elif args.command == "seal-run":
        result = seal_run(
            args.run_directory,
            args.output,
            args.status_directory,
            args.screening_report,
            args.promotion_report,
            args.package_manifest_sha256,
            args.recipe_sha256,
            args.run_contract_sha256,
            args.runtime_bindings_sha256,
            args.finalization_audit,
            args.finalization_audit_sha256,
            args.profile,
        )
    elif args.command == "write-status":
        result = write_status(args.output, args.stage, args.state, args.detail, args.seal)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
