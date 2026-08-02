from __future__ import annotations

"""Checksum-bound tar.zst spooling for distributed DALMUTI V5 collection.

This module intentionally contains no SSH client.  It provides the immutable
artifacts on both sides of a transfer:

1. verify a planned raw shard and atomically export a tar.zst spool bundle;
2. copy that bundle with any transport;
3. safely import it into the canonical collector-host shard directory;
4. publish a verified-copy receipt;
5. copy the receipt back and retire only the exact verified remote replicas.

An archive or raw shard is never removed merely because an upload command
returned success.  Retirement requires a canonical receipt that binds the
collection plan, planned shard, source manifest, archive bytes, and imported
canonical shard.
"""

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid

from v5_collection_plan import (
    V5CollectionPlan,
    V5PlannedShard,
    load_collection_plan,
    planned_shard_path,
    verify_planned_shard,
)
from v5_low_disk_stage import (
    V5ShardStorageInventory,
    inventory_v5_training_shard,
)


V5_SPOOL_FORMAT = "dalmuti-v5-checksum-bound-tar-zstd-spool"
V5_SPOOL_VERSION = 1
V5_COPY_RECEIPT_FORMAT = "dalmuti-v5-verified-copy-receipt"
V5_COPY_RECEIPT_VERSION = 1
V5_ARCHIVE_CONTRACT = "single-planned-shard-regular-files-and-directories-v1"

RAW_ROOT_NAME = "raw-shards"
SPOOL_ROOT_NAME = "spool-bundles"
INCOMING_SPOOL_ROOT_NAME = "incoming-spool-bundles"
CANONICAL_ROOT_NAME = "canonical-shards"
RECEIPT_ROOT_NAME = "verified-copy-receipts"

DEFAULT_SPOOL_RESERVE_BYTES = 2 * 1024**3
DEFAULT_IMPORT_RESERVE_BYTES = 10 * 1024**3

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_TOKEN = re.compile(r"[a-zA-Z0-9]{8,64}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _exact_keys(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} fields are non-canonical")
    return value


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _no_replace_directory_rename(source: Path, target: Path) -> None:
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("Linux renameat2(RENAME_NOREPLACE) is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result:
            number = ctypes.get_errno()
            if number in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(number, os.strerror(number), target)
            raise OSError(number, os.strerror(number), target)
        return
    if os.name == "nt":
        os.rename(source, target)
        return
    raise RuntimeError("atomic no-replace directory rename is unsupported here")


def _run_directory(path: str | Path, run_namespace: str, expected_name: str) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError(f"V5 {expected_name} root must not be a symlink")
    root = unresolved.resolve()
    if root.name != expected_name or root.parent.name != run_namespace:
        raise ValueError(
            f"V5 root must be <independent-root>/{run_namespace}/{expected_name}"
        )
    return root


def _load_plan_shard(
    plan_path: str | Path, shard_index: int
) -> tuple[V5CollectionPlan, V5PlannedShard]:
    plan = load_collection_plan(plan_path)
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < len(plan.shards)
    ):
        raise ValueError("shard_index is outside the immutable collection plan")
    return plan, plan.shards[shard_index]


def _spool_name(shard: V5PlannedShard, manifest_sha256: str) -> str:
    return f"spool-{shard.index:03d}-{manifest_sha256[:16]}"


def _receipt_name(shard: V5PlannedShard, manifest_sha256: str) -> str:
    return f"copy-{shard.index:03d}-{manifest_sha256[:16]}"


def _inventory_members(root: Path) -> list[dict[str, object]]:
    if root.is_symlink():
        raise ValueError("V5 shard root must not be a symlink")
    members: list[dict[str, object]] = [
        {
            "byteLength": 0,
            "name": f"{root.name}/",
            "sha256": None,
            "type": "directory",
        }
    ]
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("V5 shard archive input must not contain symlinks")
        relative = path.relative_to(root).as_posix()
        name = f"{root.name}/{relative}"
        if path.is_dir():
            members.append(
                {
                    "byteLength": 0,
                    "name": name + "/",
                    "sha256": None,
                    "type": "directory",
                }
            )
        elif path.is_file():
            members.append(
                {
                    "byteLength": path.stat().st_size,
                    "name": name,
                    "sha256": _sha256_file(path),
                    "type": "file",
                }
            )
        else:
            raise ValueError("V5 shard contains a non-file non-directory entry")
    return members


def validate_v5_archive_member_names(
    names: Sequence[str], *, expected_top_directory: str
) -> None:
    """Reject path traversal, alternate separators, and duplicate members."""

    if not names or len(set(names)) != len(names):
        raise ValueError("V5 archive member list is empty or contains duplicates")
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or "\x00" in name
            or name.startswith("/")
        ):
            raise ValueError("V5 archive contains an unsafe member name")
        path = PurePosixPath(name.rstrip("/"))
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != expected_top_directory
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError("V5 archive member escapes its planned top directory")


def _tar_executable() -> str:
    executable = shutil.which("tar")
    if executable is None:
        raise RuntimeError("tar executable is required for V5 spool archives")
    return executable


def _gnu_tar(executable: str) -> bool:
    result = subprocess.run(
        [executable, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0 and b"GNU tar" in result.stdout


def _create_tar_zstd(source: Path, archive: Path, *, level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 19:
        raise ValueError("zstd level must be in 1..19")
    if _path_lexists(archive):
        raise FileExistsError(f"V5 spool archive already exists: {archive}")
    tar = _tar_executable()
    zstd = shutil.which("zstd")
    if zstd is not None:
        tar_command = [tar, "-C", str(source.parent)]
        if _gnu_tar(tar):
            tar_command.extend(
                [
                    "--sort=name",
                    "--mtime=@0",
                    "--owner=0",
                    "--group=0",
                    "--numeric-owner",
                    "--format=ustar",
                ]
            )
        tar_command.extend(["-cf", "-", source.name])
        with archive.open("xb") as output:
            producer = subprocess.Popen(
                tar_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert producer.stdout is not None
            compressor = subprocess.Popen(
                [zstd, "-q", f"-{level}", "-T0", "-c"],
                stdin=producer.stdout,
                stdout=output,
                stderr=subprocess.PIPE,
            )
            producer.stdout.close()
            compressor_error = (
                compressor.stderr.read() if compressor.stderr is not None else b""
            )
            compressor_code = compressor.wait()
            producer_error = (
                producer.stderr.read() if producer.stderr is not None else b""
            )
            producer_code = producer.wait()
            output.flush()
            os.fsync(output.fileno())
        if producer_code or compressor_code:
            detail = (producer_error + compressor_error).decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(f"V5 tar.zst export failed: {detail.strip()}")
        return

    # Windows' bundled bsdtar has libzstd even when no standalone zstd exists.
    result = subprocess.run(
        [tar, "-acf", str(archive), "-C", str(source.parent), source.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        try:
            archive.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            "V5 tar.zst export failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    with archive.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _archive_listing(archive: Path) -> tuple[list[str], list[str]]:
    tar = _tar_executable()
    names_result = subprocess.run(
        [tar, "-tf", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    verbose_result = subprocess.run(
        [tar, "-tvf", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if names_result.returncode or verbose_result.returncode:
        detail = names_result.stderr + verbose_result.stderr
        raise ValueError(
            "V5 tar.zst archive could not be listed: "
            + detail.decode("utf-8", errors="replace").strip()
        )
    try:
        names = names_result.stdout.decode("utf-8").splitlines()
        verbose = verbose_result.stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("V5 archive member listing is not UTF-8") from error
    if len(names) != len(verbose):
        raise ValueError("V5 archive verbose and plain member listings disagree")
    types: list[str] = []
    for line in verbose:
        if not line or line[0] not in ("-", "d"):
            raise ValueError("V5 archive contains a link or special-file member")
        types.append("directory" if line[0] == "d" else "file")
    return names, types


def _validate_archive_against_members(
    archive: Path,
    members: Sequence[Mapping[str, object]],
    *,
    expected_top_directory: str,
) -> None:
    names, types = _archive_listing(archive)
    validate_v5_archive_member_names(
        names, expected_top_directory=expected_top_directory
    )
    expected = {
        str(record["name"]): str(record["type"]) for record in members
    }
    observed = dict(zip(names, types, strict=True))
    if len(expected) != len(members) or observed != expected:
        raise ValueError("V5 archive membership differs from its spool manifest")


def _extract_archive(archive: Path, staging: Path) -> None:
    if any(staging.iterdir()):
        raise ValueError("V5 archive extraction staging directory is not empty")
    tar = _tar_executable()
    result = subprocess.run(
        [tar, "-xf", str(archive), "-C", str(staging)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "V5 tar.zst extraction failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )


def _copy_verified_archive(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    """Copy one already-bound archive inode into private import staging."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, flags)
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    copied = 0
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ValueError("V5 incoming archive is not the expected regular file")
        destination_descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
            0o600,
        )
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("V5 private archive copy made no progress")
                view = view[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or copied != expected_bytes
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError("V5 incoming archive changed or failed checksum while copying")
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(target: Path, builder: object) -> None:
    if target.is_symlink():
        raise ValueError("V5 immutable target must not be a symlink")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.publish.lock"
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        if _path_lexists(target):
            raise FileExistsError(f"immutable V5 artifact already exists: {target}")
        staging.mkdir()
        if not callable(builder):
            raise TypeError("V5 spool builder must be callable")
        builder(staging)
        _no_replace_directory_rename(staging, target)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
        if staging.exists():
            shutil.rmtree(staging)


def _source_record(
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
    manifest_sha: str,
    inventory: V5ShardStorageInventory,
) -> dict[str, object]:
    return {
        "collectionPlanManifestSha256": plan.manifest_sha256,
        "decisionCount": inventory.decision_count,
        "fileCount": inventory.file_count,
        "logicalBytes": inventory.logical_bytes,
        "manifestSha256": manifest_sha,
        "matchCount": inventory.match_count,
        "name": shard.name,
        "nonforcedDecisionCount": inventory.nonforced_decision_count,
        "playerCount": shard.player_count,
        "plannedMatchCount": shard.match_count,
        "plannedMatchStart": shard.match_start,
        "shardIndex": shard.index,
    }


def _archive_capacity_bound(logical_bytes: int, member_count: int) -> int:
    # POSIX tar consumes at most one 512-byte header and 511 bytes of padding
    # per member plus two zero blocks.  The extra zstd allowance is deliberately
    # above the documented compress-bound overhead for an incompressible frame.
    tar_bound = logical_bytes + 1024 * (member_count + 2)
    return tar_bound + (tar_bound >> 8) + 128 * 1024


def _acquire_capacity_lock(spool: Path, *, timeout_seconds: float = 30.0) -> tuple[Path, int]:
    lock = spool / ".capacity-reservation.lock"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(
                lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            return lock, descriptor
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("V5 spool capacity reservation lock stayed busy")
            time.sleep(0.02)


def _release_capacity_lock(lock: Path, descriptor: int) -> None:
    os.close(descriptor)
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


def _active_capacity_reservations(directory: Path, *, exclude: str | None = None) -> int:
    total = 0
    children = tuple(directory.iterdir())
    if any(path.suffix != ".json" for path in children):
        raise ValueError("V5 spool capacity reservation inventory drifted")
    for path in sorted(children):
        if path.name == exclude:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("V5 spool capacity reservation is not a real file")
        raw = path.read_bytes()
        try:
            record = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("V5 spool capacity reservation is invalid") from error
        if (
            not isinstance(record, dict)
            or set(record) != {"chargeBytes", "pid", "token"}
            or _canonical_json_bytes(record) != raw
            or record["token"] != path.stem
        ):
            raise ValueError("V5 spool capacity reservation is non-canonical")
        total += _integer(record["chargeBytes"], "reservation charge", minimum=1)
        _integer(record["pid"], "reservation pid", minimum=1)
    return total


@contextmanager
def _spool_capacity_reservation(
    spool: Path, *, charge_bytes: int, reserve_bytes: int
):
    directory = spool / ".capacity-reservations"
    if directory.is_symlink():
        raise ValueError("V5 spool capacity reservation root must not be a symlink")
    directory.mkdir(exist_ok=True)
    token = uuid.uuid4().hex
    record_path = directory / f"{token}.json"
    lock, descriptor = _acquire_capacity_lock(spool)
    try:
        active = _active_capacity_reservations(directory)
        if shutil.disk_usage(spool).free - active < charge_bytes + reserve_bytes:
            raise ValueError(
                "insufficient spool space after active reservations and reserve"
            )
        _write_exclusive(
            record_path,
            _canonical_json_bytes(
                {"chargeBytes": charge_bytes, "pid": os.getpid(), "token": token}
            ),
        )
    finally:
        _release_capacity_lock(lock, descriptor)

    def verify_remaining_headroom() -> None:
        capacity_lock, capacity_descriptor = _acquire_capacity_lock(spool)
        try:
            active_other = _active_capacity_reservations(
                directory, exclude=record_path.name
            )
            if shutil.disk_usage(spool).free - active_other < reserve_bytes:
                raise ValueError(
                    "tar.zst export would violate reserved concurrent spool headroom"
                )
        finally:
            _release_capacity_lock(capacity_lock, capacity_descriptor)

    try:
        yield verify_remaining_headroom
    finally:
        try:
            record_path.unlink()
        except FileNotFoundError:
            pass


def export_v5_planned_shard_spool(
    plan_path: str | Path,
    *,
    shard_index: int,
    raw_root: str | Path,
    spool_root: str | Path,
    zstd_level: int = 3,
    minimum_free_after_export_bytes: int = DEFAULT_SPOOL_RESERVE_BYTES,
) -> Path:
    """Verify one planned raw shard and atomically publish its tar.zst bundle."""

    plan, shard = _load_plan_shard(plan_path, shard_index)
    raw = _run_directory(raw_root, plan.run_namespace, RAW_ROOT_NAME)
    spool = _run_directory(spool_root, plan.run_namespace, SPOOL_ROOT_NAME)
    source = planned_shard_path(raw, shard)
    if source.is_symlink():
        raise ValueError("planned V5 raw shard must not be a symlink")
    manifest_sha = verify_planned_shard(plan, shard, source)
    inventory = inventory_v5_training_shard(source)
    members = _inventory_members(source)
    reserve = _integer(
        minimum_free_after_export_bytes,
        "minimum_free_after_export_bytes",
    )
    spool.mkdir(parents=True, exist_ok=True)
    archive_capacity_bound = _archive_capacity_bound(
        inventory.logical_bytes, len(members)
    )
    target = spool / _spool_name(shard, manifest_sha)

    def build(staging: Path, verify_remaining_headroom: object) -> None:
        archive = staging / "shard.tar.zst"
        _create_tar_zstd(source, archive, level=zstd_level)
        if not callable(verify_remaining_headroom):
            raise TypeError("V5 spool headroom verifier must be callable")
        verify_remaining_headroom()
        archive_sha = _sha256_file(archive)
        _validate_archive_against_members(
            archive,
            members,
            expected_top_directory=shard.name,
        )
        document = {
            "archive": {
                "byteLength": archive.stat().st_size,
                "contract": V5_ARCHIVE_CONTRACT,
                "fileName": "shard.tar.zst",
                "members": members,
                "sha256": archive_sha,
                "zstdLevel": zstd_level,
            },
            "format": V5_SPOOL_FORMAT,
            "runNamespace": plan.run_namespace,
            "source": _source_record(plan, shard, manifest_sha, inventory),
            "version": V5_SPOOL_VERSION,
        }
        raw_manifest = _canonical_json_bytes(document)
        manifest_digest = _sha256_bytes(raw_manifest)
        _write_exclusive(staging / "spool.json", raw_manifest)
        _write_exclusive(
            staging / "spool.json.sha256",
            f"{manifest_digest}  spool.json\n".encode("ascii"),
        )
        _write_exclusive(
            staging / "shard.tar.zst.sha256",
            f"{archive_sha}  shard.tar.zst\n".encode("ascii"),
        )

    with _spool_capacity_reservation(
        spool,
        charge_bytes=archive_capacity_bound,
        reserve_bytes=reserve,
    ) as verify_headroom:
        _publish_directory(
            target, lambda staging: build(staging, verify_headroom)
        )
    return target


def _read_canonical_document(
    path: Path, sidecar: Path, *, label: str, sidecar_name: str
) -> tuple[Mapping[str, object], str]:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    if sidecar.read_bytes() != f"{digest}  {sidecar_name}\n".encode("ascii"):
        raise ValueError(f"{label} checksum sidecar does not match")
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not ASCII JSON") from error
    if not isinstance(document, dict) or _canonical_json_bytes(document) != raw:
        raise ValueError(f"{label} is not canonical")
    return document, digest


def _validate_member_records(value: object, top: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("V5 spool archive members are missing")
    records: list[Mapping[str, object]] = []
    names: list[str] = []
    for raw in value:
        record = _exact_keys(
            raw, {"byteLength", "name", "sha256", "type"}, "V5 archive member"
        )
        name = record["name"]
        kind = record["type"]
        size = _integer(record["byteLength"], "archive member byte length")
        if not isinstance(name, str) or kind not in ("file", "directory"):
            raise ValueError("V5 archive member fields are invalid")
        if kind == "directory":
            if size or record["sha256"] is not None or not name.endswith("/"):
                raise ValueError("V5 archive directory member is invalid")
        else:
            _sha(record["sha256"], "archive file member SHA")
            if size < 1 or name.endswith("/"):
                raise ValueError("V5 archive file member is invalid")
        names.append(name)
        records.append(record)
    validate_v5_archive_member_names(names, expected_top_directory=top)
    expected = [top + "/"] + sorted(names[1:])
    if names != expected:
        raise ValueError("V5 archive member records are non-canonical")
    return tuple(records)


def load_v5_spool_bundle(bundle_path: str | Path) -> tuple[Mapping[str, object], str]:
    unresolved = Path(bundle_path)
    if unresolved.is_symlink():
        raise ValueError("V5 spool bundle must not be a symlink")
    root = unresolved.resolve()
    expected_files = {
        "shard.tar.zst",
        "shard.tar.zst.sha256",
        "spool.json",
        "spool.json.sha256",
    }
    children = tuple(root.iterdir())
    if (
        {path.name for path in children} != expected_files
        or any(path.is_symlink() or not path.is_file() for path in children)
    ):
        raise ValueError("V5 spool bundle file inventory is incomplete")
    document, manifest_sha = _read_canonical_document(
        root / "spool.json",
        root / "spool.json.sha256",
        label="V5 spool manifest",
        sidecar_name="spool.json",
    )
    top = _exact_keys(
        document,
        {"archive", "format", "runNamespace", "source", "version"},
        "V5 spool manifest",
    )
    if top["format"] != V5_SPOOL_FORMAT or top["version"] != V5_SPOOL_VERSION:
        raise ValueError("V5 spool manifest contract is incompatible")
    source = _exact_keys(
        top["source"],
        {
            "collectionPlanManifestSha256", "decisionCount", "fileCount", "logicalBytes",
            "manifestSha256", "matchCount", "name", "nonforcedDecisionCount",
            "plannedMatchCount", "plannedMatchStart", "playerCount", "shardIndex",
        },
        "V5 spool source",
    )
    for name in (
        "collectionPlanManifestSha256",
        "manifestSha256",
    ):
        _sha(source[name], f"V5 spool source {name}")
    for name in (
        "decisionCount", "fileCount", "logicalBytes", "matchCount",
        "nonforcedDecisionCount", "plannedMatchCount", "plannedMatchStart",
        "playerCount", "shardIndex",
    ):
        _integer(source[name], f"V5 spool source {name}")
    if top["runNamespace"] is None or not isinstance(top["runNamespace"], str):
        raise ValueError("V5 spool run namespace is invalid")
    if not isinstance(source["name"], str):
        raise ValueError("V5 spool source name is invalid")
    archive_record = _exact_keys(
        top["archive"],
        {"byteLength", "contract", "fileName", "members", "sha256", "zstdLevel"},
        "V5 spool archive",
    )
    if (
        archive_record["contract"] != V5_ARCHIVE_CONTRACT
        or archive_record["fileName"] != "shard.tar.zst"
    ):
        raise ValueError("V5 spool archive contract drifted")
    archive_sha = _sha(archive_record["sha256"], "V5 spool archive SHA")
    archive_size = _integer(archive_record["byteLength"], "V5 spool archive bytes", minimum=1)
    level = _integer(archive_record["zstdLevel"], "V5 spool zstd level", minimum=1)
    if level > 19:
        raise ValueError("V5 spool zstd level is outside 1..19")
    members = _validate_member_records(archive_record["members"], str(source["name"]))
    archive = root / "shard.tar.zst"
    if archive.stat().st_size != archive_size or _sha256_file(archive) != archive_sha:
        raise ValueError("V5 spool archive byte length or checksum drifted")
    if (root / "shard.tar.zst.sha256").read_bytes() != (
        f"{archive_sha}  shard.tar.zst\n".encode("ascii")
    ):
        raise ValueError("V5 spool archive sidecar does not match")
    _validate_archive_against_members(
        archive,
        members,
        expected_top_directory=str(source["name"]),
    )
    return top, manifest_sha


def _bind_spool_to_plan(
    document: Mapping[str, object],
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
) -> Mapping[str, object]:
    source = document["source"]
    assert isinstance(source, Mapping)
    expected = {
        "collectionPlanManifestSha256": plan.manifest_sha256,
        "name": shard.name,
        "plannedMatchCount": shard.match_count,
        "plannedMatchStart": shard.match_start,
        "playerCount": shard.player_count,
        "shardIndex": shard.index,
    }
    if document["runNamespace"] != plan.run_namespace:
        raise ValueError("V5 spool run namespace differs from collection plan")
    for name, value in expected.items():
        if source.get(name) != value:
            raise ValueError(f"V5 spool differs from planned shard: {name}")
    expected_bundle_name = _spool_name(shard, str(source["manifestSha256"]))
    return {"bundleName": expected_bundle_name, **dict(source)}


def _inventory_matches_source(
    inventory: V5ShardStorageInventory,
    manifest_sha: str,
    source: Mapping[str, object],
) -> None:
    expected = {
        "decisionCount": inventory.decision_count,
        "fileCount": inventory.file_count,
        "logicalBytes": inventory.logical_bytes,
        "manifestSha256": manifest_sha,
        "matchCount": inventory.match_count,
        "nonforcedDecisionCount": inventory.nonforced_decision_count,
    }
    for name, value in expected.items():
        if source.get(name) != value:
            raise ValueError(f"V5 imported shard differs from spool source: {name}")


def _receipt_document(
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
    spool: Mapping[str, object],
    spool_manifest_sha: str,
    inventory: V5ShardStorageInventory,
    imported_manifest_sha: str,
) -> dict[str, object]:
    source = spool["source"]
    archive = spool["archive"]
    assert isinstance(source, Mapping) and isinstance(archive, Mapping)
    return {
        "archive": {
            "byteLength": archive["byteLength"],
            "sha256": archive["sha256"],
            "spoolManifestSha256": spool_manifest_sha,
        },
        "copyVerified": True,
        "format": V5_COPY_RECEIPT_FORMAT,
        "imported": {
            "decisionCount": inventory.decision_count,
            "fileCount": inventory.file_count,
            "logicalBytes": inventory.logical_bytes,
            "manifestSha256": imported_manifest_sha,
            "matchCount": inventory.match_count,
            "nonforcedDecisionCount": inventory.nonforced_decision_count,
        },
        "runNamespace": plan.run_namespace,
        "source": {
            "collectionPlanManifestSha256": plan.manifest_sha256,
            "manifestSha256": source["manifestSha256"],
            "name": shard.name,
            "shardIndex": shard.index,
        },
        "version": V5_COPY_RECEIPT_VERSION,
    }


def _publish_receipt(target: Path, document: Mapping[str, object]) -> str:
    raw = _canonical_json_bytes(dict(document))
    digest = _sha256_bytes(raw)

    def build(staging: Path) -> None:
        _write_exclusive(staging / "receipt.json", raw)
        _write_exclusive(
            staging / "receipt.json.sha256",
            f"{digest}  receipt.json\n".encode("ascii"),
        )

    _publish_directory(target, build)
    return digest


def load_v5_verified_copy_receipt(
    receipt_path: str | Path,
) -> tuple[Mapping[str, object], str]:
    unresolved = Path(receipt_path)
    if unresolved.is_symlink():
        raise ValueError("V5 verified-copy receipt must not be a symlink")
    root = unresolved.resolve()
    children = tuple(root.iterdir())
    if (
        {path.name for path in children} != {"receipt.json", "receipt.json.sha256"}
        or any(path.is_symlink() or not path.is_file() for path in children)
    ):
        raise ValueError("V5 verified-copy receipt inventory is incomplete")
    document, digest = _read_canonical_document(
        root / "receipt.json",
        root / "receipt.json.sha256",
        label="V5 verified-copy receipt",
        sidecar_name="receipt.json",
    )
    top = _exact_keys(
        document,
        {"archive", "copyVerified", "format", "imported", "runNamespace", "source", "version"},
        "V5 verified-copy receipt",
    )
    if (
        top["format"] != V5_COPY_RECEIPT_FORMAT
        or top["version"] != V5_COPY_RECEIPT_VERSION
        or top["copyVerified"] is not True
        or not isinstance(top["runNamespace"], str)
    ):
        raise ValueError("V5 verified-copy receipt contract is incompatible")
    archive = _exact_keys(
        top["archive"],
        {"byteLength", "sha256", "spoolManifestSha256"},
        "V5 receipt archive",
    )
    _integer(archive["byteLength"], "receipt archive byte length", minimum=1)
    _sha(archive["sha256"], "receipt archive SHA")
    _sha(archive["spoolManifestSha256"], "receipt spool manifest SHA")
    imported = _exact_keys(
        top["imported"],
        {"decisionCount", "fileCount", "logicalBytes", "manifestSha256", "matchCount", "nonforcedDecisionCount"},
        "V5 receipt imported shard",
    )
    _sha(imported["manifestSha256"], "receipt imported manifest SHA")
    for name in (
        "decisionCount", "fileCount", "logicalBytes", "matchCount", "nonforcedDecisionCount"
    ):
        _integer(imported[name], f"receipt imported {name}", minimum=1)
    source = _exact_keys(
        top["source"],
        {"collectionPlanManifestSha256", "manifestSha256", "name", "shardIndex"},
        "V5 receipt source",
    )
    _sha(source["collectionPlanManifestSha256"], "receipt collection plan SHA")
    _sha(source["manifestSha256"], "receipt source manifest SHA")
    if not isinstance(source["name"], str):
        raise ValueError("receipt source shard name is invalid")
    _integer(source["shardIndex"], "receipt shard index")
    return top, digest


def _bind_receipt(
    receipt: Mapping[str, object],
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
    spool: Mapping[str, object] | None = None,
    spool_manifest_sha: str | None = None,
) -> None:
    source = receipt["source"]
    imported = receipt["imported"]
    archive = receipt["archive"]
    assert isinstance(source, Mapping)
    assert isinstance(imported, Mapping)
    assert isinstance(archive, Mapping)
    expected = {
        "collectionPlanManifestSha256": plan.manifest_sha256,
        "manifestSha256": imported["manifestSha256"],
        "name": shard.name,
        "shardIndex": shard.index,
    }
    if receipt["runNamespace"] != plan.run_namespace:
        raise ValueError("verified-copy receipt run namespace differs from plan")
    for name, value in expected.items():
        if source.get(name) != value:
            raise ValueError(f"verified-copy receipt differs from planned shard: {name}")
    if spool is not None:
        spool_source = spool["source"]
        spool_archive = spool["archive"]
        assert isinstance(spool_source, Mapping) and isinstance(spool_archive, Mapping)
        if (
            source["manifestSha256"] != spool_source["manifestSha256"]
            or archive["sha256"] != spool_archive["sha256"]
            or archive["byteLength"] != spool_archive["byteLength"]
            or archive["spoolManifestSha256"] != spool_manifest_sha
        ):
            raise ValueError("verified-copy receipt differs from spool archive")
        for name in (
            "decisionCount",
            "fileCount",
            "logicalBytes",
            "matchCount",
            "nonforcedDecisionCount",
        ):
            if imported.get(name) != spool_source.get(name):
                raise ValueError(
                    f"verified-copy imported counts differ from spool source: {name}"
                )


def import_v5_planned_shard_spool(
    plan_path: str | Path,
    *,
    shard_index: int,
    bundle_path: str | Path,
    canonical_root: str | Path,
    receipt_root: str | Path,
    minimum_free_after_import_bytes: int = DEFAULT_IMPORT_RESERVE_BYTES,
) -> Path:
    """Safely import a copied bundle and publish its verified-copy receipt."""

    plan, shard = _load_plan_shard(plan_path, shard_index)
    bundle_unresolved = Path(bundle_path)
    if bundle_unresolved.is_symlink():
        raise ValueError("incoming V5 spool bundle must not be a symlink")
    bundle = bundle_unresolved.resolve()
    if (
        bundle.parent.name != INCOMING_SPOOL_ROOT_NAME
        or bundle.parent.parent.name != plan.run_namespace
    ):
        raise ValueError("incoming spool bundle is outside its independent run namespace")
    spool, spool_manifest_sha = load_v5_spool_bundle(bundle)
    source = _bind_spool_to_plan(spool, plan, shard)
    expected_bundle_name = str(source["bundleName"])
    if bundle.name != expected_bundle_name:
        raise ValueError("incoming spool bundle directory name is non-canonical")
    canonical = _run_directory(
        canonical_root, plan.run_namespace, CANONICAL_ROOT_NAME
    )
    receipts = _run_directory(receipt_root, plan.run_namespace, RECEIPT_ROOT_NAME)
    canonical.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    target = canonical / shard.name
    receipt_target = receipts / _receipt_name(
        shard, str(source["manifestSha256"])
    )
    reserve = _integer(
        minimum_free_after_import_bytes,
        "minimum_free_after_import_bytes",
    )
    needed = int(source["logicalBytes"]) + reserve

    if _path_lexists(target):
        if target.is_symlink() or not target.is_dir():
            raise ValueError("canonical V5 shard target is not a real directory")
        imported_sha = verify_planned_shard(plan, shard, target)
        inventory = inventory_v5_training_shard(target)
        _inventory_matches_source(inventory, imported_sha, source)
    else:
        if shutil.disk_usage(canonical).free < needed:
            raise ValueError("insufficient canonical storage for import plus reserve")
        # Keep the private path deliberately short: Windows test/collector hosts
        # can otherwise hit legacy MAX_PATH on long belief-array filenames.
        operation = canonical / f".i-{uuid.uuid4().hex[:12]}"
        operation.mkdir()
        try:
            archive_record = spool["archive"]
            assert isinstance(archive_record, Mapping)
            members = archive_record["members"]
            assert isinstance(members, list)
            private_archive = operation / "a.tar.zst"
            _copy_verified_archive(
                bundle / "shard.tar.zst",
                private_archive,
                expected_bytes=int(archive_record["byteLength"]),
                expected_sha256=str(archive_record["sha256"]),
            )
            _validate_archive_against_members(
                private_archive,
                members,
                expected_top_directory=shard.name,
            )
            extraction = operation / "x"
            extraction.mkdir()
            _extract_archive(private_archive, extraction)
            extracted = extraction / shard.name
            if extracted.is_symlink() or not extracted.is_dir():
                raise ValueError("V5 archive did not extract one real planned shard")
            imported_sha = verify_planned_shard(plan, shard, extracted)
            inventory = inventory_v5_training_shard(extracted)
            _inventory_matches_source(inventory, imported_sha, source)
            _no_replace_directory_rename(extracted, target)
        finally:
            if operation.exists():
                shutil.rmtree(operation)

    receipt_document = _receipt_document(
        plan,
        shard,
        spool,
        spool_manifest_sha,
        inventory,
        imported_sha,
    )
    if _path_lexists(receipt_target):
        existing, _ = load_v5_verified_copy_receipt(receipt_target)
        if existing != receipt_document:
            raise ValueError("existing verified-copy receipt differs from imported shard")
    else:
        _publish_receipt(receipt_target, receipt_document)
    return receipt_target


def _verified_retirement_inputs(
    plan_path: str | Path,
    shard_index: int,
    receipt_path: str | Path,
    bundle_path: str | Path | None,
) -> tuple[
    V5CollectionPlan,
    V5PlannedShard,
    Mapping[str, object],
    Mapping[str, object] | None,
]:
    plan, shard = _load_plan_shard(plan_path, shard_index)
    receipt_unresolved = Path(receipt_path)
    if receipt_unresolved.is_symlink():
        raise ValueError("remote verified-copy receipt must not be a symlink")
    receipt_path_value = receipt_unresolved.resolve()
    if (
        receipt_path_value.parent.name != RECEIPT_ROOT_NAME
        or receipt_path_value.parent.parent.name != plan.run_namespace
    ):
        raise ValueError("verified-copy receipt is outside its independent run namespace")
    receipt, _ = load_v5_verified_copy_receipt(receipt_path_value)
    spool: Mapping[str, object] | None = None
    spool_sha: str | None = None
    if bundle_path is not None:
        spool, spool_sha = load_v5_spool_bundle(bundle_path)
        _bind_spool_to_plan(spool, plan, shard)
    _bind_receipt(receipt, plan, shard, spool, spool_sha)
    expected_receipt_name = _receipt_name(
        shard, str(receipt["source"]["manifestSha256"])  # type: ignore[index]
    )
    if receipt_path_value.name != expected_receipt_name:
        raise ValueError("verified-copy receipt directory name is non-canonical")
    return plan, shard, receipt, spool


def _retire_exact_directory(
    target: Path,
    root: Path,
    *,
    label: str,
    verify_callback: object,
) -> None:
    if target.parent != root or target.is_symlink() or not target.is_dir():
        raise ValueError(f"{label} retirement target is not one exact real directory")
    before = target.stat()
    lock = root / f".{target.name}.retire.lock"
    descriptor: int | None = None
    quarantine = root / f".{target.name}.retiring-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        after = target.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError(f"{label} retirement target changed during verification")
        _no_replace_directory_rename(target, quarantine)
        moved = quarantine.stat()
        if (before.st_dev, before.st_ino) != (moved.st_dev, moved.st_ino):
            raise RuntimeError(f"{label} changed during atomic quarantine")
        if not callable(verify_callback):
            raise TypeError("V5 retirement verifier must be callable")
        # Verify the exact inode tree that will be removed.  On failure the
        # quarantine remains intact and recoverable; no unverified bytes are
        # deleted and no newly-created target can be touched.
        verify_callback(quarantine)
        shutil.rmtree(quarantine)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def retire_v5_verified_raw_shard(
    plan_path: str | Path,
    *,
    shard_index: int,
    raw_root: str | Path,
    receipt_path: str | Path,
    bundle_path: str | Path,
) -> Path:
    """Delete only the exact raw shard proven copied and imported elsewhere."""

    plan, shard, receipt, spool = _verified_retirement_inputs(
        plan_path, shard_index, receipt_path, bundle_path
    )
    assert spool is not None
    raw = _run_directory(raw_root, plan.run_namespace, RAW_ROOT_NAME)
    target = planned_shard_path(raw, shard)
    source = spool["source"]
    assert isinstance(source, Mapping)

    def verify_again(candidate: Path) -> None:
        raw_sha = verify_planned_shard(plan, shard, candidate)
        inventory = inventory_v5_training_shard(candidate)
        _inventory_matches_source(inventory, raw_sha, source)
        if raw_sha != receipt["source"]["manifestSha256"]:  # type: ignore[index]
            raise ValueError("raw V5 shard differs from verified-copy receipt")

    verify_again(target)
    _retire_exact_directory(
        target,
        raw,
        label="raw V5 shard",
        verify_callback=verify_again,
    )
    return target


def retire_v5_verified_spool_bundle(
    plan_path: str | Path,
    *,
    shard_index: int,
    raw_root: str | Path,
    spool_root: str | Path,
    bundle_path: str | Path,
    receipt_path: str | Path,
) -> Path:
    """Delete only the exact remote spool bundle bound by an import receipt."""

    plan, shard, receipt, spool = _verified_retirement_inputs(
        plan_path, shard_index, receipt_path, bundle_path
    )
    assert spool is not None
    raw = _run_directory(raw_root, plan.run_namespace, RAW_ROOT_NAME)
    raw_target = planned_shard_path(raw, shard)
    quarantines = tuple(raw.glob(f".{shard.name}.retiring-*"))
    if _path_lexists(raw_target) or quarantines:
        raise ValueError(
            "raw V5 shard must be completely receipt-retired before its spool bundle"
        )
    root = _run_directory(spool_root, plan.run_namespace, SPOOL_ROOT_NAME)
    source = spool["source"]
    assert isinstance(source, Mapping)
    target = Path(bundle_path).resolve()
    expected = root / _spool_name(shard, str(source["manifestSha256"]))
    if target != expected:
        raise ValueError("spool retirement target is not the exact planned bundle")
    if source["manifestSha256"] != receipt["source"]["manifestSha256"]:  # type: ignore[index]
        raise ValueError("spool bundle differs from verified-copy receipt")
    def verify_again(candidate: Path) -> None:
        refreshed, refreshed_sha = load_v5_spool_bundle(candidate)
        _bind_spool_to_plan(refreshed, plan, shard)
        _bind_receipt(receipt, plan, shard, refreshed, refreshed_sha)

    verify_again(target)
    _retire_exact_directory(
        target,
        root,
        label="V5 spool bundle",
        verify_callback=verify_again,
    )
    return target


def _json_summary(value: Mapping[str, object]) -> None:
    sys.stdout.write(_canonical_json_bytes(dict(value)).decode("ascii"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="verify and spool one planned raw shard")
    export.add_argument("--plan", required=True)
    export.add_argument("--shard-index", type=int, required=True)
    export.add_argument("--raw-root", required=True)
    export.add_argument("--spool-root", required=True)
    export.add_argument("--zstd-level", type=int, default=3)
    export.add_argument(
        "--minimum-free-after-bytes", type=int, default=DEFAULT_SPOOL_RESERVE_BYTES
    )
    imported = commands.add_parser("import", help="safely import a copied spool bundle")
    imported.add_argument("--plan", required=True)
    imported.add_argument("--shard-index", type=int, required=True)
    imported.add_argument("--bundle", required=True)
    imported.add_argument("--canonical-root", required=True)
    imported.add_argument("--receipt-root", required=True)
    imported.add_argument(
        "--minimum-free-after-bytes", type=int, default=DEFAULT_IMPORT_RESERVE_BYTES
    )
    raw = commands.add_parser("retire-raw", help="retire receipt-proven remote raw shard")
    raw.add_argument("--plan", required=True)
    raw.add_argument("--shard-index", type=int, required=True)
    raw.add_argument("--raw-root", required=True)
    raw.add_argument("--bundle", required=True)
    raw.add_argument("--receipt", required=True)
    spool = commands.add_parser(
        "retire-spool", help="retire receipt-proven remote spool bundle"
    )
    spool.add_argument("--plan", required=True)
    spool.add_argument("--shard-index", type=int, required=True)
    spool.add_argument("--raw-root", required=True)
    spool.add_argument("--spool-root", required=True)
    spool.add_argument("--bundle", required=True)
    spool.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "export":
        target = export_v5_planned_shard_spool(
            arguments.plan,
            shard_index=arguments.shard_index,
            raw_root=arguments.raw_root,
            spool_root=arguments.spool_root,
            zstd_level=arguments.zstd_level,
            minimum_free_after_export_bytes=arguments.minimum_free_after_bytes,
        )
    elif arguments.command == "import":
        target = import_v5_planned_shard_spool(
            arguments.plan,
            shard_index=arguments.shard_index,
            bundle_path=arguments.bundle,
            canonical_root=arguments.canonical_root,
            receipt_root=arguments.receipt_root,
            minimum_free_after_import_bytes=arguments.minimum_free_after_bytes,
        )
    elif arguments.command == "retire-raw":
        target = retire_v5_verified_raw_shard(
            arguments.plan,
            shard_index=arguments.shard_index,
            raw_root=arguments.raw_root,
            bundle_path=arguments.bundle,
            receipt_path=arguments.receipt,
        )
    else:
        target = retire_v5_verified_spool_bundle(
            arguments.plan,
            shard_index=arguments.shard_index,
            raw_root=arguments.raw_root,
            spool_root=arguments.spool_root,
            bundle_path=arguments.bundle,
            receipt_path=arguments.receipt,
        )
    _json_summary({"command": arguments.command, "target": str(target)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_ROOT_NAME",
    "DEFAULT_IMPORT_RESERVE_BYTES",
    "DEFAULT_SPOOL_RESERVE_BYTES",
    "INCOMING_SPOOL_ROOT_NAME",
    "RAW_ROOT_NAME",
    "RECEIPT_ROOT_NAME",
    "SPOOL_ROOT_NAME",
    "V5_ARCHIVE_CONTRACT",
    "V5_COPY_RECEIPT_FORMAT",
    "V5_COPY_RECEIPT_VERSION",
    "V5_SPOOL_FORMAT",
    "V5_SPOOL_VERSION",
    "export_v5_planned_shard_spool",
    "import_v5_planned_shard_spool",
    "load_v5_spool_bundle",
    "load_v5_verified_copy_receipt",
    "main",
    "retire_v5_verified_raw_shard",
    "retire_v5_verified_spool_bundle",
    "validate_v5_archive_member_names",
]
