from __future__ import annotations

"""Immutable V5 certification and one-shot final-holdout promotion workflow."""

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import tempfile
from typing import Iterator, Mapping, Sequence
import uuid

from v5_evaluate import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    EXACT_GATES,
    FINAL_MATCH_COUNTS,
    PLAYER_COUNTS,
    SCREENING_MATCH_COUNTS,
    derive_v5_evaluation_seed,
    validate_v5_evaluation_report,
)
from v5_export import canonical_json_bytes, v5_actor_bundle_digests
from v5_provenance import validate_v5_evaluation_provenance


V5_PROMOTION_PLAN_FORMAT = "dalmuti-v5-final-holdout-reservation"
V5_PROMOTION_PLAN_VERSION = 1
V5_CONSUMPTION_RECEIPT_FORMAT = "dalmuti-v5-final-holdout-consumption"
V5_CONSUMPTION_RECEIPT_VERSION = 1
V5_FINAL_CLAIM_FORMAT = "dalmuti-v5-final-holdout-shard-start-claim"
V5_FINAL_CLAIM_VERSION = 1
V5_CERTIFICATION_RESERVATION_FORMAT = (
    "dalmuti-v5-certification-execution-reservation"
)
V5_CERTIFICATION_RESERVATION_VERSION = 1
V5_SCREENING_RESERVATION_FORMAT = "dalmuti-v5-screening-execution-reservation"
V5_SCREENING_RESERVATION_VERSION = 1
V5_PROMOTION_LOCK_FORMAT = "dalmuti-v5-promotion-registry-lock"
V5_PROMOTION_LOCK_VERSION = 1
V5_PROMOTION_LOCK_RECOVERY_FORMAT = (
    "dalmuti-v5-promotion-registry-lock-recovery"
)
V5_PROMOTION_LOCK_RECOVERY_VERSION = 1
V5_FIRST_FINAL_SEED_BASE = 900_000_001
V5_FINAL_SEED_STEP = 20_000_000
V5_CERTIFICATION_REPORT_COUNT = 2
V5_DEVELOPMENT_GATES = {
    "minMeanChipDifference": 0.30,
    "minCluster95LowerBound": 0.20,
    "minPairwiseRate": 0.57,
}
V5_ACTOR_IDENTITY_KEYS = {
    "actorSha256",
    "manifestSha256",
    "tensorStateSha256",
    "publicContractSha256",
    "policyNumericsSha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_UTC_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_canonical_object(path: Path, label: str) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value, _sha256_bytes(raw)


def _fsync_directory(directory: Path) -> None:
    """Persist directory entries on hosts exposing a directory-fsync primitive."""

    if os.name == "nt":
        # Python has no portable Windows directory fsync.  Every file is still
        # flushed before publication; the Linux production host additionally
        # receives the directory-entry durability barrier.
        return
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary_prefix(path: Path) -> str:
    return f".{path.name}.publish-"


def _write_exclusive(path: Path, payload: bytes) -> os.stat_result:
    """Publish fsynced bytes by same-directory, no-replace hardlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=_publish_temporary_prefix(path),
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary_identity = temporary.stat()
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"immutable V5 file already exists: {path}")
        _fsync_directory(path.parent)
        final_identity = path.stat()
        if (
            temporary_identity.st_dev != final_identity.st_dev
            or temporary_identity.st_ino != final_identity.st_ino
        ):
            raise RuntimeError("promotion publication target is not its source link")
        temporary.unlink()
        _fsync_directory(path.parent)
        return final_identity
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)


def _write_canonical_with_sidecar(path: Path, value: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(dict(value))
    digest = _sha256_bytes(raw)
    sidecar = path.with_name(path.name + ".sha256")
    _write_exclusive(path, raw)
    _write_exclusive(sidecar, f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _verify_sidecar(path: Path, digest: str) -> None:
    expected = f"{digest}  {path.name}\n".encode("ascii")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        try:
            _write_exclusive(sidecar, expected)
        except FileExistsError:
            pass
    if sidecar.read_bytes() != expected:
        raise ValueError(f"{path.name} checksum sidecar does not match")


def _host_boot_id() -> str:
    linux = Path("/proc/sys/kernel/random/boot_id")
    if linux.is_file():
        value = linux.read_text(encoding="ascii").strip().lower()
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise RuntimeError("Linux kernel boot_id is invalid") from error
    if os.name == "nt":
        class _GUID(ctypes.Structure):
            _fields_ = [
                ("data1", ctypes.c_uint32),
                ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8),
            ]

        class _BOOT_ENVIRONMENT(ctypes.Structure):
            _fields_ = [
                ("identifier", _GUID),
                ("firmwareType", ctypes.c_uint32),
                ("bootFlags", ctypes.c_uint64),
            ]

        value = _BOOT_ENVIRONMENT()
        returned = ctypes.c_ulong()
        status = ctypes.windll.ntdll.NtQuerySystemInformation(  # type: ignore[attr-defined]
            90,
            ctypes.byref(value),
            ctypes.sizeof(value),
            ctypes.byref(returned),
        )
        if status != 0:
            raise RuntimeError(
                f"Windows boot identifier query failed with NTSTATUS {status}"
            )
        raw = (
            int(value.identifier.data1).to_bytes(4, "little")
            + int(value.identifier.data2).to_bytes(2, "little")
            + int(value.identifier.data3).to_bytes(2, "little")
            + bytes(value.identifier.data4)
        )
        return str(uuid.UUID(bytes_le=raw))
    raise RuntimeError("this platform does not expose a stable kernel boot identifier")


def _process_start_ticks(pid: int) -> int | None:
    if type(pid) is not int or pid < 1:
        return None
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            raw = stat.read_text(encoding="ascii")
            remainder = raw[raw.rindex(")") + 2 :].split()
            # remainder[0] is proc stat field 3; starttime is field 22.
            return int(remainder[19])
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            return None
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, pid
        )
        if not handle:
            return None
        try:
            creation = ctypes.c_uint64()
            exit_time = ctypes.c_uint64()
            kernel = ctypes.c_uint64()
            user = ctypes.c_uint64()
            success = ctypes.windll.kernel32.GetProcessTimes(  # type: ignore[attr-defined]
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            return int(creation.value) if success else None
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    return None


def _process_may_exist(pid: int) -> bool:
    if Path("/proc").is_dir():
        return Path(f"/proc/{pid}").exists()
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        # ERROR_INVALID_PARAMETER means there is no process with this PID.
        return int(ctypes.windll.kernel32.GetLastError()) != 87  # type: ignore[attr-defined]
    return True


def _registry_identity(registry: Path) -> dict[str, object]:
    resolved = registry.resolve()
    identity = resolved.stat()
    return {
        "device": int(identity.st_dev),
        "inode": int(identity.st_ino),
        "path": os.path.normcase(str(resolved)),
    }


def _validate_registry_identity(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"device", "inode", "path"}
        or type(value.get("device")) is not int
        or int(value["device"]) < 0
        or type(value.get("inode")) is not int
        or int(value["inode"]) < 0
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or not Path(str(value["path"])).is_absolute()
        or any(ord(character) < 32 for character in str(value["path"]))
    ):
        raise ValueError("promotion registry identity contract drifted")
    return dict(value)


def _validate_process_identity(value: object, label: str) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"bootId", "hostname", "pid", "processStartTicks"}
        or not isinstance(value.get("bootId"), str)
        or not value["bootId"]
        or len(str(value["bootId"])) > 128
        or any(ord(character) < 32 for character in str(value["bootId"]))
        or not isinstance(value.get("hostname"), str)
        or not value["hostname"]
        or len(str(value["hostname"])) > 255
        or any(ord(character) < 32 for character in str(value["hostname"]))
        or type(value.get("pid")) is not int
        or int(value["pid"]) < 1
        or type(value.get("processStartTicks")) is not int
        or int(value["processStartTicks"]) < 0
    ):
        raise ValueError(f"{label} process identity contract drifted")
    return dict(value)


def _current_process_identity(label: str) -> dict[str, object]:
    start = _process_start_ticks(os.getpid())
    if start is None:
        raise RuntimeError(f"could not bind {label} to its process-start identity")
    return {
        "bootId": _host_boot_id(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "processStartTicks": start,
    }


def _promotion_lock_document(registry: Path) -> dict[str, object]:
    process = _current_process_identity("promotion registry lock")
    return {
        "bootId": process["bootId"],
        "format": V5_PROMOTION_LOCK_FORMAT,
        "hostname": process["hostname"],
        "pid": process["pid"],
        "processStartTicks": process["processStartTicks"],
        "registryIdentity": _registry_identity(registry),
        "version": V5_PROMOTION_LOCK_VERSION,
    }


def _validate_promotion_lock_document(value: object) -> dict[str, object]:
    expected = {
        "bootId", "format", "hostname", "pid", "processStartTicks",
        "registryIdentity", "version",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("format") != V5_PROMOTION_LOCK_FORMAT
        or value.get("version") != V5_PROMOTION_LOCK_VERSION
    ):
        raise ValueError("promotion registry lock contract drifted")
    _validate_process_identity(
        {
            "bootId": value["bootId"],
            "hostname": value["hostname"],
            "pid": value["pid"],
            "processStartTicks": value["processStartTicks"],
        },
        "promotion registry lock",
    )
    _validate_registry_identity(value["registryIdentity"])
    return dict(value)


def _load_promotion_lock(path: Path) -> tuple[dict[str, object], bytes, str]:
    value, digest = _strict_canonical_object(path, "V5 promotion registry lock")
    verified = _validate_promotion_lock_document(value)
    return verified, canonical_json_bytes(verified), digest


def _valid_recovery_reason(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.strip() == value
        and 1 <= len(value) <= 240
        and all(ord(character) >= 32 for character in value)
    )


def _load_promotion_recovery_receipt(
    path: Path,
) -> tuple[dict[str, object], bytes]:
    value, _ = _strict_canonical_object(
        path, "V5 promotion registry-lock recovery receipt"
    )
    expected = {
        "format", "observedStaleEvidence", "observer", "oldLock",
        "oldLockSha256", "recoveredAt", "recoveryReason", "registryIdentity",
        "retiredLock", "version",
    }
    if (
        set(value) != expected
        or value.get("format") != V5_PROMOTION_LOCK_RECOVERY_FORMAT
        or value.get("version") != V5_PROMOTION_LOCK_RECOVERY_VERSION
        or value.get("observedStaleEvidence") not in {
            "process-missing", "pid-reused", "host-rebooted"
        }
        or not isinstance(value.get("oldLockSha256"), str)
        or _SHA256.fullmatch(str(value["oldLockSha256"])) is None
        or not isinstance(value.get("recoveredAt"), str)
        or _UTC_SECONDS.fullmatch(str(value["recoveredAt"])) is None
        or not _valid_recovery_reason(value.get("recoveryReason"))
        or not isinstance(value.get("retiredLock"), str)
    ):
        raise ValueError("promotion registry-lock recovery receipt contract drifted")
    old_lock = _validate_promotion_lock_document(value["oldLock"])
    observer = _validate_process_identity(value["observer"], "promotion recovery")
    registry_identity = _validate_registry_identity(value["registryIdentity"])
    old_sha = _sha256_bytes(canonical_json_bytes(old_lock))
    expected_retired = (
        ".v5-promotion-lock-recoveries/retired-locks/"
        f"{old_sha}.lock"
    )
    if (
        value["oldLockSha256"] != old_sha
        or registry_identity != old_lock["registryIdentity"]
        or value["retiredLock"] != expected_retired
        or observer["hostname"] != old_lock["hostname"]
    ):
        raise ValueError("promotion registry-lock recovery receipt bindings drifted")
    return dict(value), canonical_json_bytes(value)


def _recovery_pointer_paths(registry: Path) -> tuple[Path, Path]:
    return (
        registry / ".v5-promotion-lock-recovery-active.json",
        registry / ".v5-promotion-lock-recovery-complete.json",
    )


def _receipt_archive_path(registry: Path, lock_sha: str) -> Path:
    return registry / ".v5-promotion-lock-recoveries" / f"{lock_sha}.json"


def _retired_lock_path(registry: Path, lock_sha: str) -> Path:
    return (
        registry
        / ".v5-promotion-lock-recoveries"
        / "retired-locks"
        / f"{lock_sha}.lock"
    )


def _validate_recovery_artifacts(
    registry: Path, pointer: Path
) -> tuple[dict[str, object], bytes, Path, Path]:
    receipt, raw = _load_promotion_recovery_receipt(pointer)
    if receipt["registryIdentity"] != _registry_identity(registry):
        raise ValueError("promotion recovery belongs to another registry identity")
    old_lock = receipt["oldLock"]
    assert isinstance(old_lock, Mapping)
    if old_lock["hostname"] != socket.gethostname():
        raise RuntimeError(
            "promotion recovery belongs to another host; its evidence cannot be "
            "completed here"
        )
    lock_sha = str(receipt["oldLockSha256"])
    archive = _receipt_archive_path(registry, lock_sha)
    retired = _retired_lock_path(registry, lock_sha)
    archive_receipt, archive_raw = _load_promotion_recovery_receipt(archive)
    if archive_receipt != receipt or archive_raw != raw:
        raise ValueError("promotion recovery pointer differs from its archive")
    old_raw = canonical_json_bytes(receipt["oldLock"])  # type: ignore[arg-type]
    if retired.read_bytes() != old_raw:
        raise ValueError("retired promotion lock differs from its recovery receipt")
    return receipt, raw, archive, retired


def _clear_completed_recovery_pointers(registry: Path) -> None:
    active, complete = _recovery_pointer_paths(registry)
    if not active.exists() and not complete.exists():
        return
    if active.exists() and not complete.exists():
        raise RuntimeError("promotion lock recovery is still in progress")
    pointer = active if active.exists() else complete
    _, raw, _, _ = _validate_recovery_artifacts(registry, pointer)
    if complete.read_bytes() != raw:
        raise ValueError("promotion recovery completion pointer drifted")
    # Active first is intentional: a crash leaves a complete-only pointer,
    # which is still unambiguously safe to validate and clear on the next try.
    if active.exists():
        active.unlink()
        _fsync_directory(registry)
    complete.unlink()
    _fsync_directory(registry)


@contextmanager
def _registry_process_guard(registry: Path) -> Iterator[None]:
    """Crash-released cross-process guard for logical-lock state transitions."""

    guard = registry / ".v5-promotion.guard"
    descriptor = os.open(
        guard,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("promotion registry is active on this host") from error
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def _registry_lock(registry: Path) -> Iterator[None]:
    registry = registry.resolve()
    registry.mkdir(parents=True, exist_ok=True)
    lock = registry / ".v5-promotion.lock"
    with _registry_process_guard(registry):
        _clear_completed_recovery_pointers(registry)
        document = _promotion_lock_document(registry)
        payload = canonical_json_bytes(document)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=_publish_temporary_prefix(lock),
            suffix=".tmp",
            dir=registry,
        )
        temporary = Path(temporary_name)
        published_identity: os.stat_result | None = None
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            temporary_identity = temporary.stat()
            try:
                os.link(temporary, lock)
            except FileExistsError:
                raise FileExistsError(
                    f"immutable V5 file already exists: {lock}"
                )
            # Ownership begins at the no-replace link.  Every later failure,
            # including a durability barrier or temp cleanup failure, enters
            # the exact-identity release path below.
            published_identity = lock.stat()
            if (
                temporary_identity.st_dev != published_identity.st_dev
                or temporary_identity.st_ino != published_identity.st_ino
            ):
                raise RuntimeError(
                    "promotion registry lock is not linked to its fsynced source"
                )
            _fsync_directory(registry)
            temporary.unlink()
            _fsync_directory(registry)
            yield
        finally:
            try:
                if published_identity is not None:
                    current, raw, _ = _load_promotion_lock(lock)
                    path_identity = lock.stat()
                    if (
                        current != document
                        or raw != payload
                        or published_identity.st_dev != path_identity.st_dev
                        or published_identity.st_ino != path_identity.st_ino
                    ):
                        raise RuntimeError(
                            "promotion registry lock changed while its process was active"
                        )
                    lock.unlink()
                    _fsync_directory(registry)
            finally:
                if temporary.exists():
                    temporary.unlink()
                    _fsync_directory(registry)


def recover_v5_promotion_lock(
    registry_directory: str | Path,
    *,
    recovery_reason: str,
) -> dict[str, object]:
    """Evidence-retire one provably stale promotion lock without PID-reuse risk."""

    if not _valid_recovery_reason(recovery_reason):
        raise ValueError("recovery reason must be 1..240 printable characters")
    registry = Path(registry_directory).resolve()
    registry.mkdir(parents=True, exist_ok=True)
    lock = registry / ".v5-promotion.lock"
    active, complete = _recovery_pointer_paths(registry)
    with _registry_process_guard(registry):
        if not lock.exists():
            pointer = active if active.exists() else complete if complete.exists() else None
            if pointer is None:
                raise FileNotFoundError("there is no promotion registry lock to recover")
            receipt, raw, archive, retired = _validate_recovery_artifacts(
                registry, pointer
            )
            if receipt["recoveryReason"] != recovery_reason:
                raise ValueError("existing promotion recovery used another reason")
            if active.exists() and active.read_bytes() != raw:
                raise ValueError("promotion recovery active pointer drifted")
            if not complete.exists():
                try:
                    os.link(archive, complete)
                    _fsync_directory(registry)
                except FileExistsError:
                    if complete.read_bytes() != raw:
                        raise FileExistsError(
                            "promotion recovery completion pointer already exists"
                        )
            elif complete.read_bytes() != raw:
                raise ValueError("promotion recovery completion pointer drifted")
            return {
                "observedStaleEvidence": receipt["observedStaleEvidence"],
                "oldLockSha256": receipt["oldLockSha256"],
                "receipt": str(archive),
                "resumedRecovery": True,
                "retiredLock": str(retired),
            }

        value, raw, lock_sha = _load_promotion_lock(lock)
        registry_identity = _registry_identity(registry)
        if value["registryIdentity"] != registry_identity:
            raise ValueError("stale promotion lock belongs to another registry identity")
        observer = _current_process_identity("promotion lock recovery")
        if value["hostname"] != observer["hostname"]:
            raise RuntimeError(
                "promotion registry lock belongs to another host; remote process "
                "inactivity cannot be proven here"
            )
        same_boot = value["bootId"] == observer["bootId"]
        observed_start = (
            _process_start_ticks(int(value["pid"])) if same_boot else None
        )
        if same_boot and observed_start == value["processStartTicks"]:
            raise RuntimeError(
                "promotion registry lock still belongs to the active process"
            )
        if (
            same_boot
            and observed_start is None
            and _process_may_exist(int(value["pid"]))
        ):
            raise RuntimeError(
                "promotion registry lock PID may still be active but its "
                "process-start identity could not be read"
            )
        stale_evidence = (
            "process-missing"
            if same_boot and observed_start is None
            else "pid-reused"
            if same_boot
            else "host-rebooted"
        )
        archive = _receipt_archive_path(registry, lock_sha)
        retired = _retired_lock_path(registry, lock_sha)
        archive.parent.mkdir(parents=True, exist_ok=True)
        retired.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "format": V5_PROMOTION_LOCK_RECOVERY_FORMAT,
            "observedStaleEvidence": stale_evidence,
            "observer": observer,
            "oldLock": value,
            "oldLockSha256": lock_sha,
            "recoveredAt": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z"),
            "recoveryReason": recovery_reason,
            "registryIdentity": registry_identity,
            "retiredLock": str(retired.relative_to(registry)).replace("\\", "/"),
            "version": V5_PROMOTION_LOCK_RECOVERY_VERSION,
        }
        receipt_payload = canonical_json_bytes(document)
        resumed = False
        try:
            _write_exclusive(archive, receipt_payload)
        except FileExistsError:
            existing, existing_raw = _load_promotion_recovery_receipt(archive)
            if (
                existing_raw != receipt_payload
                and (
                    existing.get("oldLock") != value
                    or existing.get("oldLockSha256") != lock_sha
                    or existing.get("registryIdentity") != registry_identity
                    or existing.get("recoveryReason") != recovery_reason
                    or existing.get("retiredLock") != document["retiredLock"]
                )
            ):
                raise ValueError(
                    "existing promotion recovery receipt belongs to another lock"
                )
            document = existing
            receipt_payload = existing_raw
            resumed = True
        try:
            os.link(archive, active)
            _fsync_directory(registry)
        except FileExistsError:
            if active.read_bytes() != receipt_payload:
                raise FileExistsError(
                    "promotion recovery active pointer already exists"
                )
            resumed = True
        refreshed, refreshed_raw, refreshed_sha = _load_promotion_lock(lock)
        if refreshed != value or refreshed_raw != raw or refreshed_sha != lock_sha:
            raise RuntimeError("promotion registry lock changed before retirement")
        try:
            os.link(lock, retired)
            _fsync_directory(retired.parent)
        except FileExistsError:
            if not resumed or retired.read_bytes() != raw:
                raise FileExistsError("promotion retired-lock target already exists")
        if retired.read_bytes() != raw:
            raise RuntimeError("retired promotion lock differs from its source")
        source_identity = lock.stat()
        retired_identity = retired.stat()
        if (
            source_identity.st_dev != retired_identity.st_dev
            or source_identity.st_ino != retired_identity.st_ino
        ):
            raise RuntimeError("retired promotion lock is not a hardlink to its source")
        refreshed, refreshed_raw, refreshed_sha = _load_promotion_lock(lock)
        if refreshed != value or refreshed_raw != raw or refreshed_sha != lock_sha:
            raise RuntimeError("promotion registry lock changed after retirement link")
        lock.unlink()
        _fsync_directory(registry)
        try:
            os.link(archive, complete)
            _fsync_directory(registry)
        except FileExistsError:
            if complete.read_bytes() != receipt_payload:
                raise FileExistsError(
                    "promotion recovery completion pointer already exists"
                )
        return {
            "observedStaleEvidence": document["observedStaleEvidence"],
            "oldLockSha256": lock_sha,
            "receipt": str(archive),
            "resumedRecovery": resumed,
            "retiredLock": str(retired),
        }


def _exact_actor_identity(actor_bundle: str | Path) -> dict[str, str]:
    identity = v5_actor_bundle_digests(actor_bundle)
    if set(identity) != V5_ACTOR_IDENTITY_KEYS:
        raise ValueError("Actor bundle did not provide the exact five V5 digests")
    return {
        name: _require_sha(identity[name], f"Actor identity {name}")
        for name in sorted(V5_ACTOR_IDENTITY_KEYS)
    }


def _validated_evaluation_seed_set(
    family_id: str, seed_base: int, plan: Mapping[int, int]
) -> set[int]:
    expected = sum(int(matches) for matches in plan.values())
    seeds = {
        derive_v5_evaluation_seed(family_id, seed_base, player, match_index)
        for player, matches in plan.items()
        for match_index in range(matches)
    }
    if len(seeds) != expected:
        raise RuntimeError("evaluation seed schedule contains an internal collision")
    return seeds


def _certification_coordinates_for_model(
    model: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    material = canonical_json_bytes(dict(model))
    prefix = str(model["tensorStateSha256"])[:12]
    selected: list[dict[str, object]] = []
    selected_sets: list[set[int]] = []
    for ordinal, label in enumerate(("a", "b")):
        for attempt in range(1024):
            digest = hashlib.sha256(
                b"DALMUTI-V5-CERTIFICATION-COORDINATE\0"
                + material
                + bytes((ordinal,))
                + attempt.to_bytes(4, "little")
            ).digest()
            seed_base = 1 + int.from_bytes(digest[:8], "little") % 800_000_000
            if any(item["seedBase"] == seed_base for item in selected):
                continue
            family_id = f"v5-certification-{prefix}-{label}"
            try:
                seeds = _validated_evaluation_seed_set(
                    family_id, seed_base, SCREENING_MATCH_COUNTS
                )
            except RuntimeError:
                continue
            if any(seeds & prior for prior in selected_sets):
                continue
            selected.append({
                "bootstrapResamples": DEFAULT_BOOTSTRAP_RESAMPLES,
                "familyId": family_id,
                "matchPlan": {
                    str(player): SCREENING_MATCH_COUNTS[player]
                    for player in PLAYER_COUNTS
                },
                "seedBase": seed_base,
            })
            selected_sets.append(seeds)
            break
        else:
            raise RuntimeError("could not derive disjoint certification coordinates")
    return selected[0], selected[1]


def v5_certification_coordinates(
    actor_bundle: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the only two certification schedules eligible for this Actor."""

    return _certification_coordinates_for_model(_exact_actor_identity(actor_bundle))


def _validated_certification_execution_provenance(
    value: Mapping[str, object],
) -> dict[str, object]:
    verified = validate_v5_evaluation_provenance(value)
    artifacts = verified["artifacts"]
    assert isinstance(artifacts, Mapping)
    if any(
        artifacts.get(name) is None
        for name in ("gitBundleSha256", "sourceSnapshotSha256")
    ):
        raise ValueError(
            "certification execution provenance requires preserved source artifacts"
        )
    return verified


def _functional_policy_identity(model: Mapping[str, str]) -> dict[str, str]:
    return {
        "policyNumericsSha256": _require_sha(
            model.get("policyNumericsSha256"), "screening policy numerics"
        ),
        "publicContractSha256": _require_sha(
            model.get("publicContractSha256"), "screening public contract"
        ),
        "tensorStateSha256": _require_sha(
            model.get("tensorStateSha256"), "screening Actor tensor state"
        ),
    }


def _screening_coordinate_for_model(model: Mapping[str, str]) -> dict[str, object]:
    functional = _functional_policy_identity(model)
    material = canonical_json_bytes(functional)
    family_id = f"v5-screening-{functional['tensorStateSha256'][:12]}"
    certification_sets = [
        _validated_evaluation_seed_set(
            str(coordinate["familyId"]),
            int(coordinate["seedBase"]),
            SCREENING_MATCH_COUNTS,
        )
        for coordinate in _certification_coordinates_for_model(model)
    ]
    for attempt in range(4096):
        digest = hashlib.sha256(
            b"DALMUTI-V5-SCREENING-COORDINATE\0"
            + material
            + attempt.to_bytes(4, "little")
        ).digest()
        seed_base = 800_000_001 + int.from_bytes(digest[:8], "little") % 80_000_000
        try:
            seeds = _validated_evaluation_seed_set(
                family_id, seed_base, SCREENING_MATCH_COUNTS
            )
        except RuntimeError:
            continue
        if any(seeds & certification for certification in certification_sets):
            continue
        return {
            "bootstrapResamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "familyId": family_id,
            "matchPlan": {
                str(player): SCREENING_MATCH_COUNTS[player]
                for player in PLAYER_COUNTS
            },
            "seedBase": seed_base,
        }
    raise RuntimeError("could not derive a disjoint Actor screening coordinate")


def _screening_reservation_id(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"DALMUTI-V5-SCREENING-EXECUTION\0"
        + canonical_json_bytes(dict(body))
    )


def _validate_screening_execution_reservation(
    value: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "coordinate", "evaluationProvenance", "evaluationProvenanceSha256",
        "format", "functionalPolicyIdentity", "model", "outputPath",
        "reservationId", "version",
    }
    if (
        set(value) != expected
        or value.get("format") != V5_SCREENING_RESERVATION_FORMAT
        or value.get("version") != V5_SCREENING_RESERVATION_VERSION
    ):
        raise ValueError("V5 screening execution reservation contract drifted")
    model = value.get("model")
    provenance = value.get("evaluationProvenance")
    coordinate = value.get("coordinate")
    if (
        not isinstance(model, dict)
        or set(model) != V5_ACTOR_IDENTITY_KEYS
        or not isinstance(provenance, Mapping)
        or not isinstance(coordinate, Mapping)
    ):
        raise ValueError("V5 screening execution reservation structure drifted")
    normalized_model = {
        name: _require_sha(model[name], f"screening reservation model {name}")
        for name in sorted(V5_ACTOR_IDENTITY_KEYS)
    }
    verified_provenance = _validated_certification_execution_provenance(provenance)
    provenance_sha = _require_sha(
        value.get("evaluationProvenanceSha256"),
        "screening execution provenance SHA",
    )
    expected_coordinate = _screening_coordinate_for_model(normalized_model)
    functional = _functional_policy_identity(normalized_model)
    base_body: dict[str, object] = {
        "coordinate": expected_coordinate,
        "evaluationProvenance": verified_provenance,
        "evaluationProvenanceSha256": provenance_sha,
        "format": V5_SCREENING_RESERVATION_FORMAT,
        "functionalPolicyIdentity": functional,
        "model": normalized_model,
        "version": V5_SCREENING_RESERVATION_VERSION,
    }
    reservation_id = _screening_reservation_id(base_body)
    if (
        provenance_sha != verified_provenance["provenanceSha256"]
        or value.get("functionalPolicyIdentity") != functional
        or dict(coordinate) != expected_coordinate
        or value.get("reservationId") != reservation_id
        or value.get("outputPath")
        != f"screening-results/{reservation_id}/report.json"
    ):
        raise ValueError("V5 screening execution reservation identity drifted")
    return dict(value)


def load_v5_screening_execution_reservation(
    path: str | Path,
) -> dict[str, object]:
    reservation_path = Path(path).resolve()
    value, digest = _strict_canonical_object(
        reservation_path, "V5 screening execution reservation"
    )
    _verify_sidecar(reservation_path, digest)
    verified = _validate_screening_execution_reservation(value)
    if (
        reservation_path.parent.name != "screening-reservations"
        or reservation_path.name != f"{verified['reservationId']}.json"
    ):
        raise ValueError("V5 screening reservation is outside its registry")
    return verified


def reserve_v5_screening_execution(
    registry_directory: str | Path,
    actor_bundle: str | Path,
    evaluation_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Reserve the frozen functional Actor's only production screening run."""

    registry = Path(registry_directory).resolve()
    model = _exact_actor_identity(actor_bundle)
    provenance = _validated_certification_execution_provenance(
        evaluation_provenance
    )
    coordinate = _screening_coordinate_for_model(model)
    functional = _functional_policy_identity(model)
    base_body: dict[str, object] = {
        "coordinate": coordinate,
        "evaluationProvenance": provenance,
        "evaluationProvenanceSha256": provenance["provenanceSha256"],
        "format": V5_SCREENING_RESERVATION_FORMAT,
        "functionalPolicyIdentity": functional,
        "model": model,
        "version": V5_SCREENING_RESERVATION_VERSION,
    }
    reservation_id = _screening_reservation_id(base_body)
    document = _validate_screening_execution_reservation({
        **base_body,
        "outputPath": f"screening-results/{reservation_id}/report.json",
        "reservationId": reservation_id,
    })
    directory = registry / "screening-reservations"
    path = directory / f"{reservation_id}.json"
    with _registry_lock(registry):
        directory.mkdir(parents=True, exist_ok=True)
        for existing_path in sorted(directory.glob("*.json")):
            existing = load_v5_screening_execution_reservation(existing_path)
            existing_functional = existing["functionalPolicyIdentity"]
            assert isinstance(existing_functional, Mapping)
            if (
                existing_functional.get("tensorStateSha256")
                == functional["tensorStateSha256"]
            ):
                if existing == document and existing_path == path:
                    return {
                        "reservation": existing,
                        "reservationPath": str(existing_path),
                        "reservationSha256": _sha256_file(existing_path),
                    }
                raise ValueError(
                    "this frozen functional Actor already has a screening reservation"
                )
        digest = _write_canonical_with_sidecar(path, document)
    return {
        "reservation": document,
        "reservationPath": str(path),
        "reservationSha256": digest,
    }


def _validate_certification_execution_reservation(
    value: Mapping[str, object],
) -> dict[str, object]:
    if (
        set(value)
        != {
            "coordinates",
            "evaluationProvenance",
            "evaluationProvenanceSha256",
            "format",
            "model",
            "reservationId",
            "screening",
            "version",
        }
        or value.get("format") != V5_CERTIFICATION_RESERVATION_FORMAT
        or value.get("version") != V5_CERTIFICATION_RESERVATION_VERSION
    ):
        raise ValueError("V5 certification execution reservation contract drifted")
    model = value.get("model")
    provenance = value.get("evaluationProvenance")
    coordinates = value.get("coordinates")
    screening = value.get("screening")
    if (
        not isinstance(model, dict)
        or set(model) != V5_ACTOR_IDENTITY_KEYS
        or not isinstance(provenance, Mapping)
        or not isinstance(coordinates, list)
        or len(coordinates) != V5_CERTIFICATION_REPORT_COUNT
        or not isinstance(screening, Mapping)
        or set(screening) != {
            "evaluationProvenanceSha256", "reportSha256", "reservationId",
            "reservationSha256",
        }
    ):
        raise ValueError("V5 certification execution reservation structure drifted")
    for name, digest in model.items():
        _require_sha(digest, f"certification reservation model {name}")
    verified_provenance = _validated_certification_execution_provenance(provenance)
    provenance_sha = _require_sha(
        value.get("evaluationProvenanceSha256"),
        "certification execution provenance SHA",
    )
    if verified_provenance["provenanceSha256"] != provenance_sha:
        raise ValueError("certification reservation provenance SHA drifted")
    for name in (
        "evaluationProvenanceSha256", "reportSha256", "reservationId",
        "reservationSha256",
    ):
        _require_sha(screening.get(name), f"certification screening {name}")
    if screening.get("evaluationProvenanceSha256") != provenance_sha:
        raise ValueError("certification and screening execution provenance differ")
    expected_coordinates = _certification_coordinates_for_model(model)  # type: ignore[arg-type]
    base_body: dict[str, object] = {
        "coordinates": [
            {
                "familyId": coordinate["familyId"],
                "matchPlan": coordinate["matchPlan"],
                "seedBase": coordinate["seedBase"],
            }
            for coordinate in expected_coordinates
        ],
        "evaluationProvenance": verified_provenance,
        "evaluationProvenanceSha256": provenance_sha,
        "format": V5_CERTIFICATION_RESERVATION_FORMAT,
        "model": model,
        "screening": dict(screening),
        "version": V5_CERTIFICATION_RESERVATION_VERSION,
    }
    reservation_id = _certification_reservation_id(base_body)
    expected_records = [
        {
            **base_body["coordinates"][index],  # type: ignore[index]
            "label": label,
            "outputPath": (
                f"certification-results/{reservation_id}/{label}.json"
            ),
        }
        for index, label in enumerate(("a", "b"))
    ]
    if value.get("reservationId") != reservation_id or coordinates != expected_records:
        raise ValueError("V5 certification execution reservation identity drifted")
    return dict(value)


def load_v5_certification_execution_reservation(
    path: str | Path,
) -> dict[str, object]:
    reservation_path = Path(path).resolve()
    value, digest = _strict_canonical_object(
        reservation_path, "V5 certification execution reservation"
    )
    _verify_sidecar(reservation_path, digest)
    verified = _validate_certification_execution_reservation(value)
    if (
        reservation_path.parent.name != "certification-reservations"
        or reservation_path.name != f"{verified['reservationId']}.json"
    ):
        raise ValueError(
            "V5 certification execution reservation is outside its registry"
        )
    return verified


def reserve_v5_certification_execution(
    registry_directory: str | Path,
    actor_bundle: str | Path,
    evaluation_provenance: Mapping[str, object],
    *,
    screening_reservation: str | Path,
    screening_report: str | Path,
) -> dict[str, object]:
    """Reserve certification only after the Actor's one-shot screening passes."""

    registry = Path(registry_directory).resolve()
    model = _exact_actor_identity(actor_bundle)
    provenance = _validated_certification_execution_provenance(
        evaluation_provenance
    )
    screening_path = Path(screening_reservation).resolve()
    screening_execution = load_v5_screening_execution_reservation(screening_path)
    if screening_path.parent.parent != registry:
        raise ValueError("screening and certification must use the same registry")
    screening_report_path = Path(screening_report).resolve()
    screening_value, screening_sha = _load_evaluation_report(screening_report_path)
    _verify_sidecar(screening_report_path, screening_sha)
    if (
        screening_execution["model"] != model
        or screening_execution["evaluationProvenance"] != provenance
        or screening_execution["evaluationProvenanceSha256"]
        != provenance["provenanceSha256"]
    ):
        raise ValueError("screening reservation differs from certification execution")
    _resolve_registry_result_path(
        registry, screening_execution["outputPath"], screening_report_path
    )
    _validate_screening_report(
        screening_value, screening_sha, model, screening_execution
    )
    screening_record = {
        "evaluationProvenanceSha256": provenance["provenanceSha256"],
        "reportSha256": screening_sha,
        "reservationId": screening_execution["reservationId"],
        "reservationSha256": _sha256_file(screening_path),
    }
    coordinates = _certification_coordinates_for_model(model)
    base_body: dict[str, object] = {
        "coordinates": [
            {
                "familyId": coordinate["familyId"],
                "matchPlan": coordinate["matchPlan"],
                "seedBase": coordinate["seedBase"],
            }
            for coordinate in coordinates
        ],
        "evaluationProvenance": provenance,
        "evaluationProvenanceSha256": provenance["provenanceSha256"],
        "format": V5_CERTIFICATION_RESERVATION_FORMAT,
        "model": model,
        "screening": screening_record,
        "version": V5_CERTIFICATION_RESERVATION_VERSION,
    }
    reservation_id = _certification_reservation_id(base_body)
    document = {
        **base_body,
        "coordinates": [
            {
                **base_body["coordinates"][index],  # type: ignore[index]
                "label": label,
                "outputPath": (
                    f"certification-results/{reservation_id}/{label}.json"
                ),
            }
            for index, label in enumerate(("a", "b"))
        ],
        "reservationId": reservation_id,
    }
    document = _validate_certification_execution_reservation(document)
    directory = registry / "certification-reservations"
    path = directory / f"{reservation_id}.json"
    with _registry_lock(registry):
        directory.mkdir(parents=True, exist_ok=True)
        current_screening = load_v5_screening_execution_reservation(screening_path)
        current_report, current_report_sha = _load_evaluation_report(
            screening_report_path
        )
        _verify_sidecar(screening_report_path, current_report_sha)
        if (
            current_screening != screening_execution
            or _sha256_file(screening_path) != screening_record["reservationSha256"]
            or current_report != screening_value
            or current_report_sha != screening_sha
        ):
            raise ValueError("screening prerequisite changed before certification reservation")
        _validate_screening_report(
            current_report, current_report_sha, model, current_screening
        )
        for existing_path in sorted(directory.glob("*.json")):
            existing = load_v5_certification_execution_reservation(existing_path)
            if existing["model"]["tensorStateSha256"] == model["tensorStateSha256"]:  # type: ignore[index]
                if existing == document and existing_path == path:
                    return {
                        "reservation": existing,
                        "reservationPath": str(existing_path),
                        "reservationSha256": _sha256_file(existing_path),
                    }
                raise ValueError(
                    "this frozen Actor already has a certification execution reservation"
                )
        digest = _write_canonical_with_sidecar(path, document)
    return {
        "reservation": document,
        "reservationPath": str(path),
        "reservationSha256": digest,
    }


def _canonical_registry_result_path(logical_path: object) -> str:
    if not isinstance(logical_path, str) or not logical_path or "\\" in logical_path:
        raise ValueError("reserved evaluation result path is not canonical POSIX")
    logical = PurePosixPath(logical_path)
    if (
        logical.is_absolute()
        or logical.as_posix() != logical_path
        or any(part in ("", ".", "..") for part in logical.parts)
    ):
        raise ValueError("reserved evaluation result path is not canonical POSIX")
    return logical_path


def _resolve_registry_result_path(
    registry: Path, logical_path: object, actual_path: str | Path
) -> Path:
    logical_value = _canonical_registry_result_path(logical_path)
    logical = PurePosixPath(logical_value)
    expected = registry.joinpath(*logical.parts).resolve()
    try:
        expected.relative_to(registry)
    except ValueError as error:
        raise ValueError("reserved evaluation result path escaped its registry") from error
    actual = Path(actual_path).resolve()
    if actual != expected:
        raise ValueError("evaluation output path differs from its reservation")
    return expected


def _screening_report_binding(
    reservation: Mapping[str, object],
) -> dict[str, object]:
    coordinate = reservation["coordinate"]
    assert isinstance(coordinate, Mapping)
    return {
        "coordinate": {
            "familyId": coordinate["familyId"],
            "seedBase": coordinate["seedBase"],
        },
        "evaluationProvenanceSha256": reservation[
            "evaluationProvenanceSha256"
        ],
        "outputPath": reservation["outputPath"],
        "reservationId": reservation["reservationId"],
        "reservationSha256": _sha256_bytes(
            canonical_json_bytes(dict(reservation))
        ),
    }


def authorize_v5_screening_evaluation(
    screening_reservation: str | Path,
    model_identity: Mapping[str, object],
    *,
    evaluation_provenance: Mapping[str, object],
    family_id: str,
    seed_base: int,
    match_plan: Mapping[int, int],
    match_shard_count: int,
    match_shard_index: int,
    bootstrap_resamples: int,
    output_path: str | Path,
) -> dict[str, object]:
    reservation_path = Path(screening_reservation).resolve()
    reservation = load_v5_screening_execution_reservation(reservation_path)
    model = {
        name: _require_sha(model_identity.get(name), f"screening Actor {name}")
        for name in sorted(V5_ACTOR_IDENTITY_KEYS)
    }
    provenance = _validated_certification_execution_provenance(
        evaluation_provenance
    )
    coordinate = reservation["coordinate"]
    assert isinstance(coordinate, Mapping)
    expected_plan = {
        str(player): int(matches) for player, matches in match_plan.items()
    }
    if (
        reservation["model"] != model
        or reservation["evaluationProvenance"] != provenance
        or reservation["evaluationProvenanceSha256"]
        != provenance["provenanceSha256"]
        or coordinate.get("familyId") != family_id
        or coordinate.get("seedBase") != seed_base
        or coordinate.get("matchPlan") != expected_plan
        or coordinate.get("bootstrapResamples") != bootstrap_resamples
        or match_shard_count != 1
        or match_shard_index != 0
    ):
        raise ValueError("screening evaluator differs from its one-shot reservation")
    if reservation_path.parent.name != "screening-reservations":
        raise ValueError("screening reservation is outside a canonical registry")
    registry = reservation_path.parent.parent
    _resolve_registry_result_path(registry, reservation["outputPath"], output_path)
    return _screening_report_binding(reservation)


def _certification_report_binding(
    reservation: Mapping[str, object],
    coordinate: Mapping[str, object],
) -> dict[str, object]:
    return {
        "coordinate": {
            "familyId": coordinate["familyId"],
            "seedBase": coordinate["seedBase"],
        },
        "evaluationProvenanceSha256": reservation[
            "evaluationProvenanceSha256"
        ],
        "outputPath": coordinate["outputPath"],
        "reservationId": reservation["reservationId"],
        "reservationSha256": _sha256_bytes(
            canonical_json_bytes(dict(reservation))
        ),
    }


def authorize_v5_certification_evaluation(
    reservation_path: str | Path,
    model_identity: Mapping[str, object],
    *,
    evaluation_provenance: Mapping[str, object],
    family_id: str,
    seed_base: int,
    match_plan: Mapping[int, int],
    match_shard_count: int,
    match_shard_index: int,
    bootstrap_resamples: int,
    output_path: str | Path,
) -> dict[str, object]:
    """Authorize the only two Actor-derived certification runs before gameplay."""

    path = Path(reservation_path).resolve()
    reservation = load_v5_certification_execution_reservation(path)
    registry = path.parent.parent
    provenance = _validated_certification_execution_provenance(
        evaluation_provenance
    )
    expected_plan = {
        str(player): int(matches) for player, matches in match_plan.items()
    }
    coordinates = reservation["coordinates"]
    assert isinstance(coordinates, list)
    matches = [
        coordinate
        for coordinate in coordinates
        if isinstance(coordinate, Mapping)
        and coordinate.get("familyId") == family_id
        and coordinate.get("seedBase") == seed_base
    ]
    if (
        reservation["model"] != dict(model_identity)
        or reservation["evaluationProvenance"] != provenance
        or reservation["evaluationProvenanceSha256"]
        != provenance["provenanceSha256"]
        or len(matches) != 1
        or matches[0].get("matchPlan") != expected_plan
        or match_shard_count != 1
        or match_shard_index != 0
        or bootstrap_resamples != DEFAULT_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError(
            "certification evaluator configuration differs from its execution reservation"
        )
    coordinate = matches[0]
    _resolve_registry_result_path(registry, coordinate["outputPath"], output_path)
    return _certification_report_binding(reservation, coordinate)


def _load_evaluation_report(path: str | Path) -> tuple[dict[str, object], str]:
    report_path = Path(path).resolve()
    raw, digest = _strict_canonical_object(report_path, "V5 evaluation report")
    verified = validate_v5_evaluation_report(raw)
    return verified, digest


def _report_resamples(report: Mapping[str, object]) -> int:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("evaluation report has no per-player-count results")
    values: set[int] = set()
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("evaluation result is not an object")
        interval = result.get("matchClustered95")
        if not isinstance(interval, Mapping) or type(interval.get("resamples")) is not int:
            raise ValueError("evaluation report omitted bootstrap resamples")
        values.add(int(interval["resamples"]))
    if len(values) != 1:
        raise ValueError("evaluation report mixed bootstrap resample counts")
    return values.pop()


def _report_match_plan(report: Mapping[str, object]) -> dict[int, int]:
    value = report.get("matchPlan")
    if not isinstance(value, Mapping):
        raise ValueError("evaluation report omitted its match plan")
    try:
        plan = {int(player): int(matches) for player, matches in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation match plan is invalid") from error
    if set(plan) != set(PLAYER_COUNTS) or any(matches < 1 for matches in plan.values()):
        raise ValueError("evaluation match plan must cover p4..p10")
    return plan


def _report_provenance_summary(
    report: Mapping[str, object], label: str
) -> dict[str, object]:
    """Revalidate preserved evaluator/Normal source and runtime evidence."""

    records = report.get("evaluationProvenance")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} omitted evaluation provenance")
    coordinates: list[tuple[int, int]] = []
    provenance_hashes: list[str] = []
    source_bindings: set[tuple[str, str]] = set()
    for record in records:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"provenance", "shard"}
            or not isinstance(record.get("shard"), Mapping)
            or set(record["shard"]) != {"count", "index"}  # type: ignore[arg-type]
        ):
            raise ValueError(f"{label} provenance record structure drifted")
        shard = record["shard"]
        assert isinstance(shard, Mapping)
        count = shard.get("count")
        index = shard.get("index")
        if (
            type(count) is not int
            or type(index) is not int
            or count < 1
            or not 0 <= index < count
        ):
            raise ValueError(f"{label} provenance shard coordinate drifted")
        evidence = record.get("provenance")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{label} provenance evidence is missing")
        verified = validate_v5_evaluation_provenance(evidence)
        source = verified["source"]
        artifacts = verified["artifacts"]
        assert isinstance(source, Mapping) and isinstance(artifacts, Mapping)
        if any(
            artifacts.get(name) is None
            for name in ("gitBundleSha256", "sourceSnapshotSha256")
        ):
            raise ValueError(f"{label} provenance omitted preserved source artifacts")
        source_bindings.add(
            (str(source["sourceCommit"]), str(source["sourceBindingSha256"]))
        )
        coordinates.append((count, index))
        provenance_hashes.append(
            _require_sha(verified.get("provenanceSha256"), f"{label} provenance SHA")
        )
    if len(source_bindings) != 1:
        raise ValueError(f"{label} mixed evaluator/Normal source bindings")
    counts = {count for count, _ in coordinates}
    if len(counts) != 1:
        raise ValueError(f"{label} provenance mixed shard counts")
    count = next(iter(counts))
    if coordinates != [(count, index) for index in range(count)]:
        raise ValueError(f"{label} provenance shard inventory is incomplete or unordered")
    source_commit, source_binding = next(iter(source_bindings))
    return {
        "provenanceSha256": provenance_hashes,
        "sourceBindingSha256": source_binding,
        "sourceCommit": source_commit,
    }


def _validate_screening_report(
    report: Mapping[str, object],
    report_sha256: str,
    model: Mapping[str, str],
    reservation: Mapping[str, object],
) -> dict[str, object]:
    coordinate = reservation.get("coordinate")
    assert isinstance(coordinate, Mapping)
    if (
        report.get("mode") != "screening"
        or "diagnosticOnly" in report
        or report.get("model") != model
        or report.get("shard") != {"count": 1, "index": 0}
        or report.get("completeEvaluation") is not True
        or report.get("allPlayerCountsPassed") is not True
        or report.get("familyId") != coordinate.get("familyId")
        or report.get("seedBase") != coordinate.get("seedBase")
        or report.get("matchPlan") != coordinate.get("matchPlan")
        or report.get("screeningReservation")
        != _screening_report_binding(reservation)
        or _report_resamples(report) != DEFAULT_BOOTSTRAP_RESAMPLES
        or _report_match_plan(report) != SCREENING_MATCH_COUNTS
    ):
        raise ValueError(
            "screening report is incomplete, failed, or differs from its "
            "one-shot reservation"
        )
    provenance = _report_provenance_summary(report, "screening report")
    if provenance["provenanceSha256"] != [
        reservation["evaluationProvenanceSha256"]
    ]:
        raise ValueError("screening report provenance differs from its reservation")
    return {
        "reportSha256": _require_sha(report_sha256, "screening report SHA"),
        "reservationBinding": _screening_report_binding(reservation),
    }


def _validate_certification_report(
    report: Mapping[str, object],
    report_sha256: str,
    model: Mapping[str, str],
) -> dict[str, object]:
    if report.get("mode") != "certification":
        raise ValueError("promotion requires certification-mode development reports")
    if report.get("model") != model:
        raise ValueError("certification report is bound to a different frozen Actor")
    if report.get("shard") != {"count": 1, "index": 0}:
        raise ValueError("certification report must be one merged complete report")
    if report.get("completeEvaluation") is not True:
        raise ValueError("certification report is incomplete")
    if report.get("allPlayerCountsPassed") is not True:
        raise ValueError("certification report failed its evaluator gates")
    if _report_resamples(report) != DEFAULT_BOOTSTRAP_RESAMPLES:
        raise ValueError("certification requires exactly 10000 bootstrap resamples")
    match_plan = _report_match_plan(report)
    if match_plan != SCREENING_MATCH_COUNTS:
        raise ValueError("certification requires exactly 60 matches for each p4..p10")
    results = report.get("results")
    assert isinstance(results, list)
    by_player = {
        int(result["playerCount"]): result
        for result in results
        if isinstance(result, Mapping) and type(result.get("playerCount")) is int
    }
    if set(by_player) != set(PLAYER_COUNTS):
        raise ValueError("certification report omitted a player-count stratum")
    for player_count in PLAYER_COUNTS:
        result = by_player[player_count]
        interval = result.get("matchClustered95")
        pairwise = result.get("candidateBeforeNormalPairwise")
        if not isinstance(interval, Mapping) or not isinstance(pairwise, Mapping):
            raise ValueError("certification metric structure is invalid")
        try:
            mean = float(result["meanCandidateMinusNormalChipPerAct"])
            lower = float(interval["low"])
            rate = float(pairwise["rate"])
            matches = int(result["matches"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("certification metric is invalid") from error
        if not all(math.isfinite(value) for value in (mean, lower, rate)):
            raise ValueError("certification metric is non-finite")
        if matches != match_plan[player_count]:
            raise ValueError("certification result count disagrees with its plan")
        if result.get("complete") is not True:
            raise ValueError("certification result is incomplete")
        if (
            mean < V5_DEVELOPMENT_GATES["minMeanChipDifference"]
            or lower < V5_DEVELOPMENT_GATES["minCluster95LowerBound"]
            or rate < V5_DEVELOPMENT_GATES["minPairwiseRate"]
        ):
            raise ValueError(
                f"certification p{player_count} failed a stricter development gate"
            )
    family_id = report.get("familyId")
    seed_base = report.get("seedBase")
    if not isinstance(family_id, str) or not family_id or type(seed_base) is not int:
        raise ValueError("certification seed-family identity is invalid")
    eligible_coordinates = _certification_coordinates_for_model(model)
    if not any(
        family_id == coordinate["familyId"]
        and seed_base == coordinate["seedBase"]
        for coordinate in eligible_coordinates
    ):
        raise ValueError("certification report used an unreserved model-derived coordinate")
    provenance = _report_provenance_summary(report, "certification report")
    binding = report.get("certificationReservation")
    if not isinstance(binding, dict) or set(binding) != {
        "coordinate",
        "evaluationProvenanceSha256",
        "outputPath",
        "reservationId",
        "reservationSha256",
    }:
        raise ValueError("certification report omitted its execution reservation")
    coordinate = binding.get("coordinate")
    if coordinate != {"familyId": family_id, "seedBase": seed_base}:
        raise ValueError("certification report reservation coordinate drifted")
    for name in (
        "evaluationProvenanceSha256",
        "reservationId",
        "reservationSha256",
    ):
        _require_sha(binding.get(name), f"certification reservation {name}")
    output_path = _canonical_registry_result_path(binding.get("outputPath"))
    reservation_id = str(binding["reservationId"])
    if output_path not in {
        f"certification-results/{reservation_id}/a.json",
        f"certification-results/{reservation_id}/b.json",
    }:
        raise ValueError("certification report reserved output path drifted")
    if provenance["provenanceSha256"] != [
        binding["evaluationProvenanceSha256"]
    ]:
        raise ValueError(
            "certification report provenance differs from its execution reservation"
        )
    return {
        "certificationReservation": dict(binding),
        "evaluationProvenanceSha256": provenance["provenanceSha256"],
        "evaluationSourceBindingSha256": provenance["sourceBindingSha256"],
        "evaluationSourceCommit": provenance["sourceCommit"],
        "familyId": family_id,
        "matchPlan": {str(player): match_plan[player] for player in PLAYER_COUNTS},
        "reportSha256": _require_sha(report_sha256, "certification report SHA"),
        "seedBase": seed_base,
    }


def _evaluation_seed_set(family_id: str, seed_base: int, plan: Mapping[int, int]) -> set[int]:
    return _validated_evaluation_seed_set(family_id, seed_base, plan)


def _canonical_final_family(seed_base: int) -> str:
    return f"v5-final-holdout-s{seed_base}"


def _plan_id(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"DALMUTI-V5-FINAL-RESERVATION\0" + canonical_json_bytes(dict(body))
    )


def _receipt_id(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"DALMUTI-V5-FINAL-CONSUMPTION\0" + canonical_json_bytes(dict(body))
    )


def _claim_id(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"DALMUTI-V5-FINAL-SHARD-CLAIM\0" + canonical_json_bytes(dict(body))
    )


def _certification_reservation_id(body: Mapping[str, object]) -> str:
    return _sha256_bytes(
        b"DALMUTI-V5-CERTIFICATION-EXECUTION\0"
        + canonical_json_bytes(dict(body))
    )


def _validate_plan(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != {
        "certification",
        "final",
        "format",
        "model",
        "reservationId",
        "version",
    } or value.get("format") != V5_PROMOTION_PLAN_FORMAT or value.get(
        "version"
    ) != V5_PROMOTION_PLAN_VERSION:
        raise ValueError("V5 promotion plan contract drifted")
    model = value.get("model")
    certification = value.get("certification")
    final = value.get("final")
    if (
        not isinstance(model, dict)
        or set(model) != V5_ACTOR_IDENTITY_KEYS
        or not isinstance(certification, dict)
        or set(certification) != {
            "developmentGates",
            "disjointSeedFamilies",
            "executionReservation",
            "reports",
        }
        or not isinstance(final, dict)
        or set(final) != {
            "bootstrapResamples",
            "evaluationProvenanceSha256",
            "evaluationGates",
            "familyId",
            "finalReportTrainingOrTuningUseAllowed",
            "matchPlan",
            "matchShardCount",
            "seedBase",
        }
    ):
        raise ValueError("V5 promotion plan structure drifted")
    for name, digest in model.items():
        _require_sha(digest, f"promotion model {name}")
    reports = certification.get("reports")
    execution = certification.get("executionReservation")
    if (
        certification.get("developmentGates") != V5_DEVELOPMENT_GATES
        or certification.get("disjointSeedFamilies") is not True
        or not isinstance(execution, dict)
        or set(execution) != {
            "evaluationProvenanceSha256",
            "reservationId",
            "reservationSha256",
        }
        or not isinstance(reports, list)
        or len(reports) != V5_CERTIFICATION_REPORT_COUNT
    ):
        raise ValueError("V5 certification reservation fields drifted")
    for name in (
        "evaluationProvenanceSha256",
        "reservationId",
        "reservationSha256",
    ):
        _require_sha(execution.get(name), f"certification execution {name}")
    report_fields = {
        "certificationReservation",
        "evaluationProvenanceSha256",
        "evaluationSourceBindingSha256",
        "evaluationSourceCommit",
        "familyId",
        "matchPlan",
        "reportSha256",
        "seedBase",
        "snapshotPath",
    }
    certification_seed_sets: list[set[int]] = []
    for report in reports:
        if not isinstance(report, dict) or set(report) != report_fields:
            raise ValueError("V5 certification report reservation drifted")
        binding = report.get("certificationReservation")
        if not isinstance(binding, dict) or set(binding) != {
            "coordinate",
            "evaluationProvenanceSha256",
            "outputPath",
            "reservationId",
            "reservationSha256",
        }:
            raise ValueError("V5 certification execution binding drifted")
        if {
            name: binding[name]
            for name in (
                "evaluationProvenanceSha256",
                "reservationId",
                "reservationSha256",
            )
        } != execution:
            raise ValueError("V5 certification reports mixed execution reservations")
        _require_sha(report.get("reportSha256"), "certification report SHA")
        _require_sha(
            report.get("evaluationSourceBindingSha256"),
            "certification evaluator source binding SHA",
        )
        provenance_hashes = report.get("evaluationProvenanceSha256")
        if not isinstance(provenance_hashes, list) or not provenance_hashes:
            raise ValueError("V5 certification provenance hashes are missing")
        for digest in provenance_hashes:
            _require_sha(digest, "certification evaluation provenance SHA")
        if provenance_hashes != [execution["evaluationProvenanceSha256"]]:
            raise ValueError(
                "V5 certification provenance differs from its execution reservation"
            )
        if (
            not isinstance(report.get("evaluationSourceCommit"), str)
            or _GIT_COMMIT.fullmatch(str(report["evaluationSourceCommit"])) is None
        ):
            raise ValueError("V5 certification evaluation source commit drifted")
        family_id = report.get("familyId")
        seed_base = report.get("seedBase")
        match_plan = report.get("matchPlan")
        if (
            not isinstance(family_id, str)
            or not family_id
            or not family_id.isascii()
            or type(seed_base) is not int
            or not 0 <= seed_base <= 0xFFFF_FFFF
            or not isinstance(match_plan, dict)
        ):
            raise ValueError("V5 certification seed family drifted")
        try:
            numeric_plan = {
                int(player): int(matches) for player, matches in match_plan.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("V5 certification match plan drifted") from error
        if numeric_plan != SCREENING_MATCH_COUNTS:
            raise ValueError("V5 certification match plan must be exactly 60/p")
        certification_seed_sets.append(
            _evaluation_seed_set(family_id, seed_base, numeric_plan)
        )
        if binding.get("coordinate") != {
            "familyId": family_id,
            "seedBase": seed_base,
        }:
            raise ValueError("V5 certification execution coordinate drifted")
        output_path = _canonical_registry_result_path(binding.get("outputPath"))
        if output_path not in {
            f"certification-results/{execution['reservationId']}/a.json",
            f"certification-results/{execution['reservationId']}/b.json",
        }:
            raise ValueError("V5 certification execution output path drifted")
        if report.get("snapshotPath") != (
            f"certifications/{report['reportSha256']}.json"
        ):
            raise ValueError("V5 certification snapshot path drifted")
    family_pairs = {(report["familyId"], report["seedBase"]) for report in reports}
    if len(family_pairs) != V5_CERTIFICATION_REPORT_COUNT:
        raise ValueError("certification seed families are not distinct")
    if len({report["familyId"] for report in reports}) != V5_CERTIFICATION_REPORT_COUNT:
        raise ValueError("certification family IDs are not distinct")
    if {
        report["certificationReservation"]["outputPath"]  # type: ignore[index]
        for report in reports
    } != {
        f"certification-results/{execution['reservationId']}/a.json",
        f"certification-results/{execution['reservationId']}/b.json",
    }:
        raise ValueError("certification reserved output inventory is incomplete")
    if len({
        (report["evaluationSourceCommit"], report["evaluationSourceBindingSha256"])
        for report in reports
    }) != 1:
        raise ValueError("certification reports used different evaluator/Normal sources")
    expected_coordinates = {
        (coordinate["familyId"], coordinate["seedBase"])
        for coordinate in _certification_coordinates_for_model(model)  # type: ignore[arg-type]
    }
    if family_pairs != expected_coordinates:
        raise ValueError("certification coordinates are not Actor-derived and canonical")
    if certification_seed_sets[0] & certification_seed_sets[1]:
        raise ValueError("certification seed families are not disjoint")
    expected_final_plan = {
        str(player): FINAL_MATCH_COUNTS[player] for player in PLAYER_COUNTS
    }
    seed_base = final.get("seedBase")
    if (
        type(seed_base) is not int
        or seed_base < V5_FIRST_FINAL_SEED_BASE
        or seed_base > 0xFFFF_FFFF
        or (seed_base - V5_FIRST_FINAL_SEED_BASE) % V5_FINAL_SEED_STEP != 0
        or final.get("familyId") != _canonical_final_family(seed_base)
        or final.get("matchPlan") != expected_final_plan
        or type(final.get("matchShardCount")) is not int
        or int(final["matchShardCount"]) < 1
        or int(final["matchShardCount"]) > max(FINAL_MATCH_COUNTS.values())
        or final.get("bootstrapResamples") != DEFAULT_BOOTSTRAP_RESAMPLES
        or final.get("evaluationProvenanceSha256")
        != execution["evaluationProvenanceSha256"]
        or final.get("evaluationGates") != EXACT_GATES
        or final.get("finalReportTrainingOrTuningUseAllowed") is not False
    ):
        raise ValueError("V5 final holdout reservation fields drifted")
    final_seed_set = _evaluation_seed_set(
        str(final["familyId"]), int(seed_base), FINAL_MATCH_COUNTS
    )
    if any(final_seed_set & seeds for seeds in certification_seed_sets):
        raise ValueError("reserved final seed family overlaps certification")
    body = {key: value[key] for key in value if key != "reservationId"}
    if value.get("reservationId") != _plan_id(body):
        raise ValueError("V5 promotion reservation ID does not match its plan")
    return dict(value)


def load_v5_promotion_plan(path: str | Path) -> dict[str, object]:
    plan_path = Path(path).resolve()
    plan, digest = _strict_canonical_object(plan_path, "V5 promotion plan")
    _verify_sidecar(plan_path, digest)
    verified = _validate_plan(plan)
    if plan_path.name != f"{verified['reservationId']}.json":
        raise ValueError("V5 promotion plan filename does not match reservation ID")
    _validate_plan_certification_snapshots(plan_path, verified)
    return verified


def _validate_plan_certification_snapshots(
    plan_path: Path, plan: Mapping[str, object]
) -> None:
    if plan_path.parent.name != "reservations":
        raise ValueError("promotion plan is outside a canonical reservation registry")
    registry = plan_path.parent.parent
    model = plan["model"]
    certification = plan["certification"]
    assert isinstance(model, Mapping) and isinstance(certification, Mapping)
    execution = certification["executionReservation"]
    assert isinstance(execution, Mapping)
    reservation_path = (
        registry
        / "certification-reservations"
        / f"{execution['reservationId']}.json"
    )
    reservation = load_v5_certification_execution_reservation(reservation_path)
    if (
        _sha256_file(reservation_path) != execution["reservationSha256"]
        or reservation["model"] != model
        or reservation["evaluationProvenanceSha256"]
        != execution["evaluationProvenanceSha256"]
    ):
        raise ValueError(
            "certification execution reservation no longer matches promotion plan"
        )
    screening = reservation["screening"]
    assert isinstance(screening, Mapping)
    screening_path = (
        registry
        / "screening-reservations"
        / f"{screening['reservationId']}.json"
    )
    screening_reservation = load_v5_screening_execution_reservation(
        screening_path
    )
    screening_report = (
        registry
        / "screening-results"
        / str(screening["reservationId"])
        / "report.json"
    )
    screening_value, screening_sha = _load_evaluation_report(screening_report)
    _verify_sidecar(screening_report, screening_sha)
    if (
        _sha256_file(screening_path) != screening["reservationSha256"]
        or screening_sha != screening["reportSha256"]
        or screening_reservation["model"] != model
        or screening_reservation["evaluationProvenanceSha256"]
        != screening["evaluationProvenanceSha256"]
    ):
        raise ValueError("screening prerequisite snapshot drifted")
    _validate_screening_report(
        screening_value,
        screening_sha,
        model,  # type: ignore[arg-type]
        screening_reservation,
    )
    screening_coordinate = screening_reservation["coordinate"]
    final = plan["final"]
    assert isinstance(screening_coordinate, Mapping) and isinstance(final, Mapping)
    if _evaluation_seed_set(
        str(screening_coordinate["familyId"]),
        int(screening_coordinate["seedBase"]),
        SCREENING_MATCH_COUNTS,
    ) & _evaluation_seed_set(
        str(final["familyId"]), int(final["seedBase"]), FINAL_MATCH_COUNTS
    ):
        raise ValueError("screening and final holdout seed sets overlap")
    reserved_coordinates = reservation["coordinates"]
    assert isinstance(reserved_coordinates, list)
    reports = certification["reports"]
    assert isinstance(reports, list)
    for record in reports:
        assert isinstance(record, Mapping)
        matching_coordinates = [
            coordinate
            for coordinate in reserved_coordinates
            if isinstance(coordinate, Mapping)
            and coordinate.get("familyId") == record.get("familyId")
            and coordinate.get("seedBase") == record.get("seedBase")
        ]
        if (
            len(matching_coordinates) != 1
            or record.get("certificationReservation")
            != _certification_report_binding(reservation, matching_coordinates[0])
        ):
            raise ValueError(
                "certification report binding differs from canonical execution reservation"
            )
        snapshot = (registry / str(record["snapshotPath"])).resolve()
        expected_parent = (registry / "certifications").resolve()
        if snapshot.parent != expected_parent:
            raise ValueError("certification snapshot escaped its registry")
        report, digest = _load_evaluation_report(snapshot)
        _verify_sidecar(snapshot, digest)
        if digest != record["reportSha256"]:
            raise ValueError("certification snapshot checksum drifted")
        rebuilt = _validate_certification_report(report, digest, model)  # type: ignore[arg-type]
        rebuilt["snapshotPath"] = record["snapshotPath"]
        if rebuilt != dict(record):
            raise ValueError("certification snapshot evidence no longer matches plan")


def _validate_receipt_shape(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "approved",
        "certificationReportSha256",
        "consumed",
        "evaluationProvenance",
        "final",
        "finalClaims",
        "finalReportTrainingOrTuningUseAllowed",
        "format",
        "model",
        "planSha256",
        "receiptId",
        "reservationId",
        "version",
    }
    if (
        set(value) != expected
        or value.get("format") != V5_CONSUMPTION_RECEIPT_FORMAT
        or value.get("version") != V5_CONSUMPTION_RECEIPT_VERSION
        or value.get("approved") is not True
        or value.get("consumed") is not True
        or value.get("finalReportTrainingOrTuningUseAllowed") is not False
    ):
        raise ValueError("V5 final consumption receipt contract drifted")
    model = value.get("model")
    final = value.get("final")
    final_claims = value.get("finalClaims")
    provenance = value.get("evaluationProvenance")
    certifications = value.get("certificationReportSha256")
    if (
        not isinstance(model, dict)
        or set(model) != V5_ACTOR_IDENTITY_KEYS
        or not isinstance(final, dict)
        or set(final) != {"familyId", "reportSha256", "seedBase"}
        or not isinstance(final_claims, list)
        or not final_claims
        or not isinstance(provenance, dict)
        or set(provenance) != {
            "certificationProvenanceSha256",
            "finalProvenanceSha256",
            "sourceBindingSha256",
            "sourceCommit",
        }
        or not isinstance(certifications, list)
        or len(certifications) != V5_CERTIFICATION_REPORT_COUNT
    ):
        raise ValueError("V5 final consumption receipt structure drifted")
    certification_provenance = provenance["certificationProvenanceSha256"]
    final_provenance = provenance["finalProvenanceSha256"]
    if (
        not isinstance(certification_provenance, list)
        or len(certification_provenance) != V5_CERTIFICATION_REPORT_COUNT
        or any(not isinstance(group, list) or not group for group in certification_provenance)
        or not isinstance(final_provenance, list)
        or not final_provenance
    ):
        raise ValueError("consumed evaluation provenance inventory drifted")
    for digest in [
        digest
        for group in certification_provenance
        for digest in group
    ] + list(final_provenance):
        _require_sha(digest, "consumed evaluation provenance SHA")
    _require_sha(
        provenance.get("sourceBindingSha256"),
        "consumed evaluator source binding SHA",
    )
    if (
        not isinstance(provenance.get("sourceCommit"), str)
        or _GIT_COMMIT.fullmatch(str(provenance["sourceCommit"])) is None
    ):
        raise ValueError("consumed evaluator source commit drifted")
    for name, digest in model.items():
        _require_sha(digest, f"consumed model {name}")
    for digest in certifications:
        _require_sha(digest, "consumed certification report SHA")
    if len(set(certifications)) != V5_CERTIFICATION_REPORT_COUNT:
        raise ValueError("consumed certification report hashes are not distinct")
    claim_reservations: set[str] = set()
    claim_counts: set[int] = set()
    claim_indices: set[int] = set()
    claim_ids: set[str] = set()
    for claim in final_claims:
        if (
            not isinstance(claim, dict)
            or set(claim) != {
                "claimId",
                "evaluationProvenanceSha256",
                "outputPath",
                "reservationId",
                "shard",
            }
            or not isinstance(claim.get("shard"), dict)
            or set(claim["shard"]) != {"count", "index"}
        ):
            raise ValueError("consumed final claim binding structure drifted")
        claim_id = _require_sha(claim.get("claimId"), "consumed final claim ID")
        reservation_id = _require_sha(
            claim.get("reservationId"), "consumed claim reservation ID"
        )
        _require_sha(
            claim.get("evaluationProvenanceSha256"),
            "consumed claim evaluation provenance SHA",
        )
        count = claim["shard"].get("count")
        index = claim["shard"].get("index")
        if (
            type(count) is not int
            or type(index) is not int
            or count < 1
            or not 0 <= index < count
        ):
            raise ValueError("consumed final claim shard binding drifted")
        claim_ids.add(claim_id)
        claim_reservations.add(reservation_id)
        claim_counts.add(count)
        claim_indices.add(index)
        output_path = _canonical_registry_result_path(claim.get("outputPath"))
        if output_path != (
            f"final-results/{reservation_id}/shard-{index:03d}.json"
        ):
            raise ValueError("consumed final claim output path drifted")
    if (
        len(claim_ids) != len(final_claims)
        or len(claim_reservations) != 1
        or len(claim_counts) != 1
        or claim_reservations != {value.get("reservationId")}
    ):
        raise ValueError("consumed final claim inventory is mixed or duplicated")
    claim_count = next(iter(claim_counts))
    if len(final_claims) != claim_count or claim_indices != set(range(claim_count)):
        raise ValueError("consumed final claim inventory is incomplete")
    claimed_provenance = {
        claim["evaluationProvenanceSha256"] for claim in final_claims
    }
    if (
        len(claimed_provenance) != 1
        or set(final_provenance) != claimed_provenance
    ):
        raise ValueError("consumed final provenance differs from its shard claims")
    _require_sha(final.get("reportSha256"), "consumed final report SHA")
    _require_sha(value.get("planSha256"), "consumed promotion plan SHA")
    _require_sha(value.get("reservationId"), "consumed reservation ID")
    if (
        not isinstance(final.get("familyId"), str)
        or type(final.get("seedBase")) is not int
        or int(final["seedBase"]) < V5_FIRST_FINAL_SEED_BASE
        or int(final["seedBase"]) > 0xFFFF_FFFF
        or (
            int(final["seedBase"]) - V5_FIRST_FINAL_SEED_BASE
        ) % V5_FINAL_SEED_STEP != 0
        or final.get("familyId") != _canonical_final_family(int(final["seedBase"]))
    ):
        raise ValueError("consumed final seed family drifted")
    body = {key: value[key] for key in value if key != "receiptId"}
    if value.get("receiptId") != _receipt_id(body):
        raise ValueError("V5 final consumption receipt ID drifted")
    return dict(value)


def reserve_v5_final_holdout(
    registry_directory: str | Path,
    actor_bundle: str | Path,
    certification_report_paths: Sequence[str | Path],
    *,
    final_match_shard_count: int,
) -> dict[str, object]:
    """Burn the next final seed and publish its immutable pre-eval plan."""

    if len(certification_report_paths) != V5_CERTIFICATION_REPORT_COUNT:
        raise ValueError("exactly two certification reports are required")
    if (
        isinstance(final_match_shard_count, bool)
        or not isinstance(final_match_shard_count, int)
        or not 1 <= final_match_shard_count <= max(FINAL_MATCH_COUNTS.values())
    ):
        raise ValueError(
            "final_match_shard_count must be 1..2500 so every shard is non-empty"
        )
    registry = Path(registry_directory).resolve()
    model = _exact_actor_identity(actor_bundle)
    certification_records: list[dict[str, object]] = []
    certification_seed_sets: list[set[int]] = []
    for path in certification_report_paths:
        report, digest = _load_evaluation_report(path)
        record = _validate_certification_report(report, digest, model)
        record["snapshotPath"] = f"certifications/{digest}.json"
        certification_records.append(record)
        match_plan = {int(key): int(value) for key, value in record["matchPlan"].items()}  # type: ignore[union-attr]
        certification_seed_sets.append(
            _evaluation_seed_set(
                str(record["familyId"]), int(record["seedBase"]), match_plan
            )
        )
    execution_bindings = [
        record["certificationReservation"] for record in certification_records
    ]
    if any(not isinstance(value, Mapping) for value in execution_bindings):
        raise ValueError("certification reports omitted execution reservations")
    execution_tuples = {
        (
            value["reservationId"],  # type: ignore[index]
            value["reservationSha256"],  # type: ignore[index]
            value["evaluationProvenanceSha256"],  # type: ignore[index]
        )
        for value in execution_bindings
    }
    if len(execution_tuples) != 1:
        raise ValueError("certification reports used different execution reservations")
    execution_id, execution_sha, execution_provenance_sha = next(
        iter(execution_tuples)
    )
    execution_record = {
        "evaluationProvenanceSha256": execution_provenance_sha,
        "reservationId": execution_id,
        "reservationSha256": execution_sha,
    }
    certification_execution_path = (
        registry / "certification-reservations" / f"{execution_id}.json"
    )
    certification_execution = load_v5_certification_execution_reservation(
        certification_execution_path
    )
    if (
        _sha256_file(certification_execution_path) != execution_sha
        or certification_execution["model"] != model
        or certification_execution["evaluationProvenanceSha256"]
        != execution_provenance_sha
    ):
        raise ValueError(
            "certification execution reservation does not match its reports"
        )
    screening_record = certification_execution["screening"]
    assert isinstance(screening_record, Mapping)
    screening_execution_path = (
        registry
        / "screening-reservations"
        / f"{screening_record['reservationId']}.json"
    )
    screening_execution = load_v5_screening_execution_reservation(
        screening_execution_path
    )
    if (
        _sha256_file(screening_execution_path)
        != screening_record["reservationSha256"]
        or screening_execution["model"] != model
        or screening_execution["evaluationProvenanceSha256"]
        != execution_provenance_sha
    ):
        raise ValueError("screening reservation no longer matches certification")
    screening_coordinate = screening_execution["coordinate"]
    assert isinstance(screening_coordinate, Mapping)
    screening_seed_set = _evaluation_seed_set(
        str(screening_coordinate["familyId"]),
        int(screening_coordinate["seedBase"]),
        SCREENING_MATCH_COUNTS,
    )
    reserved_coordinates = certification_execution["coordinates"]
    assert isinstance(reserved_coordinates, list)
    for original_path, record in zip(
        certification_report_paths, certification_records, strict=True
    ):
        matches = [
            coordinate
            for coordinate in reserved_coordinates
            if isinstance(coordinate, Mapping)
            and coordinate.get("familyId") == record["familyId"]
            and coordinate.get("seedBase") == record["seedBase"]
        ]
        if (
            len(matches) != 1
            or record["certificationReservation"]
            != _certification_report_binding(certification_execution, matches[0])
        ):
            raise ValueError(
                "certification report does not consume an exact reserved coordinate"
            )
        binding = record["certificationReservation"]
        assert isinstance(binding, Mapping)
        _resolve_registry_result_path(
            registry, binding["outputPath"], original_path
        )
    if certification_records[0]["reportSha256"] == certification_records[1]["reportSha256"]:
        raise ValueError("certification reports must be two distinct immutable reports")
    if (
        certification_records[0]["familyId"]
        == certification_records[1]["familyId"]
        or certification_seed_sets[0] & certification_seed_sets[1]
    ):
        raise ValueError("certification seed families are not disjoint")
    if len({
        (
            record["evaluationSourceCommit"],
            record["evaluationSourceBindingSha256"],
        )
        for record in certification_records
    }) != 1:
        raise ValueError("certification reports used different evaluator/Normal sources")
    certification_records.sort(
        key=lambda value: (str(value["familyId"]), int(value["seedBase"]))
    )
    reservations = registry / "reservations"
    consumptions = registry / "consumptions"
    certifications = registry / "certifications"
    with _registry_lock(registry):
        reservations.mkdir(parents=True, exist_ok=True)
        consumptions.mkdir(parents=True, exist_ok=True)
        certifications.mkdir(parents=True, exist_ok=True)
        current_execution = load_v5_certification_execution_reservation(
            certification_execution_path
        )
        if (
            current_execution != certification_execution
            or _sha256_file(certification_execution_path) != execution_sha
        ):
            raise ValueError(
                "certification execution reservation changed before final reservation"
            )
        # Snapshot the exact reports under the registry. Approval never trusts
        # or depends on the caller's original report paths.
        for original_path in certification_report_paths:
            current_report, current_sha = _load_evaluation_report(original_path)
            matches = [
                record
                for record in certification_records
                if record["reportSha256"] == current_sha
            ]
            if len(matches) != 1:
                raise ValueError("certification report changed before reservation")
            expected_record = matches[0]
            _validate_certification_report(current_report, current_sha, model)
            binding = expected_record["certificationReservation"]
            assert isinstance(binding, Mapping)
            _resolve_registry_result_path(
                registry, binding["outputPath"], original_path
            )
            snapshot = registry / str(expected_record["snapshotPath"])
            if snapshot.exists():
                snapshot_report, snapshot_sha = _load_evaluation_report(snapshot)
                _verify_sidecar(snapshot, snapshot_sha)
                if snapshot_sha != current_sha or snapshot_report != current_report:
                    raise ValueError("certification snapshot hash collision")
            else:
                _write_canonical_with_sidecar(snapshot, current_report)
        existing: list[dict[str, object]] = []
        for path in sorted(reservations.glob("*.json")):
            existing.append(load_v5_promotion_plan(path))
        consumed: list[dict[str, object]] = []
        for path in sorted(consumptions.glob("*.json")):
            receipt, receipt_sha = _strict_canonical_object(
                path, "V5 final consumption receipt"
            )
            _verify_sidecar(path, receipt_sha)
            consumed.append(_validate_receipt_shape(receipt))
        same_actor_plans = [
            plan
            for plan in existing
            if plan["model"]["tensorStateSha256"] == model["tensorStateSha256"]  # type: ignore[index]
        ]
        if same_actor_plans:
            if len(same_actor_plans) != 1:
                raise ValueError("frozen Actor has multiple final-seed reservations")
            prior = same_actor_plans[0]
            prior_certification = prior["certification"]
            prior_final = prior["final"]
            assert isinstance(prior_certification, Mapping)
            assert isinstance(prior_final, Mapping)
            if (
                prior["model"] == model
                and prior_certification.get("developmentGates")
                == V5_DEVELOPMENT_GATES
                and prior_certification.get("disjointSeedFamilies") is True
                and prior_certification.get("executionReservation")
                == execution_record
                and prior_certification.get("reports") == certification_records
                and prior_final.get("matchShardCount") == final_match_shard_count
            ):
                prior_path = (
                    reservations / f"{prior['reservationId']}.json"
                )
                return {
                    "plan": prior,
                    "planPath": str(prior_path),
                    "planSha256": _sha256_file(prior_path),
                }
            raise ValueError("this frozen Actor already has a final-seed reservation")
        if any(
            receipt["model"]["tensorStateSha256"] == model["tensorStateSha256"]  # type: ignore[index]
            for receipt in consumed
        ):
            raise ValueError("this frozen Actor already has a final-seed reservation")
        used_seeds = {int(plan["final"]["seedBase"]) for plan in existing}  # type: ignore[index]
        used_seeds.update(int(receipt["final"]["seedBase"]) for receipt in consumed)  # type: ignore[index]
        existing_final_sets = [
            _evaluation_seed_set(
                str(plan["final"]["familyId"]),  # type: ignore[index]
                int(plan["final"]["seedBase"]),  # type: ignore[index]
                FINAL_MATCH_COUNTS,
            )
            for plan in existing
        ]
        existing_final_sets.extend(
            _evaluation_seed_set(
                str(receipt["final"]["familyId"]),  # type: ignore[index]
                int(receipt["final"]["seedBase"]),  # type: ignore[index]
                FINAL_MATCH_COUNTS,
            )
            for receipt in consumed
        )
        seed_base = V5_FIRST_FINAL_SEED_BASE
        while True:
            if seed_base > 0xFFFF_FFFF:
                raise RuntimeError("V5 final seed reservation namespace is exhausted")
            if seed_base in used_seeds:
                seed_base += V5_FINAL_SEED_STEP
                continue
            family_id = _canonical_final_family(seed_base)
            try:
                final_seed_set = _evaluation_seed_set(
                    family_id, seed_base, FINAL_MATCH_COUNTS
                )
            except RuntimeError:
                seed_base += V5_FINAL_SEED_STEP
                continue
            if (
                final_seed_set & screening_seed_set
                or any(final_seed_set & seeds for seeds in certification_seed_sets)
                or any(final_seed_set & seeds for seeds in existing_final_sets)
            ):
                seed_base += V5_FINAL_SEED_STEP
                continue
            break
        body: dict[str, object] = {
            "certification": {
                "developmentGates": dict(V5_DEVELOPMENT_GATES),
                "disjointSeedFamilies": True,
                "executionReservation": execution_record,
                "reports": certification_records,
            },
            "final": {
                "bootstrapResamples": DEFAULT_BOOTSTRAP_RESAMPLES,
                "evaluationProvenanceSha256": execution_provenance_sha,
                "evaluationGates": dict(EXACT_GATES),
                "familyId": family_id,
                "finalReportTrainingOrTuningUseAllowed": False,
                "matchPlan": {
                    str(player): FINAL_MATCH_COUNTS[player] for player in PLAYER_COUNTS
                },
                "matchShardCount": final_match_shard_count,
                "seedBase": seed_base,
            },
            "format": V5_PROMOTION_PLAN_FORMAT,
            "model": model,
            "version": V5_PROMOTION_PLAN_VERSION,
        }
        plan = {**body, "reservationId": _plan_id(body)}
        plan = _validate_plan(plan)
        plan_path = reservations / f"{plan['reservationId']}.json"
        plan_sha = _write_canonical_with_sidecar(plan_path, plan)
    return {
        "plan": plan,
        "planPath": str(plan_path),
        "planSha256": plan_sha,
    }


def _validate_claim_shape(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != {
        "claimId",
        "evaluationProvenance",
        "evaluationProvenanceSha256",
        "final",
        "format",
        "model",
        "outputPath",
        "planSha256",
        "reservationId",
        "shard",
        "started",
        "version",
    } or value.get("format") != V5_FINAL_CLAIM_FORMAT or value.get(
        "version"
    ) != V5_FINAL_CLAIM_VERSION or value.get("started") is not True:
        raise ValueError("V5 final shard claim contract drifted")
    final = value.get("final")
    shard = value.get("shard")
    model = value.get("model")
    provenance = value.get("evaluationProvenance")
    if (
        not isinstance(final, dict)
        or set(final) != {
            "bootstrapResamples",
            "familyId",
            "matchPlan",
            "seedBase",
        }
        or not isinstance(shard, dict)
        or set(shard) != {"count", "index"}
        or not isinstance(model, dict)
        or set(model) != V5_ACTOR_IDENTITY_KEYS
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("V5 final shard claim structure drifted")
    count = shard.get("count")
    index = shard.get("index")
    if (
        type(count) is not int
        or type(index) is not int
        or count < 1
        or not 0 <= index < count
    ):
        raise ValueError("V5 final shard claim index/count drifted")
    for name, digest in model.items():
        _require_sha(digest, f"claimed model {name}")
    verified_provenance = _validated_certification_execution_provenance(provenance)
    provenance_sha = _require_sha(
        value.get("evaluationProvenanceSha256"),
        "claim evaluation provenance SHA",
    )
    if verified_provenance["provenanceSha256"] != provenance_sha:
        raise ValueError("V5 final shard claim provenance SHA drifted")
    _canonical_registry_result_path(value.get("outputPath"))
    _require_sha(value.get("planSha256"), "claim plan SHA")
    _require_sha(value.get("reservationId"), "claim reservation ID")
    body = {key: value[key] for key in value if key != "claimId"}
    if value.get("claimId") != _claim_id(body):
        raise ValueError("V5 final shard claim ID drifted")
    return dict(value)


def _claim_report_binding(claim: Mapping[str, object]) -> dict[str, object]:
    return {
        "claimId": claim["claimId"],
        "evaluationProvenanceSha256": claim["evaluationProvenanceSha256"],
        "outputPath": claim["outputPath"],
        "reservationId": claim["reservationId"],
        "shard": claim["shard"],
    }


def _validate_claim_against_loaded_plan(
    path: Path,
    claim: Mapping[str, object],
    plan_path: Path,
    plan: Mapping[str, object],
    plan_sha: str,
) -> dict[str, object]:
    verified = _validate_claim_shape(claim)
    shard = verified["shard"]
    assert isinstance(shard, Mapping)
    if path.name != f"shard-{shard['index']}.json":
        raise ValueError("V5 final shard claim filename drifted")
    registry = path.parent.parent.parent
    expected_parent = registry / "claims" / str(verified["reservationId"])
    if path.parent != expected_parent:
        raise ValueError("V5 final shard claim is outside its registry")
    inferred_plan = registry / "reservations" / f"{verified['reservationId']}.json"
    if plan_path != inferred_plan:
        raise ValueError("V5 final shard claim belongs to a different promotion plan")
    final = plan["final"]
    assert isinstance(final, Mapping)
    expected_final = {
        "bootstrapResamples": final["bootstrapResamples"],
        "familyId": final["familyId"],
        "matchPlan": final["matchPlan"],
        "seedBase": final["seedBase"],
    }
    expected_output = (
        f"final-results/{verified['reservationId']}/"
        f"shard-{int(shard['index']):03d}.json"
    )
    certification = plan["certification"]
    assert isinstance(certification, Mapping)
    execution = certification["executionReservation"]
    assert isinstance(execution, Mapping)
    certification_reservation = load_v5_certification_execution_reservation(
        registry
        / "certification-reservations"
        / f"{execution['reservationId']}.json"
    )
    if (
        verified["reservationId"] != plan["reservationId"]
        or verified["planSha256"] != plan_sha
        or verified["model"] != plan["model"]
        or verified["final"] != expected_final
        or shard["count"] != final["matchShardCount"]
        or verified["evaluationProvenanceSha256"]
        != final["evaluationProvenanceSha256"]
        or verified["evaluationProvenance"]
        != certification_reservation["evaluationProvenance"]
        or verified["outputPath"] != expected_output
    ):
        raise ValueError("V5 final shard claim does not match its reservation")
    return verified


def load_v5_final_evaluation_claim(
    claim_path: str | Path,
    promotion_plan: str | Path | None = None,
) -> dict[str, object]:
    path = Path(claim_path).resolve()
    claim, digest = _strict_canonical_object(path, "V5 final shard claim")
    _verify_sidecar(path, digest)
    claim = _validate_claim_shape(claim)
    registry = path.parent.parent.parent
    inferred_plan = registry / "reservations" / f"{claim['reservationId']}.json"
    if promotion_plan is not None and Path(promotion_plan).resolve() != inferred_plan:
        raise ValueError("V5 final shard claim belongs to a different promotion plan")
    plan, plan_sha = _strict_canonical_object(inferred_plan, "V5 promotion plan")
    _verify_sidecar(inferred_plan, plan_sha)
    plan = _validate_plan(plan)
    _validate_plan_certification_snapshots(inferred_plan, plan)
    return _validate_claim_against_loaded_plan(
        path, claim, inferred_plan, plan, plan_sha
    )


def claim_v5_final_evaluation_shard(
    promotion_plan: str | Path,
    actor_bundle: str | Path,
    *,
    evaluation_provenance: Mapping[str, object],
    match_shard_count: int,
    match_shard_index: int,
) -> dict[str, object]:
    """Irreversibly claim one reserved shard before any final gameplay."""

    plan_path = Path(promotion_plan).resolve()
    plan, plan_sha = _strict_canonical_object(plan_path, "V5 promotion plan")
    _verify_sidecar(plan_path, plan_sha)
    plan = _validate_plan(plan)
    _validate_plan_certification_snapshots(plan_path, plan)
    model = _exact_actor_identity(actor_bundle)
    final = plan["final"]
    assert isinstance(final, Mapping)
    if model != plan["model"]:
        raise ValueError("final shard claim Actor does not match promotion plan")
    if (
        type(match_shard_count) is not int
        or type(match_shard_index) is not int
        or match_shard_count != final["matchShardCount"]
        or not 0 <= match_shard_index < match_shard_count
    ):
        raise ValueError("final shard claim count/index differs from reservation")
    if plan_path.parent.name != "reservations":
        raise ValueError("promotion plan is outside a canonical reservation registry")
    registry = plan_path.parent.parent
    provenance = _validated_certification_execution_provenance(
        evaluation_provenance
    )
    certification = plan["certification"]
    assert isinstance(certification, Mapping)
    execution = certification["executionReservation"]
    assert isinstance(execution, Mapping)
    certification_reservation = load_v5_certification_execution_reservation(
        registry
        / "certification-reservations"
        / f"{execution['reservationId']}.json"
    )
    if (
        provenance["provenanceSha256"]
        != final["evaluationProvenanceSha256"]
        or provenance != certification_reservation["evaluationProvenance"]
    ):
        raise ValueError(
            "final shard execution provenance differs from the promotion plan"
        )
    claims = registry / "claims" / str(plan["reservationId"])
    receipt = registry / "consumptions" / f"{plan['reservationId']}.json"
    output_path = (
        f"final-results/{plan['reservationId']}/"
        f"shard-{match_shard_index:03d}.json"
    )
    body: dict[str, object] = {
        "evaluationProvenance": provenance,
        "evaluationProvenanceSha256": provenance["provenanceSha256"],
        "final": {
            "bootstrapResamples": final["bootstrapResamples"],
            "familyId": final["familyId"],
            "matchPlan": final["matchPlan"],
            "seedBase": final["seedBase"],
        },
        "format": V5_FINAL_CLAIM_FORMAT,
        "model": model,
        "outputPath": output_path,
        "planSha256": plan_sha,
        "reservationId": plan["reservationId"],
        "shard": {"count": match_shard_count, "index": match_shard_index},
        "started": True,
        "version": V5_FINAL_CLAIM_VERSION,
    }
    expected_claim = {**body, "claimId": _claim_id(body)}
    with _registry_lock(registry):
        current_plan = load_v5_promotion_plan(plan_path)
        if current_plan != plan or _sha256_file(plan_path) != plan_sha:
            raise ValueError("promotion plan changed before final shard claim")
        if _exact_actor_identity(actor_bundle) != model:
            raise ValueError("Actor bundle changed before final shard claim")
        if receipt.exists() or receipt.with_name(receipt.name + ".sha256").exists():
            raise ValueError("final holdout reservation is already consumed")
        claims.mkdir(parents=True, exist_ok=True)
        claim_path = claims / f"shard-{match_shard_index}.json"
        if claim_path.exists() or claim_path.with_name(
            claim_path.name + ".sha256"
        ).exists():
            existing = load_v5_final_evaluation_claim(claim_path, plan_path)
            if existing != expected_claim:
                raise ValueError(
                    "existing final shard claim differs from deterministic retry"
                )
            claim_sha = _sha256_file(claim_path)
        else:
            claim_sha = _write_canonical_with_sidecar(claim_path, expected_claim)
    verified = load_v5_final_evaluation_claim(claim_path, plan_path)
    return {
        "claim": verified,
        "claimPath": str(claim_path),
        "claimSha256": claim_sha,
        "reportBinding": _claim_report_binding(verified),
    }


def authorize_v5_final_evaluation(
    promotion_plan: str | Path,
    claim_path: str | Path,
    model_identity: Mapping[str, object],
    *,
    evaluation_provenance: Mapping[str, object],
    family_id: str,
    seed_base: int,
    match_plan: Mapping[int, int],
    match_shard_count: int,
    match_shard_index: int,
    bootstrap_resamples: int,
    output_path: str | Path,
) -> dict[str, object]:
    """Validate the exact claim immediately before evaluator collection."""

    claim = load_v5_final_evaluation_claim(claim_path, promotion_plan)
    final = claim["final"]
    shard = claim["shard"]
    assert isinstance(final, Mapping) and isinstance(shard, Mapping)
    provenance = _validated_certification_execution_provenance(
        evaluation_provenance
    )
    expected_plan = {str(player): int(matches) for player, matches in match_plan.items()}
    if (
        claim["model"] != dict(model_identity)
        or final["familyId"] != family_id
        or final["seedBase"] != seed_base
        or final["matchPlan"] != expected_plan
        or final["bootstrapResamples"] != bootstrap_resamples
        or shard != {"count": match_shard_count, "index": match_shard_index}
        or claim["evaluationProvenance"] != provenance
        or claim["evaluationProvenanceSha256"]
        != provenance["provenanceSha256"]
    ):
        raise ValueError("final evaluator configuration does not match its start claim")
    claim_file = Path(claim_path).resolve()
    registry = claim_file.parent.parent.parent
    _resolve_registry_result_path(registry, claim["outputPath"], output_path)
    return _claim_report_binding(claim)


def _load_complete_claim_inventory(
    plan_path: Path, plan: Mapping[str, object]
) -> list[dict[str, object]]:
    """Reopen the exact canonical claim set and return report bindings."""

    if plan_path.parent.name != "reservations":
        raise ValueError("promotion plan is outside a canonical reservation registry")
    final = plan.get("final")
    if not isinstance(final, Mapping):
        raise ValueError("promotion plan omitted its final shard inventory")
    shard_count = final.get("matchShardCount")
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("promotion plan final shard count is invalid")
    registry = plan_path.parent.parent
    claim_directory = registry / "claims" / str(plan["reservationId"])
    if not claim_directory.is_dir():
        raise ValueError("final approval requires every reserved shard start claim")
    expected_names = {
        name
        for index in range(shard_count)
        for name in (f"shard-{index}.json", f"shard-{index}.json.sha256")
    }
    actual_names = {path.name for path in claim_directory.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "final approval requires every reserved shard start claim; "
            "canonical inventory is incomplete or drifted"
        )
    bindings: list[dict[str, object]] = []
    current_plan, plan_sha = _strict_canonical_object(plan_path, "V5 promotion plan")
    _verify_sidecar(plan_path, plan_sha)
    if current_plan != dict(plan):
        raise ValueError("promotion plan changed while loading final shard claims")
    for index in range(shard_count):
        claim_path = claim_directory / f"shard-{index}.json"
        claim, claim_sha = _strict_canonical_object(
            claim_path, "V5 final shard claim"
        )
        _verify_sidecar(claim_path, claim_sha)
        claim = _validate_claim_against_loaded_plan(
            claim_path, claim, plan_path, plan, plan_sha
        )
        if claim.get("shard") != {"count": shard_count, "index": index}:
            raise ValueError("canonical final shard claim index/count drifted")
        bindings.append(_claim_report_binding(claim))
    return bindings


def _validate_final_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    model: Mapping[str, str],
    claim_bindings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    final = plan["final"]
    assert isinstance(final, Mapping)
    if report.get("mode") != "final":
        raise ValueError("consumption requires a final-mode evaluation report")
    if report.get("familyId") != final["familyId"] or report.get("seedBase") != final["seedBase"]:
        raise ValueError("final report used an unreserved family or seed")
    if report.get("model") != model or model != plan.get("model"):
        raise ValueError("final report, plan, and Actor bundle model bindings disagree")
    if report.get("matchPlan") != final["matchPlan"]:
        raise ValueError("final report did not use the exact reserved match counts")
    if _report_resamples(report) != DEFAULT_BOOTSTRAP_RESAMPLES:
        raise ValueError("final report requires exactly 10000 bootstrap resamples")
    if report.get("shard") != {"count": 1, "index": 0}:
        raise ValueError("final approval requires one merged complete report")
    if report.get("completeEvaluation") is not True or report.get(
        "allPlayerCountsPassed"
    ) is not True:
        raise ValueError("final report is incomplete or failed an exit gate")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("final report omitted per-player-count results")
    observed = {
        int(result["playerCount"]): int(result["matches"])
        for result in results
        if isinstance(result, Mapping)
    }
    if observed != FINAL_MATCH_COUNTS:
        raise ValueError("final report does not contain every exact match count")
    report_claims = report.get("finalClaims")
    if (
        not isinstance(report_claims, list)
        or canonical_json_bytes(report_claims)
        != canonical_json_bytes([dict(value) for value in claim_bindings])
    ):
        raise ValueError(
            "final report claim bindings do not match the canonical registry inventory"
        )
    provenance = _report_provenance_summary(report, "final report")
    if set(provenance["provenanceSha256"]) != {
        final["evaluationProvenanceSha256"]
    }:
        raise ValueError(
            "final report execution provenance differs from its promotion plan"
        )
    certification = plan.get("certification")
    if not isinstance(certification, Mapping) or not isinstance(
        certification.get("reports"), list
    ):
        raise ValueError("promotion plan omitted certification provenance")
    expected_sources = {
        (
            record.get("evaluationSourceCommit"),
            record.get("evaluationSourceBindingSha256"),
        )
        for record in certification["reports"]
        if isinstance(record, Mapping)
    }
    if expected_sources != {
        (provenance["sourceCommit"], provenance["sourceBindingSha256"])
    }:
        raise ValueError(
            "final report evaluator/Normal source differs from certification"
        )
    return provenance


def _receipt_body(
    plan: Mapping[str, object],
    plan_sha: str,
    report_sha: str,
    claim_bindings: Sequence[Mapping[str, object]],
    final_provenance: Mapping[str, object],
) -> dict[str, object]:
    certification = plan["certification"]
    final = plan["final"]
    assert isinstance(certification, Mapping) and isinstance(final, Mapping)
    reports = certification["reports"]
    assert isinstance(reports, list)
    certification_provenance = [
        list(report["evaluationProvenanceSha256"])  # type: ignore[index]
        for report in reports
    ]
    return {
        "approved": True,
        "certificationReportSha256": [
            report["reportSha256"] for report in reports  # type: ignore[index]
        ],
        "consumed": True,
        "evaluationProvenance": {
            "certificationProvenanceSha256": certification_provenance,
            "finalProvenanceSha256": list(final_provenance["provenanceSha256"]),  # type: ignore[arg-type]
            "sourceBindingSha256": final_provenance["sourceBindingSha256"],
            "sourceCommit": final_provenance["sourceCommit"],
        },
        "finalClaims": [dict(value) for value in claim_bindings],
        "final": {
            "familyId": final["familyId"],
            "reportSha256": report_sha,
            "seedBase": final["seedBase"],
        },
        "finalReportTrainingOrTuningUseAllowed": False,
        "format": V5_CONSUMPTION_RECEIPT_FORMAT,
        "model": plan["model"],
        "planSha256": plan_sha,
        "reservationId": plan["reservationId"],
        "version": V5_CONSUMPTION_RECEIPT_VERSION,
    }


def approve_v5_final_holdout(
    promotion_plan: str | Path,
    actor_bundle: str | Path,
    final_report_path: str | Path,
    *,
    expected_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Approve once and exclusively consume the reserved final seed family."""

    plan_path = Path(promotion_plan).resolve()
    plan, plan_sha = _strict_canonical_object(plan_path, "V5 promotion plan")
    _verify_sidecar(plan_path, plan_sha)
    plan = _validate_plan(plan)
    _validate_plan_certification_snapshots(plan_path, plan)
    model = _exact_actor_identity(actor_bundle)
    report, report_sha = _load_evaluation_report(final_report_path)
    if expected_report is not None and canonical_json_bytes(dict(expected_report)) != canonical_json_bytes(report):
        raise ValueError("in-memory final report differs from the immutable report file")
    claim_bindings = _load_complete_claim_inventory(plan_path, plan)
    final_provenance = _validate_final_report(
        report, plan, model, claim_bindings
    )
    certification_hashes = {
        record["reportSha256"]  # type: ignore[index]
        for record in plan["certification"]["reports"]  # type: ignore[index]
    }
    if report_sha in certification_hashes:
        raise ValueError("final holdout report cannot be a certification/tuning report")
    if plan_path.parent.name != "reservations":
        raise ValueError("promotion plan is outside a canonical reservation registry")
    registry = plan_path.parent.parent
    receipt_path = registry / "consumptions" / f"{plan['reservationId']}.json"
    with _registry_lock(registry):
        # Revalidate every external snapshot inside the exclusive consumption
        # boundary immediately before writing the irreversible receipt.
        current_plan, current_plan_sha = _strict_canonical_object(
            plan_path, "V5 promotion plan"
        )
        _verify_sidecar(plan_path, current_plan_sha)
        if current_plan_sha != plan_sha or _validate_plan(current_plan) != plan:
            raise ValueError("promotion plan changed before consumption")
        _validate_plan_certification_snapshots(plan_path, plan)
        if _exact_actor_identity(actor_bundle) != model:
            raise ValueError("Actor bundle changed before consumption")
        current_report, current_report_sha = _load_evaluation_report(final_report_path)
        if current_report_sha != report_sha or current_report != report:
            raise ValueError("final report changed before consumption")
        current_claim_bindings = _load_complete_claim_inventory(plan_path, plan)
        if current_claim_bindings != claim_bindings:
            raise ValueError("final shard claim inventory changed before consumption")
        current_final_provenance = _validate_final_report(
            current_report, plan, model, current_claim_bindings
        )
        if current_final_provenance != final_provenance:
            raise ValueError("final evaluation provenance changed before consumption")
        if receipt_path.exists() or receipt_path.with_name(
            receipt_path.name + ".sha256"
        ).exists():
            raise FileExistsError("this final holdout reservation was already consumed")
        body = _receipt_body(
            plan,
            plan_sha,
            report_sha,
            current_claim_bindings,
            current_final_provenance,
        )
        receipt = {**body, "receiptId": _receipt_id(body)}
        receipt_sha = _write_canonical_with_sidecar(receipt_path, receipt)
    verified = verify_v5_final_consumption_receipt(
        receipt_path, plan_path, actor_bundle, final_report_path
    )
    return {
        "approved": True,
        "receipt": verified,
        "receiptPath": str(receipt_path),
        "receiptSha256": receipt_sha,
    }


def verify_v5_final_consumption_receipt(
    receipt_path: str | Path,
    promotion_plan: str | Path,
    actor_bundle: str | Path,
    final_report_path: str | Path,
) -> dict[str, object]:
    path = Path(receipt_path).resolve()
    receipt, receipt_sha = _strict_canonical_object(
        path, "V5 final consumption receipt"
    )
    _verify_sidecar(path, receipt_sha)
    receipt = _validate_receipt_shape(receipt)
    plan_path = Path(promotion_plan).resolve()
    plan, plan_sha = _strict_canonical_object(plan_path, "V5 promotion plan")
    _verify_sidecar(plan_path, plan_sha)
    plan = _validate_plan(plan)
    _validate_plan_certification_snapshots(plan_path, plan)
    model = _exact_actor_identity(actor_bundle)
    report, report_sha = _load_evaluation_report(final_report_path)
    claim_bindings = _load_complete_claim_inventory(plan_path, plan)
    final_provenance = _validate_final_report(report, plan, model, claim_bindings)
    body = _receipt_body(
        plan, plan_sha, report_sha, claim_bindings, final_provenance
    )
    expected = {**body, "receiptId": _receipt_id(body)}
    if receipt != expected:
        raise ValueError("V5 final consumption receipt bindings do not recompute")
    if path.name != f"{plan['reservationId']}.json":
        raise ValueError("V5 final consumption receipt filename drifted")
    expected_parent = plan_path.parent.parent / "consumptions"
    if path.parent != expected_parent:
        raise ValueError("V5 final consumption receipt is outside its registry")
    return receipt


# Workflow aliases.
reserve_final_holdout = reserve_v5_final_holdout
approve_final_holdout = approve_v5_final_holdout
verify_consumption_receipt = verify_v5_final_consumption_receipt


__all__ = [
    "V5_ACTOR_IDENTITY_KEYS",
    "V5_CERTIFICATION_RESERVATION_FORMAT",
    "V5_CERTIFICATION_REPORT_COUNT",
    "V5_CONSUMPTION_RECEIPT_FORMAT",
    "V5_DEVELOPMENT_GATES",
    "V5_FINAL_CLAIM_FORMAT",
    "V5_FINAL_SEED_STEP",
    "V5_FIRST_FINAL_SEED_BASE",
    "V5_PROMOTION_LOCK_FORMAT",
    "V5_PROMOTION_LOCK_RECOVERY_FORMAT",
    "V5_PROMOTION_PLAN_FORMAT",
    "V5_SCREENING_RESERVATION_FORMAT",
    "approve_final_holdout",
    "approve_v5_final_holdout",
    "authorize_v5_certification_evaluation",
    "authorize_v5_final_evaluation",
    "authorize_v5_screening_evaluation",
    "claim_v5_final_evaluation_shard",
    "load_v5_final_evaluation_claim",
    "load_v5_certification_execution_reservation",
    "load_v5_promotion_plan",
    "load_v5_screening_execution_reservation",
    "recover_v5_promotion_lock",
    "reserve_final_holdout",
    "reserve_v5_certification_execution",
    "reserve_v5_final_holdout",
    "reserve_v5_screening_execution",
    "v5_certification_coordinates",
    "verify_consumption_receipt",
    "verify_v5_final_consumption_receipt",
]
