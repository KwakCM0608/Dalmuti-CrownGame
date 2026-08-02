from __future__ import annotations

"""CLI for immutable DALMUTI V5 collection, calibration, and indexing.

This module intentionally contains no SSH operations.  The same commands run
locally or in a sealed remote source checkout; immutable plans and SHA-256
bindings are the coordination protocol.
"""

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
import time
from typing import Mapping, Sequence
import uuid

import torch

from v5_collect_mappo import (
    V5MAPPOCollectionConfig,
    V5PublishedCollection,
    V5TorchInferenceRuntime,
    collect_v5_mappo,
    publish_v5_mappo_collection,
)
from v5_collection_plan import (
    DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT,
    DEFAULT_ACTUAL_STRATUM_RELATIVE_TOLERANCE,
    DEFAULT_MAX_MATCHES_PER_SHARD,
    DEFAULT_TARGET_NONFORCED,
    DEFAULT_TARGET_NONFORCED_MAX,
    DEFAULT_TARGET_NONFORCED_MIN,
    DEFAULT_TOTAL_MATCHES,
    MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM,
    V5_CALIBRATION_SCHEDULE_CONTRACT,
    V5CollectionPlan,
    build_collection_plan,
    build_source_inventory,
    calibration_schedule_id,
    canonical_json_bytes,
    expected_planned_shard_metadata,
    load_collection_plan,
    load_verified_calibration_report,
    planned_shard_path,
    publish_calibration_report,
    publish_collection_plan,
    publish_verified_index,
    resume_verified_shard,
    sha256_file,
    source_inventory_sha256,
    verify_planned_shard,
)
from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_model import V5_POLICY_NUMERICS_SHA256


# This schedule supplies both exact CPU/CUDA parity evidence and the rate used
# to size a 1.5--2.0M-decision production corpus.  A smaller override remains
# useful as a parity-only diagnostic, but is rejected for production sizing.
DEFAULT_CALIBRATION_MATCH_COUNTS = tuple((player_count, 32) for player_count in range(4, 11))
V5_THROUGHPUT_PREFLIGHT_FORMAT = "dalmuti-v5-backend-throughput-preflight"
V5_THROUGHPUT_PREFLIGHT_VERSION = 1
V5_WORKER_SLOT_LOCK_FORMAT = "dalmuti-v5-worker-slot-lock"
V5_WORKER_SLOT_RECOVERY_FORMAT = "dalmuti-v5-worker-slot-recovery"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class V5LoadedBehavior:
    actor: torch.nn.Module
    critic: torch.nn.Module
    actor_sha256: str
    actor_manifest_sha256: str
    critic_sha256: str
    policy_numerics_sha256: str
    pair_id: str
    pair_manifest_sha256: str

    @property
    def hashes(self) -> dict[str, str]:
        return {
            "actorManifestSha256": self.actor_manifest_sha256,
            "actorSha256": self.actor_sha256,
            "criticSha256": self.critic_sha256,
            "pairId": self.pair_id,
            "pairManifestSha256": self.pair_manifest_sha256,
        }


def _publish_report_file(path: str | Path, value: Mapping[str, object]) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(dict(value))
    digest = hashlib.sha256(raw).hexdigest()
    with target.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    try:
        with target.with_name(target.name + ".sha256").open("xb") as sidecar:
            sidecar.write(f"{digest}  {target.name}\n".encode("ascii"))
            sidecar.flush()
            os.fsync(sidecar.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return digest


def load_verified_behavior(
    actor_bundle: str | Path,
    critic_checkpoint: str | Path,
    behavior_pair: str | Path,
) -> V5LoadedBehavior:
    """Load exact verified Actor and critic artifacts and bind file hashes."""

    from v5_export import load_v5_actor_bundle, v5_actor_bundle_digests
    from v5_train import load_v5_critic_checkpoint, load_verified_v5_behavior_pair

    bundle = Path(actor_bundle).resolve()
    checkpoint = Path(critic_checkpoint).resolve()
    digests = v5_actor_bundle_digests(bundle)
    if digests.get("publicContractSha256") != V5_PUBLIC_CONTRACT_SHA256:
        raise ValueError("behavior Actor public contract fingerprint drifted")
    if digests.get("policyNumericsSha256") != V5_POLICY_NUMERICS_SHA256:
        raise ValueError("behavior Actor policy numerics fingerprint drifted")
    actor, _ = load_v5_actor_bundle(bundle)
    critic, payload = load_v5_critic_checkpoint(checkpoint)
    critic_sha = sha256_file(checkpoint)
    pair = load_verified_v5_behavior_pair(Path(behavior_pair).resolve())
    expected_pair_hashes = {
        "actorManifestSha256": digests["manifestSha256"],
        "actorSha256": digests["actorSha256"],
        "actorTensorStateSha256": digests["tensorStateSha256"],
        "criticSha256": critic_sha,
        "criticTensorStateSha256": payload.get("tensorStateSha256")
        if isinstance(payload, Mapping)
        else None,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
    }
    for name, expected in expected_pair_hashes.items():
        if pair.get(name) != expected:
            raise ValueError(f"Actor/critic pair binding mismatch: {name}")
    # The critic loader verifies the tensor-state fingerprint.  If the
    # producer also records the file digest in metadata, it may not disagree.
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if isinstance(metadata, Mapping) and "criticSha256" in metadata and metadata[
        "criticSha256"
    ] != critic_sha:
        raise ValueError("critic checkpoint metadata file hash disagrees")
    return V5LoadedBehavior(
        actor=actor,
        critic=critic,
        actor_sha256=str(digests["actorSha256"]),
        actor_manifest_sha256=str(digests["manifestSha256"]),
        critic_sha256=critic_sha,
        policy_numerics_sha256=str(digests["policyNumericsSha256"]),
        pair_id=str(pair["pairId"]),
        pair_manifest_sha256=str(pair["pairManifestSha256"]),
    )


def _require_behavior_matches(
    expected: Mapping[str, str], actual: V5LoadedBehavior
) -> None:
    if dict(expected) != actual.hashes:
        mismatches = sorted(
            name for name in set(expected) | set(actual.hashes)
            if expected.get(name) != actual.hashes.get(name)
        )
        raise ValueError(f"behavior artifact hash mismatch: {mismatches[0]}")


def _require_source_matches(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> None:
    if dict(expected) != dict(actual):
        names = sorted(set(expected) | set(actual))
        mismatch = next(
            (name for name in names if expected.get(name) != actual.get(name)),
            "inventory",
        )
        raise ValueError(f"source inventory hash mismatch: {mismatch}")


def _device_backend(device: str | torch.device) -> str:
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("device is not a valid Torch device") from error
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError("V5 mixed collection supports only cpu and cuda backends")
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA collection was requested but CUDA is unavailable")
    return parsed.type


def _planned_worker_topology(
    plan: V5CollectionPlan, backend: str
) -> tuple[int, int]:
    policy = plan.document.get("backendPolicy")
    if not isinstance(policy, Mapping) or backend not in {"cpu", "cuda"}:
        raise ValueError("collection plan worker topology is missing")
    prefix = "cpu" if backend == "cpu" else "cuda"
    workers = policy.get(f"{prefix}WorkerCount")
    threads = policy.get(f"{prefix}TorchThreadsPerWorker")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads < 1
    ):
        raise ValueError("collection plan worker topology is invalid")
    return workers, threads


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
            # remainder[0] is field 3; starttime is proc stat field 22.
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
        # ERROR_INVALID_PARAMETER is the documented response when no process
        # has this PID. Access-denied and unknown failures remain fail-closed.
        return int(ctypes.windll.kernel32.GetLastError()) != 87  # type: ignore[attr-defined]
    return True


def _worker_lock_document(
    plan: V5CollectionPlan, backend: str, slot: int
) -> dict[str, object]:
    start = _process_start_ticks(os.getpid())
    if start is None:
        raise RuntimeError("could not obtain the worker process start identity")
    return {
        "backend": backend,
        "bootId": _host_boot_id(),
        "format": V5_WORKER_SLOT_LOCK_FORMAT,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "planSha256": plan.manifest_sha256,
        "processStartTicks": start,
        "slot": slot,
        "version": 1,
    }


def _load_worker_lock(path: Path) -> tuple[dict[str, object], bytes, str]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_json_pairs(pairs, "worker lock"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"worker lock contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("worker slot lock is not canonical JSON") from error
    expected = {
        "backend", "bootId", "format", "hostname", "pid", "planSha256",
        "processStartTicks", "slot", "version",
    }
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
        or set(value) != expected
        or value.get("format") != V5_WORKER_SLOT_LOCK_FORMAT
        or value.get("version") != 1
        or value.get("backend") not in {"cpu", "cuda"}
        or not isinstance(value.get("bootId"), str)
        or not isinstance(value.get("hostname"), str)
        or type(value.get("pid")) is not int
        or int(value["pid"]) < 1
        or type(value.get("processStartTicks")) is not int
        or int(value["processStartTicks"]) < 0
        or type(value.get("slot")) is not int
        or int(value["slot"]) < 0
        or not isinstance(value.get("planSha256"), str)
        or _SHA256.fullmatch(str(value["planSha256"])) is None
    ):
        raise ValueError("worker slot lock contract drifted")
    return value, raw, hashlib.sha256(raw).hexdigest()


def _unique_json_pairs(
    pairs: list[tuple[str, object]], label: str
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"{label} contains duplicate key {key}")
        value[key] = item
    return value


@contextmanager
def _planned_worker_slot(
    plan: V5CollectionPlan, shards_root: str | Path, backend: str
):
    workers, threads = _planned_worker_topology(plan, backend)
    root = Path(shards_root).resolve()
    slots = root / ".v5-worker-slots" / backend
    slots.mkdir(parents=True, exist_ok=True)
    claimed: Path | None = None
    descriptor: int | None = None
    claimed_payload: bytes | None = None
    for index in range(workers):
        candidate = slots / f"slot-{index:03d}.lock"
        try:
            descriptor = os.open(
                candidate,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except FileExistsError:
            continue
        claimed = candidate
        try:
            claimed_payload = canonical_json_bytes(
                _worker_lock_document(plan, backend, index)
            )
            written = 0
            while written < len(claimed_payload):
                count = os.write(descriptor, claimed_payload[written:])
                if count <= 0:
                    raise OSError("worker lock write made no progress")
                written += count
            os.fsync(descriptor)
        except Exception:
            descriptor_identity = os.fstat(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                path_identity = candidate.stat()
            except FileNotFoundError:
                pass
            else:
                if (
                    descriptor_identity.st_dev == path_identity.st_dev
                    and descriptor_identity.st_ino == path_identity.st_ino
                ):
                    candidate.unlink()
            raise
        else:
            os.close(descriptor)
            descriptor = None
        break
    if claimed is None:
        raise RuntimeError(
            f"planned {backend} worker concurrency ({workers}) is already saturated"
        )
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        yield {"slot": claimed.name, "threads": threads, "workers": workers}
    finally:
        torch.set_num_threads(prior_threads)
        if descriptor is not None:
            os.close(descriptor)
        if claimed_payload is None or claimed.read_bytes() != claimed_payload:
            raise RuntimeError("worker slot lock changed while its process was active")
        claimed.unlink()


def recover_v5_worker_slot(
    plan_path: str | Path,
    shards_root: str | Path,
    *,
    backend: str,
    slot_index: int,
    recovery_reason: str,
) -> dict[str, object]:
    """Evidence-retire one provably inactive or operator-audited stale lock."""

    plan = load_collection_plan(plan_path)
    workers, _ = _planned_worker_topology(plan, backend)
    if type(slot_index) is not int or not 0 <= slot_index < workers:
        raise ValueError("worker slot recovery index is outside the plan topology")
    if (
        not isinstance(recovery_reason, str)
        or recovery_reason.strip() != recovery_reason
        or not recovery_reason
        or len(recovery_reason) > 240
        or any(ord(character) < 32 for character in recovery_reason)
    ):
        raise ValueError("recovery reason must be 1..240 printable characters")
    root = Path(shards_root).resolve()
    lock = root / ".v5-worker-slots" / backend / f"slot-{slot_index:03d}.lock"
    value, raw, lock_sha = _load_worker_lock(lock)
    if (
        value["planSha256"] != plan.manifest_sha256
        or value["backend"] != backend
        or value["slot"] != slot_index
    ):
        raise ValueError("stale worker lock does not belong to this plan/backend/slot")
    observer = {
        "bootId": _host_boot_id(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "processStartTicks": _process_start_ticks(os.getpid()),
    }
    if observer["processStartTicks"] is None:
        raise RuntimeError("could not bind worker recovery to its observer process")
    same_boot = (
        value["hostname"] == observer["hostname"]
        and value["bootId"] == observer["bootId"]
    )
    if value["hostname"] != observer["hostname"]:
        raise RuntimeError(
            "worker slot lock belongs to another host; remote process inactivity "
            "cannot be proven here"
        )
    observed_start = (
        _process_start_ticks(int(value["pid"])) if same_boot else None
    )
    if same_boot and observed_start == value["processStartTicks"]:
        raise RuntimeError("worker slot lock still belongs to the active process")
    if (
        same_boot
        and observed_start is None
        and _process_may_exist(int(value["pid"]))
    ):
        raise RuntimeError(
            "worker slot PID may still be active but its process-start identity "
            "could not be read"
        )
    stale_evidence = (
        "process-missing"
        if same_boot and observed_start is None
        else "pid-reused"
        if same_boot
        else "host-rebooted"
    )
    recovery_directory = root / ".v5-worker-slot-recoveries" / backend
    retired_directory = recovery_directory / "retired-locks"
    recovery_directory.mkdir(parents=True, exist_ok=True)
    retired_directory.mkdir(parents=True, exist_ok=True)
    receipt = recovery_directory / f"slot-{slot_index:03d}-{lock_sha}.json"
    retired = retired_directory / f"slot-{slot_index:03d}-{lock_sha}.lock"
    document = {
        "backend": backend,
        "format": V5_WORKER_SLOT_RECOVERY_FORMAT,
        "observedStaleEvidence": stale_evidence,
        "observer": observer,
        "oldLock": value,
        "oldLockSha256": lock_sha,
        "planSha256": plan.manifest_sha256,
        "recoveredAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "recoveryReason": recovery_reason,
        "retiredLock": str(retired.relative_to(root)).replace("\\", "/"),
        "slot": slot_index,
        "version": 1,
    }
    receipt_payload = canonical_json_bytes(document)
    resumed = False
    try:
        descriptor = os.open(
            receipt,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError:
        existing = _load_recovery_receipt(receipt)
        if (
            existing.get("oldLockSha256") != lock_sha
            or existing.get("planSha256") != plan.manifest_sha256
            or existing.get("backend") != backend
            or existing.get("slot") != slot_index
            or existing.get("oldLock") != value
            or existing.get("recoveryReason") != recovery_reason
            or existing.get("retiredLock")
            != str(retired.relative_to(root)).replace("\\", "/")
        ):
            raise ValueError("existing worker recovery receipt belongs to another lock")
        resumed = True
    else:
        with os.fdopen(descriptor, "wb") as output:
            output.write(receipt_payload)
            output.flush()
            os.fsync(output.fileno())
    refreshed, refreshed_raw, refreshed_sha = _load_worker_lock(lock)
    if refreshed != value or refreshed_raw != raw or refreshed_sha != lock_sha:
        raise RuntimeError("worker slot lock changed before evidence retirement")
    try:
        os.link(lock, retired)
    except FileExistsError:
        if not resumed or retired.read_bytes() != raw:
            raise FileExistsError("worker slot retired-lock target already exists")
    try:
        if retired.read_bytes() != raw:
            raise RuntimeError("retired worker lock link differs from source lock")
        refreshed, refreshed_raw, refreshed_sha = _load_worker_lock(lock)
        if refreshed != value or refreshed_raw != raw or refreshed_sha != lock_sha:
            raise RuntimeError("worker slot lock changed after retirement link")
        lock.unlink()
    except Exception:
        # The source lock deliberately remains, so the slot stays fail-closed.
        raise
    return {
        "backend": backend,
        "oldLockSha256": lock_sha,
        "receipt": str(receipt),
        "resumedRecovery": resumed,
        "retiredLock": str(retired),
        "slot": slot_index,
    }


def _load_recovery_receipt(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_json_pairs(
                pairs, "worker recovery receipt"
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(
                    f"worker recovery receipt contains non-finite number {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("worker recovery receipt is not canonical JSON") from error
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
        or set(value) != {
            "backend", "format", "observedStaleEvidence", "observer",
            "oldLock", "oldLockSha256", "planSha256", "recoveredAt",
            "recoveryReason", "retiredLock", "slot", "version",
        }
        or value.get("format") != V5_WORKER_SLOT_RECOVERY_FORMAT
        or value.get("version") != 1
        or value.get("backend") not in {"cpu", "cuda"}
        or value.get("observedStaleEvidence") not in {
            "process-missing", "pid-reused", "host-rebooted"
        }
        or not isinstance(value.get("observer"), Mapping)
        or not isinstance(value.get("oldLock"), Mapping)
        or not isinstance(value.get("oldLockSha256"), str)
        or _SHA256.fullmatch(str(value["oldLockSha256"])) is None
        or not isinstance(value.get("planSha256"), str)
        or _SHA256.fullmatch(str(value["planSha256"])) is None
        or not isinstance(value.get("recoveredAt"), str)
        or not isinstance(value.get("recoveryReason"), str)
        or not isinstance(value.get("retiredLock"), str)
        or type(value.get("slot")) is not int
    ):
        raise ValueError("worker recovery receipt contract drifted")
    if hashlib.sha256(canonical_json_bytes(value["oldLock"])).hexdigest() != value[
        "oldLockSha256"
    ]:
        raise ValueError("worker recovery receipt old-lock SHA drifted")
    return value


def collect_calibration_snapshot(
    *,
    actor_bundle: str | Path,
    critic_checkpoint: str | Path,
    behavior_pair: str | Path,
    source_root: str | Path,
    output: str | Path,
    backend: str,
    device: str,
    run_namespace: str,
    seed_base: int,
    lane_count: int = 7,
    match_counts: tuple[tuple[int, int], ...] = DEFAULT_CALIBRATION_MATCH_COUNTS,
    source_files: Sequence[str] | None = None,
) -> V5PublishedCollection:
    actual_backend = _device_backend(device)
    if backend not in {"cpu", "cuda"} or actual_backend != backend:
        raise ValueError("calibration backend must equal the actual Torch device backend")
    behavior = load_verified_behavior(actor_bundle, critic_checkpoint, behavior_pair)
    sources = build_source_inventory(
        source_root,
        tuple(source_files) if source_files is not None else None,  # type: ignore[arg-type]
    ) if source_files is not None else build_source_inventory(source_root)
    config = V5MAPPOCollectionConfig(
        run_namespace=run_namespace,
        seed_base=seed_base,
        match_counts=match_counts,
        require_all_player_counts=True,
        lane_count=lane_count,
    )
    runtime = V5TorchInferenceRuntime(
        behavior.actor,
        behavior.critic,
        device=device,
    )
    collection = collect_v5_mappo(runtime.actor_batch, runtime.critic_batch, config)
    metadata = {
        "calibrationBackend": backend,
        "calibrationScheduleContract": V5_CALIBRATION_SCHEDULE_CONTRACT,
        "calibrationScheduleId": calibration_schedule_id(
            run_namespace, seed_base, match_counts
        ),
        "behaviorModelPairId": behavior.pair_id,
        "behaviorModelPairManifestSha256": behavior.pair_manifest_sha256,
        "sourceInventory": sources,
        "sourceInventorySha256": source_inventory_sha256(sources),
        "policyNumericsSha256": behavior.policy_numerics_sha256,
    }
    return publish_v5_mappo_collection(
        output,
        collection,
        behavior_actor_sha256=behavior.actor_sha256,
        behavior_actor_manifest_sha256=behavior.actor_manifest_sha256,
        behavior_critic_sha256=behavior.critic_sha256,
        metadata=metadata,
    )


def collect_throughput_preflight(
    *,
    actor_bundle: str | Path,
    critic_checkpoint: str | Path,
    behavior_pair: str | Path,
    source_root: str | Path,
    output: str | Path,
    backend: str,
    device: str,
    run_namespace: str,
    seed_base: int,
    scratch_root: str | Path,
    matches_per_player_count: int = MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM,
    lane_count: int = 32,
    source_files: Sequence[str] | None = None,
) -> dict[str, object]:
    """Measure actual end-to-end shard seconds/match independently for p4..p10."""

    actual_backend = _device_backend(device)
    if backend not in {"cpu", "cuda"} or actual_backend != backend:
        raise ValueError("throughput backend must equal the actual Torch device backend")
    if (
        isinstance(matches_per_player_count, bool)
        or not isinstance(matches_per_player_count, int)
        or matches_per_player_count < MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM
    ):
        raise ValueError(
            "throughput preflight requires at least 20 complete matches per p-stratum"
        )
    behavior = load_verified_behavior(actor_bundle, critic_checkpoint, behavior_pair)
    sources = (
        build_source_inventory(source_root, tuple(source_files))
        if source_files is not None
        else build_source_inventory(source_root)
    )
    scratch = Path(scratch_root).resolve()
    if scratch.is_symlink() or not scratch.is_dir():
        raise ValueError("throughput scratch root must be a real existing directory")
    runtime = V5TorchInferenceRuntime(behavior.actor, behavior.critic, device=device)
    measurements: dict[str, dict[str, object]] = {}
    for player_count in range(4, 11):
        config = V5MAPPOCollectionConfig(
            run_namespace=run_namespace,
            seed_base=seed_base,
            match_counts=((player_count, matches_per_player_count),),
            require_all_player_counts=False,
            lane_count=lane_count,
        )
        temporary = Path(
            tempfile.mkdtemp(prefix=f".v5-throughput-p{player_count}-", dir=scratch)
        )
        try:
            started = time.perf_counter()
            collection = collect_v5_mappo(
                runtime.actor_batch, runtime.critic_batch, config
            )
            published = publish_v5_mappo_collection(
                temporary / "shard",
                collection,
                behavior_actor_sha256=behavior.actor_sha256,
                behavior_actor_manifest_sha256=behavior.actor_manifest_sha256,
                behavior_critic_sha256=behavior.critic_sha256,
                metadata={
                    "behaviorModelPairId": behavior.pair_id,
                    "behaviorModelPairManifestSha256": behavior.pair_manifest_sha256,
                    "throughputPreflight": True,
                },
            )
            elapsed = time.perf_counter() - started
            if elapsed <= 0.0:
                raise RuntimeError("throughput timer did not advance")
            measurements[str(player_count)] = {
                "decisions": published.decisions,
                "elapsedSeconds": elapsed,
                "matches": published.matches,
                "nonforcedDecisions": published.nonforced_decisions,
                "secondsPerMatch": elapsed / published.matches,
            }
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    report: dict[str, object] = {
        "backend": backend,
        "behavior": behavior.hashes,
        "format": V5_THROUGHPUT_PREFLIGHT_FORMAT,
        "laneCount": lane_count,
        "matchesPerPlayerCount": matches_per_player_count,
        "measurements": measurements,
        "policyNumericsSha256": behavior.policy_numerics_sha256,
        "runNamespace": run_namespace,
        "seedBase": seed_base,
        "sourceInventory": sources,
        "sourceInventorySha256": source_inventory_sha256(sources),
        "version": V5_THROUGHPUT_PREFLIGHT_VERSION,
    }
    digest = _publish_report_file(output, report)
    return {
        "backend": backend,
        "output": str(Path(output).resolve()),
        "reportSha256": digest,
        "secondsPerMatch": {
            player: measurements[str(player)]["secondsPerMatch"]
            for player in range(4, 11)
        },
    }


def _verify_calibration_for_plan(
    report_path: str | Path,
    cpu_snapshot: str | Path,
    cuda_snapshot: str | Path,
    behavior: V5LoadedBehavior,
    sources: Mapping[str, str],
) -> tuple[Mapping[str, object], str]:
    report, digest = load_verified_calibration_report(
        report_path, cpu_snapshot, cuda_snapshot
    )
    bindings = report.get("behavior")
    if not isinstance(bindings, Mapping) or dict(bindings) != {
        **behavior.hashes,
        "policyNumericsSha256": behavior.policy_numerics_sha256,
        "sourceInventorySha256": source_inventory_sha256(sources),
    }:
        raise ValueError("calibration behavior/source hashes do not match production inputs")
    return report, digest


def _preflight_from_calibration_report(
    report: Mapping[str, object],
) -> dict[int, tuple[int, int]]:
    measurements = report.get("measurements")
    if not isinstance(measurements, Mapping) or set(measurements) != {
        str(player) for player in range(4, 11)
    }:
        raise ValueError("calibration report lacks p-specific preflight measurements")
    result: dict[int, tuple[int, int]] = {}
    for player in range(4, 11):
        record = measurements[str(player)]
        if not isinstance(record, Mapping):
            raise ValueError("calibration preflight stratum is invalid")
        matches = record.get("matches")
        nonforced = record.get("nonforcedDecisions")
        if (
            isinstance(matches, bool)
            or not isinstance(matches, int)
            or matches < MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM
            or isinstance(nonforced, bool)
            or not isinstance(nonforced, int)
            or nonforced < matches
        ):
            raise ValueError(
                "production calibration preflight requires at least "
                f"{MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM} matches per p-stratum"
            )
        result[player] = (matches, nonforced)
    return result


def create_collection_plan(
    *,
    actor_bundle: str | Path,
    critic_checkpoint: str | Path,
    behavior_pair: str | Path,
    source_root: str | Path,
    calibration_report: str | Path,
    calibration_cpu_snapshot: str | Path,
    calibration_cuda_snapshot: str | Path,
    run_namespace: str,
    seed_base: int,
    output: str | Path | None,
    dry_run: bool = False,
    total_matches: int | None = None,
    default_total_matches: int = DEFAULT_TOTAL_MATCHES,
    preflight_matches: int | None = None,
    preflight_nonforced_decisions: int | None = None,
    preflight_strata: Mapping[int, tuple[int, int]] | None = None,
    target_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED,
    minimum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MIN,
    maximum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MAX,
    cpu_matches_per_player_count: int = DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT,
    cpu_matches_by_player_count: Mapping[int, int] | None = None,
    cpu_seconds_per_match: Mapping[int, float] | None = None,
    cuda_seconds_per_match: Mapping[int, float] | None = None,
    cpu_worker_count: int = 1,
    cuda_worker_count: int = 1,
    cpu_torch_threads_per_worker: int = 1,
    cuda_torch_threads_per_worker: int = 1,
    max_matches_per_shard: int = DEFAULT_MAX_MATCHES_PER_SHARD,
    actual_stratum_relative_tolerance: float = DEFAULT_ACTUAL_STRATUM_RELATIVE_TOLERANCE,
    diagnostic_unbalanced: bool = False,
    source_files: Sequence[str] | None = None,
) -> V5CollectionPlan:
    if type(diagnostic_unbalanced) is not bool:
        raise ValueError("diagnostic_unbalanced must be an exact bool")
    aggregate_or_explicit = any(
        value is not None
        for value in (
            total_matches,
            preflight_matches,
            preflight_nonforced_decisions,
        )
    )
    if aggregate_or_explicit and not diagnostic_unbalanced:
        raise ValueError(
            "aggregate or explicit sizing requires --diagnostic-unbalanced"
        )
    behavior = load_verified_behavior(actor_bundle, critic_checkpoint, behavior_pair)
    sources = (
        build_source_inventory(source_root, tuple(source_files))
        if source_files is not None
        else build_source_inventory(source_root)
    )
    calibration_value, calibration_sha = _verify_calibration_for_plan(
        calibration_report,
        calibration_cpu_snapshot,
        calibration_cuda_snapshot,
        behavior,
        sources,
    )
    if (
        preflight_strata is None
        and not diagnostic_unbalanced
        and total_matches is None
        and preflight_matches is None
        and preflight_nonforced_decisions is None
    ):
        preflight_strata = _preflight_from_calibration_report(calibration_value)
    plan = build_collection_plan(
        run_namespace=run_namespace,
        seed_base=seed_base,
        behavior_actor_sha256=behavior.actor_sha256,
        behavior_actor_manifest_sha256=behavior.actor_manifest_sha256,
        behavior_critic_sha256=behavior.critic_sha256,
        behavior_pair_id=behavior.pair_id,
        behavior_pair_manifest_sha256=behavior.pair_manifest_sha256,
        calibration_report_sha256=calibration_sha,
        source_inventory=sources,
        total_matches=total_matches,
        default_total_matches=default_total_matches,
        preflight_matches=preflight_matches,
        preflight_nonforced_decisions=preflight_nonforced_decisions,
        preflight_strata=preflight_strata,
        target_nonforced_decisions=target_nonforced_decisions,
        minimum_nonforced_decisions=minimum_nonforced_decisions,
        maximum_nonforced_decisions=maximum_nonforced_decisions,
        cpu_matches_per_player_count=cpu_matches_per_player_count,
        cpu_matches_by_player_count=cpu_matches_by_player_count,
        cpu_seconds_per_match=cpu_seconds_per_match,
        cuda_seconds_per_match=cuda_seconds_per_match,
        cpu_worker_count=cpu_worker_count,
        cuda_worker_count=cuda_worker_count,
        cpu_torch_threads_per_worker=cpu_torch_threads_per_worker,
        cuda_torch_threads_per_worker=cuda_torch_threads_per_worker,
        max_matches_per_shard=max_matches_per_shard,
        actual_stratum_relative_tolerance=actual_stratum_relative_tolerance,
        diagnostic_unbalanced=diagnostic_unbalanced,
    )
    if dry_run:
        if output is not None:
            raise ValueError("dry-run planning does not accept or write an output path")
    else:
        if output is None:
            raise ValueError("non-dry-run planning requires an immutable output path")
        publish_collection_plan(output, plan)
    return plan


def _plan_calibration_digest(
    plan: V5CollectionPlan,
    report: str | Path,
    cpu_snapshot: str | Path,
    cuda_snapshot: str | Path,
) -> str:
    _, actual = load_verified_calibration_report(report, cpu_snapshot, cuda_snapshot)
    calibration = plan.document["calibration"]
    assert isinstance(calibration, Mapping)
    if calibration.get("reportSha256") != actual:
        raise ValueError("production calibration report hash differs from collection plan")
    return actual


def collect_planned_shard(
    *,
    plan_path: str | Path,
    shard_index: int,
    shards_root: str | Path,
    actor_bundle: str | Path,
    critic_checkpoint: str | Path,
    behavior_pair: str | Path,
    source_root: str | Path,
    calibration_report: str | Path,
    calibration_cpu_snapshot: str | Path,
    calibration_cuda_snapshot: str | Path,
    device: str,
    lane_count: int = 32,
    resume_existing: bool = False,
    source_files: Sequence[str] | None = None,
) -> V5PublishedCollection:
    plan = load_collection_plan(plan_path)
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < len(plan.shards):
        raise ValueError("shard_index is outside the immutable collection plan")
    shard = plan.shards[shard_index]
    actual_backend = _device_backend(device)
    if actual_backend != shard.backend:
        raise ValueError(
            f"planned shard {shard.index} requires {shard.backend}, not {actual_backend}"
        )
    with _planned_worker_slot(plan, shards_root, actual_backend):
        behavior = load_verified_behavior(actor_bundle, critic_checkpoint, behavior_pair)
        _require_behavior_matches(plan.behavior, behavior)
        sources = (
            build_source_inventory(source_root, tuple(source_files))
            if source_files is not None
            else build_source_inventory(source_root)
        )
        _require_source_matches(plan.source_inventory, sources)
        _plan_calibration_digest(
            plan,
            calibration_report,
            calibration_cpu_snapshot,
            calibration_cuda_snapshot,
        )
        target = planned_shard_path(shards_root, shard)
        if target.exists():
            if not resume_existing:
                raise FileExistsError(f"planned immutable shard already exists: {target}")
            manifest_sha = resume_verified_shard(plan, shard, shards_root)
            assert manifest_sha is not None
            # A verified existing shard is resumable as an all-or-nothing unit.
            from v5_dataset import load_v5_actor_shard

            loaded = load_v5_actor_shard(target)
            try:
                decisions = loaded.decision_count
                matches = loaded.match_count
                nonforced = int((~loaded.arrays["forced"]).sum())
            finally:
                loaded.close()
            return V5PublishedCollection(target, manifest_sha, matches, decisions, nonforced)
        config = V5MAPPOCollectionConfig(
            run_namespace=plan.run_namespace,
            seed_base=plan.seed_base,
            match_counts=((shard.player_count, shard.match_count),),
            match_start=shard.match_start,
            lane_count=lane_count,
        )
        runtime = V5TorchInferenceRuntime(
            behavior.actor,
            behavior.critic,
            device=device,
        )
        collection = collect_v5_mappo(runtime.actor_batch, runtime.critic_batch, config)
        published = publish_v5_mappo_collection(
            target,
            collection,
            behavior_actor_sha256=behavior.actor_sha256,
            behavior_actor_manifest_sha256=behavior.actor_manifest_sha256,
            behavior_critic_sha256=behavior.critic_sha256,
            metadata=expected_planned_shard_metadata(
                plan, shard, execution_backend=actual_backend
            ),
        )
        verified_sha = verify_planned_shard(plan, shard, target)
        if verified_sha != published.manifest_sha256:
            raise RuntimeError("new shard checksum changed during post-publish verification")
        return published


def _source_files(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("source files must be a comma-separated list")
    return result


def _match_counts(value: str) -> tuple[tuple[int, int], ...]:
    parsed: dict[int, int] = {}
    try:
        for item in value.split(","):
            player, count = item.split(":", 1)
            parsed[int(player)] = int(count)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("match counts must look like 4:1,...,10:1") from error
    result = tuple(sorted(parsed.items()))
    if tuple(player for player, _ in result) != tuple(range(4, 11)) or any(
        count < 1 for _, count in result
    ):
        raise argparse.ArgumentTypeError("calibration match counts must cover p4..p10")
    return result


def _preflight_strata(value: str | None) -> dict[int, tuple[int, int]] | None:
    if value is None:
        return None
    parsed: dict[int, tuple[int, int]] = {}
    try:
        for item in value.split(","):
            player, matches, nonforced = item.split(":", 2)
            parsed[int(player)] = (int(matches), int(nonforced))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "preflight strata must look like 4:matches:nonforced,...,10:matches:nonforced"
        ) from error
    if set(parsed) != set(range(4, 11)) or any(
        matches < 1 or nonforced < matches
        for matches, nonforced in parsed.values()
    ):
        raise argparse.ArgumentTypeError("preflight strata must cover valid p4..p10 measurements")
    return parsed


def _per_player_ints(value: str | None) -> dict[int, int] | None:
    if value is None:
        return None
    parsed: dict[int, int] = {}
    try:
        for item in value.split(","):
            player, count = item.split(":", 1)
            parsed[int(player)] = int(count)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "p-specific integer mapping must look like 4:10,...,10:10"
        ) from error
    if set(parsed) != set(range(4, 11)):
        raise argparse.ArgumentTypeError("p-specific mapping must cover p4..p10")
    return parsed


def _per_player_floats(value: str | None) -> dict[int, float] | None:
    if value is None:
        return None
    parsed: dict[int, float] = {}
    try:
        for item in value.split(","):
            player, seconds = item.split(":", 1)
            parsed[int(player)] = float(seconds)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "p-specific seconds mapping must look like 4:1.2,...,10:4.5"
        ) from error
    if set(parsed) != set(range(4, 11)):
        raise argparse.ArgumentTypeError("p-specific mapping must cover p4..p10")
    return parsed


def _add_behavior_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--actor-bundle", required=True)
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument(
        "--behavior-pair",
        required=True,
        help="verified trainer/initialization directory containing pair-manifest.json",
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--source-files",
        help="optional comma-separated logical source inventory override",
    )


def _add_calibration_triple(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--calibration-cpu-snapshot", required=True)
    parser.add_argument("--calibration-cuda-snapshot", required=True)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser(
        "calibrate-collect", help="collect one immutable CPU or CUDA calibration snapshot"
    )
    _add_behavior_arguments(calibration)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    calibration.add_argument("--device", required=True)
    calibration.add_argument("--run-namespace", required=True)
    calibration.add_argument("--seed-base", type=int, required=True)
    calibration.add_argument("--lanes", type=int, default=7)
    calibration.add_argument(
        "--match-counts", type=_match_counts, default=DEFAULT_CALIBRATION_MATCH_COUNTS
    )

    throughput = commands.add_parser(
        "benchmark-throughput",
        help="measure end-to-end p4..p10 seconds/match on one actual backend",
    )
    _add_behavior_arguments(throughput)
    throughput.add_argument("--output", required=True)
    throughput.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    throughput.add_argument("--device", required=True)
    throughput.add_argument("--run-namespace", required=True)
    throughput.add_argument("--seed-base", type=int, required=True)
    throughput.add_argument("--scratch-root", required=True)
    throughput.add_argument(
        "--matches-per-player-count",
        type=int,
        default=MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM,
    )
    throughput.add_argument("--lanes", type=int, default=32)

    comparison = commands.add_parser(
        "calibrate-compare", help="compare and seal CPU/CUDA calibration snapshots"
    )
    comparison.add_argument("--cpu-snapshot", required=True)
    comparison.add_argument("--cuda-snapshot", required=True)
    comparison.add_argument("--output", required=True)

    plan = commands.add_parser("plan", help="build or dry-run the immutable mixed plan")
    _add_behavior_arguments(plan)
    _add_calibration_triple(plan)
    plan.add_argument("--run-namespace", required=True)
    plan.add_argument("--seed-base", type=int, required=True)
    plan.add_argument("--output")
    plan.add_argument("--dry-run", action="store_true")
    plan.add_argument("--total-matches", type=int)
    plan.add_argument("--default-total-matches", type=int, default=DEFAULT_TOTAL_MATCHES)
    plan.add_argument("--preflight-matches", type=int)
    plan.add_argument("--preflight-nonforced-decisions", type=int)
    plan.add_argument(
        "--preflight-strata",
        help="p-specific measurements: 4:matches:nonforced,...,10:matches:nonforced",
    )
    plan.add_argument("--target-nonforced-decisions", type=int, default=DEFAULT_TARGET_NONFORCED)
    plan.add_argument("--minimum-nonforced-decisions", type=int, default=DEFAULT_TARGET_NONFORCED_MIN)
    plan.add_argument("--maximum-nonforced-decisions", type=int, default=DEFAULT_TARGET_NONFORCED_MAX)
    plan.add_argument("--cpu-matches-per-player-count", type=int, default=DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT)
    plan.add_argument(
        "--cpu-matches-by-player-count",
        help="explicit p4..p10 CPU counts: 4:n,...,10:n",
    )
    plan.add_argument(
        "--cpu-seconds-per-match",
        help="measured CPU seconds/match: 4:s,...,10:s",
    )
    plan.add_argument(
        "--cuda-seconds-per-match",
        help="measured CUDA seconds/match: 4:s,...,10:s",
    )
    plan.add_argument("--cpu-worker-count", type=int, default=1)
    plan.add_argument("--cuda-worker-count", type=int, default=1)
    plan.add_argument("--cpu-torch-threads-per-worker", type=int, default=1)
    plan.add_argument("--cuda-torch-threads-per-worker", type=int, default=1)
    plan.add_argument("--max-matches-per-shard", type=int, default=DEFAULT_MAX_MATCHES_PER_SHARD)
    plan.add_argument(
        "--actual-stratum-relative-tolerance",
        type=float,
        default=DEFAULT_ACTUAL_STRATUM_RELATIVE_TOLERANCE,
    )
    plan.add_argument(
        "--diagnostic-unbalanced",
        action="store_true",
        help="allow an unbalanced diagnostic plan that can never publish a production index",
    )

    collect = commands.add_parser("collect-shard", help="collect one precommitted immutable shard")
    _add_behavior_arguments(collect)
    _add_calibration_triple(collect)
    collect.add_argument("--plan", required=True)
    collect.add_argument("--shard-index", type=int, required=True)
    collect.add_argument("--shards-root", required=True)
    collect.add_argument("--device", required=True)
    collect.add_argument("--lanes", type=int, default=32)
    collect.add_argument("--resume-existing", action="store_true")

    recover = commands.add_parser(
        "recover-worker-slot",
        help="evidence-retire one stale planned worker lock without PID-reuse risk",
    )
    recover.add_argument("--plan", required=True)
    recover.add_argument("--shards-root", required=True)
    recover.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    recover.add_argument("--slot-index", type=int, required=True)
    recover.add_argument("--reason", required=True)

    index = commands.add_parser("publish-index", help="verify all shards and publish zero-copy index")
    index.add_argument("--plan", required=True)
    index.add_argument("--shards-root", required=True)
    index.add_argument("--output", required=True)
    return parser


def _summary(value: Mapping[str, object]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    source_files = _source_files(getattr(arguments, "source_files", None))
    if arguments.command == "calibrate-collect":
        result = collect_calibration_snapshot(
            actor_bundle=arguments.actor_bundle,
            critic_checkpoint=arguments.critic_checkpoint,
            behavior_pair=arguments.behavior_pair,
            source_root=arguments.source_root,
            output=arguments.output,
            backend=arguments.backend,
            device=arguments.device,
            run_namespace=arguments.run_namespace,
            seed_base=arguments.seed_base,
            lane_count=arguments.lanes,
            match_counts=arguments.match_counts,
            source_files=source_files,
        )
        _summary({
            "decisions": result.decisions,
            "manifestSha256": result.manifest_sha256,
            "matches": result.matches,
            "nonforcedDecisions": result.nonforced_decisions,
            "target": str(result.target),
        })
    elif arguments.command == "benchmark-throughput":
        result = collect_throughput_preflight(
            actor_bundle=arguments.actor_bundle,
            critic_checkpoint=arguments.critic_checkpoint,
            behavior_pair=arguments.behavior_pair,
            source_root=arguments.source_root,
            output=arguments.output,
            backend=arguments.backend,
            device=arguments.device,
            run_namespace=arguments.run_namespace,
            seed_base=arguments.seed_base,
            scratch_root=arguments.scratch_root,
            matches_per_player_count=arguments.matches_per_player_count,
            lane_count=arguments.lanes,
            source_files=source_files,
        )
        _summary(result)
    elif arguments.command == "calibrate-compare":
        digest = publish_calibration_report(
            arguments.output, arguments.cpu_snapshot, arguments.cuda_snapshot
        )
        _summary({"reportSha256": digest, "target": str(Path(arguments.output).resolve())})
    elif arguments.command == "plan":
        result = create_collection_plan(
            actor_bundle=arguments.actor_bundle,
            critic_checkpoint=arguments.critic_checkpoint,
            behavior_pair=arguments.behavior_pair,
            source_root=arguments.source_root,
            calibration_report=arguments.calibration_report,
            calibration_cpu_snapshot=arguments.calibration_cpu_snapshot,
            calibration_cuda_snapshot=arguments.calibration_cuda_snapshot,
            run_namespace=arguments.run_namespace,
            seed_base=arguments.seed_base,
            output=arguments.output,
            dry_run=arguments.dry_run,
            total_matches=arguments.total_matches,
            default_total_matches=arguments.default_total_matches,
            preflight_matches=arguments.preflight_matches,
            preflight_nonforced_decisions=arguments.preflight_nonforced_decisions,
            preflight_strata=_preflight_strata(arguments.preflight_strata),
            target_nonforced_decisions=arguments.target_nonforced_decisions,
            minimum_nonforced_decisions=arguments.minimum_nonforced_decisions,
            maximum_nonforced_decisions=arguments.maximum_nonforced_decisions,
            cpu_matches_per_player_count=arguments.cpu_matches_per_player_count,
            cpu_matches_by_player_count=_per_player_ints(
                arguments.cpu_matches_by_player_count
            ),
            cpu_seconds_per_match=_per_player_floats(
                arguments.cpu_seconds_per_match
            ),
            cuda_seconds_per_match=_per_player_floats(
                arguments.cuda_seconds_per_match
            ),
            cpu_worker_count=arguments.cpu_worker_count,
            cuda_worker_count=arguments.cuda_worker_count,
            cpu_torch_threads_per_worker=arguments.cpu_torch_threads_per_worker,
            cuda_torch_threads_per_worker=arguments.cuda_torch_threads_per_worker,
            max_matches_per_shard=arguments.max_matches_per_shard,
            actual_stratum_relative_tolerance=arguments.actual_stratum_relative_tolerance,
            diagnostic_unbalanced=arguments.diagnostic_unbalanced,
            source_files=source_files,
        )
        if arguments.dry_run:
            print(canonical_json_bytes(dict(result.document)).decode("utf-8"), end="")
        else:
            _summary({
                "planSha256": result.manifest_sha256,
                "shards": len(result.shards),
                "totalMatches": result.document["totalMatches"],
            })
    elif arguments.command == "collect-shard":
        result = collect_planned_shard(
            plan_path=arguments.plan,
            shard_index=arguments.shard_index,
            shards_root=arguments.shards_root,
            actor_bundle=arguments.actor_bundle,
            critic_checkpoint=arguments.critic_checkpoint,
            behavior_pair=arguments.behavior_pair,
            source_root=arguments.source_root,
            calibration_report=arguments.calibration_report,
            calibration_cpu_snapshot=arguments.calibration_cpu_snapshot,
            calibration_cuda_snapshot=arguments.calibration_cuda_snapshot,
            device=arguments.device,
            lane_count=arguments.lanes,
            resume_existing=arguments.resume_existing,
            source_files=source_files,
        )
        _summary({
            "decisions": result.decisions,
            "manifestSha256": result.manifest_sha256,
            "matches": result.matches,
            "nonforcedDecisions": result.nonforced_decisions,
            "target": str(result.target),
        })
    elif arguments.command == "recover-worker-slot":
        _summary(
            recover_v5_worker_slot(
                arguments.plan,
                arguments.shards_root,
                backend=arguments.backend,
                slot_index=arguments.slot_index,
                recovery_reason=arguments.reason,
            )
        )
    elif arguments.command == "publish-index":
        plan = load_collection_plan(arguments.plan)
        digest = publish_verified_index(plan, arguments.shards_root, arguments.output)
        _summary({"indexSha256": digest, "target": str(Path(arguments.output).resolve())})
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError("unhandled V5 collection command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CALIBRATION_MATCH_COUNTS",
    "V5LoadedBehavior",
    "argument_parser",
    "collect_calibration_snapshot",
    "collect_throughput_preflight",
    "collect_planned_shard",
    "create_collection_plan",
    "load_verified_behavior",
    "main",
]
