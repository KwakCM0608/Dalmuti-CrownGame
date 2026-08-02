from __future__ import annotations

"""Fail-closed hybrid staging for a large immutable DALMUTI V5 corpus.

The GPU host used for V5 has very little persistent storage but a reasonably
large ``/dev/shm``.  This module plans a deterministic split of complete V5
shards between those two tiers, verifies every transferred byte, and publishes
one ordinary V5 zero-copy index that may reference both filesystems.

No source shard is rewritten, concatenated, or deleted.  The volatile copy is
only a replica of a checksum-bound canonical corpus retained on the collector
host.  A lost tmpfs therefore requires restaging, never data reconstruction.
"""

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from v5_dataset import (
    V5_INDEX_FORMAT,
    V5_INDEX_VERSION,
    load_v5_index_manifest,
    load_v5_training_shard,
    publish_v5_index_manifest,
)


V5_LOW_DISK_STAGE_FORMAT = "dalmuti-v5-low-disk-stage-plan"
V5_LOW_DISK_STAGE_VERSION = 1
V5_LOW_DISK_STAGE_INDEX_CONTRACT = (
    "verified-direct-cross-filesystem-zero-copy-index-v1"
)
V5_LOW_DISK_PROMOTION_RECEIPT_FORMAT = (
    "dalmuti-v5-low-disk-stage-promotion-receipt"
)
V5_LOW_DISK_PROMOTION_RECEIPT_VERSION = 1
V5_LOW_DISK_PROMOTION_LOCK_FORMAT = "dalmuti-v5-low-disk-stage-promotion-lock"
V5_LOW_DISK_PROMOTION_LOCK_VERSION = 1

SOURCE_INDEX_RECORD_NAME = "source-index-record"
LOW_DISK_STAGE_PLAN_NAME = "low-disk-stage-plan"
PROMOTION_RECEIPT_ROOT_NAME = "low-disk-promotion-receipts"

PERSISTENT_TIER = "persistent"
VOLATILE_TIER = "volatile"
_TIERS = (PERSISTENT_TIER, VOLATILE_TIER)

DEFAULT_FILESYSTEM_BLOCK_BYTES = 4 * 1024
DEFAULT_PLACEMENT_QUANTUM_BYTES = 1024 * 1024
DEFAULT_PER_SHARD_OVERHEAD_BYTES = 256 * 1024
DEFAULT_PERSISTENT_RESERVE_BYTES = 6 * 1024**3
DEFAULT_VOLATILE_RESERVE_BYTES = 2 * 1024**3

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9._-]{0,126}[a-z0-9]")
_STAGED_NAME = re.compile(r"shard-[0-9]{3,}-[0-9a-f]{16}")


@dataclass(frozen=True)
class V5ShardStorageInventory:
    root: Path
    manifest_sha256: str
    logical_bytes: int
    rounded_file_bytes: int
    file_count: int
    decision_count: int
    nonforced_decision_count: int
    match_count: int
    player_counts: tuple[int, ...]


@dataclass(frozen=True)
class V5LowDiskStagePlan:
    document: Mapping[str, object]
    manifest_sha256: str

    @property
    def run_namespace(self) -> str:
        return str(self.document["runNamespace"])

    @property
    def shards(self) -> tuple[Mapping[str, object], ...]:
        raw = self.document["shards"]
        assert isinstance(raw, list)
        return tuple(raw)


@dataclass(frozen=True)
class V5SourceIndexRecord:
    document: Mapping[str, object]
    manifest_sha256: str


@dataclass(frozen=True)
class V5VerifiedHybridStage:
    plan: V5LowDiskStagePlan
    source_index: V5SourceIndexRecord
    shard_paths: tuple[Path, ...]
    shard_manifest_sha256s: tuple[str, ...]
    hybrid_index_manifest_sha256: str | None
    actual_decision_counts_by_player_count: Mapping[str, int] | None = None
    actual_match_counts_by_player_count: Mapping[str, int] | None = None
    actual_nonforced_decision_counts_by_player_count: Mapping[str, int] | None = None
    actual_corpus_gate: Mapping[str, object] | None = None


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


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _round_up(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields are non-canonical")
    return value


def _canonical_manifest(root: Path) -> tuple[Mapping[str, object], str]:
    raw = (root / "manifest.json").read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V5 shard manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ValueError("V5 shard manifest is not canonical")
    digest = _sha256_bytes(raw)
    expected = f"{digest}  manifest.json\n".encode("ascii")
    if (root / "manifest.json.sha256").read_bytes() != expected:
        raise ValueError("V5 shard manifest sidecar does not match")
    return value, digest


def _read_exact_manifest_directory(
    target: str | Path, *, label: str
) -> tuple[Path, Mapping[str, object], str]:
    unresolved = Path(target)
    if unresolved.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    root = unresolved.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{label} is missing")
    children = tuple(root.iterdir())
    if (
        {child.name for child in children}
        != {"manifest.json", "manifest.json.sha256"}
        or any(child.is_symlink() or not child.is_file() for child in children)
    ):
        raise ValueError(f"{label} inventory is non-canonical")
    document, digest = _canonical_manifest(root)
    return root, document, digest


def _validate_source_index_document(document: object) -> Mapping[str, object]:
    root = _exact_keys(
        document,
        {
            "actionCount",
            "counts",
            "format",
            "mergeMode",
            "metadata",
            "playerCounts",
            "shards",
            "version",
        },
        "V5 source-index record",
    )
    if (
        root["format"] != V5_INDEX_FORMAT
        or root["version"] != V5_INDEX_VERSION
        or root["mergeMode"] != "zero-copy immutable shard references"
    ):
        raise ValueError("V5 source-index record format/version is incompatible")
    _integer(root["actionCount"], "source-index action count", minimum=1)
    if not isinstance(root["metadata"], dict):
        raise ValueError("V5 source-index metadata must be an object")
    players = root["playerCounts"]
    if (
        not isinstance(players, list)
        or not players
        or players != sorted(set(players))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 4 <= value <= 10
            for value in players
        )
    ):
        raise ValueError("V5 source-index player counts are invalid")
    records = root["shards"]
    if not isinstance(records, list) or not records:
        raise ValueError("V5 source-index record has no shards")
    decisions = 0
    matches = 0
    relative_paths: set[str] = set()
    for raw_record in records:
        record = _exact_keys(
            raw_record,
            {"decisionCount", "manifestSha256", "matchCount", "relativePath"},
            "V5 source-index shard",
        )
        decisions += _integer(
            record["decisionCount"], "source-index shard decisions", minimum=1
        )
        matches += _integer(
            record["matchCount"], "source-index shard matches", minimum=1
        )
        _sha(record["manifestSha256"], "source-index shard manifest SHA")
        relative = record["relativePath"]
        if (
            not isinstance(relative, str)
            or not relative
            or "\x00" in relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or relative in relative_paths
        ):
            raise ValueError("V5 source-index shard relative path is invalid")
        relative_paths.add(relative)
    counts = _exact_keys(
        root["counts"], {"decisions", "matches", "shards"}, "V5 source-index counts"
    )
    if counts != {
        "decisions": decisions,
        "matches": matches,
        "shards": len(records),
    }:
        raise ValueError("V5 source-index aggregate counts drifted")
    return root


def load_v5_source_index_record(target: str | Path) -> V5SourceIndexRecord:
    root, document, digest = _read_exact_manifest_directory(
        target, label="V5 source-index record"
    )
    if root.name != SOURCE_INDEX_RECORD_NAME:
        raise ValueError(
            f"V5 source-index record directory must be named {SOURCE_INDEX_RECORD_NAME}"
        )
    return V5SourceIndexRecord(_validate_source_index_document(document), digest)


def inventory_v5_training_shard(
    shard_path: str | Path,
    *,
    filesystem_block_bytes: int = DEFAULT_FILESYSTEM_BLOCK_BYTES,
) -> V5ShardStorageInventory:
    """Fully verify and inventory one immutable training shard.

    Extra files and symlinks are rejected so a transfer map cannot silently
    omit, add, or redirect bytes outside the checksum-bound shard contract.
    """

    block = _integer(filesystem_block_bytes, "filesystem_block_bytes", minimum=512)
    if block & (block - 1):
        raise ValueError("filesystem_block_bytes must be a power of two")
    unresolved_root = Path(shard_path)
    if unresolved_root.is_symlink():
        raise ValueError("V5 staged shard root must not be a symlink")
    root = unresolved_root.resolve()
    loaded = load_v5_training_shard(root)
    try:
        manifest, digest = _canonical_manifest(root)
        partitions = manifest.get("partitions")
        if not isinstance(partitions, Mapping):
            raise ValueError("V5 shard partitions are missing")
        expected_files = {"manifest.json", "manifest.json.sha256"}
        for partition in ("actor", "privileged"):
            records = partitions.get(partition)
            if not isinstance(records, Mapping):
                raise ValueError(f"V5 {partition} partition is missing")
            for name, record in records.items():
                if not isinstance(name, str) or not isinstance(record, Mapping):
                    raise ValueError("V5 shard array record is invalid")
                relative = record.get("path")
                if relative != f"{partition}/{name}.npy":
                    raise ValueError("V5 shard array path is non-canonical")
                expected_files.add(str(relative))

        observed_files: set[str] = set()
        logical_bytes = 0
        rounded_bytes = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("V5 staged shard must not contain symlinks")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                observed_files.add(relative)
                size = path.stat().st_size
                logical_bytes += size
                rounded_bytes += _round_up(size, block)
        if observed_files != expected_files:
            missing = sorted(expected_files - observed_files)
            extra = sorted(observed_files - expected_files)
            detail = missing[0] if missing else extra[0]
            raise ValueError(f"V5 shard file inventory drifted: {detail}")

        forced = np.asarray(loaded.actor.arrays["forced"], dtype=np.bool_)
        players = tuple(
            sorted({int(value) for value in loaded.actor.arrays["player_counts"]})
        )
        return V5ShardStorageInventory(
            root=root,
            manifest_sha256=digest,
            logical_bytes=logical_bytes,
            rounded_file_bytes=rounded_bytes,
            file_count=len(observed_files),
            decision_count=loaded.actor.decision_count,
            nonforced_decision_count=int((~forced).sum(dtype=np.int64)),
            match_count=loaded.actor.match_count,
            player_counts=players,
        )
    finally:
        loaded.close()


def _charge_bytes(
    inventory: V5ShardStorageInventory,
    *,
    placement_quantum_bytes: int,
    per_shard_overhead_bytes: int,
) -> int:
    return _round_up(
        inventory.rounded_file_bytes + per_shard_overhead_bytes,
        placement_quantum_bytes,
    )


def _tier_record(
    *,
    free_bytes: int,
    reserve_bytes: int,
    minimum_reserve_bytes: int,
    assigned_bytes: int,
    quantum: int,
) -> dict[str, int]:
    free = _integer(free_bytes, "tier free bytes", minimum=1)
    reserve = _integer(reserve_bytes, "tier reserve bytes")
    minimum = _integer(minimum_reserve_bytes, "tier minimum reserve bytes")
    if reserve < minimum:
        raise ValueError("tier reserve is below its fail-closed minimum")
    if free <= reserve:
        raise ValueError("tier free space does not exceed its reserved headroom")
    usable = ((free - reserve) // quantum) * quantum
    if assigned_bytes > usable:
        raise ValueError("tier assignment exceeds reserved usable capacity")
    return {
        "assignedBytes": assigned_bytes,
        "freeBytesAtPlanning": free,
        "minimumReserveBytes": minimum,
        "postStageHeadroomBytes": free - assigned_bytes,
        "reserveBytes": reserve,
        "usableBytes": usable,
    }


def _persistent_subset(
    charges: Sequence[int], persistent_capacity: int, volatile_capacity: int
) -> set[int]:
    """Return the minimum-root-use exact two-bin assignment.

    Charges are placement-quantum multiples.  A bounded subset-sum avoids the
    fragmentation failures of greedy bin packing while remaining tiny for the
    expected 20-40 V5 shards.
    """

    total = sum(charges)
    if total > persistent_capacity + volatile_capacity:
        raise ValueError("V5 corpus exceeds combined reserved staging capacity")
    if any(value > max(persistent_capacity, volatile_capacity) for value in charges):
        raise ValueError("one V5 shard is larger than every staging tier")
    unit = math.gcd(*charges, persistent_capacity, volatile_capacity)
    scaled = [value // unit for value in charges]
    lower = max(0, (total - volatile_capacity + unit - 1) // unit)
    upper = persistent_capacity // unit
    reachable: dict[int, int] = {0: 0}
    for index, charge in enumerate(scaled):
        for subtotal, mask in tuple(sorted(reachable.items(), reverse=True)):
            candidate = subtotal + charge
            if candidate <= upper and candidate not in reachable:
                reachable[candidate] = mask | (1 << index)
    feasible = [value for value in reachable if lower <= value <= upper]
    if not feasible:
        raise ValueError("V5 shard sizes cannot fit the two reserved staging tiers")
    chosen = reachable[min(feasible)]
    persistent = {index for index in range(len(charges)) if chosen & (1 << index)}
    volatile_charge = sum(
        charge for index, charge in enumerate(charges) if index not in persistent
    )
    if volatile_charge > volatile_capacity:
        raise RuntimeError("internal V5 volatile staging assignment overflow")
    return persistent


def build_v5_low_disk_stage_plan(
    source_index: str | Path,
    *,
    run_namespace: str,
    persistent_free_bytes: int,
    volatile_free_bytes: int,
    persistent_reserve_bytes: int = DEFAULT_PERSISTENT_RESERVE_BYTES,
    volatile_reserve_bytes: int = DEFAULT_VOLATILE_RESERVE_BYTES,
    minimum_persistent_reserve_bytes: int = DEFAULT_PERSISTENT_RESERVE_BYTES,
    minimum_volatile_reserve_bytes: int = DEFAULT_VOLATILE_RESERVE_BYTES,
    filesystem_block_bytes: int = DEFAULT_FILESYSTEM_BLOCK_BYTES,
    placement_quantum_bytes: int = DEFAULT_PLACEMENT_QUANTUM_BYTES,
    per_shard_overhead_bytes: int = DEFAULT_PER_SHARD_OVERHEAD_BYTES,
) -> V5LowDiskStagePlan:
    if not isinstance(run_namespace, str) or _RUN_NAMESPACE.fullmatch(run_namespace) is None:
        raise ValueError("run_namespace is not a safe independent directory name")
    block = _integer(filesystem_block_bytes, "filesystem_block_bytes", minimum=512)
    quantum = _integer(
        placement_quantum_bytes, "placement_quantum_bytes", minimum=block
    )
    overhead = _integer(per_shard_overhead_bytes, "per_shard_overhead_bytes")
    if block & (block - 1) or quantum % block:
        raise ValueError("staging block/placement units are incompatible")

    # Compute capacities before the expensive full-corpus checksum pass.
    persistent_empty = _tier_record(
        free_bytes=persistent_free_bytes,
        reserve_bytes=persistent_reserve_bytes,
        minimum_reserve_bytes=minimum_persistent_reserve_bytes,
        assigned_bytes=0,
        quantum=quantum,
    )
    volatile_empty = _tier_record(
        free_bytes=volatile_free_bytes,
        reserve_bytes=volatile_reserve_bytes,
        minimum_reserve_bytes=minimum_volatile_reserve_bytes,
        assigned_bytes=0,
        quantum=quantum,
    )

    index_root = Path(source_index).resolve()
    index = load_v5_index_manifest(index_root)
    try:
        inventories = tuple(
            inventory_v5_training_shard(path, filesystem_block_bytes=block)
            for path in index.shard_paths
        )
        source_counts = {
            "decisions": index.decision_count,
            "matches": index.match_count,
            "shards": len(index.shard_paths),
        }
    finally:
        index.close()
    charges = tuple(
        _charge_bytes(
            item,
            placement_quantum_bytes=quantum,
            per_shard_overhead_bytes=overhead,
        )
        for item in inventories
    )
    persistent = _persistent_subset(
        charges,
        int(persistent_empty["usableBytes"]),
        int(volatile_empty["usableBytes"]),
    )

    shard_records: list[dict[str, object]] = []
    for index_number, (inventory, charge) in enumerate(
        zip(inventories, charges, strict=True)
    ):
        tier = PERSISTENT_TIER if index_number in persistent else VOLATILE_TIER
        shard_records.append(
            {
                "chargeBytes": charge,
                "decisionCount": inventory.decision_count,
                "fileCount": inventory.file_count,
                "index": index_number,
                "logicalBytes": inventory.logical_bytes,
                "manifestSha256": inventory.manifest_sha256,
                "matchCount": inventory.match_count,
                "nonforcedDecisionCount": inventory.nonforced_decision_count,
                "playerCounts": list(inventory.player_counts),
                "roundedFileBytes": inventory.rounded_file_bytes,
                "stagedName": f"shard-{index_number:03d}-{inventory.manifest_sha256[:16]}",
                "tier": tier,
            }
        )
    assigned = {
        tier: sum(
            int(record["chargeBytes"])
            for record in shard_records
            if record["tier"] == tier
        )
        for tier in _TIERS
    }
    tiers = {
        PERSISTENT_TIER: _tier_record(
            free_bytes=persistent_free_bytes,
            reserve_bytes=persistent_reserve_bytes,
            minimum_reserve_bytes=minimum_persistent_reserve_bytes,
            assigned_bytes=assigned[PERSISTENT_TIER],
            quantum=quantum,
        ),
        VOLATILE_TIER: _tier_record(
            free_bytes=volatile_free_bytes,
            reserve_bytes=volatile_reserve_bytes,
            minimum_reserve_bytes=minimum_volatile_reserve_bytes,
            assigned_bytes=assigned[VOLATILE_TIER],
            quantum=quantum,
        ),
    }
    source_manifest_sha = _sha256_file(index_root / "manifest.json")
    document: dict[str, object] = {
        "allocation": {
            "filesystemBlockBytes": block,
            "perShardOverheadBytes": overhead,
            "placementQuantumBytes": quantum,
            "policy": "minimum-persistent-use exact two-bin subset-sum",
        },
        "format": V5_LOW_DISK_STAGE_FORMAT,
        "runNamespace": run_namespace,
        "shards": shard_records,
        "source": {
            "chargedBytes": sum(charges),
            "counts": source_counts,
            "indexManifestSha256": source_manifest_sha,
            "logicalBytes": sum(item.logical_bytes for item in inventories),
            "nonforcedDecisions": sum(
                item.nonforced_decision_count for item in inventories
            ),
        },
        "tiers": tiers,
        "version": V5_LOW_DISK_STAGE_VERSION,
    }
    raw = _canonical_json_bytes(document)
    return V5LowDiskStagePlan(document, _sha256_bytes(raw))


def _write_fsynced_file(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _require_exact_published_directory(
    target: Path, files: Mapping[str, bytes]
) -> None:
    if target.is_symlink() or not target.is_dir():
        raise ValueError("immutable V5 stage artifact target is foreign")
    children = tuple(target.iterdir())
    if (
        {child.name for child in children} != set(files)
        or any(child.is_symlink() or not child.is_file() for child in children)
        or any((target / name).read_bytes() != data for name, data in files.items())
    ):
        raise ValueError("immutable V5 stage artifact target differs from expected bytes")


def _retire_publish_staging_orphans(target: Path) -> None:
    prefix = f".{target.name}.staging-"
    changed = False
    for candidate in tuple(target.parent.iterdir()):
        suffix = candidate.name[len(prefix):] if candidate.name.startswith(prefix) else ""
        if re.fullmatch(r"[0-9a-f]{32}", suffix) is None:
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("V5 stage publication staging artifact is foreign")
        shutil.rmtree(candidate)
        changed = True
    if changed:
        _fsync_directory(target.parent)


def _exclusive_publish_directory(target: Path, files: Mapping[str, bytes]) -> None:
    """Lock-free, crash-safe publication with exact idempotent reuse."""

    if target.is_symlink():
        raise ValueError("V5 stage artifact target must not be a symlink")
    if (
        not files
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(data, bytes)
            for name, data in files.items()
        )
    ):
        raise ValueError("V5 stage artifact file map is invalid")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if _path_lexists(target):
        _require_exact_published_directory(target, files)
        _retire_publish_staging_orphans(target)
        return

    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    completed = False
    try:
        staging.mkdir()
        for name, data in files.items():
            _write_fsynced_file(staging / name, data)
        _fsync_directory(staging)
        try:
            _rename_directory_noreplace(staging, target)
            _fsync_directory(target.parent)
        except Exception:
            # A concurrent exact winner, or a crash injected immediately after
            # rename, is success.  Foreign/partial bytes remain fail-closed.
            if not _path_lexists(target):
                raise
            _require_exact_published_directory(target, files)
        completed = True
    finally:
        if staging.exists():
            shutil.rmtree(staging)
            _fsync_directory(target.parent)
    if not completed:
        raise RuntimeError("immutable V5 stage artifact publication did not complete")
    _require_exact_published_directory(target, files)
    _retire_publish_staging_orphans(target)


def publish_v5_source_index_record(
    source_index: str | Path, target: str | Path
) -> str:
    """Preserve the canonical source index bytes without resolving its shards.

    The record is intentionally just the original manifest and sidecar.  It can
    therefore travel to the low-disk GPU host even though that host materializes
    the shards under different, cross-filesystem paths.
    """

    source_root, document, digest = _read_exact_manifest_directory(
        source_index, label="V5 source index"
    )
    _validate_source_index_document(document)
    destination = Path(target)
    if destination.name != SOURCE_INDEX_RECORD_NAME:
        raise ValueError(
            f"V5 source-index record target must be named {SOURCE_INDEX_RECORD_NAME}"
        )
    _exclusive_publish_directory(
        destination,
        {
            "manifest.json": (source_root / "manifest.json").read_bytes(),
            "manifest.json.sha256": (
                source_root / "manifest.json.sha256"
            ).read_bytes(),
        },
    )
    return digest


def publish_v5_low_disk_stage_plan(
    target: str | Path, plan: V5LowDiskStagePlan
) -> str:
    raw = _canonical_json_bytes(dict(plan.document))
    digest = _sha256_bytes(raw)
    if digest != plan.manifest_sha256:
        raise ValueError("V5 low-disk stage plan checksum drifted in memory")
    _exclusive_publish_directory(
        Path(target),
        {
            "plan.json": raw,
            "plan.json.sha256": f"{digest}  plan.json\n".encode("ascii"),
        },
    )
    return digest


def _validate_plan_document(document: object) -> Mapping[str, object]:
    root = _exact_keys(
        document,
        {"allocation", "format", "runNamespace", "shards", "source", "tiers", "version"},
        "V5 low-disk stage plan",
    )
    if (
        root["format"] != V5_LOW_DISK_STAGE_FORMAT
        or root["version"] != V5_LOW_DISK_STAGE_VERSION
        or not isinstance(root["runNamespace"], str)
        or _RUN_NAMESPACE.fullmatch(str(root["runNamespace"])) is None
    ):
        raise ValueError("V5 low-disk stage plan contract is incompatible")
    allocation = _exact_keys(
        root["allocation"],
        {"filesystemBlockBytes", "perShardOverheadBytes", "placementQuantumBytes", "policy"},
        "V5 stage allocation",
    )
    block = _integer(allocation["filesystemBlockBytes"], "filesystemBlockBytes", minimum=512)
    quantum = _integer(allocation["placementQuantumBytes"], "placementQuantumBytes", minimum=block)
    overhead = _integer(allocation["perShardOverheadBytes"], "perShardOverheadBytes")
    if (
        block & (block - 1)
        or quantum % block
        or allocation["policy"] != "minimum-persistent-use exact two-bin subset-sum"
    ):
        raise ValueError("V5 stage allocation units/policy drifted")

    raw_shards = root["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("V5 low-disk stage plan has no shards")
    tier_sums = {tier: 0 for tier in _TIERS}
    total_logical = 0
    total_nonforced = 0
    total_decisions = 0
    total_matches = 0
    manifests: set[str] = set()
    for expected_index, raw_record in enumerate(raw_shards):
        record = _exact_keys(
            raw_record,
            {
                "chargeBytes", "decisionCount", "fileCount", "index", "logicalBytes",
                "manifestSha256", "matchCount", "nonforcedDecisionCount", "playerCounts",
                "roundedFileBytes", "stagedName", "tier",
            },
            "V5 staged shard",
        )
        if record["index"] != expected_index:
            raise ValueError("V5 staged shard indexes are not contiguous")
        manifest_sha = _sha(record["manifestSha256"], "staged shard manifest SHA")
        expected_name = f"shard-{expected_index:03d}-{manifest_sha[:16]}"
        if record["stagedName"] != expected_name or _STAGED_NAME.fullmatch(expected_name) is None:
            raise ValueError("V5 staged shard name does not bind its identity")
        tier = record["tier"]
        if tier not in _TIERS:
            raise ValueError("V5 staged shard tier is invalid")
        logical = _integer(record["logicalBytes"], "shard logical bytes", minimum=1)
        rounded = _integer(record["roundedFileBytes"], "shard rounded bytes", minimum=logical)
        charge = _integer(record["chargeBytes"], "shard charge bytes", minimum=1)
        if charge != _round_up(rounded + overhead, quantum):
            raise ValueError("V5 staged shard capacity charge drifted")
        decisions = _integer(record["decisionCount"], "shard decisions", minimum=1)
        nonforced = _integer(record["nonforcedDecisionCount"], "shard nonforced decisions", minimum=1)
        matches = _integer(record["matchCount"], "shard matches", minimum=1)
        _integer(record["fileCount"], "shard file count", minimum=4)
        players = record["playerCounts"]
        if (
            not isinstance(players, list)
            or not players
            or players != sorted(set(players))
            or any(isinstance(value, bool) or not isinstance(value, int) or not 4 <= value <= 10 for value in players)
        ):
            raise ValueError("V5 staged shard player counts are invalid")
        tier_sums[str(tier)] += charge
        total_logical += logical
        total_nonforced += nonforced
        total_decisions += decisions
        total_matches += matches
        manifests.add(manifest_sha)

    tiers = _exact_keys(root["tiers"], set(_TIERS), "V5 staging tiers")
    for tier in _TIERS:
        record = _exact_keys(
            tiers[tier],
            {"assignedBytes", "freeBytesAtPlanning", "minimumReserveBytes", "postStageHeadroomBytes", "reserveBytes", "usableBytes"},
            f"V5 {tier} tier",
        )
        assigned = _integer(record["assignedBytes"], "tier assigned bytes")
        free = _integer(record["freeBytesAtPlanning"], "tier free bytes", minimum=1)
        reserve = _integer(record["reserveBytes"], "tier reserve bytes")
        minimum = _integer(record["minimumReserveBytes"], "tier minimum reserve bytes")
        usable = _integer(record["usableBytes"], "tier usable bytes")
        if (
            reserve < minimum
            or assigned != tier_sums[tier]
            or usable != ((free - reserve) // quantum) * quantum
            or assigned > usable
            or record["postStageHeadroomBytes"] != free - assigned
            or free - assigned < reserve
        ):
            raise ValueError(f"V5 {tier} tier capacity contract drifted")

    source = _exact_keys(
        root["source"],
        {"chargedBytes", "counts", "indexManifestSha256", "logicalBytes", "nonforcedDecisions"},
        "V5 staging source",
    )
    counts = _exact_keys(source["counts"], {"decisions", "matches", "shards"}, "V5 source counts")
    _sha(source["indexManifestSha256"], "source index manifest SHA")
    if (
        source["chargedBytes"] != sum(tier_sums.values())
        or source["logicalBytes"] != total_logical
        or source["nonforcedDecisions"] != total_nonforced
        or counts != {"decisions": total_decisions, "matches": total_matches, "shards": len(raw_shards)}
    ):
        raise ValueError("V5 staging source aggregates drifted")
    return root


def load_v5_low_disk_stage_plan(target: str | Path) -> V5LowDiskStagePlan:
    root = Path(target).resolve()
    raw = (root / "plan.json").read_bytes()
    digest = _sha256_bytes(raw)
    if (root / "plan.json.sha256").read_bytes() != f"{digest}  plan.json\n".encode("ascii"):
        raise ValueError("V5 low-disk stage plan sidecar does not match")
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V5 low-disk stage plan is not ASCII JSON") from error
    if _canonical_json_bytes(document) != raw:
        raise ValueError("V5 low-disk stage plan is not canonical")
    return V5LowDiskStagePlan(_validate_plan_document(document), digest)


def _require_namespaced_tier_root(path: str | Path, run_namespace: str, tier: str) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError(f"V5 {tier} stage root must not be a symlink")
    root = unresolved.resolve()
    expected_name = f"{tier}-shards"
    if root.name != expected_name or root.parent.name != run_namespace:
        raise ValueError(
            f"{tier} stage root must be <independent-root>/{run_namespace}/{expected_name}"
        )
    return root


def _device_id(path: Path) -> int:
    return int(os.stat(path).st_dev)


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically rename a directory while refusing every existing target."""

    if sys.platform.startswith("linux"):
        # Linux rename(2) may replace an empty destination directory.  The
        # renameat2 flag is the kernel primitive that closes that race.
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
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(target),
            rename_noreplace,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(error_number, os.strerror(error_number), target)
            raise OSError(error_number, os.strerror(error_number), target)
        return
    if os.name == "nt":
        # Windows MoveFileEx without MOVEFILE_REPLACE_EXISTING is no-replace.
        os.rename(source, target)
        return
    # There is no portable no-replace directory rename.  Silently weakening
    # the immutable contract on an untested platform would be unsafe.
    raise RuntimeError("atomic no-replace directory rename is unsupported here")


def _verify_inventory_record(
    record: Mapping[str, object], actual: V5ShardStorageInventory
) -> None:
    expected = {
        "decisionCount": actual.decision_count,
        "fileCount": actual.file_count,
        "logicalBytes": actual.logical_bytes,
        "manifestSha256": actual.manifest_sha256,
        "matchCount": actual.match_count,
        "nonforcedDecisionCount": actual.nonforced_decision_count,
        "playerCounts": list(actual.player_counts),
        "roundedFileBytes": actual.rounded_file_bytes,
    }
    for name, value in expected.items():
        if record[name] != value:
            raise ValueError(f"V5 staged shard verification drifted: {name}")


def _require_namespaced_receipt_root(
    path: str | Path, run_namespace: str
) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("V5 low-disk promotion receipt root must not be a symlink")
    root = unresolved.resolve()
    if (
        root.name != PROMOTION_RECEIPT_ROOT_NAME
        or root.parent.name != run_namespace
    ):
        raise ValueError(
            "V5 receipt root must be "
            f"<independent-root>/{run_namespace}/{PROMOTION_RECEIPT_ROOT_NAME}"
        )
    return root


def _promotion_receipt_document(
    plan: V5LowDiskStagePlan,
    record: Mapping[str, object],
    inventory: V5ShardStorageInventory,
) -> dict[str, object]:
    return {
        "format": V5_LOW_DISK_PROMOTION_RECEIPT_FORMAT,
        "inventory": {
            "decisionCount": inventory.decision_count,
            "fileCount": inventory.file_count,
            "logicalBytes": inventory.logical_bytes,
            "manifestSha256": inventory.manifest_sha256,
            "matchCount": inventory.match_count,
            "nonforcedDecisionCount": inventory.nonforced_decision_count,
            "playerCounts": list(inventory.player_counts),
            "roundedFileBytes": inventory.rounded_file_bytes,
        },
        "lowDiskStagePlanSha256": plan.manifest_sha256,
        "promoted": True,
        "runNamespace": plan.run_namespace,
        "shard": {
            "index": record["index"],
            "manifestSha256": record["manifestSha256"],
            "stagedName": record["stagedName"],
            "tier": record["tier"],
        },
        "sourceIndexManifestSha256": plan.document["source"][  # type: ignore[index]
            "indexManifestSha256"
        ],
        "version": V5_LOW_DISK_PROMOTION_RECEIPT_VERSION,
    }


def _validate_promotion_receipt_document(
    document: object,
) -> Mapping[str, object]:
    root = _exact_keys(
        document,
        {
            "format",
            "inventory",
            "lowDiskStagePlanSha256",
            "promoted",
            "runNamespace",
            "shard",
            "sourceIndexManifestSha256",
            "version",
        },
        "V5 low-disk promotion receipt",
    )
    if (
        root["format"] != V5_LOW_DISK_PROMOTION_RECEIPT_FORMAT
        or root["version"] != V5_LOW_DISK_PROMOTION_RECEIPT_VERSION
        or root["promoted"] is not True
        or not isinstance(root["runNamespace"], str)
        or _RUN_NAMESPACE.fullmatch(str(root["runNamespace"])) is None
    ):
        raise ValueError("V5 low-disk promotion receipt contract is incompatible")
    _sha(root["lowDiskStagePlanSha256"], "promotion receipt stage-plan SHA")
    _sha(root["sourceIndexManifestSha256"], "promotion receipt source-index SHA")
    shard = _exact_keys(
        root["shard"],
        {"index", "manifestSha256", "stagedName", "tier"},
        "V5 promotion receipt shard",
    )
    _integer(shard["index"], "promotion receipt shard index")
    manifest_sha = _sha(
        shard["manifestSha256"], "promotion receipt shard manifest SHA"
    )
    if (
        shard["tier"] not in _TIERS
        or shard["stagedName"]
        != f"shard-{int(shard['index']):03d}-{manifest_sha[:16]}"
    ):
        raise ValueError("V5 promotion receipt shard identity is invalid")
    inventory = _exact_keys(
        root["inventory"],
        {
            "decisionCount",
            "fileCount",
            "logicalBytes",
            "manifestSha256",
            "matchCount",
            "nonforcedDecisionCount",
            "playerCounts",
            "roundedFileBytes",
        },
        "V5 promotion receipt inventory",
    )
    _sha(inventory["manifestSha256"], "promotion receipt inventory manifest SHA")
    for name in (
        "decisionCount",
        "fileCount",
        "logicalBytes",
        "matchCount",
        "nonforcedDecisionCount",
        "roundedFileBytes",
    ):
        _integer(inventory[name], f"promotion receipt {name}", minimum=1)
    players = inventory["playerCounts"]
    if (
        not isinstance(players, list)
        or not players
        or players != sorted(set(players))
        or any(type(value) is not int or not 4 <= value <= 10 for value in players)
    ):
        raise ValueError("V5 promotion receipt player counts are invalid")
    return root


def load_v5_low_disk_promotion_receipt(
    receipt_path: str | Path,
) -> tuple[Mapping[str, object], str]:
    _, document, digest = _read_exact_manifest_directory(
        receipt_path, label="V5 low-disk promotion receipt"
    )
    # Receipt directories reuse the same two canonical filenames as every
    # immutable V5 control artifact, but their document is receipt-shaped.
    return _validate_promotion_receipt_document(document), digest


def _receipt_path(receipt_root: Path, record: Mapping[str, object]) -> Path:
    return receipt_root / str(record["stagedName"])


def _publish_or_reuse_promotion_receipt(
    target: Path, expected: Mapping[str, object]
) -> str:
    raw = _canonical_json_bytes(dict(expected))
    digest = _sha256_bytes(raw)
    _exclusive_publish_directory(
        target,
        {
            "manifest.json": raw,
            "manifest.json.sha256": f"{digest}  manifest.json\n".encode("ascii"),
        },
    )
    observed, observed_sha = load_v5_low_disk_promotion_receipt(target)
    if observed != expected or observed_sha != digest:
        raise ValueError("existing V5 promotion receipt is foreign or corrupt")
    return observed_sha


def _linux_boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except OSError:
        return None
    return value if re.fullmatch(r"[0-9a-fA-F-]{16,64}", value) else None


def _linux_process_start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    try:
        # Fields after the executable name begin at proc field 3.  starttime is
        # field 22, therefore offset 19 in this tail.
        return int(raw.rsplit(")", 1)[1].split()[19])
    except (IndexError, ValueError):
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fsync_directory(path: Path) -> None:
    """Persist a same-directory link/unlink transition on the GPU host.

    Windows does not expose the POSIX directory-fsync contract used by the
    Linux training host, so local unit tests intentionally use a no-op there.
    File contents are still flushed before every link on both platforms.
    """

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _promotion_lock_pending_prefix(lock: Path) -> str:
    return f"{lock.name}.pending-"


def _retire_promotion_lock_pending_files(root: Path, lock: Path) -> None:
    """Remove only exact pre-link crash remnants after owning the real lock."""

    prefix = _promotion_lock_pending_prefix(lock)
    changed = False
    for candidate in tuple(root.iterdir()):
        suffix = candidate.name[len(prefix):] if candidate.name.startswith(prefix) else ""
        if re.fullmatch(r"[0-9a-f]{32}", suffix) is None:
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("V5 promotion lock pending artifact is foreign")
        candidate.unlink()
        changed = True
    if changed:
        _fsync_directory(root)


def _acquire_promotion_lock(lock: Path, raw: bytes) -> tuple[int, int]:
    """Publish a complete canonical lock with an atomic no-replace hardlink.

    A crash while writing can leave only a uniquely named pending file.  The
    canonical lock name is linked only after that file is flushed, so it can
    never contain a torn JSON document and permanently brick recovery.
    """

    root = lock.parent
    pending = root / f"{_promotion_lock_pending_prefix(lock)}{uuid.uuid4().hex}"
    identity: tuple[int, int] | None = None
    try:
        try:
            with pending.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(pending, lock)
            observed = os.lstat(lock)
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError("published V5 promotion lock is not a regular file")
            identity = (int(observed.st_dev), int(observed.st_ino))
            _fsync_directory(root)
        finally:
            try:
                pending.unlink()
            except FileNotFoundError:
                pass
            else:
                _fsync_directory(root)
        if identity is None:
            raise RuntimeError("V5 promotion lock publication did not complete")
        _retire_promotion_lock_pending_files(root, lock)
    except BaseException as error:
        if identity is not None:
            # If the canonical name still denotes our exact inode and bytes,
            # retire it durably and preserve the original exception.  A
            # replacement is never unlinked.
            try:
                released = _release_exact_promotion_lock(lock, raw, identity)
            except OSError as cleanup_error:
                error.add_note(
                    f"V5 promotion lock cleanup also failed: {cleanup_error}"
                )
            else:
                if not released:
                    error.add_note(
                        "V5 promotion lock was replaced; foreign bytes were retained"
                    )
        raise
    return identity


def _release_exact_promotion_lock(
    lock: Path, raw: bytes, identity: tuple[int, int]
) -> bool:
    try:
        observed_stat = os.lstat(lock)
    except FileNotFoundError:
        return False
    if (
        stat.S_ISLNK(observed_stat.st_mode)
        or not stat.S_ISREG(observed_stat.st_mode)
        or (int(observed_stat.st_dev), int(observed_stat.st_ino)) != identity
        or lock.read_bytes() != raw
    ):
        # Never unlink a lock that changed ownership or content underneath us.
        return False
    lock.unlink()
    _fsync_directory(lock.parent)
    return True


def _promotion_lock_document(
    plan: V5LowDiskStagePlan, record: Mapping[str, object]
) -> dict[str, object]:
    return {
        "bootId": _linux_boot_id(),
        "format": V5_LOW_DISK_PROMOTION_LOCK_FORMAT,
        "hostname": socket.gethostname(),
        "lowDiskStagePlanSha256": plan.manifest_sha256,
        "pid": os.getpid(),
        "processStartTicks": _linux_process_start_ticks(os.getpid()),
        "stagedName": record["stagedName"],
        "version": V5_LOW_DISK_PROMOTION_LOCK_VERSION,
    }


def _load_promotion_lock(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V5 promotion lock is malformed") from error
    if _canonical_json_bytes(document) != raw:
        raise ValueError("V5 promotion lock is not canonical")
    root = _exact_keys(
        document,
        {
            "bootId",
            "format",
            "hostname",
            "lowDiskStagePlanSha256",
            "pid",
            "processStartTicks",
            "stagedName",
            "version",
        },
        "V5 promotion lock",
    )
    if (
        root["format"] != V5_LOW_DISK_PROMOTION_LOCK_FORMAT
        or root["version"] != V5_LOW_DISK_PROMOTION_LOCK_VERSION
        or not isinstance(root["hostname"], str)
        or not isinstance(root["stagedName"], str)
    ):
        raise ValueError("V5 promotion lock contract is incompatible")
    _sha(root["lowDiskStagePlanSha256"], "promotion lock stage-plan SHA")
    _integer(root["pid"], "promotion lock pid", minimum=1)
    for name in ("bootId", "processStartTicks"):
        if root[name] is not None and (
            isinstance(root[name], bool)
            or not isinstance(root[name], (str if name == "bootId" else int))
        ):
            raise ValueError(f"V5 promotion lock {name} is invalid")
    return root


def _promotion_lock_is_active(lock: Mapping[str, object]) -> bool:
    if lock["hostname"] != socket.gethostname():
        # A different host may share the persistent filesystem.  Without a
        # distributed liveness oracle, treating it as active is the only safe
        # choice.
        return True
    current_boot = _linux_boot_id()
    recorded_boot = lock["bootId"]
    if (
        isinstance(recorded_boot, str)
        and current_boot is not None
        and recorded_boot != current_boot
    ):
        return False
    pid = int(lock["pid"])
    if not _pid_is_alive(pid):
        return False
    recorded_start = lock["processStartTicks"]
    current_start = _linux_process_start_ticks(pid)
    if (
        isinstance(recorded_start, int)
        and current_start is not None
        and recorded_start != current_start
    ):
        return False
    return True


def verify_and_promote_v5_staged_shard(
    plan_path: str | Path,
    *,
    shard_index: int,
    incoming_path: str | Path,
    tier_root: str | Path,
    receipt_root: str | Path,
) -> Path:
    """Checksum an incoming replica, then atomically give it its final name.

    Transfer tools must write to ``.<final-name>.incoming-<nonce>``.  A partial
    SCP can therefore never masquerade as an immutable staged shard.  This
    function never removes an incoming directory or replaces an existing
    target; failed transfers remain explicit evidence for the orchestrator.
    If a process died after the atomic rename, an exact target can issue or
    reuse its deterministic receipt without copying the shard again.
    """

    plan = load_v5_low_disk_stage_plan(plan_path)
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < len(plan.shards)
    ):
        raise ValueError("shard_index is outside the immutable staging plan")
    record = plan.shards[shard_index]
    tier = str(record["tier"])
    root = _require_namespaced_tier_root(tier_root, plan.run_namespace, tier)
    receipts = _require_namespaced_receipt_root(receipt_root, plan.run_namespace)
    receipts.mkdir(parents=True, exist_ok=True)
    unresolved_incoming = Path(incoming_path)
    if unresolved_incoming.is_symlink():
        raise ValueError("incoming V5 shard root must not be a symlink")
    incoming = unresolved_incoming.resolve()
    if incoming.parent != root:
        raise ValueError("incoming V5 shard is outside its namespaced tier root")
    final_name = str(record["stagedName"])
    prefix = f".{final_name}.incoming-"
    suffix = incoming.name[len(prefix):] if incoming.name.startswith(prefix) else ""
    if not suffix or re.fullmatch(r"[a-zA-Z0-9]{8,64}", suffix) is None:
        raise ValueError("incoming V5 shard name does not use the safe staging form")
    target = root / final_name
    block = int(plan.document["allocation"]["filesystemBlockBytes"])  # type: ignore[index]
    lock = root / f".{final_name}.promote.lock"
    if _path_lexists(lock):
        if lock.is_symlink() or not lock.is_file():
            raise ValueError("V5 promotion lock is foreign or corrupt")
        prior_lock = _load_promotion_lock(lock)
        if (
            prior_lock["lowDiskStagePlanSha256"] != plan.manifest_sha256
            or prior_lock["stagedName"] != final_name
        ):
            raise ValueError("V5 promotion lock belongs to a foreign operation")
        if _promotion_lock_is_active(prior_lock):
            raise BlockingIOError("V5 shard promotion is still active")
        # The exact owner is demonstrably dead or from an earlier boot.  Remove
        # only this validated ephemeral lock, then contend normally via O_EXCL.
        lock.unlink()
        _fsync_directory(root)

    lock_document = _promotion_lock_document(plan, record)
    lock_raw = _canonical_json_bytes(lock_document)
    lock_identity: tuple[int, int] | None = None
    try:
        lock_identity = _acquire_promotion_lock(lock, lock_raw)
        if _path_lexists(target):
            if target.is_symlink() or not target.is_dir():
                raise ValueError("existing V5 staged target is foreign")
            actual = inventory_v5_training_shard(
                target, filesystem_block_bytes=block
            )
            _verify_inventory_record(record, actual)
            if _path_lexists(incoming):
                raise FileExistsError(
                    "exact staged target already exists but incoming transfer also remains"
                )
        else:
            if not incoming.is_dir():
                raise FileNotFoundError("incoming V5 staged shard is missing")
            actual = inventory_v5_training_shard(
                incoming, filesystem_block_bytes=block
            )
            _verify_inventory_record(record, actual)
            expected_receipt = _promotion_receipt_document(plan, record, actual)
            receipt_target = _receipt_path(receipts, record)
            if _path_lexists(receipt_target):
                observed, _ = load_v5_low_disk_promotion_receipt(receipt_target)
                if observed != expected_receipt:
                    raise ValueError(
                        "existing V5 promotion receipt is foreign or corrupt"
                    )
            if _path_lexists(target):
                raise FileExistsError(
                    f"immutable V5 staged shard appeared during promotion: {target}"
                )
            # Source and target share a parent/filesystem.  The platform
            # primitive is both atomic and no-replace.
            _rename_directory_noreplace(incoming, target)

        expected_receipt = _promotion_receipt_document(plan, record, actual)
        _publish_or_reuse_promotion_receipt(
            _receipt_path(receipts, record), expected_receipt
        )
    finally:
        if lock_identity is not None:
            released = _release_exact_promotion_lock(
                lock, lock_raw, lock_identity
            )
            if not released and sys.exc_info()[0] is None:
                raise ValueError(
                    "V5 promotion lock ownership changed before release"
                )
    return target


def _hybrid_index_metadata(plan: V5LowDiskStagePlan) -> dict[str, object]:
    return {
        "lowDiskStageContract": V5_LOW_DISK_STAGE_INDEX_CONTRACT,
        "lowDiskStagePlanSha256": plan.manifest_sha256,
        "runNamespace": plan.run_namespace,
        "sourceIndexManifestSha256": plan.document["source"][  # type: ignore[index]
            "indexManifestSha256"
        ],
    }


def _source_index_inventory(
    source: V5SourceIndexRecord,
) -> Counter[tuple[str, int, int]]:
    records = source.document["shards"]
    assert isinstance(records, list)
    return Counter(
        (
            str(record["manifestSha256"]),
            int(record["decisionCount"]),
            int(record["matchCount"]),
        )
        for record in records
    )


def _verify_collection_plan_stage_binding(
    collection_plan_path: str | Path,
    stage_plan: V5LowDiskStagePlan,
    source_index: V5SourceIndexRecord,
    paths: Sequence[Path],
    inventories: Sequence[V5ShardStorageInventory],
) -> tuple[
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, object],
]:
    from v5_collection_plan import (
        load_collection_plan,
        validate_actual_nonforced_corpus,
        verify_planned_collection_corpus,
    )

    collection = load_collection_plan(collection_plan_path)
    if collection.purpose != "production":
        raise ValueError("hybrid training requires a production collection plan")
    if collection.run_namespace != stage_plan.run_namespace:
        raise ValueError("low-disk stage namespace differs from collection plan")
    if len(collection.shards) != len(paths):
        raise ValueError("low-disk stage shard count differs from collection plan")
    if len(inventories) != len(paths):
        raise ValueError("low-disk staged inventory coverage is incomplete")

    verified = verify_planned_collection_corpus(
        collection,
        Path(collection_plan_path).resolve().parent,
        index_shard_paths=paths,
    )
    decisions = verified["actualDecisionCountsByPlayerCount"]
    nonforced = verified["actualNonforcedDecisionCountsByPlayerCount"]
    matches = verified["actualMatchCountsByPlayerCount"]
    manifests = verified["shardManifestSha256s"]
    if not all(
        isinstance(value, Mapping)
        for value in (decisions, nonforced, matches, manifests)
    ):
        raise ValueError("verified V5 collection corpus counts are malformed")

    corpus_gate = validate_actual_nonforced_corpus(collection, nonforced)
    expected_metadata = {
        "actualCorpusGate": corpus_gate,
        "actualDecisionCountsByPlayerCount": decisions,
        "actualMatchCountsByPlayerCount": matches,
        "actualNonforcedDecisionCountsByPlayerCount": nonforced,
        "behavior": dict(collection.behavior),
        "calibrationReportSha256": collection.document["calibration"][  # type: ignore[index]
            "reportSha256"
        ],
        "collectionPlanManifestSha256": collection.manifest_sha256,
        "completeShardIndices": verified["completeShardIndices"],
        "matchCoordinatesSha256": verified["matchCoordinatesSha256"],
        "matchProvenanceContract": verified["matchProvenanceContract"],
        "plannedMatchCountsByPlayerCount": collection.document["matchCounts"],
        "policyNumericsSha256": collection.document["policyNumericsSha256"],
        "shardManifestSha256s": manifests,
        "sourceInventorySha256": collection.document["sourceInventorySha256"],
        "totalUniqueMatches": verified["totalUniqueMatches"],
    }
    if source_index.document["metadata"] != expected_metadata:
        raise ValueError("source-index record metadata does not recompute")
    if (
        source_index.document["playerCounts"] != list(range(4, 11))
        or source_index.document["counts"]
        != {
            "decisions": sum(decisions.values()),
            "matches": sum(matches.values()),
            "shards": len(collection.shards),
        }
        or corpus_gate.get("passed") is not True
    ):
        raise ValueError("source-index record failed the production corpus gate")
    return decisions, matches, nonforced, corpus_gate


def verify_v5_hybrid_stage(
    plan_path: str | Path,
    *,
    persistent_root: str | Path,
    volatile_root: str | Path,
    source_index_record: str | Path,
    promotion_receipt_root: str | Path,
    hybrid_index: str | Path | None = None,
    collection_plan: str | Path | None = None,
) -> V5VerifiedHybridStage:
    """Purely verify one materialized cross-filesystem corpus.

    With ``hybrid_index=None`` this verifies the exact materialization before
    publishing an index.  Supplying an index adds byte-for-byte index binding;
    workflow admission calls the same path immediately before preflight/train.
    """

    plan_root = Path(plan_path).resolve()
    if plan_root.name != LOW_DISK_STAGE_PLAN_NAME:
        raise ValueError(
            f"V5 low-disk stage plan directory must be named {LOW_DISK_STAGE_PLAN_NAME}"
        )
    if Path(plan_path).is_symlink() or not plan_root.is_dir():
        raise ValueError("V5 low-disk stage plan must be a real directory")
    plan_children = tuple(plan_root.iterdir())
    if (
        {child.name for child in plan_children}
        != {"plan.json", "plan.json.sha256"}
        or any(child.is_symlink() or not child.is_file() for child in plan_children)
    ):
        raise ValueError("V5 low-disk stage plan inventory is non-canonical")
    plan = load_v5_low_disk_stage_plan(plan_root)
    source_root = Path(source_index_record).resolve()
    source = load_v5_source_index_record(source_root)
    if source.manifest_sha256 != plan.document["source"]["indexManifestSha256"]:  # type: ignore[index]
        raise ValueError("V5 source-index record differs from the low-disk stage plan")

    roots = {
        PERSISTENT_TIER: _require_namespaced_tier_root(
            persistent_root, plan.run_namespace, PERSISTENT_TIER
        ),
        VOLATILE_TIER: _require_namespaced_tier_root(
            volatile_root, plan.run_namespace, VOLATILE_TIER
        ),
    }
    receipts = _require_namespaced_receipt_root(
        promotion_receipt_root, plan.run_namespace
    )
    if roots[PERSISTENT_TIER] == roots[VOLATILE_TIER]:
        raise ValueError("V5 staging tiers resolve to the same directory")
    if _device_id(roots[PERSISTENT_TIER]) == _device_id(roots[VOLATILE_TIER]):
        raise ValueError("V5 persistent and volatile tiers must be different filesystems")

    index_path = Path(hybrid_index).resolve() if hybrid_index is not None else None
    if index_path is not None:
        if source_root.parent != index_path.parent or plan_root.parent != index_path.parent:
            raise ValueError(
                "V5 stage plan, source-index record, and hybrid index must share a control directory"
            )
        for root in roots.values():
            try:
                index_path.relative_to(root)
            except ValueError:
                continue
            raise ValueError("V5 hybrid index must be outside both shard-only tier roots")

    expected_children = {
        tier: {
            str(record["stagedName"])
            for record in plan.shards
            if record["tier"] == tier
        }
        for tier in _TIERS
    }
    tiers = plan.document["tiers"]
    assert isinstance(tiers, Mapping)
    for tier, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"V5 {tier} stage root is missing")
        children = tuple(root.iterdir())
        observed = {child.name for child in children}
        if observed != expected_children[tier]:
            raise ValueError(f"V5 {tier} stage directory coverage drifted")
        if any(child.is_symlink() or not child.is_dir() for child in children):
            raise ValueError(f"V5 {tier} staged children must be real directories")
        tier_contract = tiers[tier]
        assert isinstance(tier_contract, Mapping)
        reserve = int(tier_contract["reserveBytes"])
        if shutil.disk_usage(root).free < reserve:
            raise ValueError(f"V5 {tier} tier no longer preserves reserved headroom")

    if not receipts.is_dir():
        raise FileNotFoundError("V5 low-disk promotion receipt root is missing")
    receipt_children = tuple(receipts.iterdir())
    expected_receipts = {
        str(record["stagedName"]) for record in plan.shards
    }
    if {child.name for child in receipt_children} != expected_receipts:
        raise ValueError("V5 promotion receipt directory coverage drifted")
    if any(child.is_symlink() or not child.is_dir() for child in receipt_children):
        raise ValueError("V5 promotion receipts must be real directories")

    verified_paths: list[Path] = []
    inventories: list[V5ShardStorageInventory] = []
    block = int(plan.document["allocation"]["filesystemBlockBytes"])  # type: ignore[index]
    for record in plan.shards:
        tier = str(record["tier"])
        path = roots[tier] / str(record["stagedName"])
        actual = inventory_v5_training_shard(path, filesystem_block_bytes=block)
        _verify_inventory_record(record, actual)
        expected_receipt = _promotion_receipt_document(plan, record, actual)
        receipt, _ = load_v5_low_disk_promotion_receipt(
            _receipt_path(receipts, record)
        )
        if receipt != expected_receipt:
            raise ValueError("V5 promotion receipt differs from staged bytes")
        verified_paths.append(path)
        inventories.append(actual)

    source_counts = plan.document["source"]["counts"]  # type: ignore[index]
    source_inventory = _source_index_inventory(source)
    materialized_inventory = Counter(
        (
            item.manifest_sha256,
            item.decision_count,
            item.match_count,
        )
        for item in inventories
    )
    if source_inventory != materialized_inventory:
        raise ValueError("source-index record shard inventory differs from staged bytes")
    if source.document["counts"] != source_counts:
        raise ValueError("source-index record aggregate counts differ from stage plan")
    if sum(item.nonforced_decision_count for item in inventories) != plan.document[
        "source"
    ]["nonforcedDecisions"]:  # type: ignore[index]
        raise ValueError("staged nonforced decision total differs from stage plan")

    actual_decisions: Mapping[str, int] | None = None
    actual_matches: Mapping[str, int] | None = None
    actual_nonforced: Mapping[str, int] | None = None
    corpus_gate: Mapping[str, object] | None = None
    if collection_plan is not None:
        (
            actual_decisions,
            actual_matches,
            actual_nonforced,
            corpus_gate,
        ) = _verify_collection_plan_stage_binding(
            collection_plan, plan, source, verified_paths, inventories
        )

    index_digest: str | None = None
    if index_path is not None:
        _, index_document, index_digest = _read_exact_manifest_directory(
            index_path, label="V5 hybrid index"
        )
        index = load_v5_index_manifest(index_path)
        try:
            if (
                index.manifest != index_document
                or index.manifest.get("metadata") != _hybrid_index_metadata(plan)
                or set(index.shard_paths) != set(verified_paths)
                or index.decision_count != source_counts["decisions"]
                or index.match_count != source_counts["matches"]
                or len(index.shard_paths) != source_counts["shards"]
            ):
                raise ValueError("V5 hybrid index differs from the verified stage")
        finally:
            index.close()

    return V5VerifiedHybridStage(
        plan=plan,
        source_index=source,
        shard_paths=tuple(verified_paths),
        shard_manifest_sha256s=tuple(
            inventory.manifest_sha256 for inventory in inventories
        ),
        hybrid_index_manifest_sha256=index_digest,
        actual_decision_counts_by_player_count=actual_decisions,
        actual_match_counts_by_player_count=actual_matches,
        actual_nonforced_decision_counts_by_player_count=actual_nonforced,
        actual_corpus_gate=corpus_gate,
    )


def verify_and_publish_v5_hybrid_index(
    plan_path: str | Path,
    *,
    persistent_root: str | Path,
    volatile_root: str | Path,
    source_index_record: str | Path,
    promotion_receipt_root: str | Path,
    output_index: str | Path,
    collection_plan: str | Path | None = None,
) -> str:
    """Verify every replica and publish a standard cross-filesystem V5 index."""

    output = Path(output_index).resolve()
    for raw_root in (persistent_root, volatile_root):
        root = Path(raw_root).resolve()
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise ValueError("V5 hybrid index must be outside both shard-only tier roots")
    if Path(plan_path).resolve().parent != output.parent or Path(
        source_index_record
    ).resolve().parent != output.parent:
        raise ValueError(
            "V5 stage plan, source-index record, and output index must share a control directory"
        )
    verified = verify_v5_hybrid_stage(
        plan_path,
        persistent_root=persistent_root,
        volatile_root=volatile_root,
        source_index_record=source_index_record,
        promotion_receipt_root=promotion_receipt_root,
        collection_plan=collection_plan,
    )
    digest = publish_v5_index_manifest(
        output,
        verified.shard_paths,
        metadata=_hybrid_index_metadata(verified.plan),
    )
    final = verify_v5_hybrid_stage(
        plan_path,
        persistent_root=persistent_root,
        volatile_root=volatile_root,
        source_index_record=source_index_record,
        promotion_receipt_root=promotion_receipt_root,
        hybrid_index=output,
        collection_plan=collection_plan,
    )
    if final.hybrid_index_manifest_sha256 != digest:
        raise RuntimeError("published V5 hybrid index checksum drifted")
    return digest


def measure_tar_zstd_stream(
    shard_path: str | Path, *, level: int = 3
) -> dict[str, object]:
    """Measure a deterministic tar.zst stream without writing an archive."""

    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 19:
        raise ValueError("zstd level must be in 1..19")
    root = Path(shard_path).resolve()
    inventory = inventory_v5_training_shard(root)
    tar = shutil.which("tar")
    zstd = shutil.which("zstd")
    if tar is None or zstd is None:
        raise RuntimeError("tar and zstd executables are required for compression probing")
    tar_command = [
        tar,
        "-C", str(root.parent),
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-cf", "-",
        root.name,
    ]
    compressor_command = [zstd, "-q", f"-{level}", "-T0", "-c"]
    tar_process = subprocess.Popen(tar_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert tar_process.stdout is not None
    compressor = subprocess.Popen(
        compressor_command,
        stdin=tar_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar_process.stdout.close()
    digest = hashlib.sha256()
    compressed_bytes = 0
    assert compressor.stdout is not None
    while True:
        chunk = compressor.stdout.read(1024 * 1024)
        if not chunk:
            break
        compressed_bytes += len(chunk)
        digest.update(chunk)
    compressor_stderr = compressor.stderr.read() if compressor.stderr is not None else b""
    compressor_code = compressor.wait()
    tar_stderr = tar_process.stderr.read() if tar_process.stderr is not None else b""
    tar_code = tar_process.wait()
    if tar_code != 0 or compressor_code != 0:
        detail = (tar_stderr + compressor_stderr).decode("utf-8", errors="replace")
        raise RuntimeError(f"tar.zst compression probe failed: {detail.strip()}")
    return {
        "compressedBytes": compressed_bytes,
        "compressedSha256": digest.hexdigest(),
        "compressionRatio": compressed_bytes / inventory.logical_bytes,
        "decisionCount": inventory.decision_count,
        "logicalBytes": inventory.logical_bytes,
        "matchCount": inventory.match_count,
        "nonforcedDecisionCount": inventory.nonforced_decision_count,
        "rawBytesPerNonforcedDecision": (
            inventory.logical_bytes / inventory.nonforced_decision_count
        ),
        "tarZstdBytesPerNonforcedDecision": (
            compressed_bytes / inventory.nonforced_decision_count
        ),
    }


def _json_summary(value: Mapping[str, object]) -> None:
    sys.stdout.write(_canonical_json_bytes(dict(value)).decode("ascii"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="build an immutable hybrid capacity plan")
    plan.add_argument("--source-index", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--run-namespace", required=True)
    plan.add_argument("--persistent-free-bytes", type=int, required=True)
    plan.add_argument("--volatile-free-bytes", type=int, required=True)
    plan.add_argument(
        "--persistent-reserve-bytes", type=int, default=DEFAULT_PERSISTENT_RESERVE_BYTES
    )
    plan.add_argument(
        "--volatile-reserve-bytes", type=int, default=DEFAULT_VOLATILE_RESERVE_BYTES
    )
    source_record = commands.add_parser(
        "record-source-index",
        help="publish the raw canonical source-index manifest and sidecar",
    )
    source_record.add_argument("--source-index", required=True)
    source_record.add_argument("--output", required=True)
    verify = commands.add_parser(
        "verify-index", help="verify staged replicas and publish the cross-filesystem index"
    )
    verify.add_argument("--plan", required=True)
    verify.add_argument("--persistent-root", required=True)
    verify.add_argument("--volatile-root", required=True)
    verify.add_argument("--source-index-record", required=True)
    verify.add_argument("--receipt-root", required=True)
    verify.add_argument("--collection-plan")
    verify.add_argument("--output-index", required=True)
    audit = commands.add_parser(
        "verify-stage", help="re-open and verify an existing hybrid stage/index"
    )
    audit.add_argument("--plan", required=True)
    audit.add_argument("--persistent-root", required=True)
    audit.add_argument("--volatile-root", required=True)
    audit.add_argument("--source-index-record", required=True)
    audit.add_argument("--receipt-root", required=True)
    audit.add_argument("--hybrid-index", required=True)
    audit.add_argument("--collection-plan")
    promote = commands.add_parser(
        "promote-shard", help="verify an incoming shard and atomically promote it"
    )
    promote.add_argument("--plan", required=True)
    promote.add_argument("--shard-index", type=int, required=True)
    promote.add_argument("--incoming", required=True)
    promote.add_argument("--tier-root", required=True)
    promote.add_argument("--receipt-root", required=True)
    probe = commands.add_parser(
        "probe", help="measure a verified shard and optional streamed tar.zst"
    )
    probe.add_argument("--shard", required=True)
    probe.add_argument("--tar-zstd", action="store_true")
    probe.add_argument("--zstd-level", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "plan":
        plan = build_v5_low_disk_stage_plan(
            arguments.source_index,
            run_namespace=arguments.run_namespace,
            persistent_free_bytes=arguments.persistent_free_bytes,
            volatile_free_bytes=arguments.volatile_free_bytes,
            persistent_reserve_bytes=arguments.persistent_reserve_bytes,
            volatile_reserve_bytes=arguments.volatile_reserve_bytes,
        )
        digest = publish_v5_low_disk_stage_plan(arguments.output, plan)
        _json_summary(
            {
                "planSha256": digest,
                "shards": len(plan.shards),
                "source": plan.document["source"],
                "target": str(Path(arguments.output).resolve()),
                "tiers": plan.document["tiers"],
            }
        )
    elif arguments.command == "record-source-index":
        digest = publish_v5_source_index_record(
            arguments.source_index, arguments.output
        )
        _json_summary(
            {"sourceIndexManifestSha256": digest, "target": str(Path(arguments.output).resolve())}
        )
    elif arguments.command == "verify-index":
        digest = verify_and_publish_v5_hybrid_index(
            arguments.plan,
            persistent_root=arguments.persistent_root,
            volatile_root=arguments.volatile_root,
            source_index_record=arguments.source_index_record,
            promotion_receipt_root=arguments.receipt_root,
            output_index=arguments.output_index,
            collection_plan=arguments.collection_plan,
        )
        _json_summary(
            {"indexSha256": digest, "target": str(Path(arguments.output_index).resolve())}
        )
    elif arguments.command == "verify-stage":
        verified = verify_v5_hybrid_stage(
            arguments.plan,
            persistent_root=arguments.persistent_root,
            volatile_root=arguments.volatile_root,
            source_index_record=arguments.source_index_record,
            promotion_receipt_root=arguments.receipt_root,
            hybrid_index=arguments.hybrid_index,
            collection_plan=arguments.collection_plan,
        )
        _json_summary(
            {
                "hybridIndexManifestSha256": verified.hybrid_index_manifest_sha256,
                "shards": len(verified.shard_paths),
                "sourceIndexManifestSha256": verified.source_index.manifest_sha256,
            }
        )
    elif arguments.command == "promote-shard":
        target = verify_and_promote_v5_staged_shard(
            arguments.plan,
            shard_index=arguments.shard_index,
            incoming_path=arguments.incoming,
            tier_root=arguments.tier_root,
            receipt_root=arguments.receipt_root,
        )
        _json_summary({"shardIndex": arguments.shard_index, "target": str(target)})
    else:
        inventory = inventory_v5_training_shard(arguments.shard)
        if arguments.tar_zstd:
            summary = measure_tar_zstd_stream(arguments.shard, level=arguments.zstd_level)
        else:
            summary = {
                "decisionCount": inventory.decision_count,
                "logicalBytes": inventory.logical_bytes,
                "matchCount": inventory.match_count,
                "nonforcedDecisionCount": inventory.nonforced_decision_count,
                "rawBytesPerNonforcedDecision": (
                    inventory.logical_bytes / inventory.nonforced_decision_count
                ),
            }
        _json_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FILESYSTEM_BLOCK_BYTES",
    "DEFAULT_PER_SHARD_OVERHEAD_BYTES",
    "DEFAULT_PERSISTENT_RESERVE_BYTES",
    "DEFAULT_PLACEMENT_QUANTUM_BYTES",
    "DEFAULT_VOLATILE_RESERVE_BYTES",
    "LOW_DISK_STAGE_PLAN_NAME",
    "PERSISTENT_TIER",
    "PROMOTION_RECEIPT_ROOT_NAME",
    "SOURCE_INDEX_RECORD_NAME",
    "V5_LOW_DISK_PROMOTION_RECEIPT_FORMAT",
    "V5_LOW_DISK_PROMOTION_RECEIPT_VERSION",
    "V5_LOW_DISK_STAGE_FORMAT",
    "V5_LOW_DISK_STAGE_INDEX_CONTRACT",
    "V5_LOW_DISK_STAGE_VERSION",
    "V5LowDiskStagePlan",
    "V5ShardStorageInventory",
    "V5SourceIndexRecord",
    "V5VerifiedHybridStage",
    "VOLATILE_TIER",
    "build_v5_low_disk_stage_plan",
    "inventory_v5_training_shard",
    "load_v5_low_disk_stage_plan",
    "load_v5_low_disk_promotion_receipt",
    "load_v5_source_index_record",
    "main",
    "measure_tar_zstd_stream",
    "publish_v5_low_disk_stage_plan",
    "publish_v5_source_index_record",
    "verify_and_publish_v5_hybrid_index",
    "verify_and_promote_v5_staged_shard",
    "verify_v5_hybrid_stage",
]
