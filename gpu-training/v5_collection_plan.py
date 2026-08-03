from __future__ import annotations

"""Immutable mixed CPU/CUDA collection planning for DALMUTI V5.

The plan is deliberately independent of SSH orchestration.  It commits every
complete-match range to one backend, binds the exact behavior Actor, critic,
public contract, source inventory, and cross-backend calibration, and admits
only completely verified immutable shards when resuming or publishing the
zero-copy corpus index.
"""

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np

from v5_collect_mappo import (
    V5_MAPPO_COLLECTION_CONTRACT,
    V5_MAPPO_REWARD_CONTRACT,
    V5_MATCH_PROVENANCE_CONTRACT,
    derive_v5_collection_match_seed,
)
from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_model import V5_POLICY_NUMERICS_SHA256


V5_COLLECTION_PLAN_FORMAT = "dalmuti-v5-mixed-backend-collection-plan"
V5_COLLECTION_PLAN_VERSION = 1
V5_CALIBRATION_REPORT_FORMAT = "dalmuti-v5-cpu-cuda-calibration"
V5_CALIBRATION_REPORT_VERSION = 1
V5_CALIBRATION_SCHEDULE_CONTRACT = "p4-p10-identical-complete-match-schedule-v1"

DEFAULT_TOTAL_MATCHES = 12_000
DEFAULT_TARGET_NONFORCED_MIN = 1_500_000
# A 1.60M center keeps the first measured corpus close to the requested
# roughly 12k complete matches while retaining headroom inside 1.5--2.0M.
DEFAULT_TARGET_NONFORCED = 1_600_000
DEFAULT_TARGET_NONFORCED_MAX = 2_000_000
DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT = 128
DEFAULT_MAX_MATCHES_PER_SHARD = 600
DEFAULT_ACTUAL_STRATUM_RELATIVE_TOLERANCE = 0.08
MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM = 20

CALIBRATION_LOG_PROBABILITY_ATOL = 2.0e-5
CALIBRATION_VALUE_ATOL = 5.0e-5
CALIBRATION_VALUE_RTOL = 1.0e-5
CALIBRATION_DERIVED_VALUE_ATOL = 2.5e-4

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Exact behavior-affecting sources used by collection.  Logical POSIX names,
# rather than absolute paths, make the inventory portable across machines.
V5_COLLECTION_SOURCE_FILES = (
    "gpu-training/v5_collect_cli.py",
    "gpu-training/v5_collection_plan.py",
    "gpu-training/v5_collect_mappo.py",
    "gpu-training/v5_contract.py",
    "gpu-training/v5_public.py",
    "gpu-training/v5_model.py",
    "gpu-training/v5_dataset.py",
    "gpu-training/v5_gae.py",
    "gpu-training/v5_export.py",
    "gpu-training/v5_train.py",
    "gpu-training/v4_collect_fixed_match_ppo.py",
    "gpu-training/v4_collect_ppo.py",
    "gpu-training/v4_env.py",
    "gpu-training/v4_evaluate.py",
    "gpu-training/v3_action_conditioned.py",
)


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical UTF-8 JSON representation used here."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_canonical_json(raw: bytes, label: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _canonical_mapping_sha256(value: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes where the host exposes that primitive."""

    if os.name == "nt":
        return
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""

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
        if result != 0:
            number = ctypes.get_errno()
            if number in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(number, os.strerror(number), target)
            raise OSError(number, os.strerror(number), target)
        return
    if os.name == "nt":
        # Windows rename is no-replace when the destination exists.
        os.rename(source, target)
        return
    raise RuntimeError("atomic no-replace directory publication is unsupported")


def _exclusive_publish_directory(target: Path, files: Mapping[str, bytes]) -> None:
    if target.is_symlink():
        raise ValueError("immutable V5 artifact target must not be a symlink")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(f"immutable V5 artifact already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name in sorted(files):
            if Path(name).name != name:
                raise ValueError("published filenames must be simple basenames")
            _write_fsync(temporary / name, files[name])
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def source_inventory_sha256(inventory: Mapping[str, str]) -> str:
    normalized = _validate_source_inventory(inventory)
    return _canonical_mapping_sha256(normalized)


def calibration_schedule_id(
    run_namespace: str,
    seed_base: int,
    match_counts: Mapping[int, int] | Sequence[tuple[int, int]],
) -> str:
    counts = dict(match_counts)
    if set(counts) != set(range(4, 11)) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in counts.values()
    ):
        raise ValueError("calibration schedule must cover positive p4..p10 counts")
    if not isinstance(run_namespace, str) or _SAFE_NAMESPACE.fullmatch(run_namespace) is None:
        raise ValueError("calibration namespace is invalid")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or not 0 <= seed_base <= 0xFFFF_FFFF:
        raise ValueError("calibration seed is invalid")
    return sha256_bytes(canonical_json_bytes({
        "contract": V5_CALIBRATION_SCHEDULE_CONTRACT,
        "matchCounts": {str(player): counts[player] for player in range(4, 11)},
        "runNamespace": run_namespace,
        "seedBase": seed_base,
    }))


def _validate_source_inventory(value: Mapping[str, str] | object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source inventory must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        if not isinstance(raw_name, str):
            raise ValueError("source inventory names must be strings")
        name = raw_name.replace("\\", "/")
        pure = Path(name)
        if (
            not name
            or name.startswith("/")
            or pure.is_absolute()
            or ".." in pure.parts
            or name != raw_name
            or name in normalized
        ):
            raise ValueError("source inventory names must be canonical relative POSIX paths")
        normalized[name] = _require_sha256(raw_digest, f"source hash {name}")
    return {name: normalized[name] for name in sorted(normalized)}


def build_source_inventory(
    source_root: str | Path,
    relative_paths: Sequence[str] = V5_COLLECTION_SOURCE_FILES,
) -> dict[str, str]:
    root = Path(source_root).resolve()
    if not relative_paths:
        raise ValueError("source inventory path list cannot be empty")
    result: dict[str, str] = {}
    for raw_name in relative_paths:
        name = raw_name.replace("\\", "/")
        if name != raw_name or name in result or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("source inventory paths must be unique canonical POSIX paths")
        path = (root / Path(*name.split("/"))).resolve()
        try:
            common = os.path.commonpath((str(root), str(path)))
        except ValueError as error:
            raise ValueError("source inventory path escapes source root") from error
        if common != str(root) or not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"source inventory file is missing or unsafe: {name}")
        result[name] = sha256_file(path)
    return _validate_source_inventory(result)


@dataclass(frozen=True)
class V5PlannedShard:
    index: int
    backend: str
    player_count: int
    match_start: int
    match_count: int
    name: str

    @property
    def match_stop(self) -> int:
        return self.match_start + self.match_count

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "index": self.index,
            "matchCount": self.match_count,
            "matchStart": self.match_start,
            "name": self.name,
            "playerCount": self.player_count,
        }


@dataclass(frozen=True)
class V5CollectionPlan:
    document: Mapping[str, object]
    manifest_sha256: str
    shards: tuple[V5PlannedShard, ...]

    @property
    def run_namespace(self) -> str:
        return str(self.document["runNamespace"])

    @property
    def seed_base(self) -> int:
        return int(self.document["seedBase"])

    @property
    def purpose(self) -> str:
        return str(self.document["purpose"])

    @property
    def behavior(self) -> Mapping[str, str]:
        value = self.document["behavior"]
        assert isinstance(value, Mapping)
        return value  # type: ignore[return-value]

    @property
    def source_inventory(self) -> Mapping[str, str]:
        value = self.document["sourceInventory"]
        assert isinstance(value, Mapping)
        return value  # type: ignore[return-value]


def resolve_total_match_count(
    *,
    default_total_matches: int = DEFAULT_TOTAL_MATCHES,
    preflight_matches: int | None = None,
    preflight_nonforced_decisions: int | None = None,
    target_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED,
    minimum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MIN,
    maximum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MAX,
) -> tuple[int, dict[str, object]]:
    """Resolve corpus size from a measured preflight or the sealed 12k default."""

    for label, value in (
        ("default_total_matches", default_total_matches),
        ("target_nonforced_decisions", target_nonforced_decisions),
        ("minimum_nonforced_decisions", minimum_nonforced_decisions),
        ("maximum_nonforced_decisions", maximum_nonforced_decisions),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not minimum_nonforced_decisions <= target_nonforced_decisions <= maximum_nonforced_decisions:
        raise ValueError("nonforced decision target must lie inside its sealed range")
    if (preflight_matches is None) != (preflight_nonforced_decisions is None):
        raise ValueError("preflight match and nonforced counts are required together")
    if preflight_matches is None:
        return default_total_matches, {
            "estimatedNonforcedDecisions": target_nonforced_decisions,
            "preflight": None,
        }
    if (
        isinstance(preflight_matches, bool)
        or not isinstance(preflight_matches, int)
        or preflight_matches < 7
        or isinstance(preflight_nonforced_decisions, bool)
        or not isinstance(preflight_nonforced_decisions, int)
        or preflight_nonforced_decisions < preflight_matches
    ):
        raise ValueError("preflight counts must describe at least seven complete matches")
    rate = preflight_nonforced_decisions / preflight_matches
    minimum_matches = math.ceil(minimum_nonforced_decisions / rate)
    maximum_matches = math.floor(maximum_nonforced_decisions / rate)
    if maximum_matches < minimum_matches:
        raise ValueError("preflight rate cannot satisfy the sealed target interval")
    resolved = min(
        maximum_matches,
        max(minimum_matches, int(round(target_nonforced_decisions / rate))),
    )
    estimated = int(round(rate * resolved))
    return resolved, {
        "estimatedNonforcedDecisions": estimated,
        "preflight": {
            "matches": preflight_matches,
            "nonforcedDecisions": preflight_nonforced_decisions,
            "nonforcedDecisionsPerMatch": rate,
        },
    }


def _balanced_match_counts(total_matches: int) -> dict[int, int]:
    if isinstance(total_matches, bool) or not isinstance(total_matches, int) or total_matches < 14:
        raise ValueError("mixed p4..p10 planning requires at least fourteen matches")
    quotient, remainder = divmod(total_matches, 7)
    return {
        player_count: quotient + (offset < remainder)
        for offset, player_count in enumerate(range(4, 11))
    }


def resolve_stratified_match_counts(
    preflight_strata: Mapping[int, tuple[int, int]],
    *,
    target_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED,
    minimum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MIN,
    maximum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MAX,
) -> tuple[dict[int, int], dict[str, object]]:
    """Equalize expected useful decisions across p4..p10 using measurements.

    ``preflight_strata[p]`` is ``(complete_matches, nonforced_decisions)``.
    Integer match rounding is deterministic and the total estimate remains in
    the sealed 1.5--2.0M interval.
    """

    if set(preflight_strata) != set(range(4, 11)):
        raise ValueError("stratified preflight must cover every p4..p10 stratum")
    if not minimum_nonforced_decisions <= target_nonforced_decisions <= maximum_nonforced_decisions:
        raise ValueError("nonforced decision target must lie inside its sealed range")
    target_per_stratum = target_nonforced_decisions / 7.0
    rates: dict[int, float] = {}
    evidence: dict[str, dict[str, object]] = {}
    counts: dict[int, int] = {}
    estimates: dict[str, int] = {}
    for player_count in range(4, 11):
        raw = preflight_strata[player_count]
        if (
            not isinstance(raw, tuple)
            or len(raw) != 2
            or isinstance(raw[0], bool)
            or not isinstance(raw[0], int)
            or raw[0] < 1
            or isinstance(raw[1], bool)
            or not isinstance(raw[1], int)
            or raw[1] < raw[0]
        ):
            raise ValueError("each stratified preflight entry must be (matches, nonforced)")
        matches, nonforced = raw
        rate = nonforced / matches
        resolved = max(2, int(round(target_per_stratum / rate)))
        estimate = int(round(resolved * rate))
        rates[player_count] = rate
        counts[player_count] = resolved
        estimates[str(player_count)] = estimate
        evidence[str(player_count)] = {
            "matches": matches,
            "nonforcedDecisions": nonforced,
            "nonforcedDecisionsPerMatch": rate,
        }
    total_estimated = sum(estimates.values())
    # Rounding a stratum independently can move the total by only a few
    # matches.  If an extremely small diagnostic sample places it outside the
    # sealed interval, scale the common per-stratum target once and recompute.
    if not minimum_nonforced_decisions <= total_estimated <= maximum_nonforced_decisions:
        bounded_target = min(
            maximum_nonforced_decisions,
            max(minimum_nonforced_decisions, target_nonforced_decisions),
        )
        target_per_stratum = bounded_target / 7.0
        for player_count in range(4, 11):
            counts[player_count] = max(
                2, int(round(target_per_stratum / rates[player_count]))
            )
            estimates[str(player_count)] = int(
                round(counts[player_count] * rates[player_count])
            )
        total_estimated = sum(estimates.values())
    if not minimum_nonforced_decisions <= total_estimated <= maximum_nonforced_decisions:
        raise ValueError("stratified preflight could not size a corpus inside the sealed range")
    return counts, {
        "estimatedNonforcedByPlayerCount": estimates,
        "estimatedNonforcedDecisions": total_estimated,
        "preflight": {"kind": "stratified", "strata": evidence},
        "stratumTargetNonforcedDecisions": target_per_stratum,
    }


def _partition_count(total: int, pieces: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, pieces)
    result = tuple(quotient + (index < remainder) for index in range(pieces))
    if not result or min(result) < 1 or max(result) - min(result) > 1 or sum(result) != total:
        raise RuntimeError("internal balanced partition invariant failed")
    return result


def completion_balanced_cpu_matches(
    match_counts: Mapping[int, int],
    cpu_seconds_per_match: Mapping[int, float],
    cuda_seconds_per_match: Mapping[int, float],
    *,
    cpu_worker_count: int = 1,
    cuda_worker_count: int = 1,
) -> dict[int, int]:
    """Balance predicted CPU/CUDA completion time independently for p4..p10.

    For each stratum this seals ``n_cpu ~= N*e_cuda/(e_cpu+e_cuda)``, where
    ``e_backend = single_worker_seconds_per_match / safe_worker_count``, and
    keeps at least one complete match on each backend.  Inputs must be measured
    seconds per complete five-act match from the same preflight schedule.
    """

    for label, count in (
        ("CPU worker", cpu_worker_count),
        ("CUDA worker", cuda_worker_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{label} count must be a positive integer")
    expected = set(range(4, 11))
    if set(match_counts) != expected or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in match_counts.values()
    ):
        raise ValueError("match_counts must bind every p4..p10 stratum to at least two matches")
    results: dict[int, int] = {}
    for label, measurements in (
        ("CPU", cpu_seconds_per_match),
        ("CUDA", cuda_seconds_per_match),
    ):
        if not isinstance(measurements, Mapping) or set(measurements) != expected:
            raise ValueError(f"{label} throughput measurements must cover p4..p10")
        for player_count, value in measurements.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{label} p{player_count} seconds/match must be positive")
    for player_count in range(4, 11):
        total = int(match_counts[player_count])
        cpu_time = float(cpu_seconds_per_match[player_count]) / cpu_worker_count
        cuda_time = float(cuda_seconds_per_match[player_count]) / cuda_worker_count
        ideal = total * cuda_time / (cpu_time + cuda_time)
        # Explicit half-up rounding is stable across Python implementations.
        results[player_count] = min(total - 1, max(1, int(math.floor(ideal + 0.5))))
    return results


def _normalize_cpu_matches_by_player_count(
    match_counts: Mapping[int, int], value: Mapping[int, int] | Mapping[str, int]
) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for raw_player, raw_count in value.items():
        try:
            player = int(raw_player)
        except (TypeError, ValueError) as error:
            raise ValueError("CPU match allocation key is invalid") from error
        if (
            player in parsed
            or player not in range(4, 11)
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
        ):
            raise ValueError("CPU match allocation must contain integer p4..p10 counts")
        parsed[player] = raw_count
    if set(parsed) != set(range(4, 11)):
        raise ValueError("CPU match allocation must cover every p4..p10 stratum")
    for player, cpu_count in parsed.items():
        total = int(match_counts[player])
        if not 1 <= cpu_count < total:
            raise ValueError(
                f"CPU p{player} allocation must leave at least one match per backend"
            )
    return {player: parsed[player] for player in range(4, 11)}


def allocate_mixed_backend_shards(
    match_counts: Mapping[int, int],
    *,
    cpu_matches_per_player_count: int = DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT,
    cpu_matches_by_player_count: Mapping[int, int] | Mapping[str, int] | None = None,
    max_matches_per_shard: int = DEFAULT_MAX_MATCHES_PER_SHARD,
) -> tuple[V5PlannedShard, ...]:
    """Allocate contiguous, gap-free p-stratum ranges to fixed backends."""

    expected = set(range(4, 11))
    if set(match_counts) != expected or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in match_counts.values()
    ):
        raise ValueError("match_counts must bind every p4..p10 stratum to at least two matches")
    for label, value in (
        ("cpu_matches_per_player_count", cpu_matches_per_player_count),
        ("max_matches_per_shard", max_matches_per_shard),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    explicit_cpu = (
        _normalize_cpu_matches_by_player_count(match_counts, cpu_matches_by_player_count)
        if cpu_matches_by_player_count is not None
        else None
    )
    shards: list[V5PlannedShard] = []
    for player_count in range(4, 11):
        total = int(match_counts[player_count])
        cpu_count = (
            explicit_cpu[player_count]
            if explicit_cpu is not None
            else min(cpu_matches_per_player_count, total - 1)
        )
        cuda_total = total - cpu_count
        cpu_pieces = math.ceil(cpu_count / max_matches_per_shard)
        cuda_pieces = math.ceil(cuda_total / max_matches_per_shard)
        pieces = tuple(
            ("cpu", count) for count in _partition_count(cpu_count, cpu_pieces)
        ) + tuple(
            ("cuda", count) for count in _partition_count(cuda_total, cuda_pieces)
        )
        start = 0
        for backend, count in pieces:
            index = len(shards)
            stop = start + count
            name = (
                f"shard-{index:03d}-p{player_count}-{backend}-"
                f"m{start:06d}-{stop:06d}"
            )
            shards.append(V5PlannedShard(index, backend, player_count, start, count, name))
            start = stop
        if start != total:
            raise RuntimeError("internal shard coverage invariant failed")
    return tuple(shards)


def _normalize_behavior(
    actor_sha256: str,
    actor_manifest_sha256: str,
    critic_sha256: str,
    pair_id: str,
    pair_manifest_sha256: str,
) -> dict[str, str]:
    return {
        "actorManifestSha256": _require_sha256(
            actor_manifest_sha256, "Actor manifest SHA-256"
        ),
        "actorSha256": _require_sha256(actor_sha256, "Actor SHA-256"),
        "criticSha256": _require_sha256(critic_sha256, "critic SHA-256"),
        "pairId": _require_sha256(pair_id, "behavior pair ID"),
        "pairManifestSha256": _require_sha256(
            pair_manifest_sha256, "behavior pair manifest SHA-256"
        ),
    }


def build_collection_plan(
    *,
    run_namespace: str,
    seed_base: int,
    behavior_actor_sha256: str,
    behavior_actor_manifest_sha256: str,
    behavior_critic_sha256: str,
    behavior_pair_id: str,
    behavior_pair_manifest_sha256: str,
    calibration_report_sha256: str,
    source_inventory: Mapping[str, str],
    total_matches: int | None = None,
    default_total_matches: int = DEFAULT_TOTAL_MATCHES,
    preflight_matches: int | None = None,
    preflight_nonforced_decisions: int | None = None,
    preflight_strata: Mapping[int, tuple[int, int]] | None = None,
    target_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED,
    minimum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MIN,
    maximum_nonforced_decisions: int = DEFAULT_TARGET_NONFORCED_MAX,
    cpu_matches_per_player_count: int = DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT,
    cpu_matches_by_player_count: Mapping[int, int] | Mapping[str, int] | None = None,
    cpu_seconds_per_match: Mapping[int, float] | None = None,
    cuda_seconds_per_match: Mapping[int, float] | None = None,
    cpu_worker_count: int = 1,
    cuda_worker_count: int = 1,
    cpu_torch_threads_per_worker: int = 1,
    cuda_torch_threads_per_worker: int = 1,
    max_matches_per_shard: int = DEFAULT_MAX_MATCHES_PER_SHARD,
    actual_stratum_relative_tolerance: float = DEFAULT_ACTUAL_STRATUM_RELATIVE_TOLERANCE,
    diagnostic_unbalanced: bool = False,
) -> V5CollectionPlan:
    if type(diagnostic_unbalanced) is not bool:
        raise ValueError("diagnostic_unbalanced must be an exact bool")
    if preflight_strata is None and not diagnostic_unbalanced:
        raise ValueError(
            "production collection requires measured p4..p10 stratified preflight"
        )
    if preflight_strata is not None and diagnostic_unbalanced:
        raise ValueError(
            "measured stratified collection must use the production purpose"
        )
    if not isinstance(run_namespace, str) or _SAFE_NAMESPACE.fullmatch(run_namespace) is None:
        raise ValueError("run_namespace must be fresh safe ASCII using 1..128 characters")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or not 0 <= seed_base <= 0xFFFF_FFFF:
        raise ValueError("seed_base must be a fresh uint32")
    if (
        isinstance(actual_stratum_relative_tolerance, bool)
        or not isinstance(actual_stratum_relative_tolerance, (int, float))
        or not math.isfinite(float(actual_stratum_relative_tolerance))
        or not 0.0 < float(actual_stratum_relative_tolerance) <= 0.25
    ):
        raise ValueError("actual stratum relative tolerance must be in (0, 0.25]")
    if preflight_strata is not None:
        undersampled = [
            player_count
            for player_count, sample in preflight_strata.items()
            if (
                type(player_count) is int
                and player_count in range(4, 11)
                and isinstance(sample, tuple)
                and len(sample) == 2
                and type(sample[0]) is int
                and sample[0] < MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM
            )
        ]
        if undersampled:
            raise ValueError(
                "production stratified preflight requires at least "
                f"{MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM} complete matches "
                f"per p-stratum; p{min(undersampled)} is undersampled"
            )
        if (
            total_matches is not None
            or preflight_matches is not None
            or preflight_nonforced_decisions is not None
        ):
            raise ValueError(
                "stratified preflight is mutually exclusive with aggregate/explicit sizing"
            )
        match_counts, estimate = resolve_stratified_match_counts(
            preflight_strata,
            target_nonforced_decisions=target_nonforced_decisions,
            minimum_nonforced_decisions=minimum_nonforced_decisions,
            maximum_nonforced_decisions=maximum_nonforced_decisions,
        )
        resolved = sum(match_counts.values())
    elif total_matches is None:
        resolved, estimate = resolve_total_match_count(
            default_total_matches=default_total_matches,
            preflight_matches=preflight_matches,
            preflight_nonforced_decisions=preflight_nonforced_decisions,
            target_nonforced_decisions=target_nonforced_decisions,
            minimum_nonforced_decisions=minimum_nonforced_decisions,
            maximum_nonforced_decisions=maximum_nonforced_decisions,
        )
        match_counts = _balanced_match_counts(resolved)
        if estimate["preflight"] is None:
            # Without p-specific measurements, do not pretend equal matches
            # imply equal or even known decision volume.
            estimate["estimatedNonforcedDecisions"] = None
        estimate["estimatedNonforcedByPlayerCount"] = None
        estimate["stratumTargetNonforcedDecisions"] = None
    else:
        if preflight_matches is not None or preflight_nonforced_decisions is not None:
            raise ValueError("explicit total_matches and measured preflight are mutually exclusive")
        if isinstance(total_matches, bool) or not isinstance(total_matches, int) or total_matches < 14:
            raise ValueError("total_matches must be at least fourteen")
        resolved = total_matches
        estimate = {
            "estimatedNonforcedDecisions": None,
            "estimatedNonforcedByPlayerCount": None,
            "preflight": None,
            "stratumTargetNonforcedDecisions": None,
        }
        match_counts = _balanced_match_counts(resolved)
    if (cpu_seconds_per_match is None) != (cuda_seconds_per_match is None):
        raise ValueError("CPU and CUDA seconds/match measurements are required together")
    for label, count in (
        ("cpu_worker_count", cpu_worker_count),
        ("cuda_worker_count", cuda_worker_count),
        ("cpu_torch_threads_per_worker", cpu_torch_threads_per_worker),
        ("cuda_torch_threads_per_worker", cuda_torch_threads_per_worker),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{label} must be a positive integer")
    if cpu_matches_by_player_count is not None and cpu_seconds_per_match is not None:
        raise ValueError("explicit CPU match counts and measured throughput are mutually exclusive")
    measured_cpu: dict[int, float] | None = None
    measured_cuda: dict[int, float] | None = None
    if cpu_seconds_per_match is not None and cuda_seconds_per_match is not None:
        measured_cpu = {int(key): float(value) for key, value in cpu_seconds_per_match.items()}
        measured_cuda = {int(key): float(value) for key, value in cuda_seconds_per_match.items()}
        cpu_allocation = completion_balanced_cpu_matches(
            match_counts,
            measured_cpu,
            measured_cuda,
            cpu_worker_count=cpu_worker_count,
            cuda_worker_count=cuda_worker_count,
        )
        allocation_contract = "measured-per-stratum-equal-finish-time-v1"
        fixed_cpu_count: int | None = None
    elif cpu_matches_by_player_count is not None:
        cpu_allocation = _normalize_cpu_matches_by_player_count(
            match_counts, cpu_matches_by_player_count
        )
        allocation_contract = "explicit-per-stratum-v1"
        fixed_cpu_count = None
    else:
        cpu_allocation = {
            player: min(cpu_matches_per_player_count, match_counts[player] - 1)
            for player in range(4, 11)
        }
        allocation_contract = "fixed-cpu-cap-per-stratum-v1"
        fixed_cpu_count = cpu_matches_per_player_count
    shards = allocate_mixed_backend_shards(
        match_counts,
        cpu_matches_per_player_count=cpu_matches_per_player_count,
        cpu_matches_by_player_count=cpu_allocation,
        max_matches_per_shard=max_matches_per_shard,
    )
    sources = _validate_source_inventory(source_inventory)
    behavior = _normalize_behavior(
        behavior_actor_sha256,
        behavior_actor_manifest_sha256,
        behavior_critic_sha256,
        behavior_pair_id,
        behavior_pair_manifest_sha256,
    )
    document: dict[str, object] = {
        "backendPolicy": {
            "allocationContract": allocation_contract,
            "cpuMatchesByPlayerCount": {
                str(player): cpu_allocation[player] for player in range(4, 11)
            },
            "cpuMatchesPerPlayerCount": fixed_cpu_count,
            "cpuWorkerCount": cpu_worker_count,
            "cudaWorkerCount": cuda_worker_count,
            "cpuTorchThreadsPerWorker": cpu_torch_threads_per_worker,
            "cudaTorchThreadsPerWorker": cuda_torch_threads_per_worker,
            "cpuSecondsPerMatch": (
                None
                if measured_cpu is None
                else {str(player): measured_cpu[player] for player in range(4, 11)}
            ),
            "cudaSecondsPerMatch": (
                None
                if measured_cuda is None
                else {str(player): measured_cuda[player] for player in range(4, 11)}
            ),
            "fixedBackendPerShard": True,
            "maxMatchesPerShard": max_matches_per_shard,
            "requiredBackends": ["cpu", "cuda"],
        },
        "behavior": behavior,
        "calibration": {
            "reportSha256": _require_sha256(
                calibration_report_sha256, "calibration report SHA-256"
            ),
            "scheduleContract": V5_CALIBRATION_SCHEDULE_CONTRACT,
        },
        "collectionContract": V5_MAPPO_COLLECTION_CONTRACT,
        "format": V5_COLLECTION_PLAN_FORMAT,
        "matchCounts": {str(key): match_counts[key] for key in sorted(match_counts)},
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "purpose": "diagnostic-unbalanced" if diagnostic_unbalanced else "production",
        "rewardContract": V5_MAPPO_REWARD_CONTRACT,
        "runNamespace": run_namespace,
        "seedBase": seed_base,
        "shards": [shard.to_dict() for shard in shards],
        "sourceInventory": sources,
        "sourceInventorySha256": source_inventory_sha256(sources),
        "targets": {
            "actualStratumRelativeTolerance": float(
                actual_stratum_relative_tolerance
            ),
            "defaultMatches": default_total_matches,
            "estimatedNonforcedDecisions": estimate["estimatedNonforcedDecisions"],
            "estimatedNonforcedByPlayerCount": estimate[
                "estimatedNonforcedByPlayerCount"
            ],
            "maximumNonforcedDecisions": maximum_nonforced_decisions,
            "minimumNonforcedDecisions": minimum_nonforced_decisions,
            "preflight": estimate["preflight"],
            "resolvedMatches": resolved,
            "stratumTargetNonforcedDecisions": estimate[
                "stratumTargetNonforcedDecisions"
            ],
            "targetNonforcedDecisions": target_nonforced_decisions,
        },
        "totalMatches": resolved,
        "version": V5_COLLECTION_PLAN_VERSION,
    }
    return validate_collection_plan_document(document)


def _parse_shard(value: object, expected_index: int) -> V5PlannedShard:
    if not isinstance(value, Mapping) or set(value) != {
        "backend", "index", "matchCount", "matchStart", "name", "playerCount"
    }:
        raise ValueError("collection plan shard record is non-canonical")
    if (
        value["index"] != expected_index
        or value["backend"] not in {"cpu", "cuda"}
        or isinstance(value["playerCount"], bool)
        or not isinstance(value["playerCount"], int)
        or not 4 <= value["playerCount"] <= 10
        or isinstance(value["matchStart"], bool)
        or not isinstance(value["matchStart"], int)
        or value["matchStart"] < 0
        or isinstance(value["matchCount"], bool)
        or not isinstance(value["matchCount"], int)
        or value["matchCount"] < 1
        or int(value["matchStart"]) + int(value["matchCount"]) > 0x1_0000_0000
        or not isinstance(value["name"], str)
    ):
        raise ValueError("collection plan shard fields are invalid")
    shard = V5PlannedShard(
        expected_index,
        str(value["backend"]),
        int(value["playerCount"]),
        int(value["matchStart"]),
        int(value["matchCount"]),
        str(value["name"]),
    )
    expected_name = (
        f"shard-{shard.index:03d}-p{shard.player_count}-{shard.backend}-"
        f"m{shard.match_start:06d}-{shard.match_stop:06d}"
    )
    if shard.name != expected_name:
        raise ValueError("collection plan shard name does not encode its exact range")
    return shard


def validate_collection_plan_document(value: Mapping[str, object] | object) -> V5CollectionPlan:
    expected_keys = {
        "backendPolicy", "behavior", "calibration", "collectionContract",
        "format", "matchCounts", "policyNumericsSha256", "publicContractSha256", "purpose", "rewardContract",
        "runNamespace", "seedBase", "shards", "sourceInventory",
        "sourceInventorySha256", "targets", "totalMatches", "version",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("collection plan fields are non-canonical")
    if (
        value["format"] != V5_COLLECTION_PLAN_FORMAT
        or value["version"] != V5_COLLECTION_PLAN_VERSION
        or value["collectionContract"] != V5_MAPPO_COLLECTION_CONTRACT
        or value["rewardContract"] != V5_MAPPO_REWARD_CONTRACT
        or value["publicContractSha256"] != V5_PUBLIC_CONTRACT_SHA256
        or value["policyNumericsSha256"] != V5_POLICY_NUMERICS_SHA256
        or value["purpose"] not in {"production", "diagnostic-unbalanced"}
        or not isinstance(value["runNamespace"], str)
        or _SAFE_NAMESPACE.fullmatch(str(value["runNamespace"])) is None
        or isinstance(value["seedBase"], bool)
        or not isinstance(value["seedBase"], int)
        or not 0 <= value["seedBase"] <= 0xFFFF_FFFF
    ):
        raise ValueError("collection plan contract, namespace, or seed is invalid")
    behavior = value["behavior"]
    if not isinstance(behavior, Mapping) or set(behavior) != {
        "actorManifestSha256", "actorSha256", "criticSha256", "pairId",
        "pairManifestSha256",
    }:
        raise ValueError("collection plan behavior binding is invalid")
    normalized_behavior = _normalize_behavior(
        str(behavior.get("actorSha256")),
        str(behavior.get("actorManifestSha256")),
        str(behavior.get("criticSha256")),
        str(behavior.get("pairId")),
        str(behavior.get("pairManifestSha256")),
    )
    if dict(behavior) != normalized_behavior:
        raise ValueError("collection plan behavior binding is non-canonical")
    calibration = value["calibration"]
    if (
        not isinstance(calibration, Mapping)
        or set(calibration) != {"reportSha256", "scheduleContract"}
        or calibration["scheduleContract"] != V5_CALIBRATION_SCHEDULE_CONTRACT
    ):
        raise ValueError("collection plan calibration binding is invalid")
    _require_sha256(calibration["reportSha256"], "calibration report SHA-256")
    sources = _validate_source_inventory(value["sourceInventory"])
    if (
        value["sourceInventory"] != sources
        or value["sourceInventorySha256"] != source_inventory_sha256(sources)
    ):
        raise ValueError("collection plan source inventory fingerprint drifted")
    counts_value = value["matchCounts"]
    if not isinstance(counts_value, Mapping) or set(counts_value) != {str(p) for p in range(4, 11)}:
        raise ValueError("collection plan must cover p4..p10")
    counts: dict[int, int] = {}
    for player_count in range(4, 11):
        raw = counts_value[str(player_count)]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 2:
            raise ValueError("collection plan p-stratum counts are invalid")
        counts[player_count] = raw
    if (
        isinstance(value["totalMatches"], bool)
        or not isinstance(value["totalMatches"], int)
        or value["totalMatches"] != sum(counts.values())
    ):
        raise ValueError("collection plan total match count disagrees with strata")
    raw_shards = value["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("collection plan has no shards")
    shards = tuple(_parse_shard(item, index) for index, item in enumerate(raw_shards))
    backend_policy = value["backendPolicy"]
    if not isinstance(backend_policy, Mapping) or set(backend_policy) != {
        "allocationContract", "cpuMatchesByPlayerCount",
        "cpuMatchesPerPlayerCount", "cpuSecondsPerMatch",
        "cudaSecondsPerMatch", "cpuWorkerCount", "cudaWorkerCount",
        "cpuTorchThreadsPerWorker", "cudaTorchThreadsPerWorker",
        "fixedBackendPerShard", "maxMatchesPerShard",
        "requiredBackends",
    } or backend_policy["fixedBackendPerShard"] is not True or backend_policy[
        "requiredBackends"
    ] != ["cpu", "cuda"]:
        raise ValueError("collection plan backend policy is invalid")
    cpu_size = backend_policy["cpuMatchesPerPlayerCount"]
    max_size = backend_policy["maxMatchesPerShard"]
    if (
        isinstance(max_size, bool)
        or not isinstance(max_size, int)
        or max_size < 1
        or (
            cpu_size is not None
            and (
                isinstance(cpu_size, bool)
                or not isinstance(cpu_size, int)
                or cpu_size < 1
            )
        )
    ):
        raise ValueError("collection plan shard size policy is invalid")
    cpu_allocation = _normalize_cpu_matches_by_player_count(
        counts, backend_policy["cpuMatchesByPlayerCount"]  # type: ignore[arg-type]
    )
    worker_fields = (
        "cpuWorkerCount",
        "cudaWorkerCount",
        "cpuTorchThreadsPerWorker",
        "cudaTorchThreadsPerWorker",
    )
    if any(
        isinstance(backend_policy[name], bool)
        or not isinstance(backend_policy[name], int)
        or int(backend_policy[name]) < 1
        for name in worker_fields
    ):
        raise ValueError("collection plan worker topology is invalid")
    allocation_contract = backend_policy["allocationContract"]
    cpu_times = backend_policy["cpuSecondsPerMatch"]
    cuda_times = backend_policy["cudaSecondsPerMatch"]
    if allocation_contract == "measured-per-stratum-equal-finish-time-v1":
        if cpu_size is not None or not isinstance(cpu_times, Mapping) or not isinstance(cuda_times, Mapping):
            raise ValueError("measured backend allocation omitted p-specific timings")
        try:
            parsed_cpu_times = {int(key): float(value) for key, value in cpu_times.items()}
            parsed_cuda_times = {int(key): float(value) for key, value in cuda_times.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("measured backend timings are invalid") from error
        expected_cpu = completion_balanced_cpu_matches(
            counts,
            parsed_cpu_times,
            parsed_cuda_times,
            cpu_worker_count=int(backend_policy["cpuWorkerCount"]),
            cuda_worker_count=int(backend_policy["cudaWorkerCount"]),
        )
        if cpu_allocation != expected_cpu or cpu_times != {
            str(player): parsed_cpu_times[player] for player in range(4, 11)
        } or cuda_times != {
            str(player): parsed_cuda_times[player] for player in range(4, 11)
        }:
            raise ValueError("measured backend allocation does not recompute")
    elif allocation_contract == "explicit-per-stratum-v1":
        if cpu_size is not None or cpu_times is not None or cuda_times is not None:
            raise ValueError("explicit backend allocation contains unexpected timing policy")
    elif allocation_contract == "fixed-cpu-cap-per-stratum-v1":
        if cpu_size is None or cpu_times is not None or cuda_times is not None:
            raise ValueError("fixed backend allocation policy is incomplete")
        expected_cpu = {
            player: min(int(cpu_size), counts[player] - 1)
            for player in range(4, 11)
        }
        if cpu_allocation != expected_cpu:
            raise ValueError("fixed backend allocation does not recompute")
    else:
        raise ValueError("collection plan backend allocation contract is invalid")
    expected_shards = allocate_mixed_backend_shards(
        counts,
        cpu_matches_per_player_count=(
            int(cpu_size) if cpu_size is not None else DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT
        ),
        cpu_matches_by_player_count=cpu_allocation,
        max_matches_per_shard=int(max_size),
    )
    if shards != expected_shards:
        raise ValueError("collection plan shard ranges contain a gap, overlap, or backend drift")
    targets = value["targets"]
    if not isinstance(targets, Mapping) or set(targets) != {
        "actualStratumRelativeTolerance", "defaultMatches", "estimatedNonforcedDecisions", "maximumNonforcedDecisions",
        "estimatedNonforcedByPlayerCount", "minimumNonforcedDecisions", "preflight",
        "resolvedMatches", "stratumTargetNonforcedDecisions", "targetNonforcedDecisions",
    } or targets["resolvedMatches"] != value["totalMatches"]:
        raise ValueError("collection plan target sizing record is invalid")
    for name in (
        "defaultMatches", "maximumNonforcedDecisions", "minimumNonforcedDecisions",
        "resolvedMatches", "targetNonforcedDecisions",
    ):
        raw = targets[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError("collection plan target sizing integer is invalid")
    tolerance = targets["actualStratumRelativeTolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0.0 < float(tolerance) <= 0.25
    ):
        raise ValueError("collection plan actual stratum tolerance is invalid")
    estimated = targets["estimatedNonforcedDecisions"]
    if estimated is not None and (
        isinstance(estimated, bool) or not isinstance(estimated, int) or estimated < 1
    ):
        raise ValueError("collection plan estimated decision count is invalid")
    if estimated is not None and not (
        int(targets["minimumNonforcedDecisions"])
        <= estimated
        <= int(targets["maximumNonforcedDecisions"])
    ):
        raise ValueError("collection plan estimate is outside the sealed corpus range")
    preflight = targets["preflight"]
    if preflight is not None:
        if isinstance(preflight, Mapping) and preflight.get("kind") == "stratified":
            if set(preflight) != {"kind", "strata"} or not isinstance(
                preflight["strata"], Mapping
            ) or set(preflight["strata"]) != {str(p) for p in range(4, 11)}:
                raise ValueError("collection plan stratified preflight record is invalid")
            estimated_by_p = targets["estimatedNonforcedByPlayerCount"]
            if not isinstance(estimated_by_p, Mapping) or set(estimated_by_p) != {
                str(p) for p in range(4, 11)
            }:
                raise ValueError("collection plan per-stratum estimates are missing")
            recomputed_estimates: dict[str, int] = {}
            for player_count in range(4, 11):
                record = preflight["strata"][str(player_count)]
                if not isinstance(record, Mapping) or set(record) != {
                    "matches", "nonforcedDecisions", "nonforcedDecisionsPerMatch"
                }:
                    raise ValueError("collection plan preflight stratum is invalid")
                matches = record["matches"]
                nonforced = record["nonforcedDecisions"]
                rate = record["nonforcedDecisionsPerMatch"]
                if (
                    isinstance(matches, bool)
                    or not isinstance(matches, int)
                    or matches < 1
                    or isinstance(nonforced, bool)
                    or not isinstance(nonforced, int)
                    or nonforced < matches
                    or not isinstance(rate, (int, float))
                    or float(rate) != nonforced / matches
                ):
                    raise ValueError("collection plan preflight stratum counts drifted")
                if (
                    value.get("purpose") == "production"
                    and matches < MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM
                ):
                    raise ValueError(
                        "production collection plan preflight is statistically undersampled"
                    )
                recomputed_estimates[str(player_count)] = int(
                    round(counts[player_count] * float(rate))
                )
            if dict(estimated_by_p) != recomputed_estimates or estimated != sum(
                recomputed_estimates.values()
            ):
                raise ValueError("collection plan per-stratum decision estimates drifted")
            # Integer rounding may differ by at most roughly one match from
            # the common useful-decision target in each stratum.
            stratum_target = targets["stratumTargetNonforcedDecisions"]
            if not isinstance(stratum_target, (int, float)) or stratum_target <= 0:
                raise ValueError("collection plan per-stratum target is invalid")
            maximum_rate = max(
                float(preflight["strata"][str(p)]["nonforcedDecisionsPerMatch"])  # type: ignore[index]
                for p in range(4, 11)
            )
            if max(abs(value - float(stratum_target)) for value in recomputed_estimates.values()) > maximum_rate:
                raise ValueError("collection plan did not equalize expected useful decisions")
        else:
            if not isinstance(preflight, Mapping) or set(preflight) != {
                "matches", "nonforcedDecisions", "nonforcedDecisionsPerMatch"
            }:
                raise ValueError("collection plan aggregate preflight record is invalid")
            if any(
                isinstance(preflight[name], bool)
                or not isinstance(preflight[name], int)
                or preflight[name] < 1
                for name in ("matches", "nonforcedDecisions")
            ) or not isinstance(preflight["nonforcedDecisionsPerMatch"], (int, float)):
                raise ValueError("collection plan preflight counts are invalid")
            expected_rate = preflight["nonforcedDecisions"] / preflight["matches"]
            if float(preflight["nonforcedDecisionsPerMatch"]) != expected_rate:
                raise ValueError("collection plan preflight rate drifted")
            if targets["estimatedNonforcedByPlayerCount"] is not None or targets[
                "stratumTargetNonforcedDecisions"
            ] is not None:
                raise ValueError("aggregate preflight cannot claim per-stratum balance")
    elif targets["estimatedNonforcedByPlayerCount"] is not None or targets[
        "stratumTargetNonforcedDecisions"
    ] is not None:
        raise ValueError("unmeasured plan cannot claim per-stratum estimates")
    if value["purpose"] == "production":
        if (
            not isinstance(preflight, Mapping)
            or preflight.get("kind") != "stratified"
            or targets["stratumTargetNonforcedDecisions"] is None
        ):
            raise ValueError(
                "production collection requires measured p4..p10 stratified preflight"
            )
    elif isinstance(preflight, Mapping) and preflight.get("kind") == "stratified":
        raise ValueError("diagnostic-unbalanced plan cannot claim stratified production sizing")
    document = json.loads(canonical_json_bytes(dict(value)))
    digest = sha256_bytes(canonical_json_bytes(document))
    return V5CollectionPlan(document, digest, shards)


def publish_collection_plan(target: str | Path, plan: V5CollectionPlan) -> str:
    target_path = Path(target).resolve()
    raw = canonical_json_bytes(dict(plan.document))
    if sha256_bytes(raw) != plan.manifest_sha256:
        raise ValueError("collection plan in-memory checksum drifted")
    _exclusive_publish_directory(
        target_path,
        {
            "plan.json": raw,
            "plan.json.sha256": f"{plan.manifest_sha256}  plan.json\n".encode("ascii"),
        },
    )
    return plan.manifest_sha256


def load_collection_plan(target: str | Path) -> V5CollectionPlan:
    root = Path(target).resolve()
    if not root.is_dir() or {path.name for path in root.iterdir()} != {
        "plan.json", "plan.json.sha256"
    }:
        raise ValueError("collection plan directory inventory is incomplete")
    raw = (root / "plan.json").read_bytes()
    sidecar = (root / "plan.json.sha256").read_bytes()
    digest = sha256_bytes(raw)
    if sidecar != f"{digest}  plan.json\n".encode("ascii"):
        raise ValueError("collection plan checksum sidecar does not match")
    plan = validate_collection_plan_document(_strict_canonical_json(raw, "collection plan"))
    if plan.manifest_sha256 != digest:
        raise ValueError("collection plan canonical checksum drifted")
    return plan


def expected_planned_shard_metadata(
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
    *,
    execution_backend: str | None = None,
) -> dict[str, object]:
    backend = shard.backend if execution_backend is None else execution_backend
    if backend != shard.backend:
        raise ValueError("execution backend does not match the immutable shard plan")
    calibration = plan.document["calibration"]
    assert isinstance(calibration, Mapping)
    policy = plan.document["backendPolicy"]
    assert isinstance(policy, Mapping)
    prefix = "cpu" if backend == "cpu" else "cuda"
    behavior = plan.behavior
    return {
        "behaviorModelPairId": behavior["pairId"],
        "behaviorModelPairManifestSha256": behavior["pairManifestSha256"],
        "calibrationReportSha256": calibration["reportSha256"],
        "collectionPlanFormat": V5_COLLECTION_PLAN_FORMAT,
        "collectionPlanManifestSha256": plan.manifest_sha256,
        "collectionPlanVersion": V5_COLLECTION_PLAN_VERSION,
        "executionBackend": backend,
        "executionPlannedWorkerCount": policy[f"{prefix}WorkerCount"],
        "executionTorchThreadsPerWorker": policy[
            f"{prefix}TorchThreadsPerWorker"
        ],
        "plannedMatchCount": shard.match_count,
        "plannedMatchStart": shard.match_start,
        "plannedPlayerCount": shard.player_count,
        "plannedShardIndex": shard.index,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "sourceInventory": dict(plan.source_inventory),
        "sourceInventorySha256": plan.document["sourceInventorySha256"],
    }


def planned_shard_path(shards_root: str | Path, shard: V5PlannedShard) -> Path:
    return Path(shards_root).resolve() / shard.name


def _verify_planned_shard_details(
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
    path: str | Path,
) -> dict[str, object]:
    """Verify one shard and return its exact private match-provenance facts."""

    from v5_dataset import load_v5_training_shard

    target = Path(path).resolve()
    loaded = load_v5_training_shard(target)
    try:
        manifest = loaded.actor.manifest
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("planned shard metadata is missing")
        expected_custom = expected_planned_shard_metadata(plan, shard)
        for key, expected in expected_custom.items():
            if metadata.get(key) != expected:
                raise ValueError(f"planned shard metadata binding drifted: {key}")
        behavior = plan.behavior
        expected_base = {
            "behaviorActorManifestSha256": behavior["actorManifestSha256"],
            "behaviorActorSha256": behavior["actorSha256"],
            "behaviorCriticSha256": behavior["criticSha256"],
            "behaviorModelPairId": behavior["pairId"],
            "behaviorModelPairManifestSha256": behavior["pairManifestSha256"],
            "collectionContract": V5_MAPPO_COLLECTION_CONTRACT,
            "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "rewardContract": V5_MAPPO_REWARD_CONTRACT,
            "runNamespace": plan.run_namespace,
            "seedBase": plan.seed_base,
            "matchCounts": {str(shard.player_count): shard.match_count},
            "matchStart": shard.match_start,
            "matchShardCount": 1,
            "matchShardIndex": 0,
        }
        for key, expected in expected_base.items():
            if metadata.get(key) != expected:
                raise ValueError(f"planned shard behavior/schedule binding drifted: {key}")
        counts = manifest.get("counts")
        if not isinstance(counts, Mapping) or counts.get("matches") != shard.match_count:
            raise ValueError("planned shard manifest match count drifted")
        players = np.asarray(loaded.actor.arrays["player_counts"])
        if players.shape != (shard.match_count,) or not np.all(players == shard.player_count):
            raise ValueError("planned shard contains the wrong p-stratum")
        provenance = loaded.privileged_arrays
        if not {"match_indices", "match_seeds"} <= set(provenance):
            raise ValueError("planned shard omitted private match provenance")
        match_indices = np.asarray(provenance["match_indices"])
        match_seeds = np.asarray(provenance["match_seeds"])
        expected_indices = np.arange(
            shard.match_start,
            shard.match_stop,
            dtype=np.uint32,
        )
        expected_seeds = np.asarray(
            [
                derive_v5_collection_match_seed(
                    plan.run_namespace,
                    plan.seed_base,
                    shard.player_count,
                    match_index,
                )
                for match_index in range(shard.match_start, shard.match_stop)
            ],
            dtype=np.uint32,
        )
        if not np.array_equal(match_indices, expected_indices):
            raise ValueError(
                "planned shard private match indexes do not equal its exact range"
            )
        if not np.array_equal(match_seeds, expected_seeds):
            raise ValueError(
                "planned shard private match seeds do not recompute from its plan"
            )
        raw_manifest = (target / "manifest.json").read_bytes()
        digest = sha256_bytes(raw_manifest)
        if (target / "manifest.json.sha256").read_bytes() != f"{digest}  manifest.json\n".encode("ascii"):
            raise ValueError("planned shard manifest sidecar drifted")
        decision_count = loaded.actor.decision_count
        nonforced = int(
            decision_count - loaded.actor.arrays["forced"].sum()
        )
        return {
            "decisionCount": decision_count,
            "manifestSha256": digest,
            "matchCoordinates": tuple(
                (
                    shard.player_count,
                    int(match_index),
                    int(match_seed),
                )
                for match_index, match_seed in zip(
                    match_indices,
                    match_seeds,
                    strict=True,
                )
            ),
            "matchCount": loaded.actor.match_count,
            "nonforcedDecisionCount": nonforced,
            "path": target,
        }
    finally:
        loaded.close()


def verify_planned_shard(
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
    path: str | Path,
) -> str:
    """Verify a completed immutable shard; never treat partial state as resumable."""

    return str(_verify_planned_shard_details(plan, shard, path)["manifestSha256"])


def _declared_planned_shard_index(path: Path) -> int:
    """Read only the routing key; full immutable verification follows."""

    try:
        document = json.loads((path / "manifest.json").read_bytes())
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"indexed V5 shard has no readable manifest: {path}"
        ) from error
    if not isinstance(document, Mapping):
        raise ValueError("indexed V5 shard manifest must be an object")
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("indexed V5 shard metadata is missing")
    index = metadata.get("plannedShardIndex")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("indexed V5 shard has an invalid plannedShardIndex")
    return index


def verify_planned_collection_corpus(
    plan: V5CollectionPlan,
    shards_root: str | Path,
    *,
    index_shard_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    """Recompute exact planned-match coverage, seeds, hashes, and corpus counts.

    ``match_indices`` and ``match_seeds`` stay in the private partition; this
    verifier is the production integration boundary and never feeds them to
    the public Actor.
    """

    root = Path(shards_root).resolve()
    if index_shard_paths is None:
        scheduled = tuple(
            (shard, planned_shard_path(root, shard)) for shard in plan.shards
        )
    else:
        indexed = tuple(Path(path).resolve() for path in index_shard_paths)
        if len(indexed) != len(plan.shards) or len(indexed) != len(set(indexed)):
            raise ValueError(
                "V5 production index shard inventory differs from its exact plan"
            )
        planned_by_index = {shard.index: shard for shard in plan.shards}
        if set(planned_by_index) != set(range(len(plan.shards))):
            raise ValueError("V5 collection plan shard indexes are not canonical")
        indexed_by_plan: dict[int, Path] = {}
        for path in indexed:
            planned_index = _declared_planned_shard_index(path)
            if planned_index not in planned_by_index:
                raise ValueError(
                    "V5 production index contains a foreign planned shard index"
                )
            if planned_index in indexed_by_plan:
                raise ValueError(
                    "V5 production index repeats one planned shard index"
                )
            indexed_by_plan[planned_index] = path
        if set(indexed_by_plan) != set(planned_by_index):
            raise ValueError(
                "V5 production index shard inventory differs from its exact plan"
            )
        scheduled = tuple(
            (planned_by_index[index], indexed_by_plan[index])
            for index in range(len(plan.shards))
        )

    decisions_by_player = {str(player): 0 for player in range(4, 11)}
    nonforced_by_player = {str(player): 0 for player in range(4, 11)}
    matches_by_player = {str(player): 0 for player in range(4, 11)}
    manifest_hashes: dict[str, str] = {}
    observed: list[tuple[int, int, int]] = []
    for shard, path in scheduled:
        details = _verify_planned_shard_details(plan, shard, path)
        key = str(shard.player_count)
        decisions_by_player[key] += int(details["decisionCount"])
        nonforced_by_player[key] += int(details["nonforcedDecisionCount"])
        matches_by_player[key] += int(details["matchCount"])
        manifest_hashes[str(shard.index)] = str(details["manifestSha256"])
        coordinates = details["matchCoordinates"]
        assert isinstance(coordinates, tuple)
        observed.extend(coordinates)

    expected = [
        (
            shard.player_count,
            match_index,
            derive_v5_collection_match_seed(
                plan.run_namespace,
                plan.seed_base,
                shard.player_count,
                match_index,
            ),
        )
        for shard in plan.shards
        for match_index in range(shard.match_start, shard.match_stop)
    ]
    observed_keys = [(player, match_index) for player, match_index, _ in observed]
    expected_keys = [(player, match_index) for player, match_index, _ in expected]
    observed_seeds = [seed for _, _, seed in observed]
    if (
        len(observed_keys) != len(set(observed_keys))
        or len(observed_seeds) != len(set(observed_seeds))
        or sorted(observed) != sorted(expected)
        or set(observed_keys) != set(expected_keys)
    ):
        raise ValueError(
            "V5 planned collection match provenance has a duplicate, gap, or seed drift"
        )
    coordinates_sha = sha256_bytes(
        canonical_json_bytes([list(value) for value in sorted(observed)])
    )
    return {
        "actualDecisionCountsByPlayerCount": decisions_by_player,
        "actualMatchCountsByPlayerCount": matches_by_player,
        "actualNonforcedDecisionCountsByPlayerCount": nonforced_by_player,
        "completeShardIndices": list(range(len(plan.shards))),
        "matchCoordinatesSha256": coordinates_sha,
        "matchProvenanceContract": V5_MATCH_PROVENANCE_CONTRACT,
        "shardManifestSha256s": manifest_hashes,
        "totalUniqueMatches": len(observed),
    }


def resume_verified_shard(
    plan: V5CollectionPlan,
    shard: V5PlannedShard,
    shards_root: str | Path,
) -> str | None:
    """Return a verified digest, None if absent, and raise for every partial shard."""

    target = planned_shard_path(shards_root, shard)
    if not target.exists():
        return None
    return verify_planned_shard(plan, shard, target)


def _manifest_digest(root: Path) -> str:
    raw = (root / "manifest.json").read_bytes()
    digest = sha256_bytes(raw)
    if (root / "manifest.json.sha256").read_bytes() != f"{digest}  manifest.json\n".encode("ascii"):
        raise ValueError("calibration shard manifest sidecar does not match")
    return digest


def _calibration_metadata_pair(
    cpu_metadata: Mapping[str, object], cuda_metadata: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, str]]:
    if cpu_metadata.get("calibrationBackend") != "cpu" or cuda_metadata.get(
        "calibrationBackend"
    ) != "cuda":
        raise ValueError("calibration snapshots must identify CPU and CUDA backends")
    cpu_common = dict(cpu_metadata)
    cuda_common = dict(cuda_metadata)
    del cpu_common["calibrationBackend"]
    del cuda_common["calibrationBackend"]
    if cpu_common != cuda_common:
        raise ValueError("CPU/CUDA calibration snapshots disagree on bound schedule or sources")
    if cpu_common.get("calibrationScheduleContract") != V5_CALIBRATION_SCHEDULE_CONTRACT:
        raise ValueError("calibration schedule contract is missing")
    required_contracts = {
        "collectionContract": V5_MAPPO_COLLECTION_CONTRACT,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "rewardContract": V5_MAPPO_REWARD_CONTRACT,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
    }
    for name, expected in required_contracts.items():
        if cpu_common.get(name) != expected:
            raise ValueError(f"calibration contract binding drifted: {name}")
    raw_sources = cpu_common.get("sourceInventory")
    sources = _validate_source_inventory(raw_sources)
    if cpu_common.get("sourceInventorySha256") != source_inventory_sha256(sources):
        raise ValueError("calibration source inventory fingerprint drifted")
    raw_counts = cpu_common.get("matchCounts")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != {
        str(player) for player in range(4, 11)
    }:
        raise ValueError("calibration p4..p10 schedule is missing")
    counts: dict[int, int] = {}
    for player in range(4, 11):
        raw_count = raw_counts[str(player)]
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1:
            raise ValueError("calibration match counts are invalid")
        counts[player] = raw_count
    namespace = cpu_common.get("runNamespace")
    seed_base = cpu_common.get("seedBase")
    if cpu_common.get("calibrationScheduleId") != calibration_schedule_id(
        namespace, seed_base, counts  # type: ignore[arg-type]
    ):
        raise ValueError("calibration schedule identity drifted")
    bindings = {
        "actorManifestSha256": _require_sha256(
            cpu_common.get("behaviorActorManifestSha256"), "calibration Actor manifest"
        ),
        "actorSha256": _require_sha256(
            cpu_common.get("behaviorActorSha256"), "calibration Actor"
        ),
        "criticSha256": _require_sha256(
            cpu_common.get("behaviorCriticSha256"), "calibration critic"
        ),
        "pairId": _require_sha256(
            cpu_common.get("behaviorModelPairId"), "calibration behavior pair ID"
        ),
        "pairManifestSha256": _require_sha256(
            cpu_common.get("behaviorModelPairManifestSha256"),
            "calibration behavior pair manifest",
        ),
        "policyNumericsSha256": _require_sha256(
            cpu_common.get("policyNumericsSha256"), "calibration policy numerics"
        ),
        "sourceInventorySha256": _require_sha256(
            cpu_common.get("sourceInventorySha256"), "calibration source inventory"
        ),
    }
    if bindings["policyNumericsSha256"] != V5_POLICY_NUMERICS_SHA256:
        raise ValueError("calibration policy numerics contract drifted")
    return cpu_common, bindings


def compare_calibration_shards(
    cpu_shard: str | Path,
    cuda_shard: str | Path,
) -> dict[str, object]:
    """Recompute the sealed CPU/CUDA equivalence report and fail on drift."""

    from v5_dataset import load_v5_training_shard

    cpu_root = Path(cpu_shard).resolve()
    cuda_root = Path(cuda_shard).resolve()
    cpu = load_v5_training_shard(cpu_root)
    cuda = load_v5_training_shard(cuda_root)
    try:
        cpu_meta = cpu.actor.manifest.get("metadata")
        cuda_meta = cuda.actor.manifest.get("metadata")
        if not isinstance(cpu_meta, Mapping) or not isinstance(cuda_meta, Mapping):
            raise ValueError("calibration shard metadata is missing")
        common, bindings = _calibration_metadata_pair(cpu_meta, cuda_meta)
        if cpu.actor.decision_count != cuda.actor.decision_count or cpu.actor.match_count != cuda.actor.match_count:
            raise ValueError("CPU/CUDA calibration snapshot counts differ")
        raw_counts = common["matchCounts"]
        assert isinstance(raw_counts, Mapping)
        expected_players = np.concatenate(
            [
                np.full(int(raw_counts[str(player)]), player, dtype=np.uint8)
                for player in range(4, 11)
            ]
        )
        if not np.array_equal(
            np.sort(np.asarray(cpu.actor.arrays["player_counts"])),
            np.sort(expected_players),
        ):
            raise ValueError("calibration snapshot does not contain its declared p schedule")
        cpu_arrays = {**cpu.actor.arrays, **cpu.privileged_arrays}
        cuda_arrays = {**cuda.actor.arrays, **cuda.privileged_arrays}
        if set(cpu_arrays) != set(cuda_arrays):
            raise ValueError("CPU/CUDA calibration array inventory differs")
        tolerant = {
            "old_log_probs": (CALIBRATION_LOG_PROBABILITY_ATOL, 0.0),
            "selected_action_probabilities": (CALIBRATION_LOG_PROBABILITY_ATOL, 0.0),
            "old_values": (CALIBRATION_VALUE_ATOL, CALIBRATION_VALUE_RTOL),
            "advantages": (CALIBRATION_DERIVED_VALUE_ATOL, CALIBRATION_VALUE_RTOL),
            "returns": (CALIBRATION_DERIVED_VALUE_ATOL, CALIBRATION_VALUE_RTOL),
            "deltas": (CALIBRATION_DERIVED_VALUE_ATOL, CALIBRATION_VALUE_RTOL),
            "policy_entropies": (CALIBRATION_DERIVED_VALUE_ATOL, CALIBRATION_VALUE_RTOL),
        }
        comparisons: dict[str, dict[str, object]] = {}
        exact_names: list[str] = []
        for name in sorted(cpu_arrays):
            left = np.asarray(cpu_arrays[name])
            right = np.asarray(cuda_arrays[name])
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"calibration array shape/dtype differs: {name}")
            if name in tolerant:
                atol, rtol = tolerant[name]
                difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
                max_abs = float(difference.max(initial=0.0))
                if not np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False):
                    raise ValueError(
                        f"calibration tolerance exceeded for {name}: {max_abs}"
                    )
                comparisons[name] = {
                    "atol": atol,
                    "maxAbsoluteDifference": max_abs,
                    "rtol": rtol,
                }
            else:
                if not np.array_equal(left, right):
                    raise ValueError(f"calibration structural array differs: {name}")
                exact_names.append(name)
        if "actions" not in exact_names or "old_log_probs" not in comparisons or "old_values" not in comparisons:
            raise RuntimeError("calibration omitted a required comparison")
        measurements: dict[str, dict[str, object]] = {}
        offsets = np.asarray(cpu.actor.arrays["match_offsets"], dtype=np.int64)
        players = np.asarray(cpu.actor.arrays["player_counts"], dtype=np.int64)
        forced = np.asarray(cpu.actor.arrays["forced"], dtype=np.bool_)
        for player_count in range(4, 11):
            match_indexes = np.flatnonzero(players == player_count)
            decisions = 0
            nonforced = 0
            for match_index in match_indexes:
                start = int(offsets[match_index])
                stop = int(offsets[match_index + 1])
                decisions += stop - start
                nonforced += int((~forced[start:stop]).sum())
            if not match_indexes.size or nonforced < int(match_indexes.size):
                raise ValueError("calibration produced an invalid useful-decision measurement")
            measurements[str(player_count)] = {
                "decisions": decisions,
                "matches": int(match_indexes.size),
                "nonforcedDecisions": nonforced,
                "nonforcedDecisionsPerMatch": nonforced / int(match_indexes.size),
            }
        schedule = {
            key: common[key]
            for key in (
                "calibrationScheduleContract", "matchCounts", "matchStart",
                "runNamespace", "seedBase",
            )
            if key in common
        }
        return {
            "behavior": bindings,
            "comparisons": {
                "exactArrays": exact_names,
                "tolerantArrays": comparisons,
            },
            "cpu": {"backend": "cpu", "manifestSha256": _manifest_digest(cpu_root)},
            "cuda": {"backend": "cuda", "manifestSha256": _manifest_digest(cuda_root)},
            "format": V5_CALIBRATION_REPORT_FORMAT,
            "measurements": measurements,
            "passed": True,
            "schedule": schedule,
            "tolerances": {
                "derivedValueAbsolute": CALIBRATION_DERIVED_VALUE_ATOL,
                "logProbabilityAbsolute": CALIBRATION_LOG_PROBABILITY_ATOL,
                "valueAbsolute": CALIBRATION_VALUE_ATOL,
                "valueRelative": CALIBRATION_VALUE_RTOL,
            },
            "version": V5_CALIBRATION_REPORT_VERSION,
        }
    finally:
        cpu.close()
        cuda.close()


def publish_calibration_report(
    target: str | Path,
    cpu_shard: str | Path,
    cuda_shard: str | Path,
) -> str:
    report = compare_calibration_shards(cpu_shard, cuda_shard)
    raw = canonical_json_bytes(report)
    digest = sha256_bytes(raw)
    _exclusive_publish_directory(
        Path(target).resolve(),
        {
            "report.json": raw,
            "report.json.sha256": f"{digest}  report.json\n".encode("ascii"),
        },
    )
    return digest


def load_verified_calibration_report(
    target: str | Path,
    cpu_shard: str | Path,
    cuda_shard: str | Path,
) -> tuple[dict[str, object], str]:
    root = Path(target).resolve()
    if not root.is_dir() or {path.name for path in root.iterdir()} != {
        "report.json", "report.json.sha256"
    }:
        raise ValueError("calibration report directory inventory is incomplete")
    raw = (root / "report.json").read_bytes()
    digest = sha256_bytes(raw)
    if (root / "report.json.sha256").read_bytes() != f"{digest}  report.json\n".encode("ascii"):
        raise ValueError("calibration report checksum sidecar does not match")
    stored = _strict_canonical_json(raw, "calibration report")
    recomputed = compare_calibration_shards(cpu_shard, cuda_shard)
    if stored != recomputed:
        raise ValueError("calibration report no longer matches its immutable snapshots")
    return stored, digest


def validate_actual_nonforced_corpus(
    plan: V5CollectionPlan,
    nonforced_by_player_count: Mapping[str, int] | Mapping[int, int],
) -> dict[str, object]:
    """Enforce the sealed total-volume and equal-stratum useful-row gates."""

    if plan.purpose != "production":
        raise ValueError("only a production collection plan can pass the corpus gate")

    parsed: dict[int, int] = {}
    for raw_key, raw_value in nonforced_by_player_count.items():
        try:
            player = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError("actual nonforced p-stratum key is invalid") from error
        if (
            player not in range(4, 11)
            or player in parsed
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 1
        ):
            raise ValueError("actual nonforced p-stratum counts are invalid")
        parsed[player] = raw_value
    if set(parsed) != set(range(4, 11)):
        raise ValueError("actual nonforced counts must cover every p4..p10 stratum")
    targets = plan.document["targets"]
    assert isinstance(targets, Mapping)
    total = sum(parsed.values())
    minimum = int(targets["minimumNonforcedDecisions"])
    maximum = int(targets["maximumNonforcedDecisions"])
    if total < minimum:
        raise ValueError(
            f"actual nonforced corpus is under target: {total} < {minimum}"
        )
    if total > maximum:
        raise ValueError(
            f"actual nonforced corpus is over target: {total} > {maximum}"
        )
    target = targets["stratumTargetNonforcedDecisions"]
    if not isinstance(target, (int, float)) or float(target) <= 0.0:
        raise ValueError("production corpus plan lacks a measured equal-stratum target")
    tolerance = float(targets["actualStratumRelativeTolerance"])
    lower = float(target) * (1.0 - tolerance)
    upper = float(target) * (1.0 + tolerance)
    failures = [
        player for player in range(4, 11)
        if not lower <= parsed[player] <= upper
    ]
    if failures:
        player = failures[0]
        raise ValueError(
            "actual nonforced corpus violates equal-stratum tolerance: "
            f"p{player}={parsed[player]}, allowed=[{lower:.3f},{upper:.3f}]"
        )
    return {
        "allowedPerStratum": {"lower": lower, "upper": upper},
        "nonforcedByPlayerCount": {
            str(player): parsed[player] for player in range(4, 11)
        },
        "passed": True,
        "relativeTolerance": tolerance,
        "targetPerStratum": float(target),
        "total": total,
        "totalRange": {"maximum": maximum, "minimum": minimum},
    }


def publish_verified_index(
    plan: V5CollectionPlan,
    shards_root: str | Path,
    target: str | Path,
) -> str:
    """Verify complete plan coverage, then publish one zero-copy V5 index."""

    if plan.purpose != "production":
        raise ValueError("diagnostic-unbalanced plans cannot publish a production index")

    from v5_dataset import (
        load_v5_index_manifest,
        publish_v5_index_manifest,
    )

    paths = [planned_shard_path(shards_root, shard) for shard in plan.shards]
    verified_corpus = verify_planned_collection_corpus(plan, shards_root)
    decisions_by_player = verified_corpus["actualDecisionCountsByPlayerCount"]
    nonforced_by_player = verified_corpus[
        "actualNonforcedDecisionCountsByPlayerCount"
    ]
    matches_by_player = verified_corpus["actualMatchCountsByPlayerCount"]
    manifest_hashes = verified_corpus["shardManifestSha256s"]
    assert isinstance(nonforced_by_player, Mapping)
    metadata = {
        "actualDecisionCountsByPlayerCount": decisions_by_player,
        "actualMatchCountsByPlayerCount": matches_by_player,
        "actualNonforcedDecisionCountsByPlayerCount": nonforced_by_player,
        "behavior": dict(plan.behavior),
        "calibrationReportSha256": plan.document["calibration"]["reportSha256"],  # type: ignore[index]
        "collectionPlanManifestSha256": plan.manifest_sha256,
        "completeShardIndices": verified_corpus["completeShardIndices"],
        "matchCoordinatesSha256": verified_corpus["matchCoordinatesSha256"],
        "matchProvenanceContract": verified_corpus[
            "matchProvenanceContract"
        ],
        "plannedMatchCountsByPlayerCount": plan.document["matchCounts"],
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "shardManifestSha256s": manifest_hashes,
        "sourceInventorySha256": plan.document["sourceInventorySha256"],
        "totalUniqueMatches": verified_corpus["totalUniqueMatches"],
    }
    corpus_gate = validate_actual_nonforced_corpus(plan, nonforced_by_player)
    metadata["actualCorpusGate"] = corpus_gate
    digest = publish_v5_index_manifest(target, paths, metadata=metadata)
    index = load_v5_index_manifest(target)
    try:
        if index.match_count != int(plan.document["totalMatches"]):
            raise RuntimeError("published V5 index did not preserve planned match coverage")
    finally:
        index.close()
    return digest


__all__ = [
    "CALIBRATION_DERIVED_VALUE_ATOL",
    "CALIBRATION_LOG_PROBABILITY_ATOL",
    "CALIBRATION_VALUE_ATOL",
    "CALIBRATION_VALUE_RTOL",
    "DEFAULT_CPU_MATCHES_PER_PLAYER_COUNT",
    "DEFAULT_ACTUAL_STRATUM_RELATIVE_TOLERANCE",
    "DEFAULT_MAX_MATCHES_PER_SHARD",
    "DEFAULT_TARGET_NONFORCED",
    "DEFAULT_TARGET_NONFORCED_MAX",
    "DEFAULT_TARGET_NONFORCED_MIN",
    "DEFAULT_TOTAL_MATCHES",
    "MIN_PRODUCTION_PREFLIGHT_MATCHES_PER_STRATUM",
    "V5_CALIBRATION_REPORT_FORMAT",
    "V5_CALIBRATION_SCHEDULE_CONTRACT",
    "V5_COLLECTION_PLAN_FORMAT",
    "V5_COLLECTION_PLAN_VERSION",
    "V5_COLLECTION_SOURCE_FILES",
    "V5CollectionPlan",
    "V5PlannedShard",
    "allocate_mixed_backend_shards",
    "build_collection_plan",
    "build_source_inventory",
    "calibration_schedule_id",
    "canonical_json_bytes",
    "completion_balanced_cpu_matches",
    "compare_calibration_shards",
    "expected_planned_shard_metadata",
    "load_collection_plan",
    "load_verified_calibration_report",
    "planned_shard_path",
    "publish_calibration_report",
    "publish_collection_plan",
    "publish_verified_index",
    "resolve_total_match_count",
    "resolve_stratified_match_counts",
    "resume_verified_shard",
    "sha256_bytes",
    "sha256_file",
    "source_inventory_sha256",
    "validate_collection_plan_document",
    "validate_actual_nonforced_corpus",
    "verify_planned_shard",
    "verify_planned_collection_corpus",
]
