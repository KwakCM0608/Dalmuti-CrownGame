from __future__ import annotations

"""CUDA peak-memory admission test for DALMUTI V5 audit and PPO batches."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from v5_export import canonical_json_bytes, sha256_file, v5_actor_bundle_digests
from v5_model import V5_POLICY_NUMERICS_SHA256, configure_v5_policy_numerics
from v5_train import (
    _open_source,
    _selected_policy_statistics,
    _verify_behavior_bindings,
    load_v5_critic_checkpoint,
    verify_v5_model_pair,
)
from v5_export import load_v5_actor_bundle


V5_GPU_MEMORY_PREFLIGHT_FORMAT = "dalmuti-v5-gpu-memory-preflight"
V5_GPU_MEMORY_PREFLIGHT_VERSION = 1
DEFAULT_MINIMUM_FREE_BYTES = 1 * 1024**3
V5_GPU_MEMORY_ADMISSION_BINDING_FORMAT = (
    "dalmuti-v5-gpu-memory-admission-binding"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class V5GPUMemoryPreflightConfig:
    audit_batch_size: int = 64
    microbatch_size: int = 32
    gradient_accumulation: int = 1
    critic_batch_size: int = 256
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES
    maximum_reserved_fraction: float = 0.90
    warmup_iterations: int = 2
    timing_iterations: int = 7

    def __post_init__(self) -> None:
        for name in (
            "audit_batch_size",
            "microbatch_size",
            "gradient_accumulation",
            "critic_batch_size",
            "warmup_iterations",
            "timing_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.microbatch_size, self.gradient_accumulation) not in {
            (8, 4), (16, 2), (32, 1)
        }:
            raise ValueError(
                "Actor batching must be 8x4, 16x2, or 32x1 for effective 32"
            )
        if self.critic_batch_size not in {256, 512, 1024}:
            raise ValueError("critic_batch_size must be 256, 512, or 1024")
        if (
            isinstance(self.minimum_free_bytes, bool)
            or not isinstance(self.minimum_free_bytes, int)
            or self.minimum_free_bytes < 0
        ):
            raise ValueError("minimum_free_bytes must be a non-negative integer")
        if (
            isinstance(self.maximum_reserved_fraction, bool)
            or not isinstance(self.maximum_reserved_fraction, (int, float))
            or not math.isfinite(float(self.maximum_reserved_fraction))
            or not 0.0 < float(self.maximum_reserved_fraction) < 1.0
        ):
            raise ValueError("maximum_reserved_fraction must be in (0,1)")


def _write_report(path: str | Path, report: Mapping[str, object]) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(dict(report))
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


def _strict_canonical_report(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"V5 GPU memory report contains duplicate key {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"V5 GPU memory report contains non-finite {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V5 GPU memory report is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("V5 GPU memory report is not a canonical JSON object")
    digest = hashlib.sha256(raw).hexdigest()
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if path.with_name(path.name + ".sha256").read_bytes() != expected:
        raise ValueError("V5 GPU memory report checksum sidecar does not match")
    return value, digest


def load_v5_gpu_memory_preflight_report(
    path: str | Path,
) -> tuple[dict[str, object], str]:
    report, digest = _strict_canonical_report(Path(path).resolve())
    expected = {
        "behaviorBindings", "config", "datasetIdentitySha256",
        "datasetStatistics", "device", "failure", "format", "model",
        "modelPairId", "passed", "peaks", "policyNumericsSha256",
        "runtime", "timing", "version",
    }
    config_value = report.get("config")
    try:
        config = V5GPUMemoryPreflightConfig(**config_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("V5 GPU memory report configuration is invalid") from error
    device = report.get("device")
    statistics_value = report.get("datasetStatistics")
    model = report.get("model")
    bindings = report.get("behaviorBindings")
    peaks = report.get("peaks")
    timing = report.get("timing")
    if (
        set(report) != expected
        or report.get("format") != V5_GPU_MEMORY_PREFLIGHT_FORMAT
        or report.get("version") != V5_GPU_MEMORY_PREFLIGHT_VERSION
        or config_value != asdict(config)
        or type(report.get("passed")) is not bool
        or (report.get("failure") is not None and not isinstance(report.get("failure"), str))
        or not isinstance(bindings, Mapping)
        or set(bindings) != {
            "behaviorActorManifestSha256", "behaviorActorSha256",
            "behaviorCriticSha256", "behaviorModelPairId",
            "behaviorModelPairManifestSha256",
        }
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in bindings.values()
        )
        or not isinstance(model, Mapping)
        or set(model) != {
            "actorSha256", "manifestSha256", "policyNumericsSha256",
            "publicContractSha256", "tensorStateSha256",
        }
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in model.values()
        )
        or not isinstance(device, Mapping)
        or set(device) != {
            "capability", "name", "requested", "totalMemoryBytes", "type"
        }
        or device.get("type") != "cuda"
        or not isinstance(device.get("requested"), str)
        or not isinstance(statistics_value, Mapping)
        or set(statistics_value) != {
            "nonforcedDecisionCount", "totalDecisionCount"
        }
        or any(
            type(statistics_value.get(name)) is not int
            or int(statistics_value[name]) < 1
            for name in ("nonforcedDecisionCount", "totalDecisionCount")
        )
        or int(statistics_value["nonforcedDecisionCount"])
        > int(statistics_value["totalDecisionCount"])
        or not isinstance(report.get("datasetIdentitySha256"), str)
        or _SHA256.fullmatch(str(report["datasetIdentitySha256"])) is None
        or not isinstance(report.get("modelPairId"), str)
        or _SHA256.fullmatch(str(report["modelPairId"])) is None
        or report.get("policyNumericsSha256") != V5_POLICY_NUMERICS_SHA256
        or not isinstance(peaks, Mapping)
        or not isinstance(report.get("runtime"), Mapping)
        or (report.get("passed") is True and not isinstance(timing, Mapping))
    ):
        raise ValueError("V5 GPU memory report contract drifted")
    if not isinstance(device, Mapping) or not isinstance(peaks, Mapping):
        raise AssertionError("validated V5 GPU report mappings disappeared")
    phase_names = (
        "auditForward",
        "actorBackwardAndOptimizer",
        "criticBackwardAndOptimizer",
    )
    if set(peaks) != {
        *phase_names,
        "auditForwardCandidates",
        "allocatorMaximumReservedFraction",
        "minimumObservedDeviceFreeBytes",
    }:
        raise ValueError("V5 GPU memory peak inventory drifted")
    phase_values: list[Mapping[str, object]] = []
    for name in phase_names:
        phase = peaks[name]
        if (
            not isinstance(phase, Mapping)
            or set(phase) != {
                "allocatedBytes", "deviceFreeBytes", "deviceTotalBytes",
                "reservedBytes",
            }
            or any(
                type(phase.get(key)) is not int or int(phase[key]) < 0
                for key in phase
            )
            or int(phase["allocatedBytes"]) > int(phase["reservedBytes"])
            or int(phase["deviceTotalBytes"]) != device["totalMemoryBytes"]
            or int(phase["deviceFreeBytes"]) > int(phase["deviceTotalBytes"])
        ):
            raise ValueError(f"V5 GPU memory peak phase is invalid: {name}")
        phase_values.append(phase)
    candidates = peaks["auditForwardCandidates"]
    if not isinstance(candidates, Mapping) or set(candidates) != {
        "history", "legal", "history_x_legal"
    }:
        raise ValueError("V5 GPU audit candidate peak inventory drifted")
    candidate_phases: dict[str, Mapping[str, int]] = {}
    for criterion, phase in candidates.items():
        if (
            not isinstance(phase, Mapping)
            or set(phase) != {
                "allocatedBytes", "deviceFreeBytes", "deviceTotalBytes",
                "reservedBytes",
            }
            or any(type(phase.get(key)) is not int or int(phase[key]) < 0 for key in phase)
            or int(phase["allocatedBytes"]) > int(phase["reservedBytes"])
            or int(phase["deviceTotalBytes"]) != device["totalMemoryBytes"]
            or int(phase["deviceFreeBytes"]) > int(phase["deviceTotalBytes"])
        ):
            raise ValueError(f"V5 GPU audit candidate peak is invalid: {criterion}")
        candidate_phases[str(criterion)] = phase  # type: ignore[assignment]
    if dict(peaks["auditForward"]) != _aggregate_phase_peaks(candidate_phases):  # type: ignore[arg-type]
        raise ValueError("V5 GPU aggregate audit peak does not recompute")
    total = device.get("totalMemoryBytes")
    peak_reserved = max(int(phase["reservedBytes"]) for phase in phase_values)
    observed_free = min(int(phase["deviceFreeBytes"]) for phase in phase_values)
    expected_fraction = peak_reserved / int(total) if type(total) is int and total > 0 else math.inf
    recomputed_pass = (
        report.get("failure") is None
        and observed_free >= config.minimum_free_bytes
        and expected_fraction <= config.maximum_reserved_fraction
    )
    if (
        type(total) is not int
        or total < 1
        or peaks.get("minimumObservedDeviceFreeBytes") != observed_free
        or not isinstance(
            peaks.get("allocatorMaximumReservedFraction"), (int, float)
        )
        or float(peaks["allocatorMaximumReservedFraction"]) != expected_fraction
        or report.get("passed") is not recomputed_pass
    ):
        raise ValueError("V5 GPU memory admission gate does not recompute")
    if report.get("passed") is True:
        _validate_timing_report(timing, config, statistics_value)
    return report, digest


def _validate_timing_report(
    value: object,
    config: V5GPUMemoryPreflightConfig,
    dataset_statistics: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "actorOptimizerStep", "auditForward", "auditForwardCandidates",
        "criticOptimizerStep",
        "projectedEpoch", "projectionBasis", "warmupIterations",
    } or value.get("warmupIterations") != config.warmup_iterations:
        raise ValueError("V5 GPU memory timing inventory drifted")
    for name in ("actorOptimizerStep", "auditForward", "criticOptimizerStep"):
        record = value[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"iterations", "medianSeconds", "p95Seconds"}
            or record.get("iterations") != config.timing_iterations
            or any(
                not isinstance(record.get(key), (int, float))
                or isinstance(record.get(key), bool)
                or not math.isfinite(float(record[key]))
                or float(record[key]) <= 0.0
                for key in ("medianSeconds", "p95Seconds")
            )
            or float(record["p95Seconds"]) < float(record["medianSeconds"])
        ):
            raise ValueError(f"V5 GPU timing phase is invalid: {name}")
    audit_candidates = value["auditForwardCandidates"]
    if not isinstance(audit_candidates, Mapping) or set(audit_candidates) != {
        "history", "legal", "history_x_legal"
    }:
        raise ValueError("V5 GPU audit timing candidate inventory drifted")
    for criterion, record in audit_candidates.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != {"iterations", "medianSeconds", "p95Seconds"}
            or record.get("iterations") != config.timing_iterations
            or any(
                not isinstance(record.get(key), (int, float))
                or isinstance(record.get(key), bool)
                or not math.isfinite(float(record[key]))
                or float(record[key]) <= 0.0
                for key in ("medianSeconds", "p95Seconds")
            )
        ):
            raise ValueError(f"V5 GPU audit candidate timing is invalid: {criterion}")
    aggregate_audit = value["auditForward"]
    assert isinstance(aggregate_audit, Mapping)
    if (
        aggregate_audit["medianSeconds"]
        != max(float(record["medianSeconds"]) for record in audit_candidates.values())  # type: ignore[index]
        or aggregate_audit["p95Seconds"]
        != max(float(record["p95Seconds"]) for record in audit_candidates.values())  # type: ignore[index]
    ):
        raise ValueError("V5 GPU aggregate audit timing does not recompute")
    basis = value["projectionBasis"]
    if (
        not isinstance(basis, Mapping)
        or set(basis) != {
            "datasetNonforcedDecisions", "datasetTotalDecisions",
            "effectiveNonforcedDecisionsPerOptimizerStep", "scope",
        }
        or basis.get("effectiveNonforcedDecisionsPerOptimizerStep")
        != config.microbatch_size * config.gradient_accumulation
        or basis.get("datasetNonforcedDecisions")
        != dataset_statistics["nonforcedDecisionCount"]
        or basis.get("datasetTotalDecisions")
        != dataset_statistics["totalDecisionCount"]
    ):
        raise ValueError("V5 GPU timing projection basis drifted")
    projected = value["projectedEpoch"]
    if not isinstance(projected, Mapping) or set(projected) != {"1500000", "2000000"}:
        raise ValueError("V5 GPU timing projection targets drifted")
    expected_fields = {
        "actorOptimizerSteps", "auditForwardBatches", "criticOptimizerSteps",
        "medianSeconds", "nonforcedDecisions", "p95Seconds",
        "projectedTotalDecisions",
    }
    for key, expected_decisions in (("1500000", 1_500_000), ("2000000", 2_000_000)):
        record = projected[key]
        expected_total = math.ceil(
            expected_decisions
            * int(dataset_statistics["totalDecisionCount"])
            / int(dataset_statistics["nonforcedDecisionCount"])
        )
        expected_actor_steps = math.ceil(
            expected_decisions
            / (config.microbatch_size * config.gradient_accumulation)
        )
        expected_critic_steps = math.ceil(
            expected_total / config.critic_batch_size
        )
        expected_audit_batches = 2 * math.ceil(
            expected_decisions / config.audit_batch_size
        )
        expected_median = (
            expected_actor_steps
            * float(value["actorOptimizerStep"]["medianSeconds"])  # type: ignore[index]
            + expected_critic_steps
            * float(value["criticOptimizerStep"]["medianSeconds"])  # type: ignore[index]
            + expected_audit_batches
            * float(value["auditForward"]["medianSeconds"])  # type: ignore[index]
        )
        expected_p95 = (
            expected_actor_steps
            * float(value["actorOptimizerStep"]["p95Seconds"])  # type: ignore[index]
            + expected_critic_steps
            * float(value["criticOptimizerStep"]["p95Seconds"])  # type: ignore[index]
            + expected_audit_batches
            * float(value["auditForward"]["p95Seconds"])  # type: ignore[index]
        )
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_fields
            or record.get("nonforcedDecisions") != expected_decisions
            or record.get("projectedTotalDecisions") != expected_total
            or record.get("actorOptimizerSteps") != expected_actor_steps
            or record.get("criticOptimizerSteps") != expected_critic_steps
            or record.get("auditForwardBatches") != expected_audit_batches
            or record.get("medianSeconds") != expected_median
            or record.get("p95Seconds") != expected_p95
            or any(
                type(record.get(name)) is not int or int(record[name]) < 1
                for name in (
                    "actorOptimizerSteps", "auditForwardBatches",
                    "criticOptimizerSteps", "projectedTotalDecisions",
                )
            )
            or any(
                not isinstance(record.get(name), (int, float))
                or isinstance(record.get(name), bool)
                or not math.isfinite(float(record[name]))
                or float(record[name]) <= 0.0
                for name in ("medianSeconds", "p95Seconds")
            )
        ):
            raise ValueError("V5 GPU timing projection record drifted")


def verify_v5_gpu_memory_admission(
    report_path: str | Path,
    dataset_path: str | Path,
    model_pair_directory: str | Path,
    *,
    config: V5GPUMemoryPreflightConfig,
    device: str,
) -> dict[str, object]:
    """Bind a PASS report to the exact production corpus, pair, and batches."""

    report, digest = load_v5_gpu_memory_preflight_report(report_path)
    pair_root = Path(model_pair_directory).resolve()
    pair = verify_v5_model_pair(pair_root)
    actor_path = pair_root / "actor-bundle"
    critic_path = pair_root / "critic.pt"
    source = _open_source(dataset_path)
    try:
        bindings = _verify_behavior_bindings(source, actor_path, critic_path, pair)
        requested = str(torch.device(device))
        if (
            report.get("passed") is not True
            or report.get("failure") is not None
            or report.get("datasetIdentitySha256") != source.identity_sha256
            or report.get("modelPairId") != pair["pairId"]
            or report.get("model") != v5_actor_bundle_digests(actor_path)
            or report.get("behaviorBindings") != bindings
            or report.get("config") != asdict(config)
            or report.get("device", {}).get("requested") != requested  # type: ignore[union-attr]
        ):
            raise ValueError(
                "V5 GPU memory PASS report differs from production data/pair/config/device"
            )
        return {
            "config": {
                "audit_batch_size": config.audit_batch_size,
                "critic_batch_size": config.critic_batch_size,
                "gradient_accumulation": config.gradient_accumulation,
                "microbatch_size": config.microbatch_size,
            },
            "datasetIdentitySha256": source.identity_sha256,
            "device": requested,
            "format": V5_GPU_MEMORY_ADMISSION_BINDING_FORMAT,
            "modelPairId": pair["pairId"],
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "reportSha256": digest,
            "version": 1,
        }
    finally:
        source.close()


def _worst_case_indices(
    source: object,
    count: int,
    *,
    nonforced: bool,
    criterion: str = "history",
) -> tuple[object, np.ndarray]:
    if criterion not in {"history", "legal", "history_x_legal"}:
        raise ValueError("worst-case criterion must be history, legal, or history_x_legal")
    best: tuple[object, np.ndarray] | None = None
    best_score = -1
    for shard in source.shards:  # type: ignore[attr-defined]
        arrays = shard.actor.arrays
        players = np.asarray(arrays["global_codes"][:, 1], dtype=np.int64)
        mask = players == 10
        if nonforced:
            mask &= ~np.asarray(arrays["forced"], dtype=np.bool_)
        candidates = np.flatnonzero(mask)
        if candidates.size < count:
            continue
        ends = np.asarray(arrays["history_end"], dtype=np.int64)
        starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
        lengths = ends - starts
        if criterion == "history":
            row_scores = lengths
        else:
            legal_counts = np.unpackbits(
                np.asarray(arrays["legal_action_bits"], dtype=np.uint8),
                axis=1,
                bitorder="little",
            ).sum(axis=1, dtype=np.int64)
            row_scores = (
                legal_counts
                if criterion == "legal"
                else lengths * legal_counts
            )
        order = np.argsort(row_scores[candidates], kind="stable")[::-1]
        selected = candidates[order[:count]].astype(np.int64, copy=False)
        score = int(row_scores[selected].sum())
        if best is None or score > best_score:
            best = shard, selected
            best_score = score
    if best is None:
        label = "nonforced " if nonforced else ""
        raise ValueError(f"V5 memory preflight dataset lacks {count} p10 {label}rows")
    return best


def _aggregate_phase_peaks(
    phases: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    if not phases:
        raise ValueError("cannot aggregate zero CUDA phase peaks")
    totals = {int(value["deviceTotalBytes"]) for value in phases.values()}
    if len(totals) != 1:
        raise ValueError("CUDA device total memory changed across audit candidates")
    return {
        "allocatedBytes": max(int(value["allocatedBytes"]) for value in phases.values()),
        "deviceFreeBytes": min(int(value["deviceFreeBytes"]) for value in phases.values()),
        "deviceTotalBytes": totals.pop(),
        "reservedBytes": max(int(value["reservedBytes"]) for value in phases.values()),
    }


def _phase_peak(device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocatedBytes": int(torch.cuda.max_memory_allocated(device)),
        "deviceFreeBytes": int(free_bytes),
        "deviceTotalBytes": int(total_bytes),
        "reservedBytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _safe_phase_peak(device: torch.device) -> dict[str, int]:
    try:
        return _phase_peak(device)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            free_bytes = 0
            total_bytes = torch.cuda.get_device_properties(device).total_memory
        return {
            "allocatedBytes": int(torch.cuda.max_memory_allocated(device)),
            "deviceFreeBytes": int(free_bytes),
            "deviceTotalBytes": int(total_bytes),
            "reservedBytes": int(torch.cuda.max_memory_reserved(device)),
        }


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("CUDA timing samples must be finite and positive")
    ordered = sorted(float(value) for value in samples)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "iterations": len(ordered),
        "medianSeconds": float(statistics.median(ordered)),
        "p95Seconds": ordered[rank],
    }


def _cuda_timed(device: torch.device, operation: object) -> float:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    operation()  # type: ignore[operator]
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _step_preflight_optimizer(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object,
) -> None:
    """Mirror the trainer's AMP unscale, clip, finite-check, step order."""

    scaler.unscale_(optimizer)  # type: ignore[attr-defined]
    gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("V5 gradient norm became non-finite")
    scaler.step(optimizer)  # type: ignore[attr-defined]
    scaler.update()  # type: ignore[attr-defined]


def run_v5_gpu_memory_preflight(
    dataset_path: str | Path,
    model_pair_directory: str | Path,
    output: str | Path,
    *,
    config: V5GPUMemoryPreflightConfig | None = None,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Run audit64, Actor physical32, and all-row Critic batch256 admission."""

    cfg = config or V5GPUMemoryPreflightConfig()
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V5 GPU memory preflight requires an available CUDA device")
    numerics = configure_v5_policy_numerics(target_device)
    pair_root = Path(model_pair_directory).resolve()
    pair = verify_v5_model_pair(pair_root)
    actor_path = pair_root / "actor-bundle"
    critic_path = pair_root / "critic.pt"
    actor, _ = load_v5_actor_bundle(actor_path)
    critic, _ = load_v5_critic_checkpoint(critic_path)
    source = _open_source(dataset_path)
    failure: str | None = None
    audit_peak: dict[str, int] = {"allocatedBytes": 0, "reservedBytes": 0}
    actor_peak: dict[str, int] = {"allocatedBytes": 0, "reservedBytes": 0}
    critic_peak: dict[str, int] = {"allocatedBytes": 0, "reservedBytes": 0}
    audit_candidate_peaks: dict[str, dict[str, int]] = {}
    audit_candidate_timings: dict[str, dict[str, float | int]] = {}
    timing: dict[str, object] | None = None
    try:
        bindings = _verify_behavior_bindings(
            source, actor_path, critic_path, pair
        )
        audit_candidates = {
            criterion: _worst_case_indices(
                source,
                cfg.audit_batch_size,
                nonforced=True,
                criterion=criterion,
            )
            for criterion in ("history", "legal", "history_x_legal")
        }
        train_shard, train_indices = _worst_case_indices(
            source,
            cfg.microbatch_size * cfg.gradient_accumulation,
            nonforced=True,
            criterion="history_x_legal",
        )
        critic_shard, critic_indices = _worst_case_indices(
            source, cfg.critic_batch_size, nonforced=False
        )
        nonforced_decisions = sum(
            int((~np.asarray(shard.actor.arrays["forced"], dtype=np.bool_)).sum())
            for shard in source.shards
        )
        if nonforced_decisions < 1:
            raise ValueError("V5 memory preflight dataset has no nonforced decisions")
        actor = actor.to(target_device)
        critic = critic.to(target_device)
        actor.eval()
        critic.eval()
        torch.cuda.empty_cache()
        try:
            for criterion, (audit_shard, audit_indices) in audit_candidates.items():
                def audit_operation(
                    shard: object = audit_shard,
                    indices: np.ndarray = audit_indices,
                ) -> None:
                    actor.eval()
                    critic.eval()
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", enabled=False
                    ):
                        _selected_policy_statistics(
                            actor,
                            shard.actor.arrays,  # type: ignore[attr-defined]
                            indices,
                            target_device,
                        )

                for _ in range(cfg.warmup_iterations):
                    audit_operation()
                torch.cuda.synchronize(target_device)
                torch.cuda.reset_peak_memory_stats(target_device)
                audit_seconds = [
                    _cuda_timed(target_device, audit_operation)
                    for _ in range(cfg.timing_iterations)
                ]
                audit_candidate_peaks[criterion] = _phase_peak(target_device)
                audit_candidate_timings[criterion] = _timing_summary(
                    audit_seconds
                )
            audit_peak = _aggregate_phase_peaks(audit_candidate_peaks)

            actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1.0e-5)
            critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=3.0e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            arrays = train_shard.actor.arrays

            def actor_training_operation() -> None:
                actor.train()
                actor_optimizer.zero_grad(set_to_none=True)
                for accumulation_index in range(cfg.gradient_accumulation):
                    start = accumulation_index * cfg.microbatch_size
                    microbatch_indices = train_indices[
                        start : start + cfg.microbatch_size
                    ]
                    with torch.amp.autocast("cuda", enabled=True):
                        selected, entropy, auxiliary = _selected_policy_statistics(
                            actor, arrays, microbatch_indices, target_device
                        )
                        policy_loss = (
                            -selected.mean()
                            - 0.005 * entropy.mean()
                            + 0.01 * auxiliary.mean()
                        )
                    scaler.scale(
                        policy_loss / cfg.gradient_accumulation
                    ).backward()
                _step_preflight_optimizer(actor, actor_optimizer, scaler)

            for _ in range(cfg.warmup_iterations):
                actor_training_operation()
            torch.cuda.synchronize(target_device)
            torch.cuda.reset_peak_memory_stats(target_device)
            actor_seconds = [
                _cuda_timed(target_device, actor_training_operation)
                for _ in range(cfg.timing_iterations)
            ]
            actor_peak = _phase_peak(target_device)

            critic_arrays = critic_shard.actor.arrays
            critic_states = torch.from_numpy(
                np.ascontiguousarray(
                    critic_shard.privileged_arrays["privileged_states"][critic_indices]
                )
            ).to(device=target_device, dtype=torch.float32)
            critic_players = torch.from_numpy(
                np.ascontiguousarray(
                    critic_arrays["global_codes"][critic_indices, 1]
                )
            ).to(device=target_device, dtype=torch.long)
            critic_returns = torch.from_numpy(
                np.ascontiguousarray(critic_arrays["returns"][critic_indices])
            ).to(device=target_device, dtype=torch.float32)

            def critic_training_operation() -> None:
                critic.train()
                critic_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    values = critic(critic_states, critic_players)
                    value_loss = torch.nn.functional.huber_loss(
                        values.float(), critic_returns
                    )
                scaler.scale(value_loss).backward()
                _step_preflight_optimizer(critic, critic_optimizer, scaler)

            for _ in range(cfg.warmup_iterations):
                critic_training_operation()
            torch.cuda.synchronize(target_device)
            torch.cuda.reset_peak_memory_stats(target_device)
            critic_seconds = [
                _cuda_timed(target_device, critic_training_operation)
                for _ in range(cfg.timing_iterations)
            ]
            critic_peak = _phase_peak(target_device)
            audit_timing = {
                "iterations": cfg.timing_iterations,
                "medianSeconds": max(
                    float(value["medianSeconds"])
                    for value in audit_candidate_timings.values()
                ),
                "p95Seconds": max(
                    float(value["p95Seconds"])
                    for value in audit_candidate_timings.values()
                ),
            }
            actor_timing = _timing_summary(actor_seconds)
            critic_timing = _timing_summary(critic_seconds)
            effective_batch = cfg.microbatch_size * cfg.gradient_accumulation
            projected: dict[str, object] = {}
            for decisions in (1_500_000, 2_000_000):
                total_decisions = math.ceil(
                    decisions * source.decision_count / nonforced_decisions
                )
                actor_steps = math.ceil(decisions / effective_batch)
                critic_steps = math.ceil(total_decisions / cfg.critic_batch_size)
                audit_steps = 2 * math.ceil(
                    decisions / cfg.audit_batch_size
                )
                projected[str(decisions)] = {
                    "nonforcedDecisions": decisions,
                    "projectedTotalDecisions": total_decisions,
                    "actorOptimizerSteps": actor_steps,
                    "criticOptimizerSteps": critic_steps,
                    "auditForwardBatches": audit_steps,
                    "medianSeconds": (
                        actor_steps * float(actor_timing["medianSeconds"])
                        + critic_steps * float(critic_timing["medianSeconds"])
                        + audit_steps * float(audit_timing["medianSeconds"])
                    ),
                    "p95Seconds": (
                        actor_steps * float(actor_timing["p95Seconds"])
                        + critic_steps * float(critic_timing["p95Seconds"])
                        + audit_steps * float(audit_timing["p95Seconds"])
                    ),
                }
            timing = {
                "auditForward": audit_timing,
                "auditForwardCandidates": audit_candidate_timings,
                "projectionBasis": {
                    "effectiveNonforcedDecisionsPerOptimizerStep": effective_batch,
                    "datasetNonforcedDecisions": nonforced_decisions,
                    "datasetTotalDecisions": source.decision_count,
                    "scope": (
                        "GPU Actor/Critic compute for one PPO epoch plus initial "
                        "and post-epoch nonforced Actor audits; excludes forced "
                        "semantic vector scans and dataset IO"
                    ),
                },
                "projectedEpoch": projected,
                "actorOptimizerStep": actor_timing,
                "criticOptimizerStep": critic_timing,
                "warmupIterations": cfg.warmup_iterations,
            }
        except torch.cuda.OutOfMemoryError as error:
            failure = f"CUDA out of memory: {error}"
            observed = _safe_phase_peak(target_device)
            for criterion in ("history", "legal", "history_x_legal"):
                audit_candidate_peaks.setdefault(criterion, dict(observed))
            audit_peak = _aggregate_phase_peaks(audit_candidate_peaks)
            actor_peak = _safe_phase_peak(target_device)
            critic_peak = _safe_phase_peak(target_device)

        properties = torch.cuda.get_device_properties(target_device)
        total = int(properties.total_memory)
        peak_reserved = max(
            audit_peak["reservedBytes"],
            actor_peak["reservedBytes"],
            critic_peak["reservedBytes"],
        )
        minimum_observed_free = min(
            audit_peak["deviceFreeBytes"],
            actor_peak["deviceFreeBytes"],
            critic_peak["deviceFreeBytes"],
        )
        fraction = peak_reserved / total
        passed = (
            failure is None
            and minimum_observed_free >= cfg.minimum_free_bytes
            and fraction <= cfg.maximum_reserved_fraction
        )
        if failure is None and not passed:
            failure = (
                "insufficient CUDA headroom: "
                f"observedFree={minimum_observed_free}, "
                f"reservedFraction={fraction:.6f}"
            )
        report: dict[str, object] = {
            "behaviorBindings": bindings,
            "config": asdict(cfg),
            "datasetIdentitySha256": source.identity_sha256,
            "datasetStatistics": {
                "nonforcedDecisionCount": nonforced_decisions,
                "totalDecisionCount": source.decision_count,
            },
            "device": {
                "capability": list(torch.cuda.get_device_capability(target_device)),
                "name": str(properties.name),
                "requested": str(target_device),
                "totalMemoryBytes": total,
                "type": target_device.type,
            },
            "failure": failure,
            "format": V5_GPU_MEMORY_PREFLIGHT_FORMAT,
            "model": v5_actor_bundle_digests(actor_path),
            "modelPairId": pair["pairId"],
            "passed": passed,
            "peaks": {
                "auditForward": audit_peak,
                "auditForwardCandidates": audit_candidate_peaks,
                "actorBackwardAndOptimizer": actor_peak,
                "criticBackwardAndOptimizer": critic_peak,
                "allocatorMaximumReservedFraction": fraction,
                "minimumObservedDeviceFreeBytes": minimum_observed_free,
            },
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "runtime": {
                "cuda": str(torch.version.cuda),
                "numerics": numerics,
                "torch": str(torch.__version__),
            },
            "timing": timing,
            "version": V5_GPU_MEMORY_PREFLIGHT_VERSION,
        }
        digest = _write_report(output, report)
        result = {
            "output": str(Path(output).resolve()),
            "passed": passed,
            "report": report,
            "reportSha256": digest,
        }
        if not passed:
            raise RuntimeError(
                f"V5 GPU memory preflight failed; preserved report: {result['output']}"
            )
        return result
    finally:
        source.close()
        try:
            del actor
            del critic
        finally:
            torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-pair", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-batch-size", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--critic-batch-size", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timing-iterations", type=int, default=7)
    parser.add_argument("--minimum-free-bytes", type=int, default=DEFAULT_MINIMUM_FREE_BYTES)
    parser.add_argument("--maximum-reserved-fraction", type=float, default=0.90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_v5_gpu_memory_preflight(
        arguments.dataset,
        arguments.model_pair,
        arguments.output,
        device=arguments.device,
        config=V5GPUMemoryPreflightConfig(
            audit_batch_size=arguments.audit_batch_size,
            microbatch_size=arguments.microbatch_size,
            gradient_accumulation=arguments.gradient_accumulation,
            critic_batch_size=arguments.critic_batch_size,
            warmup_iterations=arguments.warmup_iterations,
            timing_iterations=arguments.timing_iterations,
            minimum_free_bytes=arguments.minimum_free_bytes,
            maximum_reserved_fraction=arguments.maximum_reserved_fraction,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MINIMUM_FREE_BYTES",
    "V5GPUMemoryPreflightConfig",
    "V5_GPU_MEMORY_ADMISSION_BINDING_FORMAT",
    "V5_GPU_MEMORY_PREFLIGHT_FORMAT",
    "load_v5_gpu_memory_preflight_report",
    "main",
    "run_v5_gpu_memory_preflight",
    "verify_v5_gpu_memory_admission",
]
