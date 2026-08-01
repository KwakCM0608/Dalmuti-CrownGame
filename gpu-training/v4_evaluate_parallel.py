from __future__ import annotations

"""Safe whole-player-count subprocess parallelism for the V4 evaluator.

Each shard owns complete player-count match clusters.  The parent accepts only
immutable canonical shards that exactly cover p4 through p10 once, then calls
``v4_evaluate.assemble_benchmark_report``.  Consequently parallel execution
does not add fields to, or otherwise perturb, the canonical benchmark report.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

import v4_evaluate as evaluator
from v4_evaluate import (
    ACTS_PER_MATCH,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    EVALUATION_MODES,
    PLAYER_COUNTS,
    SCREENING_MATCH_COUNTS,
    CandidatePolicyRouting,
    EnvironmentAdapter,
    EvaluationBindings,
    EvaluationSeedSchedule,
    assemble_benchmark_report,
    candidate_policy_report_metadata,
    canonical_json_bytes,
    evaluate_player_count,
    resolve_cli_evaluation_bindings,
    validate_evaluation_plan,
    write_report_exclusive,
)


SHARD_FORMAT = "dalmuti-v4-player-count-evaluation-shard"
SHARD_VERSION = 1
MAX_SUBPROCESS_WORKERS = 4

_SHARD_FIELDS = {
    "format",
    "version",
    "evaluationMode",
    "bindings",
    "actualFilesVerified",
    "seedFamily",
    "playerCounts",
    "matchCountsByPlayerCount",
    "actsPerMatch",
    "bootstrapResamples",
    "batchSize",
    "candidateBatchedForward",
    "candidatePolicy",
    "promotionThresholds",
    "finalReservationSha256",
    "results",
    "deploymentTriggered",
}


def _expected_match_counts(mode: str) -> dict[int, int]:
    return evaluator._expected_match_counts(mode)


def _expected_gates(mode: str) -> dict[str, float]:
    if mode not in EVALUATION_MODES:
        raise ValueError(f"mode must be one of {EVALUATION_MODES}")
    return evaluator._expected_gates(mode)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _reservation_sha256(
    mode: str, reservation: Mapping[str, object] | None
) -> str | None:
    if mode == "final":
        if not isinstance(reservation, Mapping):
            raise ValueError("final evaluation requires an atomic final seed reservation")
        return hashlib.sha256(canonical_json_bytes(reservation)).hexdigest()
    if reservation is not None:
        raise ValueError("only final evaluation may receive a final reservation")
    return None


def _binding_from_values(
    value: object, *, actual_files_verified: object
) -> EvaluationBindings:
    if not isinstance(value, Mapping) or set(value) != {
        "artifactSha256",
        "modelSha256",
        "observationSchemaSha256",
        "normalBaselineSha256",
        "normalBaselineSourceCommit",
    }:
        raise ValueError("evaluation shard bindings are invalid")
    if not isinstance(actual_files_verified, bool):
        raise ValueError("evaluation shard file verification flag is invalid")
    return EvaluationBindings(
        artifact_sha256=value["artifactSha256"],  # type: ignore[arg-type]
        actor_sha256=value["modelSha256"],  # type: ignore[arg-type]
        observation_contract_sha256=value["observationSchemaSha256"],  # type: ignore[arg-type]
        normal_baseline_sha256=value["normalBaselineSha256"],  # type: ignore[arg-type]
        normal_baseline_source_commit=value["normalBaselineSourceCommit"],  # type: ignore[arg-type]
        actual_files_verified=actual_files_verified,
    )


def _validate_player_counts(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("evaluation shard player counts must be a non-empty list")
    counts: list[int] = []
    for player_count in value:
        if (
            isinstance(player_count, bool)
            or not isinstance(player_count, int)
            or player_count not in PLAYER_COUNTS
        ):
            raise ValueError("evaluation shard player count must be from 4 through 10")
        counts.append(player_count)
    if counts != sorted(set(counts)):
        raise ValueError("evaluation shard player counts must be unique and ascending")
    return tuple(counts)


def validate_parallel_evaluation_plan(
    *,
    mode: str,
    seed_schedule: EvaluationSeedSchedule,
    bindings: EvaluationBindings,
    final_seed_reservation: Mapping[str, object] | None,
) -> None:
    """Validate exact mode presets and the sealed-final binding contract."""

    validate_evaluation_plan(
        mode=mode,
        match_counts=_expected_match_counts(mode),
        acts=ACTS_PER_MATCH,
        gates=_expected_gates(mode),
        seed_schedule=seed_schedule,
        final_seed_reservation=final_seed_reservation,
    )
    evaluator._validate_final_reservation_bindings(
        mode=mode,
        bindings=bindings,
        final_seed_reservation=final_seed_reservation,
    )
    _reservation_sha256(mode, final_seed_reservation)


def partition_player_counts(
    workers: int, *, match_counts: Mapping[int, int] | None = None
) -> tuple[tuple[int, ...], ...]:
    """Deterministically balance whole player counts across at most four workers."""

    workers = _positive_int(workers, "workers")
    if workers > MAX_SUBPROCESS_WORKERS:
        raise ValueError(f"workers must not exceed {MAX_SUBPROCESS_WORKERS}")
    counts = dict(match_counts or SCREENING_MATCH_COUNTS)
    if set(counts) != set(PLAYER_COUNTS):
        raise ValueError("partition match counts must cover p4 through p10 exactly")
    for player_count in PLAYER_COUNTS:
        _positive_int(counts[player_count], f"p{player_count} match count")

    bin_count = min(workers, len(PLAYER_COUNTS))
    bins: list[list[int]] = [[] for _ in range(bin_count)]
    loads = [0] * bin_count
    weighted = sorted(
        PLAYER_COUNTS,
        key=lambda player_count: (
            -(int(counts[player_count]) * player_count),
            -player_count,
        ),
    )
    for player_count in weighted:
        target = min(range(bin_count), key=lambda index: (loads[index], index))
        bins[target].append(player_count)
        loads[target] += int(counts[player_count]) * player_count
    return tuple(tuple(sorted(values)) for values in bins)


def build_evaluation_shard(
    *,
    mode: str,
    seed_schedule: EvaluationSeedSchedule,
    bindings: EvaluationBindings,
    player_counts: Sequence[int],
    results: Sequence[Mapping[str, object]],
    candidate_policy_metadata: Mapping[str, object],
    candidate_batched_forward: bool,
    bootstrap_resamples: int,
    batch_size: int,
    final_seed_reservation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    validate_parallel_evaluation_plan(
        mode=mode,
        seed_schedule=seed_schedule,
        bindings=bindings,
        final_seed_reservation=final_seed_reservation,
    )
    selected = _validate_player_counts(list(player_counts))
    _positive_int(bootstrap_resamples, "bootstrap resamples")
    _positive_int(batch_size, "batch size")
    if not isinstance(candidate_batched_forward, bool):
        raise ValueError("candidate_batched_forward must be boolean")
    if not isinstance(candidate_policy_metadata, Mapping):
        raise ValueError("candidate policy metadata must be an object")

    counts = _expected_match_counts(mode)
    all_ranges = {
        int(value["playerCount"]): value for value in seed_schedule.ranges(counts)
    }
    shard: dict[str, object] = {
        "format": SHARD_FORMAT,
        "version": SHARD_VERSION,
        "evaluationMode": mode,
        "bindings": bindings.report_value(),
        "actualFilesVerified": bindings.actual_files_verified,
        "seedFamily": {
            "id": seed_schedule.family_id,
            "mode": seed_schedule.mode,
            "baseSeed": seed_schedule.base_seed,
            "ranges": [all_ranges[value] for value in selected],
        },
        "playerCounts": list(selected),
        "matchCountsByPlayerCount": {
            str(value): counts[value] for value in selected
        },
        "actsPerMatch": ACTS_PER_MATCH,
        "bootstrapResamples": bootstrap_resamples,
        "batchSize": batch_size,
        "candidateBatchedForward": candidate_batched_forward,
        "candidatePolicy": dict(candidate_policy_metadata),
        "promotionThresholds": _expected_gates(mode),
        "finalReservationSha256": _reservation_sha256(
            mode, final_seed_reservation
        ),
        "results": [dict(value) for value in results],
        "deploymentTriggered": False,
    }
    validate_evaluation_shard(shard)
    return shard


def validate_evaluation_shard(
    shard: Mapping[str, object],
    *,
    expected_mode: str | None = None,
    expected_seed_schedule: EvaluationSeedSchedule | None = None,
    expected_bindings: EvaluationBindings | None = None,
    expected_candidate_policy_metadata: Mapping[str, object] | None = None,
    expected_candidate_batched_forward: bool | None = None,
    expected_bootstrap_resamples: int | None = None,
    expected_batch_size: int | None = None,
    expected_final_seed_reservation: Mapping[str, object] | None = None,
) -> None:
    if set(shard) != _SHARD_FIELDS:
        raise ValueError("evaluation shard fields drifted")
    if shard.get("format") != SHARD_FORMAT or shard.get("version") != SHARD_VERSION:
        raise ValueError("unsupported evaluation shard")
    mode = shard.get("evaluationMode")
    if not isinstance(mode, str) or mode not in EVALUATION_MODES:
        raise ValueError("evaluation shard mode is invalid")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError("evaluation shard mode does not match")
    if shard.get("actsPerMatch") != ACTS_PER_MATCH:
        raise ValueError("evaluation shard must use five acts")
    bootstrap_resamples = _positive_int(
        shard.get("bootstrapResamples"), "bootstrap resamples"
    )
    batch_size = _positive_int(shard.get("batchSize"), "batch size")
    if (
        expected_bootstrap_resamples is not None
        and bootstrap_resamples != expected_bootstrap_resamples
    ):
        raise ValueError("evaluation shard bootstrap resamples do not match")
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError("evaluation shard batch size does not match")
    candidate_batched_forward = shard.get("candidateBatchedForward")
    if not isinstance(candidate_batched_forward, bool):
        raise ValueError("evaluation shard batched-forward audit is invalid")
    if (
        expected_candidate_batched_forward is not None
        and candidate_batched_forward is not expected_candidate_batched_forward
    ):
        raise ValueError("evaluation shard batched-forward audit does not match")

    bindings = _binding_from_values(
        shard.get("bindings"),
        actual_files_verified=shard.get("actualFilesVerified"),
    )
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError("evaluation shard artifact bindings do not match")

    selected = _validate_player_counts(shard.get("playerCounts"))
    expected_counts = _expected_match_counts(mode)
    shard_counts = shard.get("matchCountsByPlayerCount")
    if not isinstance(shard_counts, Mapping) or dict(shard_counts) != {
        str(value): expected_counts[value] for value in selected
    }:
        raise ValueError(f"evaluation shard does not use exact {mode} match counts")
    if shard.get("promotionThresholds") != _expected_gates(mode):
        raise ValueError(f"evaluation shard does not use exact {mode} gates")

    seed_family = shard.get("seedFamily")
    if not isinstance(seed_family, Mapping) or set(seed_family) != {
        "id",
        "mode",
        "baseSeed",
        "ranges",
    }:
        raise ValueError("evaluation shard seed family is invalid")
    try:
        schedule = EvaluationSeedSchedule(
            mode,
            seed_family["id"],  # type: ignore[arg-type]
            seed_family["baseSeed"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation shard seed family is invalid") from error
    all_ranges = {
        int(value["playerCount"]): value
        for value in schedule.ranges(expected_counts)
    }
    if (
        seed_family.get("mode") != mode
        or canonical_json_bytes(seed_family.get("ranges"))
        != canonical_json_bytes([all_ranges[value] for value in selected])
    ):
        raise ValueError("evaluation shard seed ranges do not match")
    if expected_seed_schedule is not None and schedule != expected_seed_schedule:
        raise ValueError("evaluation shard seed schedule does not match")

    reservation_digest = shard.get("finalReservationSha256")
    if mode == "final":
        try:
            evaluator._require_sha256(
                reservation_digest, "final evaluation shard reservation SHA-256"
            )
        except (TypeError, ValueError) as error:
            raise ValueError("final evaluation shard reservation digest is invalid")
    elif reservation_digest is not None:
        raise ValueError("non-final evaluation shard declares a final reservation")
    if expected_final_seed_reservation is not None:
        expected_digest = _reservation_sha256(
            mode, expected_final_seed_reservation
        )
        if reservation_digest != expected_digest:
            raise ValueError("evaluation shard final reservation does not match")

    candidate_policy = shard.get("candidatePolicy")
    if not isinstance(candidate_policy, Mapping):
        raise ValueError("evaluation shard candidate policy is invalid")
    routing = candidate_policy.get("routing")
    if not isinstance(routing, Mapping):
        raise ValueError("evaluation shard candidate routing is missing")
    evaluator._routing_from_report(routing)
    if (
        expected_candidate_policy_metadata is not None
        and canonical_json_bytes(candidate_policy)
        != canonical_json_bytes(expected_candidate_policy_metadata)
    ):
        raise ValueError("evaluation shard candidate policy does not match")

    results = shard.get("results")
    if not isinstance(results, list) or len(results) != len(selected):
        raise ValueError("evaluation shard result coverage is invalid")
    result_counts: list[int] = []
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("evaluation shard result must be an object")
        player_count = result.get("playerCount")
        if (
            isinstance(player_count, bool)
            or not isinstance(player_count, int)
            or player_count not in selected
        ):
            raise ValueError("evaluation shard result player count is invalid")
        if (
            result.get("matches") != expected_counts[player_count]
            or result.get("actsPerMatch") != ACTS_PER_MATCH
        ):
            raise ValueError("evaluation shard result preset does not match")
        result_counts.append(player_count)
    if result_counts != list(selected):
        raise ValueError("evaluation shard results must follow player-count order")
    if shard.get("deploymentTriggered") is not False:
        raise ValueError("evaluation shard must never trigger deployment")


def _publish_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_evaluation_shard_exclusive(
    output_path: str | Path, shard: Mapping[str, object]
) -> dict[str, object]:
    validate_evaluation_shard(shard)
    output = Path(output_path)
    checksum_path = output.with_name(output.name + ".sha256")
    if output.exists() or checksum_path.exists():
        raise FileExistsError("evaluation shard and checksum are immutable")
    payload = canonical_json_bytes(shard)
    digest = hashlib.sha256(payload).hexdigest()
    checksum_payload = f"{digest}  {output.name}\n".encode("ascii")
    _publish_exclusive(output, payload)
    try:
        _publish_exclusive(checksum_path, checksum_payload)
    except Exception:
        if output.exists():
            output.unlink()
        raise
    return {
        "path": str(output),
        "sha256Path": str(checksum_path),
        "sha256": digest,
        "bytes": len(payload),
    }


def load_evaluation_shard(path: str | Path) -> dict[str, object]:
    shard_path = Path(path)
    checksum_path = shard_path.with_name(shard_path.name + ".sha256")
    payload = shard_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected_sidecar = f"{digest}  {shard_path.name}\n".encode("ascii")
    if checksum_path.read_bytes() != expected_sidecar:
        raise ValueError("evaluation shard checksum does not match")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation shard JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("evaluation shard JSON must contain an object")
    if canonical_json_bytes(value) != payload:
        raise ValueError("evaluation shard JSON is not canonical")
    validate_evaluation_shard(value)
    return value


def merge_evaluation_shards(
    shards: Sequence[Mapping[str, object]],
    *,
    mode: str,
    seed_schedule: EvaluationSeedSchedule,
    bindings: EvaluationBindings,
    candidate_policy_metadata: Mapping[str, object],
    candidate_batched_forward: bool,
    bootstrap_resamples: int,
    batch_size: int,
    final_seed_reservation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not shards:
        raise ValueError("at least one evaluation shard is required")
    validate_parallel_evaluation_plan(
        mode=mode,
        seed_schedule=seed_schedule,
        bindings=bindings,
        final_seed_reservation=final_seed_reservation,
    )
    _positive_int(bootstrap_resamples, "bootstrap resamples")
    _positive_int(batch_size, "batch size")

    results_by_player_count: dict[int, Mapping[str, object]] = {}
    for shard in shards:
        validate_evaluation_shard(
            shard,
            expected_mode=mode,
            expected_seed_schedule=seed_schedule,
            expected_bindings=bindings,
            expected_candidate_policy_metadata=candidate_policy_metadata,
            expected_candidate_batched_forward=candidate_batched_forward,
            expected_bootstrap_resamples=bootstrap_resamples,
            expected_batch_size=batch_size,
            expected_final_seed_reservation=final_seed_reservation,
        )
        for result in shard["results"]:  # type: ignore[index]
            player_count = int(result["playerCount"])
            if player_count in results_by_player_count:
                raise ValueError(f"duplicate evaluation shard coverage for p{player_count}")
            results_by_player_count[player_count] = result
    missing = [value for value in PLAYER_COUNTS if value not in results_by_player_count]
    if missing:
        raise ValueError(
            "evaluation shards must cover p4 through p10 exactly; missing "
            + ", ".join(f"p{value}" for value in missing)
        )

    return assemble_benchmark_report(
        mode=mode,
        seed_schedule=seed_schedule,
        bindings=bindings,
        results=[results_by_player_count[value] for value in PLAYER_COUNTS],
        candidate_policy_metadata=candidate_policy_metadata,
        candidate_batched_forward=candidate_batched_forward,
        match_counts=_expected_match_counts(mode),
        gates=_expected_gates(mode),
        acts=ACTS_PER_MATCH,
        bootstrap_resamples=bootstrap_resamples,
        batch_size=batch_size,
        final_seed_reservation=final_seed_reservation,
    )


def evaluate_partition(
    *,
    mode: str,
    seed_schedule: EvaluationSeedSchedule,
    bindings: EvaluationBindings,
    player_counts: Sequence[int],
    candidate_policy: evaluator.CandidatePolicy,
    adapter: EnvironmentAdapter | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    batch_size: int = 128,
    final_seed_reservation: Mapping[str, object] | None = None,
    candidate_policy_routing: CandidatePolicyRouting | None = None,
) -> dict[str, object]:
    routing = candidate_policy_routing or CandidatePolicyRouting()
    selected = _validate_player_counts(list(player_counts))
    validate_parallel_evaluation_plan(
        mode=mode,
        seed_schedule=seed_schedule,
        bindings=bindings,
        final_seed_reservation=final_seed_reservation,
    )
    resolved_adapter = adapter or evaluator.V4EnvAdapter()
    counts = _expected_match_counts(mode)
    gates = _expected_gates(mode)
    results = [
        evaluate_player_count(
            player_count=player_count,
            matches=counts[player_count],
            acts=ACTS_PER_MATCH,
            seed_schedule=seed_schedule,
            candidate_policy=candidate_policy,
            adapter=resolved_adapter,
            gates=gates,
            bootstrap_resamples=bootstrap_resamples,
            batch_size=batch_size,
            candidate_policy_routing=routing,
        )
        for player_count in selected
    ]
    return build_evaluation_shard(
        mode=mode,
        seed_schedule=seed_schedule,
        bindings=bindings,
        player_counts=selected,
        results=results,
        candidate_policy_metadata=candidate_policy_report_metadata(
            candidate_policy, routing
        ),
        candidate_batched_forward=callable(
            getattr(candidate_policy, "actions", None)
        ),
        bootstrap_resamples=bootstrap_resamples,
        batch_size=batch_size,
        final_seed_reservation=final_seed_reservation,
    )


def _load_reservation(path: str | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    value = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("final reservation JSON must contain an object")
    return value


def _routing_from_arguments(arguments: argparse.Namespace) -> CandidatePolicyRouting:
    return CandidatePolicyRouting(
        mode=arguments.candidate_policy_mode,
        minimum_legal_logit_margin=arguments.minimum_legal_logit_margin,
        minimum_top_probability=arguments.minimum_top_probability,
    )


def _resolve_cli_inputs(
    arguments: argparse.Namespace,
    *,
    device: str,
    compile_actor: bool,
) -> tuple[
    evaluator.CenteredLogitActorPolicy,
    EvaluationBindings,
    CandidatePolicyRouting,
    Mapping[str, object] | None,
]:
    routing = _routing_from_arguments(arguments)
    policy, actor_sha256, artifact_sha256 = evaluator._load_cli_actor_policy(
        arguments.actor_bundle,
        actor_seeds=arguments.actor_seed,
        device=device,
        compile_actor=compile_actor,
    )
    bindings = resolve_cli_evaluation_bindings(
        artifact_sha256=artifact_sha256,
        actor_sha256=actor_sha256,
        observation_contract_path=arguments.observation_contract,
        frozen_normal_source_path=arguments.frozen_normal_source,
        repository_root=arguments.repository_root,
        frozen_normal_source_commit=arguments.frozen_baseline_commit,
        expected_observation_sha256=arguments.observation_sha256,
        expected_normal_sha256=arguments.frozen_baseline_sha256,
    )
    reservation = _load_reservation(arguments.final_reservation)
    return policy, bindings, routing, reservation


def _argument_parser() -> argparse.ArgumentParser:
    parser = evaluator._argument_parser()
    parser.description = (
        "Evaluate a verified V4 public actor in up to four whole-player-count "
        "subprocess shards and emit the canonical benchmark v2 report."
    )
    parser.add_argument("--workers", type=int, default=MAX_SUBPROCESS_WORKERS)
    parser.add_argument(
        "--shard-directory",
        help="fresh directory for immutable worker shards; defaults beside --output",
    )
    parser.add_argument(
        "--internal-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-player-count",
        action="append",
        type=int,
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-shard-output", help=argparse.SUPPRESS)
    return parser


def _base_worker_command(arguments: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    for value in arguments.actor_bundle:
        command.extend(("--actor-bundle", str(Path(value).resolve())))
    for value in arguments.actor_seed:
        command.extend(("--actor-seed", str(value)))
    command.extend(
        (
            "--mode",
            arguments.mode,
            "--family-id",
            arguments.family_id,
            "--base-seed",
            str(arguments.base_seed),
            "--output",
            str(Path(arguments.output).resolve()),
            "--device",
            arguments.device,
            "--candidate-policy-mode",
            arguments.candidate_policy_mode,
            "--batch-size",
            str(arguments.batch_size),
            "--bootstrap-resamples",
            str(arguments.bootstrap_resamples),
            "--frozen-baseline-commit",
            arguments.frozen_baseline_commit,
            "--frozen-normal-source",
            str(Path(arguments.frozen_normal_source).resolve()),
            "--repository-root",
            str(Path(arguments.repository_root).resolve()),
            "--observation-contract",
            str(Path(arguments.observation_contract).resolve()),
        )
    )
    for name, value in (
        ("--minimum-legal-logit-margin", arguments.minimum_legal_logit_margin),
        ("--minimum-top-probability", arguments.minimum_top_probability),
        ("--frozen-baseline-sha256", arguments.frozen_baseline_sha256),
        ("--observation-sha256", arguments.observation_sha256),
        ("--final-reservation", arguments.final_reservation),
    ):
        if value is not None:
            resolved = (
                str(Path(value).resolve())
                if name == "--final-reservation"
                else str(value)
            )
            command.extend((name, resolved))
    if arguments.compile_actor:
        command.append("--compile-actor")
    return command


def _worker_main(arguments: argparse.Namespace) -> int:
    if not arguments.worker_player_count or arguments.worker_shard_output is None:
        raise ValueError("internal worker requires player counts and shard output")
    if arguments.workers != MAX_SUBPROCESS_WORKERS or arguments.shard_directory:
        raise ValueError("internal worker received parent-only options")
    policy, bindings, routing, reservation = _resolve_cli_inputs(
        arguments,
        device=arguments.device,
        compile_actor=arguments.compile_actor,
    )
    schedule = EvaluationSeedSchedule(
        arguments.mode, arguments.family_id, arguments.base_seed
    )
    shard = evaluate_partition(
        mode=arguments.mode,
        seed_schedule=schedule,
        bindings=bindings,
        player_counts=arguments.worker_player_count,
        candidate_policy=policy,
        bootstrap_resamples=arguments.bootstrap_resamples,
        batch_size=arguments.batch_size,
        final_seed_reservation=reservation,
        candidate_policy_routing=routing,
    )
    published = write_evaluation_shard_exclusive(
        Path(arguments.worker_shard_output).resolve(), shard
    )
    print(json.dumps(published, sort_keys=True))
    return 0


def _parent_main(arguments: argparse.Namespace) -> int:
    if arguments.worker_player_count or arguments.worker_shard_output:
        raise ValueError("worker-only options require --internal-worker")
    workers = _positive_int(arguments.workers, "workers")
    if workers > MAX_SUBPROCESS_WORKERS:
        raise ValueError(f"workers must not exceed {MAX_SUBPROCESS_WORKERS}")
    output = Path(arguments.output).resolve()
    checksum = output.with_name(output.name + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("evaluation output path is immutable and already exists")
    if arguments.mode == "final" and not arguments.final_reservation:
        raise ValueError("--final-reservation is mandatory in final mode")
    if arguments.mode != "final" and arguments.final_reservation:
        raise ValueError("--final-reservation is valid only in final mode")

    # Independent CPU preflight binds every actual actor/Normal/observation file.
    # Workers repeat the same verification before moving their actor to CUDA.
    policy, bindings, routing, reservation = _resolve_cli_inputs(
        arguments, device="cpu", compile_actor=False
    )
    expected_policy_metadata = candidate_policy_report_metadata(policy, routing)
    if arguments.compile_actor:
        expected_policy_metadata["inferenceExecution"] = (
            "torch-compile-reduce-overhead"
        )
    del policy
    gc.collect()
    schedule = EvaluationSeedSchedule(
        arguments.mode, arguments.family_id, arguments.base_seed
    )
    validate_parallel_evaluation_plan(
        mode=arguments.mode,
        seed_schedule=schedule,
        bindings=bindings,
        final_seed_reservation=reservation,
    )

    shard_directory = (
        Path(arguments.shard_directory).resolve()
        if arguments.shard_directory
        else output.with_name(output.name + ".shards")
    )
    if shard_directory.exists():
        if not shard_directory.is_dir() or any(shard_directory.iterdir()):
            raise FileExistsError("evaluation shard directory must be fresh and empty")
    else:
        shard_directory.mkdir(parents=True)

    partitions = partition_player_counts(
        workers, match_counts=_expected_match_counts(arguments.mode)
    )
    commands: list[list[str]] = []
    shard_paths: list[Path] = []
    base_command = _base_worker_command(arguments)
    for index, player_counts in enumerate(partitions, start=1):
        suffix = "-".join(f"p{value}" for value in player_counts)
        shard_path = shard_directory / f"shard-{index:03d}-{suffix}.json"
        command = [*base_command, "--internal-worker"]
        for player_count in player_counts:
            command.extend(("--worker-player-count", str(player_count)))
        command.extend(("--worker-shard-output", str(shard_path)))
        commands.append(command)
        shard_paths.append(shard_path)

    processes: list[subprocess.Popen[str]] = []
    try:
        for command in commands:
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=str(Path(__file__).resolve().parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise

    # Drain every pipe concurrently.  Waiting serially can deadlock if a later
    # worker fills its stderr pipe while an earlier long-running worker owns
    # the parent's current ``communicate`` call.
    with ThreadPoolExecutor(max_workers=len(processes)) as pool:
        outputs = list(pool.map(lambda process: process.communicate(), processes))
    failures: list[str] = []
    for index, (process, (stdout, stderr)) in enumerate(
        zip(processes, outputs), start=1
    ):
        if process.returncode != 0:
            failures.append(
                f"worker {index} exited {process.returncode}: "
                f"{stderr.strip() or stdout.strip()}"
            )
    if failures:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise RuntimeError("; ".join(failures))

    shards = [load_evaluation_shard(path) for path in shard_paths]
    report = merge_evaluation_shards(
        shards,
        mode=arguments.mode,
        seed_schedule=schedule,
        bindings=bindings,
        candidate_policy_metadata=expected_policy_metadata,
        candidate_batched_forward=True,
        bootstrap_resamples=arguments.bootstrap_resamples,
        batch_size=arguments.batch_size,
        final_seed_reservation=reservation,
    )
    published = write_report_exclusive(output, report)
    print(
        json.dumps(
            {
                "promotionPassed": report["promotionPassed"],
                "evaluationMode": report["evaluationMode"],
                "familyId": arguments.family_id,
                "workers": len(partitions),
                "shardDirectory": str(shard_directory),
                **published,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    if arguments.internal_worker:
        return _worker_main(arguments)
    return _parent_main(arguments)


__all__ = [
    "MAX_SUBPROCESS_WORKERS",
    "SHARD_FORMAT",
    "SHARD_VERSION",
    "build_evaluation_shard",
    "evaluate_partition",
    "load_evaluation_shard",
    "main",
    "merge_evaluation_shards",
    "partition_player_counts",
    "validate_evaluation_shard",
    "validate_parallel_evaluation_plan",
    "write_evaluation_shard_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
