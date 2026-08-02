from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PLAYER_COUNTS = tuple(range(4, 11))
MATCH_COUNTS = {4: 320, 5: 256, 6: 192, 7: 160, 8: 128, 9: 112, 10: 96}
CALIBRATION_COUNTS = {player_count: 1 for player_count in PLAYER_COUNTS}
BACKEND_MAP = ("cpu", "cpu", *("cuda" for _ in range(12)))
RUN_NAMESPACE = "v4-fixedid-ppo-i001-mixed-s580000001"
CALIBRATION_NAMESPACE = "v4-fixedid-mixed-calibration-s575000001"
ENVIRONMENT_SEED = 580000001
CALIBRATION_SEED = 575000001
TRAINING_SEED = 590000001
BEHAVIOR_ACTOR_SHA256 = "32f7f366c0a65d7b2b67baf5aeb2e33c49c87ddf4bcac513317bf710fc351466"
BEHAVIOR_MANIFEST_SHA256 = "6485004cfc936f1c711e84bbf6cdfe365eddf7055db6abb6a2780a24ed1c3b5c"
SCREEN_FAMILY = "attempt004-screening-seed450000001"
SCREEN_SEED = 450000001
FROZEN_BASELINE_COMMIT = "e0c52b0462d86756cf40b90f19d35a3e26b0f674"
FROZEN_BASELINE_SHA256 = "aa44743c64a23ac002d7faf09867bdb3e06232320f8efeb1df0e42724037bb61"
OBSERVATION_SCHEMA_SHA256 = "13dc7e4846669a4130dd69dd8b450c4ca3a443c2d1f64cfa08583c6a1108e99f"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_relative(value: object, label: str) -> str:
    _require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value, f"invalid {label}")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts), f"unsafe {label}")
    return str(path)


def load_recipe(path: Path) -> Mapping[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid mixed execution recipe") from error
    _require(isinstance(value, Mapping) and payload == canonical_json_bytes(value), "mixed execution recipe is not canonical JSON")
    validate_recipe(value)
    return value


def load_fixed_collection_plan_sha256(metadata_path: Path) -> str:
    """Load the trainer binding from strict merged metadata and its sidecar."""

    payload = metadata_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{metadata_path}.sha256")
    _require(
        sidecar.read_bytes() == f"{digest}  {metadata_path.name}\n".encode("ascii"),
        "merged metadata sidecar is stale or malformed",
    )
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid merged metadata") from error
    _require(
        isinstance(metadata, Mapping) and payload == canonical_json_bytes(metadata),
        "merged metadata is not canonical JSON",
    )
    loss = metadata.get("lossEligibility")
    _require(isinstance(loss, Mapping), "merged metadata lacks loss eligibility")
    plans = loss.get("fixedCollectionPlans")
    _require(
        isinstance(plans, list) and len(plans) == 1,
        "merged metadata must contain one fixed collection plan",
    )
    plan = plans[0]
    _require(isinstance(plan, Mapping), "merged fixed collection plan is invalid")
    fields = plan.get("canonicalFields")
    _require(isinstance(fields, Mapping), "merged fixed plan fields are missing")
    _require(fields.get("version") == 2, "merged fixed plan is not mixed version 2")
    _require(fields.get("matchShardCount") == 14, "merged fixed plan shard count drifted")
    _require(
        fields.get("shardBackendMap")
        == {str(index): backend for index, backend in enumerate(BACKEND_MAP)},
        "merged fixed plan backend map drifted",
    )
    plan_sha = plan.get("canonicalSha256")
    _require(
        isinstance(plan_sha, str) and SHA256_RE.fullmatch(plan_sha) is not None,
        "merged fixed plan SHA-256 is invalid",
    )
    _require(
        plan.get("opaqueId")
        == f"fixed-complete-mixed-backend-shard-plan-v2:sha256={plan_sha}",
        "merged fixed plan opaque ID drifted",
    )
    return plan_sha


def materialize_argv(
    argv: Sequence[str], replacements: Mapping[str, str]
) -> tuple[str, ...]:
    """Resolve every plan placeholder and fail if one remains unbound."""

    result: list[str] = []
    for raw in argv:
        value = raw
        for name, replacement in replacements.items():
            _require(
                isinstance(name, str)
                and name.startswith("{")
                and name.endswith("}")
                and isinstance(replacement, str)
                and replacement != "",
                "invalid plan placeholder replacement",
            )
            value = value.replace(name, replacement)
        _require("{" not in value and "}" not in value, f"unresolved plan placeholder: {value}")
        result.append(value)
    return tuple(result)


def _exact_mapping(value: object, expected: Mapping[str, object], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    for key, expected_value in expected.items():
        _require(value.get(key) == expected_value, f"{label}.{key} drifted")
    return value


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    _require(recipe.get("format") == "dalmuti-v4-mixed-package-recipe" and recipe.get("version") == 1, "unsupported mixed package recipe")
    _require(recipe.get("packageId") == RUN_NAMESPACE, "package namespace drifted")
    contract = recipe.get("runContract")
    _require(isinstance(contract, Mapping), "mixed run contract is missing")
    _require(contract.get("format") == "dalmuti-v4-fixedid-mixed-execution-contract" and contract.get("version") == 1, "unsupported mixed execution contract")
    identity = _exact_mapping(
        contract.get("identity"),
        {
            "calibrationNamespace": CALIBRATION_NAMESPACE,
            "calibrationSeed": CALIBRATION_SEED,
            "environmentSeed": ENVIRONMENT_SEED,
            "runNamespace": RUN_NAMESPACE,
            "trainingSeed": TRAINING_SEED,
        },
        "identity",
    )
    del identity
    behavior = _exact_mapping(
        contract.get("behaviorActor"),
        {"actorSha256": BEHAVIOR_ACTOR_SHA256, "manifestSha256": BEHAVIOR_MANIFEST_SHA256},
        "behaviorActor",
    )
    del behavior
    baseline = _exact_mapping(
        contract.get("baseline"),
        {
            "normalCommit": FROZEN_BASELINE_COMMIT,
            "normalSha256": FROZEN_BASELINE_SHA256,
            "normalSource": "frozen-git-bundle",
            "observationSha256": OBSERVATION_SCHEMA_SHA256,
            "observationSource": "sealed-current-source",
        },
        "baseline",
    )
    del baseline
    calibration = _exact_mapping(
        contract.get("calibration"),
        {
            "actualArtifactsRequired": ["report", "cpuNpz", "cudaNpz"],
            "backendRuns": ["cpu", "cuda"],
            "lanes": 7,
            "matchCounts": {str(key): value for key, value in CALIBRATION_COUNTS.items()},
            "matchShardCount": 1,
            "matchShardIndex": 0,
            "maximumFloatingDifference": 2e-5,
        },
        "calibration",
    )
    del calibration
    production = _exact_mapping(
        contract.get("productionCollection"),
        {
            "backendMap": list(BACKEND_MAP),
            "epsilonFloor": 0.0,
            "lanes": 16,
            "localShardIndices": [0, 1],
            "matchCounts": {str(key): value for key, value in MATCH_COUNTS.items()},
            "matchShardCount": 14,
            "matchStart": 0,
            "pairwiseCoefficient": 0.25,
            "remoteShardIndices": list(range(2, 14)),
            "remoteWaves": [list(range(2, 8)), list(range(8, 14))],
            "standardizeAdvantages": True,
            "temperature": 1.0,
        },
        "productionCollection",
    )
    del production
    replay = _exact_mapping(
        contract.get("pretrainingReplay"),
        {
            "device": "cuda",
            "fullPpoEligibleDataset": True,
            "maximumAbsoluteLogProbabilityError": 2e-5,
            "separateSealedAudit": True,
            "trainerMandatoryReplay": True,
        },
        "pretrainingReplay",
    )
    del replay
    training = _exact_mapping(
        contract.get("training"),
        {
            "actorLearningRate": 2e-5,
            "batchSize": 2,
            "bcWeight": 0.05,
            "checkpointEvery": 1,
            "clipRatio": 0.12,
            "criticLearningRate": 2e-4,
            "criticWeight": 0.2,
            "device": "cuda",
            "entropyCoefficient": 0.0005,
            "epochs": 1,
            "gamma": 1.0,
            "gradientAccumulation": 1,
            "lambda": 0.95,
            "maxGradientNorm": 1.0,
            "numWorkers": 0,
            "ppoWeight": 1.0,
            "qBoostCoefficient": 0.0,
            "weightDecay": 0.0001,
        },
        "training",
    )
    del training
    gates = _exact_mapping(
        contract.get("hardGates"),
        {
            "absoluteMaximumApproxKl": 0.02,
            "absoluteMaximumClipFraction": 0.25,
            "minimumEntropyRetention": 0.7,
            "secondEpochMaximumApproxKl": 0.0015,
            "secondEpochMaximumClipFraction": 0.03,
            "secondEpochRequiresScreenImprovement": True,
            "softMaximumApproxKl": 0.012,
            "softMaximumClipFraction": 0.15,
        },
        "hardGates",
    )
    del gates
    screening = _exact_mapping(
        contract.get("screening"),
        {
            "baseSeed": SCREEN_SEED,
            "batchSize": 64,
            "bootstrapResamples": 10000,
            "candidatePolicyMode": "pure-actor",
            "familyId": SCREEN_FAMILY,
            "matchesPerPlayerCount": 60,
            "playerCounts": list(PLAYER_COUNTS),
            "workers": 4,
        },
        "screening",
    )
    del screening
    prohibitions = contract.get("prohibitions")
    _require(
        prohibitions
        == {
            "deployment": True,
            "finalReservationSeed": True,
            "productIntegration": True,
            "resumeV3OrI2": True,
        },
        "prohibitions drifted",
    )
    promotion = contract.get("promotionGates")
    _require(
        promotion
        == {
            "allPlayerCountsRequired": True,
            "minimumClustered95LowerBound": 0.15,
            "minimumMeanChipDifferencePerAct": 0.25,
            "minimumPairwiseBeforeNormal": 0.55,
        },
        "promotion gates drifted",
    )
    topology = contract.get("topology")
    _require(
        topology
        == {
            "localCpuAndRemoteWaveOneParallel": True,
            "localCoordinator": "gpu-training/v4_mixed_local_coordinator.py",
            "remoteWorker": "gpu-training/v4_mixed_remote_worker.py",
            "resultPreservation": "local-and-remote-checksummed",
        },
        "mixed execution topology drifted",
    )
    package_screen = recipe.get("screening")
    _require(isinstance(package_screen, Mapping), "package screening binding is missing")
    _require(
        package_screen.get("familyId") == SCREEN_FAMILY
        and package_screen.get("baseSeed") == SCREEN_SEED
        and package_screen.get("matchesPerPlayerCount") == 60
        and package_screen.get("playerCounts") == list(PLAYER_COUNTS)
        and package_screen.get("actsPerMatch") == 5
        and package_screen.get("bootstrapResamples") == 10000,
        "package screening binding drifted",
    )
    for field in ("normalBaselineSha256", "observationSchemaSha256"):
        _require(isinstance(package_screen.get(field), str) and SHA256_RE.fullmatch(str(package_screen[field])) is not None, f"invalid package screening {field}")
    source_paths = recipe.get("sourcePaths")
    _require(isinstance(source_paths, list) and source_paths == sorted(source_paths) and len(source_paths) == len(set(source_paths)), "sourcePaths must be sorted and unique")
    required_sources = {
        "docs/rl-v4-execution-ledger.md",
        "gpu-training/v4_mixed_execution_recipe.json",
        "gpu-training/v4_mixed_local_coordinator.py",
        "gpu-training/v4_mixed_pretrain_replay.py",
        "gpu-training/v4_mixed_remote_worker.py",
        "gpu-training/v4_mixed_workflow.py",
        "lib/bot-strategy.ts",
        "training/v4-public-history.ts",
    }
    _require(required_sources.issubset(set(source_paths)), "mixed workflow sources are not all sealed")
    serialized = canonical_json_bytes(contract).decode("utf-8").lower()
    _require("v3-ppo-i2" not in serialized and "deploy" not in serialized.replace('"deployment":true', ""), "forbidden legacy/deployment command in contract")


def match_counts_arg(calibration: bool = False) -> str:
    counts = CALIBRATION_COUNTS if calibration else MATCH_COUNTS
    return ",".join(f"{key}:{counts[key]}" for key in PLAYER_COUNTS)


def backend_map_arg() -> str:
    return ",".join(BACKEND_MAP)


def npz_artifacts(path: str) -> tuple[str, str, str, str]:
    return (
        path,
        f"{path}.sha256",
        f"{path}.metadata.json",
        f"{path}.metadata.json.sha256",
    )


def report_artifacts(path: str) -> tuple[str, str]:
    return (path, f"{path}.sha256")


@dataclass(frozen=True)
class CommandSpec:
    command_id: str
    host: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"argv": list(self.argv), "host": self.host, "id": self.command_id, "outputs": list(self.outputs)}


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    dependencies: tuple[str, ...]
    commands: tuple[CommandSpec, ...]
    concurrency_group: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "commands": [command.to_dict() for command in self.commands],
            "concurrencyGroup": self.concurrency_group,
            "dependencies": list(self.dependencies),
            "id": self.phase_id,
        }


def collector_argv(
    *, python: str, source_root: str, actor_bundle: str, output: str,
    namespace: str, seed: int, device: str, shard_count: int, shard_index: int,
    calibration: bool, calibration_report: str | None = None,
    calibration_cpu_npz: str | None = None, calibration_cuda_npz: str | None = None,
) -> tuple[str, ...]:
    argv = [
        python,
        f"{source_root}/gpu-training/v4_collect_fixed_match_ppo.py",
        "--actor-bundle", actor_bundle,
        "--output", output,
        "--run-namespace", namespace,
        "--seed-base", str(seed),
        "--match-counts", match_counts_arg(calibration),
        "--match-start", "0",
        "--match-shard-count", str(shard_count),
        "--match-shard-index", str(shard_index),
        "--temperature", "1.0",
        "--epsilon-floor", "0.0",
        "--pairwise-coefficient", "0.25",
        "--lanes", "7" if calibration else "16",
        "--device", device,
        "--repository-root", source_root,
    ]
    if not calibration:
        _require(calibration_report is not None and calibration_cpu_npz is not None and calibration_cuda_npz is not None, "production collector lacks actual calibration triple")
        argv.extend(
            [
                "--shard-backend-map", backend_map_arg(),
                "--cross-backend-calibration-report", calibration_report,
                "--cross-backend-calibration-cpu-npz", calibration_cpu_npz,
                "--cross-backend-calibration-cuda-npz", calibration_cuda_npz,
            ]
        )
    return tuple(argv)


def training_argv(*, python: str, source_root: str, dataset: str, output: str, actor_bundle: str, plan_sha: str) -> tuple[str, ...]:
    return (
        python, f"{source_root}/gpu-training/v4_train.py",
        "--dataset", dataset, "--output", output,
        "--initialize-actor-bundle", actor_bundle,
        "--device", "cuda", "--epochs", "1", "--batch-size", "2",
        "--gradient-accumulation", "1", "--seed", str(TRAINING_SEED),
        "--actor-learning-rate", "2e-5", "--critic-learning-rate", "2e-4",
        "--weight-decay", "1e-4", "--bc-weight", "0.05", "--ppo-weight", "1.0",
        "--critic-weight", "0.2", "--q-boost-coefficient", "0.0",
        "--gamma", "1.0", "--lambda", "0.95", "--clip-ratio", "0.12",
        "--entropy-coefficient", "0.0005", "--max-gradient-norm", "1.0",
        "--num-workers", "0", "--checkpoint-every", "1",
        "--expected-fixed-collection-plan-sha256", plan_sha,
    )


def screening_argv(
    *, python: str, source_root: str, candidate: str, output: str,
    shard_directory: str, frozen_baseline: str,
) -> tuple[str, ...]:
    return (
        python, f"{source_root}/gpu-training/v4_evaluate_parallel.py",
        "--actor-bundle", candidate, "--mode", "screening",
        "--family-id", SCREEN_FAMILY, "--base-seed", str(SCREEN_SEED),
        "--output", output, "--device", "cuda",
        "--candidate-policy-mode", "pure-actor", "--batch-size", "64",
        "--bootstrap-resamples", "10000", "--workers", "4",
        "--shard-directory", shard_directory,
        "--frozen-baseline-commit", FROZEN_BASELINE_COMMIT,
        "--frozen-normal-source", f"{frozen_baseline}/lib/bot-strategy.ts",
        "--repository-root", frozen_baseline,
        "--observation-contract", f"{source_root}/training/v4-public-history.ts",
        "--frozen-baseline-sha256", FROZEN_BASELINE_SHA256,
        "--observation-sha256", OBSERVATION_SCHEMA_SHA256,
    )


def comparison_argv(
    *, python: str, source_root: str, cpu_npz: str, cuda_npz: str, output: str
) -> tuple[str, ...]:
    return (
        python,
        f"{source_root}/gpu-training/v4_compare_fixed_match_backends.py",
        "--cpu-npz",
        cpu_npz,
        "--cuda-npz",
        cuda_npz,
        "--output",
        output,
    )


def merge_argv(
    *, python: str, source_root: str, inputs: Sequence[str], output: str
) -> tuple[str, ...]:
    _require(len(inputs) == 14, "production merge requires all fourteen shards")
    argv = [python, f"{source_root}/gpu-training/v4_merge_datasets.py"]
    for path in inputs:
        argv.extend(("--input", path, "--input-checksum", f"{path}.sha256"))
    argv.extend(("--output", output))
    return tuple(argv)


def pretraining_replay_argv(
    *, python: str, source_root: str, dataset: str, actor_bundle: str, output: str
) -> tuple[str, ...]:
    return (
        python,
        f"{source_root}/gpu-training/v4_mixed_pretrain_replay.py",
        "replay",
        "--dataset",
        dataset,
        "--actor-bundle",
        actor_bundle,
        "--device",
        "cuda",
        "--maximum-absolute-log-probability-error",
        "2e-5",
        "--output",
        output,
    )


def training_gate_argv(
    *,
    python: str,
    source_root: str,
    training_result: str,
    run_manifest: str,
    candidate: str,
    output: str,
) -> tuple[str, ...]:
    return (
        python,
        f"{source_root}/gpu-training/v4_mixed_pretrain_replay.py",
        "verify-training-gates",
        "--training-result",
        training_result,
        "--run-manifest",
        run_manifest,
        "--candidate",
        candidate,
        "--maximum-approx-kl",
        "0.020",
        "--maximum-clip-fraction",
        "0.25",
        "--minimum-entropy-retention",
        "0.70",
        "--output",
        output,
    )


def promotion_gate_argv(
    *, python: str, source_root: str, screening_report: str, output: str
) -> tuple[str, ...]:
    return (
        python,
        f"{source_root}/gpu-training/v4_mixed_pretrain_replay.py",
        "verify-promotion-gates",
        "--screening-report",
        screening_report,
        "--minimum-mean-chip-difference-per-act",
        "0.25",
        "--minimum-clustered-95-lower-bound",
        "0.15",
        "--minimum-pairwise-before-normal",
        "0.55",
        "--output",
        output,
    )


def build_mixed_phase_plan(recipe: Mapping[str, Any]) -> tuple[PhaseSpec, ...]:
    """Build the immutable command DAG without executing local or remote work."""

    validate_recipe(recipe)
    local_source = "{local_source_root}"
    remote_source = "{remote_source_root}"
    local_run = "{local_run_directory}"
    remote_run = "{remote_run_directory}"
    local_actor = "{local_behavior_actor_bundle}"
    remote_actor = "{remote_behavior_actor_bundle}"
    local_python = "{local_python}"
    remote_python = "{remote_python}"
    calibration_report = f"{local_run}/calibration/backend-comparison.json"
    calibration_cpu = f"{local_run}/calibration/cpu.npz"
    calibration_cuda_local = f"{local_run}/calibration/cuda.npz"
    calibration_cuda_remote = f"{remote_run}/calibration/cuda.npz"
    remote_calibration_report = f"{remote_run}/calibration/backend-comparison.json"
    remote_calibration_cpu = f"{remote_run}/calibration/cpu.npz"
    remote_calibration_cuda = calibration_cuda_remote

    preflight = PhaseSpec(
        "preflight",
        (),
        (
            CommandSpec(
                "verify-local-actor",
                "local",
                (
                    local_python,
                    f"{local_source}/gpu-training/v4_mixed_local_coordinator.py",
                    "verify-actor",
                    "--actor-bundle",
                    local_actor,
                    "--expected-actor-sha256",
                    BEHAVIOR_ACTOR_SHA256,
                    "--expected-manifest-sha256",
                    BEHAVIOR_MANIFEST_SHA256,
                ),
                (),
            ),
            CommandSpec(
                "stage-remote-source-and-actor",
                "coordinator-transfer",
                ("stage-remote-source-and-actor", remote_run, local_actor),
                (remote_source, remote_actor),
            ),
        ),
    )

    local_calibration = PhaseSpec(
        "calibration-local-cpu",
        ("preflight",),
        (
            CommandSpec(
                "collect-calibration-cpu",
                "local",
                collector_argv(
                    python=local_python,
                    source_root=local_source,
                    actor_bundle=local_actor,
                    output=calibration_cpu,
                    namespace=CALIBRATION_NAMESPACE,
                    seed=CALIBRATION_SEED,
                    device="cpu",
                    shard_count=1,
                    shard_index=0,
                    calibration=True,
                ),
                npz_artifacts(calibration_cpu),
            ),
        ),
        "calibration-cpu-cuda",
    )
    remote_calibration = PhaseSpec(
        "calibration-remote-cuda",
        ("preflight",),
        (
            CommandSpec(
                "collect-calibration-cuda",
                "remote",
                collector_argv(
                    python=remote_python,
                    source_root=remote_source,
                    actor_bundle=remote_actor,
                    output=calibration_cuda_remote,
                    namespace=CALIBRATION_NAMESPACE,
                    seed=CALIBRATION_SEED,
                    device="cuda",
                    shard_count=1,
                    shard_index=0,
                    calibration=True,
                ),
                npz_artifacts(calibration_cuda_remote),
            ),
        ),
        "calibration-cpu-cuda",
    )
    calibration_admission = PhaseSpec(
        "calibration-admission",
        ("calibration-local-cpu", "calibration-remote-cuda"),
        (
            CommandSpec(
                "retrieve-calibration-cuda",
                "local-transfer",
                ("retrieve", calibration_cuda_remote, calibration_cuda_local),
                npz_artifacts(calibration_cuda_local),
            ),
            CommandSpec(
                "compare-calibration-backends",
                "local",
                comparison_argv(
                    python=local_python,
                    source_root=local_source,
                    cpu_npz=calibration_cpu,
                    cuda_npz=calibration_cuda_local,
                    output=calibration_report,
                ),
                report_artifacts(calibration_report),
            ),
            CommandSpec(
                "upload-calibration-triple",
                "local-transfer",
                (
                    "upload-calibration-triple",
                    calibration_report,
                    calibration_cpu,
                    calibration_cuda_local,
                    remote_calibration_report,
                    remote_calibration_cpu,
                    remote_calibration_cuda,
                ),
                (
                    *report_artifacts(remote_calibration_report),
                    *npz_artifacts(remote_calibration_cpu),
                    *npz_artifacts(remote_calibration_cuda),
                ),
            ),
        ),
    )

    def production_command(index: int, host: str) -> CommandSpec:
        is_local = host == "local"
        source = local_source if is_local else remote_source
        run = local_run if is_local else remote_run
        actor = local_actor if is_local else remote_actor
        python = local_python if is_local else remote_python
        report = calibration_report if is_local else remote_calibration_report
        cpu_npz = calibration_cpu if is_local else remote_calibration_cpu
        cuda_npz = calibration_cuda_local if is_local else remote_calibration_cuda
        output = f"{run}/rollouts/shard-{index:02d}.npz"
        return CommandSpec(
            f"collect-production-shard-{index:02d}",
            host,
            collector_argv(
                python=python,
                source_root=source,
                actor_bundle=actor,
                output=output,
                namespace=RUN_NAMESPACE,
                seed=ENVIRONMENT_SEED,
                device=BACKEND_MAP[index],
                shard_count=14,
                shard_index=index,
                calibration=False,
                calibration_report=report,
                calibration_cpu_npz=cpu_npz,
                calibration_cuda_npz=cuda_npz,
            ),
            npz_artifacts(output),
        )

    local_production = tuple(
        PhaseSpec(
            f"production-local-shard-{index:02d}",
            ("calibration-admission",),
            (production_command(index, "local"),),
            "production-local-and-remote",
        )
        for index in (0, 1)
    )
    remote_wave_one = tuple(
        PhaseSpec(
            f"production-remote-wave-one-shard-{index:02d}",
            ("calibration-admission",),
            (production_command(index, "remote"),),
            "production-local-and-remote",
        )
        for index in range(2, 8)
    )
    wave_one_ids = tuple(phase.phase_id for phase in remote_wave_one)
    remote_wave_two = tuple(
        PhaseSpec(
            f"production-remote-wave-two-shard-{index:02d}",
            wave_one_ids,
            (production_command(index, "remote"),),
            "production-local-and-remote",
        )
        for index in range(8, 14)
    )
    local_inputs = [f"{local_run}/rollouts/shard-{index:02d}.npz" for index in range(14)]
    merged_local = f"{local_run}/merged/production.npz"
    merged_remote = f"{remote_run}/merged/production.npz"
    merge = PhaseSpec(
        "retrieve-merge-upload",
        tuple(
            phase.phase_id
            for phase in (*local_production, *remote_wave_two)
        ),
        (
            CommandSpec(
                "retrieve-remote-production-shards",
                "local-transfer",
                ("retrieve-shards", "2-13", f"{remote_run}/rollouts", f"{local_run}/rollouts"),
                tuple(
                    artifact
                    for path in local_inputs[2:]
                    for artifact in npz_artifacts(path)
                ),
            ),
            CommandSpec(
                "merge-production-shards",
                "local",
                merge_argv(
                    python=local_python,
                    source_root=local_source,
                    inputs=local_inputs,
                    output=merged_local,
                ),
                npz_artifacts(merged_local),
            ),
            CommandSpec(
                "upload-merged-production",
                "local-transfer",
                (
                    "upload-merged",
                    merged_local,
                    merged_remote,
                    f"{local_run}/rollouts",
                    f"{remote_run}/rollouts",
                ),
                (
                    *npz_artifacts(merged_remote),
                    *npz_artifacts(f"{remote_run}/rollouts/shard-00.npz"),
                    *npz_artifacts(f"{remote_run}/rollouts/shard-01.npz"),
                ),
            ),
        ),
    )
    replay_report = f"{remote_run}/replay/pretraining.json"
    replay = PhaseSpec(
        "pretraining-cuda-replay",
        ("retrieve-merge-upload",),
        (
            CommandSpec(
                "replay-full-ppo-dataset",
                "remote",
                pretraining_replay_argv(
                    python=remote_python,
                    source_root=remote_source,
                    dataset=merged_remote,
                    actor_bundle=remote_actor,
                    output=replay_report,
                ),
                (replay_report, f"{replay_report}.sha256"),
            ),
        ),
    )
    training_root = f"{remote_run}/training/train-seed-{TRAINING_SEED}-run-001"
    training = PhaseSpec(
        "train-epoch-one",
        ("pretraining-cuda-replay",),
        (
            CommandSpec(
                "train-epoch-one-cuda",
                "remote",
                training_argv(
                    python=remote_python,
                    source_root=remote_source,
                    dataset=merged_remote,
                    output=training_root,
                    actor_bundle=remote_actor,
                    plan_sha="{merged_collection_plan_sha256}",
                ),
                (
                    f"{training_root}/result.json",
                    f"{training_root}/run-manifest.json",
                    f"{training_root}/candidate/actor.pt",
                    f"{training_root}/candidate/manifest.json",
                    f"{training_root}/candidate/manifest.json.sha256",
                ),
            ),
        ),
    )
    gate_report = f"{remote_run}/training/epoch-0001-hard-gates.json"
    gates = PhaseSpec(
        "post-training-hard-gates",
        ("train-epoch-one",),
        (
            CommandSpec(
                "publish-candidate-actor-sidecar",
                "remote",
                (
                    remote_python,
                    f"{remote_source}/gpu-training/v4_mixed_pretrain_replay.py",
                    "publish-candidate-sidecar",
                    "--candidate",
                    f"{training_root}/candidate",
                ),
                (f"{training_root}/candidate/actor.pt.sha256",),
            ),
            CommandSpec(
                "verify-epoch-one-hard-gates",
                "remote",
                training_gate_argv(
                    python=remote_python,
                    source_root=remote_source,
                    training_result=f"{training_root}/result.json",
                    run_manifest=f"{training_root}/run-manifest.json",
                    candidate=f"{training_root}/candidate",
                    output=gate_report,
                ),
                (gate_report, f"{gate_report}.sha256"),
            ),
        ),
    )
    screen_report = f"{remote_run}/screening/epoch-0001.json"
    screening = PhaseSpec(
        "screen-epoch-one",
        ("post-training-hard-gates",),
        (
            CommandSpec(
                "screen-epoch-one-p4-p10",
                "remote",
                screening_argv(
                    python=remote_python,
                    source_root=remote_source,
                    candidate=f"{training_root}/candidate",
                    output=screen_report,
                    shard_directory=f"{remote_run}/screening/epoch-0001.shards",
                    frozen_baseline="{remote_frozen_baseline_repository}",
                ),
                (screen_report, f"{screen_report}.sha256"),
            ),
        ),
    )
    promotion_report = f"{remote_run}/screening/epoch-0001-promotion-gates.json"
    promotion = PhaseSpec(
        "verify-all-player-promotion-gates",
        ("screen-epoch-one",),
        (
            CommandSpec(
                "verify-screening-promotion-gates",
                "remote",
                promotion_gate_argv(
                    python=remote_python,
                    source_root=remote_source,
                    screening_report=screen_report,
                    output=promotion_report,
                ),
                report_artifacts(promotion_report),
            ),
        ),
    )
    semantic_verification = PhaseSpec(
        "verify-complete-remote-screening",
        ("verify-all-player-promotion-gates",),
        (
            CommandSpec(
                "verify-complete-remote-screening",
                "remote",
                (
                    remote_python,
                    f"{remote_source}/gpu-training/v4_mixed_package_runtime.py",
                    "verify-screening",
                    "--package-dir",
                    "{remote_package_directory}",
                    "--expected-manifest-sha256",
                    "{package_manifest_sha256}",
                    "--source-root",
                    remote_source,
                    "--report",
                    screen_report,
                    "--candidate",
                    f"{training_root}/candidate",
                ),
                (),
            ),
        ),
    )
    preserve = PhaseSpec(
        "retrieve-verify-seal",
        ("verify-complete-remote-screening",),
        (
            CommandSpec(
                "finalize-remote-run",
                "coordinator-finalize",
                (
                    "finalize-remote-run",
                    remote_run,
                    f"{remote_run}/screening/epoch-0001.json",
                    f"{remote_run}/screening/epoch-0001-promotion-gates.json",
                ),
                (
                    f"{remote_run}/provenance/final-files.json",
                    f"{remote_run}/provenance/final-files.json.sha256",
                    f"{remote_run}/status/999-succeeded.json",
                    f"{remote_run}/status/999-succeeded.json.sha256",
                ),
            ),
            CommandSpec(
                "retrieve-checksummed-results",
                "local-transfer",
                (
                    "retrieve-results",
                    remote_run,
                    f"{local_run}/remote-sealed-run",
                ),
                (
                    f"{local_run}/remote-sealed-run/provenance/final-files.json",
                    f"{local_run}/remote-sealed-run/provenance/final-files.json.sha256",
                ),
            ),
            CommandSpec(
                "verify-and-seal-local-copy",
                "local",
                (
                    local_python,
                    f"{local_source}/gpu-training/v4_mixed_local_coordinator.py",
                    "verify-and-seal",
                    "--run-directory",
                    local_run,
                ),
                (
                    f"{local_run}/provenance/final-files.json",
                    f"{local_run}/provenance/final-files.json.sha256",
                ),
            ),
        ),
    )
    phases = (
        preflight,
        local_calibration,
        remote_calibration,
        calibration_admission,
        *local_production,
        *remote_wave_one,
        *remote_wave_two,
        merge,
        replay,
        training,
        gates,
        screening,
        promotion,
        semantic_verification,
        preserve,
    )
    assert_plan_is_acyclic(phases)
    return phases


def assert_plan_is_acyclic(phases: Sequence[PhaseSpec]) -> None:
    by_id = {phase.phase_id: phase for phase in phases}
    _require(len(by_id) == len(phases), "duplicate workflow phase")
    visited: set[str] = set()
    active: set[str] = set()

    def visit(name: str) -> None:
        _require(name in by_id, f"unknown phase dependency: {name}")
        if name in visited:
            return
        _require(name not in active, "workflow phase cycle")
        active.add(name)
        for dependency in by_id[name].dependencies:
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for phase in phases:
        visit(phase.phase_id)


def plan_document(phases: Sequence[PhaseSpec], recipe: Mapping[str, Any]) -> dict[str, object]:
    assert_plan_is_acyclic(phases)
    fields = {
        "format": "dalmuti-v4-mixed-workflow-plan",
        "version": 1,
        "runNamespace": RUN_NAMESPACE,
        "recipeSha256": canonical_sha256(recipe),
        "phases": [phase.to_dict() for phase in phases],
        "prohibitions": recipe["runContract"]["prohibitions"],
    }
    return {**fields, "canonicalSha256": canonical_sha256(fields)}
