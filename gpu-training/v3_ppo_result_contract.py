from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

import torch

from v3_action_conditioned import load_v3_action_conditioned_json
from v3_ppo_dataset import V3_PPO_SEMANTICS_CONTRACT_SHA256


RESULT_FORMAT = "dalmuti-v3-ppo-training-result"
STRICT_RESULT_VERSION = 2
LEGACY_RESULT_VERSION = 1
STRICT_PROVENANCE_MODE = "strict-gpu-handoff"
LEGACY_PROVENANCE_MODE = "legacy-smoke"

BASE_RESULT_FILES = (
    "checkpoint.pt",
    "v3-actor-critic-weights.json",
    "v3-ppo-metadata.json",
    "training-metrics.json",
    "hardware-report.json",
    "data-verification.json",
    "training.log",
)

METRIC_FIELDS = (
    "epoch",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "explained_variance",
    "seconds",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_hex_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty portable path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"{label} is unsafe: {value}")
    return value


def reject_symlink_components(path: Path, label: str) -> None:
    """Reject an existing symlink anywhere in an absolute path.

    Calling ``resolve()`` first would erase this provenance, including for a
    symlink that still resolves inside the approved root.
    """
    absolute = path.absolute()
    parts = absolute.parts
    if not parts:
        raise ValueError(f"{label} path is empty")
    current = Path(parts[0])
    if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
        raise ValueError(f"{label} traverses a symbolic link: {current}")
    for part in parts[1:]:
        current = current / part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symbolic link: {current}")


def _regular_file_without_symlink(path: Path, label: str) -> None:
    if (
        path.is_symlink()
        or getattr(path, "is_junction", lambda: False)()
        or not path.is_file()
    ):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _json_file(path: Path, label: str) -> dict | list:
    _regular_file_without_symlink(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{label} must contain a JSON object or array")
    return value


def _manifest_file_entries(manifest: Mapping[str, object]) -> dict[str, dict]:
    values = manifest.get("files")
    if not isinstance(values, list) or not values:
        raise ValueError("bundle manifest contains no files")
    result: dict[str, dict] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
            raise ValueError(f"bundle manifest file {index} is not an object")
        relative = safe_relative_path(value.get("path"), f"bundle file {index} path")
        size = value.get("bytes")
        digest = value.get("sha256")
        if (
            relative in result
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _is_hex_sha256(digest)
        ):
            raise ValueError(f"invalid or duplicate bundle file entry: {relative}")
        result[relative] = value
    return result


def load_source_contract(
    bundle_manifest_path: Path,
    run_config_path: Path,
    *,
    verify_source_files: bool,
) -> dict:
    raw_bundle_manifest_path = bundle_manifest_path.absolute()
    raw_run_config_path = run_config_path.absolute()
    reject_symlink_components(raw_bundle_manifest_path, "source bundle manifest")
    reject_symlink_components(raw_run_config_path, "source GPU run config")
    bundle_manifest_path = raw_bundle_manifest_path.resolve()
    run_config_path = raw_run_config_path.resolve()
    _regular_file_without_symlink(bundle_manifest_path, "bundle manifest")
    _regular_file_without_symlink(run_config_path, "GPU run config")
    if bundle_manifest_path.parent != run_config_path.parent:
        raise ValueError("bundle manifest and GPU run config must be adjacent")
    if (
        bundle_manifest_path.name != "bundle-manifest.json"
        or run_config_path.name != "gpu-run-config.json"
    ):
        raise ValueError("source contract files must use their canonical names")
    root = bundle_manifest_path.parent
    bundle = _json_file(bundle_manifest_path, "bundle manifest")
    run_config = _json_file(run_config_path, "GPU run config")
    expected_bundle_keys = {
        "format",
        "version",
        "createdAt",
        "parentModel",
        "behaviorModelSha256",
        "observationSchemaVersion",
        "observationFeatures",
        "actionCatalogueVersion",
        "actionCount",
        "rollouts",
        "dataCounts",
        "files",
        "totalBytes",
    }
    if not isinstance(bundle, dict) or (
        set(bundle) != expected_bundle_keys
        or bundle.get("format") != "dalmuti-v3-ppo-gpu-bundle"
        or bundle.get("version") != 1
        or not isinstance(bundle.get("createdAt"), str)
        or not bundle.get("createdAt")
    ):
        raise ValueError("unsupported V3 PPO source bundle manifest")
    expected_run_config_keys = {
        "format",
        "version",
        "parentModelSha256",
        "rolloutTemperature",
        "algorithm",
        "allowedTerminalRankAuxiliaryCoefficients",
        "determinism",
        "pathPolicy",
        "requiredCommandArguments",
    }
    if not isinstance(run_config, dict) or (
        set(run_config) != expected_run_config_keys
        or run_config.get("format") != "dalmuti-v3-ppo-gpu-run-config"
        or run_config.get("version") != 2
    ):
        raise ValueError("unsupported V3 PPO GPU run config")

    entries = _manifest_file_entries(bundle)
    run_relative = run_config_path.relative_to(root).as_posix()
    run_entry = entries.get(run_relative)
    if run_entry is None:
        raise ValueError("GPU run config is not bound by the bundle manifest")
    run_sha256 = sha256_file(run_config_path)
    if (
        run_entry.get("bytes") != run_config_path.stat().st_size
        or run_entry.get("sha256") != run_sha256
    ):
        raise ValueError("GPU run config does not match the bundle manifest")

    parent = bundle.get("parentModel")
    if (
        not isinstance(parent, dict)
        or set(parent) != {"filename", "format", "bytes", "sha256"}
        or not isinstance(parent.get("filename"), str)
        or PurePosixPath(parent["filename"]).name != parent["filename"]
        or parent.get("format") != "dalmuti-action-conditioned-actor-critic"
        or isinstance(parent.get("bytes"), bool)
        or not isinstance(parent.get("bytes"), int)
        or parent["bytes"] < 1
        or not _is_hex_sha256(parent.get("sha256"))
    ):
        raise ValueError("bundle parent model binding is invalid")
    if run_config.get("parentModelSha256") != parent["sha256"]:
        raise ValueError("GPU run config parent model does not match the bundle")
    if (
        bundle.get("behaviorModelSha256") != parent["sha256"]
        or bundle.get("observationSchemaVersion") != 2
        or bundle.get("observationFeatures") != 172
        or bundle.get("actionCatalogueVersion") != 1
        or bundle.get("actionCount") != 236
    ):
        raise ValueError("bundle runtime/model metadata is inconsistent")
    behavior_entry = entries.get("behavior-model.json")
    if (
        behavior_entry is None
        or behavior_entry.get("sha256") != parent["sha256"]
        or behavior_entry.get("bytes") != parent.get("bytes")
    ):
        raise ValueError("bundle behavior model inventory is inconsistent")

    rollouts = bundle.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise ValueError("bundle contains no rollout provenance")
    rollout_inventory: list[dict] = []
    seen_rollouts: set[str] = set()
    for index, rollout in enumerate(rollouts):
        expected_rollout_keys = {
            "filename",
            "bytes",
            "sha256",
            "playerCount",
            "acts",
            "seed",
            "temperature",
            "episodes",
            "learnerSamples",
            "forcedSamples",
            "nonForcedSamples",
            "environmentDecisions",
        }
        if not isinstance(rollout, dict) or set(rollout) != expected_rollout_keys:
            raise ValueError(f"bundle rollout {index} is not an object")
        filename = rollout.get("filename")
        if (
            not isinstance(filename, str)
            or filename != PurePosixPath(filename).name
            or filename in seen_rollouts
        ):
            raise ValueError(f"bundle rollout {index} has an invalid filename")
        seen_rollouts.add(filename)
        relative = f"data/{filename}"
        entry = entries.get(relative)
        if (
            entry is None
            or rollout.get("bytes") != entry.get("bytes")
            or rollout.get("sha256") != entry.get("sha256")
        ):
            raise ValueError(f"bundle rollout inventory mismatch: {filename}")
        integer_fields = {
            "playerCount": 4,
            "acts": 1,
            "seed": 1,
            "episodes": 1,
            "learnerSamples": 1,
            "forcedSamples": 0,
            "nonForcedSamples": 1,
            "environmentDecisions": 1,
        }
        if any(
            isinstance(rollout.get(key), bool)
            or not isinstance(rollout.get(key), int)
            or rollout[key] < minimum
            for key, minimum in integer_fields.items()
        ):
            raise ValueError(f"bundle rollout counts are invalid: {filename}")
        if (
            rollout["learnerSamples"]
            != rollout["forcedSamples"] + rollout["nonForcedSamples"]
            or rollout["environmentDecisions"] < rollout["learnerSamples"]
            or isinstance(rollout.get("temperature"), bool)
            or not isinstance(rollout.get("temperature"), (int, float))
            or not math.isfinite(float(rollout["temperature"]))
            or float(rollout["temperature"]) <= 0
        ):
            raise ValueError(f"bundle rollout semantics are invalid: {filename}")
        rollout_inventory.append(
            {
                "path": relative,
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "playerCount": rollout.get("playerCount"),
                "acts": rollout.get("acts"),
                "seed": rollout.get("seed"),
                "temperature": rollout.get("temperature"),
                "episodes": rollout.get("episodes"),
                "learnerSamples": rollout.get("learnerSamples"),
                "forcedSamples": rollout.get("forcedSamples"),
                "nonForcedSamples": rollout.get("nonForcedSamples"),
                "environmentDecisions": rollout.get("environmentDecisions"),
            }
        )

    player_counts = [item["playerCount"] for item in rollout_inventory]
    if sorted(player_counts) != list(range(4, 11)):
        raise ValueError("strict V3 bundle requires exactly one rollout for players 4..10")
    rollout_temperature = run_config.get("rolloutTemperature")
    if any(
        not _same_scalar(item["temperature"], rollout_temperature)
        for item in rollout_inventory
    ):
        raise ValueError("bundle rollout temperatures differ from the run config")
    expected_data_counts = {
        key: sum(int(item[key]) for item in rollout_inventory)
        for key in (
            "episodes",
            "learnerSamples",
            "forcedSamples",
            "nonForcedSamples",
            "environmentDecisions",
        )
    }
    data_counts = bundle.get("dataCounts")
    if (
        not isinstance(data_counts, dict)
        or set(data_counts) != set(expected_data_counts)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in data_counts.values()
        )
        or data_counts != expected_data_counts
    ):
        raise ValueError("bundle aggregate rollout counts are inconsistent")
    if (
        isinstance(bundle.get("totalBytes"), bool)
        or not isinstance(bundle.get("totalBytes"), int)
        or bundle.get("totalBytes")
        != sum(entry["bytes"] for entry in entries.values())
    ):
        raise ValueError("bundle total byte count is inconsistent")

    if verify_source_files:
        actual_source_paths: set[str] = set()
        ignored_roots = {
            str(run_config.get("pathPolicy", {}).get("outputRoot", "")),
            str(run_config.get("pathPolicy", {}).get("resultsRoot", "")),
        }
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            if "__pycache__" in directory_names:
                reject_symlink_components(
                    current_path / "__pycache__", "bundle Python bytecode cache"
                )
                raise ValueError(
                    "bundle source inventory contains an unmanifested Python "
                    "bytecode cache"
                )
            if current_path == root:
                for name in directory_names:
                    if name in ignored_roots:
                        reject_symlink_components(
                            current_path / name,
                            f"bundle runtime directory {name}",
                        )
                directory_names[:] = [
                    name for name in directory_names if name not in ignored_roots
                ]
            for name in directory_names:
                reject_symlink_components(
                    current_path / name, f"bundle source directory {name}"
                )
            for name in file_names:
                path = current_path / name
                reject_symlink_components(path, f"bundle source inventory {name}")
                actual_source_paths.add(path.relative_to(root).as_posix())
        expected_source_paths = set(entries) | {"bundle-manifest.json"}
        if actual_source_paths != expected_source_paths:
            raise ValueError(
                "bundle source inventory mismatch; missing="
                f"{sorted(expected_source_paths - actual_source_paths)}, "
                f"unexpected={sorted(actual_source_paths - expected_source_paths)}"
            )
        for relative, entry in entries.items():
            raw_path = root / relative
            reject_symlink_components(raw_path, f"bundle source file {relative}")
            path = raw_path.resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"bundle file escapes its root: {relative}") from error
            _regular_file_without_symlink(path, f"bundle file {relative}")
            if (
                path.stat().st_size != entry["bytes"]
                or sha256_file(path) != entry["sha256"]
            ):
                raise ValueError(f"bundle source file binding mismatch: {relative}")

    algorithm = run_config.get("algorithm")
    path_policy = run_config.get("pathPolicy")
    allowed_rank = run_config.get("allowedTerminalRankAuxiliaryCoefficients")
    determinism = run_config.get("determinism")
    if not isinstance(algorithm, dict) or not algorithm:
        raise ValueError("GPU run config has no machine-verifiable algorithm contract")
    expected_algorithm = {
        "epochs": 12,
        "batchSize": 4096,
        "learningRate": 0.0001,
        "weightDecay": 0.00001,
        "gamma": 1,
        "gaeLambda": 1,
        "skipForcedPolicyTime": True,
        "rolloutTemperature": rollout_temperature,
        "clipCoefficient": 0.2,
        "valueCoefficient": 0.5,
        "entropyCoefficient": 0.01,
        "maxGradientNorm": 0.5,
        "targetKl": 0.015,
        "bindingTolerance": 0.00002,
        "behaviorBindingBatchSize": 8192,
        "loaderWorkers": 7,
        "device": "cuda",
        "seed": 202608061,
    }
    if algorithm != expected_algorithm:
        raise ValueError("GPU run config algorithm differs from the strict V3 contract")
    required_arguments = [
        "--output",
        "models/<fresh-v3-run>",
        "--results-dir",
        "returned/<fresh-v3-run>",
        "--epochs",
        "12",
        "--batch-size",
        "4096",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "0.00001",
        "--gamma",
        "1",
        "--gae-lambda",
        "1",
        "--skip-forced-policy-time",
        "--terminal-rank-auxiliary-coefficient",
        "<0-or-0.05>",
        "--rollout-temperature",
        str(rollout_temperature),
        "--clip-coefficient",
        "0.2",
        "--value-coefficient",
        "0.5",
        "--entropy-coefficient",
        "0.01",
        "--max-gradient-norm",
        "0.5",
        "--target-kl",
        "0.015",
        "--binding-tolerance",
        "0.00002",
        "--behavior-binding-batch-size",
        "8192",
        "--loader-workers",
        "7",
        "--seed",
        "202608061",
        "--device",
        "cuda",
    ]
    if run_config.get("requiredCommandArguments") != required_arguments:
        raise ValueError("GPU run config required command arguments are inconsistent")
    if not isinstance(path_policy, dict) or not path_policy:
        raise ValueError("GPU run config has no path policy")
    expected_path_policy = {
        "bundleRoot": ".",
        "behaviorModel": "behavior-model.json",
        "dataRoot": "data",
        "outputRoot": "models",
        "resultsRoot": "returned",
        "requireFreshRunDirectories": True,
        "requireMatchingRunIds": True,
        "requireDisjointPaths": True,
        "protectBundleInputs": True,
        "rejectSymbolicLinks": True,
    }
    if path_policy != expected_path_policy:
        raise ValueError("GPU run config path policy is not the strict V3 policy")
    expected_determinism = {
        "required": True,
        "pythonDontWriteBytecode": True,
        "torchDeterministicAlgorithms": True,
        "warnOnly": False,
        "cublasWorkspaceConfig": ":4096:8",
        "cudnnDeterministic": True,
        "cudnnBenchmark": False,
        "cudaMatmulAllowTf32": False,
        "cudnnAllowTf32": False,
    }
    if determinism != expected_determinism:
        raise ValueError("GPU run config deterministic contract mismatch")
    if (
        not isinstance(allowed_rank, list)
        or not allowed_rank
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in allowed_rank
        )
    ):
        raise ValueError("GPU run config rank-auxiliary variants are invalid")
    normalized_rank = [float(value) for value in allowed_rank]
    if normalized_rank != [0.0, 0.05]:
        raise ValueError("GPU run config must define the rank0/rank0.05 A/B variants")

    return {
        "root": root,
        "bundle": bundle,
        "runConfig": run_config,
        "bundleManifestSha256": sha256_file(bundle_manifest_path),
        "runConfigSha256": run_sha256,
        "parentModelSha256": parent["sha256"],
        "rollouts": rollout_inventory,
        "dataCounts": bundle.get("dataCounts"),
        "algorithm": algorithm,
        "pathPolicy": path_policy,
        "determinism": determinism,
        "allowedTerminalRankAuxiliaryCoefficients": [
            float(value) for value in allowed_rank
        ],
    }


def expected_result_paths(completed_epochs: int) -> tuple[str, ...]:
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int) or completed_epochs < 1:
        raise ValueError("completed epoch count must be positive")
    result = list(BASE_RESULT_FILES)
    for epoch in range(1, completed_epochs + 1):
        prefix = f"checkpoints/epoch-{epoch:02d}"
        result.extend(
            (
                f"{prefix}/checkpoint.pt",
                f"{prefix}/v3-actor-critic-weights.json",
                f"{prefix}/metrics.json",
            )
        )
    return tuple(result)


def exact_directory_inventory(
    root: Path,
    expected_paths: Iterable[str],
    *,
    allow_manifest: bool,
) -> dict[str, Path]:
    raw_root = root.absolute()
    reject_symlink_components(raw_root, "result root")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError(f"result root must be a regular directory: {root}")
    expected = set(expected_paths)
    if allow_manifest:
        expected.add("result-manifest.json")
    actual: dict[str, Path] = {}
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            directory = current_path / name
            if directory.is_symlink() or getattr(
                directory, "is_junction", lambda: False
            )():
                raise ValueError(f"result inventory contains a symlink: {directory}")
        for name in file_names:
            path = current_path / name
            if (
                path.is_symlink()
                or getattr(path, "is_junction", lambda: False)()
                or not path.is_file()
            ):
                raise ValueError(f"result inventory contains a non-regular file: {path}")
            relative = path.relative_to(root).as_posix()
            safe_relative_path(relative, "result inventory path")
            if relative in actual:
                raise ValueError(f"duplicate result inventory path: {relative}")
            actual[relative] = path
    actual_paths = set(actual)
    if actual_paths != expected:
        missing = sorted(expected - actual_paths)
        unexpected = sorted(actual_paths - expected)
        raise ValueError(
            "result inventory mismatch; missing="
            f"{missing}, unexpected={unexpected}"
        )
    return actual


def _finite_number(value: object, label: str, *, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_metrics(value: object) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("training metrics contain no epochs")
    result: list[dict] = []
    for expected_epoch, metric in enumerate(value, start=1):
        if not isinstance(metric, dict) or set(metric) != set(METRIC_FIELDS):
            raise ValueError(f"epoch {expected_epoch} metrics schema mismatch")
        if metric.get("epoch") != expected_epoch:
            raise ValueError("training metric epochs must be consecutive")
        for field in METRIC_FIELDS[1:]:
            minimum = 0.0 if field in (
                "value_loss",
                "entropy",
                "approximate_kl",
                "clip_fraction",
                "seconds",
            ) else None
            _finite_number(metric.get(field), f"epoch {expected_epoch} {field}", minimum=minimum)
        if float(metric["clip_fraction"]) > 1.0:
            raise ValueError(f"epoch {expected_epoch} clip_fraction exceeds one")
        result.append(metric)
    return result


def _load_checkpoint(path: Path, label: str) -> dict:
    _regular_file_without_symlink(path, label)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # PyTorch raises multiple serialization errors.
        raise ValueError(f"{label} is not a readable checkpoint") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a checkpoint object")
    return value


def _validate_checkpoint_contract(value: dict, epoch: int, *, optimizer: bool) -> None:
    expected_keys = {
        "format",
        "version",
        "epoch",
        "modelState",
        "observationFeatures",
        "observationSchemaVersion",
        "actionCatalogueVersion",
        "actionCount",
        "actorObservationHiddenSizes",
        "actorActionHiddenSizes",
        "actorScorerHiddenSizes",
        "valueHiddenSizes",
        "optimizerStateIncluded",
        "optimizerState",
    }
    if (
        set(value) != expected_keys
        or value.get("format") != "dalmuti-v3-action-conditioned-checkpoint"
        or value.get("version") != 1
        or value.get("epoch") != epoch
        or value.get("observationFeatures") != 172
        or value.get("observationSchemaVersion") != 2
        or value.get("actionCatalogueVersion") != 1
        or value.get("actionCount") != 236
        or value.get("optimizerStateIncluded") is not optimizer
        or not isinstance(value.get("modelState"), dict)
    ):
        raise ValueError(f"epoch {epoch} checkpoint contract mismatch")
    if optimizer and not isinstance(value.get("optimizerState"), dict):
        raise ValueError(f"epoch {epoch} optimizer state is missing")
    if not optimizer and value.get("optimizerState") is not None:
        raise ValueError("final checkpoint unexpectedly contains optimizer state")


def _compare_state_dicts(left: Mapping[str, object], right: Mapping[str, object], label: str) -> None:
    if set(left) != set(right):
        raise ValueError(f"{label} state-dict keys differ")
    for key in sorted(left):
        left_value = left[key]
        right_value = right[key]
        if not isinstance(left_value, torch.Tensor) or not isinstance(right_value, torch.Tensor):
            raise ValueError(f"{label} state-dict value is not a tensor: {key}")
        if left_value.shape != right_value.shape or not torch.equal(left_value, right_value):
            raise ValueError(f"{label} state-dict tensor differs: {key}")


def _compare_json_model_to_checkpoint(model_path: Path, checkpoint: dict, label: str) -> None:
    model, payload = load_v3_action_conditioned_json(model_path)
    if (
        payload.get("observationSchemaVersion") != 2
        or payload.get("observationFeatures") != 172
        or payload.get("actionCatalogueVersion") != 1
        or payload.get("actionCount") != 236
    ):
        raise ValueError(f"{label} runtime model contract mismatch")
    checkpoint_state = checkpoint["modelState"]
    model_state = model.state_dict()
    # JSON export rounds weights to 8 decimals, so compare against the
    # checkpoint with the same explicit export tolerance rather than silently
    # accepting a different selected model.
    if set(model_state) != set(checkpoint_state):
        raise ValueError(f"{label} model/checkpoint keys differ")
    for key in sorted(model_state):
        actual = model_state[key]
        expected = checkpoint_state[key]
        if not isinstance(expected, torch.Tensor) or actual.shape != expected.shape:
            raise ValueError(f"{label} model/checkpoint shape differs: {key}")
        if not torch.allclose(actual, expected, rtol=0.0, atol=1.1e-8):
            maximum = float(torch.max(torch.abs(actual - expected)))
            raise ValueError(
                f"{label} model/checkpoint value differs: {key} ({maximum})"
            )


def _algorithm_argument_mapping() -> dict[str, str]:
    return {
        "epochs": "epochs",
        "batchSize": "batch_size",
        "learningRate": "learning_rate",
        "weightDecay": "weight_decay",
        "gamma": "gamma",
        "gaeLambda": "gae_lambda",
        "skipForcedPolicyTime": "skip_forced_policy_time",
        "rolloutTemperature": "rollout_temperature",
        "clipCoefficient": "clip_coefficient",
        "valueCoefficient": "value_coefficient",
        "entropyCoefficient": "entropy_coefficient",
        "maxGradientNorm": "max_gradient_norm",
        "targetKl": "target_kl",
        "bindingTolerance": "binding_tolerance",
        "seed": "seed",
        "device": "device",
    }


def _same_scalar(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    return left == right


def _validate_strict_metadata(
    metadata: dict,
    data_verification: dict,
    hardware: dict,
    source: dict,
    run_id: str,
    completed_epochs: int,
) -> dict:
    expected_metadata_keys = {
        "format",
        "version",
        "behaviorModel",
        "behaviorModelSha256",
        "rolloutBehaviorModelSha256",
        "modelFormat",
        "observationSchemaVersion",
        "observationFeatures",
        "actionCatalogueVersion",
        "actionCount",
        "device",
        "torchVersion",
        "cudaAvailable",
        "cudaDevice",
        "gpuIdentity",
        "sourceProvenance",
        "deterministicRuntime",
        "samples",
        "trajectories",
        "forcedSamples",
        "policySamples",
        "returnEstimator",
        "skipForcedPolicyTime",
        "terminalRankAuxiliaryCoefficient",
        "rolloutTemperature",
        "behaviorBindingsVerified",
        "bindingTolerance",
        "completedEpochs",
        "stoppedForTargetKl",
        "sourceFiles",
        "sourceData",
        "arguments",
    }
    if (
        set(metadata) != expected_metadata_keys
        or metadata.get("format") != "dalmuti-v3-ppo-training-result"
        or metadata.get("version") != 1
        or metadata.get("modelFormat") != "dalmuti-action-conditioned-actor-critic"
        or metadata.get("observationSchemaVersion") != 2
        or metadata.get("observationFeatures") != 172
        or metadata.get("actionCatalogueVersion") != 1
        or metadata.get("actionCount") != 236
    ):
        raise ValueError("V3 training metadata contract mismatch")
    if (
        metadata.get("behaviorModelSha256") != source["parentModelSha256"]
        or metadata.get("rolloutBehaviorModelSha256") != source["parentModelSha256"]
    ):
        raise ValueError("training metadata parent model binding mismatch")
    if metadata.get("device") != "cuda" or metadata.get("cudaAvailable") is not True:
        raise ValueError("strict V3 result was not trained with CUDA")
    if metadata.get("completedEpochs") != completed_epochs:
        raise ValueError("training metadata completed epoch count mismatch")
    if metadata.get("behaviorBindingsVerified") is not True:
        raise ValueError("training metadata did not verify behavior bindings")

    arguments = metadata.get("arguments")
    expected_argument_keys = {
        "data",
        "behavior_model",
        "output",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gamma",
        "gae_lambda",
        "skip_forced_policy_time",
        "terminal_rank_auxiliary_coefficient",
        "rollout_temperature",
        "clip_coefficient",
        "value_coefficient",
        "entropy_coefficient",
        "max_gradient_norm",
        "target_kl",
        "binding_tolerance",
        "seed",
        "device",
        "run_id",
        "bundle_manifest",
        "run_config",
    }
    if not isinstance(arguments, dict) or set(arguments) != expected_argument_keys:
        raise ValueError("training metadata arguments are missing")
    algorithm = source["algorithm"]
    for contract_key, argument_key in _algorithm_argument_mapping().items():
        if contract_key not in algorithm:
            raise ValueError(f"GPU algorithm contract is missing {contract_key}")
        if not _same_scalar(arguments.get(argument_key), algorithm[contract_key]):
            raise ValueError(f"training argument differs from run contract: {argument_key}")
    rank_coefficient = arguments.get("terminal_rank_auxiliary_coefficient")
    if not any(
        _same_scalar(rank_coefficient, allowed)
        for allowed in source["allowedTerminalRankAuxiliaryCoefficients"]
    ):
        raise ValueError("terminal-rank auxiliary coefficient is not an allowed A/B variant")
    if not _same_scalar(metadata.get("terminalRankAuxiliaryCoefficient"), rank_coefficient):
        raise ValueError("rank-auxiliary metadata/argument mismatch")
    for metadata_key, argument_key in (
        ("rolloutTemperature", "rollout_temperature"),
        ("bindingTolerance", "binding_tolerance"),
        ("skipForcedPolicyTime", "skip_forced_policy_time"),
    ):
        if not _same_scalar(metadata.get(metadata_key), arguments.get(argument_key)):
            raise ValueError(f"training metadata/argument mismatch: {metadata_key}")
    if metadata.get("returnEstimator") != "undiscounted-monte-carlo":
        raise ValueError("strict V3 PPO must use undiscounted Monte Carlo returns")
    if arguments.get("device") != "cuda":
        raise ValueError("strict V3 PPO arguments must request CUDA")
    if arguments.get("run_id") != run_id:
        raise ValueError("training argument run_id differs from the result run ID")
    path_argument_expectations = {
        "behavior_model": "behavior-model.json",
        "bundle_manifest": "bundle-manifest.json",
        "run_config": "gpu-run-config.json",
    }
    for key, filename in path_argument_expectations.items():
        value = arguments.get(key)
        if not isinstance(value, str) or Path(value).name != filename:
            raise ValueError(f"training path argument is not bundle canonical: {key}")
    output_argument = arguments.get("output")
    if not isinstance(output_argument, str) or Path(output_argument).name != run_id:
        raise ValueError("training output argument differs from the result run ID")
    stopped = metadata.get("stoppedForTargetKl")
    if not isinstance(stopped, bool):
        raise ValueError("target-KL stop metadata must be boolean")
    requested_epochs = algorithm.get("epochs")
    if stopped:
        if completed_epochs > requested_epochs:
            raise ValueError("target-KL run completed more epochs than requested")
    elif completed_epochs != requested_epochs:
        raise ValueError("non-stopped run did not complete every requested epoch")

    provenance = metadata.get("sourceProvenance")
    if not isinstance(provenance, dict):
        raise ValueError("training metadata source provenance is missing")
    expected_provenance = {
        "runId": run_id,
        "bundleManifestSha256": source["bundleManifestSha256"],
        "runConfigSha256": source["runConfigSha256"],
        "parentModelSha256": source["parentModelSha256"],
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(f"training source provenance mismatch: {key}")

    expected_data_verification_keys = {
        "format",
        "version",
        "files",
        "sourceFiles",
        "samples",
        "trajectories",
        "behaviorModelSha256",
        "observationShape",
        "legalMaskShape",
        "actionCatalogueVersion",
        "actionCount",
        "forcedSamples",
        "policySamples",
        "terminalSamples",
        "gamma",
        "gaeLambda",
        "skipForcedPolicyTime",
        "terminalRankAuxiliaryCoefficient",
        "rolloutTemperature",
        "rolloutSemanticsContract",
        "behaviorBinding",
        "finite",
    }
    if (
        set(data_verification) != expected_data_verification_keys
        or data_verification.get("format")
        != "dalmuti-v3-ppo-data-verification"
        or data_verification.get("version") != 2
        or data_verification.get("finite") is not True
        or data_verification.get("behaviorModelSha256")
        != source["parentModelSha256"]
        or data_verification.get("actionCatalogueVersion") != 1
        or data_verification.get("actionCount") != 236
    ):
        raise ValueError("strict V3 data-verification contract mismatch")
    semantics = data_verification.get("rolloutSemanticsContract")
    expected_semantics = {
        "sha256": V3_PPO_SEMANTICS_CONTRACT_SHA256,
        "environment": "exact-game-rules-policy-and-provenance-verified",
        "reward": "recomputed-from-terminal-finishPlace-and-playerCount",
        "summaryCounts": "recomputed-or-strictly-bounded-and-verified",
    }
    if semantics != expected_semantics:
        raise ValueError("strict V3 rollout semantics attestation mismatch")
    binding = data_verification.get("behaviorBinding")
    expected_binding = {
        "observation": "verified",
        "actionCatalogue": "verified",
        "legalMask": "verified",
        "logProbability": "recomputed-and-verified",
        "value": "recomputed-and-verified",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != set(expected_binding) | {"absoluteTolerance"}
        or any(
            binding.get(key) != expected
            for key, expected in expected_binding.items()
        )
    ):
        raise ValueError("strict V3 behavior binding report is incomplete")
    expected_filenames = [PurePosixPath(item["path"]).name for item in source["rollouts"]]
    expected_source_data = {
        PurePosixPath(item["path"]).name: {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in source["rollouts"]
    }

    def report_filenames(value: object, label: str) -> list[str]:
        if not isinstance(value, list) or any(
            not isinstance(path, str) or not path for path in value
        ):
            raise ValueError(f"{label} file provenance is invalid")
        filenames = [Path(path).name for path in value]
        if len(filenames) != len(set(filenames)):
            raise ValueError(f"{label} repeats a rollout filename")
        return filenames

    if sorted(report_filenames(data_verification.get("files"), "data verification")) != sorted(expected_filenames):
        raise ValueError("data-verification files differ from the source bundle")
    if sorted(report_filenames(metadata.get("sourceFiles"), "training metadata")) != sorted(expected_filenames):
        raise ValueError("training source files differ from the source bundle")
    if sorted(report_filenames(arguments.get("data"), "training arguments")) != sorted(expected_filenames):
        raise ValueError("training data arguments differ from the source bundle")

    def hashed_source_files(value: object, label: str) -> dict[str, dict]:
        if not isinstance(value, list) or len(value) != len(expected_source_data):
            raise ValueError(f"{label} hashed source provenance is invalid")
        result: dict[str, dict] = {}
        for index, item in enumerate(value):
            if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
                raise ValueError(f"{label} source file {index} schema mismatch")
            path = item["path"]
            size = item["bytes"]
            digest = item["sha256"]
            if (
                not isinstance(path, str)
                or not path
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not _is_hex_sha256(digest)
            ):
                raise ValueError(f"{label} source file {index} is invalid")
            filename = Path(path).name
            if filename in result:
                raise ValueError(f"{label} repeats a hashed rollout filename")
            result[filename] = {"bytes": size, "sha256": digest}
        return result

    verification_sources = hashed_source_files(
        data_verification.get("sourceFiles"), "data verification"
    )
    metadata_sources = hashed_source_files(
        metadata.get("sourceData"), "training metadata"
    )
    if (
        verification_sources != expected_source_data
        or metadata_sources != expected_source_data
    ):
        raise ValueError("hashed rollout provenance differs from the source bundle")
    counts = source.get("dataCounts")
    if not isinstance(counts, dict) or (
        data_verification.get("samples") != counts.get("learnerSamples")
        or data_verification.get("forcedSamples") != counts.get("forcedSamples")
        or data_verification.get("policySamples") != counts.get("nonForcedSamples")
    ):
        raise ValueError("data-verification counts differ from the source bundle")
    if data_verification.get("observationShape") != [
        data_verification.get("samples"),
        172,
    ] or data_verification.get("legalMaskShape") != [
        data_verification.get("samples"),
        236,
    ]:
        raise ValueError("data-verification tensor shapes are inconsistent")
    trajectories = data_verification.get("trajectories")
    if (
        isinstance(trajectories, bool)
        or not isinstance(trajectories, int)
        or trajectories < 1
        or data_verification.get("terminalSamples") != trajectories
    ):
        raise ValueError("data-verification trajectory/terminal counts are invalid")
    if not _same_scalar(data_verification.get("gamma"), algorithm.get("gamma")):
        raise ValueError("data verification gamma differs from run contract")
    if not _same_scalar(data_verification.get("gaeLambda"), algorithm.get("gaeLambda")):
        raise ValueError("data verification lambda differs from run contract")

    if metadata.get("samples") != data_verification.get("samples"):
        raise ValueError("training/data sample count mismatch")
    if metadata.get("trajectories") != data_verification.get("trajectories"):
        raise ValueError("training/data trajectory count mismatch")
    if metadata.get("forcedSamples") != data_verification.get("forcedSamples"):
        raise ValueError("training/data forced count mismatch")
    if metadata.get("policySamples") != data_verification.get("policySamples"):
        raise ValueError("training/data policy count mismatch")
    for metadata_key, verification_key in (
        ("rolloutTemperature", "rolloutTemperature"),
        ("skipForcedPolicyTime", "skipForcedPolicyTime"),
        ("terminalRankAuxiliaryCoefficient", "terminalRankAuxiliaryCoefficient"),
        ("bindingTolerance", "behaviorBinding.absoluteTolerance"),
    ):
        if verification_key.startswith("behaviorBinding."):
            verification_value = data_verification.get("behaviorBinding", {}).get(
                verification_key.split(".", 1)[1]
            )
        else:
            verification_value = data_verification.get(verification_key)
        if not _same_scalar(metadata.get(metadata_key), verification_value):
            raise ValueError(f"training/data contract mismatch: {metadata_key}")

    expected_hardware_keys = {
        "format",
        "version",
        "platform",
        "pythonVersion",
        "pythonExecutable",
        "processArchitecture",
        "cpuCount",
        "numpyVersion",
        "torchVersion",
        "torchCudaVersion",
        "cudnnVersion",
        "cudaAvailable",
        "requestedDevice",
        "deterministicRuntime",
        "bundleFreeDiskBytes",
        "nvidiaSmi",
        "gpuDevices",
    }
    if (
        set(hardware) != expected_hardware_keys
        or hardware.get("format") != "dalmuti-gpu-preflight"
        or hardware.get("version") != 1
    ):
        raise ValueError("hardware report contract mismatch")
    if (
        hardware.get("cudaAvailable") is not True
        or hardware.get("requestedDevice") != "cuda"
        or not isinstance(hardware.get("torchCudaVersion"), str)
        or not hardware.get("torchCudaVersion")
    ):
        raise ValueError("hardware report does not attest CUDA availability")
    devices = hardware.get("gpuDevices")
    if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
        raise ValueError("hardware report contains no GPU identity")
    selected_name = metadata.get("cudaDevice")
    if not isinstance(selected_name, str) or selected_name != devices[0].get("name"):
        raise ValueError("training metadata GPU identity mismatch")
    if metadata.get("gpuIdentity") != devices[0]:
        raise ValueError("training metadata does not bind the complete GPU identity")
    expected_gpu_keys = {
        "index",
        "name",
        "computeCapability",
        "totalMemoryBytes",
        "multiProcessorCount",
        "uuid",
    }
    if set(devices[0]) != expected_gpu_keys:
        raise ValueError("hardware GPU identity schema mismatch")
    if metadata.get("torchVersion") != hardware.get("torchVersion"):
        raise ValueError("training/hardware torch version mismatch")
    deterministic = metadata.get("deterministicRuntime")
    hardware_deterministic = hardware.get("deterministicRuntime")
    if not isinstance(deterministic, dict) or deterministic != hardware_deterministic:
        raise ValueError("training/hardware deterministic runtime mismatch")
    required_determinism = {
        "algorithmsEnabled": True,
        "warnOnly": False,
        "cudnnDeterministic": True,
        "cudnnBenchmark": False,
        "cublasWorkspaceConfig": ":4096:8",
        "cudaMatmulAllowTf32": False,
        "cudnnAllowTf32": False,
    }
    expected_determinism_keys = set(required_determinism) | {
        "seed",
        "pythonHashSeed",
    }
    if set(deterministic) != expected_determinism_keys:
        raise ValueError("deterministic CUDA metadata schema mismatch")
    for key, value in required_determinism.items():
        if deterministic.get(key) != value:
            raise ValueError(f"deterministic CUDA contract mismatch: {key}")
    if deterministic.get("seed") != algorithm.get("seed"):
        raise ValueError("deterministic runtime seed differs from run contract")
    if deterministic.get("pythonHashSeed") != str(algorithm.get("seed")):
        raise ValueError("PYTHONHASHSEED differs from the deterministic run seed")
    return {
        "arguments": arguments,
        "rankAuxiliaryCoefficient": float(rank_coefficient),
        "gpu": devices[0],
        "deterministicRuntime": deterministic,
    }


def validate_result_directory(
    model_dir: Path,
    *,
    source: dict | None,
    allow_legacy_smoke: bool,
    allow_manifest: bool,
    expected_run_id: str | None = None,
) -> dict:
    raw_model_dir = model_dir.absolute()
    reject_symlink_components(raw_model_dir, "V3 result model directory")
    model_dir = raw_model_dir.resolve()
    metrics_value = _json_file(model_dir / "training-metrics.json", "training metrics")
    metrics = validate_metrics(metrics_value)
    completed_epochs = len(metrics)
    inventory = exact_directory_inventory(
        model_dir,
        expected_result_paths(completed_epochs),
        allow_manifest=allow_manifest,
    )
    metadata_value = _json_file(model_dir / "v3-ppo-metadata.json", "training metadata")
    data_value = _json_file(model_dir / "data-verification.json", "data verification")
    hardware_value = _json_file(model_dir / "hardware-report.json", "hardware report")
    if not all(isinstance(value, dict) for value in (metadata_value, data_value, hardware_value)):
        raise ValueError("result metadata reports must be JSON objects")
    metadata = metadata_value
    data_verification = data_value
    hardware = hardware_value

    final_checkpoint = _load_checkpoint(model_dir / "checkpoint.pt", "final checkpoint")
    _validate_checkpoint_contract(final_checkpoint, completed_epochs, optimizer=False)
    previous_checkpoint: dict | None = None
    for epoch, metric in enumerate(metrics, start=1):
        prefix = model_dir / "checkpoints" / f"epoch-{epoch:02d}"
        epoch_metric = _json_file(prefix / "metrics.json", f"epoch {epoch} metrics")
        if epoch_metric != metric:
            raise ValueError(f"epoch {epoch} metrics file differs from training metrics")
        checkpoint = _load_checkpoint(prefix / "checkpoint.pt", f"epoch {epoch} checkpoint")
        _validate_checkpoint_contract(checkpoint, epoch, optimizer=True)
        _compare_json_model_to_checkpoint(
            prefix / "v3-actor-critic-weights.json",
            checkpoint,
            f"epoch {epoch}",
        )
        previous_checkpoint = checkpoint
    assert previous_checkpoint is not None
    _compare_state_dicts(
        final_checkpoint["modelState"],
        previous_checkpoint["modelState"],
        "final/last-epoch",
    )
    final_json = model_dir / "v3-actor-critic-weights.json"
    last_json = (
        model_dir
        / "checkpoints"
        / f"epoch-{completed_epochs:02d}"
        / "v3-actor-critic-weights.json"
    )
    if final_json.read_bytes() != last_json.read_bytes():
        raise ValueError("selected final model is not the final completed epoch")
    _compare_json_model_to_checkpoint(final_json, final_checkpoint, "final")

    run_id = expected_run_id or model_dir.name
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in (".", "..")
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("result runId is invalid")
    strict_details: dict | None = None
    if source is None:
        if not allow_legacy_smoke:
            raise ValueError("strict V3 result validation requires source provenance")
        provenance_mode = LEGACY_PROVENANCE_MODE
    else:
        if allow_legacy_smoke:
            raise ValueError("legacy smoke mode cannot be combined with strict provenance")
        strict_details = _validate_strict_metadata(
            metadata,
            data_verification,
            hardware,
            source,
            run_id,
            completed_epochs,
        )
        provenance_mode = STRICT_PROVENANCE_MODE

    return {
        "runId": run_id,
        "completedEpochs": completed_epochs,
        "selectedEpoch": {
            "strategy": "final-completed-epoch",
            "epoch": completed_epochs,
            "modelSha256": sha256_file(final_json),
            "checkpointSha256": sha256_file(model_dir / "checkpoint.pt"),
        },
        "provenanceMode": provenance_mode,
        "strictDetails": strict_details,
        "metadata": metadata,
        "dataVerification": data_verification,
        "hardware": hardware,
        "inventory": inventory,
    }
