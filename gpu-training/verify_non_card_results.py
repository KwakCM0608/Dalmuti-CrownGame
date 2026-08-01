"""Verify a DALMUTI non-card supervised result directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Mapping

import torch

from non_card_action_conditioned import (
    load_revolution_action_conditioned_json,
    load_tax_return_action_conditioned_json,
)
from non_card_counterfactual_dataset import file_sha256
from train_non_card_counterfactual import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    TRAINING_RESULT_FORMAT,
    TRAINING_RESULT_VERSION,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_RESULT_VERSIONS = tuple(range(1, TRAINING_RESULT_VERSION + 1))
SUPPORTED_CHECKPOINT_VERSIONS = tuple(range(1, CHECKPOINT_VERSION + 1))


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe manifest path: {value}")
    if path.as_posix() != value:
        raise ValueError(f"manifest path is not canonical POSIX: {value}")
    return path


def _torch_load(path: Path) -> dict[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility only for old Torch releases that predate the safer
        # weights-only unpickler. Current GPU and bundled CPU runtimes use it.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint root must be an object: {path}")
    return value


def _validate_checkpoint(
    path: Path,
    *,
    decision: str,
    epoch: int,
    require_optimizer: bool,
    result_version: int,
    behavior_cloning_coefficient: float,
    utility_target: str,
) -> dict[str, object]:
    checkpoint = _torch_load(path)
    checkpoint_version = checkpoint.get("version")
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint_version not in SUPPORTED_CHECKPOINT_VERSIONS
        or checkpoint.get("decision") != decision
        or checkpoint.get("epoch") != epoch
    ):
        raise ValueError(f"checkpoint contract mismatch: {path}")
    if result_version >= 2:
        if checkpoint_version != result_version:
            raise ValueError(
                f"result/checkpoint version mismatch: {path}"
            )
        if (
            checkpoint.get("behaviorCloningCoefficient")
            != behavior_cloning_coefficient
            or not isinstance(checkpoint.get("trainingOptions"), dict)
            or checkpoint["trainingOptions"].get(
                "behavior_cloning_coefficient"
            )
            != behavior_cloning_coefficient
        ):
            raise ValueError(
                f"checkpoint behavior-cloning coefficient mismatch: {path}"
            )
    if result_version >= 3 and (
        checkpoint.get("utilityTarget") != utility_target
        or checkpoint["trainingOptions"].get("utility_target")
        != utility_target
    ):
        raise ValueError(f"checkpoint utility target mismatch: {path}")
    if not isinstance(checkpoint.get("modelState"), dict):
        raise ValueError(f"checkpoint model state is missing: {path}")
    if require_optimizer and not isinstance(checkpoint.get("optimizerState"), dict):
        raise ValueError(f"checkpoint optimizer state is missing: {path}")
    return checkpoint


def _load_json_model(decision: str, path: Path) -> torch.nn.Module:
    if decision == "tax-return":
        model, _ = load_tax_return_action_conditioned_json(path)
    elif decision == "revolution":
        model, _ = load_revolution_action_conditioned_json(path)
    else:
        raise ValueError(f"unsupported decision: {decision}")
    return model.eval()


def _compare_checkpoint_to_json(
    checkpoint: Mapping[str, object],
    json_path: Path,
    decision: str,
) -> None:
    model = _load_json_model(decision, json_path)
    checkpoint_state = checkpoint["modelState"]
    json_state = model.state_dict()
    if set(checkpoint_state) != set(json_state):
        raise ValueError(f"PT/JSON state keys differ: {json_path}")
    for key, json_tensor in json_state.items():
        checkpoint_tensor = checkpoint_state[key]
        if not isinstance(checkpoint_tensor, torch.Tensor):
            raise ValueError(f"checkpoint tensor is invalid: {key}")
        if checkpoint_tensor.shape != json_tensor.shape:
            raise ValueError(f"PT/JSON tensor shape differs: {key}")
        if not torch.isfinite(checkpoint_tensor).all():
            raise ValueError(f"checkpoint tensor is non-finite: {key}")
        if not torch.allclose(
            checkpoint_tensor.detach().cpu().to(dtype=torch.float32),
            json_tensor.detach().cpu().to(dtype=torch.float32),
            rtol=0.0,
            atol=1.1e-7,
        ):
            raise ValueError(f"PT/JSON tensor values differ: {key}")


def _validate_split_summary(
    value: object, label: str, *, result_version: int
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _require_keys(value, {"groupSplitKey", "train", "validation"}, label)
    if value["groupSplitKey"] != "canonicalWorldKey":
        raise ValueError(f"{label} group split key mismatch")
    for partition in ("train", "validation"):
        entry = value[partition]
        if not isinstance(entry, dict):
            raise ValueError(f"{label}.{partition} must be an object")
        expected_partition_keys = {
            "samples",
            "uniqueEpisodes",
            "uniqueWorlds",
            "sampleIdsSha256",
        }
        if result_version >= 2:
            expected_partition_keys.add("targetBestEqualsBaselineRate")
        _require_keys(entry, expected_partition_keys, f"{label}.{partition}")
        if (
            isinstance(entry["samples"], bool)
            or not isinstance(entry["samples"], int)
            or entry["samples"] < 1
            or isinstance(entry["uniqueEpisodes"], bool)
            or not isinstance(entry["uniqueEpisodes"], int)
            or entry["uniqueEpisodes"] < 1
            or entry["uniqueEpisodes"] > entry["samples"]
            or isinstance(entry["uniqueWorlds"], bool)
            or not isinstance(entry["uniqueWorlds"], int)
            or entry["uniqueWorlds"] < 1
            or entry["uniqueWorlds"] > entry["uniqueEpisodes"]
        ):
            raise ValueError(f"{label}.{partition} counts are invalid")
        if not isinstance(entry["sampleIdsSha256"], str) or not SHA256_RE.fullmatch(entry["sampleIdsSha256"]):
            raise ValueError(f"{label}.{partition} sample hash is invalid")
        if result_version >= 2:
            rate = entry["targetBestEqualsBaselineRate"]
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or not 0.0 <= float(rate) <= 1.0
            ):
                raise ValueError(
                    f"{label}.{partition} baseline target rate is invalid"
                )


def _validate_decision_result(
    root: Path,
    decision: str,
    summary: object,
    *,
    result_version: int,
    behavior_cloning_coefficient: float,
    utility_target: str,
) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"metrics summary is invalid for {decision}")
    legacy_keys = {
        "decision",
        "seed",
        "completedEpochs",
        "bestEpoch",
        "bestValidationLoss",
        "stoppedEarly",
        "dataset",
        "history",
    }
    actor_keys = {
        "decision",
        "seed",
        "completedEpochs",
        "selectionMetric",
        "bestEpoch",
        "bestValidationActorSelectionLoss",
        "bestValidationValueLossAtActorBest",
        "bestValueEpoch",
        "bestValidationValueLoss",
        "stoppedEarly",
        "dataset",
        "history",
    }
    actual_keys = set(summary)
    actor_contract = actual_keys == actor_keys
    legacy_contract = actual_keys == legacy_keys
    if result_version >= 2 and not actor_contract:
        raise ValueError(f"v2 metrics require actor selection fields for {decision}")
    if result_version == 1 and not (actor_contract or legacy_contract):
        raise ValueError(f"legacy metrics fields mismatch for {decision}")

    completed = summary["completedEpochs"]
    best_epoch = summary["bestEpoch"]
    history = summary["history"]
    if (
        summary["decision"] != decision
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 1
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch < 1
        or best_epoch > completed
        or not isinstance(history, list)
        or len(history) != completed
        or not isinstance(summary["stoppedEarly"], bool)
    ):
        raise ValueError(f"metrics summary contract mismatch for {decision}")
    if actor_contract:
        best_value_epoch = summary["bestValueEpoch"]
        if (
            summary["selectionMetric"] != "validation.actorSelectionLoss"
            or isinstance(best_value_epoch, bool)
            or not isinstance(best_value_epoch, int)
            or best_value_epoch < 1
            or best_value_epoch > completed
            or any(
                not isinstance(summary[field], (int, float))
                or isinstance(summary[field], bool)
                or not math.isfinite(float(summary[field]))
                for field in (
                    "bestValidationActorSelectionLoss",
                    "bestValidationValueLossAtActorBest",
                    "bestValidationValueLoss",
                )
            )
        ):
            raise ValueError(f"actor metrics summary mismatch for {decision}")
    else:
        if (
            isinstance(summary["bestValidationLoss"], bool)
            or not isinstance(summary["bestValidationLoss"], (int, float))
            or not math.isfinite(float(summary["bestValidationLoss"]))
        ):
            raise ValueError(f"legacy best validation loss is invalid for {decision}")

    _validate_split_summary(
        summary["dataset"],
        f"metrics.{decision}.dataset",
        result_version=result_version,
    )
    base_metric_keys = {
        "totalLoss",
        "policyCrossEntropy",
        "policyKl",
        "actionValueLoss",
        "valueLoss",
        "entropy",
        "bestActionAccuracy",
        "chosenActionRegret",
    }
    if actor_contract:
        base_metric_keys.add("actorSelectionLoss")
    if result_version >= 2:
        base_metric_keys.update(
            {
                "behaviorCloningLoss",
                "baselineActionAgreement",
                "targetBestEqualsBaselineRate",
                "predictedLogitMarginVsBaseline",
                "predictedProbabilityMarginVsBaseline",
                "targetUtilityMarginVsBaseline",
            }
        )
    for expected_epoch, metric in enumerate(history, start=1):
        if not isinstance(metric, dict) or metric.get("epoch") != expected_epoch:
            raise ValueError(f"{decision} history epochs are not consecutive")
        for partition in ("train", "validation"):
            values = metric.get(partition)
            if not isinstance(values, dict) or set(values) != base_metric_keys:
                raise ValueError(
                    f"{decision} epoch {expected_epoch} {partition} metric fields mismatch"
                )
            for key in base_metric_keys:
                value = values[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"{decision} epoch {expected_epoch} has invalid {key}"
                    )
            if result_version >= 2:
                for rate_key in (
                    "baselineActionAgreement",
                    "targetBestEqualsBaselineRate",
                ):
                    if not 0.0 <= float(values[rate_key]) <= 1.0:
                        raise ValueError(
                            f"{decision} epoch {expected_epoch} has invalid {rate_key}"
                        )
                if (
                    values["behaviorCloningLoss"] < 0
                    or values["predictedLogitMarginVsBaseline"] < -1.0e-7
                    or values["predictedProbabilityMarginVsBaseline"] < -1.0e-7
                ):
                    raise ValueError(
                        f"{decision} epoch {expected_epoch} has invalid baseline margin"
                    )
                fixed_rate = summary["dataset"][partition][
                    "targetBestEqualsBaselineRate"
                ]
                if not math.isclose(
                    float(values["targetBestEqualsBaselineRate"]),
                    float(fixed_rate),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        f"{decision} epoch {expected_epoch} target/baseline rate drifted"
                    )
        epoch_directory = (
            root / decision / "checkpoints" / f"epoch-{expected_epoch:03d}"
        )
        checkpoint = _validate_checkpoint(
            epoch_directory / "checkpoint.pt",
            decision=decision,
            epoch=expected_epoch,
            require_optimizer=True,
            result_version=result_version,
            behavior_cloning_coefficient=behavior_cloning_coefficient,
            utility_target=utility_target,
        )
        _compare_checkpoint_to_json(
            checkpoint, epoch_directory / "model.json", decision
        )
        epoch_metrics = _read_object(epoch_directory / "metrics.json")
        if epoch_metrics != metric:
            raise ValueError(
                f"{decision} epoch {expected_epoch} metrics file mismatch"
            )
    best_metrics = history[best_epoch - 1]["validation"]
    if actor_contract:
        value_best_metrics = history[summary["bestValueEpoch"] - 1][
            "validation"
        ]
        if (
            summary["bestValidationActorSelectionLoss"]
            != best_metrics["actorSelectionLoss"]
            or summary["bestValidationValueLossAtActorBest"]
            != best_metrics["valueLoss"]
            or summary["bestValidationValueLoss"]
            != value_best_metrics["valueLoss"]
        ):
            raise ValueError(
                f"{decision} best actor/value metric binding mismatch"
            )
    elif summary["bestValidationLoss"] != best_metrics["totalLoss"]:
        raise ValueError(f"{decision} legacy best metric binding mismatch")
    best_checkpoint = _validate_checkpoint(
        root / decision / "best" / "checkpoint.pt",
        decision=decision,
        epoch=best_epoch,
        require_optimizer=False,
        result_version=result_version,
        behavior_cloning_coefficient=behavior_cloning_coefficient,
        utility_target=utility_target,
    )
    _compare_checkpoint_to_json(
        best_checkpoint,
        root / decision / "best" / "model.json",
        decision,
    )
    decision_metrics = _read_object(root / decision / "metrics.json")
    if decision_metrics != summary:
        raise ValueError(f"{decision} metrics file mismatch")


def verify_result_directory(directory: str | Path) -> dict[str, object]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {root}")
    manifest = _read_object(root / "training-manifest.json")
    result_version = manifest.get("version")
    if result_version not in SUPPORTED_RESULT_VERSIONS:
        raise ValueError("unsupported training result manifest version")
    manifest_keys = {
        "format",
        "version",
        "createdAt",
        "groupSplitKey",
        "decisionKinds",
        "policyTemperatureOverride",
        "files",
        "totalBytes",
    }
    if result_version >= 2:
        manifest_keys.add("behaviorCloningCoefficient")
    if result_version >= 3:
        manifest_keys.add("utilityTarget")
    _require_keys(
        manifest,
        manifest_keys,
        "training-manifest",
    )
    if (
        manifest["format"] != TRAINING_RESULT_FORMAT
        or manifest["groupSplitKey"] != "canonicalWorldKey"
    ):
        raise ValueError("unsupported training result manifest")
    decision_kinds = manifest["decisionKinds"]
    if (
        not isinstance(decision_kinds, list)
        or not decision_kinds
        or decision_kinds not in (["tax-return"], ["revolution"], ["tax-return", "revolution"])
    ):
        raise ValueError("invalid decisionKinds in training manifest")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("training manifest contains no files")
    seen: set[str] = set()
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("training manifest file entry must be an object")
        _require_keys(entry, {"path", "bytes", "sha256"}, "training manifest file")
        relative = _safe_relative_path(entry["path"])
        name = relative.as_posix()
        if name in seen:
            raise ValueError(f"duplicate training manifest path: {name}")
        seen.add(name)
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(f"training result file is missing: {name}")
        size = path.stat().st_size
        if entry["bytes"] != size:
            raise ValueError(f"training result byte count mismatch: {name}")
        if not isinstance(entry["sha256"], str) or not SHA256_RE.fullmatch(entry["sha256"]):
            raise ValueError(f"invalid training result SHA-256: {name}")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"training result SHA-256 mismatch: {name}")
        total_bytes += size
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "training-manifest.json"
    }
    if actual_files != seen:
        raise ValueError("training result contains unmanifested or missing files")
    if manifest["totalBytes"] != total_bytes:
        raise ValueError("training result total byte count mismatch")

    config = _read_object(root / "training-config.json")
    dataset = _read_object(root / "dataset-manifest.json")
    metrics = _read_object(root / "training-metrics.json")
    for label, value in (("config", config), ("dataset", dataset), ("metrics", metrics)):
        if value.get("groupSplitKey") != "canonicalWorldKey":
            raise ValueError(f"{label} group split key mismatch")
    override = manifest["policyTemperatureOverride"]
    if (
        config.get("options", {}).get("policy_temperature") != override
        or dataset.get("policyTargets", {}).get("overrideTemperature") != override
        or metrics.get("policyTemperatureOverride") != override
    ):
        raise ValueError("policy temperature metadata is inconsistent")
    behavior_cloning_coefficient = (
        manifest["behaviorCloningCoefficient"]
        if result_version >= 2
        else 0.0
    )
    if (
        isinstance(behavior_cloning_coefficient, bool)
        or not isinstance(behavior_cloning_coefficient, (int, float))
        or not math.isfinite(float(behavior_cloning_coefficient))
        or behavior_cloning_coefficient < 0
    ):
        raise ValueError("behavior-cloning coefficient is invalid")
    config_options = config.get("options")
    if not isinstance(config_options, dict):
        raise ValueError("training config options are missing")
    config_behavior_coefficient = config_options.get(
        "behavior_cloning_coefficient", 0.0
    )
    metrics_behavior_coefficient = metrics.get(
        "behaviorCloningCoefficient", 0.0
    )
    config_top_behavior_coefficient = config.get(
        "behaviorCloningCoefficient", 0.0
    )
    if result_version >= 2 and (
        config.get("version") != result_version
        or metrics.get("version") != result_version
        or dataset.get("version") != result_version
        or "behaviorCloningCoefficient" not in config
        or "behaviorCloningCoefficient" not in metrics
    ):
        raise ValueError("v2 behavior-cloning metadata is incomplete")
    if not (
        config_behavior_coefficient
        == metrics_behavior_coefficient
        == config_top_behavior_coefficient
        == behavior_cloning_coefficient
    ):
        raise ValueError("behavior-cloning coefficient metadata is inconsistent")
    utility_target = (
        manifest["utilityTarget"] if result_version >= 3 else "terminal"
    )
    if utility_target not in ("terminal", "decision-act"):
        raise ValueError("utility target is invalid")
    config_utility_target = config_options.get("utility_target", "terminal")
    config_top_utility_target = config.get("utilityTarget", "terminal")
    metrics_utility_target = metrics.get("utilityTarget", "terminal")
    dataset_utility_target = dataset.get("utilityTarget", "terminal")
    if result_version >= 3 and (
        "utilityTarget" not in config
        or "utilityTarget" not in metrics
        or "utilityTarget" not in dataset
    ):
        raise ValueError("v3 utility-target metadata is incomplete")
    if not (
        config_utility_target
        == config_top_utility_target
        == metrics_utility_target
        == dataset_utility_target
        == utility_target
    ):
        raise ValueError("utility-target metadata is inconsistent")
    dataset_decisions = dataset.get("decisions")
    metrics_decisions = metrics.get("decisions")
    if not isinstance(dataset_decisions, dict) or not isinstance(metrics_decisions, dict):
        raise ValueError("decision metadata is missing")
    if set(dataset_decisions) != set(decision_kinds) or set(metrics_decisions) != set(decision_kinds):
        raise ValueError("decision metadata does not match the training manifest")
    for decision in decision_kinds:
        _validate_split_summary(
            dataset_decisions[decision],
            f"dataset.{decision}",
            result_version=result_version,
        )
        _validate_decision_result(
            root,
            decision,
            metrics_decisions[decision],
            result_version=result_version,
            behavior_cloning_coefficient=float(
                behavior_cloning_coefficient
            ),
            utility_target=utility_target,
        )
    return {
        "directory": str(root),
        "decisionKinds": decision_kinds,
        "files": len(entries) + 1,
        "bytes": total_bytes + (root / "training-manifest.json").stat().st_size,
        "manifestSha256": file_sha256(root / "training-manifest.json"),
        "policyTemperatureOverride": override,
        "behaviorCloningCoefficient": float(
            behavior_cloning_coefficient
        ),
        "resultVersion": result_version,
        "utilityTarget": utility_target,
        "groupSplitKey": "canonicalWorldKey",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a DALMUTI non-card supervised training result."
    )
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    report = verify_result_directory(args.result_dir)
    print(
        f"Non-card result verified: {report['files']} files, "
        f"{report['bytes'] / 1024 / 1024:.2f} MiB"
    )
    print(f"Decisions: {', '.join(report['decisionKinds'])}")
    print(f"Manifest SHA-256: {report['manifestSha256']}")


if __name__ == "__main__":
    main()
