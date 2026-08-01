"""Strict verifier for tax-return advantage ensemble training results."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PurePosixPath

import torch

from non_card_counterfactual_dataset import file_sha256
from tax_return_advantage import (
    BASELINE_PROVENANCE_SHA256,
    TAX_RETURN_ADVANTAGE_MEMBER_COUNT,
    TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
    TaxReturnBilinearResidualNetwork,
    export_layer_parameters,
    member_parameters_sha256,
    read_ensemble_json,
)
from train_tax_return_advantage import RESULT_FORMAT, RESULT_VERSION, member_seed


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required result file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def _safe_relative(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise TypeError("manifest path must be a string")
    result = PurePosixPath(value)
    if (
        not value
        or result.is_absolute()
        or ".." in result.parts
        or "\\" in value
        or result.as_posix() != value
    ):
        raise ValueError(f"unsafe result path: {value}")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _actual_payload_paths(root: Path) -> set[str]:
    manifest_path = root / "training-manifest.json"
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }


def verify_result_directory(directory: str | Path) -> dict[str, object]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {root}")
    manifest = _read_object(root / "training-manifest.json")
    _keys(
        manifest,
        {
            "format",
            "version",
            "createdAt",
            "scoreSemantics",
            "baselineProvenanceSha256",
            "memberSeeds",
            "files",
            "totalBytes",
        },
        "training-manifest",
    )
    if (
        manifest["format"] != RESULT_FORMAT
        or manifest["version"] != RESULT_VERSION
        or manifest["scoreSemantics"] != TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS
        or manifest["baselineProvenanceSha256"] != BASELINE_PROVENANCE_SHA256
    ):
        raise ValueError("unsupported tax-return advantage training result")
    seeds = manifest["memberSeeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != TAX_RETURN_ADVANTAGE_MEMBER_COUNT
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("training manifest memberSeeds are invalid")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("training manifest contains no files")
    expected_paths: set[str] = set()
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("training manifest file entry must be an object")
        _keys(entry, {"path", "bytes", "sha256"}, "training manifest file")
        relative = _safe_relative(entry["path"])
        name = relative.as_posix()
        if name in expected_paths:
            raise ValueError(f"duplicate result path: {name}")
        expected_paths.add(name)
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(f"manifested result file is missing: {name}")
        size = path.stat().st_size
        if isinstance(entry["bytes"], bool) or entry["bytes"] != size:
            raise ValueError(f"result byte count mismatch: {name}")
        sha256 = entry["sha256"]
        if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
            raise ValueError(f"invalid result SHA-256: {name}")
        if file_sha256(path) != sha256:
            raise ValueError(f"result SHA-256 mismatch: {name}")
        total_bytes += size
    actual_paths = _actual_payload_paths(root)
    if actual_paths != expected_paths:
        raise ValueError("result contains unmanifested or missing files")
    if manifest["totalBytes"] != total_bytes:
        raise ValueError("training manifest totalBytes mismatch")

    config = _read_object(root / "training-config.json")
    dataset = _read_object(root / "dataset-manifest.json")
    metrics = _read_object(root / "training-metrics.json")
    model = read_ensemble_json(root / "model.json")
    if config.get("format") != RESULT_FORMAT or config.get("version") != RESULT_VERSION:
        raise ValueError("training config format mismatch")
    if config.get("baselineProvenanceSha256") != BASELINE_PROVENANCE_SHA256:
        raise ValueError("training config baseline provenance mismatch")
    options = config.get("options")
    if not isinstance(options, dict):
        raise ValueError("training config options are missing")
    base_seed = options.get("seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("training config seed is invalid")
    expected_seeds = [member_seed(base_seed, index) for index in range(5)]
    if not (seeds == config.get("memberSeeds") == expected_seeds):
        raise ValueError("member seed derivation mismatch")
    if config.get("objective") != model["objective"]:
        raise ValueError("training objective is not bound to the model")
    if dataset.get("format") != RESULT_FORMAT or dataset.get("version") != RESULT_VERSION:
        raise ValueError("dataset manifest format mismatch")
    if (
        dataset.get("groupSplitKey") != model["trainingData"]["groupSplitKey"]
        or dataset.get("sourceContract") != model["trainingData"]
        or model["objective"]["bootstrapUnit"]
        != model["trainingData"]["groupSplitKey"]
    ):
        raise ValueError("dataset/model source-contract binding mismatch")
    if dataset.get("routing") != {
        "returnCountOne": "excluded-from-training-exact-normal-fallback",
        "returnCountTwo": "trained",
    }:
        raise ValueError("dataset return-count routing mismatch")
    counts = dataset.get("counts")
    if not isinstance(counts, dict) or set(counts) != {
        "trainReturnCountTwo",
        "validationReturnCountTwo",
        "trainReturnCountOneExcluded",
        "validationReturnCountOneExcluded",
    }:
        raise ValueError("dataset exclusion counts are invalid")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("dataset exclusion counts must be non-negative integers")
    if counts["trainReturnCountTwo"] < 1 or counts["validationReturnCountTwo"] < 1:
        raise ValueError("dataset requires trained and validation returnCount=2 states")
    if metrics.get("format") != RESULT_FORMAT or metrics.get("version") != RESULT_VERSION:
        raise ValueError("training metrics format mismatch")
    if metrics.get("selectionMetric") != "paired-validation-loss":
        raise ValueError("checkpoint selection metric mismatch")
    metric_members = metrics.get("members")
    if not isinstance(metric_members, list) or len(metric_members) != 5:
        raise ValueError("training metrics must contain five members")

    for index, (member, metric) in enumerate(zip(model["members"], metric_members)):
        if not isinstance(metric, dict):
            raise ValueError("training member metric must be an object")
        if (
            metric.get("memberIndex") != index
            or metric.get("seed") != seeds[index]
            or metric.get("bestEpoch") != member["checkpointEpoch"]
            or metric.get("bestValidationPairedLoss") != member["validationPairedLoss"]
        ):
            raise ValueError("model/member metric binding mismatch")
        bootstrap = metric.get("bootstrap")
        if (
            not isinstance(bootstrap, dict)
            or bootstrap.get("unit") != model["trainingData"]["groupSplitKey"]
        ):
            raise ValueError("member bootstrap/group split binding mismatch")
        best_loss = _finite_nonnegative(
            member["validationPairedLoss"], "validationPairedLoss"
        )
        if member["parametersSha256"] != member_parameters_sha256(member):
            raise ValueError("member parameter hash mismatch")
        checkpoint_path = root / "members" / f"member-{index}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "dalmuti-tax-return-advantage-checkpoint":
            raise ValueError("member checkpoint format mismatch")
        if (
            checkpoint.get("version") != 1
            or checkpoint.get("memberIndex") != index
            or checkpoint.get("seed") != seeds[index]
            or checkpoint.get("epoch") != member["checkpointEpoch"]
        ):
            raise ValueError("member checkpoint provenance mismatch")
        if checkpoint.get("options") != options:
            raise ValueError("member checkpoint options/config binding mismatch")
        validation_metrics = checkpoint.get("validationMetrics")
        if not isinstance(validation_metrics, dict):
            raise ValueError("member checkpoint validation metrics missing")
        if _finite_nonnegative(validation_metrics.get("total"), "checkpoint paired loss") != best_loss:
            raise ValueError("member checkpoint paired loss mismatch")
        context_features = model["architecture"]["contextFeatures"]
        checkpoint_model = TaxReturnBilinearResidualNetwork(context_features)
        try:
            checkpoint_model.load_state_dict(checkpoint.get("modelState"), strict=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError("member checkpoint model state is invalid") from error
        checkpoint_parameters = export_layer_parameters(checkpoint_model)
        checkpoint_parameter_member = {
            **checkpoint_parameters,
            "parametersSha256": member["parametersSha256"],
        }
        if (
            member_parameters_sha256(checkpoint_parameter_member)
            != member["parametersSha256"]
            or checkpoint_parameters["contextLayer"] != member["contextLayer"]
            or checkpoint_parameters["bilinearWeight"] != member["bilinearWeight"]
        ):
            raise ValueError("member checkpoint/model JSON parameter mismatch")
        history = _read_object(root / "members" / f"member-{index}" / "metrics.json")
        if (
            history.get("memberIndex") != index
            or history.get("seed") != seeds[index]
            or history.get("bestEpoch") != member["checkpointEpoch"]
            or history.get("bestValidationPairedLoss") != best_loss
        ):
            raise ValueError("member history/model binding mismatch")

    return {
        "directory": str(root),
        "files": len(entries) + 1,
        "bytes": total_bytes + (root / "training-manifest.json").stat().st_size,
        "manifestSha256": file_sha256(root / "training-manifest.json"),
        "modelSha256": file_sha256(root / "model.json"),
        "memberSeeds": seeds,
        "scoreSemantics": TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
        "trainReturnCountTwo": counts["trainReturnCountTwo"],
        "validationReturnCountTwo": counts["validationReturnCountTwo"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a tax-return advantage ensemble training result."
    )
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    report = verify_result_directory(args.result_dir)
    print(
        f"Tax advantage result verified: {report['files']} files, "
        f"{report['bytes'] / 1024 / 1024:.2f} MiB"
    )
    print(f"Model SHA-256: {report['modelSha256']}")
    print(f"Manifest SHA-256: {report['manifestSha256']}")


if __name__ == "__main__":
    main()
