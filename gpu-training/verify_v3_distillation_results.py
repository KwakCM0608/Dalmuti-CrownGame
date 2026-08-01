from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Mapping

import torch

from v3_action_conditioned import load_v3_action_conditioned_json
from v3_distillation_dataset import file_sha256
from train_v3_distillation import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    TRAINING_RESULT_FORMAT,
    TRAINING_RESULT_VERSION,
)
from verify_v3_distillation_bundle import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_SPLIT,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_TRAINING,
    verify_bundle,
)


PACKAGE_FORMAT = "dalmuti-v3-distillation-result-package"
PACKAGE_VERSION = 1
PACKAGE_MANIFEST_NAME = "package-manifest.json"
ARCHIVE_ROOT = "result"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ROOT_FILES = [
    "checkpoint.pt",
    "training-manifest.json",
    "training-metrics.json",
    "v3-actor-critic-weights.json",
]
EXPECTED_EPOCH_FILES = [
    "checkpoint.pt",
    "metrics.json",
    "v3-actor-critic-weights.json",
]
EXPECTED_PROVENANCE_FILES = [
    "provenance/bundle-manifest.json",
    "provenance/gpu-run-config.json",
    "provenance/handoff-files.sha256",
    "provenance/hardware-report.json",
    "provenance/training.log",
]
EXPECTED_GPU_ARGUMENTS = {
    "data": "v3-distillation.ndjson",
    "teacher_model": "ppo4-actor-critic-weights.json",
    "output": "model",
    "epochs": EXPECTED_TRAINING["epochs"],
    "batch_size": EXPECTED_TRAINING["batchSize"],
    "learning_rate": EXPECTED_TRAINING["learningRate"],
    "weight_decay": EXPECTED_TRAINING["weightDecay"],
    "value_coefficient": EXPECTED_TRAINING["valueCoefficient"],
    "validation_fraction": EXPECTED_TRAINING["validationFraction"],
    "split_seed": EXPECTED_TRAINING["splitSeed"],
    "seed": EXPECTED_TRAINING["seed"],
    "patience": EXPECTED_TRAINING["patience"],
    "max_gradient_norm": EXPECTED_TRAINING["maxGradientNorm"],
    "binding_tolerance": EXPECTED_TRAINING["bindingTolerance"],
    "device": "cuda",
    "deterministic": True,
}


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _torch_load(path: Path) -> dict[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint root must be an object: {path}")
    return value


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")


def _validate_checkpoint(
    path: Path,
    *,
    epoch: int,
    teacher_sha256: str,
    dataset_sha256: str,
    temperature: float,
    optimizer_required: bool,
) -> dict[str, object]:
    checkpoint = _torch_load(path)
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("version") != CHECKPOINT_VERSION
        or checkpoint.get("epoch") != epoch
        or checkpoint.get("teacherSha256") != teacher_sha256
        or checkpoint.get("datasetSha256") != dataset_sha256
        or checkpoint.get("temperature") != temperature
        or not isinstance(checkpoint.get("modelState"), dict)
        or checkpoint.get("optimizerStateIncluded") is not optimizer_required
        or (
            optimizer_required
            and not isinstance(checkpoint.get("optimizerState"), dict)
        )
        or (
            not optimizer_required
            and checkpoint.get("optimizerState") is not None
        )
    ):
        raise ValueError(f"checkpoint contract mismatch: {path}")
    for key, tensor in checkpoint["modelState"].items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"invalid model state entry: {path}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite model state: {path}:{key}")
    return checkpoint


def _compare_checkpoint_json(
    checkpoint: Mapping[str, object], model_path: Path
) -> None:
    model, payload = load_v3_action_conditioned_json(model_path)
    if (
        payload.get("observationSchemaVersion") != 2
        or payload.get("observationFeatures") != 172
        or payload.get("actionCatalogueVersion") != 1
        or payload.get("actionCount") != 236
    ):
        raise ValueError(f"V3 runtime model contract mismatch: {model_path}")
    checkpoint_state = checkpoint["modelState"]
    json_state = model.state_dict()
    if set(checkpoint_state) != set(json_state):
        raise ValueError(f"PT/JSON state keys differ: {model_path}")
    for key, json_tensor in json_state.items():
        checkpoint_tensor = checkpoint_state[key]
        if (
            checkpoint_tensor.shape != json_tensor.shape
            or not torch.allclose(
                checkpoint_tensor.detach().cpu().to(dtype=torch.float32),
                json_tensor.detach().cpu().to(dtype=torch.float32),
                rtol=0.0,
                atol=1.1e-7,
            )
        ):
            raise ValueError(f"PT/JSON tensor values differ: {model_path}:{key}")


def _verify_expected_handoff_result(
    root: Path,
    manifest: Mapping[str, object],
    *,
    expected_handoff: str | Path,
    teacher_model: str | Path | None,
    dataset: str | Path | None,
) -> dict[str, object]:
    if teacher_model is None or dataset is None:
        raise ValueError(
            "expected-handoff verification requires the external teacher and dataset"
        )
    handoff_root = Path(expected_handoff).resolve()
    handoff_report = verify_bundle(
        handoff_root,
        verify_teacher_bindings=False,
    )
    handoff_manifest_path = handoff_root / "bundle-manifest.json"
    handoff_config_path = handoff_root / "gpu-run-config.json"
    handoff_checksums_path = handoff_root / "handoff-files.sha256"
    handoff_manifest = _read_object(handoff_manifest_path)
    handoff_config = _read_object(handoff_config_path)
    for filename, expected_path in (
        ("bundle-manifest.json", handoff_manifest_path),
        ("gpu-run-config.json", handoff_config_path),
        ("handoff-files.sha256", handoff_checksums_path),
    ):
        copied = root / "provenance" / filename
        if file_sha256(copied) != file_sha256(expected_path):
            raise ValueError(f"returned handoff provenance mismatch: {filename}")
    copied_manifest = _read_object(root / "provenance" / "bundle-manifest.json")
    copied_config = _read_object(root / "provenance" / "gpu-run-config.json")
    if (
        copied_manifest != handoff_manifest
        or copied_config != handoff_config
        or copied_manifest.get("runId") != handoff_report["runId"]
        or copied_config.get("runId") != handoff_report["runId"]
    ):
        raise ValueError("returned handoff manifest, run config, or run-id mismatch")
    teacher = manifest.get("teacher")
    dataset_binding = manifest.get("dataset")
    arguments = manifest.get("arguments")
    split = manifest.get("split")
    if teacher != {
        "filename": "ppo4-actor-critic-weights.json",
        "sha256": EXPECTED_TEACHER_SHA256,
        "format": "dalmuti-actor-critic",
        "actionCount": 506,
        "temperature": 2.5,
    }:
        raise ValueError("GPU result teacher contract differs from the handoff")
    if dataset_binding != {
        "filename": "v3-distillation.ndjson",
        "sha256": EXPECTED_DATASET_SHA256,
        "samples": 140000,
    }:
        raise ValueError("GPU result dataset contract differs from the handoff")
    if arguments != EXPECTED_GPU_ARGUMENTS:
        raise ValueError("GPU result training arguments differ from the fixed command")
    if not isinstance(split, dict):
        raise ValueError("GPU result split is missing")
    expected_split_fields = {
        "groupSplitKey": EXPECTED_SPLIT["groupSplitKey"],
        "train": {
            "samples": EXPECTED_SPLIT["trainSamples"],
            "uniqueGroups": EXPECTED_SPLIT["trainGroups"],
        },
        "validation": {
            "samples": EXPECTED_SPLIT["validationSamples"],
            "uniqueGroups": EXPECTED_SPLIT["validationGroups"],
        },
        "overlappingGroups": 0,
    }
    if set(split) != {"groupSplitKey", "train", "validation", "overlappingGroups"}:
        raise ValueError("GPU result split fields differ from the fixed split")
    for partition in ("train", "validation"):
        observed = split.get(partition)
        expected = expected_split_fields[partition]
        if (
            not isinstance(observed, dict)
            or set(observed) != {"samples", "uniqueGroups", "sampleIdsSha256"}
            or observed.get("samples") != expected["samples"]
            or observed.get("uniqueGroups") != expected["uniqueGroups"]
            or not isinstance(observed.get("sampleIdsSha256"), str)
            or not SHA256_RE.fullmatch(observed["sampleIdsSha256"])
        ):
            raise ValueError(f"GPU result {partition} split binding mismatch")
    if (
        split.get("groupSplitKey") != expected_split_fields["groupSplitKey"]
        or split.get("overlappingGroups") != 0
    ):
        raise ValueError("GPU result split contract mismatch")
    hardware = manifest.get("hardware")
    reproducibility = manifest.get("reproducibility")
    if (
        manifest.get("device") != "cuda"
        or not isinstance(hardware, dict)
        or hardware.get("device") != "cuda"
        or hardware.get("cudaAvailable") is not True
        or not isinstance(hardware.get("torchCudaVersion"), str)
        or not hardware.get("torchCudaVersion")
        or not isinstance(hardware.get("gpu"), dict)
        or not hardware["gpu"].get("name")
        or not isinstance(reproducibility, dict)
        or reproducibility.get("seed") != EXPECTED_TRAINING["seed"]
        or reproducibility.get("deterministicAlgorithms") is not True
        or reproducibility.get("cudnnDeterministic") is not True
        or reproducibility.get("cudnnBenchmark") is not False
        or reproducibility.get("cublasWorkspaceConfig")
        not in (":4096:8", ":16:8")
    ):
        raise ValueError("GPU result lacks the required CUDA determinism contract")
    preflight = _read_object(root / "provenance" / "hardware-report.json")
    devices = preflight.get("gpuDevices")
    if (
        preflight.get("format") != "dalmuti-gpu-preflight"
        or preflight.get("version") != 1
        or preflight.get("cudaAvailable") is not True
        or not isinstance(devices, list)
        or hardware["gpu"] not in devices
        or any(
            preflight.get(key) != hardware.get(key)
            for key in (
                "pythonVersion",
                "numpyVersion",
                "torchVersion",
                "torchCudaVersion",
                "cudnnVersion",
            )
        )
    ):
        raise ValueError("GPU preflight provenance does not match training hardware")
    training_log = (root / "provenance" / "training.log").read_text(
        encoding="utf-8"
    )
    if not training_log.strip() or "epoch 001 |" not in training_log:
        raise ValueError("GPU training log provenance is empty or incomplete")
    return {
        "runId": handoff_report["runId"],
        "handoffManifestSha256": file_sha256(handoff_manifest_path),
        "gpuRunConfigSha256": file_sha256(handoff_config_path),
        "handoffChecksumsSha256": file_sha256(handoff_checksums_path),
    }


def verify_result_directory(
    directory: str | Path,
    *,
    teacher_model: str | Path | None = None,
    dataset: str | Path | None = None,
    expected_handoff: str | Path | None = None,
    allow_legacy_inventory: bool = False,
) -> dict[str, object]:
    root = Path(directory).resolve()
    required = {
        "checkpoint.pt",
        "v3-actor-critic-weights.json",
        "training-manifest.json",
        "training-metrics.json",
    }
    missing = [name for name in sorted(required) if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("incomplete V3 distillation result: " + ", ".join(missing))
    manifest = _read_object(root / "training-manifest.json")
    if (
        manifest.get("format") != TRAINING_RESULT_FORMAT
        or manifest.get("version") != TRAINING_RESULT_VERSION
    ):
        raise ValueError("unsupported V3 distillation training result")
    teacher = manifest.get("teacher")
    dataset_binding = manifest.get("dataset")
    split = manifest.get("split")
    if (
        not isinstance(teacher, dict)
        or not isinstance(teacher.get("sha256"), str)
        or not SHA256_RE.fullmatch(teacher["sha256"])
        or teacher.get("format") != "dalmuti-actor-critic"
        or teacher.get("actionCount") != 506
        or isinstance(teacher.get("temperature"), bool)
        or not isinstance(teacher.get("temperature"), (int, float))
        or teacher["temperature"] <= 0
        or not isinstance(dataset_binding, dict)
        or not isinstance(dataset_binding.get("sha256"), str)
        or not SHA256_RE.fullmatch(dataset_binding["sha256"])
        or not isinstance(dataset_binding.get("samples"), int)
        or dataset_binding["samples"] < 2
        or not isinstance(split, dict)
        or split.get("groupSplitKey") != "sourceSha256:episodeId"
        or split.get("overlappingGroups") != 0
    ):
        raise ValueError("training provenance or group split contract mismatch")
    if teacher_model is not None and file_sha256(teacher_model) != teacher["sha256"]:
        raise ValueError("external teacher model SHA-256 mismatch")
    if dataset is not None and file_sha256(dataset) != dataset_binding["sha256"]:
        raise ValueError("external distillation dataset SHA-256 mismatch")
    metrics_path = root / "training-metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("training metrics contain no epochs")
    _finite_tree(metrics, "training metrics")
    completed_epochs = manifest.get("completedEpochs")
    best_epoch = manifest.get("bestEpoch")
    if (
        completed_epochs != len(metrics)
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch < 1
        or best_epoch > len(metrics)
    ):
        raise ValueError("completed/best epoch binding mismatch")
    checkpoints = root / "checkpoints"
    expected_directories = {
        f"epoch-{epoch:03d}" for epoch in range(1, len(metrics) + 1)
    }
    actual_directories = {
        path.name for path in checkpoints.iterdir() if path.is_dir()
    } if checkpoints.is_dir() else set()
    if actual_directories != expected_directories:
        raise ValueError("epoch checkpoint directories mismatch")
    epoch_checkpoints: dict[int, dict[str, object]] = {}
    for epoch, metric in enumerate(metrics, start=1):
        if not isinstance(metric, dict) or metric.get("epoch") != epoch:
            raise ValueError("training metric epochs must be consecutive")
        directory_path = checkpoints / f"epoch-{epoch:03d}"
        epoch_metric = _read_object(directory_path / "metrics.json")
        if epoch_metric != metric:
            raise ValueError(f"epoch metrics binding mismatch: {epoch}")
        checkpoint = _validate_checkpoint(
            directory_path / "checkpoint.pt",
            epoch=epoch,
            teacher_sha256=teacher["sha256"],
            dataset_sha256=dataset_binding["sha256"],
            temperature=float(teacher["temperature"]),
            optimizer_required=True,
        )
        _compare_checkpoint_json(
            checkpoint, directory_path / "v3-actor-critic-weights.json"
        )
        epoch_checkpoints[epoch] = checkpoint
    final_checkpoint = _validate_checkpoint(
        root / "checkpoint.pt",
        epoch=best_epoch,
        teacher_sha256=teacher["sha256"],
        dataset_sha256=dataset_binding["sha256"],
        temperature=float(teacher["temperature"]),
        optimizer_required=False,
    )
    _compare_checkpoint_json(
        final_checkpoint, root / "v3-actor-critic-weights.json"
    )
    best_state = epoch_checkpoints[best_epoch]["modelState"]
    final_state = final_checkpoint["modelState"]
    if set(best_state) != set(final_state) or any(
        not torch.equal(best_state[key], final_state[key]) for key in best_state
    ):
        raise ValueError("final model is not the selected best epoch")
    if manifest.get("bestValidation") != metrics[best_epoch - 1].get("validation"):
        raise ValueError("best validation metric binding mismatch")
    train = split.get("train")
    validation = split.get("validation")
    if (
        not isinstance(train, dict)
        or not isinstance(validation, dict)
        or train.get("samples", 0) < 1
        or validation.get("samples", 0) < 1
        or train["samples"] + validation["samples"]
        != dataset_binding["samples"]
    ):
        raise ValueError("split sample counters mismatch")
    inventory = manifest.get("resultInventory")
    if inventory is None and not allow_legacy_inventory:
        raise ValueError(
            "strict result inventory is required; use explicit legacy opt-in only "
            "for a pre-inventory local artifact"
        )
    if expected_handoff is not None and allow_legacy_inventory:
        raise ValueError("expected-handoff verification cannot allow legacy inventory")
    if inventory is not None:
        if (
            not isinstance(inventory, dict)
            or inventory
            != {
                "version": 1,
                "requiredRootFiles": EXPECTED_ROOT_FILES,
                "requiredEpochFiles": EXPECTED_EPOCH_FILES,
                "optionalProvenanceFiles": EXPECTED_PROVENANCE_FILES,
            }
        ):
            raise ValueError("strict result inventory contract mismatch")
        required_inventory = set(EXPECTED_ROOT_FILES)
        for epoch in range(1, len(metrics) + 1):
            directory_name = f"checkpoints/epoch-{epoch:03d}"
            required_inventory.update(
                f"{directory_name}/{filename}"
                for filename in EXPECTED_EPOCH_FILES
            )
        allowed_inventory = required_inventory | set(EXPECTED_PROVENANCE_FILES)
        actual_inventory = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("strict result inventory contains a symbolic link")
        if (
            not required_inventory.issubset(actual_inventory)
            or not actual_inventory.issubset(allowed_inventory)
        ):
            raise ValueError("strict result directory file inventory mismatch")
        hardware = manifest.get("hardware")
        reproducibility = manifest.get("reproducibility")
        arguments = manifest.get("arguments")
        device = manifest.get("device")
        if (
            not isinstance(hardware, dict)
            or hardware.get("format")
            != "dalmuti-v3-distillation-training-hardware"
            or hardware.get("version") != 1
            or hardware.get("device") != device
            or hardware.get("torchVersion") != manifest.get("torchVersion")
            or not isinstance(reproducibility, dict)
            or reproducibility.get("seed")
            != (arguments.get("seed") if isinstance(arguments, dict) else None)
            or reproducibility.get("deterministicAlgorithms") is not True
            or reproducibility.get("cublasWorkspaceConfig")
            not in (":4096:8", ":16:8")
            or not isinstance(arguments, dict)
            or arguments.get("deterministic") is not True
        ):
            raise ValueError("hardware or deterministic training binding mismatch")
        if str(device).startswith("cuda") and (
            hardware.get("cudaAvailable") is not True
            or not isinstance(hardware.get("torchCudaVersion"), str)
            or not hardware.get("torchCudaVersion")
            or not isinstance(hardware.get("gpu"), dict)
            or not hardware["gpu"].get("name")
        ):
            raise ValueError("CUDA result lacks GPU/CUDA identity")
    handoff_binding: dict[str, object] | None = None
    if expected_handoff is not None:
        if not isinstance(completed_epochs, int) or not (
            1 <= completed_epochs <= EXPECTED_TRAINING["epochs"]
        ):
            raise ValueError("GPU result completed-epoch count exceeds the fixed run")
        handoff_binding = _verify_expected_handoff_result(
            root,
            manifest,
            expected_handoff=expected_handoff,
            teacher_model=teacher_model,
            dataset=dataset,
        )
    return {
        "root": str(root),
        "teacherSha256": teacher["sha256"],
        "datasetSha256": dataset_binding["sha256"],
        "samples": dataset_binding["samples"],
        "temperature": teacher["temperature"],
        "completedEpochs": completed_epochs,
        "bestEpoch": best_epoch,
        "bestValidation": manifest["bestValidation"],
        "modelSha256": file_sha256(root / "v3-actor-critic-weights.json"),
        "hardware": manifest.get("hardware"),
        "reproducibility": manifest.get("reproducibility"),
        "strictInventoryVerified": inventory is not None,
        "expectedHandoffVerified": handoff_binding is not None,
        "handoff": handoff_binding,
    }


def _safe_archive_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe archive path: {value}")
    return path


def verify_result_archive(
    archive_path: str | Path,
    checksum_path: str | Path,
    *,
    extract_dir: str | Path | None = None,
    teacher_model: str | Path | None = None,
    dataset: str | Path | None = None,
    expected_handoff: str | Path | None = None,
    allow_legacy_inventory: bool = False,
) -> dict[str, object]:
    archive = Path(archive_path).resolve()
    checksum = Path(checksum_path).resolve()
    parts = checksum.read_text(encoding="ascii").split()
    actual_sha256 = file_sha256(archive)
    if (
        len(parts) != 2
        or parts[1] != archive.name
        or parts[0] != actual_sha256
    ):
        raise ValueError("V3 distillation archive checksum mismatch")
    with zipfile.ZipFile(archive, "r") as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate paths")
        for name in names:
            _safe_archive_path(name)
        if PACKAGE_MANIFEST_NAME not in names:
            raise ValueError("package manifest is missing")
        package_manifest = json.loads(bundle.read(PACKAGE_MANIFEST_NAME))
        if (
            not isinstance(package_manifest, dict)
            or set(package_manifest)
            != {
                "format",
                "version",
                "archiveRoot",
                "teacherSha256",
                "datasetSha256",
                "modelSha256",
                "files",
                "totalBytes",
            }
            or package_manifest.get("format") != PACKAGE_FORMAT
            or package_manifest.get("version") != PACKAGE_VERSION
            or package_manifest.get("archiveRoot") != ARCHIVE_ROOT
        ):
            raise ValueError("unsupported V3 distillation result package")
        entries = package_manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("package contains no files")
        expected_names = {PACKAGE_MANIFEST_NAME}
        total_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
                raise ValueError("invalid package file entry")
            relative = _safe_archive_path(entry["path"])
            name = f"{ARCHIVE_ROOT}/{relative.as_posix()}"
            expected_names.add(name)
            if name not in names:
                raise ValueError(f"package file is missing: {name}")
            payload = bundle.read(name)
            if (
                len(payload) != entry["bytes"]
                or hashlib.sha256(payload).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"package file binding mismatch: {name}")
            total_bytes += len(payload)
        if set(names) != expected_names:
            raise ValueError("archive contains unmanifested files")
        if package_manifest.get("totalBytes") != total_bytes:
            raise ValueError("package total byte count mismatch")
        with tempfile.TemporaryDirectory(prefix="dalmuti-v3-distill-verify-") as temporary:
            temporary_root = Path(temporary) / ARCHIVE_ROOT
            for entry in entries:
                target = temporary_root.joinpath(*PurePosixPath(entry["path"]).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(bundle.read(f"{ARCHIVE_ROOT}/{entry['path']}"))
            report = verify_result_directory(
                temporary_root,
                teacher_model=teacher_model,
                dataset=dataset,
                expected_handoff=expected_handoff,
                allow_legacy_inventory=allow_legacy_inventory,
            )
        if package_manifest.get("modelSha256") != report["modelSha256"]:
            raise ValueError("package final-model binding mismatch")
        if (
            package_manifest.get("teacherSha256") != report["teacherSha256"]
            or package_manifest.get("datasetSha256") != report["datasetSha256"]
        ):
            raise ValueError("package training-provenance binding mismatch")
        if extract_dir is not None:
            destination = Path(extract_dir).resolve()
            if destination.exists():
                raise FileExistsError(
                    f"V3 distillation extraction directory must be fresh: {destination}"
                )
            destination.mkdir(parents=True, exist_ok=False)
            for entry in entries:
                target = destination.joinpath(*PurePosixPath(entry["path"]).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(bundle.read(f"{ARCHIVE_ROOT}/{entry['path']}"))
    return {
        **report,
        "archive": str(archive),
        "archiveSha256": actual_sha256,
        "archiveBytes": archive.stat().st_size,
        "files": len(entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a V3 distillation result directory or package."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--result-dir")
    target.add_argument("--archive")
    parser.add_argument("--checksum")
    parser.add_argument("--extract-dir")
    parser.add_argument("--teacher-model")
    parser.add_argument("--data")
    parser.add_argument(
        "--expected-handoff",
        help=(
            "extracted GPU handoff root; enables exact CUDA, argument, split, "
            "run-id, and five-file provenance verification"
        ),
    )
    parser.add_argument(
        "--allow-legacy-inventory",
        action="store_true",
        help="explicitly accept a pre-inventory local artifact (never for GPU handoff)",
    )
    args = parser.parse_args()
    if args.result_dir:
        if args.checksum or args.extract_dir:
            parser.error("--result-dir cannot use --checksum or --extract-dir")
        report = verify_result_directory(
            args.result_dir,
            teacher_model=args.teacher_model,
            dataset=args.data,
            expected_handoff=args.expected_handoff,
            allow_legacy_inventory=args.allow_legacy_inventory,
        )
    else:
        if not args.checksum:
            parser.error("--archive requires --checksum")
        report = verify_result_archive(
            args.archive,
            args.checksum,
            extract_dir=args.extract_dir,
            teacher_model=args.teacher_model,
            dataset=args.data,
            expected_handoff=args.expected_handoff,
            allow_legacy_inventory=args.allow_legacy_inventory,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
