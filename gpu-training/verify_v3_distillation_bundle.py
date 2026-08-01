from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from v3_distillation_dataset import (  # noqa: E402
    file_sha256,
    group_split_mask,
    load_v3_distillation_data,
)


BUNDLE_FORMAT = "dalmuti-v3-distillation-gpu-handoff"
BUNDLE_VERSION = 1
MANIFEST_NAME = "bundle-manifest.json"
CHECKSUMS_NAME = "handoff-files.sha256"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TEACHER_SHA256 = (
    "3a8bc15ee05305e4cd8f9e6710cb8e927a54e0a3acf6ae0927ffabe50318535f"
)
EXPECTED_DATASET_SHA256 = (
    "cac0fa2c98592c48c3f0fffe94a77f193f0e56d833fd794c7d056b7afeb373bb"
)
EXPECTED_LOCAL_PREPARATION_RUN_ID = (
    "v3-warmstart-distill-ppo4-t25-seed-202608071-run-002"
)
EXPECTED_SOURCE_FILES = [
    {
        "filename": "ppo-i5-mc-temp25-p10.ndjson",
        "bytes": 570036452,
        "sha256": "6c27ad8c29e66a6da35316e990f40bb77ef12ffc2d930f67ba128fc3207e95a6",
        "playerCount": 10,
        "selectedSamples": 20000,
    },
    {
        "filename": "ppo-i5-mc-temp25-p4.ndjson",
        "bytes": 293204461,
        "sha256": "3c0f0629126f9ecb66897f06839638c93bbec2473f0bb91b4cbe1588aba7b548",
        "playerCount": 4,
        "selectedSamples": 20000,
    },
    {
        "filename": "ppo-i5-mc-temp25-p5.ndjson",
        "bytes": 340116127,
        "sha256": "b482b6592f84d94f5c4492012e9f4ca6570660cb40986085c315c795da8095aa",
        "playerCount": 5,
        "selectedSamples": 20000,
    },
    {
        "filename": "ppo-i5-mc-temp25-p6.ndjson",
        "bytes": 384422317,
        "sha256": "be5c4c3041f597bbb1618098059b426cab59156e5295f1d918abf3d074a5f971",
        "playerCount": 6,
        "selectedSamples": 20000,
    },
    {
        "filename": "ppo-i5-mc-temp25-p7.ndjson",
        "bytes": 433487560,
        "sha256": "e3cd3e4628ed07171f6131f3bfd899a1c7152d5fda56d334bea70094e38d3be2",
        "playerCount": 7,
        "selectedSamples": 20000,
    },
    {
        "filename": "ppo-i5-mc-temp25-p8.ndjson",
        "bytes": 484781073,
        "sha256": "a22e3277611677247641f0f3dc9db2b87fca3346f8b15d165102858b1d82ebc0",
        "playerCount": 8,
        "selectedSamples": 20000,
    },
    {
        "filename": "ppo-i5-mc-temp25-p9.ndjson",
        "bytes": 525117694,
        "sha256": "73fe24996f63ec5534c7a58d6fb71fc699ef9b36ec295c4a5444812fdf37da76",
        "playerCount": 9,
        "selectedSamples": 20000,
    },
]
EXPECTED_SPLIT = {
    "groupSplitKey": "sourceSha256:episodeId",
    "validationFraction": 0.15,
    "splitSeed": 20260801,
    "trainSamples": 117737,
    "validationSamples": 22263,
    "trainGroups": 1510,
    "validationGroups": 283,
    "overlappingGroups": 0,
}
EXPECTED_TRAINING = {
    "epochs": 50,
    "batchSize": 512,
    "learningRate": 0.0003,
    "weightDecay": 0.00001,
    "valueCoefficient": 0.25,
    "validationFraction": 0.15,
    "splitSeed": 20260801,
    "seed": 202608071,
    "patience": 8,
    "maxGradientNorm": 1,
    "bindingTolerance": 0.00002,
    "device": "cuda",
    "deterministic": True,
}
EXPECTED_RESULT_CONTRACT = {
    "format": "dalmuti-v3-distillation-result-package",
    "requiredReturnFiles": ["result.zip", "result.zip.sha256"],
    "freshRemoteWorkDirectoryRequired": True,
    "deleteRemoteOnlyAfterLocalVerification": True,
}


def expected_gpu_run_config(run_id: str) -> dict[str, object]:
    return {
        "format": "dalmuti-v3-distillation-gpu-run-config",
        "version": 1,
        "runId": run_id,
        "requiredCommandArguments": [
            "--data",
            "input/v3-distillation.ndjson",
            "--teacher-model",
            "input/ppo4-actor-critic-weights.json",
            "--output",
            "<fresh-run>/model",
            "--epochs",
            "50",
            "--batch-size",
            "512",
            "--learning-rate",
            "0.0003",
            "--weight-decay",
            "0.00001",
            "--value-coefficient",
            "0.25",
            "--validation-fraction",
            "0.15",
            "--split-seed",
            "20260801",
            "--seed",
            "202608071",
            "--patience",
            "8",
            "--max-gradient-norm",
            "1",
            "--binding-tolerance",
            "0.00002",
            "--device",
            "cuda",
            "--deterministic",
        ],
    }


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("bundle path must be a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe bundle path: {value}")
    return path


def _require_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def _parse_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or not SHA256_RE.fullmatch(parts[0])
            or parts[1] in result
        ):
            raise ValueError(
                f"malformed handoff checksum line {line_number}"
            )
        relative = _safe_relative_path(parts[1]).as_posix()
        result[relative] = parts[0]
    if not result:
        raise ValueError("handoff checksum file is empty")
    return result


def verify_bundle(
    root_path: str | Path,
    *,
    verify_teacher_bindings: bool = True,
) -> dict[str, object]:
    root = Path(root_path).resolve()
    manifest_path = root / MANIFEST_NAME
    checksums_path = root / CHECKSUMS_NAME
    manifest = _read_object(manifest_path)
    _require_keys(
        manifest,
        {
            "format",
            "version",
            "createdAt",
            "runId",
            "teacher",
            "dataset",
            "split",
            "training",
            "resultContract",
            "files",
            "totalBytes",
        },
        "bundle manifest",
    )
    if (
        manifest["format"] != BUNDLE_FORMAT
        or manifest["version"] != BUNDLE_VERSION
        or not isinstance(manifest["runId"], str)
        or not manifest["runId"]
    ):
        raise ValueError("unsupported V3 distillation bundle")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("bundle manifest contains no files")
    expected_files = {MANIFEST_NAME, CHECKSUMS_NAME}
    checked_bytes = 0
    listed_hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("bundle file entry must be an object")
        _require_keys(entry, {"path", "bytes", "sha256"}, "bundle file")
        relative = _safe_relative_path(entry["path"])
        relative_string = relative.as_posix()
        if relative_string in expected_files:
            raise ValueError(f"duplicate bundle path: {relative_string}")
        expected_files.add(relative_string)
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"bundle file is missing or is a symlink: {relative_string}"
            )
        size = path.stat().st_size
        digest = file_sha256(path)
        if entry["bytes"] != size or entry["sha256"] != digest:
            raise ValueError(f"bundle file binding mismatch: {relative_string}")
        listed_hashes[relative_string] = digest
        checked_bytes += size
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("bundle contains missing or unmanifested files")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("bundle must not contain symbolic links")
    if manifest["totalBytes"] != checked_bytes:
        raise ValueError("bundle total byte count mismatch")
    checksums = _parse_checksums(checksums_path)
    expected_checksums = {
        MANIFEST_NAME: file_sha256(manifest_path),
        **listed_hashes,
    }
    if checksums != expected_checksums:
        raise ValueError("handoff-files.sha256 does not match the bundle")

    teacher = manifest["teacher"]
    dataset = manifest["dataset"]
    split = manifest["split"]
    training = manifest["training"]
    result_contract = manifest["resultContract"]
    if not all(
        isinstance(value, dict)
        for value in (teacher, dataset, split, training, result_contract)
    ):
        raise ValueError("bundle contracts must be objects")
    _require_keys(
        dataset,
        {
            "path",
            "bytes",
            "sha256",
            "format",
            "formatVersion",
            "localPreparationRunId",
            "samples",
            "uniqueGroups",
            "sourcePlayerCounts",
            "samplesPerSource",
            "includeForced",
            "temperature",
            "sourceFiles",
        },
        "bundle dataset contract",
    )
    teacher_path = root.joinpath(*_safe_relative_path(teacher.get("path")).parts)
    dataset_path = root.joinpath(*_safe_relative_path(dataset.get("path")).parts)
    if (
        teacher
        != {
            "path": "input/ppo4-actor-critic-weights.json",
            "bytes": 2732440,
            "sha256": EXPECTED_TEACHER_SHA256,
            "format": "dalmuti-actor-critic",
            "version": 1,
            "observationFeatures": 172,
            "actionCount": 506,
        }
        or teacher.get("sha256") != file_sha256(teacher_path)
        or teacher.get("bytes") != teacher_path.stat().st_size
        or teacher.get("sha256") != EXPECTED_TEACHER_SHA256
        or dataset.get("sha256") != file_sha256(dataset_path)
        or dataset.get("bytes") != dataset_path.stat().st_size
        or dataset.get("sha256") != EXPECTED_DATASET_SHA256
        or dataset.get("format") != "dalmuti-v3-distillation-ndjson"
        or dataset.get("formatVersion") != 1
        or dataset.get("localPreparationRunId")
        != EXPECTED_LOCAL_PREPARATION_RUN_ID
        or dataset.get("samples") != 140000
        or dataset.get("uniqueGroups") != 1793
        or dataset.get("sourcePlayerCounts") != [4, 5, 6, 7, 8, 9, 10]
        or dataset.get("samplesPerSource") != 20000
        or dataset.get("includeForced") is not False
        or dataset.get("temperature") != 2.5
        or dataset.get("sourceFiles") != EXPECTED_SOURCE_FILES
        or split != EXPECTED_SPLIT
        or training != EXPECTED_TRAINING
        or result_contract != EXPECTED_RESULT_CONTRACT
    ):
        raise ValueError("fixed teacher, dataset, split, or training contract mismatch")
    run_config = _read_object(root / "gpu-run-config.json")
    if run_config != expected_gpu_run_config(str(manifest["runId"])):
        raise ValueError("GPU run configuration does not match the fixed contract")
    sidecar = dataset_path.with_suffix(f"{dataset_path.suffix}.sha256")
    sidecar_parts = sidecar.read_text(encoding="ascii").split()
    if (
        len(sidecar_parts) != 2
        or sidecar_parts[0] != dataset["sha256"]
        or sidecar_parts[1] != dataset_path.name
    ):
        raise ValueError("distillation dataset sidecar mismatch")
    summary_path = root / "input" / "dataset-summary.json"
    verification_path = root / "input" / "data-verification.json"
    summary = _read_object(summary_path)
    verification = _read_object(verification_path)
    source_counts = summary.get("sourceSampleCounts")
    if (
        summary.get("datasetSha256") != dataset["sha256"]
        or summary.get("datasetBytes") != dataset["bytes"]
        or summary.get("teacherSha256") != teacher["sha256"]
        or summary.get("samples") != dataset["samples"]
        or summary.get("uniqueGroups") != dataset["uniqueGroups"]
        or source_counts
        != {
            source["sha256"]: source["selectedSamples"]
            for source in EXPECTED_SOURCE_FILES
        }
        or verification.get("format")
        != "dalmuti-v3-distillation-data-verification"
        or verification.get("data", {}).get("sha256") != dataset["sha256"]
        or verification.get("teacher", {}).get("sha256") != teacher["sha256"]
        or verification.get("samples") != dataset["samples"]
        or verification.get("uniqueEpisodeGroups") != dataset["uniqueGroups"]
        or verification.get("split")
        != {
            "groupSplitKey": EXPECTED_SPLIT["groupSplitKey"],
            "validationFraction": EXPECTED_SPLIT["validationFraction"],
            "splitSeed": EXPECTED_SPLIT["splitSeed"],
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
        or verification.get("finite") is not True
    ):
        raise ValueError("dataset summary or local verification binding mismatch")
    schema = _read_object(root / "v3-distillation-schema.json")
    if schema.get("$id") != "https://dclab.local/schemas/dalmuti-v3-distillation-v1.json":
        raise ValueError("distillation schema binding mismatch")

    if verify_teacher_bindings:
        loaded = load_v3_distillation_data(
            dataset_path,
            teacher_model_path=teacher_path,
            binding_tolerance=float(training["bindingTolerance"]),
            verify_teacher_bindings=True,
        )
        validation_mask = group_split_mask(
            loaded.group_keys,
            validation_fraction=float(split["validationFraction"]),
            seed=int(split["splitSeed"]),
        )
        train_groups = set(loaded.group_keys[~validation_mask].tolist())
        validation_groups = set(loaded.group_keys[validation_mask].tolist())
        if (
            len(loaded) != dataset["samples"]
            or int((~validation_mask).sum()) != split["trainSamples"]
            or int(validation_mask.sum()) != split["validationSamples"]
            or len(train_groups) != split["trainGroups"]
            or len(validation_groups) != split["validationGroups"]
            or train_groups & validation_groups
            or not np.isfinite(loaded.teacher_probabilities).all()
            or not np.isfinite(loaded.teacher_values).all()
        ):
            raise ValueError("full teacher binding or split verification failed")
    return {
        "root": str(root),
        "runId": manifest["runId"],
        "files": len(entries),
        "bytes": checked_bytes,
        "teacherSha256": teacher["sha256"],
        "datasetSha256": dataset["sha256"],
        "samples": dataset["samples"],
        "uniqueGroups": dataset["uniqueGroups"],
        "teacherBindingsVerified": verify_teacher_bindings,
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the exact DALMUTI V3 distillation GPU bundle."
    )
    parser.add_argument(
        "--root",
        default=str(CODE_ROOT.parent),
        help="extracted bundle root (defaults to the parent of code/)",
    )
    parser.add_argument(
        "--skip-teacher-bindings",
        action="store_true",
        help="verify byte/schema contracts without recomputing all teacher outputs",
    )
    args = parser.parse_args()
    report = verify_bundle(
        args.root,
        verify_teacher_bindings=not args.skip_teacher_bindings,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
