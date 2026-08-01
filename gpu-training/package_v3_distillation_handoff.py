from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from actor_critic import load_behavior_model
from v3_distillation_dataset import file_sha256
from verify_v3_distillation_bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    CHECKSUMS_NAME,
    EXPECTED_DATASET_SHA256,
    EXPECTED_LOCAL_PREPARATION_RUN_ID,
    EXPECTED_RESULT_CONTRACT,
    EXPECTED_SOURCE_FILES,
    EXPECTED_SPLIT,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_TRAINING,
    MANIFEST_NAME,
    expected_gpu_run_config,
    verify_bundle,
)


CODE_FILES = (
    "actor_critic.py",
    "ppo_dataset.py",
    "v3_action_conditioned.py",
    "v3_ppo_dataset.py",
    "v3_distillation_dataset.py",
    "prepare_v3_distillation_data.py",
    "verify_v3_distillation_data.py",
    "train_v3_distillation.py",
    "verify_v3_distillation_results.py",
    "package_v3_distillation_results.py",
    "verify_v3_distillation_bundle.py",
    "test_v3_action_conditioned.py",
    "test_v3_distillation_pipeline.py",
    "preflight.py",
)
ROOT_FILES = (
    "requirements.txt",
    "v3-distillation-schema.json",
    "PROMPT_FOR_GPU_V3_DISTILLATION.md",
)
PYTHON_BYTECODE_GUARD = "export PYTHONDONTWRITEBYTECODE=1"


def _validate_runtime_prompt(path: Path) -> None:
    prompt = path.read_text(encoding="utf-8")
    guard_position = prompt.find(PYTHON_BYTECODE_GUARD)
    first_python_process = prompt.find('"$PY"')
    if (
        guard_position < 0
        or first_python_process < 0
        or guard_position > first_python_process
    ):
        raise ValueError(
            "GPU prompt must export PYTHONDONTWRITEBYTECODE=1 before the "
            "first Python process"
        )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _hash_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def _entry(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _validate_inputs(
    data_path: Path,
    teacher_path: Path,
    summary_path: Path,
    verification_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sidecar = data_path.with_suffix(f"{data_path.suffix}.sha256")
    parts = sidecar.read_text(encoding="ascii").split()
    data_sha256 = file_sha256(data_path)
    teacher_sha256 = file_sha256(teacher_path)
    if len(parts) != 2 or parts != [data_sha256, data_path.name]:
        raise ValueError("distillation dataset checksum sidecar mismatch")
    teacher_model, teacher_payload = load_behavior_model(teacher_path)
    del teacher_model
    if (
        teacher_payload.get("format") != "dalmuti-actor-critic"
        or teacher_payload.get("version") != 1
        or teacher_payload.get("observationFeatures") != 172
        or teacher_payload.get("actionCount") != 506
        or teacher_sha256 != EXPECTED_TEACHER_SHA256
        or data_sha256 != EXPECTED_DATASET_SHA256
    ):
        raise ValueError("unexpected legacy PPO4 teacher contract")
    with data_path.open("r", encoding="utf-8") as stream:
        dataset_manifest = json.loads(stream.readline())
    summary = _read_object(summary_path)
    verification = _read_object(verification_path)
    sources = dataset_manifest.get("sources")
    source_counts = summary.get("sourceSampleCounts")
    if (
        dataset_manifest.get("type") != "manifest"
        or dataset_manifest.get("format") != "dalmuti-v3-distillation-ndjson"
        or dataset_manifest.get("formatVersion") != 1
        or dataset_manifest.get("teacher", {}).get("sha256") != teacher_sha256
        or dataset_manifest.get("teacher", {}).get("temperature") != 2.5
        or dataset_manifest.get("actionSpace", {}).get("size") != 236
        or dataset_manifest.get("actionSpace", {}).get("catalogueVersion") != 1
        or dataset_manifest.get("selection", {}).get("maxSamplesPerSource")
        != 20000
        or dataset_manifest.get("selection", {}).get("includeForced") is not False
        or not isinstance(sources, list)
        or sorted(source.get("playerCount") for source in sources)
        != [4, 5, 6, 7, 8, 9, 10]
        or [
            {
                "filename": source.get("filename"),
                "bytes": source.get("bytes"),
                "sha256": source.get("sha256"),
                "playerCount": source.get("playerCount"),
                "selectedSamples": source_counts.get(source.get("sha256"))
                if isinstance(source_counts, dict)
                else None,
            }
            for source in sources
        ]
        != EXPECTED_SOURCE_FILES
        or summary.get("datasetSha256") != data_sha256
        or summary.get("datasetBytes") != data_path.stat().st_size
        or summary.get("teacherSha256") != teacher_sha256
        or summary.get("samples") != 140000
        or summary.get("uniqueGroups") != 1793
        or not isinstance(source_counts, dict)
        or source_counts
        != {
            source["sha256"]: source["selectedSamples"]
            for source in EXPECTED_SOURCE_FILES
        }
        or verification.get("format")
        != "dalmuti-v3-distillation-data-verification"
        or verification.get("data", {}).get("sha256") != data_sha256
        or verification.get("teacher", {}).get("sha256") != teacher_sha256
        or verification.get("samples") != 140000
        or verification.get("uniqueEpisodeGroups") != 1793
        or verification.get("split", {}).get("train", {}).get("samples")
        != 117737
        or verification.get("split", {}).get("validation", {}).get("samples")
        != 22263
        or verification.get("split", {}).get("overlappingGroups") != 0
        or verification.get("finite") is not True
    ):
        raise ValueError("full distillation input contract mismatch")
    return dataset_manifest, summary, verification


def package_handoff(
    *,
    data: str | Path,
    teacher_model: str | Path,
    data_verification: str | Path,
    output_directory: str | Path,
    output_archive: str | Path,
    run_id: str,
) -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    data_path = Path(data).resolve()
    teacher_path = Path(teacher_model).resolve()
    verification_path = Path(data_verification).resolve()
    summary_path = data_path.parent / "dataset-summary.json"
    dataset_manifest, summary, verification = _validate_inputs(
        data_path, teacher_path, summary_path, verification_path
    )
    output = Path(output_directory).resolve()
    archive = Path(output_archive).resolve()
    archive_checksum = archive.with_suffix(f"{archive.suffix}.sha256")
    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run-id must be a non-empty token")
    if archive.suffix.lower() != ".zip":
        raise ValueError("handoff archive must end in .zip")
    if output.exists() or archive.exists() or archive_checksum.exists():
        raise FileExistsError(
            "handoff directory, ZIP, and checksum must all be fresh"
        )
    _validate_runtime_prompt(
        package_root / "PROMPT_FOR_GPU_V3_DISTILLATION.md"
    )
    try:
        archive.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("handoff archive must be outside the bundle directory")
    output.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []
    for filename in ROOT_FILES:
        destination = output / filename
        _copy(package_root / filename, destination)
        copied.append(destination)
    for filename in CODE_FILES:
        destination = output / "code" / filename
        _copy(package_root / filename, destination)
        copied.append(destination)
    input_files = (
        (teacher_path, output / "input" / "ppo4-actor-critic-weights.json"),
        (data_path, output / "input" / "v3-distillation.ndjson"),
        (
            data_path.with_suffix(f"{data_path.suffix}.sha256"),
            output / "input" / "v3-distillation.ndjson.sha256",
        ),
        (summary_path, output / "input" / "dataset-summary.json"),
        (verification_path, output / "input" / "data-verification.json"),
    )
    for source, destination in input_files:
        _copy(source, destination)
        copied.append(destination)
    run_config = expected_gpu_run_config(run_id)
    run_config_path = output / "gpu-run-config.json"
    with run_config_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")
    copied.append(run_config_path)
    entries = [
        _entry(output, path)
        for path in sorted(
            copied, key=lambda value: value.relative_to(output).as_posix()
        )
    ]
    teacher_sha256 = file_sha256(teacher_path)
    dataset_sha256 = file_sha256(data_path)
    manifest = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "teacher": {
            "path": "input/ppo4-actor-critic-weights.json",
            "bytes": teacher_path.stat().st_size,
            "sha256": teacher_sha256,
            "format": "dalmuti-actor-critic",
            "version": 1,
            "observationFeatures": 172,
            "actionCount": 506,
        },
        "dataset": {
            "path": "input/v3-distillation.ndjson",
            "bytes": data_path.stat().st_size,
            "sha256": dataset_sha256,
            "format": "dalmuti-v3-distillation-ndjson",
            "formatVersion": 1,
            "localPreparationRunId": EXPECTED_LOCAL_PREPARATION_RUN_ID,
            "samples": summary["samples"],
            "uniqueGroups": summary["uniqueGroups"],
            "sourcePlayerCounts": sorted(
                source["playerCount"] for source in dataset_manifest["sources"]
            ),
            "samplesPerSource": 20000,
            "includeForced": False,
            "temperature": 2.5,
            "sourceFiles": [
                {
                    "filename": source["filename"],
                    "bytes": source["bytes"],
                    "sha256": source["sha256"],
                    "playerCount": source["playerCount"],
                    "selectedSamples": summary["sourceSampleCounts"][
                        source["sha256"]
                    ],
                }
                for source in dataset_manifest["sources"]
            ],
        },
        "split": dict(EXPECTED_SPLIT),
        "training": dict(EXPECTED_TRAINING),
        "resultContract": dict(EXPECTED_RESULT_CONTRACT),
        "files": entries,
        "totalBytes": sum(int(entry["bytes"]) for entry in entries),
    }
    manifest_path = output / MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    checksum_entries = [
        (MANIFEST_NAME, file_sha256(manifest_path)),
        *((entry["path"], entry["sha256"]) for entry in entries),
    ]
    checksums_path = output / CHECKSUMS_NAME
    with checksums_path.open("x", encoding="ascii", newline="\n") as stream:
        for path, digest in sorted(checksum_entries):
            stream.write(f"{digest}  {path}\n")
    bundle_report = verify_bundle(output, verify_teacher_bindings=False)

    archive.parent.mkdir(parents=True, exist_ok=True)
    all_files = sorted(
        (path for path in output.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(output).as_posix(),
    )
    with zipfile.ZipFile(
        archive,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as bundle:
        for path in all_files:
            relative = path.relative_to(output).as_posix()
            with path.open("rb") as source, bundle.open(
                _archive_info(relative), "w"
            ) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    expected_names = {
        path.relative_to(output).as_posix() for path in all_files
    }
    with zipfile.ZipFile(archive, "r") as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError("handoff ZIP paths do not match the bundle directory")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError(f"unsafe handoff ZIP path: {name}")
            with bundle.open(name, "r") as stream:
                digest, size = _hash_stream(stream)
            source = output.joinpath(*relative.parts)
            if digest != file_sha256(source) or size != source.stat().st_size:
                raise ValueError(f"handoff ZIP file binding mismatch: {name}")
    archive_sha256 = file_sha256(archive)
    with archive_checksum.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{archive_sha256}  {archive.name}\n")
    if file_sha256(archive) != archive_sha256:
        raise RuntimeError("handoff archive changed after checksum creation")
    return {
        **bundle_report,
        "directory": str(output),
        "directoryBytes": sum(path.stat().st_size for path in all_files),
        "archive": str(archive),
        "archiveBytes": archive.stat().st_size,
        "archiveSha256": archive_sha256,
        "archiveChecksum": str(archive_checksum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exclusive V3 distillation GPU handoff ZIP."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--data-verification", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = package_handoff(
        data=args.data,
        teacher_model=args.teacher_model,
        data_verification=args.data_verification,
        output_directory=args.output_dir,
        output_archive=args.archive,
        run_id=args.run_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
