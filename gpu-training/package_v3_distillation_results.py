from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from v3_distillation_dataset import file_sha256
from verify_v3_distillation_results import (
    ARCHIVE_ROOT,
    PACKAGE_FORMAT,
    PACKAGE_MANIFEST_NAME,
    PACKAGE_VERSION,
    verify_result_archive,
    verify_result_directory,
)


def _archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def package_result(
    result_directory: str | Path,
    output_archive: str | Path,
    *,
    teacher_model: str | Path | None = None,
    dataset: str | Path | None = None,
    expected_handoff: str | Path | None = None,
    allow_legacy_inventory: bool = False,
) -> dict[str, object]:
    root = Path(result_directory).resolve()
    archive = Path(output_archive).resolve()
    checksum = archive.with_suffix(f"{archive.suffix}.sha256")
    if archive.suffix.lower() != ".zip":
        raise ValueError("output archive must end in .zip")
    if archive.exists() or checksum.exists():
        raise FileExistsError(
            f"package outputs must be fresh: {archive}, {checksum}"
        )
    try:
        archive.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("output archive must be outside the result directory")
    report = verify_result_directory(
        root,
        teacher_model=teacher_model,
        dataset=dataset,
        expected_handoff=expected_handoff,
        allow_legacy_inventory=allow_legacy_inventory,
    )
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    package_manifest = {
        "format": PACKAGE_FORMAT,
        "version": PACKAGE_VERSION,
        "archiveRoot": ARCHIVE_ROOT,
        "teacherSha256": report["teacherSha256"],
        "datasetSha256": report["datasetSha256"],
        "modelSha256": report["modelSha256"],
        "files": entries,
        "totalBytes": sum(entry["bytes"] for entry in entries),
    }
    manifest_bytes = (
        json.dumps(
            package_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as bundle:
        bundle.writestr(_archive_info(PACKAGE_MANIFEST_NAME), manifest_bytes)
        for path, entry in zip(files, entries):
            name = f"{ARCHIVE_ROOT}/{entry['path']}"
            with path.open("rb") as source, bundle.open(
                _archive_info(name), "w"
            ) as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    archive_sha256 = file_sha256(archive)
    with checksum.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{archive_sha256}  {archive.name}\n")
    verified = verify_result_archive(
        archive,
        checksum,
        teacher_model=teacher_model,
        dataset=dataset,
        expected_handoff=expected_handoff,
        allow_legacy_inventory=allow_legacy_inventory,
    )
    return {**verified, "checksum": str(checksum)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a strict V3 distillation warm-start result."
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model")
    parser.add_argument("--data")
    parser.add_argument("--expected-handoff")
    parser.add_argument("--allow-legacy-inventory", action="store_true")
    args = parser.parse_args()
    report = package_result(
        args.result_dir,
        args.output,
        teacher_model=args.teacher_model,
        dataset=args.data,
        expected_handoff=args.expected_handoff,
        allow_legacy_inventory=args.allow_legacy_inventory,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
