"""Create and verify an exclusive non-card training result ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from non_card_counterfactual_dataset import file_sha256
from verify_non_card_results import verify_result_directory


PACKAGE_FORMAT = "dalmuti-non-card-supervised-result-package"
PACKAGE_VERSION = 3
SUPPORTED_PACKAGE_VERSIONS = tuple(range(1, PACKAGE_VERSION + 1))
PACKAGE_MANIFEST_NAME = "package-manifest.json"
ARCHIVE_ROOT = "result"


def _archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe archive path: {name}")
    return path


def _hash_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def verify_result_archive(path: str | Path) -> dict[str, object]:
    archive = Path(path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"result archive does not exist: {archive}")
    with zipfile.ZipFile(archive, "r") as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("result archive contains duplicate paths")
        for name in names:
            _safe_name(name)
        if PACKAGE_MANIFEST_NAME not in names:
            raise ValueError("result archive lacks package-manifest.json")
        try:
            package_manifest = json.loads(
                bundle.read(PACKAGE_MANIFEST_NAME).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid package manifest JSON") from error
        if not isinstance(package_manifest, dict):
            raise ValueError("package manifest must be an object")
        package_version = package_manifest.get("version")
        if package_version not in SUPPORTED_PACKAGE_VERSIONS:
            raise ValueError("unsupported non-card result package version")
        package_keys = {
            "format",
            "version",
            "archiveRoot",
            "trainingManifestSha256",
            "files",
            "totalBytes",
        }
        if package_version >= 2:
            package_keys.add("behaviorCloningCoefficient")
        if package_version >= 3:
            package_keys.add("utilityTarget")
        if set(package_manifest) != package_keys:
            raise ValueError("package manifest fields mismatch")
        if (
            package_manifest["format"] != PACKAGE_FORMAT
            or package_manifest["archiveRoot"] != ARCHIVE_ROOT
        ):
            raise ValueError("unsupported non-card result package")
        entries = package_manifest["files"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("package manifest contains no result files")
        expected_names = {PACKAGE_MANIFEST_NAME}
        total_bytes = 0
        training_manifest_sha256: str | None = None
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
                raise ValueError("invalid package file entry")
            relative = _safe_name(entry["path"])
            archive_name = f"{ARCHIVE_ROOT}/{relative.as_posix()}"
            if archive_name in expected_names:
                raise ValueError(f"duplicate package file path: {archive_name}")
            expected_names.add(archive_name)
            if archive_name not in names:
                raise FileNotFoundError(f"package file is missing: {archive_name}")
            with bundle.open(archive_name, "r") as stream:
                digest, size = _hash_stream(stream)
            if entry["bytes"] != size or entry["sha256"] != digest:
                raise ValueError(f"package file hash or size mismatch: {archive_name}")
            if relative.as_posix() == "training-manifest.json":
                training_manifest_sha256 = digest
            total_bytes += size
        if set(names) != expected_names:
            raise ValueError("result archive contains unmanifested files")
        if package_manifest["totalBytes"] != total_bytes:
            raise ValueError("package total byte count mismatch")
        if training_manifest_sha256 != package_manifest["trainingManifestSha256"]:
            raise ValueError("packaged training manifest binding mismatch")

        # Full schema, PT/JSON parity, and internal result hash verification.
        with tempfile.TemporaryDirectory(prefix="dalmuti-non-card-verify-") as directory:
            extraction_root = Path(directory)
            for info in bundle.infolist():
                destination = extraction_root.joinpath(*PurePosixPath(info.filename).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info, "r") as source, destination.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            result_report = verify_result_directory(extraction_root / ARCHIVE_ROOT)
        package_behavior_coefficient = (
            package_manifest["behaviorCloningCoefficient"]
            if package_version >= 2
            else 0.0
        )
        if (
            package_behavior_coefficient
            != result_report["behaviorCloningCoefficient"]
        ):
            raise ValueError(
                "package behavior-cloning coefficient binding mismatch"
            )
        package_utility_target = (
            package_manifest["utilityTarget"]
            if package_version >= 3
            else "terminal"
        )
        if package_utility_target != result_report["utilityTarget"]:
            raise ValueError("package utility-target binding mismatch")
    return {
        "archive": str(archive),
        "sha256": file_sha256(archive),
        "bytes": archive.stat().st_size,
        "files": len(entries),
        "decisionKinds": result_report["decisionKinds"],
        "trainingManifestSha256": training_manifest_sha256,
        "behaviorCloningCoefficient": result_report[
            "behaviorCloningCoefficient"
        ],
        "packageVersion": package_version,
        "utilityTarget": result_report["utilityTarget"],
    }


def package_result_directory(
    result_directory: str | Path,
    output_archive: str | Path,
) -> dict[str, object]:
    root = Path(result_directory).resolve()
    archive = Path(output_archive).resolve()
    checksum_path = archive.with_suffix(f"{archive.suffix}.sha256")
    if archive.suffix.lower() != ".zip":
        raise ValueError("output archive must end in .zip")
    if archive.exists() or checksum_path.exists():
        raise FileExistsError(
            f"package outputs must not exist: {archive}, {checksum_path}"
        )
    try:
        archive.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("output archive must be outside the result directory")
    verification = verify_result_directory(root)
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
        "trainingManifestSha256": verification["manifestSha256"],
        "behaviorCloningCoefficient": verification[
            "behaviorCloningCoefficient"
        ],
        "utilityTarget": verification["utilityTarget"],
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
            with path.open("rb") as source, bundle.open(_archive_info(name), "w") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    archive_report = verify_result_archive(archive)
    checksum = archive_report["sha256"]
    with checksum_path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{checksum}  {archive.name}\n")
    if file_sha256(archive) != checksum:
        raise RuntimeError("archive changed after verification")
    return {
        **archive_report,
        "checksumFile": str(checksum_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package or verify a DALMUTI non-card training result ZIP."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--result-dir")
    action.add_argument("--verify-archive")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.verify_archive:
        if args.output is not None:
            parser.error("--output cannot be used with --verify-archive")
        report = verify_result_archive(args.verify_archive)
        print(
            f"Non-card result package verified: {report['files']} files, "
            f"{report['bytes'] / 1024 / 1024:.2f} MiB"
        )
    else:
        if args.output is None:
            parser.error("--result-dir requires --output")
        report = package_result_directory(args.result_dir, args.output)
        print(f"Non-card result archive: {report['archive']}")
        print(f"SHA-256: {report['sha256']}")
        print(f"Checksum file: {report['checksumFile']}")


if __name__ == "__main__":
    main()
