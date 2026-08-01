"""Create and verify exclusive tax-return advantage result packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from non_card_counterfactual_dataset import file_sha256
from verify_tax_return_advantage_results import verify_result_directory


PACKAGE_FORMAT = "dalmuti-tax-return-advantage-result-package"
PACKAGE_VERSION = 1
PACKAGE_MANIFEST = "package-manifest.json"
ARCHIVE_ROOT = "result"


def _safe_name(value: str) -> PurePosixPath:
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


def _info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _stream_hash(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _verify_external_checksum(archive: Path, checksum: str | Path) -> str:
    checksum_path = Path(checksum).resolve()
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"result archive checksum does not exist: {checksum_path}"
        )
    archive_sha256 = file_sha256(archive)
    expected = f"{archive_sha256}  {archive.name}\n"
    try:
        actual = checksum_path.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError("result archive checksum is not ASCII") from error
    if actual != expected:
        raise ValueError("result archive external checksum mismatch")
    return archive_sha256


def verify_result_archive(
    path: str | Path,
    checksum: str | Path | None = None,
) -> dict[str, object]:
    archive = Path(path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"result archive does not exist: {archive}")
    archive_sha256 = (
        _verify_external_checksum(archive, checksum)
        if checksum is not None
        else file_sha256(archive)
    )
    with zipfile.ZipFile(archive, "r") as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate paths")
        for name in names:
            _safe_name(name)
        if PACKAGE_MANIFEST not in names:
            raise ValueError("archive lacks package-manifest.json")
        try:
            manifest = json.loads(bundle.read(PACKAGE_MANIFEST).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid package manifest") from error
        if not isinstance(manifest, dict) or set(manifest) != {
            "format",
            "version",
            "archiveRoot",
            "trainingManifestSha256",
            "modelSha256",
            "scoreSemantics",
            "memberSeeds",
            "files",
            "totalBytes",
        }:
            raise ValueError("package manifest fields mismatch")
        if (
            manifest["format"] != PACKAGE_FORMAT
            or manifest["version"] != PACKAGE_VERSION
            or manifest["archiveRoot"] != ARCHIVE_ROOT
        ):
            raise ValueError("unsupported tax-return advantage package")
        entries = manifest["files"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("package contains no files")
        expected = {PACKAGE_MANIFEST}
        total_bytes = 0
        hashes: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
                raise ValueError("package file entry is invalid")
            relative = _safe_name(entry["path"])
            name = f"{ARCHIVE_ROOT}/{relative.as_posix()}"
            if name in expected or name not in names:
                raise ValueError(f"duplicate or missing package path: {name}")
            expected.add(name)
            with bundle.open(name, "r") as stream:
                sha256, size = _stream_hash(stream)
            if entry["sha256"] != sha256 or entry["bytes"] != size:
                raise ValueError(f"package file hash or size mismatch: {name}")
            hashes[relative.as_posix()] = sha256
            total_bytes += size
        if set(names) != expected or manifest["totalBytes"] != total_bytes:
            raise ValueError("package file inventory mismatch")
        if hashes.get("training-manifest.json") != manifest["trainingManifestSha256"]:
            raise ValueError("package training-manifest binding mismatch")
        if hashes.get("model.json") != manifest["modelSha256"]:
            raise ValueError("package model binding mismatch")
        with tempfile.TemporaryDirectory(prefix="dalmuti-tax-advantage-verify-") as directory:
            root = Path(directory)
            for info in bundle.infolist():
                destination = root.joinpath(*PurePosixPath(info.filename).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info, "r") as source, destination.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            report = verify_result_directory(root / ARCHIVE_ROOT)
        if (
            manifest["scoreSemantics"] != report["scoreSemantics"]
            or manifest["memberSeeds"] != report["memberSeeds"]
        ):
            raise ValueError("package provenance binding mismatch")
    return {
        "archive": str(archive),
        "sha256": archive_sha256,
        "bytes": archive.stat().st_size,
        "files": len(entries),
        "modelSha256": report["modelSha256"],
        "manifestSha256": report["manifestSha256"],
        "scoreSemantics": report["scoreSemantics"],
        "memberSeeds": report["memberSeeds"],
    }


def package_result_directory(result_directory: str | Path, output: str | Path) -> dict[str, object]:
    root = Path(result_directory).resolve()
    archive = Path(output).resolve()
    checksum_path = archive.with_suffix(f"{archive.suffix}.sha256")
    if archive.suffix.lower() != ".zip":
        raise ValueError("output archive must end in .zip")
    if archive.exists() or checksum_path.exists():
        raise FileExistsError("package outputs must not already exist")
    try:
        archive.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("output archive must be outside the result directory")
    report = verify_result_directory(root)
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
    manifest = {
        "format": PACKAGE_FORMAT,
        "version": PACKAGE_VERSION,
        "archiveRoot": ARCHIVE_ROOT,
        "trainingManifestSha256": report["manifestSha256"],
        "modelSha256": report["modelSha256"],
        "scoreSemantics": report["scoreSemantics"],
        "memberSeeds": report["memberSeeds"],
        "files": entries,
        "totalBytes": sum(int(entry["bytes"]) for entry in entries),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
        bundle.writestr(_info(PACKAGE_MANIFEST), manifest_bytes)
        for path, entry in zip(files, entries):
            with path.open("rb") as source, bundle.open(
                _info(f"{ARCHIVE_ROOT}/{entry['path']}"), "w"
            ) as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    verified = verify_result_archive(archive)
    with checksum_path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{verified['sha256']}  {archive.name}\n")
    return {**verified, "checksumFile": str(checksum_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package or verify a tax-return advantage training result."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--result-dir")
    action.add_argument("--verify-archive")
    parser.add_argument("--output")
    parser.add_argument("--checksum")
    args = parser.parse_args()
    if args.verify_archive:
        if args.output is not None:
            parser.error("--output cannot be used with --verify-archive")
        if args.checksum is None:
            parser.error("--verify-archive requires --checksum")
        report = verify_result_archive(args.verify_archive, args.checksum)
    else:
        if args.output is None:
            parser.error("--result-dir requires --output")
        if args.checksum is not None:
            parser.error("--checksum can only be used with --verify-archive")
        report = package_result_directory(args.result_dir, args.output)
    print(f"Tax advantage package verified: {report['archive']}")
    print(f"SHA-256: {report['sha256']}")


if __name__ == "__main__":
    main()
