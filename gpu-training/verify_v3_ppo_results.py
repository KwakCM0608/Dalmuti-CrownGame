from __future__ import annotations

import argparse
import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path

from v3_ppo_result_contract import (
    LEGACY_PROVENANCE_MODE,
    LEGACY_RESULT_VERSION,
    RESULT_FORMAT,
    STRICT_PROVENANCE_MODE,
    STRICT_RESULT_VERSION,
    load_source_contract,
    reject_symlink_components,
    safe_relative_path,
    sha256_file,
    validate_result_directory,
)


CHECKSUM_PATTERN = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)\r?\n$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly verify a returned V3 PPO ZIP and external SHA-256."
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--expected-bundle-manifest")
    parser.add_argument("--expected-run-config")
    parser.add_argument("--extract-dir")
    parser.add_argument(
        "--allow-legacy-smoke",
        action="store_true",
        help="Explicitly accept a local smoke result without GPU provenance.",
    )
    return parser.parse_args()


def _regular_file(path: Path, label: str) -> None:
    if (
        path.is_symlink()
        or getattr(path, "is_junction", lambda: False)()
        or not path.is_file()
    ):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _disjoint(left: Path, right: Path, label: str) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise ValueError(f"{label} must be disjoint: {left}, {right}")


def _source_summary(source: dict) -> dict:
    return {
        "bundleManifestSha256": source["bundleManifestSha256"],
        "runConfigSha256": source["runConfigSha256"],
        "parentModelSha256": source["parentModelSha256"],
        "rollouts": source["rollouts"],
        "dataCounts": source["dataCounts"],
        "algorithm": source["algorithm"],
        "pathPolicy": source["pathPolicy"],
        "determinism": source["determinism"],
        "allowedTerminalRankAuxiliaryCoefficients": source[
            "allowedTerminalRankAuxiliaryCoefficients"
        ],
    }


def _safe_zip_infos(packaged: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for info in packaged.infolist():
        name = safe_relative_path(info.filename, "V3 archive entry")
        if name in result:
            raise ValueError(f"duplicate V3 archive entry: {name}")
        if info.is_dir() or name.endswith("/"):
            raise ValueError(f"directory entries are forbidden in V3 results: {name}")
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted V3 archive entries are forbidden: {name}")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in (0, stat.S_IFREG):
            raise ValueError(f"non-regular V3 archive entry: {name}")
        result[name] = info
    return result


def _manifest_entries(manifest: dict) -> dict[str, dict]:
    values = manifest.get("files")
    if not isinstance(values, list) or not values:
        raise ValueError("V3 result manifest contains no files")
    result: dict[str, dict] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
            raise ValueError(f"V3 result file {index} is not an object")
        path = safe_relative_path(value.get("path"), f"V3 result file {index}")
        size = value.get("bytes")
        digest = value.get("sha256")
        if (
            path in result
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"invalid or duplicate V3 result entry: {path}")
        result[path] = value
    return result


def _copy_and_hash(
    packaged: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> tuple[int, str]:
    import hashlib

    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with packaged.open(info, "r") as source, destination.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            output.write(chunk)
    return size, digest.hexdigest()


def _validate_checksum(archive: Path, checksum: Path) -> str:
    _regular_file(archive, "V3 result archive")
    _regular_file(checksum, "V3 result checksum")
    if checksum.parent != archive.parent or checksum.name != f"{archive.name}.sha256":
        raise ValueError("V3 checksum must be adjacent to and named after the ZIP")
    try:
        text = checksum.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError("V3 checksum file must be ASCII") from error
    match = CHECKSUM_PATTERN.fullmatch(text)
    if match is None or match.group(2) != archive.name:
        raise ValueError("V3 checksum file is malformed")
    actual = sha256_file(archive)
    if match.group(1) != actual:
        raise ValueError("V3 result archive checksum mismatch")
    return actual


def _validate_manifest_contract(
    manifest: dict,
    *,
    source: dict | None,
    allow_legacy_smoke: bool,
    entry_paths: list[str],
) -> None:
    if manifest.get("format") != RESULT_FORMAT:
        raise ValueError("unsupported V3 result manifest format")
    version = manifest.get("version")
    if version == LEGACY_RESULT_VERSION:
        if not allow_legacy_smoke:
            raise ValueError("legacy V3 result requires --allow-legacy-smoke")
        return
    if version != STRICT_RESULT_VERSION:
        raise ValueError("unsupported V3 result manifest version")
    expected_manifest_keys = {
        "format",
        "version",
        "createdAt",
        "provenanceMode",
        "runId",
        "completedEpochs",
        "selectedEpoch",
        "sourceProvenance",
        "trainingContract",
        "resultInventory",
        "files",
    }
    if (
        set(manifest) != expected_manifest_keys
        or not isinstance(manifest.get("createdAt"), str)
        or not manifest.get("createdAt")
    ):
        raise ValueError("V3 result manifest schema mismatch")
    inventory = manifest.get("resultInventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("policy")
        != "exact-files-no-symlinks-no-unmanifested-entries-v1"
        or inventory.get("paths") != sorted(entry_paths)
    ):
        raise ValueError("V3 result exact-inventory contract mismatch")
    expected_mode = (
        LEGACY_PROVENANCE_MODE if allow_legacy_smoke else STRICT_PROVENANCE_MODE
    )
    if manifest.get("provenanceMode") != expected_mode:
        raise ValueError("V3 result provenance mode mismatch")
    if source is None:
        if manifest.get("sourceProvenance") is not None:
            raise ValueError("legacy smoke unexpectedly claims strict provenance")
    elif manifest.get("sourceProvenance") != _source_summary(source):
        raise ValueError("V3 result source bundle/run-config provenance mismatch")


def _validate_cross_bindings(manifest: dict, validation: dict, source: dict | None) -> None:
    if manifest.get("version") == LEGACY_RESULT_VERSION:
        return
    if (
        manifest.get("runId") != validation["runId"]
        or manifest.get("completedEpochs") != validation["completedEpochs"]
        or manifest.get("selectedEpoch") != validation["selectedEpoch"]
        or manifest.get("provenanceMode") != validation["provenanceMode"]
        or manifest.get("trainingContract") != validation["strictDetails"]
    ):
        raise ValueError("V3 result manifest/model/training cross-binding mismatch")
    if source is not None and manifest.get("sourceProvenance") != _source_summary(source):
        raise ValueError("V3 result external provenance changed during verification")


def main() -> None:
    args = parse_args()
    raw_archive = Path(args.archive).absolute()
    raw_checksum = Path(args.checksum).absolute()
    reject_symlink_components(raw_archive, "V3 archive")
    reject_symlink_components(raw_checksum, "V3 checksum")
    archive = raw_archive.resolve()
    checksum = raw_checksum.resolve()
    _disjoint(archive, checksum, "archive and checksum paths")
    actual_sha256 = _validate_checksum(archive, checksum)

    strict_values = (args.expected_bundle_manifest, args.expected_run_config)
    if args.allow_legacy_smoke:
        if any(strict_values):
            raise ValueError(
                "legacy smoke mode cannot be combined with source provenance"
            )
        source = None
    else:
        if not all(strict_values):
            raise ValueError(
                "strict verification requires --expected-bundle-manifest and "
                "--expected-run-config"
            )
        bundle_manifest = Path(args.expected_bundle_manifest).absolute()
        run_config = Path(args.expected_run_config).absolute()
        _disjoint(archive, bundle_manifest, "archive and bundle-manifest paths")
        _disjoint(archive, run_config, "archive and run-config paths")
        source = load_source_contract(
            bundle_manifest,
            run_config,
            verify_source_files=True,
        )

    with zipfile.ZipFile(archive, "r") as packaged:
        infos = _safe_zip_infos(packaged)
        manifest_info = infos.get("result-manifest.json")
        if manifest_info is None or manifest_info.file_size > 16 * 1024 * 1024:
            raise ValueError("V3 result manifest is missing or unreasonably large")
        try:
            manifest = json.loads(packaged.read(manifest_info).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("V3 result manifest is not valid UTF-8 JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("V3 result manifest must be an object")
        entries = _manifest_entries(manifest)
        expected_archive_paths = set(entries) | {"result-manifest.json"}
        if set(infos) != expected_archive_paths:
            missing = sorted(expected_archive_paths - set(infos))
            unexpected = sorted(set(infos) - expected_archive_paths)
            raise ValueError(
                f"V3 archive inventory mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )
        _validate_manifest_contract(
            manifest,
            source=source,
            allow_legacy_smoke=args.allow_legacy_smoke,
            entry_paths=list(entries),
        )

        with tempfile.TemporaryDirectory(prefix="dalmuti-v3-verify-") as temporary:
            temporary_root = Path(temporary)
            for relative, info in infos.items():
                size, digest = _copy_and_hash(
                    packaged,
                    info,
                    temporary_root / relative,
                )
                if relative == "result-manifest.json":
                    continue
                entry = entries[relative]
                if size != entry["bytes"] or digest != entry["sha256"]:
                    raise ValueError(f"V3 result file binding mismatch: {relative}")
            validation = validate_result_directory(
                temporary_root,
                source=source,
                allow_legacy_smoke=args.allow_legacy_smoke,
                allow_manifest=True,
                expected_run_id=manifest.get("runId"),
            )
            _validate_cross_bindings(manifest, validation, source)

        if args.extract_dir:
            raw_destination = Path(args.extract_dir).absolute()
            reject_symlink_components(raw_destination, "V3 extraction directory")
            destination = raw_destination.resolve()
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"V3 extraction directory must be fresh: {destination}"
                )
            for other, label in (
                (archive, "archive/extraction"),
                (checksum, "checksum/extraction"),
            ):
                _disjoint(destination, other, f"{label} paths")
            if source is not None:
                _disjoint(destination, source["root"], "bundle/extraction paths")
            destination.mkdir(parents=True, exist_ok=False)
            for relative, info in infos.items():
                _copy_and_hash(packaged, info, destination / relative)

    print(f"Verified strict V3 PPO result: {archive} ({actual_sha256})")


if __name__ == "__main__":
    main()
