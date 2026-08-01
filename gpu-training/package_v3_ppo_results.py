from __future__ import annotations

import argparse
import json
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from v3_ppo_result_contract import (
    RESULT_FORMAT,
    STRICT_RESULT_VERSION,
    load_source_contract,
    reject_symlink_components,
    safe_relative_path,
    sha256_file,
    validate_result_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and attest one strict V3 PPO result."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--expected-bundle-manifest")
    parser.add_argument("--expected-run-config")
    parser.add_argument(
        "--allow-legacy-smoke",
        action="store_true",
        help="Explicitly package a local CPU smoke without GPU provenance.",
    )
    return parser.parse_args()


def _disjoint(left: Path, right: Path, label: str) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise ValueError(f"{label} must be disjoint: {left}, {right}")


def _source_summary(source: dict | None) -> dict | None:
    if source is None:
        return None
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


def _zip_exact_files(archive: Path, root: Path, paths: list[str]) -> None:
    with zipfile.ZipFile(
        archive,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as packaged:
        for relative in paths:
            safe_relative_path(relative, "result archive path")
            source = root / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"result archive source is not a regular file: {source}")
            info = zipfile.ZipInfo(relative)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            # Fixed timestamps keep packaging reproducible apart from the
            # explicitly recorded createdAt field in result-manifest.json.
            info.date_time = (1980, 1, 1, 0, 0, 0)
            with source.open("rb") as input_stream, packaged.open(info, "w") as output:
                while chunk := input_stream.read(1024 * 1024):
                    output.write(chunk)


def main() -> None:
    args = parse_args()
    raw_model_dir = Path(args.model_dir).absolute()
    raw_results_dir = Path(args.results_dir).absolute()
    reject_symlink_components(raw_model_dir, "V3 model directory")
    reject_symlink_components(raw_results_dir, "V3 result directory")
    model_dir = raw_model_dir.resolve()
    results_dir = raw_results_dir.resolve()
    if not model_dir.is_dir():
        raise ValueError(f"V3 model directory must be regular: {model_dir}")
    if results_dir.exists() or results_dir.is_symlink():
        raise FileExistsError(
            f"V3 result directory must be fresh: {results_dir}"
        )
    _disjoint(model_dir, results_dir, "model and returned-result directories")

    strict_paths = (
        args.expected_bundle_manifest,
        args.expected_run_config,
    )
    if args.allow_legacy_smoke:
        if any(strict_paths):
            raise ValueError(
                "legacy smoke mode cannot be combined with source provenance"
            )
        source = None
    else:
        if not all(strict_paths):
            raise ValueError(
                "strict packaging requires --expected-bundle-manifest and "
                "--expected-run-config"
            )
        bundle_manifest = Path(args.expected_bundle_manifest).absolute()
        run_config = Path(args.expected_run_config).absolute()
        _disjoint(results_dir, bundle_manifest, "result and bundle-manifest paths")
        _disjoint(results_dir, run_config, "result and run-config paths")
        source = load_source_contract(
            bundle_manifest,
            run_config,
            verify_source_files=True,
        )

    validation = validate_result_directory(
        model_dir,
        source=source,
        allow_legacy_smoke=args.allow_legacy_smoke,
        allow_manifest=False,
    )
    manifest_path = model_dir / "result-manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(
            f"V3 result manifest must not already exist: {manifest_path}"
        )

    file_paths = sorted(validation["inventory"])
    entries = [
        {
            "path": relative,
            "bytes": validation["inventory"][relative].stat().st_size,
            "sha256": sha256_file(validation["inventory"][relative]),
        }
        for relative in file_paths
    ]
    manifest = {
        "format": RESULT_FORMAT,
        "version": STRICT_RESULT_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "provenanceMode": validation["provenanceMode"],
        "runId": validation["runId"],
        "completedEpochs": validation["completedEpochs"],
        "selectedEpoch": validation["selectedEpoch"],
        "sourceProvenance": _source_summary(source),
        "trainingContract": validation["strictDetails"],
        "resultInventory": {
            "policy": "exact-files-no-symlinks-no-unmanifested-entries-v1",
            "paths": file_paths,
        },
        "files": entries,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    results_dir.mkdir(parents=True, exist_ok=False)
    archive = results_dir / f"{model_dir.name}-result.zip"
    checksum = results_dir / f"{archive.name}.sha256"
    if archive.exists() or checksum.exists():
        raise FileExistsError("V3 archive outputs must be fresh")
    _zip_exact_files(
        archive,
        model_dir,
        [*file_paths, "result-manifest.json"],
    )
    digest = sha256_file(archive)
    with checksum.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{digest}  {archive.name}\n")
    print(f"V3 PPO result archive: {archive}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
