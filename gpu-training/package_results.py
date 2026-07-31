from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "checkpoint.pt",
    "policy-weights.json",
    "policy-metadata.json",
    "training-metrics.json",
    "hardware-report.json",
    "data-verification.json",
    "training.log",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a DALMUTI GPU training result for handoff.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (model_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"cannot package incomplete result; missing: {', '.join(missing)}"
        )

    manifest_entries = []
    for filename in REQUIRED_FILES:
        path = model_dir / filename
        manifest_entries.append(
            {
                "path": filename,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "format": "dalmuti-gpu-training-result",
        "version": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "modelDirectory": model_dir.name,
        "files": manifest_entries,
    }
    manifest_path = model_dir / "result-manifest.json"
    manifest_path.write_text(
        f"{json.dumps(manifest, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    archive_base = results_dir / f"{model_dir.name}-result"
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=model_dir,
        )
    )
    checksum = sha256(archive_path)
    checksum_path = archive_path.with_suffix(f"{archive_path.suffix}.sha256")
    checksum_path.write_text(
        f"{checksum}  {archive_path.name}\n",
        encoding="utf-8",
    )
    print(f"Result archive: {archive_path}")
    print(f"SHA-256: {checksum}")
    print(f"Checksum file: {checksum_path}")


if __name__ == "__main__":
    main()
