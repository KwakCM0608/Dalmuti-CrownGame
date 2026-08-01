from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "checkpoint.pt",
    "actor-critic-weights.json",
    "ppo-metadata.json",
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
        description="Package a DALMUTI PPO update for CPU handoff.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    archive_path = results_dir / f"{model_dir.name}-result.zip"
    checksum_path = archive_path.with_suffix(f"{archive_path.suffix}.sha256")
    manifest_path = model_dir / "result-manifest.json"
    existing_outputs = [
        path
        for path in (manifest_path, archive_path, checksum_path)
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "result packaging outputs must not already exist: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (model_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"cannot package incomplete PPO result; missing: "
            f"{', '.join(missing)}"
        )

    metrics = json.loads(
        (model_dir / "training-metrics.json").read_text(encoding="utf-8")
    )
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("training metrics contain no completed epochs")
    checkpoint_files: list[Path] = []
    checkpoint_directories: list[str] = []
    completed_epochs: list[int] = []
    for metric in metrics:
        epoch = metric.get("epoch") if isinstance(metric, dict) else None
        if not isinstance(epoch, int) or epoch < 1 or epoch > 12:
            raise ValueError("training metrics contain an invalid epoch")
        completed_epochs.append(epoch)
        relative_directory = Path("checkpoints") / f"epoch-{epoch:02d}"
        directory = model_dir / relative_directory
        expected = tuple(
            directory / filename
            for filename in (
                "checkpoint.pt",
                "actor-critic-weights.json",
                "metrics.json",
            )
        )
        missing_epoch_files = [
            path.name for path in expected if not path.is_file()
        ]
        if missing_epoch_files:
            raise FileNotFoundError(
                f"epoch {epoch} checkpoint is incomplete; missing: "
                f"{', '.join(missing_epoch_files)}"
            )
        checkpoint_directories.append(relative_directory.as_posix())
        checkpoint_files.extend(expected)
    if completed_epochs != list(range(1, len(metrics) + 1)):
        raise ValueError("training metrics must contain consecutive epochs")

    entries = []
    result_files = [
        *(model_dir / filename for filename in REQUIRED_FILES),
        *checkpoint_files,
    ]
    for path in result_files:
        relative_path = path.relative_to(model_dir).as_posix()
        entries.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "format": "dalmuti-ppo-training-result",
        "version": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "modelDirectory": model_dir.name,
        "completedEpochs": len(metrics),
        "epochCheckpointDirectories": checkpoint_directories,
        "files": entries,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(
            f"{json.dumps(manifest, ensure_ascii=False, indent=2)}\n"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    archive_base = results_dir / f"{model_dir.name}-result"
    created_archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=model_dir,
        )
    )
    if created_archive_path != archive_path:
        raise RuntimeError("result archive path changed unexpectedly")
    checksum = sha256(created_archive_path)
    with checksum_path.open("x", encoding="utf-8") as stream:
        stream.write(f"{checksum}  {created_archive_path.name}\n")
    print(f"PPO result archive: {created_archive_path}")
    print(f"SHA-256: {checksum}")
    print(f"Checksum file: {checksum_path}")


if __name__ == "__main__":
    main()
