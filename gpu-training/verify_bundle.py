from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest_path = root / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked_bytes = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"bundle file is missing: {entry['path']}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"bundle file size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"bundle checksum mismatch: {entry['path']}")
        checked_bytes += size
    if checked_bytes != manifest["totalBytes"]:
        raise ValueError("bundle total byte count mismatch")
    print(
        f"Bundle verified: {len(manifest['files'])} files, "
        f"{checked_bytes / 1024 / 1024:.2f} MiB"
    )


if __name__ == "__main__":
    main()

