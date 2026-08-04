from __future__ import annotations

"""Repository-root launcher for the V5 paired-action diagnostic."""

import os
from pathlib import Path
import sys


SOURCE_ROOT = Path(
    os.environ.get("DALMUTI_V5_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
GPU_TRAINING = SOURCE_ROOT / "gpu-training"
if not (GPU_TRAINING / "v5_paired_action_counterfactual.py").is_file():
    raise RuntimeError(
        "V5 source root is invalid; set DALMUTI_V5_SOURCE_ROOT to the repository checkout"
    )
if str(GPU_TRAINING) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINING))

from v5_paired_action_counterfactual import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
