from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def nvidia_smi_output() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(
        [executable],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    return output or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the GPU computer before DALMUTI model training.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parent
    disk = shutil.disk_usage(bundle_root)
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "format": "dalmuti-gpu-preflight",
        "version": 1,
        "platform": platform.platform(),
        "pythonVersion": sys.version,
        "pythonExecutable": sys.executable,
        "processArchitecture": platform.architecture()[0],
        "cpuCount": os.cpu_count(),
        "numpyVersion": np.__version__,
        "torchVersion": torch.__version__,
        "torchCudaVersion": torch.version.cuda,
        "cudnnVersion": torch.backends.cudnn.version(),
        "cudaAvailable": cuda_available,
        "bundleFreeDiskBytes": disk.free,
        "nvidiaSmi": nvidia_smi_output(),
        "gpuDevices": [],
    }
    if cuda_available:
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "computeCapability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "totalMemoryBytes": properties.total_memory,
                    "multiProcessorCount": properties.multi_processor_count,
                }
            )
        report["gpuDevices"] = devices

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{payload}\n", encoding="utf-8")

    if args.device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA training was requested, but torch.cuda.is_available() is "
            "False. Install a CUDA-enabled PyTorch build before continuing."
        )
    if args.device == "cuda" and not report["gpuDevices"]:
        raise RuntimeError("CUDA is available, but no CUDA GPU was enumerated.")
    if disk.free < 8 * 1024**3:
        raise RuntimeError(
            "Less than 8 GiB of free disk space is available in the bundle "
            "location."
        )


if __name__ == "__main__":
    main()
