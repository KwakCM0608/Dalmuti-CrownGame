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

from train_ppo import cuda_device_identity


DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


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
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.deterministic:
        if args.seed is None:
            raise ValueError("--seed is required with --deterministic")
        if os.environ.get("PYTHONHASHSEED") != str(args.seed):
            raise RuntimeError(
                "deterministic preflight requires PYTHONHASHSEED to equal --seed"
            )
        if args.device == "cuda" and os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ) != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
            raise RuntimeError(
                "deterministic CUDA preflight requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

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
        "requestedDevice": args.device,
        "deterministicRuntime": {
            "seed": args.seed,
            "pythonHashSeed": os.environ.get("PYTHONHASHSEED"),
            "algorithmsEnabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "warnOnly": bool(
                getattr(
                    torch,
                    "is_deterministic_algorithms_warn_only_enabled",
                    lambda: False,
                )()
            ),
            "cublasWorkspaceConfig": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "cudnnDeterministic": torch.backends.cudnn.deterministic,
            "cudnnBenchmark": torch.backends.cudnn.benchmark,
            "cudaMatmulAllowTf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnnAllowTf32": torch.backends.cudnn.allow_tf32,
        },
        "bundleFreeDiskBytes": disk.free,
        "nvidiaSmi": nvidia_smi_output(),
        "gpuDevices": [],
    }
    if cuda_available:
        devices = []
        for index in range(torch.cuda.device_count()):
            devices.append(cuda_device_identity(index))
        report["gpuDevices"] = devices

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(f"{payload}\n")

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
