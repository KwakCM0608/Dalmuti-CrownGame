from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify, train, and package the DALMUTI BC warm-start policy."
        ),
    )
    parser.add_argument("--data", nargs="+", default=["data/*-p*-v2.ndjson"])
    parser.add_argument("--output", default="models/bc-warmstart-v3")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--supervised-weight", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def run_and_tee(
    command: list[str],
    log_path: Path,
    *,
    append: bool = True,
) -> None:
    mode = "a" if append else "w"
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    }
    display = subprocess.list2cmdline(command)
    print(f"\n> {display}", flush=True)
    with log_path.open(mode, encoding="utf-8") as log:
        log.write(f"\n> {display}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    output = (root / args.output).resolve()
    results_dir = (root / args.results_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.log"
    python = sys.executable

    run_and_tee(
        [python, str(root / "verify_bundle.py")],
        log_path,
        append=False,
    )
    run_and_tee(
        [
            python,
            str(root / "preflight.py"),
            "--device",
            "cuda",
            "--output",
            str(output / "hardware-report.json"),
        ],
        log_path,
    )
    verify_command = [
        python,
        str(root / "verify_data.py"),
        "--data",
        *args.data,
        "--validation-fraction",
        str(args.validation_fraction),
        "--supervised-weight",
        str(args.supervised_weight),
        "--output",
        str(output / "data-verification.json"),
    ]
    run_and_tee(verify_command, log_path)

    train_command = [
        python,
        str(root / "train_bc.py"),
        "--data",
        *args.data,
        "--output",
        str(output),
        "--device",
        "cuda",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-sizes",
        args.hidden_sizes,
        "--validation-fraction",
        str(args.validation_fraction),
        "--supervised-weight",
        str(args.supervised_weight),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
    ]
    run_and_tee(train_command, log_path)
    run_and_tee(
        [
            python,
            str(root / "package_results.py"),
            "--model-dir",
            str(output),
            "--results-dir",
            str(results_dir),
        ],
        log_path,
    )
    print(
        "\nGPU warm-start training finished. Return the result ZIP and its "
        f"SHA-256 file from {results_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
