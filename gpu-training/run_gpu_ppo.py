from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify, train, and package one DALMUTI PPO update.",
    )
    parser.add_argument("--data", nargs="+", default=["data/*.ndjson"])
    parser.add_argument("--behavior-model", default="behavior-model.json")
    parser.add_argument("--output", default="models/ppo-iteration")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument(
        "--skip-forced-policy-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--terminal-rank-auxiliary-coefficient",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument("--clip-coefficient", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=20260801)
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


def create_unique_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"training output must not already exist: {path}"
        )
    path.mkdir(parents=True, exist_ok=False)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 12:
        raise ValueError("epochs must be from 1 to 12")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    output = (root / args.output).resolve()
    results_dir = (root / args.results_dir).resolve()
    behavior_model = (root / args.behavior_model).resolve()
    create_unique_output_directory(output)
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
    verification_command = [
        python,
        str(root / "verify_ppo_data.py"),
        "--data",
        *args.data,
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--terminal-rank-auxiliary-coefficient",
        str(args.terminal_rank_auxiliary_coefficient),
        "--rollout-temperature",
        str(args.rollout_temperature),
        "--output",
        str(output / "data-verification.json"),
    ]
    if args.skip_forced_policy_time:
        verification_command.append("--skip-forced-policy-time")
    else:
        verification_command.append("--no-skip-forced-policy-time")
    run_and_tee(
        verification_command,
        log_path,
    )
    training_command = [
        python,
        str(root / "train_ppo.py"),
        "--data",
        *args.data,
        "--behavior-model",
        str(behavior_model),
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
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--terminal-rank-auxiliary-coefficient",
        str(args.terminal_rank_auxiliary_coefficient),
        "--rollout-temperature",
        str(args.rollout_temperature),
        "--clip-coefficient",
        str(args.clip_coefficient),
        "--value-coefficient",
        str(args.value_coefficient),
        "--entropy-coefficient",
        str(args.entropy_coefficient),
        "--target-kl",
        str(args.target_kl),
        "--seed",
        str(args.seed),
    ]
    if args.skip_forced_policy_time:
        training_command.append("--skip-forced-policy-time")
    else:
        training_command.append("--no-skip-forced-policy-time")
    run_and_tee(
        training_command,
        log_path,
    )
    run_and_tee(
        [
            python,
            str(root / "package_ppo_results.py"),
            "--model-dir",
            str(output),
            "--results-dir",
            str(results_dir),
        ],
        log_path,
    )
    print(
        "\nPPO update finished. Return the PPO result ZIP and SHA-256 "
        f"file from {results_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
