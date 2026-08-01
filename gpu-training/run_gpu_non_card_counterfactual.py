"""One-command GPU training and exclusive result packaging."""

from __future__ import annotations

import argparse
from pathlib import Path

from package_non_card_results import package_result_directory
from train_non_card_counterfactual import TrainingOptions, train_non_card_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train DALMUTI tax/revolution policies in a fresh directory and "
            "package the fully verified result."
        )
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-zip", required=True)
    parser.add_argument(
        "--decision",
        choices=("all", "tax-return", "revolution"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--policy-coefficient", type=float, default=0.5)
    parser.add_argument("--action-value-coefficient", type=float, default=1.0)
    parser.add_argument("--value-coefficient", type=float, default=0.25)
    parser.add_argument(
        "--behavior-cloning-coefficient", type=float, default=0.0
    )
    parser.add_argument(
        "--utility-target",
        choices=("terminal", "decision-act"),
        default="terminal",
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--policy-temperature", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    options = TrainingOptions(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        policy_coefficient=args.policy_coefficient,
        action_value_coefficient=args.action_value_coefficient,
        value_coefficient=args.value_coefficient,
        behavior_cloning_coefficient=(
            args.behavior_cloning_coefficient
        ),
        utility_target=args.utility_target,
        entropy_coefficient=args.entropy_coefficient,
        huber_delta=args.huber_delta,
        max_gradient_norm=args.max_gradient_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        seed=args.seed,
        policy_temperature=args.policy_temperature,
        device=args.device,
        deterministic=args.deterministic,
    )
    train_non_card_models(
        data_patterns=args.data,
        output_directory=args.output,
        decision=args.decision,
        options=options,
    )
    report = package_result_directory(args.output, args.result_zip)
    print(f"Verified result ZIP: {Path(report['archive'])}")
    print(f"SHA-256: {report['sha256']}")


if __name__ == "__main__":
    main()
