from __future__ import annotations

import argparse
import json
from pathlib import Path

from v3_ppo_dataset import (
    build_v3_ppo_data_verification,
    load_v3_ppo_rollouts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly verify V3 action-conditioned PPO rollouts."
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--behavior-model", required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument(
        "--skip-forced-policy-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--terminal-rank-auxiliary-coefficient", type=float, default=0.0
    )
    parser.add_argument("--rollout-temperature", type=float, required=True)
    parser.add_argument("--binding-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--behavior-binding-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--behavior-binding-batch-size", type=int, default=2048
    )
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    source_files: list[dict[str, object]] = []
    loaded = load_v3_ppo_rollouts(
        args.data,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        skip_forced_policy_time=args.skip_forced_policy_time,
        terminal_rank_auxiliary_coefficient=(
            args.terminal_rank_auxiliary_coefficient
        ),
        behavior_model_path=args.behavior_model,
        binding_tolerance=args.binding_tolerance,
        behavior_binding_device=args.behavior_binding_device,
        behavior_binding_batch_size=args.behavior_binding_batch_size,
        loader_workers=args.loader_workers,
        source_files_out=source_files,
    )
    result = build_v3_ppo_data_verification(
        loaded,
        source_files=source_files,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        rollout_temperature=args.rollout_temperature,
        binding_tolerance=args.binding_tolerance,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(f"{payload}\n")


if __name__ == "__main__":
    main()
