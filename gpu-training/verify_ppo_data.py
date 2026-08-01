from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ppo_dataset import load_ppo_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate DALMUTI on-policy PPO rollouts.",
    )
    parser.add_argument("--data", nargs="+", required=True)
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
    parser.add_argument("--output")
    args = parser.parse_args()
    loaded = load_ppo_rollouts(
        args.data,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        skip_forced_policy_time=args.skip_forced_policy_time,
        terminal_rank_auxiliary_coefficient=(
            args.terminal_rank_auxiliary_coefficient
        ),
    )
    if not np.isclose(
        loaded.behavior_temperature,
        args.rollout_temperature,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "rollout-temperature does not match the PPO rollout manifest"
        )
    result = {
        "files": list(loaded.files),
        "samples": len(loaded),
        "trajectories": loaded.trajectory_count,
        "behaviorModelSha256": loaded.behavior_model_sha256,
        "observationShape": list(loaded.observations.shape),
        "legalMaskShape": list(loaded.legal_masks.shape),
        "forcedSamples": int(loaded.forced.sum()),
        "policySamples": int((~loaded.forced).sum()),
        "terminalSamples": int(loaded.terminals.sum()),
        "returnEstimator": (
            "undiscounted-monte-carlo"
            if args.gamma == 1.0 and args.gae_lambda == 1.0
            else "gae"
        ),
        "gamma": args.gamma,
        "gaeLambda": args.gae_lambda,
        "skipForcedPolicyTime": loaded.skip_forced_policy_time,
        "terminalRankAuxiliaryCoefficient": (
            loaded.terminal_rank_auxiliary_coefficient
        ),
        "rolloutTemperature": args.rollout_temperature,
        "manifestRolloutTemperature": loaded.behavior_temperature,
        "meanReward": float(loaded.rewards.mean()),
        "meanRankAuxiliaryReward": float(
            loaded.rank_auxiliary_rewards.mean()
        ),
        "meanEffectiveReward": float(loaded.effective_rewards.mean()),
        "meanAdvantage": float(loaded.advantages.mean()),
        "advantageStd": float(loaded.advantages.std()),
        "meanReturn": float(loaded.returns.mean()),
        "finite": bool(
            np.isfinite(loaded.observations).all()
            and np.isfinite(loaded.old_log_probabilities).all()
            and np.isfinite(loaded.old_values).all()
            and np.isfinite(loaded.advantages).all()
            and np.isfinite(loaded.returns).all()
        ),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{payload}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
