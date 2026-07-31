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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--output")
    args = parser.parse_args()
    loaded = load_ppo_rollouts(
        args.data,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    result = {
        "files": list(loaded.files),
        "samples": len(loaded),
        "trajectories": loaded.trajectory_count,
        "behaviorModelSha256": loaded.behavior_model_sha256,
        "observationShape": list(loaded.observations.shape),
        "legalMaskShape": list(loaded.legal_masks.shape),
        "forcedSamples": int(loaded.forced.sum()),
        "terminalSamples": int(loaded.terminals.sum()),
        "meanReward": float(loaded.rewards.mean()),
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
