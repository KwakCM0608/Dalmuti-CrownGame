from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from actor_critic import (
    ActorCriticNetwork,
    export_actor_critic_json,
    load_behavior_model,
)
from ppo_dataset import PpoRollouts, load_ppo_rollouts


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one DALMUTI action-masked PPO update.",
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--behavior-model", required=True)
    parser.add_argument("--output", default="models/ppo-iteration-1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coefficient", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_dataset(rollouts: PpoRollouts) -> TensorDataset:
    non_forced = ~rollouts.forced
    if not non_forced.any():
        raise ValueError("PPO rollouts contain no non-forced actions")
    mean = float(rollouts.advantages[non_forced].mean())
    standard_deviation = float(rollouts.advantages[non_forced].std())
    normalized_advantages = (
        (rollouts.advantages - mean) / max(standard_deviation, 1.0e-8)
    ).astype(np.float32)
    return TensorDataset(
        torch.from_numpy(rollouts.observations),
        torch.from_numpy(rollouts.legal_masks),
        torch.from_numpy(rollouts.actions),
        torch.from_numpy(rollouts.old_log_probabilities),
        torch.from_numpy(rollouts.old_values),
        torch.from_numpy(rollouts.returns),
        torch.from_numpy(normalized_advantages),
        torch.from_numpy(non_forced.astype(np.float32)),
    )


def explained_variance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    target_variance = torch.var(targets, unbiased=False)
    if float(target_variance) < 1.0e-12:
        return 0.0
    residual_variance = torch.var(targets - predictions, unbiased=False)
    return float(1.0 - residual_variance / target_variance)


def train_epoch(
    model: ActorCriticNetwork,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> tuple[EpochMetrics, bool]:
    started_at = time.perf_counter()
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approximate_kl": 0.0,
        "clip_fraction": 0.0,
        "explained_variance": 0.0,
    }
    batches = 0
    stop_for_kl = False
    model.train()
    for (
        observations,
        legal_masks,
        actions,
        old_log_probabilities,
        old_values,
        returns,
        advantages,
        policy_weights,
    ) in loader:
        observations = observations.to(device, non_blocking=True)
        legal_masks = legal_masks.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        old_log_probabilities = old_log_probabilities.to(
            device,
            non_blocking=True,
        )
        old_values = old_values.to(device, non_blocking=True)
        returns = returns.to(device, non_blocking=True)
        advantages = advantages.to(device, non_blocking=True)
        policy_weights = policy_weights.to(device, non_blocking=True)

        logits, values = model(observations, legal_masks)
        log_probabilities = torch.log_softmax(logits, dim=1)
        selected_log_probabilities = log_probabilities.gather(
            1,
            actions[:, None],
        ).squeeze(1)
        log_ratio = selected_log_probabilities - old_log_probabilities
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio,
            1.0 - args.clip_coefficient,
            1.0 + args.clip_coefficient,
        ) * advantages
        policy_denominator = policy_weights.sum().clamp_min(1.0)
        policy_loss = -(
            torch.minimum(unclipped, clipped) * policy_weights
        ).sum() / policy_denominator

        value_delta = values - old_values
        clipped_values = old_values + torch.clamp(
            value_delta,
            -args.clip_coefficient,
            args.clip_coefficient,
        )
        value_losses = torch.square(values - returns)
        clipped_value_losses = torch.square(clipped_values - returns)
        value_loss = 0.5 * torch.maximum(
            value_losses,
            clipped_value_losses,
        ).mean()

        probabilities = torch.softmax(logits, dim=1)
        entropy_by_sample = -(
            probabilities * log_probabilities
        ).sum(dim=1)
        entropy = (
            entropy_by_sample * policy_weights
        ).sum() / policy_denominator
        loss = (
            policy_loss
            + args.value_coefficient * value_loss
            - args.entropy_coefficient * entropy
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            args.max_gradient_norm,
        )
        optimizer.step()

        with torch.no_grad():
            approximate_kl = (
                ((ratio - 1.0) - log_ratio) * policy_weights
            ).sum() / policy_denominator
            clip_fraction = (
                ((torch.abs(ratio - 1.0) > args.clip_coefficient).float())
                * policy_weights
            ).sum() / policy_denominator
            totals["policy_loss"] += float(policy_loss)
            totals["value_loss"] += float(value_loss)
            totals["entropy"] += float(entropy)
            totals["approximate_kl"] += float(approximate_kl)
            totals["clip_fraction"] += float(clip_fraction)
            totals["explained_variance"] += explained_variance(
                values,
                returns,
            )
            batches += 1
            if float(approximate_kl) > args.target_kl:
                stop_for_kl = True
                break

    if batches < 1:
        raise RuntimeError("PPO loader produced no batches")
    averaged = {
        key: value / batches
        for key, value in totals.items()
    }
    return (
        EpochMetrics(
            epoch=0,
            policy_loss=averaged["policy_loss"],
            value_loss=averaged["value_loss"],
            entropy=averaged["entropy"],
            approximate_kl=averaged["approximate_kl"],
            clip_fraction=averaged["clip_fraction"],
            explained_variance=averaged["explained_variance"],
            seconds=time.perf_counter() - started_at,
        ),
        stop_for_kl,
    )


def save_outputs(
    output: Path,
    model: ActorCriticNetwork,
    rollouts: PpoRollouts,
    metrics: list[EpochMetrics],
    device: torch.device,
    behavior_model: Path,
    args: argparse.Namespace,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dalmuti-actor-critic-checkpoint",
            "version": 1,
            "modelState": model.state_dict(),
            "observationFeatures": model.observation_features,
            "actionCount": model.action_count,
            "hiddenSizes": model.hidden_sizes,
            "optimizerStateIncluded": False,
        },
        output / "checkpoint.pt",
    )
    export_actor_critic_json(
        model,
        output / "actor-critic-weights.json",
    )
    metadata = {
        "format": "dalmuti-ppo-training-result",
        "version": 1,
        "behaviorModel": str(behavior_model),
        "behaviorModelSha256": file_sha256(behavior_model),
        "rolloutBehaviorModelSha256": rollouts.behavior_model_sha256,
        "device": str(device),
        "torchVersion": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDevice": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "samples": len(rollouts),
        "trajectories": rollouts.trajectory_count,
        "forcedSamples": int(rollouts.forced.sum()),
        "sourceFiles": list(rollouts.files),
        "arguments": vars(args),
    }
    (output / "ppo-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "training-metrics.json").write_text(
        json.dumps(
            [asdict(metric) for metric in metrics],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.target_kl <= 0 or args.max_gradient_norm <= 0:
        raise ValueError("target-kl and max-gradient-norm must be positive")
    set_seeds(args.seed)
    device = choose_device(args.device)
    behavior_model = Path(args.behavior_model).resolve()
    behavior_sha256 = file_sha256(behavior_model)
    rollouts = load_ppo_rollouts(
        args.data,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    if behavior_sha256 != rollouts.behavior_model_sha256:
        raise ValueError(
            "behavior model SHA-256 does not match the PPO rollout manifest"
        )
    model, _ = load_behavior_model(behavior_model)
    model = model.to(device)
    print(
        f"Loaded {len(rollouts):,} samples from "
        f"{rollouts.trajectory_count:,} trajectories. Device: {device}."
    )
    print(
        f"Behavior model: {behavior_sha256}; "
        f"forced samples: {int(rollouts.forced.sum()):,}."
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        tensor_dataset(rollouts),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    metrics: list[EpochMetrics] = []
    for epoch in range(1, args.epochs + 1):
        epoch_metrics, stop_for_kl = train_epoch(
            model,
            loader,
            device,
            optimizer,
            args,
        )
        epoch_metrics = EpochMetrics(
            **{
                **asdict(epoch_metrics),
                "epoch": epoch,
            }
        )
        metrics.append(epoch_metrics)
        print(
            f"epoch {epoch:02d} | "
            f"policy {epoch_metrics.policy_loss:.5f} "
            f"value {epoch_metrics.value_loss:.5f} "
            f"entropy {epoch_metrics.entropy:.5f} | "
            f"KL {epoch_metrics.approximate_kl:.6f} "
            f"clip {epoch_metrics.clip_fraction:.2%} "
            f"EV {epoch_metrics.explained_variance:.4f} | "
            f"{epoch_metrics.seconds:.1f}s"
        )
        if stop_for_kl:
            print(
                f"Stopped after epoch {epoch}: target KL "
                f"{args.target_kl} was exceeded."
            )
            break

    save_outputs(
        Path(args.output),
        model,
        rollouts,
        metrics,
        device,
        behavior_model,
        args,
    )
    print(f"Saved PPO update to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
