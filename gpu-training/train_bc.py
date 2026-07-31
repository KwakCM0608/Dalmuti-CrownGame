from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import LoadedRollouts, RolloutSplit, load_rollouts
from model import PolicyNetwork, export_policy_json


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    validation_top3_accuracy: float
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DALMUTI V2 behavior-cloning policy.",
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", default="models/bc-v2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--hidden-sizes",
        default="256,256",
        help="comma-separated hidden layer sizes",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--supervised-weight", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--include-forced", action="store_true")
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


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("hidden-sizes must contain integers") from error
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("hidden-sizes must contain positive integers")
    return sizes


def tensor_dataset(split: RolloutSplit) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(split.observations),
        torch.from_numpy(split.legal_masks),
        torch.from_numpy(split.actions),
        torch.from_numpy(split.weights),
    )


def evaluate(
    model: PolicyNetwork,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top3 = 0
    total_samples = 0
    with torch.inference_mode():
        total_weight = 0.0
        for observations, legal_masks, actions, weights in loader:
            observations = observations.to(device, non_blocking=True)
            legal_masks = legal_masks.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            logits = model(observations, legal_masks)
            losses = criterion(logits, actions)
            loss = (losses * weights).sum() / weights.sum()
            batch_size = actions.shape[0]
            total_loss += float((losses * weights).sum())
            total_weight += float(weights.sum())
            total_correct += int((logits.argmax(dim=1) == actions).sum())
            total_top3 += int(
                (logits.topk(k=3, dim=1).indices == actions[:, None])
                .any(dim=1)
                .sum()
            )
            total_samples += batch_size
    return (
        total_loss / total_weight,
        total_correct / total_samples,
        total_top3 / total_samples,
    )


def train_epoch(
    model: PolicyNetwork,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_weight = 0.0
    for observations, legal_masks, actions, weights in loader:
        observations = observations.to(device, non_blocking=True)
        legal_masks = legal_masks.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(observations, legal_masks)
        losses = criterion(logits, actions)
        loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        batch_size = actions.shape[0]
        total_loss += float((losses.detach() * weights).sum())
        total_weight += float(weights.sum())
        total_correct += int((logits.argmax(dim=1) == actions).sum())
        total_samples += batch_size
    return total_loss / total_weight, total_correct / total_samples


def save_outputs(
    output: Path,
    model: PolicyNetwork,
    loaded: LoadedRollouts,
    metrics: list[EpochMetrics],
    best_epoch: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "modelState": model.state_dict(),
            "observationFeatures": model.observation_features,
            "actionCount": model.action_count,
            "hiddenSizes": model.hidden_sizes,
            "bestEpoch": best_epoch,
        },
        checkpoint_path,
    )
    export_policy_json(model, output / "policy-weights.json")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "format": "dalmuti-bc-training-result",
        "version": 1,
        "teacherPolicy": "normal",
        "bestEpoch": best_epoch,
        "device": str(device),
        "torchVersion": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDevice": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "trainSamples": len(loaded.train),
        "validationSamples": len(loaded.validation),
        "forcedSamplesSkipped": loaded.forced_samples_skipped,
        "parameterCount": parameter_count,
        "sourceFiles": loaded.files,
        "arguments": vars(args),
    }
    (output / "policy-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "training-metrics.json").write_text(
        json.dumps(
            [asdict(epoch) for epoch in metrics],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive")
    set_seeds(args.seed)
    device = choose_device(args.device)
    loaded = load_rollouts(
        args.data,
        validation_fraction=args.validation_fraction,
        include_forced=args.include_forced,
        max_samples=args.max_samples,
        supervised_weight=args.supervised_weight,
    )
    print(
        f"Loaded {len(loaded.train):,} train and "
        f"{len(loaded.validation):,} validation samples "
        f"from {len(loaded.files)} files."
    )
    print(
        f"Skipped {loaded.forced_samples_skipped:,} forced samples. "
        f"Training device: {device}."
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        tensor_dataset(loaded.train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        tensor_dataset(loaded.validation),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = PolicyNetwork(
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    metrics: list[EpochMetrics] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started_at = time.perf_counter()
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
        )
        validation_loss, validation_accuracy, validation_top3 = evaluate(
            model,
            validation_loader,
            device,
            criterion,
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            validation_top3_accuracy=validation_top3,
            seconds=time.perf_counter() - started_at,
        )
        metrics.append(epoch_metrics)
        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.2%} | "
            f"val loss {validation_loss:.4f} "
            f"acc {validation_accuracy:.2%} "
            f"top3 {validation_top3:.2%} | "
            f"{epoch_metrics.seconds:.1f}s"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    save_outputs(
        Path(args.output),
        model,
        loaded,
        metrics,
        best_epoch,
        device,
        args,
    )
    print(f"Saved best epoch {best_epoch} to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
