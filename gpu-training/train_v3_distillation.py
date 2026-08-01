from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from actor_critic import load_behavior_model
from v3_action_conditioned import (
    V3ActionConditionedActorCriticNetwork,
    export_v3_action_conditioned_json,
)
from v3_distillation_dataset import (
    file_sha256,
    group_split_mask,
    load_v3_distillation_data,
)


TRAINING_RESULT_FORMAT = "dalmuti-v3-distillation-training-result"
TRAINING_RESULT_VERSION = 1
CHECKPOINT_FORMAT = "dalmuti-v3-distillation-checkpoint"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class EvaluationMetrics:
    samples: int
    policyCrossEntropy: float
    teacherEntropy: float
    policyKl: float
    argmaxAgreement: float
    valueMse: float
    valueRmse: float
    valueMae: float
    selectionMetric: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a strict 236-action V3 warm start from legacy PPO."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--value-coefficient", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--seed", type=int, default=202608071)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--binding-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _linear_layers(module: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in module if isinstance(layer, nn.Linear)]


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seeds(seed: int, deterministic: bool) -> None:
    if deterministic:
        # CUDA GEMM determinism requires this before the first CUDA operation.
        # Preserve an explicit caller-provided choice, but supply PyTorch's
        # recommended deterministic workspace when none was configured.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _hardware_report(device: torch.device) -> dict[str, object]:
    gpu: dict[str, object] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "totalMemoryBytes": properties.total_memory,
            "computeCapability": f"{properties.major}.{properties.minor}",
            "multiProcessorCount": properties.multi_processor_count,
        }
    return {
        "format": "dalmuti-v3-distillation-training-hardware",
        "version": 1,
        "platform": platform.platform(),
        "pythonVersion": sys.version,
        "pythonExecutable": sys.executable,
        "numpyVersion": np.__version__,
        "torchVersion": torch.__version__,
        "torchCudaVersion": torch.version.cuda,
        "cudnnVersion": torch.backends.cudnn.version(),
        "cudaAvailable": torch.cuda.is_available(),
        "device": str(device),
        "gpu": gpu,
    }


def initialize_from_teacher(
    model: V3ActionConditionedActorCriticNetwork,
    teacher: nn.Module,
) -> None:
    teacher_trunk = _linear_layers(teacher.trunk)
    actor_trunk = _linear_layers(model.actor_observation_trunk)
    value_layers = _linear_layers(model.value_network)
    if (
        len(teacher_trunk) != len(actor_trunk)
        or len(value_layers) != len(teacher_trunk) + 1
    ):
        raise ValueError("V3 warm-start trunks do not match the legacy teacher")
    with torch.no_grad():
        for teacher_layer, actor_layer, value_layer in zip(
            teacher_trunk, actor_trunk, value_layers[:-1]
        ):
            actor_layer.weight.copy_(teacher_layer.weight)
            actor_layer.bias.copy_(teacher_layer.bias)
            value_layer.weight.copy_(teacher_layer.weight)
            value_layer.bias.copy_(teacher_layer.bias)
        value_layers[-1].weight.copy_(teacher.value_head.weight)
        value_layers[-1].bias.copy_(teacher.value_head.bias)


def _tensor_dataset(data, indices: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(data.observations[indices]),
        torch.from_numpy(data.legal_masks[indices]),
        torch.from_numpy(data.teacher_probabilities[indices]),
        torch.from_numpy(data.teacher_values[indices]),
        torch.from_numpy(data.teacher_argmax_actions[indices]),
    )


def _losses(
    model: V3ActionConditionedActorCriticNetwork,
    observations: torch.Tensor,
    legal_masks: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    teacher_values: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = model(observations, legal_masks)
    log_probabilities = torch.log_softmax(logits / temperature, dim=-1)
    policy_cross_entropy = -(
        teacher_probabilities * log_probabilities
    ).sum(dim=-1)
    safe_teacher_log = torch.where(
        teacher_probabilities > 0,
        torch.log(teacher_probabilities.clamp_min(1.0e-30)),
        torch.zeros_like(teacher_probabilities),
    )
    teacher_entropy = -(
        teacher_probabilities * safe_teacher_log
    ).sum(dim=-1)
    value_squared_error = (values - teacher_values).square()
    predictions = logits.argmax(dim=-1)
    return (
        policy_cross_entropy,
        teacher_entropy,
        value_squared_error,
        (values - teacher_values).abs(),
        predictions,
    )


def evaluate(
    model: V3ActionConditionedActorCriticNetwork,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float,
    value_coefficient: float,
) -> EvaluationMetrics:
    model.eval()
    sample_count = 0
    cross_entropy_sum = 0.0
    entropy_sum = 0.0
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    agreement_sum = 0
    with torch.no_grad():
        for batch in loader:
            observations, masks, targets, values, argmax = (
                tensor.to(device) for tensor in batch
            )
            cross_entropy, entropy, squared_error, absolute_error, predictions = (
                _losses(
                    model,
                    observations,
                    masks,
                    targets,
                    values,
                    temperature=temperature,
                )
            )
            count = observations.shape[0]
            sample_count += count
            cross_entropy_sum += float(cross_entropy.sum())
            entropy_sum += float(entropy.sum())
            squared_error_sum += float(squared_error.sum())
            absolute_error_sum += float(absolute_error.sum())
            agreement_sum += int((predictions == argmax).sum())
    cross_entropy = cross_entropy_sum / sample_count
    entropy = entropy_sum / sample_count
    policy_kl = max(0.0, cross_entropy - entropy)
    value_mse = squared_error_sum / sample_count
    return EvaluationMetrics(
        samples=sample_count,
        policyCrossEntropy=cross_entropy,
        teacherEntropy=entropy,
        policyKl=policy_kl,
        argmaxAgreement=agreement_sum / sample_count,
        valueMse=value_mse,
        valueRmse=math.sqrt(value_mse),
        valueMae=absolute_error_sum / sample_count,
        selectionMetric=policy_kl + value_coefficient * value_mse,
    )


def _checkpoint_payload(
    model: V3ActionConditionedActorCriticNetwork,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    teacher_sha256: str,
    dataset_sha256: str,
    temperature: float,
) -> dict[str, object]:
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "teacherSha256": teacher_sha256,
        "datasetSha256": dataset_sha256,
        "temperature": temperature,
        "modelState": model.state_dict(),
        "optimizerStateIncluded": optimizer is not None,
        "optimizerState": optimizer.state_dict() if optimizer is not None else None,
    }


def _export_preserving_device(
    model: V3ActionConditionedActorCriticNetwork, path: Path
) -> None:
    original_device = next(model.parameters()).device
    try:
        export_v3_action_conditioned_json(model.to("cpu"), path)
    finally:
        model.to(original_device)


def _sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
        or not math.isfinite(args.value_coefficient)
        or args.value_coefficient < 0
        or args.patience < 1
        or not math.isfinite(args.max_gradient_norm)
        or args.max_gradient_norm <= 0
    ):
        raise ValueError("invalid V3 distillation training argument")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"V3 distillation output directory must be fresh: {output}"
        )
    _set_seeds(args.seed, args.deterministic)
    device = _device(args.device)
    hardware = _hardware_report(device)
    data_path = Path(args.data).resolve()
    teacher_path = Path(args.teacher_model).resolve()
    data = load_v3_distillation_data(
        data_path,
        teacher_model_path=teacher_path,
        binding_tolerance=args.binding_tolerance,
        verify_teacher_bindings=True,
    )
    validation_mask = group_split_mask(
        data.group_keys,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_groups = set(data.group_keys[train_indices].tolist())
    validation_groups = set(data.group_keys[validation_indices].tolist())
    if train_groups & validation_groups:
        raise RuntimeError("episode/world split leaked between partitions")
    teacher, teacher_payload = load_behavior_model(teacher_path)
    hidden_sizes = tuple(int(value) for value in teacher_payload["hiddenSizes"])
    model = V3ActionConditionedActorCriticNetwork(
        observation_features=172,
        observation_schema_version=2,
        actor_observation_hidden_sizes=hidden_sizes,
        actor_action_hidden_sizes=(64, 64),
        actor_scorer_hidden_sizes=(256, 128),
        value_hidden_sizes=hidden_sizes,
    )
    initialize_from_teacher(model, teacher)
    # The V3 critic has the exact same 172->legacy-hidden->1 topology as the
    # legacy actor-critic critic path. Keeping this copied network frozen is an
    # exact, lossless value bridge; only the new shared action scorer needs
    # distillation.
    for parameter in model.value_network.parameters():
        parameter.requires_grad_(False)
    model = model.to(device)
    train_loader = DataLoader(
        _tensor_dataset(data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        _tensor_dataset(data, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output.mkdir(parents=True, exist_ok=False)
    dataset_sha256 = file_sha256(data_path)
    metrics: list[dict[str, object]] = []
    best_epoch = 0
    best_metric = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    baseline_validation = evaluate(
        model,
        validation_loader,
        device,
        temperature=data.temperature,
        value_coefficient=args.value_coefficient,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch in train_loader:
            observations, masks, targets, values, _ = (
                tensor.to(device) for tensor in batch
            )
            optimizer.zero_grad(set_to_none=True)
            cross_entropy, _, value_squared_error, _, _ = _losses(
                model,
                observations,
                masks,
                targets,
                values,
                temperature=data.temperature,
            )
            loss = (
                cross_entropy.mean() * data.temperature**2
                + args.value_coefficient * value_squared_error.mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_gradient_norm
            )
            optimizer.step()
            train_loss_sum += float(loss.detach()) * observations.shape[0]
            train_samples += observations.shape[0]
        train_evaluation = evaluate(
            model,
            DataLoader(
                _tensor_dataset(data, train_indices),
                batch_size=args.batch_size,
                shuffle=False,
            ),
            device,
            temperature=data.temperature,
            value_coefficient=args.value_coefficient,
        )
        validation_evaluation = evaluate(
            model,
            validation_loader,
            device,
            temperature=data.temperature,
            value_coefficient=args.value_coefficient,
        )
        epoch_metrics = {
            "epoch": epoch,
            "optimizationLoss": train_loss_sum / train_samples,
            "train": asdict(train_evaluation),
            "validation": asdict(validation_evaluation),
        }
        metrics.append(epoch_metrics)
        checkpoint_directory = output / "checkpoints" / f"epoch-{epoch:03d}"
        checkpoint_directory.mkdir(parents=True, exist_ok=False)
        torch.save(
            _checkpoint_payload(
                model,
                epoch=epoch,
                optimizer=optimizer,
                teacher_sha256=data.teacher_sha256,
                dataset_sha256=dataset_sha256,
                temperature=data.temperature,
            ),
            checkpoint_directory / "checkpoint.pt",
        )
        _export_preserving_device(
            model, checkpoint_directory / "v3-actor-critic-weights.json"
        )
        (checkpoint_directory / "metrics.json").write_text(
            json.dumps(epoch_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"epoch {epoch:03d} | val KL {validation_evaluation.policyKl:.6f} "
            f"argmax {validation_evaluation.argmaxAgreement:.4f} "
            f"value RMSE {validation_evaluation.valueRmse:.6f}"
        )
        if validation_evaluation.selectionMetric < best_metric - 1.0e-10:
            best_metric = validation_evaluation.selectionMetric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("V3 distillation produced no best checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        _checkpoint_payload(
            model,
            epoch=best_epoch,
            optimizer=None,
            teacher_sha256=data.teacher_sha256,
            dataset_sha256=dataset_sha256,
            temperature=data.temperature,
        ),
        output / "checkpoint.pt",
    )
    _export_preserving_device(model, output / "v3-actor-critic-weights.json")
    (output / "training-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    split_summary = {
        "groupSplitKey": "sourceSha256:episodeId",
        "train": {
            "samples": len(train_indices),
            "uniqueGroups": len(train_groups),
            "sampleIdsSha256": _sample_ids_sha256(
                [data.sample_ids[index] for index in train_indices]
            ),
        },
        "validation": {
            "samples": len(validation_indices),
            "uniqueGroups": len(validation_groups),
            "sampleIdsSha256": _sample_ids_sha256(
                [data.sample_ids[index] for index in validation_indices]
            ),
        },
        "overlappingGroups": 0,
    }
    training_manifest = {
        "format": TRAINING_RESULT_FORMAT,
        "version": TRAINING_RESULT_VERSION,
        "teacher": {
            "filename": teacher_path.name,
            "sha256": data.teacher_sha256,
            "format": teacher_payload["format"],
            "actionCount": teacher_payload["actionCount"],
            "temperature": data.temperature,
        },
        "dataset": {
            "filename": data_path.name,
            "sha256": dataset_sha256,
            "samples": len(data),
        },
        "split": split_summary,
        "architecture": {
            "actorObservationHiddenSizes": list(hidden_sizes),
            "actorActionHiddenSizes": [64, 64],
            "actorScorerHiddenSizes": [256, 128],
            "valueHiddenSizes": list(hidden_sizes),
            "initialization": (
                "legacy teacher trunk copied to actor observation and value trunks; "
                "legacy value head copied exactly and the value network frozen"
            ),
        },
        "objective": {
            "policy": "temperature-scaled legal-action distribution cross entropy",
            "value": "lossless frozen copy of the legacy teacher critic",
            "valueCoefficient": args.value_coefficient,
            "selection": "validation policy KL + valueCoefficient * value MSE",
        },
        "arguments": {
            **vars(args),
            "data": data_path.name,
            "teacher_model": teacher_path.name,
            "output": output.name,
        },
        "device": str(device),
        "torchVersion": torch.__version__,
        "hardware": hardware,
        "reproducibility": {
            "seed": args.seed,
            "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnnDeterministic": (
                torch.backends.cudnn.deterministic
                if torch.backends.cudnn.is_available()
                else None
            ),
            "cudnnBenchmark": (
                torch.backends.cudnn.benchmark
                if torch.backends.cudnn.is_available()
                else None
            ),
            "cublasWorkspaceConfig": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "claim": (
                "deterministic algorithms requested; exact cross-host bitwise "
                "identity still depends on matching hardware and libraries"
            ),
        },
        "resultInventory": {
            "version": 1,
            "requiredRootFiles": [
                "checkpoint.pt",
                "training-manifest.json",
                "training-metrics.json",
                "v3-actor-critic-weights.json",
            ],
            "requiredEpochFiles": [
                "checkpoint.pt",
                "metrics.json",
                "v3-actor-critic-weights.json",
            ],
            "optionalProvenanceFiles": [
                "provenance/bundle-manifest.json",
                "provenance/gpu-run-config.json",
                "provenance/handoff-files.sha256",
                "provenance/hardware-report.json",
                "provenance/training.log",
            ],
        },
        "baselineValidation": asdict(baseline_validation),
        "completedEpochs": len(metrics),
        "bestEpoch": best_epoch,
        "bestValidation": metrics[best_epoch - 1]["validation"],
    }
    (output / "training-manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "bestEpoch": best_epoch,
                "bestValidation": metrics[best_epoch - 1]["validation"],
                "completedEpochs": len(metrics),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
