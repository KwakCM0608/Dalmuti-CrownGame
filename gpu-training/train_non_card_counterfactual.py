"""Supervised GPU trainer for exhaustive DALMUTI non-card decisions.

Each record supplies every legal root action in one paired hidden world.  The
actor is trained with confidence-weighted soft policy cross entropy and a
centered action-value regression on all legal actions.  The value network is
trained toward the soft-target policy's expected terminal chip score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from non_card_action_conditioned import (
    RevolutionActionConditionedActorCriticNetwork,
    TaxReturnActionConditionedActorCriticNetwork,
    export_revolution_action_conditioned_json,
    export_tax_return_action_conditioned_json,
)
from non_card_counterfactual_dataset import (
    DecisionArrays,
    DecisionSplit,
    NonCardCounterfactualDatasets,
    file_sha256,
    load_non_card_counterfactuals,
)


TRAINING_RESULT_FORMAT = "dalmuti-non-card-supervised-training-result"
TRAINING_RESULT_VERSION = 3
CHECKPOINT_FORMAT = "dalmuti-non-card-supervised-checkpoint"
CHECKPOINT_VERSION = 3


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    policy_coefficient: float = 0.5
    action_value_coefficient: float = 1.0
    value_coefficient: float = 0.25
    behavior_cloning_coefficient: float = 0.0
    utility_target: str = "terminal"
    entropy_coefficient: float = 0.0
    huber_delta: float = 1.0
    max_gradient_norm: float = 1.0
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1.0e-4
    validation_fraction: float = 0.2
    split_seed: int = 20260801
    seed: int = 20260801
    policy_temperature: float | None = None
    device: str = "auto"
    deterministic: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train separate tax-return and revolution policies from strict "
            "paired counterfactual NDJSON."
        )
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
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
        "--behavior-cloning-coefficient",
        type=float,
        default=0.0,
        help=(
            "Anchor the actor to the deterministic normal baseline with "
            "coefficient * NLL(baselineActionIndex)."
        ),
    )
    parser.add_argument(
        "--utility-target",
        choices=("terminal", "decision-act"),
        default="terminal",
        help=(
            "Train on terminal cumulative chips or the normalized chip award "
            "from the act containing the decision."
        ),
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--policy-temperature",
        type=float,
        help=(
            "Optional ablation temperature. Recompute every legal soft target "
            "from centeredUtility; otherwise use the collector probabilities."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_options(options: TrainingOptions) -> None:
    integer_fields = {
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "early_stopping_patience": options.early_stopping_patience,
    }
    for label, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    for label, value in {
        "learning_rate": options.learning_rate,
        "huber_delta": options.huber_delta,
        "max_gradient_norm": options.max_gradient_norm,
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and greater than zero")
    for label, value in {
        "weight_decay": options.weight_decay,
        "policy_coefficient": options.policy_coefficient,
        "action_value_coefficient": options.action_value_coefficient,
        "value_coefficient": options.value_coefficient,
        "behavior_cloning_coefficient": (
            options.behavior_cloning_coefficient
        ),
        "entropy_coefficient": options.entropy_coefficient,
        "early_stopping_min_delta": options.early_stopping_min_delta,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")
    if (
        options.policy_coefficient
        + options.action_value_coefficient
        + options.value_coefficient
        + options.behavior_cloning_coefficient
        <= 0
    ):
        raise ValueError("at least one supervised loss coefficient must be positive")
    if not 0.0 < options.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if options.split_seed < 0 or options.seed < 0:
        raise ValueError("split_seed and seed must be non-negative")
    if options.policy_temperature is not None and (
        not math.isfinite(options.policy_temperature)
        or options.policy_temperature <= 0
    ):
        raise ValueError("policy_temperature must be finite and greater than zero")
    if options.utility_target not in ("terminal", "decision-act"):
        raise ValueError("utility_target must be terminal or decision-act")


def set_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _json_write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _sample_ids_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _softmax_legal_targets(
    action_values: np.ndarray,
    legal_masks: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    masked = np.where(
        legal_masks,
        action_values.astype(np.float64) / temperatures[:, None],
        -np.inf,
    )
    maximum = np.max(masked, axis=1, keepdims=True)
    exponentials = np.where(legal_masks, np.exp(masked - maximum), 0.0)
    totals = exponentials.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("utility target produced an invalid policy distribution")
    policy_targets = (exponentials / totals).astype(np.float32)
    if not np.allclose(policy_targets.sum(axis=1), 1.0, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError("utility target policy probabilities are inconsistent")
    return policy_targets


def _training_targets(
    arrays: DecisionArrays,
    *,
    utility_target: str,
    temperature: float | None,
) -> DecisionArrays:
    if len(arrays) == 0:
        return arrays
    if utility_target == "terminal":
        if temperature is None:
            return arrays
        temperatures = np.full(len(arrays), temperature, dtype=np.float64)
        policy_targets = _softmax_legal_targets(
            arrays.action_value_targets,
            arrays.legal_masks,
            temperatures,
        )
        value_targets = (
            arrays.value_targets.astype(np.float64)
            + np.sum(
                (
                    policy_targets.astype(np.float64)
                    - arrays.policy_targets.astype(np.float64)
                )
                * arrays.action_value_targets.astype(np.float64),
                axis=1,
            )
        ).astype(np.float32)
        return replace(
            arrays,
            policy_targets=policy_targets,
            value_targets=value_targets,
        )
    if utility_target != "decision-act":
        raise ValueError("utility_target must be terminal or decision-act")

    legal = arrays.legal_masks.astype(np.float64)
    raw_utilities = arrays.decision_act_utilities.astype(np.float64)
    legal_counts = legal.sum(axis=1, keepdims=True)
    centers = (raw_utilities * legal).sum(axis=1, keepdims=True) / legal_counts
    centered_utilities = np.where(
        arrays.legal_masks,
        raw_utilities - centers,
        0.0,
    ).astype(np.float32)
    temperatures = (
        np.full(len(arrays), temperature, dtype=np.float64)
        if temperature is not None
        else arrays.source_policy_temperatures.astype(np.float64)
    )
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise ValueError("decision-act policy temperatures are invalid")
    policy_targets = _softmax_legal_targets(
        centered_utilities,
        arrays.legal_masks,
        temperatures,
    )
    value_targets = np.sum(
        policy_targets.astype(np.float64) * raw_utilities,
        axis=1,
    ).astype(np.float32)
    masked_raw = np.where(arrays.legal_masks, raw_utilities, -np.inf)
    best_actions = np.argmax(masked_raw, axis=1).astype(np.int64)
    return replace(
        arrays,
        policy_targets=policy_targets,
        action_value_targets=centered_utilities,
        value_targets=value_targets,
        best_actions=best_actions,
    )


def apply_utility_target(
    split: DecisionSplit,
    *,
    utility_target: str,
    temperature: float | None,
) -> DecisionSplit:
    return DecisionSplit(
        train=_training_targets(
            split.train,
            utility_target=utility_target,
            temperature=temperature,
        ),
        validation=_training_targets(
            split.validation,
            utility_target=utility_target,
            temperature=temperature,
        ),
    )


def _policy_targets_at_temperature(
    arrays: DecisionArrays,
    temperature: float | None,
) -> DecisionArrays:
    return _training_targets(
        arrays,
        utility_target="terminal",
        temperature=temperature,
    )


def apply_policy_temperature(
    split: DecisionSplit,
    temperature: float | None,
) -> DecisionSplit:
    return DecisionSplit(
        train=_policy_targets_at_temperature(split.train, temperature),
        validation=_policy_targets_at_temperature(split.validation, temperature),
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-8)


def supervised_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    legal_masks: torch.Tensor,
    policy_targets: torch.Tensor,
    action_value_targets: torch.Tensor,
    action_weights: torch.Tensor,
    value_targets: torch.Tensor,
    best_actions: torch.Tensor,
    baseline_actions: torch.Tensor,
    sample_weights: torch.Tensor,
    options: TrainingOptions,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute exhaustive legal-action losses and useful validation metrics."""

    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    legal = legal_masks.to(dtype=logits.dtype)
    confidence = action_weights * legal

    weighted_policy = policy_targets * confidence
    weighted_policy_total = weighted_policy.sum(dim=1).clamp_min(1.0e-8)
    policy_cross_entropy_by_sample = -(
        weighted_policy * log_probabilities
    ).sum(dim=1) / weighted_policy_total
    policy_cross_entropy = _weighted_mean(
        policy_cross_entropy_by_sample, sample_weights
    )

    confidence_weighted_targets = (
        weighted_policy / weighted_policy_total[:, None]
    )
    target_log_probabilities = torch.where(
        confidence_weighted_targets > 0,
        torch.log(confidence_weighted_targets.clamp_min(1.0e-12)),
        torch.zeros_like(confidence_weighted_targets),
    )
    policy_kl_by_sample = (
        confidence_weighted_targets
        * (target_log_probabilities - log_probabilities)
    ).sum(dim=1)
    policy_kl = _weighted_mean(policy_kl_by_sample, sample_weights)

    behavior_cloning_by_sample = -log_probabilities.gather(
        1, baseline_actions[:, None]
    ).squeeze(1)
    behavior_cloning_loss = _weighted_mean(
        behavior_cloning_by_sample, sample_weights
    )

    legal_count = legal.sum(dim=1).clamp_min(1.0)
    centered_logits = logits - (logits * legal).sum(dim=1, keepdim=True) / legal_count[:, None]
    action_losses = torch.nn.functional.huber_loss(
        centered_logits,
        action_value_targets,
        reduction="none",
        delta=options.huber_delta,
    )
    action_denominator = confidence.sum(dim=1).clamp_min(1.0e-8)
    action_value_by_sample = (action_losses * confidence).sum(dim=1) / action_denominator
    action_value_loss = _weighted_mean(action_value_by_sample, sample_weights)

    value_by_sample = torch.nn.functional.huber_loss(
        values,
        value_targets,
        reduction="none",
        delta=options.huber_delta,
    )
    value_loss = _weighted_mean(value_by_sample, sample_weights)

    entropy_by_sample = -(probabilities * log_probabilities).sum(dim=1)
    entropy = _weighted_mean(entropy_by_sample, sample_weights)
    predicted_actions = logits.argmax(dim=1)
    accuracy = _weighted_mean(
        (predicted_actions == best_actions).to(dtype=logits.dtype),
        sample_weights,
    )
    best_targets = action_value_targets.gather(1, best_actions[:, None]).squeeze(1)
    chosen_targets = action_value_targets.gather(1, predicted_actions[:, None]).squeeze(1)
    regret = _weighted_mean(best_targets - chosen_targets, sample_weights)
    baseline_agreement = _weighted_mean(
        (predicted_actions == baseline_actions).to(dtype=logits.dtype),
        sample_weights,
    )
    target_best_equals_baseline = _weighted_mean(
        (best_actions == baseline_actions).to(dtype=logits.dtype),
        sample_weights,
    )
    predicted_logits = logits.gather(
        1, predicted_actions[:, None]
    ).squeeze(1)
    baseline_logits = logits.gather(
        1, baseline_actions[:, None]
    ).squeeze(1)
    predicted_logit_margin = _weighted_mean(
        predicted_logits - baseline_logits, sample_weights
    )
    predicted_probabilities = probabilities.gather(
        1, predicted_actions[:, None]
    ).squeeze(1)
    baseline_probabilities = probabilities.gather(
        1, baseline_actions[:, None]
    ).squeeze(1)
    predicted_probability_margin = _weighted_mean(
        predicted_probabilities - baseline_probabilities, sample_weights
    )
    baseline_targets = action_value_targets.gather(
        1, baseline_actions[:, None]
    ).squeeze(1)
    target_utility_margin = _weighted_mean(
        chosen_targets - baseline_targets, sample_weights
    )

    actor_selection_loss = (
        options.policy_coefficient * policy_cross_entropy
        + options.action_value_coefficient * action_value_loss
        + options.behavior_cloning_coefficient * behavior_cloning_loss
        - options.entropy_coefficient * entropy
    )
    total = actor_selection_loss + options.value_coefficient * value_loss
    return total, {
        "totalLoss": total.detach(),
        "actorSelectionLoss": actor_selection_loss.detach(),
        "policyCrossEntropy": policy_cross_entropy.detach(),
        "policyKl": policy_kl.detach(),
        "behaviorCloningLoss": behavior_cloning_loss.detach(),
        "actionValueLoss": action_value_loss.detach(),
        "valueLoss": value_loss.detach(),
        "entropy": entropy.detach(),
        "bestActionAccuracy": accuracy.detach(),
        "chosenActionRegret": regret.detach(),
        "baselineActionAgreement": baseline_agreement.detach(),
        "targetBestEqualsBaselineRate": (
            target_best_equals_baseline.detach()
        ),
        "predictedLogitMarginVsBaseline": (
            predicted_logit_margin.detach()
        ),
        "predictedProbabilityMarginVsBaseline": (
            predicted_probability_margin.detach()
        ),
        "targetUtilityMarginVsBaseline": target_utility_margin.detach(),
    }


def _batch(
    arrays: DecisionArrays,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    numpy_indices = indices.cpu().numpy()
    return (
        torch.from_numpy(arrays.observations[numpy_indices]).to(device),
        torch.from_numpy(arrays.legal_masks[numpy_indices]).to(device),
        torch.from_numpy(arrays.policy_targets[numpy_indices]).to(device),
        torch.from_numpy(arrays.action_value_targets[numpy_indices]).to(device),
        torch.from_numpy(arrays.action_weights[numpy_indices]).to(device),
        torch.from_numpy(arrays.value_targets[numpy_indices]).to(device),
        torch.from_numpy(arrays.best_actions[numpy_indices]).to(device),
        torch.from_numpy(arrays.baseline_actions[numpy_indices]).to(device),
        torch.from_numpy(arrays.sample_weights[numpy_indices]).to(device),
    )


def _run_epoch(
    model: nn.Module,
    arrays: DecisionArrays,
    *,
    options: TrainingOptions,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        if generator is None:
            raise ValueError("training requires a deterministic shuffle generator")
        order = torch.randperm(len(arrays), generator=generator)
    else:
        order = torch.arange(len(arrays))
    totals: dict[str, float] = {}
    total_weight = 0.0
    for start in range(0, len(arrays), options.batch_size):
        indices = order[start : start + options.batch_size]
        batch = _batch(arrays, indices, device)
        observations, legal_masks = batch[0], batch[1]
        with torch.set_grad_enabled(training):
            logits, values = model(observations, legal_masks)
            total_loss, metrics = supervised_loss(
                logits,
                values,
                legal_masks,
                *batch[2:],
                options,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), options.max_gradient_norm)
                optimizer.step()
        weight = float(batch[-1].sum().detach().cpu())
        total_weight += weight
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * weight
    if total_weight <= 0:
        raise ValueError(f"{arrays.decision} split contains no samples")
    result = {key: value / total_weight for key, value in totals.items()}
    result["targetBestEqualsBaselineRate"] = float(
        np.mean(arrays.best_actions == arrays.baseline_actions)
    )
    return result


def _export_json(decision: str, model: nn.Module, path: Path) -> None:
    if decision == "tax-return":
        export_tax_return_action_conditioned_json(model, path)
    else:
        export_revolution_action_conditioned_json(model, path)


def _checkpoint_payload(
    *,
    decision: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: Mapping[str, object],
    options: TrainingOptions,
    dataset_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "decision": decision,
        "epoch": epoch,
        "behaviorCloningCoefficient": (
            options.behavior_cloning_coefficient
        ),
        "utilityTarget": options.utility_target,
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
        "metrics": dict(metrics),
        "trainingOptions": asdict(options),
        "dataset": dict(dataset_summary),
    }


def _model_factory(decision: str) -> nn.Module:
    if decision == "tax-return":
        return TaxReturnActionConditionedActorCriticNetwork()
    if decision == "revolution":
        return RevolutionActionConditionedActorCriticNetwork()
    raise ValueError(f"unsupported decision: {decision}")


def _split_summary(split: DecisionSplit) -> dict[str, object]:
    def target_best_equals_baseline_rate(arrays: DecisionArrays) -> float | None:
        if len(arrays) == 0:
            return None
        return float(np.mean(arrays.best_actions == arrays.baseline_actions))

    return {
        "groupSplitKey": "canonicalWorldKey",
        "train": {
            "samples": len(split.train),
            "uniqueEpisodes": len(set(split.train.episode_ids)),
            "uniqueWorlds": len(set(split.train.world_keys)),
            "sampleIdsSha256": _sample_ids_sha256(split.train.sample_ids),
            "targetBestEqualsBaselineRate": (
                target_best_equals_baseline_rate(split.train)
            ),
        },
        "validation": {
            "samples": len(split.validation),
            "uniqueEpisodes": len(set(split.validation.episode_ids)),
            "uniqueWorlds": len(set(split.validation.world_keys)),
            "sampleIdsSha256": _sample_ids_sha256(split.validation.sample_ids),
            "targetBestEqualsBaselineRate": (
                target_best_equals_baseline_rate(split.validation)
            ),
        },
    }


def train_decision(
    *,
    decision: str,
    split: DecisionSplit,
    output_directory: Path,
    options: TrainingOptions,
    device: torch.device,
    decision_seed: int,
    model_factory: Callable[[], nn.Module] | None = None,
) -> dict[str, object]:
    if len(split.train) == 0 or len(split.validation) == 0:
        raise ValueError(
            f"{decision} requires non-empty train and validation partitions; "
            f"got {len(split.train)} and {len(split.validation)}"
        )
    if set(split.train.world_keys) & set(split.validation.world_keys):
        raise ValueError(
            f"{decision} has canonicalWorldKey leakage between partitions"
        )
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoints_directory = output_directory / "checkpoints"
    checkpoints_directory.mkdir()
    set_seeds(decision_seed, options.deterministic)
    model = (model_factory or (lambda: _model_factory(decision)))().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(decision_seed)
    dataset_summary = _split_summary(split)
    history: list[dict[str, object]] = []
    best_validation_actor_selection_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_value_loss = math.inf
    best_value_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, options.epochs + 1):
        started_at = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            split.train,
            options=options,
            device=device,
            optimizer=optimizer,
            generator=shuffle_generator,
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model,
                split.validation,
                options=options,
                device=device,
                optimizer=None,
                generator=None,
            )
        improved = (
            validation_metrics["actorSelectionLoss"]
            < best_validation_actor_selection_loss
            - options.early_stopping_min_delta
        )
        if improved:
            best_validation_actor_selection_loss = validation_metrics[
                "actorSelectionLoss"
            ]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if validation_metrics["valueLoss"] < best_validation_value_loss:
            best_validation_value_loss = validation_metrics["valueLoss"]
            best_value_epoch = epoch
        epoch_metrics: dict[str, object] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started_at,
            "train": train_metrics,
            "validation": validation_metrics,
            "improved": improved,
        }
        history.append(epoch_metrics)
        epoch_directory = checkpoints_directory / f"epoch-{epoch:03d}"
        epoch_directory.mkdir()
        torch.save(
            _checkpoint_payload(
                decision=decision,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=epoch_metrics,
                options=options,
                dataset_summary=dataset_summary,
            ),
            epoch_directory / "checkpoint.pt",
        )
        _export_json(decision, model.eval(), epoch_directory / "model.json")
        _json_write_exclusive(epoch_directory / "metrics.json", epoch_metrics)
        if epochs_without_improvement >= options.early_stopping_patience:
            break

    if best_state is None or best_epoch < 1:
        raise RuntimeError(f"{decision} training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    best_directory = output_directory / "best"
    best_directory.mkdir()
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "decision": decision,
            "epoch": best_epoch,
            "behaviorCloningCoefficient": (
                options.behavior_cloning_coefficient
            ),
            "utilityTarget": options.utility_target,
            "modelState": best_state,
            "metrics": history[best_epoch - 1],
            "trainingOptions": asdict(options),
            "dataset": dataset_summary,
        },
        best_directory / "checkpoint.pt",
    )
    _export_json(decision, model, best_directory / "model.json")
    summary = {
        "decision": decision,
        "seed": decision_seed,
        "completedEpochs": len(history),
        "selectionMetric": "validation.actorSelectionLoss",
        "bestEpoch": best_epoch,
        "bestValidationActorSelectionLoss": (
            best_validation_actor_selection_loss
        ),
        "bestValidationValueLossAtActorBest": history[best_epoch - 1][
            "validation"
        ]["valueLoss"],
        "bestValueEpoch": best_value_epoch,
        "bestValidationValueLoss": best_validation_value_loss,
        "stoppedEarly": len(history) < options.epochs,
        "dataset": dataset_summary,
        "history": history,
    }
    _json_write_exclusive(output_directory / "metrics.json", summary)
    return summary


def _dataset_manifest(
    datasets: NonCardCounterfactualDatasets,
    options: TrainingOptions,
    adjusted_splits: Mapping[str, DecisionSplit],
) -> dict[str, object]:
    source_temperatures = sorted(
        {
            float(report.manifest["collection"]["policyTemperature"])
            for report in datasets.files
        }
    )
    decisions: dict[str, object] = {}
    for decision, split in adjusted_splits.items():
        decisions[decision] = _split_summary(split)
    return {
        "format": "dalmuti-non-card-supervised-dataset-binding",
        "version": 3,
        "groupSplitKey": datasets.group_split_key,
        "utilityTarget": options.utility_target,
        "splitSeed": datasets.split_seed,
        "validationFraction": datasets.validation_fraction,
        "policyTargets": {
            "source": (
                "recomputed-from-decisionActUtility"
                if options.utility_target == "decision-act"
                else "recomputed-from-centeredUtility"
                if options.policy_temperature is not None
                else "record-softTargetProbability"
            ),
            "overrideTemperature": options.policy_temperature,
            "sourceManifestTemperatures": source_temperatures,
        },
        "files": [
            {
                "path": report.path,
                "bytes": report.bytes,
                "sha256": report.sha256,
                "decisions": report.decisions,
                "actionEvaluations": report.action_evaluations,
                "manifest": report.manifest,
            }
            for report in datasets.files
        ],
        "decisions": decisions,
        "privacy": {
            "opponentCardIdentitiesIncluded": False,
            "physicalCardIdsIncluded": False,
        },
    }


def _all_result_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path != root / "training-manifest.json"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def train_non_card_models(
    *,
    data_patterns: Sequence[str],
    output_directory: str | Path,
    decision: str = "all",
    options: TrainingOptions = TrainingOptions(),
) -> dict[str, object]:
    validate_options(options)
    if decision not in ("all", "tax-return", "revolution"):
        raise ValueError("decision must be all, tax-return, or revolution")
    output = Path(output_directory).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    device = choose_device(options.device)
    set_seeds(options.seed, options.deterministic)
    config = {
        "format": "dalmuti-non-card-supervised-training-config",
        "version": 3,
        "decision": decision,
        "groupSplitKey": "canonicalWorldKey",
        "behaviorCloningCoefficient": (
            options.behavior_cloning_coefficient
        ),
        "utilityTarget": options.utility_target,
        "options": asdict(options),
        "resolvedDevice": str(device),
        "torchVersion": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDevice": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "lossContract": {
            "policy": (
                "confidence-weighted soft-target cross-entropy; equivalent "
                "to confidence-weighted KL up to target entropy"
            ),
            "actionValue": (
                "confidence-weighted Huber regression from mean-centered "
                "legal actor scores to centeredUtility"
            ),
            "value": (
                "Huber regression to soft-target expected terminal "
                "cumulative chip score"
            ),
            "actionConfidenceWeight": "1 / (1 + standardError^2)",
            "sampleWeight": "sqrt(targetSampleCount)",
            "behaviorCloning": (
                "NLL of the collector baselineActionIndex; coefficient zero "
                "preserves the counterfactual-only objective"
            ),
            "checkpointSelection": (
                "minimum validation actorSelectionLoss = policyCoefficient * "
                "policyCrossEntropy + actionValueCoefficient * "
                "actionValueLoss + behaviorCloningCoefficient * "
                "behaviorCloningLoss - entropyCoefficient * entropy"
            ),
            "valueSelection": (
                "valueLoss is trained and reported separately but does not "
                "select the production actor checkpoint"
            ),
        },
    }
    _json_write_exclusive(output / "training-config.json", config)

    requested = ("tax-return", "revolution") if decision == "all" else (decision,)
    datasets = load_non_card_counterfactuals(
        data_patterns,
        validation_fraction=options.validation_fraction,
        split_seed=options.split_seed,
        allow_mixed_policy_temperatures=options.policy_temperature is not None,
    )
    adjusted_splits: dict[str, DecisionSplit] = {}
    for kind in requested:
        source_split = (
            datasets.tax_return
            if kind == "tax-return"
            else datasets.revolution
        )
        if source_split is None:
            raise ValueError(f"input data contains no {kind} decisions")
        adjusted_splits[kind] = apply_utility_target(
            source_split,
            utility_target=options.utility_target,
            temperature=options.policy_temperature,
        )
    dataset_manifest = _dataset_manifest(
        datasets, options, adjusted_splits
    )
    _json_write_exclusive(output / "dataset-manifest.json", dataset_manifest)

    summaries: dict[str, object] = {}
    for decision_index, kind in enumerate(requested):
        summaries[kind] = train_decision(
            decision=kind,
            split=adjusted_splits[kind],
            output_directory=output / kind,
            options=options,
            device=device,
            decision_seed=options.seed + decision_index,
        )

    metrics_payload = {
        "format": "dalmuti-non-card-supervised-training-metrics",
        "version": 3,
        "groupSplitKey": "canonicalWorldKey",
        "policyTemperatureOverride": options.policy_temperature,
        "behaviorCloningCoefficient": (
            options.behavior_cloning_coefficient
        ),
        "utilityTarget": options.utility_target,
        "decisions": summaries,
    }
    _json_write_exclusive(output / "training-metrics.json", metrics_payload)
    files = _all_result_files(output)
    manifest = {
        "format": TRAINING_RESULT_FORMAT,
        "version": TRAINING_RESULT_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "groupSplitKey": "canonicalWorldKey",
        "decisionKinds": list(requested),
        "policyTemperatureOverride": options.policy_temperature,
        "behaviorCloningCoefficient": (
            options.behavior_cloning_coefficient
        ),
        "utilityTarget": options.utility_target,
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files
        ],
        "totalBytes": sum(path.stat().st_size for path in files),
    }
    _json_write_exclusive(output / "training-manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
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
    manifest = train_non_card_models(
        data_patterns=args.data,
        output_directory=args.output,
        decision=args.decision,
        options=options,
    )
    print(
        f"Non-card training completed: {len(manifest['decisionKinds'])} policies, "
        f"{len(manifest['files'])} verified files"
    )
    print(f"Result directory: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
