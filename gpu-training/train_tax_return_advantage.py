"""Train a five-member conservative tax-return advantage ensemble.

Only great-Dalmuti two-card return states are optimized.  Lesser-Dalmuti
one-card return records remain useful for an exclusion audit but never enter a
gradient or checkpoint-selection metric; inference routes them to the exact
normal heuristic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from non_card_action_conditioned import TAX_RETURN_ACTION_FEATURE_LAYOUT
from non_card_counterfactual_dataset import file_sha256
from tax_return_advantage import (
    BASELINE_PROVENANCE,
    BASELINE_PROVENANCE_SHA256,
    TAX_RETURN_ACTION_CATALOGUE_VERSION,
    TAX_RETURN_ACTION_COUNT,
    TAX_RETURN_ACTION_FEATURE_COUNT,
    TAX_RETURN_ADVANTAGE_CONTEXT_FEATURES,
    TAX_RETURN_ADVANTAGE_DEFAULT_MINIMUM_CHIPS,
    TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT,
    TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION,
    TAX_RETURN_ADVANTAGE_MEMBER_COUNT,
    TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
    TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT,
    TAX_RETURN_ADVANTAGE_Z_VALUE,
    TAX_RETURN_OBSERVATION_FEATURE_COUNT,
    TaxReturnBilinearResidualNetwork,
    canonical_json_bytes,
    export_layer_parameters,
    member_parameters_sha256,
    validate_ensemble_payload,
    write_ensemble_json,
)
from tax_return_advantage_dataset import (
    TaxAdvantageArrays,
    load_tax_return_advantage_dataset,
)


RESULT_FORMAT = "dalmuti-tax-return-advantage-training-result"
RESULT_VERSION = 1
@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    huber_delta_chips: float = 0.5
    regression_coefficient: float = 1.0
    sign_coefficient: float = 0.25
    sign_temperature_chips: float = 0.25
    tie_epsilon_chips: float = 1.0e-9
    context_features: int = TAX_RETURN_ADVANTAGE_CONTEXT_FEATURES
    patience: int = 50
    validation_fraction: float = 0.2
    split_seed: int = 20260801
    seed: int = 202608041
    deterministic: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DALMUTI tax-return baseline-advantage ensemble."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--huber-delta-chips", type=float, default=0.5)
    parser.add_argument("--regression-coefficient", type=float, default=1.0)
    parser.add_argument("--sign-coefficient", type=float, default=0.25)
    parser.add_argument("--sign-temperature-chips", type=float, default=0.25)
    parser.add_argument("--tie-epsilon-chips", type=float, default=1.0e-9)
    parser.add_argument("--context-features", type=int, default=16)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--seed", type=int, default=202608041)
    parser.add_argument(
        "--nondeterministic", action="store_true", help="allow nondeterministic CUDA kernels"
    )
    return parser.parse_args()


def validate_options(options: TrainingOptions) -> None:
    integer_positive = {
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "context_features": options.context_features,
        "patience": options.patience,
    }
    for name, value in integer_positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(options.seed, bool) or options.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(options.split_seed, bool) or options.split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    positive = {
        "learning_rate": options.learning_rate,
        "huber_delta_chips": options.huber_delta_chips,
        "regression_coefficient": options.regression_coefficient,
        "sign_temperature_chips": options.sign_temperature_chips,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    nonnegative = {
        "weight_decay": options.weight_decay,
        "sign_coefficient": options.sign_coefficient,
        "tie_epsilon_chips": options.tie_epsilon_chips,
    }
    for name, value in nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not math.isfinite(options.validation_fraction) or not 0 < options.validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def member_seed(base_seed: int, member_index: int) -> int:
    if not 0 <= member_index < TAX_RETURN_ADVANTAGE_MEMBER_COUNT:
        raise ValueError("member index is out of range")
    return base_seed + member_index * 1_000_003


def group_bootstrap_indices(group_keys: Sequence[str], seed: int) -> np.ndarray:
    groups = sorted(set(group_keys))
    if not groups:
        raise ValueError("group bootstrap requires at least one information state")
    indices_by_group: dict[str, list[int]] = {group: [] for group in groups}
    for index, group in enumerate(group_keys):
        indices_by_group[group].append(index)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(groups, dtype=object), size=len(groups), replace=True)
    indices = [
        index
        for group in sampled.tolist()
        for index in indices_by_group[str(group)]
    ]
    if not indices:
        raise RuntimeError("information-state bootstrap produced no states")
    return np.asarray(indices, dtype=np.int64)


def _tensor_batch(
    arrays: TaxAdvantageArrays,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(arrays.observations[indices]).to(device=device),
        torch.from_numpy(arrays.legal_masks[indices]).to(device=device),
        torch.from_numpy(arrays.baseline_actions[indices]).to(device=device),
        torch.from_numpy(arrays.target_advantages[indices]).to(device=device),
    )


def paired_advantage_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    legal_masks: torch.Tensor,
    baseline_actions: torch.Tensor,
    options: TrainingOptions,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if predicted.shape != targets.shape or predicted.shape != legal_masks.shape:
        raise ValueError("paired advantage tensor shapes do not match")
    action_indices = torch.arange(
        predicted.shape[1], device=predicted.device
    )[None, :]
    paired_mask = legal_masks & (action_indices != baseline_actions[:, None])
    counts = paired_mask.sum(dim=1)
    if not (counts > 0).all():
        raise ValueError("every training state requires a non-baseline legal action")
    regression = functional.smooth_l1_loss(
        predicted,
        targets,
        reduction="none",
        beta=options.huber_delta_chips,
    )
    labels = torch.where(
        targets > options.tie_epsilon_chips,
        torch.ones_like(targets),
        torch.where(
            targets < -options.tie_epsilon_chips,
            torch.zeros_like(targets),
            torch.full_like(targets, 0.5),
        ),
    )
    sign = functional.binary_cross_entropy_with_logits(
        predicted / options.sign_temperature_chips,
        labels,
        reduction="none",
    )

    def state_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * paired_mask).sum(dim=1) / counts

    state_regression = state_mean(regression)
    state_sign = state_mean(sign)
    state_total = (
        options.regression_coefficient * state_regression
        + options.sign_coefficient * state_sign
    )
    return state_total.mean(), {
        "total": state_total.mean(),
        "regression": state_regression.mean(),
        "tieAwareSign": state_sign.mean(),
        "meanAbsoluteError": state_mean((predicted - targets).abs()).mean(),
    }


def _epoch(
    model: TaxReturnBilinearResidualNetwork,
    arrays: TaxAdvantageArrays,
    indices: np.ndarray,
    options: TrainingOptions,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    shuffle_seed: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    ordered = indices.copy()
    if training:
        np.random.default_rng(shuffle_seed).shuffle(ordered)
    totals = {name: 0.0 for name in ("total", "regression", "tieAwareSign", "meanAbsoluteError")}
    state_count = 0
    for start in range(0, len(ordered), options.batch_size):
        batch_indices = ordered[start : start + options.batch_size]
        observations, legal_masks, baseline_actions, targets = _tensor_batch(
            arrays, batch_indices, device
        )
        with torch.set_grad_enabled(training):
            predicted = model(observations, baseline_actions, legal_masks)
            loss, metrics = paired_advantage_loss(
                predicted,
                targets,
                legal_masks,
                baseline_actions,
                options,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_size = len(batch_indices)
        state_count += batch_size
        for name, value in metrics.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / state_count for name, value in totals.items()}


def _checkpoint_payload(
    model: TaxReturnBilinearResidualNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    member_index: int,
    seed: int,
    epoch: int,
    validation_metrics: dict[str, float],
    options: TrainingOptions,
) -> dict[str, object]:
    return {
        "format": "dalmuti-tax-return-advantage-checkpoint",
        "version": 1,
        "memberIndex": member_index,
        "seed": seed,
        "epoch": epoch,
        "validationMetrics": validation_metrics,
        "options": asdict(options),
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
    }


def train_member(
    train: TaxAdvantageArrays,
    validation: TaxAdvantageArrays,
    *,
    member_index: int,
    seed: int,
    options: TrainingOptions,
    device: torch.device,
    member_directory: Path,
    bootstrap_unit: str,
) -> tuple[dict[str, object], dict[str, object]]:
    set_seeds(seed, options.deterministic)
    model = TaxReturnBilinearResidualNetwork(options.context_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    bootstrap_indices = group_bootstrap_indices(train.group_keys, seed)
    validation_indices = np.arange(len(validation), dtype=np.int64)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    best_checkpoint = member_directory / "best.pt"
    member_directory.mkdir(parents=True, exist_ok=False)

    for epoch in range(1, options.epochs + 1):
        train_metrics = _epoch(
            model,
            train,
            bootstrap_indices,
            options,
            device,
            optimizer,
            seed + epoch * 17,
        )
        validation_metrics = _epoch(
            model,
            validation,
            validation_indices,
            options,
            device,
            None,
            seed,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        if validation_metrics["total"] < best_loss - 1.0e-12:
            best_loss = validation_metrics["total"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    member_index=member_index,
                    seed=seed,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                    options=options,
                ),
                best_checkpoint,
            )
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0 or stale_epochs >= options.patience:
            print(
                f"member {member_index + 1}/5 epoch {epoch}: "
                f"train={train_metrics['total']:.6f} "
                f"validation={validation_metrics['total']:.6f}"
            )
        if stale_epochs >= options.patience:
            break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["modelState"])
    model.eval()
    parameters = export_layer_parameters(model)
    member = {
        "memberIndex": member_index,
        "seed": seed,
        "checkpointEpoch": best_epoch,
        "validationPairedLoss": best_loss,
        "parametersSha256": "",
        **parameters,
    }
    member["parametersSha256"] = member_parameters_sha256(member)
    metrics = {
        "memberIndex": member_index,
        "seed": seed,
        "bootstrap": {
            "unit": bootstrap_unit,
            "sourceInformationStates": len(set(train.group_keys)),
            "sampledInformationStateDraws": len(set(train.group_keys)),
            "sampledStatesWithMultiplicity": int(len(bootstrap_indices)),
            "sampledIndexSha256": hashlib.sha256(
                bootstrap_indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "completedEpochs": len(history),
        "bestEpoch": best_epoch,
        "bestValidationPairedLoss": best_loss,
        "bestValidationMetrics": checkpoint["validationMetrics"],
        "history": history,
    }
    (member_directory / "metrics.json").write_bytes(canonical_json_bytes(metrics))
    return member, metrics


def _sample_ids_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_files(root: Path) -> list[dict[str, object]]:
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "training-manifest.json"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]


def train_tax_return_advantage(
    input_patterns: Sequence[str],
    result_directory: str | Path,
    *,
    options: TrainingOptions,
    device: torch.device,
) -> dict[str, object]:
    validate_options(options)
    root = Path(result_directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        dataset = load_tax_return_advantage_dataset(
            input_patterns,
            validation_fraction=options.validation_fraction,
            split_seed=options.split_seed,
        )
        train = dataset.train
        validation = dataset.validation
        member_seeds = [
            member_seed(options.seed, index)
            for index in range(TAX_RETURN_ADVANTAGE_MEMBER_COUNT)
        ]
        members = []
        member_metrics = []
        for index, seed in enumerate(member_seeds):
            member, metrics = train_member(
                train,
                validation,
                member_index=index,
                seed=seed,
                options=options,
                device=device,
                member_directory=root / "members" / f"member-{index}",
                bootstrap_unit=dataset.group_split_key,
            )
            members.append(member)
            member_metrics.append(metrics)

        objective = {
            "utilityTarget": "decision-act-current-chip-advantage",
            "utilityScale": "chip-units",
            "weighting": "equal-per-state",
            "regression": {
                "loss": "huber-paired-action-vs-baseline",
                "coefficient": options.regression_coefficient,
                "deltaChips": options.huber_delta_chips,
            },
            "tieAwareSign": {
                "loss": "binary-cross-entropy-with-logits",
                "coefficient": options.sign_coefficient,
                "temperatureChips": options.sign_temperature_chips,
                "tieTarget": 0.5,
                "tieEpsilonChips": options.tie_epsilon_chips,
            },
            "checkpointSelection": "paired-validation-loss",
            "bootstrapUnit": dataset.group_split_key,
        }
        model_payload = {
            "format": TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT,
            "version": TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION,
            "decisionKind": "tax-return",
            "scoreSemantics": TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
            "observationSchemaVersion": 1,
            "observationFeatures": TAX_RETURN_OBSERVATION_FEATURE_COUNT,
            "actionCatalogueVersion": TAX_RETURN_ACTION_CATALOGUE_VERSION,
            "actionCount": TAX_RETURN_ACTION_COUNT,
            "actionFeatures": TAX_RETURN_ACTION_FEATURE_COUNT,
            "actionFeatureLayout": list(TAX_RETURN_ACTION_FEATURE_LAYOUT),
            "trainingData": dict(dataset.source_contract),
            "architecture": {
                "contextFeatures": options.context_features,
                "contextActivation": "tanh",
                "score": "raw(s,a)-raw(s,normalBaselineAction)",
                "weightLayout": TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT,
            },
            "baseline": {
                "provenance": BASELINE_PROVENANCE,
                "provenanceSha256": BASELINE_PROVENANCE_SHA256,
                "score": "exactly-zero-by-residualization",
            },
            "objective": objective,
            "routing": {
                "returnCountOne": "exact-normal-fallback",
                "returnCountTwo": "ensemble-lower-confidence-bound",
                "roleRouting": {
                    "great-dalmuti": "ensemble-lower-confidence-bound",
                    "lesser-dalmuti": "exact-normal-fallback",
                    "other-roles": "not-applicable",
                },
                "memberCount": TAX_RETURN_ADVANTAGE_MEMBER_COUNT,
                "unanimityRule": "all-member-advantages-strictly-positive",
                "lowerConfidenceBound": "mean-minus-z-times-sample-sd",
                "zValue": TAX_RETURN_ADVANTAGE_Z_VALUE,
                "defaultMinimumChipAdvantage": (
                    TAX_RETURN_ADVANTAGE_DEFAULT_MINIMUM_CHIPS
                ),
                "selection": "maximum-eligible-lcb",
                "tieBreak": "baseline-then-lowest-action-index",
            },
            "members": members,
        }
        validate_ensemble_payload(model_payload)
        write_ensemble_json(model_payload, root / "model.json")
        config = {
            "format": RESULT_FORMAT,
            "version": RESULT_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "torchVersion": torch.__version__,
            "options": asdict(options),
            "memberSeeds": member_seeds,
            "baselineProvenanceSha256": BASELINE_PROVENANCE_SHA256,
            "objective": objective,
        }
        (root / "training-config.json").write_bytes(canonical_json_bytes(config))
        dataset_manifest = {
            "format": RESULT_FORMAT,
            "version": RESULT_VERSION,
            "groupSplitKey": dataset.group_split_key,
            "validationFraction": options.validation_fraction,
            "splitSeed": options.split_seed,
            "sourceFiles": list(dataset.source_files),
            "sourceContract": dict(dataset.source_contract),
            "routing": {
                "returnCountOne": "excluded-from-training-exact-normal-fallback",
                "returnCountTwo": "trained",
            },
            "counts": dict(dataset.exclusion_counts),
            "train": {
                "states": len(train),
                "informationStates": len(set(train.group_keys)),
                "sampleIdsSha256": _sample_ids_sha256(train.sample_ids),
            },
            "validation": {
                "states": len(validation),
                "informationStates": len(set(validation.group_keys)),
                "sampleIdsSha256": _sample_ids_sha256(validation.sample_ids),
            },
        }
        (root / "dataset-manifest.json").write_bytes(
            canonical_json_bytes(dataset_manifest)
        )
        aggregate_metrics = {
            "format": RESULT_FORMAT,
            "version": RESULT_VERSION,
            "selectionMetric": "paired-validation-loss",
            "memberCount": len(member_metrics),
            "members": [
                {
                    key: value
                    for key, value in metrics.items()
                    if key != "history"
                }
                for metrics in member_metrics
            ],
        }
        (root / "training-metrics.json").write_bytes(
            canonical_json_bytes(aggregate_metrics)
        )
        files = _manifest_files(root)
        manifest = {
            "format": RESULT_FORMAT,
            "version": RESULT_VERSION,
            "createdAt": config["createdAt"],
            "scoreSemantics": TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS,
            "baselineProvenanceSha256": BASELINE_PROVENANCE_SHA256,
            "memberSeeds": member_seeds,
            "files": files,
            "totalBytes": sum(int(entry["bytes"]) for entry in files),
        }
        (root / "training-manifest.json").write_bytes(canonical_json_bytes(manifest))
        return {
            "resultDirectory": str(root),
            "modelPath": str(root / "model.json"),
            "members": len(members),
            "trainStates": len(train),
            "validationStates": len(validation),
            "manifestSha256": file_sha256(root / "training-manifest.json"),
        }
    except BaseException:
        # Preserve the fresh run directory and partial artifacts for diagnosis.
        # A new attempt must use a new directory, never resume this one.
        raise


def main() -> None:
    args = parse_args()
    options = TrainingOptions(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        huber_delta_chips=args.huber_delta_chips,
        regression_coefficient=args.regression_coefficient,
        sign_coefficient=args.sign_coefficient,
        sign_temperature_chips=args.sign_temperature_chips,
        tie_epsilon_chips=args.tie_epsilon_chips,
        context_features=args.context_features,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        seed=args.seed,
        deterministic=not args.nondeterministic,
    )
    device = choose_device(args.device)
    report = train_tax_return_advantage(
        args.input,
        args.result_dir,
        options=options,
        device=device,
    )
    print(
        f"Tax advantage ensemble trained: {report['members']} members, "
        f"{report['trainStates']} train / {report['validationStates']} validation states"
    )
    print(f"Model: {report['modelPath']}")
    print(f"Manifest SHA-256: {report['manifestSha256']}")


if __name__ == "__main__":
    main()
