from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_ppo import (
    EpochMetrics,
    choose_device,
    cuda_device_identity,
    deterministic_runtime_metadata,
    file_sha256,
    return_estimator_name,
    set_seeds,
    tensor_dataset,
    train_epoch,
)
from v3_action_conditioned import (
    V3ActionConditionedActorCriticNetwork,
    export_v3_action_conditioned_json,
    load_v3_action_conditioned_json,
)
from v3_ppo_dataset import (
    PpoRollouts,
    build_v3_ppo_data_verification,
    load_v3_ppo_rollouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one strict V3 action-conditioned PPO update."
    )
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--behavior-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-verification-output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
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
    parser.add_argument("--clip-coefficient", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--binding-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--behavior-binding-batch-size", type=int, default=8192
    )
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-id")
    parser.add_argument("--bundle-manifest")
    parser.add_argument("--run-config")
    return parser.parse_args()


def checkpoint_payload(
    model: V3ActionConditionedActorCriticNetwork,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    return {
        "format": "dalmuti-v3-action-conditioned-checkpoint",
        "version": 1,
        "epoch": epoch,
        "modelState": model.state_dict(),
        "observationFeatures": model.observation_features,
        "observationSchemaVersion": model.observation_schema_version,
        "actionCatalogueVersion": 1,
        "actionCount": 236,
        "actorObservationHiddenSizes": model.actor_observation_hidden_sizes,
        "actorActionHiddenSizes": model.actor_action_hidden_sizes,
        "actorScorerHiddenSizes": model.actor_scorer_hidden_sizes,
        "valueHiddenSizes": model.value_hidden_sizes,
        "optimizerStateIncluded": optimizer is not None,
        "optimizerState": optimizer.state_dict() if optimizer is not None else None,
    }


def export_preserving_device(
    model: V3ActionConditionedActorCriticNetwork, path: Path
) -> None:
    device = next(model.parameters()).device
    try:
        export_v3_action_conditioned_json(model.to("cpu"), path)
    finally:
        model.to(device)


def save_epoch(
    output: Path,
    model: V3ActionConditionedActorCriticNetwork,
    optimizer: torch.optim.Optimizer,
    metrics: EpochMetrics,
) -> None:
    directory = output / "checkpoints" / f"epoch-{metrics.epoch:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(
        checkpoint_payload(model, metrics.epoch, optimizer),
        directory / "checkpoint.pt",
    )
    export_preserving_device(model, directory / "v3-actor-critic-weights.json")
    (directory / "metrics.json").write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_outputs(
    output: Path,
    model: V3ActionConditionedActorCriticNetwork,
    rollouts: PpoRollouts,
    metrics: list[EpochMetrics],
    device: torch.device,
    behavior_model: Path,
    args: argparse.Namespace,
    stopped_for_target_kl: bool,
    runtime: dict[str, object],
    bundle_manifest: Path | None,
    run_config: Path | None,
) -> None:
    final_epoch = metrics[-1].epoch
    torch.save(
        checkpoint_payload(model, final_epoch, None), output / "checkpoint.pt"
    )
    export_preserving_device(model, output / "v3-actor-critic-weights.json")
    metadata = {
        "format": "dalmuti-v3-ppo-training-result",
        "version": 1,
        "behaviorModel": str(behavior_model),
        "behaviorModelSha256": file_sha256(behavior_model),
        "rolloutBehaviorModelSha256": rollouts.behavior_model_sha256,
        "modelFormat": "dalmuti-action-conditioned-actor-critic",
        "observationSchemaVersion": 2,
        "observationFeatures": 172,
        "actionCatalogueVersion": 1,
        "actionCount": 236,
        "device": str(device),
        "torchVersion": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDevice": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "gpuIdentity": (
            cuda_device_identity(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            if device.type == "cuda"
            else None
        ),
        "sourceProvenance": (
            {
                "runId": args.run_id,
                "bundleManifestSha256": file_sha256(bundle_manifest),
                "runConfigSha256": file_sha256(run_config),
                "parentModelSha256": file_sha256(behavior_model),
            }
            if bundle_manifest is not None and run_config is not None
            else None
        ),
        "deterministicRuntime": runtime,
        "samples": len(rollouts),
        "trajectories": rollouts.trajectory_count,
        "forcedSamples": int(rollouts.forced.sum()),
        "policySamples": int((~rollouts.forced).sum()),
        "returnEstimator": return_estimator_name(args.gamma, args.gae_lambda),
        "skipForcedPolicyTime": rollouts.skip_forced_policy_time,
        "terminalRankAuxiliaryCoefficient": (
            rollouts.terminal_rank_auxiliary_coefficient
        ),
        "rolloutTemperature": args.rollout_temperature,
        "behaviorBindingsVerified": True,
        "bindingTolerance": args.binding_tolerance,
        "completedEpochs": final_epoch,
        "stoppedForTargetKl": stopped_for_target_kl,
        "sourceFiles": list(rollouts.files),
        "sourceData": [
            {
                "path": f"data/{Path(value).name}",
                "bytes": Path(value).stat().st_size,
                "sha256": file_sha256(Path(value)),
            }
            for value in rollouts.files
        ],
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "data_verification_output",
                "behavior_binding_batch_size",
                "loader_workers",
            }
        },
    }
    (output / "v3-ppo-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "training-metrics.json").write_text(
        json.dumps([asdict(metric) for metric in metrics], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.epochs > 12
        or args.batch_size < 1
        or args.behavior_binding_batch_size < 1
        or args.loader_workers < 1
    ):
        raise ValueError(
            "epochs must be 1..12 and batch sizes must be positive"
        )
    if args.target_kl < 0 or args.max_gradient_norm <= 0:
        raise ValueError("target-kl must be non-negative and max-gradient-norm positive")
    if not np.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0:
        raise ValueError("rollout-temperature must be finite and positive")
    provenance_values = (
        args.run_id,
        args.bundle_manifest,
        args.run_config,
    )
    if any(value is not None for value in provenance_values) and not all(
        value is not None for value in provenance_values
    ):
        raise ValueError(
            "run-id, bundle-manifest, and run-config must be supplied together"
        )
    bundle_manifest = (
        Path(args.bundle_manifest).resolve()
        if args.bundle_manifest is not None
        else None
    )
    run_config = (
        Path(args.run_config).resolve()
        if args.run_config is not None
        else None
    )
    for label, path in (
        ("bundle manifest", bundle_manifest),
        ("run config", run_config),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"V3 {label} is not a regular file: {path}")
    set_seeds(
        args.seed,
        deterministic=True,
        requested_device=args.device,
    )
    device = choose_device(args.device)
    runtime = deterministic_runtime_metadata(args.seed)
    if not runtime["algorithmsEnabled"] or runtime["warnOnly"]:
        raise RuntimeError("strict V3 PPO requires deterministic algorithms")
    if device.type == "cuda" and (
        runtime["pythonHashSeed"] != str(args.seed)
        or runtime["cublasWorkspaceConfig"] != ":4096:8"
    ):
        raise RuntimeError("strict V3 CUDA determinism environment is incomplete")
    behavior_model = Path(args.behavior_model).resolve()
    output = Path(args.output).resolve()
    data_verification_output = Path(args.data_verification_output).resolve()
    if data_verification_output != output / "data-verification.json":
        raise ValueError(
            "data-verification-output must be the canonical fresh report "
            "inside the V3 output directory"
        )
    protected = (
        data_verification_output,
        output / "checkpoint.pt",
        output / "v3-actor-critic-weights.json",
        output / "v3-ppo-metadata.json",
        output / "training-metrics.json",
        output / "checkpoints",
    )
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "V3 training outputs must not already exist: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    source_files: list[dict[str, object]] = []
    rollouts = load_v3_ppo_rollouts(
        args.data,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        skip_forced_policy_time=args.skip_forced_policy_time,
        terminal_rank_auxiliary_coefficient=(
            args.terminal_rank_auxiliary_coefficient
        ),
        behavior_model_path=behavior_model,
        binding_tolerance=args.binding_tolerance,
        behavior_binding_device=device,
        behavior_binding_batch_size=args.behavior_binding_batch_size,
        loader_workers=args.loader_workers,
        source_files_out=source_files,
    )
    data_verification = build_v3_ppo_data_verification(
        rollouts,
        source_files=source_files,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        rollout_temperature=args.rollout_temperature,
        binding_tolerance=args.binding_tolerance,
    )
    with data_verification_output.open("x", encoding="utf-8") as stream:
        json.dump(data_verification, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    model, _ = load_v3_action_conditioned_json(behavior_model)
    if file_sha256(behavior_model) != rollouts.behavior_model_sha256:
        raise RuntimeError("V3 behavior model changed after binding verification")
    model = model.to(device)
    loader = DataLoader(
        tensor_dataset(rollouts),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    metrics: list[EpochMetrics] = []
    stopped = False
    for epoch in range(1, args.epochs + 1):
        values, stop = train_epoch(model, loader, device, optimizer, args)
        values = EpochMetrics(**{**asdict(values), "epoch": epoch})
        metrics.append(values)
        save_epoch(output, model, optimizer, values)
        print(
            f"epoch {epoch:02d} | policy {values.policy_loss:.5f} "
            f"value {values.value_loss:.5f} entropy {values.entropy:.5f} "
            f"KL {values.approximate_kl:.6f}"
        )
        if stop:
            stopped = True
            break
    save_outputs(
        output,
        model,
        rollouts,
        metrics,
        device,
        behavior_model,
        args,
        stopped,
        runtime,
        bundle_manifest,
        run_config,
    )
    print(f"Saved strict V3 PPO update to {output}")


if __name__ == "__main__":
    main()
