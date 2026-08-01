from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from v4_dataset import (
    V4_LOSS_MASK_NAMES,
    V4TrajectoryDataset,
    create_v4_smoke_dataset,
    load_v4_dataset_npz,
)
from v4_export import (
    canonical_json_bytes,
    export_v4_actor_bundle,
    load_v4_actor_checkpoint,
    sha256_file,
    verify_v4_actor_bundle,
)
from v4_model import (
    V4_ACTION_COUNT,
    V4ActorConfig,
    V4CriticConfig,
    V4PrivilegedQCritic,
    V4PublicActor,
    assert_actor_critic_parameter_isolation,
)
from v4_objectives import (
    action_q_regression_loss,
    expected_sarsa_lambda_targets,
    masked_behavior_cloning_loss,
    vrpo_clipped_policy_loss,
)


V4_TRAINING_CHECKPOINT_FORMAT = "dalmuti-v4-training-checkpoint"
V4_TRAINING_CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class V4TrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation: int = 1
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    # Safe defaults are BC-only.  PPO/critic use must be explicit and is
    # admitted only for samples provenance-bound to the PPO collector.
    bc_weight: float = 1.0
    ppo_weight: float = 0.0
    critic_weight: float = 0.0
    q_boost_coefficient: float = 0.0
    gamma: float = 1.0
    lambda_: float = 0.95
    clip_ratio: float = 0.15
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 1.0
    seed: int = 20260801
    amp: bool = True
    num_workers: int = 0
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "gradient_accumulation",
            "checkpoint_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "max_gradient_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "weight_decay",
            "bc_weight",
            "ppo_weight",
            "critic_weight",
            "q_boost_coefficient",
            "entropy_coefficient",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("gamma", "lambda_"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not 0.0 < float(self.clip_ratio) < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")
        if self.bc_weight == 0.0 and self.ppo_weight == 0.0:
            raise ValueError("training requires a positive BC or PPO Actor loss")
        if self.q_boost_coefficient > 0.0 and (
            self.ppo_weight == 0.0 or self.critic_weight == 0.0
        ):
            raise ValueError(
                "Q boost requires positive PPO and critic loss weights"
            )
        if self.entropy_coefficient > 0.0 and self.ppo_weight == 0.0:
            raise ValueError("entropy regularization requires a positive PPO loss")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: Path, device: torch.device) -> dict[str, object]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict):
        raise ValueError("V4 training checkpoint must contain an object")
    return value


def _save_checkpoint(
    output: Path,
    epoch: int,
    global_step: int,
    actor: V4PublicActor,
    critic: V4PrivilegedQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scaler: object,
    dataset: V4TrajectoryDataset,
    training_config: V4TrainingConfig,
) -> Path:
    checkpoint = {
        "format": V4_TRAINING_CHECKPOINT_FORMAT,
        "version": V4_TRAINING_CHECKPOINT_VERSION,
        "completedEpoch": epoch,
        "globalStep": global_step,
        "datasetFingerprint": dataset.fingerprint,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "actorConfig": actor.config.to_dict(),
        "criticConfig": critic.config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "actorState": actor.state_dict(),
        "criticState": critic.state_dict(),
        "actorOptimizerState": actor_optimizer.state_dict(),
        "criticOptimizerState": critic_optimizer.state_dict(),
        "scalerState": scaler.state_dict(),
        "torchRngState": torch.get_rng_state(),
        "numpyRngState": np.random.get_state(),
        "pythonRngState": random.getstate(),
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    path = output / "checkpoints" / f"epoch-{epoch:04d}.pt"
    _atomic_write(path, buffer.getvalue())
    latest = {
        "format": V4_TRAINING_CHECKPOINT_FORMAT,
        "version": V4_TRAINING_CHECKPOINT_VERSION,
        "completedEpoch": epoch,
        "globalStep": global_step,
        "datasetFingerprint": dataset.fingerprint,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "checkpoint": str(path.relative_to(output)).replace("\\", "/"),
        "sha256": sha256_file(path),
    }
    _atomic_write(output / "latest.json", canonical_json_bytes(latest))
    return path


def _resolve_resume(output: Path, resume: str | Path | None) -> Path | None:
    if resume is None:
        return None
    if str(resume) != "latest":
        return Path(resume).resolve()
    latest_path = output / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = output / latest["checkpoint"]
    if latest.get("sha256") != sha256_file(checkpoint):
        raise ValueError("latest V4 checkpoint checksum does not match")
    return checkpoint.resolve()


def _resume_training(
    checkpoint_path: Path,
    actor: V4PublicActor,
    critic: V4PrivilegedQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scaler: object,
    dataset: V4TrajectoryDataset,
    training_config: V4TrainingConfig,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = _torch_load(checkpoint_path, device)
    if (
        checkpoint.get("format") != V4_TRAINING_CHECKPOINT_FORMAT
        or checkpoint.get("version") != V4_TRAINING_CHECKPOINT_VERSION
    ):
        raise ValueError("unsupported V4 training checkpoint")
    if checkpoint.get("datasetFingerprint") != dataset.fingerprint:
        raise ValueError("resume dataset fingerprint does not match")
    if checkpoint.get("lossContractFingerprint") != dataset.loss_contract_fingerprint:
        raise ValueError("resume loss eligibility contract does not match")
    if checkpoint.get("actorConfig") != actor.config.to_dict():
        raise ValueError("resume actor configuration does not match")
    if checkpoint.get("criticConfig") != critic.config.to_dict():
        raise ValueError("resume critic configuration does not match")
    old_training = checkpoint.get("trainingConfig")
    if not isinstance(old_training, dict):
        raise ValueError("resume training configuration is missing")
    current_training = training_config.to_dict()
    for name, value in old_training.items():
        if name == "epochs":
            continue
        if current_training.get(name) != value:
            raise ValueError(f"resume training setting changed: {name}")
    actor.load_state_dict(checkpoint["actorState"], strict=True)
    critic.load_state_dict(checkpoint["criticState"], strict=True)
    actor_optimizer.load_state_dict(checkpoint["actorOptimizerState"])
    critic_optimizer.load_state_dict(checkpoint["criticOptimizerState"])
    scaler.load_state_dict(checkpoint["scalerState"])
    torch.set_rng_state(checkpoint["torchRngState"].cpu())
    np.random.set_state(checkpoint["numpyRngState"])
    random.setstate(checkpoint["pythonRngState"])
    completed_epoch = int(checkpoint["completedEpoch"])
    global_step = int(checkpoint["globalStep"])
    if completed_epoch >= training_config.epochs:
        raise ValueError("resume checkpoint already reached the requested epochs")
    return completed_epoch, global_step


def _flatten_time(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _batch_to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device=device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def _last_used_column(mask: torch.Tensor) -> int:
    """Return the exclusive width of the last non-padding column.

    V4 NPZ files retain the full portable p10/192-event shapes.  Feeding all
    of those zero-padded tokens through attention would waste most of the RTX
    3080 memory, so each CPU batch is narrowed before it is copied to CUDA.
    """

    if mask.dtype != torch.bool or mask.ndim < 2:
        raise ValueError("padding masks must be boolean with a feature axis")
    columns = mask.reshape(-1, mask.shape[-1]).any(dim=0)
    used = columns.nonzero(as_tuple=False)
    return 0 if used.numel() == 0 else int(used[-1, 0]) + 1


def _trim_public_padding(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Trim only contiguous public-token padding; trajectory time is intact."""

    player_width = _last_used_column(batch["player_mask"])
    if player_width < 1:
        raise ValueError("every V4 batch requires at least one public player")
    history_width = _last_used_column(batch["history_mask"])
    result = dict(batch)
    result["player_features"] = batch["player_features"][..., :player_width, :]
    result["player_mask"] = batch["player_mask"][..., :player_width]
    result["history_features"] = batch["history_features"][..., :history_width, :]
    result["history_mask"] = batch["history_mask"][..., :history_width]
    return result


def _epoch_metrics(output: Path) -> list[dict[str, object]]:
    metrics_directory = output / "metrics"
    if not metrics_directory.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(metrics_directory.glob("epoch-*.json"))
    ]


def _resolve_training_contract(
    dataset: V4TrajectoryDataset,
    training_config: V4TrainingConfig,
    *,
    resume: str | Path | None,
    initial_actor_sha256: str | None,
) -> dict[str, object]:
    """Fail closed before model/output creation when a loss lacks provenance."""

    eligibility = dataset.loss_eligibility
    if eligibility is None or dataset.loss_contract_fingerprint is None:
        raise ValueError("V4 dataset has no bound loss eligibility contract")
    counts = {
        "behaviorCloning": int(eligibility.behavior_cloning.sum()),
        "ppo": int(eligibility.ppo.sum()),
        "critic": int(eligibility.critic.sum()),
    }
    requested = {
        "behaviorCloning": training_config.bc_weight,
        "ppo": training_config.ppo_weight,
        "critic": training_config.critic_weight,
    }
    for name, weight in requested.items():
        if weight > 0.0 and counts[name] == 0:
            raise ValueError(
                f"{name} loss was requested but the dataset has no eligible samples"
            )
    if (
        training_config.bc_weight == 0.0
        and not torch.equal(
            eligibility.ppo, dataset.tensors.valid_masks
        )
    ):
        raise ValueError(
            "PPO-only training requires every valid sample to be PPO-eligible; "
            "use a positive BC weight for mixed data"
        )
    if training_config.ppo_weight > 0.0:
        actor_hashes = eligibility.behavior_actor_sha256s
        if len(actor_hashes) != 1:
            raise ValueError(
                "PPO training requires exactly one bound behavior Actor checkpoint"
            )
        if resume is None:
            if initial_actor_sha256 is None:
                raise ValueError(
                    "fresh PPO training requires --initialize-actor-bundle"
                )
            if initial_actor_sha256 != actor_hashes[0]:
                raise ValueError(
                    "PPO initialization Actor does not match the collector behavior Actor"
                )
    return {
        "version": 1,
        "preparationFormat": eligibility.preparation_format,
        "preparationVersion": eligibility.preparation_version,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "masks": dict(V4_LOSS_MASK_NAMES),
        "eligibleSampleCounts": counts,
        "requestedWeights": requested,
        "ppoBehaviorActorSha256s": list(eligibility.behavior_actor_sha256s),
        "normalAndDaggerAreBcOnly": True,
        "ppoAndCriticAdmitOnlyPpoCollectorSamples": True,
    }


def train_v4(
    dataset: V4TrajectoryDataset,
    output_directory: str | Path,
    training_config: V4TrainingConfig,
    *,
    device: str | torch.device = "cpu",
    resume: str | Path | None = None,
    initialize_actor_bundle: str | Path | None = None,
    include_onnx: bool = False,
) -> dict[str, object]:
    output = Path(output_directory).resolve()
    if resume is not None and initialize_actor_bundle is not None:
        raise ValueError("resume and fresh Actor initialization are mutually exclusive")
    bundle_path: Path | None = None
    bundle_manifest: dict[str, object] | None = None
    initial_actor_sha256: str | None = None
    if initialize_actor_bundle is not None:
        bundle_path = Path(initialize_actor_bundle).resolve()
        bundle_manifest = verify_v4_actor_bundle(bundle_path)
        files = bundle_manifest.get("files")
        if not isinstance(files, dict) or not isinstance(files.get("actor.pt"), dict):
            raise ValueError("initial Actor bundle lacks its actor checkpoint binding")
        actor_record = files["actor.pt"]
        initial_actor_sha256 = actor_record.get("sha256")
        if not isinstance(initial_actor_sha256, str):
            raise ValueError("initial Actor bundle lacks its actor SHA-256")
    training_contract = _resolve_training_contract(
        dataset,
        training_config,
        resume=resume,
        initial_actor_sha256=initial_actor_sha256,
    )
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    output.mkdir(parents=True, exist_ok=True)
    use_amp = bool(training_config.amp and device_value.type == "cuda")
    random.seed(training_config.seed)
    np.random.seed(training_config.seed % (2**32))
    torch.manual_seed(training_config.seed)
    if device_value.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)

    actor = V4PublicActor(dataset.actor_config).to(device_value)
    critic = V4PrivilegedQCritic(dataset.critic_config).to(device_value)
    assert_actor_critic_parameter_isolation(actor, critic)
    initial_actor: dict[str, object] | None = None
    if bundle_path is not None:
        assert bundle_manifest is not None
        initialized_actor, _ = load_v4_actor_checkpoint(
            bundle_path / "actor.pt"
        )
        if not isinstance(initialized_actor, V4PublicActor):
            raise ValueError("fresh training initialization requires one Actor")
        initialized_actor = initialized_actor.to(device_value)
        if initialized_actor.config.to_dict() != actor.config.to_dict():
            raise ValueError("initial Actor bundle configuration does not match dataset")
        actor.load_state_dict(initialized_actor.state_dict(), strict=True)
        initial_actor = {
            "actorSha256": initial_actor_sha256,
            "manifestSha256": sha256_file(bundle_path / "manifest.json"),
        }
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=training_config.actor_learning_rate,
        weight_decay=training_config.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=training_config.critic_learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    global_step = 0
    checkpoint_path = _resolve_resume(output, resume)
    if checkpoint_path is not None:
        start_epoch, global_step = _resume_training(
            checkpoint_path,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            scaler,
            dataset,
            training_config,
            device_value,
        )
    elif (output / "latest.json").exists():
        raise FileExistsError(
            "the output already contains a run; pass resume='latest' or use a new directory"
        )

    run_manifest = {
        "format": "dalmuti-v4-training-run",
        "version": 2,
        "datasetFingerprint": dataset.fingerprint,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "actorConfig": dataset.actor_config.to_dict(),
        "criticConfig": dataset.critic_config.to_dict(),
        "trainingConfig": training_config.to_dict(),
        "device": str(device_value),
        "ampEnabled": use_amp,
        "initialActor": initial_actor,
        "trainingContract": training_contract,
        "privilegedCriticExported": False,
    }
    manifest_path = output / "run-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_current = dict(run_manifest)
        comparable_existing.get("trainingConfig", {}).pop("epochs", None)
        comparable_current.get("trainingConfig", {}).pop("epochs", None)
        if comparable_existing != comparable_current:
            raise ValueError("existing V4 run manifest does not match this run")
    else:
        _atomic_write(manifest_path, canonical_json_bytes(run_manifest))

    loader_generator = torch.Generator()
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch + 1, training_config.epochs + 1):
        loader_generator.manual_seed(training_config.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            generator=loader_generator,
            pin_memory=device_value.type == "cuda",
        )
        actor.train()
        critic.train()
        totals = {
            "loss": 0.0,
            "policyLoss": 0.0,
            "behaviorCloningLoss": 0.0,
            "criticLoss": 0.0,
            "entropy": 0.0,
            "approxKl": 0.0,
            "clipFraction": 0.0,
            "meanQBoost": 0.0,
        }
        eligible_samples = {
            "behaviorCloning": 0,
            "ppo": 0,
            "critic": 0,
        }
        batches = 0
        optimizer_steps = 0
        for batch_index, cpu_batch in enumerate(loader):
            batch = _batch_to_device(
                _trim_public_padding(cpu_batch), device_value
            )
            valid_flat = batch["valid_masks"].reshape(-1)
            bc_flat = (
                batch[V4_LOSS_MASK_NAMES["behaviorCloning"]].reshape(-1)
                & valid_flat
            )
            ppo_flat = batch[V4_LOSS_MASK_NAMES["ppo"]].reshape(-1) & valid_flat
            critic_flat = (
                batch[V4_LOSS_MASK_NAMES["critic"]].reshape(-1) & valid_flat
            )
            eligible_samples["behaviorCloning"] += int(bc_flat.sum())
            eligible_samples["ppo"] += int(ppo_flat.sum())
            eligible_samples["critic"] += int(critic_flat.sum())
            legal_flat = _flatten_time(batch["legal_masks"]).clone()
            legal_flat[~valid_flat, 0] = True
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_flat = actor(
                    _flatten_time(batch["global_features"]),
                    _flatten_time(batch["rank_features"]),
                    _flatten_time(batch["player_features"]),
                    _flatten_time(batch["player_mask"]),
                    _flatten_time(batch["memory_trace_features"]),
                    _flatten_time(batch["history_features"]),
                    _flatten_time(batch["history_mask"]),
                    legal_flat,
                )
                q_flat = critic(
                    _flatten_time(batch["privileged_states"]), legal_flat
                )
            batch_size, time_steps = batch["actions"].shape
            logits_time = logits_flat.float().reshape(
                batch_size, time_steps, V4_ACTION_COUNT
            ).transpose(0, 1)
            q_time = q_flat.float().reshape(
                batch_size, time_steps, V4_ACTION_COUNT
            ).transpose(0, 1)
            legal_time = legal_flat.reshape(
                batch_size, time_steps, V4_ACTION_COUNT
            ).transpose(0, 1)
            actor_zero = logits_flat.float().sum() * 0.0
            critic_zero = q_flat.float().sum() * 0.0
            policy_loss = actor_zero
            entropy = actor_zero.detach()
            approx_kl = actor_zero.detach()
            clip_fraction = actor_zero.detach()
            mean_q_boost = actor_zero.detach()
            if training_config.ppo_weight > 0.0 and bool(ppo_flat.any()):
                policy_result = vrpo_clipped_policy_loss(
                    logits_flat.float()[ppo_flat],
                    legal_flat[ppo_flat],
                    batch["actions"].reshape(-1)[ppo_flat],
                    batch["old_action_log_probs"].reshape(-1)[ppo_flat].float(),
                    batch["advantages"].reshape(-1)[ppo_flat].float(),
                    q_values=(
                        q_flat.float()[ppo_flat]
                        if training_config.q_boost_coefficient > 0.0
                        else None
                    ),
                    q_boost_coefficient=training_config.q_boost_coefficient,
                    clip_ratio=training_config.clip_ratio,
                    entropy_coefficient=training_config.entropy_coefficient,
                    normalize_advantages=True,
                )
                policy_loss = policy_result.loss
                entropy = policy_result.entropy
                approx_kl = policy_result.approx_kl
                clip_fraction = policy_result.clip_fraction
                mean_q_boost = policy_result.mean_q_boost
                policy_loss_metric = policy_result.policy_loss
            else:
                policy_loss_metric = actor_zero.detach()
            if training_config.bc_weight > 0.0 and bool(bc_flat.any()):
                bc_loss = masked_behavior_cloning_loss(
                    logits_flat.float()[bc_flat],
                    legal_flat[bc_flat],
                    batch["expert_actions"].reshape(-1)[bc_flat],
                )
            else:
                bc_loss = actor_zero
            if training_config.critic_weight > 0.0 and bool(critic_flat.any()):
                targets_time = expected_sarsa_lambda_targets(
                    batch["rewards"].float().transpose(0, 1),
                    batch["dones"].transpose(0, 1),
                    q_time.detach(),
                    logits_time.detach(),
                    legal_time,
                    gamma=training_config.gamma,
                    lambda_=training_config.lambda_,
                    valid_masks=batch[
                        V4_LOSS_MASK_NAMES["critic"]
                    ].transpose(0, 1),
                )
                critic_loss = action_q_regression_loss(
                    q_flat.float()[critic_flat],
                    legal_flat[critic_flat],
                    batch["actions"].reshape(-1)[critic_flat],
                    targets_time.transpose(0, 1).reshape(-1)[critic_flat],
                )
            else:
                critic_loss = critic_zero
            total_loss = actor_zero
            if training_config.ppo_weight > 0.0:
                total_loss = total_loss + training_config.ppo_weight * policy_loss
            if training_config.bc_weight > 0.0:
                total_loss = total_loss + training_config.bc_weight * bc_loss
            if training_config.critic_weight > 0.0:
                total_loss = total_loss + training_config.critic_weight * critic_loss
            group_start = (
                batch_index // training_config.gradient_accumulation
            ) * training_config.gradient_accumulation
            accumulation_group_size = min(
                training_config.gradient_accumulation,
                len(loader) - group_start,
            )
            scaled_loss = total_loss / accumulation_group_size
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % training_config.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(actor_optimizer)
                nn.utils.clip_grad_norm_(
                    actor.parameters(), training_config.max_gradient_norm
                )
                scaler.step(actor_optimizer)
                if any(parameter.grad is not None for parameter in critic.parameters()):
                    scaler.unscale_(critic_optimizer)
                    nn.utils.clip_grad_norm_(
                        critic.parameters(), training_config.max_gradient_norm
                    )
                    scaler.step(critic_optimizer)
                scaler.update()
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                global_step += 1
                optimizer_steps += 1
            batch_metrics = {
                "loss": total_loss,
                "policyLoss": policy_loss_metric,
                "behaviorCloningLoss": bc_loss,
                "criticLoss": critic_loss,
                "entropy": entropy,
                "approxKl": approx_kl,
                "clipFraction": clip_fraction,
                "meanQBoost": mean_q_boost,
            }
            for name, value in batch_metrics.items():
                totals[name] += float(value.detach().cpu())
            batches += 1
        epoch_metrics: dict[str, object] = {
            "epoch": epoch,
            "globalStep": global_step,
            "batches": batches,
            "optimizerSteps": optimizer_steps,
            "eligibleSamplesSeen": eligible_samples,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in epoch_metrics.values()
        ):
            raise RuntimeError("V4 training produced non-finite metrics")
        _atomic_write(
            output / "metrics" / f"epoch-{epoch:04d}.json",
            canonical_json_bytes(epoch_metrics),
        )
        if epoch % training_config.checkpoint_every == 0 or epoch == training_config.epochs:
            _save_checkpoint(
                output,
                epoch,
                global_step,
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                scaler,
                dataset,
                training_config,
            )

    actor.eval()
    candidate_manifest = export_v4_actor_bundle(
        actor,
        output / "candidate",
        metadata={
            "seed": training_config.seed,
            "datasetFingerprint": dataset.fingerprint,
            "lossContractFingerprint": dataset.loss_contract_fingerprint,
            "trainingContract": training_contract,
            "completedEpochs": training_config.epochs,
            "globalStep": global_step,
            "initialActor": initial_actor,
        },
        include_onnx=include_onnx,
    )
    result = {
        "format": "dalmuti-v4-training-result",
        "version": 2,
        "completedEpochs": training_config.epochs,
        "globalStep": global_step,
        "datasetFingerprint": dataset.fingerprint,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "trainingContract": training_contract,
        "metrics": _epoch_metrics(output),
        "candidate": candidate_manifest,
        "privilegedCriticExported": False,
    }
    _atomic_write(output / "result.json", canonical_json_bytes(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the DALMUTI V4 public Transformer with a privileged critic."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", help="checkpoint path or 'latest'")
    parser.add_argument(
        "--initialize-actor-bundle",
        type=Path,
        help="verified public Actor bundle used to initialize a fresh run",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--include-onnx", action="store_true")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--feedforward", type=int)
    parser.add_argument("--action-hidden", type=int)
    parser.add_argument("--max-history", type=int)
    parser.add_argument("--privileged-features", type=int)
    parser.add_argument("--critic-d-model", type=int)
    parser.add_argument("--critic-layers", type=int)
    parser.add_argument("--critic-action-hidden", type=int)
    parser.add_argument("--actor-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--ppo-weight", type=float, default=0.0)
    parser.add_argument("--critic-weight", type=float, default=0.0)
    parser.add_argument("--q-boost-coefficient", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dataset:
        dataset = load_v4_dataset_npz(args.dataset)
    else:
        actor_config = V4ActorConfig(
            # A bare --smoke command is intentionally tiny and CPU-runnable;
            # explicitly supplied values can still exercise production shape.
            d_model=args.d_model or 24,
            layers=args.layers or 1,
            heads=args.heads or 4,
            feedforward=args.feedforward or 48,
            action_hidden=args.action_hidden or 16,
            max_history=args.max_history or 4,
        )
        critic_config = V4CriticConfig(
            privileged_features=args.privileged_features or 24,
            d_model=args.critic_d_model or 24,
            hidden_layers=args.critic_layers or 1,
            action_hidden=args.critic_action_hidden or 16,
        )
        dataset = create_v4_smoke_dataset(
            actor_config,
            critic_config,
            trajectories=max(2, args.batch_size),
            time_steps=3,
            seed=args.seed,
        )
    training_config = V4TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        weight_decay=args.weight_decay,
        bc_weight=args.bc_weight,
        ppo_weight=args.ppo_weight,
        critic_weight=args.critic_weight,
        q_boost_coefficient=args.q_boost_coefficient,
        gamma=args.gamma,
        lambda_=args.lambda_,
        clip_ratio=args.clip_ratio,
        entropy_coefficient=args.entropy_coefficient,
        max_gradient_norm=args.max_gradient_norm,
        seed=args.seed,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
    )
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    result = train_v4(
        dataset,
        args.output,
        training_config,
        device=device,
        resume=args.resume,
        initialize_actor_bundle=args.initialize_actor_bundle,
        include_onnx=args.include_onnx,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "completedEpochs": result["completedEpochs"],
        "globalStep": result["globalStep"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V4TrainingConfig", "main", "train_v4"]
