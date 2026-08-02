from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from v4_dataset import (
    V4_FIXED_PPO_SOURCE_CONTRACT,
    V4_LOSS_MASK_NAMES,
    V4_MERGED_PREPARATION_FORMAT,
    V4TrajectoryDataset,
    create_v4_smoke_dataset,
    fixed_collection_plan_sha256,
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
    canonical_v4_policy_numerics_contract,
    configure_v4_policy_numerics,
)
from v4_objectives import (
    action_q_regression_loss,
    expected_sarsa_lambda_targets,
    masked_behavior_cloning_loss,
    masked_log_probabilities,
    nonforced_policy_eligibility,
    vrpo_clipped_policy_loss,
)


V4_TRAINING_CHECKPOINT_FORMAT = "dalmuti-v4-training-checkpoint"
V4_TRAINING_CHECKPOINT_VERSION = 2
V4_BALANCED_PLAYER_COUNTS = tuple(range(4, 11))
V4_PLAYER_COUNT_BALANCE_VERSION = 1
V4_FIXED_PPO_EXECUTION_CONTRACT_VERSION = 2
V4_INITIAL_POLICY_REPRODUCTION_AUDIT_VERSION = 2
V4_POST_EPOCH_POLICY_DRIFT_AUDIT_VERSION = 1
V4_POST_EPOCH_POLICY_DRIFT_AUDIT_CONTRACT_VERSION = 1
V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE = 2.0e-5
V4_ENTROPY_COLLAPSE_FRACTION = 0.30
# Policy replays are part of the sealed numerical contract, not a throughput
# workload.  Keep the CUDA batch small and fixed so the audit exercises the
# same kernel shape in pre-training replay and every pre/post-update audit.
V4_CUDA_POLICY_AUDIT_BATCH_SIZE = 4
V4_ACTOR_STATE_SHA256_CONTRACT_VERSION = 1
V4_FIXED_CHECKPOINT_RNG_CONTRACT_VERSION = 1
V4_LATEST_COMMON_FIELDS = frozenset(
    {
        "format",
        "version",
        "completedEpoch",
        "globalStep",
        "datasetFingerprint",
        "lossContractFingerprint",
        "checkpoint",
        "sha256",
    }
)
V4_LATEST_FIXED_FIELDS = frozenset(
    {
        "balanceContractFingerprint",
        "fixedCollectionPlanSha256",
        "fixedPpoExecutionContractFingerprint",
        "initialPolicyReproductionAudit",
        "initialPolicyReproductionAuditFingerprint",
        "fixedCheckpointRngContractVersion",
        "cudaRngStateCount",
        "postEpochPolicyDriftAuditContractFingerprint",
        "postEpochPolicyDriftAudit",
        "postEpochPolicyDriftAuditFingerprint",
    }
)


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _actor_state_sha256(state_dict: Mapping[str, object]) -> str:
    """Hash exact tensor names, dtypes, shapes, and contiguous CPU bytes."""

    digest = hashlib.sha256()
    digest.update(
        f"dalmuti-v4-actor-state-v{V4_ACTOR_STATE_SHA256_CONTRACT_VERSION}\0".encode(
            "ascii"
        )
    )
    if not state_dict:
        raise ValueError("V4 Actor state dictionary is empty")
    if any(not isinstance(name, str) for name in state_dict):
        raise ValueError("V4 Actor state dictionary keys are non-canonical")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("V4 Actor state dictionary is non-canonical")
        contiguous = tensor.detach().cpu().contiguous()
        metadata = canonical_json_bytes(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            }
        )
        raw_octets = contiguous.reshape(-1).view(torch.uint8).reshape(
            -1, contiguous.element_size()
        )
        if sys.byteorder == "big" and contiguous.element_size() > 1:
            component_size = {
                torch.complex64: 4,
                torch.complex128: 8,
            }.get(contiguous.dtype, contiguous.element_size())
            raw_octets = raw_octets.reshape(-1, component_size).flip(-1)
        raw = raw_octets.numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


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
    expected_fixed_collection_plan_sha256: str | None = None

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
        expected_plan = self.expected_fixed_collection_plan_sha256
        if expected_plan is not None and (
            not isinstance(expected_plan, str)
            or len(expected_plan) != 64
            or any(character not in "0123456789abcdef" for character in expected_plan)
        ):
            raise ValueError(
                "expected_fixed_collection_plan_sha256 must be a lowercase SHA-256"
            )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        # Preserve the byte shape of every pre-plan legacy training contract.
        if self.expected_fixed_collection_plan_sha256 is None:
            value.pop("expected_fixed_collection_plan_sha256")
        return value


def _balanced_batch_estimator_multiplier(
    *,
    trajectory_count: int,
    batch_trajectory_count: int,
    total_eligible_rows: int,
) -> float:
    """Return N/(B*C) for a trajectory-uniform balanced-loss minibatch.

    DataLoader samples trajectories, not individual decision rows.  Dividing
    by the number (or total weight) of rows present in a minibatch therefore
    biases variable-length trajectories.  For a uniformly shuffled batch of
    B out of N trajectories, N/(B*C) times the batch weighted-row sum is an
    unbiased estimate of the global C-row objective.
    """

    for value, label in (
        (trajectory_count, "trajectory_count"),
        (batch_trajectory_count, "batch_trajectory_count"),
        (total_eligible_rows, "total_eligible_rows"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if batch_trajectory_count > trajectory_count:
        raise ValueError("batch_trajectory_count cannot exceed trajectory_count")
    return trajectory_count / (batch_trajectory_count * total_eligible_rows)


def _player_count_balance_contract(
    dataset: V4TrajectoryDataset,
    fixed_collection_plan_sha256: str | None = None,
) -> dict[str, object] | None:
    """Build the exact p4..p10 loss contract for fixed-match PPO data."""

    eligibility = dataset.loss_eligibility
    if eligibility is None:
        return None
    fixed_source = V4_FIXED_PPO_SOURCE_CONTRACT
    sources = eligibility.ppo_source_contracts
    contains_fixed = fixed_source in sources
    fixed_only = sources == (fixed_source,)
    if contains_fixed and not fixed_only:
        raise ValueError(
            "mixed fixed-match and legacy PPO source contracts cannot be trained together"
        )
    if eligibility.requires_player_count_balanced_loss != fixed_only:
        raise ValueError(
            "fixed-only PPO source provenance and balanced-loss requirement disagree"
        )
    if not fixed_only:
        return None
    if (
        len(eligibility.ppo_reward_contracts) != 1
        or len(eligibility.ppo_behavior_policy_contracts) != 1
    ):
        raise ValueError(
            "fixed-only PPO training requires one canonical reward and behavior contract"
        )
    if fixed_collection_plan_sha256 is None:
        raise ValueError(
            "fixed-only PPO balanced loss requires one expected collection plan"
        )

    valid = dataset.tensors.valid_masks
    player_count_rows = dataset.tensors.player_mask.sum(dim=-1).to(torch.long)
    valid_player_counts = player_count_rows[valid]
    if (
        valid_player_counts.numel() == 0
        or (valid_player_counts < V4_BALANCED_PLAYER_COUNTS[0]).any()
        or (valid_player_counts > V4_BALANCED_PLAYER_COUNTS[-1]).any()
    ):
        raise ValueError("fixed-match player counts must be present from p4 through p10")
    encoded_player_counts = dataset.tensors.global_features[..., 0]
    expected_encoding = (player_count_rows.to(torch.float32) - 4.0) / 6.0
    if not torch.allclose(
        encoded_player_counts[valid].to(torch.float32),
        expected_encoding[valid],
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise ValueError("fixed-match public player-count tensors disagree")

    masks = {
        "behaviorCloning": nonforced_policy_eligibility(
            dataset.tensors.legal_masks,
            eligibility.behavior_cloning & valid,
        ),
        "ppo": nonforced_policy_eligibility(
            dataset.tensors.legal_masks,
            eligibility.ppo & valid,
        ),
        "critic": eligibility.critic & valid,
    }
    counts_by_loss: dict[str, dict[str, int]] = {}
    totals_by_loss: dict[str, int] = {}
    weights_by_loss: dict[str, dict[str, float]] = {}
    masses_by_loss: dict[str, dict[str, float]] = {}
    total_masses_by_loss: dict[str, float] = {}
    group_count = len(V4_BALANCED_PLAYER_COUNTS)
    for loss_name, mask in masks.items():
        counts = {
            str(player_count): int(
                (mask & (player_count_rows == player_count)).sum().item()
            )
            for player_count in V4_BALANCED_PLAYER_COUNTS
        }
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            raise ValueError(
                f"fixed-match {loss_name} balanced loss is missing eligible rows for "
                + ", ".join(f"p{name}" for name in missing)
            )
        total = sum(counts.values())
        runtime_weights = {
            name: float(np.float32(total / (group_count * count)))
            for name, count in counts.items()
        }
        runtime_masses = {
            name: float(count * runtime_weights[name])
            for name, count in counts.items()
        }
        counts_by_loss[loss_name] = counts
        totals_by_loss[loss_name] = total
        weights_by_loss[loss_name] = runtime_weights
        masses_by_loss[loss_name] = runtime_masses
        total_masses_by_loss[loss_name] = float(sum(runtime_masses.values()))

    contract: dict[str, object] = {
        "version": V4_PLAYER_COUNT_BALANCE_VERSION,
        "playerCounts": list(V4_BALANCED_PLAYER_COUNTS),
        "playerCountGroupCount": group_count,
        "trajectoryCount": len(dataset),
        "samplingUnit": "trajectory",
        "samplingContract": "uniform shuffled permutation without replacement",
        "eligibleRowCountsByLossAndPlayerCount": counts_by_loss,
        "totalEligibleRowsByLoss": totals_by_loss,
        "runtimeFloat32WeightsByLossAndPlayerCount": weights_by_loss,
        "runtimeWeightMassByLossAndPlayerCount": masses_by_loss,
        "runtimeTotalWeightMassByLoss": total_masses_by_loss,
        "exactWeightFormula": "C_total_loss / (7 * C_loss_player_count)",
        "actorEligibility": (
            "loss eligibility AND valid row AND legal-action count greater than one"
        ),
        "criticEligibility": "critic loss eligibility AND valid row",
        "optimizerEstimator": (
            "objective_weighted_mean * batch_weight_sum / "
            "(actual_batch_trajectory_count * C_total_loss / trajectory_count)"
        ),
        "equivalentOptimizerEstimator": (
            "trajectory_count / (actual_batch_trajectory_count * C_total_loss) "
            "* sum_batch(runtime_float32_weight * row_loss)"
        ),
        "minibatchWeightRenormalization": False,
        "epochDiagnosticReduction": (
            "sum_epoch(runtime_float32_weight * metric_row) / "
            "sum_epoch(runtime_float32_weight)"
        ),
        "ppoRewardContract": eligibility.ppo_reward_contracts[0],
        "ppoBehaviorPolicyContract": (
            eligibility.ppo_behavior_policy_contracts[0]
        ),
        "fixedCollectionPlanId": eligibility.fixed_collection_plan_ids[0],
        "fixedCollectionPlanSha256": fixed_collection_plan_sha256,
        "actorDropout": float(dataset.actor_config.dropout),
        "rolloutTrainerModeDistributionParity": (
            "actorConfig.dropout=0.0; raw masked softmax uses the sealed "
            "FP32 MHA-slowpath math-SDP contract in rollout and train"
        ),
        "fixedPpoActorForwardDtype": "torch.float32",
        "fixedPpoActorAutocastDisabled": True,
        "fixedPpoPolicyNumerics": canonical_v4_policy_numerics_contract(),
        "criticAutocastMayRemainEnabled": True,
        "initialOldCurrentRatioMathematicallyOneForFrozenActor": True,
        "initialOldCurrentLogProbabilityAbsoluteTolerance": (
            V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE
        ),
        "requiresFullDatasetInitialPolicyReproductionAudit": True,
        "requiredDeterministicExecution": {
            "torchDeterministicAlgorithms": True,
            "cudaMatmulTf32": False,
            "cudnnTf32": False,
            "cudnnDeterministic": True,
            "cudnnBenchmark": False,
            "cublasWorkspaceConfig": ":4096:8",
        },
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    contract["balanceContractFingerprint"] = fingerprint
    return contract


def _resolve_fixed_collection_plan_sha256(
    dataset: V4TrajectoryDataset,
    training_config: V4TrainingConfig,
) -> str | None:
    """Admit one precommitted, completely merged fixed collection plan."""

    eligibility = dataset.loss_eligibility
    if eligibility is None:
        return None
    fixed_only = eligibility.ppo_source_contracts == (
        V4_FIXED_PPO_SOURCE_CONTRACT,
    )
    expected = training_config.expected_fixed_collection_plan_sha256
    plan_ids = tuple(getattr(eligibility, "fixed_collection_plan_ids", ()))
    if not fixed_only:
        if plan_ids:
            raise ValueError(
                "non-fixed PPO data unexpectedly carries fixed collection plans"
            )
        if expected is not None:
            raise ValueError(
                "expected_fixed_collection_plan_sha256 is valid only for "
                "fixed-only PPO training"
            )
        return None
    if eligibility.preparation_format != V4_MERGED_PREPARATION_FORMAT:
        raise ValueError(
            "fixed-only PPO training requires one completely merged collection plan"
        )
    if float(dataset.actor_config.dropout) != 0.0:
        raise ValueError(
            "fixed-only PPO training requires actorConfig.dropout=0.0 so "
            "rollout eval and trainer train distributions are identical"
        )
    if expected is None:
        raise ValueError(
            "fixed-only PPO training requires "
            "expected_fixed_collection_plan_sha256"
        )
    if len(plan_ids) != 1:
        raise ValueError(
            "fixed-only PPO training requires exactly one collection plan"
        )
    plan_id = plan_ids[0]
    try:
        actual = fixed_collection_plan_sha256(plan_id)
    except ValueError as error:
        raise ValueError("fixed collection plan ID is non-canonical") from error
    if expected != actual:
        raise ValueError(
            "expected fixed collection plan SHA-256 does not match the corpus"
        )
    return actual


def _configure_fixed_ppo_execution(device: torch.device) -> dict[str, object]:
    """Apply and report the collector-compatible deterministic execution mode."""

    cuda = device.type == "cuda"
    policy_numerics = configure_v4_policy_numerics(device)
    cublas_workspace = (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") if cuda else None
    )
    contract: dict[str, object] = {
        "version": V4_FIXED_PPO_EXECUTION_CONTRACT_VERSION,
        "torchVersion": torch.__version__,
        "numpyVersion": np.__version__,
        "device": str(device),
        "actorForwardDtype": "torch.float32",
        "actorAutocastEnabled": False,
        "criticAutocastControlledByTrainingAmp": True,
        "torchDeterministicAlgorithms": True,
        "cudaMatmulTf32": False if cuda else None,
        "cudnnTf32": False if cuda else None,
        "cudnnDeterministic": True if cuda else None,
        "cudnnBenchmark": False if cuda else None,
        "cublasWorkspaceConfig": cublas_workspace,
        "policyNumerics": policy_numerics,
    }
    contract["executionContractFingerprint"] = hashlib.sha256(
        canonical_json_bytes(contract)
    ).hexdigest()
    return contract


def _fixed_nonforced_ppo_balance_spec(
    dataset: V4TrajectoryDataset,
) -> dict[str, object]:
    """Reconstruct the fixed-policy p4..p10 reduction without trusting metadata."""

    eligibility = dataset.loss_eligibility
    if eligibility is None:
        raise ValueError("fixed PPO policy audit requires loss eligibility")
    valid = dataset.tensors.valid_masks
    eligible = eligibility.ppo & valid
    nonforced = nonforced_policy_eligibility(
        dataset.tensors.legal_masks,
        eligible,
    )
    forced = eligible & ~nonforced
    player_counts = dataset.tensors.player_mask.sum(dim=-1).to(torch.long)
    counts = {
        str(player_count): int(
            (nonforced & (player_counts == player_count)).sum().item()
        )
        for player_count in V4_BALANCED_PLAYER_COUNTS
    }
    if any(count < 1 for count in counts.values()):
        raise ValueError(
            "fixed PPO policy audit requires nonforced rows for every p4..p10"
        )
    total = sum(counts.values())
    weights = {
        name: float(np.float32(total / (len(V4_BALANCED_PLAYER_COUNTS) * count)))
        for name, count in counts.items()
    }
    masses = {
        name: float(counts[name] * weights[name]) for name in counts
    }
    forced_counts = {
        str(player_count): int(
            (forced & (player_counts == player_count)).sum().item()
        )
        for player_count in V4_BALANCED_PLAYER_COUNTS
    }
    return {
        "counts": counts,
        "total": total,
        "weights": weights,
        "masses": masses,
        "totalMass": float(sum(masses.values())),
        "eligibleTotal": int(eligible.sum().item()),
        "forcedCounts": forced_counts,
        "forcedTotal": sum(forced_counts.values()),
    }


def _validate_policy_balance_spec(
    dataset: V4TrajectoryDataset,
    balance_contract: Mapping[str, object] | None,
) -> dict[str, object]:
    spec = _fixed_nonforced_ppo_balance_spec(dataset)
    if balance_contract is None:
        return spec
    if (
        balance_contract.get("eligibleRowCountsByLossAndPlayerCount", {}).get(
            "ppo"
        )
        != spec["counts"]
        or balance_contract.get("totalEligibleRowsByLoss", {}).get("ppo")
        != spec["total"]
        or balance_contract.get("runtimeFloat32WeightsByLossAndPlayerCount", {}).get(
            "ppo"
        )
        != spec["weights"]
        or balance_contract.get("runtimeWeightMassByLossAndPlayerCount", {}).get(
            "ppo"
        )
        != spec["masses"]
        or balance_contract.get("runtimeTotalWeightMassByLoss", {}).get("ppo")
        != spec["totalMass"]
    ):
        raise ValueError("fixed PPO policy audit balance contract drifted")
    return spec


def _policy_audit_batch_size(
    training_config: V4TrainingConfig,
    device: torch.device,
) -> int:
    return (
        V4_CUDA_POLICY_AUDIT_BATCH_SIZE
        if device.type == "cuda"
        else training_config.batch_size
    )


def _post_epoch_policy_drift_audit_contract(
    training_config: V4TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    audit_batch_size = _policy_audit_batch_size(training_config, device)
    contract: dict[str, object] = {
        "version": V4_POST_EPOCH_POLICY_DRIFT_AUDIT_CONTRACT_VERSION,
        "timing": "after every optimizer update in the epoch and before checkpoint",
        "actorMode": "eval",
        "actorForwardDtype": "torch.float32",
        "actorAutocastEnabled": False,
        "datasetTraversal": "full dataset; shuffle=false",
        "auditBatchSelectionRule": (
            "cuda uses fixed 4; cpu uses trainingConfig.batch_size"
        ),
        "auditBatchSize": audit_batch_size,
        "auditMustNotMutateOptimizationRngOrWeights": True,
        "actorModeRestoration": (
            "restore exact pre-audit actor.training boolean in finally"
        ),
        "ppoEligibleRows": "ppo loss eligibility AND valid row",
        "policyMetricRows": (
            "ppo eligible rows AND legal-action count greater than one"
        ),
        "forcedSingletonRows": (
            "reported separately and excluded from every policy aggregate"
        ),
        "playerCountReduction": (
            "sum(runtime_float32_ppo_weight * row_metric) / "
            "sum(runtime_float32_ppo_weight)"
        ),
        "logRatio": (
            "current selected-action FP32 log probability minus stored "
            "behavior selected-action FP32 log probability"
        ),
        "approxKl": "expm1(logRatio) - logRatio, evaluated in float64",
        "clipFraction": (
            "abs(expm1(logRatio)) greater than configured clip ratio"
        ),
        "clipRatio": float(training_config.clip_ratio),
        "entropy": "current legal-action categorical entropy",
        "accumulationDtype": "torch.float64",
        "entropyCollapseThresholdFraction": V4_ENTROPY_COLLAPSE_FRACTION,
        "initialEntropyBinding": (
            "initialPolicyReproductionAudit.nonforcedBalancedEntropy"
        ),
        "actorStateSha256ContractVersion": (
            V4_ACTOR_STATE_SHA256_CONTRACT_VERSION
        ),
        "actorStateSha256Contract": (
            "SHA-256 over sorted state tensor name, dtype, shape, and exact "
            "canonical little-endian contiguous CPU bytes"
        ),
        "checkpointRngContractVersion": V4_FIXED_CHECKPOINT_RNG_CONTRACT_VERSION,
        "checkpointRngContract": (
            "fixed checkpoints store CPU torch, every CUDA device, NumPy, and "
            "Python RNG states; CPU fixed checkpoints bind cudaRngStates=null"
        ),
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    contract["auditContractFingerprint"] = fingerprint
    return contract


def _replay_full_ppo_policy(
    actor: V4PublicActor,
    dataset: V4TrajectoryDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    clip_ratio: float,
    balance_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate every fixed PPO row against its stored behavior action log-prob."""

    eligibility = dataset.loss_eligibility
    if eligibility is None:
        raise ValueError("fixed PPO policy audit requires loss eligibility")
    spec = _validate_policy_balance_spec(dataset, balance_contract)
    weights_by_player_count = spec["weights"]
    assert isinstance(weights_by_player_count, Mapping)
    audit_generator = torch.Generator()
    audit_generator.manual_seed(0)
    torch_rng_state = torch.get_rng_state().clone()
    cuda_rng_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if device.type == "cuda"
        else None
    )
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=audit_generator,
    )
    metric_names = (
        "approxKl",
        "clipFraction",
        "entropy",
        "meanLogRatio",
        "meanAbsoluteLogRatio",
    )
    weighted_numerators = {name: 0.0 for name in metric_names}
    weighted_mass = 0.0
    per_player_count = {
        str(player_count): {
            "count": 0,
            "metricSums": {name: 0.0 for name in metric_names},
            "maximumAbsoluteLogRatio": 0.0,
        }
        for player_count in V4_BALANCED_PLAYER_COUNTS
    }
    forced_counts = {str(player_count): 0 for player_count in V4_BALANCED_PLAYER_COUNTS}
    eligible_count = 0
    nonforced_count = 0
    forced_count = 0
    all_absolute_error_sum = 0.0
    all_maximum_absolute_error = 0.0
    forced_maximum_absolute_log_ratio = 0.0
    maximum_absolute_log_ratio = 0.0
    was_training = actor.training
    actor.eval()
    try:
        with torch.no_grad():
            for cpu_batch in loader:
                batch = _batch_to_device(_trim_public_padding(cpu_batch), device)
                valid = batch["valid_masks"].reshape(-1)
                eligible = (
                    batch[V4_LOSS_MASK_NAMES["ppo"]].reshape(-1) & valid
                )
                if not bool(eligible.any()):
                    continue
                legal = _flatten_time(batch["legal_masks"]).clone()
                legal[~valid, 0] = True
                with torch.cuda.amp.autocast(enabled=False):
                    logits = actor(
                        _flatten_time(batch["global_features"]).float(),
                        _flatten_time(batch["rank_features"]).float(),
                        _flatten_time(batch["player_features"]).float(),
                        _flatten_time(batch["player_mask"]),
                        _flatten_time(batch["memory_trace_features"]).float(),
                        _flatten_time(batch["history_features"]).float(),
                        _flatten_time(batch["history_mask"]),
                        legal,
                    ).float()
                eligible_legal = legal[eligible]
                log_probabilities = masked_log_probabilities(
                    logits[eligible], eligible_legal
                ).float()
                actions = batch["actions"].reshape(-1)[eligible]
                selected_log_probabilities = log_probabilities.gather(
                    1, actions[:, None]
                ).squeeze(1)
                old_log_probabilities = batch["old_action_log_probs"].reshape(-1)[
                    eligible
                ].float()
                log_ratios = (
                    selected_log_probabilities.to(torch.float64)
                    - old_log_probabilities.to(torch.float64)
                )
                if not torch.isfinite(log_ratios).all():
                    raise ValueError(
                        "fixed PPO full policy replay produced non-finite log ratios"
                    )
                absolute_errors = log_ratios.abs()
                eligible_rows = int(log_ratios.numel())
                eligible_count += eligible_rows
                all_absolute_error_sum += float(absolute_errors.sum().cpu())
                all_maximum_absolute_error = max(
                    all_maximum_absolute_error,
                    float(absolute_errors.max().cpu()),
                )

                nonforced = eligible_legal.sum(dim=-1) > 1
                forced = ~nonforced
                player_counts = _flatten_time(batch["player_mask"])[eligible].sum(
                    dim=-1
                ).to(torch.long)
                if bool(forced.any()):
                    forced_log_ratios = log_ratios[forced]
                    forced_count += int(forced.sum().item())
                    forced_maximum_absolute_log_ratio = max(
                        forced_maximum_absolute_log_ratio,
                        float(forced_log_ratios.abs().max().cpu()),
                    )
                    forced_player_counts = player_counts[forced]
                    for player_count in V4_BALANCED_PLAYER_COUNTS:
                        forced_counts[str(player_count)] += int(
                            (forced_player_counts == player_count).sum().item()
                        )
                if not bool(nonforced.any()):
                    continue

                nonforced_count += int(nonforced.sum().item())
                selected_log_ratios = log_ratios[nonforced]
                selected_legal = eligible_legal[nonforced]
                selected_log_prob_matrix = log_probabilities[nonforced].to(
                    torch.float64
                )
                probabilities = selected_log_prob_matrix.exp().masked_fill(
                    ~selected_legal, 0.0
                )
                entropy_rows = -(
                    probabilities
                    * selected_log_prob_matrix.masked_fill(~selected_legal, 0.0)
                ).sum(dim=-1)
                ratio_deltas = torch.expm1(selected_log_ratios)
                metric_rows = {
                    "approxKl": ratio_deltas - selected_log_ratios,
                    "clipFraction": (ratio_deltas.abs() > float(clip_ratio)).to(
                        torch.float64
                    ),
                    "entropy": entropy_rows,
                    "meanLogRatio": selected_log_ratios,
                    "meanAbsoluteLogRatio": selected_log_ratios.abs(),
                }
                if any(not torch.isfinite(rows).all() for rows in metric_rows.values()):
                    raise ValueError(
                        "fixed PPO full policy replay produced non-finite metrics"
                    )
                selected_player_counts = player_counts[nonforced]
                row_weights = torch.zeros_like(selected_log_ratios)
                for player_count in V4_BALANCED_PLAYER_COUNTS:
                    row_weights[selected_player_counts == player_count] = float(
                        weights_by_player_count[str(player_count)]
                    )
                if (row_weights <= 0.0).any():
                    raise ValueError(
                        "fixed PPO full policy replay encountered an unbound player count"
                    )
                weighted_mass += float(row_weights.sum().cpu())
                for name, rows in metric_rows.items():
                    weighted_numerators[name] += float(
                        (row_weights * rows).sum().cpu()
                    )
                maximum_absolute_log_ratio = max(
                    maximum_absolute_log_ratio,
                    float(selected_log_ratios.abs().max().cpu()),
                )
                for player_count in V4_BALANCED_PLAYER_COUNTS:
                    key = str(player_count)
                    selected = selected_player_counts == player_count
                    count = int(selected.sum().item())
                    if count == 0:
                        continue
                    record = per_player_count[key]
                    record["count"] = int(record["count"]) + count
                    metric_sums = record["metricSums"]
                    assert isinstance(metric_sums, dict)
                    for name, rows in metric_rows.items():
                        metric_sums[name] = float(metric_sums[name]) + float(
                            rows[selected].sum().cpu()
                        )
                    record["maximumAbsoluteLogRatio"] = max(
                        float(record["maximumAbsoluteLogRatio"]),
                        float(selected_log_ratios[selected].abs().max().cpu()),
                    )
    finally:
        actor.train(was_training)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)

    expected_counts = spec["counts"]
    expected_total = int(spec["total"])
    if (
        nonforced_count != expected_total
        or {
            key: int(value["count"])
            for key, value in per_player_count.items()
        }
        != expected_counts
        or forced_count != spec["forcedTotal"]
        or forced_counts != spec["forcedCounts"]
        or eligible_count != spec["eligibleTotal"]
    ):
        raise ValueError("fixed PPO full policy replay row counts drifted")
    expected_mass = float(spec["totalMass"])
    if not math.isclose(weighted_mass, expected_mass, rel_tol=2.0e-12, abs_tol=2.0e-10):
        raise ValueError("fixed PPO full policy replay weight mass drifted")
    serialized_by_player_count: dict[str, object] = {}
    masses = spec["masses"]
    assert isinstance(masses, Mapping)
    for key, record in per_player_count.items():
        count = int(record["count"])
        metric_sums = record["metricSums"]
        assert isinstance(metric_sums, Mapping)
        serialized_by_player_count[key] = {
            "count": count,
            "runtimeFloat32Weight": float(weights_by_player_count[key]),
            "weightMass": float(masses[key]),
            **{
                name: float(metric_sums[name]) / count for name in metric_names
            },
            "maximumAbsoluteLogRatio": float(
                record["maximumAbsoluteLogRatio"]
            ),
        }
    return {
        "ppoEligibleRowCount": eligible_count,
        "effectiveNonforcedPpoRowCount": nonforced_count,
        "forcedSingletonPpoRowCount": forced_count,
        "forcedSingletonRowsByPlayerCount": forced_counts,
        "nonforcedRowsByPlayerCount": dict(expected_counts),
        "nonforcedWeightMassByPlayerCount": dict(masses),
        "nonforcedTotalWeightMass": expected_mass,
        "allMeanAbsoluteLogProbabilityError": (
            all_absolute_error_sum / max(1, eligible_count)
        ),
        "allMaximumAbsoluteLogProbabilityError": all_maximum_absolute_error,
        **{
            name: weighted_numerators[name] / weighted_mass
            for name in metric_names
        },
        "maximumAbsoluteLogRatio": maximum_absolute_log_ratio,
        "forcedMaximumAbsoluteLogRatio": forced_maximum_absolute_log_ratio,
        "perPlayerCount": serialized_by_player_count,
    }


def _validate_initial_policy_reproduction_audit(
    value: object,
    dataset: V4TrajectoryDataset,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("fixed PPO initial policy reproduction audit is missing")
    expected_fields = {
        "version",
        "ppoEligibleRowCount",
        "effectiveNonforcedPpoRowCount",
        "forcedSingletonPpoRowCount",
        "forcedSingletonRowsByPlayerCount",
        "nonforcedRowsByPlayerCount",
        "nonforcedWeightMassByPlayerCount",
        "nonforcedTotalWeightMass",
        "nonforcedEntropyByPlayerCount",
        "nonforcedBalancedEntropy",
        "auditBatchSize",
        "actorMode",
        "actorForwardDtype",
        "actorAutocastEnabled",
        "storedOldActionLogProbabilityDtype",
        "absoluteTolerance",
        "maximumAbsoluteLogProbabilityError",
        "meanAbsoluteLogProbabilityError",
        "forcedMaximumAbsoluteLogProbabilityError",
        "passed",
    }
    if set(value) != expected_fields:
        raise ValueError(
            "fixed PPO initial policy reproduction audit fields are non-canonical"
        )
    eligibility = dataset.loss_eligibility
    assert eligibility is not None
    expected_eligible_count = int(
        (eligibility.ppo & dataset.tensors.valid_masks).sum().item()
    )
    spec = _fixed_nonforced_ppo_balance_spec(dataset)
    nonforced_count = value.get("effectiveNonforcedPpoRowCount")
    forced_count = value.get("forcedSingletonPpoRowCount")
    maximum = value.get("maximumAbsoluteLogProbabilityError")
    mean = value.get("meanAbsoluteLogProbabilityError")
    forced_maximum = value.get("forcedMaximumAbsoluteLogProbabilityError")
    entropy = value.get("nonforcedBalancedEntropy")
    tolerance = value.get("absoluteTolerance")
    entropy_by_player_count = value.get("nonforcedEntropyByPlayerCount")
    forced_rows_by_player_count = value.get(
        "forcedSingletonRowsByPlayerCount"
    )
    if (
        value.get("version") != V4_INITIAL_POLICY_REPRODUCTION_AUDIT_VERSION
        or value.get("ppoEligibleRowCount") != expected_eligible_count
        or isinstance(nonforced_count, bool)
        or nonforced_count != spec["total"]
        or isinstance(forced_count, bool)
        or forced_count != expected_eligible_count - int(spec["total"])
        or value.get("nonforcedRowsByPlayerCount") != spec["counts"]
        or value.get("nonforcedWeightMassByPlayerCount") != spec["masses"]
        or value.get("nonforcedTotalWeightMass") != spec["totalMass"]
        or not isinstance(forced_rows_by_player_count, Mapping)
        or set(forced_rows_by_player_count)
        != {str(player_count) for player_count in V4_BALANCED_PLAYER_COUNTS}
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for count in forced_rows_by_player_count.values()
        )
        or dict(forced_rows_by_player_count) != spec["forcedCounts"]
        or sum(forced_rows_by_player_count.values()) != forced_count
        or not isinstance(entropy_by_player_count, Mapping)
        or set(entropy_by_player_count)
        != {str(player_count) for player_count in V4_BALANCED_PLAYER_COUNTS}
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in entropy_by_player_count.values()
        )
        or not isinstance(entropy, (int, float))
        or isinstance(entropy, bool)
        or not math.isfinite(float(entropy))
        or float(entropy) <= 0.0
        or isinstance(value.get("auditBatchSize"), bool)
        or not isinstance(value.get("auditBatchSize"), int)
        or int(value["auditBatchSize"]) < 1
        or value.get("actorMode") != "eval"
        or value.get("actorForwardDtype") != "torch.float32"
        or value.get("actorAutocastEnabled") is not False
        or value.get("storedOldActionLogProbabilityDtype") != "torch.float32"
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or float(tolerance)
        != V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in (maximum, mean, forced_maximum)
        )
        or float(mean) > float(maximum) + 1.0e-12
        or float(forced_maximum) > float(maximum) + 1.0e-12
        or float(maximum) > float(tolerance)
        or value.get("passed") is not True
    ):
        raise ValueError(
            "fixed PPO initial policy reproduction audit is non-canonical or failed"
        )
    masses = spec["masses"]
    assert isinstance(masses, Mapping)
    reconstructed_entropy = sum(
        float(entropy_by_player_count[key]) * float(masses[key])
        for key in masses
    ) / float(spec["totalMass"])
    if not math.isclose(
        float(entropy), reconstructed_entropy, rel_tol=2.0e-10, abs_tol=2.0e-12
    ):
        raise ValueError("fixed PPO initial entropy reduction drifted")
    return dict(value)


def _audit_initial_policy_reproduction(
    actor: V4PublicActor,
    dataset: V4TrajectoryDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    balance_contract: Mapping[str, object] | None = None,
    clip_ratio: float = 0.15,
) -> dict[str, object]:
    """Replay every PPO row with the frozen FP32 Actor before any update."""

    replay = _replay_full_ppo_policy(
        actor,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_ratio=clip_ratio,
        balance_contract=balance_contract,
    )
    maximum_absolute_error = float(
        replay["allMaximumAbsoluteLogProbabilityError"]
    )
    per_player_count = replay["perPlayerCount"]
    assert isinstance(per_player_count, Mapping)
    record: dict[str, object] = {
        "version": V4_INITIAL_POLICY_REPRODUCTION_AUDIT_VERSION,
        "ppoEligibleRowCount": replay["ppoEligibleRowCount"],
        "effectiveNonforcedPpoRowCount": replay[
            "effectiveNonforcedPpoRowCount"
        ],
        "forcedSingletonPpoRowCount": replay["forcedSingletonPpoRowCount"],
        "forcedSingletonRowsByPlayerCount": replay[
            "forcedSingletonRowsByPlayerCount"
        ],
        "nonforcedRowsByPlayerCount": replay["nonforcedRowsByPlayerCount"],
        "nonforcedWeightMassByPlayerCount": replay[
            "nonforcedWeightMassByPlayerCount"
        ],
        "nonforcedTotalWeightMass": replay["nonforcedTotalWeightMass"],
        "nonforcedEntropyByPlayerCount": {
            key: value["entropy"] for key, value in per_player_count.items()
        },
        "nonforcedBalancedEntropy": replay["entropy"],
        "auditBatchSize": batch_size,
        "actorMode": "eval",
        "actorForwardDtype": "torch.float32",
        "actorAutocastEnabled": False,
        "storedOldActionLogProbabilityDtype": "torch.float32",
        "absoluteTolerance": V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE,
        "maximumAbsoluteLogProbabilityError": maximum_absolute_error,
        "meanAbsoluteLogProbabilityError": replay[
            "allMeanAbsoluteLogProbabilityError"
        ],
        "forcedMaximumAbsoluteLogProbabilityError": replay[
            "forcedMaximumAbsoluteLogRatio"
        ],
        "passed": (
            int(replay["ppoEligibleRowCount"]) > 0
            and maximum_absolute_error
            <= V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE
        ),
    }
    if record["passed"] is not True:
        raise ValueError(
            "fixed PPO initial policy reproduction exceeded absolute log-probability "
            f"tolerance: max={maximum_absolute_error:.9g}, "
            f"tolerance={V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE:.9g}"
        )
    return _validate_initial_policy_reproduction_audit(record, dataset)


def _validate_post_epoch_policy_drift_audit(
    value: object,
    dataset: V4TrajectoryDataset,
    training_config: V4TrainingConfig,
    balance_contract: Mapping[str, object],
    initial_policy_reproduction_audit: Mapping[str, object],
    *,
    expected_epoch: int,
    expected_global_step: int,
    fixed_collection_plan_sha256: str,
    fixed_ppo_execution_contract_fingerprint: str,
    audit_contract_fingerprint: str,
    audit_batch_size: int,
    expected_actor_state_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("fixed PPO post-epoch policy drift audit is missing")
    expected_fields = {
        "version",
        "epoch",
        "globalStep",
        "datasetFingerprint",
        "lossContractFingerprint",
        "balanceContractFingerprint",
        "fixedCollectionPlanSha256",
        "fixedPpoExecutionContractFingerprint",
        "initialPolicyReproductionAuditFingerprint",
        "auditContractFingerprint",
        "actorMode",
        "actorForwardDtype",
        "actorAutocastEnabled",
        "storedOldActionLogProbabilityDtype",
        "clipRatio",
        "auditBatchSize",
        "actorStateSha256",
        "ppoEligibleRowCount",
        "effectiveNonforcedPpoRowCount",
        "forcedSingletonPpoRowCount",
        "forcedSingletonRowsByPlayerCount",
        "nonforcedRowsByPlayerCount",
        "nonforcedWeightMassByPlayerCount",
        "nonforcedTotalWeightMass",
        "approxKl",
        "clipFraction",
        "entropy",
        "initialBehaviorEntropy",
        "entropyRetentionRatio",
        "entropyCollapseFraction",
        "entropyCollapseExceeds30Percent",
        "meanLogRatio",
        "meanAbsoluteLogRatio",
        "maximumAbsoluteLogRatio",
        "forcedMaximumAbsoluteLogRatio",
        "perPlayerCount",
    }
    if set(value) != expected_fields:
        raise ValueError(
            "fixed PPO post-epoch policy drift audit fields are non-canonical"
        )
    initial_audit = _validate_initial_policy_reproduction_audit(
        initial_policy_reproduction_audit, dataset
    )
    if initial_audit.get("auditBatchSize") != audit_batch_size:
        raise ValueError(
            "fixed PPO initial/post policy audit batch selection drifted"
        )
    initial_fingerprint = hashlib.sha256(
        canonical_json_bytes(initial_audit)
    ).hexdigest()
    spec = _validate_policy_balance_spec(dataset, balance_contract)
    eligibility = dataset.loss_eligibility
    assert eligibility is not None
    eligible_count = int(
        (eligibility.ppo & dataset.tensors.valid_masks).sum().item()
    )
    expected_forced_count = eligible_count - int(spec["total"])
    fixed_bindings_match = (
        value.get("version") == V4_POST_EPOCH_POLICY_DRIFT_AUDIT_VERSION
        and value.get("epoch") == expected_epoch
        and value.get("globalStep") == expected_global_step
        and value.get("datasetFingerprint") == dataset.fingerprint
        and value.get("lossContractFingerprint")
        == dataset.loss_contract_fingerprint
        and value.get("balanceContractFingerprint")
        == balance_contract.get("balanceContractFingerprint")
        and value.get("fixedCollectionPlanSha256")
        == fixed_collection_plan_sha256
        and value.get("fixedPpoExecutionContractFingerprint")
        == fixed_ppo_execution_contract_fingerprint
        and value.get("initialPolicyReproductionAuditFingerprint")
        == initial_fingerprint
        and value.get("auditContractFingerprint")
        == audit_contract_fingerprint
        and value.get("actorMode") == "eval"
        and value.get("actorForwardDtype") == "torch.float32"
        and value.get("actorAutocastEnabled") is False
        and value.get("storedOldActionLogProbabilityDtype") == "torch.float32"
        and value.get("clipRatio") == float(training_config.clip_ratio)
        and value.get("auditBatchSize") == audit_batch_size
        and _is_lower_sha256(value.get("actorStateSha256"))
        and value.get("actorStateSha256") == expected_actor_state_sha256
        and value.get("ppoEligibleRowCount") == eligible_count
        and value.get("effectiveNonforcedPpoRowCount") == spec["total"]
        and value.get("forcedSingletonPpoRowCount") == expected_forced_count
        and value.get("nonforcedRowsByPlayerCount") == spec["counts"]
        and value.get("nonforcedWeightMassByPlayerCount") == spec["masses"]
        and value.get("nonforcedTotalWeightMass") == spec["totalMass"]
    )
    if not fixed_bindings_match:
        raise ValueError("fixed PPO post-epoch policy drift audit binding drifted")

    forced_counts = value.get("forcedSingletonRowsByPlayerCount")
    if (
        not isinstance(forced_counts, Mapping)
        or set(forced_counts)
        != {str(player_count) for player_count in V4_BALANCED_PLAYER_COUNTS}
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in forced_counts.values()
        )
        or dict(forced_counts) != spec["forcedCounts"]
        or sum(int(count) for count in forced_counts.values())
        != expected_forced_count
    ):
        raise ValueError("fixed PPO post-epoch forced-row evidence drifted")

    metric_names = (
        "approxKl",
        "clipFraction",
        "entropy",
        "meanLogRatio",
        "meanAbsoluteLogRatio",
    )

    def finite_number(item: object) -> bool:
        return (
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
        )

    numeric_names = metric_names + (
        "initialBehaviorEntropy",
        "entropyRetentionRatio",
        "entropyCollapseFraction",
        "maximumAbsoluteLogRatio",
        "forcedMaximumAbsoluteLogRatio",
    )
    if any(not finite_number(value.get(name)) for name in numeric_names):
        raise ValueError("fixed PPO post-epoch policy drift metrics are non-finite")
    if (
        float(value["approxKl"]) < -1.0e-14
        or not 0.0 <= float(value["clipFraction"]) <= 1.0
        or float(value["entropy"]) < 0.0
        or float(value["initialBehaviorEntropy"]) <= 0.0
        or float(value["meanAbsoluteLogRatio"]) < 0.0
        or float(value["maximumAbsoluteLogRatio"]) < 0.0
        or float(value["forcedMaximumAbsoluteLogRatio"]) < 0.0
        or float(value["meanAbsoluteLogRatio"])
        > float(value["maximumAbsoluteLogRatio"]) + 1.0e-12
        or float(value["forcedMaximumAbsoluteLogRatio"])
        > V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE
    ):
        raise ValueError("fixed PPO post-epoch policy drift metric bounds failed")

    per_player_count = value.get("perPlayerCount")
    if (
        not isinstance(per_player_count, Mapping)
        or set(per_player_count)
        != {str(player_count) for player_count in V4_BALANCED_PLAYER_COUNTS}
    ):
        raise ValueError("fixed PPO post-epoch per-player-count audit drifted")
    expected_weights = spec["weights"]
    expected_masses = spec["masses"]
    expected_counts = spec["counts"]
    assert isinstance(expected_weights, Mapping)
    assert isinstance(expected_masses, Mapping)
    assert isinstance(expected_counts, Mapping)
    per_fields = {
        "count",
        "runtimeFloat32Weight",
        "weightMass",
        *metric_names,
        "maximumAbsoluteLogRatio",
    }
    reconstructed = {name: 0.0 for name in metric_names}
    reconstructed_maximum = 0.0
    for key, raw_record in per_player_count.items():
        if not isinstance(raw_record, Mapping) or set(raw_record) != per_fields:
            raise ValueError("fixed PPO post-epoch per-player-count fields drifted")
        if (
            raw_record.get("count") != expected_counts[key]
            or raw_record.get("runtimeFloat32Weight") != expected_weights[key]
            or raw_record.get("weightMass") != expected_masses[key]
            or any(not finite_number(raw_record.get(name)) for name in metric_names)
            or not finite_number(raw_record.get("maximumAbsoluteLogRatio"))
            or float(raw_record["approxKl"]) < -1.0e-14
            or not 0.0 <= float(raw_record["clipFraction"]) <= 1.0
            or float(raw_record["entropy"]) < 0.0
            or float(raw_record["meanAbsoluteLogRatio"]) < 0.0
            or float(raw_record["maximumAbsoluteLogRatio"]) < 0.0
            or float(raw_record["meanAbsoluteLogRatio"])
            > float(raw_record["maximumAbsoluteLogRatio"]) + 1.0e-12
        ):
            raise ValueError("fixed PPO post-epoch per-player-count metric drifted")
        for name in metric_names:
            reconstructed[name] += float(raw_record[name]) * float(
                expected_masses[key]
            )
        reconstructed_maximum = max(
            reconstructed_maximum,
            float(raw_record["maximumAbsoluteLogRatio"]),
        )
    total_mass = float(spec["totalMass"])
    for name in metric_names:
        reconstructed[name] /= total_mass
        if not math.isclose(
            float(value[name]),
            reconstructed[name],
            rel_tol=2.0e-10,
            abs_tol=2.0e-12,
        ):
            raise ValueError(
                f"fixed PPO post-epoch p-balanced {name} reduction drifted"
            )
    if not math.isclose(
        float(value["maximumAbsoluteLogRatio"]),
        reconstructed_maximum,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    ):
        raise ValueError("fixed PPO post-epoch maximum log-ratio drifted")

    initial_entropy = float(initial_audit["nonforcedBalancedEntropy"])
    entropy = float(value["entropy"])
    retention = entropy / initial_entropy
    collapse = 1.0 - retention
    if (
        not math.isclose(
            float(value["initialBehaviorEntropy"]),
            initial_entropy,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(value["entropyRetentionRatio"]),
            retention,
            rel_tol=2.0e-12,
            abs_tol=2.0e-12,
        )
        or not math.isclose(
            float(value["entropyCollapseFraction"]),
            collapse,
            rel_tol=2.0e-12,
            abs_tol=2.0e-12,
        )
        or value.get("entropyCollapseExceeds30Percent")
        is not (collapse > V4_ENTROPY_COLLAPSE_FRACTION)
    ):
        raise ValueError("fixed PPO post-epoch entropy baseline binding drifted")
    return dict(value)


def _audit_post_epoch_policy_drift(
    actor: V4PublicActor,
    dataset: V4TrajectoryDataset,
    training_config: V4TrainingConfig,
    balance_contract: Mapping[str, object],
    initial_policy_reproduction_audit: Mapping[str, object],
    *,
    epoch: int,
    global_step: int,
    device: torch.device,
    fixed_collection_plan_sha256: str,
    fixed_ppo_execution_contract_fingerprint: str,
    audit_contract_fingerprint: str,
    audit_batch_size: int,
) -> dict[str, object]:
    replay = _replay_full_ppo_policy(
        actor,
        dataset,
        device=device,
        batch_size=audit_batch_size,
        num_workers=training_config.num_workers,
        clip_ratio=training_config.clip_ratio,
        balance_contract=balance_contract,
    )
    initial_audit = _validate_initial_policy_reproduction_audit(
        initial_policy_reproduction_audit, dataset
    )
    initial_fingerprint = hashlib.sha256(
        canonical_json_bytes(initial_audit)
    ).hexdigest()
    initial_entropy = float(initial_audit["nonforcedBalancedEntropy"])
    entropy = float(replay["entropy"])
    retention = entropy / initial_entropy
    collapse = 1.0 - retention
    actor_state_sha256 = _actor_state_sha256(actor.state_dict())
    record: dict[str, object] = {
        "version": V4_POST_EPOCH_POLICY_DRIFT_AUDIT_VERSION,
        "epoch": epoch,
        "globalStep": global_step,
        "datasetFingerprint": dataset.fingerprint,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "balanceContractFingerprint": balance_contract[
            "balanceContractFingerprint"
        ],
        "fixedCollectionPlanSha256": fixed_collection_plan_sha256,
        "fixedPpoExecutionContractFingerprint": (
            fixed_ppo_execution_contract_fingerprint
        ),
        "initialPolicyReproductionAuditFingerprint": initial_fingerprint,
        "auditContractFingerprint": audit_contract_fingerprint,
        "actorMode": "eval",
        "actorForwardDtype": "torch.float32",
        "actorAutocastEnabled": False,
        "storedOldActionLogProbabilityDtype": "torch.float32",
        "clipRatio": float(training_config.clip_ratio),
        "auditBatchSize": audit_batch_size,
        "actorStateSha256": actor_state_sha256,
        "ppoEligibleRowCount": replay["ppoEligibleRowCount"],
        "effectiveNonforcedPpoRowCount": replay[
            "effectiveNonforcedPpoRowCount"
        ],
        "forcedSingletonPpoRowCount": replay["forcedSingletonPpoRowCount"],
        "forcedSingletonRowsByPlayerCount": replay[
            "forcedSingletonRowsByPlayerCount"
        ],
        "nonforcedRowsByPlayerCount": replay["nonforcedRowsByPlayerCount"],
        "nonforcedWeightMassByPlayerCount": replay[
            "nonforcedWeightMassByPlayerCount"
        ],
        "nonforcedTotalWeightMass": replay["nonforcedTotalWeightMass"],
        "approxKl": replay["approxKl"],
        "clipFraction": replay["clipFraction"],
        "entropy": entropy,
        "initialBehaviorEntropy": initial_entropy,
        "entropyRetentionRatio": retention,
        "entropyCollapseFraction": collapse,
        "entropyCollapseExceeds30Percent": (
            collapse > V4_ENTROPY_COLLAPSE_FRACTION
        ),
        "meanLogRatio": replay["meanLogRatio"],
        "meanAbsoluteLogRatio": replay["meanAbsoluteLogRatio"],
        "maximumAbsoluteLogRatio": replay["maximumAbsoluteLogRatio"],
        "forcedMaximumAbsoluteLogRatio": replay[
            "forcedMaximumAbsoluteLogRatio"
        ],
        "perPlayerCount": replay["perPlayerCount"],
    }
    return _validate_post_epoch_policy_drift_audit(
        record,
        dataset,
        training_config,
        balance_contract,
        initial_audit,
        expected_epoch=epoch,
        expected_global_step=global_step,
        fixed_collection_plan_sha256=fixed_collection_plan_sha256,
        fixed_ppo_execution_contract_fingerprint=(
            fixed_ppo_execution_contract_fingerprint
        ),
        audit_contract_fingerprint=audit_contract_fingerprint,
        audit_batch_size=audit_batch_size,
        expected_actor_state_sha256=actor_state_sha256,
    )


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


def _checkpoint_sha256_sidecar_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}.sha256")


def _write_checkpoint_sha256_sidecar(checkpoint: Path, checksum: str) -> None:
    if not _is_lower_sha256(checksum):
        raise ValueError("checkpoint SHA-256 is non-canonical")
    _atomic_write(
        _checkpoint_sha256_sidecar_path(checkpoint),
        f"{checksum}  {checkpoint.name}\n".encode("ascii"),
    )


def _verify_checkpoint_sha256_sidecar(
    checkpoint: Path,
    *,
    required: bool,
) -> str | None:
    sidecar = _checkpoint_sha256_sidecar_path(checkpoint)
    if not sidecar.is_file():
        if required:
            raise ValueError("fixed checkpoint SHA-256 sidecar is missing")
        return None
    try:
        text = sidecar.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError("checkpoint SHA-256 sidecar is malformed") from error
    parts = text[:-1].split() if text.endswith("\n") else []
    if (
        text.count("\n") != 1
        or len(parts) != 2
        or not _is_lower_sha256(parts[0])
        or parts[1] != checkpoint.name
    ):
        raise ValueError("checkpoint SHA-256 sidecar is malformed")
    actual = sha256_file(checkpoint)
    if parts[0] != actual:
        raise ValueError("checkpoint SHA-256 sidecar checksum does not match")
    return str(parts[0])


def _validate_latest_checkpoint_record(
    value: object,
    *,
    fixed: bool,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("latest V4 checkpoint record must be an object")
    actual_fields = set(value)
    if actual_fields == V4_LATEST_COMMON_FIELDS:
        record_is_fixed = False
    elif actual_fields == V4_LATEST_COMMON_FIELDS | V4_LATEST_FIXED_FIELDS:
        record_is_fixed = True
    else:
        raise ValueError("latest V4 checkpoint record key set is non-canonical")
    if record_is_fixed != fixed:
        raise ValueError("latest V4 checkpoint record variant does not match")
    completed_epoch = value.get("completedEpoch")
    global_step = value.get("globalStep")
    if (
        value.get("format") != V4_TRAINING_CHECKPOINT_FORMAT
        or type(value.get("version")) is not int
        or value.get("version") != V4_TRAINING_CHECKPOINT_VERSION
        or type(completed_epoch) is not int
        or completed_epoch < 1
        or type(global_step) is not int
        or global_step < 1
        or not _is_lower_sha256(value.get("datasetFingerprint"))
        or not _is_lower_sha256(value.get("lossContractFingerprint"))
        or not _is_lower_sha256(value.get("sha256"))
    ):
        raise ValueError("latest V4 checkpoint record fields are non-canonical")
    canonical_checkpoint = f"checkpoints/epoch-{completed_epoch:04d}.pt"
    if (
        not isinstance(value.get("checkpoint"), str)
        or value["checkpoint"] != canonical_checkpoint
    ):
        raise ValueError("latest V4 checkpoint path is not canonical")
    if record_is_fixed:
        fixed_fingerprints = (
            "balanceContractFingerprint",
            "fixedCollectionPlanSha256",
            "fixedPpoExecutionContractFingerprint",
            "initialPolicyReproductionAuditFingerprint",
            "postEpochPolicyDriftAuditContractFingerprint",
            "postEpochPolicyDriftAuditFingerprint",
        )
        if any(not _is_lower_sha256(value.get(name)) for name in fixed_fingerprints):
            raise ValueError(
                "latest fixed V4 checkpoint fingerprints are non-canonical"
            )
        if not isinstance(value.get("initialPolicyReproductionAudit"), Mapping):
            raise ValueError(
                "latest initial policy reproduction audit must be an object"
            )
        if not isinstance(value.get("postEpochPolicyDriftAudit"), Mapping):
            raise ValueError(
                "latest post-epoch policy drift audit must be an object"
            )
        if (
            type(value.get("fixedCheckpointRngContractVersion")) is not int
            or value.get("fixedCheckpointRngContractVersion")
            != V4_FIXED_CHECKPOINT_RNG_CONTRACT_VERSION
        ):
            raise ValueError(
                "latest fixed checkpoint RNG contract version is non-canonical"
            )
        cuda_rng_state_count = value.get("cudaRngStateCount")
        if type(cuda_rng_state_count) is not int or cuda_rng_state_count < 0:
            raise ValueError(
                "latest fixed checkpoint CUDA RNG state count is non-canonical"
            )
        # This boundary can validate the JSON containers and their fingerprints,
        # but it has no dataset with which to reconstruct row counts or the
        # balance contract.  Full nested audit semantics remain fail-closed in
        # _resume_training before any state is installed.
    return dict(value)


def _path_is_link_or_reparse_alias(path: Path) -> bool:
    """Reject symlink, junction, and other Windows reparse aliases safely."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
        if os.name == "nt":
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError("cannot safely inspect V4 checkpoint path") from error
    return False


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
    balance_contract_fingerprint: str | None = None,
    fixed_collection_plan_sha256: str | None = None,
    fixed_ppo_execution_contract_fingerprint: str | None = None,
    initial_policy_reproduction_audit: Mapping[str, object] | None = None,
    post_epoch_policy_drift_audit_contract_fingerprint: str | None = None,
    post_epoch_policy_drift_audit: Mapping[str, object] | None = None,
) -> Path:
    if post_epoch_policy_drift_audit is not None:
        expected_actor_state_sha256 = post_epoch_policy_drift_audit.get(
            "actorStateSha256"
        )
        actual_actor_state_sha256 = _actor_state_sha256(actor.state_dict())
        if expected_actor_state_sha256 != actual_actor_state_sha256:
            raise ValueError(
                "post-epoch policy drift audit Actor state SHA-256 does not match"
            )
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
    if post_epoch_policy_drift_audit is not None:
        actor_device = next(actor.parameters()).device
        checkpoint["fixedCheckpointRngContractVersion"] = (
            V4_FIXED_CHECKPOINT_RNG_CONTRACT_VERSION
        )
        checkpoint["cudaRngStates"] = (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if actor_device.type == "cuda"
            else None
        )
    if balance_contract_fingerprint is not None:
        checkpoint["balanceContractFingerprint"] = balance_contract_fingerprint
    if fixed_collection_plan_sha256 is not None:
        checkpoint["fixedCollectionPlanSha256"] = fixed_collection_plan_sha256
    if fixed_ppo_execution_contract_fingerprint is not None:
        checkpoint["fixedPpoExecutionContractFingerprint"] = (
            fixed_ppo_execution_contract_fingerprint
        )
    if initial_policy_reproduction_audit is not None:
        audit = dict(initial_policy_reproduction_audit)
        checkpoint["initialPolicyReproductionAudit"] = audit
        checkpoint["initialPolicyReproductionAuditFingerprint"] = hashlib.sha256(
            canonical_json_bytes(audit)
        ).hexdigest()
    if post_epoch_policy_drift_audit is not None:
        if post_epoch_policy_drift_audit_contract_fingerprint is None:
            raise ValueError("post-epoch policy drift audit contract is missing")
        post_audit = dict(post_epoch_policy_drift_audit)
        checkpoint["postEpochPolicyDriftAuditContractFingerprint"] = (
            post_epoch_policy_drift_audit_contract_fingerprint
        )
        checkpoint["postEpochPolicyDriftAudit"] = post_audit
        checkpoint["postEpochPolicyDriftAuditFingerprint"] = hashlib.sha256(
            canonical_json_bytes(post_audit)
        ).hexdigest()
    elif post_epoch_policy_drift_audit_contract_fingerprint is not None:
        raise ValueError("post-epoch policy drift audit is missing")
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    path = output / "checkpoints" / f"epoch-{epoch:04d}.pt"
    _atomic_write(path, buffer.getvalue())
    checkpoint_sha256 = sha256_file(path)
    if post_epoch_policy_drift_audit is not None:
        _write_checkpoint_sha256_sidecar(path, checkpoint_sha256)
    latest = {
        "format": V4_TRAINING_CHECKPOINT_FORMAT,
        "version": V4_TRAINING_CHECKPOINT_VERSION,
        "completedEpoch": epoch,
        "globalStep": global_step,
        "datasetFingerprint": dataset.fingerprint,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "checkpoint": str(path.relative_to(output)).replace("\\", "/"),
        "sha256": checkpoint_sha256,
    }
    if balance_contract_fingerprint is not None:
        latest["balanceContractFingerprint"] = balance_contract_fingerprint
    if fixed_collection_plan_sha256 is not None:
        latest["fixedCollectionPlanSha256"] = fixed_collection_plan_sha256
    if fixed_ppo_execution_contract_fingerprint is not None:
        latest["fixedPpoExecutionContractFingerprint"] = (
            fixed_ppo_execution_contract_fingerprint
        )
    if initial_policy_reproduction_audit is not None:
        audit = dict(initial_policy_reproduction_audit)
        latest["initialPolicyReproductionAudit"] = audit
        latest["initialPolicyReproductionAuditFingerprint"] = hashlib.sha256(
            canonical_json_bytes(audit)
        ).hexdigest()
    if post_epoch_policy_drift_audit is not None:
        assert post_epoch_policy_drift_audit_contract_fingerprint is not None
        post_audit = dict(post_epoch_policy_drift_audit)
        latest["fixedCheckpointRngContractVersion"] = (
            V4_FIXED_CHECKPOINT_RNG_CONTRACT_VERSION
        )
        latest["cudaRngStateCount"] = (
            len(checkpoint["cudaRngStates"])
            if isinstance(checkpoint["cudaRngStates"], list)
            else 0
        )
        latest["postEpochPolicyDriftAuditContractFingerprint"] = (
            post_epoch_policy_drift_audit_contract_fingerprint
        )
        latest["postEpochPolicyDriftAudit"] = post_audit
        latest["postEpochPolicyDriftAuditFingerprint"] = hashlib.sha256(
            canonical_json_bytes(post_audit)
        ).hexdigest()
    _atomic_write(output / "latest.json", canonical_json_bytes(latest))
    return path


def _resolve_resume(
    output: Path,
    resume: str | Path | None,
    *,
    require_checkpoint_sidecar: bool = False,
) -> Path | None:
    if resume is None:
        return None
    if str(resume) != "latest":
        checkpoint = Path(resume).resolve()
        # Existing legacy explicit checkpoints predate sidecars and remain
        # loadable for compatibility. Every fixed checkpoint requires one.
        _verify_checkpoint_sha256_sidecar(
            checkpoint, required=require_checkpoint_sidecar
        )
        return checkpoint
    latest_path = output / "latest.json"
    latest = _validate_latest_checkpoint_record(
        json.loads(latest_path.read_text(encoding="utf-8")),
        fixed=require_checkpoint_sidecar,
    )
    output_root = output.resolve()
    checkpoint_directory_literal = output_root / "checkpoints"
    if _path_is_link_or_reparse_alias(checkpoint_directory_literal):
        raise ValueError("latest V4 checkpoint directory is a path alias")
    checkpoint_directory = checkpoint_directory_literal.resolve()
    if checkpoint_directory.parent != output_root:
        raise ValueError("latest V4 checkpoint directory escapes output")
    checkpoint_literal = output_root / str(latest["checkpoint"])
    expected_checkpoint_literal = (
        checkpoint_directory
        / f'epoch-{int(latest["completedEpoch"]):04d}.pt'
    )
    if checkpoint_literal != expected_checkpoint_literal:
        raise ValueError("latest V4 checkpoint literal path is not canonical")
    if _path_is_link_or_reparse_alias(checkpoint_literal):
        raise ValueError("latest V4 checkpoint file is a path alias")
    checkpoint = checkpoint_literal.resolve()
    expected_checkpoint = expected_checkpoint_literal.resolve()
    if checkpoint != expected_checkpoint or checkpoint.parent != checkpoint_directory:
        raise ValueError("latest V4 checkpoint path escapes output/checkpoints")
    actual = sha256_file(checkpoint)
    if latest.get("sha256") != actual:
        raise ValueError("latest V4 checkpoint checksum does not match")
    sidecar_checksum = _verify_checkpoint_sha256_sidecar(
        checkpoint, required=require_checkpoint_sidecar
    )
    if sidecar_checksum is not None and sidecar_checksum != actual:
        raise ValueError("latest checkpoint and sidecar SHA-256 disagree")
    return checkpoint.resolve()


def _validated_checkpoint_cuda_rng_states(
    value: object,
    device: torch.device,
) -> list[torch.Tensor] | None:
    if device.type != "cuda":
        if value is not None:
            raise ValueError("resume CPU fixed checkpoint carries CUDA RNG states")
        return None
    expected_count = torch.cuda.device_count()
    if (
        expected_count < 1
        or not isinstance(value, list)
        or len(value) != expected_count
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() < 1
            for state in value
        )
    ):
        raise ValueError("resume CUDA RNG states are non-canonical")
    return [state.detach().cpu().clone() for state in value]


def _resume_training_impl(
    checkpoint_path: Path,
    actor: V4PublicActor,
    critic: V4PrivilegedQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scaler: object,
    dataset: V4TrajectoryDataset,
    training_config: V4TrainingConfig,
    device: torch.device,
    balance_contract_fingerprint: str | None = None,
    fixed_collection_plan_sha256: str | None = None,
    fixed_ppo_execution_contract_fingerprint: str | None = None,
    balance_contract: Mapping[str, object] | None = None,
    post_epoch_policy_drift_audit_contract_fingerprint: str | None = None,
    policy_audit_batch_size: int | None = None,
) -> tuple[
    int,
    int,
    dict[str, object] | None,
    dict[str, object] | None,
]:
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
    if checkpoint.get("balanceContractFingerprint") != balance_contract_fingerprint:
        raise ValueError("resume player-count balance contract does not match")
    if checkpoint.get("fixedCollectionPlanSha256") != fixed_collection_plan_sha256:
        raise ValueError("resume fixed collection plan does not match")
    if (
        checkpoint.get("fixedPpoExecutionContractFingerprint")
        != fixed_ppo_execution_contract_fingerprint
    ):
        raise ValueError("resume fixed PPO execution contract does not match")
    completed_epoch = checkpoint.get("completedEpoch")
    global_step = checkpoint.get("globalStep")
    if (
        isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or completed_epoch < 1
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 1
    ):
        raise ValueError("resume checkpoint epoch/global step is invalid")
    raw_audit = checkpoint.get("initialPolicyReproductionAudit")
    cuda_rng_states_to_restore: list[torch.Tensor] | None = None
    if fixed_collection_plan_sha256 is None:
        # Legacy BC/DAGGER/PPO checkpoints intentionally retain their former
        # CUDA-resume behavior and file shape. The new exact all-device RNG
        # contract is mandatory only for fixed-match checkpoints.
        if raw_audit is not None or checkpoint.get(
            "initialPolicyReproductionAuditFingerprint"
        ) is not None:
            raise ValueError(
                "legacy resume checkpoint unexpectedly carries a fixed PPO audit"
            )
        audit = None
    else:
        if (
            checkpoint.get("fixedCheckpointRngContractVersion")
            != V4_FIXED_CHECKPOINT_RNG_CONTRACT_VERSION
            or "cudaRngStates" not in checkpoint
        ):
            raise ValueError("resume fixed checkpoint RNG contract is missing")
        cuda_rng_states_to_restore = _validated_checkpoint_cuda_rng_states(
            checkpoint["cudaRngStates"], device
        )
        audit = _validate_initial_policy_reproduction_audit(raw_audit, dataset)
        if audit.get("auditBatchSize") != policy_audit_batch_size:
            raise ValueError(
                "resume initial policy audit batch selection does not match"
            )
        audit_fingerprint = hashlib.sha256(
            canonical_json_bytes(audit)
        ).hexdigest()
        if checkpoint.get(
            "initialPolicyReproductionAuditFingerprint"
        ) != audit_fingerprint:
            raise ValueError(
                "resume initial policy reproduction audit fingerprint does not match"
            )
    raw_post_audit = checkpoint.get("postEpochPolicyDriftAudit")
    raw_post_fingerprint = checkpoint.get("postEpochPolicyDriftAuditFingerprint")
    raw_post_contract_fingerprint = checkpoint.get(
        "postEpochPolicyDriftAuditContractFingerprint"
    )
    if fixed_collection_plan_sha256 is None:
        if any(
            value is not None
            for value in (
                raw_post_audit,
                raw_post_fingerprint,
                raw_post_contract_fingerprint,
            )
        ):
            raise ValueError(
                "legacy resume checkpoint unexpectedly carries a post-epoch audit"
            )
        post_audit = None
    else:
        if (
            balance_contract is None
            or fixed_ppo_execution_contract_fingerprint is None
            or post_epoch_policy_drift_audit_contract_fingerprint is None
            or policy_audit_batch_size is None
        ):
            raise ValueError("resume fixed PPO post-epoch audit contract is missing")
        if (
            raw_post_contract_fingerprint
            != post_epoch_policy_drift_audit_contract_fingerprint
        ):
            raise ValueError(
                "resume post-epoch policy drift audit contract does not match"
            )
        assert audit is not None
        checkpoint_actor_state = checkpoint.get("actorState")
        if not isinstance(checkpoint_actor_state, Mapping):
            raise ValueError("resume checkpoint Actor state is missing")
        checkpoint_actor_state_sha256 = _actor_state_sha256(
            checkpoint_actor_state
        )
        if (
            not isinstance(raw_post_audit, Mapping)
            or raw_post_audit.get("actorStateSha256")
            != checkpoint_actor_state_sha256
        ):
            raise ValueError(
                "resume checkpoint Actor state SHA-256 does not match post-epoch audit"
            )
        post_audit = _validate_post_epoch_policy_drift_audit(
            raw_post_audit,
            dataset,
            training_config,
            balance_contract,
            audit,
            expected_epoch=completed_epoch,
            expected_global_step=global_step,
            fixed_collection_plan_sha256=fixed_collection_plan_sha256,
            fixed_ppo_execution_contract_fingerprint=(
                fixed_ppo_execution_contract_fingerprint
            ),
            audit_contract_fingerprint=(
                post_epoch_policy_drift_audit_contract_fingerprint
            ),
            audit_batch_size=policy_audit_batch_size,
            expected_actor_state_sha256=checkpoint_actor_state_sha256,
        )
        post_fingerprint = hashlib.sha256(
            canonical_json_bytes(post_audit)
        ).hexdigest()
        if raw_post_fingerprint != post_fingerprint:
            raise ValueError(
                "resume post-epoch policy drift audit fingerprint does not match"
            )
    if checkpoint.get("actorConfig") != actor.config.to_dict():
        raise ValueError("resume actor configuration does not match")
    if checkpoint.get("criticConfig") != critic.config.to_dict():
        raise ValueError("resume critic configuration does not match")
    old_training = checkpoint.get("trainingConfig")
    if not isinstance(old_training, dict):
        raise ValueError("resume training configuration is missing")
    current_training = training_config.to_dict()
    if set(old_training) != set(current_training):
        raise ValueError("resume training configuration key set changed")
    for name, value in current_training.items():
        if name == "epochs":
            if isinstance(old_training[name], bool) or not isinstance(
                old_training[name], int
            ):
                raise ValueError("resume training epochs type changed")
            continue
        if type(old_training[name]) is not type(value) or old_training[name] != value:
            raise ValueError(f"resume training setting changed: {name}")
    actor.load_state_dict(checkpoint["actorState"], strict=True)
    critic.load_state_dict(checkpoint["criticState"], strict=True)
    actor_optimizer.load_state_dict(checkpoint["actorOptimizerState"])
    critic_optimizer.load_state_dict(checkpoint["criticOptimizerState"])
    scaler.load_state_dict(checkpoint["scalerState"])
    torch.set_rng_state(checkpoint["torchRngState"].cpu())
    if cuda_rng_states_to_restore is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states_to_restore)
    np.random.set_state(checkpoint["numpyRngState"])
    random.setstate(checkpoint["pythonRngState"])
    if completed_epoch >= training_config.epochs:
        raise ValueError("resume checkpoint already reached the requested epochs")
    return completed_epoch, global_step, audit, post_audit


def _capture_global_rng_state() -> dict[str, object]:
    raw_numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "numpy": (
            raw_numpy_state[0],
            raw_numpy_state[1].copy(),
            *raw_numpy_state[2:],
        ),
        "python": random.getstate(),
    }


def _restore_global_rng_state(snapshot: Mapping[str, object]) -> None:
    torch_state = snapshot["torch"]
    numpy_state = snapshot["numpy"]
    if not isinstance(torch_state, torch.Tensor) or not isinstance(
        numpy_state, tuple
    ):
        raise ValueError("global RNG snapshot is non-canonical")
    torch.set_rng_state(torch_state)
    cuda_states = snapshot["cuda"]
    if cuda_states is not None:
        if not isinstance(cuda_states, list):
            raise ValueError("global CUDA RNG snapshot is non-canonical")
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(numpy_state)
    random.setstate(snapshot["python"])


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
    balance_contract_fingerprint: str | None = None,
    fixed_collection_plan_sha256: str | None = None,
    fixed_ppo_execution_contract_fingerprint: str | None = None,
    balance_contract: Mapping[str, object] | None = None,
    post_epoch_policy_drift_audit_contract_fingerprint: str | None = None,
    policy_audit_batch_size: int | None = None,
) -> tuple[
    int,
    int,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Load one checkpoint transactionally with respect to every global RNG."""

    rng_snapshot = _capture_global_rng_state()
    try:
        return _resume_training_impl(
            checkpoint_path,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            scaler,
            dataset,
            training_config,
            device,
            balance_contract_fingerprint,
            fixed_collection_plan_sha256,
            fixed_ppo_execution_contract_fingerprint,
            balance_contract,
            post_epoch_policy_drift_audit_contract_fingerprint,
            policy_audit_batch_size,
        )
    except BaseException:
        _restore_global_rng_state(rng_snapshot)
        raise


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
    effective_actor_masks = {
        "behaviorCloning": nonforced_policy_eligibility(
            dataset.tensors.legal_masks,
            eligibility.behavior_cloning,
        ),
        "ppo": nonforced_policy_eligibility(
            dataset.tensors.legal_masks,
            eligibility.ppo,
        ),
    }
    effective_actor_counts = {
        name: int(mask.sum()) for name, mask in effective_actor_masks.items()
    }
    forced_actor_counts = {
        name: counts[name] - effective_actor_counts[name]
        for name in effective_actor_counts
    }
    requested = {
        "behaviorCloning": training_config.bc_weight,
        "ppo": training_config.ppo_weight,
        "critic": training_config.critic_weight,
    }
    fixed_source = V4_FIXED_PPO_SOURCE_CONTRACT
    expected_qboost_zero = fixed_source in eligibility.ppo_source_contracts
    if eligibility.requires_qboost_coefficient_zero != expected_qboost_zero:
        raise ValueError(
            "fixed-match PPO source and q-boost prohibition binding disagree"
        )
    requires_qboost_zero = eligibility.requires_qboost_coefficient_zero
    if requires_qboost_zero and training_config.q_boost_coefficient != 0.0:
        raise ValueError(
            "evaluation-aligned fixed-match PPO data requires q_boost_coefficient=0"
        )
    if (
        fixed_source in eligibility.ppo_source_contracts
        and eligibility.ppo_source_contracts != (fixed_source,)
    ):
        raise ValueError(
            "mixed fixed-match and legacy PPO source contracts cannot be trained together"
        )
    if eligibility.requires_player_count_balanced_loss != (
        eligibility.ppo_source_contracts == (fixed_source,)
    ):
        raise ValueError(
            "fixed-only PPO source provenance and balanced-loss requirement disagree"
        )
    fixed_collection_plan_sha256 = _resolve_fixed_collection_plan_sha256(
        dataset,
        training_config,
    )
    balance_contract = _player_count_balance_contract(
        dataset,
        fixed_collection_plan_sha256,
    )
    if balance_contract is not None and resume is None:
        actor_hashes = eligibility.behavior_actor_sha256s
        if len(actor_hashes) != 1:
            raise ValueError(
                "fixed-only PPO training requires exactly one bound behavior Actor"
            )
        if initial_actor_sha256 is None:
            raise ValueError(
                "fresh fixed-only PPO training requires --initialize-actor-bundle "
                "for the full-dataset initial policy reproduction audit"
            )
        if initial_actor_sha256 != actor_hashes[0]:
            raise ValueError(
                "fixed-only PPO initialization Actor does not match the collector "
                "behavior Actor"
            )
    for name, weight in requested.items():
        admitted_count = (
            effective_actor_counts[name]
            if name in effective_actor_counts
            else counts[name]
        )
        if weight > 0.0 and admitted_count == 0:
            raise ValueError(
                f"{name} loss was requested but the dataset has no eligible samples "
                "after forced singleton-action rows are excluded"
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
    contract: dict[str, object] = {
        "version": 1,
        "preparationFormat": eligibility.preparation_format,
        "preparationVersion": eligibility.preparation_version,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "masks": dict(V4_LOSS_MASK_NAMES),
        "eligibleSampleCounts": counts,
        "effectiveNonforcedActorSampleCounts": effective_actor_counts,
        "forcedActorSamplesExcluded": forced_actor_counts,
        "actorPolicyMask": "loss eligibility AND legal-action count greater than one",
        "advantageNormalization": (
            "collector-bound advantages; no minibatch recentering or rescaling"
        ),
        "requestedWeights": requested,
        "ppoBehaviorActorSha256s": list(eligibility.behavior_actor_sha256s),
        "ppoSourceContracts": list(eligibility.ppo_source_contracts),
        "requiresPlayerCountBalancedLoss": eligibility.requires_player_count_balanced_loss,
        "normalAndDaggerAreBcOnly": True,
        "ppoAndCriticAdmitOnlyPpoCollectorSamples": True,
        "fixedMatchPpoRequiresQBoostCoefficientZero": requires_qboost_zero,
    }
    if balance_contract is not None:
        contract["version"] = 3
        contract["playerCountBalancedLoss"] = balance_contract
        contract["balanceContractFingerprint"] = balance_contract[
            "balanceContractFingerprint"
        ]
        contract["ppoRewardContracts"] = list(
            eligibility.ppo_reward_contracts
        )
        contract["ppoBehaviorPolicyContracts"] = list(
            eligibility.ppo_behavior_policy_contracts
        )
        contract["fixedCollectionPlanIds"] = list(
            eligibility.fixed_collection_plan_ids
        )
        contract["fixedCollectionPlanSha256"] = fixed_collection_plan_sha256
        contract["fixedPpoActorForwardDtype"] = "torch.float32"
        contract["fixedPpoActorAutocastDisabled"] = True
        contract["initialPolicyReproductionAbsoluteTolerance"] = (
            V4_FIXED_INITIAL_LOG_PROBABILITY_ABSOLUTE_TOLERANCE
        )
        contract["requiresFullDatasetInitialPolicyReproductionAudit"] = True
    return contract


def _train_v4_impl(
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
    raw_balance_contract = training_contract.get("playerCountBalancedLoss")
    balance_contract = (
        raw_balance_contract
        if isinstance(raw_balance_contract, Mapping)
        else None
    )
    balance_contract_fingerprint = (
        str(training_contract["balanceContractFingerprint"])
        if balance_contract is not None
        else None
    )
    fixed_collection_plan_sha256 = (
        str(training_contract["fixedCollectionPlanSha256"])
        if balance_contract is not None
        else None
    )
    device_value = torch.device(device)
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    fixed_ppo_execution_contract = (
        _configure_fixed_ppo_execution(device_value)
        if balance_contract is not None
        else None
    )
    fixed_ppo_execution_contract_fingerprint = (
        str(fixed_ppo_execution_contract["executionContractFingerprint"])
        if fixed_ppo_execution_contract is not None
        else None
    )
    policy_audit_batch_size = (
        _policy_audit_batch_size(training_config, device_value)
        if balance_contract is not None
        else None
    )
    post_epoch_policy_drift_audit_contract = (
        _post_epoch_policy_drift_audit_contract(training_config, device_value)
        if balance_contract is not None
        else None
    )
    post_epoch_policy_drift_audit_contract_fingerprint = (
        str(
            post_epoch_policy_drift_audit_contract[
                "auditContractFingerprint"
            ]
        )
        if post_epoch_policy_drift_audit_contract is not None
        else None
    )
    use_amp = bool(training_config.amp and device_value.type == "cuda")
    random.seed(training_config.seed)
    np.random.seed(training_config.seed % (2**32))
    torch.manual_seed(training_config.seed)
    if device_value.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)

    balance_weight_lookups: dict[str, torch.Tensor] = {}
    if balance_contract is not None:
        raw_weights = balance_contract["runtimeFloat32WeightsByLossAndPlayerCount"]
        assert isinstance(raw_weights, Mapping)
        for loss_name in ("behaviorCloning", "ppo", "critic"):
            values = torch.zeros(11, dtype=torch.float32, device=device_value)
            loss_weights = raw_weights[loss_name]
            assert isinstance(loss_weights, Mapping)
            for player_count in V4_BALANCED_PLAYER_COUNTS:
                values[player_count] = float(loss_weights[str(player_count)])
            balance_weight_lookups[loss_name] = values

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
    initial_policy_reproduction_audit: dict[str, object] | None = None
    latest_post_epoch_policy_drift_audit: dict[str, object] | None = None
    checkpoint_path = _resolve_resume(
        output,
        resume,
        require_checkpoint_sidecar=balance_contract is not None,
    )
    if checkpoint_path is not None:
        (
            start_epoch,
            global_step,
            initial_policy_reproduction_audit,
            latest_post_epoch_policy_drift_audit,
        ) = _resume_training(
            checkpoint_path,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            scaler,
            dataset,
            training_config,
            device_value,
            balance_contract_fingerprint,
            fixed_collection_plan_sha256,
            fixed_ppo_execution_contract_fingerprint,
            balance_contract,
            post_epoch_policy_drift_audit_contract_fingerprint,
            policy_audit_batch_size,
        )
        if str(resume) == "latest":
            latest = json.loads(
                (output / "latest.json").read_text(encoding="utf-8")
            )
            if latest.get("fixedPpoExecutionContractFingerprint") != (
                fixed_ppo_execution_contract_fingerprint
            ):
                raise ValueError(
                    "latest fixed PPO execution contract does not match"
                )
            if latest.get("initialPolicyReproductionAudit") != (
                initial_policy_reproduction_audit
            ):
                raise ValueError(
                    "latest initial policy reproduction audit does not match"
                )
            audit_fingerprint = (
                hashlib.sha256(
                    canonical_json_bytes(initial_policy_reproduction_audit)
                ).hexdigest()
                if initial_policy_reproduction_audit is not None
                else None
            )
            if latest.get(
                "initialPolicyReproductionAuditFingerprint"
            ) != audit_fingerprint:
                raise ValueError(
                    "latest initial policy reproduction audit fingerprint does not match"
                )
            if balance_contract is not None:
                assert latest_post_epoch_policy_drift_audit is not None
                expected_cuda_rng_state_count = (
                    torch.cuda.device_count()
                    if device_value.type == "cuda"
                    else 0
                )
                if (
                    latest.get("fixedCheckpointRngContractVersion")
                    != V4_FIXED_CHECKPOINT_RNG_CONTRACT_VERSION
                    or latest.get("cudaRngStateCount")
                    != expected_cuda_rng_state_count
                ):
                    raise ValueError(
                        "latest fixed checkpoint RNG contract does not match"
                    )
                if latest.get(
                    "postEpochPolicyDriftAuditContractFingerprint"
                ) != post_epoch_policy_drift_audit_contract_fingerprint:
                    raise ValueError(
                        "latest post-epoch policy drift audit contract does not match"
                    )
                if latest.get("postEpochPolicyDriftAudit") != (
                    latest_post_epoch_policy_drift_audit
                ):
                    raise ValueError(
                        "latest post-epoch policy drift audit does not match"
                    )
                post_audit_fingerprint = hashlib.sha256(
                    canonical_json_bytes(latest_post_epoch_policy_drift_audit)
                ).hexdigest()
                if latest.get("postEpochPolicyDriftAuditFingerprint") != (
                    post_audit_fingerprint
                ):
                    raise ValueError(
                        "latest post-epoch policy drift audit fingerprint does not match"
                    )
    elif (output / "latest.json").exists():
        raise FileExistsError(
            "the output already contains a run; pass resume='latest' or use a new directory"
        )
    elif balance_contract is not None:
        assert policy_audit_batch_size is not None
        initial_policy_reproduction_audit = (
            _audit_initial_policy_reproduction(
                actor,
                dataset,
                device=device_value,
                batch_size=policy_audit_batch_size,
                num_workers=training_config.num_workers,
                balance_contract=balance_contract,
                clip_ratio=training_config.clip_ratio,
            )
        )

    if fixed_ppo_execution_contract is not None:
        assert initial_policy_reproduction_audit is not None
        audit_fingerprint = hashlib.sha256(
            canonical_json_bytes(initial_policy_reproduction_audit)
        ).hexdigest()
        training_contract["fixedPpoExecutionContract"] = (
            fixed_ppo_execution_contract
        )
        training_contract["fixedPpoExecutionContractFingerprint"] = (
            fixed_ppo_execution_contract_fingerprint
        )
        training_contract["initialPolicyReproductionAudit"] = (
            initial_policy_reproduction_audit
        )
        training_contract["initialPolicyReproductionAuditFingerprint"] = (
            audit_fingerprint
        )
        assert post_epoch_policy_drift_audit_contract is not None
        training_contract["postEpochPolicyDriftAuditContract"] = (
            post_epoch_policy_drift_audit_contract
        )
        training_contract[
            "postEpochPolicyDriftAuditContractFingerprint"
        ] = post_epoch_policy_drift_audit_contract_fingerprint

    # Fixed fresh runs reach this point only after plan/dropout/FP32 policy
    # reproduction admission has passed, so a failed preflight leaves no output.
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run-manifest.json"
    if checkpoint_path is not None and manifest_path.exists():
        existing_initial_actor = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("initialActor")
        if existing_initial_actor is not None and not isinstance(
            existing_initial_actor, Mapping
        ):
            raise ValueError("existing V4 run manifest has an invalid initial Actor")
        initial_actor = (
            dict(existing_initial_actor)
            if isinstance(existing_initial_actor, Mapping)
            else None
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
        effective_actor_samples = {
            "behaviorCloning": 0,
            "ppo": 0,
        }
        forced_actor_samples_excluded = {
            "behaviorCloning": 0,
            "ppo": 0,
        }
        balanced_metric_numerators = {
            "policyLoss": 0.0,
            "behaviorCloningLoss": 0.0,
            "criticLoss": 0.0,
            "entropy": 0.0,
            "approxKl": 0.0,
            "clipFraction": 0.0,
            "meanQBoost": 0.0,
        }
        balanced_metric_weight_sums = {
            "behaviorCloning": 0.0,
            "ppo": 0.0,
            "critic": 0.0,
        }
        balanced_seen_counts = {
            loss_name: {
                str(player_count): 0
                for player_count in V4_BALANCED_PLAYER_COUNTS
            }
            for loss_name in ("behaviorCloning", "ppo", "critic")
        }
        batches = 0
        optimizer_steps = 0
        for batch_index, cpu_batch in enumerate(loader):
            batch = _batch_to_device(
                _trim_public_padding(cpu_batch), device_value
            )
            valid_flat = batch["valid_masks"].reshape(-1)
            bc_eligible_flat = (
                batch[V4_LOSS_MASK_NAMES["behaviorCloning"]].reshape(-1)
                & valid_flat
            )
            ppo_eligible_flat = (
                batch[V4_LOSS_MASK_NAMES["ppo"]].reshape(-1) & valid_flat
            )
            critic_flat = (
                batch[V4_LOSS_MASK_NAMES["critic"]].reshape(-1) & valid_flat
            )
            bc_flat = nonforced_policy_eligibility(
                batch["legal_masks"].reshape(-1, V4_ACTION_COUNT),
                bc_eligible_flat,
            )
            ppo_flat = nonforced_policy_eligibility(
                batch["legal_masks"].reshape(-1, V4_ACTION_COUNT),
                ppo_eligible_flat,
            )
            eligible_samples["behaviorCloning"] += int(bc_eligible_flat.sum())
            eligible_samples["ppo"] += int(ppo_eligible_flat.sum())
            eligible_samples["critic"] += int(critic_flat.sum())
            effective_actor_samples["behaviorCloning"] += int(bc_flat.sum())
            effective_actor_samples["ppo"] += int(ppo_flat.sum())
            forced_actor_samples_excluded["behaviorCloning"] += int(
                bc_eligible_flat.sum() - bc_flat.sum()
            )
            forced_actor_samples_excluded["ppo"] += int(
                ppo_eligible_flat.sum() - ppo_flat.sum()
            )
            batch_balance_weights: dict[str, torch.Tensor | None] = {
                "behaviorCloning": None,
                "ppo": None,
                "critic": None,
            }
            if balance_contract is not None:
                player_counts_flat = batch["player_mask"].sum(dim=-1).reshape(-1)
                loss_masks = {
                    "behaviorCloning": bc_flat,
                    "ppo": ppo_flat,
                    "critic": critic_flat,
                }
                for loss_name, loss_mask in loss_masks.items():
                    selected_player_counts = player_counts_flat[loss_mask].to(torch.long)
                    weights = balance_weight_lookups[loss_name][
                        selected_player_counts
                    ]
                    if weights.numel() > 0 and (weights <= 0.0).any():
                        raise RuntimeError(
                            f"{loss_name} encountered an unbound player-count weight"
                        )
                    batch_balance_weights[loss_name] = weights
                    for player_count in V4_BALANCED_PLAYER_COUNTS:
                        balanced_seen_counts[loss_name][str(player_count)] += int(
                            (selected_player_counts == player_count).sum().item()
                        )
            legal_flat = _flatten_time(batch["legal_masks"]).clone()
            legal_flat[~valid_flat, 0] = True
            # The fixed collector records FP32 Actor logits.  Keep the Actor
            # outside AMP for every fixed PPO optimization forward; the
            # privileged critic may still use AMP independently.
            with torch.cuda.amp.autocast(
                enabled=use_amp and balance_contract is None
            ):
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
            with torch.cuda.amp.autocast(enabled=use_amp):
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
                ppo_weights = batch_balance_weights["ppo"]
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
                    normalize_advantages=False,
                    weights=ppo_weights,
                )
                policy_loss = policy_result.loss
                entropy = policy_result.entropy
                approx_kl = policy_result.approx_kl
                clip_fraction = policy_result.clip_fraction
                mean_q_boost = policy_result.mean_q_boost
                policy_loss_metric = policy_result.policy_loss
                if balance_contract is not None:
                    assert ppo_weights is not None
                    ppo_weight_sum = float(ppo_weights.sum().detach().cpu())
                    ppo_multiplier = _balanced_batch_estimator_multiplier(
                        trajectory_count=int(balance_contract["trajectoryCount"]),
                        batch_trajectory_count=batch_size,
                        total_eligible_rows=int(
                            balance_contract["totalEligibleRowsByLoss"]["ppo"]
                        ),
                    )
                    policy_loss = policy_loss * (ppo_weights.sum() * ppo_multiplier)
                    balanced_metric_weight_sums["ppo"] += ppo_weight_sum
                    for name, value in (
                        ("policyLoss", policy_result.policy_loss),
                        ("entropy", policy_result.entropy),
                        ("approxKl", policy_result.approx_kl),
                        ("clipFraction", policy_result.clip_fraction),
                        ("meanQBoost", policy_result.mean_q_boost),
                    ):
                        balanced_metric_numerators[name] += (
                            float(value.detach().cpu()) * ppo_weight_sum
                        )
            else:
                policy_loss_metric = actor_zero.detach()
            if training_config.bc_weight > 0.0 and bool(bc_flat.any()):
                bc_weights = batch_balance_weights["behaviorCloning"]
                bc_metric_loss = masked_behavior_cloning_loss(
                    logits_flat.float()[bc_flat],
                    legal_flat[bc_flat],
                    batch["expert_actions"].reshape(-1)[bc_flat],
                    weights=bc_weights,
                )
                bc_loss = bc_metric_loss
                if balance_contract is not None:
                    assert bc_weights is not None
                    bc_weight_sum = float(bc_weights.sum().detach().cpu())
                    bc_multiplier = _balanced_batch_estimator_multiplier(
                        trajectory_count=int(balance_contract["trajectoryCount"]),
                        batch_trajectory_count=batch_size,
                        total_eligible_rows=int(
                            balance_contract["totalEligibleRowsByLoss"][
                                "behaviorCloning"
                            ]
                        ),
                    )
                    bc_loss = bc_loss * (bc_weights.sum() * bc_multiplier)
                    balanced_metric_weight_sums[
                        "behaviorCloning"
                    ] += bc_weight_sum
                    balanced_metric_numerators[
                        "behaviorCloningLoss"
                    ] += float(bc_metric_loss.detach().cpu()) * bc_weight_sum
            else:
                bc_loss = actor_zero
                bc_metric_loss = actor_zero.detach()
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
                critic_weights = batch_balance_weights["critic"]
                critic_metric_loss = action_q_regression_loss(
                    q_flat.float()[critic_flat],
                    legal_flat[critic_flat],
                    batch["actions"].reshape(-1)[critic_flat],
                    targets_time.transpose(0, 1).reshape(-1)[critic_flat],
                    weights=critic_weights,
                )
                critic_loss = critic_metric_loss
                if balance_contract is not None:
                    assert critic_weights is not None
                    critic_weight_sum = float(critic_weights.sum().detach().cpu())
                    critic_multiplier = _balanced_batch_estimator_multiplier(
                        trajectory_count=int(balance_contract["trajectoryCount"]),
                        batch_trajectory_count=batch_size,
                        total_eligible_rows=int(
                            balance_contract["totalEligibleRowsByLoss"]["critic"]
                        ),
                    )
                    critic_loss = critic_loss * (
                        critic_weights.sum() * critic_multiplier
                    )
                    balanced_metric_weight_sums["critic"] += critic_weight_sum
                    balanced_metric_numerators[
                        "criticLoss"
                    ] += float(critic_metric_loss.detach().cpu()) * critic_weight_sum
            else:
                critic_loss = critic_zero
                critic_metric_loss = critic_zero.detach()
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
                "behaviorCloningLoss": bc_metric_loss,
                "criticLoss": critic_metric_loss,
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
            "effectiveNonforcedActorSamplesSeen": effective_actor_samples,
            "forcedActorSamplesExcluded": forced_actor_samples_excluded,
        }
        if balance_contract is None:
            epoch_metrics.update(
                {
                    name: value / max(1, batches)
                    for name, value in totals.items()
                }
            )
        else:
            expected_counts = balance_contract[
                "eligibleRowCountsByLossAndPlayerCount"
            ]
            if balanced_seen_counts != expected_counts:
                raise RuntimeError(
                    "epoch player-count eligible rows do not match the balance contract"
                )
            runtime_weights = balance_contract[
                "runtimeFloat32WeightsByLossAndPlayerCount"
            ]
            observed_masses = {
                loss_name: {
                    player_count: float(
                        count
                        * runtime_weights[loss_name][player_count]
                    )
                    for player_count, count in counts.items()
                }
                for loss_name, counts in balanced_seen_counts.items()
            }
            if observed_masses != balance_contract[
                "runtimeWeightMassByLossAndPlayerCount"
            ]:
                raise RuntimeError(
                    "epoch player-count weight mass does not match the balance contract"
                )
            requested_metric_loss = {
                "behaviorCloning": training_config.bc_weight,
                "ppo": training_config.ppo_weight,
                "critic": training_config.critic_weight,
            }
            for loss_name, requested_weight in requested_metric_loss.items():
                if requested_weight <= 0.0:
                    continue
                expected_mass = float(
                    balance_contract["runtimeTotalWeightMassByLoss"][loss_name]
                )
                if not math.isclose(
                    balanced_metric_weight_sums[loss_name],
                    expected_mass,
                    rel_tol=2.0e-6,
                    abs_tol=2.0e-5,
                ):
                    raise RuntimeError(
                        f"epoch {loss_name} diagnostic weight mass drifted from contract"
                    )

            def balanced_metric(name: str, loss_name: str) -> float:
                denominator = balanced_metric_weight_sums[loss_name]
                return (
                    balanced_metric_numerators[name] / denominator
                    if denominator > 0.0
                    else 0.0
                )

            policy_metric = balanced_metric("policyLoss", "ppo")
            entropy_metric = balanced_metric("entropy", "ppo")
            bc_metric = balanced_metric(
                "behaviorCloningLoss", "behaviorCloning"
            )
            critic_metric = balanced_metric("criticLoss", "critic")
            balanced_values = {
                "policyLoss": policy_metric,
                "behaviorCloningLoss": bc_metric,
                "criticLoss": critic_metric,
                "entropy": entropy_metric,
                "approxKl": balanced_metric("approxKl", "ppo"),
                "clipFraction": balanced_metric("clipFraction", "ppo"),
                "meanQBoost": balanced_metric("meanQBoost", "ppo"),
            }
            balanced_values["loss"] = (
                training_config.ppo_weight
                * (
                    policy_metric
                    - training_config.entropy_coefficient * entropy_metric
                )
                + training_config.bc_weight * bc_metric
                + training_config.critic_weight * critic_metric
            )
            optimization_pass_diagnostics = {
                "timing": (
                    "each shuffled minibatch before its optimizer update; "
                    "not final Actor drift"
                ),
                "entropy": balanced_values["entropy"],
                "approxKl": balanced_values["approxKl"],
                "clipFraction": balanced_values["clipFraction"],
                "policyLoss": balanced_values["policyLoss"],
                "meanQBoost": balanced_values["meanQBoost"],
            }
            epoch_metrics.update(
                {
                    "balanceContractFingerprint": balance_contract_fingerprint,
                    "fixedCollectionPlanSha256": fixed_collection_plan_sha256,
                    "fixedPpoExecutionContractFingerprint": (
                        fixed_ppo_execution_contract_fingerprint
                    ),
                    "initialPolicyReproductionAuditFingerprint": (
                        training_contract[
                            "initialPolicyReproductionAuditFingerprint"
                        ]
                    ),
                    "balancedEligibleRowsSeenByLossAndPlayerCount": (
                        balanced_seen_counts
                    ),
                    "balancedWeightMassSeenByLossAndPlayerCount": observed_masses,
                    "balancedDiagnosticWeightMassByLoss": (
                        balanced_metric_weight_sums
                    ),
                    "optimizationPassDiagnostics": (
                        optimization_pass_diagnostics
                    ),
                    **balanced_values,
                }
            )
            assert initial_policy_reproduction_audit is not None
            assert fixed_collection_plan_sha256 is not None
            assert fixed_ppo_execution_contract_fingerprint is not None
            assert post_epoch_policy_drift_audit_contract_fingerprint is not None
            assert policy_audit_batch_size is not None
            latest_post_epoch_policy_drift_audit = (
                _audit_post_epoch_policy_drift(
                    actor,
                    dataset,
                    training_config,
                    balance_contract,
                    initial_policy_reproduction_audit,
                    epoch=epoch,
                    global_step=global_step,
                    device=device_value,
                    fixed_collection_plan_sha256=fixed_collection_plan_sha256,
                    fixed_ppo_execution_contract_fingerprint=(
                        fixed_ppo_execution_contract_fingerprint
                    ),
                    audit_contract_fingerprint=(
                        post_epoch_policy_drift_audit_contract_fingerprint
                    ),
                    audit_batch_size=policy_audit_batch_size,
                )
            )
            post_audit_fingerprint = hashlib.sha256(
                canonical_json_bytes(latest_post_epoch_policy_drift_audit)
            ).hexdigest()
            epoch_metrics.update(
                {
                    "postEpochPolicyDriftAuditContractFingerprint": (
                        post_epoch_policy_drift_audit_contract_fingerprint
                    ),
                    "postEpochPolicyDriftAudit": (
                        latest_post_epoch_policy_drift_audit
                    ),
                    "postEpochPolicyDriftAuditFingerprint": (
                        post_audit_fingerprint
                    ),
                    # Operational KL/clip/entropy gates must read the final
                    # Actor, never the moving pre-update training pass.
                    "approxKl": latest_post_epoch_policy_drift_audit[
                        "approxKl"
                    ],
                    "clipFraction": latest_post_epoch_policy_drift_audit[
                        "clipFraction"
                    ],
                    "entropy": latest_post_epoch_policy_drift_audit["entropy"],
                }
            )
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
                balance_contract_fingerprint,
                fixed_collection_plan_sha256,
                fixed_ppo_execution_contract_fingerprint,
                initial_policy_reproduction_audit,
                post_epoch_policy_drift_audit_contract_fingerprint,
                latest_post_epoch_policy_drift_audit,
            )

    actor.eval()
    candidate_metadata: dict[str, object] = {
        "seed": training_config.seed,
        "datasetFingerprint": dataset.fingerprint,
        "lossContractFingerprint": dataset.loss_contract_fingerprint,
        "trainingContract": training_contract,
        "completedEpochs": training_config.epochs,
        "globalStep": global_step,
        "initialActor": initial_actor,
    }
    if latest_post_epoch_policy_drift_audit is not None:
        assert post_epoch_policy_drift_audit_contract_fingerprint is not None
        if latest_post_epoch_policy_drift_audit.get(
            "actorStateSha256"
        ) != _actor_state_sha256(actor.state_dict()):
            raise ValueError(
                "final post-epoch audit Actor state SHA-256 does not match"
            )
        candidate_metadata["finalPostEpochPolicyDriftAudit"] = (
            latest_post_epoch_policy_drift_audit
        )
        candidate_metadata[
            "finalPostEpochPolicyDriftAuditFingerprint"
        ] = hashlib.sha256(
            canonical_json_bytes(latest_post_epoch_policy_drift_audit)
        ).hexdigest()
        candidate_metadata[
            "postEpochPolicyDriftAuditContractFingerprint"
        ] = post_epoch_policy_drift_audit_contract_fingerprint
    candidate_manifest = export_v4_actor_bundle(
        actor,
        output / "candidate",
        metadata=candidate_metadata,
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
    if latest_post_epoch_policy_drift_audit is not None:
        result["finalPostEpochPolicyDriftAudit"] = (
            latest_post_epoch_policy_drift_audit
        )
        result["finalPostEpochPolicyDriftAuditFingerprint"] = hashlib.sha256(
            canonical_json_bytes(latest_post_epoch_policy_drift_audit)
        ).hexdigest()
        result["postEpochPolicyDriftAuditContractFingerprint"] = (
            post_epoch_policy_drift_audit_contract_fingerprint
        )
    _atomic_write(output / "result.json", canonical_json_bytes(result))
    return result


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
    """Run training; failed resume attempts roll every global RNG back."""

    if resume is None:
        return _train_v4_impl(
            dataset,
            output_directory,
            training_config,
            device=device,
            resume=resume,
            initialize_actor_bundle=initialize_actor_bundle,
            include_onnx=include_onnx,
        )
    rng_snapshot = _capture_global_rng_state()
    try:
        return _train_v4_impl(
            dataset,
            output_directory,
            training_config,
            device=device,
            resume=resume,
            initialize_actor_bundle=initialize_actor_bundle,
            include_onnx=include_onnx,
        )
    except BaseException:
        _restore_global_rng_state(rng_snapshot)
        raise


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
    parser.add_argument(
        "--expected-fixed-collection-plan-sha256",
        help=(
            "required precommitted collection-plan SHA-256 for fixed-only PPO data"
        ),
    )
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
        expected_fixed_collection_plan_sha256=(
            args.expected_fixed_collection_plan_sha256
        ),
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
