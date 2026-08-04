from __future__ import annotations

"""Leakage-safe V6 offline action-Q and public delta pilot trainer.

The centralized V/Q network is training-only.  Its three Q heads see the
sealed privileged state, but receive a Monte-Carlo label only at the action
actually recorded in the corpus.  The deployable delta heads receive public
V5 observations only.  They are distilled at the recorded behavior action
and at the Normal anchor; unobserved legal actions receive only a conservative
positive-score retention hinge, never a copied counterfactual target.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from v5_export import (
    canonical_json_bytes,
    load_v5_actor_bundle,
    sha256_file,
    tensor_state_sha256,
    v5_actor_bundle_digests,
)
from v5_model import pack_legal_actions
from v5_public import actor_batch_from_packed_arrays
from v5_train import load_v5_critic_checkpoint, verify_v5_model_pair
from v6_override import (
    ACTION_COUNT,
    BOOTSTRAP_HEADS,
    CENTRAL_Q_CONTRACT,
    V6CentralBootstrapActionQCritic,
    V6CentralBootstrapQConfig,
    V6PublicDeltaConfig,
    V6PublicDeltaScorer,
    deterministic_bootstrap_membership,
    public_delta_api_has_no_privileged_input,
)
from v6_pretrain import (
    PLAYER_COUNTS,
    V6_PRETRAIN_FORMAT,
    V6MatchView,
    V6SplitDataset,
    _verify_json_sidecar,
    load_v6_split_dataset,
    monte_carlo_targets_by_shard,
)
from v6_targets import V6_MC_RETURN_CONTRACT


V6_OFFLINE_FORMAT = "dalmuti-v6-offline-central-q-public-delta-pilot"
V6_OFFLINE_VERSION = 2
V6_BEHAVIOR_BALANCE_CONTRACT = "dalmuti-v6-equal-p-normal-alternative-mass-v1"
V6_DISTILL_CONTRACT = "dalmuti-v6-observed-behavior-normal-anchor-distill-v1"
V6_PRIVATE_PROHIBITION_CONTRACT = "dalmuti-v6-public-checkpoint-no-private-input-v1"
PILOT_FRACTIONS = (0.05, 0.10)


def _strict_export_object(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical ASCII JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


@dataclass(frozen=True)
class V6OfflineConfig:
    seed: int = 860_200_001
    pilot_fraction: float = 0.10
    q_epochs: int = 4
    distill_epochs: int = 4
    q_batch_size: int = 512
    distill_batch_size: int = 64
    validation_batch_size: int = 64
    q_learning_rate: float = 3.0e-4
    distill_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    huber_delta: float = 1.0
    value_loss_weight: float = 1.0
    q_loss_weight: float = 1.0
    alternative_mass_fraction: float = 0.50
    distill_alternative_mass_fraction: float = 0.50
    retention_hinge_weight: float = 0.25
    retention_margin: float = 0.0
    maximum_gradient_norm: float = 1.0
    use_amp: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFF_FFFF
        ):
            raise ValueError("seed must be an explicit uint32")
        if not any(
            math.isclose(float(self.pilot_fraction), value, rel_tol=0.0, abs_tol=1e-12)
            for value in PILOT_FRACTIONS
        ):
            raise ValueError("pilot_fraction must be exactly 0.05 or 0.10")
        for name, maximum in (
            ("q_epochs", 1000),
            ("distill_epochs", 1000),
            ("q_batch_size", 65_536),
            ("distill_batch_size", 4096),
            ("validation_batch_size", 65_536),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1,{maximum}]")
        for name in (
            "q_learning_rate",
            "distill_learning_rate",
            "huber_delta",
            "value_loss_weight",
            "q_loss_weight",
            "maximum_gradient_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "retention_hinge_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "alternative_mass_fraction",
            "distill_alternative_mass_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite in (0,1)")
        if not math.isfinite(float(self.retention_margin)):
            raise ValueError("retention_margin must be finite")
        if type(self.use_amp) is not bool:
            raise ValueError("use_amp must be an exact bool")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def balanced_behavior_player_weights(
    player_counts: object,
    behavior_actions: object,
    normal_actions: object,
    *,
    alternative_mass_fraction: float = 0.50,
) -> tuple[np.ndarray, dict[str, object]]:
    """Give each p equal mass and a configurable Normal/alternative mixture."""

    players = np.asarray(player_counts)
    behavior = np.asarray(behavior_actions)
    normal = np.asarray(normal_actions)
    if (
        players.ndim != 1
        or behavior.shape != players.shape
        or normal.shape != players.shape
        or not np.issubdtype(players.dtype, np.integer)
        or not np.issubdtype(behavior.dtype, np.integer)
        or not np.issubdtype(normal.dtype, np.integer)
        or np.issubdtype(players.dtype, np.bool_)
    ):
        raise ValueError("player/action arrays must be aligned integer vectors")
    if players.size < 1 or np.any((players < 4) | (players > 10)):
        raise ValueError("player_counts must contain only p4..p10")
    if np.any((behavior < 0) | (behavior >= ACTION_COUNT)) or np.any(
        (normal < 0) | (normal >= ACTION_COUNT)
    ):
        raise ValueError("behavior and Normal actions must use the fixed catalogue")
    alternative_fraction = float(alternative_mass_fraction)
    if not math.isfinite(alternative_fraction) or not 0.0 < alternative_fraction < 1.0:
        raise ValueError("alternative_mass_fraction must be finite in (0,1)")

    alternative = behavior != normal
    weights = np.zeros(players.shape, dtype=np.float32)
    p_mass = players.size / len(PLAYER_COUNTS)
    counts: dict[str, object] = {}
    masses: dict[str, object] = {}
    for player_count in PLAYER_COUNTS:
        counts[str(player_count)] = {}
        masses[str(player_count)] = {}
        for label, selected_kind in (("normal", ~alternative), ("alternative", alternative)):
            selected = (players == player_count) & selected_kind
            count = int(selected.sum())
            if count == 0:
                raise ValueError(
                    f"balanced Q training requires {label} rows for p{player_count}"
                )
            kind_fraction = (
                alternative_fraction if label == "alternative" else 1.0 - alternative_fraction
            )
            weights[selected] = np.float32(p_mass * kind_fraction / count)
            counts[str(player_count)][label] = count  # type: ignore[index]
            masses[str(player_count)][label] = float(  # type: ignore[index]
                weights[selected].sum(dtype=np.float64)
            )
    return weights, {
        "contract": V6_BEHAVIOR_BALANCE_CONTRACT,
        "counts": counts,
        "lossMass": masses,
        "normalRows": int((~alternative).sum()),
        "alternativeRows": int(alternative.sum()),
        "alternativeMassFraction": alternative_fraction,
        "normalMassFraction": 1.0 - alternative_fraction,
        "minimumRowWeight": float(weights.min()),
        "maximumRowWeight": float(weights.max()),
        "maximumToMinimumRowWeightRatio": float(weights.max() / weights.min()),
        "totalMass": float(weights.sum(dtype=np.float64)),
        "meanRowWeight": float(weights.mean(dtype=np.float64)),
    }


def equal_player_value_weights(
    player_counts: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """Equal-p weights for state-V, independent of behavior type."""

    players = np.asarray(player_counts)
    if (
        players.ndim != 1
        or not np.issubdtype(players.dtype, np.integer)
        or np.issubdtype(players.dtype, np.bool_)
        or players.size < 1
        or np.any((players < 4) | (players > 10))
    ):
        raise ValueError("value player_counts must be an integer p4..p10 vector")
    weights = np.zeros(players.shape, dtype=np.float32)
    counts: dict[str, int] = {}
    masses: dict[str, float] = {}
    p_mass = players.size / len(PLAYER_COUNTS)
    for player_count in PLAYER_COUNTS:
        selected = players == player_count
        count = int(selected.sum())
        if count == 0:
            raise ValueError(f"state-V training requires rows for p{player_count}")
        weights[selected] = np.float32(p_mass / count)
        counts[str(player_count)] = count
        masses[str(player_count)] = float(weights[selected].sum(dtype=np.float64))
    return weights, {
        "contract": "dalmuti-v6-equal-p-state-value-mass-v1",
        "counts": counts,
        "lossMass": masses,
        "minimumRowWeight": float(weights.min()),
        "maximumRowWeight": float(weights.max()),
        "maximumToMinimumRowWeightRatio": float(weights.max() / weights.min()),
        "totalMass": float(weights.sum(dtype=np.float64)),
        "meanRowWeight": float(weights.mean(dtype=np.float64)),
    }


def selected_packed_positions(
    packed_action_ids: torch.Tensor,
    packed_action_mask: torch.Tensor,
    selected_actions: torch.Tensor,
    *,
    label: str = "behavior",
) -> torch.Tensor:
    """Require each logged action to occur exactly once in its packed legal set."""

    if (
        packed_action_ids.dtype != torch.long
        or packed_action_mask.dtype != torch.bool
        or packed_action_ids.ndim != 2
        or packed_action_mask.shape != packed_action_ids.shape
        or selected_actions.dtype != torch.long
        or selected_actions.shape != (packed_action_ids.shape[0],)
        or len({packed_action_ids.device, packed_action_mask.device, selected_actions.device}) != 1
    ):
        raise ValueError("packed actions and selected actions have incompatible contracts")
    matches = packed_action_mask & (
        packed_action_ids == selected_actions.unsqueeze(1)
    )
    counts = matches.sum(dim=1)
    if not torch.equal(counts, torch.ones_like(counts)):
        bad = int((counts != 1).nonzero(as_tuple=False)[0, 0])
        raise ValueError(
            f"{label} action must occur exactly once in packed legal actions; "
            f"row {bad} had {int(counts[bad])} occurrences"
        )
    return matches.to(torch.int64).argmax(dim=1)


def weighted_central_v_q_loss(
    *,
    values: torch.Tensor,
    q_values: torch.Tensor,
    selected_positions: torch.Tensor,
    targets: torch.Tensor,
    value_row_weights: torch.Tensor,
    q_row_weights: torch.Tensor,
    bootstrap_membership: torch.Tensor,
    q_head_weight_mass_per_value_row: torch.Tensor | None = None,
    huber_delta: float = 1.0,
    value_loss_weight: float = 1.0,
    q_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit V and each Q head; Q supervision touches logged actions only."""

    batch_size = values.shape[0]
    if (
        values.shape != (batch_size,)
        or q_values.ndim != 3
        or q_values.shape[0] != batch_size
        or q_values.shape[2] != BOOTSTRAP_HEADS
        or selected_positions.dtype != torch.long
        or selected_positions.shape != (batch_size,)
        or targets.shape != (batch_size,)
        or value_row_weights.shape != (batch_size,)
        or q_row_weights.shape != (batch_size,)
        or bootstrap_membership.dtype != torch.bool
        or bootstrap_membership.shape != (batch_size, BOOTSTRAP_HEADS)
    ):
        raise ValueError("central V/Q loss tensors have incompatible shapes or dtypes")
    if len(
        {
            values.device,
            q_values.device,
            selected_positions.device,
            targets.device,
            value_row_weights.device,
            q_row_weights.device,
            bootstrap_membership.device,
        }
    ) != 1:
        raise ValueError("central V/Q loss tensors must share a device")
    if q_head_weight_mass_per_value_row is None:
        q_head_weight_mass_per_value_row = torch.ones(
            BOOTSTRAP_HEADS, dtype=torch.float32, device=q_values.device
        )
    if (
        q_head_weight_mass_per_value_row.shape != (BOOTSTRAP_HEADS,)
        or not q_head_weight_mass_per_value_row.dtype.is_floating_point
        or q_head_weight_mass_per_value_row.device != q_values.device
        or not torch.isfinite(q_head_weight_mass_per_value_row).all()
        or (q_head_weight_mass_per_value_row <= 0).any()
    ):
        raise ValueError("q_head_weight_mass_per_value_row must be positive float [3]")
    if (
        (selected_positions < 0).any()
        or (selected_positions >= q_values.shape[1]).any()
        or not torch.isfinite(values).all()
        or not torch.isfinite(targets).all()
        or not torch.isfinite(value_row_weights).all()
        or not torch.isfinite(q_row_weights).all()
        or (value_row_weights <= 0).any()
        or (q_row_weights < 0).any()
    ):
        raise ValueError("central V/Q loss received an invalid value")
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    selected = q_values[
        torch.arange(batch_size, device=q_values.device), selected_positions
    ]
    value_rows = F.huber_loss(values.float(), targets.float(), delta=delta, reduction="none")
    # The globally constructed weights have mean one.  Dividing by local
    # weight mass would cancel them in p-specific shards and small batches, so
    # use the uniformly sampled row count as the fixed SGD denominator.
    value_loss = (value_rows * value_row_weights).sum() / batch_size
    q_head_losses: list[torch.Tensor] = []
    q_head_active: list[bool] = []
    q_eligible = q_row_weights > 0
    if bool(q_eligible.any()):
        for head in range(BOOTSTRAP_HEADS):
            membership = bootstrap_membership[:, head] & q_eligible
            if not bool(membership.any()):
                q_head_losses.append(q_values.sum() * 0.0)
                q_head_active.append(False)
                continue
            head_weights = q_row_weights[membership]
            head_rows = F.huber_loss(
                selected[membership, head].float(),
                targets[membership].float(),
                delta=delta,
                reduction="none",
            )
            q_head_losses.append(
                (head_rows * head_weights).sum()
                / (batch_size * q_head_weight_mass_per_value_row[head])
            )
            q_head_active.append(True)
    else:
        # A shuffled batch can contain only forced decisions.  Those rows train
        # state-V but must contribute exactly zero Q gradient.
        q_head_losses = [q_values.sum() * 0.0 for _ in range(BOOTSTRAP_HEADS)]
        q_head_active = [False] * BOOTSTRAP_HEADS
    # Always average all three heads.  A batch-local inactive head contributes
    # zero; renormalizing over active heads would overweight sparse memberships.
    q_loss = torch.stack(q_head_losses).mean()
    total = float(value_loss_weight) * value_loss + float(q_loss_weight) * q_loss
    return total, {
        "value": value_loss,
        "q": q_loss,
        "qHeads": torch.stack(q_head_losses),
        "qActiveHeads": torch.tensor(
            q_head_active, dtype=torch.bool, device=q_values.device
        ),
    }


def observed_delta_distillation_loss(
    *,
    student_scores: torch.Tensor,
    action_ids: torch.Tensor,
    action_mask: torch.Tensor,
    behavior_positions: torch.Tensor,
    normal_positions: torch.Tensor,
    behavior_actions: torch.Tensor,
    normal_actions: torch.Tensor,
    teacher_behavior_deltas: torch.Tensor,
    row_weights: torch.Tensor,
    retention_hinge_weight: float = 0.25,
    retention_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill only behavior deltas and zero Normal anchors.

    Other legal actions have no counterfactual label.  A one-sided hinge may
    keep their unobserved score from becoming spuriously positive, but never
    treats a teacher extrapolation as their ground truth.
    """

    batch_size, width = action_ids.shape
    if (
        student_scores.shape != (batch_size, width, BOOTSTRAP_HEADS)
        or action_ids.dtype != torch.long
        or action_mask.dtype != torch.bool
        or action_mask.shape != action_ids.shape
        or behavior_positions.dtype != torch.long
        or normal_positions.dtype != torch.long
        or behavior_positions.shape != (batch_size,)
        or normal_positions.shape != (batch_size,)
        or behavior_actions.dtype != torch.long
        or normal_actions.dtype != torch.long
        or behavior_actions.shape != (batch_size,)
        or normal_actions.shape != (batch_size,)
        or teacher_behavior_deltas.shape != (batch_size, BOOTSTRAP_HEADS)
        or row_weights.shape != (batch_size,)
    ):
        raise ValueError("public delta distillation tensors have incompatible contracts")
    devices = {
        student_scores.device,
        action_ids.device,
        action_mask.device,
        behavior_positions.device,
        normal_positions.device,
        behavior_actions.device,
        normal_actions.device,
        teacher_behavior_deltas.device,
        row_weights.device,
    }
    if len(devices) != 1 or (row_weights <= 0).any():
        raise ValueError("distillation tensors must share a device and positive weights")
    rows = torch.arange(batch_size, device=student_scores.device)
    if (
        not action_mask[rows, behavior_positions].all()
        or not action_mask[rows, normal_positions].all()
        or not torch.equal(action_ids[rows, behavior_positions], behavior_actions)
        or not torch.equal(action_ids[rows, normal_positions], normal_actions)
    ):
        raise ValueError("behavior/Normal positions do not identify their packed actions")
    behavior_scores = student_scores[rows, behavior_positions]
    normal_scores = student_scores[rows, normal_positions]
    alternative = behavior_actions != normal_actions
    behavior_rows = F.smooth_l1_loss(
        behavior_scores.float(), teacher_behavior_deltas.float(), reduction="none"
    ).mean(dim=1)
    anchor_rows = F.smooth_l1_loss(
        normal_scores.float(), torch.zeros_like(normal_scores, dtype=torch.float32), reduction="none"
    ).mean(dim=1)
    # Normal rows have one anchor.  Alternative rows split their supervised
    # row mass equally between the observed behavior target and Normal anchor.
    supervised_rows = torch.where(
        alternative,
        0.5 * (behavior_rows + anchor_rows),
        anchor_rows,
    )
    # As above, the row weights encode a global p x behavior mixture.  A local
    # weight-sum denominator would erase that mixture in p-specific shards.
    supervised = (supervised_rows * row_weights).sum() / batch_size

    observed = torch.zeros_like(action_mask)
    observed[rows, behavior_positions] = True
    observed[rows, normal_positions] = True
    unobserved = action_mask & ~observed
    relative_scores = student_scores.float() - normal_scores.float().unsqueeze(1)
    positive = F.relu(relative_scores - float(retention_margin)).square().mean(dim=2)
    per_row_count = unobserved.sum(dim=1)
    retention_rows = (
        (positive * unobserved).sum(dim=1)
        / per_row_count.clamp_min(1).to(positive.dtype)
    )
    eligible = per_row_count > 0
    if bool(eligible.any()):
        retention = (
            retention_rows[eligible] * row_weights[eligible]
        ).sum() / batch_size
    else:
        retention = supervised.new_zeros(())
    total = supervised + float(retention_hinge_weight) * retention
    return total, {
        "supervised": supervised,
        "retention": retention,
        "alternativeRows": alternative.sum(),
        "unobservedLegalItems": unobserved.sum(),
    }


def _autocast(device: torch.device, enabled: bool):  # type: ignore[no-untyped-def]
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device.type, enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def _make_scaler(enabled: bool):  # type: ignore[no-untyped-def]
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _emit_progress(
    *,
    phase: str,
    epoch: int,
    epochs: int,
    batches: int,
    rows: int,
    total_rows: int,
    started: float,
    device: torch.device,
) -> None:
    elapsed = time.monotonic() - started
    rate = rows / elapsed if elapsed > 0.0 else 0.0
    remaining = max(0, total_rows - rows)
    payload = {
        "event": "v6-offline-progress",
        "phase": phase,
        "epoch": epoch,
        "epochs": epochs,
        "batches": batches,
        "rows": rows,
        "totalRows": total_rows,
        "elapsedSeconds": elapsed,
        "rowsPerSecond": rate,
        "estimatedEpochSecondsRemaining": remaining / rate if rate > 0.0 else None,
        "cudaMaxMemoryBytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _shuffled_batches(
    rows_by_shard: Mapping[str, np.ndarray], batch_size: int, seed: int
) -> Iterator[tuple[str, np.ndarray]]:
    """Globally interleave batches while retaining compact per-shard rows."""

    generator = np.random.default_rng(seed)
    orders: dict[str, np.ndarray] = {}
    descriptors: list[tuple[str, int, int]] = []
    for digest in sorted(rows_by_shard):
        rows = rows_by_shard[digest]
        orders[digest] = generator.permutation(len(rows))
        for start in range(0, len(rows), batch_size):
            descriptors.append((digest, start, min(start + batch_size, len(rows))))
    for descriptor_index in generator.permutation(len(descriptors)):
        digest, start, stop = descriptors[int(descriptor_index)]
        yield digest, rows_by_shard[digest][orders[digest][start:stop]]


def _dense_lookup(
    dataset: V6SplitDataset,
    rows_by_shard: Mapping[str, np.ndarray],
    values_by_shard: Mapping[str, np.ndarray],
    *,
    dtype: np.dtype,
    fill_value: object,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for digest, rows in rows_by_shard.items():
        values = np.asarray(values_by_shard[digest], dtype=dtype)
        if values.shape[0] != rows.shape[0]:
            raise ValueError("dense lookup values are not row-aligned")
        shape = (dataset.shards[digest].actor.decision_count, *values.shape[1:])
        lookup = np.full(shape, fill_value, dtype=dtype)
        lookup[rows] = values
        output[digest] = lookup
    return output


def _view_bootstrap_membership_lookups(
    dataset: V6SplitDataset,
    view: V6MatchView,
    seed: int,
) -> dict[str, np.ndarray]:
    """Hash each match once, then broadcast three booleans to its decisions."""

    output = {
        digest: np.zeros(
            (dataset.shards[digest].actor.decision_count, BOOTSTRAP_HEADS),
            dtype=np.bool_,
        )
        for digest in {record.shard_manifest_sha256 for record in view.matches}
    }
    occupied = {
        digest: np.zeros(dataset.shards[digest].actor.decision_count, dtype=np.bool_)
        for digest in output
    }
    match_membership = deterministic_bootstrap_membership(
        [record.split_hash for record in view.matches], seed=seed
    )
    for record, membership in zip(view.matches, match_membership, strict=True):
        digest = record.shard_manifest_sha256
        selected = occupied[digest][record.decision_start : record.decision_end]
        if bool(selected.any()):
            raise ValueError("selected complete matches overlap local decision rows")
        output[digest][record.decision_start : record.decision_end] = membership
        occupied[digest][record.decision_start : record.decision_end] = True
    return output


def _packed_batch(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    action_features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = np.asarray(arrays["legal_action_bits"][indices], dtype=np.uint8)
    dense = np.unpackbits(bits, axis=1, bitorder="little")[:, :ACTION_COUNT].astype(
        np.bool_, copy=False
    )
    legal = torch.from_numpy(np.ascontiguousarray(dense)).to(device=device, dtype=torch.bool)
    action_ids, action_mask = pack_legal_actions(legal)
    behavior = torch.from_numpy(
        np.ascontiguousarray(arrays["actions"][indices])
    ).to(device=device, dtype=torch.long)
    normal = torch.from_numpy(
        np.ascontiguousarray(arrays["normal_actions"][indices])
    ).to(device=device, dtype=torch.long)
    behavior_positions = selected_packed_positions(
        action_ids, action_mask, behavior, label="behavior"
    )
    # Validate Normal here as well; downstream distillation depends on both.
    selected_packed_positions(action_ids, action_mask, normal, label="Normal")
    safe = action_ids.clamp(0, ACTION_COUNT - 1)
    features = action_features.index_select(0, safe.reshape(-1)).reshape(
        *safe.shape, action_features.shape[1]
    )
    features = features.masked_fill(~action_mask.unsqueeze(-1), 0.0)
    return action_ids, action_mask, features, behavior_positions, normal


def _initialize_central_from_pretrained_value(
    central: V6CentralBootstrapActionQCritic, pretrained: torch.nn.Module
) -> dict[str, object]:
    """Copy the architecture-identical V6 warm-start V trunk exactly."""

    cfg = central.config
    source_cfg = getattr(pretrained, "config", None)
    expected = {
        "privileged_features": cfg.privileged_features,
        "d_model": cfg.d_model,
        "hidden_layers": cfg.hidden_layers,
        "player_count_embedding": cfg.player_count_embedding,
        "dropout": cfg.dropout,
    }
    if source_cfg is None or source_cfg.to_dict() != expected:
        raise ValueError("pretrained V critic architecture cannot initialize V6 central Q")
    central.player_count_embedding.load_state_dict(
        pretrained.player_count_embedding.state_dict(), strict=True
    )
    source_trunk = pretrained.value_network[:-1]
    central.state_encoder.load_state_dict(source_trunk.state_dict(), strict=True)
    central.value_output.load_state_dict(pretrained.value_output.state_dict(), strict=True)
    return {
        "contract": "dalmuti-v6-exact-pretrained-value-trunk-copy-v1",
        "sourceTensorStateSha256": tensor_state_sha256(pretrained.state_dict()),
        "copiedValueTensorStateSha256": tensor_state_sha256(
            {
                **{f"player_count_embedding.{k}": v for k, v in central.player_count_embedding.state_dict().items()},
                **{f"state_encoder.{k}": v for k, v in central.state_encoder.state_dict().items()},
                **{f"value_output.{k}": v for k, v in central.value_output.state_dict().items()},
            }
        ),
    }


def _training_tables(
    dataset: V6SplitDataset,
    view: V6MatchView,
    seed: int,
    alternative_mass_fraction: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
]:
    targets = monte_carlo_targets_by_shard(view, dataset.shards)
    rows_by_shard = {digest: rows for digest, (rows, _) in targets.items()}
    target_values = {digest: values for digest, (_, values) in targets.items()}
    flat_value_players = np.concatenate(
        [
            np.asarray(dataset.shards[d].actor.arrays["global_codes"][rows, 1], np.int64)
            for d, rows in sorted(rows_by_shard.items())
        ]
    )
    flat_value_weights, value_weight_report = equal_player_value_weights(
        flat_value_players
    )
    q_rows_by_shard = {
        digest: rows[
            ~np.asarray(
                dataset.shards[digest].actor.arrays["forced"][rows], dtype=np.bool_
            )
        ]
        for digest, rows in rows_by_shard.items()
    }
    flat_q_players = np.concatenate(
        [
            np.asarray(dataset.shards[d].actor.arrays["global_codes"][rows, 1], np.int64)
            for d, rows in sorted(q_rows_by_shard.items())
        ]
    )
    flat_q_behavior = np.concatenate(
        [
            np.asarray(dataset.shards[d].actor.arrays["actions"][rows], np.int64)
            for d, rows in sorted(q_rows_by_shard.items())
        ]
    )
    flat_q_normal = np.concatenate(
        [
            np.asarray(dataset.shards[d].actor.arrays["normal_actions"][rows], np.int64)
            for d, rows in sorted(q_rows_by_shard.items())
        ]
    )
    flat_q_weights, q_weight_report = balanced_behavior_player_weights(
        flat_q_players,
        flat_q_behavior,
        flat_q_normal,
        alternative_mass_fraction=alternative_mass_fraction,
    )
    membership_lookup = _view_bootstrap_membership_lookups(dataset, view, seed)

    value_weight_values: dict[str, np.ndarray] = {}
    q_weight_values: dict[str, np.ndarray] = {}
    offset = 0
    for digest, rows in sorted(rows_by_shard.items()):
        stop = offset + len(rows)
        value_weight_values[digest] = flat_value_weights[offset:stop]
        offset = stop
    offset = 0
    for digest, rows in sorted(q_rows_by_shard.items()):
        stop = offset + len(rows)
        q_weight_values[digest] = flat_q_weights[offset:stop]
        offset = stop
    value_weights = _dense_lookup(
        dataset,
        rows_by_shard,
        value_weight_values,
        dtype=np.dtype(np.float32),
        fill_value=0.0,
    )
    q_weights = _dense_lookup(
        dataset,
        q_rows_by_shard,
        q_weight_values,
        dtype=np.dtype(np.float32),
        fill_value=0.0,
    )
    target_lookup = _dense_lookup(
        dataset, rows_by_shard, target_values, dtype=np.dtype(np.float32), fill_value=np.nan
    )
    q_membership = np.concatenate(
        [membership_lookup[d][rows] for d, rows in sorted(q_rows_by_shard.items())]
    )
    if not q_membership.any(axis=0).all():
        raise ValueError("a deterministic bootstrap head received no nonforced Q rows")
    q_alternative = flat_q_behavior != flat_q_normal
    q_head_mass_per_value_row = np.asarray(
        [
            float(
                flat_q_weights[q_membership[:, head]].sum(dtype=np.float64)
                / flat_value_players.size
            )
            for head in range(BOOTSTRAP_HEADS)
        ],
        dtype=np.float64,
    )
    if np.any(q_head_mass_per_value_row <= 0.0):
        raise ValueError("bootstrap Q head has zero global weighted loss mass")
    per_head_cells: dict[str, object] = {}
    for head in range(BOOTSTRAP_HEADS):
        per_head_cells[str(head)] = {}
        for player_count in PLAYER_COUNTS:
            per_head_cells[str(head)][str(player_count)] = {}  # type: ignore[index]
            for label, kind in (("normal", ~q_alternative), ("alternative", q_alternative)):
                selected = q_membership[:, head] & (flat_q_players == player_count) & kind
                per_head_cells[str(head)][str(player_count)][label] = {  # type: ignore[index]
                    "rows": int(selected.sum()),
                    "qLossMass": float(flat_q_weights[selected].sum(dtype=np.float64)),
                }
    bootstrap_report = {
        "contract": "dalmuti-v6-match-key-bootstrap-membership-v1",
        "seed": seed,
        "nonforcedQRowsPerHead": [
            int(q_membership[:, head].sum()) for head in range(BOOTSTRAP_HEADS)
        ],
        "uniqueMatches": view.match_count,
        "membershipIsWholeMatch": True,
        "hashesComputedOncePerMatch": True,
        "qHeadWeightMassPerValueRow": q_head_mass_per_value_row.tolist(),
        "fixedGlobalSgdDenominator": True,
        "perHeadPlayerBehaviorCells": per_head_cells,
    }
    return rows_by_shard, target_lookup, value_weights, q_weights, membership_lookup, {
        "valueRows": sum(len(rows) for rows in rows_by_shard.values()),
        "qNonforcedRows": sum(len(rows) for rows in q_rows_by_shard.values()),
        "forcedRowsExcludedFromQ": sum(len(rows) for rows in rows_by_shard.values())
        - sum(len(rows) for rows in q_rows_by_shard.values()),
        "valueLossWeighting": value_weight_report,
        "qLossWeighting": q_weight_report,
        "bootstrap": bootstrap_report,
    }


def _train_central_q(
    central: V6CentralBootstrapActionQCritic,
    actor_action_features: torch.Tensor,
    dataset: V6SplitDataset,
    view: V6MatchView,
    config: V6OfflineConfig,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object], float]:
    rows, targets, value_weights, q_weights, memberships, reports = _training_tables(
        dataset, view, config.seed + 101, config.alternative_mass_fraction
    )
    constant_numerator = 0.0
    constant_mass = 0.0
    for digest, selected in rows.items():
        local_weights = value_weights[digest][selected].astype(np.float64)
        local_targets = targets[digest][selected].astype(np.float64)
        constant_numerator += float((local_targets * local_weights).sum())
        constant_mass += float(local_weights.sum())
    constant = constant_numerator / constant_mass

    optimizer = torch.optim.AdamW(
        central.parameters(), lr=config.q_learning_rate, weight_decay=config.weight_decay
    )
    amp = bool(config.use_amp and device.type == "cuda")
    scaler = _make_scaler(amp)
    epochs: list[dict[str, object]] = []
    q_head_normalizers = torch.tensor(
        reports["bootstrap"]["qHeadWeightMassPerValueRow"],  # type: ignore[index]
        dtype=torch.float32,
        device=device,
    )
    central.train()
    for epoch in range(config.q_epochs):
        epoch_started = time.monotonic()
        sums = {"total": 0.0, "value": 0.0, "q": 0.0}
        mass = 0.0
        batches = 0
        rows_seen = 0
        for digest, indices in _shuffled_batches(
            rows, config.q_batch_size, config.seed + 10_000 + epoch
        ):
            shard = dataset.shards[digest]
            _, mask, action_features, selected_positions, _ = _packed_batch(
                shard.actor.arrays, indices, actor_action_features, device
            )
            privileged = torch.from_numpy(
                np.ascontiguousarray(shard.privileged_arrays["privileged_states"][indices])
            ).to(device=device, dtype=torch.float32)
            players = torch.from_numpy(
                np.ascontiguousarray(shard.actor.arrays["global_codes"][indices, 1])
            ).to(device=device, dtype=torch.long)
            target = torch.from_numpy(np.ascontiguousarray(targets[digest][indices])).to(
                device=device, dtype=torch.float32
            )
            value_row_weight = torch.from_numpy(
                np.ascontiguousarray(value_weights[digest][indices])
            ).to(
                device=device, dtype=torch.float32
            )
            q_row_weight = torch.from_numpy(
                np.ascontiguousarray(q_weights[digest][indices])
            ).to(device=device, dtype=torch.float32)
            membership = torch.from_numpy(
                np.ascontiguousarray(memberships[digest][indices])
            ).to(device=device, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                output = central(privileged, action_features, mask, players)
                loss, parts = weighted_central_v_q_loss(
                    values=output.values,
                    q_values=output.q_values,
                    selected_positions=selected_positions,
                    targets=target,
                    value_row_weights=value_row_weight,
                    q_row_weights=q_row_weight,
                    bootstrap_membership=membership,
                    q_head_weight_mass_per_value_row=q_head_normalizers,
                    huber_delta=config.huber_delta,
                    value_loss_weight=config.value_loss_weight,
                    q_loss_weight=config.q_loss_weight,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V6 central V/Q loss became non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(central.parameters(), config.maximum_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            local_mass = float(len(indices))
            sums["total"] += float(loss.detach().cpu()) * local_mass
            sums["value"] += float(parts["value"].detach().cpu()) * local_mass
            sums["q"] += float(parts["q"].detach().cpu()) * local_mass
            mass += local_mass
            batches += 1
            rows_seen += len(indices)
            if batches % 100 == 0:
                _emit_progress(
                    phase="central-v-q",
                    epoch=epoch + 1,
                    epochs=config.q_epochs,
                    batches=batches,
                    rows=rows_seen,
                    total_rows=reports["valueRows"],  # type: ignore[arg-type]
                    started=epoch_started,
                    device=device,
                )
        _emit_progress(
            phase="central-v-q",
            epoch=epoch + 1,
            epochs=config.q_epochs,
            batches=batches,
            rows=rows_seen,
            total_rows=reports["valueRows"],  # type: ignore[arg-type]
            started=epoch_started,
            device=device,
        )
        epochs.append(
            {
                "epoch": epoch + 1,
                "batches": batches,
                "valueRows": reports["valueRows"],
                "qNonforcedRows": reports["qNonforcedRows"],
                "weightedTotal": sums["total"] / mass,
                "weightedValueHuber": sums["value"] / mass,
                "weightedSelectedActionQHuber": sums["q"] / mass,
                "elapsedSeconds": time.monotonic() - epoch_started,
                "rowsPerSecond": reports["valueRows"]
                / max(time.monotonic() - epoch_started, 1.0e-9),
                "batchOrder": "globally-random-interleaved-across-shards",
            }
        )
    central.eval()
    return epochs, reports, constant


def _metrics(target: np.ndarray, prediction: np.ndarray, delta: float) -> dict[str, float]:
    target64 = np.asarray(target, dtype=np.float64)
    predicted64 = np.asarray(prediction, dtype=np.float64)
    if target64.shape != predicted64.shape or target64.size == 0:
        raise ValueError("metric arrays must be aligned and non-empty")
    error = predicted64 - target64
    absolute = np.abs(error)
    quadratic = np.minimum(absolute, delta)
    huber = 0.5 * np.square(quadratic) + delta * (absolute - quadratic)
    variance = float(np.var(target64))
    explained = 0.0 if variance <= 0.0 else 1.0 - float(np.var(error)) / variance
    return {
        "count": int(target64.size),
        "huber": float(huber.mean()),
        "mae": float(absolute.mean()),
        "explainedVariance": explained,
    }


def _evaluate_central_q(
    central: V6CentralBootstrapActionQCritic,
    action_features: torch.Tensor,
    dataset: V6SplitDataset,
    view: V6MatchView,
    device: torch.device,
    batch_size: int,
    huber_delta: float,
    constant: float,
) -> dict[str, object]:
    target_map = monte_carlo_targets_by_shard(view, dataset.shards)
    q_predictions: list[np.ndarray] = []
    value_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    alternatives: list[np.ndarray] = []
    player_values: list[np.ndarray] = []
    forced_values: list[np.ndarray] = []
    per_head: list[list[np.ndarray]] = [[] for _ in range(BOOTSTRAP_HEADS)]
    central.eval()
    with torch.inference_mode():
        for digest, (rows, target_values) in target_map.items():
            shard = dataset.shards[digest]
            for start in range(0, len(rows), batch_size):
                indices = rows[start : start + batch_size]
                local_targets = target_values[start : start + batch_size]
                _, mask, features, positions, normal = _packed_batch(
                    shard.actor.arrays, indices, action_features, device
                )
                privileged = torch.from_numpy(
                    np.ascontiguousarray(shard.privileged_arrays["privileged_states"][indices])
                ).to(device=device, dtype=torch.float32)
                players = torch.from_numpy(
                    np.ascontiguousarray(shard.actor.arrays["global_codes"][indices, 1])
                ).to(device=device, dtype=torch.long)
                output = central(privileged, features, mask, players)
                selected = output.q_values[
                    torch.arange(len(indices), device=device), positions
                ].float().cpu().numpy()
                q_predictions.append(selected.mean(axis=1))
                for head in range(BOOTSTRAP_HEADS):
                    per_head[head].append(selected[:, head])
                value_predictions.append(output.values.float().cpu().numpy())
                targets.append(np.asarray(local_targets, dtype=np.float32))
                behavior = np.asarray(shard.actor.arrays["actions"][indices], np.int64)
                alternatives.append(behavior != normal.cpu().numpy())
                forced_values.append(
                    np.asarray(shard.actor.arrays["forced"][indices], dtype=np.bool_)
                )
                player_values.append(
                    np.asarray(
                        shard.actor.arrays["global_codes"][indices, 1], dtype=np.int64
                    )
                )
    y = np.concatenate(targets)
    q = np.concatenate(q_predictions)
    value = np.concatenate(value_predictions)
    alternative = np.concatenate(alternatives)
    player_counts = np.concatenate(player_values)
    forced = np.concatenate(forced_values)

    def population(selected: np.ndarray) -> dict[str, object]:
        if not bool(selected.any()):
            raise ValueError("central-Q metric population is empty")
        def model_record(prediction: np.ndarray) -> dict[str, object]:
            pooled = _metrics(y[selected], prediction[selected], huber_delta)
            per_player: dict[str, object] = {}
            for player_count in PLAYER_COUNTS:
                local = selected & (player_counts == player_count)
                if not bool(local.any()):
                    raise ValueError(
                        f"held-out population omitted p{player_count}"
                    )
                per_player[str(player_count)] = _metrics(
                    y[local], prediction[local], huber_delta
                )
            pooled["perPlayerCount"] = per_player  # type: ignore[index]
            pooled["equalPlayerCountMean"] = {  # type: ignore[index]
                metric: float(
                    np.mean(
                        [
                            per_player[str(player_count)][metric]  # type: ignore[index]
                            for player_count in PLAYER_COUNTS
                        ]
                    )
                )
                for metric in ("huber", "mae", "explainedVariance")
            }
            return pooled

        q_record = model_record(q)
        value_record = model_record(value)
        constant_record = model_record(np.full(y.shape, constant, dtype=np.float64))
        return {
            "selectedActionQ": q_record,
            "valueOnly": value_record,
            "constantBaseline": constant_record,
            "perQHead": [
                model_record(np.concatenate(per_head[head]))
                for head in range(BOOTSTRAP_HEADS)
            ],
            "comparisons": {
                "qPooledHuberBelowValueOnly": q_record["huber"] < value_record["huber"],
                "qPooledHuberBelowConstant": q_record["huber"] < constant_record["huber"],
                "qEqualPHuberBelowValueOnly": (
                    q_record["equalPlayerCountMean"]["huber"]  # type: ignore[index]
                    < value_record["equalPlayerCountMean"]["huber"]  # type: ignore[index]
                ),
                "qEqualPHuberBelowConstant": (
                    q_record["equalPlayerCountMean"]["huber"]  # type: ignore[index]
                    < constant_record["equalPlayerCountMean"]["huber"]  # type: ignore[index]
                ),
            },
        }

    all_decision = population(np.ones(y.shape, dtype=np.bool_))
    nonforced_all = population(~forced)
    alternative_nonforced = population((~forced) & alternative)
    normal_nonforced = population((~forced) & (~alternative))
    forced_reference = population(forced)

    def wins_both(record: Mapping[str, object]) -> bool:
        comparisons = record["comparisons"]
        assert isinstance(comparisons, Mapping)
        return all(bool(comparisons[name]) for name in (
            "qPooledHuberBelowValueOnly",
            "qPooledHuberBelowConstant",
            "qEqualPHuberBelowValueOnly",
            "qEqualPHuberBelowConstant",
        ))

    return {
        "population": "held-out complete-match validation decisions; Q gate excludes forced rows",
        "allDecisionValueReference": all_decision,
        "nonforcedAll": nonforced_all,
        "alternativeBehaviorNonforced": alternative_nonforced,
        "normalBehaviorNonforced": normal_nonforced,
        "forcedReferenceNotUsedForQGate": forced_reference,
        "trainingConstant": constant,
        "offlineQPilotGate": {
            "requiresNonforcedQBelowVAndConstantPooledAndEqualP": True,
            "requiresAlternativeQBelowVAndConstantPooledAndEqualP": True,
            "nonforcedPassed": wins_both(nonforced_all),
            "alternativePassed": wins_both(alternative_nonforced),
            "passed": wins_both(nonforced_all) and wins_both(alternative_nonforced),
            "gamePromotionClaim": False,
        },
    }


def _distill_rows(view: V6MatchView, dataset: V6SplitDataset) -> dict[str, np.ndarray]:
    return view.rows_by_shard(nonforced_only=True, shards=dataset.shards)


def _distill_weight_lookups(
    dataset: V6SplitDataset,
    rows: Mapping[str, np.ndarray],
    alternative_mass_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    players = np.concatenate(
        [np.asarray(dataset.shards[d].actor.arrays["global_codes"][r, 1], np.int64) for d, r in sorted(rows.items())]
    )
    behavior = np.concatenate(
        [np.asarray(dataset.shards[d].actor.arrays["actions"][r], np.int64) for d, r in sorted(rows.items())]
    )
    normal = np.concatenate(
        [np.asarray(dataset.shards[d].actor.arrays["normal_actions"][r], np.int64) for d, r in sorted(rows.items())]
    )
    flat, report = balanced_behavior_player_weights(
        players,
        behavior,
        normal,
        alternative_mass_fraction=alternative_mass_fraction,
    )
    values: dict[str, np.ndarray] = {}
    offset = 0
    for digest, selected in sorted(rows.items()):
        stop = offset + len(selected)
        values[digest] = flat[offset:stop]
        offset = stop
    return _dense_lookup(
        dataset, rows, values, dtype=np.dtype(np.float32), fill_value=0.0
    ), report


def _teacher_action_deltas(
    central: V6CentralBootstrapActionQCritic,
    privileged: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    players: torch.Tensor,
    behavior_positions: torch.Tensor,
    normal_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output = central(privileged, features, mask, players)
        rows = torch.arange(features.shape[0], device=features.device)
        normal_q = output.q_values[rows, normal_positions]
        all_deltas = (output.q_values - normal_q.unsqueeze(1)).float()
        return all_deltas[rows, behavior_positions], all_deltas


def _teacher_deltas(
    central: V6CentralBootstrapActionQCritic,
    privileged: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    players: torch.Tensor,
    behavior_positions: torch.Tensor,
    normal_positions: torch.Tensor,
) -> torch.Tensor:
    behavior, _ = _teacher_action_deltas(
        central,
        privileged,
        features,
        mask,
        players,
        behavior_positions,
        normal_positions,
    )
    return behavior


def _train_public_delta(
    scorer: V6PublicDeltaScorer,
    central: V6CentralBootstrapActionQCritic,
    action_features: torch.Tensor,
    dataset: V6SplitDataset,
    view: V6MatchView,
    config: V6OfflineConfig,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = _distill_rows(view, dataset)
    weights, weight_report = _distill_weight_lookups(
        dataset, rows, config.distill_alternative_mass_fraction
    )
    parameters = [parameter for parameter in scorer.delta_heads.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.requires_grad for parameter in scorer.public_actor.parameters()):
        raise RuntimeError("public distillation must train delta heads only")
    optimizer = torch.optim.AdamW(
        parameters, lr=config.distill_learning_rate, weight_decay=config.weight_decay
    )
    amp = bool(config.use_amp and device.type == "cuda")
    scaler = _make_scaler(amp)
    epochs: list[dict[str, object]] = []
    central.eval()
    for epoch in range(config.distill_epochs):
        epoch_started = time.monotonic()
        scorer.train()
        scorer.public_actor.eval()
        sums = {"total": 0.0, "supervised": 0.0, "retention": 0.0}
        mass = 0.0
        item_counts = {"alternativeRows": 0, "unobservedLegalItems": 0}
        batches = 0
        rows_seen = 0
        total_rows = sum(len(value) for value in rows.values())
        for digest, indices in _shuffled_batches(
            rows, config.distill_batch_size, config.seed + 20_000 + epoch
        ):
            shard = dataset.shards[digest]
            action_ids, mask, features, behavior_positions, normal = _packed_batch(
                shard.actor.arrays, indices, action_features, device
            )
            behavior = torch.from_numpy(
                np.ascontiguousarray(shard.actor.arrays["actions"][indices])
            ).to(device=device, dtype=torch.long)
            normal_positions = selected_packed_positions(
                action_ids, mask, normal, label="Normal"
            )
            privileged = torch.from_numpy(
                np.ascontiguousarray(shard.privileged_arrays["privileged_states"][indices])
            ).to(device=device, dtype=torch.float32)
            players = torch.from_numpy(
                np.ascontiguousarray(shard.actor.arrays["global_codes"][indices, 1])
            ).to(device=device, dtype=torch.long)
            teacher = _teacher_deltas(
                central, privileged, features, mask, players, behavior_positions, normal_positions
            )
            public = actor_batch_from_packed_arrays(shard.actor.arrays, indices, device)
            row_weight = torch.from_numpy(np.ascontiguousarray(weights[digest][indices])).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                output = scorer(public, action_ids, mask)
                loss, parts = observed_delta_distillation_loss(
                    student_scores=output.head_scores,
                    action_ids=action_ids,
                    action_mask=mask,
                    behavior_positions=behavior_positions,
                    normal_positions=normal_positions,
                    behavior_actions=behavior,
                    normal_actions=normal,
                    teacher_behavior_deltas=teacher,
                    row_weights=row_weight,
                    retention_hinge_weight=config.retention_hinge_weight,
                    retention_margin=config.retention_margin,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V6 public delta loss became non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.maximum_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            local_mass = float(len(indices))
            for name in sums:
                sums[name] += float((loss if name == "total" else parts[name]).detach().cpu()) * local_mass
            mass += local_mass
            item_counts["alternativeRows"] += int(parts["alternativeRows"].detach().cpu())
            item_counts["unobservedLegalItems"] += int(parts["unobservedLegalItems"].detach().cpu())
            batches += 1
            rows_seen += len(indices)
            if batches % 100 == 0:
                _emit_progress(
                    phase="public-delta-distill",
                    epoch=epoch + 1,
                    epochs=config.distill_epochs,
                    batches=batches,
                    rows=rows_seen,
                    total_rows=total_rows,
                    started=epoch_started,
                    device=device,
                )
        _emit_progress(
            phase="public-delta-distill",
            epoch=epoch + 1,
            epochs=config.distill_epochs,
            batches=batches,
            rows=rows_seen,
            total_rows=total_rows,
            started=epoch_started,
            device=device,
        )
        epochs.append(
            {
                "epoch": epoch + 1,
                "batches": batches,
                "rows": sum(len(value) for value in rows.values()),
                "weightedTotal": sums["total"] / mass,
                "weightedSupervised": sums["supervised"] / mass,
                "weightedRetentionHinge": sums["retention"] / mass,
                **item_counts,
                "elapsedSeconds": time.monotonic() - epoch_started,
                "rowsPerSecond": sum(len(value) for value in rows.values())
                / max(time.monotonic() - epoch_started, 1.0e-9),
                "batchOrder": "globally-random-interleaved-across-shards",
            }
        )
    scorer.eval()
    return epochs, {"lossWeighting": weight_report}


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left.astype(np.float64), right.astype(np.float64))[0, 1])


def _evaluate_public_delta(
    scorer: V6PublicDeltaScorer,
    central: V6CentralBootstrapActionQCritic,
    action_features: torch.Tensor,
    dataset: V6SplitDataset,
    view: V6MatchView,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    rows_by_shard = _distill_rows(view, dataset)
    teacher_values: list[np.ndarray] = []
    student_values: list[np.ndarray] = []
    alternative_values: list[np.ndarray] = []
    player_values: list[np.ndarray] = []
    override_values: list[np.ndarray] = []
    candidate_teacher_positive_values: list[np.ndarray] = []
    total = 0
    anchor_absolute = 0.0
    scorer.eval()
    central.eval()
    with torch.inference_mode():
        for digest, rows in rows_by_shard.items():
            shard = dataset.shards[digest]
            for start in range(0, len(rows), batch_size):
                indices = rows[start : start + batch_size]
                ids, mask, features, behavior_positions, normal = _packed_batch(
                    shard.actor.arrays, indices, action_features, device
                )
                behavior = torch.from_numpy(
                    np.ascontiguousarray(shard.actor.arrays["actions"][indices])
                ).to(device=device, dtype=torch.long)
                normal_positions = selected_packed_positions(ids, mask, normal, label="Normal")
                privileged = torch.from_numpy(
                    np.ascontiguousarray(shard.privileged_arrays["privileged_states"][indices])
                ).to(device=device, dtype=torch.float32)
                players = torch.from_numpy(
                    np.ascontiguousarray(shard.actor.arrays["global_codes"][indices, 1])
                ).to(device=device, dtype=torch.long)
                teacher, all_teacher_deltas = _teacher_action_deltas(
                    central, privileged, features, mask, players, behavior_positions, normal_positions
                )
                public = actor_batch_from_packed_arrays(shard.actor.arrays, indices, device)
                output = scorer(public, ids, mask)
                batch_rows = torch.arange(len(indices), device=device)
                normal_scores = output.head_scores[batch_rows, normal_positions]
                behavior_scores = output.head_scores[batch_rows, behavior_positions]
                student = behavior_scores - normal_scores
                teacher_values.append(teacher.cpu().numpy())
                student_values.append(student.float().cpu().numpy())
                alternative_values.append((behavior != normal).cpu().numpy())
                player_values.append(players.cpu().numpy())
                anchor_absolute += float(normal_scores.float().abs().sum().cpu())
                deltas = output.head_scores.float() - normal_scores.unsqueeze(1)
                lcb = deltas.mean(dim=2) - deltas.std(dim=2, correction=0)
                eligible = mask.clone()
                eligible[batch_rows, normal_positions] = False
                eligible_lcb = lcb.masked_fill(~eligible, -torch.inf)
                best_positions = eligible_lcb.argmax(dim=1)
                best = eligible_lcb[batch_rows, best_positions]
                overrides = best > 0.0
                candidate_teacher_mean = all_teacher_deltas[
                    batch_rows, best_positions
                ].mean(dim=1)
                override_values.append(overrides.cpu().numpy())
                candidate_teacher_positive_values.append(
                    (candidate_teacher_mean > 0.0).cpu().numpy()
                )
                total += len(indices)
    teacher = np.concatenate(teacher_values)
    student = np.concatenate(student_values)
    alternative = np.concatenate(alternative_values)
    player_counts = np.concatenate(player_values)
    overrides = np.concatenate(override_values)
    candidate_teacher_positive = np.concatenate(candidate_teacher_positive_values)
    if not bool(alternative.any()):
        raise ValueError("validation has no alternative behavior rows")
    t_mean = teacher.mean(axis=1)
    s_mean = student.mean(axis=1)
    selected = alternative
    def logged_delta_record(mask: np.ndarray) -> dict[str, object]:
        if not bool(mask.any()):
            raise ValueError("logged delta validation subset is empty")
        predicted = s_mean[mask]
        truth = t_mean[mask]
        predicted_positive_local = predicted > 0.0
        truth_positive_local = truth > 0.0
        nonzero = np.abs(truth) > 1.0e-8
        return {
            "rows": int(mask.sum()),
            "maeAllHeads": float(np.abs(student[mask] - teacher[mask]).mean()),
            "maeEnsemble": float(np.abs(predicted - truth).mean()),
            "correlation": _correlation(predicted, truth),
            "signAgreement": (
                float((np.sign(predicted[nonzero]) == np.sign(truth[nonzero])).mean())
                if bool(nonzero.any()) else None
            ),
            "positiveSignPrecision": (
                float(truth_positive_local[predicted_positive_local].mean())
                if bool(predicted_positive_local.any()) else None
            ),
            "predictedPositiveRows": int(predicted_positive_local.sum()),
            "label": "logged behavior action only",
        }

    def safe_gate_record(mask: np.ndarray) -> dict[str, object]:
        if not bool(mask.any()):
            raise ValueError("safe-gate validation subset is empty")
        local_override = overrides[mask]
        local_teacher_positive = candidate_teacher_positive[mask]
        true_positive = local_override & local_teacher_positive
        return {
            "rows": int(mask.sum()),
            "normalFallbackRows": int((~local_override).sum()),
            "normalFallbackRate": float((~local_override).mean()),
            "overrideRows": int(local_override.sum()),
            "overrideRate": float(local_override.mean()),
            "selectedCandidateTeacherPositiveRows": int(local_teacher_positive.sum()),
            "selectedCandidateTeacherPositivePrecision": (
                float(true_positive.sum() / local_override.sum())
                if bool(local_override.any()) else None
            ),
            "selectedCandidateTeacherPositiveRecall": (
                float(true_positive.sum() / local_teacher_positive.sum())
                if bool(local_teacher_positive.any()) else None
            ),
        }

    per_player: dict[str, object] = {}
    for player_count in PLAYER_COUNTS:
        local = player_counts == player_count
        per_player[str(player_count)] = {
            "allNonforced": safe_gate_record(local),
            "loggedNormalBehavior": safe_gate_record(local & ~alternative),
            "loggedAlternativeBehavior": safe_gate_record(local & alternative),
            "loggedAlternativeDelta": logged_delta_record(local & alternative),
        }

    return {
        "population": "held-out complete-match validation nonforced decisions",
        "allRows": int(total),
        "alternativeRows": int(selected.sum()),
        "loggedAlternativeBehaviorDistillation": logged_delta_record(alternative),
        "normalAnchorMeanAbsoluteScore": anchor_absolute / (total * BOOTSTRAP_HEADS),
        "safeGate": {
            "beta": 1.0,
            "threshold": 0.0,
            "overallNonforced": safe_gate_record(np.ones(total, dtype=np.bool_)),
            "loggedNormalBehavior": safe_gate_record(~alternative),
            "loggedAlternativeBehavior": safe_gate_record(alternative),
            "perPlayerCount": per_player,
            "selectedCandidateTeacherMetricWarning": (
                "central-Q sign is not counterfactual ground truth for an unlogged candidate"
            ),
            "note": "diagnostic only; no calibration or game promotion implied",
        },
        "teacherIdentificationLimit": (
            "Q(Normal) at an alternative-behavior state is model extrapolation; "
            "the logged corpus provides no same-state counterfactual MC label"
        ),
    }


def _torch_bytes(payload: object) -> bytes:
    output = io.BytesIO()
    torch.save(payload, output)
    return output.getvalue()


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in sorted(module.state_dict().items())
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _publish_output(
    output_directory: Path,
    central: V6CentralBootstrapActionQCritic,
    scorer: V6PublicDeltaScorer,
    result: dict[str, object],
) -> dict[str, object]:
    target = output_directory.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"immutable V6 offline output already exists: {target}")
    lock = target.parent / f".{target.name}.publish.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        central_state = _state(central)
        public_state = _state(scorer.delta_heads)
        central_payload = {
            "format": V6_OFFLINE_FORMAT,
            "version": V6_OFFLINE_VERSION,
            "kind": "centralized-training-only-v-plus-three-advantage-heads",
            "centralQContract": CENTRAL_Q_CONTRACT,
            "deployExportAllowed": False,
            "containsRawPrivateRows": False,
            "config": asdict(central.config),
            "tensorStateSha256": tensor_state_sha256(central_state),
            "stateDict": central_state,
        }
        public_payload = {
            "format": V6_OFFLINE_FORMAT,
            "version": V6_OFFLINE_VERSION,
            "kind": "public-delta-heads-only",
            "privilegedInputAllowed": False,
            "containsRawPrivateRows": False,
            "config": asdict(scorer.config),
            "baseActor": result["sources"]["publicActor"],  # type: ignore[index]
            "publicActorDModel": scorer.public_actor.config.d_model,
            "tensorStateSha256": tensor_state_sha256(public_state),
            "stateDict": public_state,
        }
        _write_exclusive(staging / "central-q.pt", _torch_bytes(central_payload))
        _write_exclusive(staging / "public-delta.pt", _torch_bytes(public_payload))
        result_raw = canonical_json_bytes(result)
        _write_exclusive(staging / "result.json", result_raw)
        inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir(), key=lambda value: value.name)
            if path.is_file()
        }
        manifest = {
            "format": V6_OFFLINE_FORMAT,
            "version": V6_OFFLINE_VERSION,
            "files": inventory,
            "resultSha256": hashlib.sha256(result_raw).hexdigest(),
            "sources": result["sources"],
            "privateDataProhibition": result["privateDataProhibition"],
        }
        manifest_raw = canonical_json_bytes(manifest)
        _write_exclusive(staging / "manifest.json", manifest_raw)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        _write_exclusive(
            staging / "manifest.json.sha256",
            f"{manifest_sha}  manifest.json\n".encode("ascii"),
        )
        os.rename(staging, target)
        return {
            "outputDirectory": str(target),
            "manifest": manifest,
            "manifestSha256": manifest_sha,
            "result": result,
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def verify_v6_offline_output(output_directory: str | Path) -> dict[str, object]:
    root = Path(output_directory).resolve(strict=True)
    expected = {
        "central-q.pt",
        "public-delta.pt",
        "result.json",
        "manifest.json",
        "manifest.json.sha256",
    }
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("V6 offline output has missing or untracked files")
    digest = sha256_file(root / "manifest.json")
    if (root / "manifest.json.sha256").read_bytes() != f"{digest}  manifest.json\n".encode("ascii"):
        raise ValueError("V6 offline manifest checksum sidecar disagrees")
    raw = (root / "manifest.json").read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError("V6 offline manifest is not canonical")
    if value.get("format") != V6_OFFLINE_FORMAT or value.get("version") != V6_OFFLINE_VERSION:
        raise ValueError("unsupported V6 offline output")
    if set(value) != {
        "files",
        "format",
        "privateDataProhibition",
        "resultSha256",
        "sources",
        "version",
    }:
        raise ValueError("V6 offline manifest fields drifted")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != {"central-q.pt", "public-delta.pt", "result.json"}:
        raise ValueError("V6 offline inventory drifted")
    for name, record in files.items():
        path = root / name
        if not isinstance(record, dict) or record != {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }:
            raise ValueError(f"V6 offline inventory mismatch: {name}")
    result = _strict_export_object(root / "result.json", "V6 offline result")
    if (
        value.get("resultSha256") != sha256_file(root / "result.json")
        or result.get("format") != V6_OFFLINE_FORMAT
        or result.get("version") != V6_OFFLINE_VERSION
        or value.get("sources") != result.get("sources")
        or value.get("privateDataProhibition")
        != result.get("privateDataProhibition")
    ):
        raise ValueError("V6 offline result binding drifted")
    try:
        public = torch.load(root / "public-delta.pt", map_location="cpu", weights_only=True)
        central = torch.load(root / "central-q.pt", map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("V6 offline checkpoint could not be safely loaded") from error
    public_expected = {
        "baseActor",
        "config",
        "containsRawPrivateRows",
        "format",
        "kind",
        "privilegedInputAllowed",
        "publicActorDModel",
        "stateDict",
        "tensorStateSha256",
        "version",
    }
    if (
        not isinstance(public, dict)
        or set(public) != public_expected
        or public.get("format") != V6_OFFLINE_FORMAT
        or public.get("version") != V6_OFFLINE_VERSION
        or public.get("kind") != "public-delta-heads-only"
        or public.get("privilegedInputAllowed") is not False
        or public.get("containsRawPrivateRows") is not False
        or not isinstance(public.get("stateDict"), dict)
        or not public["stateDict"]
        or any("privileged" in str(name).lower() for name in public["stateDict"])
        or public.get("tensorStateSha256") != tensor_state_sha256(public["stateDict"])
        or public.get("baseActor") != result["sources"]["publicActor"]  # type: ignore[index]
    ):
        raise ValueError("public delta checkpoint violated private-data prohibition")
    public_config = public.get("config")
    public_d_model = public.get("publicActorDModel")
    if not isinstance(public_config, dict) or not isinstance(public_d_model, int):
        raise ValueError("public delta checkpoint config is malformed")
    try:
        parsed_public_config = V6PublicDeltaConfig(**public_config)
        public_heads = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.LayerNorm(public_d_model),
                torch.nn.Linear(public_d_model, parsed_public_config.hidden_features),
                torch.nn.GELU(),
                torch.nn.Linear(parsed_public_config.hidden_features, 1),
            )
            for _ in range(BOOTSTRAP_HEADS)
        )
        public_heads.load_state_dict(public["stateDict"], strict=True)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("public delta tensor shapes/config drifted") from error
    central_expected = {
        "centralQContract",
        "config",
        "containsRawPrivateRows",
        "deployExportAllowed",
        "format",
        "kind",
        "stateDict",
        "tensorStateSha256",
        "version",
    }
    if (
        not isinstance(central, dict)
        or set(central) != central_expected
        or central.get("format") != V6_OFFLINE_FORMAT
        or central.get("version") != V6_OFFLINE_VERSION
        or central.get("kind")
        != "centralized-training-only-v-plus-three-advantage-heads"
        or central.get("centralQContract") != CENTRAL_Q_CONTRACT
        or central.get("deployExportAllowed") is not False
        or central.get("containsRawPrivateRows") is not False
        or not isinstance(central.get("stateDict"), dict)
        or not central["stateDict"]
        or central.get("tensorStateSha256") != tensor_state_sha256(central["stateDict"])
    ):
        raise ValueError("central Q checkpoint contract drifted")
    try:
        central_model = V6CentralBootstrapActionQCritic(
            V6CentralBootstrapQConfig(**central["config"])
        )
        central_model.load_state_dict(central["stateDict"], strict=True)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("central Q tensor shapes/config drifted") from error
    return value


def _runtime(device: torch.device, elapsed: float) -> dict[str, object]:
    return {
        "elapsedSeconds": elapsed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cudaAvailable": bool(torch.cuda.is_available()),
        "cudaDeviceName": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _verified_pretrain_receipt(
    pair_root: Path, pair: Mapping[str, object]
) -> dict[str, object]:
    """Bind offline training to the exact V6 pretrain result, not only its pair."""

    manifest_path = pair_root / "manifest.json"
    result_path = pair_root / "result.json"
    manifest = _strict_export_object(manifest_path, "V6 pretrain manifest")
    manifest_sha = _verify_json_sidecar(manifest_path)
    result = _strict_export_object(result_path, "V6 pretrain result")
    result_sha = sha256_file(result_path)
    if (
        manifest.get("format") != V6_PRETRAIN_FORMAT
        or result.get("format") != V6_PRETRAIN_FORMAT
        or manifest.get("resultSha256") != result_sha
        or manifest.get("modelPairId") != pair.get("pairId")
        or result.get("outputModelPair") != pair
        or manifest.get("sources") != result.get("sources")
    ):
        raise ValueError("V6 pretrain result/manifest/model-pair binding drifted")
    config = result.get("config")
    views = result.get("views")
    sources = result.get("sources")
    if not isinstance(config, dict) or not isinstance(views, dict) or not isinstance(sources, dict):
        raise ValueError("V6 pretrain result omitted config, views, or sources")
    fraction = config.get("train_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("V6 pretrain result omitted its train fraction")
    return {
        "manifestSha256": manifest_sha,
        "resultSha256": result_sha,
        "trainFraction": float(fraction),
        "config": config,
        "views": views,
        "sources": sources,
        "progressGates": result.get("progressGates"),
    }


def train_v6_offline_pilot(
    dataset_index: str | Path,
    split_manifest: str | Path,
    pretrain_model_pair: str | Path,
    output_directory: str | Path,
    *,
    public_actor_bundle: str | Path | None = None,
    config: V6OfflineConfig | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Run one immutable 5% or 10% complete-match offline V6 pilot."""

    started = time.monotonic()
    cfg = config or V6OfflineConfig()
    target_device = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    _seed_all(cfg.seed)
    if not public_delta_api_has_no_privileged_input():
        raise RuntimeError("deployable V6 scorer API unexpectedly accepts private input")

    supplied = Path(pretrain_model_pair).resolve(strict=True)
    pair_root = supplied.parent if supplied.is_file() else supplied
    pair = verify_v5_model_pair(supplied)
    pretrain_receipt = _verified_pretrain_receipt(pair_root, pair)
    paired_actor, paired_actor_manifest = load_v5_actor_bundle(pair_root / "actor-bundle")
    pretrained_value, critic_payload = load_v5_critic_checkpoint(pair_root / "critic.pt")
    paired_actor_metadata = paired_actor_manifest.get("metadata")
    critic_metadata = critic_payload.get("metadata")
    if (
        not isinstance(paired_actor_metadata, Mapping)
        or not isinstance(critic_metadata, Mapping)
        or paired_actor_metadata.get("v6PretrainFormat") != V6_PRETRAIN_FORMAT
        or critic_metadata.get("v6PretrainFormat") != V6_PRETRAIN_FORMAT
    ):
        raise ValueError("offline V6 requires a verified V6 pretrain model pair")
    public_actor_path = (
        Path(public_actor_bundle).resolve(strict=True)
        if public_actor_bundle is not None
        else pair_root / "actor-bundle"
    )
    actor, public_actor_manifest = load_v5_actor_bundle(public_actor_path)
    if actor.config != paired_actor.config:
        raise ValueError("public Actor override must have the pretrain Actor architecture")
    del paired_actor

    central = V6CentralBootstrapActionQCritic(V6CentralBootstrapQConfig())
    initialization = _initialize_central_from_pretrained_value(central, pretrained_value)
    scorer = V6PublicDeltaScorer(actor, V6PublicDeltaConfig(freeze_public_backbone=True))
    central = central.to(target_device)
    scorer = scorer.to(target_device)
    action_features = scorer.public_actor.action_features.detach().to(
        device=target_device, dtype=torch.float32
    )

    with load_v6_split_dataset(
        dataset_index, split_manifest, train_fraction=cfg.pilot_fraction
    ) as dataset:
        if (
            paired_actor_metadata.get("splitManifestSha256") != dataset.split_manifest_sha256
            or critic_metadata.get("splitManifestSha256") != dataset.split_manifest_sha256
        ):
            raise ValueError("pretrain model pair and offline corpus split identities disagree")
        pretrain_sources = pretrain_receipt["sources"]
        assert isinstance(pretrain_sources, Mapping)
        if pretrain_sources.get("corpusIdentitySha256") != dataset.corpus_identity_sha256:
            raise ValueError("pretrain result and offline dataset corpus identities disagree")
        q_epochs, q_reports, constant = _train_central_q(
            central,
            action_features,
            dataset,
            dataset.views["train"],
            cfg,
            target_device,
        )
        central_validation = _evaluate_central_q(
            central,
            action_features,
            dataset,
            dataset.views["validation"],
            target_device,
            cfg.validation_batch_size,
            cfg.huber_delta,
            constant,
        )
        distill_epochs, distill_reports = _train_public_delta(
            scorer,
            central,
            action_features,
            dataset,
            dataset.views["train"],
            cfg,
            target_device,
        )
        public_validation = _evaluate_public_delta(
            scorer,
            central,
            action_features,
            dataset,
            dataset.views["validation"],
            target_device,
            cfg.validation_batch_size,
        )
        sources = {
            "datasetIndex": str(dataset.index_root),
            "datasetIndexManifestSha256": dataset.index_manifest_sha256,
            "corpusIdentitySha256": dataset.corpus_identity_sha256,
            "splitManifest": str(Path(split_manifest).resolve()),
            "splitManifestSha256": dataset.split_manifest_sha256,
            "pretrainModelPair": pair,
            "pretrainResult": pretrain_receipt,
            "pretrainActor": v5_actor_bundle_digests(pair_root / "actor-bundle"),
            "publicActor": v5_actor_bundle_digests(public_actor_path),
            "publicActorPath": str(public_actor_path),
            "publicActorUsesPretrainFinal": public_actor_path == pair_root / "actor-bundle",
            "publicActorMetadata": public_actor_manifest.get("metadata"),
            "pretrainCriticSha256": sha256_file(pair_root / "critic.pt"),
            "implementationSha256": {
                "v6OfflineTrain": sha256_file(Path(__file__).resolve()),
                "v6Override": sha256_file(Path(__file__).resolve().with_name("v6_override.py")),
                "v6Pretrain": sha256_file(Path(__file__).resolve().with_name("v6_pretrain.py")),
                "v6Targets": sha256_file(Path(__file__).resolve().with_name("v6_targets.py")),
            },
        }
        result: dict[str, object] = {
            "format": V6_OFFLINE_FORMAT,
            "version": V6_OFFLINE_VERSION,
            "config": cfg.to_dict(),
            "views": {
                "trainPilot": dataset.views["train"].summary(),
                "validationComplete": dataset.views["validation"].summary(),
                "testReserved": dataset.views["test"].summary(),
                "selectionUnit": "complete-five-act-match",
            },
            "target": {
                "contract": V6_MC_RETURN_CONTRACT,
                "centralQContract": CENTRAL_Q_CONTRACT,
                "gamma": 1.0,
                "qSupervision": "logged behavior action only",
                "normalCounterfactualAtAlternativeRowsObserved": False,
            },
            "initialization": initialization,
            "training": {
                "centralVQ": {"epochs": q_epochs, **q_reports},
                "publicDelta": {
                    "epochs": distill_epochs,
                    **distill_reports,
                    "contract": V6_DISTILL_CONTRACT,
                    "observedTargets": ["logged-behavior", "Normal-zero-anchor"],
                    "unobservedLegalActionTreatment": "positive-score-retention-hinge-only",
                },
            },
            "validation": {
                "centralVQ": central_validation,
                "publicDistillation": public_validation,
                "gamePromotionClaim": False,
            },
            "privateDataProhibition": {
                "contract": V6_PRIVATE_PROHIBITION_CONTRACT,
                "publicForwardAcceptsPrivilegedInput": False,
                "publicCheckpointContainsOnlyDeltaHeadTensors": True,
                "rawPrivilegedRowsPublished": False,
                "centralCheckpointTrainingOnly": True,
            },
            "limitations": [
                "A logged one-action corpus does not identify same-state counterfactual action values.",
                "Q(Normal) on alternative-behavior rows is model extrapolation, not a Monte-Carlo label.",
                "This pilot requires separate counterfactual and full-game evaluation before promotion.",
            ],
            "sources": sources,
            "runtime": _runtime(target_device, time.monotonic() - started),
        }
        return _publish_output(Path(output_directory), central, scorer, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--pretrain-model-pair", required=True, type=Path)
    parser.add_argument("--public-actor-bundle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pilot-fraction", type=float, choices=PILOT_FRACTIONS, default=0.10)
    parser.add_argument("--q-epochs", type=int, default=4)
    parser.add_argument("--distill-epochs", type=int, default=4)
    parser.add_argument("--q-batch-size", type=int, default=512)
    parser.add_argument("--distill-batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--alternative-mass-fraction", type=float, default=0.50)
    parser.add_argument(
        "--distill-alternative-mass-fraction", type=float, default=0.50
    )
    parser.add_argument("--seed", type=int, default=860_200_001)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = V6OfflineConfig(
        seed=arguments.seed,
        pilot_fraction=arguments.pilot_fraction,
        q_epochs=arguments.q_epochs,
        distill_epochs=arguments.distill_epochs,
        q_batch_size=arguments.q_batch_size,
        distill_batch_size=arguments.distill_batch_size,
        validation_batch_size=arguments.validation_batch_size,
        alternative_mass_fraction=arguments.alternative_mass_fraction,
        distill_alternative_mass_fraction=arguments.distill_alternative_mass_fraction,
        use_amp=not arguments.no_amp,
    )
    result = train_v6_offline_pilot(
        arguments.dataset_index,
        arguments.split_manifest,
        arguments.pretrain_model_pair,
        arguments.output,
        public_actor_bundle=arguments.public_actor_bundle,
        config=config,
        device=arguments.device,
    )
    print(
        json.dumps(
            {
                "manifestSha256": result["manifestSha256"],
                "outputDirectory": result["outputDirectory"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PILOT_FRACTIONS",
    "V6OfflineConfig",
    "V6_BEHAVIOR_BALANCE_CONTRACT",
    "V6_DISTILL_CONTRACT",
    "V6_OFFLINE_FORMAT",
    "balanced_behavior_player_weights",
    "equal_player_value_weights",
    "observed_delta_distillation_loss",
    "selected_packed_positions",
    "train_v6_offline_pilot",
    "verify_v6_offline_output",
    "weighted_central_v_q_loss",
]
