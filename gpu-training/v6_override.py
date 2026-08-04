from __future__ import annotations

"""Leakage-safe V6 action-Q bootstrap and conservative policy override.

The centralized critic in this module is training-only and may consume the
sealed 512-value privileged state.  The deployable scorer deliberately has no
privileged argument: it reuses a :class:`V5PublicActor` public encoder and
scores only packed legal actions.  ``choose_safe_override`` is implemented in
NumPy as the final, auditable policy gate and therefore remains usable on
machines without PyTorch.
"""

from dataclasses import dataclass
import hashlib
import inspect
import math
from typing import Any, Iterable

import numpy as np


ACTION_COUNT = 236
ACTION_FEATURE_COUNT = 22
PRIVILEGED_STATE_SIZE = 512
BOOTSTRAP_HEADS = 3
BOOTSTRAP_MEMBERSHIP_CONTRACT = "dalmuti-v6-match-key-bootstrap-membership-v1"
SAFE_OVERRIDE_CONTRACT = "dalmuti-v6-public-lcb-safe-override-v1"
NORMAL_PARITY_CONTRACT = "dalmuti-v6-exact-normal-parity-v1"
CENTRAL_Q_CONTRACT = "dalmuti-v6-stopgrad-value-plus-action-advantage-v1"


try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    from v5_model import V5PublicActor, pack_legal_actions
    from v5_public import V5ActorPublicBatch

    TORCH_AVAILABLE = True
except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
    if error.name != "torch":
        raise
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _finite_float(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" at least {minimum}" if minimum is not None else ""
        raise ValueError(f"{label} must be finite and{suffix}")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _canonical_match_key(value: object) -> bytes:
    """Return an unambiguous, process-independent match-key encoding."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        prefix = b"b:"
    elif isinstance(value, str):
        payload = value.encode("utf-8")
        prefix = b"s:"
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        payload = str(int(value)).encode("ascii")
        prefix = b"i:"
    else:
        raise TypeError("match keys must be strings, bytes, or integers")
    if not payload:
        raise ValueError("match keys must not be empty")
    return prefix + len(payload).to_bytes(8, "big") + payload


def deterministic_bootstrap_membership(
    match_keys: Iterable[object],
    *,
    seed: int = 0,
    head_count: int = BOOTSTRAP_HEADS,
    inclusion_probability: float = 1.0 - math.exp(-1.0),
) -> np.ndarray:
    """Assign whole matches to deterministic Bernoulli bootstrap heads.

    A match key is hashed independently for each head.  Every decision from a
    match must be passed the same key and therefore receives identical
    membership.  The default inclusion probability is the asymptotic fraction
    of unique samples in an ordinary size-N bootstrap (``1-exp(-1)``).
    """

    heads = _positive_integer(head_count, "head_count")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed64 = int(seed)
    if seed64 < 0 or seed64 >= 1 << 64:
        raise ValueError("seed must be in [0, 2**64)")
    probability = _finite_float(inclusion_probability, "inclusion_probability")
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("inclusion_probability must be in (0, 1)")

    encoded = tuple(_canonical_match_key(value) for value in match_keys)
    result = np.empty((len(encoded), heads), dtype=np.bool_)
    domain = BOOTSTRAP_MEMBERSHIP_CONTRACT.encode("ascii") + b"\x00"
    seed_bytes = seed64.to_bytes(8, "big")
    threshold = int(probability * (1 << 64))
    for row, key in enumerate(encoded):
        for head in range(heads):
            digest = hashlib.sha256(
                domain + seed_bytes + head.to_bytes(4, "big") + key
            ).digest()
            result[row, head] = int.from_bytes(digest[:8], "big") < threshold
    return result


@dataclass(frozen=True)
class SafeOverrideDecision:
    action_id: int
    normal_action_id: int
    overridden: bool
    best_alternative_id: int | None
    best_lcb: float | None
    action_ids: np.ndarray
    legal_mask: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray
    delta_lcb: np.ndarray

    def __post_init__(self) -> None:
        for value in (
            self.action_ids,
            self.legal_mask,
            self.delta_mean,
            self.delta_std,
            self.delta_lcb,
        ):
            value.setflags(write=False)


def _resolve_numpy_normal(
    action_ids: np.ndarray,
    legal_mask: np.ndarray,
    normal_action: int,
) -> tuple[int, int]:
    legal_positions = np.flatnonzero(legal_mask)
    if legal_positions.size == 0:
        raise ValueError("every decision requires at least one legal action")
    if legal_positions.size == 1:
        position = int(legal_positions[0])
        resolved = int(action_ids[position])
        if normal_action not in (-1, resolved):
            raise ValueError("a forced Normal action must be -1 or the sole legal action")
        return resolved, position
    matches = np.flatnonzero(legal_mask & (action_ids == normal_action))
    if matches.size != 1:
        raise ValueError("Normal must identify exactly one legal action")
    return int(normal_action), int(matches[0])


def choose_safe_override(
    *,
    action_ids: object,
    legal_mask: object,
    head_scores: object,
    normal_action: int,
    beta: float = 1.0,
    threshold: float = 0.0,
) -> SafeOverrideDecision:
    """Choose a public-only alternative when its bootstrap LCB is positive.

    ``head_scores`` has shape ``[packed_actions, 3]``.  Each head is converted
    to a Normal-relative delta before its population standard deviation and
    lower confidence bound are calculated.  Illegal actions never compete,
    forced rows remain Normal, the threshold comparison is strict, and equal
    LCBs are resolved by the lowest catalogue action id.
    """

    ids = np.asarray(action_ids)
    legal = np.asarray(legal_mask)
    scores = np.asarray(head_scores)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or np.issubdtype(ids.dtype, np.bool_)
        or ids.size < 1
    ):
        raise ValueError("action_ids must be a non-empty integer vector")
    if legal.shape != ids.shape or legal.dtype != np.dtype(np.bool_):
        raise ValueError("legal_mask must be canonical bool with action_ids shape")
    legal_ids = ids[legal]
    if (
        np.any((legal_ids < 0) | (legal_ids >= ACTION_COUNT))
        or np.unique(legal_ids).size != legal_ids.size
    ):
        raise ValueError("legal action_ids must be unique fixed-catalogue indices")
    if (
        scores.shape != (ids.size, BOOTSTRAP_HEADS)
        or not np.issubdtype(scores.dtype, np.floating)
    ):
        raise ValueError("head_scores must be floating [packed_actions, 3]")
    if not np.isfinite(scores[legal]).all():
        raise ValueError("legal head scores must be finite")
    resolved_normal, normal_position = _resolve_numpy_normal(
        ids.astype(np.int64, copy=False), legal, int(normal_action)
    )
    beta_value = _finite_float(beta, "beta", minimum=0.0)
    if isinstance(threshold, bool):
        raise ValueError("threshold must be a number or positive infinity")
    threshold_value = float(threshold)
    if math.isnan(threshold_value) or threshold_value == -math.inf:
        raise ValueError("threshold must be finite or positive infinity")

    relative = scores.astype(np.float64, copy=False) - scores[
        normal_position
    ].astype(np.float64, copy=False)[None, :]
    means = relative.mean(axis=1)
    standard_deviations = relative.std(axis=1, ddof=0)
    lcbs = means - beta_value * standard_deviations
    # Make the Normal reference bit-exact, independent of reduction noise.
    means[normal_position] = 0.0
    standard_deviations[normal_position] = 0.0
    lcbs[normal_position] = 0.0

    legal_positions = np.flatnonzero(legal)
    alternative_positions = legal_positions[legal_positions != normal_position]
    if alternative_positions.size == 0:
        return SafeOverrideDecision(
            resolved_normal,
            resolved_normal,
            False,
            None,
            None,
            ids.astype(np.int64, copy=True),
            legal.copy(),
            means,
            standard_deviations,
            lcbs,
        )

    best_value = float(np.max(lcbs[alternative_positions]))
    tied = alternative_positions[lcbs[alternative_positions] == best_value]
    best_position = int(tied[np.argmin(ids[tied])])
    best_action = int(ids[best_position])
    override = best_value > threshold_value
    return SafeOverrideDecision(
        best_action if override else resolved_normal,
        resolved_normal,
        override,
        best_action,
        best_value,
        ids.astype(np.int64, copy=True),
        legal.copy(),
        means,
        standard_deviations,
        lcbs,
    )


def assert_exact_normal_parity(
    *,
    action_ids: object,
    legal_masks: object,
    head_scores: object,
    normal_actions: object,
    beta: float = 1.0,
    threshold: float = 0.0,
) -> str:
    """Assert a complete batch chooses Normal and return a canonical digest."""

    ids = np.asarray(action_ids)
    masks = np.asarray(legal_masks)
    scores = np.asarray(head_scores)
    normals = np.asarray(normal_actions)
    if ids.ndim == 1:
        if masks.ndim != 2 or scores.ndim != 3:
            raise ValueError("batched masks/scores are required")
        batch_ids = np.broadcast_to(ids, masks.shape)
    elif ids.ndim == 2:
        batch_ids = ids
    else:
        raise ValueError("action_ids must be [actions] or [batch,actions]")
    if (
        masks.dtype != np.dtype(np.bool_)
        or masks.ndim != 2
        or batch_ids.shape != masks.shape
        or scores.shape != (*masks.shape, BOOTSTRAP_HEADS)
        or normals.shape != (masks.shape[0],)
    ):
        raise ValueError("parity arrays have incompatible shapes or dtypes")

    selected = np.empty(masks.shape[0], dtype=np.int64)
    resolved = np.empty_like(selected)
    for row in range(masks.shape[0]):
        decision = choose_safe_override(
            action_ids=batch_ids[row],
            legal_mask=masks[row],
            head_scores=scores[row],
            normal_action=int(normals[row]),
            beta=beta,
            threshold=threshold,
        )
        selected[row] = decision.action_id
        resolved[row] = decision.normal_action_id
    if not np.array_equal(selected, resolved):
        mismatches = np.flatnonzero(selected != resolved)
        first = int(mismatches[0])
        raise AssertionError(
            "safe override diverged from Normal on "
            f"{mismatches.size} rows; first row {first}: "
            f"selected={selected[first]}, normal={resolved[first]}"
        )

    digest = hashlib.sha256()
    digest.update(NORMAL_PARITY_CONTRACT.encode("ascii") + b"\x00")
    digest.update(np.asarray(masks.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(batch_ids, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(masks, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(resolved, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(selected, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(scores, dtype="<f8").tobytes())
    digest.update(np.asarray([float(beta), float(threshold)], dtype="<f8").tobytes())
    return digest.hexdigest()


def assert_zero_head_exact_normal_parity(
    *,
    action_ids: object,
    legal_masks: object,
    head_scores: object,
    normal_actions: object,
    beta: float = 1.0,
    threshold: float = 0.0,
) -> str:
    """Verify reset heads are exactly zero and seal their Normal parity."""

    scores = np.asarray(head_scores)
    if not np.issubdtype(scores.dtype, np.floating) or not np.isfinite(scores).all():
        raise ValueError("reset head_scores must be finite floating-point")
    if np.count_nonzero(scores) != 0:
        raise AssertionError("reset public delta heads are not exactly zero")
    return assert_exact_normal_parity(
        action_ids=action_ids,
        legal_masks=legal_masks,
        head_scores=scores,
        normal_actions=normal_actions,
        beta=beta,
        threshold=threshold,
    )


@dataclass(frozen=True)
class V6PublicDeltaConfig:
    bootstrap_heads: int = BOOTSTRAP_HEADS
    hidden_features: int = 288
    freeze_public_backbone: bool = True

    def __post_init__(self) -> None:
        if self.bootstrap_heads != BOOTSTRAP_HEADS:
            raise ValueError("V6 requires exactly three bootstrap heads")
        _positive_integer(self.hidden_features, "hidden_features")
        if not isinstance(self.freeze_public_backbone, bool):
            raise ValueError("freeze_public_backbone must be bool")


@dataclass(frozen=True)
class V6CentralBootstrapQConfig:
    privileged_features: int = PRIVILEGED_STATE_SIZE
    action_features: int = ACTION_FEATURE_COUNT
    d_model: int = 512
    action_hidden: int = 256
    hidden_layers: int = 3
    player_count_embedding: int = 32
    bootstrap_heads: int = BOOTSTRAP_HEADS
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "privileged_features",
            "action_features",
            "d_model",
            "action_hidden",
            "hidden_layers",
            "player_count_embedding",
        ):
            _positive_integer(getattr(self, name), name)
        if self.privileged_features != PRIVILEGED_STATE_SIZE:
            raise ValueError("V6 requires the sealed 512-value privileged state")
        if self.action_features != ACTION_FEATURE_COUNT:
            raise ValueError("V6 requires the sealed 22-value action features")
        if self.bootstrap_heads != BOOTSTRAP_HEADS:
            raise ValueError("V6 requires exactly three bootstrap heads")
        dropout = _finite_float(self.dropout, "dropout", minimum=0.0)
        if dropout >= 1.0:
            raise ValueError("dropout must be in [0,1)")


if TORCH_AVAILABLE:

    @dataclass(frozen=True)
    class V6PublicDeltaOutput:
        head_scores: torch.Tensor
        action_indices: torch.Tensor
        action_mask: torch.Tensor


    @dataclass(frozen=True)
    class V6CentralBootstrapQOutput:
        values: torch.Tensor
        q_values: torch.Tensor
        action_mask: torch.Tensor


    class V6PublicDeltaScorer(nn.Module):
        """Three public-only Normal-relative score heads.

        The scorer deliberately exposes only ``V5ActorPublicBatch`` and legal
        action packing in ``forward``.  It reuses the V5 public Transformer,
        action feature encoder, and action cross-attention stack; privileged
        state cannot enter this inference path.
        """

        def __init__(
            self,
            public_actor: V5PublicActor,
            config: V6PublicDeltaConfig | None = None,
        ) -> None:
            super().__init__()
            if type(public_actor) is not V5PublicActor:
                raise TypeError("public_actor must be exactly V5PublicActor")
            self.config = config or V6PublicDeltaConfig(
                hidden_features=public_actor.config.d_model
            )
            self.public_actor = public_actor
            d_model = public_actor.config.d_model
            self.delta_heads = nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, self.config.hidden_features),
                    nn.GELU(),
                    nn.Linear(self.config.hidden_features, 1),
                )
                for _ in range(BOOTSTRAP_HEADS)
            )
            if self.config.freeze_public_backbone:
                for parameter in self.public_actor.parameters():
                    parameter.requires_grad_(False)
            self.reset_delta_heads()

        def reset_delta_heads(self) -> None:
            """Zero final layers so the safe gate is exactly Normal."""

            for head in self.delta_heads:
                output = head[-1]
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)

        def forward(
            self,
            public_batch: V5ActorPublicBatch,
            legal_action_indices: torch.Tensor | None = None,
            legal_action_mask: torch.Tensor | None = None,
        ) -> V6PublicDeltaOutput:
            if type(public_batch) is not V5ActorPublicBatch:
                raise TypeError("public_batch must be exactly V5ActorPublicBatch")
            if legal_action_indices is None or legal_action_mask is None:
                if legal_action_indices is not None or legal_action_mask is not None:
                    raise ValueError("packed action indices and mask must be supplied together")
                legal_action_indices, legal_action_mask = pack_legal_actions(
                    public_batch.legal_mask
                )
            if (
                legal_action_indices.dtype != torch.long
                or legal_action_mask.dtype != torch.bool
                or legal_action_indices.ndim != 2
                or legal_action_mask.shape != legal_action_indices.shape
                or legal_action_indices.shape[0] != public_batch.global_codes.shape[0]
            ):
                raise ValueError("packed actions must be int64/bool [batch,actions]")
            selected = torch.zeros_like(public_batch.legal_mask)
            rows, positions = legal_action_mask.nonzero(as_tuple=True)
            chosen = legal_action_indices[rows, positions]
            if (
                chosen.numel() == 0
                or int(chosen.min()) < 0
                or int(chosen.max()) >= ACTION_COUNT
            ):
                raise ValueError("packed legal action escaped the catalogue")
            selected[rows, chosen] = True
            if (
                not torch.equal(selected, public_batch.legal_mask)
                or not torch.equal(
                    legal_action_mask.sum(dim=-1),
                    selected.sum(dim=-1),
                )
            ):
                raise ValueError("packed actions must exactly represent the public legal mask")

            public_core, core_mask = self.public_actor.encode_public_core(public_batch)
            safe_indices = legal_action_indices.clamp(0, ACTION_COUNT - 1)
            action_features = self.public_actor.action_features.to(
                dtype=public_core.dtype
            ).index_select(0, safe_indices.reshape(-1)).reshape(
                *safe_indices.shape, ACTION_FEATURE_COUNT
            )
            action_hidden = self.public_actor.action_query_encoder(action_features)
            for layer in self.public_actor.action_cross_attention:
                action_hidden = layer(action_hidden, public_core, core_mask)
            scores = torch.stack(
                [head(action_hidden).squeeze(-1) for head in self.delta_heads],
                dim=-1,
            )
            scores = scores.masked_fill(~legal_action_mask.unsqueeze(-1), 0.0)
            return V6PublicDeltaOutput(scores, legal_action_indices, legal_action_mask)


    class V6CentralBootstrapActionQCritic(nn.Module):
        """Training-only centralized V plus three packed legal-action Q heads."""

        def __init__(self, config: V6CentralBootstrapQConfig | None = None) -> None:
            super().__init__()
            self.config = config or V6CentralBootstrapQConfig()
            cfg = self.config
            self.player_count_embedding = nn.Embedding(11, cfg.player_count_embedding)
            state_input = cfg.privileged_features + cfg.player_count_embedding
            state_layers: list[nn.Module] = [
                nn.LayerNorm(state_input),
                nn.Linear(state_input, cfg.d_model),
                nn.GELU(),
            ]
            for _ in range(cfg.hidden_layers - 1):
                state_layers.extend(
                    (
                        nn.LayerNorm(cfg.d_model),
                        nn.Linear(cfg.d_model, cfg.d_model),
                        nn.GELU(),
                        nn.Dropout(cfg.dropout),
                    )
                )
            state_layers.append(nn.LayerNorm(cfg.d_model))
            self.state_encoder = nn.Sequential(*state_layers)
            self.value_output = nn.Linear(cfg.d_model, 1)
            self.state_to_action = nn.Linear(cfg.d_model, cfg.action_hidden)
            self.action_encoder = nn.Sequential(
                nn.LayerNorm(cfg.action_features),
                nn.Linear(cfg.action_features, cfg.action_hidden),
                nn.GELU(),
                nn.Linear(cfg.action_hidden, cfg.action_hidden),
            )
            self.q_heads = nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(cfg.action_hidden),
                    nn.Linear(cfg.action_hidden, cfg.action_hidden),
                    nn.GELU(),
                    nn.Linear(cfg.action_hidden, 1),
                )
                for _ in range(BOOTSTRAP_HEADS)
            )
            nn.init.zeros_(self.value_output.weight)
            nn.init.zeros_(self.value_output.bias)
            for head in self.q_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

        def forward(
            self,
            privileged_states: torch.Tensor,
            packed_action_features: torch.Tensor,
            packed_action_mask: torch.Tensor,
            player_counts: torch.Tensor,
        ) -> V6CentralBootstrapQOutput:
            cfg = self.config
            if (
                privileged_states.ndim != 2
                or privileged_states.shape[1] != PRIVILEGED_STATE_SIZE
                or not privileged_states.dtype.is_floating_point
            ):
                raise ValueError("privileged_states must be float [batch,512]")
            batch_size = privileged_states.shape[0]
            if (
                packed_action_features.ndim != 3
                or packed_action_features.shape[0] != batch_size
                or packed_action_features.shape[2] != ACTION_FEATURE_COUNT
                or not packed_action_features.dtype.is_floating_point
            ):
                raise ValueError("packed_action_features must be float [batch,legal,22]")
            if (
                packed_action_mask.dtype != torch.bool
                or packed_action_mask.shape != packed_action_features.shape[:2]
                or not packed_action_mask.any(dim=-1).all()
            ):
                raise ValueError("packed_action_mask must select at least one action per row")
            if (
                player_counts.dtype != torch.long
                or player_counts.shape != (batch_size,)
                or ((player_counts < 4) | (player_counts > 10)).any()
            ):
                raise ValueError("player_counts must be int64 p4..p10")
            devices = {
                privileged_states.device,
                packed_action_features.device,
                packed_action_mask.device,
                player_counts.device,
            }
            if len(devices) != 1 or next(self.parameters()).device not in devices:
                raise ValueError("all critic inputs and parameters must share a device")
            if not torch.isfinite(privileged_states).all():
                raise ValueError("privileged_states must be finite")
            if not torch.isfinite(packed_action_features[packed_action_mask]).all():
                raise ValueError("legal packed action features must be finite")

            state = self.state_encoder(
                torch.cat(
                    (privileged_states, self.player_count_embedding(player_counts)),
                    dim=-1,
                )
            )
            values = self.value_output(state).squeeze(-1)
            # The pretrained state-value path is the baseline.  Logged-action
            # Q supervision may only train the action-specific advantage path;
            # otherwise a noisy one-action corpus can damage the stronger V
            # representation shared by every action.
            detached_state = state.detach()
            action = self.action_encoder(packed_action_features)
            joint = F.gelu(
                action + self.state_to_action(detached_state).unsqueeze(1)
            )
            advantages = torch.stack(
                [head(joint).squeeze(-1) for head in self.q_heads], dim=-1
            )
            q_values = values.detach().unsqueeze(1).unsqueeze(2) + advantages
            masked_value = max(-1.0e9, torch.finfo(q_values.dtype).min / 2.0)
            q_values = q_values.masked_fill(
                ~packed_action_mask.unsqueeze(-1), masked_value
            )
            return V6CentralBootstrapQOutput(values, q_values, packed_action_mask)


    def bootstrap_action_q_huber_loss(
        output: V6CentralBootstrapQOutput,
        selected_action_positions: torch.Tensor,
        q_targets: torch.Tensor,
        bootstrap_membership: torch.Tensor,
        *,
        delta: float = 1.0,
    ) -> torch.Tensor:
        """Huber loss for logged actions and whole-match bootstrap heads.

        The offline corpus contains a Monte Carlo target only for the action
        that was actually taken.  It is invalid to copy that target onto every
        other legal action, so this contract requires one packed action
        position and one scalar target per row.
        """

        if (
            selected_action_positions.dtype != torch.long
            or selected_action_positions.shape != (output.action_mask.shape[0],)
        ):
            raise ValueError("selected_action_positions must be int64 [batch]")
        if (
            q_targets.shape != (output.action_mask.shape[0],)
            or not q_targets.dtype.is_floating_point
        ):
            raise ValueError("q_targets must be float [batch]")
        if (
            bootstrap_membership.dtype != torch.bool
            or bootstrap_membership.shape
            != (output.action_mask.shape[0], BOOTSTRAP_HEADS)
        ):
            raise ValueError("bootstrap_membership must be bool [batch,3]")
        if (
            selected_action_positions.device != output.q_values.device
            or q_targets.device != output.q_values.device
            or bootstrap_membership.device != output.q_values.device
        ):
            raise ValueError("Q loss tensors must share a device")
        if (
            (selected_action_positions < 0).any()
            or (selected_action_positions >= output.action_mask.shape[1]).any()
        ):
            raise ValueError("selected action position escaped the packed actions")
        selected_is_legal = output.action_mask.gather(
            1, selected_action_positions.unsqueeze(1)
        ).squeeze(1)
        if not selected_is_legal.all():
            raise ValueError("selected action position must be legal")
        if not torch.isfinite(q_targets).all():
            raise ValueError("q_targets must be finite")
        if not bootstrap_membership.any():
            raise ValueError("bootstrap batch selected no Q targets")
        delta_value = _finite_float(delta, "delta", minimum=0.0)
        if delta_value <= 0.0:
            raise ValueError("delta must be positive")

        batch_indices = torch.arange(
            output.action_mask.shape[0], device=output.q_values.device
        )
        chosen_q = output.q_values[batch_indices, selected_action_positions]
        targets = q_targets.unsqueeze(-1).expand_as(chosen_q)
        return F.huber_loss(
            chosen_q[bootstrap_membership],
            targets[bootstrap_membership],
            delta=delta_value,
            reduction="mean",
        )


else:

    @dataclass(frozen=True)
    class V6PublicDeltaOutput:
        head_scores: Any
        action_indices: Any
        action_mask: Any


    @dataclass(frozen=True)
    class V6CentralBootstrapQOutput:
        values: Any
        q_values: Any
        action_mask: Any


    class V6PublicDeltaScorer:  # pragma: no cover - exercised on GPU host
        def __init__(
            self,
            public_actor: Any,
            config: V6PublicDeltaConfig | None = None,
        ) -> None:
            raise RuntimeError("V6PublicDeltaScorer requires PyTorch")

        def forward(
            self,
            public_batch: Any,
            legal_action_indices: Any | None = None,
            legal_action_mask: Any | None = None,
        ) -> V6PublicDeltaOutput:
            raise RuntimeError("V6PublicDeltaScorer requires PyTorch")


    class V6CentralBootstrapActionQCritic:  # pragma: no cover
        def __init__(self, config: V6CentralBootstrapQConfig | None = None) -> None:
            raise RuntimeError("V6CentralBootstrapActionQCritic requires PyTorch")


    def bootstrap_action_q_huber_loss(*args: object, **kwargs: object) -> Any:
        raise RuntimeError("bootstrap_action_q_huber_loss requires PyTorch")


def public_delta_api_has_no_privileged_input() -> bool:
    """Audit the deployable forward signature for accidental private inputs."""

    names = tuple(inspect.signature(V6PublicDeltaScorer.forward).parameters)
    forbidden = ("privileged", "private", "opponent_hand", "hidden_hand")
    return not any(token in name.lower() for name in names for token in forbidden)


__all__ = [
    "ACTION_COUNT",
    "ACTION_FEATURE_COUNT",
    "BOOTSTRAP_HEADS",
    "BOOTSTRAP_MEMBERSHIP_CONTRACT",
    "CENTRAL_Q_CONTRACT",
    "NORMAL_PARITY_CONTRACT",
    "PRIVILEGED_STATE_SIZE",
    "SAFE_OVERRIDE_CONTRACT",
    "SafeOverrideDecision",
    "TORCH_AVAILABLE",
    "V6CentralBootstrapActionQCritic",
    "V6CentralBootstrapQConfig",
    "V6CentralBootstrapQOutput",
    "V6PublicDeltaConfig",
    "V6PublicDeltaOutput",
    "V6PublicDeltaScorer",
    "assert_exact_normal_parity",
    "assert_zero_head_exact_normal_parity",
    "bootstrap_action_q_huber_loss",
    "choose_safe_override",
    "deterministic_bootstrap_membership",
    "public_delta_api_has_no_privileged_input",
]
