from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from v3_action_conditioned import V3_ACTION_FEATURE_COUNT, V3_ACTION_FEATURES
from v5_contract import (
    V5_ACTION_COUNT,
    V5_DECK_COUNTS,
    V5_GLOBAL_FIELDS,
    V5_HISTORY_FIELDS,
    V5_MAX_HISTORY,
    V5_MAX_OPPONENTS,
    V5_MAX_PLAYERS,
    V5_PLAYER_FIELDS,
    V5_PUBLIC_SCHEMA_VERSION,
    V5_RANK_COUNT,
    V5_TABLE_FIELDS,
)
from v5_public import V5ActorPublicBatch


V5_ACTION_FEATURE_COUNT = V3_ACTION_FEATURE_COUNT
V5_MASKED_LOGIT = -1.0e9
V5_POLICY_NUMERICS_CONTRACT_VERSION = 1
V5_POLICY_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def canonical_v5_policy_numerics_contract() -> dict[str, object]:
    fields: dict[str, object] = {
        "contract": "dalmuti-v5-deterministic-fp32-policy-numerics",
        "version": V5_POLICY_NUMERICS_CONTRACT_VERSION,
        "deterministicAlgorithms": True,
        "mhaFastPath": False,
        "flashSdp": False,
        "memoryEfficientSdp": False,
        "mathSdp": True,
        "cudnnSdp": False,
        "cudaMatmulTf32": False,
        "cudnnTf32": False,
        "cudnnDeterministic": True,
        "cudnnBenchmark": False,
        "requiredCudaCublasWorkspaceConfig": V5_POLICY_CUBLAS_WORKSPACE_CONFIG,
    }
    digest = hashlib.sha256(
        json.dumps(
            fields,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {**fields, "contractSha256": digest}


V5_POLICY_NUMERICS_SHA256 = str(
    canonical_v5_policy_numerics_contract()["contractSha256"]
)


def validate_v5_policy_numerics_contract(value: object) -> dict[str, object]:
    expected = canonical_v5_policy_numerics_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("V5 policy numerics contract is missing or non-canonical")
    return expected


def configure_v5_policy_numerics(
    device: str | torch.device,
) -> dict[str, object]:
    """Apply the sealed CPU/CUDA policy math controls and verify them."""

    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("V5 policy numerics supports only CPU or CUDA")
    cuda = target.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA policy numerics requested but CUDA is unavailable")
    mha = getattr(torch.backends, "mha", None)
    cuda_backends = getattr(torch.backends, "cuda", None)
    cudnn = getattr(torch.backends, "cudnn", None)
    required_mha = ("set_fastpath_enabled", "get_fastpath_enabled")
    required_cuda = (
        "enable_flash_sdp",
        "flash_sdp_enabled",
        "enable_mem_efficient_sdp",
        "mem_efficient_sdp_enabled",
        "enable_math_sdp",
        "math_sdp_enabled",
        "enable_cudnn_sdp",
        "cudnn_sdp_enabled",
    )
    if mha is None or any(not hasattr(mha, name) for name in required_mha):
        raise RuntimeError("PyTorch lacks the required MHA fast-path controls")
    if cuda_backends is None or any(
        not hasattr(cuda_backends, name) for name in required_cuda
    ):
        raise RuntimeError("PyTorch lacks the required SDP backend controls")
    if cudnn is None:
        raise RuntimeError("PyTorch lacks the required cuDNN backend controls")
    if cuda:
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing not in (None, V5_POLICY_CUBLAS_WORKSPACE_CONFIG):
            raise ValueError(
                "V5 CUDA policy numerics requires "
                f"CUBLAS_WORKSPACE_CONFIG={V5_POLICY_CUBLAS_WORKSPACE_CONFIG}"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = V5_POLICY_CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True)
    mha.set_fastpath_enabled(False)
    cuda_backends.enable_flash_sdp(False)
    cuda_backends.enable_mem_efficient_sdp(False)
    cuda_backends.enable_math_sdp(True)
    cuda_backends.enable_cudnn_sdp(False)
    cuda_backends.matmul.allow_tf32 = False
    cudnn.allow_tf32 = False
    cudnn.deterministic = True
    cudnn.benchmark = False
    if (
        not torch.are_deterministic_algorithms_enabled()
        or bool(mha.get_fastpath_enabled())
        or bool(cuda_backends.flash_sdp_enabled())
        or bool(cuda_backends.mem_efficient_sdp_enabled())
        or not bool(cuda_backends.math_sdp_enabled())
        or bool(cuda_backends.cudnn_sdp_enabled())
        or bool(cuda_backends.matmul.allow_tf32)
        or bool(cudnn.allow_tf32)
        or not bool(cudnn.deterministic)
        or bool(cudnn.benchmark)
        or (
            cuda
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != V5_POLICY_CUBLAS_WORKSPACE_CONFIG
        )
    ):
        raise RuntimeError("V5 policy numerics controls did not settle exactly")
    return canonical_v5_policy_numerics_contract()


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _dropout_probability(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result >= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1)")
    return result


def _prior_probability(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.5 or result >= 1.0:
        raise ValueError("normal_prior_probability must be finite and in (0.5, 1)")
    return result


def _masked_value(dtype: torch.dtype) -> float:
    return max(V5_MASKED_LOGIT, torch.finfo(dtype).min / 2.0)


def _prior_accumulation_dtype(residual: torch.Tensor) -> torch.dtype:
    # AMP may produce fp16/bfloat16 residuals while torch.log promotes the
    # categorical Normal-prior margin.  Preserve the small prior/logit head in
    # FP32 instead of quantizing it or relying on dtype-sensitive scatter.
    return (
        torch.float32
        if residual.dtype in (torch.float16, torch.bfloat16)
        else residual.dtype
    )


def _validate_legal_masks(legal_masks: torch.Tensor, batch_size: int) -> None:
    if legal_masks.dtype != torch.bool or legal_masks.shape != (
        batch_size,
        V5_ACTION_COUNT,
    ):
        raise ValueError("legal masks must be bool [batch, 236]")
    if not torch.jit.is_tracing() and not legal_masks.any(dim=-1).all():
        raise ValueError("every observation requires at least one legal action")


def _mask_illegal_logits(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != V5_ACTION_COUNT:
        raise ValueError("logits must have shape [batch, 236]")
    _validate_legal_masks(legal_masks, logits.shape[0])
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must use a floating-point dtype")
    return logits.masked_fill(~legal_masks, _masked_value(logits.dtype))


def _validate_normal_actions(
    normal_actions: torch.Tensor,
    batch_size: int,
) -> None:
    if normal_actions.dtype != torch.long or normal_actions.shape != (batch_size,):
        raise ValueError("normal actions must be int64 [batch]")


def resolve_normal_actions(
    legal_masks: torch.Tensor,
    normal_actions: torch.Tensor,
) -> torch.Tensor:
    """Resolve Normal, allowing -1 only when exactly one action is legal."""

    batch_size = legal_masks.shape[0]
    _validate_legal_masks(legal_masks, batch_size)
    _validate_normal_actions(normal_actions, batch_size)
    if normal_actions.device != legal_masks.device:
        raise ValueError("normal actions and legal masks must share a device")
    legal_counts = legal_masks.sum(dim=-1)
    in_range = (normal_actions >= 0) & (normal_actions < V5_ACTION_COUNT)
    safe_indices = normal_actions.clamp(0, V5_ACTION_COUNT - 1)
    supplied_is_legal = in_range & legal_masks.gather(
        1, safe_indices.unsqueeze(1)
    ).squeeze(1)
    forced = legal_counts == 1
    invalid_nonforced = ~forced & ~supplied_is_legal
    invalid_forced = forced & ~(supplied_is_legal | (normal_actions == -1))
    if not torch.jit.is_tracing() and (invalid_nonforced | invalid_forced).any():
        raise ValueError("normal action must be legal; only forced rows may use -1")
    forced_actions = legal_masks.to(torch.int64).argmax(dim=-1)
    return torch.where(forced, forced_actions, safe_indices)


def normal_prior_logits(
    legal_masks: torch.Tensor,
    normal_actions: torch.Tensor,
    probability: float = 0.9,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the exact Normal categorical prior.

    For L > 1 legal actions, Normal receives
    ``log(p * (L - 1) / (1 - p))`` while every alternative receives zero.
    At p=.9 the required margin is exactly ``log(9 * (L - 1))``.  A forced
    row uses zero on its sole legal action and remains deterministic.
    """

    prior_probability = _prior_probability(probability)
    resolved = resolve_normal_actions(legal_masks, normal_actions)
    legal_counts = legal_masks.sum(dim=-1)
    odds = prior_probability / (1.0 - prior_probability)
    margins = torch.where(
        legal_counts > 1,
        torch.log((legal_counts - 1).to(dtype=dtype) * odds),
        torch.zeros_like(legal_counts, dtype=dtype),
    )
    logits = torch.zeros(
        (legal_masks.shape[0], V5_ACTION_COUNT),
        dtype=dtype,
        device=legal_masks.device,
    )
    logits.scatter_(1, resolved.unsqueeze(1), margins.unsqueeze(1))
    return _mask_illegal_logits(logits, legal_masks), resolved


def pack_legal_actions(
    legal_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack variable legal sets into [batch, max_legal] tensors."""

    if legal_masks.ndim != 2:
        raise ValueError("legal masks must have rank two")
    _validate_legal_masks(legal_masks, legal_masks.shape[0])
    counts = legal_masks.sum(dim=-1)
    maximum = int(counts.max().item())
    packed_indices = torch.zeros(
        (legal_masks.shape[0], maximum),
        dtype=torch.long,
        device=legal_masks.device,
    )
    packed_mask = (
        torch.arange(maximum, device=legal_masks.device).unsqueeze(0)
        < counts.unsqueeze(1)
    )
    batch_indices, action_indices = legal_masks.nonzero(as_tuple=True)
    positions = legal_masks.to(torch.int64).cumsum(dim=-1)[
        batch_indices, action_indices
    ] - 1
    packed_indices[batch_indices, positions] = action_indices
    return packed_indices, packed_mask


@dataclass(frozen=True)
class V5ActorConfig:
    history_latents: int = 8
    d_model: int = 288
    core_layers: int = 7
    action_layers: int = 2
    heads: int = 8
    feedforward: int = 768
    dropout: float = 0.0
    normal_prior_probability: float = 0.9
    observation_schema_version: int = V5_PUBLIC_SCHEMA_VERSION
    action_catalogue_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "history_latents",
            "d_model",
            "core_layers",
            "action_layers",
            "heads",
            "feedforward",
            "observation_schema_version",
            "action_catalogue_version",
        ):
            _positive_integer(getattr(self, name), name)
        _dropout_probability(self.dropout, "dropout")
        _prior_probability(self.normal_prior_probability)
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")
        if self.action_layers != 2:
            raise ValueError("the V5 action head requires exactly two cross-attention layers")
        if self.normal_prior_probability != 0.9:
            raise ValueError("the V5 actor requires the canonical 90% Normal prior")
        if self.observation_schema_version != V5_PUBLIC_SCHEMA_VERSION:
            raise ValueError("actor must use the canonical V5 public schema")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V5CriticConfig:
    privileged_features: int = 512
    d_model: int = 512
    hidden_layers: int = 3
    player_count_embedding: int = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "privileged_features",
            "d_model",
            "hidden_layers",
            "player_count_embedding",
        ):
            _positive_integer(getattr(self, name), name)
        _dropout_probability(self.dropout, "dropout")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V5ActorOutput:
    logits: torch.Tensor
    residual_logits: torch.Tensor
    normal_auxiliary_logits: torch.Tensor
    normal_actions: torch.Tensor


@dataclass(frozen=True)
class V5PackedActorOutput:
    logits: torch.Tensor
    residual_logits: torch.Tensor
    normal_auxiliary_logits: torch.Tensor
    action_indices: torch.Tensor
    action_mask: torch.Tensor
    normal_actions: torch.Tensor

    def greedy_actions(self) -> torch.Tensor:
        positions = self.logits.argmax(dim=-1)
        return self.action_indices.gather(1, positions.unsqueeze(1)).squeeze(1)


class _SwiGLUCrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention followed by a pointwise SwiGLU."""

    def __init__(self, config: V5ActorConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.d_model)
        self.context_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.gate_and_value = nn.Linear(config.d_model, 2 * config.feedforward)
        self.feedforward_output = nn.Linear(config.feedforward, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_context = self.context_norm(context)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_context,
            normalized_context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        hidden = queries + self.dropout(attended)
        gate, value = self.gate_and_value(
            self.feedforward_norm(hidden)
        ).chunk(2, dim=-1)
        return hidden + self.dropout(
            self.feedforward_output(F.silu(gate) * value)
        )


class V5PublicActor(nn.Module):
    """Public-belief Normal-residual policy over the fixed 236 actions.

    The actor directly accepts the privacy-enforcing ``V5ActorPublicBatch``.
    Own/public/unknown cards, table and players, categorical ordered history,
    and exact public hypergeometric beliefs all become model tokens.  History
    is compressed before the compact public Transformer.  Structured action
    queries then use two nonlinear cross-attention scoring layers.  There is
    no actor method capable of accepting a privileged state.
    """

    _HISTORY_CARDINALITIES = (5, 10, 21, 21, 14, 13, 3, 15, 5, 4, 11, 11)

    def __init__(self, config: V5ActorConfig | None = None) -> None:
        super().__init__()
        self.config = config or V5ActorConfig()
        cfg = self.config
        d_model = cfg.d_model
        compact_hidden = max(cfg.heads, d_model // 2)
        belief_hidden = max(cfg.heads, (2 * d_model) // 3)

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.history_null_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.history_latent_queries = nn.Parameter(
            torch.empty(1, cfg.history_latents, d_model)
        )

        self.global_continuous_projection = nn.Sequential(
            nn.Linear(4, compact_hidden),
            nn.GELU(),
            nn.Linear(compact_hidden, d_model),
        )
        self.global_player_count_embedding = nn.Embedding(11, d_model)
        self.global_role_embedding = nn.Embedding(5, d_model)
        self.global_revolution_embedding = nn.Embedding(3, d_model)

        self.rank_index_embedding = nn.Embedding(V5_RANK_COUNT, d_model)
        self.rank_card_projection = nn.Sequential(
            nn.Linear(3, compact_hidden),
            nn.GELU(),
            nn.Linear(compact_hidden, d_model),
        )

        self.player_offset_embedding = nn.Embedding(V5_MAX_PLAYERS, d_model)
        self.player_remaining_embedding = nn.Embedding(21, d_model)
        self.player_role_embedding = nn.Embedding(5, d_model)
        self.player_flag_embeddings = nn.ModuleList(
            nn.Embedding(2, d_model) for _ in range(3)
        )

        self.table_present_embedding = nn.Embedding(2, d_model)
        self.table_rank_embedding = nn.Embedding(14, d_model)
        self.table_required_embedding = nn.Embedding(15, d_model)
        self.table_natural_embedding = nn.Embedding(13, d_model)
        self.table_joker_embedding = nn.Embedding(3, d_model)
        self.table_actor_embedding = nn.Embedding(10, d_model)

        # Every opponent token retains all three rank-wise hypergeometric
        # statistics (13*3) plus exact response feasibility.
        self.belief_projection = nn.Sequential(
            nn.Linear(V5_RANK_COUNT * 3 + 1, belief_hidden),
            nn.GELU(),
            nn.Linear(belief_hidden, d_model),
        )
        self.history_field_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_model)
            for cardinality in self._HISTORY_CARDINALITIES
        )
        self.history_position_embedding = nn.Embedding(
            V5_MAX_HISTORY + 1, d_model
        )
        self.history_compactor = _SwiGLUCrossAttentionLayer(cfg)

        self.token_type_embedding = nn.Embedding(7, d_model)
        maximum_core_tokens = (
            2
            + V5_RANK_COUNT
            + V5_MAX_PLAYERS
            + 1
            + V5_MAX_OPPONENTS
            + cfg.history_latents
        )
        self.core_position_embedding = nn.Embedding(
            maximum_core_tokens, d_model
        )
        self.public_input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.heads,
            dim_feedforward=cfg.feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.public_core = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.core_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.action_query_encoder = nn.Sequential(
            nn.Linear(V5_ACTION_FEATURE_COUNT, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.action_cross_attention = nn.ModuleList(
            _SwiGLUCrossAttentionLayer(cfg) for _ in range(cfg.action_layers)
        )
        self.residual_hidden = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.residual_output = nn.Linear(d_model, 1)
        self.normal_auxiliary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.register_buffer(
            "action_features",
            torch.tensor(V3_ACTION_FEATURES, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "deck_counts",
            torch.tensor(V5_DECK_COUNTS, dtype=torch.float32),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.history_null_token, mean=0.0, std=0.02)
        nn.init.normal_(self.history_latent_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.history_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.core_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    @staticmethod
    def _range_check(
        tensor: torch.Tensor,
        minimum: int,
        maximum: int,
        label: str,
    ) -> None:
        if not torch.jit.is_tracing() and (
            (tensor < minimum).any() or (tensor > maximum).any()
        ):
            raise ValueError(f"{label} is outside [{minimum}, {maximum}]")

    def _validate_public_batch(self, batch: V5ActorPublicBatch) -> int:
        if type(batch) is not V5ActorPublicBatch:
            raise TypeError("actor input must be exactly V5ActorPublicBatch")
        if batch.global_codes.ndim != 2 or batch.global_codes.shape[1] != len(V5_GLOBAL_FIELDS):
            raise ValueError("global_codes must be int64 [batch, 6]")
        batch_size = batch.global_codes.shape[0]
        integer_shapes = {
            "global_codes": (batch_size, len(V5_GLOBAL_FIELDS)),
            "own_rank_counts": (batch_size, V5_RANK_COUNT),
            "public_played_counts": (batch_size, V5_RANK_COUNT),
            "player_codes": (batch_size, V5_MAX_PLAYERS, len(V5_PLAYER_FIELDS)),
            "table_codes": (batch_size, len(V5_TABLE_FIELDS)),
            "history_codes": (batch_size, V5_MAX_HISTORY, len(V5_HISTORY_FIELDS)),
            "belief_unknown_rank_counts": (batch_size, V5_RANK_COUNT),
        }
        for name, shape in integer_shapes.items():
            value = getattr(batch, name)
            if value.dtype != torch.long or value.shape != shape:
                raise ValueError(f"{name} must be int64 with shape {shape}")
        boolean_shapes = {
            "player_mask": (batch_size, V5_MAX_PLAYERS),
            "history_mask": (batch_size, V5_MAX_HISTORY),
            "legal_mask": (batch_size, V5_ACTION_COUNT),
            "opponent_mask": (batch_size, V5_MAX_OPPONENTS),
        }
        for name, shape in boolean_shapes.items():
            value = getattr(batch, name)
            if value.dtype != torch.bool or value.shape != shape:
                raise ValueError(f"{name} must be bool with shape {shape}")
        continuous_shapes = {
            "belief_expected_counts": (
                batch_size, V5_MAX_OPPONENTS, V5_RANK_COUNT
            ),
            "belief_probability_at_least_one": (
                batch_size, V5_MAX_OPPONENTS, V5_RANK_COUNT
            ),
            "belief_probability_at_least_required": (
                batch_size, V5_MAX_OPPONENTS, V5_RANK_COUNT
            ),
            "belief_response_feasibility": (batch_size, V5_MAX_OPPONENTS),
        }
        for name, shape in continuous_shapes.items():
            value = getattr(batch, name)
            if not value.dtype.is_floating_point or value.shape != shape:
                raise ValueError(f"{name} must be floating-point with shape {shape}")
            if not torch.jit.is_tracing() and not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        devices = {
            getattr(batch, name).device
            for name in (*integer_shapes, *boolean_shapes, *continuous_shapes)
        }
        if len(devices) != 1:
            raise ValueError("all public batch tensors must use one device")
        if next(self.parameters()).device not in devices:
            raise ValueError("public batch and actor parameters must share a device")
        if not torch.jit.is_tracing():
            _validate_legal_masks(batch.legal_mask, batch_size)
            global_codes = batch.global_codes
            if not global_codes[:, 0].eq(V5_PUBLIC_SCHEMA_VERSION).all():
                raise ValueError("global_codes uses a non-V5 schema")
            self._range_check(global_codes[:, 1], 4, 10, "player count")
            self._range_check(global_codes[:, 2], 1, 1_000_000, "act")
            self._range_check(global_codes[:, 3], 0, 4, "actor role")
            self._range_check(global_codes[:, 4], 0, 2, "revolution")
            self._range_check(global_codes[:, 5], 0, 1_000_000_000, "truncation")
            expected_player_mask = (
                torch.arange(V5_MAX_PLAYERS, device=global_codes.device)[None, :]
                < global_codes[:, 1, None]
            )
            expected_opponent_mask = (
                torch.arange(V5_MAX_OPPONENTS, device=global_codes.device)[None, :]
                < (global_codes[:, 1, None] - 1)
            )
            if not torch.equal(batch.player_mask, expected_player_mask):
                raise ValueError("player_mask disagrees with player count")
            if not torch.equal(batch.opponent_mask, expected_opponent_mask):
                raise ValueError("opponent_mask disagrees with player count")
            deck = self.deck_counts.to(dtype=torch.long)
            own = batch.own_rank_counts
            played = batch.public_played_counts
            unknown = batch.belief_unknown_rank_counts
            if (
                (own < 0).any()
                or (played < 0).any()
                or (unknown < 0).any()
                or (own > deck).any()
                or (played > deck).any()
                or not torch.equal(unknown, deck - own - played)
            ):
                raise ValueError("card counts violate the public unseen-card identity")
            player = batch.player_codes
            for column, maximum in enumerate((9, 20, 4, 1, 1, 1)):
                self._range_check(player[..., column], 0, maximum, "player_codes")
            table = batch.table_codes
            for column, maximum in enumerate((1, 13, 14, 12, 2, 9)):
                self._range_check(table[..., column], 0, maximum, "table_codes")
            for column, maximum in enumerate(
                cardinality - 1 for cardinality in self._HISTORY_CARDINALITIES
            ):
                self._range_check(
                    batch.history_codes[..., column], 0, maximum, "history_codes"
                )
            probability_values = (
                batch.belief_probability_at_least_one,
                batch.belief_probability_at_least_required,
                batch.belief_response_feasibility,
            )
            if any(((value < 0.0) | (value > 1.0)).any() for value in probability_values):
                raise ValueError("belief probabilities must be in [0, 1]")
            if (batch.belief_expected_counts < 0.0).any():
                raise ValueError("belief expected counts must be non-negative")
        return batch_size

    def encode_public_core(
        self,
        batch: V5ActorPublicBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._validate_public_batch(batch)
        cfg = self.config
        dtype = self.cls_token.dtype

        global_codes = batch.global_codes
        global_continuous = torch.stack(
            (
                global_codes[:, 0].to(dtype=dtype) / V5_PUBLIC_SCHEMA_VERSION,
                global_codes[:, 1].to(dtype=dtype) / 10.0,
                torch.log1p(global_codes[:, 2].to(dtype=dtype)) / math.log1p(1_000_000),
                torch.log1p(global_codes[:, 5].to(dtype=dtype)) / math.log1p(1_000_000_000),
            ),
            dim=-1,
        )
        global_token = (
            self.global_continuous_projection(global_continuous)
            + self.global_player_count_embedding(global_codes[:, 1])
            + self.global_role_embedding(global_codes[:, 3])
            + self.global_revolution_embedding(global_codes[:, 4])
        ).unsqueeze(1)

        deck = self.deck_counts.to(dtype=dtype).unsqueeze(0)
        rank_values = torch.stack(
            (
                batch.own_rank_counts.to(dtype=dtype) / deck,
                batch.public_played_counts.to(dtype=dtype) / deck,
                batch.belief_unknown_rank_counts.to(dtype=dtype) / deck,
            ),
            dim=-1,
        )
        rank_tokens = self.rank_card_projection(rank_values) + self.rank_index_embedding(
            torch.arange(V5_RANK_COUNT, device=global_codes.device)
        ).unsqueeze(0)

        player = batch.player_codes
        player_tokens = (
            self.player_offset_embedding(player[..., 0])
            + self.player_remaining_embedding(player[..., 1])
            + self.player_role_embedding(player[..., 2])
        )
        for flag_index, embedding in enumerate(self.player_flag_embeddings, start=3):
            player_tokens = player_tokens + embedding(player[..., flag_index])

        table = batch.table_codes
        table_token = (
            self.table_present_embedding(table[:, 0])
            + self.table_rank_embedding(table[:, 1])
            + self.table_required_embedding(table[:, 2])
            + self.table_natural_embedding(table[:, 3])
            + self.table_joker_embedding(table[:, 4])
            + self.table_actor_embedding(table[:, 5])
        ).unsqueeze(1)

        belief_values = torch.cat(
            (
                batch.belief_expected_counts,
                batch.belief_probability_at_least_one,
                batch.belief_probability_at_least_required,
                batch.belief_response_feasibility.unsqueeze(-1),
            ),
            dim=-1,
        ).to(dtype=dtype)
        # The stable actor-relative opponent identity is supplied by the core
        # absolute position embedding below.
        belief_tokens = self.belief_projection(belief_values)

        history_tokens = torch.zeros(
            (batch_size, V5_MAX_HISTORY, cfg.d_model),
            dtype=dtype,
            device=global_codes.device,
        )
        for column, embedding in enumerate(self.history_field_embeddings):
            history_tokens = history_tokens + embedding(batch.history_codes[..., column])
        history_tokens = torch.cat(
            (
                self.history_null_token.expand(batch_size, -1, -1),
                history_tokens,
            ),
            dim=1,
        )
        history_positions = torch.arange(
            V5_MAX_HISTORY + 1, device=global_codes.device
        )
        history_tokens = history_tokens + self.history_position_embedding(
            history_positions
        ).unsqueeze(0)
        history_valid = torch.cat(
            (
                torch.ones(
                    (batch_size, 1), dtype=torch.bool, device=global_codes.device
                ),
                batch.history_mask,
            ),
            dim=1,
        )
        history_latents = self.history_compactor(
            self.history_latent_queries.expand(batch_size, -1, -1),
            history_tokens,
            history_valid,
        )

        tokens = torch.cat(
            (
                self.cls_token.expand(batch_size, -1, -1),
                global_token,
                rank_tokens,
                player_tokens,
                table_token,
                belief_tokens,
                history_latents,
            ),
            dim=1,
        )
        type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=global_codes.device),
                torch.ones(1, dtype=torch.long, device=global_codes.device),
                torch.full((V5_RANK_COUNT,), 2, dtype=torch.long, device=global_codes.device),
                torch.full((V5_MAX_PLAYERS,), 3, dtype=torch.long, device=global_codes.device),
                torch.full((1,), 4, dtype=torch.long, device=global_codes.device),
                torch.full((V5_MAX_OPPONENTS,), 5, dtype=torch.long, device=global_codes.device),
                torch.full((cfg.history_latents,), 6, dtype=torch.long, device=global_codes.device),
            )
        )
        positions = torch.arange(tokens.shape[1], device=global_codes.device)
        tokens = self.public_input_norm(
            tokens
            + self.token_type_embedding(type_ids).unsqueeze(0)
            + self.core_position_embedding(positions).unsqueeze(0)
        )
        core_valid = torch.cat(
            (
                torch.ones(
                    (batch_size, 2 + V5_RANK_COUNT),
                    dtype=torch.bool,
                    device=global_codes.device,
                ),
                batch.player_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=global_codes.device),
                batch.opponent_mask,
                torch.ones(
                    (batch_size, cfg.history_latents),
                    dtype=torch.bool,
                    device=global_codes.device,
                ),
            ),
            dim=1,
        )
        encoded = self.public_core(tokens, src_key_padding_mask=~core_valid)
        return encoded, core_valid

    def _score_action_indices(
        self,
        public_core: torch.Tensor,
        core_mask: torch.Tensor,
        action_indices: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_indices.dtype != torch.long or action_indices.ndim != 2:
            raise ValueError("action indices must be int64 [batch, actions]")
        if action_mask.dtype != torch.bool or action_mask.shape != action_indices.shape:
            raise ValueError("action mask must be bool [batch, actions]")
        if action_indices.shape[0] != public_core.shape[0]:
            raise ValueError("action indices batch size mismatch")
        if not torch.jit.is_tracing():
            if not action_mask.any(dim=-1).all():
                raise ValueError("every row requires at least one action")
            valid_indices = action_indices[action_mask]
            if int(valid_indices.min()) < 0 or int(valid_indices.max()) >= V5_ACTION_COUNT:
                raise ValueError("action index is outside the catalogue")
            for row_indices, row_mask in zip(action_indices, action_mask):
                selected = row_indices[row_mask]
                if torch.unique(selected).numel() != selected.numel():
                    raise ValueError("packed legal action indices must be unique")
        safe_indices = action_indices.clamp(0, V5_ACTION_COUNT - 1)
        features = self.action_features.to(dtype=public_core.dtype).index_select(
            0, safe_indices.reshape(-1)
        ).reshape(*safe_indices.shape, V5_ACTION_FEATURE_COUNT)
        action_hidden = self.action_query_encoder(features)
        for layer in self.action_cross_attention:
            action_hidden = layer(action_hidden, public_core, core_mask)
        residual = self.residual_output(
            self.residual_hidden(action_hidden)
        ).squeeze(-1)
        auxiliary = self.normal_auxiliary_head(action_hidden).squeeze(-1)
        masked = _masked_value(residual.dtype)
        return (
            residual.masked_fill(~action_mask, masked),
            auxiliary.masked_fill(~action_mask, masked),
        )

    def forward_with_auxiliary(
        self,
        batch: V5ActorPublicBatch,
        normal_actions: torch.Tensor,
    ) -> V5ActorOutput:
        public_core, core_mask = self.encode_public_core(batch)
        batch_size = batch.global_codes.shape[0]
        _validate_legal_masks(batch.legal_mask, batch_size)
        action_indices = torch.arange(
            V5_ACTION_COUNT, device=public_core.device
        ).unsqueeze(0).expand(batch_size, -1)
        residual, auxiliary = self._score_action_indices(
            public_core, core_mask, action_indices, batch.legal_mask
        )
        prior_dtype = _prior_accumulation_dtype(residual)
        prior, resolved = normal_prior_logits(
            batch.legal_mask,
            normal_actions,
            self.config.normal_prior_probability,
            dtype=prior_dtype,
        )
        logits = _mask_illegal_logits(
            prior + residual.to(dtype=prior_dtype), batch.legal_mask
        )
        return V5ActorOutput(logits, residual, auxiliary, resolved)

    def forward(
        self,
        batch: V5ActorPublicBatch,
        normal_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_auxiliary(batch, normal_actions).logits

    def forward_batch(
        self,
        batch: V5ActorPublicBatch,
        normal_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self(batch, normal_actions)

    def forward_packed_batch(
        self,
        batch: V5ActorPublicBatch,
        normal_actions: torch.Tensor,
        legal_action_indices: torch.Tensor | None = None,
        legal_action_mask: torch.Tensor | None = None,
    ) -> V5PackedActorOutput:
        if legal_action_indices is None or legal_action_mask is None:
            if legal_action_indices is not None or legal_action_mask is not None:
                raise ValueError("packed action indices and mask must be supplied together")
            legal_action_indices, legal_action_mask = pack_legal_actions(
                batch.legal_mask
            )
        if legal_action_indices.shape != legal_action_mask.shape:
            raise ValueError("packed action indices and mask shapes must match")
        if (
            legal_action_indices.dtype != torch.long
            or legal_action_mask.dtype != torch.bool
            or legal_action_indices.ndim != 2
            or legal_action_indices.shape[0] != batch.global_codes.shape[0]
            or legal_action_indices.device != batch.legal_mask.device
            or legal_action_mask.device != batch.legal_mask.device
        ):
            raise ValueError("packed actions must be int64/bool [batch, actions] on the actor device")
        selected_legal = torch.zeros_like(batch.legal_mask)
        selected_rows, selected_positions = legal_action_mask.nonzero(as_tuple=True)
        selected_indices = legal_action_indices[selected_rows, selected_positions]
        if not torch.jit.is_tracing() and (
            selected_indices.numel() == 0
            or int(selected_indices.min()) < 0
            or int(selected_indices.max()) >= V5_ACTION_COUNT
        ):
            raise ValueError("packed action index is outside the catalogue")
        selected_legal[selected_rows, selected_indices] = True
        if not torch.jit.is_tracing() and not torch.equal(
            selected_legal, batch.legal_mask
        ):
            raise ValueError("packed actions must exactly represent the legal mask")
        public_core, core_mask = self.encode_public_core(batch)
        residual, auxiliary = self._score_action_indices(
            public_core, core_mask, legal_action_indices, legal_action_mask
        )
        resolved = resolve_normal_actions(batch.legal_mask, normal_actions)
        selected_matches = (
            legal_action_indices == resolved.unsqueeze(1)
        ) & legal_action_mask
        if not torch.jit.is_tracing() and not selected_matches.sum(dim=-1).eq(1).all():
            raise ValueError("packed actions must contain the resolved Normal action")
        legal_counts = legal_action_mask.sum(dim=-1)
        odds = self.config.normal_prior_probability / (
            1.0 - self.config.normal_prior_probability
        )
        prior_dtype = _prior_accumulation_dtype(residual)
        margins = torch.where(
            legal_counts > 1,
            torch.log((legal_counts - 1).to(dtype=prior_dtype) * odds),
            torch.zeros_like(legal_counts, dtype=prior_dtype),
        )
        prior = torch.zeros_like(residual, dtype=prior_dtype).scatter(
            1,
            selected_matches.to(torch.int64).argmax(dim=-1, keepdim=True),
            margins.unsqueeze(1),
        )
        logits = (prior + residual.to(dtype=prior_dtype)).masked_fill(
            ~legal_action_mask, _masked_value(prior_dtype)
        )
        return V5PackedActorOutput(
            logits,
            residual,
            auxiliary,
            legal_action_indices,
            legal_action_mask,
            resolved,
        )


def normal_action_auxiliary_loss(
    auxiliary_logits: torch.Tensor,
    normal_actions: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    if auxiliary_logits.ndim != 2 or auxiliary_logits.shape[1] != V5_ACTION_COUNT:
        raise ValueError("auxiliary logits must have shape [batch, 236]")
    _validate_normal_actions(normal_actions, auxiliary_logits.shape[0])
    if normal_actions.device != auxiliary_logits.device:
        raise ValueError("normal actions and auxiliary logits must share a device")
    if ((normal_actions < 0) | (normal_actions >= V5_ACTION_COUNT)).any():
        raise ValueError("resolved Normal actions must be catalogue indices")
    return F.cross_entropy(auxiliary_logits, normal_actions, reduction=reduction)


class V5CentralStateValueCritic(nn.Module):
    """Centralized training-only state-V critic with separate parameters."""

    def __init__(self, config: V5CriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or V5CriticConfig()
        cfg = self.config
        self.player_count_embedding = nn.Embedding(11, cfg.player_count_embedding)
        input_features = cfg.privileged_features + cfg.player_count_embedding
        layers: list[nn.Module] = [
            nn.LayerNorm(input_features),
            nn.Linear(input_features, cfg.d_model),
            nn.GELU(),
        ]
        for _ in range(cfg.hidden_layers - 1):
            layers.extend(
                (
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                )
            )
        self.value_output = nn.Linear(cfg.d_model, 1)
        layers.extend((nn.LayerNorm(cfg.d_model), self.value_output))
        self.value_network = nn.Sequential(*layers)
        nn.init.zeros_(self.value_output.weight)
        nn.init.zeros_(self.value_output.bias)

    def forward(
        self,
        privileged_states: torch.Tensor,
        player_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if privileged_states.ndim != 2 or privileged_states.shape[1] != cfg.privileged_features:
            raise ValueError("privileged states must be [batch, privileged_features]")
        if not privileged_states.dtype.is_floating_point:
            raise ValueError("privileged states must be floating-point tensors")
        if not torch.jit.is_tracing() and not torch.isfinite(privileged_states).all():
            raise ValueError("privileged states must be finite")
        batch_size = privileged_states.shape[0]
        if player_counts is None:
            player_counts = torch.zeros(
                batch_size, dtype=torch.long, device=privileged_states.device
            )
        elif player_counts.dtype != torch.long or player_counts.shape != (batch_size,):
            raise ValueError("player counts must be int64 [batch]")
        elif player_counts.device != privileged_states.device:
            raise ValueError("player counts and privileged states must share a device")
        elif not torch.jit.is_tracing() and (
            ((player_counts < 4) | (player_counts > 10)).any()
        ):
            raise ValueError("player counts must be from 4 to 10")
        count_hidden = self.player_count_embedding(player_counts)
        return self.value_network(
            torch.cat((privileged_states, count_hidden), dim=-1)
        ).squeeze(-1)


def assert_actor_critic_parameter_isolation(
    actor: nn.Module,
    critic: nn.Module,
) -> None:
    actor_parameters = tuple(actor.parameters())
    critic_parameters = tuple(critic.parameters())
    if {id(value) for value in actor_parameters} & {
        id(value) for value in critic_parameters
    }:
        raise ValueError("actor and critic share Parameter objects")

    def storage_key(parameter: nn.Parameter) -> tuple[int, int]:
        storage = (
            parameter.untyped_storage()
            if hasattr(parameter, "untyped_storage")
            else parameter.storage()
        )
        return storage.data_ptr(), parameter.storage_offset()

    if {storage_key(value) for value in actor_parameters} & {
        storage_key(value) for value in critic_parameters
    }:
        raise ValueError("actor and critic share parameter storage")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "V5_ACTION_COUNT",
    "V5_ACTION_FEATURE_COUNT",
    "V5_MASKED_LOGIT",
    "V5_POLICY_CUBLAS_WORKSPACE_CONFIG",
    "V5_POLICY_NUMERICS_CONTRACT_VERSION",
    "V5_POLICY_NUMERICS_SHA256",
    "V5ActorConfig",
    "V5ActorOutput",
    "V5CentralStateValueCritic",
    "V5CriticConfig",
    "V5PackedActorOutput",
    "V5PublicActor",
    "assert_actor_critic_parameter_isolation",
    "canonical_v5_policy_numerics_contract",
    "configure_v5_policy_numerics",
    "normal_action_auxiliary_loss",
    "normal_prior_logits",
    "pack_legal_actions",
    "resolve_normal_actions",
    "trainable_parameter_count",
    "validate_v5_policy_numerics_contract",
]
