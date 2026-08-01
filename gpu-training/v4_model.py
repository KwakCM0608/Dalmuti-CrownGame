from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn

from v3_action_conditioned import (
    V3_ACTION_COUNT,
    V3_ACTION_FEATURE_COUNT,
    V3_ACTION_FEATURES,
)


V4_ACTION_COUNT = V3_ACTION_COUNT
V4_ACTION_FEATURE_COUNT = V3_ACTION_FEATURE_COUNT
V4_MASKED_LOGIT = -1.0e9

# These sizes describe the first public-history tensor contract. They are
# configurable so a future encoder revision does not require a model rewrite.
V4_GLOBAL_FEATURES = 12
V4_RANK_FEATURES = 6
V4_PLAYER_FEATURES = 12
V4_HISTORY_FEATURES = 20
V4_MEMORY_FEATURES = 20
V4_RANK_TOKENS = 13
V4_MAX_PLAYERS = 10
V4_MAX_HISTORY = 192
V4_MEMORY_TOKENS = 4


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _probability(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result >= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1)")
    return result


def _validate_legal_masks(
    legal_masks: torch.Tensor,
    batch_size: int,
    action_count: int = V4_ACTION_COUNT,
) -> None:
    if legal_masks.dtype != torch.bool or legal_masks.shape != (
        batch_size,
        action_count,
    ):
        raise ValueError(
            f"legal masks must be bool [batch, {action_count}]"
        )
    if not torch.jit.is_tracing() and not legal_masks.any(dim=-1).all():
        raise ValueError("every observation requires at least one legal action")


def mask_illegal_logits(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    masked_value: float = V4_MASKED_LOGIT,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != V4_ACTION_COUNT:
        raise ValueError("logits must have shape [batch, 236]")
    _validate_legal_masks(legal_masks, logits.shape[0])
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must use a floating-point dtype")
    finite_floor = torch.finfo(logits.dtype).min / 2.0
    dtype_safe_value = max(float(masked_value), finite_floor)
    return logits.masked_fill(~legal_masks, dtype_safe_value)


@dataclass(frozen=True)
class V4ActorConfig:
    global_features: int = V4_GLOBAL_FEATURES
    rank_features: int = V4_RANK_FEATURES
    player_features: int = V4_PLAYER_FEATURES
    history_features: int = V4_HISTORY_FEATURES
    memory_features: int = V4_MEMORY_FEATURES
    rank_tokens: int = V4_RANK_TOKENS
    max_players: int = V4_MAX_PLAYERS
    max_history: int = V4_MAX_HISTORY
    memory_tokens: int = V4_MEMORY_TOKENS
    d_model: int = 384
    layers: int = 8
    heads: int = 12
    feedforward: int = 1024
    dropout: float = 0.0
    action_hidden: int = 384
    observation_schema_version: int = 4
    action_catalogue_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "global_features",
            "rank_features",
            "player_features",
            "history_features",
            "memory_features",
            "rank_tokens",
            "max_players",
            "max_history",
            "memory_tokens",
            "d_model",
            "layers",
            "heads",
            "feedforward",
            "action_hidden",
            "observation_schema_version",
            "action_catalogue_version",
        ):
            _positive_integer(getattr(self, name), name)
        _probability(self.dropout, "dropout")
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V4CriticConfig:
    privileged_features: int = 512
    d_model: int = 512
    hidden_layers: int = 3
    action_hidden: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "privileged_features",
            "d_model",
            "hidden_layers",
            "action_hidden",
        ):
            _positive_integer(getattr(self, name), name)
        _probability(self.dropout, "dropout")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class V4PublicActor(nn.Module):
    """Public-information-only entity/history Transformer policy.

    No privileged tensor is accepted by this module. The input contract has
    separate global, physical-rank, relative-player, and ordered public-event
    tensors. Scores are conditioned on the fixed 236-action semantic catalogue.
    """

    def __init__(self, config: V4ActorConfig | None = None) -> None:
        super().__init__()
        self.config = config or V4ActorConfig()
        cfg = self.config

        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.global_projection = nn.Linear(cfg.global_features, cfg.d_model)
        self.rank_projection = nn.Linear(cfg.rank_features, cfg.d_model)
        self.player_projection = nn.Linear(cfg.player_features, cfg.d_model)
        self.history_projection = nn.Linear(cfg.history_features, cfg.d_model)
        self.memory_projection = nn.Linear(cfg.memory_features, cfg.d_model)
        self.token_type_embedding = nn.Embedding(6, cfg.d_model)
        maximum_tokens = (
            2
            + cfg.rank_tokens
            + cfg.max_players
            + cfg.memory_tokens
            + cfg.max_history
        )
        self.position_embedding = nn.Embedding(maximum_tokens, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.heads,
            dim_feedforward=cfg.feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.layers,
            norm=nn.LayerNorm(cfg.d_model),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(V4_ACTION_FEATURE_COUNT, cfg.action_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_hidden, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.action_bias = nn.Linear(V4_ACTION_FEATURE_COUNT, 1)
        self.register_buffer(
            "action_features",
            torch.tensor(V3_ACTION_FEATURES, dtype=torch.float32),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embedding.weight, mean=0.0, std=0.02)

    def _validate_public_inputs(
        self,
        global_features: torch.Tensor,
        rank_features: torch.Tensor,
        player_features: torch.Tensor,
        player_mask: torch.Tensor,
        memory_trace_features: torch.Tensor,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[int, int, int]:
        cfg = self.config
        if global_features.ndim != 2 or global_features.shape[1] != cfg.global_features:
            raise ValueError("global features have an invalid shape")
        batch_size = global_features.shape[0]
        if rank_features.shape != (
            batch_size,
            cfg.rank_tokens,
            cfg.rank_features,
        ):
            raise ValueError("rank features have an invalid shape")
        if (
            player_features.ndim != 3
            or player_features.shape[0] != batch_size
            or player_features.shape[2] != cfg.player_features
            or player_features.shape[1] > cfg.max_players
            or player_features.shape[1] < 1
        ):
            raise ValueError("player features have an invalid shape")
        player_count = player_features.shape[1]
        if player_mask.dtype != torch.bool or player_mask.shape != (
            batch_size,
            player_count,
        ):
            raise ValueError("player mask must be bool [batch, players]")
        if (
            history_features.ndim != 3
            or history_features.shape[0] != batch_size
            or history_features.shape[2] != cfg.history_features
            or history_features.shape[1] > cfg.max_history
        ):
            raise ValueError("history features have an invalid shape")
        history_count = history_features.shape[1]
        if history_mask.dtype != torch.bool or history_mask.shape != (
            batch_size,
            history_count,
        ):
            raise ValueError("history mask must be bool [batch, history]")
        if memory_trace_features.shape != (
            batch_size,
            cfg.memory_tokens,
            cfg.memory_features,
        ):
            raise ValueError("memory-trace features have an invalid shape")
        if not torch.jit.is_tracing():
            public_tensors = (
                global_features,
                rank_features,
                player_features,
                memory_trace_features,
                history_features,
            )
            if any(not tensor.dtype.is_floating_point for tensor in public_tensors):
                raise ValueError("public actor inputs must use floating-point tensors")
            if any(not torch.isfinite(tensor).all() for tensor in public_tensors):
                raise ValueError("public actor inputs must be finite")
        return batch_size, player_count, history_count

    def encode_public(
        self,
        global_features: torch.Tensor,
        rank_features: torch.Tensor,
        player_features: torch.Tensor,
        player_mask: torch.Tensor,
        memory_trace_features: torch.Tensor,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, player_count, history_count = self._validate_public_inputs(
            global_features,
            rank_features,
            player_features,
            player_mask,
            memory_trace_features,
            history_features,
            history_mask,
        )
        cfg = self.config
        cls = self.cls_token.expand(batch_size, -1, -1)
        global_token = self.global_projection(global_features).unsqueeze(1)
        tokens = torch.cat(
            (
                cls,
                global_token,
                self.rank_projection(rank_features),
                self.player_projection(player_features),
                self.memory_projection(memory_trace_features),
                self.history_projection(history_features),
            ),
            dim=1,
        )

        type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=tokens.device),
                torch.ones(1, dtype=torch.long, device=tokens.device),
                torch.full(
                    (cfg.rank_tokens,), 2, dtype=torch.long, device=tokens.device
                ),
                torch.full(
                    (player_count,), 3, dtype=torch.long, device=tokens.device
                ),
                torch.full(
                    (cfg.memory_tokens,), 4, dtype=torch.long, device=tokens.device
                ),
                torch.full(
                    (history_count,), 5, dtype=torch.long, device=tokens.device
                ),
            )
        )
        # Use contract-stable segment offsets.  Player/history padding may be
        # trimmed for throughput, but that must never move memory/history
        # semantics to a different learned absolute position.
        fixed_count = 2 + cfg.rank_tokens
        memory_offset = fixed_count + cfg.max_players
        history_offset = memory_offset + cfg.memory_tokens
        positions = torch.cat(
            (
                torch.arange(fixed_count, device=tokens.device),
                torch.arange(
                    fixed_count,
                    fixed_count + player_count,
                    device=tokens.device,
                ),
                torch.arange(
                    memory_offset,
                    memory_offset + cfg.memory_tokens,
                    device=tokens.device,
                ),
                torch.arange(
                    history_offset,
                    history_offset + history_count,
                    device=tokens.device,
                ),
            )
        )
        tokens = self.input_norm(
            tokens
            + self.token_type_embedding(type_ids).unsqueeze(0)
            + self.position_embedding(positions).unsqueeze(0)
        )
        fixed_valid = torch.ones(
            (batch_size, 2 + cfg.rank_tokens),
            dtype=torch.bool,
            device=tokens.device,
        )
        valid_tokens = torch.cat(
            (
                fixed_valid,
                player_mask,
                torch.ones(
                    (batch_size, cfg.memory_tokens),
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                history_mask,
            ),
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~valid_tokens)
        return self.state_projection(encoded[:, 0])

    def forward(
        self,
        global_features: torch.Tensor,
        rank_features: torch.Tensor,
        player_features: torch.Tensor,
        player_mask: torch.Tensor,
        memory_trace_features: torch.Tensor,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.encode_public(
            global_features,
            rank_features,
            player_features,
            player_mask,
            memory_trace_features,
            history_features,
            history_mask,
        )
        action_features = self.action_features.to(dtype=state.dtype)
        action_hidden = self.action_encoder(action_features)
        logits = (
            torch.matmul(state, action_hidden.transpose(0, 1))
            / math.sqrt(self.config.d_model)
            + self.action_bias(action_features).transpose(0, 1)
        )
        if legal_masks is not None:
            logits = mask_illegal_logits(logits, legal_masks)
        return logits


class V4PrivilegedQCritic(nn.Module):
    """Training-only centralized action-Q critic.

    This module deliberately has no actor reference and owns all of its
    parameters. Its privileged input and weights must never enter actor export.
    """

    def __init__(self, config: V4CriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or V4CriticConfig()
        cfg = self.config
        state_layers: list[nn.Module] = [
            nn.LayerNorm(cfg.privileged_features),
            nn.Linear(cfg.privileged_features, cfg.d_model),
            nn.GELU(),
        ]
        for _ in range(cfg.hidden_layers - 1):
            state_layers.extend(
                (
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                )
            )
        state_layers.append(nn.LayerNorm(cfg.d_model))
        self.state_encoder = nn.Sequential(*state_layers)
        self.action_encoder = nn.Sequential(
            nn.Linear(V4_ACTION_FEATURE_COUNT, cfg.action_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_hidden, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.state_query = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.state_bias = nn.Linear(cfg.d_model, 1)
        self.action_bias = nn.Linear(V4_ACTION_FEATURE_COUNT, 1)
        self.register_buffer(
            "action_features",
            torch.tensor(V3_ACTION_FEATURES, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        privileged_states: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if privileged_states.ndim != 2 or privileged_states.shape[1] != cfg.privileged_features:
            raise ValueError("privileged states must be [batch, privileged_features]")
        if not privileged_states.dtype.is_floating_point:
            raise ValueError("privileged states must use a floating-point tensor")
        if not torch.jit.is_tracing() and not torch.isfinite(privileged_states).all():
            raise ValueError("privileged states must be finite")
        states = self.state_encoder(privileged_states)
        action_features = self.action_features.to(dtype=states.dtype)
        actions = self.action_encoder(action_features)
        q_values = (
            torch.matmul(self.state_query(states), actions.transpose(0, 1))
            / math.sqrt(cfg.d_model)
            + self.state_bias(states)
            + self.action_bias(action_features).transpose(0, 1)
        )
        if legal_masks is not None:
            q_values = mask_illegal_logits(q_values, legal_masks)
        return q_values


def assert_actor_critic_parameter_isolation(
    actor: nn.Module, critic: nn.Module
) -> None:
    actor_parameters = tuple(actor.parameters())
    critic_parameters = tuple(critic.parameters())
    actor_ids = {id(parameter) for parameter in actor_parameters}
    critic_ids = {id(parameter) for parameter in critic_parameters}
    if actor_ids & critic_ids:
        raise ValueError("actor and critic share Parameter objects")
    def storage_key(parameter: nn.Parameter) -> tuple[int, int]:
        # untyped_storage is the modern API; storage keeps PyTorch 1.12 GPU
        # hosts compatible without weakening the alias check.
        storage = (
            parameter.untyped_storage()
            if hasattr(parameter, "untyped_storage")
            else parameter.storage()
        )
        return storage.data_ptr(), parameter.storage_offset()

    actor_storage = {storage_key(parameter) for parameter in actor_parameters}
    critic_storage = {storage_key(parameter) for parameter in critic_parameters}
    if actor_storage & critic_storage:
        raise ValueError("actor and critic share parameter storage")


def centered_legal_logits(
    logits: torch.Tensor, legal_masks: torch.Tensor | None = None
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != V4_ACTION_COUNT:
        raise ValueError("logits must have shape [batch, 236]")
    if legal_masks is None:
        return logits - logits.mean(dim=-1, keepdim=True)
    _validate_legal_masks(legal_masks, logits.shape[0])
    legal_float = legal_masks.to(dtype=logits.dtype)
    mean = (logits * legal_float).sum(dim=-1, keepdim=True) / legal_float.sum(
        dim=-1, keepdim=True
    )
    return mask_illegal_logits(logits - mean, legal_masks)


class V4CenteredLogitEnsemble(nn.Module):
    """Three independently seeded actors averaged after legal-logit centering."""

    def __init__(
        self,
        actors: Sequence[V4PublicActor],
        seeds: Sequence[int],
    ) -> None:
        super().__init__()
        if len(actors) != 3 or len(seeds) != 3:
            raise ValueError("the V4 ensemble requires exactly three actors and seeds")
        if len(set(int(seed) for seed in seeds)) != 3:
            raise ValueError("ensemble seeds must be unique")
        if any(actor.config != actors[0].config for actor in actors[1:]):
            raise ValueError("ensemble actors must use the same configuration")
        self.actors = nn.ModuleList(actors)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.config = actors[0].config

    @classmethod
    def from_seeds(
        cls,
        config: V4ActorConfig | None = None,
        seeds: Sequence[int] = (41, 43, 47),
    ) -> "V4CenteredLogitEnsemble":
        cfg = config or V4ActorConfig()
        actors: list[V4PublicActor] = []
        for seed in seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                actors.append(V4PublicActor(cfg))
        return cls(actors, seeds)

    def forward(
        self,
        global_features: torch.Tensor,
        rank_features: torch.Tensor,
        player_features: torch.Tensor,
        player_mask: torch.Tensor,
        memory_trace_features: torch.Tensor,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        centered = [
            centered_legal_logits(
                actor(
                    global_features,
                    rank_features,
                    player_features,
                    player_mask,
                    memory_trace_features,
                    history_features,
                    history_mask,
                    legal_masks,
                ),
                legal_masks,
            )
            for actor in self.actors
        ]
        result = torch.stack(centered, dim=0).mean(dim=0)
        if legal_masks is not None:
            result = mask_illegal_logits(result, legal_masks)
        return result


def trainable_parameter_ids(module: nn.Module) -> frozenset[int]:
    return frozenset(id(parameter) for parameter in module.parameters())


__all__ = [
    "V4_ACTION_COUNT",
    "V4_ACTION_FEATURE_COUNT",
    "V4_MASKED_LOGIT",
    "V4ActorConfig",
    "V4CriticConfig",
    "V4PublicActor",
    "V4PrivilegedQCritic",
    "V4CenteredLogitEnsemble",
    "assert_actor_critic_parameter_isolation",
    "centered_legal_logits",
    "mask_illegal_logits",
    "trainable_parameter_ids",
]
