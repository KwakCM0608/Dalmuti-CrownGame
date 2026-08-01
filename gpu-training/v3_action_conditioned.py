from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import nn


V3_ACTION_CATALOGUE_VERSION = 1
V3_NORMAL_RANK_COUNT = 12
V3_ACTION_COUNT = 236
V3_ACTION_FEATURE_LAYOUT = (
    "type.pass",
    "type.solo-joker",
    "type.play",
    *(f"rank.{rank}" for rank in range(1, V3_NORMAL_RANK_COUNT + 1)),
    "joker-count.0",
    "joker-count.1",
    "joker-count.2",
    "rank-strength",
    "natural-count",
    "total-count",
    "joker-fraction",
)
V3_ACTION_FEATURE_COUNT = len(V3_ACTION_FEATURE_LAYOUT)


def _create_action_catalogue() -> tuple[Mapping[str, object], ...]:
    actions: list[Mapping[str, object]] = [
        MappingProxyType({"type": "pass"}),
        MappingProxyType({"type": "solo-joker"}),
    ]
    for rank in range(1, V3_NORMAL_RANK_COUNT + 1):
        for natural_count in range(1, rank + 1):
            for joker_count in range(3):
                actions.append(
                    MappingProxyType({
                        "type": "play",
                        "rank": rank,
                        "count": natural_count + joker_count,
                        "jokerCount": joker_count,
                    })
                )
    if len(actions) != V3_ACTION_COUNT:
        raise RuntimeError(f"V3 action catalogue has {len(actions)} entries")
    return tuple(actions)


V3_ACTION_CATALOGUE = _create_action_catalogue()


def _integer_in_range(value: object, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be from {minimum} to {maximum}")
    return value


def encode_v3_semantic_action(action: Mapping[str, object]) -> int:
    action_type = action.get("type")
    if action_type == "pass":
        return 0
    if action_type == "solo-joker":
        return 1
    if action_type != "play":
        raise ValueError("unsupported V3 semantic action type")
    rank = _integer_in_range(action.get("rank"), 1, 12, "rank")
    count = _integer_in_range(action.get("count"), 1, 14, "count")
    joker_count = _integer_in_range(
        action.get("jokerCount"), 0, 2, "jokerCount"
    )
    natural_count = count - joker_count
    if natural_count < 1 or natural_count > rank:
        raise ValueError(
            f"rank {rank} requires a natural-card count from 1 to {rank}"
        )
    first_rank_index = 2 + (3 * (rank - 1) * rank) // 2
    return first_rank_index + (natural_count - 1) * 3 + joker_count


def decode_v3_semantic_action(action_index: int) -> dict:
    index = _integer_in_range(
        action_index, 0, V3_ACTION_COUNT - 1, "action_index"
    )
    return dict(V3_ACTION_CATALOGUE[index])


def encode_v3_action_features(action_or_index: Mapping[str, object] | int) -> tuple[float, ...]:
    if isinstance(action_or_index, int) and not isinstance(action_or_index, bool):
        action = decode_v3_semantic_action(action_or_index)
    else:
        action = decode_v3_semantic_action(
            encode_v3_semantic_action(action_or_index)
        )
    features = [0.0] * V3_ACTION_FEATURE_COUNT
    action_type = action["type"]
    if action_type == "pass":
        features[0] = 1.0
    elif action_type == "solo-joker":
        features[1] = 1.0
        features[16] = 1.0
        features[20] = 1.0 / 14.0
        features[21] = 1.0
    else:
        rank = int(action["rank"])
        count = int(action["count"])
        joker_count = int(action["jokerCount"])
        natural_count = count - joker_count
        features[2] = 1.0
        features[3 + rank - 1] = 1.0
        features[15 + joker_count] = 1.0
        features[18] = (13.0 - rank) / 12.0
        features[19] = natural_count / 12.0
        features[20] = count / 14.0
        features[21] = joker_count / count
    return tuple(features)


V3_ACTION_FEATURES = tuple(
    encode_v3_action_features(index) for index in range(V3_ACTION_COUNT)
)


def _validate_hidden_sizes(
    sizes: Sequence[int], label: str, require_layer: bool = True
) -> tuple[int, ...]:
    result = tuple(sizes)
    if require_layer and not result:
        raise ValueError(f"{label} must contain at least one hidden layer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise ValueError(f"{label} must contain positive integers")
    return result


def _hidden_trunk(input_size: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(current_size, hidden_size), nn.ReLU()))
        current_size = hidden_size
    return nn.Sequential(*layers)


def _output_network(
    input_size: int, hidden_sizes: Sequence[int], output_size: int
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(current_size, hidden_size), nn.ReLU()))
        current_size = hidden_size
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)


class V3ActionConditionedActorCriticNetwork(nn.Module):
    """Scores shared semantic actions without exposing opponents' hidden cards.

    The caller supplies a fixed-size observation assembled from the acting
    player's private hand and public information. The actor observation trunk,
    actor action trunk, and critic network do not share parameters.
    """

    def __init__(
        self,
        observation_features: int = 172,
        observation_schema_version: int = 2,
        actor_observation_hidden_sizes: Sequence[int] = (256, 128),
        actor_action_hidden_sizes: Sequence[int] = (64, 64),
        actor_scorer_hidden_sizes: Sequence[int] = (256, 128),
        value_hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if (
            isinstance(observation_features, bool)
            or not isinstance(observation_features, int)
            or observation_features < 1
            or isinstance(observation_schema_version, bool)
            or not isinstance(observation_schema_version, int)
            or observation_schema_version < 1
        ):
            raise ValueError("observation metadata must use positive integers")
        actor_observation_sizes = _validate_hidden_sizes(
            actor_observation_hidden_sizes, "actor observation trunk"
        )
        actor_action_sizes = _validate_hidden_sizes(
            actor_action_hidden_sizes, "actor action trunk"
        )
        actor_scorer_sizes = _validate_hidden_sizes(
            actor_scorer_hidden_sizes, "actor scorer", require_layer=False
        )
        value_sizes = _validate_hidden_sizes(value_hidden_sizes, "value network")

        self.actor_observation_trunk = _hidden_trunk(
            observation_features, actor_observation_sizes
        )
        self.actor_action_trunk = _hidden_trunk(
            V3_ACTION_FEATURE_COUNT, actor_action_sizes
        )
        self.actor_scorer = _output_network(
            actor_observation_sizes[-1] + actor_action_sizes[-1],
            actor_scorer_sizes,
            1,
        )
        self.value_network = _output_network(
            observation_features, value_sizes, 1
        )
        self.register_buffer(
            "action_features",
            torch.tensor(V3_ACTION_FEATURES, dtype=torch.float32),
            persistent=False,
        )
        self.observation_features = observation_features
        self.observation_schema_version = observation_schema_version
        self.actor_observation_hidden_sizes = actor_observation_sizes
        self.actor_action_hidden_sizes = actor_action_sizes
        self.actor_scorer_hidden_sizes = actor_scorer_sizes
        self.value_hidden_sizes = value_sizes

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if observations.ndim != 2:
            raise ValueError("observations must have shape [batch, features]")
        if observations.shape[1] != self.observation_features:
            raise ValueError("observation feature count mismatch")

    def _score_hidden_pairs(
        self, state_hidden: torch.Tensor, action_hidden: torch.Tensor
    ) -> torch.Tensor:
        batch_size = state_hidden.shape[0]
        action_count = action_hidden.shape[0]
        states = state_hidden[:, None, :].expand(-1, action_count, -1)
        actions = action_hidden[None, :, :].expand(batch_size, -1, -1)
        combined = torch.cat((states, actions), dim=-1)
        return self.actor_scorer(combined).squeeze(-1)

    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_observations(observations)
        state_hidden = self.actor_observation_trunk(observations)
        action_hidden = self.actor_action_trunk(self.action_features)
        if legal_masks is None:
            logits = self._score_hidden_pairs(state_hidden, action_hidden)
        else:
            expected_shape = (observations.shape[0], V3_ACTION_COUNT)
            if legal_masks.shape != expected_shape or legal_masks.dtype != torch.bool:
                raise ValueError("legal masks must be bool [batch, 236]")
            if not legal_masks.any(dim=1).all():
                raise ValueError("every observation requires a legal action")
            batch_indices, action_indices = legal_masks.nonzero(as_tuple=True)
            combined = torch.cat(
                (
                    state_hidden.index_select(0, batch_indices),
                    action_hidden.index_select(0, action_indices),
                ),
                dim=-1,
            )
            legal_scores = self.actor_scorer(combined).squeeze(-1)
            logits = observations.new_full(expected_shape, -1.0e9)
            logits[batch_indices, action_indices] = legal_scores
        values = self.value_network(observations).squeeze(-1)
        return logits, values

    def forward_legal(
        self,
        observations: torch.Tensor,
        legal_action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores one shared, non-empty legal-action list for a batch."""
        self._validate_observations(observations)
        if legal_action_indices.ndim != 1 or legal_action_indices.numel() < 1:
            raise ValueError("legal action indices must be a non-empty vector")
        if legal_action_indices.dtype != torch.long:
            raise ValueError("legal action indices must use torch.long")
        if (
            int(legal_action_indices.min()) < 0
            or int(legal_action_indices.max()) >= V3_ACTION_COUNT
            or torch.unique(legal_action_indices).numel()
            != legal_action_indices.numel()
        ):
            raise ValueError("legal action indices are invalid or duplicated")
        selected_features = self.action_features.index_select(
            0, legal_action_indices.to(self.action_features.device)
        )
        state_hidden = self.actor_observation_trunk(observations)
        action_hidden = self.actor_action_trunk(selected_features)
        logits = self._score_hidden_pairs(state_hidden, action_hidden)
        values = self.value_network(observations).squeeze(-1)
        return logits, values


def _linear_layers(module: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in module if isinstance(layer, nn.Linear)]


def _export_layer(layer: nn.Linear) -> dict:
    return {
        "inFeatures": layer.in_features,
        "outFeatures": layer.out_features,
        "weight": [
            round(float(value), 8)
            for value in layer.weight.detach().cpu().flatten().tolist()
        ],
        "bias": [
            round(float(value), 8)
            for value in layer.bias.detach().cpu().tolist()
        ],
    }


def export_v3_action_conditioned_json(
    model: V3ActionConditionedActorCriticNetwork,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    payload = {
        "format": "dalmuti-action-conditioned-actor-critic",
        "version": 1,
        "observationSchemaVersion": model.observation_schema_version,
        "observationFeatures": model.observation_features,
        "actionCatalogueVersion": V3_ACTION_CATALOGUE_VERSION,
        "actionCount": V3_ACTION_COUNT,
        "actionFeatures": V3_ACTION_FEATURE_COUNT,
        "actionFeatureLayout": list(V3_ACTION_FEATURE_LAYOUT),
        "actorObservationHiddenSizes": list(
            model.actor_observation_hidden_sizes
        ),
        "actorActionHiddenSizes": list(model.actor_action_hidden_sizes),
        "actorScorerHiddenSizes": list(model.actor_scorer_hidden_sizes),
        "valueHiddenSizes": list(model.value_hidden_sizes),
        "activation": "relu",
        "weightLayout": "row-major [out_features, in_features]",
        "actorObservationLayers": [
            _export_layer(layer)
            for layer in _linear_layers(model.actor_observation_trunk)
        ],
        "actorActionLayers": [
            _export_layer(layer)
            for layer in _linear_layers(model.actor_action_trunk)
        ],
        "actorScorerLayers": [
            _export_layer(layer) for layer in _linear_layers(model.actor_scorer)
        ],
        "valueLayers": [
            _export_layer(layer) for layer in _linear_layers(model.value_network)
        ],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _require_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _require_sizes(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1
        for size in value
    ):
        raise ValueError(f"{key} must contain positive integers")
    return tuple(value)


def _copy_json_layer(layer: nn.Linear, value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if (
        value.get("inFeatures") != layer.in_features
        or value.get("outFeatures") != layer.out_features
    ):
        raise ValueError(f"{label} dimensions do not connect")
    try:
        weight = torch.tensor(value["weight"], dtype=torch.float32)
        bias = torch.tensor(value["bias"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} parameters are invalid") from error
    if weight.numel() != layer.in_features * layer.out_features:
        raise ValueError(f"{label} weight size mismatch")
    if bias.numel() != layer.out_features:
        raise ValueError(f"{label} bias size mismatch")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError(f"{label} parameters must be finite")
    with torch.no_grad():
        layer.weight.copy_(weight.reshape(layer.out_features, layer.in_features))
        layer.bias.copy_(bias)


def _copy_json_layers(
    module: nn.Sequential, values: object, label: str
) -> None:
    layers = _linear_layers(module)
    if not isinstance(values, list) or len(values) != len(layers):
        raise ValueError(f"{label} layer count mismatch")
    for index, layer in enumerate(layers):
        _copy_json_layer(layer, values[index], f"{label} layer {index}")


def load_v3_action_conditioned_json(
    path: str | Path,
) -> tuple[V3ActionConditionedActorCriticNetwork, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V3 model must be a JSON object")
    if (
        payload.get("format") != "dalmuti-action-conditioned-actor-critic"
        or payload.get("version") != 1
        or payload.get("activation") != "relu"
        or payload.get("weightLayout")
        != "row-major [out_features, in_features]"
    ):
        raise ValueError("unsupported V3 actor-critic model format")
    if (
        payload.get("actionCatalogueVersion") != V3_ACTION_CATALOGUE_VERSION
        or payload.get("actionCount") != V3_ACTION_COUNT
        or payload.get("actionFeatures") != V3_ACTION_FEATURE_COUNT
        or payload.get("actionFeatureLayout") != list(V3_ACTION_FEATURE_LAYOUT)
    ):
        raise ValueError("V3 action catalogue contract mismatch")

    model = V3ActionConditionedActorCriticNetwork(
        observation_features=_require_integer(payload, "observationFeatures"),
        observation_schema_version=_require_integer(
            payload, "observationSchemaVersion"
        ),
        actor_observation_hidden_sizes=_require_sizes(
            payload, "actorObservationHiddenSizes"
        ),
        actor_action_hidden_sizes=_require_sizes(
            payload, "actorActionHiddenSizes"
        ),
        actor_scorer_hidden_sizes=_require_sizes(
            payload, "actorScorerHiddenSizes"
        ),
        value_hidden_sizes=_require_sizes(payload, "valueHiddenSizes"),
    )
    _copy_json_layers(
        model.actor_observation_trunk,
        payload.get("actorObservationLayers"),
        "actor observation trunk",
    )
    _copy_json_layers(
        model.actor_action_trunk,
        payload.get("actorActionLayers"),
        "actor action trunk",
    )
    _copy_json_layers(
        model.actor_scorer,
        payload.get("actorScorerLayers"),
        "actor scorer",
    )
    _copy_json_layers(
        model.value_network, payload.get("valueLayers"), "value network"
    )
    return model, payload
