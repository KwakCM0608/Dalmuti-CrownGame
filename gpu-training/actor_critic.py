from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


OBSERVATION_FEATURES = 172
ACTION_COUNT = 506
DEFAULT_HIDDEN_SIZES = (256, 256)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        observation_features: int = OBSERVATION_FEATURES,
        action_count: int = ACTION_COUNT,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("actor-critic requires at least one hidden layer")
        trunk_layers: list[nn.Module] = []
        input_size = observation_features
        for hidden_size in hidden_sizes:
            trunk_layers.append(nn.Linear(input_size, hidden_size))
            trunk_layers.append(nn.ReLU())
            input_size = hidden_size
        self.trunk = nn.Sequential(*trunk_layers)
        self.policy_head = nn.Linear(input_size, action_count)
        self.value_head = nn.Linear(input_size, 1)
        self.observation_features = observation_features
        self.action_count = action_count
        self.hidden_sizes = tuple(hidden_sizes)

    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        logits = self.policy_head(hidden)
        if legal_masks is not None:
            logits = logits.masked_fill(~legal_masks, -1.0e9)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def _linear_layers(module: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in module if isinstance(layer, nn.Linear)]


def _copy_json_layer(layer: nn.Linear, value: dict, label: str) -> None:
    if value.get("inFeatures") != layer.in_features:
        raise ValueError(f"{label} input size mismatch")
    if value.get("outFeatures") != layer.out_features:
        raise ValueError(f"{label} output size mismatch")
    weight = torch.tensor(value.get("weight"), dtype=torch.float32)
    bias = torch.tensor(value.get("bias"), dtype=torch.float32)
    if weight.numel() != layer.in_features * layer.out_features:
        raise ValueError(f"{label} weight size mismatch")
    if bias.numel() != layer.out_features:
        raise ValueError(f"{label} bias size mismatch")
    with torch.no_grad():
        layer.weight.copy_(weight.reshape(layer.out_features, layer.in_features))
        layer.bias.copy_(bias)


def load_behavior_model(
    path: str | Path,
) -> tuple[ActorCriticNetwork, dict]:
    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    model_format = payload.get("format")
    if payload.get("observationFeatures") != OBSERVATION_FEATURES:
        raise ValueError("behavior model observation size mismatch")
    if payload.get("actionCount") != ACTION_COUNT:
        raise ValueError("behavior model action size mismatch")
    hidden_sizes = tuple(int(value) for value in payload.get("hiddenSizes", []))
    model = ActorCriticNetwork(hidden_sizes=hidden_sizes)
    trunk_layers = _linear_layers(model.trunk)

    if model_format == "dalmuti-mlp-policy":
        if payload.get("version") != 1:
            raise ValueError("unsupported MLP policy version")
        layers = payload.get("layers")
        if not isinstance(layers, list) or len(layers) != len(trunk_layers) + 1:
            raise ValueError("behavior MLP layer count mismatch")
        for index, layer in enumerate(trunk_layers):
            _copy_json_layer(layer, layers[index], f"trunk layer {index}")
        _copy_json_layer(model.policy_head, layers[-1], "policy layer")
        with torch.no_grad():
            model.value_head.weight.zero_()
            model.value_head.bias.zero_()
    elif model_format == "dalmuti-actor-critic":
        if payload.get("version") != 1:
            raise ValueError("unsupported actor-critic version")
        layers = payload.get("trunkLayers")
        if not isinstance(layers, list) or len(layers) != len(trunk_layers):
            raise ValueError("actor-critic trunk layer count mismatch")
        for index, layer in enumerate(trunk_layers):
            _copy_json_layer(layer, layers[index], f"trunk layer {index}")
        _copy_json_layer(model.policy_head, payload["policyLayer"], "policy layer")
        _copy_json_layer(model.value_head, payload["valueLayer"], "value layer")
    else:
        raise ValueError(f"unsupported behavior model format: {model_format}")
    return model, payload


def _export_layer(layer: nn.Linear) -> dict:
    return {
        "inFeatures": layer.in_features,
        "outFeatures": layer.out_features,
        "weight": [
            round(float(value), 8)
            for value in layer.weight.detach().flatten().tolist()
        ],
        "bias": [
            round(float(value), 8)
            for value in layer.bias.detach().tolist()
        ],
    }


def export_actor_critic_json(
    model: ActorCriticNetwork,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.to("cpu").eval()
    payload = {
        "format": "dalmuti-actor-critic",
        "version": 1,
        "observationFeatures": model.observation_features,
        "actionCount": model.action_count,
        "hiddenSizes": list(model.hidden_sizes),
        "activation": "relu",
        "weightLayout": "row-major [out_features, in_features]",
        "trunkLayers": [
            _export_layer(layer)
            for layer in _linear_layers(cpu_model.trunk)
        ],
        "policyLayer": _export_layer(cpu_model.policy_head),
        "valueLayer": _export_layer(cpu_model.value_head),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
