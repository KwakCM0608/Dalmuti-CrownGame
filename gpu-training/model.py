from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


OBSERVATION_FEATURES = 172
ACTION_COUNT = 506
DEFAULT_HIDDEN_SIZES = (256, 256)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_features: int = OBSERVATION_FEATURES,
        action_count: int = ACTION_COUNT,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        sizes = [observation_features, *hidden_sizes, action_count]
        layers: list[nn.Module] = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.observation_features = observation_features
        self.action_count = action_count
        self.hidden_sizes = tuple(hidden_sizes)

    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.network(observations)
        if legal_masks is not None:
            logits = logits.masked_fill(~legal_masks, -1.0e9)
        return logits


def linear_layers(model: PolicyNetwork) -> list[nn.Linear]:
    return [
        layer
        for layer in model.network
        if isinstance(layer, nn.Linear)
    ]


def export_policy_json(model: PolicyNetwork, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.to("cpu").eval()
    layers = []
    for layer in linear_layers(cpu_model):
        layers.append(
            {
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
        )
    payload = {
        "format": "dalmuti-mlp-policy",
        "version": 1,
        "observationFeatures": model.observation_features,
        "actionCount": model.action_count,
        "hiddenSizes": list(model.hidden_sizes),
        "activation": "relu",
        "weightLayout": "row-major [out_features, in_features]",
        "layers": layers,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
