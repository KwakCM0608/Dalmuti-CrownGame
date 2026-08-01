"""Action-conditioned actor-critic networks for DALMUTI non-card decisions.

The constants and feature derivations in this module intentionally mirror
``training/non-card-observation.ts`` and
``training/non-card-action-space.ts``.  Tax-return and revolution decisions
use distinct networks and serialized formats so either policy can be trained,
promoted, or rolled back independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn


NON_CARD_OBSERVATION_SCHEMA_VERSION = 1

TAX_RETURN_OBSERVATION_FEATURE_COUNT = 103
REVOLUTION_OBSERVATION_FEATURE_COUNT = 102

TAX_RETURN_ACTION_CATALOGUE_VERSION = 1
REVOLUTION_ACTION_CATALOGUE_VERSION = 1

TAX_RETURN_ACTION_FEATURE_LAYOUT = (
    "returns-one-card",
    "returns-two-cards",
    *(f"rank-{rank}-fraction" for rank in range(1, 14)),
)
TAX_RETURN_ACTION_FEATURE_COUNT = len(TAX_RETURN_ACTION_FEATURE_LAYOUT)

REVOLUTION_ACTION_FEATURE_LAYOUT = (
    "decline",
    "declare-normal-revolution",
    "declare-great-revolution",
)
REVOLUTION_ACTION_FEATURE_COUNT = len(REVOLUTION_ACTION_FEATURE_LAYOUT)

TAX_RETURN_MODEL_FORMAT = (
    "dalmuti-tax-return-action-conditioned-actor-critic"
)
REVOLUTION_MODEL_FORMAT = (
    "dalmuti-revolution-action-conditioned-actor-critic"
)
NON_CARD_MODEL_FORMAT_VERSION = 1
MASKED_LOGIT = -1.0e9

# Offsets from COMMON_FEATURE_GROUP_DEFINITIONS in non-card-observation.ts.
ACTOR_ROLE_FEATURE_OFFSET = 3
GREAT_PEON_ROLE_OFFSET = 4
GREAT_PEON_ROLE_FEATURE_INDEX = (
    ACTOR_ROLE_FEATURE_OFFSET + GREAT_PEON_ROLE_OFFSET
)
OWN_HAND_COUNTS_FEATURE_OFFSET = 8
OWN_HAND_COUNTS_FEATURE_COUNT = 13
TAX_RETURN_COUNT_FEATURE_OFFSET = 101


def _create_tax_return_catalogue() -> tuple[tuple[int, ...], ...]:
    actions: list[tuple[int, ...]] = []
    actions.extend((rank,) for rank in range(1, 14))
    for first_rank in range(1, 14):
        for second_rank in range(first_rank, 14):
            if first_rank == 1 and second_rank == 1:
                continue
            actions.append((first_rank, second_rank))
    result = tuple(actions)
    if len(result) != 103:
        raise RuntimeError(
            f"tax-return catalogue has {len(result)} actions; expected 103"
        )
    return result


TAX_RETURN_ACTION_CATALOGUE = _create_tax_return_catalogue()
TAX_RETURN_ACTION_COUNT = len(TAX_RETURN_ACTION_CATALOGUE)
REVOLUTION_ACTION_COUNT = 2


def _tax_required_rank_counts(
    ranks: Sequence[int],
) -> tuple[int, ...]:
    counts = [0] * 13
    for rank in ranks:
        counts[rank - 1] += 1
    return tuple(counts)


TAX_RETURN_ACTION_SIZES = tuple(
    len(ranks) for ranks in TAX_RETURN_ACTION_CATALOGUE
)
TAX_RETURN_ACTION_REQUIRED_COUNTS = tuple(
    _tax_required_rank_counts(ranks)
    for ranks in TAX_RETURN_ACTION_CATALOGUE
)


def encode_tax_return_action_features(
    ranks: Sequence[int],
) -> tuple[float, ...]:
    canonical_ranks = tuple(sorted(ranks))
    if canonical_ranks not in TAX_RETURN_ACTION_CATALOGUE:
        raise ValueError("tax-return ranks are structurally impossible")
    features = [
        1.0 if len(canonical_ranks) == 1 else 0.0,
        1.0 if len(canonical_ranks) == 2 else 0.0,
        *([0.0] * 13),
    ]
    for rank in canonical_ranks:
        features[rank + 1] += 0.5
    return tuple(features)


TAX_RETURN_ACTION_FEATURES = tuple(
    encode_tax_return_action_features(ranks)
    for ranks in TAX_RETURN_ACTION_CATALOGUE
)


def _validate_observations(
    observations: torch.Tensor,
    expected_features: int,
) -> None:
    if observations.ndim != 2:
        raise ValueError("observations must have shape [batch, features]")
    if observations.shape[1] != expected_features:
        raise ValueError(
            f"observation feature count mismatch: expected {expected_features}"
        )
    if not torch.is_floating_point(observations):
        raise ValueError("observations must use a floating-point dtype")
    if not torch.isfinite(observations).all():
        raise ValueError("observations must contain only finite values")


def _validate_mask(
    legal_masks: torch.Tensor,
    batch_size: int,
    action_count: int,
) -> None:
    if legal_masks.dtype != torch.bool:
        raise ValueError("legal masks must use torch.bool")
    if legal_masks.shape != (batch_size, action_count):
        raise ValueError(
            f"legal masks must have shape [batch, {action_count}]"
        )
    if not legal_masks.any(dim=1).all():
        raise ValueError("every observation requires at least one legal action")


def _role_indices_from_observations(
    observations: torch.Tensor,
) -> torch.Tensor:
    role_features = observations[
        :, ACTOR_ROLE_FEATURE_OFFSET : ACTOR_ROLE_FEATURE_OFFSET + 5
    ]
    rounded_roles = role_features.round()
    if (
        (role_features - rounded_roles).abs() > 1.0e-4
    ).any() or not (rounded_roles.sum(dim=1) == 1).all():
        raise ValueError("actor-role features must be one-hot")
    return rounded_roles.argmax(dim=1)


def legal_tax_return_masks_from_observations(
    observations: torch.Tensor,
) -> torch.Tensor:
    """Reconstructs the exact TS tax mask from encoded observations.

    The observation contains normalized physical-rank counts and a one-hot
    return size.  Deriving the mask here prevents an invalid training batch
    from ever assigning probability or policy gradient to an illegal return.
    """

    _validate_observations(
        observations, TAX_RETURN_OBSERVATION_FEATURE_COUNT
    )
    return_count_features = observations[
        :,
        TAX_RETURN_COUNT_FEATURE_OFFSET : TAX_RETURN_COUNT_FEATURE_OFFSET + 2,
    ]
    return_one = return_count_features[:, 0] > 0.5
    return_two = return_count_features[:, 1] > 0.5
    if not torch.logical_xor(return_one, return_two).all():
        raise ValueError("tax return-count features must be one-hot")
    return_counts = torch.where(
        return_one,
        torch.ones_like(return_one, dtype=torch.long),
        torch.full_like(return_one, 2, dtype=torch.long),
    )
    role_indices = _role_indices_from_observations(observations)
    expected_return_counts = torch.where(
        role_indices == 0,
        torch.full_like(role_indices, 2),
        torch.where(
            role_indices == 1,
            torch.ones_like(role_indices),
            torch.zeros_like(role_indices),
        ),
    )
    if not (expected_return_counts == return_counts).all():
        raise ValueError(
            "tax return count does not match the encoded Dalmuti role"
        )

    normalized_counts = observations[
        :,
        OWN_HAND_COUNTS_FEATURE_OFFSET : (
            OWN_HAND_COUNTS_FEATURE_OFFSET
            + OWN_HAND_COUNTS_FEATURE_COUNT
        ),
    ]
    deck_copies = observations.new_tensor(
        [*range(1, 13), 2], dtype=observations.dtype
    )
    scaled_counts = normalized_counts * deck_copies
    rounded_counts = scaled_counts.round()
    if (
        (scaled_counts - rounded_counts).abs() > 1.0e-4
    ).any() or (rounded_counts < 0).any() or (
        rounded_counts > deck_copies
    ).any():
        raise ValueError("tax hand-count features are not valid card counts")
    hand_counts = rounded_counts.to(dtype=torch.long)

    required_counts = torch.tensor(
        TAX_RETURN_ACTION_REQUIRED_COUNTS,
        dtype=torch.long,
        device=observations.device,
    )
    action_sizes = torch.tensor(
        TAX_RETURN_ACTION_SIZES,
        dtype=torch.long,
        device=observations.device,
    )

    size_matches = return_counts[:, None] == action_sizes[None, :]
    fits_hand = (
        required_counts[None, :, :] <= hand_counts[:, None, :]
    ).all(dim=2)
    legal_masks = size_matches & fits_hand
    if not legal_masks.any(dim=1).all():
        raise ValueError("tax observation produced no legal return action")
    return legal_masks


def revolution_action_features_from_observations(
    observations: torch.Tensor,
) -> torch.Tensor:
    """Builds per-observation action features exactly as the TS encoder does."""

    _validate_observations(
        observations, REVOLUTION_OBSERVATION_FEATURE_COUNT
    )
    role_indices = _role_indices_from_observations(observations)
    great_peon = (role_indices == GREAT_PEON_ROLE_OFFSET).to(
        dtype=observations.dtype
    )
    result = observations.new_zeros(
        (observations.shape[0], REVOLUTION_ACTION_COUNT, 3)
    )
    result[:, 0, 0] = 1.0
    result[:, 1, 1] = 1.0 - great_peon
    result[:, 1, 2] = great_peon
    return result


def legal_revolution_masks_from_observations(
    observations: torch.Tensor,
) -> torch.Tensor:
    _validate_observations(
        observations, REVOLUTION_OBSERVATION_FEATURE_COUNT
    )
    return torch.ones(
        (observations.shape[0], REVOLUTION_ACTION_COUNT),
        dtype=torch.bool,
        device=observations.device,
    )


def _validate_hidden_sizes(
    sizes: Sequence[int],
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    result = tuple(sizes)
    if not allow_empty and not result:
        raise ValueError(f"{label} must contain at least one hidden layer")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1
        for size in result
    ):
        raise ValueError(f"{label} must contain positive integers")
    return result


def _hidden_trunk(
    input_size: int, hidden_sizes: Sequence[int]
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(current_size, hidden_size), nn.ReLU()))
        current_size = hidden_size
    return nn.Sequential(*layers)


def _output_network(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(current_size, hidden_size), nn.ReLU()))
        current_size = hidden_size
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)


class _ActionConditionedActorCriticNetwork(nn.Module):
    def __init__(
        self,
        *,
        observation_features: int,
        action_features: int,
        action_count: int,
        actor_observation_hidden_sizes: Sequence[int],
        actor_action_hidden_sizes: Sequence[int],
        actor_scorer_hidden_sizes: Sequence[int],
        value_hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        observation_sizes = _validate_hidden_sizes(
            actor_observation_hidden_sizes, "actor observation trunk"
        )
        action_sizes = _validate_hidden_sizes(
            actor_action_hidden_sizes, "actor action trunk"
        )
        scorer_sizes = _validate_hidden_sizes(
            actor_scorer_hidden_sizes,
            "actor scorer",
            allow_empty=True,
        )
        value_sizes = _validate_hidden_sizes(
            value_hidden_sizes, "value network"
        )

        self.actor_observation_trunk = _hidden_trunk(
            observation_features, observation_sizes
        )
        self.actor_action_trunk = _hidden_trunk(
            action_features, action_sizes
        )
        self.actor_scorer = _output_network(
            observation_sizes[-1] + action_sizes[-1], scorer_sizes, 1
        )
        self.value_network = _output_network(
            observation_features, value_sizes, 1
        )
        self.observation_features = observation_features
        self.action_feature_count = action_features
        self.action_count = action_count
        self.actor_observation_hidden_sizes = observation_sizes
        self.actor_action_hidden_sizes = action_sizes
        self.actor_scorer_hidden_sizes = scorer_sizes
        self.value_hidden_sizes = value_sizes

    def _score_legal_pairs(
        self,
        observations: torch.Tensor,
        action_hidden: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.actor_observation_trunk(observations)
        batch_indices, action_indices = legal_masks.nonzero(as_tuple=True)
        if action_hidden.ndim == 2:
            selected_action_hidden = action_hidden.index_select(
                0, action_indices
            )
        elif action_hidden.ndim == 3:
            selected_action_hidden = action_hidden[
                batch_indices, action_indices
            ]
        else:
            raise RuntimeError("action trunk produced an invalid shape")
        combined = torch.cat(
            (
                state_hidden.index_select(0, batch_indices),
                selected_action_hidden,
            ),
            dim=-1,
        )
        legal_scores = self.actor_scorer(combined).squeeze(-1)
        logits = observations.new_full(
            (observations.shape[0], self.action_count), MASKED_LOGIT
        )
        logits[batch_indices, action_indices] = legal_scores
        values = self.value_network(observations).squeeze(-1)
        return logits, values


class TaxReturnActionConditionedActorCriticNetwork(
    _ActionConditionedActorCriticNetwork
):
    """Actor-critic that scores only legal semantic tax-return actions."""

    def __init__(
        self,
        actor_observation_hidden_sizes: Sequence[int] = (128, 64),
        actor_action_hidden_sizes: Sequence[int] = (32, 32),
        actor_scorer_hidden_sizes: Sequence[int] = (128, 64),
        value_hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__(
            observation_features=TAX_RETURN_OBSERVATION_FEATURE_COUNT,
            action_features=TAX_RETURN_ACTION_FEATURE_COUNT,
            action_count=TAX_RETURN_ACTION_COUNT,
            actor_observation_hidden_sizes=actor_observation_hidden_sizes,
            actor_action_hidden_sizes=actor_action_hidden_sizes,
            actor_scorer_hidden_sizes=actor_scorer_hidden_sizes,
            value_hidden_sizes=value_hidden_sizes,
        )
        self.register_buffer(
            "action_features",
            torch.tensor(
                TAX_RETURN_ACTION_FEATURES, dtype=torch.float32
            ),
            persistent=False,
        )

    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        derived_masks = legal_tax_return_masks_from_observations(observations)
        if legal_masks is not None:
            _validate_mask(
                legal_masks,
                observations.shape[0],
                TAX_RETURN_ACTION_COUNT,
            )
            if not torch.equal(legal_masks, derived_masks):
                raise ValueError(
                    "tax legal mask does not match the encoded observation"
                )
        action_hidden = self.actor_action_trunk(
            self.action_features.to(dtype=observations.dtype)
        )
        return self._score_legal_pairs(
            observations, action_hidden, derived_masks
        )


class RevolutionActionConditionedActorCriticNetwork(
    _ActionConditionedActorCriticNetwork
):
    """Actor-critic for decline/declare with role-conditioned declaration."""

    def __init__(
        self,
        actor_observation_hidden_sizes: Sequence[int] = (128, 64),
        actor_action_hidden_sizes: Sequence[int] = (16, 16),
        actor_scorer_hidden_sizes: Sequence[int] = (64, 32),
        value_hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__(
            observation_features=REVOLUTION_OBSERVATION_FEATURE_COUNT,
            action_features=REVOLUTION_ACTION_FEATURE_COUNT,
            action_count=REVOLUTION_ACTION_COUNT,
            actor_observation_hidden_sizes=actor_observation_hidden_sizes,
            actor_action_hidden_sizes=actor_action_hidden_sizes,
            actor_scorer_hidden_sizes=actor_scorer_hidden_sizes,
            value_hidden_sizes=value_hidden_sizes,
        )

    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        derived_masks = legal_revolution_masks_from_observations(observations)
        if legal_masks is not None:
            _validate_mask(
                legal_masks,
                observations.shape[0],
                REVOLUTION_ACTION_COUNT,
            )
            if not torch.equal(legal_masks, derived_masks):
                raise ValueError(
                    "revolution legal mask must enable decline and declare"
                )
        action_features = revolution_action_features_from_observations(
            observations
        )
        action_hidden = self.actor_action_trunk(
            action_features.reshape(-1, REVOLUTION_ACTION_FEATURE_COUNT)
        ).reshape(observations.shape[0], REVOLUTION_ACTION_COUNT, -1)
        return self._score_legal_pairs(
            observations, action_hidden, derived_masks
        )


def _linear_layers(module: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in module if isinstance(layer, nn.Linear)]


def _export_layer(layer: nn.Linear) -> dict[str, object]:
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


def _export_payload(
    model: _ActionConditionedActorCriticNetwork,
    *,
    model_format: str,
    decision_kind: str,
    action_catalogue_version: int,
    action_feature_layout: Sequence[str],
) -> dict[str, object]:
    return {
        "format": model_format,
        "version": NON_CARD_MODEL_FORMAT_VERSION,
        "decisionKind": decision_kind,
        "observationSchemaVersion": NON_CARD_OBSERVATION_SCHEMA_VERSION,
        "observationFeatures": model.observation_features,
        "actionCatalogueVersion": action_catalogue_version,
        "actionCount": model.action_count,
        "actionFeatures": model.action_feature_count,
        "actionFeatureLayout": list(action_feature_layout),
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
            _export_layer(layer)
            for layer in _linear_layers(model.actor_scorer)
        ],
        "valueLayers": [
            _export_layer(layer)
            for layer in _linear_layers(model.value_network)
        ],
    }


def _write_payload(payload: Mapping[str, object], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def export_tax_return_action_conditioned_json(
    model: TaxReturnActionConditionedActorCriticNetwork,
    output_path: str | Path,
) -> None:
    model.eval()
    _write_payload(
        _export_payload(
            model,
            model_format=TAX_RETURN_MODEL_FORMAT,
            decision_kind="tax-return",
            action_catalogue_version=TAX_RETURN_ACTION_CATALOGUE_VERSION,
            action_feature_layout=TAX_RETURN_ACTION_FEATURE_LAYOUT,
        ),
        output_path,
    )


def export_revolution_action_conditioned_json(
    model: RevolutionActionConditionedActorCriticNetwork,
    output_path: str | Path,
) -> None:
    model.eval()
    payload = _export_payload(
        model,
        model_format=REVOLUTION_MODEL_FORMAT,
        decision_kind="revolution",
        action_catalogue_version=REVOLUTION_ACTION_CATALOGUE_VERSION,
        action_feature_layout=REVOLUTION_ACTION_FEATURE_LAYOUT,
    )
    payload["greatPeonRoleFeatureIndex"] = GREAT_PEON_ROLE_FEATURE_INDEX
    _write_payload(payload, output_path)


def _require_sizes(
    payload: Mapping[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{key} must not be empty")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1
        for size in value
    ):
        raise ValueError(f"{key} must contain positive integers")
    return tuple(value)


def _copy_json_layer(
    layer: nn.Linear, value: object, label: str
) -> None:
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
        layer.weight.copy_(
            weight.reshape(layer.out_features, layer.in_features)
        )
        layer.bias.copy_(bias)


def _copy_json_layers(
    module: nn.Sequential, values: object, label: str
) -> None:
    layers = _linear_layers(module)
    if not isinstance(values, list) or len(values) != len(layers):
        raise ValueError(f"{label} layer count mismatch")
    for index, layer in enumerate(layers):
        _copy_json_layer(layer, values[index], f"{label} layer {index}")


def _read_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("non-card model must be a JSON object")
    return payload


def _validate_contract(
    payload: Mapping[str, object],
    *,
    model_format: str,
    decision_kind: str,
    observation_features: int,
    action_catalogue_version: int,
    action_count: int,
    action_feature_count: int,
    action_feature_layout: Sequence[str],
) -> None:
    if (
        payload.get("format") != model_format
        or payload.get("version") != NON_CARD_MODEL_FORMAT_VERSION
        or payload.get("decisionKind") != decision_kind
        or payload.get("activation") != "relu"
        or payload.get("weightLayout")
        != "row-major [out_features, in_features]"
    ):
        raise ValueError("unsupported non-card actor-critic model format")
    if (
        payload.get("observationSchemaVersion")
        != NON_CARD_OBSERVATION_SCHEMA_VERSION
        or payload.get("observationFeatures") != observation_features
    ):
        raise ValueError("non-card observation contract mismatch")
    if (
        payload.get("actionCatalogueVersion")
        != action_catalogue_version
        or payload.get("actionCount") != action_count
        or payload.get("actionFeatures") != action_feature_count
        or payload.get("actionFeatureLayout")
        != list(action_feature_layout)
    ):
        raise ValueError("non-card action catalogue contract mismatch")


def _load_layers(
    model: _ActionConditionedActorCriticNetwork,
    payload: Mapping[str, object],
) -> None:
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
        model.value_network,
        payload.get("valueLayers"),
        "value network",
    )


def load_tax_return_action_conditioned_json(
    path: str | Path,
) -> tuple[TaxReturnActionConditionedActorCriticNetwork, dict[str, object]]:
    payload = _read_payload(path)
    _validate_contract(
        payload,
        model_format=TAX_RETURN_MODEL_FORMAT,
        decision_kind="tax-return",
        observation_features=TAX_RETURN_OBSERVATION_FEATURE_COUNT,
        action_catalogue_version=TAX_RETURN_ACTION_CATALOGUE_VERSION,
        action_count=TAX_RETURN_ACTION_COUNT,
        action_feature_count=TAX_RETURN_ACTION_FEATURE_COUNT,
        action_feature_layout=TAX_RETURN_ACTION_FEATURE_LAYOUT,
    )
    model = TaxReturnActionConditionedActorCriticNetwork(
        actor_observation_hidden_sizes=_require_sizes(
            payload, "actorObservationHiddenSizes"
        ),
        actor_action_hidden_sizes=_require_sizes(
            payload, "actorActionHiddenSizes"
        ),
        actor_scorer_hidden_sizes=_require_sizes(
            payload, "actorScorerHiddenSizes", allow_empty=True
        ),
        value_hidden_sizes=_require_sizes(payload, "valueHiddenSizes"),
    )
    _load_layers(model, payload)
    return model, payload


def load_revolution_action_conditioned_json(
    path: str | Path,
) -> tuple[RevolutionActionConditionedActorCriticNetwork, dict[str, object]]:
    payload = _read_payload(path)
    _validate_contract(
        payload,
        model_format=REVOLUTION_MODEL_FORMAT,
        decision_kind="revolution",
        observation_features=REVOLUTION_OBSERVATION_FEATURE_COUNT,
        action_catalogue_version=REVOLUTION_ACTION_CATALOGUE_VERSION,
        action_count=REVOLUTION_ACTION_COUNT,
        action_feature_count=REVOLUTION_ACTION_FEATURE_COUNT,
        action_feature_layout=REVOLUTION_ACTION_FEATURE_LAYOUT,
    )
    if payload.get("greatPeonRoleFeatureIndex") != GREAT_PEON_ROLE_FEATURE_INDEX:
        raise ValueError("revolution role-conditioned action contract mismatch")
    model = RevolutionActionConditionedActorCriticNetwork(
        actor_observation_hidden_sizes=_require_sizes(
            payload, "actorObservationHiddenSizes"
        ),
        actor_action_hidden_sizes=_require_sizes(
            payload, "actorActionHiddenSizes"
        ),
        actor_scorer_hidden_sizes=_require_sizes(
            payload, "actorScorerHiddenSizes", allow_empty=True
        ),
        value_hidden_sizes=_require_sizes(payload, "valueHiddenSizes"),
    )
    _load_layers(model, payload)
    return model, payload
