"""Conservative tax-return advantage ensemble model.

This training-only model predicts the *residual* chip advantage of each legal
two-card return against the exact current normal-heuristic action.  Its scorer
is intentionally compact::

    raw(s, a) = tanh(A s + b)^T W phi(a)
    advantage(s, a | b) = raw(s, a) - raw(s, b)

Subtracting the baseline raw score makes the baseline advantage exactly zero
in both PyTorch and the TypeScript evaluator, independent of floating-point
rounding in the learned parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from non_card_action_conditioned import (
    NON_CARD_OBSERVATION_SCHEMA_VERSION,
    TAX_RETURN_ACTION_CATALOGUE_VERSION,
    TAX_RETURN_ACTION_COUNT,
    TAX_RETURN_ACTION_FEATURE_COUNT,
    TAX_RETURN_ACTION_FEATURE_LAYOUT,
    TAX_RETURN_ACTION_FEATURES,
    TAX_RETURN_OBSERVATION_FEATURE_COUNT,
    legal_tax_return_masks_from_observations,
)


TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT = (
    "dalmuti-tax-return-bilinear-residual-ensemble"
)
TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION = 2
TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS = (
    "chip-advantage-vs-normal-baseline"
)
TAX_RETURN_ADVANTAGE_MEMBER_COUNT = 5
TAX_RETURN_ADVANTAGE_Z_VALUE = 1.645
TAX_RETURN_ADVANTAGE_DEFAULT_MINIMUM_CHIPS = 0.5
TAX_RETURN_ADVANTAGE_CONTEXT_FEATURES = 16
TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT = (
    "row-major [context_features, action_features]"
)
TAX_RETURN_COUNT_FEATURE_OFFSET = 101
DETERMINIZATION_ALGORITHM = (
    "target-act-opponent-physical-card-fisher-yates-v1"
)
DETERMINIZATION_ALGORITHM_VERSION = 1
DETERMINIZATION_ALGORITHM_CONTRACT_SHA256 = (
    "368240f14f2e5d84bb3085610a176ad4519bc6e5ae288b70de549f63212905c4"
)
DETERMINIZATION_CANDIDATE_SEED_DERIVATION = (
    "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,"
    "worldIndex,attempt)))"
)
DETERMINIZATION_CONTINUATION_SEED_DERIVATION = (
    "uint32(first-8-hex(sha256(rootSeed,canonicalInformationStateKey,"
    "worldIndex,continuationIndex,continuation)))"
)

BASELINE_PROVENANCE = {
    "implementation": "lib/bot-strategy.ts#chooseBotTaxReturn",
    "semanticEncoding": (
        "training/non-card-action-space.ts#encodeTaxReturnAction"
    ),
    "difficulty": "normal",
}


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


BASELINE_PROVENANCE_SHA256 = canonical_sha256(BASELINE_PROVENANCE)


class TaxReturnBilinearResidualNetwork(nn.Module):
    """Small action-conditioned score network with exact residualization."""

    def __init__(self, context_features: int = TAX_RETURN_ADVANTAGE_CONTEXT_FEATURES):
        super().__init__()
        if isinstance(context_features, bool) or not isinstance(context_features, int):
            raise TypeError("context_features must be an integer")
        if context_features < 1:
            raise ValueError("context_features must be positive")
        self.context_features = context_features
        self.context = nn.Linear(
            TAX_RETURN_OBSERVATION_FEATURE_COUNT,
            context_features,
        )
        self.bilinear_weight = nn.Parameter(
            torch.empty(context_features, TAX_RETURN_ACTION_FEATURE_COUNT)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.context.weight)
        nn.init.zeros_(self.context.bias)
        nn.init.xavier_uniform_(self.bilinear_weight)

    def raw_scores(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != TAX_RETURN_OBSERVATION_FEATURE_COUNT:
            raise ValueError(
                "observations must have shape "
                f"[batch, {TAX_RETURN_OBSERVATION_FEATURE_COUNT}]"
            )
        if not torch.is_floating_point(observations) or not torch.isfinite(observations).all():
            raise ValueError("observations must contain finite floating-point values")
        context = torch.tanh(self.context(observations))
        action_features = observations.new_tensor(TAX_RETURN_ACTION_FEATURES)
        return context @ self.bilinear_weight @ action_features.transpose(0, 1)

    def forward(
        self,
        observations: torch.Tensor,
        baseline_actions: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.raw_scores(observations)
        if baseline_actions.dtype != torch.long or baseline_actions.shape != (
            observations.shape[0],
        ):
            raise ValueError("baseline_actions must be torch.long with shape [batch]")
        if (baseline_actions < 0).any() or (
            baseline_actions >= TAX_RETURN_ACTION_COUNT
        ).any():
            raise ValueError("baseline action index is out of range")
        if legal_masks is None:
            legal_masks = legal_tax_return_masks_from_observations(observations)
        if legal_masks.dtype != torch.bool or legal_masks.shape != scores.shape:
            raise ValueError(
                f"legal_masks must be bool with shape {tuple(scores.shape)}"
            )
        if not legal_masks.gather(1, baseline_actions[:, None]).all():
            raise ValueError("baseline action must be legal")
        baseline_scores = scores.gather(1, baseline_actions[:, None])
        advantages = scores - baseline_scores
        # Assignment, rather than a second subtraction, guarantees an exact
        # +0.0 serialized/inference contract for every baseline action.
        advantages = advantages.scatter(
            1,
            baseline_actions[:, None],
            torch.zeros_like(baseline_scores),
        )
        return advantages.masked_fill(~legal_masks, 0.0)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_list(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise TypeError(f"{label} must contain {length} numbers")
    return [_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def export_layer_parameters(model: TaxReturnBilinearResidualNetwork) -> dict[str, object]:
    return {
        "contextLayer": {
            "inFeatures": TAX_RETURN_OBSERVATION_FEATURE_COUNT,
            "outFeatures": model.context_features,
            "weight": model.context.weight.detach().cpu().reshape(-1).tolist(),
            "bias": model.context.bias.detach().cpu().tolist(),
        },
        "bilinearWeight": model.bilinear_weight.detach().cpu().reshape(-1).tolist(),
    }


def load_layer_parameters(
    model: TaxReturnBilinearResidualNetwork,
    member: Mapping[str, object],
) -> None:
    layer = member.get("contextLayer")
    if not isinstance(layer, dict) or set(layer) != {
        "inFeatures",
        "outFeatures",
        "weight",
        "bias",
    }:
        raise ValueError("member contextLayer fields mismatch")
    if layer["inFeatures"] != TAX_RETURN_OBSERVATION_FEATURE_COUNT:
        raise ValueError("member context input size mismatch")
    if layer["outFeatures"] != model.context_features:
        raise ValueError("member context output size mismatch")
    context_weight = _finite_list(
        layer["weight"],
        TAX_RETURN_OBSERVATION_FEATURE_COUNT * model.context_features,
        "member context weight",
    )
    context_bias = _finite_list(
        layer["bias"], model.context_features, "member context bias"
    )
    bilinear_weight = _finite_list(
        member.get("bilinearWeight"),
        model.context_features * TAX_RETURN_ACTION_FEATURE_COUNT,
        "member bilinear weight",
    )
    with torch.no_grad():
        model.context.weight.copy_(
            torch.tensor(context_weight, dtype=model.context.weight.dtype).reshape(
                model.context_features,
                TAX_RETURN_OBSERVATION_FEATURE_COUNT,
            )
        )
        model.context.bias.copy_(
            torch.tensor(context_bias, dtype=model.context.bias.dtype)
        )
        model.bilinear_weight.copy_(
            torch.tensor(
                bilinear_weight,
                dtype=model.bilinear_weight.dtype,
            ).reshape(
                model.context_features,
                TAX_RETURN_ACTION_FEATURE_COUNT,
            )
        )


def member_parameters_sha256(member: Mapping[str, object]) -> str:
    layer = member.get("contextLayer")
    if not isinstance(layer, dict):
        raise TypeError("member contextLayer must be an object")
    in_features = layer.get("inFeatures")
    out_features = layer.get("outFeatures")
    weight = layer.get("weight")
    bias = layer.get("bias")
    bilinear = member.get("bilinearWeight")
    if (
        isinstance(in_features, bool)
        or not isinstance(in_features, int)
        or isinstance(out_features, bool)
        or not isinstance(out_features, int)
        or not isinstance(weight, list)
        or not isinstance(bias, list)
        or not isinstance(bilinear, list)
    ):
        raise TypeError("member parameter hash payload is malformed")
    digest = hashlib.sha256()
    digest.update(b"dalmuti-tax-return-bilinear-residual-member-v1\0")
    digest.update(
        struct.pack(
            "<IIIII",
            in_features,
            out_features,
            len(weight),
            len(bias),
            len(bilinear),
        )
    )
    for values in (weight, bias, bilinear):
        for value in values:
            digest.update(struct.pack("<d", float(value)))
    return digest.hexdigest()


def validate_ensemble_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("tax-return advantage ensemble must be an object")
    expected = {
        "format",
        "version",
        "decisionKind",
        "scoreSemantics",
        "observationSchemaVersion",
        "observationFeatures",
        "actionCatalogueVersion",
        "actionCount",
        "actionFeatures",
        "actionFeatureLayout",
        "trainingData",
        "architecture",
        "baseline",
        "objective",
        "routing",
        "members",
    }
    if set(value) != expected:
        raise ValueError("tax-return advantage ensemble fields mismatch")
    if (
        value["format"] != TAX_RETURN_ADVANTAGE_ENSEMBLE_FORMAT
        or value["version"] != TAX_RETURN_ADVANTAGE_ENSEMBLE_VERSION
        or value["decisionKind"] != "tax-return"
        or value["scoreSemantics"] != TAX_RETURN_ADVANTAGE_SCORE_SEMANTICS
    ):
        raise ValueError("unsupported tax-return advantage ensemble format")
    if (
        value["observationSchemaVersion"] != NON_CARD_OBSERVATION_SCHEMA_VERSION
        or value["observationFeatures"] != TAX_RETURN_OBSERVATION_FEATURE_COUNT
        or value["actionCatalogueVersion"] != TAX_RETURN_ACTION_CATALOGUE_VERSION
        or value["actionCount"] != TAX_RETURN_ACTION_COUNT
        or value["actionFeatures"] != TAX_RETURN_ACTION_FEATURE_COUNT
        or value["actionFeatureLayout"] != list(TAX_RETURN_ACTION_FEATURE_LAYOUT)
    ):
        raise ValueError("tax-return advantage feature contract mismatch")
    training_data = value["trainingData"]
    if not isinstance(training_data, dict) or set(training_data) != {
        "sourceFormatVersions",
        "groupSplitKey",
        "determinizationSchema",
        "worldCountPerInformationState",
        "continuationCountPerHiddenWorld",
        "effectiveIndependentWorldsPerInformationState",
        "rawContinuationEvaluationsPerInformationState",
        "standardErrorEstimable",
        "determinizationAlgorithm",
        "determinizationAlgorithmVersion",
        "determinizationAlgorithmContractSha256",
        "candidateSeedDerivation",
        "continuationSeedDerivation",
        "targetField",
        "targetTransform",
        "stateWeighting",
    }:
        raise ValueError("tax-return training-data fields mismatch")
    versions = training_data["sourceFormatVersions"]
    if versions not in ([1], [2]):
        raise ValueError("tax-return source format versions must be [1] or [2]")
    version = versions[0]
    expected_group_key = (
        "canonicalWorldKey" if version == 1 else "canonicalInformationStateKey"
    )
    if training_data["groupSplitKey"] != expected_group_key:
        raise ValueError("tax-return training-data group key mismatch")
    expected_schema = (
        None
        if version == 1
        else "world-clustered-paired-baseline-advantages-v2"
    )
    if training_data["determinizationSchema"] != expected_schema:
        raise ValueError("tax-return determinization schema mismatch")
    expected_determinization_contract = (
        {
            "determinizationAlgorithm": None,
            "determinizationAlgorithmVersion": None,
            "determinizationAlgorithmContractSha256": None,
            "candidateSeedDerivation": None,
            "continuationSeedDerivation": None,
        }
        if version == 1
        else {
            "determinizationAlgorithm": DETERMINIZATION_ALGORITHM,
            "determinizationAlgorithmVersion": DETERMINIZATION_ALGORITHM_VERSION,
            "determinizationAlgorithmContractSha256": (
                DETERMINIZATION_ALGORITHM_CONTRACT_SHA256
            ),
            "candidateSeedDerivation": DETERMINIZATION_CANDIDATE_SEED_DERIVATION,
            "continuationSeedDerivation": (
                DETERMINIZATION_CONTINUATION_SEED_DERIVATION
            ),
        }
    )
    if any(
        training_data[key] != expected
        for key, expected in expected_determinization_contract.items()
    ):
        raise ValueError("tax-return determinization algorithm contract mismatch")
    world_count = training_data["worldCountPerInformationState"]
    continuation_count = training_data["continuationCountPerHiddenWorld"]
    effective_worlds = training_data[
        "effectiveIndependentWorldsPerInformationState"
    ]
    raw_continuation_evaluations = training_data[
        "rawContinuationEvaluationsPerInformationState"
    ]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in (
            world_count,
            continuation_count,
            effective_worlds,
            raw_continuation_evaluations,
        )
    ) or (
        effective_worlds != world_count
        or raw_continuation_evaluations != world_count * continuation_count
    ):
        raise ValueError("tax-return hidden-world/continuation binding mismatch")
    if version == 1 and (
        world_count,
        continuation_count,
        effective_worlds,
        raw_continuation_evaluations,
    ) != (1, 1, 1, 1):
        raise ValueError("tax-return v1 source must bind exactly one replay sample")
    if training_data["standardErrorEstimable"] is not (world_count > 1):
        raise ValueError("tax-return standard-error estimability mismatch")
    expected_target_field = (
        "actions[].decisionActUtility-minus-baseline.decisionActUtility"
        if version == 1
        else "actions[].pairedDecisionActBaselineAdvantage.mean"
    )
    if training_data["targetField"] != expected_target_field:
        raise ValueError("tax-return target field mismatch")
    if training_data["targetTransform"] != {
        "scoreUnit": "chip-units",
        "sourceUnit": "(roundChipAward-2)/2",
        "operation": "multiply-source-baseline-advantage-by-2",
        "multiplier": 2.0,
    }:
        raise ValueError("tax-return target unit transform mismatch")
    if training_data["stateWeighting"] != (
        "one-per-information-state-independent-of-worldCount"
    ):
        raise ValueError("tax-return information-state weighting mismatch")
    architecture = value["architecture"]
    if not isinstance(architecture, dict) or set(architecture) != {
        "contextFeatures",
        "contextActivation",
        "score",
        "weightLayout",
    }:
        raise ValueError("tax-return advantage architecture fields mismatch")
    context_features = architecture["contextFeatures"]
    if isinstance(context_features, bool) or not isinstance(context_features, int) or context_features < 1:
        raise ValueError("contextFeatures must be a positive integer")
    if architecture != {
        "contextFeatures": context_features,
        "contextActivation": "tanh",
        "score": "raw(s,a)-raw(s,normalBaselineAction)",
        "weightLayout": TAX_RETURN_ADVANTAGE_WEIGHT_LAYOUT,
    }:
        raise ValueError("unsupported tax-return advantage architecture")
    baseline = value["baseline"]
    if not isinstance(baseline, dict) or set(baseline) != {
        "provenance",
        "provenanceSha256",
        "score",
    }:
        raise ValueError("tax-return baseline fields mismatch")
    if baseline["provenance"] != BASELINE_PROVENANCE:
        raise ValueError("tax-return normal-baseline provenance mismatch")
    if baseline["provenanceSha256"] != BASELINE_PROVENANCE_SHA256:
        raise ValueError("tax-return normal-baseline provenance hash mismatch")
    if baseline["score"] != "exactly-zero-by-residualization":
        raise ValueError("tax-return baseline score contract mismatch")
    objective = value["objective"]
    required_objective = {
        "utilityTarget",
        "utilityScale",
        "weighting",
        "regression",
        "tieAwareSign",
        "checkpointSelection",
        "bootstrapUnit",
    }
    if not isinstance(objective, dict) or set(objective) != required_objective:
        raise ValueError("tax-return objective fields mismatch")
    if objective["utilityTarget"] != "decision-act-current-chip-advantage":
        raise ValueError("tax-return utility target mismatch")
    if objective["utilityScale"] != "chip-units":
        raise ValueError("tax-return utility scale mismatch")
    if objective["weighting"] != "equal-per-state":
        raise ValueError("tax-return weighting mismatch")
    if objective["checkpointSelection"] != "paired-validation-loss":
        raise ValueError("tax-return checkpoint selection mismatch")
    if objective["bootstrapUnit"] != expected_group_key:
        raise ValueError("tax-return bootstrap/group split unit mismatch")
    regression = objective["regression"]
    if not isinstance(regression, dict) or set(regression) != {
        "loss",
        "coefficient",
        "deltaChips",
    }:
        raise ValueError("tax-return regression objective fields mismatch")
    if regression["loss"] != "huber-paired-action-vs-baseline":
        raise ValueError("tax-return regression loss mismatch")
    if _finite_number(regression["coefficient"], "regression coefficient") <= 0:
        raise ValueError("tax-return regression coefficient must be positive")
    if _finite_number(regression["deltaChips"], "Huber delta") <= 0:
        raise ValueError("tax-return Huber delta must be positive")
    tie_sign = objective["tieAwareSign"]
    if not isinstance(tie_sign, dict) or set(tie_sign) != {
        "loss",
        "coefficient",
        "temperatureChips",
        "tieTarget",
        "tieEpsilonChips",
    }:
        raise ValueError("tax-return tie-aware sign objective fields mismatch")
    if (
        tie_sign["loss"] != "binary-cross-entropy-with-logits"
        or tie_sign["tieTarget"] != 0.5
    ):
        raise ValueError("tax-return tie-aware sign loss mismatch")
    if _finite_number(tie_sign["coefficient"], "sign coefficient") < 0:
        raise ValueError("tax-return sign coefficient must be non-negative")
    if _finite_number(tie_sign["temperatureChips"], "sign temperature") <= 0:
        raise ValueError("tax-return sign temperature must be positive")
    if _finite_number(tie_sign["tieEpsilonChips"], "tie epsilon") < 0:
        raise ValueError("tax-return tie epsilon must be non-negative")
    routing = value["routing"]
    if not isinstance(routing, dict) or set(routing) != {
        "returnCountOne",
        "returnCountTwo",
        "roleRouting",
        "memberCount",
        "unanimityRule",
        "lowerConfidenceBound",
        "zValue",
        "defaultMinimumChipAdvantage",
        "selection",
        "tieBreak",
    }:
        raise ValueError("tax-return routing fields mismatch")
    if (
        routing["returnCountOne"] != "exact-normal-fallback"
        or routing["returnCountTwo"] != "ensemble-lower-confidence-bound"
        or routing["memberCount"] != TAX_RETURN_ADVANTAGE_MEMBER_COUNT
        or routing["unanimityRule"] != "all-member-advantages-strictly-positive"
        or routing["lowerConfidenceBound"] != "mean-minus-z-times-sample-sd"
        or routing["zValue"] != TAX_RETURN_ADVANTAGE_Z_VALUE
        or routing["selection"] != "maximum-eligible-lcb"
        or routing["tieBreak"] != "baseline-then-lowest-action-index"
    ):
        raise ValueError("unsupported tax-return routing contract")
    if routing["roleRouting"] != {
        "great-dalmuti": "ensemble-lower-confidence-bound",
        "lesser-dalmuti": "exact-normal-fallback",
        "other-roles": "not-applicable",
    }:
        raise ValueError("unsupported tax-return role routing")
    minimum = _finite_number(
        routing["defaultMinimumChipAdvantage"],
        "defaultMinimumChipAdvantage",
    )
    if minimum < 0:
        raise ValueError("defaultMinimumChipAdvantage must be non-negative")
    members = value["members"]
    if not isinstance(members, list) or len(members) != TAX_RETURN_ADVANTAGE_MEMBER_COUNT:
        raise ValueError("tax-return ensemble must contain exactly five members")
    seen_seeds: set[int] = set()
    for index, member in enumerate(members):
        if not isinstance(member, dict) or set(member) != {
            "memberIndex",
            "seed",
            "checkpointEpoch",
            "validationPairedLoss",
            "parametersSha256",
            "contextLayer",
            "bilinearWeight",
        }:
            raise ValueError(f"tax-return member {index} fields mismatch")
        if member["memberIndex"] != index:
            raise ValueError("tax-return member indices must be canonical")
        seed = member["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed in seen_seeds:
            raise ValueError("tax-return member seeds must be distinct non-negative integers")
        seen_seeds.add(seed)
        epoch = member["checkpointEpoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("tax-return member checkpointEpoch must be positive")
        loss = _finite_number(member["validationPairedLoss"], "validationPairedLoss")
        if loss < 0:
            raise ValueError("validationPairedLoss must be non-negative")
        expected_hash = member_parameters_sha256(member)
        if member["parametersSha256"] != expected_hash:
            raise ValueError("tax-return member parameter hash mismatch")
        model = TaxReturnBilinearResidualNetwork(context_features)
        load_layer_parameters(model, member)
    return value


def write_ensemble_json(value: Mapping[str, object], output_path: str | Path) -> None:
    validated = validate_ensemble_payload(dict(value))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(validated))


def read_ensemble_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid tax-return ensemble JSON: {source}") from error
    return validate_ensemble_payload(value)
