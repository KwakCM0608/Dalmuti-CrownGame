from __future__ import annotations

"""Leakage-safe V6 calibration and fixed-identity Normal evaluation.

The V5 evaluator intentionally accepts exactly ``V5PublicActor``.  V6 uses a
separate wrapper instead of weakening that sealed contract: public V5 features
are scored by three Normal-relative delta heads and an alternative is selected
only through :func:`v6_override.choose_safe_override`.

Calibration and screening are deliberately separate APIs.  A calibration run
is exactly 10 complete five-act matches for each p4..p10 (70 total) and has no
screening coordinate in its inputs or report.  Only after that report exists
can :func:`bind_v6_screening_plan` bind a disjoint 60-per-player-count (420
match) screening family to the immutable calibration report digest.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError as error:  # pure plan/audit tests run without torch
    if error.name != "torch":
        raise
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from v6_override import (
    BOOTSTRAP_HEADS,
    SAFE_OVERRIDE_CONTRACT,
    V6PublicDeltaConfig,
    V6PublicDeltaScorer,
    choose_safe_override,
    public_delta_api_has_no_privileged_input,
)


ACTS_PER_MATCH = 5
PLAYER_COUNTS = tuple(range(4, 11))
NAMESPACE_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
V6_EVALUATION_FORMAT = "dalmuti-v6-public-safe-override-normal-comparison"
V6_EVALUATION_VERSION = 1
V6_CALIBRATION_RECEIPT_FORMAT = "dalmuti-v6-calibration-receipt"
V6_CANDIDATE_FORMAT = "dalmuti-v6-public-safe-override-candidate"
V6_CANDIDATE_VERSION = 1
V6_PUBLIC_SCORER_ADAPTER_CONTRACT = "dalmuti-v6-public-batch-scorer-adapter-v1"
CALIBRATION_MATCH_COUNTS = {player_count: 10 for player_count in PLAYER_COUNTS}
SCREENING_MATCH_COUNTS = {player_count: 60 for player_count in PLAYER_COUNTS}
_BASE_ACTOR_IDENTITY_KEYS = {
    "actorSha256",
    "manifestSha256",
    "tensorStateSha256",
    "publicContractSha256",
    "policyNumericsSha256",
}
_CANDIDATE_IDENTITY_KEYS = {
    "format",
    "version",
    "baseActor",
    "publicDelta",
    "scorerConfigSha256",
    "safeOverrideContract",
}
_PUBLIC_DELTA_IDENTITY_KEYS = {
    "checkpointSha256",
    "format",
    "kind",
    "tensorStateSha256",
    "version",
}
_DECISION_AUDIT_FIELDS = {
    "normalAction",
    "selectedAction",
    "deviated",
    "forced",
    "bestAlternativeAction",
    "bestAlternativeLcb",
    "bestAlternativeUncertainty",
}
_DEVIATION_AUDIT_FIELDS = _DECISION_AUDIT_FIELDS | {
    "playerCount",
    "matchIndex",
    "decisionIndex",
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def derive_v5_evaluation_seed(
    family_id: str, seed_base: int, player_count: int, match_index: int
) -> int:
    value = int.from_bytes(
        hashlib.sha256(canonical_json_bytes([
            family_id,
            seed_base,
            "v5-exact-normal-evaluation-match",
            player_count,
            match_index,
        ])).digest()[:4],
        "little",
    )
    return value or 1


def _validate_coordinate(family_id: object, seed_base: object) -> tuple[str, int]:
    if (
        not isinstance(family_id, str)
        or not 1 <= len(family_id) <= 128
        or family_id[0] not in NAMESPACE_CHARACTERS - {".", "_", "-"}
        or any(character not in NAMESPACE_CHARACTERS for character in family_id)
    ):
        raise ValueError("family_id must use 1..128 safe ASCII characters")
    if (
        isinstance(seed_base, bool)
        or not isinstance(seed_base, int)
        or not 0 <= seed_base <= 0xFFFF_FFFF
    ):
        raise ValueError("seed_base must be uint32")
    return family_id, seed_base


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("V6 game evaluation and checkpoint loading require PyTorch")


def _require_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _threshold(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("threshold must be finite or positive infinity")
    result = float(value)
    if math.isnan(result) or result == -math.inf:
        raise ValueError("threshold must be finite or positive infinity")
    return result


def _json_threshold(value: float) -> float | str:
    return "positive-infinity" if value == math.inf else value


def _parse_json_threshold(value: object) -> float:
    if value == "positive-infinity":
        return math.inf
    return _threshold(value)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _assert_public_mapping(value: object, label: str) -> None:
    """Fail closed if a published identity/audit even names private material."""

    forbidden = ("privileged", "private", "opponenthand", "hiddenhand", "rawhand")

    def walk(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} has a non-string key at {path}")
                compact = "".join(character for character in key.lower() if character.isalnum())
                # Negative publication-boundary markers such as
                # ``containsRawHands: false`` are allowed; positive/private
                # payload fields are not.
                if any(token in compact for token in forbidden) and child is not False:
                    raise ValueError(f"{label} names private data at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, label)


def validate_v6_candidate_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_IDENTITY_KEYS:
        raise ValueError("V6 candidate identity fields drifted")
    result = dict(value)
    if result["format"] != V6_CANDIDATE_FORMAT or result["version"] != 1:
        raise ValueError("unsupported V6 candidate identity")
    if result["safeOverrideContract"] != SAFE_OVERRIDE_CONTRACT:
        raise ValueError("V6 candidate uses another safe-override contract")
    base = result["baseActor"]
    if not isinstance(base, Mapping) or set(base) != _BASE_ACTOR_IDENTITY_KEYS:
        raise ValueError("V6 candidate base Actor identity drifted")
    for key, digest in base.items():
        _require_sha(digest, f"baseActor.{key}")
    public = result["publicDelta"]
    if not isinstance(public, Mapping) or set(public) != _PUBLIC_DELTA_IDENTITY_KEYS:
        raise ValueError("V6 public-delta identity drifted")
    _require_sha(public["checkpointSha256"], "public delta checkpoint")
    _require_sha(public["tensorStateSha256"], "public delta tensor state")
    if (
        not isinstance(public["format"], str)
        or not public["format"]
        or type(public["version"]) is not int
        or int(public["version"]) < 1
        or public["kind"] != "public-delta-heads-only"
    ):
        raise ValueError("V6 public-delta format identity is invalid")
    _require_sha(result["scorerConfigSha256"], "V6 scorer config")
    _assert_public_mapping(result, "candidateIdentity")
    # This also proves it can be represented without NaN/Infinity or custom values.
    canonical_json_bytes(result)
    return result


def v6_candidate_identity_sha256(value: Mapping[str, object]) -> str:
    return _canonical_sha(validate_v6_candidate_identity(value))


@dataclass(frozen=True)
class V6PolicyParameters:
    beta: float = 1.0
    threshold: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta", _finite_nonnegative(self.beta, "beta"))
        object.__setattr__(self, "threshold", _threshold(self.threshold))

    def to_dict(self) -> dict[str, object]:
        return {
            "beta": self.beta,
            "threshold": _json_threshold(self.threshold),
            "selectionContract": SAFE_OVERRIDE_CONTRACT,
            "forcedActionPolicy": "exact-production-Normal",
        }


def _validate_policy_mapping(value: object) -> V6PolicyParameters:
    expected = {"beta", "threshold", "selectionContract", "forcedActionPolicy"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("V6 policy fields drifted")
    if (
        value["selectionContract"] != SAFE_OVERRIDE_CONTRACT
        or value["forcedActionPolicy"] != "exact-production-Normal"
    ):
        raise ValueError("V6 policy contract drifted")
    return V6PolicyParameters(
        beta=_finite_nonnegative(value["beta"], "beta"),
        threshold=_parse_json_threshold(value["threshold"]),
    )


@dataclass(frozen=True)
class V6CalibrationPlan:
    family_id: str
    seed_base: int
    policy: V6PolicyParameters
    lane_count: int = 32
    bootstrap_resamples: int = 10_000

    @property
    def stage(self) -> str:
        return "calibration"

    @property
    def v5_config(self) -> Any:
        _require_torch()
        from v5_evaluate import V5EvaluationConfig

        return V5EvaluationConfig(
            "screening",
            self.family_id,
            self.seed_base,
            match_counts=tuple(CALIBRATION_MATCH_COUNTS.items()),
            lane_count=self.lane_count,
            bootstrap_resamples=self.bootstrap_resamples,
        )

    def __post_init__(self) -> None:
        if type(self.policy) is not V6PolicyParameters:
            raise TypeError("calibration policy must be V6PolicyParameters")
        _validate_coordinate(self.family_id, self.seed_base)
        _positive_integer(self.lane_count, "lane_count")
        _positive_integer(self.bootstrap_resamples, "bootstrap_resamples")


@dataclass(frozen=True)
class V6ScreeningPlan:
    family_id: str
    seed_base: int
    policy: V6PolicyParameters
    calibration_receipt: Mapping[str, object]
    lane_count: int = 32
    bootstrap_resamples: int = 10_000

    @property
    def stage(self) -> str:
        return "screening"

    @property
    def v5_config(self) -> Any:
        _require_torch()
        from v5_evaluate import V5EvaluationConfig

        return V5EvaluationConfig(
            "screening",
            self.family_id,
            self.seed_base,
            match_counts=tuple(SCREENING_MATCH_COUNTS.items()),
            lane_count=self.lane_count,
            bootstrap_resamples=self.bootstrap_resamples,
        )

    def __post_init__(self) -> None:
        if type(self.policy) is not V6PolicyParameters:
            raise TypeError("screening policy must be V6PolicyParameters")
        _validate_coordinate(self.family_id, self.seed_base)
        _positive_integer(self.lane_count, "lane_count")
        _positive_integer(self.bootstrap_resamples, "bootstrap_resamples")
        receipt = validate_v6_calibration_receipt(self.calibration_receipt)
        if receipt["policy"] != self.policy.to_dict():
            raise ValueError("screening policy differs from calibrated policy")
        calibration_coordinate = receipt["coordinate"]
        assert isinstance(calibration_coordinate, Mapping)
        if calibration_coordinate == {
            "familyId": self.family_id,
            "seedBase": self.seed_base,
        }:
            raise ValueError("calibration and screening coordinates must differ")
        calibration_seeds = {
            derive_v5_evaluation_seed(
                str(calibration_coordinate["familyId"]),
                int(calibration_coordinate["seedBase"]),
                player_count,
                match_index,
            )
            for player_count, count in CALIBRATION_MATCH_COUNTS.items()
            for match_index in range(count)
        }
        screening_seeds = {
            derive_v5_evaluation_seed(
                self.family_id, self.seed_base, player_count, match_index
            )
            for player_count, count in SCREENING_MATCH_COUNTS.items()
            for match_index in range(count)
        }
        if calibration_seeds & screening_seeds:
            raise ValueError("calibration and screening derived match seeds overlap")


@dataclass(frozen=True)
class V6PublicScoreBatch:
    action_indices: np.ndarray
    action_mask: np.ndarray
    head_scores: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.action_indices.ndim != 2
            or self.action_indices.dtype != np.dtype(np.int64)
            or self.action_mask.shape != self.action_indices.shape
            or self.action_mask.dtype != np.dtype(np.bool_)
            or self.head_scores.shape
            != (*self.action_indices.shape, BOOTSTRAP_HEADS)
            or not np.issubdtype(self.head_scores.dtype, np.floating)
        ):
            raise ValueError("public score batch has incompatible arrays")
        if not self.action_mask.any(axis=1).all():
            raise ValueError("every public score row needs a legal action")
        if not np.isfinite(self.head_scores[self.action_mask]).all():
            raise ValueError("public legal action scores must be finite")
        self.action_indices.setflags(write=False)
        self.action_mask.setflags(write=False)
        self.head_scores.setflags(write=False)


class V6PublicBatchScorer(Protocol):
    public_only_contract: str

    def score_public(
        self, observations: Sequence[Any]
    ) -> V6PublicScoreBatch: ...


class TorchV6PublicBatchScorer:
    """Thin torch adapter whose only model input is a V5 public batch."""

    public_only_contract = V6_PUBLIC_SCORER_ADAPTER_CONTRACT

    def __init__(
        self,
        scorer: V6PublicDeltaScorer,
        device: object = "cpu",
    ) -> None:
        _require_torch()
        from v5_model import configure_v5_policy_numerics

        if type(scorer) is not V6PublicDeltaScorer:
            raise TypeError("torch V6 adapter requires exactly V6PublicDeltaScorer")
        if not public_delta_api_has_no_privileged_input():
            raise RuntimeError("V6 public scorer API unexpectedly accepts private input")
        if float(scorer.public_actor.config.dropout) != 0.0:
            raise ValueError("V6 evaluation requires public Actor dropout=0")
        self.device = torch.device(device)
        self.policy_numerics = configure_v5_policy_numerics(self.device)
        self.scorer = scorer.to(self.device).eval()

    def score_public(
        self, observations: Sequence[Any]
    ) -> V6PublicScoreBatch:
        from v5_public import (
            V5PublicObservation,
            stack_v5_actor_public_features,
            tensorize_v5_public_observation,
        )

        if not observations or any(
            type(value) is not V5PublicObservation for value in observations
        ):
            raise TypeError("V6 scorer requires non-empty exact public observations")
        batch = stack_v5_actor_public_features(
            [tensorize_v5_public_observation(value) for value in observations],
            device=self.device,
        )
        with torch.inference_mode():
            output = self.scorer(batch)
        return V6PublicScoreBatch(
            output.action_indices.detach().cpu().numpy().astype(np.int64, copy=False),
            output.action_mask.detach().cpu().numpy().astype(np.bool_, copy=False),
            output.head_scores.detach().cpu().numpy(),
        )


@dataclass(frozen=True)
class V6PolicyBatchResult:
    actions: tuple[int, ...]
    audits: tuple[dict[str, object], ...]


class V6SafeOverridePolicy:
    """Apply the pure public three-head conservative override to a batch."""

    def __init__(
        self,
        scorer: V6PublicBatchScorer,
        parameters: V6PolicyParameters,
    ) -> None:
        if getattr(scorer, "public_only_contract", None) != (
            V6_PUBLIC_SCORER_ADAPTER_CONTRACT
        ):
            raise TypeError("V6 scorer adapter did not attest the public-only contract")
        if type(parameters) is not V6PolicyParameters:
            raise TypeError("parameters must be V6PolicyParameters")
        self.scorer = scorer
        self.parameters = parameters

    def actions(
        self,
        observations: Sequence[Any],
        normal_actions: Sequence[int],
    ) -> V6PolicyBatchResult:
        if len(observations) != len(normal_actions):
            raise TypeError("V6 policy requires matched public observations/actions")
        if not observations:
            return V6PolicyBatchResult((), ())

        actions: list[int | None] = [None] * len(observations)
        audits: list[dict[str, object] | None] = [None] * len(observations)
        scored_positions: list[int] = []
        scored_observations: list[Any] = []
        for index, (observation, normal_raw) in enumerate(
            zip(observations, normal_actions, strict=True)
        ):
            normal = int(normal_raw)
            raw_legal = getattr(observation, "legal_mask", None)
            legal = np.asarray(raw_legal)
            if legal.dtype != np.dtype(np.bool_) or legal.ndim != 1 or legal.size != 236:
                raise TypeError("V6 policy observation needs canonical bool legal_mask[236]")
            legal_actions = np.flatnonzero(legal)
            if legal_actions.size < 1 or not 0 <= normal < legal.size or not legal[normal]:
                raise ValueError("production Normal must be a legal catalogue action")
            if legal_actions.size == 1:
                # Forced decisions bypass the learned scorer entirely.
                forced = int(legal_actions[0])
                if normal != forced:
                    raise RuntimeError("production Normal disagrees with the forced action")
                actions[index] = normal
                audits[index] = {
                    "normalAction": normal,
                    "selectedAction": normal,
                    "deviated": False,
                    "forced": True,
                    "bestAlternativeAction": None,
                    "bestAlternativeLcb": None,
                    "bestAlternativeUncertainty": None,
                }
            else:
                scored_positions.append(index)
                scored_observations.append(observation)

        if scored_observations:
            scored = self.scorer.score_public(scored_observations)
            if scored.action_indices.shape[0] != len(scored_observations):
                raise ValueError("public scorer returned another batch size")
            for row_index, original_index in enumerate(scored_positions):
                observation = observations[original_index]
                normal = int(normal_actions[original_index])
                ids = scored.action_indices[row_index]
                mask = scored.action_mask[row_index]
                represented = np.zeros_like(observation.legal_mask, dtype=np.bool_)
                legal_ids = ids[mask]
                if (
                    np.any((legal_ids < 0) | (legal_ids >= represented.size))
                    or np.unique(legal_ids).size != legal_ids.size
                ):
                    raise ValueError("public scorer packed invalid legal action IDs")
                represented[legal_ids] = True
                if not np.array_equal(represented, observation.legal_mask):
                    raise ValueError("public scorer changed the observation legal mask")
                decision = choose_safe_override(
                    action_ids=ids,
                    legal_mask=mask,
                    head_scores=scored.head_scores[row_index],
                    normal_action=normal,
                    beta=self.parameters.beta,
                    threshold=self.parameters.threshold,
                )
                best_position = (
                    None
                    if decision.best_alternative_id is None
                    else int(np.flatnonzero(
                        decision.action_ids == decision.best_alternative_id
                    )[0])
                )
                actions[original_index] = decision.action_id
                audits[original_index] = {
                    "normalAction": decision.normal_action_id,
                    "selectedAction": decision.action_id,
                    "deviated": decision.overridden,
                    "forced": False,
                    "bestAlternativeAction": decision.best_alternative_id,
                    "bestAlternativeLcb": decision.best_lcb,
                    "bestAlternativeUncertainty": (
                        None
                        if best_position is None
                        else float(decision.delta_std[best_position])
                    ),
                }

        if any(value is None for value in actions) or any(value is None for value in audits):
            raise RuntimeError("V6 policy failed to resolve every decision")
        verified_audits: list[dict[str, object]] = []
        for audit in audits:
            assert isinstance(audit, dict)
            if set(audit) != _DECISION_AUDIT_FIELDS:
                raise RuntimeError("V6 decision audit fields drifted")
            _assert_public_mapping(audit, "decisionAudit")
            verified_audits.append(audit)
        return V6PolicyBatchResult(
            tuple(int(value) for value in actions),  # type: ignore[arg-type]
            tuple(verified_audits),
        )


@dataclass(frozen=True)
class V6EvaluationCollection:
    match_clusters: tuple[dict[str, object], ...]
    decision_audits: tuple[dict[str, object], ...]


def collect_v6_evaluation_clusters(
    policy: V6SafeOverridePolicy,
    config: Any,
) -> V6EvaluationCollection:
    """Collect exact V5-compatible match clusters with a separate public audit."""

    _require_torch()
    from v4_collect_fixed_match_ppo import evaluator_group_reward_components
    from v5_evaluate import V5EvaluationConfig, _finish_lane, _new_lane
    from v5_public import V5PublicObservation, v5_public_from_v4_actor_observation

    if type(config) is not V5EvaluationConfig:
        raise TypeError("V6 collection requires exact V5EvaluationConfig")
    if type(policy) is not V6SafeOverridePolicy:
        raise TypeError("V6 collection requires V6SafeOverridePolicy")
    if config.mode != "screening":
        raise ValueError("V6 calibration/screening adapter only accepts screening mode")
    specs = [
        (player_count, match_index)
        for player_count, count in sorted(config.resolved_match_counts.items())
        for match_index in range(count)
        if match_index % config.match_shard_count == config.match_shard_index
    ]
    if not specs:
        raise ValueError("requested V6 evaluation shard is empty")
    seeds = {
        derive_v5_evaluation_seed(
            config.family_id, config.seed_base, player_count, match_index
        )
        for player_count, match_index in specs
    }
    if len(seeds) != len(specs):
        raise RuntimeError("V6 derived match seeds collided")

    next_spec = min(len(specs), config.lane_count)
    lanes = [_new_lane(config, *spec) for spec in specs[:next_spec]]
    complete: list[dict[str, object]] = []
    decision_audits: list[dict[str, object]] = []
    while lanes:
        candidate_lanes = []
        candidate_publics: list[V5PublicObservation] = []
        candidate_normals: list[int] = []
        actions: dict[tuple[int, int], int] = {}
        for lane in lanes:
            normal = int(lane.env.normal_action())
            if not bool(lane.observation.public.legal_mask[normal]):
                raise RuntimeError("production Normal selected an illegal action")
            if int(lane.env.current_player_id) in lane.candidate_ids:
                candidate_lanes.append(lane)
                candidate_publics.append(
                    v5_public_from_v4_actor_observation(lane.observation.public)
                )
                candidate_normals.append(normal)
            else:
                actions[(lane.player_count, lane.match_index)] = normal
        batch_result = policy.actions(candidate_publics, candidate_normals)
        for lane, action, audit in zip(
            candidate_lanes,
            batch_result.actions,
            batch_result.audits,
            strict=True,
        ):
            key = (lane.player_count, lane.match_index)
            actions[key] = action
            contextual = {
                "playerCount": lane.player_count,
                "matchIndex": lane.match_index,
                "decisionIndex": lane.decisions + 1,
                **audit,
            }
            if set(contextual) != _DEVIATION_AUDIT_FIELDS:
                raise RuntimeError("V6 contextual audit fields drifted")
            decision_audits.append(contextual)

        remaining = []
        for lane in lanes:
            key = (lane.player_count, lane.match_index)
            expected_candidate_ids = frozenset(
                lane.initial_order[seat] for seat in lane.candidate_initial_seats
            )
            if lane.candidate_ids != expected_candidate_ids:
                raise RuntimeError("candidate identity routing changed")
            lane.decisions += 1
            result = lane.env.step(actions[key])
            lane.observation = result.observation
            if result.act_ended:
                act_result = result.info.get("act_result")
                if not isinstance(act_result, Mapping):
                    raise RuntimeError("ended act omitted its exact result")
                finish_order = tuple(int(value) for value in act_result["finish_order"])
                chip_awards = act_result["chip_awards"]
                if (
                    len(finish_order) != lane.player_count
                    or set(finish_order) != set(lane.initial_order)
                    or not isinstance(chip_awards, Mapping)
                ):
                    raise RuntimeError("act result changed physical player identities")
                components = evaluator_group_reward_components(
                    finish_order, chip_awards, lane.candidate_ids
                )
                lane.act_records.append({
                    "act": len(lane.act_records) + 1,
                    "finishOrder": list(finish_order),
                    "candidatePhysicalIds": sorted(lane.candidate_ids),
                    "candidateMeanChip": components[0],
                    "normalMeanChip": components[1],
                    "meanChipDifference": components[2],
                    "candidateBefore": components[3],
                    "comparisons": components[4],
                    "pairwiseRate": components[5],
                })
            if result.terminated:
                complete.append(_finish_lane(lane))
                if next_spec < len(specs):
                    remaining.append(_new_lane(config, *specs[next_spec]))
                    next_spec += 1
            else:
                remaining.append(lane)
        lanes = remaining

    complete.sort(key=lambda item: (int(item["playerCount"]), int(item["matchIndex"])))
    decision_audits.sort(
        key=lambda item: (
            int(item["playerCount"]),
            int(item["matchIndex"]),
            int(item["decisionIndex"]),
        )
    )
    if len(complete) != len(specs):
        raise RuntimeError("V6 evaluation did not complete every requested match")
    return V6EvaluationCollection(tuple(complete), tuple(decision_audits))


def _distribution(values: Sequence[float]) -> dict[str, object] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("V6 public audit contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def summarize_v6_action_audit(
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    validated: list[dict[str, object]] = []
    for raw in events:
        if not isinstance(raw, Mapping) or set(raw) != _DEVIATION_AUDIT_FIELDS:
            raise ValueError("V6 public decision audit fields drifted")
        event = dict(raw)
        _assert_public_mapping(event, "decisionAudit")
        for key in ("playerCount", "matchIndex", "decisionIndex", "normalAction", "selectedAction"):
            if type(event[key]) is not int:
                raise ValueError(f"V6 audit {key} must be integer")
        if event["playerCount"] not in PLAYER_COUNTS:
            raise ValueError("V6 audit player count is invalid")
        if event["matchIndex"] < 0 or event["decisionIndex"] < 1:
            raise ValueError("V6 audit coordinate is invalid")
        if type(event["deviated"]) is not bool or type(event["forced"]) is not bool:
            raise ValueError("V6 audit flags must be bool")
        if event["deviated"] != (event["selectedAction"] != event["normalAction"]):
            raise ValueError("V6 audit deviation flag disagrees with actions")
        if event["forced"] and (
            event["deviated"]
            or any(event[key] is not None for key in (
                "bestAlternativeAction", "bestAlternativeLcb",
                "bestAlternativeUncertainty",
            ))
        ):
            raise ValueError("forced V6 audit must be exact Normal without scores")
        if not event["forced"]:
            if type(event["bestAlternativeAction"]) is not int:
                raise ValueError("scored V6 audit omitted its best alternative")
            for key in (
                "bestAlternativeLcb", "bestAlternativeUncertainty",
            ):
                if isinstance(event[key], bool) or not math.isfinite(float(event[key])):
                    raise ValueError("scored V6 audit contains an invalid public statistic")
        validated.append(event)

    keys = [
        (int(item["playerCount"]), int(item["matchIndex"]), int(item["decisionIndex"]))
        for item in validated
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("V6 decision audit duplicated a decision coordinate")
    validated.sort(key=lambda item: (
        int(item["playerCount"]), int(item["matchIndex"]), int(item["decisionIndex"])
    ))

    def aggregate(group: Sequence[Mapping[str, object]]) -> dict[str, object]:
        forced = sum(int(bool(item["forced"])) for item in group)
        deviations = [dict(item) for item in group if bool(item["deviated"])]
        scored = [item for item in group if not bool(item["forced"])]
        return {
            "candidateDecisions": len(group),
            "forcedDecisions": forced,
            "scoredDecisions": len(scored),
            "deviations": len(deviations),
            "deviationRateAmongScored": (
                len(deviations) / len(scored) if scored else 0.0
            ),
            "bestAlternativeLcb": _distribution([
                float(item["bestAlternativeLcb"]) for item in scored
            ]),
            "bestAlternativeUncertainty": _distribution([
                float(item["bestAlternativeUncertainty"]) for item in scored
            ]),
            # Publishing only deviations keeps reports bounded; no observation or hand
            # tensor is ever copied into this module.
            "deviationRecords": deviations,
        }

    by_player_count = [
        {"playerCount": player_count, **aggregate([
            item for item in validated if item["playerCount"] == player_count
        ])}
        for player_count in PLAYER_COUNTS
        if any(item["playerCount"] == player_count for item in validated)
    ]
    result = {
        "contract": "dalmuti-v6-public-deviation-lcb-uncertainty-audit-v1",
        "publishedDecisionFields": sorted(_DEVIATION_AUDIT_FIELDS),
        "containsObservationFeatures": False,
        "overall": aggregate(validated),
        "byPlayerCount": by_player_count,
        "completeDecisionAuditSha256": _canonical_sha(validated),
    }
    _assert_public_mapping(result, "actionAudit")
    return result


def _coordinate(plan: V6CalibrationPlan | V6ScreeningPlan) -> dict[str, object]:
    return {"familyId": plan.family_id, "seedBase": plan.seed_base}


def summarize_v6_evaluation(
    collection: V6EvaluationCollection,
    plan: V6CalibrationPlan | V6ScreeningPlan,
    candidate_identity: Mapping[str, object],
) -> dict[str, object]:
    _require_torch()
    from v5_evaluate import (
        summarize_v5_evaluation_clusters,
        validate_v5_evaluation_report,
    )

    identity = validate_v6_candidate_identity(candidate_identity)
    config = plan.v5_config
    statistics = summarize_v5_evaluation_clusters(collection.match_clusters, config)
    validate_v5_evaluation_report(statistics)
    report: dict[str, object] = {
        "format": V6_EVALUATION_FORMAT,
        "version": V6_EVALUATION_VERSION,
        "stage": plan.stage,
        "coordinate": _coordinate(plan),
        "matchPlan": {
            str(key): value for key, value in sorted(config.resolved_match_counts.items())
        },
        "candidate": identity,
        "candidateIdentitySha256": v6_candidate_identity_sha256(identity),
        "policy": plan.policy.to_dict(),
        "normalComparison": statistics,
        "actionAudit": summarize_v6_action_audit(collection.decision_audits),
        "publicationBoundary": {
            "publicInputsOnly": True,
            "publishedModelAudit": "candidate-action-deviation-lcb-uncertainty-only",
            "containsRawHands": False,
            "containsObservationFeatures": False,
        },
    }
    if isinstance(plan, V6ScreeningPlan):
        report["calibrationBinding"] = validate_v6_calibration_receipt(
            plan.calibration_receipt
        )
    _assert_public_mapping(report, "evaluationReport")
    return report


def evaluate_v6_calibration(
    policy: V6SafeOverridePolicy,
    plan: V6CalibrationPlan,
    candidate_identity: Mapping[str, object],
) -> dict[str, object]:
    if policy.parameters != plan.policy:
        raise ValueError("runtime V6 policy differs from calibration plan")
    return summarize_v6_evaluation(
        collect_v6_evaluation_clusters(policy, plan.v5_config),
        plan,
        candidate_identity,
    )


def evaluate_v6_screening(
    policy: V6SafeOverridePolicy,
    plan: V6ScreeningPlan,
    candidate_identity: Mapping[str, object],
) -> dict[str, object]:
    if policy.parameters != plan.policy:
        raise ValueError("runtime V6 policy differs from screening plan")
    receipt = validate_v6_calibration_receipt(plan.calibration_receipt)
    if receipt["candidateIdentitySha256"] != v6_candidate_identity_sha256(
        candidate_identity
    ):
        raise ValueError("screening candidate differs from calibrated candidate")
    return summarize_v6_evaluation(
        collect_v6_evaluation_clusters(policy, plan.v5_config),
        plan,
        candidate_identity,
    )


def validate_v6_evaluation_report(value: object) -> dict[str, object]:
    _require_torch()
    from v5_evaluate import V5EvaluationConfig, validate_v5_evaluation_report

    base_fields = {
        "format", "version", "stage", "coordinate", "matchPlan", "candidate",
        "candidateIdentitySha256", "policy", "normalComparison", "actionAudit",
        "publicationBoundary",
    }
    if not isinstance(value, Mapping):
        raise ValueError("V6 evaluation report must be an object")
    report = dict(value)
    stage = report.get("stage")
    expected_fields = base_fields | ({"calibrationBinding"} if stage == "screening" else set())
    if set(report) != expected_fields:
        raise ValueError("V6 evaluation report fields drifted")
    if (
        report["format"] != V6_EVALUATION_FORMAT
        or report["version"] != V6_EVALUATION_VERSION
        or stage not in {"calibration", "screening"}
    ):
        raise ValueError("unsupported V6 evaluation report")
    identity = validate_v6_candidate_identity(report["candidate"])
    if report["candidateIdentitySha256"] != v6_candidate_identity_sha256(identity):
        raise ValueError("V6 report candidate digest disagrees")
    policy = _validate_policy_mapping(report["policy"])
    coordinate = report["coordinate"]
    if not isinstance(coordinate, Mapping) or set(coordinate) != {"familyId", "seedBase"}:
        raise ValueError("V6 report coordinate fields drifted")
    counts = CALIBRATION_MATCH_COUNTS if stage == "calibration" else SCREENING_MATCH_COUNTS
    expected_plan = {str(key): value for key, value in counts.items()}
    if report["matchPlan"] != expected_plan:
        raise ValueError("V6 report stage has the wrong match plan")
    config = V5EvaluationConfig(
        "screening",
        str(coordinate["familyId"]),
        int(coordinate["seedBase"]),
        match_counts=tuple(counts.items()),
        bootstrap_resamples=int(
            report["normalComparison"]["results"][0]["matchClustered95"]["resamples"]  # type: ignore[index]
        ),
    )
    statistics = validate_v5_evaluation_report(report["normalComparison"])  # type: ignore[arg-type]
    if (
        statistics["familyId"] != config.family_id
        or statistics["seedBase"] != config.seed_base
        or statistics["matchPlan"] != expected_plan
    ):
        raise ValueError("V6 report and Normal-comparison coordinates disagree")
    audit = report["actionAudit"]
    if not isinstance(audit, Mapping) or audit.get("containsObservationFeatures") is not False:
        raise ValueError("V6 report public audit boundary drifted")
    boundary = report["publicationBoundary"]
    if boundary != {
        "publicInputsOnly": True,
        "publishedModelAudit": "candidate-action-deviation-lcb-uncertainty-only",
        "containsRawHands": False,
        "containsObservationFeatures": False,
    }:
        raise ValueError("V6 report publication boundary drifted")
    _assert_public_mapping(report, "evaluationReport")
    if stage == "screening":
        receipt = validate_v6_calibration_receipt(report["calibrationBinding"])
        if receipt["candidateIdentitySha256"] != report["candidateIdentitySha256"]:
            raise ValueError("V6 screening calibration candidate binding drifted")
        if receipt["policy"] != policy.to_dict():
            raise ValueError("V6 screening calibration policy binding drifted")
        # Re-run coordinate and seed-disjoint validation.
        V6ScreeningPlan(
            config.family_id,
            config.seed_base,
            policy,
            receipt,
            bootstrap_resamples=config.bootstrap_resamples,
        )
    canonical_json_bytes(report)
    return report


def build_v6_calibration_receipt(
    calibration_report: Mapping[str, object],
) -> dict[str, object]:
    report = validate_v6_evaluation_report(calibration_report)
    if report["stage"] != "calibration":
        raise ValueError("only a calibration report can create a receipt")
    # There is intentionally no screening family or seed in this object.
    return {
        "format": V6_CALIBRATION_RECEIPT_FORMAT,
        "version": 1,
        "calibrationReportSha256": _canonical_sha(report),
        "coordinate": dict(report["coordinate"]),  # type: ignore[arg-type]
        "matchPlan": dict(report["matchPlan"]),  # type: ignore[arg-type]
        "candidateIdentitySha256": report["candidateIdentitySha256"],
        "policy": dict(report["policy"]),  # type: ignore[arg-type]
    }


def validate_v6_calibration_receipt(value: object) -> dict[str, object]:
    fields = {
        "format", "version", "calibrationReportSha256", "coordinate",
        "matchPlan", "candidateIdentitySha256", "policy",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("V6 calibration receipt fields drifted")
    receipt = dict(value)
    if receipt["format"] != V6_CALIBRATION_RECEIPT_FORMAT or receipt["version"] != 1:
        raise ValueError("unsupported V6 calibration receipt")
    _require_sha(receipt["calibrationReportSha256"], "calibration report")
    _require_sha(receipt["candidateIdentitySha256"], "calibration candidate")
    coordinate = receipt["coordinate"]
    if not isinstance(coordinate, Mapping) or set(coordinate) != {"familyId", "seedBase"}:
        raise ValueError("V6 calibration coordinate fields drifted")
    # Validate family and seed without allowing any screening coordinate input.
    _validate_coordinate(coordinate["familyId"], coordinate["seedBase"])
    if receipt["matchPlan"] != {
        str(key): value for key, value in CALIBRATION_MATCH_COUNTS.items()
    }:
        raise ValueError("V6 calibration receipt match plan drifted")
    _validate_policy_mapping(receipt["policy"])
    canonical_json_bytes(receipt)
    return receipt


def bind_v6_screening_plan(
    calibration_report: Mapping[str, object],
    *,
    family_id: str,
    seed_base: int,
    lane_count: int = 32,
    bootstrap_resamples: int = 10_000,
) -> V6ScreeningPlan:
    """Bind screening only after calibration bytes and policy are fixed."""

    report = validate_v6_evaluation_report(calibration_report)
    if report["stage"] != "calibration":
        raise ValueError("screening requires a calibration-stage report")
    policy = _validate_policy_mapping(report["policy"])
    return V6ScreeningPlan(
        family_id,
        seed_base,
        policy,
        build_v6_calibration_receipt(report),
        lane_count,
        bootstrap_resamples,
    )


def validate_v6_screening_calibration_binding(
    screening_report: Mapping[str, object],
    calibration_report: Mapping[str, object],
) -> dict[str, object]:
    """Verify a screening report against the exact external calibration bytes."""

    screening = validate_v6_evaluation_report(screening_report)
    calibration = validate_v6_evaluation_report(calibration_report)
    if screening["stage"] != "screening" or calibration["stage"] != "calibration":
        raise ValueError("binding requires screening and calibration reports in order")
    expected = build_v6_calibration_receipt(calibration)
    if screening["calibrationBinding"] != expected:
        raise ValueError("screening report is not bound to the supplied calibration report")
    if screening["candidateIdentitySha256"] != calibration["candidateIdentitySha256"]:
        raise ValueError("screening and calibration candidates differ")
    if screening["policy"] != calibration["policy"]:
        raise ValueError("screening and calibration policies differ")
    return screening


def _load_public_delta_payload(path: Path) -> dict[str, object]:
    _require_torch()
    from v5_export import tensor_state_sha256

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("V6 public delta checkpoint could not be safely loaded") from error
    required = {
        "format", "version", "kind", "privilegedInputAllowed",
        "containsRawPrivateRows", "config", "baseActor",
        "publicActorDModel", "tensorStateSha256", "stateDict",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("V6 public delta checkpoint fields drifted")
    if (
        not isinstance(value["format"], str)
        or not value["format"]
        or type(value["version"]) is not int
        or value["version"] < 1
        or value["kind"] != "public-delta-heads-only"
        or value["privilegedInputAllowed"] is not False
        or value["containsRawPrivateRows"] is not False
        or type(value["publicActorDModel"]) is not int
        or value["publicActorDModel"] < 1
        or not isinstance(value["stateDict"], dict)
        or any(
            token in str(name).lower()
            for name in value["stateDict"]
            for token in ("privileged", "private", "central")
        )
        or value["tensorStateSha256"] != tensor_state_sha256(value["stateDict"])
    ):
        raise ValueError("V6 public delta checkpoint violated its public-only contract")
    validate_v6_candidate_base_actor(value["baseActor"])
    config = value["config"]
    if not isinstance(config, Mapping):
        raise ValueError("V6 public delta scorer config is missing")
    parsed = V6PublicDeltaConfig(**dict(config))
    if asdict(parsed) != dict(config):
        raise ValueError("V6 public delta scorer config is non-canonical")
    return value


def validate_v6_candidate_base_actor(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _BASE_ACTOR_IDENTITY_KEYS:
        raise ValueError("V6 public delta base Actor identity drifted")
    result = dict(value)
    for key, digest in result.items():
        _require_sha(digest, f"baseActor.{key}")
    return result


def load_v6_public_candidate(
    base_actor_bundle: str | Path,
    public_delta_checkpoint: str | Path,
    *,
    device: object = "cpu",
) -> tuple[TorchV6PublicBatchScorer, dict[str, object]]:
    """Load the current public-head payload through a deliberately thin adapter."""

    _require_torch()
    from v5_export import (
        load_v5_actor_bundle,
        v5_actor_bundle_digests,
    )

    actor_root = Path(base_actor_bundle).resolve(strict=True)
    checkpoint = Path(public_delta_checkpoint).resolve(strict=True)
    payload = _load_public_delta_payload(checkpoint)
    actor, _ = load_v5_actor_bundle(actor_root)
    base_identity = v5_actor_bundle_digests(actor_root)
    if payload["baseActor"] != base_identity:
        raise ValueError("V6 public delta checkpoint belongs to another base Actor")
    if payload["publicActorDModel"] != actor.config.d_model:
        raise ValueError("V6 public delta checkpoint Actor width drifted")
    config = V6PublicDeltaConfig(**dict(payload["config"]))  # type: ignore[arg-type]
    scorer = V6PublicDeltaScorer(actor, config)
    try:
        scorer.delta_heads.load_state_dict(payload["stateDict"], strict=True)  # type: ignore[arg-type]
    except Exception as error:
        raise ValueError("V6 public delta state does not fit the declared scorer") from error
    candidate = validate_v6_candidate_identity({
        "format": V6_CANDIDATE_FORMAT,
        "version": V6_CANDIDATE_VERSION,
        "baseActor": base_identity,
        "publicDelta": {
            "checkpointSha256": sha256_file(checkpoint),
            "format": payload["format"],
            "version": payload["version"],
            "kind": payload["kind"],
            "tensorStateSha256": payload["tensorStateSha256"],
        },
        "scorerConfigSha256": _canonical_sha(dict(payload["config"])),  # type: ignore[arg-type]
        "safeOverrideContract": SAFE_OVERRIDE_CONTRACT,
    })
    return TorchV6PublicBatchScorer(scorer, device=device), candidate


def write_v6_evaluation_report(path: str | Path, report: Mapping[str, object]) -> str:
    verified = validate_v6_evaluation_report(report)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(verified)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = target.with_name(target.name + ".sha256")
    if target.exists() or sidecar.exists():
        raise FileExistsError("immutable V6 evaluation output already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        report_stage = staging / target.name
        sidecar_stage = staging / sidecar.name
        report_stage.write_bytes(raw)
        sidecar_stage.write_bytes(f"{digest}  {target.name}\n".encode("ascii"))
        os.replace(report_stage, target)
        os.replace(sidecar_stage, sidecar)
    finally:
        try:
            staging.rmdir()
        except OSError:
            pass
    return digest


def load_v6_evaluation_report(path: str | Path) -> dict[str, object]:
    target = Path(path).resolve(strict=True)
    raw = target.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if raw != canonical_json_bytes(value):
        raise ValueError("V6 evaluation report is not canonical JSON")
    sidecar = target.with_name(target.name + ".sha256")
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar.read_bytes() != f"{digest}  {target.name}\n".encode("ascii"):
        raise ValueError("V6 evaluation report checksum sidecar disagrees")
    return validate_v6_evaluation_report(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "screen"):
        command = commands.add_parser(name)
        command.add_argument("--base-actor-bundle", required=True, type=Path)
        command.add_argument("--public-delta-checkpoint", required=True, type=Path)
        command.add_argument("--family-id", required=True)
        command.add_argument("--seed-base", required=True, type=int)
        command.add_argument("--device", default="cpu")
        command.add_argument("--lane-count", type=int, default=32)
        command.add_argument("--output", required=True, type=Path)
    calibration = commands.choices["calibrate"]
    calibration.add_argument("--beta", type=float, default=1.0)
    calibration.add_argument("--threshold", type=float, required=True)
    screening = commands.choices["screen"]
    screening.add_argument("--calibration-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    scorer, identity = load_v6_public_candidate(
        arguments.base_actor_bundle,
        arguments.public_delta_checkpoint,
        device=arguments.device,
    )
    if arguments.command == "calibrate":
        parameters = V6PolicyParameters(arguments.beta, arguments.threshold)
        plan: V6CalibrationPlan | V6ScreeningPlan = V6CalibrationPlan(
            arguments.family_id,
            arguments.seed_base,
            parameters,
            arguments.lane_count,
        )
        report = evaluate_v6_calibration(
            V6SafeOverridePolicy(scorer, parameters), plan, identity
        )
    else:
        calibration = load_v6_evaluation_report(arguments.calibration_report)
        plan = bind_v6_screening_plan(
            calibration,
            family_id=arguments.family_id,
            seed_base=arguments.seed_base,
            lane_count=arguments.lane_count,
        )
        report = evaluate_v6_screening(
            V6SafeOverridePolicy(scorer, plan.policy), plan, identity
        )
    digest = write_v6_evaluation_report(arguments.output, report)
    print(json.dumps({"output": str(arguments.output.resolve()), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_MATCH_COUNTS",
    "SCREENING_MATCH_COUNTS",
    "TorchV6PublicBatchScorer",
    "V6CalibrationPlan",
    "V6EvaluationCollection",
    "V6PolicyBatchResult",
    "V6PolicyParameters",
    "V6PublicScoreBatch",
    "V6SafeOverridePolicy",
    "V6ScreeningPlan",
    "bind_v6_screening_plan",
    "build_v6_calibration_receipt",
    "collect_v6_evaluation_clusters",
    "evaluate_v6_calibration",
    "evaluate_v6_screening",
    "load_v6_evaluation_report",
    "load_v6_public_candidate",
    "summarize_v6_action_audit",
    "summarize_v6_evaluation",
    "validate_v6_calibration_receipt",
    "validate_v6_candidate_identity",
    "validate_v6_evaluation_report",
    "validate_v6_screening_calibration_binding",
    "v6_candidate_identity_sha256",
    "write_v6_evaluation_report",
]
