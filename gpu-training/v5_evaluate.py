from __future__ import annotations

"""Exact fixed-identity V5 Actor evaluation against production Normal."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from v4_collect_fixed_match_ppo import (
    ACTS_PER_MATCH,
    evaluation_candidate_initial_seats,
    evaluator_group_reward_components,
)
from v4_collect_ppo import NAMESPACE_CHARACTERS, _derive_uint32
from v4_evaluate import deterministic_cluster_bootstrap95
from v4_env import DalmutiScalarEnv, V4EnvironmentObservation
from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_export import (
    canonical_json_bytes,
    load_v5_actor_bundle,
    sha256_file,
    tensor_state_sha256,
    v5_actor_bundle_digests,
)
from v5_model import (
    V5_POLICY_NUMERICS_SHA256,
    V5PublicActor,
    configure_v5_policy_numerics,
)
from v5_public import (
    V5PublicObservation,
    stack_v5_actor_public_features,
    tensorize_v5_public_observation,
    v5_public_from_v4_actor_observation,
)
from v5_provenance import (
    build_v5_evaluation_provenance,
    validate_v5_evaluation_provenance,
)


V5_EVALUATION_FORMAT = "dalmuti-v5-fixed-identity-normal-comparison"
V5_EVALUATION_VERSION = 1
PLAYER_COUNTS = tuple(range(4, 11))
SCREENING_MATCH_COUNTS = {player_count: 60 for player_count in PLAYER_COUNTS}
FINAL_MATCH_COUNTS = {
    4: 2500,
    5: 1700,
    6: 900,
    7: 600,
    8: 400,
    9: 400,
    10: 300,
}
EXACT_GATES = {
    "minMeanChipDifference": 0.25,
    "minCluster95LowerBound": 0.15,
    "minPairwiseRate": 0.55,
}
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
EVALUATION_MODES = ("screening", "certification", "final")
_FINAL_MODEL_IDENTITY_KEYS = {
    "actorSha256",
    "manifestSha256",
    "tensorStateSha256",
    "publicContractSha256",
    "policyNumericsSha256",
}
_PROVENANCE_RECORD_KEYS = {"provenance", "shard"}


@dataclass(frozen=True)
class V5EvaluationConfig:
    mode: str
    family_id: str
    seed_base: int
    match_counts: tuple[tuple[int, int], ...] | None = None
    match_shard_count: int = 1
    match_shard_index: int = 0
    lane_count: int = 32
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES

    def __post_init__(self) -> None:
        if self.mode not in EVALUATION_MODES:
            raise ValueError(f"mode must be one of {EVALUATION_MODES}")
        if (
            not isinstance(self.family_id, str)
            or not 1 <= len(self.family_id) <= 128
            or self.family_id[0] not in NAMESPACE_CHARACTERS - {".", "_", "-"}
            or any(value not in NAMESPACE_CHARACTERS for value in self.family_id)
        ):
            raise ValueError("family_id must use 1..128 safe ASCII characters")
        if (
            isinstance(self.seed_base, bool)
            or not isinstance(self.seed_base, int)
            or not 0 <= self.seed_base <= 0xFFFF_FFFF
        ):
            raise ValueError("seed_base must be uint32")
        for name in ("match_shard_count", "lane_count", "bootstrap_resamples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.match_shard_index, bool)
            or not isinstance(self.match_shard_index, int)
            or not 0 <= self.match_shard_index < self.match_shard_count
        ):
            raise ValueError("match_shard_index/count are invalid")
        if self.match_counts is not None:
            if not isinstance(self.match_counts, tuple) or not self.match_counts:
                raise ValueError("match_counts must be a non-empty tuple")
            players: list[int] = []
            for item in self.match_counts:
                if (
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or isinstance(item[0], bool)
                    or not isinstance(item[0], int)
                    or item[0] not in PLAYER_COUNTS
                    or isinstance(item[1], bool)
                    or not isinstance(item[1], int)
                    or item[1] < 1
                ):
                    raise ValueError("match_counts entries must be p4..p10 positive pairs")
                players.append(item[0])
            if players != sorted(set(players)):
                raise ValueError("match_counts player counts must be sorted and unique")
            if self.mode == "final" and dict(self.match_counts) != FINAL_MATCH_COUNTS:
                raise ValueError(
                    "final evaluation requires the exact canonical p4..p10 match plan"
                )
            if (
                self.mode == "certification"
                and dict(self.match_counts) != SCREENING_MATCH_COUNTS
            ):
                raise ValueError(
                    "certification requires the canonical 60-match p4..p10 plan"
                )
        if self.mode in {"certification", "final"} and (
            self.bootstrap_resamples != DEFAULT_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError(
                f"{self.mode} evaluation requires exactly 10000 bootstrap resamples"
            )
        if self.mode == "certification" and (
            self.match_shard_count != 1 or self.match_shard_index != 0
        ):
            raise ValueError("certification evaluation must be one unsharded report")

    @property
    def resolved_match_counts(self) -> dict[int, int]:
        if self.match_counts is not None:
            return dict(self.match_counts)
        return dict(
            SCREENING_MATCH_COUNTS
            if self.mode in {"screening", "certification"}
            else FINAL_MATCH_COUNTS
        )


@dataclass
class _Lane:
    player_count: int
    match_index: int
    seed: int
    env: DalmutiScalarEnv
    initial_order: tuple[int, ...]
    candidate_initial_seats: tuple[int, ...]
    candidate_ids: frozenset[int]
    observation: V4EnvironmentObservation
    act_records: list[dict[str, object]]
    decisions: int = 0


class V5GreedyActorPolicy:
    """Packed-legal, batched inference over validated public inputs only."""

    def __init__(self, actor: V5PublicActor, device: str | torch.device = "cpu") -> None:
        if type(actor) is not V5PublicActor:
            raise TypeError("V5 evaluator requires exactly one V5PublicActor")
        if float(actor.config.dropout) != 0.0:
            raise ValueError("V5 evaluation requires dropout=0")
        self.device = torch.device(device)
        self.policy_numerics = configure_v5_policy_numerics(self.device)
        self.actor = actor.to(self.device).eval()

    def actions(
        self,
        observations: Sequence[V5PublicObservation],
        normal_actions: Sequence[int],
    ) -> list[int]:
        if not observations:
            return []
        if len(observations) != len(normal_actions) or any(
            type(value) is not V5PublicObservation for value in observations
        ):
            raise TypeError("V5 policy requires matched public observations/actions")
        batch = stack_v5_actor_public_features(
            [tensorize_v5_public_observation(value) for value in observations],
            device=self.device,
        )
        normal = torch.tensor(normal_actions, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            packed = self.actor.forward_packed_batch(batch, normal)
            positions = packed.logits.argmax(dim=-1)
            actions = packed.action_indices.gather(1, positions[:, None]).squeeze(1)
        result = [int(value) for value in actions.detach().cpu().tolist()]
        for row, action in zip(observations, result, strict=True):
            if not bool(row.legal_mask[action]):
                raise RuntimeError("V5 Actor selected an illegal packed action")
        return result


def derive_v5_evaluation_seed(
    family_id: str, seed_base: int, player_count: int, match_index: int
) -> int:
    return _derive_uint32(
        family_id,
        seed_base,
        "v5-exact-normal-evaluation-match",
        player_count,
        match_index,
    )


def _bootstrap_seed(config: V5EvaluationConfig, player_count: int) -> int:
    return _derive_uint32(
        config.family_id,
        config.seed_base,
        "v5-exact-normal-evaluation-bootstrap",
        player_count,
    ) or 1


def _new_lane(config: V5EvaluationConfig, player_count: int, match_index: int) -> _Lane:
    seed = derive_v5_evaluation_seed(
        config.family_id, config.seed_base, player_count, match_index
    )
    env = DalmutiScalarEnv(player_count, acts=ACTS_PER_MATCH, seed=seed, device="cpu")
    initial_order = tuple(int(value) for value in env._order)
    if len(initial_order) != player_count or len(set(initial_order)) != player_count:
        raise RuntimeError("environment initial physical identities are invalid")
    seats = evaluation_candidate_initial_seats(player_count, match_index)
    candidate_ids = frozenset(initial_order[seat] for seat in seats)
    if not candidate_ids or len(candidate_ids) >= player_count:
        raise RuntimeError("candidate rotation must choose a non-empty proper group")
    return _Lane(
        player_count, match_index, seed, env, initial_order, seats,
        candidate_ids, env.observe(), [],
    )


def _finish_lane(lane: _Lane) -> dict[str, object]:
    if len(lane.act_records) != ACTS_PER_MATCH:
        raise RuntimeError("V5 evaluation requires complete five-act match clusters")
    differences = [float(record["meanChipDifference"]) for record in lane.act_records]
    before = sum(int(record["candidateBefore"]) for record in lane.act_records)
    comparisons = sum(int(record["comparisons"]) for record in lane.act_records)
    return {
        "playerCount": lane.player_count,
        "matchIndex": lane.match_index,
        "seed": lane.seed,
        "initialOrder": list(lane.initial_order),
        "candidateInitialSeats": list(lane.candidate_initial_seats),
        "candidatePhysicalIds": sorted(lane.candidate_ids),
        "acts": lane.act_records,
        "decisions": lane.decisions,
        "meanChipDifference": sum(differences) / ACTS_PER_MATCH,
        "candidateBefore": before,
        "comparisons": comparisons,
    }


def collect_v5_evaluation_clusters(
    actor: V5PublicActor,
    config: V5EvaluationConfig,
    *,
    device: str | torch.device = "cpu",
    certification_authorization: Mapping[str, object] | None = None,
    final_authorization: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Play the requested match shard; candidate physical IDs never change."""

    if config.mode == "certification":
        if not isinstance(certification_authorization, Mapping):
            raise ValueError(
                "certification collection requires prior execution reservation authorization"
            )
        coordinate = certification_authorization.get("coordinate")
        if coordinate != {
            "familyId": config.family_id,
            "seedBase": config.seed_base,
        }:
            raise ValueError("certification collection authorization coordinate drifted")
        if final_authorization is not None:
            raise ValueError("certification collection cannot use final authorization")
    elif config.mode == "final":
        if not isinstance(final_authorization, Mapping):
            raise ValueError("final collection requires prior claim authorization")
        shard = final_authorization.get("shard")
        if shard != {
            "count": config.match_shard_count,
            "index": config.match_shard_index,
        }:
            raise ValueError("final collection authorization shard drifted")
        if certification_authorization is not None:
            raise ValueError("final collection cannot use certification authorization")
    elif certification_authorization is not None or final_authorization is not None:
        raise ValueError("screening collection cannot use promotion authorization")

    specs = [
        (player_count, match_index)
        for player_count, count in sorted(config.resolved_match_counts.items())
        for match_index in range(count)
        if match_index % config.match_shard_count == config.match_shard_index
    ]
    if not specs:
        raise ValueError("requested evaluation shard is empty")
    seeds = {
        derive_v5_evaluation_seed(
            config.family_id, config.seed_base, player_count, match_index
        )
        for player_count, match_index in specs
    }
    if len(seeds) != len(specs):
        raise RuntimeError("V5 derived match seeds collided")
    policy = V5GreedyActorPolicy(actor, device=device)
    next_spec = min(len(specs), config.lane_count)
    lanes = [_new_lane(config, *spec) for spec in specs[:next_spec]]
    complete: list[dict[str, object]] = []
    while lanes:
        candidate_lanes: list[_Lane] = []
        candidate_publics: list[V5PublicObservation] = []
        candidate_normals: list[int] = []
        actions: dict[tuple[int, int], int] = {}
        for lane in lanes:
            env = lane.env
            normal = int(env.normal_action())
            public_v4 = lane.observation.public
            if not bool(public_v4.legal_mask[normal]):
                raise RuntimeError("production Normal selected an illegal action")
            if int(env.current_player_id) in lane.candidate_ids:
                candidate_lanes.append(lane)
                candidate_publics.append(v5_public_from_v4_actor_observation(public_v4))
                candidate_normals.append(normal)
            else:
                actions[(lane.player_count, lane.match_index)] = normal
        candidate_actions = policy.actions(candidate_publics, candidate_normals)
        for lane, action in zip(candidate_lanes, candidate_actions, strict=True):
            actions[(lane.player_count, lane.match_index)] = action

        remaining: list[_Lane] = []
        for lane in lanes:
            env = lane.env
            key = (lane.player_count, lane.match_index)
            action = actions[key]
            # Re-check the immutable routing identity at every decision.
            expected_candidate_ids = frozenset(
                lane.initial_order[seat] for seat in lane.candidate_initial_seats
            )
            if lane.candidate_ids != expected_candidate_ids:
                raise RuntimeError("candidate identity routing changed")
            lane.decisions += 1
            result = env.step(action)
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
                (
                    candidate_mean,
                    normal_mean,
                    difference,
                    before,
                    comparisons,
                    rate,
                    _,
                ) = evaluator_group_reward_components(
                    finish_order, chip_awards, lane.candidate_ids
                )
                lane.act_records.append({
                    "act": len(lane.act_records) + 1,
                    "finishOrder": list(finish_order),
                    "candidatePhysicalIds": sorted(lane.candidate_ids),
                    "candidateMeanChip": candidate_mean,
                    "normalMeanChip": normal_mean,
                    "meanChipDifference": difference,
                    "candidateBefore": before,
                    "comparisons": comparisons,
                    "pairwiseRate": rate,
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
    if len(complete) != len(specs):
        raise RuntimeError("evaluation failed to complete every requested match")
    return complete


def _authorize_certification_actor_evaluation(
    actor: V5PublicActor,
    config: V5EvaluationConfig,
    *,
    model_identity: Mapping[str, object] | None,
    certification_reservation: str | Path | None,
    evaluation_provenance: Mapping[str, object] | None,
    output_path: str | Path | None,
) -> dict[str, object]:
    if (
        model_identity is None
        or certification_reservation is None
        or evaluation_provenance is None
        or output_path is None
    ):
        raise ValueError(
            "certification evaluation requires its Actor identity, execution "
            "reservation, provenance, and canonical output"
        )
    from v5_promotion import (
        authorize_v5_certification_evaluation,
        load_v5_certification_execution_reservation,
    )

    reservation = load_v5_certification_execution_reservation(
        certification_reservation
    )
    model = reservation.get("model")
    if not isinstance(model, Mapping) or model != dict(model_identity):
        raise ValueError(
            "loaded certification Actor identity differs from its reservation"
        )
    if model.get("tensorStateSha256") != tensor_state_sha256(actor.state_dict()):
        raise ValueError(
            "loaded certification Actor tensor state differs from its reserved bundle"
        )
    return authorize_v5_certification_evaluation(
        certification_reservation,
        model_identity,
        evaluation_provenance=evaluation_provenance,
        family_id=config.family_id,
        seed_base=config.seed_base,
        match_plan=config.resolved_match_counts,
        match_shard_count=config.match_shard_count,
        match_shard_index=config.match_shard_index,
        bootstrap_resamples=config.bootstrap_resamples,
        output_path=output_path,
    )


def _authorize_final_actor_evaluation(
    actor: V5PublicActor,
    config: V5EvaluationConfig,
    *,
    model_identity: Mapping[str, object] | None,
    promotion_plan: str | Path | None,
    final_claim: str | Path | None,
    evaluation_provenance: Mapping[str, object] | None,
    output_path: str | Path | None,
) -> dict[str, object]:
    if (
        model_identity is None
        or promotion_plan is None
        or final_claim is None
        or evaluation_provenance is None
        or output_path is None
    ):
        raise ValueError(
            "final evaluation requires plan, claim, provenance, and canonical output"
        )
    from v5_promotion import authorize_v5_final_evaluation, load_v5_final_evaluation_claim

    claim = load_v5_final_evaluation_claim(final_claim, promotion_plan)
    model = claim.get("model")
    if not isinstance(model, Mapping) or model != dict(model_identity):
        raise ValueError("loaded final Actor identity differs from its start claim")
    if model.get("tensorStateSha256") != tensor_state_sha256(actor.state_dict()):
        raise ValueError("loaded final Actor tensor state differs from its claimed bundle")
    return authorize_v5_final_evaluation(
        promotion_plan,
        final_claim,
        model_identity,
        family_id=config.family_id,
        seed_base=config.seed_base,
        match_plan=config.resolved_match_counts,
        match_shard_count=config.match_shard_count,
        match_shard_index=config.match_shard_index,
        bootstrap_resamples=config.bootstrap_resamples,
        evaluation_provenance=evaluation_provenance,
        output_path=output_path,
    )


def _validate_cluster(record: Mapping[str, object]) -> dict[str, object]:
    required = {
        "playerCount", "matchIndex", "seed", "initialOrder",
        "candidateInitialSeats", "candidatePhysicalIds", "acts", "decisions",
        "meanChipDifference", "candidateBefore", "comparisons",
    }
    if set(record) != required:
        raise ValueError("V5 match-cluster fields drifted")
    player_count = record["playerCount"]
    match_index = record["matchIndex"]
    if type(player_count) is not int or player_count not in PLAYER_COUNTS:
        raise ValueError("cluster player count is invalid")
    if type(match_index) is not int or match_index < 0:
        raise ValueError("cluster match index is invalid")
    initial = record["initialOrder"]
    seats = record["candidateInitialSeats"]
    candidate = record["candidatePhysicalIds"]
    acts = record["acts"]
    if (
        not isinstance(initial, list)
        or len(initial) != player_count
        or len(set(initial)) != player_count
        or seats != list(evaluation_candidate_initial_seats(player_count, match_index))
        or not isinstance(candidate, list)
        or candidate != sorted(initial[seat] for seat in seats)
        or not isinstance(acts, list)
        or len(acts) != ACTS_PER_MATCH
    ):
        raise ValueError("cluster fixed-identity schedule is invalid")
    differences: list[float] = []
    before = comparisons = 0
    for index, act in enumerate(acts, start=1):
        if not isinstance(act, dict) or act.get("act") != index:
            raise ValueError("cluster acts are incomplete or unordered")
        if act.get("candidatePhysicalIds") != candidate:
            raise ValueError("candidate physical IDs changed within a match")
        finish = act.get("finishOrder")
        if not isinstance(finish, list) or set(finish) != set(initial):
            raise ValueError("cluster finish order is invalid")
        # Reconstruct exact score and pairwise metrics; do not trust report values.
        awards = {
            actor_id: 4 if place == 1 else 3 if place == 2 else
            1 if place == player_count - 1 else 0 if place == player_count else 2
            for place, actor_id in enumerate(finish, start=1)
        }
        components = evaluator_group_reward_components(finish, awards, candidate)
        expected = {
            "candidateMeanChip": components[0],
            "normalMeanChip": components[1],
            "meanChipDifference": components[2],
            "candidateBefore": components[3],
            "comparisons": components[4],
            "pairwiseRate": components[5],
        }
        for key, value in expected.items():
            if act.get(key) != value:
                raise ValueError(f"cluster act metric drifted: {key}")
        differences.append(components[2])
        before += components[3]
        comparisons += components[4]
    if (
        record.get("meanChipDifference") != sum(differences) / ACTS_PER_MATCH
        or record.get("candidateBefore") != before
        or record.get("comparisons") != comparisons
    ):
        raise ValueError("cluster aggregate metric drifted")
    return dict(record)


def _validate_evaluation_provenance_records(
    values: Sequence[Mapping[str, object]] | None,
    config: V5EvaluationConfig,
    *,
    final_claims: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]] | None:
    """Validate direct-shard or complete-merge execution provenance."""

    if values is None:
        if config.mode in {"certification", "final"}:
            raise ValueError(
                f"{config.mode} report requires evaluation source/runtime provenance"
            )
        return None
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("evaluation provenance must be a non-empty sequence")
    records: list[dict[str, object]] = []
    coordinates: set[tuple[int, int]] = set()
    source_bindings: set[tuple[str, str]] = set()
    for raw in values:
        if not isinstance(raw, Mapping) or set(raw) != _PROVENANCE_RECORD_KEYS:
            raise ValueError("evaluation provenance record fields drifted")
        shard = raw.get("shard")
        if (
            not isinstance(shard, Mapping)
            or set(shard) != {"count", "index"}
            or type(shard.get("count")) is not int
            or type(shard.get("index")) is not int
            or int(shard["count"]) < 1
            or not 0 <= int(shard["index"]) < int(shard["count"])
        ):
            raise ValueError("evaluation provenance shard coordinate is invalid")
        coordinate = (int(shard["count"]), int(shard["index"]))
        if coordinate in coordinates:
            raise ValueError("evaluation provenance duplicated a shard coordinate")
        coordinates.add(coordinate)
        evidence = raw.get("provenance")
        if not isinstance(evidence, Mapping):
            raise ValueError("evaluation provenance evidence is missing")
        verified = validate_v5_evaluation_provenance(evidence)
        source = verified["source"]
        artifacts = verified["artifacts"]
        assert isinstance(source, Mapping) and isinstance(artifacts, Mapping)
        source_bindings.add(
            (str(source["sourceCommit"]), str(source["sourceBindingSha256"]))
        )
        if config.mode in {"screening", "certification", "final"} and any(
            artifacts.get(name) is None
            for name in ("gitBundleSha256", "sourceSnapshotSha256")
        ):
            raise ValueError(
                f"{config.mode} provenance requires preserved source snapshot and Git bundle"
            )
        records.append({
            "provenance": verified,
            "shard": {"count": coordinate[0], "index": coordinate[1]},
        })
    if len(source_bindings) != 1:
        raise ValueError("evaluation shards used different evaluator/Normal sources")
    shard_counts = {count for count, _ in coordinates}
    if len(shard_counts) != 1:
        raise ValueError("evaluation provenance mixed shard counts")
    evidence_count = next(iter(shard_counts))
    direct = coordinates == {
        (config.match_shard_count, config.match_shard_index)
    }
    complete_merge = (
        config.match_shard_count == 1
        and config.match_shard_index == 0
        and coordinates == {(evidence_count, index) for index in range(evidence_count)}
    )
    if not direct and not complete_merge:
        raise ValueError(
            "evaluation provenance does not cover the exact direct or merged shard inventory"
        )
    if config.mode == "final":
        if final_claims is None:
            raise ValueError("final provenance cannot be checked without shard claims")
        claim_coordinates = {
            (int(value["shard"]["count"]), int(value["shard"]["index"]))  # type: ignore[index]
            for value in final_claims
        }
        if coordinates != claim_coordinates:
            raise ValueError("final provenance and claim shard inventories disagree")
        claim_provenance = {
            (
                int(value["shard"]["count"]),  # type: ignore[index]
                int(value["shard"]["index"]),  # type: ignore[index]
            ): value["evaluationProvenanceSha256"]
            for value in final_claims
        }
        observed_provenance = {
            (
                int(value["shard"]["count"]),  # type: ignore[index]
                int(value["shard"]["index"]),  # type: ignore[index]
            ): value["provenance"]["provenanceSha256"]  # type: ignore[index]
            for value in records
        }
        if claim_provenance != observed_provenance:
            raise ValueError("final claim and execution provenance SHA bindings disagree")
    records.sort(key=lambda value: int(value["shard"]["index"]))  # type: ignore[index]
    return records


def _canonical_logical_result_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is not a canonical registry-relative POSIX path")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or logical.as_posix() != value
        or any(part in ("", ".", "..") for part in logical.parts)
    ):
        raise ValueError(f"{label} is not a canonical registry-relative POSIX path")
    return value


def _validate_certification_reservation_binding(
    value: Mapping[str, object] | None,
    config: V5EvaluationConfig,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "coordinate",
        "evaluationProvenanceSha256",
        "outputPath",
        "reservationId",
        "reservationSha256",
    }:
        raise ValueError("certification reservation report binding fields drifted")
    coordinate = value.get("coordinate")
    if coordinate != {"familyId": config.family_id, "seedBase": config.seed_base}:
        raise ValueError("certification reservation report coordinate drifted")
    for name in (
        "evaluationProvenanceSha256",
        "reservationId",
        "reservationSha256",
    ):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"certification reservation report has invalid {name}")
    output_path = _canonical_logical_result_path(
        value.get("outputPath"), "certification reservation output path"
    )
    reservation_id = str(value["reservationId"])
    if output_path not in {
        f"certification-results/{reservation_id}/a.json",
        f"certification-results/{reservation_id}/b.json",
    }:
        raise ValueError("certification reservation output path drifted")
    return dict(value)


def _validate_screening_reservation_binding(
    value: Mapping[str, object] | None,
    config: V5EvaluationConfig,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "coordinate",
        "evaluationProvenanceSha256",
        "outputPath",
        "reservationId",
        "reservationSha256",
    }:
        raise ValueError("screening reservation report binding fields drifted")
    coordinate = value.get("coordinate")
    if coordinate != {"familyId": config.family_id, "seedBase": config.seed_base}:
        raise ValueError("screening reservation report coordinate drifted")
    for name in (
        "evaluationProvenanceSha256",
        "reservationId",
        "reservationSha256",
    ):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"screening reservation report has invalid {name}")
    output_path = _canonical_logical_result_path(
        value.get("outputPath"), "screening reservation output path"
    )
    if output_path != (
        f"screening-results/{value['reservationId']}/report.json"
    ):
        raise ValueError("screening reservation output path drifted")
    return dict(value)


def summarize_v5_evaluation_clusters(
    records: Sequence[Mapping[str, object]],
    config: V5EvaluationConfig,
    *,
    model_identity: Mapping[str, object] | None = None,
    screening_reservation: Mapping[str, object] | None = None,
    certification_reservation: Mapping[str, object] | None = None,
    final_claims: Sequence[Mapping[str, object]] | None = None,
    evaluation_provenance: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize zero complete matches")
    validated = [_validate_cluster(record) for record in records]
    keys = [(int(item["playerCount"]), int(item["matchIndex"])) for item in validated]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate match cluster")
    for item in validated:
        player_count = int(item["playerCount"])
        match_index = int(item["matchIndex"])
        if match_index % config.match_shard_count != config.match_shard_index:
            raise ValueError("match cluster belongs to a different shard")
        if int(item["seed"]) != derive_v5_evaluation_seed(
            config.family_id, config.seed_base, player_count, match_index
        ):
            raise ValueError("match cluster seed is not the deterministic derived seed")
    validated.sort(key=lambda item: (int(item["playerCount"]), int(item["matchIndex"])))
    planned = config.resolved_match_counts
    results: list[dict[str, object]] = []
    overall_complete = True
    for player_count in sorted(planned):
        group = [item for item in validated if item["playerCount"] == player_count]
        expected_indices = set(range(planned[player_count]))
        actual_indices = {int(item["matchIndex"]) for item in group}
        if not actual_indices <= expected_indices:
            raise ValueError("cluster match index escaped the evaluation plan")
        complete = actual_indices == expected_indices
        overall_complete &= complete
        if not group:
            continue
        samples = [float(item["meanChipDifference"]) for item in group]
        interval = deterministic_cluster_bootstrap95(
            samples,
            seed=_bootstrap_seed(config, player_count),
            resamples=config.bootstrap_resamples,
        )
        before = sum(int(item["candidateBefore"]) for item in group)
        comparisons = sum(int(item["comparisons"]) for item in group)
        pairwise = before / comparisons
        mean = float(interval["mean"])
        lower = float(interval["low"])
        point_pass = mean >= EXACT_GATES["minMeanChipDifference"]
        lower_pass = lower >= EXACT_GATES["minCluster95LowerBound"]
        pair_pass = pairwise >= EXACT_GATES["minPairwiseRate"]
        provisional = point_pass and lower_pass and pair_pass
        cluster_bytes = canonical_json_bytes(group)
        results.append({
            "playerCount": player_count,
            "matches": len(group),
            "plannedMatches": planned[player_count],
            "complete": complete,
            "meanCandidateMinusNormalChipPerAct": mean,
            "matchClustered95": {
                "method": "deterministic-percentile-bootstrap",
                "unit": "complete-five-act-match",
                "clusters": len(group),
                "resamples": config.bootstrap_resamples,
                "low": lower,
                "high": float(interval["high"]),
            },
            "candidateBeforeNormalPairwise": {
                "candidateBefore": before,
                "comparisons": comparisons,
                "rate": pairwise,
            },
            "clusterRecordsSha256": hashlib.sha256(cluster_bytes).hexdigest(),
            "gate": {
                "meanChipPassed": point_pass,
                "cluster95LowerPassed": lower_pass,
                "pairwisePassed": pair_pass,
                "provisionalPassed": provisional,
                "passed": complete and provisional,
            },
        })
    expected_players = set(planned)
    result_players = {int(item["playerCount"]) for item in results}
    overall_complete &= result_players == expected_players
    all_passed = overall_complete and all(bool(item["gate"]["passed"]) for item in results)  # type: ignore[index]
    identity = dict(model_identity or {})
    identity.setdefault("publicContractSha256", V5_PUBLIC_CONTRACT_SHA256)
    identity.setdefault("policyNumericsSha256", V5_POLICY_NUMERICS_SHA256)
    if identity.get("publicContractSha256") != V5_PUBLIC_CONTRACT_SHA256:
        raise ValueError("evaluation model identity has wrong public contract")
    if identity.get("policyNumericsSha256") != V5_POLICY_NUMERICS_SHA256:
        raise ValueError("evaluation model identity has wrong policy numerics")
    if config.mode in {"certification", "final"} or screening_reservation is not None:
        if set(identity) != _FINAL_MODEL_IDENTITY_KEYS:
            raise ValueError(
                f"{config.mode} evaluation requires the complete verified Actor identity"
            )
        for name in ("actorSha256", "manifestSha256", "tensorStateSha256"):
            digest = identity.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"final evaluation model identity has invalid {name}")
    screening_binding: dict[str, object] | None = None
    if config.mode == "screening" and screening_reservation is not None:
        screening_binding = _validate_screening_reservation_binding(
            screening_reservation, config
        )
    elif config.mode != "screening" and screening_reservation is not None:
        raise ValueError(
            "non-screening report cannot contain a screening reservation"
        )
    certification_binding: dict[str, object] | None = None
    if config.mode == "certification":
        certification_binding = _validate_certification_reservation_binding(
            certification_reservation, config
        )
    elif certification_reservation is not None:
        raise ValueError(
            "non-certification report cannot contain a certification reservation"
        )
    claim_records: list[dict[str, object]] | None = None
    if config.mode == "final":
        if not isinstance(final_claims, Sequence) or not final_claims:
            raise ValueError("final report requires one-shot shard claim bindings")
        claim_records = [dict(value) for value in final_claims]
        for claim in claim_records:
            if set(claim) != {
                "claimId",
                "evaluationProvenanceSha256",
                "outputPath",
                "reservationId",
                "shard",
            }:
                raise ValueError("final report claim binding fields drifted")
            shard_value = claim.get("shard")
            if (
                not isinstance(shard_value, dict)
                or set(shard_value) != {"count", "index"}
                or type(shard_value.get("count")) is not int
                or type(shard_value.get("index")) is not int
                or not 0 <= shard_value["index"] < shard_value["count"]
            ):
                raise ValueError("final report claim shard binding is invalid")
            for name in (
                "claimId",
                "evaluationProvenanceSha256",
                "reservationId",
            ):
                digest = claim.get(name)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"final report claim has invalid {name}")
            output_path = _canonical_logical_result_path(
                claim.get("outputPath"), "final report claim output path"
            )
            expected_output = (
                f"final-results/{claim['reservationId']}/"
                f"shard-{shard_value['index']:03d}.json"
            )
            if output_path != expected_output:
                raise ValueError("final report claim output path drifted")
        reservations = {value["reservationId"] for value in claim_records}
        counts = {value["shard"]["count"] for value in claim_records}  # type: ignore[index]
        indices = {value["shard"]["index"] for value in claim_records}  # type: ignore[index]
        if len(reservations) != 1 or len(counts) != 1:
            raise ValueError("final report mixed reservations or shard counts")
        claimed_count = int(next(iter(counts)))
        direct_shard = (
            len(claim_records) == 1
            and claim_records[0]["shard"]
            == {"count": config.match_shard_count, "index": config.match_shard_index}
        )
        complete_merge = (
            config.match_shard_count == 1
            and config.match_shard_index == 0
            and len(claim_records) == claimed_count
            and indices == set(range(claimed_count))
        )
        if not direct_shard and not complete_merge:
            raise ValueError("final report does not bind the exact claimed shard inventory")
        claim_records.sort(key=lambda value: int(value["shard"]["index"]))  # type: ignore[index]
    elif final_claims is not None:
        raise ValueError("non-final report cannot contain final holdout claims")
    provenance_records = _validate_evaluation_provenance_records(
        evaluation_provenance,
        config,
        final_claims=claim_records,
    )
    if certification_binding is not None:
        if (
            provenance_records is None
            or len(provenance_records) != 1
            or provenance_records[0]["provenance"]["provenanceSha256"]  # type: ignore[index]
            != certification_binding["evaluationProvenanceSha256"]
        ):
            raise ValueError(
                "certification reservation and execution provenance SHA disagree"
            )
    if screening_binding is not None:
        if (
            provenance_records is None
            or len(provenance_records) != 1
            or provenance_records[0]["provenance"]["provenanceSha256"]  # type: ignore[index]
            != screening_binding["evaluationProvenanceSha256"]
        ):
            raise ValueError(
                "screening reservation and execution provenance SHA disagree"
            )
    report = {
        "format": V5_EVALUATION_FORMAT,
        "version": V5_EVALUATION_VERSION,
        "mode": config.mode,
        "familyId": config.family_id,
        "seedBase": config.seed_base,
        "actsPerMatch": ACTS_PER_MATCH,
        "gates": dict(EXACT_GATES),
        "matchPlan": {str(key): value for key, value in sorted(planned.items())},
        "shard": {
            "count": config.match_shard_count,
            "index": config.match_shard_index,
        },
        "model": identity,
        "completeEvaluation": overall_complete,
        "allPlayerCountsPassed": all_passed,
        "results": results,
        "matchClusters": validated,
    }
    if screening_binding is not None:
        report["screeningReservation"] = screening_binding
    if certification_binding is not None:
        report["certificationReservation"] = certification_binding
    if claim_records is not None:
        report["finalClaims"] = claim_records
    if provenance_records is not None:
        report["evaluationProvenance"] = provenance_records
    return report


def evaluate_v5_actor(
    actor: V5PublicActor,
    config: V5EvaluationConfig,
    *,
    device: str | torch.device = "cpu",
    model_identity: Mapping[str, object] | None = None,
    evaluation_provenance: Mapping[str, object] | None = None,
    screening_reservation: Mapping[str, object] | None = None,
    certification_reservation: str | Path | None = None,
    promotion_plan: str | Path | None = None,
    final_claim: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    certification_binding: dict[str, object] | None = None
    claim_binding: dict[str, object] | None = None
    if config.mode == "certification":
        certification_binding = _authorize_certification_actor_evaluation(
            actor,
            config,
            model_identity=model_identity,
            certification_reservation=certification_reservation,
            evaluation_provenance=evaluation_provenance,
            output_path=output_path,
        )
    elif config.mode == "final":
        claim_binding = _authorize_final_actor_evaluation(
            actor,
            config,
            model_identity=model_identity,
            promotion_plan=promotion_plan,
            final_claim=final_claim,
            evaluation_provenance=evaluation_provenance,
            output_path=output_path,
        )
    records = collect_v5_evaluation_clusters(
        actor,
        config,
        device=device,
        certification_authorization=certification_binding,
        final_authorization=claim_binding,
    )
    return summarize_v5_evaluation_clusters(
        records,
        config,
        model_identity=model_identity,
        screening_reservation=screening_reservation,
        certification_reservation=certification_binding,
        final_claims=[claim_binding] if claim_binding is not None else None,
        evaluation_provenance=(
            [{
                "provenance": dict(evaluation_provenance),
                "shard": {
                    "count": config.match_shard_count,
                    "index": config.match_shard_index,
                },
            }]
            if evaluation_provenance is not None
            else None
        ),
    )


def merge_v5_evaluation_reports(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not reports:
        raise ValueError("at least one V5 shard report is required")
    verified_reports = [validate_v5_evaluation_report(report) for report in reports]
    first = verified_reports[0]
    if first.get("mode") == "certification":
        raise ValueError("certification is reserved as one canonical unsharded report")
    required_shared = (
        "format", "version", "mode", "familyId", "seedBase", "actsPerMatch",
        "gates", "matchPlan", "model",
    )
    for report in verified_reports:
        if any(report.get(key) != first.get(key) for key in required_shared):
            raise ValueError("V5 shard report contracts do not match")
    shard_count = len(verified_reports)
    indices: set[int] = set()
    records: list[Mapping[str, object]] = []
    final_claims: list[Mapping[str, object]] | None = (
        [] if first.get("mode") == "final" else None
    )
    provenance_records: list[Mapping[str, object]] | None = (
        [] if "evaluationProvenance" in first else None
    )
    for report in verified_reports:
        if ("evaluationProvenance" in report) != (provenance_records is not None):
            raise ValueError("V5 shard reports mixed provenance presence")
        shard = report.get("shard")
        if not isinstance(shard, Mapping) or shard.get("count") != shard_count:
            raise ValueError("V5 shard inventory is incomplete")
        index = shard.get("index")
        if type(index) is not int or not 0 <= index < shard_count or index in indices:
            raise ValueError("V5 shard index is invalid or duplicated")
        indices.add(index)
        clusters = report.get("matchClusters")
        if not isinstance(clusters, list):
            raise ValueError("V5 shard omitted match cluster records")
        records.extend(clusters)
        if final_claims is not None:
            claims = report.get("finalClaims")
            if not isinstance(claims, list):
                raise ValueError("V5 final shard omitted its one-shot claim binding")
            final_claims.extend(claims)
        if provenance_records is not None:
            evidence = report.get("evaluationProvenance")
            if not isinstance(evidence, list):
                raise ValueError("V5 shard omitted evaluation provenance")
            provenance_records.extend(evidence)
    if indices != set(range(shard_count)):
        raise ValueError("V5 shard report set has a gap")
    plan = first.get("matchPlan")
    if not isinstance(plan, Mapping):
        raise ValueError("V5 shard match plan is invalid")
    config = V5EvaluationConfig(
        mode=str(first["mode"]),
        family_id=str(first["familyId"]),
        seed_base=int(first["seedBase"]),
        match_counts=tuple(sorted((int(key), int(value)) for key, value in plan.items())),
        bootstrap_resamples=int(
            verified_reports[0]["results"][0]["matchClustered95"]["resamples"]  # type: ignore[index]
        ),
    )
    return summarize_v5_evaluation_clusters(
        records,
        config,
        model_identity=first.get("model"),  # type: ignore[arg-type]
        final_claims=final_claims,
        evaluation_provenance=provenance_records,
    )


def validate_v5_evaluation_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Recompute every metric and binding solely from complete match records."""

    if not isinstance(report, Mapping):
        raise TypeError("V5 evaluation report must be a mapping")
    optional = {"diagnosticOnly", "initialNormalParity"}
    expected = {
        "format", "version", "mode", "familyId", "seedBase", "actsPerMatch",
        "gates", "matchPlan", "shard", "model", "completeEvaluation",
        "allPlayerCountsPassed", "results", "matchClusters",
    }
    if report.get("mode") == "final":
        expected.add("finalClaims")
    if report.get("mode") == "certification":
        expected.add("certificationReservation")
    if "screeningReservation" in report:
        if report.get("mode") != "screening":
            raise ValueError("non-screening report contains a screening reservation")
        expected.add("screeningReservation")
    if "evaluationProvenance" in report:
        expected.add("evaluationProvenance")
    if not expected <= set(report) or set(report) - expected - optional:
        raise ValueError("V5 evaluation report fields drifted")
    if (
        report.get("format") != V5_EVALUATION_FORMAT
        or report.get("version") != V5_EVALUATION_VERSION
        or report.get("actsPerMatch") != ACTS_PER_MATCH
        or report.get("gates") != EXACT_GATES
    ):
        raise ValueError("V5 evaluation report contract drifted")
    plan = report.get("matchPlan")
    shard = report.get("shard")
    results = report.get("results")
    clusters = report.get("matchClusters")
    model = report.get("model")
    if (
        not isinstance(plan, Mapping)
        or not isinstance(shard, Mapping)
        or set(shard) != {"count", "index"}
        or not isinstance(results, list)
        or not results
        or not isinstance(clusters, list)
        or not isinstance(model, Mapping)
    ):
        raise ValueError("V5 evaluation report structure is invalid")
    try:
        resamples = results[0]["matchClustered95"]["resamples"]
        config = V5EvaluationConfig(
            mode=str(report["mode"]),
            family_id=str(report["familyId"]),
            seed_base=int(report["seedBase"]),
            match_counts=tuple(
                sorted((int(key), int(value)) for key, value in plan.items())
            ),
            match_shard_count=int(shard["count"]),
            match_shard_index=int(shard["index"]),
            bootstrap_resamples=int(resamples),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("V5 evaluation report plan is invalid") from error
    rebuilt = summarize_v5_evaluation_clusters(
        clusters,
        config,
        model_identity=model,
        screening_reservation=(
            report.get("screeningReservation")
            if config.mode == "screening"
            else None
        ),  # type: ignore[arg-type]
        certification_reservation=(
            report.get("certificationReservation")
            if config.mode == "certification"
            else None
        ),  # type: ignore[arg-type]
        final_claims=(
            report.get("finalClaims") if config.mode == "final" else None
        ),  # type: ignore[arg-type]
        evaluation_provenance=report.get("evaluationProvenance"),  # type: ignore[arg-type]
    )
    base = {key: report[key] for key in expected}
    if canonical_json_bytes(base) != canonical_json_bytes(rebuilt):
        raise ValueError("V5 evaluation report metrics or checksums do not recompute")
    if "initialNormalParity" in report:
        proof = report["initialNormalParity"]
        if (
            not isinstance(proof, Mapping)
            or proof.get("contract")
            != "dalmuti-v5-initial-greedy-normal-parity-v1"
            or proof.get("passed") is not True
        ):
            raise ValueError("V5 initial-Normal parity proof is invalid")
    if "diagnosticOnly" in report and report["diagnosticOnly"] is not True:
        raise ValueError("V5 diagnosticOnly marker must be exactly true")
    return dict(report)


def approve_v5_final_evaluation_report(
    report: Mapping[str, object],
    actor_bundle: str | Path,
    *,
    promotion_plan: str | Path | None = None,
    final_report_path: str | Path | None = None,
) -> dict[str, object]:
    """Fail closed unless a canonical final report proves the exact exit gates.

    General report validation intentionally remains useful for screening and
    sharded partial results.  This separate approval boundary is the only API
    that may certify the V5 stopping criteria, and binds that decision to the
    verified, preserved Actor bundle on disk.
    """

    verified = validate_v5_evaluation_report(report)
    if verified.get("mode") != "final":
        raise ValueError("only a final evaluation report can approve V5 exit gates")
    if promotion_plan is None or final_report_path is None:
        raise ValueError(
            "final approval requires an immutable promotion plan and report file"
        )
    from v5_promotion import approve_v5_final_holdout

    return approve_v5_final_holdout(
        promotion_plan,
        actor_bundle,
        final_report_path,
        expected_report=verified,
    )


def prove_initial_v5_normal_parity(
    actor: V5PublicActor,
    *,
    family_id: str = "v5-initial-normal-parity",
    seed_base: int = 1,
    matches_per_player_count: int = 1,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Prove zero-residual greedy parity on complete actual p4..p10 games."""

    # A fresh actor with the same configuration is the actual initialization;
    # its residual output is contractually zero-initialized.
    fresh = V5PublicActor(actor.config)
    policy = V5GreedyActorPolicy(fresh, device=device)
    compared = decisions = 0
    per_player: dict[str, int] = {}
    for player_count in PLAYER_COUNTS:
        count = 0
        for match_index in range(matches_per_player_count):
            seed = derive_v5_evaluation_seed(
                family_id, seed_base, player_count, match_index
            )
            env = DalmutiScalarEnv(
                player_count, acts=ACTS_PER_MATCH, seed=seed, device="cpu"
            )
            observation = env.observe()
            while not env.terminated:
                normal = int(env.normal_action())
                public = v5_public_from_v4_actor_observation(observation.public)
                action = policy.actions([public], [normal])[0]
                decisions += 1
                if action != normal:
                    raise AssertionError(
                        f"initial V5 Actor diverged from Normal at p{player_count}"
                    )
                compared += 1
                count += 1
                observation = env.step(action).observation
        per_player[str(player_count)] = count
    return {
        "contract": "dalmuti-v5-initial-greedy-normal-parity-v1",
        "completeActualFiveActMatchesPerPlayerCount": matches_per_player_count,
        "playerCounts": list(PLAYER_COUNTS),
        "decisionsCompared": compared,
        "environmentDecisions": decisions,
        "decisionsByPlayerCount": per_player,
        "passed": compared > 0 and compared == decisions,
    }


def write_v5_evaluation_report(
    output_path: str | Path, report: Mapping[str, object]
) -> str:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(validate_v5_evaluation_report(report))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"V5 evaluation report already exists: {path}")
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--mode", choices=EVALUATION_MODES, required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--seed-base", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lane-count", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--match-shard-count", type=int, default=1)
    parser.add_argument("--match-shard-index", type=int, default=0)
    parser.add_argument("--certification-reservation")
    parser.add_argument("--promotion-plan")
    parser.add_argument("--final-claim")
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parent.parent),
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--source-snapshot")
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--git-bundle")
    parser.add_argument("--git-bundle-sha256")
    parser.add_argument("--prove-initial-normal-parity", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode != "screening":
        raise RuntimeError(
            "production certification/final evaluation is available only through "
            "v5_workflow.py so its one-shot execution marker cannot be bypassed"
        )
    actor, _ = load_v5_actor_bundle(arguments.bundle)
    digests = v5_actor_bundle_digests(arguments.bundle)
    config = V5EvaluationConfig(
        mode=arguments.mode,
        family_id=arguments.family_id,
        seed_base=arguments.seed_base,
        match_shard_count=arguments.match_shard_count,
        match_shard_index=arguments.match_shard_index,
        lane_count=arguments.lane_count,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    if arguments.mode == "certification":
        if arguments.certification_reservation is None:
            raise ValueError(
                "certification CLI evaluation requires --certification-reservation"
            )
        if arguments.promotion_plan is not None or arguments.final_claim is not None:
            raise ValueError(
                "--promotion-plan/--final-claim are reserved for final evaluation"
            )
    elif arguments.mode == "final":
        if arguments.promotion_plan is None or arguments.final_claim is None:
            raise ValueError(
                "final CLI evaluation requires --promotion-plan and --final-claim"
            )
        if arguments.certification_reservation is not None:
            raise ValueError(
                "--certification-reservation is reserved for certification"
            )
    elif (
        arguments.certification_reservation is not None
        or arguments.promotion_plan is not None
        or arguments.final_claim is not None
    ):
        raise ValueError(
            "promotion reservations are not allowed for screening evaluation"
        )
    provenance_arguments_present = any((
        arguments.source_commit,
        arguments.source_snapshot,
        arguments.source_snapshot_sha256,
        arguments.git_bundle,
        arguments.git_bundle_sha256,
    ))
    if arguments.mode in {"certification", "final"}:
        if (
            arguments.source_commit is None
            or arguments.source_snapshot is None
            or arguments.git_bundle is None
        ):
            raise ValueError(
                f"{arguments.mode} CLI evaluation requires --source-commit, "
                "--source-snapshot, and --git-bundle"
            )
    elif provenance_arguments_present and arguments.source_commit is None:
        raise ValueError("evaluation provenance artifacts require --source-commit")
    evaluation_provenance = None
    if arguments.source_commit is not None:
        evaluation_provenance = build_v5_evaluation_provenance(
            arguments.repository_root,
            arguments.source_commit,
            backend=str(torch.device(arguments.device).type),
            source_snapshot=arguments.source_snapshot,
            source_snapshot_sha256=arguments.source_snapshot_sha256,
            git_bundle=arguments.git_bundle,
            git_bundle_sha256=arguments.git_bundle_sha256,
        )
    report = evaluate_v5_actor(
        actor,
        config,
        device=arguments.device,
        model_identity=digests,
        evaluation_provenance=evaluation_provenance,
        certification_reservation=arguments.certification_reservation,
        promotion_plan=arguments.promotion_plan,
        final_claim=arguments.final_claim,
        output_path=arguments.output,
    )
    report["diagnosticOnly"] = True
    if arguments.prove_initial_normal_parity:
        report["initialNormalParity"] = prove_initial_v5_normal_parity(
            actor, device=arguments.device
        )
    digest = write_v5_evaluation_report(arguments.output, report)
    print(json.dumps({"report": str(Path(arguments.output).resolve()), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTS_PER_MATCH",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "EVALUATION_MODES",
    "EXACT_GATES",
    "FINAL_MATCH_COUNTS",
    "PLAYER_COUNTS",
    "SCREENING_MATCH_COUNTS",
    "V5EvaluationConfig",
    "V5GreedyActorPolicy",
    "approve_v5_final_evaluation_report",
    "collect_v5_evaluation_clusters",
    "derive_v5_evaluation_seed",
    "evaluate_v5_actor",
    "evaluation_candidate_initial_seats",
    "merge_v5_evaluation_reports",
    "prove_initial_v5_normal_parity",
    "summarize_v5_evaluation_clusters",
    "validate_v5_evaluation_report",
    "write_v5_evaluation_report",
]
