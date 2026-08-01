from __future__ import annotations

"""Canonical leakage-safe PPO return/advantage derivation for V4.

This module intentionally has no dependency on the collector, merger, or
dataset loader.  All three can therefore use the same implementation without
creating an import cycle.  In particular, a merged dataset is not trusted to
carry forward per-shard baselines: they are derived again from the complete
merged PPO trajectory population and can be independently verified on load.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np


BASELINE_FALLBACK_HIERARCHY = (
    "same-player-count-role-act",
    "same-player-count-role",
    "same-player-count-act",
    "same-player-count",
    "all-player-counts",
    "zero-no-other-match",
)
ROLE_NAMES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)
MERGED_PPO_ADVANTAGE_CONTRACT = "dalmuti-v4-global-merged-lomo-advantages-v2"
MERGED_BASELINE_MIN_REFERENCES = 16
MERGED_GLOBAL_SCALE_FLOOR = 0.5
MERGED_PPO_ADVANTAGE_ARRAYS = (
    "raw_returns",
    "baseline_values",
    "raw_advantages",
    "advantage_scales",
    "baseline_tiers",
    "baseline_reference_counts",
    "advantages",
    "rewards",
    "dones",
    "valid_masks",
    "ppo_eligible_masks",
    "trajectory_player_counts",
    "trajectory_roles",
    "trajectory_acts",
    "trajectory_match_clusters",
    "trajectory_monte_carlo_gammas",
)


@dataclass(frozen=True)
class BaselineRecord:
    player_count: int
    role: str
    act: int
    match_cluster: str
    value: float


@dataclass(frozen=True)
class BaselineResult:
    baseline: float
    scale: float
    tier: int
    reference_count: int


@dataclass(frozen=True)
class MergedAdvantageResult:
    ppo_trajectories: int
    ppo_samples: int
    standardized: bool
    fallback_counts: Mapping[str, int]
    global_population_scale: float
    array_binding_sha256: str


def _ordered_reference_values(records: Sequence[BaselineRecord]) -> list[float]:
    # Sorting plus fsum makes the reduction independent of input/shard order.
    ordered = sorted(
        records,
        key=lambda item: (
            item.player_count,
            item.role,
            item.act,
            item.match_cluster,
            item.value,
        ),
    )
    return [float(item.value) for item in ordered]


def leave_one_match_out_baselines(
    records: Sequence[BaselineRecord],
) -> tuple[BaselineResult, ...]:
    """Compute deterministic population baselines without own-cluster leakage."""

    output: list[BaselineResult] = []
    for target in records:
        other = [
            record
            for record in records
            if record.match_cluster != target.match_cluster
        ]
        filters = (
            lambda value: (
                value.player_count == target.player_count
                and value.role == target.role
                and value.act == target.act
            ),
            lambda value: (
                value.player_count == target.player_count
                and value.role == target.role
            ),
            lambda value: (
                value.player_count == target.player_count
                and value.act == target.act
            ),
            lambda value: value.player_count == target.player_count,
            lambda value: True,
        )
        references: list[BaselineRecord] = []
        tier = len(BASELINE_FALLBACK_HIERARCHY) - 1
        for index, predicate in enumerate(filters):
            references = [record for record in other if predicate(record)]
            if references:
                tier = index
                break
        if references:
            values = _ordered_reference_values(references)
            baseline = math.fsum(values) / len(values)
            variance = math.fsum(
                (value - baseline) * (value - baseline) for value in values
            ) / len(values)
            deviation = math.sqrt(max(0.0, variance))
            scale = deviation if deviation >= 1.0e-8 else 1.0
        else:
            baseline = 0.0
            scale = 1.0
        output.append(BaselineResult(baseline, scale, tier, len(references)))
    return tuple(output)


def _canonical_array_digest(
    arrays: Mapping[str, np.ndarray], names: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    for name in names:
        if name not in arrays:
            raise ValueError(f"merged PPO advantage contract lacks array {name}")
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def merged_ppo_advantage_array_sha256(
    arrays: Mapping[str, np.ndarray],
) -> str:
    return _canonical_array_digest(arrays, MERGED_PPO_ADVANTAGE_ARRAYS)


def _require_shape(
    arrays: Mapping[str, np.ndarray], name: str, expected: tuple[int, ...]
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"merged PPO advantage contract lacks array {name}")
    value = np.asarray(arrays[name])
    if value.shape != expected:
        raise ValueError(f"merged PPO advantage array {name} has an invalid shape")
    return value


def _recomputed_values(
    arrays: Mapping[str, np.ndarray], *, standardized: bool
) -> tuple[
    dict[str, np.ndarray],
    tuple[int, ...],
    tuple[BaselineResult, ...],
    float,
]:
    valid = np.asarray(arrays.get("valid_masks"))
    ppo = np.asarray(arrays.get("ppo_eligible_masks"))
    if valid.ndim != 2 or valid.dtype != np.dtype(np.bool_):
        raise ValueError("merged PPO valid_masks must be bool [trajectory,time]")
    if ppo.shape != valid.shape or ppo.dtype != np.dtype(np.bool_):
        raise ValueError("merged PPO eligibility must be bool [trajectory,time]")
    trajectory_count, time_steps = valid.shape
    player_counts = _require_shape(
        arrays, "trajectory_player_counts", (trajectory_count,)
    )
    roles = _require_shape(arrays, "trajectory_roles", (trajectory_count,))
    acts = _require_shape(arrays, "trajectory_acts", (trajectory_count,))
    clusters = _require_shape(
        arrays, "trajectory_match_clusters", (trajectory_count,)
    )
    gammas = _require_shape(
        arrays, "trajectory_monte_carlo_gammas", (trajectory_count,)
    )
    rewards = _require_shape(arrays, "rewards", valid.shape)
    dones = _require_shape(arrays, "dones", valid.shape)
    raw_returns = _require_shape(arrays, "raw_returns", valid.shape)
    for name in (
        "baseline_values",
        "raw_advantages",
        "advantage_scales",
        "baseline_tiers",
        "baseline_reference_counts",
        "advantages",
    ):
        _require_shape(arrays, name, valid.shape)

    ppo_indices = tuple(int(value) for value in np.flatnonzero(ppo.any(axis=1)))
    ppo_trajectory_mask = ppo.any(axis=1)
    if np.any(np.asarray(gammas)[~ppo_trajectory_mask] != 0.0):
        raise ValueError("non-PPO trajectories must use the canonical zero gamma")
    if np.any(ppo & ~valid):
        raise ValueError("merged PPO eligibility marks an invalid suffix")
    for index in ppo_indices:
        if not np.array_equal(ppo[index], valid[index]):
            raise ValueError("merged PPO eligibility must cover complete trajectories")
        if not str(clusters[index]):
            raise ValueError("merged PPO trajectory has an empty match cluster")
        player_count = int(player_counts[index])
        role = int(roles[index])
        act = int(acts[index])
        gamma = float(gammas[index])
        if not 4 <= player_count <= 10 or not 0 <= role < len(ROLE_NAMES) or act < 1:
            raise ValueError("merged PPO trajectory identity is invalid")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("merged PPO trajectory gamma is invalid")
        positions = np.flatnonzero(valid[index])
        if positions.size < 1 or not np.array_equal(
            positions, np.arange(positions.size)
        ):
            raise ValueError("merged PPO trajectory is not a contiguous prefix")
        if int(np.count_nonzero(dones[index, positions])) != 1 or not bool(
            dones[index, positions[-1]]
        ):
            raise ValueError("merged PPO trajectory is not terminal and complete")
        running = 0.0
        expected = np.zeros(positions.size, dtype=np.float64)
        for offset in range(positions.size - 1, -1, -1):
            running = float(rewards[index, offset]) + gamma * running
            expected[offset] = running
        if not np.allclose(
            raw_returns[index, positions], expected, rtol=0.0, atol=2.0e-6
        ):
            raise ValueError("merged PPO raw returns violate their Monte Carlo binding")

    records = tuple(
        BaselineRecord(
            int(player_counts[index]),
            ROLE_NAMES[int(roles[index])],
            int(acts[index]),
            str(clusters[index]),
            float(raw_returns[index, int(valid[index].sum()) - 1]),
        )
        for index in ppo_indices
    )
    # Merged v2 deliberately does not reuse a shard's tiny reference stratum.
    # It accepts the first hierarchy tier with at least 16 trajectories after
    # removing the target's entire match cluster.  If even the global tier is
    # too small (useful only for smoke fixtures), the baseline is zero and its
    # reference count is zero rather than pretending a tiny estimate is safe.
    baseline_values: list[BaselineResult] = []
    for target in records:
        other = [
            record
            for record in records
            if record.match_cluster != target.match_cluster
        ]
        filters = (
            lambda value: (
                value.player_count == target.player_count
                and value.role == target.role
                and value.act == target.act
            ),
            lambda value: (
                value.player_count == target.player_count
                and value.role == target.role
            ),
            lambda value: (
                value.player_count == target.player_count
                and value.act == target.act
            ),
            lambda value: value.player_count == target.player_count,
            lambda value: True,
        )
        accepted: list[BaselineRecord] = []
        tier = len(BASELINE_FALLBACK_HIERARCHY) - 1
        for index, predicate in enumerate(filters):
            references = [record for record in other if predicate(record)]
            if len(references) >= MERGED_BASELINE_MIN_REFERENCES:
                accepted = references
                tier = index
                break
        if accepted:
            values = _ordered_reference_values(accepted)
            baseline = math.fsum(values) / len(values)
            baseline_values.append(
                BaselineResult(baseline, 1.0, tier, len(accepted))
            )
        else:
            baseline_values.append(
                BaselineResult(
                    0.0,
                    1.0,
                    len(BASELINE_FALLBACK_HIERARCHY) - 1,
                    0,
                )
            )
    baselines = tuple(baseline_values)
    terminal_raw_advantages = [
        record.value - baseline.baseline
        for record, baseline in zip(records, baselines, strict=True)
    ]
    if terminal_raw_advantages:
        mean = math.fsum(terminal_raw_advantages) / len(terminal_raw_advantages)
        variance = math.fsum(
            (value - mean) * (value - mean)
            for value in sorted(terminal_raw_advantages)
        ) / len(terminal_raw_advantages)
        global_scale = max(math.sqrt(max(0.0, variance)), MERGED_GLOBAL_SCALE_FLOOR)
    else:
        global_scale = MERGED_GLOBAL_SCALE_FLOOR
    expected = {
        "baseline_values": np.zeros(valid.shape, np.float32),
        "raw_advantages": np.zeros(valid.shape, np.float32),
        "advantage_scales": np.ones(valid.shape, np.float32),
        "baseline_tiers": np.full(valid.shape, -1, np.int8),
        "baseline_reference_counts": np.zeros(valid.shape, np.int32),
        # Only PPO rows are meaningful here.  The caller preserves all non-PPO
        # standard advantages byte-for-byte.
        "advantages": np.zeros(valid.shape, np.float32),
    }
    for index, baseline in zip(ppo_indices, baselines, strict=True):
        mask = valid[index]
        raw = raw_returns[index, mask].astype(np.float64) - baseline.baseline
        training = raw / global_scale if standardized else raw
        expected["baseline_values"][index, mask] = baseline.baseline
        expected["raw_advantages"][index, mask] = raw
        expected["advantage_scales"][index, mask] = global_scale
        expected["baseline_tiers"][index, mask] = baseline.tier
        expected["baseline_reference_counts"][index, mask] = (
            baseline.reference_count
        )
        expected["advantages"][index, mask] = training
    return expected, ppo_indices, baselines, global_scale


def recompute_merged_ppo_advantages(
    arrays: MutableMapping[str, np.ndarray], *, standardized: bool
) -> MergedAdvantageResult:
    """Replace only PPO-derived arrays with the global merged derivation."""

    expected, ppo_indices, baselines, global_scale = _recomputed_values(
        arrays, standardized=standardized
    )
    ppo = np.asarray(arrays["ppo_eligible_masks"])
    for name, derived in expected.items():
        target = np.asarray(arrays[name])
        if name == "advantages":
            target[ppo] = derived[ppo]
        else:
            # Canonical auxiliary defaults are part of the merged contract,
            # including non-PPO rows and invalid suffixes.
            target[...] = derived
    fallback_counts = {name: 0 for name in BASELINE_FALLBACK_HIERARCHY}
    for baseline in baselines:
        fallback_counts[BASELINE_FALLBACK_HIERARCHY[baseline.tier]] += 1
    return MergedAdvantageResult(
        ppo_trajectories=len(ppo_indices),
        ppo_samples=int(ppo.sum()),
        standardized=standardized,
        fallback_counts=fallback_counts,
        global_population_scale=global_scale,
        array_binding_sha256=merged_ppo_advantage_array_sha256(arrays),
    )


def validate_merged_ppo_advantages(
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, object],
) -> MergedAdvantageResult:
    """Fail closed unless every merged PPO advantage is canonically derived."""

    expected_fields = {
        "format",
        "version",
        "standardized",
        "baseline",
        "fallbackHierarchy",
        "minimumReferenceCount",
        "insufficientReferenceFallback",
        "ownMatchClusterExcludedAtEveryTier",
        "referencePopulation",
        "recomputedAfterCompleteMerge",
        "rawReturn",
        "rawAdvantage",
        "standardizationScale",
        "globalScaleFloor",
        "globalPopulationScale",
        "fallbackCounts",
        "ppoTrajectoryCount",
        "ppoSampleCount",
        "arrayBindingSha256",
    }
    if (
        set(contract) != expected_fields
        or contract.get("format") != MERGED_PPO_ADVANTAGE_CONTRACT
        or contract.get("version") != 2
        or contract.get("standardized") is not True
        or contract.get("baseline")
        != "first >=16-reference leave-one-entire-match-cluster-out mean over all merged PPO trajectories"
        or contract.get("fallbackHierarchy")
        != list(BASELINE_FALLBACK_HIERARCHY[:-1])
        or contract.get("minimumReferenceCount") != MERGED_BASELINE_MIN_REFERENCES
        or contract.get("insufficientReferenceFallback")
        != "zero baseline and zero reference count"
        or contract.get("referencePopulation")
        != "all complete PPO-eligible trajectories in this final merged artifact"
        or contract.get("standardizationScale")
        != "one population std of merged PPO trajectory terminal raw advantages; floor 0.5; no recenter"
        or contract.get("globalScaleFloor") != MERGED_GLOBAL_SCALE_FLOOR
        or contract.get("rawReturn")
        != "per-trajectory Monte Carlo return recomputed from rewards and trajectory_monte_carlo_gammas"
        or contract.get("rawAdvantage")
        != "raw_returns minus the trajectory baseline; no recenter"
        or contract.get("ownMatchClusterExcludedAtEveryTier") is not True
        or contract.get("recomputedAfterCompleteMerge") is not True
    ):
        raise ValueError("merged PPO advantage derivation contract is missing or incompatible")
    expected, ppo_indices, baselines, global_scale = _recomputed_values(
        arrays, standardized=bool(contract["standardized"])
    )
    ppo = np.asarray(arrays["ppo_eligible_masks"])
    for name, derived in expected.items():
        actual = np.asarray(arrays[name])
        mask = ppo if name == "advantages" else np.ones(ppo.shape, dtype=np.bool_)
        if np.issubdtype(actual.dtype, np.floating):
            matches = np.allclose(actual[mask], derived[mask], rtol=0.0, atol=2.0e-6)
        else:
            matches = np.array_equal(actual[mask], derived[mask])
        if not matches:
            raise ValueError(f"merged PPO {name} is stale or has been tampered with")
    binding = merged_ppo_advantage_array_sha256(arrays)
    if contract.get("arrayBindingSha256") != binding:
        raise ValueError("merged PPO advantage array binding checksum does not match")
    fallback_counts = {name: 0 for name in BASELINE_FALLBACK_HIERARCHY}
    for baseline in baselines:
        fallback_counts[BASELINE_FALLBACK_HIERARCHY[baseline.tier]] += 1
    if contract.get("fallbackCounts") != fallback_counts:
        raise ValueError("merged PPO fallback counts do not match recomputed baselines")
    if contract.get("ppoTrajectoryCount") != len(ppo_indices) or contract.get(
        "ppoSampleCount"
    ) != int(ppo.sum()):
        raise ValueError("merged PPO advantage population counts do not match")
    declared_scale = contract.get("globalPopulationScale")
    if (
        isinstance(declared_scale, bool)
        or not isinstance(declared_scale, (int, float))
        or not math.isclose(
            float(declared_scale), global_scale, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise ValueError("merged PPO global population scale does not match")
    return MergedAdvantageResult(
        ppo_trajectories=len(ppo_indices),
        ppo_samples=int(ppo.sum()),
        standardized=bool(contract["standardized"]),
        fallback_counts=fallback_counts,
        global_population_scale=global_scale,
        array_binding_sha256=binding,
    )


__all__ = [
    "BASELINE_FALLBACK_HIERARCHY",
    "MERGED_PPO_ADVANTAGE_ARRAYS",
    "MERGED_PPO_ADVANTAGE_CONTRACT",
    "MERGED_BASELINE_MIN_REFERENCES",
    "MERGED_GLOBAL_SCALE_FLOOR",
    "BaselineRecord",
    "BaselineResult",
    "MergedAdvantageResult",
    "leave_one_match_out_baselines",
    "merged_ppo_advantage_array_sha256",
    "recompute_merged_ppo_advantages",
    "validate_merged_ppo_advantages",
]


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a merged V4 PPO advantage artifact."
    )
    parser.add_argument("npz", type=Path)
    arguments = parser.parse_args()
    with np.load(arguments.npz, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        contract = metadata.get("returnsAndAdvantages")
        if not isinstance(contract, Mapping):
            raise ValueError("merged artifact lacks returnsAndAdvantages metadata")
        result = validate_merged_ppo_advantages(archive, contract)
        ppo = np.asarray(archive["ppo_eligible_masks"])
        forced = np.asarray(archive["forced_masks"])
        nonforced = ppo & ~forced
        references = np.asarray(archive["baseline_reference_counts"])[nonforced]
        advantages = np.asarray(archive["advantages"])[nonforced]
        if references.size < 1:
            raise ValueError("merged artifact has no non-forced PPO decisions")
        print(json.dumps(
            {
                "arrayBindingSha256": result.array_binding_sha256,
                "globalPopulationScale": result.global_population_scale,
                "maxAbsAdvantage": float(np.max(np.abs(advantages))),
                "nonforcedCount": int(nonforced.sum()),
                "ppoSampleCount": result.ppo_samples,
                "ppoTrajectoryCount": result.ppo_trajectories,
                "referenceMax": int(references.max()),
                "referenceMedian": float(np.median(references)),
                "referenceMin": int(references.min()),
            },
            sort_keys=True,
        ))


if __name__ == "__main__":
    _main()
