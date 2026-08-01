from __future__ import annotations

"""Exact, seed-matched V4 evaluation against the frozen Normal policy.

The evaluator deliberately owns no game rules.  ``V4EnvAdapter`` imports the
reference environment lazily, while tests and future fused environments can
inject the same small adapter surface.  Normal actions always come from the
adapter's exact ``normal_action`` method; only candidate seats use the injected
public actor callback.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from time import perf_counter
from typing import Callable, Mapping, Protocol, Sequence


PLAYER_COUNTS = tuple(range(4, 11))
ACTS_PER_MATCH = 5
SCREENING_MATCH_COUNTS = {player_count: 60 for player_count in PLAYER_COUNTS}
DEVELOPMENT_MATCH_COUNTS = {
    4: 500,
    5: 350,
    6: 200,
    7: 150,
    8: 120,
    9: 120,
    10: 100,
}
FINAL_MATCH_COUNTS = {
    4: 2500,
    5: 1700,
    6: 900,
    7: 600,
    8: 400,
    9: 400,
    10: 300,
}
DEVELOPMENT_GATES = {
    "minPointDifference": 0.30,
    "minLowerBound": 0.20,
    "minPairwiseRate": 0.57,
}
FINAL_GATES = {
    "minPointDifference": 0.25,
    "minLowerBound": 0.15,
    "minPairwiseRate": 0.55,
}
EVALUATION_MODES = ("screening", "development", "final")
ROLES = (
    "great-dalmuti",
    "lesser-dalmuti",
    "merchant",
    "lesser-peon",
    "great-peon",
)
FINAL_SEED_START = 900_000_001
FINAL_SEED_STEP = 20_000_000
MAX_UINT32 = 0xFFFF_FFFF
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
HISTORY_INFERENCE_BUCKETS = (0, 16, 32, 64, 96, 128, 160, 192)
CANDIDATE_POLICY_MODES = ("pure-actor", "confidence-fallback", "exact-normal")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FAMILY_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


CandidatePolicy = Callable[[object], int]


@dataclass(frozen=True)
class CandidatePolicyRouting:
    """Explicitly binds how actor suggestions are routed at candidate seats."""

    mode: str = "pure-actor"
    minimum_legal_logit_margin: float | None = None
    minimum_top_probability: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in CANDIDATE_POLICY_MODES:
            raise ValueError(
                f"candidate policy mode must be one of {CANDIDATE_POLICY_MODES}"
            )
        margin = self.minimum_legal_logit_margin
        probability = self.minimum_top_probability
        if margin is not None:
            margin = _require_finite(margin, "minimum legal-logit margin")
            if margin < 0.0:
                raise ValueError("minimum legal-logit margin must be non-negative")
            object.__setattr__(self, "minimum_legal_logit_margin", margin)
        if probability is not None:
            probability = _require_finite(
                probability, "minimum top probability"
            )
            if probability < 0.0 or probability > 1.0:
                raise ValueError("minimum top probability must be from 0 through 1")
            object.__setattr__(self, "minimum_top_probability", probability)
        if self.mode == "confidence-fallback":
            if margin is None and probability is None:
                raise ValueError(
                    "confidence-fallback requires at least one confidence threshold"
                )
        elif margin is not None or probability is not None:
            raise ValueError(
                "confidence thresholds are valid only in confidence-fallback mode"
            )

    def report_value(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "minimumLegalLogitMargin": self.minimum_legal_logit_margin,
            "minimumTopProbability": self.minimum_top_probability,
            "thresholdComparison": "all-configured-thresholds-met-inclusive",
            "forcedDecisionRule": (
                "actor-path" if self.mode == "pure-actor" else "exact-normal"
            ),
            "runtimeErrorFallback": False,
        }


@dataclass(frozen=True)
class ActorActionDiagnostics:
    action: int
    legal_logit_margin: float | None
    top_probability: float
    legal_action_count: int


class EnvironmentAdapter(Protocol):
    def make_env(self, player_count: int, acts: int, seed: int) -> object: ...
    def current_player_id(self, env: object) -> int: ...
    def player_order(self, env: object) -> Sequence[int]: ...
    def observe(self, env: object) -> object: ...
    def normal_action(self, env: object) -> int: ...
    def legal_mask(self, env: object) -> object: ...
    def step(self, env: object, action: int) -> object: ...
    def terminated(self, env: object) -> bool: ...


class V4EnvAdapter:
    """Loose adapter for ``v4_env.DalmutiScalarEnv``.

    The lazy import lets statistical and contract tests run on a CPU machine
    without PyTorch.  ``_order`` is intentionally isolated here so the
    evaluator does not otherwise depend on reference-environment internals.
    """

    def make_env(self, player_count: int, acts: int, seed: int) -> object:
        from v4_env import DalmutiScalarEnv

        return DalmutiScalarEnv(player_count, acts=acts, seed=seed, device="cpu")

    def current_player_id(self, env: object) -> int:
        return int(getattr(env, "current_player_id"))

    def player_order(self, env: object) -> Sequence[int]:
        return tuple(int(value) for value in getattr(env, "_order"))

    def observe(self, env: object) -> object:
        return getattr(env, "observe")()

    def normal_action(self, env: object) -> int:
        return int(getattr(env, "normal_action")())

    def legal_mask(self, env: object) -> object:
        return getattr(env, "legal_mask")()

    def step(self, env: object, action: int) -> object:
        return getattr(env, "step")(action)

    def terminated(self, env: object) -> bool:
        return bool(getattr(env, "terminated"))


@dataclass(frozen=True)
class EvaluationBindings:
    artifact_sha256: str
    actor_sha256: str
    observation_contract_sha256: str
    normal_baseline_sha256: str
    normal_baseline_source_commit: str
    actual_files_verified: bool = False

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_sha256, "actor bundle artifact SHA-256")
        _require_sha256(self.actor_sha256, "actor SHA-256")
        _require_sha256(
            self.observation_contract_sha256,
            "observation contract SHA-256",
        )
        _require_sha256(self.normal_baseline_sha256, "Normal baseline SHA-256")
        if not _GIT_COMMIT_RE.fullmatch(self.normal_baseline_source_commit):
            raise ValueError("Normal baseline source commit must be 40 lowercase hex")
        if not isinstance(self.actual_files_verified, bool):
            raise ValueError("actual_files_verified must be boolean")

    def report_value(self) -> dict[str, str]:
        return {
            "artifactSha256": self.artifact_sha256,
            "modelSha256": self.actor_sha256,
            "observationSchemaSha256": self.observation_contract_sha256,
            "normalBaselineSha256": self.normal_baseline_sha256,
            "normalBaselineSourceCommit": self.normal_baseline_source_commit,
        }


@dataclass(frozen=True)
class EvaluationSeedSchedule:
    """Injected, contiguous match-seed schedule for one immutable family."""

    mode: str
    family_id: str
    base_seed: int

    def __post_init__(self) -> None:
        if self.mode not in EVALUATION_MODES:
            raise ValueError(f"mode must be one of {EVALUATION_MODES}")
        if not _FAMILY_RE.fullmatch(self.family_id):
            raise ValueError("family_id must be a lowercase kebab-case identifier")
        _require_positive_int(self.base_seed, "base seed")
        if self.mode == "final":
            if self.base_seed < FINAL_SEED_START:
                raise ValueError("final evaluation must use the reserved final namespace")
            if (self.base_seed - FINAL_SEED_START) % FINAL_SEED_STEP != 0:
                raise ValueError("final base seed is outside the reservation sequence")
        elif self.base_seed >= FINAL_SEED_START:
            raise ValueError("non-final evaluation must not consume final seed namespace")

    def seed_for(self, player_count: int, match_index: int) -> int:
        if player_count not in PLAYER_COUNTS:
            raise ValueError("player_count must be from 4 through 10")
        if isinstance(match_index, bool) or not isinstance(match_index, int) or match_index < 0:
            raise ValueError("match_index must be a non-negative integer")
        seed = self.base_seed + player_count * 1_000_000 + match_index
        if seed > MAX_UINT32:
            raise ValueError("match seed exceeds uint32")
        return seed

    def ranges(self, match_counts: Mapping[int, int]) -> list[dict[str, int]]:
        return [
            {
                "playerCount": player_count,
                "matches": int(match_counts[player_count]),
                "start": self.seed_for(player_count, 0),
                "end": self.seed_for(
                    player_count, int(match_counts[player_count]) - 1
                ),
            }
            for player_count in PLAYER_COUNTS
        ]


class _Mulberry32:
    def __init__(self, seed: int):
        self.state = int(seed) & 0xFFFF_FFFF

    def next_uint32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFF_FFFF
        value = self.state
        value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFF_FFFF
        value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
        return (value ^ (value >> 14)) & 0xFFFF_FFFF

    def integer(self, maximum_exclusive: int) -> int:
        if maximum_exclusive < 1:
            raise ValueError("maximum_exclusive must be positive")
        limit = (1 << 32) - ((1 << 32) % maximum_exclusive)
        while True:
            value = self.next_uint32()
            if value < limit:
                return value % maximum_exclusive


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hex")
    return value


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def sha256_file(path: str | Path) -> str:
    """Hash one required input file without trusting caller-supplied metadata."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"required binding input is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_digest(actual: str, expected: str | None, label: str) -> str:
    if expected is not None:
        expected_value = _require_sha256(expected, f"declared {label}")
        if actual != expected_value:
            raise ValueError(f"declared {label} does not match the actual file")
    return actual


def verify_frozen_normal_source(
    source_path: str | Path,
    *,
    repository_root: str | Path,
    source_commit: str,
    expected_sha256: str | None = None,
) -> str:
    """Bind Normal to both current bytes and the exact Git blob at a real commit.

    Merely accepting a 40-hex string lets a report claim arbitrary provenance.
    This verifier resolves the commit in the supplied repository and requires the
    evaluated source bytes to be byte-identical to that commit's blob.
    """

    if not _GIT_COMMIT_RE.fullmatch(source_commit):
        raise ValueError("Normal baseline source commit must be 40 lowercase hex")
    root = Path(repository_root).resolve()
    source = Path(source_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root is missing: {root}")
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("frozen Normal source must be inside repository root") from error
    actual_bytes = source.read_bytes()
    actual_sha256 = hashlib.sha256(actual_bytes).hexdigest()
    _expected_digest(actual_sha256, expected_sha256, "Normal baseline SHA-256")

    git_prefix = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    try:
        resolved = subprocess.run(
            [*git_prefix, "rev-parse", "--verify", f"{source_commit}^{{commit}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("ascii").strip()
        committed_bytes = subprocess.run(
            [*git_prefix, "show", f"{source_commit}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Normal baseline source commit cannot be verified") from error
    if resolved != source_commit:
        raise ValueError("Normal baseline source commit is not the resolved full commit")
    if committed_bytes != actual_bytes:
        raise ValueError("frozen Normal source differs from the bound Git commit")
    return actual_sha256


def resolve_cli_evaluation_bindings(
    *,
    artifact_sha256: str,
    actor_sha256: str,
    observation_contract_path: str | Path,
    frozen_normal_source_path: str | Path,
    repository_root: str | Path,
    frozen_normal_source_commit: str,
    expected_observation_sha256: str | None = None,
    expected_normal_sha256: str | None = None,
) -> EvaluationBindings:
    """Build evaluator bindings exclusively from verified on-disk inputs."""

    observation_sha256 = sha256_file(observation_contract_path)
    _expected_digest(
        observation_sha256,
        expected_observation_sha256,
        "observation contract SHA-256",
    )
    normal_sha256 = verify_frozen_normal_source(
        frozen_normal_source_path,
        repository_root=repository_root,
        source_commit=frozen_normal_source_commit,
        expected_sha256=expected_normal_sha256,
    )
    return EvaluationBindings(
        artifact_sha256=artifact_sha256,
        actor_sha256=actor_sha256,
        observation_contract_sha256=observation_sha256,
        normal_baseline_sha256=normal_sha256,
        normal_baseline_source_commit=frozen_normal_source_commit,
        actual_files_verified=True,
    )


def history_inference_bucket(length: int, *, max_history: int = 192) -> int:
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("history length must be a non-negative integer")
    _require_positive_int(max_history, "maximum history")
    if length > max_history:
        raise ValueError("history length exceeds actor config")
    boundaries = sorted(
        {min(boundary, max_history) for boundary in HISTORY_INFERENCE_BUCKETS}
    )
    if max_history not in boundaries:
        boundaries.append(max_history)
        boundaries.sort()
    return next(boundary for boundary in boundaries if length <= boundary)


def _expected_match_counts(mode: str) -> dict[int, int]:
    if mode == "screening":
        return dict(SCREENING_MATCH_COUNTS)
    if mode == "development":
        return dict(DEVELOPMENT_MATCH_COUNTS)
    if mode == "final":
        return dict(FINAL_MATCH_COUNTS)
    raise ValueError(f"mode must be one of {EVALUATION_MODES}")


def _expected_gates(mode: str) -> dict[str, float]:
    return dict(FINAL_GATES if mode == "final" else DEVELOPMENT_GATES)


def validate_evaluation_plan(
    *,
    mode: str,
    match_counts: Mapping[int, int],
    acts: int,
    gates: Mapping[str, float],
    seed_schedule: EvaluationSeedSchedule,
    final_seed_reservation: Mapping[str, object] | None = None,
) -> None:
    if mode not in EVALUATION_MODES:
        raise ValueError(f"mode must be one of {EVALUATION_MODES}")
    if acts != ACTS_PER_MATCH:
        raise ValueError("V4 evaluation requires exactly five acts per match")
    expected_counts = _expected_match_counts(mode)
    normalized_counts = {
        int(key): value for key, value in match_counts.items()
    }
    if set(normalized_counts) != set(PLAYER_COUNTS):
        raise ValueError("match counts must contain exactly p4 through p10")
    for player_count, expected in expected_counts.items():
        value = normalized_counts[player_count]
        _require_positive_int(value, f"p{player_count} match count")
        if value != expected:
            raise ValueError(
                f"{mode} p{player_count} match count must be exactly {expected}"
            )
    expected_gates = _expected_gates(mode)
    if set(gates) != set(expected_gates):
        raise ValueError("promotion gates contain unexpected keys")
    for key, expected in expected_gates.items():
        if _require_finite(gates[key], key) != expected:
            raise ValueError(f"{mode} {key} gate must be exactly {expected}")
    if seed_schedule.mode != mode:
        raise ValueError("seed schedule mode does not match evaluation mode")
    ranges = seed_schedule.ranges(expected_counts)
    if mode != "final":
        if final_seed_reservation is not None:
            raise ValueError("only final evaluation may receive a final reservation")
        if any(value["end"] >= FINAL_SEED_START for value in ranges):
            raise ValueError("non-final match ranges overlap the final namespace")
        return
    if not isinstance(final_seed_reservation, Mapping):
        raise ValueError("final evaluation requires an atomic final seed reservation")
    if (
        final_seed_reservation.get("format")
        != "dalmuti-v4-final-seed-reservation"
        or final_seed_reservation.get("version") != 1
        or final_seed_reservation.get("baseSeed") != seed_schedule.base_seed
        or final_seed_reservation.get("reuseForbidden") is not True
        or final_seed_reservation.get("finalFeedbackPolicy")
        != "sealed-holdout-not-a-training-input"
    ):
        raise ValueError("final seed reservation contract is invalid")
    reservation_ranges = final_seed_reservation.get("matchSeedRanges")
    if canonical_json_bytes(reservation_ranges) != canonical_json_bytes(ranges):
        raise ValueError("final seed reservation ranges do not match evaluation")


def role_for_seat(seat_index: int, player_count: int) -> str:
    if seat_index == 0:
        return "great-dalmuti"
    if seat_index == 1:
        return "lesser-dalmuti"
    if seat_index == player_count - 2:
        return "lesser-peon"
    if seat_index == player_count - 1:
        return "great-peon"
    return "merchant"


def rotating_candidate_seats(player_count: int, match_index: int) -> tuple[int, ...]:
    _require_positive_int(player_count, "player count")
    if player_count not in PLAYER_COUNTS:
        raise ValueError("player count must be from 4 through 10")
    if isinstance(match_index, bool) or not isinstance(match_index, int) or match_index < 0:
        raise ValueError("match index must be a non-negative integer")
    lower = player_count // 2
    candidate_count = (
        lower
        if player_count % 2 == 0 or match_index % 2 == 1
        else lower + 1
    )
    # Consume a single cyclic stream of seat assignments.  Unlike advancing
    # the start by one per match, this keeps every prefix balanced even when
    # the preset match count is not a multiple of the player count.
    extras_before = (match_index + 1) // 2 if player_count % 2 else 0
    assigned_before = match_index * lower + extras_before
    start = assigned_before % player_count
    return tuple((start + offset) % player_count for offset in range(candidate_count))


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    location = probability * (len(sorted_values) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return float(sorted_values[lower])
    weight = location - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def deterministic_cluster_bootstrap95(
    samples: Sequence[float],
    *,
    seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    if not samples:
        raise ValueError("cluster bootstrap needs at least one match")
    values = [_require_finite(value, "cluster sample") for value in samples]
    _require_positive_int(seed, "bootstrap seed")
    _require_positive_int(resamples, "bootstrap resamples")
    mean = sum(values) / len(values)
    if len(values) == 1:
        low = high = mean
    else:
        rng = _Mulberry32(seed)
        bootstrap_means = []
        for _ in range(resamples):
            total = 0.0
            for _ in values:
                total += values[rng.integer(len(values))]
            bootstrap_means.append(total / len(values))
        bootstrap_means.sort()
        low = _percentile(bootstrap_means, 0.025)
        high = _percentile(bootstrap_means, 0.975)
    return {
        "unit": "seed-matched-match",
        "method": "deterministic-percentile-bootstrap",
        "clusters": len(values),
        "resamples": resamples,
        "seed": seed,
        "mean": mean,
        "low": low,
        "high": high,
    }


def _bootstrap_seed(schedule: EvaluationSeedSchedule, player_count: int) -> int:
    material = (
        f"dalmuti-v4-bootstrap-v1:{schedule.mode}:{schedule.family_id}:"
        f"{schedule.base_seed}:{player_count}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
    return value or 1


def _outcome_totals() -> dict[str, float | int]:
    return {"chips": 0.0, "places": 0, "firsts": 0, "lasts": 0, "seatActs": 0}


def _record_outcome(
    totals: dict[str, float | int],
    *,
    chips: float,
    place: int,
    player_count: int,
) -> None:
    totals["chips"] = float(totals["chips"]) + chips
    totals["places"] = int(totals["places"]) + place
    totals["firsts"] = int(totals["firsts"]) + int(place == 1)
    totals["lasts"] = int(totals["lasts"]) + int(place == player_count)
    totals["seatActs"] = int(totals["seatActs"]) + 1


def _summarize_outcome(totals: Mapping[str, float | int]) -> dict[str, float | int | None]:
    seat_acts = int(totals["seatActs"])
    if seat_acts == 0:
        return {
            "meanChip": None,
            "meanPlace": None,
            "firstRate": None,
            "lastRate": None,
            "seatActs": 0,
        }
    return {
        "meanChip": float(totals["chips"]) / seat_acts,
        "meanPlace": int(totals["places"]) / seat_acts,
        "firstRate": int(totals["firsts"]) / seat_acts,
        "lastRate": int(totals["lasts"]) / seat_acts,
        "seatActs": seat_acts,
    }


def _expected_chip_award(place: int, player_count: int) -> int:
    if place == 1:
        return 4
    if place == 2:
        return 3
    if place == player_count - 1:
        return 1
    if place == player_count:
        return 0
    return 2


def _step_info(step_result: object) -> Mapping[str, object]:
    info = getattr(step_result, "info", None)
    if info is None and isinstance(step_result, Mapping):
        info = step_result.get("info")
    if not isinstance(info, Mapping):
        raise ValueError("environment step did not return an info mapping")
    return info


def _is_legal(mask: object, action: int) -> bool:
    if isinstance(action, bool) or not isinstance(action, int) or action < 0:
        return False
    try:
        value = mask[action]  # type: ignore[index]
    except (IndexError, KeyError):
        return False
    except Exception as error:
        raise ValueError("adapter legal mask is not indexable") from error
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _legal_action_count(mask: object) -> int:
    sum_method = getattr(mask, "sum", None)
    if callable(sum_method):
        try:
            total = sum_method()
            if hasattr(total, "item"):
                total = total.item()
            if isinstance(total, bool) or not isinstance(total, (int, float)):
                raise TypeError("mask sum is not numeric")
            count = int(total)
            if float(total) != count:
                raise ValueError("adapter legal mask is not boolean-valued")
            if count < 1:
                raise ValueError("adapter legal mask contains no legal action")
            return count
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("adapter legal mask cannot be summed") from error
    try:
        length = len(mask)  # type: ignore[arg-type]
    except Exception as error:
        raise ValueError("adapter legal mask has no finite length") from error
    count = sum(int(_is_legal(mask, action)) for action in range(length))
    if count < 1:
        raise ValueError("adapter legal mask contains no legal action")
    return count


_DECISION_COUNT_KEYS = (
    "candidateDecisions",
    "actorDecisions",
    "fallbackDecisions",
    "forcedDecisions",
    "deviationsFromNormal",
)


def _decision_counts() -> dict[str, int]:
    return {key: 0 for key in _DECISION_COUNT_KEYS}


def _record_candidate_decision(
    counts: dict[str, int],
    *,
    source: str,
    forced: bool,
    deviated_from_normal: bool,
) -> None:
    if source not in ("actor", "fallback"):
        raise ValueError("candidate action source must be actor or fallback")
    counts["candidateDecisions"] += 1
    counts["actorDecisions" if source == "actor" else "fallbackDecisions"] += 1
    counts["forcedDecisions"] += int(forced)
    counts["deviationsFromNormal"] += int(deviated_from_normal)


def _merge_decision_counts(
    target: dict[str, int], source: Mapping[str, object]
) -> None:
    for key in _DECISION_COUNT_KEYS:
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("candidate decision audit contains invalid counts")
        target[key] += value


def _summarize_decision_counts(counts: Mapping[str, object]) -> dict[str, object]:
    normalized = _decision_counts()
    _merge_decision_counts(normalized, counts)
    decisions = normalized["candidateDecisions"]
    if normalized["actorDecisions"] + normalized["fallbackDecisions"] != decisions:
        raise ValueError("candidate decision routing counts do not sum")
    if normalized["forcedDecisions"] > decisions:
        raise ValueError("forced candidate decisions exceed total decisions")
    if normalized["deviationsFromNormal"] > normalized["actorDecisions"]:
        raise ValueError("only actor-routed decisions may deviate from Normal")
    denominator = float(decisions) if decisions else None
    return {
        **normalized,
        "actorRate": (
            None if denominator is None else normalized["actorDecisions"] / denominator
        ),
        "fallbackRate": (
            None
            if denominator is None
            else normalized["fallbackDecisions"] / denominator
        ),
        "forcedRate": (
            None if denominator is None else normalized["forcedDecisions"] / denominator
        ),
        "deviationRate": (
            None
            if denominator is None
            else normalized["deviationsFromNormal"] / denominator
        ),
    }


def _routing_from_report(value: object) -> CandidatePolicyRouting:
    if not isinstance(value, Mapping):
        raise ValueError("candidate routing audit is missing")
    routing = CandidatePolicyRouting(
        mode=value.get("mode"),  # type: ignore[arg-type]
        minimum_legal_logit_margin=value.get("minimumLegalLogitMargin"),  # type: ignore[arg-type]
        minimum_top_probability=value.get("minimumTopProbability"),  # type: ignore[arg-type]
    )
    if dict(value) != routing.report_value():
        raise ValueError("candidate routing audit is invalid")
    return routing


def _validate_decision_summary(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("candidate decision summary is missing")
    counts = {key: value.get(key) for key in _DECISION_COUNT_KEYS}
    expected = _summarize_decision_counts(counts)
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError("candidate decision audit rates disagree with counts")
    return {key: int(expected[key]) for key in _DECISION_COUNT_KEYS}


def _validate_candidate_decision_audit(
    value: object, *, include_player_counts: bool
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("candidate decision audit is missing")
    overall = _validate_decision_summary(value.get("overall"))
    by_act = value.get("byAct")
    if not isinstance(by_act, list) or len(by_act) != ACTS_PER_MATCH:
        raise ValueError("candidate decision by-act audit is invalid")
    by_act_total = _decision_counts()
    for expected_act, act_value in enumerate(by_act, start=1):
        if not isinstance(act_value, Mapping) or act_value.get("act") != expected_act:
            raise ValueError("candidate decision act identity is invalid")
        _merge_decision_counts(by_act_total, _validate_decision_summary(act_value))
    if by_act_total != overall:
        raise ValueError("candidate decision by-act counts do not sum to overall")
    if include_player_counts:
        by_player_count = value.get("byPlayerCount")
        if not isinstance(by_player_count, list) or len(by_player_count) != len(
            PLAYER_COUNTS
        ):
            raise ValueError("candidate decision by-player-count audit is invalid")
    elif "byPlayerCount" in value:
        raise ValueError("per-player result must not contain nested player counts")
    return overall


def _candidate_batch_diagnostics(
    candidate_policy: CandidatePolicy,
    observations: Sequence[object],
) -> list[ActorActionDiagnostics]:
    if not observations:
        return []
    method = getattr(candidate_policy, "action_diagnostics", None)
    if not callable(method):
        raise ValueError(
            "confidence-fallback requires actor logit diagnostics; "
            "automatic policy fallback is forbidden"
        )
    raw_values = list(method(observations))
    if len(raw_values) != len(observations):
        raise ValueError("candidate diagnostics returned the wrong action count")
    values: list[ActorActionDiagnostics] = []
    for value in raw_values:
        if not isinstance(value, ActorActionDiagnostics):
            raise ValueError("candidate diagnostics returned an invalid record")
        if isinstance(value.action, bool) or not isinstance(value.action, int):
            raise ValueError("candidate diagnostics action must be an integer")
        _require_positive_int(value.legal_action_count, "legal action count")
        probability = _require_finite(value.top_probability, "top probability")
        if probability < 0.0 or probability > 1.0:
            raise ValueError("actor top probability must be from 0 through 1")
        if value.legal_action_count == 1:
            if value.legal_logit_margin is not None:
                raise ValueError("a forced actor decision must have a null margin")
        else:
            margin = _require_finite(value.legal_logit_margin, "legal-logit margin")
            if margin < 0.0:
                raise ValueError("actor legal-logit margin must be non-negative")
        values.append(value)
    return values


def _candidate_ids(
    initial_order: Sequence[int], player_count: int, match_index: int
) -> tuple[int, ...]:
    seats = rotating_candidate_seats(player_count, match_index)
    return tuple(int(initial_order[seat]) for seat in seats)


def _pairwise_finish(
    finish_order: Sequence[int], candidate_ids: set[int]
) -> tuple[int, int]:
    before = comparisons = 0
    for left_index, left in enumerate(finish_order):
        left_candidate = int(left) in candidate_ids
        for right in finish_order[left_index + 1 :]:
            right_candidate = int(right) in candidate_ids
            if left_candidate == right_candidate:
                continue
            comparisons += 1
            before += int(left_candidate)
    if comparisons < 1:
        raise ValueError("a match must contain candidate-Normal comparisons")
    return before, comparisons


def _evaluate_effect_gate(
    mean_difference: float,
    lower_bound: float,
    pairwise_rate: float,
    gates: Mapping[str, float],
) -> dict[str, bool]:
    point = mean_difference >= gates["minPointDifference"]
    lower = lower_bound >= gates["minLowerBound"]
    pairwise = pairwise_rate >= gates["minPairwiseRate"]
    return {
        "pointDifferencePassed": point,
        "lowerBoundPassed": lower,
        "pairwiseRatePassed": pairwise,
        "passed": point and lower and pairwise,
    }


@dataclass
class _EvaluationLane:
    match_index: int
    seed: int
    env: object
    candidate_seats: tuple[int, ...]
    candidate_ids: set[int]
    order: tuple[int, ...]
    match_groups: dict[str, dict[str, float | int]]
    pairwise_before: int = 0
    pairwise_comparisons: int = 0
    completed_acts: int = 0


def _candidate_batch_actions(
    candidate_policy: CandidatePolicy,
    observations: Sequence[object],
) -> list[int]:
    if not observations:
        return []
    batch_method = getattr(candidate_policy, "actions", None)
    if callable(batch_method):
        values = list(batch_method(observations))
    else:
        values = [candidate_policy(observation) for observation in observations]
    if len(values) != len(observations):
        raise ValueError("candidate batch policy returned the wrong action count")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("candidate policy must return integer actions")
    return values


def evaluate_player_count(
    *,
    player_count: int,
    matches: int,
    acts: int,
    seed_schedule: EvaluationSeedSchedule,
    candidate_policy: CandidatePolicy,
    adapter: EnvironmentAdapter,
    gates: Mapping[str, float],
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    batch_size: int = 1,
    candidate_policy_routing: CandidatePolicyRouting | None = None,
) -> dict[str, object]:
    if player_count not in PLAYER_COUNTS:
        raise ValueError("player_count must be from 4 through 10")
    _require_positive_int(matches, "matches")
    if acts != ACTS_PER_MATCH:
        raise ValueError("evaluation requires five acts")
    _require_positive_int(batch_size, "batch size")
    routing = candidate_policy_routing or CandidatePolicyRouting()
    role_totals = {
        role: {"candidate": _outcome_totals(), "normal": _outcome_totals()}
        for role in ROLES
    }
    groups = {"candidate": _outcome_totals(), "normal": _outcome_totals()}
    initial_seat_candidate_counts = [0] * player_count
    cluster_differences: list[float | None] = [None] * matches
    cluster_records: list[dict[str, object] | None] = [None] * matches
    pairwise_before = pairwise_comparisons = decisions = 0
    candidate_decision_counts = _decision_counts()
    candidate_decision_counts_by_act = [
        _decision_counts() for _ in range(ACTS_PER_MATCH)
    ]

    def make_lane(match_index: int) -> _EvaluationLane:
        match_seed = seed_schedule.seed_for(player_count, match_index)
        env = adapter.make_env(player_count, acts, match_seed)
        initial_order = tuple(int(value) for value in adapter.player_order(env))
        if len(initial_order) != player_count or len(set(initial_order)) != player_count:
            raise ValueError("environment initial player order is invalid")
        candidate_seats = rotating_candidate_seats(player_count, match_index)
        for seat in candidate_seats:
            initial_seat_candidate_counts[seat] += 1
        return _EvaluationLane(
            match_index=match_index,
            seed=match_seed,
            env=env,
            candidate_seats=candidate_seats,
            candidate_ids=set(
                _candidate_ids(initial_order, player_count, match_index)
            ),
            order=initial_order,
            match_groups={
                "candidate": _outcome_totals(),
                "normal": _outcome_totals(),
            },
        )

    # A rolling pool avoids waiting for the slowest lane in a fixed chunk.
    # Lane ordering remains stable and all cluster records are stored by match
    # index, so this scheduling optimization cannot change report bytes.
    for _ in (0,):
        next_match_index = min(matches, batch_size)
        lanes = [make_lane(index) for index in range(next_match_index)]

        active = lanes
        while active:
            actions_by_match: dict[int, int] = {}
            candidate_rows: list[
                tuple[_EvaluationLane, object, object, int, int]
            ] = []
            for lane in active:
                actor_id = int(adapter.current_player_id(lane.env))
                if actor_id in lane.candidate_ids:
                    legal_mask = adapter.legal_mask(lane.env)
                    legal_count = _legal_action_count(legal_mask)
                    normal_action = adapter.normal_action(lane.env)
                    if not _is_legal(legal_mask, normal_action):
                        raise ValueError(
                            f"exact Normal selected illegal action {normal_action}"
                        )
                    candidate_rows.append(
                        (
                            lane,
                            adapter.observe(lane.env),
                            legal_mask,
                            legal_count,
                            normal_action,
                        )
                    )
                else:
                    actions_by_match[lane.match_index] = adapter.normal_action(lane.env)

            routed_actions: list[tuple[_EvaluationLane, int, int, str, bool]] = []
            if routing.mode == "pure-actor":
                actor_actions = _candidate_batch_actions(
                    candidate_policy, [row[1] for row in candidate_rows]
                )
                for row, action in zip(candidate_rows, actor_actions):
                    lane, _, _, legal_count, normal_action = row
                    routed_actions.append(
                        (lane, action, normal_action, "actor", legal_count == 1)
                    )
            elif routing.mode == "exact-normal":
                for lane, _, _, legal_count, normal_action in candidate_rows:
                    routed_actions.append(
                        (lane, normal_action, normal_action, "fallback", legal_count == 1)
                    )
            else:
                nonforced_rows = [row for row in candidate_rows if row[3] > 1]
                diagnostics = _candidate_batch_diagnostics(
                    candidate_policy, [row[1] for row in nonforced_rows]
                )
                diagnostics_by_match = {
                    row[0].match_index: diagnostic
                    for row, diagnostic in zip(nonforced_rows, diagnostics)
                }
                for lane, _, legal_mask, legal_count, normal_action in candidate_rows:
                    if legal_count == 1:
                        routed_actions.append(
                            (lane, normal_action, normal_action, "fallback", True)
                        )
                        continue
                    diagnostic = diagnostics_by_match[lane.match_index]
                    if diagnostic.legal_action_count != legal_count:
                        raise ValueError(
                            "actor diagnostic legal-action count disagrees with environment"
                        )
                    if not _is_legal(legal_mask, diagnostic.action):
                        raise ValueError(
                            f"actor diagnostics selected illegal action {diagnostic.action}"
                        )
                    margin_passed = (
                        routing.minimum_legal_logit_margin is None
                        or float(diagnostic.legal_logit_margin)
                        >= routing.minimum_legal_logit_margin
                    )
                    probability_passed = (
                        routing.minimum_top_probability is None
                        or diagnostic.top_probability
                        >= routing.minimum_top_probability
                    )
                    if margin_passed and probability_passed:
                        routed_actions.append(
                            (
                                lane,
                                diagnostic.action,
                                normal_action,
                                "actor",
                                False,
                            )
                        )
                    else:
                        routed_actions.append(
                            (lane, normal_action, normal_action, "fallback", False)
                        )

            for lane, action, normal_action, source, forced in routed_actions:
                actions_by_match[lane.match_index] = action
                _record_candidate_decision(
                    candidate_decision_counts,
                    source=source,
                    forced=forced,
                    deviated_from_normal=action != normal_action,
                )
                _record_candidate_decision(
                    candidate_decision_counts_by_act[lane.completed_acts],
                    source=source,
                    forced=forced,
                    deviated_from_normal=action != normal_action,
                )

            next_active: list[_EvaluationLane] = []
            for lane in active:
                action = actions_by_match[lane.match_index]
                if not _is_legal(adapter.legal_mask(lane.env), action):
                    raise ValueError(f"policy selected illegal action {action}")
                result = adapter.step(lane.env, action)
                decisions += 1
                info = _step_info(result)
                act_result = info.get("act_result")
                if act_result is not None:
                    if not isinstance(act_result, Mapping):
                        raise ValueError("act_result must be a mapping")
                    finish_order = tuple(
                        int(value) for value in act_result.get("finish_order", ())
                    )
                    chip_awards = act_result.get("chip_awards")
                    if (
                        len(finish_order) != player_count
                        or set(finish_order) != set(lane.order)
                        or not isinstance(chip_awards, Mapping)
                    ):
                        raise ValueError("environment returned an invalid act result")
                    roles_by_id = {
                        player_id: role_for_seat(index, player_count)
                        for index, player_id in enumerate(lane.order)
                    }
                    for place, player_id in enumerate(finish_order, start=1):
                        chips_value = chip_awards.get(player_id)
                        if chips_value is None:
                            chips_value = chip_awards.get(str(player_id))
                        chips = _require_finite(chips_value, "chip award")
                        if chips != _expected_chip_award(place, player_count):
                            raise ValueError(
                                "chip award disagrees with the exact round score"
                            )
                        group = (
                            "candidate"
                            if player_id in lane.candidate_ids
                            else "normal"
                        )
                        _record_outcome(
                            groups[group],
                            chips=chips,
                            place=place,
                            player_count=player_count,
                        )
                        _record_outcome(
                            lane.match_groups[group],
                            chips=chips,
                            place=place,
                            player_count=player_count,
                        )
                        _record_outcome(
                            role_totals[roles_by_id[player_id]][group],
                            chips=chips,
                            place=place,
                            player_count=player_count,
                        )
                    act_before, act_comparisons = _pairwise_finish(
                        finish_order, lane.candidate_ids
                    )
                    lane.pairwise_before += act_before
                    lane.pairwise_comparisons += act_comparisons
                    pairwise_before += act_before
                    pairwise_comparisons += act_comparisons
                    lane.completed_acts += 1
                    lane.order = finish_order

                if not adapter.terminated(lane.env):
                    next_active.append(lane)
                    continue
                if lane.completed_acts != acts:
                    raise ValueError(
                        f"environment completed {lane.completed_acts} acts; expected {acts}"
                    )
                candidate_summary = _summarize_outcome(
                    lane.match_groups["candidate"]
                )
                normal_summary = _summarize_outcome(lane.match_groups["normal"])
                candidate_mean = candidate_summary["meanChip"]
                normal_mean = normal_summary["meanChip"]
                if candidate_mean is None or normal_mean is None:
                    raise ValueError("both policy groups require outcomes")
                difference = float(candidate_mean) - float(normal_mean)
                cluster_differences[lane.match_index] = difference
                cluster_records[lane.match_index] = {
                    "matchIndex": lane.match_index,
                    "seed": lane.seed,
                    "candidateInitialSeats": list(lane.candidate_seats),
                    "meanChipDifference": difference,
                    "candidateBefore": lane.pairwise_before,
                    "comparisons": lane.pairwise_comparisons,
                }
            while len(next_active) < batch_size and next_match_index < matches:
                next_active.append(make_lane(next_match_index))
                next_match_index += 1
            active = next_active

    if any(value is None for value in cluster_differences) or any(
        value is None for value in cluster_records
    ):
        raise RuntimeError("evaluation did not complete every match cluster")
    completed_differences = [float(value) for value in cluster_differences]
    completed_records = [value for value in cluster_records if value is not None]

    interval = deterministic_cluster_bootstrap95(
        completed_differences,
        seed=_bootstrap_seed(seed_schedule, player_count),
        resamples=bootstrap_resamples,
    )
    candidate = _summarize_outcome(groups["candidate"])
    normal = _summarize_outcome(groups["normal"])
    # The inferential and point-estimate unit is the seed-matched match.  This
    # also prevents an odd-player table's alternating  floor/ceil group size
    # from silently reweighting clusters in the point estimate.
    mean_difference = float(interval["mean"])
    pooled_seat_act_difference = (
        float(candidate["meanChip"]) - float(normal["meanChip"])
    )
    pairwise_rate = pairwise_before / pairwise_comparisons
    role_summary: dict[str, object] = {}
    for role in ROLES:
        candidate_role = _summarize_outcome(role_totals[role]["candidate"])
        normal_role = _summarize_outcome(role_totals[role]["normal"])
        candidate_count = int(candidate_role["seatActs"])
        normal_count = int(normal_role["seatActs"])
        assignment_count = candidate_count + normal_count
        role_summary[role] = {
            "candidate": candidate_role,
            "normal": normal_role,
            "candidateAssignmentRate": (
                None if assignment_count == 0 else candidate_count / assignment_count
            ),
            "meanChipDifference": (
                None
                if candidate_role["meanChip"] is None or normal_role["meanChip"] is None
                else float(candidate_role["meanChip"]) - float(normal_role["meanChip"])
            ),
        }
    minimum = min(initial_seat_candidate_counts)
    maximum = max(initial_seat_candidate_counts)
    cluster_bytes = canonical_json_bytes(completed_records)
    result: dict[str, object] = {
        "playerCount": player_count,
        "matches": matches,
        "actsPerMatch": acts,
        "decisions": decisions,
        "candidate": candidate,
        "normal": normal,
        "meanChipDifference": mean_difference,
        "pooledSeatActMeanChipDifference": pooled_seat_act_difference,
        "meanChipDifference95": {
            "low": interval["low"],
            "high": interval["high"],
        },
        "meanChipDifferenceInference": interval,
        "pairwiseCandidateBeforeNormal": {
            "candidateBefore": pairwise_before,
            "comparisons": pairwise_comparisons,
            "sampleCount": pairwise_comparisons,
            "rate": pairwise_rate,
        },
        "roleAudit": {
            "assignmentUnit": "fixed-policy-identity-within-match",
            "initialCandidateSeats": initial_seat_candidate_counts,
            "initialSeatBalance": {
                "minimum": minimum,
                "maximum": maximum,
                "spread": maximum - minimum,
                "passed": maximum - minimum <= 1,
            },
            "allActRoles": role_summary,
            "totalSeatActs": matches * acts * player_count,
        },
        "matchClusters": {
            "unit": "seed-matched-match",
            "count": matches,
            "sha256": hashlib.sha256(cluster_bytes).hexdigest(),
        },
        "statisticallyAboveNormal": float(interval["low"]) > 0.0,
        "candidateDecisionAudit": {
            "overall": _summarize_decision_counts(candidate_decision_counts),
            "byAct": [
                {
                    "act": act_index + 1,
                    **_summarize_decision_counts(counts),
                }
                for act_index, counts in enumerate(candidate_decision_counts_by_act)
            ],
        },
    }
    result["effectSizeGate"] = _evaluate_effect_gate(
        mean_difference,
        float(interval["low"]),
        pairwise_rate,
        gates,
    )
    return result


def evaluate_benchmark(
    *,
    mode: str,
    seed_schedule: EvaluationSeedSchedule,
    candidate_policy: CandidatePolicy,
    bindings: EvaluationBindings,
    adapter: EnvironmentAdapter | None = None,
    match_counts: Mapping[int, int] | None = None,
    gates: Mapping[str, float] | None = None,
    acts: int = ACTS_PER_MATCH,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    batch_size: int = 1,
    final_seed_reservation: Mapping[str, object] | None = None,
    candidate_policy_routing: CandidatePolicyRouting | None = None,
) -> dict[str, object]:
    counts = dict(match_counts or _expected_match_counts(mode))
    thresholds = dict(gates or _expected_gates(mode))
    validate_evaluation_plan(
        mode=mode,
        match_counts=counts,
        acts=acts,
        gates=thresholds,
        seed_schedule=seed_schedule,
        final_seed_reservation=final_seed_reservation,
    )
    if mode == "final":
        reservation_bindings = final_seed_reservation.get("bindings")  # type: ignore[union-attr]
        if not isinstance(reservation_bindings, Mapping):
            raise ValueError("final reservation is missing artifact bindings")
        expected_reservation_bindings = {
            "artifactSha256": bindings.artifact_sha256,
            "modelSha256": bindings.actor_sha256,
            "observationSchemaSha256": bindings.observation_contract_sha256,
            "normalBaselineSha256": bindings.normal_baseline_sha256,
            "normalBaselineSourceCommit": bindings.normal_baseline_source_commit,
        }
        for key, expected in expected_reservation_bindings.items():
            if reservation_bindings.get(key) != expected:
                raise ValueError(f"final reservation {key} does not match evaluation")
    _require_positive_int(bootstrap_resamples, "bootstrap resamples")
    _require_positive_int(batch_size, "batch size")
    routing = candidate_policy_routing or CandidatePolicyRouting()
    resolved_adapter = adapter or V4EnvAdapter()
    results = [
        evaluate_player_count(
            player_count=player_count,
            matches=counts[player_count],
            acts=acts,
            seed_schedule=seed_schedule,
            candidate_policy=candidate_policy,
            adapter=resolved_adapter,
            gates=thresholds,
            bootstrap_resamples=bootstrap_resamples,
            batch_size=batch_size,
            candidate_policy_routing=routing,
        )
        for player_count in PLAYER_COUNTS
    ]
    policy_metadata = getattr(candidate_policy, "audit_metadata", None)
    if not isinstance(policy_metadata, Mapping):
        policy_metadata = {
            "kind": "injected-public-actor-callback",
            "actorCount": 1,
            "seeds": None,
            "inferenceExecution": "deterministic-eager",
            "compileAutomaticFallback": False,
            "historyInferenceBuckets": list(HISTORY_INFERENCE_BUCKETS),
            "playerWidthBucketing": "callback-defined",
        }
    policy_metadata = dict(policy_metadata)
    policy_metadata["routing"] = routing.report_value()
    overall_decision_counts = _decision_counts()
    decision_counts_by_act = [_decision_counts() for _ in range(ACTS_PER_MATCH)]
    decision_audit_by_player_count = []
    for result in results:
        decision_audit = result["candidateDecisionAudit"]
        overall = decision_audit["overall"]  # type: ignore[index]
        _merge_decision_counts(overall_decision_counts, overall)
        decision_audit_by_player_count.append(
            {"playerCount": result["playerCount"], **dict(overall)}
        )
        for act_index, value in enumerate(decision_audit["byAct"]):  # type: ignore[index]
            _merge_decision_counts(decision_counts_by_act[act_index], value)
    report: dict[str, object] = {
        "format": "dalmuti-model-benchmark",
        "version": 2,
        "evaluationMode": mode,
        "modelSha256": bindings.actor_sha256,
        "bindings": bindings.report_value(),
        "bindingEvidence": {
            "format": "dalmuti-v4-actual-input-binding-evidence",
            "version": 1,
            "actualFilesVerified": bindings.actual_files_verified,
            "actorBundleArtifactSha256": bindings.artifact_sha256,
            "actorModelSha256": bindings.actor_sha256,
            "observationContractSha256": bindings.observation_contract_sha256,
            "normalSourceSha256": bindings.normal_baseline_sha256,
            "normalSourceCommit": bindings.normal_baseline_source_commit,
            "normalCommitBlobMatchesWorkingSource": bindings.actual_files_verified,
        },
        "seed": seed_schedule.base_seed,
        "seedFamily": {
            "id": seed_schedule.family_id,
            "mode": seed_schedule.mode,
            "ranges": seed_schedule.ranges(counts),
        },
        "playerCounts": list(PLAYER_COUNTS),
        "matchCountsByPlayerCount": counts,
        "actsPerMatch": acts,
        "candidatePolicy": policy_metadata,
        "normalPolicy": {
            "name": "exact-v4-env-normal-callback",
            "sha256": bindings.normal_baseline_sha256,
            "sourceCommit": bindings.normal_baseline_source_commit,
        },
        "evaluationDesign": {
            "seedMatchedMatchClusters": True,
            "candidateAssignment": "rotating-balanced-initial-seat",
            "normalControl": "exact-reference-environment-callback",
            "inferenceBatchSize": batch_size,
            "candidateBatchedForward": callable(
                getattr(candidate_policy, "actions", None)
            ),
            "finalMatchCountPreset": mode == "final",
            "familyId": seed_schedule.family_id,
            "developmentFamiliesRequired": 2 if mode == "development" else None,
            "finalFeedbackPolicy": (
                "sealed-holdout-not-a-training-input" if mode == "final" else None
            ),
        },
        "promotionThresholds": thresholds,
        "candidateDecisionAudit": {
            "overall": _summarize_decision_counts(overall_decision_counts),
            "byPlayerCount": decision_audit_by_player_count,
            "byAct": [
                {
                    "act": act_index + 1,
                    **_summarize_decision_counts(counts),
                }
                for act_index, counts in enumerate(decision_counts_by_act)
            ],
        },
        "promotionPassed": all(
            bool(result["effectSizeGate"]["passed"]) for result in results  # type: ignore[index]
        ),
        "results": results,
        "deploymentTriggered": False,
    }
    validate_benchmark_report(report, expected_mode=mode)
    return report


def validate_benchmark_report(
    report: Mapping[str, object], *, expected_mode: str | None = None
) -> None:
    if report.get("format") != "dalmuti-model-benchmark" or report.get("version") != 2:
        raise ValueError("unsupported benchmark report")
    mode = report.get("evaluationMode")
    if not isinstance(mode, str) or mode not in EVALUATION_MODES:
        raise ValueError("benchmark evaluation mode is invalid")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError("benchmark mode does not match expectation")
    if report.get("actsPerMatch") != ACTS_PER_MATCH:
        raise ValueError("benchmark must use five acts")
    if report.get("playerCounts") != list(PLAYER_COUNTS):
        raise ValueError("benchmark must contain p4 through p10 in order")
    counts_value = report.get("matchCountsByPlayerCount")
    if not isinstance(counts_value, Mapping):
        raise ValueError("benchmark match counts are missing")
    counts = {int(key): value for key, value in counts_value.items()}
    expected_counts = _expected_match_counts(mode)
    if counts != expected_counts:
        raise ValueError(f"benchmark does not use exact {mode} match counts")
    thresholds = report.get("promotionThresholds")
    if not isinstance(thresholds, Mapping) or dict(thresholds) != _expected_gates(mode):
        raise ValueError(f"benchmark does not use exact {mode} gates")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("benchmark bindings are missing")
    expected_binding_fields = {
        "artifactSha256",
        "modelSha256",
        "observationSchemaSha256",
        "normalBaselineSha256",
        "normalBaselineSourceCommit",
    }
    if set(bindings) != expected_binding_fields:
        raise ValueError("benchmark bindings must contain the exact V4 attempt fields")
    for field in (
        "artifactSha256",
        "modelSha256",
        "observationSchemaSha256",
        "normalBaselineSha256",
    ):
        _require_sha256(bindings.get(field), field)
    source_commit = bindings.get("normalBaselineSourceCommit")
    if not isinstance(source_commit, str) or not _GIT_COMMIT_RE.fullmatch(source_commit):
        raise ValueError("Normal baseline source commit binding is invalid")
    if report.get("modelSha256") != bindings.get("modelSha256"):
        raise ValueError("model SHA does not match actor binding")
    binding_evidence = report.get("bindingEvidence")
    expected_evidence = {
        "format": "dalmuti-v4-actual-input-binding-evidence",
        "version": 1,
        "actualFilesVerified": binding_evidence.get("actualFilesVerified")
        if isinstance(binding_evidence, Mapping)
        else None,
        "actorBundleArtifactSha256": bindings.get("artifactSha256"),
        "actorModelSha256": bindings.get("modelSha256"),
        "observationContractSha256": bindings.get("observationSchemaSha256"),
        "normalSourceSha256": bindings.get("normalBaselineSha256"),
        "normalSourceCommit": bindings.get("normalBaselineSourceCommit"),
        "normalCommitBlobMatchesWorkingSource": binding_evidence.get(
            "actualFilesVerified"
        )
        if isinstance(binding_evidence, Mapping)
        else None,
    }
    if (
        not isinstance(binding_evidence, Mapping)
        or not isinstance(binding_evidence.get("actualFilesVerified"), bool)
        or dict(binding_evidence) != expected_evidence
    ):
        raise ValueError("benchmark actual input binding evidence is invalid")
    seed_family = report.get("seedFamily")
    if not isinstance(seed_family, Mapping):
        raise ValueError("benchmark seed family is missing")
    family_id = seed_family.get("id")
    base_seed = report.get("seed")
    if not isinstance(family_id, str) or isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("benchmark seed family identity is invalid")
    schedule = EvaluationSeedSchedule(mode, family_id, base_seed)
    if (
        seed_family.get("mode") != mode
        or canonical_json_bytes(seed_family.get("ranges"))
        != canonical_json_bytes(schedule.ranges(expected_counts))
    ):
        raise ValueError("benchmark seed ranges do not match deterministic schedule")
    design = report.get("evaluationDesign")
    if (
        not isinstance(design, Mapping)
        or design.get("familyId") != family_id
        or design.get("finalMatchCountPreset") is not (mode == "final")
        or design.get("seedMatchedMatchClusters") is not True
        or design.get("normalControl") != "exact-reference-environment-callback"
        or isinstance(design.get("inferenceBatchSize"), bool)
        or not isinstance(design.get("inferenceBatchSize"), int)
        or design.get("inferenceBatchSize") < 1
        or not isinstance(design.get("candidateBatchedForward"), bool)
    ):
        raise ValueError("benchmark evaluation design is invalid")
    if report.get("normalPolicy") != {
        "name": "exact-v4-env-normal-callback",
        "sha256": bindings.get("normalBaselineSha256"),
        "sourceCommit": bindings.get("normalBaselineSourceCommit"),
    }:
        raise ValueError("Normal policy is not bound to the frozen exact callback")
    candidate_policy = report.get("candidatePolicy")
    if not isinstance(candidate_policy, Mapping) or candidate_policy.get("actorCount") not in (1, 3):
        raise ValueError("candidate policy must bind one actor or three actors")
    # Pre-bucketing screening processes may finish after this module is
    # updated.  Their immutable reports lack these additive audit fields and
    # remain valid eager evidence.  Any report that declares the new execution
    # contract must declare all of it exactly.
    execution = candidate_policy.get("inferenceExecution")
    if execution is not None and (
        execution not in ("deterministic-eager", "torch-compile-reduce-overhead")
        or candidate_policy.get("compileAutomaticFallback") is not False
        or candidate_policy.get("historyInferenceBuckets")
        != list(HISTORY_INFERENCE_BUCKETS)
    ):
        raise ValueError("candidate inference execution audit is invalid")
    if candidate_policy.get("actorCount") == 3:
        seeds = candidate_policy.get("seeds")
        if (
            not isinstance(seeds, list)
            or len(seeds) != 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(set(seeds)) != 3
            or candidate_policy.get("ensembleRule")
            != "mean-of-per-actor-logits-centered-over-legal-actions"
        ):
            raise ValueError("three-actor candidate is not a centered three-seed ensemble")
    bundle_actor_hashes = candidate_policy.get("bundleActorSha256s")
    if bundle_actor_hashes is not None:
        expected_members = int(candidate_policy["actorCount"])
        if (
            not isinstance(bundle_actor_hashes, list)
            or len(bundle_actor_hashes) not in (1, expected_members)
        ):
            raise ValueError("candidate bundle actor inventory is invalid")
        for digest in bundle_actor_hashes:
            _require_sha256(digest, "candidate bundle actor SHA-256")
        if len(bundle_actor_hashes) == 1:
            if report.get("modelSha256") != bundle_actor_hashes[0]:
                raise ValueError("actor bundle hash does not match model binding")
        elif expected_members == 3:
            composite = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "format": "dalmuti-v4-centered-logit-ensemble-binding",
                        "version": 1,
                        "actorSha256s": bundle_actor_hashes,
                        "seeds": candidate_policy["seeds"],
                        "rule": "mean-of-per-actor-logits-centered-over-legal-actions",
                    }
                )
            ).hexdigest()
            if report.get("modelSha256") != composite:
                raise ValueError("ensemble member hashes do not match model binding")
    bundle_manifest_hashes = candidate_policy.get("bundleManifestSha256s")
    bundle_artifact_sha256 = candidate_policy.get("bundleArtifactSha256")
    if binding_evidence.get("actualFilesVerified") is True:
        expected_members = int(candidate_policy["actorCount"])
        if (
            not isinstance(bundle_actor_hashes, list)
            or not isinstance(bundle_manifest_hashes, list)
            or len(bundle_actor_hashes) not in (1, expected_members)
            or len(bundle_manifest_hashes) not in (1, expected_members)
            or any(
                not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
                for digest in bundle_manifest_hashes
            )
            or bundle_artifact_sha256 != bindings.get("artifactSha256")
        ):
            raise ValueError("verified actor bundle artifact evidence is invalid")
        if len(bundle_manifest_hashes) == 1:
            expected_artifact_sha256 = bundle_manifest_hashes[0]
        else:
            expected_artifact_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "format": "dalmuti-v4-actor-bundle-set-binding",
                        "version": 1,
                        "manifestSha256s": bundle_manifest_hashes,
                        "actorSha256s": bundle_actor_hashes,
                        "seeds": candidate_policy.get("seeds"),
                    }
                )
            ).hexdigest()
        if bundle_artifact_sha256 != expected_artifact_sha256:
            raise ValueError("actor bundle artifact digest is not canonical")
    routing_value = candidate_policy.get("routing")
    routing = None if routing_value is None else _routing_from_report(routing_value)
    root_decision_audit = report.get("candidateDecisionAudit")
    if routing is None:
        if root_decision_audit is not None:
            raise ValueError("legacy candidate policy cannot declare routing counts")
    else:
        _validate_candidate_decision_audit(
            root_decision_audit, include_player_counts=True
        )
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(PLAYER_COUNTS):
        raise ValueError("benchmark results are incomplete")
    passed = True
    result_decision_counts = _decision_counts()
    result_decision_summaries: list[dict[str, object]] = []
    for player_count, result in zip(PLAYER_COUNTS, results):
        if not isinstance(result, Mapping):
            raise ValueError("benchmark result is invalid")
        if result.get("playerCount") != player_count or result.get("matches") != expected_counts[player_count]:
            raise ValueError(f"p{player_count} result count is invalid")
        if result.get("actsPerMatch") != ACTS_PER_MATCH:
            raise ValueError(f"p{player_count} act count is invalid")
        mean = _require_finite(result.get("meanChipDifference"), "mean chip difference")
        interval = result.get("meanChipDifference95")
        pairwise = result.get("pairwiseCandidateBeforeNormal")
        if not isinstance(interval, Mapping) or not isinstance(pairwise, Mapping):
            raise ValueError(f"p{player_count} inference is missing")
        low = _require_finite(interval.get("low"), "lower confidence bound")
        inference = result.get("meanChipDifferenceInference")
        if (
            not isinstance(inference, Mapping)
            or inference.get("unit") != "seed-matched-match"
            or inference.get("method") != "deterministic-percentile-bootstrap"
            or inference.get("clusters") != expected_counts[player_count]
            or _require_finite(inference.get("mean"), "bootstrap mean") != mean
            or _require_finite(inference.get("low"), "bootstrap lower bound") != low
        ):
            raise ValueError(f"p{player_count} match-clustered bootstrap is invalid")
        rate = _require_finite(pairwise.get("rate"), "pairwise rate")
        sample_count = pairwise.get("sampleCount")
        _require_positive_int(sample_count, "pairwise sample count")
        comparisons = pairwise.get("comparisons")
        candidate_before = pairwise.get("candidateBefore")
        if (
            comparisons != sample_count
            or isinstance(candidate_before, bool)
            or not isinstance(candidate_before, int)
            or candidate_before < 0
            or candidate_before > sample_count
            or rate != candidate_before / sample_count
        ):
            raise ValueError(f"p{player_count} pairwise totals disagree")
        role_audit = result.get("roleAudit")
        expected_initial_counts = [
            sum(
                seat in rotating_candidate_seats(player_count, match_index)
                for match_index in range(expected_counts[player_count])
            )
            for seat in range(player_count)
        ]
        if (
            not isinstance(role_audit, Mapping)
            or role_audit.get("initialCandidateSeats") != expected_initial_counts
            or role_audit.get("totalSeatActs")
            != expected_counts[player_count] * ACTS_PER_MATCH * player_count
            or not isinstance(role_audit.get("initialSeatBalance"), Mapping)
            or role_audit["initialSeatBalance"].get("passed") is not True  # type: ignore[index]
        ):
            raise ValueError(f"p{player_count} role/seat assignment audit is invalid")
        clusters = result.get("matchClusters")
        if (
            not isinstance(clusters, Mapping)
            or clusters.get("unit") != "seed-matched-match"
            or clusters.get("count") != expected_counts[player_count]
        ):
            raise ValueError(f"p{player_count} cluster audit is invalid")
        _require_sha256(clusters.get("sha256"), "match cluster SHA-256")
        result_decision_audit = result.get("candidateDecisionAudit")
        if routing is None:
            if result_decision_audit is not None:
                raise ValueError("legacy result cannot declare routing counts")
        else:
            per_player_counts = _validate_candidate_decision_audit(
                result_decision_audit, include_player_counts=False
            )
            _merge_decision_counts(result_decision_counts, per_player_counts)
            per_player_overall = result_decision_audit["overall"]  # type: ignore[index]
            result_decision_summaries.append(
                {"playerCount": player_count, **dict(per_player_overall)}
            )
        expected_gate = _evaluate_effect_gate(mean, low, rate, thresholds)  # type: ignore[arg-type]
        if result.get("effectSizeGate") != expected_gate:
            raise ValueError(f"p{player_count} effect gate disagrees with metrics")
        passed = passed and expected_gate["passed"]
    if routing is not None:
        root_overall = root_decision_audit["overall"]  # type: ignore[index]
        if result_decision_counts != {
            key: root_overall[key] for key in _DECISION_COUNT_KEYS  # type: ignore[index]
        }:
            raise ValueError("candidate decision player-count totals disagree")
        if root_decision_audit["byPlayerCount"] != result_decision_summaries:  # type: ignore[index]
            raise ValueError("candidate decision player-count audit disagrees")
        if routing.mode == "pure-actor" and result_decision_counts["fallbackDecisions"] != 0:
            raise ValueError("pure actor evaluation contains fallback decisions")
        if routing.mode == "exact-normal" and (
            result_decision_counts["actorDecisions"] != 0
            or result_decision_counts["fallbackDecisions"]
            != result_decision_counts["candidateDecisions"]
            or result_decision_counts["deviationsFromNormal"] != 0
        ):
            raise ValueError("exact-Normal routing audit is not behaviorally exact")
        if routing.mode != "pure-actor" and (
            result_decision_counts["forcedDecisions"]
            > result_decision_counts["fallbackDecisions"]
        ):
            raise ValueError("safe routing must send forced decisions through Normal")
    if report.get("promotionPassed") is not passed:
        raise ValueError("promotionPassed disagrees with per-player gates")
    if report.get("deploymentTriggered") is not False:
        raise ValueError("V4 evaluation must never trigger deployment")


def certify_development_families(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(reports) != 2:
        raise ValueError("development certification requires exactly two families")
    for report in reports:
        validate_benchmark_report(report, expected_mode="development")
    family_ids = [str(report["seedFamily"]["id"]) for report in reports]  # type: ignore[index]
    if len(set(family_ids)) != 2:
        raise ValueError("development family ids must be distinct")
    binding_values = [canonical_json_bytes(report["bindings"]) for report in reports]
    if binding_values[0] != binding_values[1]:
        raise ValueError("development families must evaluate identical bindings")
    occupied: list[tuple[int, int]] = []
    for report in reports:
        ranges = report["seedFamily"]["ranges"]  # type: ignore[index]
        for value in ranges:
            start, end = int(value["start"]), int(value["end"])
            if any(not (end < other_start or start > other_end) for other_start, other_end in occupied):
                raise ValueError("development seed families overlap")
            occupied.append((start, end))
    family_passes = [bool(report["promotionPassed"]) for report in reports]
    return {
        "format": "dalmuti-v4-development-certification",
        "version": 1,
        "requiredFamilies": 2,
        "familyIds": family_ids,
        "familyReportSha256": [
            hashlib.sha256(canonical_json_bytes(report)).hexdigest() for report in reports
        ],
        "bindings": dict(reports[0]["bindings"]),
        "eachFamilyPassed": family_passes,
        "promotionEligibleForFinal": all(family_passes),
        "deploymentTriggered": False,
    }


class CenteredLogitActorPolicy:
    """Greedy callback for one actor or a three-seed centered-logit ensemble."""

    def __init__(
        self,
        actors: Sequence[object],
        *,
        seeds: Sequence[int] | None = None,
        device: str = "cpu",
        compile_actor: bool = False,
    ):
        if len(actors) not in (1, 3):
            raise ValueError("actor policy requires one actor or exactly three actors")
        if seeds is not None and any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 1
            for seed in seeds
        ):
            raise ValueError("actor seeds must be positive integers")
        if len(actors) == 3:
            if seeds is None or len(seeds) != 3 or len(set(int(seed) for seed in seeds)) != 3:
                raise ValueError("three-actor ensemble requires three unique seeds")
        elif seeds is not None and len(seeds) not in (0, 1):
            raise ValueError("single actor accepts at most one seed")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for actor inference") from error
        self._torch = torch
        eager_actors = tuple(getattr(actor, "to")(device).eval() for actor in actors)
        configs = [getattr(actor, "config", None) for actor in eager_actors]
        if configs[0] is None or any(config != configs[0] for config in configs[1:]):
            raise ValueError("all candidate actors must share one explicit config")
        self.config = configs[0]
        self.compile_actor = bool(compile_actor)
        if self.compile_actor:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "--compile-actor requires torch.compile; automatic fallback is forbidden"
                )
            try:
                self.actors = tuple(
                    torch.compile(actor, mode="reduce-overhead")
                    for actor in eager_actors
                )
            except Exception as error:
                raise RuntimeError(
                    "torch.compile actor setup failed; automatic fallback is forbidden"
                ) from error
        else:
            self.actors = eager_actors
        self.device = device
        self.seeds = None if seeds is None else tuple(int(seed) for seed in seeds)
        self.last_batch_audit: dict[str, object] | None = None
        self.audit_metadata = {
            "kind": "single-actor" if len(actors) == 1 else "centered-logit-ensemble",
            "actorCount": len(actors),
            "seeds": None if self.seeds is None else list(self.seeds),
            "ensembleRule": (
                None
                if len(actors) == 1
                else "mean-of-per-actor-logits-centered-over-legal-actions"
            ),
            "selection": "deterministic-greedy-lowest-index-tie-break",
            "inferenceExecution": (
                "torch-compile-reduce-overhead" if self.compile_actor else "deterministic-eager"
            ),
            "compileAutomaticFallback": False,
            "historyInferenceBuckets": list(HISTORY_INFERENCE_BUCKETS),
            "playerWidthBucketing": "exact-valid-player-prefix",
        }

    def _valid_prefix(self, mask: object, *, label: str) -> int:
        torch = self._torch
        if getattr(mask, "ndim", None) != 1:
            raise ValueError(f"{label} mask must be one-dimensional")
        boolean = mask.to(dtype=torch.bool)
        indexes = torch.nonzero(boolean, as_tuple=False).flatten()
        if indexes.numel() == 0:
            return 0
        length = int(indexes[-1].item()) + 1
        if not bool(boolean[:length].all().item()) or bool(boolean[length:].any().item()):
            raise ValueError(f"{label} mask must be a contiguous valid prefix")
        return length

    def _pad_rows(self, tensor: object, target: int) -> object:
        if tensor.shape[0] == target:
            return tensor
        padding = tensor.new_zeros((target - tensor.shape[0], *tensor.shape[1:]))
        return self._torch.cat((tensor, padding), dim=0)

    def _pad_mask(self, tensor: object, target: int) -> object:
        if tensor.shape[0] == target:
            return tensor
        padding = tensor.new_zeros((target - tensor.shape[0],), dtype=self._torch.bool)
        return self._torch.cat((tensor.to(dtype=self._torch.bool), padding), dim=0)

    def _history_bucket(self, length: int) -> int:
        return history_inference_bucket(
            length, max_history=int(self.config.max_history)
        )

    def _prepare_rows(self, observations: Sequence[object]) -> list[dict[str, object]]:
        torch = self._torch
        rows: list[dict[str, object]] = []
        for original_index, observation in enumerate(observations):
            public = getattr(observation, "public", observation)
            player_features = getattr(public, "player_features")
            player_mask = getattr(public, "player_mask")
            history_features = getattr(public, "history_features")
            history_mask = getattr(public, "history_mask")
            player_count = self._valid_prefix(player_mask, label="player")
            if player_count < 1 or player_count > self.config.max_players:
                raise ValueError("actor player count is outside model config")
            history_count = self._valid_prefix(history_mask, label="history")
            history_start = max(0, history_count - self.config.max_history)
            rows.append(
                {
                    "originalIndex": original_index,
                    "global": getattr(public, "global_features"),
                    "rank": getattr(public, "rank_features"),
                    "player": player_features[:player_count],
                    "playerMask": player_mask[:player_count].to(dtype=torch.bool),
                    "memory": getattr(public, "memory_trace_features"),
                    "history": history_features[history_start:history_count],
                    "historyMask": history_mask[history_start:history_count].to(
                        dtype=torch.bool
                    ),
                    "legal": getattr(public, "legal_mask").to(dtype=torch.bool),
                }
            )
        return rows

    def _forward_row_diagnostics(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        player_width: int,
        history_width: int,
    ) -> list[ActorActionDiagnostics]:
        if not rows:
            return []
        torch = self._torch
        if any(row["player"].shape[0] > player_width for row in rows):
            raise ValueError("player bucket is narrower than an observation")
        if any(row["history"].shape[0] > history_width for row in rows):
            raise ValueError("history bucket is narrower than an observation")
        tensors = (
            torch.stack([row["global"] for row in rows]).to(self.device),
            torch.stack([row["rank"] for row in rows]).to(self.device),
            torch.stack(
                [self._pad_rows(row["player"], player_width) for row in rows]
            ).to(self.device),
            torch.stack(
                [self._pad_mask(row["playerMask"], player_width) for row in rows]
            ).to(self.device),
            torch.stack([row["memory"] for row in rows]).to(self.device),
            torch.stack(
                [self._pad_rows(row["history"], history_width) for row in rows]
            ).to(self.device),
            torch.stack(
                [self._pad_mask(row["historyMask"], history_width) for row in rows]
            ).to(self.device),
            torch.stack([row["legal"] for row in rows]).to(self.device),
        )
        legal = tensors[-1]
        if tuple(legal.shape) != (len(rows), 236) or not bool(legal.any(dim=-1).all().item()):
            raise ValueError("actor observation has no legal action")
        try:
            with torch.inference_mode():
                logits_by_actor = [actor(*tensors) for actor in self.actors]
                for logits in logits_by_actor:
                    if tuple(logits.shape) != (len(rows), 236) or not bool(
                        torch.isfinite(logits[legal]).all().item()
                    ):
                        raise ValueError("actor returned invalid logits")
                if len(logits_by_actor) == 1:
                    logits = logits_by_actor[0]
                else:
                    centered = []
                    legal_float = legal.to(dtype=logits_by_actor[0].dtype)
                    denominator = legal_float.sum(dim=-1, keepdim=True)
                    for member_logits in logits_by_actor:
                        mean = member_logits.masked_fill(~legal, 0.0).sum(
                            dim=-1, keepdim=True
                        ) / denominator
                        centered.append(member_logits - mean)
                    logits = torch.stack(centered, dim=0).mean(dim=0)
                masked = logits.masked_fill(~legal, float("-inf"))
                selected = torch.argmax(masked, dim=-1)
                probabilities = torch.softmax(masked, dim=-1)
                top_probabilities = probabilities.gather(
                    1, selected.unsqueeze(1)
                ).squeeze(1)
                legal_counts = legal.sum(dim=-1)
                without_selected = masked.clone()
                without_selected.scatter_(1, selected.unsqueeze(1), float("-inf"))
                second = without_selected.max(dim=-1).values
                selected_logits = masked.gather(1, selected.unsqueeze(1)).squeeze(1)
                margins = selected_logits - second
                selected_values = selected.detach().cpu().tolist()
                probability_values = top_probabilities.detach().float().cpu().tolist()
                legal_count_values = legal_counts.detach().cpu().tolist()
                margin_values = margins.detach().float().cpu().tolist()
                return [
                    ActorActionDiagnostics(
                        action=int(action),
                        legal_logit_margin=(
                            None if int(legal_count) == 1 else float(margin)
                        ),
                        top_probability=float(probability),
                        legal_action_count=int(legal_count),
                    )
                    for action, margin, probability, legal_count in zip(
                        selected_values,
                        margin_values,
                        probability_values,
                        legal_count_values,
                    )
                ]
        except Exception as error:
            if self.compile_actor:
                raise RuntimeError(
                    "compiled actor inference failed; automatic fallback is forbidden"
                ) from error
            raise

    def _forward_rows(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        player_width: int,
        history_width: int,
    ) -> list[int]:
        return [
            value.action
            for value in self._forward_row_diagnostics(
                rows,
                player_width=player_width,
                history_width=history_width,
            )
        ]

    def actions_unbucketed(self, observations: Sequence[object]) -> list[int]:
        """Reference eager shape plan used to audit bucket action parity."""

        if not observations:
            return []
        rows = self._prepare_rows(observations)
        return self._forward_rows(
            rows,
            player_width=max(row["player"].shape[0] for row in rows),
            history_width=max(row["history"].shape[0] for row in rows),
        )

    def action_diagnostics(
        self, observations: Sequence[object]
    ) -> list[ActorActionDiagnostics]:
        if not observations:
            self.last_batch_audit = {
                "observations": 0,
                "groups": [],
                "historyTokenRatioVsUnbucketed": 0.0,
            }
            return []
        rows = self._prepare_rows(observations)
        groups: dict[tuple[int, int], list[dict[str, object]]] = {}
        for row in rows:
            player_width = int(row["player"].shape[0])
            history_width = self._history_bucket(int(row["history"].shape[0]))
            groups.setdefault((player_width, history_width), []).append(row)
        results: list[ActorActionDiagnostics | None] = [None] * len(rows)
        audit_groups = []
        bucketed_history_tokens = 0
        bucketed_player_tokens = 0
        for (player_width, history_width), group_rows in sorted(groups.items()):
            group_diagnostics = self._forward_row_diagnostics(
                group_rows,
                player_width=player_width,
                history_width=history_width,
            )
            for row, diagnostic in zip(group_rows, group_diagnostics):
                results[int(row["originalIndex"])] = diagnostic
            count = len(group_rows)
            bucketed_history_tokens += count * history_width
            bucketed_player_tokens += count * player_width
            audit_groups.append(
                {
                    "playerWidth": player_width,
                    "historyBucket": history_width,
                    "observations": count,
                }
            )
        if any(value is None for value in results):
            raise RuntimeError("bucketed actor inference lost an observation")
        unbucketed_history = len(rows) * max(
            int(row["history"].shape[0]) for row in rows
        )
        unbucketed_players = len(rows) * max(
            int(row["player"].shape[0]) for row in rows
        )
        self.last_batch_audit = {
            "observations": len(rows),
            "groups": audit_groups,
            "historyTokens": bucketed_history_tokens,
            "unbucketedHistoryTokens": unbucketed_history,
            "historyTokenRatioVsUnbucketed": (
                0.0 if unbucketed_history == 0 else bucketed_history_tokens / unbucketed_history
            ),
            "playerTokens": bucketed_player_tokens,
            "unbucketedPlayerTokens": unbucketed_players,
            "playerTokenRatioVsUnbucketed": bucketed_player_tokens / unbucketed_players,
        }
        return [value for value in results if value is not None]

    def actions(self, observations: Sequence[object]) -> list[int]:
        return [value.action for value in self.action_diagnostics(observations)]

    def __call__(self, observation: object) -> int:
        return self.actions((observation,))[0]


def make_centered_logit_actor_policy(
    actor_or_actors: object | Sequence[object],
    *,
    seeds: Sequence[int] | None = None,
    device: str = "cpu",
    compile_actor: bool = False,
) -> CenteredLogitActorPolicy:
    if hasattr(actor_or_actors, "actors") and hasattr(actor_or_actors, "seeds"):
        actors = tuple(getattr(actor_or_actors, "actors"))
        resolved_seeds = tuple(getattr(actor_or_actors, "seeds"))
    elif isinstance(actor_or_actors, Sequence) and not isinstance(actor_or_actors, (str, bytes)):
        actors = tuple(actor_or_actors)
        resolved_seeds = seeds
    else:
        actors = (actor_or_actors,)
        resolved_seeds = seeds
    return CenteredLogitActorPolicy(
        actors,
        seeds=resolved_seeds,
        device=device,
        compile_actor=compile_actor,
    )


def benchmark_actor_policy_batching(
    policy: CenteredLogitActorPolicy,
    observations: Sequence[object],
    *,
    warmup_iterations: int = 2,
    measured_iterations: int = 10,
) -> dict[str, object]:
    """Benchmark bucketed and reference shape plans on the policy's device."""

    if not observations:
        raise ValueError("actor benchmark requires observations")
    if (
        isinstance(warmup_iterations, bool)
        or not isinstance(warmup_iterations, int)
        or warmup_iterations < 0
    ):
        raise ValueError("warmup_iterations must be a non-negative integer")
    _require_positive_int(measured_iterations, "measured iterations")
    torch = policy._torch

    def synchronize() -> None:
        if str(policy.device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
            torch.cuda.synchronize()

    for _ in range(warmup_iterations):
        policy.actions(observations)
        policy.actions_unbucketed(observations)
    synchronize()

    started = perf_counter()
    bucketed_actions = []
    for _ in range(measured_iterations):
        bucketed_actions = policy.actions(observations)
    synchronize()
    bucketed_seconds = perf_counter() - started
    bucket_audit = dict(policy.last_batch_audit or {})

    started = perf_counter()
    unbucketed_actions = []
    for _ in range(measured_iterations):
        unbucketed_actions = policy.actions_unbucketed(observations)
    synchronize()
    unbucketed_seconds = perf_counter() - started
    if bucketed_actions != unbucketed_actions:
        raise RuntimeError("bucketed actor benchmark changed selected actions")
    decisions = len(observations) * measured_iterations
    return {
        "format": "dalmuti-v4-actor-batching-benchmark",
        "version": 1,
        "device": str(policy.device),
        "compileActor": policy.compile_actor,
        "observationsPerIteration": len(observations),
        "warmupIterations": warmup_iterations,
        "measuredIterations": measured_iterations,
        "actionsIdentical": True,
        "bucketedSeconds": bucketed_seconds,
        "unbucketedSeconds": unbucketed_seconds,
        "bucketedDecisionsPerSecond": decisions / bucketed_seconds,
        "unbucketedDecisionsPerSecond": decisions / unbucketed_seconds,
        "speedup": unbucketed_seconds / bucketed_seconds,
        "bucketAudit": bucket_audit,
    }


def _publish_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_report_exclusive(
    output_path: str | Path, report: Mapping[str, object]
) -> dict[str, object]:
    validate_benchmark_report(report)
    output = Path(output_path)
    checksum_path = output.with_name(output.name + ".sha256")
    if output.exists() or checksum_path.exists():
        raise FileExistsError("evaluation report and checksum are immutable")
    payload = canonical_json_bytes(report)
    digest = hashlib.sha256(payload).hexdigest()
    checksum_payload = f"{digest}  {output.name}\n".encode("ascii")
    _publish_exclusive(output, payload)
    try:
        _publish_exclusive(checksum_path, checksum_payload)
    except Exception:
        # Remove only the inode published by this call; callers can safely retry
        # in a fresh immutable run directory.
        if output.exists():
            output.unlink()
        raise
    return {
        "path": str(output),
        "sha256Path": str(checksum_path),
        "sha256": digest,
        "bytes": len(payload),
    }


def _load_cli_actor_policy(
    bundle_paths: Sequence[str],
    *,
    actor_seeds: Sequence[int],
    device: str,
    compile_actor: bool,
) -> tuple[CenteredLogitActorPolicy, str, str]:
    if len(bundle_paths) not in (1, 3):
        raise ValueError("--actor-bundle must be supplied once or exactly three times")
    if len(bundle_paths) == 3 and len(actor_seeds) != 3:
        raise ValueError("three actor bundles require exactly three --actor-seed values")
    if len(bundle_paths) == 1 and len(actor_seeds) > 1:
        raise ValueError("one actor bundle accepts at most one --actor-seed")
    if len(actor_seeds) != len(set(actor_seeds)):
        raise ValueError("actor seeds must be unique")

    from v4_export import (
        load_v4_actor_checkpoint,
        sha256_file,
        verify_v4_actor_bundle,
    )

    models: list[object] = []
    payloads: list[Mapping[str, object]] = []
    actor_hashes: list[str] = []
    manifest_hashes: list[str] = []
    for value in bundle_paths:
        bundle = Path(value).resolve()
        manifest = verify_v4_actor_bundle(bundle)
        model, payload = load_v4_actor_checkpoint(bundle / "actor.pt")
        models.append(model)
        payloads.append(payload)
        actor_hashes.append(str(manifest["files"]["actor.pt"]["sha256"]))  # type: ignore[index]
        manifest_hashes.append(sha256_file(bundle / "manifest.json"))

    if len(models) == 3:
        if any(payload.get("kind") != "actor" for payload in payloads):
            raise ValueError("three-bundle evaluation requires three single-actor bundles")
        if len(set(actor_hashes)) != 3:
            raise ValueError("three-bundle ensemble requires three distinct actors")
        policy = CenteredLogitActorPolicy(
            models,
            seeds=actor_seeds,
            device=device,
            compile_actor=compile_actor,
        )
        actor_binding = hashlib.sha256(
            canonical_json_bytes(
                {
                    "format": "dalmuti-v4-centered-logit-ensemble-binding",
                    "version": 1,
                    "actorSha256s": actor_hashes,
                    "seeds": list(actor_seeds),
                    "rule": "mean-of-per-actor-logits-centered-over-legal-actions",
                }
            )
        ).hexdigest()
        artifact_binding = hashlib.sha256(
            canonical_json_bytes(
                {
                    "format": "dalmuti-v4-actor-bundle-set-binding",
                    "version": 1,
                    "manifestSha256s": manifest_hashes,
                    "actorSha256s": actor_hashes,
                    "seeds": list(actor_seeds),
                }
            )
        ).hexdigest()
    else:
        payload = payloads[0]
        if payload.get("kind") == "centered-logit-ensemble":
            if actor_seeds:
                raise ValueError(
                    "an exported ensemble already binds its seeds; omit --actor-seed"
                )
            policy = make_centered_logit_actor_policy(
                models[0], device=device, compile_actor=compile_actor
            )
        else:
            policy = make_centered_logit_actor_policy(
                models[0],
                seeds=actor_seeds or None,
                device=device,
                compile_actor=compile_actor,
            )
        actor_binding = actor_hashes[0]
        artifact_binding = manifest_hashes[0]
    policy.audit_metadata.update(
        {
            "bundleActorSha256s": actor_hashes,
            "bundleManifestSha256s": manifest_hashes,
            "bundleArtifactSha256": artifact_binding,
        }
    )
    return policy, actor_binding, artifact_binding


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a verified V4 public actor against exact frozen Normal. "
            "Match-count overrides are intentionally unsupported."
        )
    )
    parser.add_argument(
        "--actor-bundle",
        action="append",
        required=True,
        help="verified V4 actor bundle directory; supply once or three times",
    )
    parser.add_argument(
        "--actor-seed",
        action="append",
        type=int,
        default=[],
        help="actor training seed; required three times for three bundles",
    )
    parser.add_argument("--mode", required=True, choices=EVALUATION_MODES)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--candidate-policy-mode",
        choices=CANDIDATE_POLICY_MODES,
        default="pure-actor",
        help=(
            "pure actor (default), confidence-gated exact-Normal fallback, "
            "or explicit all-Normal routing"
        ),
    )
    parser.add_argument(
        "--minimum-legal-logit-margin",
        type=float,
        help="inclusive legal top-1 minus top-2 logit threshold",
    )
    parser.add_argument(
        "--minimum-top-probability",
        type=float,
        help="inclusive softmax probability threshold over legal actions",
    )
    parser.add_argument(
        "--compile-actor",
        action="store_true",
        help=(
            "use torch.compile(mode=reduce-overhead); any setup/runtime failure "
            "is fatal and never falls back to eager"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--frozen-baseline-commit", required=True)
    parser.add_argument(
        "--frozen-normal-source",
        required=True,
        help="Normal source file whose bytes must equal the bound Git commit blob",
    )
    parser.add_argument(
        "--repository-root",
        required=True,
        help="Git worktree used to resolve and verify the frozen Normal commit",
    )
    parser.add_argument(
        "--observation-contract",
        required=True,
        help="observation contract file hashed directly by the evaluator",
    )
    parser.add_argument(
        "--frozen-baseline-sha256",
        help="optional expected digest; actual Normal source bytes remain authoritative",
    )
    parser.add_argument(
        "--observation-sha256",
        help="optional expected digest; actual observation contract bytes remain authoritative",
    )
    parser.add_argument(
        "--final-reservation",
        help="atomic final seed reservation JSON; mandatory in final mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    routing = CandidatePolicyRouting(
        mode=arguments.candidate_policy_mode,
        minimum_legal_logit_margin=arguments.minimum_legal_logit_margin,
        minimum_top_probability=arguments.minimum_top_probability,
    )
    output = Path(arguments.output).resolve()
    checksum = output.with_name(output.name + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("evaluation output path is immutable and already exists")
    if arguments.mode == "final" and not arguments.final_reservation:
        raise ValueError("--final-reservation is mandatory in final mode")
    if arguments.mode != "final" and arguments.final_reservation:
        raise ValueError("--final-reservation is valid only in final mode")

    policy, actor_sha256, artifact_sha256 = _load_cli_actor_policy(
        arguments.actor_bundle,
        actor_seeds=arguments.actor_seed,
        device=arguments.device,
        compile_actor=arguments.compile_actor,
    )
    bindings = resolve_cli_evaluation_bindings(
        artifact_sha256=artifact_sha256,
        actor_sha256=actor_sha256,
        observation_contract_path=arguments.observation_contract,
        frozen_normal_source_path=arguments.frozen_normal_source,
        repository_root=arguments.repository_root,
        frozen_normal_source_commit=arguments.frozen_baseline_commit,
        expected_observation_sha256=arguments.observation_sha256,
        expected_normal_sha256=arguments.frozen_baseline_sha256,
    )
    reservation = None
    if arguments.final_reservation:
        reservation_path = Path(arguments.final_reservation).resolve()
        reservation_value = json.loads(reservation_path.read_text(encoding="utf-8"))
        if not isinstance(reservation_value, Mapping):
            raise ValueError("final reservation JSON must contain an object")
        reservation = reservation_value
    schedule = EvaluationSeedSchedule(
        arguments.mode, arguments.family_id, arguments.base_seed
    )
    report = evaluate_benchmark(
        mode=arguments.mode,
        seed_schedule=schedule,
        candidate_policy=policy,
        bindings=bindings,
        batch_size=arguments.batch_size,
        bootstrap_resamples=arguments.bootstrap_resamples,
        final_seed_reservation=reservation,
        candidate_policy_routing=routing,
    )
    published = write_report_exclusive(output, report)
    print(
        json.dumps(
            {
                "promotionPassed": report["promotionPassed"],
                "evaluationMode": report["evaluationMode"],
                "familyId": arguments.family_id,
                **published,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ACTS_PER_MATCH",
    "ActorActionDiagnostics",
    "CANDIDATE_POLICY_MODES",
    "CandidatePolicyRouting",
    "CenteredLogitActorPolicy",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEVELOPMENT_GATES",
    "DEVELOPMENT_MATCH_COUNTS",
    "EnvironmentAdapter",
    "EvaluationBindings",
    "EvaluationSeedSchedule",
    "FINAL_GATES",
    "FINAL_MATCH_COUNTS",
    "HISTORY_INFERENCE_BUCKETS",
    "PLAYER_COUNTS",
    "SCREENING_MATCH_COUNTS",
    "V4EnvAdapter",
    "benchmark_actor_policy_batching",
    "canonical_json_bytes",
    "certify_development_families",
    "deterministic_cluster_bootstrap95",
    "evaluate_benchmark",
    "evaluate_player_count",
    "history_inference_bucket",
    "make_centered_logit_actor_policy",
    "main",
    "resolve_cli_evaluation_bindings",
    "role_for_seat",
    "rotating_candidate_seats",
    "validate_benchmark_report",
    "validate_evaluation_plan",
    "verify_frozen_normal_source",
    "write_report_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
