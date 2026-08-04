from __future__ import annotations

"""Paired diagnostic sweep over the V5 Normal-residual policy strength.

This is deliberately outside the promotion registry.  It reloads the verified
Actor bundle for every scale, multiplies only the final residual projection,
and evaluates every variant on identical match seeds.  Scale zero is therefore
an exact greedy-Normal control while scale one is the unchanged trained Actor.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch


SOURCE_ROOT = Path(
    os.environ.get("DALMUTI_V5_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
GPU_TRAINING = SOURCE_ROOT / "gpu-training"
if not (GPU_TRAINING / "v5_evaluate.py").is_file():
    raise RuntimeError(
        "V5 source root is invalid; set DALMUTI_V5_SOURCE_ROOT to the repository checkout"
    )
if str(GPU_TRAINING) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINING))

from v5_evaluate import (  # noqa: E402
    V5EvaluationConfig,
    collect_v5_evaluation_clusters,
    summarize_v5_evaluation_clusters,
)
from v5_export import (  # noqa: E402
    canonical_json_bytes,
    load_v5_actor_bundle,
    sha256_file,
    tensor_state_sha256,
    v5_actor_bundle_digests,
)


FORMAT = "dalmuti-v5-residual-scale-ablation"
VERSION = 1
PLAYER_COUNTS = tuple(range(4, 11))


def _finite_scale(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 8.0:
        raise argparse.ArgumentTypeError("residual scales must be finite in [0, 8]")
    return parsed


def _scale_label(scale: float) -> str:
    rendered = format(scale, ".8g").replace("-", "m").replace(".", "p")
    return f"scale-{rendered}"


def scale_v5_actor_residual(actor: torch.nn.Module, scale: float) -> None:
    """Scale exactly the trained residual logits, leaving the Normal prior fixed."""

    if not math.isfinite(scale) or not 0.0 <= scale <= 8.0:
        raise ValueError("residual scale must be finite in [0, 8]")
    output = getattr(actor, "residual_output", None)
    if not isinstance(output, torch.nn.Linear) or output.out_features != 1:
        raise TypeError("V5 Actor residual_output contract changed")
    with torch.no_grad():
        output.weight.mul_(scale)
        if output.bias is not None:
            output.bias.mul_(scale)


def _report_metrics(report: Mapping[str, object]) -> dict[str, object]:
    raw_results = report.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(PLAYER_COUNTS):
        raise ValueError("ablation report omitted a player-count stratum")
    results = [dict(item) for item in raw_results if isinstance(item, Mapping)]
    if len(results) != len(raw_results):
        raise ValueError("ablation report result is malformed")
    means = [float(item["meanCandidateMinusNormalChipPerAct"]) for item in results]
    before = sum(
        int(dict(item["candidateBeforeNormalPairwise"])["candidateBefore"])
        for item in results
    )
    comparisons = sum(
        int(dict(item["candidateBeforeNormalPairwise"])["comparisons"])
        for item in results
    )
    return {
        "equalStrataMeanChipDifferencePerAct": sum(means) / len(means),
        "worstPlayerCountMeanChipDifferencePerAct": min(means),
        "aggregatePairwiseEarlierFinishRate": before / comparisons,
        "allPlayerCountsPassed": bool(report.get("allPlayerCountsPassed")),
        "perPlayerCount": {
            str(item["playerCount"]): {
                "meanChipDifferencePerAct": item[
                    "meanCandidateMinusNormalChipPerAct"
                ],
                "cluster95Low": dict(item["matchClustered95"])["low"],
                "pairwiseEarlierFinishRate": dict(
                    item["candidateBeforeNormalPairwise"]
                )["rate"],
            }
            for item in results
        },
    }


def _cluster_values(report: Mapping[str, object]) -> dict[tuple[int, int], float]:
    clusters = report.get("matchClusters")
    if not isinstance(clusters, list):
        raise ValueError("ablation report omitted match clusters")
    values: dict[tuple[int, int], float] = {}
    for raw in clusters:
        if not isinstance(raw, Mapping):
            raise ValueError("ablation match cluster is malformed")
        key = (int(raw["playerCount"]), int(raw["matchIndex"]))
        if key in values:
            raise ValueError("ablation match clusters contain a duplicate")
        values[key] = float(raw["meanChipDifference"])
    return values


def _paired_delta(
    control: Mapping[tuple[int, int], float],
    candidate: Mapping[tuple[int, int], float],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    if set(control) != set(candidate):
        raise ValueError("paired ablation variants do not share exact match coordinates")
    deltas = {key: candidate[key] - control[key] for key in sorted(control)}
    per_player: dict[str, float] = {}
    for player_count in PLAYER_COUNTS:
        selected = [value for (players, _), value in deltas.items() if players == player_count]
        if not selected:
            raise ValueError("paired ablation omitted a player-count stratum")
        per_player[str(player_count)] = sum(selected) / len(selected)
    rng = np.random.default_rng(bootstrap_seed)
    strata = {
        player_count: np.asarray(
            [value for (players, _), value in deltas.items() if players == player_count],
            dtype=np.float64,
        )
        for player_count in PLAYER_COUNTS
    }
    bootstrap = np.empty(bootstrap_resamples, dtype=np.float64)
    for index in range(bootstrap_resamples):
        bootstrap[index] = float(np.mean([
            values[rng.integers(0, len(values), size=len(values))].mean()
            for values in strata.values()
        ]))
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    mean_delta = sum(deltas.values()) / len(deltas)
    return {
        "meanChipDifferencePerActVersusScaleZero": mean_delta,
        "stratifiedMatchClusterBootstrap95": {
            "unit": "complete-five-act-match",
            "strata": "player-count",
            "resamples": bootstrap_resamples,
            "low": float(low),
            "high": float(high),
        },
        "positiveMatches": sum(value > 0.0 for value in deltas.values()),
        "tiedMatches": sum(value == 0.0 for value in deltas.values()),
        "negativeMatches": sum(value < 0.0 for value in deltas.values()),
        "perPlayerCountMeanDelta": per_player,
    }


def _write_json(path: Path, value: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def run_ablation(arguments: argparse.Namespace) -> dict[str, object]:
    output = Path(arguments.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("ablation output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    bundle = Path(arguments.actor_bundle).resolve(strict=True)
    base_actor, _ = load_v5_actor_bundle(bundle)
    base_identity = v5_actor_bundle_digests(bundle)
    base_tensor_state = tensor_state_sha256(base_actor.state_dict())
    del base_actor

    config = V5EvaluationConfig(
        mode="screening",
        family_id=arguments.family_id,
        seed_base=arguments.seed_base,
        match_counts=tuple((value, arguments.matches_per_player) for value in PLAYER_COUNTS),
        lane_count=arguments.lane_count,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    started = time.time()
    entries: list[dict[str, object]] = []
    cluster_sets: dict[float, dict[tuple[int, int], float]] = {}
    for scale in arguments.scales:
        actor, _ = load_v5_actor_bundle(bundle)
        scale_v5_actor_residual(actor, scale)
        scaled_state = tensor_state_sha256(actor.state_dict())
        variant_started = time.time()
        records = collect_v5_evaluation_clusters(actor, config, device=arguments.device)
        report = summarize_v5_evaluation_clusters(
            records,
            config,
            model_identity={
                "diagnosticKind": FORMAT,
                "baseActorIdentity": copy.deepcopy(base_identity),
                "baseTensorStateSha256": base_tensor_state,
                "residualScale": scale,
                "scaledTensorStateSha256": scaled_state,
            },
        )
        report["diagnosticOnly"] = True
        report["diagnosticPurpose"] = (
            "residual-scale ablation; not valid for promotion/certification/final"
        )
        report_path = output / _scale_label(scale) / "report.json"
        report_sha = _write_json(report_path, report)
        cluster_sets[scale] = _cluster_values(report)
        entries.append({
            "residualScale": scale,
            "scaledTensorStateSha256": scaled_state,
            "reportPath": report_path.relative_to(output).as_posix(),
            "reportSha256": report_sha,
            "elapsedSeconds": time.time() - variant_started,
            "metrics": _report_metrics(report),
        })
        print(json.dumps({
            "event": "variant-complete",
            "residualScale": scale,
            "elapsedSeconds": entries[-1]["elapsedSeconds"],
            "reportSha256": report_sha,
            "metrics": entries[-1]["metrics"],
        }, sort_keys=True), flush=True)
        del actor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if 0.0 not in cluster_sets:
        raise ValueError("ablation scales must include zero as the paired Normal control")
    control = cluster_sets[0.0]
    for entry in entries:
        scale = float(entry["residualScale"])
        seed_material = (
            f"{config.family_id}|{config.seed_base}|{scale:.17g}|paired-bootstrap"
        ).encode("ascii")
        paired_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        entry["pairedVersusScaleZero"] = _paired_delta(
            control,
            cluster_sets[scale],
            bootstrap_seed=paired_seed,
            bootstrap_resamples=arguments.bootstrap_resamples,
        )
    best = max(
        entries,
        key=lambda item: (
            float(dict(dict(item["pairedVersusScaleZero"])[
                "stratifiedMatchClusterBootstrap95"
            ])["low"]),
            float(dict(item["pairedVersusScaleZero"])[
                "meanChipDifferencePerActVersusScaleZero"
            ]),
            float(dict(item["metrics"])["worstPlayerCountMeanChipDifferencePerAct"]),
        ),
    )
    result = {
        "format": FORMAT,
        "version": VERSION,
        "createdUnixSeconds": int(time.time()),
        "elapsedSeconds": time.time() - started,
        "purpose": "diagnostic-only; not a promotion, certification, or final result",
        "baseActorBundle": str(bundle),
        "baseActorIdentity": base_identity,
        "baseTensorStateSha256": base_tensor_state,
        "pairedEvaluation": {
            "familyId": config.family_id,
            "seedBase": config.seed_base,
            "matchesPerPlayerCount": arguments.matches_per_player,
            "totalMatchesPerScale": arguments.matches_per_player * len(PLAYER_COUNTS),
            "bootstrapResamples": arguments.bootstrap_resamples,
            "laneCount": arguments.lane_count,
            "device": arguments.device,
        },
        "variants": entries,
        "runtime": {
            "sourceCommit": arguments.source_commit,
            "scriptSha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
            "torch": torch.__version__,
            "cudaRuntime": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "diagnosticBestEvidenceOnlyNotPromotion": {
            "residualScale": best["residualScale"],
            "metrics": best["metrics"],
            "pairedVersusScaleZero": best["pairedVersusScaleZero"],
        },
    }
    _write_json(output / "ablation.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--family-id", default="v5-run007-residual-scale-ablation-001")
    parser.add_argument("--seed-base", type=int, default=860_070_001)
    parser.add_argument("--matches-per-player", type=int, default=10)
    parser.add_argument("--lane-count", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scales",
        nargs="+",
        type=_finite_scale,
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.matches_per_player < 1:
        raise ValueError("matches-per-player must be positive")
    if len(arguments.scales) != len(set(arguments.scales)):
        raise ValueError("residual scales must be unique")
    labels = [_scale_label(value) for value in arguments.scales]
    if len(labels) != len(set(labels)):
        raise ValueError("residual scales collide after output label normalization")
    if 0.0 not in arguments.scales:
        raise ValueError("residual scales must include zero")
    result = run_ablation(arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
