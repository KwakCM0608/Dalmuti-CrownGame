from __future__ import annotations

"""Attach privacy-safe V4 search-teacher labels to complete trajectories."""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from v4_prepare_dataset import (
    BASE_SAMPLE_KEYS,
    _json_load,
    _read_and_validate,
    _sha256_file,
    _verify_sidecar,
)
from v4_search import (
    V4SearchConfig,
    determinize_v4_unseen_hands,
    run_v4_search_teacher,
)
from v4_search_env_adapter import DalmutiV4SearchEnvAdapter


TEACHER_METADATA_FORMAT = "dalmuti-v4-search-teacher-metadata"
TEACHER_METADATA_VERSION = 1
EPISODE_SHARD_ASSIGNMENT_DOMAIN = (
    "dalmuti-v4-search-teacher-episode-shard-v1"
)
ACTION_COUNT = 236
ROLE_IDS = {
    "great-dalmuti": 0,
    "lesser-dalmuti": 1,
    "merchant": 2,
    "lesser-peon": 3,
    "great-peon": 4,
}

# The name used by the execution plan.  The implementation itself has the
# more explicit class name in v4_search_env_adapter.py.
V4ScalarSearchAdapter = DalmutiV4SearchEnvAdapter


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _canonical_json_line(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_hash(repository_root: Path, relative: str) -> str:
    path = repository_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required V4 source is missing: {path}")
    return _sha256_file(path)


def _output_paths(output: Path) -> tuple[Path, Path, Path, Path]:
    metadata = output.with_suffix(".teacher-metadata.json")
    return (
        output,
        Path(f"{output}.sha256"),
        metadata,
        Path(f"{metadata}.sha256"),
    )


def _assert_outputs_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"output already exists: {existing[0]}")


def _stage_bytes(path: Path, payload: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _promote_exclusive(partial: Path, final: Path) -> None:
    # Hard-link creation is atomic and refuses to replace an existing name.
    os.link(partial, final)
    partial.unlink()


def _publish_atomically(payloads: Sequence[tuple[Path, bytes]]) -> None:
    finals = [path for path, _ in payloads]
    _assert_outputs_absent(finals)
    partials: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for final, payload in payloads:
            partials.append((_stage_bytes(final, payload), final))
        # Re-check immediately before promotion so a concurrent writer cannot
        # be replaced; os.link remains the final exclusive authority.
        _assert_outputs_absent(finals)
        for partial, final in partials:
            _promote_exclusive(partial, final)
            promoted.append(final)
    except Exception:
        for partial, _ in partials:
            partial.unlink(missing_ok=True)
        for final in promoted:
            final.unlink(missing_ok=True)
        raise


def _trajectory_groups(
    records: Sequence[Mapping[str, object]], player_count: int
) -> dict[str, list[tuple[int, Mapping[str, object]]]]:
    groups: dict[str, list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for record_index, record in enumerate(records):
        trajectory_id = str(record["trajectoryId"])
        groups[trajectory_id].append((record_index, record))
    if not groups:
        raise ValueError("V4 teacher input contains no trajectories")
    for trajectory_id, entries in groups.items():
        entries.sort(key=lambda item: int(item[1]["step"]))
        first = entries[0][1]
        previous_step = -1
        terminal_count = 0
        for position, (_, record) in enumerate(entries):
            step = int(record["step"])
            if step <= previous_step:
                raise ValueError(f"trajectory {trajectory_id} steps are not strictly increasing")
            previous_step = step
            for field in (
                "trajectoryId",
                "episodeId",
                "act",
                "actorId",
                "actorSeat",
                "actorRole",
                "finishPlace",
            ):
                if record[field] != first[field]:
                    raise ValueError(
                        f"trajectory {trajectory_id} changes its bound field {field}"
                    )
            terminal = record["actorTerminal"]
            if not isinstance(terminal, bool):
                raise ValueError(f"trajectory {trajectory_id} terminal flag is not boolean")
            terminal_count += int(terminal)
            if terminal and position != len(entries) - 1:
                raise ValueError(f"trajectory {trajectory_id} has data after actor terminal")
        if terminal_count != 1 or entries[-1][1]["actorTerminal"] is not True:
            raise ValueError(
                f"trajectory {trajectory_id} must contain one final actor terminal"
            )
        seat = int(first["actorSeat"])
        if not 0 <= seat < player_count:
            raise ValueError(f"trajectory {trajectory_id} actor seat is out of range")
    return dict(groups)


def _selection_digest(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _episode_shard_index(
    episode_id: str, *, seed: int, shard_count: int
) -> int:
    """Assign one complete episode using a stable, cross-process hash rule.

    The hashed byte payload is UTF-8 ``domain + NUL + base-10 seed + NUL +
    episodeId``.  Its SHA-256 digest is interpreted as one unsigned big-endian
    integer and reduced modulo ``shard_count``.  Python's randomized ``hash``
    is deliberately never involved.
    """

    shard_count = _positive_integer(shard_count, "shard-count")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode ID must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    payload = b"\0".join((
        EPISODE_SHARD_ASSIGNMENT_DOMAIN.encode("ascii"),
        str(seed).encode("ascii"),
        episode_id.encode("utf-8"),
    ))
    digest_integer = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return digest_integer % shard_count


def _partition_episode_trajectories(
    groups: Mapping[str, Sequence[tuple[int, Mapping[str, object]]]],
    *,
    seed: int,
    shard_count: int,
    shard_index: int,
) -> tuple[
    dict[str, Sequence[tuple[int, Mapping[str, object]]]],
    set[str],
    dict[str, int],
]:
    """Return only whole-episode trajectory groups assigned to one shard."""

    shard_count = _positive_integer(shard_count, "shard-count")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("shard-index must be an integer from 0 to shard-count - 1")
    episode_to_trajectories: dict[str, list[str]] = defaultdict(list)
    for trajectory_id, entries in groups.items():
        if not entries:
            raise ValueError(f"trajectory {trajectory_id} is empty")
        episode_id = str(entries[0][1]["episodeId"])
        episode_to_trajectories[episode_id].append(trajectory_id)
    assignments = {
        episode_id: _episode_shard_index(
            episode_id, seed=seed, shard_count=shard_count
        )
        for episode_id in episode_to_trajectories
    }
    eligible_episodes = {
        episode_id
        for episode_id, assigned_index in assignments.items()
        if assigned_index == shard_index
    }
    eligible_groups = {
        trajectory_id: entries
        for trajectory_id, entries in groups.items()
        if str(entries[0][1]["episodeId"]) in eligible_episodes
    }
    if not eligible_episodes or not eligible_groups:
        raise ValueError(
            f"episode shard {shard_index} of {shard_count} is empty"
        )
    return eligible_groups, eligible_episodes, assignments


def _select_balanced_trajectories(
    groups: Mapping[str, Sequence[tuple[int, Mapping[str, object]]]],
    *,
    player_count: int,
    seed: int,
    target: int,
) -> tuple[set[str], dict[str, int]]:
    if target > len(groups):
        raise ValueError(
            f"target-trajectories {target} exceeds available trajectories {len(groups)}"
        )
    buckets: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for trajectory_id, entries in groups.items():
        first = entries[0][1]
        role = str(first["actorRole"])
        if role not in ROLE_IDS:
            raise ValueError(f"trajectory {trajectory_id} has an unsupported role")
        key = (player_count, ROLE_IDS[role], int(first["act"]))
        buckets[key].append(trajectory_id)
    for key, values in buckets.items():
        values.sort(
            key=lambda trajectory_id: (
                _selection_digest(seed, "trajectory", *key, trajectory_id),
                trajectory_id,
            )
        )
    stratum_order = sorted(
        buckets,
        key=lambda key: (_selection_digest(seed, "stratum", *key), key),
    )
    selected: set[str] = set()
    cursors = {key: 0 for key in stratum_order}
    while len(selected) < target:
        progressed = False
        for key in stratum_order:
            cursor = cursors[key]
            if cursor >= len(buckets[key]):
                continue
            selected.add(buckets[key][cursor])
            cursors[key] = cursor + 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            raise AssertionError("balanced trajectory reservoir exhausted early")
    counts = {
        f"p{key[0]}:role-{key[1]}:act-{key[2]}": sum(
            trajectory_id in selected for trajectory_id in buckets[key]
        )
        for key in sorted(buckets)
    }
    return selected, counts


def _selected_ids_hash(selected: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(selected), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return _digest_bytes(payload)


def _sample_seed(seed: int, trajectory_id: str, step: int) -> int:
    payload = f"{seed}\0teacher\0{trajectory_id}\0{step}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _legal_mask(indices: object) -> tuple[bool, ...]:
    if not isinstance(indices, list) or not indices:
        raise ValueError("teacher sample legalActionIndices must be a non-empty list")
    mask = [False] * ACTION_COUNT
    previous = -1
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < ACTION_COUNT:
            raise ValueError("teacher sample legal action is out of range")
        if value <= previous:
            raise ValueError("teacher sample legal actions must be unique and sorted")
        mask[value] = True
        previous = value
    return tuple(mask)


def _inject_expert(raw_line: bytes, expert_action: int) -> bytes:
    if not raw_line.endswith(b"}\n"):
        raise ValueError("strict V4 sample lines must end with a compact JSON object")
    # Preserve every original sample byte, including action/reward/critic
    # values, and append only the new expert field.
    return (
        raw_line[:-2]
        + f',"expertActionIndex":{expert_action}}}\n'.encode("ascii")
    )


def label_v4_search_teacher(
    input_path: str | Path,
    output_path: str | Path,
    *,
    checksum_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    seed: int = 20260801,
    target_trajectories: int = 1,
    hypotheses: int = 8,
    rollouts_per_action: int = 1,
    max_evaluations: int = 4096,
    max_rollout_steps: int = 512,
    selection: str = "lcb",
    lcb_z: float = 1.0,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, object]:
    source = Path(input_path).resolve()
    checksum = (
        Path(checksum_path).resolve()
        if checksum_path is not None
        else Path(f"{source}.sha256")
    )
    output = Path(output_path).resolve()
    if output.suffix != ".ndjson":
        raise ValueError("V4 search-teacher output must end in .ndjson")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    shard_count = _positive_integer(shard_count, "shard-count")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("shard-index must be an integer from 0 to shard-count - 1")
    target_trajectories = _positive_integer(target_trajectories, "target-trajectories")
    hypotheses = _positive_integer(hypotheses, "hypotheses")
    rollouts_per_action = _positive_integer(
        rollouts_per_action, "rollouts-per-action"
    )
    max_evaluations = _positive_integer(max_evaluations, "max-evals")
    max_rollout_steps = _positive_integer(max_rollout_steps, "max-steps")
    lcb_z = _finite_nonnegative(lcb_z, "lcb-z")
    if selection not in {"mean", "lcb"}:
        raise ValueError("selection must be mean or lcb")

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parent.parent
    )
    outputs = _output_paths(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_outputs_absent(outputs)

    input_sha = _verify_sidecar(source, checksum)
    manifest_info, normalized_samples, input_summary, raw_data = _read_and_validate(
        source, root
    )
    if manifest_info.get("optionalFields"):
        raise ValueError("search teacher requires an unlabeled strict Normal input")
    lines = raw_data.splitlines(keepends=True)
    raw_manifest = _json_load(lines[0][:-1], "manifest line 1")
    raw_samples = [
        _json_load(line[:-1], f"sample line {index}")
        for index, line in enumerate(lines[1:-1], start=2)
    ]
    raw_sample_lines = list(lines[1:-1])
    if len(raw_samples) != len(normalized_samples):
        raise AssertionError("validated and raw V4 sample counts diverged")
    for record in raw_samples:
        if set(record) != BASE_SAMPLE_KEYS:
            raise ValueError("search teacher input samples must have the base Normal schema")
    player_count = int(manifest_info["playerCount"])
    groups = _trajectory_groups(raw_samples, player_count)
    eligible_groups, eligible_episodes, episode_assignments = (
        _partition_episode_trajectories(
            groups,
            seed=seed,
            shard_count=shard_count,
            shard_index=shard_index,
        )
    )
    selected, balance_counts = _select_balanced_trajectories(
        eligible_groups,
        player_count=player_count,
        seed=seed,
        target=target_trajectories,
    )

    adapter = V4ScalarSearchAdapter(device="cpu")
    config_values = {
        "seed": seed,
        "targetTrajectories": target_trajectories,
        "hypotheses": hypotheses,
        "rolloutsPerAction": rollouts_per_action,
        "maxEvaluations": max_evaluations,
        "maxRolloutSteps": max_rollout_steps,
        "selection": selection,
        "lcbZ": lcb_z,
        "shardCount": shard_count,
        "shardIndex": shard_index,
        "maxSeconds": None,
        "distributionTemperature": 0.2,
    }
    expert_by_index: dict[int, int] = {}
    changed = 0
    selected_nonforced = 0
    selected_forced = 0
    diagnostics_totals: Counter[str] = Counter()
    stopped_reasons: Counter[str] = Counter()
    for index, record in enumerate(raw_samples):
        if str(record["trajectoryId"]) not in selected:
            continue
        legal = _legal_mask(record["legalActionIndices"])
        normal_action = int(record["actionIndex"])
        if bool(record["forced"]):
            selected_forced += 1
            expert = normal_action
        else:
            selected_nonforced += 1
            sample_search_seed = _sample_seed(
                seed, str(record["trajectoryId"]), int(record["step"])
            )
            normal_determinization = determinize_v4_unseen_hands(
                record["actorObservation"], seed=sample_search_seed
            )
            normal_root = adapter.build_root(
                record["actorObservation"],
                normal_determinization,
                sample_search_seed,
            )
            if normal_root.env.normal_action() != normal_action:
                raise ValueError(
                    "selected input actionIndex is not the exact public-information "
                    "Normal action"
                )
            search_config = V4SearchConfig(
                seed=sample_search_seed,
                hypotheses=hypotheses,
                rollouts_per_action=rollouts_per_action,
                max_evaluations=max_evaluations,
                max_seconds=None,
                max_rollout_steps=max_rollout_steps,
                selection=selection,
                lcb_z=lcb_z,
                distribution_temperature=0.2,
            )
            result = run_v4_search_teacher(
                record["actorObservation"],
                legal,
                adapter,
                config=search_config,
                rollout_policy=None,
                batched_leaf_evaluator=None,
            )
            diagnostic = result.diagnostics
            if diagnostic.incomplete_legal_actions:
                raise RuntimeError(
                    "search max-evals left legal actions without an exact terminal sample"
                )
            if (
                diagnostic.batched_leaf_evaluations != 0
                or diagnostic.terminal_evaluations != diagnostic.evaluations
            ):
                raise RuntimeError("search teacher produced a non-terminal or learned leaf")
            expert = int(result.teacher_action)
            diagnostics_totals.update({
                "searchCalls": 1,
                "legalActions": diagnostic.legal_action_count,
                "hypothesesGenerated": diagnostic.hypotheses_generated,
                "uniqueDeterminizationsSummed": diagnostic.unique_determinizations,
                "evaluations": diagnostic.evaluations,
                "terminalEvaluations": diagnostic.terminal_evaluations,
                "batchedLeafEvaluations": diagnostic.batched_leaf_evaluations,
            })
            stopped_reasons[diagnostic.stopped_reason] += 1
            changed += int(expert != normal_action)
        if not legal[expert]:
            raise AssertionError("V4 search teacher selected an illegal action")
        expert_by_index[index] = expert
    if selected_nonforced < 1:
        raise ValueError("selected trajectories contain no non-forced teacher decisions")

    output_manifest = copy.deepcopy(dict(raw_manifest))
    output_manifest["environment"]["collection"][
        "targetNonForcedDecisions"
    ] = selected_nonforced
    manifest_line = _canonical_json_line(output_manifest)
    selected_sample_lines: list[bytes] = []
    selected_episodes: set[str] = set()
    for index, (record, raw_line) in enumerate(zip(raw_samples, raw_sample_lines)):
        if index not in expert_by_index:
            continue
        selected_sample_lines.append(_inject_expert(raw_line, expert_by_index[index]))
        selected_episodes.add(str(record["episodeId"]))
    selected_samples = len(selected_sample_lines)
    if selected_samples != selected_forced + selected_nonforced:
        raise AssertionError("selected V4 teacher sample accounting drifted")
    records_before_summary = manifest_line + b"".join(selected_sample_lines)
    records_sha = _digest_bytes(records_before_summary)
    summary = {
        "type": "summary",
        "episodes": len(selected_episodes),
        "samples": selected_samples,
        "forcedSamples": selected_forced,
        "nonForcedSamples": selected_nonforced,
        "targetNonForcedDecisions": selected_nonforced,
        "recordsBeforeSummarySha256": records_sha,
    }
    ndjson_bytes = records_before_summary + _canonical_json_line(summary)
    output_sha = _digest_bytes(ndjson_bytes)
    selected_ids_sha = _selected_ids_hash(tuple(selected))
    eligible_trajectory_ids_sha = _selected_ids_hash(tuple(eligible_groups))
    eligible_episode_ids_sha = _selected_ids_hash(tuple(eligible_episodes))
    selected_episode_ids_sha = _selected_ids_hash(tuple(selected_episodes))
    source_sha256 = {
        "search": _source_hash(root, "gpu-training/v4_search.py"),
        "adapter": _source_hash(root, "gpu-training/v4_search_env_adapter.py"),
        "environment": _source_hash(root, "gpu-training/v4_env.py"),
        "labeler": _source_hash(root, "gpu-training/v4_label_search_teacher.py"),
    }
    metadata: dict[str, object] = {
        "format": TEACHER_METADATA_FORMAT,
        "version": TEACHER_METADATA_VERSION,
        "inputSha256": input_sha,
        "outputNdjsonSha256": output_sha,
        "recordsBeforeSummarySha256": records_sha,
        "sourceSha256": source_sha256,
        "config": config_values,
        "selection": {
            "availableTrajectories": len(eligible_groups),
            "selectedTrajectories": len(selected),
            "selectedTrajectoryIdsSha256": selected_ids_sha,
            "balancedStrata": balance_counts,
        },
        "episodeSharding": {
            "shardCount": shard_count,
            "shardIndex": shard_index,
            "assignmentRule": {
                "domain": EPISODE_SHARD_ASSIGNMENT_DOMAIN,
                "payload": (
                    "UTF-8(domain + NUL + base-10 seed + NUL + episodeId)"
                ),
                "digest": "SHA-256",
                "digestInteger": "unsigned-big-endian",
                "assignment": "digestInteger modulo shardCount",
            },
            "inputEpisodes": len(episode_assignments),
            "inputTrajectories": len(groups),
            "eligibleEpisodes": len(eligible_episodes),
            "eligibleTrajectories": len(eligible_groups),
            "eligibleEpisodeIdsSha256": eligible_episode_ids_sha,
            "eligibleTrajectoryIdsSha256": eligible_trajectory_ids_sha,
            "selectedEpisodes": len(selected_episodes),
            "selectedTrajectories": len(selected),
            "selectedEpisodeIdsSha256": selected_episode_ids_sha,
            "selectedTrajectoryIdsSha256": selected_ids_sha,
            "outputNamespace": (
                f"v4-search-teacher:{input_sha}:seed-{seed}:"
                f"shard-{shard_index}-of-{shard_count}"
            ),
        },
        "samples": {
            "total": selected_samples,
            "forced": selected_forced,
            "nonForced": selected_nonforced,
        },
        "changedVsNormal": {
            "count": changed,
            "nonForcedDenominator": selected_nonforced,
            "rate": changed / selected_nonforced,
        },
        "searchDiagnosticsTotals": {
            **dict(sorted(diagnostics_totals.items())),
            "stoppedReasons": dict(sorted(stopped_reasons.items())),
        },
        "inputSummary": input_summary,
        "privacy": {
            "rule": (
                "search receives canonical actorObservation only; opponent hands are "
                "sampled from public unseen cards; privilegedCriticState, actionIndex, "
                "reward, and real hidden ownership never enter the teacher"
            ),
            "privilegedFieldsCopiedUnchanged": True,
            "compressedHistory": (
                "truncatedHistoryCount>0 permits exact Normal terminal rollouts only; "
                "injected policies and public leaves are forbidden"
            ),
        },
    }
    metadata_bytes = (
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    metadata_sha = _digest_bytes(metadata_bytes)
    output_path_final, output_checksum, metadata_path, metadata_checksum = outputs
    _publish_atomically((
        (output_path_final, ndjson_bytes),
        (output_checksum, f"{output_sha}\n".encode("ascii")),
        (metadata_path, metadata_bytes),
        (metadata_checksum, f"{metadata_sha}\n".encode("ascii")),
    ))
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label complete strict V4 Normal trajectories with exact search teachers."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-checksum", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--target-trajectories", type=int, required=True)
    parser.add_argument("--hypotheses", type=int, default=8)
    parser.add_argument("--rollouts-per-action", type=int, default=1)
    parser.add_argument("--max-evals", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--selection", choices=("mean", "lcb"), default="lcb")
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = label_v4_search_teacher(
        args.input,
        args.output,
        checksum_path=args.input_checksum,
        repository_root=args.repository_root,
        seed=args.seed,
        target_trajectories=args.target_trajectories,
        hypotheses=args.hypotheses,
        rollouts_per_action=args.rollouts_per_action,
        max_evaluations=args.max_evals,
        max_rollout_steps=args.max_steps,
        selection=args.selection,
        lcb_z=args.lcb_z,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "outputSha256": metadata["outputNdjsonSha256"],
        "selectedTrajectories": metadata["selection"]["selectedTrajectories"],
        "changedVsNormal": metadata["changedVsNormal"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TEACHER_METADATA_FORMAT",
    "EPISODE_SHARD_ASSIGNMENT_DOMAIN",
    "V4ScalarSearchAdapter",
    "_episode_shard_index",
    "_partition_episode_trajectories",
    "label_v4_search_teacher",
    "main",
]
