from __future__ import annotations

"""Read-only, two-stage paired counterfactual diagnostics for V5 Actors.

Stage one scans only the immutable Actor partition and emits a deliberately
minimal selection file.  Stage two is a separate replay boundary: it opens the
training shard, reconstructs matches from their saved seed, replays the saved
behaviour actions, and forks selected roots to the final-Actor and exact-Normal
actions.  No Actor inference is performed during replay and no raw hand or
privileged-state value is included in the report.

This module is diagnostic-only.  Its output is not a promotion, certification,
or deployment artifact.
"""

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np
import torch

from v4_collect_fixed_match_ppo import ACTS_PER_MATCH
from v4_env import ACTION_COUNT, DalmutiScalarEnv
from v4_exact_search_screen import ActOutcome, _rollout_root_action
from v5_collect_mappo import (
    canonicalize_v5_privileged_state,
    derive_v5_collection_match_seed,
)
from v5_dataset import (
    V5ActorShard,
    V5TrainingShard,
    load_v5_actor_shard,
    load_v5_training_shard,
)
from v5_export import (
    canonical_json_bytes,
    load_v5_actor_bundle,
    sha256_file,
    v5_actor_bundle_digests,
)
from v5_model import configure_v5_policy_numerics
from v5_public import (
    actor_batch_from_packed_arrays,
    pack_v5_public_observations,
    v5_public_from_v4_actor_observation,
)


SELECTION_KEYS = frozenset(
    {
        "shardManifestSha256",
        "localRow",
        "finalAction",
        "normalAction",
        "margin",
    }
)
REPORT_FORMAT = "dalmuti-v5-paired-action-counterfactual-diagnostic"
REPORT_VERSION = 1
SCAN_RECEIPT_FORMAT = "dalmuti-v5-public-deviation-selection-receipt"
SCAN_RECEIPT_VERSION = 1
PLAYER_COUNTS = tuple(range(4, 11))
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PUBLIC_ROW_NAMES = (
    "global_codes",
    "own_rank_counts",
    "public_played_counts",
    "player_codes",
    "player_masks",
    "table_codes",
    "legal_action_bits",
    "belief_response_feasibility",
)
_ACTOR_IDENTITY_KEYS = frozenset(
    {
        "actorSha256",
        "manifestSha256",
        "tensorStateSha256",
        "publicContractSha256",
        "policyNumericsSha256",
    }
)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class CounterfactualCaps:
    """Independent per-player-count match and root ceilings."""

    match_caps: tuple[tuple[int, int], ...]
    root_caps: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        for name in ("match_caps", "root_caps"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a non-empty tuple")
            players: list[int] = []
            for item in values:
                if (
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or isinstance(item[0], bool)
                    or not isinstance(item[0], int)
                    or item[0] not in PLAYER_COUNTS
                ):
                    raise ValueError(f"{name} entries must be (p4..p10, positive cap)")
                _positive_integer(item[1], f"{name} cap")
                players.append(item[0])
            if players != sorted(set(players)):
                raise ValueError(f"{name} player counts must be sorted and unique")
        if tuple(player for player, _ in self.match_caps) != tuple(
            player for player, _ in self.root_caps
        ):
            raise ValueError("match and root caps must cover the same player counts")

    @property
    def matches(self) -> dict[int, int]:
        return dict(self.match_caps)

    @property
    def roots(self) -> dict[int, int]:
        return dict(self.root_caps)

    def report_value(self) -> dict[str, dict[str, int]]:
        return {
            "matchesPerPlayerCount": {
                str(player): count for player, count in self.match_caps
            },
            "rootsPerPlayerCount": {
                str(player): count for player, count in self.root_caps
            },
        }


@dataclass(frozen=True)
class _Deviation:
    shard_sha256: str
    local_row: int
    local_match: int
    player_count: int
    final_action: int
    normal_action: int
    margin: float

    def selection_value(self) -> dict[str, object]:
        # This exact five-field boundary is intentional.  In particular it has
        # no match seed, state tensor, observation, player identity, or hand.
        return {
            "shardManifestSha256": self.shard_sha256,
            "localRow": self.local_row,
            "finalAction": self.final_action,
            "normalAction": self.normal_action,
            "margin": self.margin,
        }


class _ReplayMismatch(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _resolved_shard_paths(shard_paths: Sequence[str | Path]) -> tuple[Path, ...]:
    if not shard_paths:
        raise ValueError("at least one source shard is required")
    paths = tuple(Path(path).resolve(strict=True) for path in shard_paths)
    if len(set(paths)) != len(paths):
        raise ValueError("source shard paths must be unique")
    return paths


def _shard_manifest_sha256(path: Path) -> str:
    return _require_sha256(sha256_file(path / "manifest.json"), "shard manifest SHA")


def _match_for_rows(offsets: np.ndarray, rows: np.ndarray) -> np.ndarray:
    matches = np.searchsorted(offsets, rows, side="right") - 1
    if (
        np.any(matches < 0)
        or np.any(matches >= len(offsets) - 1)
        or np.any(rows >= offsets[matches + 1])
    ):
        raise ValueError("decision row does not belong to an Actor match interval")
    return matches.astype(np.int64, copy=False)


def _infer_deviations(
    actor: torch.nn.Module,
    shard: V5ActorShard,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> list[tuple[int, int, int, float]]:
    """Return ``(row, final, normal, margin)`` for greedy deviations only."""

    output: list[tuple[int, int, int, float]] = []
    for lower in range(0, len(rows), batch_size):
        selected = rows[lower : lower + batch_size]
        if selected.size == 0:
            continue
        public = actor_batch_from_packed_arrays(shard.arrays, selected, device)
        normal = torch.as_tensor(
            np.asarray(shard.arrays["normal_actions"])[selected].astype(
                np.int64, copy=False
            ),
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            packed = actor.forward_packed_batch(public, normal)
            final = packed.greedy_actions()
            normal_positions = (
                (packed.action_indices == normal[:, None]) & packed.action_mask
            )
            if not bool(normal_positions.sum(dim=1).eq(1).all()):
                raise RuntimeError("packed final Actor omitted the legal Normal action")
            final_positions = (
                (packed.action_indices == final[:, None]) & packed.action_mask
            )
            if not bool(final_positions.sum(dim=1).eq(1).all()):
                raise RuntimeError("packed final Actor omitted its greedy action")
            final_logits = packed.logits.gather(
                1, final_positions.to(torch.int64).argmax(dim=1, keepdim=True)
            ).squeeze(1)
            normal_logits = packed.logits.gather(
                1, normal_positions.to(torch.int64).argmax(dim=1, keepdim=True)
            ).squeeze(1)
            margins = final_logits - normal_logits
        cpu_final = final.detach().cpu().numpy()
        cpu_normal = normal.detach().cpu().numpy()
        cpu_margins = margins.detach().cpu().to(torch.float64).numpy()
        for local_row, final_action, normal_action, margin in zip(
            selected, cpu_final, cpu_normal, cpu_margins, strict=True
        ):
            if int(final_action) == int(normal_action):
                continue
            numeric_margin = float(margin)
            if not math.isfinite(numeric_margin) or numeric_margin < 0.0:
                raise RuntimeError("greedy Actor deviation has an invalid margin")
            output.append(
                (
                    int(local_row),
                    int(final_action),
                    int(normal_action),
                    numeric_margin,
                )
            )
    return output


def scan_actor_deviations(
    actor: torch.nn.Module,
    shard_paths: Sequence[str | Path],
    caps: CounterfactualCaps,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> list[dict[str, object]]:
    """Stage one: scan public Actor mmaps and select deterministic deviations.

    This function deliberately calls only :func:`load_v5_actor_shard`.  Match
    provenance and privileged arrays are neither loaded nor inspected.
    """

    _positive_integer(batch_size, "batch size")
    resolved_device = torch.device(device)
    configure_v5_policy_numerics(resolved_device)
    if float(getattr(getattr(actor, "config", None), "dropout", 0.0)) != 0.0:
        raise ValueError("counterfactual scan requires Actor dropout=0")
    actor = actor.to(resolved_device).eval()
    paths = _resolved_shard_paths(shard_paths)
    requested_players = set(caps.matches)
    best_by_match: dict[tuple[str, int], _Deviation] = {}
    path_by_sha: dict[str, Path] = {}

    # First public-only pass: retain only the strongest deviation per match so
    # match caps can be chosen without storing millions of decision records.
    for path in paths:
        shard = load_v5_actor_shard(path)
        try:
            shard_sha = _shard_manifest_sha256(path)
            if shard_sha in path_by_sha:
                raise ValueError("source shards contain a duplicate manifest SHA")
            path_by_sha[shard_sha] = path
            forced = np.asarray(shard.arrays["forced"], dtype=np.bool_)
            rows = np.flatnonzero(~forced).astype(np.int64, copy=False)
            deviations = _infer_deviations(
                actor,
                shard,
                rows,
                device=resolved_device,
                batch_size=batch_size,
            )
            offsets = np.asarray(shard.arrays["match_offsets"], dtype=np.int64)
            if deviations:
                deviation_rows = np.asarray(
                    [item[0] for item in deviations], dtype=np.int64
                )
                local_matches = _match_for_rows(offsets, deviation_rows)
            else:
                local_matches = np.empty((0,), dtype=np.int64)
            players = np.asarray(shard.arrays["player_counts"], dtype=np.int64)
            for values, local_match in zip(
                deviations, local_matches.tolist(), strict=True
            ):
                local_row, final_action, normal_action, margin = values
                player_count = int(players[local_match])
                if player_count not in requested_players:
                    continue
                candidate = _Deviation(
                    shard_sha,
                    local_row,
                    int(local_match),
                    player_count,
                    final_action,
                    normal_action,
                    margin,
                )
                key = (shard_sha, int(local_match))
                previous = best_by_match.get(key)
                if previous is None or (
                    -candidate.margin,
                    candidate.shard_sha256,
                    candidate.local_row,
                ) < (
                    -previous.margin,
                    previous.shard_sha256,
                    previous.local_row,
                ):
                    best_by_match[key] = candidate
        finally:
            shard.close()

    selected_matches: dict[int, set[tuple[str, int]]] = {
        player: set() for player in requested_players
    }
    for player_count, match_cap in caps.match_caps:
        candidates = sorted(
            (
                value
                for value in best_by_match.values()
                if value.player_count == player_count
            ),
            key=lambda value: (
                -value.margin,
                value.shard_sha256,
                value.local_match,
                value.local_row,
            ),
        )
        selected_matches[player_count] = {
            (value.shard_sha256, value.local_match)
            for value in candidates[:match_cap]
        }

    # Second public-only pass: the selected matches are small, so retain all of
    # their deviations and apply the independent per-stratum root cap exactly.
    deviations_by_player: dict[int, list[_Deviation]] = {
        player: [] for player in requested_players
    }
    selected_by_sha: dict[str, set[int]] = {}
    for matches in selected_matches.values():
        for shard_sha, local_match in matches:
            selected_by_sha.setdefault(shard_sha, set()).add(local_match)
    for shard_sha, local_matches in sorted(selected_by_sha.items()):
        shard = load_v5_actor_shard(path_by_sha[shard_sha])
        try:
            offsets = np.asarray(shard.arrays["match_offsets"], dtype=np.int64)
            players = np.asarray(shard.arrays["player_counts"], dtype=np.int64)
            rows = np.concatenate(
                [
                    np.arange(
                        int(offsets[local_match]),
                        int(offsets[local_match + 1]),
                        dtype=np.int64,
                    )
                    for local_match in sorted(local_matches)
                ]
            )
            forced = np.asarray(shard.arrays["forced"], dtype=np.bool_)
            rows = rows[~forced[rows]]
            deviations = _infer_deviations(
                actor,
                shard,
                rows,
                device=resolved_device,
                batch_size=batch_size,
            )
            local_match_rows = _match_for_rows(
                offsets,
                np.asarray([item[0] for item in deviations], dtype=np.int64),
            ) if deviations else np.empty((0,), dtype=np.int64)
            for values, local_match in zip(
                deviations, local_match_rows.tolist(), strict=True
            ):
                local_row, final_action, normal_action, margin = values
                player_count = int(players[local_match])
                if (shard_sha, int(local_match)) not in selected_matches[player_count]:
                    raise RuntimeError("public selection match routing drifted")
                deviations_by_player[player_count].append(
                    _Deviation(
                        shard_sha,
                        local_row,
                        int(local_match),
                        player_count,
                        final_action,
                        normal_action,
                        margin,
                    )
                )
        finally:
            shard.close()

    chosen: list[_Deviation] = []
    for player_count, root_cap in caps.root_caps:
        candidates = sorted(
            deviations_by_player[player_count],
            key=lambda value: (
                -value.margin,
                value.shard_sha256,
                value.local_row,
            ),
        )
        chosen.extend(candidates[:root_cap])
    chosen.sort(key=lambda value: (value.shard_sha256, value.local_row))
    records = [value.selection_value() for value in chosen]
    _validate_selection_records(records)
    return records


def _validate_selection_records(
    values: object,
) -> list[dict[str, object]]:
    if not isinstance(values, list):
        raise ValueError("selection file must be a canonical JSON list")
    output: list[dict[str, object]] = []
    coordinates: set[tuple[str, int]] = set()
    for value in values:
        if not isinstance(value, dict) or set(value) != SELECTION_KEYS:
            raise ValueError("selection record fields drifted")
        shard_sha = _require_sha256(
            value["shardManifestSha256"], "selection shard manifest SHA"
        )
        row = value["localRow"]
        final = value["finalAction"]
        normal = value["normalAction"]
        margin = value["margin"]
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or row < 0
            or isinstance(final, bool)
            or not isinstance(final, int)
            or not 0 <= final < ACTION_COUNT
            or isinstance(normal, bool)
            or not isinstance(normal, int)
            or not 0 <= normal < ACTION_COUNT
            or final == normal
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
            or float(margin) < 0.0
        ):
            raise ValueError("selection record scalar is invalid")
        coordinate = (shard_sha, row)
        if coordinate in coordinates:
            raise ValueError("selection records contain a duplicate shard row")
        coordinates.add(coordinate)
        output.append(
            {
                "shardManifestSha256": shard_sha,
                "localRow": row,
                "finalAction": final,
                "normalAction": normal,
                "margin": float(margin),
            }
        )
    if output != sorted(
        output, key=lambda item: (str(item["shardManifestSha256"]), int(item["localRow"]))
    ):
        raise ValueError("selection records must be sorted by shard SHA and local row")
    return output


def write_selection_file(
    output_path: str | Path, records: Sequence[Mapping[str, object]]
) -> str:
    """Publish the bare five-field selection list with a checksum sidecar."""

    path = Path(output_path).resolve()
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError("selection output must be a new path")
    normalized = _validate_selection_records([dict(value) for value in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(normalized))
    temporary.replace(path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def load_selection_file(path: str | Path) -> tuple[list[dict[str, object]], str]:
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"selection contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selection is not canonical ASCII JSON") from error
    if canonical_json_bytes(value) != raw:
        raise ValueError("selection is not canonical JSON")
    return _validate_selection_records(value), sha256_file(source)


def _validate_actor_identity(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _ACTOR_IDENTITY_KEYS:
        raise ValueError("scan receipt final Actor identity fields drifted")
    return {
        name: _require_sha256(value[name], f"final Actor {name}")
        for name in sorted(_ACTOR_IDENTITY_KEYS)
    }


def build_scan_receipt(
    actor_identity: Mapping[str, object],
    selection_sha256: str,
    selection_records: Sequence[Mapping[str, object]],
    source_shard_sha256: Sequence[str],
    caps: CounterfactualCaps,
) -> dict[str, object]:
    records = _validate_selection_records(
        [dict(value) for value in selection_records]
    )
    selection_sha = _require_sha256(selection_sha256, "selection SHA")
    source_shas = sorted(
        {_require_sha256(value, "source shard manifest SHA") for value in source_shard_sha256}
    )
    if not source_shas:
        raise ValueError("scan receipt requires at least one source shard")
    if not {
        str(value["shardManifestSha256"]) for value in records
    }.issubset(source_shas):
        raise ValueError("selection references a shard outside its scan receipt")
    return {
        "format": SCAN_RECEIPT_FORMAT,
        "version": SCAN_RECEIPT_VERSION,
        "diagnosticOnly": True,
        "promotionEligible": False,
        "publicActorPartitionOnly": True,
        "privatePartitionOpened": False,
        "finalActorIdentity": _validate_actor_identity(dict(actor_identity)),
        "selectionSha256": selection_sha,
        "selectedRoots": len(records),
        "caps": caps.report_value(),
        "sourceShardManifestSha256": source_shas,
    }


def _validate_scan_receipt(
    value: object,
    *,
    selection_sha256: str,
    selection_count: int,
    caps: CounterfactualCaps,
) -> dict[str, object]:
    required = {
        "format",
        "version",
        "diagnosticOnly",
        "promotionEligible",
        "publicActorPartitionOnly",
        "privatePartitionOpened",
        "finalActorIdentity",
        "selectionSha256",
        "selectedRoots",
        "caps",
        "sourceShardManifestSha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("public scan receipt fields drifted")
    if (
        value["format"] != SCAN_RECEIPT_FORMAT
        or type(value["version"]) is not int
        or value["version"] != SCAN_RECEIPT_VERSION
        or value["diagnosticOnly"] is not True
        or value["promotionEligible"] is not False
        or value["publicActorPartitionOnly"] is not True
        or value["privatePartitionOpened"] is not False
        or value["selectionSha256"] != selection_sha256
        or type(value["selectedRoots"]) is not int
        or value["selectedRoots"] != selection_count
        or value["caps"] != caps.report_value()
    ):
        raise ValueError("public scan receipt contract or selection binding drifted")
    actor_identity = _validate_actor_identity(value["finalActorIdentity"])
    raw_shas = value["sourceShardManifestSha256"]
    if (
        not isinstance(raw_shas, list)
        or not raw_shas
        or any(not isinstance(item, str) for item in raw_shas)
        or raw_shas != sorted(set(raw_shas))
    ):
        raise ValueError("public scan receipt source shard list is invalid")
    source_shas = [
        _require_sha256(item, "scan receipt source shard SHA") for item in raw_shas
    ]
    return {**value, "finalActorIdentity": actor_identity, "sourceShardManifestSha256": source_shas}


def write_scan_receipt(output_path: str | Path, receipt: Mapping[str, object]) -> str:
    path = Path(output_path).resolve()
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError("scan receipt output must be a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(dict(receipt))
    # The builder owns validation; this round trip rejects non-JSON payloads.
    json.loads(raw)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def load_scan_receipt(path: str | Path) -> tuple[dict[str, object], str]:
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"scan receipt contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("scan receipt is not canonical ASCII JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("scan receipt is not a canonical JSON object")
    return value, sha256_file(source)


def _verify_replayed_row(
    training: V5TrainingShard,
    local_row: int,
    env: DalmutiScalarEnv,
) -> None:
    actor = training.actor
    arrays = actor.arrays
    observation = env.observe()
    public = v5_public_from_v4_actor_observation(observation.public)
    packed, history, history_end = pack_v5_public_observations([public])
    for name in _PUBLIC_ROW_NAMES:
        if not np.array_equal(np.asarray(arrays[name][local_row]), packed[name][0]):
            raise _ReplayMismatch("public-row-mismatch")
    if int(history_end[0]) != len(history) or not np.array_equal(
        actor.history(local_row), history
    ):
        raise _ReplayMismatch("public-row-mismatch")
    if (
        int(arrays["decision_actor_ids"][local_row]) != env.current_player_id
        or int(arrays["decision_acts"][local_row]) != int(env._act)
        or int(arrays["normal_actions"][local_row]) != int(env.normal_action())
        or bool(arrays["forced"][local_row])
        != (int(public.legal_mask.sum()) == 1)
    ):
        raise _ReplayMismatch("structural-row-mismatch")
    canonical_private = (
        canonicalize_v5_privileged_state(observation.privileged_state)
        .numpy()
        .astype(np.float16, copy=False)
    )
    persisted_private = np.asarray(
        training.privileged_arrays["privileged_states"][local_row],
        dtype=np.float16,
    )
    if not np.array_equal(canonical_private, persisted_private):
        raise _ReplayMismatch("private-row-mismatch")


def _state_fingerprint_sha256(value: tuple[object, ...]) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _outcome_value(value: ActOutcome) -> dict[str, object]:
    return {
        "chipAward": value.chip_award,
        "finishPlace": value.finish_place,
        "environmentReward": value.environment_reward,
        "simulatedSteps": value.simulated_steps,
        "exactNormalContinuationSteps": value.exact_normal_continuation_steps,
    }


def _counterfactual_root(
    env: DalmutiScalarEnv,
    selection: Mapping[str, object],
    *,
    max_rollout_steps: int,
) -> dict[str, object]:
    local_row = int(selection["localRow"])
    final_action = int(selection["finalAction"])
    normal_action = int(selection["normalAction"])
    if int(env.normal_action()) != normal_action:
        raise _ReplayMismatch("selection-normal-mismatch")
    legal = env.legal_mask()
    if not bool(legal[final_action]) or not bool(legal[normal_action]):
        raise _ReplayMismatch("selection-action-illegal")
    root_actor = int(env.current_player_id)
    root_act = int(env._act)
    source_fingerprint = env.state_fingerprint()
    source_rng_state = env._rng.state
    final_branch = copy.deepcopy(env)
    normal_branch = copy.deepcopy(env)
    if (
        final_branch.state_fingerprint() != source_fingerprint
        or normal_branch.state_fingerprint() != source_fingerprint
    ):
        raise _ReplayMismatch("branch-state-mismatch")
    if (
        final_branch._rng.state != source_rng_state
        or normal_branch._rng.state != source_rng_state
    ):
        raise _ReplayMismatch("branch-rng-mismatch")
    try:
        final_outcome = _rollout_root_action(
            final_branch,
            final_action,
            root_actor_id=root_actor,
            max_rollout_steps=max_rollout_steps,
        )
        normal_outcome = _rollout_root_action(
            normal_branch,
            normal_action,
            root_actor_id=root_actor,
            max_rollout_steps=max_rollout_steps,
        )
    except (RuntimeError, ValueError) as error:
        raise _ReplayMismatch("counterfactual-rollout-failed") from error
    if (
        int(final_branch._act) not in (root_act, root_act + 1)
        or int(normal_branch._act) not in (root_act, root_act + 1)
    ):
        raise _ReplayMismatch("counterfactual-crossed-root-act")
    if env.state_fingerprint() != source_fingerprint or env._rng.state != source_rng_state:
        raise _ReplayMismatch("source-mutated-by-branch")
    if int(env._act) != root_act:
        raise _ReplayMismatch("source-crossed-root-act")
    return {
        "localRow": local_row,
        "actorId": root_actor,
        "act": root_act,
        "finalAction": final_action,
        "normalAction": normal_action,
        "margin": float(selection["margin"]),
        "stateFingerprintSha256": _state_fingerprint_sha256(source_fingerprint),
        "branchStateVerified": True,
        "rngStateVerified": True,
        "currentActBoundVerified": True,
        "finalActorOutcome": _outcome_value(final_outcome),
        "normalOutcome": _outcome_value(normal_outcome),
        "finalMinusNormalChip": final_outcome.chip_award - normal_outcome.chip_award,
        "normalMinusFinalPlace": normal_outcome.finish_place - final_outcome.finish_place,
    }


def _cluster_id(shard_sha: str, local_match: int) -> str:
    material = f"{shard_sha}\0{local_match}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _replay_match(
    training: V5TrainingShard,
    shard_sha: str,
    local_match: int,
    selected_by_row: Mapping[int, Mapping[str, object]],
    *,
    max_rollout_steps: int,
) -> dict[str, object]:
    arrays = training.actor.arrays
    offsets = np.asarray(arrays["match_offsets"], dtype=np.int64)
    start = int(offsets[local_match])
    stop = int(offsets[local_match + 1])
    player_count = int(arrays["player_counts"][local_match])
    bitset = int(arrays["candidate_bitsets"][local_match])
    match_index = int(training.privileged_arrays["match_indices"][local_match])
    seed = int(training.privileged_arrays["match_seeds"][local_match])
    cluster_id = _cluster_id(shard_sha, local_match)
    roots: list[dict[str, object]] = []
    try:
        metadata = training.actor.manifest.get("metadata")
        if isinstance(metadata, Mapping) and {
            "runNamespace", "seedBase"
        }.issubset(metadata):
            try:
                derived_seed = derive_v5_collection_match_seed(
                    str(metadata["runNamespace"]),
                    int(metadata["seedBase"]),
                    player_count,
                    match_index,
                )
            except (TypeError, ValueError) as error:
                raise _ReplayMismatch("saved-seed-provenance-invalid") from error
            if derived_seed != seed:
                raise _ReplayMismatch("saved-seed-provenance-mismatch")
        env = DalmutiScalarEnv(
            player_count,
            acts=ACTS_PER_MATCH,
            seed=seed,
            device="cpu",
        )
        row = start
        total_steps = 0
        while not env.terminated:
            total_steps += 1
            if total_steps > 100_000:
                raise _ReplayMismatch("saved-behavior-replay-step-limit")
            actor_id = int(env.current_player_id)
            normal_action = int(env.normal_action())
            if bitset & (1 << actor_id):
                if row >= stop:
                    raise _ReplayMismatch("saved-behavior-row-overrun")
                _verify_replayed_row(training, row, env)
                selection = selected_by_row.get(row)
                if selection is not None:
                    roots.append(
                        _counterfactual_root(
                            env,
                            selection,
                            max_rollout_steps=max_rollout_steps,
                        )
                    )
                behavior_action = int(arrays["actions"][row])
                if not bool(env.legal_mask()[behavior_action]):
                    raise _ReplayMismatch("saved-behavior-action-illegal")
                row += 1
            else:
                behavior_action = normal_action
            env.step(behavior_action)
        if row != stop:
            raise _ReplayMismatch("saved-behavior-row-underrun")
        if set(selected_by_row) != {int(value["localRow"]) for value in roots}:
            raise _ReplayMismatch("selected-root-not-replayed")
    except _ReplayMismatch as error:
        # Buffering roots until the complete match validates ensures one bad
        # public/private row invalidates the entire match, never a suffix only.
        return {
            "clusterId": cluster_id,
            "playerCount": player_count,
            "status": "failed",
            "failureCode": error.code,
            "selectedRootCount": 0,
            "roots": [],
        }
    return {
        "clusterId": cluster_id,
        "playerCount": player_count,
        "status": "complete",
        "selectedRootCount": len(roots),
        "roots": roots,
    }


def replay_paired_counterfactuals(
    selection_records: Sequence[Mapping[str, object]],
    selection_sha256: str,
    scan_receipt: Mapping[str, object],
    scan_receipt_sha256: str,
    shard_paths: Sequence[str | Path],
    caps: CounterfactualCaps,
    *,
    max_rollout_steps: int = 2048,
) -> dict[str, object]:
    """Stage two: exact saved-behaviour replay and paired root rollouts."""

    _positive_integer(max_rollout_steps, "maximum rollout steps")
    selection_sha = _require_sha256(selection_sha256, "selection SHA")
    selections = _validate_selection_records(
        [dict(value) for value in selection_records]
    )
    scan_receipt_sha = _require_sha256(scan_receipt_sha256, "scan receipt SHA")
    validated_scan_receipt = _validate_scan_receipt(
        dict(scan_receipt),
        selection_sha256=selection_sha,
        selection_count=len(selections),
        caps=caps,
    )
    paths = _resolved_shard_paths(shard_paths)
    path_by_sha: dict[str, Path] = {}
    for path in paths:
        digest = _shard_manifest_sha256(path)
        if digest in path_by_sha:
            raise ValueError("source shards contain a duplicate manifest SHA")
        path_by_sha[digest] = path
    requested_shas = {str(value["shardManifestSha256"]) for value in selections}
    missing = requested_shas - set(path_by_sha)
    if missing:
        raise ValueError("selection references an unavailable source shard")
    receipt_source_shas = set(
        validated_scan_receipt["sourceShardManifestSha256"]  # type: ignore[arg-type]
    )
    if requested_shas - receipt_source_shas:
        raise ValueError("selection references a shard outside its public scan receipt")

    clusters: list[dict[str, object]] = []
    cap_matches: dict[int, set[tuple[str, int]]] = {
        player: set() for player in caps.matches
    }
    cap_roots = {player: 0 for player in caps.roots}
    for shard_sha in sorted(requested_shas):
        selected = {
            int(value["localRow"]): value
            for value in selections
            if value["shardManifestSha256"] == shard_sha
        }
        training = load_v5_training_shard(path_by_sha[shard_sha])
        try:
            if not {
                "match_indices", "match_seeds", "privileged_states"
            }.issubset(training.privileged_arrays):
                raise ValueError("training shard lacks replay provenance")
            offsets = np.asarray(
                training.actor.arrays["match_offsets"], dtype=np.int64
            )
            rows = np.asarray(sorted(selected), dtype=np.int64)
            if np.any(rows < 0) or np.any(rows >= training.actor.decision_count):
                raise ValueError("selection row is outside its source shard")
            local_matches = _match_for_rows(offsets, rows)
            by_match: dict[int, dict[int, Mapping[str, object]]] = {}
            for row, local_match in zip(rows.tolist(), local_matches.tolist(), strict=True):
                record = selected[row]
                arrays = training.actor.arrays
                if (
                    int(arrays["normal_actions"][row]) != int(record["normalAction"])
                    or int(record["finalAction"]) == int(record["normalAction"])
                    or not bool(training.actor.legal_mask(row)[int(record["finalAction"])])
                ):
                    raise ValueError("selection does not bind its source Actor row")
                player_count = int(arrays["player_counts"][local_match])
                if player_count not in cap_matches:
                    raise ValueError("selection uses a player count outside its caps")
                match_coordinate = (shard_sha, int(local_match))
                if (
                    match_coordinate not in cap_matches[player_count]
                    and len(cap_matches[player_count])
                    >= caps.matches[player_count]
                ):
                    raise ValueError(
                        "selection exceeds a per-player-count match cap"
                    )
                if cap_roots[player_count] >= caps.roots[player_count]:
                    raise ValueError(
                        "selection exceeds a per-player-count root cap"
                    )
                cap_matches[player_count].add(match_coordinate)
                cap_roots[player_count] += 1
                by_match.setdefault(int(local_match), {})[row] = record
            for local_match, match_selections in sorted(by_match.items()):
                clusters.append(
                    _replay_match(
                        training,
                        shard_sha,
                        local_match,
                        match_selections,
                        max_rollout_steps=max_rollout_steps,
                    )
                )
        finally:
            training.close()

    for player_count in sorted(cap_matches):
        if len(cap_matches[player_count]) > caps.matches[player_count]:
            raise ValueError("selection exceeds a per-player-count match cap")
        if cap_roots[player_count] > caps.roots[player_count]:
            raise ValueError("selection exceeds a per-player-count root cap")
    clusters.sort(key=lambda value: str(value["clusterId"]))
    complete = [value for value in clusters if value["status"] == "complete"]
    failed = [value for value in clusters if value["status"] == "failed"]
    completed_roots = sum(int(value["selectedRootCount"]) for value in complete)
    chip_differences = [
        int(root["finalMinusNormalChip"])
        for cluster in complete
        for root in cluster["roots"]  # type: ignore[index]
        if isinstance(root, Mapping)
    ]
    report: dict[str, object] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "diagnosticOnly": True,
        "promotionEligible": False,
        "actorInferenceDuringReplay": False,
        "selectionSha256": selection_sha,
        "scanReceiptSha256": scan_receipt_sha,
        "finalActorIdentity": validated_scan_receipt["finalActorIdentity"],
        "caps": caps.report_value(),
        "sourceShardManifestSha256": sorted(requested_shas),
        "summary": {
            "selectedRoots": len(selections),
            "completeMatchClusters": len(complete),
            "failedMatchClusters": len(failed),
            "completedRoots": completed_roots,
            "discardedRoots": len(selections) - completed_roots,
            "meanFinalMinusNormalChip": (
                sum(chip_differences) / len(chip_differences)
                if chip_differences
                else None
            ),
            "positiveChipRoots": sum(value > 0 for value in chip_differences),
            "tiedChipRoots": sum(value == 0 for value in chip_differences),
            "negativeChipRoots": sum(value < 0 for value in chip_differences),
        },
        "matchClusters": clusters,
    }
    # A canonical round trip is also a final guard against tensors, arrays, or
    # accidental non-finite values escaping the diagnostic boundary.
    json.loads(canonical_json_bytes(report))
    return report


def write_report(output_path: str | Path, report: Mapping[str, object]) -> str:
    path = Path(output_path).resolve()
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError("counterfactual report output must be a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(dict(report))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _parse_cap(value: str) -> tuple[int, int]:
    raw = value.lower().removeprefix("p")
    pieces = raw.split("=", 1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("caps must use p4=COUNT through p10=COUNT")
    try:
        player_count, count = (int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError("caps must contain integers") from error
    if player_count not in PLAYER_COUNTS or count < 1:
        raise argparse.ArgumentTypeError("caps must use p4..p10 and a positive count")
    return player_count, count


def _caps_from_arguments(arguments: argparse.Namespace) -> CounterfactualCaps:
    matches = tuple(sorted(arguments.match_cap))
    roots = tuple(sorted(arguments.root_cap))
    return CounterfactualCaps(matches, roots)


def _add_common_caps(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--match-cap",
        action="append",
        type=_parse_cap,
        required=True,
        help="repeat per stratum, for example --match-cap p4=4",
    )
    parser.add_argument(
        "--root-cap",
        action="append",
        type=_parse_cap,
        required=True,
        help="repeat per stratum, for example --root-cap p4=8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="public-only final Actor deviation scan")
    scan.add_argument("--actor-bundle", required=True)
    scan.add_argument("--shard", action="append", required=True)
    scan.add_argument("--selection-output", required=True)
    scan.add_argument("--scan-receipt-output", required=True)
    scan.add_argument("--device", default="cuda")
    scan.add_argument("--batch-size", type=int, default=64)
    _add_common_caps(scan)
    replay = commands.add_parser("replay", help="private-bound saved behaviour replay")
    replay.add_argument("--selection", required=True)
    replay.add_argument("--scan-receipt", required=True)
    replay.add_argument("--shard", action="append", required=True)
    replay.add_argument("--report-output", required=True)
    replay.add_argument("--max-rollout-steps", type=int, default=2048)
    _add_common_caps(replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    caps = _caps_from_arguments(arguments)
    if arguments.command == "scan":
        actor, _ = load_v5_actor_bundle(arguments.actor_bundle)
        records = scan_actor_deviations(
            actor,
            arguments.shard,
            caps,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        digest = write_selection_file(arguments.selection_output, records)
        source_shas = [
            _shard_manifest_sha256(path)
            for path in _resolved_shard_paths(arguments.shard)
        ]
        receipt = build_scan_receipt(
            v5_actor_bundle_digests(arguments.actor_bundle),
            digest,
            records,
            source_shas,
            caps,
        )
        receipt_digest = write_scan_receipt(
            arguments.scan_receipt_output, receipt
        )
        print(
            json.dumps(
                {
                    "selectionSha256": digest,
                    "scanReceiptSha256": receipt_digest,
                    "selectedRoots": len(records),
                },
                sort_keys=True,
            )
        )
        return 0
    records, selection_sha = load_selection_file(arguments.selection)
    scan_receipt, scan_receipt_sha = load_scan_receipt(arguments.scan_receipt)
    report = replay_paired_counterfactuals(
        records,
        selection_sha,
        scan_receipt,
        scan_receipt_sha,
        arguments.shard,
        caps,
        max_rollout_steps=arguments.max_rollout_steps,
    )
    digest = write_report(arguments.report_output, report)
    print(
        json.dumps(
            {"reportSha256": digest, **dict(report["summary"])},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CounterfactualCaps",
    "REPORT_FORMAT",
    "REPORT_VERSION",
    "SCAN_RECEIPT_FORMAT",
    "SCAN_RECEIPT_VERSION",
    "build_scan_receipt",
    "load_scan_receipt",
    "load_selection_file",
    "main",
    "replay_paired_counterfactuals",
    "scan_actor_deviations",
    "write_report",
    "write_scan_receipt",
    "write_selection_file",
]
