from __future__ import annotations

"""Deterministic, public-only DAgger collection for the V4 DALMUTI actor.

The game transition and expert are both supplied by :mod:`v4_env`.  Every
player uses the same seeded mixture of the candidate's greedy action and the
exact Normal action.  The candidate is therefore exposed to its own state
distribution, while every saved state still has an exact Normal supervision
label.

Actor observations and the 512-value privileged critic vector are copied into
different arrays.  In particular, candidate inference receives
``V4ActorObservation`` objects and has no privileged-state argument.
"""

import argparse
import copy
from dataclasses import dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch

from v4_dataset import (
    V4_DATASET_FORMAT,
    V4_DATASET_VERSION,
    V4TrajectoryDataset,
    V4TrajectoryTensors,
)
from v4_env import (
    ACTION_COUNT,
    MAX_HISTORY,
    MAX_PLAYERS,
    PRIVILEGED_STATE_LAYOUT,
    PRIVILEGED_STATE_LAYOUT_ID,
    PRIVILEGED_STATE_LAYOUT_SHA256,
    PRIVILEGED_STATE_SIZE,
    ROLES,
    DalmutiScalarEnv,
    V4ActorObservation,
    role_for_index,
)
from v4_export import (
    canonical_json_bytes,
    load_v4_actor_checkpoint,
    sha256_file,
    verify_v4_actor_bundle,
)
from v4_model import V4ActorConfig, V4CriticConfig


DAGGER_PREPARATION_FORMAT = "dalmuti-v4-dagger-direct-npz"
DAGGER_PREPARATION_VERSION = 1
NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PUBLIC_MODEL_INPUT_FIELDS = (
    "global_features",
    "rank_features",
    "player_features",
    "player_mask",
    "memory_trace_features",
    "history_features",
    "history_mask",
    "legal_mask",
)
SOURCE_FILES = (
    "gpu-training/v4_collect_dagger.py",
    "gpu-training/v4_env.py",
    "gpu-training/v4_model.py",
    "gpu-training/v4_export.py",
    "gpu-training/v4_dataset.py",
    "gpu-training/v3_action_conditioned.py",
    "lib/bot-strategy.ts",
)


@dataclass(frozen=True)
class DaggerCollectionConfig:
    run_namespace: str
    seed_base: int
    player_counts: tuple[int, ...] = tuple(range(4, 11))
    matches_per_player_count: int = 1
    acts: int = 5
    candidate_beta: float = 0.5
    lane_count: int = 32
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not isinstance(self.run_namespace, str) or not NAMESPACE_PATTERN.fullmatch(
            self.run_namespace
        ):
            raise ValueError(
                "run_namespace must use 1..128 ASCII letters, digits, '.', '_', or '-'"
            )
        if (
            isinstance(self.seed_base, bool)
            or not isinstance(self.seed_base, int)
            or not 0 <= self.seed_base <= 0xFFFF_FFFF
        ):
            raise ValueError("seed_base must be a uint32 integer")
        if (
            not isinstance(self.player_counts, tuple)
            or not self.player_counts
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 4 <= value <= 10
                for value in self.player_counts
            )
            or tuple(sorted(set(self.player_counts))) != self.player_counts
        ):
            raise ValueError("player_counts must be a sorted unique tuple from 4 through 10")
        for name in ("matches_per_player_count", "acts", "lane_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        beta = float(self.candidate_beta)
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("candidate_beta must be finite and in [0, 1]")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string")


@dataclass(frozen=True)
class DaggerCollectionResult:
    output_path: Path
    metadata_path: Path
    checksum_path: Path
    metadata_checksum_path: Path
    npz_sha256: str
    metadata_sha256: str
    fingerprint: str
    trajectories: int
    samples: int


@dataclass(frozen=True)
class _MatchSpec:
    player_count: int
    match_index: int
    seed: int


@dataclass
class _Lane:
    spec: _MatchSpec
    env: DalmutiScalarEnv
    decision_index: int = 0


@dataclass
class _RateCounter:
    decisions: int = 0
    forced: int = 0
    candidate_selected: int = 0
    candidate_expert_disagreements: int = 0
    behavior_expert_changes: int = 0

    def add(
        self,
        *,
        forced: bool,
        candidate_selected: bool,
        candidate_action: int,
        expert_action: int,
        behavior_action: int,
    ) -> None:
        self.decisions += 1
        self.forced += int(forced)
        self.candidate_selected += int(candidate_selected)
        self.candidate_expert_disagreements += int(candidate_action != expert_action)
        self.behavior_expert_changes += int(behavior_action != expert_action)

    def to_dict(self) -> dict[str, object]:
        denominator = max(1, self.decisions)
        selected = max(1, self.candidate_selected)
        return {
            "decisions": self.decisions,
            "forcedDecisions": self.forced,
            "candidateSelected": self.candidate_selected,
            "candidateSelectionRate": self.candidate_selected / denominator,
            "candidateExpertDisagreements": self.candidate_expert_disagreements,
            "candidateExpertDisagreementRate": (
                self.candidate_expert_disagreements / denominator
            ),
            "behaviorExpertChanges": self.behavior_expert_changes,
            "behaviorExpertChangeRate": self.behavior_expert_changes / denominator,
            "behaviorExpertChangeRateWhenCandidateSelected": (
                self.behavior_expert_changes / selected
            ),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _derive_uint32(namespace: str, seed_base: int, *parts: object) -> int:
    payload = canonical_json_bytes([namespace, seed_base, *parts])
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return value if value != 0 else 1


def _mixture_uniform(
    namespace: str,
    seed_base: int,
    spec: _MatchSpec,
    act: int,
    actor_id: int,
    decision_index: int,
) -> float:
    payload = canonical_json_bytes(
        [
            "v4-dagger-mixture-v1",
            namespace,
            seed_base,
            spec.player_count,
            spec.match_index,
            spec.seed,
            act,
            actor_id,
            decision_index,
        ]
    )
    # Exactly reproducible 53-bit [0, 1) value; independent of batch ordering.
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
    return integer / float(1 << 53)


def _source_hashes(repository_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = repository_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required DAgger source is missing: {relative}")
        hashes[relative] = sha256_file(source)
    return hashes


def _validate_actor_contract(
    config: V4ActorConfig, player_counts: Sequence[int]
) -> None:
    expected = {
        "global_features": 12,
        "rank_features": 6,
        "player_features": 12,
        "history_features": 20,
        "memory_features": 20,
        "rank_tokens": 13,
        "memory_tokens": 4,
        "observation_schema_version": 4,
        "action_catalogue_version": 1,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"candidate actor has incompatible {name}")
    if config.max_players > MAX_PLAYERS or max(player_counts) > config.max_players:
        raise ValueError("candidate actor max_players cannot represent the requested tables")
    if config.max_history > MAX_HISTORY:
        raise ValueError("candidate actor max_history exceeds the exact environment")


def _valid_prefix(mask: torch.Tensor, label: str) -> int:
    boolean = mask.detach().cpu().to(dtype=torch.bool)
    if boolean.ndim != 1:
        raise ValueError(f"{label} mask must be one-dimensional")
    indexes = torch.nonzero(boolean, as_tuple=False).flatten()
    if indexes.numel() == 0:
        return 0
    length = int(indexes[-1].item()) + 1
    if not bool(boolean[:length].all().item()) or bool(boolean[length:].any().item()):
        raise ValueError(f"{label} mask must be a contiguous valid prefix")
    return length


def _history_bucket(length: int, maximum: int) -> int:
    if not 0 <= length <= maximum:
        raise ValueError("history length is outside the actor configuration")
    buckets = sorted({0, maximum, *(value for value in (16, 32, 64, 96, 128, 160, 192) if value <= maximum)})
    return next(value for value in buckets if value >= length)


def _trim_public_for_model(
    public: V4ActorObservation, config: V4ActorConfig
) -> tuple[torch.Tensor, ...]:
    player_count = _valid_prefix(public.player_mask, "player")
    if not 1 <= player_count <= config.max_players:
        raise ValueError("public player count is outside the actor configuration")
    history_count = _valid_prefix(public.history_mask, "history")
    history_start = max(0, history_count - config.max_history)
    history = public.history_features[history_start:history_count]
    history_mask = public.history_mask[history_start:history_count]
    return (
        public.global_features,
        public.rank_features,
        public.player_features[:player_count],
        public.player_mask[:player_count],
        public.memory_trace_features,
        history,
        history_mask,
        public.legal_mask,
    )


def _pad_rows(value: torch.Tensor, target: int) -> torch.Tensor:
    if value.shape[0] == target:
        return value
    return torch.cat(
        (value, value.new_zeros((target - value.shape[0], *value.shape[1:]))), dim=0
    )


def _pad_mask(value: torch.Tensor, target: int) -> torch.Tensor:
    if value.shape[0] == target:
        return value.to(dtype=torch.bool)
    return torch.cat(
        (
            value.to(dtype=torch.bool),
            value.new_zeros((target - value.shape[0],), dtype=torch.bool),
        ),
        dim=0,
    )


def _batch_candidate_actions(
    model: object,
    observations: Sequence[V4ActorObservation],
    device: torch.device,
) -> list[int]:
    """Run one greedy candidate forward pass per public shape bucket.

    This function intentionally accepts public observations, not environment
    observations.  A candidate implementation therefore has no API route to
    the separate privileged critic vector.
    """

    if not observations:
        return []
    config = getattr(model, "config", None)
    if not isinstance(config, V4ActorConfig):
        raise ValueError("candidate model is missing a V4ActorConfig")
    rows = [_trim_public_for_model(public, config) for public in observations]
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        player_width = int(row[2].shape[0])
        history_width = _history_bucket(int(row[5].shape[0]), config.max_history)
        groups.setdefault((player_width, history_width), []).append(index)
    results: list[int | None] = [None] * len(rows)
    with torch.inference_mode():
        for (player_width, history_width), indexes in sorted(groups.items()):
            selected = [rows[index] for index in indexes]
            tensors = (
                torch.stack([row[0] for row in selected]).to(device),
                torch.stack([row[1] for row in selected]).to(device),
                torch.stack([_pad_rows(row[2], player_width) for row in selected]).to(device),
                torch.stack([_pad_mask(row[3], player_width) for row in selected]).to(device),
                torch.stack([row[4] for row in selected]).to(device),
                torch.stack([_pad_rows(row[5], history_width) for row in selected]).to(device),
                torch.stack([_pad_mask(row[6], history_width) for row in selected]).to(device),
                torch.stack([row[7].to(dtype=torch.bool) for row in selected]).to(device),
            )
            legal = tensors[-1]
            logits = model(*tensors)
            if tuple(logits.shape) != (len(indexes), ACTION_COUNT):
                raise ValueError("candidate returned invalid logits shape")
            if not bool(torch.isfinite(logits[legal]).all().item()):
                raise ValueError("candidate returned non-finite legal logits")
            actions = torch.argmax(
                logits.masked_fill(~legal, float("-inf")), dim=-1
            ).detach().cpu().tolist()
            for index, action in zip(indexes, actions, strict=True):
                if not bool(rows[index][7][int(action)].item()):
                    raise RuntimeError("candidate selected an illegal action")
                results[index] = int(action)
    if any(value is None for value in results):
        raise RuntimeError("candidate batching lost an observation")
    return [int(value) for value in results]


def _assert_public_equal(left: V4ActorObservation, right: V4ActorObservation) -> None:
    if left.actor_id != right.actor_id:
        raise RuntimeError("hidden-state audit changed actor identity")
    for name in ("valid", *PUBLIC_MODEL_INPUT_FIELDS):
        if not torch.equal(getattr(left, name), getattr(right, name)):
            raise RuntimeError(f"private hidden state leaked into actor tensor: {name}")


def audit_hidden_state_privacy(env: DalmutiScalarEnv, seed: int) -> dict[str, object]:
    """Prove that opponent-hand reassignments cannot change actor tensors."""

    if env.terminated:
        raise ValueError("privacy audit requires an active environment")
    before = env.observe()
    privileged_changed = False
    for attempt in range(1, 9):
        probe = copy.deepcopy(env)
        after = probe.resample_hidden_hands((int(seed) + attempt) & 0xFFFF_FFFF)
        _assert_public_equal(before.public, after.public)
        privileged_changed = privileged_changed or not torch.equal(
            before.privileged_state, after.privileged_state
        )
    if not privileged_changed:
        raise RuntimeError("privacy audit did not perturb the privileged state")
    return {
        "publicInvariantAcrossEightOpponentHandResamples": True,
        "privilegedStateChanged": True,
        "candidateInferenceArgument": "V4ActorObservation",
        "actorInputFields": list(PUBLIC_MODEL_INPUT_FIELDS),
        "opponentPhysicalHandsExcluded": True,
        "taxCardIdentitiesExcluded": True,
        "privilegedCriticArraySeparate": True,
    }


def _snapshot_public(
    public: V4ActorObservation, config: V4ActorConfig
) -> dict[str, np.ndarray]:
    player_count = _valid_prefix(public.player_mask, "player")
    history_count = _valid_prefix(public.history_mask, "history")
    if player_count > config.max_players:
        raise ValueError("observation exceeds candidate max_players")
    history_start = max(0, history_count - config.max_history)
    history = public.history_features[history_start:history_count]
    history_mask = public.history_mask[history_start:history_count]
    player_features = np.zeros(
        (config.max_players, config.player_features), dtype=np.float32
    )
    player_mask = np.zeros(config.max_players, dtype=np.bool_)
    player_features[:player_count] = (
        public.player_features[:player_count].detach().cpu().numpy()
    )
    player_mask[:player_count] = True
    history_features = np.zeros(
        (config.max_history, config.history_features), dtype=np.float32
    )
    output_history_mask = np.zeros(config.max_history, dtype=np.bool_)
    if history.shape[0]:
        history_features[: history.shape[0]] = history.detach().cpu().numpy()
        output_history_mask[: history.shape[0]] = history_mask.detach().cpu().numpy()
    legal = public.legal_mask.detach().cpu().to(dtype=torch.bool).numpy().copy()
    if legal.shape != (ACTION_COUNT,) or not bool(legal.any()):
        raise ValueError("each collected actor observation requires a legal action")
    return {
        "global_features": public.global_features.detach().cpu().numpy().astype(np.float32, copy=True),
        "rank_features": public.rank_features.detach().cpu().numpy().astype(np.float32, copy=True),
        "player_features": player_features,
        "player_mask": player_mask,
        "memory_trace_features": public.memory_trace_features.detach().cpu().numpy().astype(np.float32, copy=True),
        "history_features": history_features,
        "history_mask": output_history_mask,
        "legal_masks": legal,
    }


def _new_lane(spec: _MatchSpec, acts: int) -> _Lane:
    return _Lane(
        spec=spec,
        env=DalmutiScalarEnv(
            spec.player_count, acts=acts, seed=spec.seed, device="cpu"
        ),
    )


def _trajectory_id(
    namespace: str, spec: _MatchSpec, act: int, actor_id: int
) -> str:
    return (
        f"v4-dagger-{namespace}-p{spec.player_count}-m{spec.match_index + 1}"
        f"-seed{spec.seed:08x}-act{act}-actor{actor_id}"
    )


def _scope_counter(
    counters: dict[str, _RateCounter], key: str
) -> _RateCounter:
    if key not in counters:
        counters[key] = _RateCounter()
    return counters[key]


def _record_rate(
    counters: dict[str, _RateCounter],
    *,
    player_count: int,
    role: str,
    act: int,
    forced: bool,
    candidate_selected: bool,
    candidate_action: int,
    expert_action: int,
    behavior_action: int,
) -> None:
    keys = (
        "overall",
        f"playerCount:{player_count}",
        f"role:{role}",
        f"act:{act}",
        f"playerRoleAct:{player_count}|{role}|{act}",
    )
    for key in keys:
        _scope_counter(counters, key).add(
            forced=forced,
            candidate_selected=candidate_selected,
            candidate_action=candidate_action,
            expert_action=expert_action,
            behavior_action=behavior_action,
        )


def _rate_report(counters: Mapping[str, _RateCounter]) -> dict[str, object]:
    return {
        "overall": counters["overall"].to_dict(),
        "byPlayerCount": {
            key.split(":", 1)[1]: value.to_dict()
            for key, value in sorted(counters.items())
            if key.startswith("playerCount:")
        },
        "byRole": {
            key.split(":", 1)[1]: value.to_dict()
            for key, value in sorted(counters.items())
            if key.startswith("role:")
        },
        "byAct": {
            key.split(":", 1)[1]: value.to_dict()
            for key, value in sorted(counters.items())
            if key.startswith("act:")
        },
        "byPlayerRoleAct": {
            key.split(":", 1)[1]: value.to_dict()
            for key, value in sorted(counters.items())
            if key.startswith("playerRoleAct:")
        },
    }


def _trajectory_balance(trajectories: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scopes: dict[str, dict[str, int]] = {
        "byPlayerCount": {},
        "byRole": {},
        "byAct": {},
        "byPlayerRoleAct": {},
    }
    for trajectory in trajectories:
        samples = len(trajectory["rows"])
        values = {
            "byPlayerCount": str(trajectory["player_count"]),
            "byRole": str(trajectory["role"]),
            "byAct": str(trajectory["act"]),
            "byPlayerRoleAct": (
                f"{trajectory['player_count']}|{trajectory['role']}|{trajectory['act']}"
            ),
        }
        for scope, key in values.items():
            record = scopes[scope].setdefault(key, {"trajectories": 0, "samples": 0})
            record["trajectories"] += 1
            record["samples"] += samples
    return scopes


def _allocate_arrays(
    trajectory_count: int,
    max_steps: int,
    actor: V4ActorConfig,
    critic: V4CriticConfig,
) -> dict[str, np.ndarray]:
    prefix = (trajectory_count, max_steps)
    return {
        "global_features": np.zeros((*prefix, actor.global_features), np.float32),
        "rank_features": np.zeros((*prefix, actor.rank_tokens, actor.rank_features), np.float32),
        "player_features": np.zeros((*prefix, actor.max_players, actor.player_features), np.float32),
        "player_mask": np.zeros((*prefix, actor.max_players), np.bool_),
        "memory_trace_features": np.zeros((*prefix, actor.memory_tokens, actor.memory_features), np.float32),
        "history_features": np.zeros((*prefix, actor.max_history, actor.history_features), np.float32),
        "history_mask": np.zeros((*prefix, actor.max_history), np.bool_),
        "legal_masks": np.zeros((*prefix, ACTION_COUNT), np.bool_),
        "actions": np.zeros(prefix, np.int64),
        "expert_actions": np.zeros(prefix, np.int64),
        "old_action_log_probs": np.zeros(prefix, np.float32),
        "advantages": np.zeros(prefix, np.float32),
        "rewards": np.zeros(prefix, np.float32),
        "dones": np.zeros(prefix, np.bool_),
        "valid_masks": np.zeros(prefix, np.bool_),
        "privileged_states": np.zeros((*prefix, critic.privileged_features), np.float32),
        "candidate_actions": np.full(prefix, -1, np.int64),
        "behavior_sources": np.full(prefix, -1, np.int8),
        "forced_masks": np.zeros(prefix, np.bool_),
        "finish_places": np.zeros(prefix, np.int16),
        "environment_terminals": np.zeros(prefix, np.bool_),
        "source_decision_indices": np.full(prefix, -1, np.int64),
    }


def _build_arrays(
    trajectories: Sequence[Mapping[str, object]],
    actor: V4ActorConfig,
    critic: V4CriticConfig,
) -> tuple[dict[str, np.ndarray], V4TrajectoryDataset]:
    if not trajectories:
        raise RuntimeError("DAgger collection produced no trajectories")
    maximum = max(len(trajectory["rows"]) for trajectory in trajectories)
    arrays = _allocate_arrays(len(trajectories), maximum, actor, critic)
    public_names = (
        "global_features",
        "rank_features",
        "player_features",
        "player_mask",
        "memory_trace_features",
        "history_features",
        "history_mask",
        "legal_masks",
    )
    for trajectory_index, trajectory in enumerate(trajectories):
        rows = trajectory["rows"]
        if not rows or not bool(rows[-1]["done"]):
            raise RuntimeError("each actor trajectory requires a terminal final row")
        if sum(int(bool(row["done"])) for row in rows) != 1:
            raise RuntimeError("each actor trajectory requires exactly one terminal")
        for time_index, row in enumerate(rows):
            for name in public_names:
                arrays[name][trajectory_index, time_index] = row[name]
            for name in (
                "actions",
                "expert_actions",
                "old_action_log_probs",
                "advantages",
                "rewards",
                "dones",
                "privileged_states",
                "candidate_actions",
                "behavior_sources",
                "forced_masks",
                "finish_places",
                "environment_terminals",
                "source_decision_indices",
            ):
                arrays[name][trajectory_index, time_index] = row[name]
            arrays["valid_masks"][trajectory_index, time_index] = True
    standard_names = {field.name for field in fields(V4TrajectoryTensors)}
    boolean_names = {"player_mask", "history_mask", "legal_masks", "dones", "valid_masks"}
    integer_names = {"actions", "expert_actions"}
    tensors: dict[str, torch.Tensor] = {}
    for name in standard_names:
        tensor = torch.from_numpy(arrays[name])
        if name in boolean_names:
            tensor = tensor.to(dtype=torch.bool)
        elif name in integer_names:
            tensor = tensor.to(dtype=torch.long)
        else:
            tensor = tensor.to(dtype=torch.float32)
        tensors[name] = tensor
    dataset = V4TrajectoryDataset(
        V4TrajectoryTensors(**tensors), actor, critic
    )
    return arrays, dataset


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asanyarray(value), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.asanyarray(arrays[name])))
    return output.getvalue()


def _exclusive_publish(payloads: Mapping[Path, bytes]) -> None:
    if not payloads:
        raise ValueError("publish requires at least one file")
    directories = {path.parent.resolve() for path in payloads}
    if len(directories) != 1:
        raise ValueError("atomic DAgger outputs must share one directory")
    directory = next(iter(directories))
    directory.mkdir(parents=True, exist_ok=True)
    for path in payloads:
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    temporary: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for target, payload in payloads.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".partial", dir=directory
            )
            temp = Path(temporary_name)
            temporary[target] = temp
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for target, temp in temporary.items():
            os.link(temp, target)
            promoted.append(target)
        for temp in temporary.values():
            temp.unlink(missing_ok=True)
    except Exception:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)
        for target in promoted:
            target.unlink(missing_ok=True)
        raise


def _configure_determinism(device: torch.device) -> dict[str, object]:
    torch.use_deterministic_algorithms(True)
    cuda = device.type == "cuda"
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA collection requested but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return {
        "torchVersion": torch.__version__,
        "numpyVersion": np.__version__,
        "device": str(device),
        "cudaAvailable": torch.cuda.is_available(),
        "deterministicAlgorithms": True,
        "tf32Allowed": False if cuda else None,
    }


def collect_v4_dagger(
    bundle_directory: str | Path,
    output_path: str | Path,
    config: DaggerCollectionConfig,
    *,
    repository_root: str | Path | None = None,
) -> DaggerCollectionResult:
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("DAgger output must end in .npz")
    metadata_path = Path(f"{output}.metadata.json")
    checksum_path = Path(f"{output}.sha256")
    metadata_checksum_path = Path(f"{metadata_path}.sha256")
    for target in (output, metadata_path, checksum_path, metadata_checksum_path):
        if target.exists():
            raise FileExistsError(f"output already exists: {target}")

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parent.parent
    )
    bundle = Path(bundle_directory).resolve()
    manifest = verify_v4_actor_bundle(bundle)
    model, checkpoint_payload = load_v4_actor_checkpoint(bundle / "actor.pt")
    actor_config = getattr(model, "config", None)
    if not isinstance(actor_config, V4ActorConfig):
        raise ValueError("verified bundle did not load a V4 actor configuration")
    _validate_actor_contract(actor_config, config.player_counts)
    if checkpoint_payload.get("criticExcluded") is not True:
        raise ValueError("candidate actor checkpoint did not exclude the critic")
    device = torch.device(config.device)
    execution = _configure_determinism(device)
    model = model.to(device).eval()

    specs = [
        _MatchSpec(
            player_count=player_count,
            match_index=match_index,
            seed=_derive_uint32(
                config.run_namespace,
                config.seed_base,
                "environment",
                player_count,
                match_index,
            ),
        )
        for player_count in config.player_counts
        for match_index in range(config.matches_per_player_count)
    ]
    seeds = [spec.seed for spec in specs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("derived environment seed schedule contains a collision")
    next_spec = 0
    lanes: list[_Lane] = []
    while next_spec < len(specs) and len(lanes) < config.lane_count:
        lanes.append(_new_lane(specs[next_spec], config.acts))
        next_spec += 1

    trajectories_by_id: dict[str, dict[str, object]] = {}
    rate_counters: dict[str, _RateCounter] = {"overall": _RateCounter()}
    privacy_audits: dict[str, object] = {}
    total_decisions = 0
    while lanes:
        publics = [lane.env.public_observation() for lane in lanes]
        expert_actions = [lane.env.normal_action() for lane in lanes]
        candidate_actions = _batch_candidate_actions(model, publics, device)
        if len(candidate_actions) != len(lanes):
            raise RuntimeError("candidate action batch length mismatch")
        replacements: list[_Lane] = []
        for lane, public, expert_action, candidate_action in zip(
            lanes, publics, expert_actions, candidate_actions, strict=True
        ):
            env = lane.env
            spec = lane.spec
            act = int(env._act)
            actor_id = env.current_player_id
            actor_position = env._order.index(actor_id)
            role = role_for_index(actor_position, spec.player_count)
            legal = public.legal_mask
            if not bool(legal[expert_action].item()):
                raise RuntimeError("exact Normal expert selected an illegal action")
            if not 0 <= candidate_action < ACTION_COUNT or not bool(
                legal[candidate_action].item()
            ):
                raise RuntimeError("candidate selected an illegal action")
            if str(spec.player_count) not in privacy_audits:
                audit_seed = _derive_uint32(
                    config.run_namespace,
                    config.seed_base,
                    "privacy-audit",
                    spec.player_count,
                )
                privacy_audits[str(spec.player_count)] = audit_hidden_state_privacy(
                    env, audit_seed
                )
            mixture = _mixture_uniform(
                config.run_namespace,
                config.seed_base,
                spec,
                act,
                actor_id,
                lane.decision_index,
            )
            candidate_selected = mixture < config.candidate_beta
            behavior_action = candidate_action if candidate_selected else expert_action
            if candidate_action == expert_action:
                old_log_probability = 0.0
            else:
                probability = (
                    config.candidate_beta
                    if candidate_selected
                    else 1.0 - config.candidate_beta
                )
                if probability <= 0.0:
                    raise RuntimeError("mixture selected a zero-probability branch")
                old_log_probability = math.log(probability)
            forced = int(legal.sum().item()) == 1
            identifier = _trajectory_id(
                config.run_namespace, spec, act, actor_id
            )
            trajectory = trajectories_by_id.setdefault(
                identifier,
                {
                    "id": identifier,
                    "player_count": spec.player_count,
                    "match_index": spec.match_index,
                    "match_seed": spec.seed,
                    "act": act,
                    "actor_id": actor_id,
                    "role": role,
                    "rows": [],
                    "finish_place": None,
                },
            )
            if trajectory["role"] != role or trajectory["act"] != act:
                raise RuntimeError("actor trajectory identity changed within an act")
            snapshot = _snapshot_public(public, actor_config)
            row: dict[str, object] = {
                **snapshot,
                "actions": behavior_action,
                "expert_actions": expert_action,
                "old_action_log_probs": old_log_probability,
                "advantages": 0.0,
                "rewards": 0.0,
                "dones": False,
                "privileged_states": env.privileged_state().detach().cpu().numpy().astype(np.float32, copy=True),
                "candidate_actions": candidate_action,
                "behavior_sources": 1 if candidate_selected else 0,
                "forced_masks": forced,
                "finish_places": 0,
                "environment_terminals": False,
                "source_decision_indices": lane.decision_index,
                "done": False,
            }
            trajectory["rows"].append(row)
            _record_rate(
                rate_counters,
                player_count=spec.player_count,
                role=role,
                act=act,
                forced=forced,
                candidate_selected=candidate_selected,
                candidate_action=candidate_action,
                expert_action=expert_action,
                behavior_action=behavior_action,
            )
            lane.decision_index += 1
            total_decisions += 1
            result = env.step(behavior_action)
            if result.act_ended:
                act_result = result.info.get("act_result")
                if not isinstance(act_result, Mapping):
                    raise RuntimeError("act terminal is missing its exact result")
                finish_order = tuple(int(value) for value in act_result["finish_order"])
                if len(finish_order) != spec.player_count:
                    raise RuntimeError("act terminal finish order is incomplete")
                for finish_index, finished_actor in enumerate(finish_order, start=1):
                    terminal_id = _trajectory_id(
                        config.run_namespace, spec, act, finished_actor
                    )
                    if terminal_id not in trajectories_by_id:
                        raise RuntimeError("finished actor has no collected decisions")
                    terminal_trajectory = trajectories_by_id[terminal_id]
                    terminal_rows = terminal_trajectory["rows"]
                    if not terminal_rows or terminal_rows[-1]["done"]:
                        raise RuntimeError("actor trajectory terminal is duplicated")
                    reward = float(result.rewards[finished_actor].item())
                    terminal_rows[-1]["rewards"] = reward
                    terminal_rows[-1]["dones"] = True
                    terminal_rows[-1]["done"] = True
                    terminal_rows[-1]["finish_places"] = finish_index
                    terminal_trajectory["finish_place"] = finish_index
                if result.terminated:
                    row["environment_terminals"] = True
            if result.terminated:
                if next_spec < len(specs):
                    replacements.append(_new_lane(specs[next_spec], config.acts))
                    next_spec += 1
            else:
                replacements.append(lane)
        lanes = replacements

    trajectories = [trajectories_by_id[key] for key in sorted(trajectories_by_id)]
    expected_trajectories = sum(
        spec.player_count * config.acts for spec in specs
    )
    if len(trajectories) != expected_trajectories:
        raise RuntimeError(
            f"expected {expected_trajectories} actor-act trajectories, got {len(trajectories)}"
        )
    if any(trajectory["finish_place"] is None for trajectory in trajectories):
        raise RuntimeError("collection contains an unterminated actor trajectory")

    critic_config = V4CriticConfig(privileged_features=PRIVILEGED_STATE_SIZE)
    arrays, dataset = _build_arrays(trajectories, actor_config, critic_config)
    trajectory_ids = np.asarray(
        [str(trajectory["id"]) for trajectory in trajectories], dtype=np.str_
    )
    arrays.update(
        {
            "trajectory_ids": trajectory_ids,
            "trajectory_player_counts": np.asarray(
                [trajectory["player_count"] for trajectory in trajectories], np.int16
            ),
            "trajectory_roles": np.asarray(
                [ROLES.index(str(trajectory["role"])) for trajectory in trajectories],
                np.int8,
            ),
            "trajectory_acts": np.asarray(
                [trajectory["act"] for trajectory in trajectories], np.int16
            ),
            "trajectory_actor_ids": np.asarray(
                [trajectory["actor_id"] for trajectory in trajectories], np.int16
            ),
            "trajectory_match_indices": np.asarray(
                [trajectory["match_index"] for trajectory in trajectories], np.int32
            ),
            "trajectory_match_seeds": np.asarray(
                [trajectory["match_seed"] for trajectory in trajectories], np.uint32
            ),
        }
    )
    source_hashes = _source_hashes(root)
    manifest_path = bundle / "manifest.json"
    actor_path = bundle / "actor.pt"
    metadata: dict[str, object] = {
        "format": V4_DATASET_FORMAT,
        "version": V4_DATASET_VERSION,
        "preparationFormat": DAGGER_PREPARATION_FORMAT,
        "preparationVersion": DAGGER_PREPARATION_VERSION,
        "fingerprint": dataset.fingerprint,
        "actorConfig": actor_config.to_dict(),
        "criticConfig": critic_config.to_dict(),
        "collection": {
            "algorithm": "DAgger",
            "expert": "exact-v4-env-Normal",
            "behavior": "deterministic keyed mixture of candidate greedy and exact Normal",
            "candidateBeta": config.candidate_beta,
            "candidateBetaMeaning": "probability of selecting candidate greedy behavior",
            "allActorsUseSameMixtureContract": True,
            "actsPerMatch": config.acts,
            "matchesPerPlayerCount": config.matches_per_player_count,
            "playerCounts": list(config.player_counts),
            "rollingCpuEnvironmentLanes": min(config.lane_count, len(specs)),
            "batchedCandidateInference": True,
            "expertLabelForEveryDecision": True,
            "advantages": "zero; DAgger data is intended for supervised BC",
            "reward": "zero except actor trajectory terminal; (round chip award - 2) / 2",
            "done": "exactly one, on the final decision of each actor-act trajectory",
        },
        "shard": {
            "runNamespace": config.run_namespace,
            "seedBase": config.seed_base,
            "identitySha256": _sha256_bytes(
                canonical_json_bytes(
                    [
                        DAGGER_PREPARATION_FORMAT,
                        config.run_namespace,
                        config.seed_base,
                        list(config.player_counts),
                        config.matches_per_player_count,
                        config.acts,
                    ]
                )
            ),
            "environmentSeeds": [spec.seed for spec in specs],
            "trajectoryIdsIncludeNamespaceAndDerivedSeed": True,
        },
        "modelBinding": {
            "bundleManifestSha256": sha256_file(manifest_path),
            "actorCheckpointSha256": sha256_file(actor_path),
            "manifestFormat": manifest.get("format"),
            "manifestVersion": manifest.get("version"),
            "modelKind": manifest.get("model", {}).get("kind"),
            "criticExcluded": True,
        },
        "environmentBinding": {
            "implementation": "DalmutiScalarEnv",
            "normalExpertCallback": "DalmutiScalarEnv.normal_action",
            "v4EnvSha256": source_hashes["gpu-training/v4_env.py"],
            "cpuStepping": True,
        },
        "privilegedCriticLayout": {
            "id": PRIVILEGED_STATE_LAYOUT_ID,
            "sha256": PRIVILEGED_STATE_LAYOUT_SHA256,
            "layout": PRIVILEGED_STATE_LAYOUT,
            "featureCount": PRIVILEGED_STATE_SIZE,
            "matchesTypescriptNormalContract": True,
        },
        "sourceHashes": source_hashes,
        "execution": execution,
        "privacy": {
            "actorPublicOnly": True,
            "actorInputFields": list(PUBLIC_MODEL_INPUT_FIELDS),
            "opponentPhysicalHandsExcluded": True,
            "taxCardIdentitiesExcluded": True,
            "privilegedCriticStateSeparate": True,
            "privilegedCriticExportAllowed": False,
            "perPlayerCountAudits": privacy_audits,
        },
        "trajectoryCount": len(trajectories),
        "sampleCount": total_decisions,
        "maxTimeSteps": int(arrays["actions"].shape[1]),
        "balance": _trajectory_balance(trajectories),
        "changedActionRates": _rate_report(rate_counters),
        "auxiliaryArrays": [
            "candidate_actions",
            "behavior_sources",
            "forced_masks",
            "finish_places",
            "environment_terminals",
            "source_decision_indices",
            "trajectory_ids",
            "trajectory_player_counts",
            "trajectory_roles",
            "trajectory_acts",
            "trajectory_actor_ids",
            "trajectory_match_indices",
            "trajectory_match_seeds",
        ],
        "padding": "zero-valued invalid suffix after the sole actor terminal; auxiliary sentinel values are -1 where documented",
    }
    arrays["metadata_json"] = np.asarray(_canonical_text(metadata))
    npz_bytes = _deterministic_npz_bytes(arrays)
    npz_sha256 = _sha256_bytes(npz_bytes)
    external_metadata = dict(metadata)
    external_metadata["npzSha256"] = npz_sha256
    external_metadata_bytes = (_canonical_text(external_metadata) + "\n").encode("utf-8")
    metadata_sha256 = _sha256_bytes(external_metadata_bytes)
    checksum_bytes = f"{npz_sha256}  {output.name}\n".encode("ascii")
    metadata_checksum_bytes = (
        f"{metadata_sha256}  {metadata_path.name}\n".encode("ascii")
    )
    _exclusive_publish(
        {
            output: npz_bytes,
            metadata_path: external_metadata_bytes,
            checksum_path: checksum_bytes,
            metadata_checksum_path: metadata_checksum_bytes,
        }
    )
    return DaggerCollectionResult(
        output_path=output,
        metadata_path=metadata_path,
        checksum_path=checksum_path,
        metadata_checksum_path=metadata_checksum_path,
        npz_sha256=npz_sha256,
        metadata_sha256=metadata_sha256,
        fingerprint=dataset.fingerprint,
        trajectories=len(trajectories),
        samples=total_decisions,
    )


def _parse_player_counts(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("player counts must be comma-separated integers") from error
    if not parsed or any(not 4 <= item <= 10 for item in parsed):
        raise argparse.ArgumentTypeError("player counts must be from 4 through 10")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a deterministic public-only V4 DAgger trajectory NPZ."
    )
    parser.add_argument("--actor-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-namespace", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--player-counts", type=_parse_player_counts, default=tuple(range(4, 11)))
    parser.add_argument("--matches-per-player-count", type=int, default=1)
    parser.add_argument("--acts", type=int, default=5)
    parser.add_argument("--candidate-beta", type=float, default=0.5)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = DaggerCollectionConfig(
        run_namespace=arguments.run_namespace,
        seed_base=arguments.seed_base,
        player_counts=arguments.player_counts,
        matches_per_player_count=arguments.matches_per_player_count,
        acts=arguments.acts,
        candidate_beta=arguments.candidate_beta,
        lane_count=arguments.lanes,
        device=arguments.device,
    )
    result = collect_v4_dagger(
        arguments.actor_bundle,
        arguments.output,
        config,
        repository_root=arguments.repository_root,
    )
    print(
        _canonical_text(
            {
                "output": str(result.output_path),
                "npzSha256": result.npz_sha256,
                "metadataSha256": result.metadata_sha256,
                "fingerprint": result.fingerprint,
                "trajectories": result.trajectories,
                "samples": result.samples,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DAGGER_PREPARATION_FORMAT",
    "DAGGER_PREPARATION_VERSION",
    "DaggerCollectionConfig",
    "DaggerCollectionResult",
    "audit_hidden_state_privacy",
    "collect_v4_dagger",
    "main",
]
