from __future__ import annotations

"""Leakage-safe V6 Normal/MC warm-start pretrainer.

The V6 warm start deliberately leaves the deployed residual policy at the
exact V5 Normal prior.  It first fits the centralized critic to undiscounted
match Monte-Carlo returns, then pretrains the Actor's dense Normal auxiliary
head and shared public encoder.  The residual output is frozen and reset to
bit-exact zero before and after Actor training, so the published Actor remains
greedy-identical to Normal while providing a substantially better starting
representation for the next on-policy iteration.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from v5_dataset import (
    V5TrainingShard,
    load_v5_index_manifest,
    load_v5_training_shard,
)
from v5_export import (
    canonical_json_bytes,
    export_v5_actor_bundle,
    load_v5_actor_bundle,
    sha256_file,
    v5_actor_bundle_digests,
)
from v5_model import (
    V5CentralStateValueCritic,
    V5PublicActor,
    assert_actor_critic_parameter_isolation,
    configure_v5_policy_numerics,
)
from v5_public import actor_batch_from_packed_arrays
from v5_train import (
    export_v5_critic_checkpoint,
    load_v5_critic_checkpoint,
    publish_v5_model_pair_manifest,
    verify_v5_model_pair,
)
from v6_targets import (
    V6_MC_RETURN_CONTRACT,
    V6_NORMAL_CLASS_WEIGHT_CONTRACT,
    balanced_normal_action_weights,
    compute_v6_monte_carlo_returns,
)


V6_SPLIT_FORMAT = "dalmuti-v6-match-disjoint-split"
V6_SPLIT_VERSION = 1
V6_PRETRAIN_FORMAT = "dalmuti-v6-normal-mc-warm-start"
V6_PRETRAIN_VERSION = 1
PLAYER_COUNTS = tuple(range(4, 11))
SPLIT_NAMES = ("train", "validation", "test")
_SHA256 = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - _SHA256)
    )


def _strict_canonical_object(
    path: Path,
    label: str,
    *,
    producer: str,
) -> dict[str, object]:
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid ASCII JSON") from error
    if producer == "v6-split-builder":
        expected = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
    elif producer == "v5-dataset-index":
        # v5_dataset._canonical_json is the authority for V5 index files.  A
        # raw non-ASCII byte was already rejected above; ensure_ascii=False is
        # still important because an escaped Unicode spelling is not what the
        # producer would have emitted.
        expected = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    else:
        raise ValueError("unknown canonical JSON producer")
    if not isinstance(value, dict) or raw != expected:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _strict_split_manifest(path: Path) -> dict[str, object]:
    return _strict_canonical_object(
        path, "V6 split manifest", producer="v6-split-builder"
    )


def _strict_dataset_index(path: Path) -> dict[str, object]:
    return _strict_canonical_object(
        path, "dataset index", producer="v5-dataset-index"
    )


def _verify_json_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if path.with_name(path.name + ".sha256").read_bytes() != expected:
        raise ValueError(f"checksum sidecar does not match {path.name}")
    return digest


@dataclass(frozen=True)
class V6PretrainConfig:
    seed: int = 860_100_001
    train_fraction: float = 1.0
    critic_epochs: int = 6
    actor_epochs: int = 6
    critic_batch_size: int = 512
    actor_batch_size: int = 16
    actor_gradient_accumulation: int = 2
    validation_batch_size: int = 64
    critic_learning_rate: float = 3.0e-4
    actor_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    huber_delta: float = 1.0
    class_weight_exponent: float = 0.5
    maximum_class_weight_ratio: float = 10.0
    maximum_gradient_norm: float = 1.0
    use_amp: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFF_FFFF
        ):
            raise ValueError("seed must be an explicit uint32")
        if (
            isinstance(self.train_fraction, bool)
            or not math.isfinite(float(self.train_fraction))
            or not 0.0 < float(self.train_fraction) <= 1.0
        ):
            raise ValueError("train_fraction must be finite in (0,1]")
        for name, maximum in (
            ("critic_epochs", 1000),
            ("actor_epochs", 1000),
            ("critic_batch_size", 65_536),
            ("actor_batch_size", 4096),
            ("actor_gradient_accumulation", 4096),
            ("validation_batch_size", 65_536),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{name} must be an integer in [1,{maximum}]")
        for name in (
            "critic_learning_rate",
            "actor_learning_rate",
            "huber_delta",
            "maximum_gradient_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            not math.isfinite(float(self.class_weight_exponent))
            or not 0.0 <= float(self.class_weight_exponent) <= 1.0
        ):
            raise ValueError("class_weight_exponent must be finite in [0,1]")
        if (
            not math.isfinite(float(self.maximum_class_weight_ratio))
            or float(self.maximum_class_weight_ratio) < 1.0
        ):
            raise ValueError("maximum_class_weight_ratio must be at least one")
        if type(self.use_amp) is not bool:
            raise ValueError("use_amp must be an exact bool")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V6MatchRecord:
    split: str
    player_count: int
    split_hash: str
    shard_manifest_sha256: str
    local_match_index: int
    decision_start: int
    decision_end: int
    match_index: int
    match_seed: int
    nonforced_decision_count: int

    @property
    def decision_count(self) -> int:
        return self.decision_end - self.decision_start


@dataclass(frozen=True)
class V6MatchView:
    split: str
    matches: tuple[V6MatchRecord, ...]

    @property
    def match_count(self) -> int:
        return len(self.matches)

    @property
    def decision_count(self) -> int:
        return sum(record.decision_count for record in self.matches)

    @property
    def nonforced_decision_count(self) -> int:
        return sum(record.nonforced_decision_count for record in self.matches)

    def rows_by_shard(
        self,
        *,
        nonforced_only: bool,
        shards: Mapping[str, V5TrainingShard],
    ) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = {}
        for record in self.matches:
            rows = np.arange(record.decision_start, record.decision_end, dtype=np.int64)
            if nonforced_only:
                forced = np.asarray(
                    shards[record.shard_manifest_sha256].actor.arrays["forced"][rows],
                    dtype=np.bool_,
                )
                rows = rows[~forced]
            grouped.setdefault(record.shard_manifest_sha256, []).append(rows)
        return {
            digest: np.concatenate(parts) if parts else np.empty(0, np.int64)
            for digest, parts in grouped.items()
        }

    def summary(self) -> dict[str, object]:
        per_player: dict[str, object] = {}
        for player_count in PLAYER_COUNTS:
            selected = [
                record for record in self.matches
                if record.player_count == player_count
            ]
            per_player[str(player_count)] = {
                "matches": len(selected),
                "decisions": sum(record.decision_count for record in selected),
                "nonforcedDecisions": sum(
                    record.nonforced_decision_count for record in selected
                ),
            }
        return {
            "matches": self.match_count,
            "decisions": self.decision_count,
            "nonforcedDecisions": self.nonforced_decision_count,
            "perPlayerCount": per_player,
        }


@dataclass
class V6SplitDataset:
    index_root: Path
    index_manifest_sha256: str
    split_manifest_sha256: str
    corpus_identity_sha256: str
    shards: dict[str, V5TrainingShard]
    views: dict[str, V6MatchView]

    def close(self) -> None:
        for shard in self.shards.values():
            shard.close()

    def __enter__(self) -> "V6SplitDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _pilot_matches(
    records: Sequence[V6MatchRecord], fraction: float
) -> tuple[V6MatchRecord, ...]:
    """Select complete SHA-ordered matches, with at least one per p stratum."""

    selected: list[V6MatchRecord] = []
    for player_count in PLAYER_COUNTS:
        stratum = sorted(
            (record for record in records if record.player_count == player_count),
            key=lambda record: (
                record.split_hash,
                record.match_index,
                record.match_seed,
            ),
        )
        if not stratum:
            raise ValueError("every V6 view must contain p4..p10")
        take = len(stratum) if fraction == 1.0 else max(
            1, int(math.ceil(len(stratum) * fraction))
        )
        selected.extend(stratum[:take])
    selected.sort(
        key=lambda record: (
            record.player_count,
            record.split_hash,
            record.match_index,
        )
    )
    return tuple(selected)


def _record_from_manifest(raw: object, split: str) -> V6MatchRecord:
    expected = {
        "decisionEnd",
        "decisionStart",
        "decisionCount",
        "localMatchIndex",
        "matchIndex",
        "matchSeed",
        "nonforcedDecisionCount",
        "playerCount",
        "shardManifestSha256",
        "shardOrdinal",
        "shardRelativePath",
        "splitHash",
        "split",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("V6 split match record fields drifted")
    if raw.get("split") != split:
        raise ValueError("V6 split record is stored under the wrong view")
    digest = raw.get("shardManifestSha256")
    split_hash = raw.get("splitHash")
    if not _is_sha256(digest) or not _is_sha256(split_hash):
        raise ValueError("V6 split record contains a malformed SHA-256")

    integer_names = (
        "decisionEnd",
        "decisionStart",
        "decisionCount",
        "localMatchIndex",
        "matchIndex",
        "matchSeed",
        "nonforcedDecisionCount",
        "playerCount",
        "shardOrdinal",
    )
    for name in integer_names:
        if isinstance(raw.get(name), bool) or not isinstance(raw.get(name), int):
            raise ValueError(f"V6 split record {name} must be an integer")
    if int(raw["shardOrdinal"]) < 0 or not isinstance(
        raw.get("shardRelativePath"), str
    ):
        raise ValueError("V6 split record historical shard locator is malformed")
    start = int(raw["decisionStart"])
    end = int(raw["decisionEnd"])
    decision_count = int(raw["decisionCount"])
    nonforced = int(raw["nonforcedDecisionCount"])
    player_count = int(raw["playerCount"])
    if (
        start < 0
        or end <= start
        or end - start != decision_count
        or not 0 <= nonforced <= decision_count
        or player_count not in PLAYER_COUNTS
        or int(raw["localMatchIndex"]) < 0
        or int(raw["matchIndex"]) < 0
        or not 0 <= int(raw["matchSeed"]) <= 0xFFFF_FFFF
    ):
        raise ValueError("V6 split match record scalar range is invalid")
    return V6MatchRecord(
        split=split,
        player_count=player_count,
        split_hash=str(split_hash),
        shard_manifest_sha256=str(digest),
        local_match_index=int(raw["localMatchIndex"]),
        decision_start=start,
        decision_end=end,
        match_index=int(raw["matchIndex"]),
        match_seed=int(raw["matchSeed"]),
        nonforced_decision_count=nonforced,
    )


def _validate_record_against_shard(
    record: V6MatchRecord, shard: V5TrainingShard
) -> None:
    arrays = shard.actor.arrays
    private = shard.privileged_arrays
    local = record.local_match_index
    offsets = np.asarray(arrays["match_offsets"], dtype=np.int64)
    if local >= shard.actor.match_count:
        raise ValueError("V6 local match index escaped its remapped shard")
    if (
        int(offsets[local]) != record.decision_start
        or int(offsets[local + 1]) != record.decision_end
        or int(arrays["player_counts"][local]) != record.player_count
        or int(private["match_indices"][local]) != record.match_index
        or int(private["match_seeds"][local]) != record.match_seed
    ):
        raise ValueError("V6 split coordinate disagrees with its SHA-remapped shard")
    forced = np.asarray(
        arrays["forced"][record.decision_start : record.decision_end],
        dtype=np.bool_,
    )
    if int((~forced).sum()) != record.nonforced_decision_count:
        raise ValueError("V6 split nonforced count disagrees with its shard")


def load_v6_split_dataset(
    dataset_index: str | Path,
    split_manifest: str | Path,
    *,
    train_fraction: float = 1.0,
) -> V6SplitDataset:
    """Open a portable split, remapping every shard solely by manifest SHA."""

    if (
        isinstance(train_fraction, bool)
        or not math.isfinite(float(train_fraction))
        or not 0.0 < float(train_fraction) <= 1.0
    ):
        raise ValueError("train_fraction must be finite in (0,1]")
    index_root = Path(dataset_index).resolve(strict=True)
    split_path = Path(split_manifest).resolve(strict=True)
    split_value = _strict_split_manifest(split_path)
    split_sha = _verify_json_sidecar(split_path)
    if (
        split_value.get("format") != V6_SPLIT_FORMAT
        or split_value.get("version") != V6_SPLIT_VERSION
    ):
        raise ValueError("unsupported V6 split manifest")
    corpus_sha = split_value.get("corpusIdentitySha256")
    if not _is_sha256(corpus_sha):
        raise ValueError("V6 split corpus identity is malformed")

    index_manifest_path = index_root / "manifest.json"
    index_sha = _verify_json_sidecar(index_manifest_path)
    index_value = _strict_dataset_index(index_manifest_path)
    metadata = index_value.get("metadata")
    portable_source_sha = (
        metadata.get("sourceIndexManifestSha256")
        if isinstance(metadata, dict)
        else None
    )
    if corpus_sha not in (index_sha, portable_source_sha):
        raise ValueError(
            "V6 split corpus SHA is not bound to this index or its portable source"
        )

    index = load_v5_index_manifest(index_root)
    path_by_sha: dict[str, Path] = {}
    for path in index.shard_paths:
        digest = sha256_file(path / "manifest.json")
        if digest in path_by_sha:
            raise ValueError("dataset index has duplicate shard manifest identities")
        path_by_sha[digest] = path

    raw_splits = split_value.get("splits")
    if not isinstance(raw_splits, dict) or set(raw_splits) != set(SPLIT_NAMES):
        raise ValueError("V6 split views are missing or unknown")
    records_by_split: dict[str, list[V6MatchRecord]] = {}
    required_shards: set[str] = set()
    coordinates: set[tuple[int, int]] = set()
    seeds: set[int] = set()
    for split in SPLIT_NAMES:
        values = raw_splits[split]
        if not isinstance(values, list) or not values:
            raise ValueError(f"V6 {split} view is empty")
        parsed = [_record_from_manifest(raw, split) for raw in values]
        for record in parsed:
            coordinate = (record.player_count, record.match_index)
            if coordinate in coordinates or record.match_seed in seeds:
                raise ValueError("V6 views overlap match coordinates or seeds")
            coordinates.add(coordinate)
            seeds.add(record.match_seed)
            required_shards.add(record.shard_manifest_sha256)
        records_by_split[split] = parsed
    missing = required_shards - set(path_by_sha)
    if missing:
        raise ValueError(
            "portable V6 split cannot remap shard manifest SHA " + sorted(missing)[0]
        )

    shards: dict[str, V5TrainingShard] = {}
    try:
        for digest in sorted(required_shards):
            shards[digest] = load_v5_training_shard(path_by_sha[digest])
        for records in records_by_split.values():
            for record in records:
                _validate_record_against_shard(
                    record, shards[record.shard_manifest_sha256]
                )
        # Pilot fraction limits optimization data only.  Validation remains the
        # complete match-disjoint split so the published parity and progress
        # metrics never become sample-only claims; test stays wholly reserved.
        views = {
            "train": V6MatchView(
                "train",
                _pilot_matches(records_by_split["train"], float(train_fraction)),
            ),
            "validation": V6MatchView(
                "validation",
                _pilot_matches(records_by_split["validation"], 1.0),
            ),
            "test": V6MatchView(
                "test",
                _pilot_matches(records_by_split["test"], 1.0),
            ),
        }
        return V6SplitDataset(
            index_root=index_root,
            index_manifest_sha256=index_sha,
            split_manifest_sha256=split_sha,
            corpus_identity_sha256=str(corpus_sha),
            shards=shards,
            views=views,
        )
    except Exception:
        for shard in shards.values():
            shard.close()
        raise


def _match_mc_returns(
    record: V6MatchRecord, shard: V5TrainingShard
) -> np.ndarray:
    arrays = shard.actor.arrays
    start, stop = record.decision_start, record.decision_end
    successors = np.asarray(arrays["next_decision"][start:stop], dtype=np.int64).copy()
    successors[successors >= 0] -= start
    return compute_v6_monte_carlo_returns(
        reward_to_next=arrays["reward_to_next"][start:stop],
        next_decision=successors.astype(np.int32),
        done=arrays["done"][start:stop],
        match_offsets=np.asarray([0, stop - start], dtype=np.uint32),
        decision_actor_ids=arrays["decision_actor_ids"][start:stop],
        player_counts=np.asarray([record.player_count], dtype=np.uint8),
        candidate_bitsets=np.asarray(
            [arrays["candidate_bitsets"][record.local_match_index]],
            dtype=np.uint16,
        ),
    )


def monte_carlo_targets_by_shard(
    view: V6MatchView, shards: Mapping[str, V5TrainingShard]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return selected row ids and exact gamma=1 targets for a match view."""

    rows: dict[str, list[np.ndarray]] = {}
    values: dict[str, list[np.ndarray]] = {}
    for record in view.matches:
        digest = record.shard_manifest_sha256
        rows.setdefault(digest, []).append(
            np.arange(record.decision_start, record.decision_end, dtype=np.int64)
        )
        values.setdefault(digest, []).append(_match_mc_returns(record, shards[digest]))
    return {
        digest: (
            np.concatenate(rows[digest]),
            np.concatenate(values[digest]).astype(np.float32, copy=False),
        )
        for digest in sorted(rows)
    }


def _row_player_counts(
    shard: V5TrainingShard, indices: np.ndarray
) -> np.ndarray:
    return np.asarray(
        shard.actor.arrays["global_codes"][indices, 1], dtype=np.int64
    )


def _equal_p_weights(player_counts: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    represented = tuple(
        player for player in PLAYER_COUNTS if bool((player_counts == player).any())
    )
    if represented != PLAYER_COUNTS:
        raise ValueError("training view must represent every p4..p10 stratum")
    total = int(player_counts.size)
    weights = np.zeros(player_counts.shape, dtype=np.float32)
    counts: dict[str, int] = {}
    for player in represented:
        count = int((player_counts == player).sum())
        counts[str(player)] = count
        weights[player_counts == player] = np.float32(total / (len(represented) * count))
    masses = {
        str(player): float(weights[player_counts == player].sum(dtype=np.float64))
        for player in represented
    }
    return weights, {
        "contract": "dalmuti-v6-equal-player-count-loss-mass-v1",
        "counts": counts,
        "massPerPlayerCount": masses,
        "meanRowWeight": float(weights.mean(dtype=np.float64)),
    }


def _actor_training_weights(
    labels: np.ndarray,
    player_counts: np.ndarray,
    *,
    exponent: float,
    maximum_ratio: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Class-balance within each p, then assign exactly equal aggregate p mass."""

    if labels.shape != player_counts.shape or labels.ndim != 1:
        raise ValueError("Actor labels and player counts must be aligned vectors")
    result = np.zeros(labels.shape, dtype=np.float32)
    reports: dict[str, object] = {}
    total = len(labels)
    for player in PLAYER_COUNTS:
        selected = player_counts == player
        if not bool(selected.any()):
            raise ValueError("Actor training needs eligible rows for every p4..p10")
        local, report = balanced_normal_action_weights(
            labels[selected],
            np.ones(int(selected.sum()), dtype=np.bool_),
            exponent=exponent,
            maximum_ratio=maximum_ratio,
        )
        p_scale = total / (len(PLAYER_COUNTS) * int(selected.sum()))
        result[selected] = local * np.float32(p_scale)
        reports[str(player)] = report
    return result, {
        "classContract": V6_NORMAL_CLASS_WEIGHT_CONTRACT,
        "method": "bounded-class-balance-within-p-then-equal-p-mass",
        "maximumClassWeightRatio": maximum_ratio,
        "perPlayerCount": reports,
        "massPerPlayerCount": {
            str(player): float(result[player_counts == player].sum(dtype=np.float64))
            for player in PLAYER_COUNTS
        },
        "meanRowWeight": float(result.mean(dtype=np.float64)),
    }


def freeze_and_zero_residual(actor: V5PublicActor) -> dict[str, object]:
    with torch.no_grad():
        actor.residual_output.weight.zero_()
        actor.residual_output.bias.zero_()
    for parameter in actor.residual_output.parameters():
        parameter.requires_grad_(False)
    return residual_zero_receipt(actor)


def residual_zero_receipt(actor: V5PublicActor) -> dict[str, object]:
    digest = hashlib.sha256()
    exact = True
    frozen = True
    tensors: dict[str, object] = {}
    for name, tensor in actor.residual_output.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        raw = value.numpy().tobytes(order="C")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(raw)
        is_zero = bool(torch.equal(value, torch.zeros_like(value)))
        exact = exact and is_zero
        tensors[name] = {"shape": list(value.shape), "exactZero": is_zero}
    for parameter in actor.residual_output.parameters():
        frozen = frozen and not parameter.requires_grad
    return {
        "contract": "dalmuti-v6-frozen-exact-zero-residual-output-v1",
        "exactZero": exact,
        "frozen": frozen,
        "sha256": digest.hexdigest(),
        "tensors": tensors,
    }


def _autocast(device: torch.device, enabled: bool):  # type: ignore[no-untyped-def]
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_scaler(enabled: bool):  # type: ignore[no-untyped-def]
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shuffled_batches(
    rows_by_shard: Mapping[str, np.ndarray],
    batch_size: int,
    seed: int,
) -> Iterator[tuple[str, np.ndarray]]:
    generator = np.random.default_rng(seed)
    digests = sorted(rows_by_shard)
    for shard_position in generator.permutation(len(digests)):
        digest = digests[int(shard_position)]
        rows = rows_by_shard[digest]
        order = generator.permutation(len(rows))
        for start in range(0, len(rows), batch_size):
            yield digest, rows[order[start : start + batch_size]]


def _flat_row_metadata(
    rows_by_shard: Mapping[str, np.ndarray],
    shards: Mapping[str, V5TrainingShard],
    array_name: str,
    dtype: np.dtype,
) -> np.ndarray:
    values = [
        np.asarray(
            shards[digest].actor.arrays[array_name][rows_by_shard[digest]],
            dtype=dtype,
        )
        for digest in sorted(rows_by_shard)
    ]
    if not values:
        raise ValueError("V6 row metadata cannot be empty")
    return np.concatenate(values, axis=0)


def _dense_lookup_by_shard(
    rows_by_shard: Mapping[str, np.ndarray],
    shards: Mapping[str, V5TrainingShard],
    flat_values: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """Map aligned flat values back to local rows using compact NumPy arrays."""

    output: dict[str, np.ndarray] = {}
    offset = 0
    for digest in sorted(rows_by_shard):
        rows = rows_by_shard[digest]
        stop = offset + len(rows)
        lookup = np.full(
            shards[digest].actor.decision_count,
            fill_value,
            dtype=flat_values.dtype,
        )
        lookup[rows] = flat_values[offset:stop]
        output[digest] = lookup
        offset = stop
    if offset != len(flat_values):
        raise RuntimeError("flat V6 row metadata did not map back exactly")
    return output


def _train_critic(
    critic: V5CentralStateValueCritic,
    dataset: V6SplitDataset,
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    config: V6PretrainConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    rows_by_shard = {digest: rows for digest, (rows, _) in targets.items()}
    all_players = _flat_row_metadata(
        rows_by_shard, dataset.shards, "global_codes", np.int32
    )
    # global_codes is rank two, so take its player-count column explicitly.
    if all_players.ndim != 2 or all_players.shape[1] < 2:
        raise ValueError("global_codes metadata is malformed")
    player_counts = all_players[:, 1].astype(np.int64, copy=False)
    weights, weight_report = _equal_p_weights(player_counts)
    weights_by_shard = _dense_lookup_by_shard(
        rows_by_shard, dataset.shards, weights
    )
    target_by_shard = {
        digest: np.full(
            dataset.shards[digest].actor.decision_count,
            np.nan,
            dtype=np.float32,
        )
        for digest in rows_by_shard
    }
    for digest, (rows, values) in targets.items():
        target_by_shard[digest][rows] = values
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        weight_decay=config.weight_decay,
    )
    amp = bool(config.use_amp and device.type == "cuda")
    scaler = _make_scaler(amp)
    epochs: list[dict[str, object]] = []
    critic.train()
    for epoch in range(config.critic_epochs):
        total_loss = 0.0
        rows_seen = 0
        batches = 0
        for digest, indices in _shuffled_batches(
            rows_by_shard,
            config.critic_batch_size,
            config.seed + 10_000 + epoch,
        ):
            shard = dataset.shards[digest]
            privileged = torch.from_numpy(
                np.ascontiguousarray(shard.privileged_arrays["privileged_states"][indices])
            ).to(device=device, dtype=torch.float32)
            players = torch.from_numpy(
                np.ascontiguousarray(shard.actor.arrays["global_codes"][indices, 1])
            ).to(device=device, dtype=torch.long)
            target = torch.from_numpy(
                np.ascontiguousarray(target_by_shard[digest][indices])
            ).to(device=device, dtype=torch.float32)
            row_weight = torch.from_numpy(
                np.ascontiguousarray(weights_by_shard[digest][indices])
            ).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                predicted = critic(privileged, players).float()
                losses = F.huber_loss(
                    predicted, target, delta=config.huber_delta, reduction="none"
                )
                loss = (losses * row_weight).sum() / row_weight.sum()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V6 critic loss became non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.maximum_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float((losses * row_weight).sum().detach().cpu())
            rows_seen += len(indices)
            batches += 1
        epochs.append(
            {
                "epoch": epoch + 1,
                "batches": batches,
                "rows": rows_seen,
                "weightedHuber": total_loss / max(1, rows_seen),
                "weighting": weight_report,
            }
        )
    critic.eval()
    return epochs


def _train_actor(
    actor: V5PublicActor,
    dataset: V6SplitDataset,
    view: V6MatchView,
    config: V6PretrainConfig,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows_by_shard = view.rows_by_shard(
        nonforced_only=True, shards=dataset.shards
    )
    labels = _flat_row_metadata(
        rows_by_shard, dataset.shards, "normal_actions", np.int64
    )
    player_matrix = _flat_row_metadata(
        rows_by_shard, dataset.shards, "global_codes", np.int32
    )
    player_counts = player_matrix[:, 1].astype(np.int64, copy=False)
    weights, weight_report = _actor_training_weights(
        labels,
        player_counts,
        exponent=config.class_weight_exponent,
        maximum_ratio=config.maximum_class_weight_ratio,
    )
    weights_by_shard = _dense_lookup_by_shard(
        rows_by_shard, dataset.shards, weights
    )
    receipt = freeze_and_zero_residual(actor)
    if not receipt["exactZero"] or not receipt["frozen"]:
        raise RuntimeError("V6 residual output could not be frozen at exact zero")
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.actor_learning_rate,
        weight_decay=config.weight_decay,
    )
    amp = bool(config.use_amp and device.type == "cuda")
    scaler = _make_scaler(amp)
    epochs: list[dict[str, object]] = []
    actor.train()
    for epoch in range(config.actor_epochs):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        total_numerator = 0.0
        total_mass = 0.0
        rows_seen = 0
        batches = 0
        optimizer_steps = 0
        for digest, indices in _shuffled_batches(
            rows_by_shard,
            config.actor_batch_size,
            config.seed + 20_000 + epoch,
        ):
            shard = dataset.shards[digest]
            public = actor_batch_from_packed_arrays(
                shard.actor.arrays, indices, device
            )
            normal = torch.from_numpy(
                np.ascontiguousarray(shard.actor.arrays["normal_actions"][indices])
            ).to(device=device, dtype=torch.long)
            row_weight = torch.from_numpy(
                np.ascontiguousarray(weights_by_shard[digest][indices])
            ).to(device=device, dtype=torch.float32)
            with _autocast(device, amp):
                output = actor.forward_packed_batch(public, normal)
                target = (
                    (output.action_indices == output.normal_actions.unsqueeze(1))
                    & output.action_mask
                ).to(torch.int64).argmax(dim=-1)
                losses = F.cross_entropy(
                    output.normal_auxiliary_logits.float(), target, reduction="none"
                )
                loss = (losses * row_weight).sum() / row_weight.sum()
                scaled_loss = loss / config.actor_gradient_accumulation
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V6 Actor Normal CE became non-finite")
            scaler.scale(scaled_loss).backward()
            accumulated += 1
            if accumulated == config.actor_gradient_accumulation:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, config.maximum_gradient_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0
            total_numerator += float((losses * row_weight).sum().detach().cpu())
            total_mass += float(row_weight.sum().detach().cpu())
            rows_seen += len(indices)
            batches += 1
        if accumulated:
            scaler.unscale_(optimizer)
            # Each partial-group loss was divided by the configured group
            # size.  Restore an exact mean over the actually accumulated rows.
            correction = config.actor_gradient_accumulation / accumulated
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(parameters, config.maximum_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        epochs.append(
            {
                "epoch": epoch + 1,
                "batches": batches,
                "optimizerSteps": optimizer_steps,
                "rows": rows_seen,
                "weightedNormalCrossEntropy": total_numerator / total_mass,
            }
        )
    actor.eval()
    final_receipt = freeze_and_zero_residual(actor)
    if not final_receipt["exactZero"] or not final_receipt["frozen"]:
        raise RuntimeError("V6 residual output drifted from frozen exact zero")
    return epochs, {"weighting": weight_report, "residual": final_receipt}


def _explained_variance(target: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.var(target.astype(np.float64)))
    if variance <= 0.0:
        return 0.0
    return 1.0 - float(np.var((target - prediction).astype(np.float64))) / variance


def _critic_metrics(
    critic: V5CentralStateValueCritic,
    dataset: V6SplitDataset,
    view: V6MatchView,
    device: torch.device,
    batch_size: int,
    huber_delta: float,
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, object]:
    mc = (
        monte_carlo_targets_by_shard(view, dataset.shards)
        if targets is None
        else dict(targets)
    )
    output: dict[str, object] = {}
    all_mc: list[np.ndarray] = []
    all_stored: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_players: list[np.ndarray] = []
    critic.eval()
    with torch.no_grad():
        for digest, (rows, mc_values) in mc.items():
            shard = dataset.shards[digest]
            if "returns" not in shard.actor.arrays:
                raise ValueError("V6 comparison requires stored V5 GAE returns")
            for start in range(0, len(rows), batch_size):
                indices = rows[start : start + batch_size]
                privileged = torch.from_numpy(
                    np.ascontiguousarray(
                        shard.privileged_arrays["privileged_states"][indices]
                    )
                ).to(device=device, dtype=torch.float32)
                players = torch.from_numpy(
                    np.ascontiguousarray(
                        shard.actor.arrays["global_codes"][indices, 1]
                    )
                ).to(device=device, dtype=torch.long)
                predicted = critic(privileged, players).float().cpu().numpy()
                all_predictions.append(predicted)
                all_players.append(players.cpu().numpy())
            all_mc.append(mc_values)
            all_stored.append(
                np.asarray(shard.actor.arrays["returns"][rows], dtype=np.float32)
            )
    predictions = np.concatenate(all_predictions)
    mc_values = np.concatenate(all_mc)
    stored_values = np.concatenate(all_stored)
    players = np.concatenate(all_players)

    def metrics(target: np.ndarray, selected: np.ndarray) -> dict[str, float]:
        prediction = predictions[selected]
        truth = target[selected]
        losses = F.huber_loss(
            torch.from_numpy(prediction),
            torch.from_numpy(truth),
            delta=huber_delta,
            reduction="none",
        ).numpy()
        return {
            "huber": float(losses.mean(dtype=np.float64)),
            "explainedVariance": _explained_variance(truth, prediction),
        }

    all_rows = np.ones(len(predictions), dtype=np.bool_)
    output["selectionTarget"] = "exactGamma1MonteCarlo"
    output["monteCarlo"] = metrics(mc_values, all_rows)
    output["storedGae"] = metrics(stored_values, all_rows)
    output["perPlayerCount"] = {
        str(player): {
            "monteCarlo": metrics(mc_values, players == player),
            "storedGae": metrics(stored_values, players == player),
            "rows": int((players == player).sum()),
        }
        for player in PLAYER_COUNTS
    }
    return output


def _actor_normal_metrics(
    actor: V5PublicActor,
    dataset: V6SplitDataset,
    view: V6MatchView,
    device: torch.device,
    batch_size: int,
    *,
    require_greedy_parity: bool,
) -> dict[str, object]:
    rows_by_shard = view.rows_by_shard(
        nonforced_only=False, shards=dataset.shards
    )
    def accumulator() -> dict[str, object]:
        return {
            "ce": 0.0,
            "rows": 0,
            "top1": 0,
            "top3": 0,
            "parity": 0,
            "perP": {
                player: {
                    "ce": 0.0,
                    "rows": 0,
                    "top1": 0,
                    "top3": 0,
                    "parity": 0,
                }
                for player in PLAYER_COUNTS
            },
        }

    populations = {"allRows": accumulator(), "nonforced": accumulator()}

    def update(
        record: dict[str, object],
        selected: torch.Tensor,
        player_values: torch.Tensor,
        losses: torch.Tensor,
        hit1: torch.Tensor,
        hit3: torch.Tensor,
        equal: torch.Tensor,
    ) -> None:
        count = int(selected.sum().cpu())
        if not count:
            return
        record["ce"] = float(record["ce"]) + float(losses[selected].sum().cpu())
        record["rows"] = int(record["rows"]) + count
        record["top1"] = int(record["top1"]) + int(hit1[selected].sum().cpu())
        record["top3"] = int(record["top3"]) + int(hit3[selected].sum().cpu())
        record["parity"] = int(record["parity"]) + int(equal[selected].sum().cpu())
        per_p = record["perP"]
        assert isinstance(per_p, dict)
        for player in PLAYER_COUNTS:
            player_selected = selected & player_values.eq(player)
            player_count = int(player_selected.sum().cpu())
            if not player_count:
                continue
            player_record = per_p[player]
            assert isinstance(player_record, dict)
            player_record["ce"] = float(player_record["ce"]) + float(
                losses[player_selected].sum().cpu()
            )
            player_record["rows"] = int(player_record["rows"]) + player_count
            player_record["top1"] = int(player_record["top1"]) + int(
                hit1[player_selected].sum().cpu()
            )
            player_record["top3"] = int(player_record["top3"]) + int(
                hit3[player_selected].sum().cpu()
            )
            player_record["parity"] = int(player_record["parity"]) + int(
                equal[player_selected].sum().cpu()
            )

    actor.eval()
    with torch.no_grad():
        for digest in sorted(rows_by_shard):
            shard = dataset.shards[digest]
            selected_rows = rows_by_shard[digest]
            for start in range(0, len(selected_rows), batch_size):
                indices = selected_rows[start : start + batch_size]
                public = actor_batch_from_packed_arrays(
                    shard.actor.arrays, indices, device
                )
                normal = torch.from_numpy(
                    np.ascontiguousarray(shard.actor.arrays["normal_actions"][indices])
                ).to(device=device, dtype=torch.long)
                output = actor.forward_packed_batch(public, normal)
                target = (
                    (output.action_indices == output.normal_actions.unsqueeze(1))
                    & output.action_mask
                ).to(torch.int64).argmax(dim=-1)
                losses = F.cross_entropy(
                    output.normal_auxiliary_logits.float(), target, reduction="none"
                )
                predicted_top1 = output.normal_auxiliary_logits.argmax(dim=-1)
                k = min(3, output.normal_auxiliary_logits.shape[1])
                predicted_top3 = output.normal_auxiliary_logits.topk(k, dim=-1).indices
                hit1 = predicted_top1.eq(target)
                hit3 = predicted_top3.eq(target.unsqueeze(1)).any(dim=-1)
                greedy = output.greedy_actions()
                equal = greedy.eq(output.normal_actions)
                player_values = public.global_codes[:, 1]
                all_selected = torch.ones_like(equal, dtype=torch.bool)
                forced = torch.from_numpy(
                    np.ascontiguousarray(shard.actor.arrays["forced"][indices])
                ).to(device=device, dtype=torch.bool)
                update(
                    populations["allRows"],
                    all_selected,
                    player_values,
                    losses,
                    hit1,
                    hit3,
                    equal,
                )
                update(
                    populations["nonforced"],
                    ~forced,
                    player_values,
                    losses,
                    hit1,
                    hit3,
                    equal,
                )
    all_rows = populations["allRows"]
    if int(all_rows["rows"]) == 0:
        raise ValueError("V6 Actor validation view has no rows")
    disagreements = int(all_rows["rows"]) - int(all_rows["parity"])
    if require_greedy_parity and disagreements:
        raise RuntimeError(
            "V6 exact greedy-Normal parity failed for "
            f"{disagreements}/{all_rows['rows']} rows"
        )

    def finish(record: Mapping[str, object]) -> dict[str, object]:
        count = int(record["rows"])
        if count == 0:
            raise ValueError("V6 Actor validation omitted a player-count stratum")
        return {
            "rows": count,
            "normalCrossEntropy": float(record["ce"]) / count,
            "normalTop1": int(record["top1"]) / count,
            "normalTop3": int(record["top3"]) / count,
            "greedyNormalParity": int(record["parity"]) / count,
        }

    def finish_population(record: Mapping[str, object]) -> dict[str, object]:
        count = int(record["rows"])
        if count == 0:
            raise ValueError("V6 Actor validation population is empty")
        per_p = record["perP"]
        assert isinstance(per_p, Mapping)
        finished_per_p = {
            str(player): finish(per_p[player]) for player in PLAYER_COUNTS
        }
        return {
            "rows": count,
            "normalCrossEntropy": float(record["ce"]) / count,
            "normalTop1": int(record["top1"]) / count,
            "normalTop3": int(record["top3"]) / count,
            "greedyNormalParity": int(record["parity"]) / count,
            "greedyNormalDisagreements": count - int(record["parity"]),
            "equalPlayerCountMean": {
                name: math.fsum(
                    float(finished_per_p[str(player)][name])
                    for player in PLAYER_COUNTS
                )
                / len(PLAYER_COUNTS)
                for name in (
                    "normalCrossEntropy",
                    "normalTop1",
                    "normalTop3",
                    "greedyNormalParity",
                )
            },
            "perPlayerCount": finished_per_p,
        }

    return {
        "allRows": finish_population(populations["allRows"]),
        "nonforced": finish_population(populations["nonforced"]),
        "progressMetricPopulation": "nonforced",
        "greedyParityGate": {
            "population": "allRows",
            "required": 1.0,
            "passed": disagreements == 0,
        },
    }


def _input_paths(
    initial_model_pair: str | Path | None,
    initial_actor: str | Path | None,
    initial_critic: str | Path | None,
) -> tuple[Path, Path, dict[str, object]]:
    if initial_model_pair is not None:
        if initial_actor is not None or initial_critic is not None:
            raise ValueError(
                "use either --initial-model-pair or explicit Actor/critic paths"
            )
        root = Path(initial_model_pair).resolve(strict=True)
        actor_path = root / "actor-bundle"
        critic_path = root / "critic.pt"
    else:
        if initial_actor is None or initial_critic is None:
            raise ValueError("explicit initialization requires both Actor and critic")
        actor_path = Path(initial_actor).resolve(strict=True)
        critic_path = Path(initial_critic).resolve(strict=True)
        if actor_path.parent != critic_path.parent:
            raise ValueError("initial Actor and critic must belong to one model-pair root")
        root = actor_path.parent
    pair = verify_v5_model_pair(root)
    return actor_path, critic_path, pair


def _initialization_metadata(
    actor_manifest: Mapping[str, object], critic_payload: Mapping[str, object]
) -> dict[str, int]:
    actor_metadata = actor_manifest.get("metadata")
    critic_metadata = critic_payload.get("metadata")
    if not isinstance(actor_metadata, Mapping) or not isinstance(
        critic_metadata, Mapping
    ):
        raise ValueError("initial model pair omitted seed metadata")
    output: dict[str, int] = {}
    for name in (
        "initializationSeed",
        "actorInitializationSeed",
        "criticInitializationSeed",
    ):
        value = actor_metadata.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != critic_metadata.get(name)
        ):
            raise ValueError(f"initial model-pair seed mismatch: {name}")
        output[name] = value
    return output


def _runtime_report(device: torch.device, elapsed_seconds: float) -> dict[str, object]:
    cuda = device.type == "cuda"
    return {
        "elapsedSeconds": elapsed_seconds,
        "python": sys.version.split()[0],
        "processArgv": [str(value) for value in sys.argv],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cudaAvailable": bool(torch.cuda.is_available()),
        "cudaRuntime": torch.version.cuda,
        "cudnnVersion": torch.backends.cudnn.version(),
        "cudaDeviceName": torch.cuda.get_device_name(device) if cuda else None,
        "cudaCapability": list(torch.cuda.get_device_capability(device)) if cuda else None,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _publish_output(
    output_directory: Path,
    actor: V5PublicActor,
    critic: V5CentralStateValueCritic,
    seeds: Mapping[str, int],
    result: dict[str, object],
) -> dict[str, object]:
    target = output_directory.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"immutable V6 output already exists: {target}")
    lock = target.parent / f".{target.name}.publish.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shared_metadata = {
            **dict(seeds),
            "v6PretrainFormat": V6_PRETRAIN_FORMAT,
            "splitManifestSha256": result["sources"]["splitManifestSha256"],  # type: ignore[index]
        }
        export_v5_actor_bundle(
            actor, staging / "actor-bundle", metadata=shared_metadata
        )
        export_v5_critic_checkpoint(
            critic,
            staging / "critic.pt",
            metadata={**shared_metadata, "trainingOnly": True},
        )
        pair = publish_v5_model_pair_manifest(staging)
        result["outputModelPair"] = pair
        result_raw = canonical_json_bytes(result)
        _write_exclusive(staging / "result.json", result_raw)
        inventory: dict[str, object] = {}
        for path in sorted(
            (value for value in staging.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(staging).as_posix(),
        ):
            relative = path.relative_to(staging).as_posix()
            inventory[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "format": V6_PRETRAIN_FORMAT,
            "version": V6_PRETRAIN_VERSION,
            "files": inventory,
            "modelPairId": pair["pairId"],
            "resultSha256": hashlib.sha256(result_raw).hexdigest(),
            "residual": result["residual"],
            "sources": result["sources"],
        }
        manifest_raw = canonical_json_bytes(manifest)
        _write_exclusive(staging / "manifest.json", manifest_raw)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        _write_exclusive(
            staging / "manifest.json.sha256",
            f"{manifest_sha}  manifest.json\n".encode("ascii"),
        )
        if target.exists():
            raise FileExistsError("immutable V6 output appeared during publication")
        os.rename(staging, target)
        return {
            "manifest": manifest,
            "manifestSha256": manifest_sha,
            "outputDirectory": str(target),
            "result": result,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def train_v6_warm_start(
    dataset_index: str | Path,
    split_manifest: str | Path,
    output_directory: str | Path,
    *,
    initial_model_pair: str | Path | None = None,
    initial_actor: str | Path | None = None,
    initial_critic: str | Path | None = None,
    config: V6PretrainConfig | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Train and atomically publish one V6 Actor/critic warm-start pair."""

    started = time.monotonic()
    cfg = config or V6PretrainConfig()
    target_device = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    numerics = configure_v5_policy_numerics(target_device)
    _seed_all(cfg.seed)
    actor_path, critic_path, input_pair = _input_paths(
        initial_model_pair, initial_actor, initial_critic
    )
    actor, actor_manifest = load_v5_actor_bundle(actor_path)
    critic, critic_payload = load_v5_critic_checkpoint(critic_path)
    seeds = _initialization_metadata(actor_manifest, critic_payload)
    assert_actor_critic_parameter_isolation(actor, critic)
    input_residual = residual_zero_receipt(actor)
    reset_residual = freeze_and_zero_residual(actor)
    if not reset_residual["exactZero"] or not reset_residual["frozen"]:
        raise RuntimeError("V6 could not establish its exact-Normal start state")
    actor = actor.to(target_device)
    critic = critic.to(target_device)

    with load_v6_split_dataset(
        dataset_index,
        split_manifest,
        train_fraction=cfg.train_fraction,
    ) as dataset:
        train_view = dataset.views["train"]
        validation_view = dataset.views["validation"]
        train_mc = monte_carlo_targets_by_shard(train_view, dataset.shards)
        validation_mc = monte_carlo_targets_by_shard(
            validation_view, dataset.shards
        )

        initial_metrics = {
            "criticValidation": _critic_metrics(
                critic,
                dataset,
                validation_view,
                target_device,
                cfg.validation_batch_size,
                cfg.huber_delta,
                validation_mc,
            ),
            "actorValidation": _actor_normal_metrics(
                actor,
                dataset,
                validation_view,
                target_device,
                cfg.validation_batch_size,
                require_greedy_parity=True,
            ),
        }
        critic_epochs = _train_critic(
            critic, dataset, train_mc, cfg, target_device
        )
        actor_epochs, actor_training = _train_actor(
            actor, dataset, train_view, cfg, target_device
        )
        final_actor_validation = _actor_normal_metrics(
            actor,
            dataset,
            validation_view,
            target_device,
            cfg.validation_batch_size,
            require_greedy_parity=True,
        )
        final_critic_validation = _critic_metrics(
            critic,
            dataset,
            validation_view,
            target_device,
            cfg.validation_batch_size,
            cfg.huber_delta,
            validation_mc,
        )
        residual = residual_zero_receipt(actor)
        if not residual["exactZero"] or not residual["frozen"]:
            raise RuntimeError("V6 publication requires frozen exact-zero residual")
        source_record = {
            "corpusIdentitySha256": dataset.corpus_identity_sha256,
            "datasetIndex": str(dataset.index_root),
            "datasetIndexManifestSha256": dataset.index_manifest_sha256,
            "splitManifest": str(Path(split_manifest).resolve()),
            "splitManifestSha256": dataset.split_manifest_sha256,
            "initialActor": v5_actor_bundle_digests(actor_path),
            "initialCriticSha256": sha256_file(critic_path),
            "initialModelPair": input_pair,
            "implementationSha256": {
                "v6Pretrain": sha256_file(Path(__file__).resolve()),
                "v6Targets": sha256_file(
                    Path(__file__).resolve().with_name("v6_targets.py")
                ),
            },
        }
        initial_actor_progress = initial_metrics["actorValidation"]["nonforced"][
            "equalPlayerCountMean"
        ]
        final_actor_progress = final_actor_validation["nonforced"][
            "equalPlayerCountMean"
        ]
        initial_critic_progress = initial_metrics["criticValidation"]["monteCarlo"]
        final_critic_progress = final_critic_validation["monteCarlo"]
        actor_progress_gate = {
            "population": "validation nonforced rows, equal p4..p10 mean",
            "initialNormalCrossEntropy": initial_actor_progress["normalCrossEntropy"],
            "finalNormalCrossEntropy": final_actor_progress["normalCrossEntropy"],
            "initialNormalTop1": initial_actor_progress["normalTop1"],
            "finalNormalTop1": final_actor_progress["normalTop1"],
            "requiresCrossEntropyStrictDecrease": True,
            "requiresTop1NonRegression": True,
            "passed": (
                final_actor_progress["normalCrossEntropy"]
                < initial_actor_progress["normalCrossEntropy"]
                and final_actor_progress["normalTop1"]
                >= initial_actor_progress["normalTop1"]
            ),
        }
        critic_progress_gate = {
            "selectionTarget": "exactGamma1MonteCarlo",
            "initialValidationHuber": initial_critic_progress["huber"],
            "finalValidationHuber": final_critic_progress["huber"],
            "requiresHuberStrictDecrease": True,
            "passed": (
                final_critic_progress["huber"]
                < initial_critic_progress["huber"]
            ),
        }
        result: dict[str, object] = {
            "format": V6_PRETRAIN_FORMAT,
            "version": V6_PRETRAIN_VERSION,
            "config": cfg.to_dict(),
            "stageOrder": [
                "critic-exact-gamma1-monte-carlo",
                "actor-dense-normal-auxiliary",
                "full-validation-greedy-normal-parity",
            ],
            "targetContracts": {
                "critic": V6_MC_RETURN_CONTRACT,
                "actorClassWeight": V6_NORMAL_CLASS_WEIGHT_CONTRACT,
            },
            "views": {
                "train": train_view.summary(),
                "validation": validation_view.summary(),
                "testReserved": dataset.views["test"].summary(),
                "selectionUnit": "complete-five-act-match",
            },
            "initialMetrics": initial_metrics,
            "initialResidual": {
                "input": input_residual,
                "afterRequiredZeroReset": reset_residual,
            },
            "training": {
                "critic": {
                    "selectionTarget": "exactGamma1MonteCarlo",
                    "epochs": critic_epochs,
                },
                "actor": {
                    "target": "Normal auxiliary logits over every legal action",
                    "epochs": actor_epochs,
                    **actor_training,
                },
            },
            "finalMetrics": {
                "criticValidation": final_critic_validation,
                "actorValidation": final_actor_validation,
            },
            "progressGates": {
                "actor": actor_progress_gate,
                "critic": critic_progress_gate,
                "greedyNormalParity": final_actor_validation["greedyParityGate"],
            },
            "residual": residual,
            "policyNumerics": numerics,
            "runtime": _runtime_report(
                target_device, time.monotonic() - started
            ),
            "sources": source_record,
        }
        return _publish_output(
            Path(output_directory), actor, critic, seeds, result
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--initial-model-pair", type=Path)
    group.add_argument("--initial-actor", type=Path)
    parser.add_argument("--initial-critic", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--actor-epochs", type=int, default=6)
    parser.add_argument("--critic-epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=860_100_001)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.initial_actor is not None and arguments.initial_critic is None:
        raise SystemExit("--initial-actor requires --initial-critic")
    if arguments.initial_model_pair is not None and arguments.initial_critic is not None:
        raise SystemExit("--initial-critic is only valid with --initial-actor")
    config = V6PretrainConfig(
        seed=arguments.seed,
        train_fraction=arguments.train_fraction,
        critic_epochs=arguments.critic_epochs,
        actor_epochs=arguments.actor_epochs,
        use_amp=not arguments.no_amp,
    )
    result = train_v6_warm_start(
        arguments.dataset_index,
        arguments.split_manifest,
        arguments.output,
        initial_model_pair=arguments.initial_model_pair,
        initial_actor=arguments.initial_actor,
        initial_critic=arguments.initial_critic,
        config=config,
        device=arguments.device,
    )
    print(
        json.dumps(
            {
                "manifestSha256": result["manifestSha256"],
                "outputDirectory": result["outputDirectory"],
                "pairId": result["result"]["outputModelPair"]["pairId"],  # type: ignore[index]
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V6MatchRecord",
    "V6MatchView",
    "V6PretrainConfig",
    "V6SplitDataset",
    "freeze_and_zero_residual",
    "load_v6_split_dataset",
    "monte_carlo_targets_by_shard",
    "residual_zero_receipt",
    "train_v6_warm_start",
]
