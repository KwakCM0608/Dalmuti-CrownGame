from __future__ import annotations

"""Mmap-native, privacy-separated V5 dataset shards.

One shard is an immutable directory of ordinary ``.npy`` files.  Actor-visible
arrays and privileged critic arrays live in different subdirectories and have
independent manifest partitions.  The actor-only loader intentionally never
stats, hashes, or opens a privileged file.

Merging is a canonical index directory containing references to immutable
shards.  It does not concatenate or rewrite any array, so loading continues to
use NumPy memory maps and a merge requires essentially no additional storage.
"""

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
import uuid

import numpy as np

from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_gae import compute_smdp_gae


V5_ACTION_COUNT = 236
V5_SHARD_FORMAT = "dalmuti-v5-mmap-npy-shard"
V5_SHARD_VERSION = 2
V5_INDEX_FORMAT = "dalmuti-v5-zero-copy-index"
V5_INDEX_VERSION = 1
V5_SEQUENCE_CONTRACT = "complete-match-shared-candidate-decision-smdp-v1"
V5_SEMANTIC_VALIDATION_CONTRACT = (
    "dalmuti-v5-packed-public-exact-semantics-v1"
)

V5_REQUIRED_PUBLIC_CORE_ARRAYS = frozenset(
    {
        "global_codes",
        "own_rank_counts",
        "public_played_counts",
        "player_codes",
        "player_masks",
        "table_codes",
        "legal_action_bits",
        "belief_response_feasibility",
        "history_events",
        "history_end",
    }
)
V5_REQUIRED_SEQUENCE_ARRAYS = frozenset(
    {
        "match_offsets",
        "candidate_bitsets",
        "player_counts",
        "decision_actor_ids",
        "decision_acts",
        "normal_actions",
        "actions",
        "old_log_probs",
        "old_values",
        "reward_to_next",
        "done",
        "forced",
        "next_decision",
    }
)
V5_REQUIRED_ACTOR_ARRAYS = (
    V5_REQUIRED_PUBLIC_CORE_ARRAYS | V5_REQUIRED_SEQUENCE_ARRAYS
)
V5_REQUIRED_PRIVILEGED_ARRAYS = frozenset({"privileged_states"})
V5_MATCH_PROVENANCE_ARRAYS = frozenset({"match_indices", "match_seeds"})
V5_DERIVED_ACTOR_ARRAYS = frozenset(
    {
        "advantages",
        "returns",
        "deltas",
        "policy_mask",
        "value_mask",
        "policy_loss_weights",
        "value_loss_weights",
    }
)
V5_OPTIONAL_ACTOR_ARRAYS = V5_DERIVED_ACTOR_ARRAYS | frozenset(
    {"selected_action_probabilities", "policy_entropies"}
)

_ARRAY_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STRUCTURAL_DTYPES = {
    "match_offsets": np.dtype(np.uint32),
    "candidate_bitsets": np.dtype(np.uint16),
    "player_counts": np.dtype(np.uint8),
    "decision_actor_ids": np.dtype(np.uint8),
    "decision_acts": np.dtype(np.uint8),
    "normal_actions": np.dtype(np.uint16),
    "actions": np.dtype(np.uint16),
    "old_log_probs": np.dtype(np.float32),
    "old_values": np.dtype(np.float32),
    "reward_to_next": np.dtype(np.float32),
    "done": np.dtype(np.bool_),
    "forced": np.dtype(np.bool_),
    "next_decision": np.dtype(np.int32),
    "legal_action_bits": np.dtype(np.uint8),
    "history_events": np.dtype(np.uint8),
    "history_end": np.dtype(np.uint32),
    "global_codes": np.dtype(np.int32),
    "own_rank_counts": np.dtype(np.uint8),
    "public_played_counts": np.dtype(np.uint8),
    "player_codes": np.dtype(np.uint8),
    "player_masks": np.dtype(np.bool_),
    "table_codes": np.dtype(np.uint8),
    "belief_response_feasibility": np.dtype(np.float32),
}
_DERIVED_DTYPES = {
    "advantages": np.dtype(np.float32),
    "returns": np.dtype(np.float32),
    "deltas": np.dtype(np.float32),
    "policy_mask": np.dtype(np.bool_),
    "value_mask": np.dtype(np.bool_),
    "policy_loss_weights": np.dtype(np.float32),
    "value_loss_weights": np.dtype(np.float32),
    "selected_action_probabilities": np.dtype(np.float32),
    "policy_entropies": np.dtype(np.float32),
}

_VERIFIED_ACTOR_ARRAYS_AUTHORITY = object()


class V5VerifiedActorArrays(Mapping[str, np.ndarray]):
    """Read-only actor arrays carrying a loader-verified semantic receipt.

    The marker is intentionally attached to the mapping object rather than an
    ordinary dictionary key.  Copying these arrays into a plain ``dict`` drops
    the fast-path authority, so ad-hoc decoder mappings must recompute exact
    response feasibility for their selected rows.
    """

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        receipt_sha256: str,
        *,
        _authority: object,
    ) -> None:
        if _authority is not _VERIFIED_ACTOR_ARRAYS_AUTHORITY:
            raise TypeError(
                "verified actor arrays can only be created by the shard loader"
            )
        self._arrays = MappingProxyType(dict(arrays))
        self._receipt_sha256 = receipt_sha256

    @property
    def __v5_exact_public_semantics__(self) -> tuple[str, str]:
        return V5_PUBLIC_CONTRACT_SHA256, self._receipt_sha256

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)


@dataclass(frozen=True)
class V5ActorShard:
    root: Path
    manifest: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]

    @property
    def decision_count(self) -> int:
        return int(self.manifest["counts"]["decisions"])  # type: ignore[index]

    @property
    def match_count(self) -> int:
        return int(self.manifest["counts"]["matches"])  # type: ignore[index]

    @property
    def action_count(self) -> int:
        return int(self.manifest["actionCount"])

    def history(self, decision_index: int) -> np.ndarray:
        if isinstance(decision_index, bool) or not 0 <= decision_index < self.decision_count:
            raise IndexError("decision_index is outside this shard")
        ends = self.arrays["history_end"]
        start = 0 if decision_index == 0 else int(ends[decision_index - 1])
        stop = int(ends[decision_index])
        return self.arrays["history_events"][start:stop]

    def legal_mask(self, decision_index: int) -> np.ndarray:
        if isinstance(decision_index, bool) or not 0 <= decision_index < self.decision_count:
            raise IndexError("decision_index is outside this shard")
        packed = np.asarray(self.arrays["legal_action_bits"][decision_index])
        return np.unpackbits(packed, bitorder="little")[: self.action_count].astype(
            np.bool_, copy=False
        )

    def close(self) -> None:
        """Release mmap handles eagerly (required before deletion on Windows)."""

        for array in self.arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def __enter__(self) -> "V5ActorShard":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class V5TrainingShard:
    actor: V5ActorShard
    privileged_arrays: Mapping[str, np.ndarray]

    def close(self) -> None:
        self.actor.close()
        for array in self.privileged_arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def __enter__(self) -> "V5TrainingShard":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class V5ShardIndex:
    root: Path
    manifest: Mapping[str, object]
    shard_paths: tuple[Path, ...]

    @property
    def decision_count(self) -> int:
        return int(self.manifest["counts"]["decisions"])  # type: ignore[index]

    @property
    def match_count(self) -> int:
        return int(self.manifest["counts"]["matches"])  # type: ignore[index]

    def close(self) -> None:
        """Index metadata owns no mmap; provided for uniform context handling."""

    def __enter__(self) -> "V5ShardIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("manifest content is not canonical JSON data") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _semantic_validation_receipt(
    actor_records: Mapping[str, object],
    *,
    action_count: int,
    counts: Mapping[str, object],
) -> dict[str, object]:
    actor_records_sha256 = _sha256_bytes(_canonical_json(dict(actor_records)))
    base: dict[str, object] = {
        "actionCount": action_count,
        "actorRecordsSha256": actor_records_sha256,
        "checks": [
            "categorical-shape-range-padding",
            "roles-finished-table-leader",
            "deck-conservation",
            "legal-action-mask-exact",
            "history-event-semantics",
            "response-feasibility-exact-float32",
        ],
        "contract": V5_SEMANTIC_VALIDATION_CONTRACT,
        "counts": dict(counts),
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "validatorVersion": 1,
    }
    digest = hashlib.sha256(
        b"DALMUTI-V5-SEMANTIC-VALIDATION\x00" + _canonical_json(base)
    ).hexdigest()
    return {**base, "receiptSha256": digest}


def _validate_semantic_validation_receipt(
    value: object,
    actor_records: Mapping[str, object],
    *,
    action_count: int,
    counts: Mapping[str, object],
) -> dict[str, object]:
    expected = _semantic_validation_receipt(
        actor_records, action_count=action_count, counts=counts
    )
    if not isinstance(value, dict) or value != expected:
        raise ValueError(
            "V5 shard semantic-validation receipt is missing or incompatible"
        )
    return expected


def _read_canonical_json(path: Path) -> tuple[Mapping[str, object], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canonical manifest: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise ValueError("manifest JSON is not canonically encoded")
    digest = _sha256_bytes(raw)
    sidecar = path.with_name(path.name + ".sha256")
    try:
        declared = sidecar.read_text(encoding="ascii")
    except OSError as error:
        raise ValueError("manifest checksum sidecar is missing") from error
    if declared != f"{digest}  {path.name}\n":
        raise ValueError("manifest checksum sidecar does not match canonical bytes")
    return MappingProxyType(value), digest


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has missing or unknown fields")


def _safe_array_name(name: object) -> str:
    if not isinstance(name, str) or _ARRAY_NAME.fullmatch(name) is None:
        raise ValueError("array names must use canonical lowercase snake_case")
    return name


def _as_array_mapping(value: Mapping[str, object], label: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    output: dict[str, np.ndarray] = {}
    for raw_name, raw_array in value.items():
        name = _safe_array_name(raw_name)
        array = np.asarray(raw_array)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise ValueError(f"{label}.{name} cannot use object or structured dtype")
        if array.ndim < 1:
            raise ValueError(f"{label}.{name} must have at least one dimension")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        output[name] = array
    return output


def _validate_exact_dtype(name: str, array: np.ndarray) -> None:
    expected = _STRUCTURAL_DTYPES.get(name, _DERIVED_DTYPES.get(name))
    if expected is not None and array.dtype != expected:
        raise ValueError(f"{name} must use canonical dtype {expected.name}")


def _validate_extra_public_dtype(name: str, array: np.ndarray) -> None:
    if array.dtype.kind in "iu" and array.dtype.itemsize > 4:
        raise ValueError(f"{name} must use a compact <=32-bit integer dtype")
    if array.dtype.kind == "f" and array.dtype.itemsize > 4:
        raise ValueError(f"{name} must not use float64 in an mmap training shard")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} uses an unsupported actor-visible dtype")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite actor-visible value")


def _validate_actor_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    action_count: int,
    verify_response_feasibility: bool,
) -> tuple[int, int, int]:
    missing = V5_REQUIRED_ACTOR_ARRAYS - set(arrays)
    if missing:
        raise ValueError(f"actor partition is missing required array: {sorted(missing)[0]}")
    unknown = set(arrays) - V5_REQUIRED_ACTOR_ARRAYS - V5_OPTIONAL_ACTOR_ARRAYS
    if unknown:
        raise ValueError(
            f"actor partition contains an unknown or private array: {sorted(unknown)[0]}"
        )
    if action_count != V5_ACTION_COUNT:
        raise ValueError("action_count must match the fixed V5 catalogue")
    for name, array in arrays.items():
        _validate_exact_dtype(name, array)
        _validate_extra_public_dtype(name, array)

    offsets = arrays["match_offsets"]
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError("match_offsets must have shape [match+1]")
    match_count = int(offsets.size - 1)
    offset64 = offsets.astype(np.int64, copy=False)
    decision_count = int(offset64[-1])
    if decision_count < 1 or int(offset64[0]) != 0 or np.any(
        offset64[1:] <= offset64[:-1]
    ):
        raise ValueError("match_offsets must encode non-empty complete matches")

    for name in ("candidate_bitsets", "player_counts"):
        if arrays[name].shape != (match_count,):
            raise ValueError(f"{name} must have shape [match]")
    decision_vectors = (
        "decision_actor_ids",
        "decision_acts",
        "normal_actions",
        "actions",
        "old_log_probs",
        "old_values",
        "reward_to_next",
        "done",
        "forced",
        "next_decision",
        "history_end",
    )
    for name in decision_vectors:
        if arrays[name].shape != (decision_count,):
            raise ValueError(f"{name} must have shape [decision]")
    public_shapes = {
        "global_codes": (decision_count, 6),
        "own_rank_counts": (decision_count, 13),
        "public_played_counts": (decision_count, 13),
        "player_codes": (decision_count, 10, 6),
        "player_masks": (decision_count, 10),
        "table_codes": (decision_count, 6),
        "belief_response_feasibility": (decision_count, 9),
    }
    for name, shape in public_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} has a non-canonical public core shape")
    packed_width = (action_count + 7) // 8
    if arrays["legal_action_bits"].shape != (decision_count, packed_width):
        raise ValueError("legal_action_bits must be uint8 [decision,ceil(action/8)]")
    history = arrays["history_events"]
    if history.ndim != 2 or history.shape[1] != 12:
        raise ValueError("history_events must be uint8 [event,12]")
    history_end = arrays["history_end"].astype(np.int64, copy=False)
    if np.any(history_end[1:] < history_end[:-1]) or int(history_end[-1]) != int(
        history.shape[0]
    ):
        raise ValueError("history_end must be monotone prefixes ending at history size")

    counts = arrays["player_counts"].astype(np.int64, copy=False)
    if np.any((counts < 4) | (counts > 10)):
        raise ValueError("player_counts must contain only p4..p10")
    candidate_bits = arrays["candidate_bitsets"].astype(np.int64, copy=False)
    for index, (bits, players) in enumerate(zip(candidate_bits, counts, strict=True)):
        if bits <= 0 or bits & ~((1 << int(players)) - 1):
            raise ValueError(f"candidate_bitsets[{index}] is empty or out of range")
    acts = arrays["decision_acts"].astype(np.int64, copy=False)
    if np.any((acts < 1) | (acts > 5)):
        raise ValueError("decision_acts must use the canonical five-act range 1..5")
    decision_players = np.repeat(counts, np.diff(offset64))
    global_codes = arrays["global_codes"]
    if (
        np.any(global_codes[:, 0] != 5)
        or not np.array_equal(global_codes[:, 1], decision_players)
        or not np.array_equal(global_codes[:, 2], acts)
    ):
        raise ValueError(
            "global_codes schema/player-count/act must match sequence metadata"
        )
    expected_player_masks = np.arange(10)[None, :] < decision_players[:, None]
    if not np.array_equal(arrays["player_masks"], expected_player_masks):
        raise ValueError("player_masks must match the decision player count")
    deck = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2], np.int16)
    unknown = (
        deck[None, :]
        - arrays["own_rank_counts"].astype(np.int16)
        - arrays["public_played_counts"].astype(np.int16)
    )
    if np.any(unknown < 0):
        raise ValueError("public card counts imply negative unseen cards")
    opponent_counts = arrays["player_codes"][:, 1:, 1].astype(np.int64)
    if not np.array_equal(
        opponent_counts.sum(axis=1, dtype=np.int64),
        unknown.sum(axis=1, dtype=np.int64),
    ):
        raise ValueError("opponent remaining counts disagree with unseen cards")
    response = arrays["belief_response_feasibility"]
    if (
        not np.isfinite(response).all()
        or np.any(response < 0.0)
        or np.any(response > 1.0)
    ):
        raise ValueError("belief_response_feasibility must remain finite in [0,1]")
    actions = arrays["actions"].astype(np.int64, copy=False)
    normal = arrays["normal_actions"].astype(np.int64, copy=False)
    if np.any((actions < 0) | (actions >= action_count)) or np.any(
        (normal < 0) | (normal >= action_count)
    ):
        raise ValueError("actions and normal_actions must be valid action ids")
    if not np.isfinite(arrays["old_log_probs"]).all() or np.any(
        arrays["old_log_probs"] > np.float32(2.0e-6)
    ):
        raise ValueError("old_log_probs must be finite log probabilities <= 0")
    for name in ("old_values", "reward_to_next"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite")

    legal = np.unpackbits(arrays["legal_action_bits"], axis=1, bitorder="little")
    legal = legal[:, :action_count]
    if action_count % 8:
        trailing_mask = np.uint8((0xFF << (action_count % 8)) & 0xFF)
        if np.any(arrays["legal_action_bits"][:, -1] & trailing_mask):
            raise ValueError("legal_action_bits has non-zero unused trailing bits")
    rows = np.arange(decision_count)
    if not np.all(legal[rows, actions]) or not np.all(legal[rows, normal]):
        raise ValueError("recorded and Normal actions must both be legal")
    legal_counts = legal.sum(axis=1)
    if np.any(legal_counts < 1) or not np.array_equal(
        arrays["forced"], legal_counts == 1
    ):
        raise ValueError("forced must exactly identify singleton legal action sets")
    history_starts = np.concatenate(
        (np.zeros(1, np.int64), history_end[:-1])
    )
    if np.any(history_end - history_starts > 192):
        raise ValueError("a decision history cannot exceed 192 public events")

    if "selected_action_probabilities" in arrays:
        probabilities = arrays["selected_action_probabilities"]
        if np.any(probabilities <= 0.0) or np.any(probabilities > 1.0):
            raise ValueError("selected_action_probabilities must be in (0,1]")
        if not np.allclose(
            np.log(probabilities.astype(np.float64)),
            arrays["old_log_probs"].astype(np.float64),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError("old_log_probs do not replay selected probabilities")
        if not np.all(probabilities[arrays["forced"]] == np.float32(1.0)):
            raise ValueError("forced decisions must have selected probability one")
    if "policy_entropies" in arrays and np.any(arrays["policy_entropies"] < 0.0):
        raise ValueError("policy_entropies cannot be negative")

    # This is the canonical packed/dense public boundary.  Publication pays
    # the exact combinatorial response check once; checksum/receipt-verified
    # loads repeat every inexpensive semantic check without redoing it.
    from v5_public import validate_packed_v5_public_semantics

    validate_packed_v5_public_semantics(
        arrays,
        verify_response_feasibility=verify_response_feasibility,
    )

    gae = compute_smdp_gae(
        reward_to_next=arrays["reward_to_next"],
        next_decision=arrays["next_decision"],
        done=arrays["done"],
        old_values=arrays["old_values"],
        match_offsets=offsets,
        decision_actor_ids=arrays["decision_actor_ids"],
        player_counts=arrays["player_counts"],
        forced=arrays["forced"],
        candidate_bitsets=arrays["candidate_bitsets"],
    )
    for name in V5_DERIVED_ACTOR_ARRAYS & set(arrays):
        if arrays[name].shape != (decision_count,):
            raise ValueError(f"{name} must have shape [decision]")
        if not np.array_equal(arrays[name], getattr(gae, name)):
            if arrays[name].dtype.kind != "f" or not np.allclose(
                arrays[name], getattr(gae, name), rtol=0.0, atol=2.0e-6
            ):
                raise ValueError(f"{name} is stale or violates canonical SMDP GAE")
    for name, array in arrays.items():
        if name not in V5_REQUIRED_ACTOR_ARRAYS and array.shape != (decision_count,):
            raise ValueError(f"optional actor array {name} must have shape [decision]")
    return match_count, decision_count, int(history.shape[0])


def _validate_privileged_arrays(
    arrays: Mapping[str, np.ndarray],
    decision_count: int,
    match_count: int,
    player_counts: np.ndarray,
) -> None:
    missing = V5_REQUIRED_PRIVILEGED_ARRAYS - set(arrays)
    if missing:
        raise ValueError(
            f"privileged partition is missing required array: {sorted(missing)[0]}"
        )
    present_provenance = V5_MATCH_PROVENANCE_ARRAYS & set(arrays)
    if present_provenance and present_provenance != V5_MATCH_PROVENANCE_ARRAYS:
        raise ValueError(
            "match_indices and match_seeds must be stored as one provenance pair"
        )
    for name, array in arrays.items():
        if name in V5_MATCH_PROVENANCE_ARRAYS:
            continue
        if array.shape[0] != decision_count:
            raise ValueError(f"privileged array {name} must start with [decision]")
        if array.dtype.kind not in "biuf" or (
            array.dtype.kind in "iuf" and array.dtype.itemsize > 4
        ):
            raise ValueError(f"privileged array {name} must use compact numeric data")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"privileged array {name} contains a non-finite value")
    states = arrays["privileged_states"]
    if states.dtype != np.dtype(np.float16) or states.ndim != 2 or states.shape[1] < 1:
        raise ValueError("privileged_states must be float16 [decision,feature]")
    if present_provenance:
        match_indices = arrays["match_indices"]
        match_seeds = arrays["match_seeds"]
        if (
            match_indices.dtype != np.dtype(np.uint32)
            or match_indices.shape != (match_count,)
            or match_seeds.dtype != np.dtype(np.uint32)
            or match_seeds.shape != (match_count,)
        ):
            raise ValueError(
                "private match provenance must be uint32 [match] indexes and seeds"
            )
        keys = {
            (int(player), int(index))
            for player, index in zip(player_counts, match_indices, strict=True)
        }
        if len(keys) != match_count or np.unique(match_seeds).size != match_count:
            raise ValueError("private match provenance coordinates or seeds repeat")


def _write_npy(path: Path, array: np.ndarray) -> tuple[int, str]:
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return path.stat().st_size, _sha256_file(path)


def _array_record(
    relative_path: str, array: np.ndarray, size: int, digest: str
) -> dict[str, object]:
    return {
        "byteLength": size,
        "dtype": array.dtype.str,
        "path": relative_path,
        "sha256": digest,
        "shape": list(array.shape),
    }


def _write_manifest(directory: Path, manifest: Mapping[str, object]) -> str:
    data = _canonical_json(manifest)
    digest = _sha256_bytes(data)
    manifest_path = directory / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    sidecar = directory / "manifest.json.sha256"
    with sidecar.open("xb") as handle:
        handle.write(f"{digest}  manifest.json\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes where the host exposes that primitive."""

    if os.name == "nt":
        return
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [Path(directory) for directory, _, _ in os.walk(root)]
    for directory in reversed(directories):
        _fsync_directory(directory)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""

    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("Linux renameat2(RENAME_NOREPLACE) is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result != 0:
            number = ctypes.get_errno()
            if number in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(number, os.strerror(number), target)
            raise OSError(number, os.strerror(number), target)
        return
    if os.name == "nt":
        # Windows rename is no-replace when the destination exists.
        os.rename(source, target)
        return
    raise RuntimeError("atomic no-replace directory publication is unsupported")


def _exclusive_publish_directory(target: Path, builder: Callable[[Path], None]) -> None:
    if target.is_symlink():
        raise ValueError("immutable V5 artifact target must not be a symlink")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    try:
        if os.path.lexists(os.fspath(target)):
            raise FileExistsError(f"immutable V5 artifact already exists: {target}")
        staging.mkdir()
        builder(staging)
        _fsync_directory_tree(staging)
        _rename_directory_noreplace(staging, target)
        _fsync_directory(target.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_v5_shard(
    target: str | Path,
    actor_arrays: Mapping[str, object],
    privileged_arrays: Mapping[str, object] | None = None,
    *,
    metadata: Mapping[str, object] | None = None,
    action_count: int = V5_ACTION_COUNT,
) -> str:
    """Validate and exclusively publish one immutable mmap-native shard.

    Returns the canonical ``manifest.json`` SHA-256.  No existing target is
    overwritten.  ``privileged_arrays=None`` is permitted for actor/BC-only
    shards; if supplied it must contain ``privileged_states``.
    """

    target_path = Path(target)
    actors = _as_array_mapping(actor_arrays, "actor_arrays")
    match_count, decision_count, history_count = _validate_actor_arrays(
        actors,
        action_count=action_count,
        verify_response_feasibility=True,
    )
    privileged = (
        None
        if privileged_arrays is None
        else _as_array_mapping(privileged_arrays, "privileged_arrays")
    )
    if privileged is not None:
        _validate_privileged_arrays(
            privileged,
            decision_count,
            match_count,
            actors["player_counts"],
        )
    metadata_value: Mapping[str, object] = {} if metadata is None else metadata
    # Canonicalize before touching disk so unsupported values cannot leave a
    # staging directory behind.
    metadata_roundtrip = json.loads(_canonical_json(dict(metadata_value)))

    manifest_digest: list[str] = []

    def build(staging: Path) -> None:
        partitions: dict[str, dict[str, dict[str, object]]] = {
            "actor": {},
            "privileged": {},
        }
        actor_dir = staging / "actor"
        actor_dir.mkdir()
        for name in sorted(actors):
            path = actor_dir / f"{name}.npy"
            size, digest = _write_npy(path, actors[name])
            partitions["actor"][name] = _array_record(
                f"actor/{name}.npy", actors[name], size, digest
            )
        if privileged is not None:
            privileged_dir = staging / "privileged"
            privileged_dir.mkdir()
            for name in sorted(privileged):
                path = privileged_dir / f"{name}.npy"
                size, digest = _write_npy(path, privileged[name])
                partitions["privileged"][name] = _array_record(
                    f"privileged/{name}.npy", privileged[name], size, digest
                )
        counts = {
            "decisions": decision_count,
            "historyEvents": history_count,
            "matches": match_count,
        }
        semantic_validation = _semantic_validation_receipt(
            partitions["actor"],
            action_count=action_count,
            counts=counts,
        )
        manifest = {
            "actionCount": action_count,
            "counts": counts,
            "format": V5_SHARD_FORMAT,
            "metadata": metadata_roundtrip,
            "partitions": partitions,
            "privacy": {
                "actorLoaderMayOpenPrivilegedFiles": False,
                "privilegedCriticExportAllowed": False,
                "privilegedStateSeparate": True,
            },
            "semanticValidation": semantic_validation,
            "sequenceContract": V5_SEQUENCE_CONTRACT,
            "version": V5_SHARD_VERSION,
        }
        manifest_digest.append(_write_manifest(staging, manifest))

    _exclusive_publish_directory(target_path, build)
    return manifest_digest[0]


def _validated_manifest(root: Path) -> tuple[Mapping[str, object], str]:
    manifest, digest = _read_canonical_json(root / "manifest.json")
    _exact_keys(
        manifest,
        {
            "actionCount",
            "counts",
            "format",
            "metadata",
            "partitions",
            "privacy",
            "semanticValidation",
            "sequenceContract",
            "version",
        },
        "V5 shard manifest",
    )
    if (
        manifest["format"] != V5_SHARD_FORMAT
        or manifest["version"] != V5_SHARD_VERSION
        or manifest["sequenceContract"] != V5_SEQUENCE_CONTRACT
    ):
        raise ValueError("V5 shard manifest format/version/sequence contract is incompatible")
    privacy = manifest["privacy"]
    if not isinstance(privacy, dict) or privacy != {
        "actorLoaderMayOpenPrivilegedFiles": False,
        "privilegedCriticExportAllowed": False,
        "privilegedStateSeparate": True,
    }:
        raise ValueError("V5 shard privacy contract is missing or incompatible")
    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise ValueError("V5 shard counts must be an object")
    _exact_keys(counts, {"decisions", "historyEvents", "matches"}, "counts")
    for name, minimum in (("decisions", 1), ("historyEvents", 0), ("matches", 1)):
        value = counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"V5 shard count {name} is invalid")
    action_count = manifest["actionCount"]
    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 2:
        raise ValueError("V5 shard actionCount is invalid")
    partitions = manifest["partitions"]
    if not isinstance(partitions, dict):
        raise ValueError("V5 shard partitions must be an object")
    _exact_keys(partitions, {"actor", "privileged"}, "partitions")
    actor_records = partitions["actor"]
    if not isinstance(actor_records, dict):
        raise ValueError("actor partition records must be an object")
    _validate_semantic_validation_receipt(
        manifest["semanticValidation"],
        actor_records,
        action_count=action_count,
        counts=counts,
    )
    return manifest, digest


def _safe_manifest_file(root: Path, relative: object, partition: str, name: str) -> Path:
    expected = f"{partition}/{name}.npy"
    if relative != expected:
        raise ValueError("array manifest path is not canonical for its partition")
    path = root / expected
    try:
        if os.path.commonpath((str(root.resolve()), str(path.resolve()))) != str(root.resolve()):
            raise ValueError("array manifest path escapes the shard")
    except ValueError as error:
        raise ValueError("array manifest path escapes the shard") from error
    return path


def _load_partition(
    root: Path, manifest: Mapping[str, object], partition: str
) -> dict[str, np.ndarray]:
    partitions = manifest["partitions"]
    assert isinstance(partitions, dict)
    records = partitions[partition]
    if not isinstance(records, dict):
        raise ValueError(f"{partition} partition must be an object")
    output: dict[str, np.ndarray] = {}
    for raw_name in sorted(records):
        name = _safe_array_name(raw_name)
        record = records[name]
        if not isinstance(record, dict):
            raise ValueError("array manifest record must be an object")
        _exact_keys(record, {"byteLength", "dtype", "path", "sha256", "shape"}, name)
        if (
            isinstance(record["byteLength"], bool)
            or not isinstance(record["byteLength"], int)
            or record["byteLength"] < 1
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or not isinstance(record["dtype"], str)
            or not isinstance(record["shape"], list)
            or not record["shape"]
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in record["shape"]
            )
        ):
            raise ValueError("array manifest record has invalid scalar fields")
        path = _safe_manifest_file(root, record["path"], partition, name)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ValueError(f"array file is missing: {partition}/{name}") from error
        if size != record["byteLength"] or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"array checksum or byte length mismatch: {partition}/{name}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"array is not a valid mmap-able NPY: {partition}/{name}") from error
        if array.dtype.str != record["dtype"] or list(array.shape) != record["shape"]:
            raise ValueError(f"array NPY header does not match manifest: {partition}/{name}")
        if not isinstance(array, np.memmap):
            raise ValueError(f"array did not load as a memory map: {partition}/{name}")
        output[name] = array
    return output


def load_v5_actor_shard(target: str | Path) -> V5ActorShard:
    """Load only actor-visible mmaps; privileged files are never opened."""

    root = Path(target).resolve()
    manifest, _ = _validated_manifest(root)
    actor_arrays = _load_partition(root, manifest, "actor")
    _validate_actor_arrays(
        actor_arrays,
        action_count=int(manifest["actionCount"]),
        verify_response_feasibility=False,
    )
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    if (
        int(counts["matches"]) != actor_arrays["match_offsets"].size - 1
        or int(counts["decisions"]) != actor_arrays["decision_actor_ids"].size
        or int(counts["historyEvents"]) != actor_arrays["history_events"].shape[0]
    ):
        raise ValueError("V5 shard manifest counts do not match actor arrays")
    semantic_validation = manifest["semanticValidation"]
    assert isinstance(semantic_validation, dict)
    receipt_sha256 = semantic_validation["receiptSha256"]
    assert isinstance(receipt_sha256, str)
    verified_arrays = V5VerifiedActorArrays(
        actor_arrays,
        receipt_sha256,
        _authority=_VERIFIED_ACTOR_ARRAYS_AUTHORITY,
    )
    return V5ActorShard(root, manifest, verified_arrays)


def load_v5_training_shard(target: str | Path) -> V5TrainingShard:
    actor = load_v5_actor_shard(target)
    privileged = _load_partition(actor.root, actor.manifest, "privileged")
    if not privileged:
        raise ValueError("training shard has no privileged critic partition")
    _validate_privileged_arrays(
        privileged,
        actor.decision_count,
        actor.match_count,
        actor.arrays["player_counts"],
    )
    return V5TrainingShard(actor, MappingProxyType(privileged))


def publish_v5_index_manifest(
    target: str | Path,
    shard_paths: Sequence[str | Path],
    *,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Publish a zero-copy index over already immutable V5 shard directories."""

    target_path = Path(target).resolve()
    if not shard_paths:
        raise ValueError("a V5 index requires at least one shard")
    loaded: list[
        tuple[Path, str, int, int, int, tuple[int, ...]]
    ] = []
    seen: set[Path] = set()
    action_count: int | None = None
    for raw_path in shard_paths:
        root = Path(raw_path).resolve()
        if root in seen:
            raise ValueError("a V5 index cannot contain a duplicate shard")
        seen.add(root)
        _, digest = _validated_manifest(root)
        # Validate actor arrays now.  This both closes checksum holes and still
        # leaves merge zero-copy: no array is materialized or rewritten.
        actor = load_v5_actor_shard(root)
        try:
            if action_count is None:
                action_count = actor.action_count
            elif action_count != actor.action_count:
                raise ValueError("indexed shards disagree on actionCount")
            loaded.append(
                (
                    root,
                    digest,
                    actor.decision_count,
                    actor.match_count,
                    actor.action_count,
                    tuple(int(value) for value in actor.arrays["player_counts"]),
                )
            )
        finally:
            actor.close()
    loaded.sort(key=lambda item: (item[1], item[0].as_posix().casefold()))
    metadata_value: Mapping[str, object] = {} if metadata is None else metadata
    metadata_roundtrip = json.loads(_canonical_json(dict(metadata_value)))
    records: list[dict[str, object]] = []
    total_decisions = 0
    total_matches = 0
    player_counts: set[int] = set()
    for root, digest, decision_count, match_count, _, shard_players in loaded:
        relative = os.path.relpath(root, target_path.parent).replace(os.sep, "/")
        records.append(
            {
                "decisionCount": decision_count,
                "manifestSha256": digest,
                "matchCount": match_count,
                "relativePath": relative,
            }
        )
        total_decisions += decision_count
        total_matches += match_count
        player_counts.update(shard_players)
    manifest = {
        "actionCount": action_count,
        "counts": {
            "decisions": total_decisions,
            "matches": total_matches,
            "shards": len(records),
        },
        "format": V5_INDEX_FORMAT,
        "mergeMode": "zero-copy immutable shard references",
        "metadata": metadata_roundtrip,
        "playerCounts": sorted(player_counts),
        "shards": records,
        "version": V5_INDEX_VERSION,
    }
    digest_holder: list[str] = []

    def build(staging: Path) -> None:
        digest_holder.append(_write_manifest(staging, manifest))

    _exclusive_publish_directory(target_path, build)
    return digest_holder[0]


def load_v5_index_manifest(target: str | Path) -> V5ShardIndex:
    root = Path(target).resolve()
    manifest, _ = _read_canonical_json(root / "manifest.json")
    _exact_keys(
        manifest,
        {
            "actionCount",
            "counts",
            "format",
            "mergeMode",
            "metadata",
            "playerCounts",
            "shards",
            "version",
        },
        "V5 index manifest",
    )
    if (
        manifest["format"] != V5_INDEX_FORMAT
        or manifest["version"] != V5_INDEX_VERSION
        or manifest["mergeMode"] != "zero-copy immutable shard references"
    ):
        raise ValueError("V5 index format/version/merge mode is incompatible")
    records = manifest["shards"]
    counts = manifest["counts"]
    if not isinstance(records, list) or not records or not isinstance(counts, dict):
        raise ValueError("V5 index shard records or counts are invalid")
    _exact_keys(counts, {"decisions", "matches", "shards"}, "index counts")
    paths: list[Path] = []
    decisions = 0
    matches = 0
    seen: set[Path] = set()
    observed_players: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("V5 index shard record must be an object")
        _exact_keys(
            record,
            {"decisionCount", "manifestSha256", "matchCount", "relativePath"},
            "index shard",
        )
        if (
            not isinstance(record["relativePath"], str)
            or Path(record["relativePath"]).is_absolute()
            or not isinstance(record["manifestSha256"], str)
            or _SHA256.fullmatch(record["manifestSha256"]) is None
        ):
            raise ValueError("V5 index shard identity is invalid")
        path = (root.parent / record["relativePath"]).resolve()
        if path in seen:
            raise ValueError("V5 index resolves duplicate shard paths")
        seen.add(path)
        _, digest = _validated_manifest(path)
        if digest != record["manifestSha256"]:
            raise ValueError("V5 index shard manifest checksum no longer matches")
        actor = load_v5_actor_shard(path)
        try:
            if actor.action_count != manifest["actionCount"]:
                raise ValueError("V5 index actionCount disagrees with a shard")
            if (
                actor.decision_count != record["decisionCount"]
                or actor.match_count != record["matchCount"]
            ):
                raise ValueError("V5 index shard counts no longer match")
            decisions += actor.decision_count
            matches += actor.match_count
            observed_players.update(
                int(value) for value in actor.arrays["player_counts"]
            )
        finally:
            actor.close()
        paths.append(path)
    if counts != {"decisions": decisions, "matches": matches, "shards": len(paths)}:
        raise ValueError("V5 index aggregate counts do not match its shards")
    if manifest["playerCounts"] != sorted(observed_players):
        raise ValueError("V5 index playerCounts do not match its shards")
    return V5ShardIndex(root, manifest, tuple(paths))


def load_v5_actor_index(target: str | Path) -> tuple[V5ActorShard, ...]:
    """Load all actor partitions named by a zero-copy index."""

    index = load_v5_index_manifest(target)
    return tuple(load_v5_actor_shard(path) for path in index.shard_paths)


# Clear collector/training-friendly aliases.
write_v5_shard = publish_v5_shard
merge_v5_shards = publish_v5_index_manifest
load_actor_shard = load_v5_actor_shard
load_training_shard = load_v5_training_shard


__all__ = [
    "V5_ACTION_COUNT",
    "V5_DERIVED_ACTOR_ARRAYS",
    "V5_INDEX_FORMAT",
    "V5_INDEX_VERSION",
    "V5_MATCH_PROVENANCE_ARRAYS",
    "V5_REQUIRED_ACTOR_ARRAYS",
    "V5_REQUIRED_PRIVILEGED_ARRAYS",
    "V5_SEMANTIC_VALIDATION_CONTRACT",
    "V5_SEQUENCE_CONTRACT",
    "V5_SHARD_FORMAT",
    "V5_SHARD_VERSION",
    "V5ActorShard",
    "V5ShardIndex",
    "V5TrainingShard",
    "V5VerifiedActorArrays",
    "load_actor_shard",
    "load_training_shard",
    "load_v5_actor_index",
    "load_v5_actor_shard",
    "load_v5_index_manifest",
    "load_v5_training_shard",
    "merge_v5_shards",
    "publish_v5_index_manifest",
    "publish_v5_shard",
    "write_v5_shard",
]
