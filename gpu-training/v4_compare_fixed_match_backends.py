from __future__ import annotations

"""Fail-closed CPU/CUDA calibration for fixed-match V4 PPO rollouts.

The comparator is deliberately narrower than a generic NPZ diff.  Both inputs
must independently satisfy the public V4 fixed-match dataset contract, their
external metadata and checksum sidecars must be canonical, and their complete
collection identity must match.  Every array is exact unless it is named in
``POLICY_NUMERIC_TOLERANCES``.  This makes a newly introduced or otherwise
unclassified backend-dependent array a hard failure instead of silently
expanding the calibration exception surface.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from v4_dataset import (
    V4_FIXED_MATCH_PPO_PREPARATION_FORMAT,
    fixed_match_shard_identity_sha256,
    load_v4_dataset_npz,
)
from v4_export import canonical_json_bytes, sha256_file


CALIBRATION_FORMAT = "dalmuti-v4-fixed-match-cpu-cuda-calibration"
CALIBRATION_VERSION = 1
OLD_ACTION_LOG_PROBABILITY_MAX_ABS = 2.0e-5

# These are the complete and exclusive numeric backend exception surface.
# Each value is an independently enforced maximum absolute difference.
POLICY_NUMERIC_TOLERANCES: dict[str, float] = {
    "old_action_log_probs": OLD_ACTION_LOG_PROBABILITY_MAX_ABS,
    "selected_action_probabilities": 2.0e-5,
    "policy_entropies": 2.0e-5,
}

_BACKEND_METADATA_PATHS = (
    "collection.batchedGpuMaskedLogitInference",
    "execution.torchVersion",
    "execution.numpyVersion",
    "execution.device",
    "execution.cudaAvailable",
    "execution.tf32Allowed",
    "execution.cublasWorkspaceConfig",
)
_POLICY_METADATA_PATHS = (
    "fingerprint",
    "policyEntropy",
)
_PUBLIC_OBSERVATION_ARRAYS = (
    "global_features",
    "rank_features",
    "player_features",
    "player_mask",
    "memory_trace_features",
    "history_features",
    "history_mask",
)
_GENERATED_EXACT_ARRAYS = (
    *_PUBLIC_OBSERVATION_ARRAYS,
    "legal_masks",
    "actions",
    "expert_actions",
    "advantages",
    "rewards",
    "dones",
    "valid_masks",
    "privileged_states",
    "raw_returns",
    "baseline_values",
    "raw_advantages",
    "advantage_scales",
    "baseline_tiers",
    "baseline_reference_counts",
    "terminal_chip_awards",
    "forced_masks",
    "source_decision_indices",
    "raw_act_candidate_mean_chips",
    "raw_act_normal_mean_chips",
    "raw_act_group_chip_differences",
    "raw_act_pairwise_rates",
    "raw_act_pairwise_centered_rewards",
    "raw_act_total_rewards",
    "suffix_group_chip_sums",
    "suffix_pairwise_centered_returns",
    "suffix_total_returns",
    "pairwise_candidate_before_normal_counts",
    "pairwise_candidate_normal_comparison_counts",
    "trajectory_ids",
    "trajectory_complete_match_ids",
    "trajectory_player_counts",
    "trajectory_roles",
    "trajectory_acts",
    "trajectory_actor_ids",
    "trajectory_match_indices",
    "trajectory_match_seeds",
    "trajectory_match_clusters",
    "trajectory_finish_places",
    "trajectory_learner_initial_seats",
    "trajectory_initial_player_orders",
    "trajectory_candidate_initial_seats",
    "trajectory_candidate_ids",
    "trajectory_act_player_orders",
    "trajectory_act_finish_orders",
    "trajectory_act_chip_awards_by_physical_id",
    "trajectory_act_candidate_mean_chips",
    "trajectory_act_normal_mean_chips",
    "trajectory_act_group_chip_differences",
    "trajectory_act_pairwise_rates",
    "trajectory_act_pairwise_centered_rewards",
    "trajectory_act_total_rewards",
    "trajectory_suffix_group_chip_sums",
    "trajectory_suffix_pairwise_centered_returns",
    "trajectory_suffix_total_returns",
)
_GENERATED_EXACT_ARRAY_NAMES = tuple(sorted(_GENERATED_EXACT_ARRAYS))
_GENERATED_ARRAY_NAMES = tuple(
    sorted((*_GENERATED_EXACT_ARRAYS, *POLICY_NUMERIC_TOLERANCES, "metadata_json"))
)
_EXECUTION_BACKEND_FIELDS = {
    "torchVersion",
    "numpyVersion",
    "device",
    "cudaAvailable",
    "tf32Allowed",
    "cublasWorkspaceConfig",
}
_REQUIRED_EXECUTION_FIELDS = _EXECUTION_BACKEND_FIELDS | {
    "deterministicAlgorithms",
}
_HEX_CHARACTERS = frozenset("0123456789abcdef")
_CANONICAL_MATCH_COUNTS = {str(player_count): 1 for player_count in range(4, 11)}
_CANONICAL_COMPLETE_MATCH_COUNT = len(_CANONICAL_MATCH_COUNTS)
_CANONICAL_TRAJECTORY_COUNT = _CANONICAL_COMPLETE_MATCH_COUNT * 5
_COMPARATOR_SOURCE = "gpu-training/v4_compare_fixed_match_backends.py"
_REPORT_TOP_LEVEL_FIELDS = {
    "format",
    "version",
    "result",
    "inputs",
    "collectionBinding",
    "modelAndSourceBinding",
    "comparisonContract",
    "exactArrays",
    "policyNumericDifferences",
    "selectedActionOldLogProbabilityDifference",
    "comparatorSourceSha256",
}


@dataclass(frozen=True)
class CalibrationResult:
    output_path: Path
    checksum_path: Path
    report_sha256: str
    cpu_npz_sha256: str
    cuda_npz_sha256: str


@dataclass(frozen=True)
class _ImmutableFileSnapshot:
    label: str
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]

    def recheck_unchanged(self) -> None:
        current = _snapshot_file(self.path, self.label)
        if (
            current.identity != self.identity
            or current.sha256 != self.sha256
            or current.payload != self.payload
        ):
            raise ValueError(f"{self.label} changed after immutable verification")


@dataclass(frozen=True)
class FixedMatchBackendCalibrationVerification:
    """Long-lived binding for one report and its exact ten source files."""

    report_sha256: str
    snapshots: tuple[_ImmutableFileSnapshot, ...]

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return tuple(snapshot.path for snapshot in self.snapshots)

    def recheck_unchanged(self) -> None:
        for snapshot in self.snapshots:
            snapshot.recheck_unchanged()


@dataclass(frozen=True)
class _InputArtifact:
    role: str
    npz_path: Path
    metadata_path: Path
    npz_checksum_path: Path
    metadata_checksum_path: Path
    npz_sha256: str
    metadata_sha256: str
    metadata: dict[str, object]
    array_names: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    snapshots: tuple[_ImmutableFileSnapshot, ...]


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(stat.S_IMODE(value.st_mode)),
    )


def _path_and_descriptor_match(
    path_stat: os.stat_result, descriptor_stat: os.stat_result
) -> bool:
    if int(path_stat.st_size) != int(descriptor_stat.st_size):
        return False
    path_inode = int(path_stat.st_ino)
    descriptor_inode = int(descriptor_stat.st_ino)
    if path_inode and descriptor_inode:
        return (
            int(path_stat.st_dev) == int(descriptor_stat.st_dev)
            and path_inode == descriptor_inode
        )
    return True


def _absolute_without_following_leaf(path_value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path_value)))


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse_flag and attributes & reparse_flag)


def _snapshot_file(path_value: str | Path, label: str) -> _ImmutableFileSnapshot:
    path = _absolute_without_following_leaf(path_value)
    try:
        before_path = path.lstat()
    except OSError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if (
        not stat.S_ISREG(before_path.st_mode)
        or path.is_symlink()
        or _is_reparse_point(before_path)
    ):
        raise ValueError(f"{label} is not a regular non-link file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode) or not _path_and_descriptor_match(
            before_path, before_fd
        ):
            raise ValueError(f"{label} changed while opening")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} disappeared while reading") from error
    if (
        _stat_identity(before_fd) != _stat_identity(after_fd)
        or _stat_identity(before_path) != _stat_identity(after_path)
        or not stat.S_ISREG(after_path.st_mode)
        or path.is_symlink()
        or _is_reparse_point(after_path)
        or not _path_and_descriptor_match(after_path, after_fd)
    ):
        raise ValueError(f"{label} changed while reading")
    payload = b"".join(chunks)
    if len(payload) != int(after_fd.st_size):
        raise ValueError(f"{label} was read incompletely")
    return _ImmutableFileSnapshot(
        label=label,
        path=path,
        payload=payload,
        sha256=digest.hexdigest(),
        identity=_stat_identity(after_path),
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_CHARACTERS for character in value)
    )


def _canonical_json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON with one final newline")
    return value


def _canonical_embedded_metadata(value: np.ndarray, label: str) -> dict[str, object]:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{label} embedded metadata_json is not a scalar string")
    text = str(value.item())
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} embedded metadata_json is invalid") from error
    if not isinstance(metadata, dict) or text.encode("utf-8") != canonical_json_bytes(metadata):
        raise ValueError(f"{label} embedded metadata_json is not canonical")
    return metadata


def _verify_sidecar_snapshot(
    snapshot: _ImmutableFileSnapshot,
    target: _ImmutableFileSnapshot,
    label: str,
) -> None:
    expected = f"{target.sha256}  {target.path.name}\n".encode("ascii")
    if snapshot.payload != expected:
        raise ValueError(f"{label} checksum sidecar is non-canonical or mismatched")


def _comparison_contract() -> dict[str, object]:
    return {
        "failClosed": True,
        "allUnclassifiedArraysRequireExactEquality": True,
        "requiredExactPublicObservationArrays": list(_PUBLIC_OBSERVATION_ARRAYS),
        "requiredExactSelectedActionArray": "actions",
        "requiredExactLegalMaskArray": "legal_masks",
        "requiredExactPrivilegedObservationArray": "privileged_states",
        "requiredExactNormalLabelArray": "expert_actions",
        "allowedPolicyNumericArrays": dict(POLICY_NUMERIC_TOLERANCES),
        "oldActionLogProbabilityMaxAbsTolerance": (
            OLD_ACTION_LOG_PROBABILITY_MAX_ABS
        ),
        "allowedMetadataDifferencePaths": list(
            (*_BACKEND_METADATA_PATHS, *_POLICY_METADATA_PATHS)
        ),
    }


def _validate_policy_entropy_summary(metadata: Mapping[str, object], role: str) -> None:
    summary = metadata.get("policyEntropy")
    expected_keys = {"count", "mean", "std", "min", "max"}
    if not isinstance(summary, Mapping) or set(summary) != expected_keys:
        raise ValueError(f"{role} policyEntropy summary is non-canonical")
    count = summary.get("count")
    sample_count = metadata.get("sampleCount")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or count != sample_count
    ):
        raise ValueError(f"{role} policyEntropy count does not bind every sample")
    values: dict[str, float] = {}
    for name in ("mean", "std", "min", "max"):
        value = summary.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{role} policyEntropy {name} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{role} policyEntropy {name} is not finite")
        values[name] = number
    if (
        values["std"] < 0.0
        or values["min"] > values["max"]
        or not values["min"] <= values["mean"] <= values["max"]
    ):
        raise ValueError(f"{role} policyEntropy summary is inconsistent")


def _validate_execution_role(metadata: Mapping[str, object], role: str) -> None:
    execution = metadata.get("execution")
    collection = metadata.get("collection")
    if not isinstance(execution, Mapping) or not isinstance(collection, Mapping):
        raise ValueError(f"{role} execution metadata is missing")
    if not _REQUIRED_EXECUTION_FIELDS <= set(execution):
        missing = sorted(_REQUIRED_EXECUTION_FIELDS - set(execution))[0]
        raise ValueError(f"{role} execution metadata lacks required field {missing}")
    torch_version = execution.get("torchVersion")
    numpy_version = execution.get("numpyVersion")
    if (
        not isinstance(torch_version, str)
        or not torch_version
        or torch_version != torch_version.strip()
        or not isinstance(numpy_version, str)
        or not numpy_version
        or numpy_version != numpy_version.strip()
    ):
        raise ValueError(f"{role} execution library versions are invalid")
    lowered_torch_version = torch_version.lower()
    if role == "cpu" and "+cu" in lowered_torch_version:
        raise ValueError("cpu execution cannot claim a CUDA torch build")
    if role == "cuda" and "+cu" not in lowered_torch_version:
        raise ValueError("cuda execution must identify its CUDA torch build")
    device = execution.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError(f"{role} execution device is missing")
    try:
        device_type = torch.device(device).type
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} execution device is invalid") from error
    if device_type != role:
        raise ValueError(f"{role} input execution metadata declares {device_type}")
    if execution.get("deterministicAlgorithms") is not True:
        raise ValueError(f"{role} collection did not enable deterministic algorithms")
    if not isinstance(execution.get("cudaAvailable"), bool):
        raise ValueError(f"{role} cudaAvailable execution flag is invalid")
    expected_batched = role == "cuda"
    if collection.get("batchedGpuMaskedLogitInference") is not expected_batched:
        raise ValueError(f"{role} batched GPU inference role is invalid")
    if role == "cpu":
        if (
            execution.get("cudaAvailable") is not False
            or execution.get("tf32Allowed") is not None
            or execution.get("cublasWorkspaceConfig") is not None
        ):
            raise ValueError(
                "cpu cudaAvailable must be false and CUDA numeric settings absent"
            )
    else:
        if (
            execution.get("cudaAvailable") is not True
            or execution.get("tf32Allowed") is not False
            or execution.get("cublasWorkspaceConfig") not in {":16:8", ":4096:8"}
        ):
            raise ValueError("cuda execution lacks deterministic CUDA numeric settings")


def _load_input(path_value: str | Path, role: str) -> _InputArtifact:
    path = _absolute_without_following_leaf(path_value)
    if role not in {"cpu", "cuda"}:
        raise ValueError("calibration role must be cpu or cuda")
    if path.suffix.lower() != ".npz":
        raise FileNotFoundError(f"{role} fixed-match NPZ is missing: {path}")
    metadata_path = Path(f"{path}.metadata.json")
    npz_checksum_path = Path(f"{path}.sha256")
    metadata_checksum_path = Path(f"{metadata_path}.sha256")

    # Capture the full four-file artifact through descriptors.  The returned
    # snapshots retain both bytes and path identity so an identical-byte
    # replacement remains detectable for the collector's entire lifetime.
    npz_snapshot = _snapshot_file(path, f"{role} calibration NPZ")
    npz_checksum_snapshot = _snapshot_file(
        npz_checksum_path, f"{role} calibration NPZ sidecar"
    )
    metadata_snapshot = _snapshot_file(
        metadata_path, f"{role} calibration metadata"
    )
    metadata_checksum_snapshot = _snapshot_file(
        metadata_checksum_path, f"{role} calibration metadata sidecar"
    )
    _verify_sidecar_snapshot(npz_checksum_snapshot, npz_snapshot, role)
    _verify_sidecar_snapshot(
        metadata_checksum_snapshot, metadata_snapshot, f"{role} metadata"
    )
    npz_bytes = npz_snapshot.payload
    metadata_bytes = metadata_snapshot.payload
    npz_sha256 = npz_snapshot.sha256
    metadata_sha256 = metadata_snapshot.sha256
    external = _canonical_json_object(metadata_bytes, f"{role} metadata")

    # This public loader performs the authoritative tensor, privacy, identity,
    # reward, baseline, fixed-match, and fingerprint contract validation.  It
    # accepts paths, so materialize the already-hashed bytes in a private
    # temporary directory instead of reopening the mutable caller path.
    with tempfile.TemporaryDirectory(prefix="v4-calibration-snapshot-") as directory:
        snapshot_path = Path(directory) / path.name
        snapshot_path.write_bytes(npz_bytes)
        dataset = load_v4_dataset_npz(snapshot_path)
        del dataset

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError(f"{role} NPZ contains duplicate array names")
        if "metadata_json" not in archive.files:
            raise ValueError(f"{role} NPZ lacks embedded metadata_json")
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
        embedded = _canonical_embedded_metadata(arrays["metadata_json"], role)
        array_names = tuple(sorted(archive.files))
    expected_external = dict(embedded)
    expected_external["npzSha256"] = npz_sha256
    if canonical_json_bytes(external) != canonical_json_bytes(expected_external):
        raise ValueError(f"{role} external metadata does not exactly bind embedded metadata and NPZ")
    if embedded.get("preparationFormat") != V4_FIXED_MATCH_PPO_PREPARATION_FORMAT:
        raise ValueError(f"{role} input is not a direct fixed-match PPO shard")
    _validate_execution_role(embedded, role)
    _validate_policy_entropy_summary(embedded, role)
    return _InputArtifact(
        role=role,
        npz_path=path,
        metadata_path=metadata_path,
        npz_checksum_path=npz_checksum_path,
        metadata_checksum_path=metadata_checksum_path,
        npz_sha256=npz_sha256,
        metadata_sha256=metadata_sha256,
        metadata=embedded,
        array_names=array_names,
        arrays=arrays,
        snapshots=(
            npz_snapshot,
            npz_checksum_snapshot,
            metadata_snapshot,
            metadata_checksum_snapshot,
        ),
    )


def _assert_input_artifact_unchanged(artifact: _InputArtifact) -> None:
    """Reject a caller path that changed while its snapshot was compared."""

    try:
        for snapshot in artifact.snapshots:
            snapshot.recheck_unchanged()
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"{artifact.role} calibration artifact changed during validation"
        ) from error


def _normalized_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    value = deepcopy(dict(metadata))
    value.pop("fingerprint", None)
    value.pop("policyEntropy", None)
    collection = value.get("collection")
    execution = value.get("execution")
    if not isinstance(collection, dict) or not isinstance(execution, dict):
        raise ValueError("fixed-match metadata lacks mutable collection/execution objects")
    collection.pop("batchedGpuMaskedLogitInference", None)
    for name in _EXECUTION_BACKEND_FIELDS:
        execution.pop(name, None)
    return value


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_exactly_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Require shape, dtype, values, signed-zero, and byte representation."""

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _composite_array_sha256(bindings: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(sorted(bindings.items())))).hexdigest()


def _validate_difference_stats(value: object, label: str) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != {
        "count",
        "differingCount",
        "maxAbsDifference",
        "meanAbsDifference",
    }:
        raise ValueError(f"{label} statistics are non-canonical")
    count = value.get("count")
    differing = value.get("differingCount")
    maximum = value.get("maxAbsDifference")
    mean = value.get("meanAbsDifference")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(differing, bool)
        or not isinstance(differing, int)
        or not 0 <= differing <= count
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or isinstance(mean, bool)
        or not isinstance(mean, (int, float))
    ):
        raise ValueError(f"{label} statistics are invalid")
    maximum_value = float(maximum)
    mean_value = float(mean)
    if (
        not math.isfinite(maximum_value)
        or not math.isfinite(mean_value)
        or maximum_value < 0.0
        or mean_value < 0.0
        or mean_value > maximum_value
        or (count == 0 and (differing != 0 or maximum_value != 0.0 or mean_value != 0.0))
        or ((differing == 0) != (maximum_value == 0.0))
        or ((differing == 0) != (mean_value == 0.0))
    ):
        raise ValueError(f"{label} statistics are invalid")
    return {
        "count": count,
        "differingCount": differing,
        "maxAbsDifference": maximum_value,
        "meanAbsDifference": mean_value,
    }


def _validate_report_input_binding(value: object, role: str) -> None:
    fields_value = {
        "role",
        "npzFileName",
        "npzSha256",
        "metadataFileName",
        "metadataSha256",
        "torchVersion",
        "numpyVersion",
        "device",
        "cudaAvailable",
    }
    if not isinstance(value, Mapping) or set(value) != fields_value:
        raise ValueError(f"calibration report {role} input binding is non-canonical")
    npz_name = value.get("npzFileName")
    metadata_name = value.get("metadataFileName")
    device = value.get("device")
    if (
        value.get("role") != role
        or not isinstance(npz_name, str)
        or Path(npz_name).name != npz_name
        or not npz_name.endswith(".npz")
        or not isinstance(metadata_name, str)
        or Path(metadata_name).name != metadata_name
        or metadata_name != f"{npz_name}.metadata.json"
        or not _is_sha256(value.get("npzSha256"))
        or not _is_sha256(value.get("metadataSha256"))
        or not isinstance(value.get("torchVersion"), str)
        or not value.get("torchVersion")
        or not isinstance(value.get("numpyVersion"), str)
        or not value.get("numpyVersion")
        or not isinstance(device, str)
        or not device
        or value.get("cudaAvailable") is not (role == "cuda")
    ):
        raise ValueError(f"calibration report {role} input binding is invalid")
    try:
        device_type = torch.device(device).type
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"calibration report {role} input device is invalid"
        ) from error
    if device_type != role:
        raise ValueError(f"calibration report {role} input device is invalid")


def _validate_calibration_report_object(
    report: Mapping[str, object],
    *,
    expected_actor_checkpoint_sha256: str,
    expected_bundle_manifest_sha256: str,
    expected_source_hashes: Mapping[str, str],
) -> None:
    """Validate the complete reusable calibration authority fail closed."""

    if set(report) != _REPORT_TOP_LEVEL_FIELDS:
        raise ValueError("calibration report fields are non-canonical")
    if (
        report.get("format") != CALIBRATION_FORMAT
        or report.get("version") != CALIBRATION_VERSION
        or report.get("result") != "pass"
    ):
        raise ValueError("calibration report did not record a canonical pass")
    expected_sources = dict(sorted(expected_source_hashes.items()))
    if (
        not _is_sha256(expected_actor_checkpoint_sha256)
        or not _is_sha256(expected_bundle_manifest_sha256)
        or not expected_sources
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in expected_sources.items()
        )
    ):
        raise ValueError("expected calibration model/source hashes are invalid")
    comparator_sha256 = sha256_file(Path(__file__).resolve())
    if (
        expected_sources.get(_COMPARATOR_SOURCE) != comparator_sha256
        or report.get("comparatorSourceSha256") != comparator_sha256
    ):
        raise ValueError("calibration report does not bind the current comparator source")

    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"cpu", "cuda"}:
        raise ValueError("calibration report input bindings are non-canonical")
    _validate_report_input_binding(inputs["cpu"], "cpu")
    _validate_report_input_binding(inputs["cuda"], "cuda")
    if inputs["cpu"] == inputs["cuda"]:
        raise ValueError("calibration report CPU/CUDA inputs are not distinct")

    collection = report.get("collectionBinding")
    collection_fields = {
        "runNamespace",
        "seedBase",
        "matchCounts",
        "matchStart",
        "matchShardCount",
        "matchShardIndex",
        "identitySha256",
        "completeUnshardedLearnerAssignmentSha256",
        "completeMatchCount",
        "trajectoryCount",
        "sampleCount",
        "trajectoryCoverageSha256",
    }
    if not isinstance(collection, Mapping) or set(collection) != collection_fields:
        raise ValueError("calibration report collection binding is non-canonical")
    sample_count = collection.get("sampleCount")
    if (
        collection.get("matchCounts") != _CANONICAL_MATCH_COUNTS
        or collection.get("matchStart") != 0
        or collection.get("matchShardCount") != 1
        or collection.get("matchShardIndex") != 0
        or collection.get("completeMatchCount") != _CANONICAL_COMPLETE_MATCH_COUNT
        or collection.get("trajectoryCount") != _CANONICAL_TRAJECTORY_COUNT
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or not _is_sha256(
            collection.get("completeUnshardedLearnerAssignmentSha256")
        )
        or not _is_sha256(collection.get("trajectoryCoverageSha256"))
    ):
        raise ValueError(
            "calibration report must cover exactly one unsharded p4-p10 match"
        )
    identity_fields = {
        "runNamespace": collection.get("runNamespace"),
        "seedBase": collection.get("seedBase"),
        "matchCounts": collection.get("matchCounts"),
        "matchStart": collection.get("matchStart"),
        "matchShardCount": collection.get("matchShardCount"),
        "matchShardIndex": collection.get("matchShardIndex"),
    }
    if collection.get("identitySha256") != fixed_match_shard_identity_sha256(
        identity_fields
    ):
        raise ValueError("calibration report shard identitySha256 is invalid")

    model = report.get("modelAndSourceBinding")
    model_fields = {
        "actorCheckpointSha256",
        "bundleManifestSha256",
        "sourceHashes",
        "sourceHashesSha256",
        "normalizedMetadataSha256",
    }
    if not isinstance(model, Mapping) or set(model) != model_fields:
        raise ValueError("calibration report model/source binding is non-canonical")
    if (
        model.get("actorCheckpointSha256") != expected_actor_checkpoint_sha256
        or model.get("bundleManifestSha256") != expected_bundle_manifest_sha256
        or model.get("sourceHashes") != expected_sources
        or model.get("sourceHashesSha256")
        != hashlib.sha256(canonical_json_bytes(expected_sources)).hexdigest()
        or not _is_sha256(model.get("normalizedMetadataSha256"))
    ):
        raise ValueError("calibration report model/source binding does not match")

    if report.get("comparisonContract") != _comparison_contract():
        raise ValueError("calibration report comparison contract is non-canonical")
    exact = report.get("exactArrays")
    if not isinstance(exact, Mapping) or set(exact) != {
        "count",
        "names",
        "sha256ByName",
        "compositeSha256",
    }:
        raise ValueError("calibration report exact-array binding is non-canonical")
    names = exact.get("names")
    hashes = exact.get("sha256ByName")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or names != list(_GENERATED_EXACT_ARRAY_NAMES)
        or exact.get("count") != len(names)
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(names)
        or any(not _is_sha256(value) for value in hashes.values())
        or exact.get("compositeSha256") != _composite_array_sha256(hashes)
    ):
        raise ValueError("calibration report exact-array binding is invalid")

    policy = report.get("policyNumericDifferences")
    if not isinstance(policy, Mapping) or set(policy) != set(POLICY_NUMERIC_TOLERANCES):
        raise ValueError("calibration report policy evidence is non-canonical")
    expected_player_keys = {f"p{player_count}" for player_count in range(4, 11)}
    for name, tolerance in POLICY_NUMERIC_TOLERANCES.items():
        record = policy.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "tolerance",
            "cpuArraySha256",
            "cudaArraySha256",
            "total",
            "byShard",
            "byPlayerCount",
        }:
            raise ValueError(f"calibration report {name} evidence is non-canonical")
        total = _validate_difference_stats(record.get("total"), f"{name} total")
        by_shard = record.get("byShard")
        by_player = record.get("byPlayerCount")
        if (
            record.get("tolerance") != tolerance
            or not _is_sha256(record.get("cpuArraySha256"))
            or not _is_sha256(record.get("cudaArraySha256"))
            or not isinstance(by_shard, Mapping)
            or set(by_shard) != {"0"}
            or not isinstance(by_player, Mapping)
            or set(by_player) != expected_player_keys
            or total["count"] != sample_count
            or float(total["maxAbsDifference"]) > tolerance
        ):
            raise ValueError(f"calibration report {name} evidence is invalid")
        shard_stats = _validate_difference_stats(
            by_shard["0"], f"{name} shard 0"
        )
        if shard_stats != total:
            raise ValueError(f"calibration report {name} shard evidence drifted")
        player_total = 0
        player_differing_total = 0
        player_maximum = 0.0
        player_weighted_difference = 0.0
        for player_count in range(4, 11):
            stats = _validate_difference_stats(
                by_player[f"p{player_count}"], f"{name} p{player_count}"
            )
            if int(stats["count"]) < 1:
                raise ValueError(
                    f"calibration report {name} lacks nonzero p{player_count} evidence"
                )
            player_total += int(stats["count"])
            player_differing_total += int(stats["differingCount"])
            player_maximum = max(
                player_maximum, float(stats["maxAbsDifference"])
            )
            player_weighted_difference += (
                int(stats["count"]) * float(stats["meanAbsDifference"])
            )
        weighted_mean = player_weighted_difference / player_total
        if (
            player_total != sample_count
            or player_differing_total != int(total["differingCount"])
            or player_maximum != float(total["maxAbsDifference"])
            or not math.isclose(
                weighted_mean,
                float(total["meanAbsDifference"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(f"calibration report {name} coverage is inconsistent")
    if report.get("selectedActionOldLogProbabilityDifference") != policy.get(
        "old_action_log_probs"
    ):
        raise ValueError("calibration report selected-action evidence drifted")


def load_verified_fixed_match_backend_calibration(
    report_path: str | Path,
    cpu_npz: str | Path,
    cuda_npz: str | Path,
    *,
    expected_actor_checkpoint_sha256: str,
    expected_bundle_manifest_sha256: str,
    expected_source_hashes: Mapping[str, str],
) -> FixedMatchBackendCalibrationVerification:
    """Strictly load and retain the exact ten-file calibration identity."""

    path = _absolute_without_following_leaf(report_path)
    if path.suffix.lower() != ".json":
        raise FileNotFoundError(f"cross-backend calibration report is missing: {path}")
    report_snapshot = _snapshot_file(path, "calibration report")
    report_sidecar_snapshot = _snapshot_file(
        Path(f"{path}.sha256"), "calibration report sidecar"
    )
    _verify_sidecar_snapshot(
        report_sidecar_snapshot, report_snapshot, "calibration report"
    )
    raw = report_snapshot.payload
    report = _canonical_json_object(raw, "calibration report")
    _validate_calibration_report_object(
        report,
        expected_actor_checkpoint_sha256=expected_actor_checkpoint_sha256,
        expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
        expected_source_hashes=expected_source_hashes,
    )
    cpu = _load_input(cpu_npz, "cpu")
    cuda = _load_input(cuda_npz, "cuda")
    rebuilt = _build_calibration_report(cpu, cuda)
    if canonical_json_bytes(rebuilt) + b"\n" != raw:
        raise ValueError(
            "calibration report does not exactly match the canonical report "
            "recomputed from the supplied CPU/CUDA inputs"
        )
    _assert_input_artifact_unchanged(cpu)
    _assert_input_artifact_unchanged(cuda)
    snapshots = (
        report_snapshot,
        report_sidecar_snapshot,
        *cpu.snapshots,
        *cuda.snapshots,
    )
    if len(snapshots) != 10 or len({snapshot.path for snapshot in snapshots}) != 10:
        raise ValueError("calibration artifact inventory must contain ten unique files")
    verification = FixedMatchBackendCalibrationVerification(
        report_sha256=report_snapshot.sha256,
        snapshots=snapshots,
    )
    verification.recheck_unchanged()
    return verification


def load_fixed_match_backend_calibration_report(
    report_path: str | Path,
    cpu_npz: str | Path,
    cuda_npz: str | Path,
    *,
    expected_actor_checkpoint_sha256: str,
    expected_bundle_manifest_sha256: str,
    expected_source_hashes: Mapping[str, str],
) -> str:
    """Compatibility wrapper returning the verified report digest."""

    return load_verified_fixed_match_backend_calibration(
        report_path,
        cpu_npz,
        cuda_npz,
        expected_actor_checkpoint_sha256=expected_actor_checkpoint_sha256,
        expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
        expected_source_hashes=expected_source_hashes,
    ).report_sha256


def _first_difference(left: np.ndarray, right: np.ndarray) -> str:
    if left.shape != right.shape:
        return f"shape {left.shape} != {right.shape}"
    if left.dtype != right.dtype:
        return f"dtype {left.dtype} != {right.dtype}"
    differing = np.flatnonzero(np.not_equal(left, right).reshape(-1))
    if differing.size == 0:
        return "non-byte-identical array representation"
    flat_index = int(differing[0])
    index = np.unravel_index(flat_index, left.shape)
    return f"first difference at {index}: {left[index]!r} != {right[index]!r}"


def _difference_stats(
    absolute_difference: np.ndarray,
    selection: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(absolute_difference[selection], dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "differingCount": 0,
            "maxAbsDifference": 0.0,
            "meanAbsDifference": 0.0,
        }
    if not np.isfinite(values).all():
        raise ValueError("policy numeric difference contains a non-finite value")
    return {
        "count": int(values.size),
        "differingCount": int(np.count_nonzero(values)),
        "maxAbsDifference": float(values.max()),
        "meanAbsDifference": float(values.mean()),
    }


def _policy_difference_report(
    name: str,
    cpu: np.ndarray,
    cuda: np.ndarray,
    valid: np.ndarray,
    player_counts: np.ndarray,
    shard_index: int,
) -> dict[str, object]:
    if cpu.shape != cuda.shape or cpu.dtype != cuda.dtype or cpu.shape != valid.shape:
        raise ValueError(f"policy numeric array {name} shape/dtype drifted")
    if cpu.dtype.kind not in "fc" or not np.isfinite(cpu).all() or not np.isfinite(cuda).all():
        raise ValueError(f"policy numeric array {name} is not finite floating point")
    invalid = ~valid
    cpu_invalid = np.ascontiguousarray(cpu[invalid])
    cuda_invalid = np.ascontiguousarray(cuda[invalid])
    canonical_zero = np.zeros(cpu_invalid.shape, dtype=cpu.dtype)
    if (
        not _arrays_exactly_equal(cpu_invalid, canonical_zero)
        or not _arrays_exactly_equal(cuda_invalid, canonical_zero)
    ):
        raise ValueError(
            f"policy numeric array {name} invalid suffix must be exact positive zero"
        )
    difference = np.abs(cpu.astype(np.float64) - cuda.astype(np.float64))
    total = _difference_stats(difference, valid)
    tolerance = POLICY_NUMERIC_TOLERANCES[name]
    if float(total["maxAbsDifference"]) > tolerance:
        raise ValueError(
            f"policy numeric array {name} exceeds max abs tolerance "
            f"{tolerance:.8g}: {total['maxAbsDifference']:.8g}"
        )
    by_player_count: dict[str, object] = {}
    for player_count in range(4, 11):
        selection = valid & (player_counts[:, None] == player_count)
        by_player_count[f"p{player_count}"] = _difference_stats(difference, selection)
    return {
        "tolerance": tolerance,
        "cpuArraySha256": _array_sha256(name, cpu),
        "cudaArraySha256": _array_sha256(name, cuda),
        "total": total,
        "byShard": {str(shard_index): total},
        "byPlayerCount": by_player_count,
    }


def _binding_record(artifact: _InputArtifact) -> dict[str, object]:
    execution = artifact.metadata["execution"]
    assert isinstance(execution, Mapping)
    return {
        "role": artifact.role,
        "npzFileName": artifact.npz_path.name,
        "npzSha256": artifact.npz_sha256,
        "metadataFileName": artifact.metadata_path.name,
        "metadataSha256": artifact.metadata_sha256,
        "torchVersion": execution["torchVersion"],
        "numpyVersion": execution["numpyVersion"],
        "device": execution["device"],
        "cudaAvailable": execution["cudaAvailable"],
    }


def _exclusive_publish(output: Path, payload: bytes) -> tuple[Path, str]:
    checksum = Path(f"{output}.sha256")
    output = output.resolve()
    checksum = checksum.resolve()
    if output.parent != checksum.parent:
        raise ValueError("calibration report and checksum must share one directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    for target in (output, checksum):
        if target.exists():
            raise FileExistsError(f"output already exists: {target}")
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = f"{digest}  {output.name}\n".encode("ascii")
    temporary: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for target, value in ((output, payload), (checksum, sidecar)):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".partial", dir=output.parent
            )
            temp = Path(temporary_name)
            temporary[target] = temp
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value)
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
    return checksum, digest


def _build_calibration_report(
    cpu: _InputArtifact,
    cuda: _InputArtifact,
) -> dict[str, object]:
    """Purely derive the canonical report from two authoritative inputs."""

    if cpu.npz_path == cuda.npz_path:
        raise ValueError("cpu and cuda calibration inputs must be distinct files")
    if cpu.array_names != cuda.array_names:
        missing_cpu = sorted(set(cuda.array_names) - set(cpu.array_names))
        missing_cuda = sorted(set(cpu.array_names) - set(cuda.array_names))
        detail = missing_cpu[0] if missing_cpu else missing_cuda[0]
        raise ValueError(f"CPU/CUDA array inventory drifted at {detail}")
    if cpu.array_names != _GENERATED_ARRAY_NAMES:
        expected = set(_GENERATED_ARRAY_NAMES)
        actual = set(cpu.array_names)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = f"missing {missing[0]}" if missing else f"unexpected {extra[0]}"
        raise ValueError(f"calibration input array inventory is non-canonical: {detail}")

    cpu_normalized = _normalized_metadata(cpu.metadata)
    cuda_normalized = _normalized_metadata(cuda.metadata)
    if canonical_json_bytes(cpu_normalized) != canonical_json_bytes(cuda_normalized):
        raise ValueError(
            "CPU/CUDA metadata differs outside the explicitly classified "
            "backend and policy numeric fields"
        )
    model = cpu.metadata.get("modelBinding")
    sources = cpu.metadata.get("sourceHashes")
    shard = cpu.metadata.get("shard")
    if (
        not isinstance(model, Mapping)
        or not _is_sha256(model.get("actorCheckpointSha256"))
        or not _is_sha256(model.get("bundleManifestSha256"))
        or not isinstance(sources, Mapping)
        or not sources
        or any(not isinstance(name, str) or not _is_sha256(value) for name, value in sources.items())
        or not isinstance(shard, Mapping)
    ):
        raise ValueError("actor, manifest, source, or shard hash binding is invalid")
    shard_index = shard.get("matchShardIndex")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise ValueError("fixed-match shard index is invalid")

    exact_bindings: dict[str, str] = {}
    policy_reports: dict[str, object] = {}
    valid_cpu = np.asarray(cpu.arrays["valid_masks"])
    valid_cuda = np.asarray(cuda.arrays["valid_masks"])
    player_cpu = np.asarray(cpu.arrays["trajectory_player_counts"])
    player_cuda = np.asarray(cuda.arrays["trajectory_player_counts"])
    if not _arrays_exactly_equal(valid_cpu, valid_cuda) or not _arrays_exactly_equal(
        player_cpu, player_cuda
    ):
        raise ValueError("CPU/CUDA trajectory coverage drifted")
    for name in cpu.array_names:
        if name == "metadata_json":
            continue
        left = np.asarray(cpu.arrays[name])
        right = np.asarray(cuda.arrays[name])
        if name in POLICY_NUMERIC_TOLERANCES:
            policy_reports[name] = _policy_difference_report(
                name,
                left,
                right,
                valid_cpu,
                player_cpu,
                shard_index,
            )
            continue
        if not _arrays_exactly_equal(left, right):
            raise ValueError(
                f"unclassified or exact array {name} drifted: "
                f"{_first_difference(left, right)}"
            )
        exact_bindings[name] = _array_sha256(name, left)

    trajectory_ids = np.asarray(cpu.arrays["trajectory_ids"])
    trajectory_coverage_sha256 = _array_sha256("trajectory_ids", trajectory_ids)
    complete_match_count = int(cpu.metadata["completeMatchCount"])
    trajectory_count = int(cpu.metadata["trajectoryCount"])
    sample_count = int(valid_cpu.sum())

    source_hashes = dict(sorted((str(name), str(value)) for name, value in sources.items()))
    normalized_metadata_sha256 = hashlib.sha256(
        canonical_json_bytes(cpu_normalized)
    ).hexdigest()
    report: dict[str, object] = {
        "format": CALIBRATION_FORMAT,
        "version": CALIBRATION_VERSION,
        "result": "pass",
        "inputs": {
            "cpu": _binding_record(cpu),
            "cuda": _binding_record(cuda),
        },
        "collectionBinding": {
            "runNamespace": shard.get("runNamespace"),
            "seedBase": shard.get("seedBase"),
            "matchCounts": shard.get("matchCounts"),
            "matchStart": shard.get("matchStart"),
            "matchShardCount": shard.get("matchShardCount"),
            "matchShardIndex": shard_index,
            "identitySha256": shard.get("identitySha256"),
            "completeUnshardedLearnerAssignmentSha256": shard.get(
                "completeUnshardedLearnerAssignmentSha256"
            ),
            "completeMatchCount": complete_match_count,
            "trajectoryCount": trajectory_count,
            "sampleCount": sample_count,
            "trajectoryCoverageSha256": trajectory_coverage_sha256,
        },
        "modelAndSourceBinding": {
            "actorCheckpointSha256": model["actorCheckpointSha256"],
            "bundleManifestSha256": model["bundleManifestSha256"],
            "sourceHashes": source_hashes,
            "sourceHashesSha256": hashlib.sha256(
                canonical_json_bytes(source_hashes)
            ).hexdigest(),
            "normalizedMetadataSha256": normalized_metadata_sha256,
        },
        "comparisonContract": _comparison_contract(),
        "exactArrays": {
            "count": len(exact_bindings),
            "names": sorted(exact_bindings),
            "sha256ByName": dict(sorted(exact_bindings.items())),
            "compositeSha256": _composite_array_sha256(exact_bindings),
        },
        "policyNumericDifferences": dict(sorted(policy_reports.items())),
        "selectedActionOldLogProbabilityDifference": policy_reports[
            "old_action_log_probs"
        ],
        "comparatorSourceSha256": sha256_file(Path(__file__).resolve()),
    }
    _validate_calibration_report_object(
        report,
        expected_actor_checkpoint_sha256=str(model["actorCheckpointSha256"]),
        expected_bundle_manifest_sha256=str(model["bundleManifestSha256"]),
        expected_source_hashes=source_hashes,
    )
    return report


def compare_fixed_match_backends(
    cpu_npz: str | Path,
    cuda_npz: str | Path,
    output_path: str | Path,
) -> CalibrationResult:
    """Validate and exclusively publish one CPU/CUDA calibration report."""

    output = Path(output_path).resolve()
    checksum = Path(f"{output}.sha256")
    if output.suffix.lower() != ".json":
        raise ValueError("calibration output must end in .json")
    if output.exists() or checksum.exists():
        existing = output if output.exists() else checksum
        raise FileExistsError(f"output already exists: {existing}")

    cpu = _load_input(cpu_npz, "cpu")
    cuda = _load_input(cuda_npz, "cuda")
    report = _build_calibration_report(cpu, cuda)
    _assert_input_artifact_unchanged(cpu)
    _assert_input_artifact_unchanged(cuda)
    report_bytes = canonical_json_bytes(report) + b"\n"
    checksum_path, report_sha256 = _exclusive_publish(output, report_bytes)
    try:
        _assert_input_artifact_unchanged(cpu)
        _assert_input_artifact_unchanged(cuda)
    except BaseException:
        for published in (output, checksum_path):
            published.unlink(missing_ok=True)
        raise
    return CalibrationResult(
        output_path=output,
        checksum_path=checksum_path,
        report_sha256=report_sha256,
        cpu_npz_sha256=cpu.npz_sha256,
        cuda_npz_sha256=cuda.npz_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed comparison of identical V4 fixed-match PPO "
            "collections executed on CPU and CUDA"
        )
    )
    parser.add_argument("--cpu-npz", type=Path, required=True)
    parser.add_argument("--cuda-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = compare_fixed_match_backends(
        arguments.cpu_npz,
        arguments.cuda_npz,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "sha256": result.report_sha256,
                "cpuNpzSha256": result.cpu_npz_sha256,
                "cudaNpzSha256": result.cuda_npz_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_FORMAT",
    "CALIBRATION_VERSION",
    "CalibrationResult",
    "OLD_ACTION_LOG_PROBABILITY_MAX_ABS",
    "POLICY_NUMERIC_TOLERANCES",
    "compare_fixed_match_backends",
    "load_fixed_match_backend_calibration_report",
]
