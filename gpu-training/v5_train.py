from __future__ import annotations

"""Production one-epoch MAPPO/PPO trainer for DALMUTI V5.

The public Actor and centralized critic deliberately cross different data
boundaries.  Actor batches are decoded only from the actor mmap partition;
``privileged_states`` are read only immediately before the critic forward.
Every behavior-policy row is replayed in deterministic FP32 before the first
update, and the complete policy is replayed again after the epoch for the
sealed KL/clip/entropy hard gates.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_dataset import (
    V5_INDEX_FORMAT,
    V5_SHARD_FORMAT,
    V5TrainingShard,
    load_v5_index_manifest,
    load_v5_training_shard,
)
from v5_export import (
    canonical_json_bytes,
    export_v5_actor_bundle,
    load_v5_actor_bundle,
    sha256_file,
    tensor_state_sha256,
    v5_actor_bundle_digests,
    verify_v5_actor_bundle,
)
from v5_model import (
    V5_POLICY_NUMERICS_SHA256,
    V5CentralStateValueCritic,
    V5ActorConfig,
    V5CriticConfig,
    V5PublicActor,
    assert_actor_critic_parameter_isolation,
    canonical_v5_policy_numerics_contract,
    configure_v5_policy_numerics,
)
from v5_public import actor_batch_from_packed_arrays


V5_TRAINING_FORMAT = "dalmuti-v5-mappo-training-result"
V5_TRAINING_VERSION = 1
V5_CRITIC_FORMAT = "dalmuti-v5-central-state-value-critic"
V5_CRITIC_VERSION = 1
V5_MODEL_PAIR_FORMAT = "dalmuti-v5-actor-critic-pair"
V5_MODEL_PAIR_VERSION = 1
V5_OPTIMIZER_FORMAT = "dalmuti-v5-mappo-optimizer"
V5_CHECKPOINT_FORMAT = "dalmuti-v5-mappo-resume-checkpoint"
V5_BEHAVIOR_LOG_PROBABILITY_ABSOLUTE_TOLERANCE = 2.0e-5
V5_MAXIMUM_APPROX_KL = 0.020
V5_MAXIMUM_CLIP_FRACTION = 0.25
V5_MINIMUM_ENTROPY_RETENTION = 0.70
V5_CRITIC_PER_PLAYER_HUBER_REGRESSION_FACTOR = 1.10
V5_CRITIC_HUBER_ZERO_EPSILON = 1.0e-8
V5_CRITIC_EXPLAINED_VARIANCE_REGRESSION_TOLERANCE = 0.02
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_LEARNING_ARRAYS = {
    "advantages",
    "returns",
    "policy_mask",
    "value_mask",
    "policy_loss_weights",
    "value_loss_weights",
}


def derive_v5_initialization_seeds(seed: int) -> dict[str, int]:
    """Domain-separate one required family seed into Actor/critic RNG seeds."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFF_FFFF:
        raise ValueError("initialization seed must be an explicit uint32 integer")

    def derive(label: str) -> int:
        digest = hashlib.sha256(
            b"DALMUTI-V5-INITIALIZATION\0"
            + seed.to_bytes(4, "little")
            + label.encode("ascii")
        ).digest()
        return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF

    actor_seed = derive("actor")
    critic_seed = derive("critic")
    if actor_seed == critic_seed:
        raise RuntimeError("V5 initialization seed domains collided")
    return {
        "initializationSeed": seed,
        "actorInitializationSeed": actor_seed,
        "criticInitializationSeed": critic_seed,
    }


@dataclass(frozen=True)
class V5TrainingConfig:
    seed: int = 750_000_001
    epochs: int = 1
    microbatch_size: int = 8
    gradient_accumulation: int = 4
    critic_batch_size: int = 256
    audit_batch_size: int = 64
    actor_learning_rate: float = 1.0e-5
    critic_learning_rate: float = 3.0e-5
    weight_decay: float = 0.01
    clip_ratio: float = 0.15
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.005
    normal_auxiliary_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    huber_delta: float = 1.0
    normalize_advantages: bool = True
    use_amp: bool = True
    require_all_player_counts: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 0xFFFF_FFFF:
            raise ValueError("seed must be a uint32 integer")
        if self.epochs != 1:
            raise ValueError("V5 on-policy training requires exactly one epoch")
        for name, maximum in (
            ("microbatch_size", 64),
            ("gradient_accumulation", 1024),
            ("critic_batch_size", 1024),
            ("audit_batch_size", 4096),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if (self.microbatch_size, self.gradient_accumulation) not in {
            (8, 4),
            (16, 2),
            (32, 1),
        }:
            raise ValueError(
                "V5 Actor physical batching must be 8x4, 16x2, or 32x1 "
                "for effective batch 32"
            )
        if self.critic_batch_size not in {256, 512, 1024}:
            raise ValueError("critic_batch_size must be one of 256, 512, or 1024")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "max_gradient_norm",
            "huber_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "weight_decay",
            "value_coefficient",
            "entropy_coefficient",
            "normal_auxiliary_coefficient",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.clip_ratio) != 0.15:
            raise ValueError("V5 PPO requires the canonical clip ratio 0.15")
        if (
            type(self.normalize_advantages) is not bool
            or type(self.use_amp) is not bool
            or type(self.require_all_player_counts) is not bool
        ):
            raise ValueError("boolean training switches must be exact bool values")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Source:
    identity_sha256: str
    shards: tuple[V5TrainingShard, ...]
    counts: tuple[int, ...]
    offsets: tuple[int, ...]

    @property
    def decision_count(self) -> int:
        return self.offsets[-1]

    def close(self) -> None:
        for shard in self.shards:
            shard.close()


@dataclass(frozen=True)
class _GlobalWeightContract:
    policy_counts: Mapping[int, int]
    value_counts: Mapping[int, int]
    policy_weights: Mapping[int, float]
    value_weights: Mapping[int, float]
    policy_mass: float
    value_mass: float

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "dalmuti-v5-global-equal-player-count-loss-mass",
            "playerCounts": sorted(self.policy_counts),
            "policy": {
                str(player): {
                    "eligibleRows": self.policy_counts[player],
                    "runtimeFloat32Weight": self.policy_weights[player],
                    "weightMass": self.policy_counts[player]
                    * self.policy_weights[player],
                }
                for player in self.policy_counts
            },
            "policyTotalWeightMass": self.policy_mass,
            "value": {
                str(player): {
                    "eligibleRows": self.value_counts[player],
                    "runtimeFloat32Weight": self.value_weights[player],
                    "weightMass": self.value_counts[player]
                    * self.value_weights[player],
                }
                for player in self.value_counts
            },
            "valueTotalWeightMass": self.value_mass,
        }


def _strict_canonical_json(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"{label} contains duplicate key {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid canonical JSON") from error
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
    if not isinstance(value, dict) or expected != raw:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _canonical_metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    raw = canonical_json_bytes(dict(value or {}))
    result = json.loads(raw.decode("ascii"))
    if not isinstance(result, dict):
        raise TypeError("metadata must canonicalize to an object")
    return result


def _tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in module.state_dict().items()
    }


def _torch_bytes(value: object) -> bytes:
    output = io.BytesIO()
    torch.save(value, output)
    return output.getvalue()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _critic_payload(
    critic: V5CentralStateValueCritic,
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    if type(critic) is not V5CentralStateValueCritic:
        raise TypeError("critic checkpoint requires exactly V5CentralStateValueCritic")
    state = _tensor_state(critic)
    return {
        "config": critic.config.to_dict(),
        "deployExportAllowed": False,
        "format": V5_CRITIC_FORMAT,
        "metadata": _canonical_metadata(metadata),
        "stateDict": state,
        "tensorStateSha256": tensor_state_sha256(state),
        "version": V5_CRITIC_VERSION,
    }


def export_v5_critic_checkpoint(
    critic: V5CentralStateValueCritic,
    checkpoint_path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Exclusively write the training-only centralized critic."""

    path = Path(checkpoint_path).resolve()
    payload = _critic_payload(critic, metadata)
    _write_exclusive(path, _torch_bytes(payload))
    return {
        "criticSha256": sha256_file(path),
        "tensorStateSha256": str(payload["tensorStateSha256"]),
    }


def load_v5_critic_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[V5CentralStateValueCritic, dict[str, object]]:
    path = Path(checkpoint_path).resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("V5 critic checkpoint could not be safely loaded") from error
    expected = {
        "config",
        "deployExportAllowed",
        "format",
        "metadata",
        "stateDict",
        "tensorStateSha256",
        "version",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("format") != V5_CRITIC_FORMAT
        or payload.get("version") != V5_CRITIC_VERSION
        or payload.get("deployExportAllowed") is not False
    ):
        raise ValueError("unsupported V5 critic checkpoint contract")
    raw_config = payload.get("config")
    expected_config_fields = {field.name for field in fields(V5CriticConfig)}
    if not isinstance(raw_config, dict) or set(raw_config) != expected_config_fields:
        raise ValueError("V5 critic configuration fields drifted")
    try:
        config = V5CriticConfig(**raw_config)
    except (TypeError, ValueError) as error:
        raise ValueError("V5 critic configuration is invalid") from error
    if config.to_dict() != raw_config:
        raise ValueError("V5 critic configuration is non-canonical")
    if _canonical_metadata(payload.get("metadata")) != payload.get("metadata"):
        raise ValueError("V5 critic metadata is non-canonical")
    state = payload.get("stateDict")
    if not isinstance(state, dict) or not state:
        raise ValueError("V5 critic tensor state is missing")
    actual_state_sha = tensor_state_sha256(state)
    if payload.get("tensorStateSha256") != actual_state_sha:
        raise ValueError("V5 critic tensor-state checksum does not match")
    critic = V5CentralStateValueCritic(config)
    try:
        critic.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("V5 critic tensor state does not match its configuration") from error
    if tensor_state_sha256(_tensor_state(critic)) != actual_state_sha:
        raise ValueError("V5 critic tensor state changed while loading")
    return critic.eval(), payload


def _read_ascii_canonical_object(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid ASCII JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _model_pair_identity(
    root: Path,
    actor_bundle_name: str,
    critic_checkpoint_name: str,
) -> dict[str, object]:
    if actor_bundle_name != "actor-bundle" or critic_checkpoint_name != "critic.pt":
        raise ValueError("V5 model pair requires canonical actor-bundle/critic.pt paths")
    actor_bundle = root / actor_bundle_name
    critic_path = root / critic_checkpoint_name
    actor_digests = v5_actor_bundle_digests(actor_bundle)
    _, actor_manifest = load_v5_actor_bundle(actor_bundle)
    _, critic_payload = load_v5_critic_checkpoint(critic_path)
    seeds = _validate_initialization_seed_binding(actor_manifest, critic_payload)
    critic_state_sha = _require_sha(
        critic_payload.get("tensorStateSha256"), "critic tensor-state SHA"
    )
    if actor_digests.get("policyNumericsSha256") != V5_POLICY_NUMERICS_SHA256:
        raise ValueError("paired Actor policy numerics binding drifted")
    if actor_digests.get("publicContractSha256") != V5_PUBLIC_CONTRACT_SHA256:
        raise ValueError("paired Actor public contract binding drifted")
    return {
        "actor": {
            "bundlePath": actor_bundle_name,
            "checkpointSha256": _require_sha(
                actor_digests["actorSha256"], "Actor checkpoint SHA"
            ),
            "manifestSha256": _require_sha(
                actor_digests["manifestSha256"], "Actor manifest SHA"
            ),
            "tensorStateSha256": _require_sha(
                actor_digests["tensorStateSha256"], "Actor tensor-state SHA"
            ),
        },
        "critic": {
            "checkpointPath": critic_checkpoint_name,
            "checkpointSha256": _require_sha(
                sha256_file(critic_path), "critic checkpoint SHA"
            ),
            "tensorStateSha256": critic_state_sha,
        },
        "initializationSeeds": seeds,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
    }


def _model_pair_id(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(
        b"DALMUTI-V5-ACTOR-CRITIC-PAIR\0" + canonical_json_bytes(dict(identity))
    ).hexdigest()


def _flat_pair_record(
    manifest: Mapping[str, object], manifest_sha: str
) -> dict[str, object]:
    identity = manifest["identity"]
    assert isinstance(identity, Mapping)
    actor = identity["actor"]
    critic = identity["critic"]
    assert isinstance(actor, Mapping) and isinstance(critic, Mapping)
    return {
        "actorManifestSha256": actor["manifestSha256"],
        "actorSha256": actor["checkpointSha256"],
        "actorTensorStateSha256": actor["tensorStateSha256"],
        "criticSha256": critic["checkpointSha256"],
        "criticTensorStateSha256": critic["tensorStateSha256"],
        "initializationSeeds": identity["initializationSeeds"],
        "pairId": manifest["pairId"],
        "pairManifestSha256": manifest_sha,
        "policyNumericsSha256": identity["policyNumericsSha256"],
        "publicContractSha256": identity["publicContractSha256"],
    }


def publish_v5_model_pair_manifest(
    output_directory: str | Path,
    *,
    actor_bundle_name: str = "actor-bundle",
    critic_checkpoint_name: str = "critic.pt",
) -> dict[str, object]:
    root = Path(output_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError("V5 model-pair output directory is missing")
    manifest_path = root / "model-pair.json"
    sidecar_path = root / "model-pair.json.sha256"
    if manifest_path.exists() or sidecar_path.exists():
        raise FileExistsError("V5 model-pair manifest is immutable")
    identity = _model_pair_identity(root, actor_bundle_name, critic_checkpoint_name)
    manifest = {
        "format": V5_MODEL_PAIR_FORMAT,
        "identity": identity,
        "pairId": _model_pair_id(identity),
        "version": V5_MODEL_PAIR_VERSION,
    }
    raw = canonical_json_bytes(manifest)
    _write_exclusive(manifest_path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        _write_exclusive(
            sidecar_path, f"{digest}  model-pair.json\n".encode("ascii")
        )
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    return _flat_pair_record(manifest, digest)


def verify_v5_model_pair(output_directory: str | Path) -> dict[str, object]:
    supplied = Path(output_directory).resolve()
    if supplied.is_file():
        if supplied.name != "model-pair.json":
            raise ValueError("V5 model-pair file must be named model-pair.json")
        root = supplied.parent
    else:
        root = supplied
    manifest_path = root / "model-pair.json"
    manifest = _read_ascii_canonical_object(manifest_path, "V5 model-pair manifest")
    if set(manifest) != {"format", "identity", "pairId", "version"} or (
        manifest.get("format") != V5_MODEL_PAIR_FORMAT
        or manifest.get("version") != V5_MODEL_PAIR_VERSION
    ):
        raise ValueError("unsupported V5 model-pair manifest")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
        "actor",
        "critic",
        "initializationSeeds",
        "policyNumericsSha256",
        "publicContractSha256",
    }:
        raise ValueError("V5 model-pair identity fields drifted")
    actor = identity.get("actor")
    critic = identity.get("critic")
    if not isinstance(actor, dict) or set(actor) != {
        "bundlePath",
        "checkpointSha256",
        "manifestSha256",
        "tensorStateSha256",
    } or not isinstance(critic, dict) or set(critic) != {
        "checkpointPath",
        "checkpointSha256",
        "tensorStateSha256",
    }:
        raise ValueError("V5 model-pair Actor/critic records drifted")
    rebuilt = _model_pair_identity(
        root,
        str(actor.get("bundlePath")),
        str(critic.get("checkpointPath")),
    )
    if identity != rebuilt or manifest.get("pairId") != _model_pair_id(rebuilt):
        raise ValueError("V5 model-pair files or pair id no longer match")
    raw_sha = sha256_file(manifest_path)
    expected_sidecar = f"{raw_sha}  model-pair.json\n".encode("ascii")
    if (root / "model-pair.json.sha256").read_bytes() != expected_sidecar:
        raise ValueError("V5 model-pair manifest sidecar does not match")
    return _flat_pair_record(manifest, raw_sha)


load_verified_v5_behavior_pair = verify_v5_model_pair


def _open_source(dataset_path: str | Path) -> _Source:
    root = Path(dataset_path).resolve()
    manifest = _strict_canonical_json(root / "manifest.json", "V5 dataset manifest")
    format_name = manifest.get("format")
    if format_name == V5_SHARD_FORMAT:
        paths = (root,)
    elif format_name == V5_INDEX_FORMAT:
        index = load_v5_index_manifest(root)
        paths = index.shard_paths
    else:
        raise ValueError("dataset is neither a V5 immutable shard nor zero-copy index")
    shards: list[V5TrainingShard] = []
    try:
        for path in paths:
            shard = load_v5_training_shard(path)
            missing = _REQUIRED_LEARNING_ARRAYS - set(shard.actor.arrays)
            if missing:
                shard.close()
                raise ValueError(
                    f"V5 training shard omitted learning array {sorted(missing)[0]}"
                )
            shards.append(shard)
    except Exception:
        for shard in shards:
            shard.close()
        raise
    counts = tuple(shard.actor.decision_count for shard in shards)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return _Source(
        sha256_file(root / "manifest.json"),
        tuple(shards),
        counts,
        tuple(offsets),
    )


def _global_weight_contract(
    source: _Source, *, require_all_player_counts: bool
) -> _GlobalWeightContract:
    policy_counts = {player: 0 for player in range(4, 11)}
    value_counts = {player: 0 for player in range(4, 11)}
    for shard in source.shards:
        arrays = shard.actor.arrays
        for start in range(0, shard.actor.decision_count, 65_536):
            _validate_forced_audit_semantics(
                arrays,
                np.arange(
                    start,
                    min(start + 65_536, shard.actor.decision_count),
                    dtype=np.int64,
                ),
            )
        counts = np.asarray(arrays["global_codes"][:, 1], dtype=np.int64)
        policy_mask = np.asarray(arrays["policy_mask"], dtype=np.bool_)
        value_mask = np.asarray(arrays["value_mask"], dtype=np.bool_)
        forced = np.asarray(arrays["forced"], dtype=np.bool_)
        stored_policy = np.asarray(arrays["policy_loss_weights"], dtype=np.float32)
        stored_value = np.asarray(arrays["value_loss_weights"], dtype=np.float32)
        if (
            not np.array_equal(policy_mask, ~forced)
            or not value_mask.all()
            or not np.array_equal(stored_policy > 0.0, policy_mask)
            or not np.array_equal(stored_value > 0.0, value_mask)
        ):
            raise ValueError("V5 stored eligibility/weight masks are inconsistent")
        for player in range(4, 11):
            policy_counts[player] += int(np.sum(policy_mask & (counts == player)))
            value_counts[player] += int(np.sum(value_mask & (counts == player)))
    represented_policy = tuple(
        player for player, count in policy_counts.items() if count > 0
    )
    represented_value = tuple(
        player for player, count in value_counts.items() if count > 0
    )
    expected = tuple(range(4, 11))
    if require_all_player_counts and (
        represented_policy != expected or represented_value != expected
    ):
        raise ValueError(
            "global V5 training corpus must represent policy and value rows for p4..p10"
        )
    if not represented_policy or represented_policy != represented_value:
        raise ValueError("global V5 policy/value player-count strata disagree")
    policy_counts = {player: policy_counts[player] for player in represented_policy}
    value_counts = {player: value_counts[player] for player in represented_value}

    def weights(counts: Mapping[int, int]) -> tuple[dict[int, float], float]:
        eligible = sum(counts.values())
        target = eligible / len(counts)
        runtime = {
            player: float(np.float32(target / count))
            for player, count in counts.items()
        }
        total_mass = math.fsum(counts[player] * runtime[player] for player in counts)
        return runtime, total_mass

    policy_weights, policy_mass = weights(policy_counts)
    value_weights, value_mass = weights(value_counts)
    # Each represented p stratum must carry equal aggregate mass to tight
    # float32 runtime tolerance; single-p shard-local weights are never reused.
    for counts, runtime, label in (
        (policy_counts, policy_weights, "policy"),
        (value_counts, value_weights, "value"),
    ):
        masses = [counts[player] * runtime[player] for player in counts]
        if max(masses) - min(masses) > max(2.0e-5, max(masses) * 2.0e-7):
            raise RuntimeError(f"global equal-p {label} weight mass could not be represented")
    return _GlobalWeightContract(
        policy_counts,
        value_counts,
        policy_weights,
        value_weights,
        policy_mass,
        value_mass,
    )


def _row_weights(
    player_counts: np.ndarray,
    lookup: Mapping[int, float],
) -> np.ndarray:
    result = np.zeros(player_counts.shape, dtype=np.float32)
    for player, weight in lookup.items():
        result[player_counts == player] = np.float32(weight)
    if np.any(result <= 0.0):
        raise ValueError("global loss weighting encountered an unbound player count")
    return result


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _verify_behavior_bindings(
    source: _Source,
    actor_bundle: Path,
    critic_checkpoint: Path,
    model_pair: Mapping[str, object],
) -> dict[str, str]:
    actor_digests = v5_actor_bundle_digests(actor_bundle)
    expected = {
        "behaviorActorSha256": _require_sha(actor_digests["actorSha256"], "Actor SHA"),
        "behaviorActorManifestSha256": _require_sha(
            actor_digests["manifestSha256"], "Actor manifest SHA"
        ),
        "behaviorCriticSha256": _require_sha(
            sha256_file(critic_checkpoint), "critic SHA"
        ),
        "behaviorModelPairId": _require_sha(model_pair.get("pairId"), "model pair id"),
        "behaviorModelPairManifestSha256": _require_sha(
            model_pair.get("pairManifestSha256"), "model pair manifest SHA"
        ),
    }
    for shard in source.shards:
        metadata = shard.actor.manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("V5 shard metadata is missing")
        for name, digest in expected.items():
            if metadata.get(name) != digest:
                raise ValueError(f"V5 behavior binding mismatch: {name}")
        if metadata.get("publicContractSha256") not in (
            None,
            V5_PUBLIC_CONTRACT_SHA256,
        ):
            raise ValueError("V5 shard public contract binding drifted")
        if metadata.get("policyNumericsSha256") not in (
            None,
            V5_POLICY_NUMERICS_SHA256,
        ):
            raise ValueError("V5 shard policy numerics binding drifted")
    return expected


def _validate_gpu_memory_preflight_binding(
    value: Mapping[str, object] | None,
    *,
    source: _Source,
    model_pair: Mapping[str, object],
    config: V5TrainingConfig,
    device: torch.device,
    allow_unadmitted_cpu: bool,
) -> dict[str, object] | None:
    """Bind the workflow-verified, actual-corpus GPU admission report."""

    if value is None:
        if allow_unadmitted_cpu and device.type == "cpu":
            return None
        raise ValueError(
            "V5 training requires a bound GPU memory preflight; only an "
            "explicit unadmitted CPU smoke/test run may omit it"
        )
    if not isinstance(value, Mapping):
        raise TypeError("gpu_memory_preflight must be a mapping or None")
    record = _canonical_metadata(value)
    required = {
        "config",
        "datasetIdentitySha256",
        "device",
        "format",
        "modelPairId",
        "policyNumericsSha256",
        "reportSha256",
        "version",
    }
    if set(record) != required:
        raise ValueError("GPU memory preflight binding fields drifted")
    if (
        record.get("format") != "dalmuti-v5-gpu-memory-admission-binding"
        or record.get("version") != 1
    ):
        raise ValueError("GPU memory preflight binding format is unsupported")
    expected_sha_fields = {
        "datasetIdentitySha256": source.identity_sha256,
        "modelPairId": _require_sha(model_pair.get("pairId"), "model pair id"),
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
    }
    for name, expected in expected_sha_fields.items():
        if _require_sha(record.get(name), name) != expected:
            raise ValueError(f"GPU memory preflight {name} mismatch")
    _require_sha(record.get("reportSha256"), "GPU preflight report SHA")
    if record.get("device") != str(device):
        raise ValueError("GPU memory preflight device mismatch")
    preflight_config = record.get("config")
    if not isinstance(preflight_config, dict) or set(preflight_config) != {
        "audit_batch_size",
        "critic_batch_size",
        "gradient_accumulation",
        "microbatch_size",
    }:
        raise ValueError("GPU memory preflight config fields drifted")
    expected_config = {
        "audit_batch_size": config.audit_batch_size,
        "critic_batch_size": config.critic_batch_size,
        "gradient_accumulation": config.gradient_accumulation,
        "microbatch_size": config.microbatch_size,
    }
    for name, expected in expected_config.items():
        actual = preflight_config.get(name)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValueError(f"GPU memory preflight config mismatch: {name}")
    return record


def _validate_initialization_seed_binding(
    actor_manifest: Mapping[str, object],
    critic_payload: Mapping[str, object],
) -> dict[str, int]:
    actor_metadata = actor_manifest.get("metadata")
    critic_metadata = critic_payload.get("metadata")
    if not isinstance(actor_metadata, Mapping) or not isinstance(
        critic_metadata, Mapping
    ):
        raise ValueError("V5 initial model metadata omitted initialization seeds")
    names = (
        "initializationSeed",
        "actorInitializationSeed",
        "criticInitializationSeed",
    )
    seeds: dict[str, int] = {}
    for name in names:
        actor_value = actor_metadata.get(name)
        critic_value = critic_metadata.get(name)
        if (
            isinstance(actor_value, bool)
            or not isinstance(actor_value, int)
            or actor_value < 0
            or actor_value != critic_value
        ):
            raise ValueError(f"V5 Actor/critic initialization seed mismatch: {name}")
        seeds[name] = actor_value
    if derive_v5_initialization_seeds(seeds["initializationSeed"]) != seeds:
        raise ValueError("V5 initialization seed derivation binding drifted")
    return seeds


def _selected_policy_statistics(
    actor: V5PublicActor,
    actor_arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected log-p, entropy, and Normal auxiliary NLL."""

    public = actor_batch_from_packed_arrays(actor_arrays, indices, device)
    normal = torch.from_numpy(
        np.ascontiguousarray(actor_arrays["normal_actions"][indices])
    ).to(device=device, dtype=torch.long)
    actions = torch.from_numpy(
        np.ascontiguousarray(actor_arrays["actions"][indices])
    ).to(device=device, dtype=torch.long)
    output = actor.forward_packed_batch(public, normal)
    matches = (output.action_indices == actions.unsqueeze(1)) & output.action_mask
    if not bool(matches.sum(dim=-1).eq(1).all()):
        raise ValueError("selected action is absent from its packed legal actions")
    selected_positions = matches.to(torch.int64).argmax(dim=-1)
    log_probabilities = F.log_softmax(output.logits.float(), dim=-1)
    selected = log_probabilities.gather(
        1, selected_positions.unsqueeze(1)
    ).squeeze(1)
    probabilities = log_probabilities.exp().masked_fill(~output.action_mask, 0.0)
    entropy = -(
        probabilities
        * log_probabilities.masked_fill(~output.action_mask, 0.0)
    ).sum(dim=-1)
    normal_matches = (
        output.action_indices == output.normal_actions.unsqueeze(1)
    ) & output.action_mask
    if not bool(normal_matches.sum(dim=-1).eq(1).all()):
        raise ValueError("Normal action is absent from packed legal actions")
    normal_positions = normal_matches.to(torch.int64).argmax(dim=-1)
    auxiliary_nll = F.cross_entropy(
        output.normal_auxiliary_logits.float(),
        normal_positions,
        reduction="none",
    )
    for label, value in (
        ("selected log probabilities", selected),
        ("policy entropy", entropy),
        ("Normal auxiliary loss", auxiliary_nll),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"V5 Actor produced non-finite {label}")
    return selected, entropy, auxiliary_nll


def _weighted_metric(rows: np.ndarray, weights: np.ndarray) -> float:
    mass = math.fsum(float(value) for value in weights)
    if mass <= 0.0:
        raise ValueError("policy metric weight mass must be positive")
    return math.fsum(
        float(value) * float(weight)
        for value, weight in zip(rows, weights, strict=True)
    ) / mass


def _validate_forced_audit_semantics(
    actor_arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> int:
    """Prove singleton rows have policy-independent zero log probability."""

    forced = np.asarray(actor_arrays["forced"][indices], dtype=np.bool_)
    policy_mask = np.asarray(actor_arrays["policy_mask"][indices], dtype=np.bool_)
    if not np.array_equal(policy_mask, ~forced):
        raise ValueError("forced rows must be policy-ineligible exactly")
    selected = indices[forced]
    if selected.size == 0:
        return 0
    old = np.asarray(actor_arrays["old_log_probs"][selected], dtype=np.float32)
    if not np.array_equal(old, np.zeros(old.shape, dtype=np.float32)):
        raise ValueError("forced rows must store exact zero old log probability")
    bits = np.asarray(actor_arrays["legal_action_bits"][selected], dtype=np.uint8)
    legal_counts = np.unpackbits(bits, axis=1, bitorder="little").sum(axis=1)
    actions = np.asarray(actor_arrays["actions"][selected], dtype=np.int64)
    normal = np.asarray(actor_arrays["normal_actions"][selected], dtype=np.int64)

    def legal_at(values: np.ndarray) -> np.ndarray:
        return (
            bits[np.arange(len(values)), values // 8]
            & (np.uint8(1) << (values % 8).astype(np.uint8))
        ) != 0

    if (
        not np.all(legal_counts == 1)
        or not np.all(legal_at(actions))
        or not np.all(legal_at(normal))
        or not np.array_equal(actions, normal)
    ):
        raise ValueError(
            "forced rows must select the same sole legal Actor and Normal action"
        )
    probabilities = actor_arrays.get("selected_action_probabilities")
    if probabilities is not None and not np.all(
        np.asarray(probabilities[selected], dtype=np.float32) == np.float32(1.0)
    ):
        raise ValueError("forced rows must store selected probability one")
    return int(selected.size)


def _audit_policy(
    actor: V5PublicActor,
    source: _Source,
    weight_contract: _GlobalWeightContract,
    *,
    device: torch.device,
    batch_size: int,
    clip_ratio: float,
) -> dict[str, object]:
    actor_was_training = actor.training
    actor.eval()
    all_log_ratios: list[np.ndarray] = []
    all_entropies: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    maximum_replay_error = 0.0
    replay_error_sum = 0.0
    replay_rows = 0
    actor_forward_rows = 0
    forced_semantic_rows = 0
    try:
        with torch.no_grad():
            for shard in source.shards:
                arrays = shard.actor.arrays
                decision_count = shard.actor.decision_count
                for start in range(0, decision_count, 65_536):
                    forced_semantic_rows += _validate_forced_audit_semantics(
                        arrays,
                        np.arange(
                            start,
                            min(start + 65_536, decision_count),
                            dtype=np.int64,
                        ),
                    )
                replay_rows += decision_count
                all_policy_indices = np.flatnonzero(
                    np.asarray(arrays["policy_mask"], dtype=np.bool_)
                ).astype(np.int64, copy=False)
                for start in range(0, len(all_policy_indices), batch_size):
                    policy_indices = all_policy_indices[
                        start : start + batch_size
                    ]
                    with torch.amp.autocast(device.type, enabled=False):
                        selected, entropy, _ = _selected_policy_statistics(
                            actor, arrays, policy_indices, device
                        )
                    current = selected.detach().cpu().to(torch.float64).numpy()
                    old = np.asarray(
                        arrays["old_log_probs"][policy_indices], dtype=np.float64
                    )
                    log_ratio = current - old
                    if not np.isfinite(log_ratio).all():
                        raise ValueError("full policy replay produced non-finite log ratios")
                    absolute = np.abs(log_ratio)
                    maximum_replay_error = max(
                        maximum_replay_error, float(absolute.max(initial=0.0))
                    )
                    replay_error_sum += float(absolute.sum(dtype=np.float64))
                    actor_forward_rows += len(policy_indices)
                    global_codes = np.asarray(arrays["global_codes"][policy_indices])
                    all_log_ratios.append(log_ratio)
                    all_entropies.append(
                        entropy.detach().cpu().to(torch.float64).numpy()
                    )
                    selected_counts = np.asarray(
                        global_codes[:, 1], dtype=np.uint8
                    )
                    all_weights.append(
                        _row_weights(
                            selected_counts, weight_contract.policy_weights
                        ).astype(np.float64)
                    )
                    all_counts.append(selected_counts)
    finally:
        actor.train(actor_was_training)
    if replay_rows != source.decision_count:
        raise RuntimeError("full policy replay did not visit every decision row")
    log_ratios = np.concatenate(all_log_ratios)
    entropies = np.concatenate(all_entropies)
    weights = np.concatenate(all_weights)
    player_counts = np.concatenate(all_counts)
    if (
        actor_forward_rows + forced_semantic_rows != replay_rows
        or log_ratios.size != actor_forward_rows
    ):
        raise RuntimeError("V5 audit Actor-forward/forced row accounting drifted")
    if log_ratios.size == 0 or np.any(weights <= 0.0):
        raise ValueError("V5 dataset contains no weighted nonforced policy rows")
    ratio_deltas = np.expm1(log_ratios)
    approx_kl_rows = ratio_deltas - log_ratios
    clip_rows = (np.abs(ratio_deltas) > clip_ratio).astype(np.float64)

    def record(mask: np.ndarray) -> dict[str, object]:
        selected_weights = weights[mask]
        selected_ratios = log_ratios[mask]
        return {
            "approxKl": _weighted_metric(approx_kl_rows[mask], selected_weights),
            "clipFraction": _weighted_metric(clip_rows[mask], selected_weights),
            "count": int(mask.sum()),
            "entropy": _weighted_metric(entropies[mask], selected_weights),
            "maximumAbsoluteLogRatio": float(
                np.abs(selected_ratios).max(initial=0.0)
            ),
            "meanAbsoluteLogRatio": _weighted_metric(
                np.abs(selected_ratios), selected_weights
            ),
            "meanLogRatio": _weighted_metric(selected_ratios, selected_weights),
            "weightMass": math.fsum(float(value) for value in selected_weights),
        }

    per_player_count = {
        str(player_count): record(player_counts == player_count)
        for player_count in range(4, 11)
        if bool(np.any(player_counts == player_count))
    }
    output = record(np.ones(log_ratios.shape, dtype=np.bool_))
    output.update(
        {
            "allRowMeanAbsoluteOldLogProbabilityError": replay_error_sum / replay_rows,
            "allRowMaximumAbsoluteOldLogProbabilityError": maximum_replay_error,
            "allRowsReplayed": replay_rows,
            "allRowsReplayedMeaning": (
                "every row semantically visited; Actor forward only on nonforced rows"
            ),
            "allRowsSemanticallyVisited": replay_rows,
            "actorForwardRows": actor_forward_rows,
            "forcedActorForwardRowsSkipped": forced_semantic_rows,
            "forcedMaximumAbsoluteLogRatio": 0.0,
            "forcedRows": forced_semantic_rows,
            "nonforcedRows": int(log_ratios.size),
            "perPlayerCount": per_player_count,
        }
    )
    return output


def _initial_replay_or_raise(audit: Mapping[str, object]) -> None:
    maximum = float(audit["allRowMaximumAbsoluteOldLogProbabilityError"])
    if maximum > V5_BEHAVIOR_LOG_PROBABILITY_ABSOLUTE_TOLERANCE:
        raise ValueError(
            "V5 behavior Actor replay exceeded absolute old-log-probability "
            f"tolerance: max={maximum:.9g}, "
            f"tolerance={V5_BEHAVIOR_LOG_PROBABILITY_ABSOLUTE_TOLERANCE:.9g}"
        )


def enforce_v5_training_hard_gates(audit: Mapping[str, object]) -> None:
    """Fail closed on the canonical post-epoch policy-drift gates."""

    try:
        approx_kl = float(audit["approxKl"])
        clip_fraction = float(audit["clipFraction"])
        retention = float(audit["entropyRetentionRatio"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("V5 post-epoch hard-gate audit is incomplete") from error
    if not all(math.isfinite(value) for value in (approx_kl, clip_fraction, retention)):
        raise ValueError("V5 post-epoch hard-gate metrics are non-finite")
    failures: list[str] = []
    if not approx_kl < V5_MAXIMUM_APPROX_KL:
        failures.append(f"approxKl={approx_kl:.9g} is not < {V5_MAXIMUM_APPROX_KL}")
    if not clip_fraction < V5_MAXIMUM_CLIP_FRACTION:
        failures.append(
            f"clipFraction={clip_fraction:.9g} is not < {V5_MAXIMUM_CLIP_FRACTION}"
        )
    if not retention > V5_MINIMUM_ENTROPY_RETENTION:
        failures.append(
            f"entropyRetention={retention:.9g} is not > {V5_MINIMUM_ENTROPY_RETENTION}"
        )
    if failures:
        raise RuntimeError("V5 post-epoch policy hard gate failed: " + "; ".join(failures))


def _empty_critic_audit_moments() -> dict[str, float]:
    return {
        "count": 0.0,
        "mass": 0.0,
        "weightedHuberSum": 0.0,
        "weightedSquaredErrorSum": 0.0,
        "weightedReturnSum": 0.0,
        "weightedReturnSquaredSum": 0.0,
        "weightedErrorSum": 0.0,
        "weightedErrorSquaredSum": 0.0,
    }


def _update_critic_audit_moments(
    moments: dict[str, float],
    *,
    returns: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    huber_delta: float,
) -> None:
    target = np.asarray(returns, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    selected_weights = np.asarray(weights, dtype=np.float64)
    if (
        target.ndim != 1
        or predicted.shape != target.shape
        or selected_weights.shape != target.shape
        or target.size == 0
        or not np.isfinite(target).all()
        or not np.isfinite(predicted).all()
        or not np.isfinite(selected_weights).all()
        or np.any(selected_weights <= 0.0)
    ):
        raise ValueError("V5 critic audit received invalid weighted rows")
    error = predicted - target
    absolute_error = np.abs(error)
    huber = np.where(
        absolute_error <= huber_delta,
        0.5 * error * error,
        huber_delta * (absolute_error - 0.5 * huber_delta),
    )
    moments["count"] += float(target.size)
    moments["mass"] += float(selected_weights.sum(dtype=np.float64))
    moments["weightedHuberSum"] += float(np.dot(huber, selected_weights))
    moments["weightedSquaredErrorSum"] += float(
        np.dot(error * error, selected_weights)
    )
    moments["weightedReturnSum"] += float(np.dot(target, selected_weights))
    moments["weightedReturnSquaredSum"] += float(
        np.dot(target * target, selected_weights)
    )
    moments["weightedErrorSum"] += float(np.dot(error, selected_weights))
    moments["weightedErrorSquaredSum"] += float(
        np.dot(error * error, selected_weights)
    )


def _critic_audit_record(moments: Mapping[str, float]) -> dict[str, object]:
    mass = float(moments["mass"])
    count = int(moments["count"])
    if count < 1 or not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("V5 critic audit population is empty")
    return_mean = float(moments["weightedReturnSum"]) / mass
    error_mean = float(moments["weightedErrorSum"]) / mass
    return_variance = max(
        0.0,
        float(moments["weightedReturnSquaredSum"]) / mass
        - return_mean * return_mean,
    )
    error_variance = max(
        0.0,
        float(moments["weightedErrorSquaredSum"]) / mass
        - error_mean * error_mean,
    )
    if return_variance <= V5_CRITIC_HUBER_ZERO_EPSILON:
        explained_variance = (
            1.0 if error_variance <= V5_CRITIC_HUBER_ZERO_EPSILON else 0.0
        )
    else:
        explained_variance = 1.0 - error_variance / return_variance
    output: dict[str, object] = {
        "count": count,
        "weightMass": mass,
        "weightedHuberLoss": float(moments["weightedHuberSum"]) / mass,
        "weightedMse": float(moments["weightedSquaredErrorSum"]) / mass,
        "explainedVariance": explained_variance,
        "weightedReturnMean": return_mean,
        "weightedReturnVariance": return_variance,
    }
    if not all(
        math.isfinite(float(value))
        for key, value in output.items()
        if key != "count"
    ):
        raise ValueError("V5 critic audit produced non-finite metrics")
    return output


def _audit_critic(
    critic: V5CentralStateValueCritic,
    source: _Source,
    weight_contract: _GlobalWeightContract,
    *,
    device: torch.device,
    batch_size: int,
    huber_delta: float,
) -> dict[str, object]:
    """Replay the centralized critic over every value-eligible row in FP32."""

    critic_was_training = critic.training
    critic.eval()
    global_moments = _empty_critic_audit_moments()
    per_player_moments = {
        player: _empty_critic_audit_moments()
        for player in sorted(weight_contract.value_counts)
    }
    try:
        with torch.no_grad():
            for shard in source.shards:
                arrays = shard.actor.arrays
                decision_count = shard.actor.decision_count
                for start in range(0, decision_count, batch_size):
                    indexes = np.arange(
                        start, min(start + batch_size, decision_count), dtype=np.int64
                    )
                    value_mask = np.asarray(
                        arrays["value_mask"][indexes], dtype=np.bool_
                    )
                    if not bool(value_mask.all()):
                        raise ValueError(
                            "every V5 decision, including forced rows, must audit value"
                        )
                    player_counts = np.asarray(
                        arrays["global_codes"][indexes, 1], dtype=np.uint8
                    )
                    row_weights = _row_weights(
                        player_counts, weight_contract.value_weights
                    ).astype(np.float64)
                    states = torch.from_numpy(
                        np.ascontiguousarray(
                            shard.privileged_arrays["privileged_states"][indexes]
                        )
                    ).to(device=device, dtype=torch.float32)
                    counts_tensor = torch.from_numpy(
                        np.ascontiguousarray(player_counts)
                    ).to(device=device, dtype=torch.long)
                    with torch.amp.autocast(device.type, enabled=False):
                        predictions = critic(states, counts_tensor).float()
                    predicted_cpu = (
                        predictions.detach().cpu().to(torch.float64).numpy()
                    )
                    returns = np.asarray(
                        arrays["returns"][indexes], dtype=np.float64
                    )
                    _update_critic_audit_moments(
                        global_moments,
                        returns=returns,
                        predictions=predicted_cpu,
                        weights=row_weights,
                        huber_delta=huber_delta,
                    )
                    for player, moments in per_player_moments.items():
                        mask = player_counts == player
                        if bool(mask.any()):
                            _update_critic_audit_moments(
                                moments,
                                returns=returns[mask],
                                predictions=predicted_cpu[mask],
                                weights=row_weights[mask],
                                huber_delta=huber_delta,
                            )
    finally:
        critic.train(critic_was_training)
    if int(global_moments["count"]) != source.decision_count:
        raise RuntimeError("V5 critic audit did not visit every decision row")
    output = _critic_audit_record(global_moments)
    output.update(
        {
            "allRowsAudited": source.decision_count,
            "auditBatchSize": batch_size,
            "perPlayerCount": {
                str(player): _critic_audit_record(moments)
                for player, moments in per_player_moments.items()
            },
        }
    )
    return output


def enforce_v5_critic_hard_gates(
    initial: Mapping[str, object], post: Mapping[str, object]
) -> None:
    """Reject a critic that fails to improve globally or regresses a p stratum."""

    try:
        initial_huber = float(initial["weightedHuberLoss"])
        post_huber = float(post["weightedHuberLoss"])
        initial_ev = float(initial["explainedVariance"])
        post_ev = float(post["explainedVariance"])
        initial_per_player = initial["perPlayerCount"]
        post_per_player = post["perPlayerCount"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("V5 critic hard-gate audit is incomplete") from error
    if not isinstance(initial_per_player, Mapping) or not isinstance(
        post_per_player, Mapping
    ):
        raise ValueError("V5 critic per-player hard-gate audit is incomplete")
    if set(initial_per_player) != set(post_per_player):
        raise ValueError("V5 critic per-player audit populations drifted")
    failures: list[str] = []
    if not all(
        math.isfinite(value)
        for value in (initial_huber, post_huber, initial_ev, post_ev)
    ):
        raise ValueError("V5 critic hard-gate metrics are non-finite")
    if not post_huber < initial_huber:
        failures.append(
            f"global weightedHuberLoss={post_huber:.9g} is not strictly below "
            f"initial={initial_huber:.9g}"
        )
    for player in sorted(initial_per_player):
        initial_record = initial_per_player[player]
        post_record = post_per_player[player]
        if not isinstance(initial_record, Mapping) or not isinstance(
            post_record, Mapping
        ):
            raise ValueError("V5 critic per-player hard-gate record is invalid")
        try:
            initial_player_huber = float(initial_record["weightedHuberLoss"])
            post_player_huber = float(post_record["weightedHuberLoss"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("V5 critic per-player hard-gate record is incomplete") from error
        if not math.isfinite(initial_player_huber) or not math.isfinite(
            post_player_huber
        ):
            raise ValueError("V5 critic per-player hard-gate metric is non-finite")
        ceiling = (
            initial_player_huber * V5_CRITIC_PER_PLAYER_HUBER_REGRESSION_FACTOR
            + V5_CRITIC_HUBER_ZERO_EPSILON
        )
        if post_player_huber > ceiling:
            failures.append(
                f"p{player} weightedHuberLoss={post_player_huber:.9g} exceeds "
                f"ceiling={ceiling:.9g}"
            )
    ev_floor = initial_ev - V5_CRITIC_EXPLAINED_VARIANCE_REGRESSION_TOLERANCE
    if post_ev < ev_floor:
        failures.append(
            f"global explainedVariance={post_ev:.9g} is below floor={ev_floor:.9g}"
        )
    if failures:
        raise RuntimeError("V5 post-epoch critic hard gate failed: " + "; ".join(failures))


def _batching_seed(seed: int, label: str, ordinal: int) -> int:
    digest = hashlib.sha256(
        b"DALMUTI-V5-SHARD-LOCAL-BATCHING\0"
        + seed.to_bytes(4, "little")
        + label.encode("ascii")
        + ordinal.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _deterministic_shard_local_microbatches(
    source: _Source,
    *,
    microbatch_size: int,
    seed: int,
    population: str = "all",
) -> list[tuple[int, np.ndarray]]:
    """Shuffle rows within shards, then globally shuffle whole microbatches.

    A microbatch never crosses an mmap shard, avoiding up to ``batch`` tiny
    Actor/critic forwards after a global row permutation.  ``all`` visits
    every row; ``policy`` visits every and only policy-eligible nonforced row.
    The selected population occurs exactly once, including one final partial
    microbatch per shard.
    """

    if population not in {"all", "policy"}:
        raise ValueError("V5 batching population must be all or policy")
    batches: list[tuple[int, np.ndarray]] = []
    for shard_id, count in enumerate(source.counts):
        if population == "all":
            eligible = np.arange(count, dtype=np.int64)
        else:
            arrays = source.shards[shard_id].actor.arrays
            eligible = np.flatnonzero(
                np.asarray(arrays["policy_mask"], dtype=np.bool_)
            ).astype(np.int64, copy=False)
        generator = np.random.Generator(
            np.random.PCG64(
                _batching_seed(seed, f"{population}-shard", shard_id)
            )
        )
        order = generator.permutation(eligible).astype(np.int64, copy=False)
        batches.extend(
            (shard_id, order[start : start + microbatch_size])
            for start in range(0, len(order), microbatch_size)
        )
    if not batches:
        raise ValueError(f"cannot batch an empty V5 {population} population")
    generator = np.random.Generator(
        np.random.PCG64(
            _batching_seed(seed, f"{population}-global", len(batches))
        )
    )
    batch_order = generator.permutation(len(batches))
    return [batches[int(index)] for index in batch_order]


def _accumulation_loss_scale(
    microbatches: Sequence[tuple[int, np.ndarray]],
    batch_index: int,
    gradient_accumulation: int,
) -> float:
    """Weight partial microbatches by rows within one optimizer step."""

    group_start = (batch_index // gradient_accumulation) * gradient_accumulation
    group_end = min(group_start + gradient_accumulation, len(microbatches))
    group_rows = sum(len(microbatches[index][1]) for index in range(group_start, group_end))
    if group_rows < 1:
        raise RuntimeError("V5 accumulation group contains no decision rows")
    return len(microbatches[batch_index][1]) / group_rows


def _deterministic_optimizer_groups(
    batches: Sequence[tuple[int, np.ndarray]],
    *,
    target_rows: int,
    seed: int,
    seed_domain: str,
) -> list[list[tuple[int, np.ndarray]]]:
    """Pack shard-local remainders so AdamW never steps once per remainder.

    Full physical batches remain single-fragment optimizer groups.  Shard
    remainders are sliced and packed to exactly ``target_rows``; the sole
    global tail is merged into the preceding group when possible.  Thus at
    most one optimizer group differs from the target size, while no physical
    forward ever crosses an mmap shard.
    """

    if target_rows < 1:
        raise ValueError("optimizer target_rows must be positive")
    if not seed_domain or not seed_domain.isascii():
        raise ValueError("optimizer seed domain must be non-empty ASCII")
    full_groups: list[list[tuple[int, np.ndarray]]] = []
    remainders: list[tuple[int, np.ndarray]] = []
    for shard_id, indexes in batches:
        if len(indexes) == target_rows:
            full_groups.append([(shard_id, indexes)])
        elif 0 < len(indexes) < target_rows:
            remainders.append((shard_id, indexes))
        else:
            raise ValueError("physical batch size exceeds its optimizer target")

    packed: list[list[tuple[int, np.ndarray]]] = []
    current: list[tuple[int, np.ndarray]] = []
    current_rows = 0
    for shard_id, indexes in remainders:
        start = 0
        while start < len(indexes):
            take = min(target_rows - current_rows, len(indexes) - start)
            current.append((shard_id, indexes[start : start + take]))
            current_rows += take
            start += take
            if current_rows == target_rows:
                packed.append(current)
                current = []
                current_rows = 0
    groups = full_groups + packed
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    if not groups:
        raise ValueError("cannot group an empty V5 critic population")
    generator = np.random.Generator(
        np.random.PCG64(
            _batching_seed(seed, f"{seed_domain}-optimizer-groups", len(groups))
        )
    )
    order = generator.permutation(len(groups))
    return [groups[int(index)] for index in order]


def _deterministic_critic_optimizer_groups(
    batches: Sequence[tuple[int, np.ndarray]],
    *,
    target_rows: int,
    seed: int,
) -> list[list[tuple[int, np.ndarray]]]:
    return _deterministic_optimizer_groups(
        batches,
        target_rows=target_rows,
        seed=seed,
        seed_domain="critic",
    )


def _optimizer_group_loss_scale(
    group: Sequence[tuple[int, np.ndarray]], fragment_index: int
) -> float:
    group_rows = sum(len(indexes) for _, indexes in group)
    if group_rows < 1:
        raise RuntimeError("V5 optimizer group contains no decision rows")
    return len(group[fragment_index][1]) / group_rows


def _advantage_moments(
    source: _Source, weight_contract: _GlobalWeightContract
) -> tuple[float, float]:
    numerator = 0.0
    second = 0.0
    mass = 0.0
    for shard in source.shards:
        arrays = shard.actor.arrays
        mask = np.asarray(arrays["policy_mask"], dtype=np.bool_)
        values = np.asarray(arrays["advantages"], dtype=np.float64)[mask]
        counts = np.asarray(arrays["global_codes"][:, 1], dtype=np.uint8)[mask]
        weights = _row_weights(counts, weight_contract.policy_weights).astype(
            np.float64
        )
        numerator += float(np.dot(values, weights))
        second += float(np.dot(values * values, weights))
        mass += float(weights.sum(dtype=np.float64))
    if mass <= 0.0:
        raise ValueError("cannot normalize an empty policy population")
    mean = numerator / mass
    variance = max(0.0, second / mass - mean * mean)
    return mean, max(math.sqrt(variance), 1.0e-8)


def _to_cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _step_accumulated_optimizers(
    *,
    actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scaler: object,
    max_gradient_norm: float,
    device: torch.device,
) -> bool:
    """Step value always, but never decay an Actor with no policy gradient."""

    scaler.unscale_(critic_optimizer)  # type: ignore[attr-defined]
    critic_norm = nn.utils.clip_grad_norm_(critic.parameters(), max_gradient_norm)
    actor_has_gradient = any(parameter.grad is not None for parameter in actor.parameters())
    if actor_has_gradient:
        scaler.unscale_(actor_optimizer)  # type: ignore[attr-defined]
        actor_norm = nn.utils.clip_grad_norm_(actor.parameters(), max_gradient_norm)
    else:
        actor_norm = torch.zeros((), device=device)
    if not bool(torch.isfinite(actor_norm)) or not bool(torch.isfinite(critic_norm)):
        raise RuntimeError("V5 gradient norm became non-finite")
    if actor_has_gradient:
        scaler.step(actor_optimizer)  # type: ignore[attr-defined]
    scaler.step(critic_optimizer)  # type: ignore[attr-defined]
    scaler.update()  # type: ignore[attr-defined]
    return actor_has_gradient


def _step_single_optimizer(
    *,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object,
    max_gradient_norm: float,
) -> None:
    scaler.unscale_(optimizer)  # type: ignore[attr-defined]
    gradient_norm = nn.utils.clip_grad_norm_(module.parameters(), max_gradient_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("V5 gradient norm became non-finite")
    scaler.step(optimizer)  # type: ignore[attr-defined]
    scaler.update()  # type: ignore[attr-defined]


def _training_epoch(
    actor: V5PublicActor,
    critic: V5CentralStateValueCritic,
    source: _Source,
    weight_contract: _GlobalWeightContract,
    config: V5TrainingConfig,
    device: torch.device,
) -> tuple[dict[str, object], torch.optim.Optimizer, torch.optim.Optimizer, object]:
    actor.train()
    critic.train()
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.actor_learning_rate,
        weight_decay=config.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    advantage_mean, advantage_scale = (
        _advantage_moments(source, weight_contract)
        if config.normalize_advantages
        else (0.0, 1.0)
    )
    policy_population = sum(weight_contract.policy_counts.values())
    actor_microbatches = _deterministic_shard_local_microbatches(
        source,
        microbatch_size=config.microbatch_size,
        seed=config.seed,
        population="policy",
    )
    actor_optimizer_groups = _deterministic_optimizer_groups(
        actor_microbatches,
        target_rows=config.microbatch_size * config.gradient_accumulation,
        seed=config.seed,
        seed_domain="actor",
    )
    critic_batches = _deterministic_shard_local_microbatches(
        source,
        microbatch_size=config.critic_batch_size,
        seed=config.seed,
        population="all",
    )
    critic_optimizer_groups = _deterministic_critic_optimizer_groups(
        critic_batches,
        target_rows=config.critic_batch_size,
        seed=config.seed,
    )

    actor_totals = {
        "policyLoss": 0.0,
        "entropy": 0.0,
        "normalAuxiliaryLoss": 0.0,
        "actorObjective": 0.0,
    }
    actor_rows_seen = 0
    actor_optimizer_steps = 0
    actor_physical_batches = 0
    for optimizer_group in actor_optimizer_groups:
        actor_optimizer.zero_grad(set_to_none=True)
        for fragment_index, (shard_id, local_indexes) in enumerate(optimizer_group):
            shard = source.shards[shard_id]
            arrays = shard.actor.arrays
            policy_mask = np.asarray(
                arrays["policy_mask"][local_indexes], dtype=np.bool_
            )
            forced = np.asarray(arrays["forced"][local_indexes], dtype=np.bool_)
            if not bool(policy_mask.all()) or bool(forced.any()):
                raise ValueError("Actor training batch included a forced policy row")
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                selected, entropy, auxiliary = _selected_policy_statistics(
                    actor, arrays, local_indexes, device
                )
                old = torch.from_numpy(
                    np.ascontiguousarray(arrays["old_log_probs"][local_indexes])
                ).to(device=device, dtype=torch.float32)
                advantages = torch.from_numpy(
                    np.ascontiguousarray(arrays["advantages"][local_indexes])
                ).to(device=device, dtype=torch.float32)
                advantages = (advantages - advantage_mean) / advantage_scale
                weights = torch.from_numpy(
                    _row_weights(
                        np.asarray(
                            arrays["global_codes"][local_indexes, 1], dtype=np.uint8
                        ),
                        weight_contract.policy_weights,
                    )
                ).to(device=device, dtype=torch.float32)
                ratio = torch.exp(selected - old)
                surrogate = torch.minimum(
                    ratio * advantages,
                    ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                    * advantages,
                )
                estimator_scale = policy_population / (
                    weight_contract.policy_mass * len(local_indexes)
                )
                policy_loss = -(surrogate * weights).sum() * estimator_scale
                entropy_mean = (entropy * weights).sum() * estimator_scale
                auxiliary_loss = (auxiliary * weights).sum() * estimator_scale
                actor_objective = (
                    policy_loss
                    - config.entropy_coefficient * entropy_mean
                    + config.normal_auxiliary_coefficient * auxiliary_loss
                )
            loss_scale = _optimizer_group_loss_scale(
                optimizer_group, fragment_index
            )
            scaler.scale(actor_objective * loss_scale).backward()
            actor_physical_batches += 1
            actor_rows_seen += len(local_indexes)
            for name, value in (
                ("policyLoss", policy_loss),
                ("entropy", entropy_mean),
                ("normalAuxiliaryLoss", auxiliary_loss),
                ("actorObjective", actor_objective),
            ):
                number = float(value.detach().cpu())
                if not math.isfinite(number):
                    raise RuntimeError(f"V5 {name} became non-finite")
                actor_totals[name] += number * len(local_indexes)
        _step_single_optimizer(
            module=actor,
            optimizer=actor_optimizer,
            scaler=scaler,
            max_gradient_norm=config.max_gradient_norm,
        )
        actor_optimizer_steps += 1
    if actor_rows_seen != policy_population:
        raise RuntimeError("V5 Actor epoch did not consume each nonforced row once")

    critic_totals = {"valueLoss": 0.0, "criticObjective": 0.0}
    critic_rows_seen = 0
    critic_optimizer_steps = 0
    critic_physical_batches = 0
    for optimizer_group in critic_optimizer_groups:
        critic_optimizer.zero_grad(set_to_none=True)
        for fragment_index, (shard_id, local_indexes) in enumerate(optimizer_group):
            shard = source.shards[shard_id]
            arrays = shard.actor.arrays
            value_mask = np.asarray(
                arrays["value_mask"][local_indexes], dtype=np.bool_
            )
            if not bool(value_mask.all()):
                raise ValueError(
                    "every V5 decision, including forced rows, must train value"
                )
            # This is the sole privileged read in the optimization path.
            states = torch.from_numpy(
                np.ascontiguousarray(
                    shard.privileged_arrays["privileged_states"][local_indexes]
                )
            ).to(device=device, dtype=torch.float32)
            player_counts_cpu = np.asarray(
                arrays["global_codes"][local_indexes, 1], dtype=np.uint8
            )
            player_counts = torch.from_numpy(
                np.ascontiguousarray(player_counts_cpu)
            ).to(device=device, dtype=torch.long)
            returns = torch.from_numpy(
                np.ascontiguousarray(arrays["returns"][local_indexes])
            ).to(device=device, dtype=torch.float32)
            value_weights = torch.from_numpy(
                _row_weights(player_counts_cpu, weight_contract.value_weights)
            ).to(device=device, dtype=torch.float32)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                predicted_values = critic(states, player_counts)
                huber = F.huber_loss(
                    predicted_values.float(),
                    returns,
                    reduction="none",
                    delta=config.huber_delta,
                )
                estimator_scale = source.decision_count / (
                    weight_contract.value_mass * len(local_indexes)
                )
                value_loss = (huber * value_weights).sum() * estimator_scale
                critic_objective = config.value_coefficient * value_loss
            loss_scale = _optimizer_group_loss_scale(
                optimizer_group, fragment_index
            )
            scaler.scale(critic_objective * loss_scale).backward()
            critic_physical_batches += 1
            critic_rows_seen += len(local_indexes)
            for name, value in (
                ("valueLoss", value_loss),
                ("criticObjective", critic_objective),
            ):
                number = float(value.detach().cpu())
                if not math.isfinite(number):
                    raise RuntimeError(f"V5 {name} became non-finite")
                critic_totals[name] += number * len(local_indexes)
        _step_single_optimizer(
            module=critic,
            optimizer=critic_optimizer,
            scaler=scaler,
            max_gradient_norm=config.max_gradient_norm,
        )
        critic_optimizer_steps += 1
    if critic_rows_seen != source.decision_count:
        raise RuntimeError("V5 critic epoch did not consume each decision row once")
    return (
        {
            "advantageMean": advantage_mean,
            "advantageScale": advantage_scale,
            "ampEnabled": amp_enabled,
            "batching": {
                "contract": "deterministic-shard-local-separated-actor-critic-v1",
                "actor": {
                    "effectiveBatchSize": (
                        config.microbatch_size * config.gradient_accumulation
                    ),
                    "gradientAccumulation": config.gradient_accumulation,
                    "microbatchSize": config.microbatch_size,
                    "optimizerGroupRows": [
                        sum(len(indexes) for _, indexes in group)
                        for group in actor_optimizer_groups
                    ],
                    "partialMicrobatches": sum(
                        int(len(indexes) < config.microbatch_size)
                        for _, indexes in actor_microbatches
                    ),
                    "partialBatchGradientWeighting": "rows / optimizer-group rows",
                    "population": "every policy-eligible nonforced row exactly once",
                    "seedDomain": "policy + actor-optimizer-groups",
                },
                "critic": {
                    "batchSize": config.critic_batch_size,
                    "optimizerGroupRows": [
                        sum(len(indexes) for _, indexes in group)
                        for group in critic_optimizer_groups
                    ],
                    "partialBatchGradientWeighting": "rows / optimizer-group rows",
                    "population": "every decision row exactly once",
                    "seedDomain": "all + critic-optimizer-groups",
                },
                "shardCount": len(source.shards),
            },
            "batches": actor_physical_batches + critic_physical_batches,
            "actorBatches": actor_physical_batches,
            "actorDecisionRowsSeen": actor_rows_seen,
            "criticBatches": critic_physical_batches,
            "criticDecisionRowsSeen": critic_rows_seen,
            "decisionRowsSeen": critic_rows_seen,
            "epoch": 1,
            "optimizerSteps": actor_optimizer_steps + critic_optimizer_steps,
            "actorOptimizerSteps": actor_optimizer_steps,
            "criticOptimizerSteps": critic_optimizer_steps,
            "valueOnlyOptimizerSteps": critic_optimizer_steps,
            **{
                name: value / actor_rows_seen for name, value in actor_totals.items()
            },
            **{
                name: value / critic_rows_seen
                for name, value in critic_totals.items()
            },
        },
        actor_optimizer,
        critic_optimizer,
        scaler,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _exclusive_output_directory(target: Path) -> tuple[Path, Path]:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"immutable V5 training output already exists: {target}")
    lock = target.parent / f".{target.name}.publish.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
    os.fsync(lock_fd)
    os.close(lock_fd)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    return staging, lock


def publish_seeded_v5_initialization(
    output_directory: str | Path,
    *,
    seed: int,
    actor_config: V5ActorConfig | None = None,
    critic_config: V5CriticConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically publish reproducible zero-residual Actor/zero-value critic.

    Model construction happens in two independently seeded RNG scopes.  No
    ambient/default Torch RNG state participates, and the caller's RNG state
    is restored on return.
    """

    seeds = derive_v5_initialization_seeds(seed)
    target = Path(output_directory).resolve()
    staging, lock = _exclusive_output_directory(target)
    try:
        configure_v5_policy_numerics("cpu")
        base_metadata = _canonical_metadata(metadata)
        seed_metadata = {**base_metadata, **seeds}
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seeds["actorInitializationSeed"])
            actor = V5PublicActor(actor_config or V5ActorConfig())
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seeds["criticInitializationSeed"])
            critic = V5CentralStateValueCritic(critic_config or V5CriticConfig())
        export_v5_actor_bundle(
            actor,
            staging / "actor-bundle",
            metadata=seed_metadata,
        )
        critic_digests = export_v5_critic_checkpoint(
            critic,
            staging / "critic.pt",
            metadata=seed_metadata,
        )
        actor_digests = v5_actor_bundle_digests(staging / "actor-bundle")
        pair_record = publish_v5_model_pair_manifest(staging)
        record: dict[str, object] = {
            "actor": actor_digests,
            "critic": critic_digests,
            "format": "dalmuti-v5-seeded-initialization",
            "metadata": base_metadata,
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
            "seeds": seeds,
            "modelPair": pair_record,
            "version": 1,
        }
        raw = canonical_json_bytes(record)
        _write_exclusive(staging / "initialization.json", raw)
        digest = hashlib.sha256(raw).hexdigest()
        _write_exclusive(
            staging / "initialization.json.sha256",
            f"{digest}  initialization.json\n".encode("ascii"),
        )
        if target.exists():
            raise FileExistsError(
                f"immutable V5 initialization appeared while publishing: {target}"
            )
        os.rename(staging, target)
        return {
            **record,
            "actorBundle": str(target / "actor-bundle"),
            "criticCheckpoint": str(target / "critic.pt"),
            "initializationSha256": digest,
            "outputDirectory": str(target),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _publish_training_output(
    *,
    target: Path,
    actor: V5PublicActor,
    critic: V5CentralStateValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scaler: object,
    config: V5TrainingConfig,
    result: dict[str, object],
) -> dict[str, object]:
    staging, lock = _exclusive_output_directory(target)
    try:
        actor_bundle = staging / "actor-bundle"
        export_v5_actor_bundle(
            actor,
            actor_bundle,
            metadata={
                "datasetIdentitySha256": result["datasetIdentitySha256"],
                "trainingFormat": V5_TRAINING_FORMAT,
                **result["initializationSeeds"],  # type: ignore[arg-type]
            },
        )
        verify_v5_actor_bundle(actor_bundle)
        critic_record = export_v5_critic_checkpoint(
            critic,
            staging / "critic.pt",
            metadata={
                "trainingOnly": True,
                **result["initializationSeeds"],  # type: ignore[arg-type]
            },
        )
        actor_digests = v5_actor_bundle_digests(actor_bundle)
        pair_record = publish_v5_model_pair_manifest(staging)
        optimizer_payload = {
            "actorOptimizer": _to_cpu_tree(actor_optimizer.state_dict()),
            "criticOptimizer": _to_cpu_tree(critic_optimizer.state_dict()),
            "format": V5_OPTIMIZER_FORMAT,
            "gradientScaler": _to_cpu_tree(scaler.state_dict()),  # type: ignore[attr-defined]
            "modelPairId": pair_record["pairId"],
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "version": 1,
        }
        _write_exclusive(staging / "optimizer.pt", _torch_bytes(optimizer_payload))
        checkpoint_payload = {
            "actorStateSha256": actor_digests["tensorStateSha256"],
            "actorStateDict": _tensor_state(actor),
            "config": config.to_dict(),
            "criticStateSha256": critic_record["tensorStateSha256"],
            "criticStateDict": _tensor_state(critic),
            "datasetIdentitySha256": result["datasetIdentitySha256"],
            "epoch": 1,
            "format": V5_CHECKPOINT_FORMAT,
            "modelPairId": pair_record["pairId"],
            "numpyRngState": _to_cpu_tree(np.random.get_state()),
            "optimizerSha256": sha256_file(staging / "optimizer.pt"),
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "pythonRngState": random.getstate(),
            "torchCpuRngState": torch.get_rng_state().cpu(),
            "torchCudaRngStates": (
                [value.cpu() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
            "version": 1,
        }
        _write_exclusive(
            staging / "training-checkpoint.pt", _torch_bytes(checkpoint_payload)
        )
        result.update(
            {
                "outputActor": actor_digests,
                "outputCritic": critic_record,
                "outputModelPair": pair_record,
            }
        )
        result_bytes = canonical_json_bytes(result)
        _write_exclusive(staging / "result.json", result_bytes)
        inventory: dict[str, dict[str, object]] = {}
        for path in sorted(
            (
                path
                for path in staging.rglob("*")
                if path.is_file()
            ),
            key=lambda value: value.relative_to(staging).as_posix(),
        ):
            relative = path.relative_to(staging).as_posix()
            inventory[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "actorBundleContainsCritic": False,
            "datasetIdentitySha256": result["datasetIdentitySha256"],
            "files": inventory,
            "format": V5_TRAINING_FORMAT,
            "gpuMemoryPreflight": result["gpuMemoryPreflight"],
            "initialBehaviorBindings": result["initialBehaviorBindings"],
            "policyNumerics": canonical_v5_policy_numerics_contract(),
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
            "resultSha256": hashlib.sha256(result_bytes).hexdigest(),
            "version": V5_TRAINING_VERSION,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        _write_exclusive(staging / "manifest.json", manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        _write_exclusive(
            staging / "manifest.json.sha256",
            f"{manifest_sha}  manifest.json\n".encode("ascii"),
        )
        if target.exists():
            raise FileExistsError(f"immutable V5 output appeared while publishing: {target}")
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


def train_v5_mappo(
    dataset_path: str | Path,
    initial_actor_bundle: str | Path,
    initial_critic_checkpoint: str | Path,
    output_directory: str | Path,
    *,
    config: V5TrainingConfig | None = None,
    device: str | torch.device | None = None,
    gpu_memory_preflight: Mapping[str, object] | None = None,
    allow_unadmitted_cpu: bool = False,
) -> dict[str, object]:
    """Run one immutable, behavior-bound V5 PPO epoch and publish its output."""

    training_config = config or V5TrainingConfig()
    if type(allow_unadmitted_cpu) is not bool:
        raise TypeError("allow_unadmitted_cpu must be an exact bool")
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if gpu_memory_preflight is None and not (
        allow_unadmitted_cpu and target_device.type == "cpu"
    ):
        raise ValueError(
            "V5 training requires a bound GPU memory preflight; only an "
            "explicit unadmitted CPU smoke/test run may omit it"
        )
    numerics = configure_v5_policy_numerics(target_device)
    if numerics.get("contractSha256") != V5_POLICY_NUMERICS_SHA256:
        raise RuntimeError("V5 policy numerics contract failed to bind")
    _set_seed(training_config.seed)
    actor_bundle_path = Path(initial_actor_bundle).resolve()
    critic_checkpoint_path = Path(initial_critic_checkpoint).resolve()
    if (
        actor_bundle_path.name != "actor-bundle"
        or critic_checkpoint_path.name != "critic.pt"
        or actor_bundle_path.parent != critic_checkpoint_path.parent
    ):
        raise ValueError("initial Actor and critic must be siblings in one V5 model pair")
    initial_model_pair = verify_v5_model_pair(actor_bundle_path.parent)
    actor, actor_manifest = load_v5_actor_bundle(actor_bundle_path)
    critic, critic_payload = load_v5_critic_checkpoint(critic_checkpoint_path)
    initialization_seeds = _validate_initialization_seed_binding(
        actor_manifest, critic_payload
    )
    assert_actor_critic_parameter_isolation(actor, critic)
    actor = actor.to(target_device)
    critic = critic.to(target_device)
    source = _open_source(dataset_path)
    try:
        weight_contract = _global_weight_contract(
            source,
            require_all_player_counts=training_config.require_all_player_counts,
        )
        bindings = _verify_behavior_bindings(
            source,
            actor_bundle_path,
            critic_checkpoint_path,
            initial_model_pair,
        )
        preflight_binding = _validate_gpu_memory_preflight_binding(
            gpu_memory_preflight,
            source=source,
            model_pair=initial_model_pair,
            config=training_config,
            device=target_device,
            allow_unadmitted_cpu=allow_unadmitted_cpu,
        )
        initial_actor_state_sha = tensor_state_sha256(_tensor_state(actor))
        initial_critic_state_sha = tensor_state_sha256(_tensor_state(critic))
        initial_audit = _audit_policy(
            actor,
            source,
            weight_contract,
            device=target_device,
            batch_size=training_config.audit_batch_size,
            clip_ratio=training_config.clip_ratio,
        )
        _initial_replay_or_raise(initial_audit)
        initial_critic_audit = _audit_critic(
            critic,
            source,
            weight_contract,
            device=target_device,
            batch_size=training_config.critic_batch_size,
            huber_delta=training_config.huber_delta,
        )
        epoch, actor_optimizer, critic_optimizer, scaler = _training_epoch(
            actor,
            critic,
            source,
            weight_contract,
            training_config,
            target_device,
        )
        post_audit = _audit_policy(
            actor,
            source,
            weight_contract,
            device=target_device,
            batch_size=training_config.audit_batch_size,
            clip_ratio=training_config.clip_ratio,
        )
        post_critic_audit = _audit_critic(
            critic,
            source,
            weight_contract,
            device=target_device,
            batch_size=training_config.critic_batch_size,
            huber_delta=training_config.huber_delta,
        )
        initial_entropy = float(initial_audit["entropy"])
        if initial_entropy <= 0.0:
            raise ValueError("initial behavior entropy must be positive")
        retention = float(post_audit["entropy"]) / initial_entropy
        post_audit["initialBehaviorEntropy"] = initial_entropy
        post_audit["entropyRetentionRatio"] = retention
        initial_per_p = initial_audit["perPlayerCount"]
        post_per_p = post_audit["perPlayerCount"]
        assert isinstance(initial_per_p, Mapping) and isinstance(post_per_p, dict)
        for key, record in post_per_p.items():
            assert isinstance(record, dict)
            baseline = initial_per_p[key]
            assert isinstance(baseline, Mapping)
            record["initialBehaviorEntropy"] = float(baseline["entropy"])
            record["entropyRetentionRatio"] = (
                float(record["entropy"]) / float(baseline["entropy"])
            )
        enforce_v5_training_hard_gates(post_audit)
        enforce_v5_critic_hard_gates(initial_critic_audit, post_critic_audit)
        result: dict[str, object] = {
            "config": training_config.to_dict(),
            "datasetDecisionCount": source.decision_count,
            "datasetIdentitySha256": source.identity_sha256,
            "datasetShardCount": len(source.shards),
            "device": str(target_device),
            "epoch": epoch,
            "format": V5_TRAINING_FORMAT,
            "gpuMemoryPreflight": preflight_binding,
            "hardGates": {
                "approxKlLessThan": V5_MAXIMUM_APPROX_KL,
                "clipFractionLessThan": V5_MAXIMUM_CLIP_FRACTION,
                "entropyRetentionGreaterThan": V5_MINIMUM_ENTROPY_RETENTION,
                "criticGlobalWeightedHuberStrictImprovement": True,
                "criticPerPlayerHuberMaximumRegressionFactor": (
                    V5_CRITIC_PER_PLAYER_HUBER_REGRESSION_FACTOR
                ),
                "criticPerPlayerHuberZeroEpsilon": V5_CRITIC_HUBER_ZERO_EPSILON,
                "criticExplainedVarianceMaximumRegression": (
                    V5_CRITIC_EXPLAINED_VARIANCE_REGRESSION_TOLERANCE
                ),
                "passed": True,
            },
            "initialActorStateSha256": initial_actor_state_sha,
            "initialBehaviorBindings": bindings,
            "initialCriticStateSha256": initial_critic_state_sha,
            "initializationSeeds": initialization_seeds,
            "globalLossWeightContract": weight_contract.to_dict(),
            "initialCriticAudit": initial_critic_audit,
            "initialPolicyReplay": initial_audit,
            "policyNumerics": numerics,
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "postEpochPolicyAudit": post_audit,
            "postEpochCriticAudit": post_critic_audit,
            "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
            "version": V5_TRAINING_VERSION,
        }
        return _publish_training_output(
            target=Path(output_directory).resolve(),
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            scaler=scaler,
            config=training_config,
            result=result,
        )
    finally:
        source.close()


# Stable workflow aliases.
train_v5 = train_v5_mappo
save_v5_critic_checkpoint = export_v5_critic_checkpoint


__all__ = [
    "V5_BEHAVIOR_LOG_PROBABILITY_ABSOLUTE_TOLERANCE",
    "V5_CHECKPOINT_FORMAT",
    "V5_CRITIC_EXPLAINED_VARIANCE_REGRESSION_TOLERANCE",
    "V5_CRITIC_FORMAT",
    "V5_CRITIC_HUBER_ZERO_EPSILON",
    "V5_CRITIC_PER_PLAYER_HUBER_REGRESSION_FACTOR",
    "V5_CRITIC_VERSION",
    "V5_MAXIMUM_APPROX_KL",
    "V5_MAXIMUM_CLIP_FRACTION",
    "V5_MINIMUM_ENTROPY_RETENTION",
    "V5_MODEL_PAIR_FORMAT",
    "V5_MODEL_PAIR_VERSION",
    "V5_TRAINING_FORMAT",
    "V5_TRAINING_VERSION",
    "V5TrainingConfig",
    "derive_v5_initialization_seeds",
    "enforce_v5_critic_hard_gates",
    "enforce_v5_training_hard_gates",
    "export_v5_critic_checkpoint",
    "load_v5_critic_checkpoint",
    "load_verified_v5_behavior_pair",
    "publish_seeded_v5_initialization",
    "publish_v5_model_pair_manifest",
    "save_v5_critic_checkpoint",
    "train_v5",
    "train_v5_mappo",
    "verify_v5_model_pair",
]
