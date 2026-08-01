from __future__ import annotations

from dataclasses import dataclass, fields
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import torch
from torch import nn

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
    V3_ACTION_FEATURES,
)
from v4_model import (
    V4_ACTION_COUNT,
    V4ActorConfig,
    V4CenteredLogitEnsemble,
    V4PublicActor,
)


V4_ACTOR_FORMAT = "dalmuti-v4-public-transformer-actor"
V4_ACTOR_FORMAT_VERSION = 2
V4_LEGACY_ACTOR_FORMAT_VERSION = 1
V4_MANIFEST_FORMAT = "dalmuti-v4-candidate-manifest"
V4_MANIFEST_VERSION = 2
V4_LEGACY_MANIFEST_VERSION = 1
V4_SEMANTIC_CONTRACT_VERSION = 1
SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
ONNX_ACTOR_SHA_METADATA = "dalmuti.actorPtSha256"
ONNX_SEMANTIC_SHA_METADATA = "dalmuti.semanticContractSha256"


class V4ONNXUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class V4ONNXExportResult:
    exported: bool
    path: Path | None
    sha256: str | None
    reason: str | None


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(raw: bytes, label: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _json_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    value = dict(metadata or {})
    # Both manifest.json and actor.pt must bind the same JSON value.  A JSON
    # round-trip prevents tuples or custom mapping subclasses from acquiring
    # different meanings in the two serializers.
    return _strict_json_object(canonical_json_bytes(value), "V4 actor metadata")


def _canonical_action_catalogue_bytes() -> bytes:
    # Keep this byte contract identical to rollout preparation and dataset
    # merging: property order is deliberately version, then catalogue.
    return json.dumps(
        {
            "version": V3_ACTION_CATALOGUE_VERSION,
            "catalogue": [dict(action) for action in V3_ACTION_CATALOGUE],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _action_catalogue_sha256() -> str:
    return sha256_bytes(_canonical_action_catalogue_bytes())


def _legacy_action_catalogue_sha256() -> str:
    # Manifest v1 used canonical_json_bytes(list), which sorted each action's
    # object keys and omitted the catalogue version.  It remains readable only
    # behind the explicit v1 legacy boundary.
    return sha256_bytes(
        canonical_json_bytes([dict(action) for action in V3_ACTION_CATALOGUE])
    )


def _action_space_contract(*, legacy: bool = False) -> dict[str, object]:
    return {
        "catalogueVersion": V3_ACTION_CATALOGUE_VERSION,
        "count": V4_ACTION_COUNT,
        "catalogueSha256": (
            _legacy_action_catalogue_sha256()
            if legacy else _action_catalogue_sha256()
        ),
    }


def _public_input_contract(config: V4ActorConfig) -> dict[str, object]:
    return {
        "global": ["batch", config.global_features],
        "ranks": ["batch", config.rank_tokens, config.rank_features],
        "players": ["batch", config.max_players, config.player_features],
        "memoryTrace": [
            "batch", config.memory_tokens, config.memory_features,
        ],
        "history": ["batch", config.max_history, config.history_features],
        "historyIsPublicOnly": True,
    }


def _validate_current_observation_contract(config: V4ActorConfig) -> None:
    canonical = V4ActorConfig()
    exact_fields = (
        "global_features",
        "rank_features",
        "player_features",
        "history_features",
        "memory_features",
        "rank_tokens",
        "memory_tokens",
        "observation_schema_version",
        "action_catalogue_version",
    )
    for name in exact_fields:
        if getattr(config, name) != getattr(canonical, name):
            raise ValueError(f"V4 actor {name} does not match the current public contract")
    if not 4 <= config.max_players <= canonical.max_players:
        raise ValueError("V4 actor max_players is outside the public contract")
    if not 1 <= config.max_history <= canonical.max_history:
        raise ValueError("V4 actor max_history is outside the public contract")
    if config.action_catalogue_version != V3_ACTION_CATALOGUE_VERSION:
        raise ValueError("V4 actor action catalogue version is not current")
    if V4_ACTION_COUNT != len(V3_ACTION_CATALOGUE):
        raise RuntimeError("V4 action count and current catalogue length diverged")


def _actor_config(value: object, label: str) -> V4ActorConfig:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing its configuration")
    expected_keys = {field.name for field in fields(V4ActorConfig)}
    if set(value) != expected_keys:
        raise ValueError(f"{label} configuration fields drifted")
    try:
        config = V4ActorConfig(**value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} has an invalid configuration") from error
    if config.to_dict() != value:
        raise ValueError(f"{label} configuration is not canonical")
    _validate_current_observation_contract(config)
    return config


def _model_identity(
    kind: object, seeds: object,
) -> tuple[str, list[int] | None]:
    if kind == "actor":
        if seeds is not None:
            raise ValueError("single actor checkpoint must not contain seeds")
        return "actor", None
    if kind == "centered-logit-ensemble":
        if (
            not isinstance(seeds, list)
            or len(seeds) != 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(set(seeds)) != 3
        ):
            raise ValueError("ensemble checkpoint has invalid seeds")
        return "centered-logit-ensemble", list(seeds)
    raise ValueError("unsupported V4 actor checkpoint kind")


def _semantic_contract(
    kind: str,
    config: V4ActorConfig,
    seeds: Sequence[int] | None,
) -> dict[str, object]:
    return {
        "version": V4_SEMANTIC_CONTRACT_VERSION,
        "kind": kind,
        "config": config.to_dict(),
        "seeds": list(seeds) if seeds is not None else None,
        "publicInputContract": _public_input_contract(config),
        "actionSpace": _action_space_contract(),
    }


def _semantic_contract_sha256(contract: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(contract)))


def _json_contract_equal(left: object, right: object) -> bool:
    """Compare JSON contracts without Python's bool/int equality aliases."""
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _validate_action_space_contract(
    value: object,
    expected: Mapping[str, object],
    label: str,
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"catalogueVersion", "count", "catalogueSha256"}
        or type(value.get("catalogueVersion")) is not int
        or type(value.get("count")) is not int
        or not isinstance(value.get("catalogueSha256"), str)
        or not _json_contract_equal(value, dict(expected))
    ):
        raise ValueError(f"{label} action-space contract drifted")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _checkpoint_payload(
    model: V4PublicActor | V4CenteredLogitEnsemble,
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    if isinstance(model, V4CenteredLogitEnsemble):
        kind = "centered-logit-ensemble"
        seeds: list[int] | None = list(model.seeds)
    elif isinstance(model, V4PublicActor):
        kind = "actor"
        seeds = None
    else:
        raise TypeError("only a V4 public actor or actor ensemble can be exported")
    _validate_current_observation_contract(model.config)
    contract = _semantic_contract(kind, model.config, seeds)
    return {
        "format": V4_ACTOR_FORMAT,
        "version": V4_ACTOR_FORMAT_VERSION,
        "kind": kind,
        "config": model.config.to_dict(),
        "seeds": seeds,
        "criticExcluded": True,
        "metadata": _json_metadata(metadata),
        "publicInputContract": contract["publicInputContract"],
        "actionSpace": contract["actionSpace"],
        "semanticContractSha256": _semantic_contract_sha256(contract),
        "stateDict": _cpu_state_dict(model),
    }


def save_v4_actor_checkpoint(
    model: V4PublicActor | V4CenteredLogitEnsemble,
    output_path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> str:
    buffer = io.BytesIO()
    torch.save(_checkpoint_payload(model, metadata), buffer)
    output = Path(output_path)
    _atomic_write(output, buffer.getvalue())
    return sha256_file(output)


def load_v4_actor_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[V4PublicActor | V4CenteredLogitEnsemble, dict[str, object]]:
    # requirements.txt pins torch>=2.4, so never fall back to unrestricted
    # pickle loading for a bundle supplied to the verifier.
    payload = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict):
        raise ValueError("V4 actor checkpoint must contain an object")
    version = payload.get("version")
    if (
        payload.get("format") != V4_ACTOR_FORMAT
        or type(version) is not int
        or version not in {
            V4_LEGACY_ACTOR_FORMAT_VERSION,
            V4_ACTOR_FORMAT_VERSION,
        }
        or payload.get("criticExcluded") is not True
    ):
        raise ValueError("unsupported V4 actor checkpoint contract")
    legacy_keys = {
        "format", "version", "kind", "config", "seeds", "criticExcluded",
        "metadata", "stateDict",
    }
    current_keys = legacy_keys | {
        "publicInputContract", "actionSpace", "semanticContractSha256",
    }
    if set(payload) != (
        legacy_keys if version == V4_LEGACY_ACTOR_FORMAT_VERSION else current_keys
    ):
        raise ValueError("V4 actor checkpoint fields do not match its version")
    config = _actor_config(payload.get("config"), "V4 actor checkpoint")
    kind, seeds = _model_identity(payload.get("kind"), payload.get("seeds"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("V4 actor checkpoint metadata must be an object")
    if version == V4_ACTOR_FORMAT_VERSION:
        contract = _semantic_contract(kind, config, seeds)
        if not _json_contract_equal(
            payload.get("publicInputContract"), contract["publicInputContract"]
        ):
            raise ValueError("V4 actor checkpoint public input contract drifted")
        _validate_action_space_contract(
            payload.get("actionSpace"),
            contract["actionSpace"],
            "V4 actor checkpoint",
        )
        if payload.get("semanticContractSha256") != _semantic_contract_sha256(contract):
            raise ValueError("V4 actor checkpoint semantic contract checksum drifted")
    model: V4PublicActor | V4CenteredLogitEnsemble
    if kind == "actor":
        model = V4PublicActor(config)
    else:
        assert seeds is not None
        model = V4CenteredLogitEnsemble.from_seeds(config, seeds)
    state_dict = payload.get("stateDict")
    if not isinstance(state_dict, dict):
        raise ValueError("V4 actor checkpoint is missing its state dictionary")
    if any("critic" in str(name).lower() for name in state_dict):
        raise ValueError("V4 public actor checkpoint contains critic parameters")
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("V4 actor state dictionary does not match its model contract") from error
    canonical_action_features = torch.tensor(
        V3_ACTION_FEATURES, dtype=torch.float32
    )
    actors = model.actors if isinstance(model, V4CenteredLogitEnsemble) else (model,)
    if any(
        not torch.equal(
            actor.action_features.detach().cpu(), canonical_action_features
        )
        for actor in actors
    ):
        raise ValueError(
            "V4 actor checkpoint action features do not match the current catalogue"
        )
    return model.eval(), payload


def _onnx_is_available() -> bool:
    return importlib.util.find_spec("onnx") is not None


def make_v4_export_inputs(
    config: V4ActorConfig,
    *,
    batch_size: int = 1,
) -> tuple[torch.Tensor, ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    global_features = torch.zeros(batch_size, config.global_features)
    rank_features = torch.zeros(
        batch_size, config.rank_tokens, config.rank_features
    )
    player_features = torch.zeros(
        batch_size, config.max_players, config.player_features
    )
    player_mask = torch.ones(batch_size, config.max_players, dtype=torch.bool)
    memory_trace_features = torch.zeros(
        batch_size, config.memory_tokens, config.memory_features
    )
    history_features = torch.zeros(
        batch_size, config.max_history, config.history_features
    )
    history_mask = torch.zeros(batch_size, config.max_history, dtype=torch.bool)
    legal_masks = torch.ones(batch_size, V4_ACTION_COUNT, dtype=torch.bool)
    return (
        global_features,
        rank_features,
        player_features,
        player_mask,
        memory_trace_features,
        history_features,
        history_mask,
        legal_masks,
    )


def export_v4_onnx(
    model: V4PublicActor | V4CenteredLogitEnsemble,
    output_path: str | Path,
    *,
    opset_version: int = 17,
) -> str:
    if not _onnx_is_available():
        raise V4ONNXUnavailableError(
            "optional dependency 'onnx' is not installed; the PyTorch actor "
            "checkpoint remains usable"
        )
    if not isinstance(model, (V4PublicActor, V4CenteredLogitEnsemble)):
        raise TypeError("only a V4 public actor or actor ensemble can be exported")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu().eval()
    example_inputs = make_v4_export_inputs(export_model.config)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.onnx.export(
            export_model,
            example_inputs,
            temporary,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[
                "global_features",
                "rank_features",
                "player_features",
                "player_mask",
                "memory_trace_features",
                "history_features",
                "history_mask",
                "legal_masks",
            ],
            output_names=["masked_logits"],
            dynamic_axes={
                name: {0: "batch"}
                for name in (
                    "global_features",
                    "rank_features",
                    "player_features",
                    "player_mask",
                    "memory_trace_features",
                    "history_features",
                    "history_mask",
                    "legal_masks",
                    "masked_logits",
                )
            },
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return sha256_file(output)


def try_export_v4_onnx(
    model: V4PublicActor | V4CenteredLogitEnsemble,
    output_path: str | Path,
    *,
    opset_version: int = 17,
) -> V4ONNXExportResult:
    output = Path(output_path)
    try:
        checksum = export_v4_onnx(
            model, output, opset_version=opset_version
        )
        return V4ONNXExportResult(True, output, checksum, None)
    except V4ONNXUnavailableError as error:
        return V4ONNXExportResult(False, None, None, str(error))


def _bind_v4_onnx_contract(
    path: Path,
    *,
    actor_sha256: str,
    semantic_sha256: str,
) -> str:
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - guarded by export
        raise V4ONNXUnavailableError(
            "optional dependency 'onnx' is required to bind a V4 ONNX bundle"
        ) from error
    try:
        model = onnx.load_model_from_string(path.read_bytes())
        metadata_keys = [entry.key for entry in model.metadata_props]
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("V4 ONNX model contains duplicate metadata keys")
        if (
            ONNX_ACTOR_SHA_METADATA in metadata_keys
            or ONNX_SEMANTIC_SHA_METADATA in metadata_keys
        ):
            raise ValueError("V4 ONNX model already contains reserved metadata")
        for key, value in (
            (ONNX_ACTOR_SHA_METADATA, actor_sha256),
            (ONNX_SEMANTIC_SHA_METADATA, semantic_sha256),
        ):
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.checker.check_model(model)
        _atomic_write(path, model.SerializeToString())
    except V4ONNXUnavailableError:
        raise
    except Exception as error:
        raise ValueError("V4 ONNX model could not be contract-bound") from error
    return sha256_file(path)


def _verify_v4_onnx_contract(
    path: Path,
    *,
    actor_sha256: str,
    semantic_sha256: str,
) -> None:
    try:
        import onnx
    except ImportError as error:
        raise V4ONNXUnavailableError(
            "optional dependency 'onnx' is required to verify a V4 ONNX bundle"
        ) from error
    try:
        model = onnx.load_model_from_string(path.read_bytes())
        onnx.checker.check_model(model)
        metadata: dict[str, str] = {}
        for entry in model.metadata_props:
            if entry.key in metadata:
                raise ValueError("duplicate ONNX metadata key")
            metadata[entry.key] = entry.value
    except Exception as error:
        raise ValueError("V4 ONNX model is malformed") from error
    if (
        metadata.get(ONNX_ACTOR_SHA_METADATA) != actor_sha256
        or metadata.get(ONNX_SEMANTIC_SHA_METADATA) != semantic_sha256
    ):
        raise ValueError("V4 ONNX model is not bound to actor.pt")


def export_v4_actor_bundle(
    model: V4PublicActor | V4CenteredLogitEnsemble,
    output_directory: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
    include_onnx: bool = False,
    require_onnx: bool = False,
) -> dict[str, object]:
    if require_onnx and not include_onnx:
        raise ValueError("require_onnx requires include_onnx")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    normalized_metadata = _json_metadata(metadata)
    model_kind = (
        "centered-logit-ensemble"
        if isinstance(model, V4CenteredLogitEnsemble)
        else "actor"
    )
    seeds: Sequence[int] | None = (
        model.seeds if isinstance(model, V4CenteredLogitEnsemble) else None
    )
    semantic_contract = _semantic_contract(model_kind, model.config, seeds)
    semantic_sha256 = _semantic_contract_sha256(semantic_contract)
    checkpoint_path = output / "actor.pt"
    checkpoint_sha256 = save_v4_actor_checkpoint(
        model, checkpoint_path, metadata=normalized_metadata
    )
    files: dict[str, object] = {
        checkpoint_path.name: {
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
        }
    }
    onnx_status: dict[str, object] = {
        "requested": include_onnx,
        "exported": False,
        "reason": None,
    }
    onnx_path = output / "actor.onnx"
    if onnx_path.exists():
        if not onnx_path.is_file():
            raise ValueError("V4 bundle actor.onnx path is not a file")
        onnx_path.unlink()
    if include_onnx:
        onnx_result = try_export_v4_onnx(model, onnx_path)
        onnx_status = {
            "requested": True,
            "exported": onnx_result.exported,
            "reason": onnx_result.reason,
        }
        if onnx_result.exported and onnx_result.path and onnx_result.sha256:
            bound_sha256 = _bind_v4_onnx_contract(
                onnx_result.path,
                actor_sha256=checkpoint_sha256,
                semantic_sha256=semantic_sha256,
            )
            files[onnx_result.path.name] = {
                "sha256": bound_sha256,
                "bytes": onnx_result.path.stat().st_size,
            }
        elif require_onnx:
            raise V4ONNXUnavailableError(onnx_result.reason or "ONNX export failed")
    manifest: dict[str, object] = {
        "format": V4_MANIFEST_FORMAT,
        "version": V4_MANIFEST_VERSION,
        "model": {
            "format": V4_ACTOR_FORMAT,
            "formatVersion": V4_ACTOR_FORMAT_VERSION,
            "kind": model_kind,
            "config": model.config.to_dict(),
            "seeds": list(seeds) if seeds is not None else None,
            "ensembleRule": (
                "mean of per-actor logits centered over legal actions"
                if seeds is not None
                else None
            ),
            "criticExcluded": True,
            "payloadSemanticContractSha256": semantic_sha256,
        },
        "publicInputContract": semantic_contract["publicInputContract"],
        "actionSpace": semantic_contract["actionSpace"],
        "onnx": onnx_status,
        "metadata": normalized_metadata,
        "files": files,
    }
    manifest_path = output / "manifest.json"
    _atomic_write(manifest_path, canonical_json_bytes(manifest))
    manifest_checksum = sha256_file(manifest_path)
    _atomic_write(
        output / "manifest.json.sha256",
        f"{manifest_checksum}  manifest.json\n".encode("ascii"),
    )
    return manifest


def verify_v4_actor_bundle(output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    manifest_path = output / "manifest.json"
    sidecar_path = output / "manifest.json.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("V4 bundle manifest or sidecar is missing")
    sidecar_text = sidecar_path.read_text(encoding="ascii")
    if not sidecar_text.endswith("\n") or sidecar_text.count("\n") != 1:
        raise ValueError("V4 manifest checksum sidecar is malformed")
    sidecar_parts = sidecar_text[:-1].split()
    if (
        len(sidecar_parts) != 2
        or not SHA256_RE.fullmatch(sidecar_parts[0])
        or sidecar_parts[1] != "manifest.json"
    ):
        raise ValueError("V4 manifest checksum sidecar is malformed")
    if sidecar_parts[0] != sha256_file(manifest_path):
        raise ValueError("V4 manifest checksum does not match")
    manifest = _strict_json_object(
        manifest_path.read_bytes(), "V4 bundle manifest"
    )
    version = manifest.get("version")
    if (
        manifest.get("format") != V4_MANIFEST_FORMAT
        or type(version) is not int
        or version not in {
            V4_LEGACY_MANIFEST_VERSION,
            V4_MANIFEST_VERSION,
        }
    ):
        raise ValueError("unsupported V4 bundle manifest")
    if set(manifest) != {
        "format", "version", "model", "publicInputContract", "actionSpace",
        "onnx", "metadata", "files",
    }:
        raise ValueError("V4 bundle manifest fields drifted")
    files = manifest.get("files")
    if not isinstance(files, dict) or "actor.pt" not in files:
        raise ValueError("V4 bundle file inventory is invalid")
    if not set(files).issubset({"actor.pt", "actor.onnx"}):
        raise ValueError("V4 bundle file inventory contains an unsupported artifact")
    actual_onnx_path = output / "actor.onnx"
    if actual_onnx_path.exists() and not actual_onnx_path.is_file():
        raise ValueError("V4 bundle actor.onnx path is not a file")
    if actual_onnx_path.is_file() != ("actor.onnx" in files):
        raise ValueError("V4 bundle contains an untracked or missing actor.onnx")
    for name, record in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(record, dict)
            or set(record) != {"sha256", "bytes"}
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or int(record["bytes"]) < 1
            or not isinstance(record.get("sha256"), str)
            or not SHA256_RE.fullmatch(str(record["sha256"]))
        ):
            raise ValueError("V4 bundle file inventory is invalid")
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(f"V4 bundle file is missing: {name}")
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"V4 bundle file size does not match: {name}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"V4 bundle checksum does not match: {name}")

    onnx = manifest.get("onnx")
    if (
        not isinstance(onnx, dict)
        or set(onnx) != {"requested", "exported", "reason"}
        or not isinstance(onnx.get("requested"), bool)
        or not isinstance(onnx.get("exported"), bool)
        or not (onnx.get("reason") is None or isinstance(onnx.get("reason"), str))
        or (onnx["exported"] and not onnx["requested"])
        or (onnx["exported"] != ("actor.onnx" in files))
    ):
        raise ValueError("V4 bundle ONNX inventory contract is invalid")

    _, payload = load_v4_actor_checkpoint(output / "actor.pt")
    expected_actor_version = (
        V4_LEGACY_ACTOR_FORMAT_VERSION
        if version == V4_LEGACY_MANIFEST_VERSION
        else V4_ACTOR_FORMAT_VERSION
    )
    if payload.get("version") != expected_actor_version:
        raise ValueError("V4 manifest and actor checkpoint versions do not match")
    model = manifest.get("model")
    legacy_model_keys = {
        "format", "formatVersion", "kind", "config", "seeds", "ensembleRule",
        "criticExcluded",
    }
    current_model_keys = legacy_model_keys | {"payloadSemanticContractSha256"}
    if not isinstance(model, dict) or set(model) != (
        legacy_model_keys
        if version == V4_LEGACY_MANIFEST_VERSION else current_model_keys
    ):
        raise ValueError("V4 bundle model manifest fields drifted")
    if (
        model.get("format") != V4_ACTOR_FORMAT
        or type(model.get("formatVersion")) is not int
        or model.get("formatVersion") != expected_actor_version
        or model.get("criticExcluded") is not True
    ):
        raise ValueError("V4 bundle model format is incompatible")
    config = _actor_config(model.get("config"), "V4 bundle manifest model")
    kind, seeds = _model_identity(model.get("kind"), model.get("seeds"))
    ensemble_rule = (
        "mean of per-actor logits centered over legal actions"
        if kind == "centered-logit-ensemble" else None
    )
    if model.get("ensembleRule") != ensemble_rule:
        raise ValueError("V4 bundle ensemble rule does not match its model kind")
    if (
        payload.get("kind") != kind
        or payload.get("config") != config.to_dict()
        or payload.get("seeds") != seeds
        or payload.get("criticExcluded") is not True
    ):
        raise ValueError("V4 manifest model identity does not match actor.pt")

    expected_public = _public_input_contract(config)
    expected_action = _action_space_contract(
        legacy=version == V4_LEGACY_MANIFEST_VERSION
    )
    if not _json_contract_equal(
        manifest.get("publicInputContract"), expected_public
    ):
        raise ValueError("V4 bundle public observation contract drifted")
    try:
        _validate_action_space_contract(
            manifest.get("actionSpace"), expected_action, "V4 bundle"
        )
    except ValueError as error:
        raise ValueError("V4 bundle action catalogue contract drifted") from error
    if version == V4_MANIFEST_VERSION:
        semantic_contract = _semantic_contract(kind, config, seeds)
        semantic_sha = _semantic_contract_sha256(semantic_contract)
        if (
            model.get("payloadSemanticContractSha256") != semantic_sha
            or payload.get("semanticContractSha256") != semantic_sha
            or not _json_contract_equal(
                payload.get("publicInputContract"), expected_public
            )
        ):
            raise ValueError("V4 manifest and actor semantic contracts do not match")
        _validate_action_space_contract(
            payload.get("actionSpace"), expected_action, "V4 actor checkpoint"
        )
        if "actor.onnx" in files:
            actor_record = files["actor.pt"]
            assert isinstance(actor_record, dict)
            _verify_v4_onnx_contract(
                output / "actor.onnx",
                actor_sha256=str(actor_record["sha256"]),
                semantic_sha256=semantic_sha,
            )
    metadata = manifest.get("metadata")
    if (
        not isinstance(metadata, dict)
        or not _json_contract_equal(payload.get("metadata"), metadata)
    ):
        raise ValueError("V4 manifest metadata does not match actor.pt")
    return manifest


__all__ = [
    "V4_ACTOR_FORMAT_VERSION",
    "V4_LEGACY_ACTOR_FORMAT_VERSION",
    "V4_LEGACY_MANIFEST_VERSION",
    "V4_MANIFEST_VERSION",
    "V4ONNXExportResult",
    "V4ONNXUnavailableError",
    "canonical_json_bytes",
    "export_v4_actor_bundle",
    "export_v4_onnx",
    "load_v4_actor_checkpoint",
    "make_v4_export_inputs",
    "save_v4_actor_checkpoint",
    "sha256_bytes",
    "sha256_file",
    "try_export_v4_onnx",
    "verify_v4_actor_bundle",
]
