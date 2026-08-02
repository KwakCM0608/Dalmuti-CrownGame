from __future__ import annotations

"""Fail-closed, Actor-only export contract for DALMUTI V5."""

from dataclasses import fields
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping

import torch

from v5_contract import (
    V5_PUBLIC_CONTRACT_SHA256,
    canonical_v5_public_contract,
    validate_v5_public_contract,
)
from v5_model import (
    V5_POLICY_NUMERICS_SHA256,
    V5ActorConfig,
    V5PublicActor,
    canonical_v5_policy_numerics_contract,
    validate_v5_policy_numerics_contract,
)


V5_ACTOR_FORMAT = "dalmuti-v5-public-normal-residual-actor"
V5_ACTOR_FORMAT_VERSION = 1
V5_MANIFEST_FORMAT = "dalmuti-v5-actor-bundle"
V5_MANIFEST_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_FILES = {
    "actor.pt",
    "config.json",
    "public-contract.json",
    "manifest.json",
    "manifest.json.sha256",
}
_PRIVATE_NAME_PARTS = ("critic", "privileged", "opponent_hand", "private_tax")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(raw: bytes, label: str) -> dict[str, object]:
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
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    return _strict_json(canonical_json_bytes(dict(value or {})), "V5 metadata")


def _actor_config(value: object) -> V5ActorConfig:
    if not isinstance(value, dict) or set(value) != {
        field.name for field in fields(V5ActorConfig)
    }:
        raise ValueError("V5 Actor configuration fields drifted")
    try:
        config = V5ActorConfig(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("V5 Actor configuration is invalid") from error
    if config.to_dict() != value:
        raise ValueError("V5 Actor configuration is non-canonical")
    return config


def _cpu_state_dict(actor: V5PublicActor) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in actor.state_dict().items()
    }
    if not state:
        raise ValueError("V5 Actor state dictionary is empty")
    for name, tensor in state.items():
        lowered = name.lower()
        if any(part in lowered for part in _PRIVATE_NAME_PARTS):
            raise ValueError(f"Actor state contains forbidden private key {name}")
        if not isinstance(tensor, torch.Tensor) or tensor.is_sparse:
            raise TypeError("V5 Actor state must contain dense tensors only")
    return state


def tensor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash exact tensor names, dtypes, shapes, and little-endian bytes."""

    digest = hashlib.sha256(b"DALMUTI-V5-TENSOR-STATE\0")
    if not state_dict:
        raise ValueError("tensor state cannot be empty")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor state requires non-empty string keys and tensors")
        value = tensor.detach().cpu().contiguous()
        if value.is_sparse:
            raise TypeError("sparse tensors are not supported")
        header = canonical_json_bytes({
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        })
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        # view(uint8) works for every dense torch dtype, including bfloat16.
        raw = value.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _checkpoint_bytes(
    actor: V5PublicActor, metadata: Mapping[str, object] | None
) -> tuple[bytes, str]:
    if type(actor) is not V5PublicActor:
        raise TypeError("only one exact V5PublicActor may be exported")
    state = _cpu_state_dict(actor)
    state_sha = tensor_state_sha256(state)
    payload: dict[str, object] = {
        "format": V5_ACTOR_FORMAT,
        "version": V5_ACTOR_FORMAT_VERSION,
        "config": actor.config.to_dict(),
        "criticIncluded": False,
        "criticExcluded": True,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "policyNumerics": canonical_v5_policy_numerics_contract(),
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "tensorStateSha256": state_sha,
        "metadata": _metadata(metadata),
        "stateDict": state,
    }
    output = io.BytesIO()
    torch.save(payload, output)
    return output.getvalue(), state_sha


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def load_v5_actor_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[V5PublicActor, dict[str, object]]:
    try:
        payload = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError("V5 Actor checkpoint could not be safely loaded") from error
    expected = {
        "format", "version", "config", "criticIncluded", "criticExcluded",
        "publicContractSha256", "policyNumerics", "policyNumericsSha256",
        "tensorStateSha256", "metadata", "stateDict",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("format") != V5_ACTOR_FORMAT
        or type(payload.get("version")) is not int
        or payload.get("version") != V5_ACTOR_FORMAT_VERSION
        or payload.get("criticIncluded") is not False
        or payload.get("criticExcluded") is not True
        or payload.get("publicContractSha256") != V5_PUBLIC_CONTRACT_SHA256
        or payload.get("policyNumericsSha256") != V5_POLICY_NUMERICS_SHA256
    ):
        raise ValueError("unsupported V5 Actor checkpoint contract")
    validate_v5_policy_numerics_contract(payload.get("policyNumerics"))
    config = _actor_config(payload.get("config"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or _metadata(metadata) != metadata:
        raise ValueError("V5 Actor checkpoint metadata is invalid")
    state = payload.get("stateDict")
    if not isinstance(state, dict):
        raise ValueError("V5 Actor checkpoint omitted its tensor state")
    for name in state:
        if not isinstance(name, str) or any(
            part in name.lower() for part in _PRIVATE_NAME_PARTS
        ):
            raise ValueError("V5 Actor checkpoint contains private/critic state")
    actual_state_sha = tensor_state_sha256(state)
    if (
        not isinstance(payload.get("tensorStateSha256"), str)
        or payload["tensorStateSha256"] != actual_state_sha
    ):
        raise ValueError("V5 Actor tensor-state checksum does not match")
    actor = V5PublicActor(config)
    try:
        actor.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("V5 Actor tensor state does not match its configuration") from error
    # Loading into the canonical class must not reinterpret any bytes.
    if tensor_state_sha256(_cpu_state_dict(actor)) != actual_state_sha:
        raise ValueError("V5 Actor tensor state changed while loading")
    return actor.eval(), payload


def export_v5_actor_bundle(
    actor: V5PublicActor,
    output_directory: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish an immutable Actor-only directory by exclusive atomic rename."""

    target = Path(output_directory).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"V5 Actor bundle already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        normalized_metadata = _metadata(metadata)
        checkpoint, state_sha = _checkpoint_bytes(actor, normalized_metadata)
        config_bytes = canonical_json_bytes(actor.config.to_dict())
        contract_bytes = canonical_json_bytes(canonical_v5_public_contract())
        _write_fsync(temporary / "actor.pt", checkpoint)
        _write_fsync(temporary / "config.json", config_bytes)
        _write_fsync(temporary / "public-contract.json", contract_bytes)
        inventory = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": sha256_file(temporary / name),
            }
            for name in ("actor.pt", "config.json", "public-contract.json")
        }
        manifest: dict[str, object] = {
            "format": V5_MANIFEST_FORMAT,
            "version": V5_MANIFEST_VERSION,
            "model": {
                "format": V5_ACTOR_FORMAT,
                "formatVersion": V5_ACTOR_FORMAT_VERSION,
                "kind": "shared-public-actor",
                "criticIncluded": False,
                "criticExcluded": True,
                "tensorStateSha256": state_sha,
            },
            "config": actor.config.to_dict(),
            "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
            "policyNumerics": canonical_v5_policy_numerics_contract(),
            "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
            "criticIncluded": False,
            "actorPtSha256": inventory["actor.pt"]["sha256"],
            "actorStateSha256": state_sha,
            "metadata": normalized_metadata,
            "files": inventory,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        _write_fsync(temporary / "manifest.json", manifest_bytes)
        manifest_sha = sha256_bytes(manifest_bytes)
        _write_fsync(
            temporary / "manifest.json.sha256",
            f"{manifest_sha}  manifest.json\n".encode("ascii"),
        )
        os.rename(temporary, target)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_v5_actor_bundle(output_directory: str | Path) -> dict[str, object]:
    target = Path(output_directory).resolve()
    if not target.is_dir():
        raise FileNotFoundError("V5 Actor bundle directory is missing")
    actual_names = {path.name for path in target.iterdir()}
    if actual_names != _BUNDLE_FILES or any(not path.is_file() for path in target.iterdir()):
        raise ValueError("V5 Actor bundle contains missing or untracked files")
    sidecar = (target / "manifest.json.sha256").read_bytes()
    match = re.fullmatch(rb"([0-9a-f]{64})  manifest\.json\n", sidecar)
    if match is None or match.group(1).decode("ascii") != sha256_file(target / "manifest.json"):
        raise ValueError("V5 manifest checksum sidecar does not match")
    manifest = _strict_json((target / "manifest.json").read_bytes(), "V5 manifest")
    if set(manifest) != {
        "format", "version", "model", "config", "publicContractSha256",
        "policyNumerics", "policyNumericsSha256", "criticIncluded",
        "actorPtSha256", "actorStateSha256", "metadata", "files",
    } or manifest.get("format") != V5_MANIFEST_FORMAT or manifest.get("version") != 1:
        raise ValueError("unsupported V5 bundle manifest")
    if manifest.get("publicContractSha256") != V5_PUBLIC_CONTRACT_SHA256:
        raise ValueError("V5 public contract fingerprint drifted")
    validate_v5_policy_numerics_contract(manifest.get("policyNumerics"))
    if manifest.get("policyNumericsSha256") != V5_POLICY_NUMERICS_SHA256:
        raise ValueError("V5 policy numerics fingerprint drifted")
    if manifest.get("criticIncluded") is not False:
        raise ValueError("V5 Actor manifest may not include a critic")
    config = _strict_json((target / "config.json").read_bytes(), "V5 config")
    contract = _strict_json(
        (target / "public-contract.json").read_bytes(), "V5 public contract"
    )
    validate_v5_public_contract(contract)
    if config != manifest.get("config"):
        raise ValueError("V5 config and manifest disagree")
    _actor_config(config)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        "actor.pt", "config.json", "public-contract.json"
    }:
        raise ValueError("V5 bundle inventory is invalid")
    for name, record in files.items():
        path = target / name
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or type(record.get("bytes")) is not int
            or not isinstance(record.get("sha256"), str)
            or not _SHA256.fullmatch(str(record["sha256"]))
            or record["bytes"] != path.stat().st_size
            or record["sha256"] != sha256_file(path)
        ):
            raise ValueError(f"V5 bundle inventory mismatch: {name}")
    actor_record = files["actor.pt"]
    assert isinstance(actor_record, dict)
    if manifest.get("actorPtSha256") != actor_record["sha256"]:
        raise ValueError("V5 manifest actor.pt checksum binding drifted")
    actor, payload = load_v5_actor_checkpoint(target / "actor.pt")
    model = manifest.get("model")
    if not isinstance(model, dict) or model != {
        "format": V5_ACTOR_FORMAT,
        "formatVersion": V5_ACTOR_FORMAT_VERSION,
        "kind": "shared-public-actor",
        "criticIncluded": False,
        "criticExcluded": True,
        "tensorStateSha256": payload["tensorStateSha256"],
    }:
        raise ValueError("V5 manifest model identity drifted")
    if manifest.get("actorStateSha256") != payload["tensorStateSha256"]:
        raise ValueError("V5 manifest exact tensor-state binding drifted")
    if (
        payload.get("config") != config
        or payload.get("metadata") != manifest.get("metadata")
        or payload.get("policyNumerics") != manifest.get("policyNumerics")
        or payload.get("policyNumericsSha256")
        != manifest.get("policyNumericsSha256")
        or actor.config.to_dict() != config
    ):
        raise ValueError("V5 manifest and Actor checkpoint disagree")
    return manifest


def load_v5_actor_bundle(
    output_directory: str | Path,
) -> tuple[V5PublicActor, dict[str, object]]:
    manifest = verify_v5_actor_bundle(output_directory)
    actor, _ = load_v5_actor_checkpoint(Path(output_directory) / "actor.pt")
    return actor, manifest


def v5_actor_bundle_digests(output_directory: str | Path) -> dict[str, str]:
    """Return the verified immutable identities used by rollout/training."""

    target = Path(output_directory).resolve()
    manifest = verify_v5_actor_bundle(target)
    model = manifest["model"]
    files = manifest["files"]
    assert isinstance(model, dict) and isinstance(files, dict)
    actor_record = files["actor.pt"]
    assert isinstance(actor_record, dict)
    return {
        "actorSha256": str(actor_record["sha256"]),
        "manifestSha256": sha256_file(target / "manifest.json"),
        "tensorStateSha256": str(model["tensorStateSha256"]),
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
    }


__all__ = [
    "V5_ACTOR_FORMAT",
    "V5_ACTOR_FORMAT_VERSION",
    "V5_MANIFEST_FORMAT",
    "V5_MANIFEST_VERSION",
    "canonical_json_bytes",
    "export_v5_actor_bundle",
    "load_v5_actor_bundle",
    "load_v5_actor_checkpoint",
    "sha256_bytes",
    "sha256_file",
    "tensor_state_sha256",
    "v5_actor_bundle_digests",
    "verify_v5_actor_bundle",
]
