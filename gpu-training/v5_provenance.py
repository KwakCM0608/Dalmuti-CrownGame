from __future__ import annotations

"""Fail-closed source and runtime provenance for DALMUTI V5 evaluation.

This is provenance for a trusted evaluator execution, not a remote-attestation
scheme.  It proves that the files visible to the verifier are byte-identical
to real blobs in one exact Git commit and records the software/backend policy
used by the evaluator.  It cannot prove that an adversarial process actually
executed those bytes; independent gameplay replay is required for that threat
model.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
from typing import Mapping, Sequence

import numpy as np
import torch

from v5_model import (
    V5_POLICY_CUBLAS_WORKSPACE_CONFIG,
    V5_POLICY_NUMERICS_SHA256,
    configure_v5_policy_numerics,
)


V5_EVALUATION_PROVENANCE_FORMAT = "dalmuti-v5-evaluation-provenance"
V5_EVALUATION_PROVENANCE_VERSION = 1

# Conservative closure of Python modules imported by the V5 evaluator and of
# the TypeScript production/reference rules that the Python environment ports.
# Keeping a few import-time-only V4 modules is deliberate: changes to their
# constants or side effects can still alter the loaded evaluator process.
V5_EVALUATION_SOURCE_FILES = tuple(sorted((
    "gpu-training/v3_action_conditioned.py",
    "gpu-training/v4_collect_dagger.py",
    "gpu-training/v4_collect_fixed_match_ppo.py",
    "gpu-training/v4_collect_ppo.py",
    "gpu-training/v4_compare_fixed_match_backends.py",
    "gpu-training/v4_dataset.py",
    "gpu-training/v4_env.py",
    "gpu-training/v4_evaluate.py",
    "gpu-training/v4_export.py",
    "gpu-training/v4_model.py",
    "gpu-training/v4_ppo_advantages.py",
    "gpu-training/v5_contract.py",
    "gpu-training/v5_evaluate.py",
    "gpu-training/v5_export.py",
    "gpu-training/v5_model.py",
    "gpu-training/v5_promotion.py",
    "gpu-training/v5_provenance.py",
    "gpu-training/v5_public.py",
    "gpu-training/v5_workflow.py",
    "lib/bot-strategy.ts",
    "lib/dealing.ts",
    "lib/round-score.ts",
    "training/action-space.ts",
    "training/non-card-action-space.ts",
    "training/non-card-observation.ts",
    "training/observation.ts",
    "training/random.ts",
    "training/simulator.ts",
    "training/v4-public-history.ts",
)))

V5_EVALUATION_BACKEND_POLICY: Mapping[str, object] = {
    "actorBackends": ["cpu", "cuda"],
    "actorInference": "eval-fp32-no-autocast-greedy-packed-legal-argmax",
    "dropout": 0.0,
    "environmentBackend": "cpu-scalar",
    "normalOpponent": "DalmutiScalarEnv.normal_action",
    "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
}

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a real regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _run_git(root: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"could not verify V5 evaluation {label}") from error


def _logical_source_files(source_files: Sequence[str]) -> tuple[str, ...]:
    if isinstance(source_files, (str, bytes)) or not source_files:
        raise ValueError("evaluation source inventory must be a non-empty sequence")
    result: list[str] = []
    for raw in source_files:
        if not isinstance(raw, str) or not raw or "\\" in raw:
            raise ValueError("evaluation source paths must be logical POSIX paths")
        logical = PurePosixPath(raw)
        if (
            logical.is_absolute()
            or raw != logical.as_posix()
            or any(part in ("", ".", "..") for part in logical.parts)
        ):
            raise ValueError("evaluation source path escaped its repository")
        result.append(raw)
    ordered = tuple(sorted(result))
    if len(ordered) != len(set(ordered)):
        raise ValueError("evaluation source inventory contains duplicate paths")
    return ordered


def resolve_v5_evaluation_source_binding(
    repository_root: str | Path,
    source_commit: str,
    *,
    source_files: Sequence[str] = V5_EVALUATION_SOURCE_FILES,
) -> dict[str, object]:
    """Verify every working source against a blob in one exact clean commit."""

    if not isinstance(source_commit, str) or _GIT_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("V5 evaluation source commit must be full 40 lowercase hex")
    unresolved_root = Path(repository_root)
    if unresolved_root.is_symlink():
        raise FileNotFoundError("V5 evaluation repository root is missing or a symlink")
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError("V5 evaluation repository root is missing or a symlink")
    top = Path(
        _run_git(root, ("rev-parse", "--show-toplevel"), "repository root")
        .decode("utf-8")
        .strip()
    ).resolve()
    if os.path.normcase(str(top)) != os.path.normcase(str(root)):
        raise ValueError("V5 evaluation repository root is not the Git worktree root")
    resolved_commit = _run_git(
        root, ("rev-parse", "--verify", f"{source_commit}^{{commit}}"), "source commit"
    ).decode("ascii").strip()
    if resolved_commit != source_commit:
        raise ValueError("V5 evaluation source commit did not resolve exactly")

    logical_files = _logical_source_files(source_files)
    status = _run_git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all", "--", *logical_files),
        "source cleanliness",
    )
    if status:
        first = status.decode("utf-8", errors="replace").splitlines()[0]
        raise ValueError(f"V5 evaluation source inventory is dirty: {first}")

    hashes: dict[str, str] = {}
    for logical in logical_files:
        unresolved = root.joinpath(*PurePosixPath(logical).parts)
        if unresolved.is_symlink():
            raise ValueError(f"V5 evaluation source must not be a symlink: {logical}")
        actual = unresolved.resolve()
        try:
            actual.relative_to(root)
        except ValueError as error:
            raise ValueError(f"V5 evaluation source escaped its repository: {logical}") from error
        if not actual.is_file():
            raise FileNotFoundError(f"required V5 evaluation source is missing: {logical}")
        working_bytes = actual.read_bytes()
        committed_bytes = _run_git(
            root, ("show", f"{source_commit}:{logical}"), f"commit blob {logical}"
        )
        if working_bytes != committed_bytes:
            raise ValueError(
                f"V5 evaluation source differs from bound commit blob: {logical}"
            )
        hashes[logical] = hashlib.sha256(working_bytes).hexdigest()

    inventory_sha = _domain_sha256(
        b"DALMUTI-V5-EVALUATION-SOURCE-INVENTORY\0", hashes
    )
    body: dict[str, object] = {
        "environmentSourceSha256": hashes.get("gpu-training/v4_env.py"),
        "evaluatorSourceSha256": hashes.get("gpu-training/v5_evaluate.py"),
        "files": hashes,
        "normalSourceCommit": source_commit,
        "normalSourceSha256": hashes.get("lib/bot-strategy.ts"),
        "simulatorSourceSha256": hashes.get("training/simulator.ts"),
        "sourceCommit": source_commit,
        "sourceInventorySha256": inventory_sha,
    }
    body["sourceBindingSha256"] = _domain_sha256(
        b"DALMUTI-V5-EVALUATION-SOURCE-BINDING\0", body
    )
    return body


def _artifact_digest(
    path: str | Path | None,
    expected_sha256: str | None,
    label: str,
) -> str | None:
    if path is None:
        if expected_sha256 is not None:
            raise ValueError(f"{label} digest was supplied without its file")
        return None
    digest = _sha256_file(Path(path), label)
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
        or digest != expected_sha256
    ):
        raise ValueError(f"{label} SHA-256 does not match its actual file")
    return digest


def _verify_git_bundle_contains_commit(
    repository_root: Path,
    bundle_path: str | Path | None,
    source_commit: str,
) -> None:
    if bundle_path is None:
        return
    bundle = Path(bundle_path).resolve()
    _run_git(
        repository_root,
        ("bundle", "verify", str(bundle)),
        "Git bundle integrity",
    )
    heads = _run_git(
        repository_root,
        ("bundle", "list-heads", str(bundle)),
        "Git bundle heads",
    ).decode("ascii", errors="strict").splitlines()
    if not any(line == source_commit or line.startswith(source_commit + " ") for line in heads):
        raise ValueError("V5 evaluation Git bundle does not expose the source commit")


def v5_evaluation_runtime_provenance(backend: str) -> dict[str, object]:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("V5 evaluation backend must be cpu or cuda")
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA provenance requested but CUDA is unavailable")
    # Apply, then record, the actual process-wide controls.  Recording a
    # constant policy without first settling the runtime would only describe
    # what the caller intended to run.
    configure_v5_policy_numerics(backend)
    cuda_backend = torch.backends.cuda
    cudnn = torch.backends.cudnn
    mha = torch.backends.mha
    cudnn_version = cudnn.version()
    cuda = backend == "cuda"
    body: dict[str, object] = {
        "backend": backend,
        "cublasWorkspaceConfig": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") if cuda else None
        ),
        "cudaRuntimeVersion": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnnVersion": None if cudnn_version is None else int(cudnn_version),
        "deviceCapability": (
            list(torch.cuda.get_device_capability()) if cuda else None
        ),
        "deviceName": torch.cuda.get_device_name() if cuda else None,
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "mhaFastPath": bool(mha.get_fastpath_enabled()),
        "flashSdp": bool(cuda_backend.flash_sdp_enabled()),
        "memoryEfficientSdp": bool(cuda_backend.mem_efficient_sdp_enabled()),
        "mathSdp": bool(cuda_backend.math_sdp_enabled()),
        "cudnnSdp": bool(cuda_backend.cudnn_sdp_enabled()),
        "cudaMatmulTf32": bool(cuda_backend.matmul.allow_tf32),
        "cudnnTf32": bool(cudnn.allow_tf32),
        "cudnnDeterministic": bool(cudnn.deterministic),
        "cudnnBenchmark": bool(cudnn.benchmark),
        "numpyVersion": str(np.__version__),
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "torchVersion": str(torch.__version__),
    }
    expected_controls = {
        "deterministicAlgorithms": True,
        "mhaFastPath": False,
        "flashSdp": False,
        "memoryEfficientSdp": False,
        "mathSdp": True,
        "cudnnSdp": False,
        "cudaMatmulTf32": False,
        "cudnnTf32": False,
        "cudnnDeterministic": True,
        "cudnnBenchmark": False,
    }
    if any(body[name] != expected for name, expected in expected_controls.items()):
        raise RuntimeError("V5 evaluation runtime numerics controls drifted")
    if cuda and body["cublasWorkspaceConfig"] != V5_POLICY_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("V5 evaluation CUDA cuBLAS workspace control drifted")
    body["runtimeSha256"] = _domain_sha256(
        b"DALMUTI-V5-EVALUATION-RUNTIME\0", body
    )
    return body


def build_v5_evaluation_provenance(
    repository_root: str | Path,
    source_commit: str,
    *,
    backend: str,
    source_snapshot: str | Path | None = None,
    source_snapshot_sha256: str | None = None,
    git_bundle: str | Path | None = None,
    git_bundle_sha256: str | None = None,
    source_files: Sequence[str] = V5_EVALUATION_SOURCE_FILES,
) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    source = resolve_v5_evaluation_source_binding(
        repository, source_commit, source_files=source_files
    )
    _verify_git_bundle_contains_commit(repository, git_bundle, source_commit)
    body: dict[str, object] = {
        "artifacts": {
            "gitBundleSha256": _artifact_digest(
                git_bundle, git_bundle_sha256, "V5 evaluation Git bundle"
            ),
            "sourceSnapshotSha256": _artifact_digest(
                source_snapshot,
                source_snapshot_sha256,
                "V5 evaluation source snapshot",
            ),
        },
        "backendPolicy": dict(V5_EVALUATION_BACKEND_POLICY),
        "format": V5_EVALUATION_PROVENANCE_FORMAT,
        "runtime": v5_evaluation_runtime_provenance(backend),
        "source": source,
        "version": V5_EVALUATION_PROVENANCE_VERSION,
    }
    result = {
        **body,
        "provenanceSha256": _domain_sha256(
            b"DALMUTI-V5-EVALUATION-PROVENANCE\0", body
        ),
    }
    return validate_v5_evaluation_provenance(result, source_files=source_files)


def validate_v5_evaluation_provenance(
    value: Mapping[str, object],
    *,
    repository_root: str | Path | None = None,
    source_snapshot: str | Path | None = None,
    git_bundle: str | Path | None = None,
    source_files: Sequence[str] = V5_EVALUATION_SOURCE_FILES,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "artifacts",
        "backendPolicy",
        "format",
        "provenanceSha256",
        "runtime",
        "source",
        "version",
    }:
        raise ValueError("V5 evaluation provenance fields drifted")
    if (
        value.get("format") != V5_EVALUATION_PROVENANCE_FORMAT
        or value.get("version") != V5_EVALUATION_PROVENANCE_VERSION
        or value.get("backendPolicy") != V5_EVALUATION_BACKEND_POLICY
    ):
        raise ValueError("V5 evaluation provenance contract drifted")

    source = value.get("source")
    source_fields = {
        "environmentSourceSha256",
        "evaluatorSourceSha256",
        "files",
        "normalSourceCommit",
        "normalSourceSha256",
        "simulatorSourceSha256",
        "sourceBindingSha256",
        "sourceCommit",
        "sourceInventorySha256",
    }
    if not isinstance(source, Mapping) or set(source) != source_fields:
        raise ValueError("V5 evaluation source provenance fields drifted")
    commit = source.get("sourceCommit")
    if (
        not isinstance(commit, str)
        or _GIT_COMMIT.fullmatch(commit) is None
        or source.get("normalSourceCommit") != commit
    ):
        raise ValueError("V5 evaluation source commit binding is invalid")
    files = source.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("V5 evaluation source inventory is missing")
    logical_files = _logical_source_files(tuple(files.keys()))
    expected_logical_files = _logical_source_files(source_files)
    if logical_files != expected_logical_files:
        raise ValueError("V5 evaluation source inventory is incomplete or unexpected")
    if tuple(files.keys()) != logical_files:
        raise ValueError("V5 evaluation source inventory is not canonically ordered")
    for digest in files.values():
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("V5 evaluation source inventory contains an invalid SHA-256")
    expected_special = {
        "environmentSourceSha256": files.get("gpu-training/v4_env.py"),
        "evaluatorSourceSha256": files.get("gpu-training/v5_evaluate.py"),
        "normalSourceSha256": files.get("lib/bot-strategy.ts"),
        "simulatorSourceSha256": files.get("training/simulator.ts"),
    }
    if any(source.get(name) != digest for name, digest in expected_special.items()):
        raise ValueError("V5 evaluation named source digest disagrees with inventory")
    inventory_sha = _domain_sha256(
        b"DALMUTI-V5-EVALUATION-SOURCE-INVENTORY\0", dict(files)
    )
    if source.get("sourceInventorySha256") != inventory_sha:
        raise ValueError("V5 evaluation source inventory SHA-256 drifted")
    source_without_binding = {
        key: source[key] for key in source if key != "sourceBindingSha256"
    }
    if source.get("sourceBindingSha256") != _domain_sha256(
        b"DALMUTI-V5-EVALUATION-SOURCE-BINDING\0", source_without_binding
    ):
        raise ValueError("V5 evaluation source binding SHA-256 drifted")

    runtime = value.get("runtime")
    runtime_fields = {
        "backend",
        "cublasWorkspaceConfig",
        "cudaRuntimeVersion",
        "cudaMatmulTf32",
        "cudnnBenchmark",
        "cudnnDeterministic",
        "cudnnSdp",
        "cudnnTf32",
        "cudnnVersion",
        "deterministicAlgorithms",
        "deviceCapability",
        "deviceName",
        "flashSdp",
        "mathSdp",
        "memoryEfficientSdp",
        "mhaFastPath",
        "numpyVersion",
        "pythonImplementation",
        "pythonVersion",
        "runtimeSha256",
        "torchVersion",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_fields:
        raise ValueError("V5 evaluation runtime provenance fields drifted")
    if runtime.get("backend") not in {"cpu", "cuda"}:
        raise ValueError("V5 evaluation runtime backend is invalid")
    for name in (
        "numpyVersion",
        "pythonImplementation",
        "pythonVersion",
        "torchVersion",
    ):
        if not isinstance(runtime.get(name), str) or not runtime[name]:
            raise ValueError(f"V5 evaluation runtime {name} is invalid")
    if runtime.get("cudaRuntimeVersion") is not None and not isinstance(
        runtime.get("cudaRuntimeVersion"), str
    ):
        raise ValueError("V5 evaluation CUDA runtime version is invalid")
    if runtime.get("cudnnVersion") is not None and type(runtime.get("cudnnVersion")) is not int:
        raise ValueError("V5 evaluation cuDNN version is invalid")
    cuda = runtime.get("backend") == "cuda"
    capability = runtime.get("deviceCapability")
    if cuda:
        if (
            runtime.get("cublasWorkspaceConfig") != V5_POLICY_CUBLAS_WORKSPACE_CONFIG
            or not isinstance(runtime.get("deviceName"), str)
            or not runtime["deviceName"]
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)
        ):
            raise ValueError("V5 evaluation CUDA device/runtime binding is invalid")
    elif any(
        runtime.get(name) is not None
        for name in ("cublasWorkspaceConfig", "deviceCapability", "deviceName")
    ):
        raise ValueError("V5 CPU provenance contains CUDA-only device fields")
    expected_controls = {
        "deterministicAlgorithms": True,
        "mhaFastPath": False,
        "flashSdp": False,
        "memoryEfficientSdp": False,
        "mathSdp": True,
        "cudnnSdp": False,
        "cudaMatmulTf32": False,
        "cudnnTf32": False,
        "cudnnDeterministic": True,
        "cudnnBenchmark": False,
    }
    if any(runtime.get(name) != expected for name, expected in expected_controls.items()):
        raise ValueError("V5 evaluation runtime numerics controls are non-canonical")
    runtime_without_sha = {
        key: runtime[key] for key in runtime if key != "runtimeSha256"
    }
    if runtime.get("runtimeSha256") != _domain_sha256(
        b"DALMUTI-V5-EVALUATION-RUNTIME\0", runtime_without_sha
    ):
        raise ValueError("V5 evaluation runtime SHA-256 drifted")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "gitBundleSha256",
        "sourceSnapshotSha256",
    }:
        raise ValueError("V5 evaluation source artifact fields drifted")
    for digest in artifacts.values():
        if digest is not None and (
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("V5 evaluation source artifact SHA-256 is invalid")

    body = {key: value[key] for key in value if key != "provenanceSha256"}
    if value.get("provenanceSha256") != _domain_sha256(
        b"DALMUTI-V5-EVALUATION-PROVENANCE\0", body
    ):
        raise ValueError("V5 evaluation provenance SHA-256 drifted")

    if repository_root is not None:
        rebuilt_source = resolve_v5_evaluation_source_binding(
            repository_root, commit, source_files=expected_logical_files
        )
        if rebuilt_source != dict(source):
            raise ValueError("V5 evaluation source provenance no longer matches disk")
    for supplied, name in (
        (git_bundle, "gitBundleSha256"),
        (source_snapshot, "sourceSnapshotSha256"),
    ):
        if supplied is not None:
            actual = _sha256_file(Path(supplied), f"V5 evaluation {name}")
            if artifacts[name] != actual:
                raise ValueError(f"V5 evaluation {name} no longer matches disk")
    return dict(value)


__all__ = [
    "V5_EVALUATION_BACKEND_POLICY",
    "V5_EVALUATION_PROVENANCE_FORMAT",
    "V5_EVALUATION_PROVENANCE_VERSION",
    "V5_EVALUATION_SOURCE_FILES",
    "build_v5_evaluation_provenance",
    "resolve_v5_evaluation_source_binding",
    "v5_evaluation_runtime_provenance",
    "validate_v5_evaluation_provenance",
]
