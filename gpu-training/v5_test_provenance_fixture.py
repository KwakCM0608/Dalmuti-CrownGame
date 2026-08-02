from __future__ import annotations

"""Canonical synthetic provenance used only by V5 unit tests."""

import hashlib
import json

from v5_provenance import (
    V5_EVALUATION_BACKEND_POLICY,
    V5_EVALUATION_PROVENANCE_FORMAT,
    V5_EVALUATION_PROVENANCE_VERSION,
    V5_EVALUATION_SOURCE_FILES,
    v5_evaluation_runtime_provenance,
    validate_v5_evaluation_provenance,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _domain(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical(value)).hexdigest()


def synthetic_v5_evaluation_provenance(
    marker: str = "unit",
) -> dict[str, object]:
    files = {
        path: hashlib.sha256(f"{marker}:{path}".encode("utf-8")).hexdigest()
        for path in V5_EVALUATION_SOURCE_FILES
    }
    commit = hashlib.sha1(f"commit:{marker}".encode("utf-8")).hexdigest()
    inventory_sha = _domain(
        b"DALMUTI-V5-EVALUATION-SOURCE-INVENTORY\0", files
    )
    source_body: dict[str, object] = {
        "environmentSourceSha256": files["gpu-training/v4_env.py"],
        "evaluatorSourceSha256": files["gpu-training/v5_evaluate.py"],
        "files": files,
        "normalSourceCommit": commit,
        "normalSourceSha256": files["lib/bot-strategy.ts"],
        "simulatorSourceSha256": files["training/simulator.ts"],
        "sourceCommit": commit,
        "sourceInventorySha256": inventory_sha,
    }
    source = {
        **source_body,
        "sourceBindingSha256": _domain(
            b"DALMUTI-V5-EVALUATION-SOURCE-BINDING\0", source_body
        ),
    }
    body: dict[str, object] = {
        "artifacts": {
            "gitBundleSha256": hashlib.sha256(
                f"bundle:{marker}".encode("utf-8")
            ).hexdigest(),
            "sourceSnapshotSha256": hashlib.sha256(
                f"snapshot:{marker}".encode("utf-8")
            ).hexdigest(),
        },
        "backendPolicy": dict(V5_EVALUATION_BACKEND_POLICY),
        "format": V5_EVALUATION_PROVENANCE_FORMAT,
        "runtime": v5_evaluation_runtime_provenance("cpu"),
        "source": source,
        "version": V5_EVALUATION_PROVENANCE_VERSION,
    }
    provenance = {
        **body,
        "provenanceSha256": _domain(
            b"DALMUTI-V5-EVALUATION-PROVENANCE\0", body
        ),
    }
    return validate_v5_evaluation_provenance(provenance)


__all__ = ["synthetic_v5_evaluation_provenance"]
