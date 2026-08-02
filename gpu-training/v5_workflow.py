from __future__ import annotations

"""Fail-closed local control plane for the DALMUTI V5 training workflow.

The high-volume CPU/CUDA collector remains in :mod:`v5_collect_cli`.  This
module supplies the missing immutable entry points around it: source sealing,
seeded initialization, one-epoch training, exact screening/certification/final
evaluation, shard-report merging, and final-holdout promotion.  It deliberately
contains no SSH, product integration, or deployment operations.
"""

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Mapping, Sequence

import torch

from v5_collection_plan import build_source_inventory, source_inventory_sha256
from v5_evaluate import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    FINAL_MATCH_COUNTS,
    SCREENING_MATCH_COUNTS,
    V5EvaluationConfig,
    evaluate_v5_actor,
    merge_v5_evaluation_reports,
    validate_v5_evaluation_report,
    write_v5_evaluation_report,
)
from v5_export import (
    canonical_json_bytes,
    load_v5_actor_bundle,
    sha256_file,
    v5_actor_bundle_digests,
)
from v5_promotion import (
    approve_v5_final_holdout,
    authorize_v5_certification_evaluation,
    authorize_v5_final_evaluation,
    authorize_v5_screening_evaluation,
    claim_v5_final_evaluation_shard,
    load_v5_certification_execution_reservation,
    load_v5_final_evaluation_claim,
    load_v5_promotion_plan,
    load_v5_screening_execution_reservation,
    reserve_v5_certification_execution,
    reserve_v5_final_holdout,
    reserve_v5_screening_execution,
    recover_v5_promotion_lock,
    v5_certification_coordinates,
)
from v5_provenance import (
    V5_EVALUATION_SOURCE_FILES,
    build_v5_evaluation_provenance,
    resolve_v5_evaluation_source_binding,
)
from v5_train import (
    V5TrainingConfig,
    publish_seeded_v5_initialization,
    train_v5_mappo,
    verify_v5_model_pair,
)


V5_WORKFLOW_FORMAT = "dalmuti-v5-immutable-workflow"
V5_WORKFLOW_VERSION = 1
V5_SOURCE_SEAL_FORMAT = "dalmuti-v5-source-seal"
V5_SOURCE_SEAL_VERSION = 1

V5_INITIALIZATION_SEED_BASE = 830_000_001
V5_CALIBRATION_SEED_BASE = 835_000_001
V5_COLLECTION_SEED_BASE = 840_000_001
V5_TRAINING_SEED_BASE = 850_000_001
V5_SCREENING_SEED_BASE = 860_000_001
V5_ITERATION_SEED_STRIDE = 1_000_000
V5_RUN_SEED_STRIDE = 10_000
V5_MAX_ITERATION = 39
V5_MAX_RUN_NUMBER = 100
V5_PRODUCTION_TRAINING_BATCHES: Mapping[str, int] = {
    "audit_batch_size": 64,
    "critic_batch_size": 256,
    "gradient_accumulation": 1,
    "microbatch_size": 32,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


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
            raw.decode("utf-8"),
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


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _write_canonical_with_sidecar(path: Path, value: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(dict(value))
    digest = hashlib.sha256(raw).hexdigest()
    _write_exclusive(path, raw)
    try:
        _write_exclusive(
            path.with_name(path.name + ".sha256"),
            f"{digest}  {path.name}\n".encode("ascii"),
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return digest


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes where the host exposes that primitive."""

    if os.name == "nt":
        # Windows does not expose a portable Python directory fsync.  Files
        # themselves are flushed before each no-replace link; Linux, the V5
        # collection/training host, additionally receives the directory fsync.
        return
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree_force_writable(path: Path) -> None:
    """Remove unpublished bootstrap staging, including read-only Git files.

    Git for Windows can mark generated object files read-only.  Bootstrap has
    no published run in which to retain a failed private staging tree, so its
    cleanup must not mask the original exception.
    """

    def make_writable_and_retry(
        operation: object,
        raw_path: str,
        error_info: tuple[type[BaseException], BaseException, object],
    ) -> None:
        error = error_info[1]
        if not isinstance(error, PermissionError) or not callable(operation):
            raise error
        target = Path(raw_path)
        mode = os.stat(target, follow_symlinks=False).st_mode
        os.chmod(target, mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        operation(raw_path)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def _atomic_link_bytes_noreplace(path: Path, payload: bytes) -> None:
    """Publish complete fsynced bytes with a same-directory no-replace link."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"immutable V5 file already exists: {path}")
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_canonical_with_sidecar(path: Path, value: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(dict(value))
    digest = hashlib.sha256(raw).hexdigest()
    _atomic_link_bytes_noreplace(path, raw)
    sidecar = path.with_name(path.name + ".sha256")
    try:
        _atomic_link_bytes_noreplace(
            sidecar, f"{digest}  {path.name}\n".encode("ascii")
        )
    except Exception:
        # The complete, fsynced primary is intentionally preserved.  A later
        # verifier can safely reconstruct only its checksum sidecar.
        raise
    return digest


def _ensure_checksum_sidecar(path: Path, digest: str) -> None:
    expected = f"{digest}  {path.name}\n".encode("ascii")
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        if sidecar.read_bytes() != expected:
            raise ValueError(f"checksum sidecar drifted: {sidecar}")
        return
    _atomic_link_bytes_noreplace(sidecar, expected)


def _load_canonical_with_sidecar(path: str | Path, label: str) -> tuple[dict[str, object], str]:
    target = Path(path).resolve()
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = f"{digest}  {target.name}\n".encode("ascii")
    if target.with_name(target.name + ".sha256").read_bytes() != expected:
        raise ValueError(f"{label} checksum sidecar does not match")
    return _strict_json(raw, label), digest


def _validate_source_seal(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "artifacts", "collectionSourceInventory",
        "collectionSourceInventorySha256", "evaluationSource", "format",
        "sealId", "sourceCommit", "version",
    }
    if (
        set(value) != expected
        or value.get("format") != V5_SOURCE_SEAL_FORMAT
        or value.get("version") != V5_SOURCE_SEAL_VERSION
        or not isinstance(value.get("sourceCommit"), str)
        or _GIT_COMMIT.fullmatch(str(value["sourceCommit"])) is None
    ):
        raise ValueError("V5 source seal contract drifted")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "gitBundle", "gitBundleSha256", "sourceSnapshot",
        "sourceSnapshotSha256",
    }:
        raise ValueError("V5 source seal artifact inventory drifted")
    for filename_key, digest_key in (
        ("gitBundle", "gitBundleSha256"),
        ("sourceSnapshot", "sourceSnapshotSha256"),
    ):
        filename = artifacts.get(filename_key)
        digest = artifacts.get(digest_key)
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("V5 source seal artifact record drifted")
    inventory = value.get("collectionSourceInventory")
    if (
        not isinstance(inventory, Mapping)
        or not inventory
        or source_inventory_sha256(inventory)  # type: ignore[arg-type]
        != value.get("collectionSourceInventorySha256")
    ):
        raise ValueError("V5 source seal collection inventory drifted")
    evaluation = value.get("evaluationSource")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("sourceCommit") != value["sourceCommit"]
        or not isinstance(evaluation.get("sourceBindingSha256"), str)
        or _SHA256.fullmatch(str(evaluation["sourceBindingSha256"])) is None
    ):
        raise ValueError("V5 source seal evaluation binding drifted")
    body = {key: value[key] for key in value if key != "sealId"}
    expected_id = hashlib.sha256(
        b"DALMUTI-V5-SOURCE-SEAL\0" + canonical_json_bytes(body)
    ).hexdigest()
    if value.get("sealId") != expected_id:
        raise ValueError("V5 source seal identity drifted")
    return dict(value)


def v5_seed_schedule(iteration: int, run_number: int) -> dict[str, int]:
    """Return the only development seed schedule accepted by this control plane."""

    if (
        type(iteration) is not int
        or type(run_number) is not int
        or not 1 <= iteration <= V5_MAX_ITERATION
        or not 1 <= run_number <= V5_MAX_RUN_NUMBER
    ):
        raise ValueError(
            f"iteration/run must be within 1..{V5_MAX_ITERATION} and "
            f"1..{V5_MAX_RUN_NUMBER}"
        )
    offset = (
        (iteration - 1) * V5_ITERATION_SEED_STRIDE
        + (run_number - 1) * V5_RUN_SEED_STRIDE
    )
    seeds = {
        "initialization": V5_INITIALIZATION_SEED_BASE + offset,
        "calibration": V5_CALIBRATION_SEED_BASE + offset,
        "collection": V5_COLLECTION_SEED_BASE + offset,
        "training": V5_TRAINING_SEED_BASE + offset,
        "screening": V5_SCREENING_SEED_BASE + offset,
    }
    if len(set(seeds.values())) != len(seeds) or max(seeds.values()) >= 900_000_001:
        raise RuntimeError("V5 development seed schedule overlaps the final holdout namespace")
    return seeds


def v5_run_namespace(iteration: int, run_number: int) -> str:
    seeds = v5_seed_schedule(iteration, run_number)
    return (
        f"v5-mappo-normalresidual-i{iteration:03d}-"
        f"s{seeds['collection']}-run-{run_number:03d}"
    )


def v5_run_directory_name(iteration: int, run_number: int) -> str:
    return v5_run_namespace(iteration, run_number)


def _run_git(repository: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "-C",
                str(repository),
                *arguments,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"V5 source seal Git command failed: {' '.join(arguments)}") from error


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any existing target."""

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
        # Windows MoveFile/rename is no-replace when the destination exists.
        os.rename(source, target)
        return
    raise RuntimeError("atomic no-replace directory publication is unsupported")


def _require_clean_exact_head(repository: Path, source_commit: str) -> None:
    if not isinstance(source_commit, str) or _GIT_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("V5 source seal requires a full 40-character lowercase commit")
    top = Path(_run_git(repository, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if os.path.normcase(str(top)) != os.path.normcase(str(repository)):
        raise ValueError("V5 source seal repository root is not the Git worktree root")
    head = _run_git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    if head != source_commit:
        raise ValueError("V5 source seal commit must be the exact current HEAD")
    dirty = _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        first = dirty.decode("utf-8", errors="replace").splitlines()[0]
        raise ValueError(f"V5 source seal requires a clean worktree: {first}")


def _publish_source_seal(
    repository: Path,
    source_commit: str,
    output_directory: Path,
) -> tuple[dict[str, object], str]:
    """Create a Git bundle and Git archive bound to one clean exact HEAD."""

    # The caller performs the whole-worktree cleanliness check before it
    # creates any in-worktree staging directory.  From here onward we bind and
    # re-open the exact behavior-affecting files around archive creation.
    head = _run_git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    if head != source_commit:
        raise ValueError("V5 source seal commit changed before archive creation")
    evaluation_binding = resolve_v5_evaluation_source_binding(
        repository, source_commit
    )
    collection_inventory = build_source_inventory(repository)
    output_directory.mkdir(parents=False, exist_ok=False)
    bundle = output_directory / "source.bundle"
    snapshot = output_directory / "source.tar"
    _run_git(repository, "bundle", "create", str(bundle), "HEAD")
    _run_git(repository, "bundle", "verify", str(bundle))
    _run_git(
        repository,
        "archive",
        "--format=tar",
        "--output",
        str(snapshot),
        source_commit,
    )
    with tarfile.open(snapshot, mode="r:") as archive:
        archived = {member.name.rstrip("/") for member in archive.getmembers()}
    missing = sorted(set(V5_EVALUATION_SOURCE_FILES) - archived)
    if missing:
        raise ValueError(f"V5 source snapshot omitted required source: {missing[0]}")
    # Re-open the behavior-affecting files after archive creation.  This makes
    # a concurrent edit fail before the seal is published.
    if resolve_v5_evaluation_source_binding(repository, source_commit) != evaluation_binding:
        raise ValueError("V5 evaluation sources changed while sealing")
    if build_source_inventory(repository) != collection_inventory:
        raise ValueError("V5 collection sources changed while sealing")
    body: dict[str, object] = {
        "artifacts": {
            "gitBundle": bundle.name,
            "gitBundleSha256": sha256_file(bundle),
            "sourceSnapshot": snapshot.name,
            "sourceSnapshotSha256": sha256_file(snapshot),
        },
        "collectionSourceInventory": collection_inventory,
        "collectionSourceInventorySha256": source_inventory_sha256(
            collection_inventory
        ),
        "evaluationSource": evaluation_binding,
        "format": V5_SOURCE_SEAL_FORMAT,
        "sourceCommit": source_commit,
        "version": V5_SOURCE_SEAL_VERSION,
    }
    seal_id = hashlib.sha256(
        b"DALMUTI-V5-SOURCE-SEAL\0" + canonical_json_bytes(body)
    ).hexdigest()
    document = _validate_source_seal({**body, "sealId": seal_id})
    digest = _write_canonical_with_sidecar(
        output_directory / "manifest.json", document
    )
    return document, digest


def bootstrap_v5_run(
    run_root: str | Path,
    *,
    repository_root: str | Path,
    source_commit: str,
    iteration: int,
    run_number: int,
) -> dict[str, object]:
    """Atomically publish one new source-sealed run and its initial model pair."""

    target = Path(run_root).resolve()
    expected_name = v5_run_directory_name(iteration, run_number)
    if target.name != expected_name:
        raise ValueError(f"V5 run directory must be named exactly {expected_name}")
    if target.exists():
        raise FileExistsError(f"V5 run directory already exists: {target}")
    repository = Path(repository_root).resolve()
    seeds = v5_seed_schedule(iteration, run_number)
    namespace = v5_run_namespace(iteration, run_number)
    target.parent.mkdir(parents=True, exist_ok=True)
    # This must happen before an in-repository staging path is created, or the
    # staging path itself may make an otherwise clean worktree appear dirty.
    _require_clean_exact_head(repository, source_commit)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        seal, seal_sha = _publish_source_seal(
            repository, source_commit, staging / "source-seal"
        )
        initialization = publish_seeded_v5_initialization(
            staging / "initialization",
            seed=seeds["initialization"],
            metadata={
                "runNamespace": namespace,
                "sourceCommit": source_commit,
                "sourceSealSha256": seal_sha,
            },
        )
        pair = verify_v5_model_pair(staging / "initialization")
        body: dict[str, object] = {
            "directories": {
                "collectionPlan": "collection/plan",
                "collectionShards": "canonical-shards",
                "datasetIndex": "collection/index",
                "evaluation": "evaluation",
                "initialization": "initialization",
                "sourceSeal": "source-seal",
                "training": "training",
            },
            "format": V5_WORKFLOW_FORMAT,
            "initialModelPair": {
                "pairId": pair["pairId"],
                "pairManifestSha256": pair["pairManifestSha256"],
            },
            "iteration": iteration,
            "runDirectoryName": expected_name,
            "runNamespace": namespace,
            "runNumber": run_number,
            "seeds": seeds,
            "sourceCommit": source_commit,
            "sourceSeal": {
                "manifestSha256": seal_sha,
                "sealId": seal["sealId"],
            },
            "version": V5_WORKFLOW_VERSION,
        }
        workflow_sha = _write_canonical_with_sidecar(
            staging / "workflow.json", body
        )
        _rename_directory_noreplace(staging, target)
        return {
            "initialActorBundle": str(target / "initialization" / "actor-bundle"),
            "initialCriticCheckpoint": str(target / "initialization" / "critic.pt"),
            "runRoot": str(target),
            "sourceSealSha256": seal_sha,
            "workflow": body,
            "workflowSha256": workflow_sha,
            "initializationSha256": initialization["initializationSha256"],
        }
    finally:
        if staging.exists():
            _remove_tree_force_writable(staging)


def materialize_v5_source_checkout(
    run_root: str | Path,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Clone the sealed bundle and detach at the exact source commit.

    Remote collection/evaluation must execute from this Git worktree.  Merely
    extracting ``source.tar`` would not provide the commit/blob/cleanliness
    evidence required by :mod:`v5_provenance`.
    """

    workflow = load_v5_run(run_root)
    root = Path(run_root).resolve()
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else root / "source-checkout"
    )
    if target != root / "source-checkout":
        raise ValueError("sealed V5 source checkout must be <run-root>/source-checkout")
    if target.exists():
        raise FileExistsError(f"sealed V5 source checkout already exists: {target}")
    seal, _ = _load_canonical_with_sidecar(
        root / "source-seal" / "manifest.json", "V5 source seal"
    )
    seal = _validate_source_seal(seal)
    artifacts = seal["artifacts"]
    assert isinstance(artifacts, Mapping)
    bundle = root / "source-seal" / str(artifacts["gitBundle"])
    source_commit = str(workflow["sourceCommit"])
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    # git clone requires a path it can create itself.
    staging.rmdir()
    try:
        try:
            subprocess.run(
                ["git", "clone", "--no-checkout", str(bundle), str(staging)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("could not clone the sealed V5 Git bundle") from error
        # Materialized bytes are part of the cryptographic source binding.
        # Never inherit a host-level core.autocrlf setting that rewrites LF
        # blobs to CRLF during checkout (notably the bundled Git on Windows).
        _run_git(staging, "config", "--local", "core.autocrlf", "false")
        _run_git(staging, "config", "--local", "core.eol", "lf")
        _run_git(staging, "checkout", "--detach", source_commit)
        _require_clean_exact_head(staging, source_commit)
        evaluation = resolve_v5_evaluation_source_binding(staging, source_commit)
        collection = build_source_inventory(staging)
        if evaluation != seal.get("evaluationSource"):
            raise ValueError("materialized V5 evaluation sources differ from the seal")
        if (
            collection != seal.get("collectionSourceInventory")
            or source_inventory_sha256(collection)
            != seal.get("collectionSourceInventorySha256")
        ):
            raise ValueError("materialized V5 collection sources differ from the seal")
        _rename_directory_noreplace(staging, target)
        return {
            "repositoryRoot": str(target),
            "sourceCommit": source_commit,
            "sourceSealId": seal["sealId"],
        }
    except BaseException as error:
        # Preserve an unsuccessful checkout byte-for-byte as failure evidence.
        # The canonical target remains absent, so an operator can inspect the
        # hidden tree and retry this same sealed run without ambiguity.
        try:
            error.add_note(f"failed sealed checkout preserved at {staging}")
        except (AttributeError, TypeError):
            pass
        raise


def load_v5_run(run_root: str | Path) -> dict[str, object]:
    root = Path(run_root).resolve()
    workflow, _ = _load_canonical_with_sidecar(
        root / "workflow.json", "V5 workflow"
    )
    expected_fields = {
        "directories",
        "format",
        "initialModelPair",
        "iteration",
        "runDirectoryName",
        "runNamespace",
        "runNumber",
        "seeds",
        "sourceCommit",
        "sourceSeal",
        "version",
    }
    expected_directories = {
        "collectionPlan": "collection/plan",
        "collectionShards": "canonical-shards",
        "datasetIndex": "collection/index",
        "evaluation": "evaluation",
        "initialization": "initialization",
        "sourceSeal": "source-seal",
        "training": "training",
    }
    if (
        set(workflow) != expected_fields
        or workflow.get("format") != V5_WORKFLOW_FORMAT
        or workflow.get("version") != V5_WORKFLOW_VERSION
        or type(workflow.get("iteration")) is not int
        or type(workflow.get("runNumber")) is not int
        or workflow.get("directories") != expected_directories
    ):
        raise ValueError("V5 workflow contract drifted")
    iteration = int(workflow["iteration"])
    run_number = int(workflow["runNumber"])
    if (
        workflow.get("runDirectoryName") != v5_run_directory_name(iteration, run_number)
        or root.name != workflow["runDirectoryName"]
        or workflow.get("runNamespace") != v5_run_namespace(iteration, run_number)
        or workflow.get("seeds") != v5_seed_schedule(iteration, run_number)
        or not isinstance(workflow.get("sourceCommit"), str)
        or _GIT_COMMIT.fullmatch(str(workflow["sourceCommit"])) is None
    ):
        raise ValueError("V5 workflow identity or seed schedule drifted")
    seal, seal_sha = _load_canonical_with_sidecar(
        root / "source-seal" / "manifest.json", "V5 source seal"
    )
    seal = _validate_source_seal(seal)
    if (
        seal.get("format") != V5_SOURCE_SEAL_FORMAT
        or seal.get("version") != V5_SOURCE_SEAL_VERSION
        or seal.get("sourceCommit") != workflow["sourceCommit"]
        or not isinstance(workflow.get("sourceSeal"), Mapping)
        or workflow["sourceSeal"].get("manifestSha256") != seal_sha  # type: ignore[union-attr]
        or workflow["sourceSeal"].get("sealId") != seal.get("sealId")  # type: ignore[union-attr]
    ):
        raise ValueError("V5 workflow source-seal binding drifted")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("V5 source seal artifact inventory is missing")
    for name, digest_name in (
        ("gitBundle", "gitBundleSha256"),
        ("sourceSnapshot", "sourceSnapshotSha256"),
    ):
        filename = artifacts.get(name)
        expected = artifacts.get(digest_name)
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
            or sha256_file(root / "source-seal" / filename) != expected
        ):
            raise ValueError(f"V5 source seal artifact drifted: {name}")
    pair = verify_v5_model_pair(root / "initialization")
    if workflow.get("initialModelPair") != {
        "pairId": pair["pairId"],
        "pairManifestSha256": pair["pairManifestSha256"],
    }:
        raise ValueError("V5 workflow initial model-pair binding drifted")
    return workflow


def train_v5_run(
    run_root: str | Path,
    dataset_index: str | Path,
    *,
    device: str,
    repository_root: str | Path,
    gpu_memory_preflight: str | Path,
    initial_model_pair: str | Path | None = None,
    low_disk_persistent_root: str | Path | None = None,
    low_disk_volatile_root: str | Path | None = None,
    low_disk_promotion_receipt_root: str | Path | None = None,
    config_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    workflow = load_v5_run(run_root)
    root = Path(run_root).resolve()
    _verify_v5_training_execution_source(root, workflow, repository_root)
    pair_root = root / "initialization"
    if (
        initial_model_pair is not None
        and Path(initial_model_pair).resolve() != pair_root
    ):
        raise ValueError(
            "external V5 initial model-pair overrides are not source-sealed; "
            "bootstrap a new run with a bound pair instead"
        )
    pair = verify_v5_model_pair(pair_root)
    canonical_dataset = _verify_v5_training_dataset(
        root,
        workflow,
        dataset_index,
        pair,
        low_disk_persistent_root=low_disk_persistent_root,
        low_disk_volatile_root=low_disk_volatile_root,
        low_disk_promotion_receipt_root=low_disk_promotion_receipt_root,
    )
    values = dict(config_overrides or {})
    if "seed" in values and values["seed"] != workflow["seeds"]["training"]:  # type: ignore[index]
        raise ValueError("V5 training seed must match the immutable run schedule")
    values["seed"] = workflow["seeds"]["training"]  # type: ignore[index]
    for name, required in V5_PRODUCTION_TRAINING_BATCHES.items():
        supplied = values.get(name, required)
        if supplied != required:
            raise ValueError(
                f"V5 production workflow requires {name}={required} from "
                "the RTX 3080 admission calibration"
            )
        values[name] = required
    config = V5TrainingConfig(**values)
    from v5_gpu_memory_preflight import (
        V5GPUMemoryPreflightConfig,
        load_v5_gpu_memory_preflight_report,
        verify_v5_gpu_memory_admission,
    )

    preflight_report, _ = load_v5_gpu_memory_preflight_report(
        gpu_memory_preflight
    )
    raw_preflight_config = preflight_report.get("config")
    if not isinstance(raw_preflight_config, Mapping):
        raise ValueError("V5 GPU memory preflight omitted its configuration")
    preflight_config = V5GPUMemoryPreflightConfig(**raw_preflight_config)
    expected_batches = {
        "audit_batch_size": config.audit_batch_size,
        "critic_batch_size": config.critic_batch_size,
        "gradient_accumulation": config.gradient_accumulation,
        "microbatch_size": config.microbatch_size,
    }
    if any(
        getattr(preflight_config, name) != expected
        for name, expected in expected_batches.items()
    ):
        raise ValueError(
            "V5 training Actor/Critic/audit batches differ from GPU admission"
        )
    admission = verify_v5_gpu_memory_admission(
        gpu_memory_preflight,
        canonical_dataset,
        pair_root,
        config=preflight_config,
        device=device,
    )
    return train_v5_mappo(
        canonical_dataset,
        pair_root / "actor-bundle",
        pair_root / "critic.pt",
        root / "training",
        config=config,
        device=device,
        gpu_memory_preflight=admission,
    )


def _verify_v5_training_dataset(
    run_root: Path,
    workflow: Mapping[str, object],
    dataset_index: str | Path,
    initial_pair: Mapping[str, object],
    *,
    low_disk_persistent_root: str | Path | None = None,
    low_disk_volatile_root: str | Path | None = None,
    low_disk_promotion_receipt_root: str | Path | None = None,
) -> Path:
    """Re-open the one production index bound to this immutable run.

    The trainer is intentionally not a generic ``--dataset`` entry point.  A
    production run may consume only the index published below its own
    ``collection`` directory, after the collection plan, every shard, the
    index manifest, and all checksum sidecars have been revalidated.
    """

    from v5_collection_plan import (
        load_collection_plan,
        validate_actual_nonforced_corpus,
        verify_planned_collection_corpus,
    )
    from v5_dataset import load_v5_index_manifest
    from v5_low_disk_stage import (
        LOW_DISK_STAGE_PLAN_NAME,
        SOURCE_INDEX_RECORD_NAME,
        verify_v5_hybrid_stage,
    )

    canonical = (run_root / "collection" / "index").resolve()
    supplied = Path(dataset_index).resolve()
    if os.path.normcase(str(supplied)) != os.path.normcase(str(canonical)):
        raise ValueError(
            "V5 production training dataset must be the canonical "
            "<run-root>/collection/index"
        )
    if not canonical.is_dir() or {path.name for path in canonical.iterdir()} != {
        "manifest.json",
        "manifest.json.sha256",
    }:
        raise ValueError("canonical V5 production index inventory drifted")

    plan_root = (run_root / "collection" / "plan").resolve()
    plan = load_collection_plan(plan_root)
    if plan.purpose != "production":
        raise ValueError("diagnostic V5 collection plans cannot reach training")
    seeds = workflow.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ValueError("V5 workflow seed schedule is missing")
    if (
        plan.run_namespace != workflow.get("runNamespace")
        or plan.seed_base != seeds.get("collection")
    ):
        raise ValueError("V5 collection plan namespace/seed differs from its run")

    pair_projection = {
        "actorManifestSha256": initial_pair.get("actorManifestSha256"),
        "actorSha256": initial_pair.get("actorSha256"),
        "criticSha256": initial_pair.get("criticSha256"),
        "pairId": initial_pair.get("pairId"),
        "pairManifestSha256": initial_pair.get("pairManifestSha256"),
    }
    if plan.behavior != pair_projection:
        raise ValueError("V5 collection behavior is not the run's initial model pair")
    workflow_pair = workflow.get("initialModelPair")
    if not isinstance(workflow_pair, Mapping) or dict(workflow_pair) != {
        "pairId": initial_pair.get("pairId"),
        "pairManifestSha256": initial_pair.get("pairManifestSha256"),
    }:
        raise ValueError("V5 workflow initial model-pair binding drifted")

    seal, _ = _load_canonical_with_sidecar(
        run_root / "source-seal" / "manifest.json", "V5 source seal"
    )
    seal = _validate_source_seal(seal)
    if (
        plan.source_inventory != seal.get("collectionSourceInventory")
        or plan.document.get("sourceInventorySha256")
        != seal.get("collectionSourceInventorySha256")
        or seal.get("sourceCommit") != workflow.get("sourceCommit")
    ):
        raise ValueError("V5 collection plan source binding differs from its run seal")

    targets = plan.document.get("targets")
    if not isinstance(targets, Mapping) or (
        targets.get("targetNonforcedDecisions") != 1_600_000
        or targets.get("minimumNonforcedDecisions") != 1_500_000
        or targets.get("maximumNonforcedDecisions") != 2_000_000
        or targets.get("actualStratumRelativeTolerance") != 0.08
    ):
        raise ValueError(
            "V5 production corpus gates must remain target=1.6M, "
            "range=1.5M..2.0M, and +/-8%"
        )

    low_disk_controls = {
        "plan": run_root / "collection" / LOW_DISK_STAGE_PLAN_NAME,
        "sourceIndex": run_root / "collection" / SOURCE_INDEX_RECORD_NAME,
    }
    low_disk_control_presence = {
        name: os.path.lexists(os.fspath(path))
        for name, path in low_disk_controls.items()
    }
    supplied_low_disk_roots = {
        "persistent": low_disk_persistent_root,
        "volatile": low_disk_volatile_root,
        "receipts": low_disk_promotion_receipt_root,
    }
    hybrid_stage = None
    if any(low_disk_control_presence.values()):
        if not all(low_disk_control_presence.values()):
            missing = sorted(
                name
                for name, present in low_disk_control_presence.items()
                if not present
            )
            raise ValueError(
                "partial V5 low-disk stage cannot reach training; missing "
                + ", ".join(missing)
            )
        if any(value is None for value in supplied_low_disk_roots.values()):
            raise ValueError(
                "V5 low-disk training requires explicit persistent, volatile, "
                "and promotion-receipt roots"
            )
        low_disk_roots = {
            # Preserve the caller's unresolved path so the low-disk verifier
            # can reject a symlink before resolving its exact namespaced root.
            name: Path(value)  # type: ignore[arg-type]
            for name, value in supplied_low_disk_roots.items()
        }
        hybrid_stage = verify_v5_hybrid_stage(
            low_disk_controls["plan"],
            persistent_root=low_disk_roots["persistent"],
            volatile_root=low_disk_roots["volatile"],
            source_index_record=low_disk_controls["sourceIndex"],
            promotion_receipt_root=low_disk_roots["receipts"],
            hybrid_index=canonical,
            collection_plan=plan_root,
        )
    elif any(value is not None for value in supplied_low_disk_roots.values()):
        raise ValueError(
            "low-disk roots were supplied without this run's stage controls"
        )

    index = load_v5_index_manifest(canonical)
    try:
        verified_corpus = verify_planned_collection_corpus(
            plan,
            run_root / "canonical-shards",
            index_shard_paths=index.shard_paths,
        )
        decisions_by_player = verified_corpus["actualDecisionCountsByPlayerCount"]
        nonforced_by_player = verified_corpus[
            "actualNonforcedDecisionCountsByPlayerCount"
        ]
        matches_by_player = verified_corpus["actualMatchCountsByPlayerCount"]
        manifest_hashes = verified_corpus["shardManifestSha256s"]
        assert isinstance(nonforced_by_player, Mapping)
        corpus_gate = validate_actual_nonforced_corpus(plan, nonforced_by_player)
        expected_metadata = {
            "actualCorpusGate": corpus_gate,
            "actualDecisionCountsByPlayerCount": decisions_by_player,
            "actualMatchCountsByPlayerCount": matches_by_player,
            "actualNonforcedDecisionCountsByPlayerCount": nonforced_by_player,
            "behavior": dict(plan.behavior),
            "calibrationReportSha256": plan.document["calibration"]["reportSha256"],  # type: ignore[index]
            "collectionPlanManifestSha256": plan.manifest_sha256,
            "completeShardIndices": verified_corpus["completeShardIndices"],
            "matchCoordinatesSha256": verified_corpus[
                "matchCoordinatesSha256"
            ],
            "matchProvenanceContract": verified_corpus[
                "matchProvenanceContract"
            ],
            "plannedMatchCountsByPlayerCount": plan.document["matchCounts"],
            "policyNumericsSha256": plan.document["policyNumericsSha256"],
            "shardManifestSha256s": manifest_hashes,
            "sourceInventorySha256": plan.document["sourceInventorySha256"],
            "totalUniqueMatches": verified_corpus["totalUniqueMatches"],
        }
        metadata_subject = (
            hybrid_stage.source_index.document.get("metadata")
            if hybrid_stage is not None
            else index.manifest.get("metadata")
        )
        if metadata_subject != expected_metadata:
            raise ValueError("V5 production index metadata does not recompute")
        if hybrid_stage is not None and (
            tuple(Path(path).resolve() for path in index.shard_paths)
            != tuple(Path(path).resolve() for path in hybrid_stage.shard_paths)
            or hybrid_stage.actual_decision_counts_by_player_count
            != decisions_by_player
            or hybrid_stage.actual_match_counts_by_player_count
            != matches_by_player
            or hybrid_stage.actual_nonforced_decision_counts_by_player_count
            != nonforced_by_player
            or hybrid_stage.actual_corpus_gate != corpus_gate
        ):
            raise ValueError("V5 hybrid stage differs from the recomputed corpus")
        if (
            index.match_count != int(plan.document["totalMatches"])
            or verified_corpus["totalUniqueMatches"]
            != int(plan.document["totalMatches"])
            or index.manifest.get("playerCounts") != list(range(4, 11))
            or corpus_gate.get("passed") is not True
            or not 1_500_000 <= int(corpus_gate["total"]) <= 2_000_000
            or corpus_gate.get("relativeTolerance") != 0.08
        ):
            raise ValueError("V5 production index failed its actual corpus gate")
    finally:
        index.close()
    return canonical


def _verify_v5_training_execution_source(
    run_root: Path,
    workflow: Mapping[str, object],
    repository_root: str | Path,
) -> None:
    repository = Path(repository_root).resolve()
    source_commit = str(workflow["sourceCommit"])
    _require_clean_exact_head(repository, source_commit)
    seal, _ = _load_canonical_with_sidecar(
        run_root / "source-seal" / "manifest.json", "V5 source seal"
    )
    seal = _validate_source_seal(seal)
    if (
        resolve_v5_evaluation_source_binding(repository, source_commit)
        != seal["evaluationSource"]
        or build_source_inventory(repository)
        != seal["collectionSourceInventory"]
    ):
        raise ValueError("V5 training execution sources differ from the run seal")
    expected_workflow = repository / "gpu-training" / "v5_workflow.py"
    expected_trainer = repository / "gpu-training" / "v5_train.py"
    actual_workflow = Path(__file__).resolve()
    actual_trainer = Path(train_v5_mappo.__code__.co_filename).resolve()
    if (
        os.path.normcase(str(actual_workflow))
        != os.path.normcase(str(expected_workflow))
        or os.path.normcase(str(actual_trainer))
        != os.path.normcase(str(expected_trainer))
    ):
        raise ValueError(
            "V5 train must execute v5_workflow.py and v5_train.py from the "
            "materialized sealed checkout"
        )


def _load_evaluation_report(path: str | Path) -> dict[str, object]:
    target = Path(path).resolve()
    report = _strict_json(target.read_bytes(), "V5 evaluation report")
    return validate_v5_evaluation_report(report)


def _load_workflow_evaluation_report(path: str | Path) -> dict[str, object]:
    target = Path(path).resolve()
    report = _load_evaluation_report(target)
    _ensure_checksum_sidecar(target, sha256_file(target))
    return report


def _write_workflow_evaluation_report(
    path: str | Path, report: Mapping[str, object]
) -> str:
    target = Path(path).resolve()
    digest = write_v5_evaluation_report(target, report)
    # The evaluator publishes a fully fsynced same-directory temp with a
    # no-replace hard link.  Persist that directory entry and then publish the
    # independently recoverable checksum sidecar.
    _fsync_directory(target.parent)
    if not target.is_file() or sha256_file(target) != digest:
        raise RuntimeError("V5 evaluator did not publish its exact immutable report")
    _ensure_checksum_sidecar(target, digest)
    return digest


def merge_v5_evaluation_report_files(
    report_paths: Sequence[str | Path], output: str | Path
) -> dict[str, object]:
    if not report_paths:
        raise ValueError("at least one evaluation shard report is required")
    reports = [_load_evaluation_report(path) for path in report_paths]
    merged = merge_v5_evaluation_reports(reports)
    digest = _write_workflow_evaluation_report(output, merged)
    return {
        "completeEvaluation": merged["completeEvaluation"],
        "allPlayerCountsPassed": merged["allPlayerCountsPassed"],
        "output": str(Path(output).resolve()),
        "reportSha256": digest,
    }


def _source_provenance_for_run(
    run_root: Path, workflow: Mapping[str, object], repository_root: str | Path, device: str
) -> dict[str, object]:
    seal, _ = _load_canonical_with_sidecar(
        run_root / "source-seal" / "manifest.json", "V5 source seal"
    )
    seal = _validate_source_seal(seal)
    artifacts = seal["artifacts"]
    assert isinstance(artifacts, Mapping)
    repository = Path(repository_root).resolve()
    source_commit = str(workflow["sourceCommit"])
    # build_v5_evaluation_provenance independently reopens every source file,
    # archive, and digest.  Passing the recorded digest is intentional.
    return build_v5_evaluation_provenance(
        repository,
        source_commit,
        backend=str(torch.device(device).type),
        source_snapshot=run_root / "source-seal" / str(artifacts["sourceSnapshot"]),
        source_snapshot_sha256=str(artifacts["sourceSnapshotSha256"]),
        git_bundle=run_root / "source-seal" / str(artifacts["gitBundle"]),
        git_bundle_sha256=str(artifacts["gitBundleSha256"]),
    )


def _passed_screening_for_actor(
    screening_report: str | Path,
    model: Mapping[str, object],
    screening_reservation: str | Path,
    expected_provenance: Mapping[str, object],
) -> dict[str, object]:
    reservation_path = Path(screening_reservation).resolve()
    reservation = load_v5_screening_execution_reservation(reservation_path)
    coordinate = reservation["coordinate"]
    assert isinstance(coordinate, Mapping)
    counts = {
        int(key): int(value)
        for key, value in coordinate["matchPlan"].items()  # type: ignore[union-attr]
    }
    report_path = Path(screening_report).resolve()
    registry = reservation_path.parent.parent
    expected_output = _reserved_registry_path(
        registry, reservation["outputPath"], "screening result path"
    )
    if report_path != expected_output:
        raise ValueError("screening report is outside its one-shot reservation")
    config = V5EvaluationConfig(
        mode="screening",
        family_id=str(coordinate["familyId"]),
        seed_base=int(coordinate["seedBase"]),
        match_counts=tuple(sorted(counts.items())),
        match_shard_count=1,
        match_shard_index=0,
        lane_count=32,
        bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    binding = authorize_v5_screening_evaluation(
        reservation_path,
        model,
        evaluation_provenance=expected_provenance,
        family_id=config.family_id,
        seed_base=config.seed_base,
        match_plan=counts,
        match_shard_count=1,
        match_shard_index=0,
        bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        output_path=report_path,
    )
    screening = _verify_exact_evaluation_execution(
        _load_workflow_evaluation_report(report_path),
        config,
        model,
        provenance=expected_provenance,
        certification_binding=None,
        final_binding=None,
        screening_binding=binding,
    )
    markers = _execution_markers(report_path)
    if not markers:
        raise ValueError("screening report lacks its workflow execution marker")
    identity = markers[-1][1].get("executionIdentity")
    if not isinstance(identity, Mapping) or identity != _evaluation_execution_identity(
        report_path,
        stage="screening",
        device=str(identity.get("device")),
        config=config,
        model=model,
        provenance=expected_provenance,
        binding=binding,
    ):
        raise ValueError("screening execution marker binding drifted")
    if (
        screening.get("completeEvaluation") is not True
        or screening.get("allPlayerCountsPassed") is not True
    ):
        raise ValueError("certification prerequisite screening failed")
    return screening


def reserve_v5_screening_run(
    run_root: str | Path,
    registry: str | Path,
    actor_bundle: str | Path,
    *,
    repository_root: str | Path,
    device: str,
) -> dict[str, object]:
    """Reserve this functional Actor's only production screening execution."""

    workflow = load_v5_run(run_root)
    root = Path(run_root).resolve()
    provenance = _source_provenance_for_run(
        root, workflow, repository_root, device
    )
    return reserve_v5_screening_execution(
        registry, actor_bundle, provenance
    )


def reserve_v5_certification_run(
    run_root: str | Path,
    registry: str | Path,
    actor_bundle: str | Path,
    screening_report: str | Path,
    *,
    screening_reservation: str | Path,
    repository_root: str | Path,
    device: str,
) -> dict[str, object]:
    """Reserve both certification coordinates after the passed screening gate."""

    workflow = load_v5_run(run_root)
    root = Path(run_root).resolve()
    bundle = Path(actor_bundle).resolve()
    model = v5_actor_bundle_digests(bundle)
    provenance = _source_provenance_for_run(
        root, workflow, repository_root, device
    )
    _passed_screening_for_actor(
        screening_report, model, screening_reservation, provenance
    )
    return reserve_v5_certification_execution(
        registry,
        bundle,
        provenance,
        screening_reservation=screening_reservation,
        screening_report=screening_report,
    )


def claim_v5_final_run_shard(
    run_root: str | Path,
    promotion_plan: str | Path,
    actor_bundle: str | Path,
    *,
    repository_root: str | Path,
    device: str,
    match_shard_count: int,
    match_shard_index: int,
) -> dict[str, object]:
    """Create the exact immutable final claim with sealed execution provenance."""

    workflow = load_v5_run(run_root)
    root = Path(run_root).resolve()
    provenance = _source_provenance_for_run(
        root, workflow, repository_root, device
    )
    return claim_v5_final_evaluation_shard(
        promotion_plan,
        actor_bundle,
        evaluation_provenance=provenance,
        match_shard_count=match_shard_count,
        match_shard_index=match_shard_index,
    )


def _reserved_registry_path(
    registry: Path, logical_path: object, label: str
) -> Path:
    if not isinstance(logical_path, str) or not logical_path or "\\" in logical_path:
        raise ValueError(f"{label} is not canonical POSIX")
    logical = PurePosixPath(logical_path)
    if (
        logical.is_absolute()
        or logical.as_posix() != logical_path
        or any(part in ("", ".", "..") for part in logical.parts)
    ):
        raise ValueError(f"{label} is not canonical POSIX")
    target = registry.joinpath(*logical.parts).resolve()
    try:
        target.relative_to(registry)
    except ValueError as error:
        raise ValueError(f"{label} escaped its registry") from error
    return target


def _select_exact_output(
    supplied: str | Path | None, expected: Path, label: str
) -> Path:
    if supplied is not None and Path(supplied).resolve() != expected:
        raise ValueError(f"{label} output differs from its canonical reservation")
    return expected


def _verify_exact_evaluation_execution(
    report: Mapping[str, object],
    config: V5EvaluationConfig,
    model: Mapping[str, object],
    *,
    provenance: Mapping[str, object] | None,
    certification_binding: Mapping[str, object] | None,
    final_binding: Mapping[str, object] | None,
    screening_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    verified = validate_v5_evaluation_report(report)
    expected_plan = {
        str(player): matches
        for player, matches in sorted(config.resolved_match_counts.items())
    }
    expected_shard = {
        "count": config.match_shard_count,
        "index": config.match_shard_index,
    }
    if (
        verified.get("mode") != config.mode
        or verified.get("familyId") != config.family_id
        or verified.get("seedBase") != config.seed_base
        or verified.get("matchPlan") != expected_plan
        or verified.get("shard") != expected_shard
        or verified.get("model") != dict(model)
    ):
        raise ValueError("existing V5 evaluation report belongs to another execution")
    expected_provenance = (
        [{"provenance": dict(provenance), "shard": expected_shard}]
        if provenance is not None
        else None
    )
    if verified.get("evaluationProvenance") != expected_provenance:
        raise ValueError("existing V5 evaluation report provenance drifted")
    if screening_binding is not None:
        if verified.get("screeningReservation") != dict(screening_binding):
            raise ValueError("existing screening report reservation drifted")
    elif "screeningReservation" in verified:
        raise ValueError("non-screening execution contains a screening binding")
    if certification_binding is not None:
        if (
            verified.get("certificationReservation")
            != dict(certification_binding)
            or "finalClaims" in verified
        ):
            raise ValueError("existing certification report reservation drifted")
    elif "certificationReservation" in verified:
        raise ValueError("non-certification execution contains a certification binding")
    if final_binding is not None:
        if verified.get("finalClaims") != [dict(final_binding)]:
            raise ValueError("existing final report claim drifted")
    elif "finalClaims" in verified:
        raise ValueError("non-final execution contains final claims")
    expected_keys = {
        (player, match_index)
        for player, matches in config.resolved_match_counts.items()
        for match_index in range(matches)
        if match_index % config.match_shard_count == config.match_shard_index
    }
    clusters = verified.get("matchClusters")
    if not isinstance(clusters, list):
        raise ValueError("existing V5 evaluation report omitted match clusters")
    actual_keys = {
        (int(cluster["playerCount"]), int(cluster["matchIndex"]))
        for cluster in clusters
        if isinstance(cluster, Mapping)
    }
    if len(actual_keys) != len(clusters) or actual_keys != expected_keys:
        raise ValueError("existing V5 evaluation report is not the exact completed shard")
    results = verified.get("results")
    if (
        not isinstance(results, list)
        or any(
            not isinstance(result, Mapping)
            or not isinstance(result.get("matchClustered95"), Mapping)
            or result["matchClustered95"].get("resamples")  # type: ignore[index]
            != config.bootstrap_resamples
            for result in results
        )
    ):
        raise ValueError("existing V5 evaluation report bootstrap plan drifted")
    return verified


def _execution_marker_path(output: Path, attempt: int = 1) -> Path:
    if type(attempt) is not int or attempt < 1:
        raise ValueError("V5 evaluation attempt must be a positive integer")
    return output.with_name(
        output.name + f".execution-attempt-{attempt:03d}.json"
    )


def _execution_recovery_directory(output: Path) -> Path:
    return output.parent / ".v5-evaluation-recoveries" / output.name


def _evaluation_execution_identity(
    output: Path,
    *,
    stage: str,
    device: str,
    config: V5EvaluationConfig,
    model: Mapping[str, object],
    provenance: Mapping[str, object] | None,
    binding: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "binding": dict(binding) if binding is not None else None,
        "config": {
            "bootstrapResamples": config.bootstrap_resamples,
            "familyId": config.family_id,
            "laneCount": config.lane_count,
            "matchPlan": {
                str(player): matches
                for player, matches in sorted(config.resolved_match_counts.items())
            },
            "mode": config.mode,
            "seedBase": config.seed_base,
            "shard": {
                "count": config.match_shard_count,
                "index": config.match_shard_index,
            },
        },
        "device": str(torch.device(device)),
        "model": dict(model),
        "outputPath": str(output.resolve()),
        "provenance": dict(provenance) if provenance is not None else None,
        "stage": stage,
    }


def _execution_identity_sha256(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(
        b"DALMUTI-V5-EVALUATION-EXECUTION\0"
        + canonical_json_bytes(dict(identity))
    ).hexdigest()


def _current_execution_process_identity() -> dict[str, object]:
    from v5_collect_cli import _host_boot_id, _process_start_ticks

    start = _process_start_ticks(os.getpid())
    if start is None:
        raise RuntimeError("could not obtain evaluation process-start identity")
    body: dict[str, object] = {
        "bootId": _host_boot_id(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "processStartTicks": start,
    }
    return {
        **body,
        "identitySha256": hashlib.sha256(
            b"DALMUTI-V5-EVALUATION-PROCESS\0" + canonical_json_bytes(body)
        ).hexdigest(),
    }


def _execution_marker_document(
    identity: Mapping[str, object], attempt: int
) -> dict[str, object]:
    return {
        "attempt": attempt,
        "executionIdentity": dict(identity),
        "executionIdentitySha256": _execution_identity_sha256(identity),
        "format": "dalmuti-v5-evaluation-execution-attempt",
        "process": _current_execution_process_identity(),
        "startedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "version": 1,
    }


def _load_execution_marker(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    marker = _strict_json(raw, "V5 evaluation execution marker")
    digest = hashlib.sha256(raw).hexdigest()
    _ensure_checksum_sidecar(path, digest)
    process = marker.get("process")
    identity = marker.get("executionIdentity")
    if (
        set(marker) != {
            "attempt", "executionIdentity", "executionIdentitySha256", "format",
            "process", "startedAt", "version",
        }
        or marker.get("format") != "dalmuti-v5-evaluation-execution-attempt"
        or marker.get("version") != 1
        or type(marker.get("attempt")) is not int
        or int(marker["attempt"]) < 1
        or not isinstance(identity, Mapping)
        or marker.get("executionIdentitySha256")
        != _execution_identity_sha256(identity)
        or not isinstance(marker.get("startedAt"), str)
        or not isinstance(process, Mapping)
        or set(process) != {
            "bootId", "hostname", "identitySha256", "pid", "processStartTicks"
        }
        or not isinstance(process.get("bootId"), str)
        or not isinstance(process.get("hostname"), str)
        or type(process.get("pid")) is not int
        or int(process["pid"]) < 1
        or type(process.get("processStartTicks")) is not int
        or int(process["processStartTicks"]) < 0
    ):
        raise ValueError("V5 evaluation execution marker contract drifted")
    process_body = {
        key: process[key]
        for key in ("bootId", "hostname", "pid", "processStartTicks")
    }
    if process.get("identitySha256") != hashlib.sha256(
        b"DALMUTI-V5-EVALUATION-PROCESS\0"
        + canonical_json_bytes(process_body)
    ).hexdigest():
        raise ValueError("V5 evaluation process identity checksum drifted")
    output_value = identity.get("outputPath")
    if not isinstance(output_value, str) or path.name != _execution_marker_path(
        Path(output_value), int(marker["attempt"])
    ).name:
        raise ValueError("V5 evaluation attempt marker filename drifted")
    return marker, digest


def _execution_markers(
    output: Path, identity: Mapping[str, object] | None = None
) -> list[tuple[Path, dict[str, object], str]]:
    prefix = output.name + ".execution-attempt-"
    values: list[tuple[int, Path, dict[str, object], str]] = []
    for path in output.parent.glob(prefix + "*.json"):
        suffix = path.name[len(prefix) : -len(".json")]
        if len(suffix) != 3 or not suffix.isdecimal():
            raise ValueError("V5 evaluation directory contains a malformed attempt marker")
        marker, digest = _load_execution_marker(path)
        attempt = int(suffix)
        if marker.get("attempt") != attempt:
            raise ValueError("V5 evaluation attempt filename/document disagree")
        if identity is not None and marker.get("executionIdentity") != dict(identity):
            raise ValueError(
                "existing V5 evaluation attempt belongs to another "
                "Actor/provenance/configuration"
            )
        values.append((attempt, path, marker, digest))
    values.sort(key=lambda item: item[0])
    if values and [item[0] for item in values] != list(
        range(1, values[-1][0] + 1)
    ):
        raise ValueError("V5 evaluation attempt chain has a gap")
    return [(path, marker, digest) for _, path, marker, digest in values]


def _claim_evaluation_execution_once(
    output: Path,
    *,
    stage: str,
    device: str,
    config: V5EvaluationConfig,
    model: Mapping[str, object],
    provenance: Mapping[str, object] | None,
    binding: Mapping[str, object] | None,
) -> Path:
    identity = _evaluation_execution_identity(
        output,
        stage=stage,
        device=device,
        config=config,
        model=model,
        provenance=provenance,
        binding=binding,
    )
    if _execution_markers(output, identity):
        raise RuntimeError(
            "V5 evaluation already started without a canonical result; "
            "an explicit crash-recovery receipt is required"
        )
    marker = _execution_marker_path(output, 1)
    _atomic_canonical_with_sidecar(
        marker, _execution_marker_document(identity, 1)
    )
    return marker


def _validate_recovery_reason(reason: str | None) -> str:
    if (
        not isinstance(reason, str)
        or reason.strip() != reason
        or not reason
        or len(reason) > 240
        or any(ord(character) < 32 for character in reason)
    ):
        raise ValueError("evaluation recovery reason must be 1..240 printable characters")
    return reason


def _prove_execution_process_inactive(
    marker: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    from v5_collect_cli import (
        _host_boot_id,
        _process_may_exist,
        _process_start_ticks,
    )

    process = marker["process"]
    assert isinstance(process, Mapping)
    observer = _current_execution_process_identity()
    if process["hostname"] != observer["hostname"]:
        raise RuntimeError(
            "evaluation attempt belongs to another host; inactivity cannot be "
            "proven locally"
        )
    same_boot = process["bootId"] == _host_boot_id()
    observed_start = (
        _process_start_ticks(int(process["pid"])) if same_boot else None
    )
    if same_boot and observed_start == process["processStartTicks"]:
        raise RuntimeError("evaluation attempt process is still active")
    if (
        same_boot
        and observed_start is None
        and _process_may_exist(int(process["pid"]))
    ):
        raise RuntimeError(
            "evaluation PID may still be active but its process-start identity "
            "cannot be read"
        )
    evidence = (
        "process-missing"
        if same_boot and observed_start is None
        else "pid-reused"
        if same_boot
        else "host-rebooted"
    )
    return evidence, observer


def _retire_evaluation_artifact(
    source: Path,
    recovery_root: Path,
    *,
    label: str,
) -> dict[str, object]:
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    retired = recovery_root / "retired" / f"{label}-{digest}.bin"
    retired.parent.mkdir(parents=True, exist_ok=True)
    if retired.exists():
        if retired.read_bytes() != raw:
            raise ValueError("evaluation recovery retired artifact checksum drifted")
    else:
        try:
            os.link(source, retired)
        except FileExistsError:
            if retired.read_bytes() != raw:
                raise ValueError("evaluation recovery retired artifact collision")
        _fsync_directory(retired.parent)
    return {
        "bytes": len(raw),
        "retiredPath": str(retired.relative_to(source.parent)).replace("\\", "/"),
        "sha256": digest,
        "sourceName": source.name,
    }


def _load_evaluation_recovery_receipt(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    receipt = _strict_json(raw, "V5 evaluation recovery receipt")
    digest = hashlib.sha256(raw).hexdigest()
    _ensure_checksum_sidecar(path, digest)
    if (
        not isinstance(receipt.get("artifacts"), list)
        or receipt.get("format") != "dalmuti-v5-evaluation-crash-recovery"
        or receipt.get("version") != 1
        or set(receipt) != {
            "action", "artifacts", "executionIdentitySha256", "format",
            "fromAttempt", "markerSha256", "observedCrashEvidence", "observer",
            "reason", "recoveredAt", "recoveryId", "restoreArtifactSha256",
            "toAttempt", "version",
        }
    ):
        raise ValueError("V5 evaluation recovery receipt contract drifted")
    body = {key: receipt[key] for key in receipt if key != "recoveryId"}
    expected_id = hashlib.sha256(
        b"DALMUTI-V5-EVALUATION-RECOVERY\0" + canonical_json_bytes(body)
    ).hexdigest()
    if receipt.get("recoveryId") != expected_id:
        raise ValueError("V5 evaluation recovery receipt identity drifted")
    return receipt


def _finish_evaluation_recovery(
    output: Path,
    receipt: Mapping[str, object],
    identity: Mapping[str, object],
) -> bool:
    artifacts = receipt["artifacts"]
    assert isinstance(artifacts, list)
    recovery_root = _execution_recovery_directory(output)
    for record in artifacts:
        if not isinstance(record, Mapping) or set(record) != {
            "bytes", "retiredPath", "sha256", "sourceName"
        }:
            raise ValueError("V5 evaluation recovery artifact record drifted")
        source_name = record["sourceName"]
        retired_name = record["retiredPath"]
        if (
            not isinstance(source_name, str)
            or Path(source_name).name != source_name
            or not isinstance(retired_name, str)
        ):
            raise ValueError("V5 evaluation recovery artifact path drifted")
        retired = (output.parent / retired_name).resolve()
        try:
            retired.relative_to(recovery_root.resolve())
        except ValueError as error:
            raise ValueError("retired evaluation artifact escaped recovery root") from error
        raw = retired.read_bytes()
        if (
            len(raw) != record["bytes"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("retired evaluation artifact no longer matches receipt")

    action = receipt["action"]
    restore_sha = receipt["restoreArtifactSha256"]
    # A crash after the receipt publication but before source retirement is
    # resumed by deleting only bytes whose digest is already preserved in the
    # immutable retired inventory.
    for record in artifacts:
        assert isinstance(record, Mapping)
        source = output.parent / str(record["sourceName"])
        if source.exists():
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("evaluation crash artifact changed before retirement")
            source.unlink()
    _fsync_directory(output.parent)

    if action == "restore-output":
        matches = [
            record for record in artifacts
            if isinstance(record, Mapping) and record.get("sha256") == restore_sha
        ]
        if not matches:
            raise ValueError("evaluation recovery restore artifact is missing")
        selected = matches[0]
        retired = (output.parent / str(selected["retiredPath"])).resolve()
        if output.exists():
            if sha256_file(output) != restore_sha:
                raise ValueError("canonical evaluation output conflicts with recovery")
        else:
            os.link(retired, output)
            _fsync_directory(output.parent)
        digest = sha256_file(output)
        _ensure_checksum_sidecar(output, digest)
        restored = True
    elif action == "retry-deterministic-execution":
        if output.exists():
            raise ValueError("corrupt evaluation output was not retired before retry")
        restored = False
    else:
        raise ValueError("V5 evaluation recovery action drifted")

    if not restored:
        attempt = int(receipt["toAttempt"])
        marker_path = _execution_marker_path(output, attempt)
        expected_document_identity = dict(identity)
        if marker_path.exists():
            marker, _ = _load_execution_marker(marker_path)
            if marker.get("executionIdentity") != expected_document_identity:
                raise ValueError("recovered evaluation retry marker identity drifted")
        else:
            _atomic_canonical_with_sidecar(
                marker_path,
                _execution_marker_document(expected_document_identity, attempt),
            )
    return restored


def _recover_evaluation_execution(
    output: Path,
    *,
    reason: str,
    identity: Mapping[str, object],
    config: V5EvaluationConfig,
    model: Mapping[str, object],
    provenance: Mapping[str, object] | None,
    screening_binding: Mapping[str, object] | None,
    certification_binding: Mapping[str, object] | None,
    final_binding: Mapping[str, object] | None,
) -> bool:
    reason = _validate_recovery_reason(reason)
    markers = _execution_markers(output, identity)
    if not markers:
        raise RuntimeError("evaluation recovery requires a prior execution marker")
    _, marker, marker_sha = markers[-1]
    attempt = int(marker["attempt"])
    recovery_root = _execution_recovery_directory(output)
    receipt_path = recovery_root / f"attempt-{attempt:03d}-{marker_sha}.json"
    if receipt_path.exists():
        receipt = _load_evaluation_recovery_receipt(receipt_path)
        if (
            receipt.get("executionIdentitySha256")
            != _execution_identity_sha256(identity)
            or receipt.get("markerSha256") != marker_sha
            or receipt.get("fromAttempt") != attempt
            or receipt.get("reason") != reason
        ):
            raise ValueError("existing evaluation recovery receipt belongs elsewhere")
        return _finish_evaluation_recovery(output, receipt, identity)

    crash_evidence, observer = _prove_execution_process_inactive(marker)
    artifacts: list[dict[str, object]] = []
    target_is_foreign = False
    if output.exists():
        try:
            candidate = _load_evaluation_report(output)
        except (OSError, ValueError):
            candidate = None
        if candidate is not None:
            try:
                _verify_exact_evaluation_execution(
                    candidate,
                    config,
                    model,
                    provenance=provenance,
                    certification_binding=certification_binding,
                    final_binding=final_binding,
                    screening_binding=screening_binding,
                )
            except ValueError:
                target_is_foreign = True
        if target_is_foreign:
            raise ValueError(
                "canonical evaluation target contains a valid foreign execution; "
                "it cannot be retired or overwritten"
            )

    candidates: list[tuple[Path, str]] = []
    for temporary in sorted(output.parent.glob(f".{output.name}.*.tmp")):
        try:
            report = _load_evaluation_report(temporary)
            _verify_exact_evaluation_execution(
                report,
                config,
                model,
                provenance=provenance,
                certification_binding=certification_binding,
                final_binding=final_binding,
                screening_binding=screening_binding,
            )
        except (OSError, ValueError):
            continue
        candidates.append((temporary, sha256_file(temporary)))
    candidate_hashes = {digest for _, digest in candidates}
    if len(candidate_hashes) > 1:
        raise RuntimeError(
            "evaluation crash left conflicting valid outputs; deterministic "
            "replay cannot be established"
        )
    action = (
        "restore-output"
        if not output.exists() and len(candidate_hashes) == 1
        else "retry-deterministic-execution"
    )
    sources: list[Path] = []
    if output.exists():
        sources.append(output)
    output_sidecar = output.with_name(output.name + ".sha256")
    if output_sidecar.exists():
        sources.append(output_sidecar)
    sources.extend(sorted(output.parent.glob(f".{output.name}.*.tmp")))
    recovery_root.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources):
        artifacts.append(
            _retire_evaluation_artifact(
                source,
                recovery_root,
                label=f"attempt-{attempt:03d}-{index:03d}",
            )
        )
    restore_sha = next(iter(candidate_hashes), None) if action == "restore-output" else None
    body: dict[str, object] = {
        "action": action,
        "artifacts": artifacts,
        "executionIdentitySha256": _execution_identity_sha256(identity),
        "format": "dalmuti-v5-evaluation-crash-recovery",
        "fromAttempt": attempt,
        "markerSha256": marker_sha,
        "observedCrashEvidence": crash_evidence,
        "observer": observer,
        "reason": reason,
        "recoveredAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "restoreArtifactSha256": restore_sha,
        "toAttempt": attempt if action == "restore-output" else attempt + 1,
        "version": 1,
    }
    receipt = {
        **body,
        "recoveryId": hashlib.sha256(
            b"DALMUTI-V5-EVALUATION-RECOVERY\0" + canonical_json_bytes(body)
        ).hexdigest(),
    }
    _atomic_canonical_with_sidecar(receipt_path, receipt)

    # Only after durable evidence exists may the canonical corrupt bytes be
    # removed.  Exact-valid orphan bytes are linked back by the finisher.
    for record in artifacts:
        source = output.parent / str(record["sourceName"])
        if source.exists():
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("evaluation crash artifact changed during recovery")
            source.unlink()
    _fsync_directory(output.parent)
    return _finish_evaluation_recovery(output, receipt, identity)


def evaluate_v5_run_stage(
    run_root: str | Path,
    actor_bundle: str | Path,
    output: str | Path | None,
    *,
    stage: str,
    device: str,
    lane_count: int = 32,
    match_shard_count: int = 1,
    match_shard_index: int = 0,
    repository_root: str | Path | None = None,
    screening_report: str | Path | None = None,
    screening_reservation: str | Path | None = None,
    certification_reservation: str | Path | None = None,
    promotion_plan: str | Path | None = None,
    final_claim: str | Path | None = None,
    recovery_reason: str | None = None,
) -> dict[str, object]:
    if lane_count != 32:
        raise ValueError("V5 production evaluation requires exactly 32 lanes")
    workflow = load_v5_run(run_root)
    root = Path(run_root).resolve()
    bundle = Path(actor_bundle).resolve()
    actor, _ = load_v5_actor_bundle(bundle)
    model = v5_actor_bundle_digests(bundle)
    provenance: dict[str, object] | None = None
    screening_binding: dict[str, object] | None = None
    certification_binding: dict[str, object] | None = None
    final_binding: dict[str, object] | None = None
    if stage == "screening":
        if repository_root is None or screening_reservation is None:
            raise ValueError(
                "production screening requires repository root and its global reservation"
            )
        if any(
            value is not None
            for value in (
                screening_report,
                certification_reservation,
                promotion_plan,
                final_claim,
            )
        ):
            raise ValueError("screening does not accept certification/final inputs")
        reservation_path = Path(screening_reservation).resolve()
        reservation = load_v5_screening_execution_reservation(reservation_path)
        if reservation.get("model") != model:
            raise ValueError("screening reservation belongs to a different Actor")
        coordinate = reservation["coordinate"]
        assert isinstance(coordinate, Mapping)
        mode = "screening"
        family = str(coordinate["familyId"])
        seed = int(coordinate["seedBase"])
        counts = {
            int(key): int(value)
            for key, value in coordinate["matchPlan"].items()  # type: ignore[union-attr]
        }
        provenance = _source_provenance_for_run(
            root, workflow, repository_root, device
        )
        registry = reservation_path.parent.parent
        target = _select_exact_output(
            output,
            _reserved_registry_path(
                registry, reservation["outputPath"], "screening result path"
            ),
            "screening",
        )
    elif stage in {"certification-a", "certification-b"}:
        if (
            repository_root is None
            or screening_report is None
            or certification_reservation is None
        ):
            raise ValueError(
                "certification requires repository root, passed screening, and reservation"
            )
        if (
            promotion_plan is not None
            or final_claim is not None
            or screening_reservation is not None
        ):
            raise ValueError("certification cannot consume final-holdout inputs")
        reservation_path = Path(certification_reservation).resolve()
        reservation = load_v5_certification_execution_reservation(reservation_path)
        coordinates = reservation["coordinates"]
        assert isinstance(coordinates, list)
        expected_label = "a" if stage == "certification-a" else "b"
        matches = [
            value
            for value in coordinates
            if isinstance(value, Mapping) and value.get("label") == expected_label
        ]
        if len(matches) != 1:
            raise ValueError("certification reservation omitted its exact coordinate")
        coordinate = matches[0]
        mode = "certification"
        family = str(coordinate["familyId"])
        seed = int(coordinate["seedBase"])
        counts = {int(key): int(value) for key, value in coordinate["matchPlan"].items()}  # type: ignore[union-attr]
        provenance = _source_provenance_for_run(root, workflow, repository_root, device)
        registry = reservation_path.parent.parent
        screening_record = reservation["screening"]
        assert isinstance(screening_record, Mapping)
        screening_execution_path = (
            registry
            / "screening-reservations"
            / f"{screening_record['reservationId']}.json"
        )
        if (
            Path(screening_report).resolve()
            != registry
            / "screening-results"
            / str(screening_record["reservationId"])
            / "report.json"
            or sha256_file(screening_report) != screening_record["reportSha256"]
        ):
            raise ValueError("certification received another screening report")
        _passed_screening_for_actor(
            screening_report,
            model,
            screening_execution_path,
            provenance,
        )
        target = _select_exact_output(
            output,
            _reserved_registry_path(
                registry, coordinate["outputPath"], "certification result path"
            ),
            "certification",
        )
    elif stage == "final":
        if repository_root is None or promotion_plan is None or final_claim is None:
            raise ValueError("final evaluation requires repository root, plan, and one-shot claim")
        if (
            screening_report is not None
            or screening_reservation is not None
            or certification_reservation is not None
        ):
            raise ValueError("final holdout must not accept a tuning/screening report")
        plan = load_v5_promotion_plan(promotion_plan)
        if plan.get("model") != model:
            raise ValueError("final promotion plan belongs to a different Actor")
        final = plan["final"]
        assert isinstance(final, Mapping)
        mode = "final"
        family = str(final["familyId"])
        seed = int(final["seedBase"])
        counts = {int(key): int(value) for key, value in final["matchPlan"].items()}  # type: ignore[union-attr]
        if counts != FINAL_MATCH_COUNTS:
            raise ValueError("final promotion plan match counts drifted")
        provenance = _source_provenance_for_run(root, workflow, repository_root, device)
        claim_path = Path(final_claim).resolve()
        claim = load_v5_final_evaluation_claim(claim_path, promotion_plan)
        registry = claim_path.parent.parent.parent
        target = _select_exact_output(
            output,
            _reserved_registry_path(
                registry, claim["outputPath"], "final result path"
            ),
            "final",
        )
    else:
        raise ValueError("stage must be screening, certification-a/b, or final")
    config = V5EvaluationConfig(
        mode=mode,
        family_id=family,
        seed_base=seed,
        match_counts=tuple(sorted(counts.items())),
        match_shard_count=match_shard_count,
        match_shard_index=match_shard_index,
        lane_count=lane_count,
        bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    if mode == "screening":
        screening_binding = authorize_v5_screening_evaluation(
            screening_reservation,  # type: ignore[arg-type]
            model,
            evaluation_provenance=provenance,  # type: ignore[arg-type]
            family_id=family,
            seed_base=seed,
            match_plan=counts,
            match_shard_count=match_shard_count,
            match_shard_index=match_shard_index,
            bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            output_path=target,
        )
    elif mode == "certification":
        certification_binding = authorize_v5_certification_evaluation(
            certification_reservation,  # type: ignore[arg-type]
            model,
            evaluation_provenance=provenance,  # type: ignore[arg-type]
            family_id=family,
            seed_base=seed,
            match_plan=counts,
            match_shard_count=match_shard_count,
            match_shard_index=match_shard_index,
            bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            output_path=target,
        )
    elif mode == "final":
        final_binding = authorize_v5_final_evaluation(
            promotion_plan,  # type: ignore[arg-type]
            final_claim,  # type: ignore[arg-type]
            model,
            evaluation_provenance=provenance,  # type: ignore[arg-type]
            family_id=family,
            seed_base=seed,
            match_plan=counts,
            match_shard_count=match_shard_count,
            match_shard_index=match_shard_index,
            bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            output_path=target,
        )
    binding = (
        screening_binding
        if mode == "screening"
        else certification_binding
        if mode == "certification"
        else final_binding
    )
    identity = _evaluation_execution_identity(
        target,
        stage=stage,
        device=device,
        config=config,
        model=model,
        provenance=provenance,
        binding=binding,
    )

    def completed_result() -> dict[str, object]:
        report = _load_workflow_evaluation_report(target)
        verified = _verify_exact_evaluation_execution(
            report,
            config,
            model,
            provenance=provenance,
            certification_binding=certification_binding,
            final_binding=final_binding,
            screening_binding=screening_binding,
        )
        if not _execution_markers(target, identity):
            raise ValueError(
                "workflow evaluation report lacks its exact execution-attempt marker"
            )
        return {
            "allPlayerCountsPassed": verified["allPlayerCountsPassed"],
            "completeEvaluation": verified["completeEvaluation"],
            "familyId": family,
            "output": str(target),
            "reportSha256": sha256_file(target),
            "reusedExistingReport": True,
            "seedBase": seed,
        }

    if target.exists():
        try:
            return completed_result()
        except (OSError, ValueError):
            # A structurally valid report for a foreign execution is rejected
            # by the recovery helper.  Corrupt/partial bytes can be retired
            # only with explicit operator evidence.
            if recovery_reason is None:
                raise
    markers = _execution_markers(target, identity)
    if markers:
        if recovery_reason is None:
            raise RuntimeError(
                "V5 evaluation already started without a canonical result; "
                "an explicit crash-recovery receipt is required"
            )
        restored = _recover_evaluation_execution(
            target,
            reason=recovery_reason,
            identity=identity,
            config=config,
            model=model,
            provenance=provenance,
            screening_binding=screening_binding,
            certification_binding=certification_binding,
            final_binding=final_binding,
        )
        if restored:
            return completed_result()
    else:
        if recovery_reason is not None:
            raise RuntimeError(
                "evaluation recovery was requested but no prior attempt exists"
            )
        _claim_evaluation_execution_once(
            target,
            stage=stage,
            device=device,
            config=config,
            model=model,
            provenance=provenance,
            binding=binding,
        )
    report = evaluate_v5_actor(
        actor,
        config,
        device=device,
        model_identity=model,
        evaluation_provenance=provenance,
        screening_reservation=screening_binding,
        certification_reservation=certification_reservation,
        promotion_plan=promotion_plan,
        final_claim=final_claim,
        output_path=target,
    )
    report = _verify_exact_evaluation_execution(
        report,
        config,
        model,
        provenance=provenance,
        certification_binding=certification_binding,
        final_binding=final_binding,
        screening_binding=screening_binding,
    )
    digest = _write_workflow_evaluation_report(target, report)
    return {
        "allPlayerCountsPassed": report["allPlayerCountsPassed"],
        "completeEvaluation": report["completeEvaluation"],
        "familyId": family,
        "output": str(target),
        "reportSha256": digest,
        "reusedExistingReport": False,
        "seedBase": seed,
    }


def _training_overrides(arguments: argparse.Namespace) -> dict[str, object]:
    names = (
        "microbatch_size",
        "gradient_accumulation",
        "critic_batch_size",
        "audit_batch_size",
        "actor_learning_rate",
        "critic_learning_rate",
        "weight_decay",
        "entropy_coefficient",
        "normal_auxiliary_coefficient",
        "max_gradient_norm",
    )
    return {
        name: getattr(arguments, name)
        for name in names
        if getattr(arguments, name) is not None
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--run-root", required=True)
    bootstrap.add_argument("--repository-root", required=True)
    bootstrap.add_argument("--source-commit", required=True)
    bootstrap.add_argument("--iteration", type=int, required=True)
    bootstrap.add_argument("--run-number", type=int, required=True)

    describe = commands.add_parser("describe")
    describe.add_argument("--run-root", required=True)

    materialize = commands.add_parser("materialize-source")
    materialize.add_argument("--run-root", required=True)
    materialize.add_argument("--output")

    train = commands.add_parser("train")
    train.add_argument("--run-root", required=True)
    train.add_argument("--dataset-index", required=True)
    train.add_argument("--device", required=True)
    train.add_argument("--repository-root", required=True)
    train.add_argument("--gpu-memory-preflight", required=True)
    train.add_argument("--initial-model-pair")
    train.add_argument("--low-disk-persistent-root")
    train.add_argument("--low-disk-volatile-root")
    train.add_argument("--low-disk-promotion-receipt-root")
    for option, converter in (
        ("microbatch-size", int),
        ("gradient-accumulation", int),
        ("critic-batch-size", int),
        ("audit-batch-size", int),
        ("actor-learning-rate", float),
        ("critic-learning-rate", float),
        ("weight-decay", float),
        ("entropy-coefficient", float),
        ("normal-auxiliary-coefficient", float),
        ("max-gradient-norm", float),
    ):
        train.add_argument(f"--{option}", type=converter)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-root", required=True)
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument(
        "--output",
        help="required for screening; certification/final must equal the reserved path",
    )
    evaluate.add_argument(
        "--stage",
        choices=("screening", "certification-a", "certification-b", "final"),
        required=True,
    )
    evaluate.add_argument("--device", required=True)
    evaluate.add_argument("--lanes", type=int, default=32)
    evaluate.add_argument("--match-shard-count", type=int, default=1)
    evaluate.add_argument("--match-shard-index", type=int, default=0)
    evaluate.add_argument("--repository-root")
    evaluate.add_argument("--screening-report")
    evaluate.add_argument("--screening-reservation")
    evaluate.add_argument("--certification-reservation")
    evaluate.add_argument("--promotion-plan")
    evaluate.add_argument("--final-claim")
    evaluate.add_argument(
        "--recover-crashed-attempt-reason",
        help="explicit evidence reason for retrying the same inactive deterministic execution",
    )

    merge = commands.add_parser("merge-evaluations")
    merge.add_argument("--output", required=True)
    merge.add_argument("reports", nargs="+")

    coordinates = commands.add_parser("certification-coordinates")
    coordinates.add_argument("--bundle", required=True)

    reserve_screening = commands.add_parser("reserve-screening")
    reserve_screening.add_argument("--run-root", required=True)
    reserve_screening.add_argument("--registry", required=True)
    reserve_screening.add_argument("--bundle", required=True)
    reserve_screening.add_argument("--repository-root", required=True)
    reserve_screening.add_argument("--device", required=True)

    reserve_certification = commands.add_parser("reserve-certification")
    reserve_certification.add_argument("--run-root", required=True)
    reserve_certification.add_argument("--registry", required=True)
    reserve_certification.add_argument("--bundle", required=True)
    reserve_certification.add_argument("--screening-report", required=True)
    reserve_certification.add_argument("--screening-reservation", required=True)
    reserve_certification.add_argument("--repository-root", required=True)
    reserve_certification.add_argument("--device", required=True)

    reserve = commands.add_parser("reserve-final")
    reserve.add_argument("--registry", required=True)
    reserve.add_argument("--bundle", required=True)
    reserve.add_argument("--certification-report", action="append", required=True)
    reserve.add_argument("--final-shards", type=int, required=True)

    claim = commands.add_parser("claim-final")
    claim.add_argument("--run-root", required=True)
    claim.add_argument("--plan", required=True)
    claim.add_argument("--bundle", required=True)
    claim.add_argument("--repository-root", required=True)
    claim.add_argument("--device", required=True)
    claim.add_argument("--match-shard-count", type=int, required=True)
    claim.add_argument("--match-shard-index", type=int, required=True)

    recover_promotion = commands.add_parser("recover-promotion-lock")
    recover_promotion.add_argument("--registry", required=True)
    recover_promotion.add_argument("--reason", required=True)

    approve = commands.add_parser("approve-final")
    approve.add_argument("--plan", required=True)
    approve.add_argument("--bundle", required=True)
    approve.add_argument("--final-report", required=True)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    if arguments.command == "bootstrap":
        result = bootstrap_v5_run(
            arguments.run_root,
            repository_root=arguments.repository_root,
            source_commit=arguments.source_commit,
            iteration=arguments.iteration,
            run_number=arguments.run_number,
        )
    elif arguments.command == "describe":
        result = load_v5_run(arguments.run_root)
    elif arguments.command == "materialize-source":
        result = materialize_v5_source_checkout(
            arguments.run_root, arguments.output
        )
    elif arguments.command == "train":
        result = train_v5_run(
            arguments.run_root,
            arguments.dataset_index,
            device=arguments.device,
            repository_root=arguments.repository_root,
            gpu_memory_preflight=arguments.gpu_memory_preflight,
            initial_model_pair=arguments.initial_model_pair,
            low_disk_persistent_root=arguments.low_disk_persistent_root,
            low_disk_volatile_root=arguments.low_disk_volatile_root,
            low_disk_promotion_receipt_root=(
                arguments.low_disk_promotion_receipt_root
            ),
            config_overrides=_training_overrides(arguments),
        )
    elif arguments.command == "evaluate":
        result = evaluate_v5_run_stage(
            arguments.run_root,
            arguments.bundle,
            arguments.output,
            stage=arguments.stage,
            device=arguments.device,
            lane_count=arguments.lanes,
            match_shard_count=arguments.match_shard_count,
            match_shard_index=arguments.match_shard_index,
            repository_root=arguments.repository_root,
            screening_report=arguments.screening_report,
            screening_reservation=arguments.screening_reservation,
            certification_reservation=arguments.certification_reservation,
            promotion_plan=arguments.promotion_plan,
            final_claim=arguments.final_claim,
            recovery_reason=arguments.recover_crashed_attempt_reason,
        )
    elif arguments.command == "merge-evaluations":
        result = merge_v5_evaluation_report_files(arguments.reports, arguments.output)
    elif arguments.command == "certification-coordinates":
        result = list(v5_certification_coordinates(arguments.bundle))
    elif arguments.command == "reserve-screening":
        result = reserve_v5_screening_run(
            arguments.run_root,
            arguments.registry,
            arguments.bundle,
            repository_root=arguments.repository_root,
            device=arguments.device,
        )
    elif arguments.command == "reserve-certification":
        result = reserve_v5_certification_run(
            arguments.run_root,
            arguments.registry,
            arguments.bundle,
            arguments.screening_report,
            screening_reservation=arguments.screening_reservation,
            repository_root=arguments.repository_root,
            device=arguments.device,
        )
    elif arguments.command == "reserve-final":
        if len(arguments.certification_report) != 2:
            raise ValueError("reserve-final requires exactly two --certification-report values")
        result = reserve_v5_final_holdout(
            arguments.registry,
            arguments.bundle,
            arguments.certification_report,
            final_match_shard_count=arguments.final_shards,
        )
    elif arguments.command == "recover-promotion-lock":
        result = recover_v5_promotion_lock(
            arguments.registry, recovery_reason=arguments.reason
        )
    elif arguments.command == "claim-final":
        result = claim_v5_final_run_shard(
            arguments.run_root,
            arguments.plan,
            arguments.bundle,
            repository_root=arguments.repository_root,
            device=arguments.device,
            match_shard_count=arguments.match_shard_count,
            match_shard_index=arguments.match_shard_index,
        )
    elif arguments.command == "approve-final":
        result = approve_v5_final_holdout(
            arguments.plan, arguments.bundle, arguments.final_report
        )
    else:  # pragma: no cover
        raise AssertionError("unhandled V5 workflow command")
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V5_CALIBRATION_SEED_BASE",
    "V5_COLLECTION_SEED_BASE",
    "V5_INITIALIZATION_SEED_BASE",
    "V5_SCREENING_SEED_BASE",
    "V5_TRAINING_SEED_BASE",
    "V5_PRODUCTION_TRAINING_BATCHES",
    "V5_WORKFLOW_FORMAT",
    "argument_parser",
    "bootstrap_v5_run",
    "evaluate_v5_run_stage",
    "load_v5_run",
    "main",
    "materialize_v5_source_checkout",
    "merge_v5_evaluation_report_files",
    "reserve_v5_certification_run",
    "claim_v5_final_run_shard",
    "train_v5_run",
    "v5_run_directory_name",
    "v5_run_namespace",
    "v5_seed_schedule",
]
