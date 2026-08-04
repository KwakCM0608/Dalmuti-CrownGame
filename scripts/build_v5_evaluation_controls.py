from __future__ import annotations

"""Build fail-closed V5 evaluation controls for a newly sealed run.

The builder is intentionally local-only.  It never opens a network connection and
never starts training or evaluation.  It extracts the audited evaluation operations
from run-004, binds them to the target run's training verifier, and adapts the
run-006 durable launcher/validator templates.  Every identity is derived from and
checked against the already-materialized target workflow/source seal before output
is installed.
"""

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN004_NAMESPACE = "v5-mappo-normalresidual-i001-s840030001-run-004"
RUN006_NAMESPACE = "v5-mappo-normalresidual-i001-s840050001-run-006"
RUN004_CONTROL_TEMPLATE = (
    REPOSITORY_ROOT
    / "artifacts/rl/v5-runs"
    / RUN004_NAMESPACE
    / "controls/run_training_iteration_001.py"
)
RUN006_LAUNCHER_TEMPLATE = (
    REPOSITORY_ROOT
    / "artifacts/rl/v5-runs"
    / RUN006_NAMESPACE
    / "controls/launch_durable_evaluation_pipeline.py"
)
RUN006_VALIDATOR_TEMPLATE = (
    REPOSITORY_ROOT
    / "artifacts/rl/v5-runs"
    / RUN006_NAMESPACE
    / "controls/validate_durable_evaluation_pipeline.py"
)
TEMPLATE_SHA256 = {
    "run004Control": "0f9831682a56e4470ebbc83474a4ba6384436229addd91c487458acc05598846",
    "run006Launcher": "45478649f1a52219a55419668d94ff5c3a3162ea2c250a5251b0bbd0bbb7593c",
    "run006Validator": "e9e66d8c0e188f611009f2a4ac408646891509416ad2c38a18565dfa2f3797a8",
}
EVALUATION_CONTROL_NAME = "run_evaluation_iteration_001.py"
TRAINING_CONTROL_NAME = "run_training_iteration_001.py"
RECOVERY_VERIFIER_NAME = "verify_completed_training_recovery.py"
RECOVERY_RECEIPT_RELATIVE = Path(
    "training-recovery/completed-training-verification.json"
)
LAUNCHER_NAME = "launch_durable_evaluation_pipeline_recovery_r3.py"
VALIDATOR_NAME = "validate_durable_evaluation_pipeline_recovery_r3.py"
RECEIPT_NAME = "evaluation-controls-build-recovery-r3.json"
EVALUATION_ATTEMPT_RELATIVE = "controls/durable-evaluation-attempts-recovery-r3"
MINIMUM_EVALUATION_FREE_BYTES = 6 * 1024**3
EXPECTED_RUNTIME_PYTHON_PATH_COUNT = 143
EXPECTED_CHECKED_SOURCE_PATH_COUNT = 154
EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256 = (
    "fdf1aebbdf29606997f4b4ff4965a73f3240c2b2a4bff445cfdc48698442117f"
)
EXPECTED_SOURCE_UNION_INVENTORY_SHA256 = (
    "7208de2a882c4941675dd8a5e9036f44095d6b5857bb7dd18889efcf5054d2ab"
)
FAILURE_ANALYSIS_REFERENCE = (
    REPOSITORY_ROOT
    / "artifacts/rl/v5-runs"
    / "bootstrap-v5-mappo-normalresidual-i001-s840060001-run-007"
    / "failure-analysis-r1"
)
FAILURE_REFERENCE_SHA256 = {
    "gpu-memory.json": "a35236b12a519a4521b61ff2bed04f5402a9d5e6e4d3be299ac19be78467e978",
    "manifest.json": "39afab56b4fcd2284ceacbef9e23787f81613d0d60ca943bdcb07301116bef0f",
    "model-pair.json": "f2929b7ea9f7fc0b113a2b25a6e0a84692ffc3ff9e61747ee7a3d119d77cac49",
    "result.json": "bb13b852e60bd82b339b7665e769778e9f463513860aeb9507d4e8dfba67290d",
    "stage-stderr.log": "89fa51ef76c3a025bd7d1a0bf5aa501cc4365f929adb33b383f64ccb358ff9ce",
    "terminal.json": "3e7c0217c66a7414e4bf6317b67a51d71b74caea46c3a06b3d66e1588cc42f99",
    "training-attempt.json": "27e07ad6cb0b8ee405390ef425da3cf8aa6e82daf20d578f06440a69274fa00c",
}
EPOCH_ONLY_PROOF_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/rl/v5-runs"
    / "bootstrap-v5-mappo-normalresidual-i001-s840060001-run-007"
    / "failure-analysis"
)
EPOCH_ONLY_PROOF_SHA256 = (
    "44bf79f326439ed90cfb92069109a8a0bfe92946bbdb71ea0257436513053509"
)
EPOCH_ONLY_PROBE_SHA256 = (
    "15d3ecf05bf8e537825867399ee068659fcf48d4fb73cc0d96604ebd8f3a56c2"
)
EXPECTED_DURABLE_TRAIN_INTENT_SHA256 = (
    "e1c62e44d6cbcf2c99ecc0d11a0f47a25d527fb2fcfaee00a428ac76565e40fe"
)
EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256 = (
    "405f0715c58808c9dea79df5cbc2c4f60d8023466487bb57b7e1a834e2100765"
)
EXPECTED_DURABLE_TRAIN_PROCESS_SHA256 = (
    "a317c480df9acf6f2c7ff7e10d3675128c7f3add205fca55f091a434b1dfb959"
)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
NAMESPACE_RE = re.compile(r"^v5-[a-z0-9-]+-run-\d{3}$")
EXTRACTED_FUNCTIONS = (
    "_require_registry_path",
    "reserve_screening",
    "evaluate_screening",
    "reserve_certification",
    "evaluate_certification",
    "reserve_final",
    "claim_final",
    "evaluate_final",
    "merge_final_reports",
    "approve_final",
)
EVALUATION_STAGES = (
    "reserve-screening",
    "screening",
    "reserve-certification",
    "certification-a",
    "certification-b",
    "reserve-final",
    "claim-final",
    "final",
    "merge-final",
    "approve-final",
)


def _named_assignment(tree: ast.Module, name: str) -> ast.expr:
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise ValueError(f"sealed evaluation source assignment drifted: {name}")
    return matches[0]


def _evaluation_contract(run_root: Path) -> dict[str, object]:
    """Read the fixed match/gate contract without importing Torch-backed code."""

    source = run_root / "source-checkout/gpu-training/v5_evaluate.py"
    promotion = run_root / "source-checkout/gpu-training/v5_promotion.py"
    source_tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    promotion_tree = ast.parse(
        promotion.read_text(encoding="utf-8"), filename=str(promotion)
    )

    players = _named_assignment(source_tree, "PLAYER_COUNTS")
    if (
        not isinstance(players, ast.Call)
        or not isinstance(players.func, ast.Name)
        or players.func.id != "tuple"
        or len(players.args) != 1
        or not isinstance(players.args[0], ast.Call)
        or not isinstance(players.args[0].func, ast.Name)
        or players.args[0].func.id != "range"
        or [ast.literal_eval(value) for value in players.args[0].args] != [4, 11]
    ):
        raise ValueError("sealed evaluation player-count contract drifted")
    player_counts = list(range(4, 11))

    screening = _named_assignment(source_tree, "SCREENING_MATCH_COUNTS")
    if (
        not isinstance(screening, ast.DictComp)
        or len(screening.generators) != 1
        or not isinstance(screening.key, ast.Name)
        or screening.key.id != "player_count"
        or ast.literal_eval(screening.value) != 60
        or not isinstance(screening.generators[0].target, ast.Name)
        or screening.generators[0].target.id != "player_count"
        or not isinstance(screening.generators[0].iter, ast.Name)
        or screening.generators[0].iter.id != "PLAYER_COUNTS"
        or screening.generators[0].ifs
    ):
        raise ValueError("sealed screening match-plan contract drifted")
    screening_plan = {player: 60 for player in player_counts}

    final_plan = ast.literal_eval(_named_assignment(source_tree, "FINAL_MATCH_COUNTS"))
    exact_gates = ast.literal_eval(_named_assignment(source_tree, "EXACT_GATES"))
    bootstrap_resamples = ast.literal_eval(
        _named_assignment(source_tree, "DEFAULT_BOOTSTRAP_RESAMPLES")
    )
    development_gates = ast.literal_eval(
        _named_assignment(promotion_tree, "V5_DEVELOPMENT_GATES")
    )
    certification_count = ast.literal_eval(
        _named_assignment(promotion_tree, "V5_CERTIFICATION_REPORT_COUNT")
    )
    expected_final = {4: 2500, 5: 1700, 6: 900, 7: 600, 8: 400, 9: 400, 10: 300}
    if (
        final_plan != expected_final
        or exact_gates
        != {
            "minMeanChipDifference": 0.25,
            "minCluster95LowerBound": 0.15,
            "minPairwiseRate": 0.55,
        }
        or development_gates
        != {
            "minMeanChipDifference": 0.30,
            "minCluster95LowerBound": 0.20,
            "minPairwiseRate": 0.57,
        }
        or bootstrap_resamples != 10_000
        or certification_count != 2
    ):
        raise ValueError("sealed evaluation gates or promotion schedule drifted")
    totals = {
        "screening": sum(screening_plan.values()),
        "certification-a": sum(screening_plan.values()),
        "certification-b": sum(screening_plan.values()),
    }
    if list(totals.values()) != [420, 420, 420]:
        raise ValueError("screening/certification totals are not exactly 420/420/420")
    return {
        "bootstrapResamples": bootstrap_resamples,
        "certificationReportCount": certification_count,
        "developmentGates": development_gates,
        "exactGates": exact_gates,
        "finalMatchPlan": {str(key): value for key, value in final_plan.items()},
        "finalTotalMatches": sum(final_plan.values()),
        "playerCounts": player_counts,
        "screeningMatchPlan": {
            str(key): value for key, value in screening_plan.items()
        },
        "stageMatchTotals": totals,
    }


def _load_failure_reference() -> dict[str, object]:
    values: dict[str, object] = {}
    for name, expected_sha in FAILURE_REFERENCE_SHA256.items():
        path = FAILURE_ANALYSIS_REFERENCE / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"run-007 failure reference is absent: {name}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"run-007 failure reference SHA-256 drifted: {name}"
            )
        raw = path.read_bytes()
        if name.endswith(".json"):
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"failure reference is not UTF-8 JSON: {name}") from error
            if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
                raise ValueError(f"failure reference is not canonical JSON: {name}")
            values[name] = value
        else:
            values[name] = raw
    manifest = values["manifest.json"]
    result = values["result.json"]
    model_pair = values["model-pair.json"]
    preflight = values["gpu-memory.json"]
    terminal = values["terminal.json"]
    training_attempt = values["training-attempt.json"]
    stderr = values["stage-stderr.log"]
    assert isinstance(manifest, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(model_pair, Mapping)
    assert isinstance(preflight, Mapping)
    assert isinstance(terminal, Mapping)
    assert isinstance(training_attempt, Mapping)
    assert isinstance(stderr, bytes)
    inventory = manifest.get("files")
    epoch = result.get("epoch")
    if (
        manifest.get("resultSha256") != FAILURE_REFERENCE_SHA256["result.json"]
        or not isinstance(inventory, Mapping)
        or inventory.get("model-pair.json", {}).get("sha256")
        != FAILURE_REFERENCE_SHA256["model-pair.json"]
        or inventory.get("training-checkpoint.pt", {}).get("sha256")
        != "b69b35aeac28f56b7c961c490ffe33e0b89469c07e2bff75d1a5d2d0b1c8664a"
        or result.get("outputModelPair") is None
        or result.get("outputModelPair", {}).get("pairManifestSha256")
        != FAILURE_REFERENCE_SHA256["model-pair.json"]
        or not isinstance(epoch, Mapping)
        or epoch.get("epoch") != 1
        or result.get("hardGates", {}).get("passed") is not True
        or result.get("config", {}).get("epochs") != 1
        or result.get("config", {}).get("use_amp") is not False
        or preflight.get("passed") is not True
        or training_attempt.get("gpuMemoryPreflightReportSha256")
        != FAILURE_REFERENCE_SHA256["gpu-memory.json"]
        or terminal
        != {
            "exitCode": 1,
            "format": "dalmuti-v5-run007-durable-terminal",
            "intentSha256": EXPECTED_DURABLE_TRAIN_INTENT_SHA256,
            "passed": False,
            "processSha256": EXPECTED_DURABLE_TRAIN_PROCESS_SHA256,
            "runNamespace": "v5-mappo-normalresidual-i001-s840060001-run-007",
            "stage": "train",
            "stderrSha256": FAILURE_REFERENCE_SHA256["stage-stderr.log"],
            "stdoutSha256": EMPTY_SHA256,
            "version": 1,
        }
        or len(stderr) != 1006
        or not stderr.endswith(
            b"ValueError: run-007 training result/config/preflight binding drifted\n"
        )
    ):
        raise ValueError("run-007 completed-training failure reference drifted")
    values["epochSha256"] = sha256_bytes(canonical_json_bytes(epoch))
    values["hardGatesSha256"] = sha256_bytes(
        canonical_json_bytes(result["hardGates"])
    )
    values["initialBehaviorBindingsSha256"] = sha256_bytes(
        canonical_json_bytes(result["initialBehaviorBindings"])
    )
    values["stderrBase64"] = base64.b64encode(stderr).decode("ascii")
    proof_path = EPOCH_ONLY_PROOF_ROOT / "epoch-only-verifier-proof.json"
    proof, proof_sha = _read_json_pair(proof_path, "epoch-only verifier proof")
    probe = EPOCH_ONLY_PROOF_ROOT / "epoch-only-verifier-probe.py"
    if (
        proof_sha != EPOCH_ONLY_PROOF_SHA256
        or not probe.is_file()
        or probe.is_symlink()
        or sha256_file(probe) != EPOCH_ONLY_PROBE_SHA256
        or proof.get("format") != "dalmuti-v5-run007-epoch-only-verifier-proof"
        or proof.get("passed") is not True
        or proof.get("epochPatchCalls") != 1
        or proof.get("remoteMutationCount") != 0
        or proof.get("originalEpochType") != "dict"
        or proof.get("originalEpochInner") != 1
        or proof.get("probeScriptSha256") != EPOCH_ONLY_PROBE_SHA256
        or proof.get("manifestSha256") != FAILURE_REFERENCE_SHA256["manifest.json"]
        or proof.get("resultSha256") != FAILURE_REFERENCE_SHA256["result.json"]
        or proof.get("trainingAttemptSha256")
        != FAILURE_REFERENCE_SHA256["training-attempt.json"]
        or proof.get("candidatePairId")
        != result.get("outputModelPair", {}).get("pairId")
        or proof.get("candidateActorSha256")
        != result.get("outputModelPair", {}).get("actorSha256")
    ):
        raise ValueError("epoch-only verifier proof binding drifted")
    values["epochOnlyProof"] = proof
    values["epochOnlyProofSha256"] = proof_sha
    return values


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_pair(path: Path, label: str) -> tuple[dict[str, object], str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is absent or unsafe: {path}")
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.is_symlink()
        or sidecar.read_bytes() != f"{digest}  {path.name}\n".encode("ascii")
    ):
        raise ValueError(f"{label} checksum sidecar drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, digest


def _require_template(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} template is absent or unsafe: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} template SHA-256 drifted: expected {expected}, got {actual}"
        )
    return path.read_text(encoding="utf-8")


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label} anchor count drifted: expected 1, got {count}")
    return source.replace(old, new, 1)


def _replace_regex_once(
    source: str, pattern: str, replacement: str, label: str
) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"{label} regex anchor count drifted: expected 1, got {count}")
    return updated


def _validate_target(
    run_root: Path, run_namespace: str, source_commit: str
) -> dict[str, object]:
    if NAMESPACE_RE.fullmatch(run_namespace) is None:
        raise ValueError("run namespace has an unsafe or unsupported shape")
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a lowercase 40-character Git object id")
    if run_root.is_symlink():
        raise ValueError("target run root must not be a symlink")
    root = run_root.resolve(strict=True)
    if root.name != run_namespace:
        raise ValueError("target run directory name differs from the requested namespace")
    workflow, workflow_sha = _read_json_pair(root / "workflow.json", "workflow")
    if (
        workflow.get("runNamespace") != run_namespace
        or workflow.get("runDirectoryName") != run_namespace
        or workflow.get("sourceCommit") != source_commit
    ):
        raise ValueError("target workflow identity differs from build inputs")
    manifest_path = root / "source-seal/manifest.json"
    manifest, manifest_sha = _read_json_pair(manifest_path, "source seal manifest")
    source_seal = workflow.get("sourceSeal")
    if (
        not isinstance(source_seal, Mapping)
        or source_seal.get("manifestSha256") != manifest_sha
        or manifest.get("sourceCommit") != source_commit
    ):
        raise ValueError("workflow/source-seal commit binding drifted")
    evaluation = manifest.get("evaluationSource")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("sourceCommit") != source_commit
        or not isinstance(evaluation.get("sourceInventorySha256"), str)
        or SHA256_RE.fullmatch(str(evaluation.get("sourceInventorySha256"))) is None
        or not isinstance(evaluation.get("files"), Mapping)
        or not evaluation["files"]
    ):
        raise ValueError("source seal evaluation inventory is absent or invalid")
    source_checkout = root / "source-checkout"
    files = evaluation["files"]
    assert isinstance(files, Mapping)
    for relative, expected in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or SHA256_RE.fullmatch(expected) is None
        ):
            raise ValueError("source seal evaluation inventory entry is invalid")
        path = source_checkout / Path(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"materialized evaluation source drifted: {relative}")
    # v5_low_disk_stage is a lazy runtime dependency that the historical source
    # seal records outside evaluationSource.files.  Bind its materialized bytes
    # independently; the run007 source-seal preflight is responsible for proving
    # that every materialized source byte also equals the commit/bundle/tar byte.
    low_disk = source_checkout / "gpu-training/v5_low_disk_stage.py"
    if not low_disk.is_file() or low_disk.is_symlink():
        raise FileNotFoundError("materialized v5_low_disk_stage.py is absent or unsafe")
    low_disk_sha = sha256_file(low_disk)
    lineage = workflow.get("corpusLineage")
    low_disk_plan, low_disk_plan_sha = _read_json_pair(
        root / "collection/corpus-lineage/low-disk-stage-plan/plan.json",
        "corpus low-disk stage plan",
    )
    tiers = low_disk_plan.get("tiers")
    persistent = tiers.get("persistent") if isinstance(tiers, Mapping) else None
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("sourceLowDiskStagePlanSha256") != low_disk_plan_sha
        or not isinstance(persistent, Mapping)
        or persistent.get("minimumReserveBytes")
        != MINIMUM_EVALUATION_FREE_BYTES
        or persistent.get("reserveBytes") != MINIMUM_EVALUATION_FREE_BYTES
    ):
        raise ValueError("corpus low-disk plan/free-space contract drifted")
    controls = root / "controls"
    training_control = controls / TRAINING_CONTROL_NAME
    durable_training_launcher = controls / "launch_durable_training_stage.py"
    control_common = controls / "control_common.py"
    for path, label in (
        (training_control, "training control"),
        (durable_training_launcher, "durable training launcher"),
        (control_common, "control common helper"),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} is absent or unsafe: {path}")
    return {
        "controlCommonSha256": sha256_file(control_common),
        "corpusLowDiskPlanSha256": low_disk_plan_sha,
        "durableTrainingLauncherSha256": sha256_file(durable_training_launcher),
        "evaluationSourceInventorySha256": evaluation["sourceInventorySha256"],
        "lowDiskStageSha256": low_disk_sha,
        "manifest": manifest,
        "sourceManifestSha256": manifest_sha,
        "trainingControlSha256": sha256_file(training_control),
        "workflowSha256": workflow_sha,
    }


def _extract_run004_functions(source: str) -> str:
    tree = ast.parse(source)
    by_name = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lines = source.splitlines(keepends=True)
    chunks: list[str] = []
    for name in EXTRACTED_FUNCTIONS:
        node = by_name.get(name)
        if not isinstance(node, ast.FunctionDef) or node.end_lineno is None:
            raise ValueError(f"run-004 evaluation template function is absent: {name}")
        chunks.append("".join(lines[node.lineno - 1 : node.end_lineno]).rstrip() + "\n")
    extracted = "\n\n".join(chunks)
    if "recover_promotion_lock" in extracted or "TRAINING_CONFIG" in extracted:
        raise ValueError("run-004 extraction crossed an evaluation-only boundary")
    return extracted.replace("run-004", "sealed run")


def _render_training_recovery_verifier(
    *,
    run_namespace: str,
    source_commit: str,
    training_control_sha256: str,
    durable_training_launcher_sha256: str,
    source_manifest_sha256: str,
    workflow_sha256: str,
    reference: Mapping[str, object],
) -> bytes:
    manifest = reference["manifest.json"]
    result = reference["result.json"]
    model_pair = reference["model-pair.json"]
    terminal = reference["terminal.json"]
    training_attempt = reference["training-attempt.json"]
    preflight = reference["gpu-memory.json"]
    assert isinstance(manifest, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(model_pair, Mapping)
    assert isinstance(terminal, Mapping)
    assert isinstance(training_attempt, Mapping)
    assert isinstance(preflight, Mapping)
    template = r'''from __future__ import annotations

"""Verify the completed run-007 training after its sealed verifier bug.

The original failed durable terminal and every training artifact remain immutable.
This control independently proves the completed output, exact failure traceback,
and the single known epoch-shape bug before publishing one O_EXCL receipt.
"""

import argparse
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Mapping, Sequence

from control_common import (
    HYBRID_INDEX_SHA256,
    PAIR_ID,
    PAIR_MANIFEST_SHA256,
    RUN_NAME,
    SOURCE_COMMIT,
    TRAINING_SEED,
    canonical_json_bytes,
    publish_json_pair_exclusive,
    read_canonical_json,
    script_run_root,
    sha256_file,
    validate_run_root,
    verify_source_blob_admission,
    verify_sidecar,
)


EXPECTED_RUN_NAMESPACE = __RUN_NAMESPACE__
EXPECTED_SOURCE_COMMIT = __SOURCE_COMMIT__
TRAINING_CONTROL_SHA256 = __TRAINING_CONTROL_SHA256__
DURABLE_TRAINING_LAUNCHER_SHA256 = __DURABLE_TRAINING_LAUNCHER_SHA256__
SOURCE_MANIFEST_SHA256 = __SOURCE_MANIFEST_SHA256__
WORKFLOW_SHA256 = __WORKFLOW_SHA256__
REMOTE_PARENT = Path("/home/pangmin/dalmuti")
REMOTE_PYTHON_TEXT = "/home/pangmin/dalmuti/gpu-bundle-v3/.venv/bin/python"
REMOTE_PYTHON = Path(REMOTE_PYTHON_TEXT)
DEVICE = "cuda:0"
MINIMUM_NOFILE = 65_536
EXPECTED_RUNTIME_PYTHON_PATH_COUNT = 143
EXPECTED_CHECKED_SOURCE_PATH_COUNT = 154
EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256 = (
    "fdf1aebbdf29606997f4b4ff4965a73f3240c2b2a4bff445cfdc48698442117f"
)
EXPECTED_SOURCE_UNION_INVENTORY_SHA256 = (
    "7208de2a882c4941675dd8a5e9036f44095d6b5857bb7dd18889efcf5054d2ab"
)
RECOVERY_RECEIPT_RELATIVE = Path(__RECOVERY_RECEIPT_RELATIVE__)
EXPECTED_PREFLIGHT_SHA256 = __EXPECTED_PREFLIGHT_SHA256__
EXPECTED_TRAINING_MANIFEST_SHA256 = __EXPECTED_TRAINING_MANIFEST_SHA256__
EXPECTED_TRAINING_RESULT_SHA256 = __EXPECTED_TRAINING_RESULT_SHA256__
EXPECTED_MODEL_PAIR_SHA256 = __EXPECTED_MODEL_PAIR_SHA256__
EXPECTED_TRAINING_ATTEMPT_SHA256 = __EXPECTED_TRAINING_ATTEMPT_SHA256__
EXPECTED_DURABLE_TRAIN_INTENT_SHA256 = __EXPECTED_DURABLE_TRAIN_INTENT_SHA256__
EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256 = __EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256__
EXPECTED_DURABLE_TRAIN_PROCESS_SHA256 = __EXPECTED_DURABLE_TRAIN_PROCESS_SHA256__
EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256 = __EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256__
EXPECTED_DURABLE_TRAIN_STDERR_SHA256 = __EXPECTED_DURABLE_TRAIN_STDERR_SHA256__
EXPECTED_EMPTY_SHA256 = __EXPECTED_EMPTY_SHA256__
EXPECTED_EPOCH_SHA256 = __EXPECTED_EPOCH_SHA256__
EXPECTED_HARD_GATES_SHA256 = __EXPECTED_HARD_GATES_SHA256__
EXPECTED_INITIAL_BINDINGS_SHA256 = __EXPECTED_INITIAL_BINDINGS_SHA256__
EXPECTED_EPOCH_ONLY_PROOF_SHA256 = __EXPECTED_EPOCH_ONLY_PROOF_SHA256__
EXPECTED_EPOCH_ONLY_PROBE_SHA256 = __EXPECTED_EPOCH_ONLY_PROBE_SHA256__
EXPECTED_FAILURE_STDERR = base64.b64decode(__EXPECTED_FAILURE_STDERR_B64__)
EXPECTED_MANIFEST = json.loads(__EXPECTED_MANIFEST_JSON__)
EXPECTED_MODEL_PAIR = json.loads(__EXPECTED_MODEL_PAIR_JSON__)
EXPECTED_TRAINING_ATTEMPT = json.loads(__EXPECTED_TRAINING_ATTEMPT_JSON__)
EXPECTED_DURABLE_TRAIN_TERMINAL = json.loads(__EXPECTED_TERMINAL_JSON__)
EXPECTED_TRAINING_CONFIG = json.loads(__EXPECTED_TRAINING_CONFIG_JSON__)
EXPECTED_PREFLIGHT_CONFIG = json.loads(__EXPECTED_PREFLIGHT_CONFIG_JSON__)
EXPECTED_GPU_ADMISSION = json.loads(__EXPECTED_GPU_ADMISSION_JSON__)
EXPECTED_HARD_GATES = json.loads(__EXPECTED_HARD_GATES_JSON__)
EXPECTED_INITIAL_BINDINGS = json.loads(__EXPECTED_INITIAL_BINDINGS_JSON__)
EXPECTED_OUTPUT_PAIR = json.loads(__EXPECTED_OUTPUT_PAIR_JSON__)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

if RUN_NAME != EXPECTED_RUN_NAMESPACE or SOURCE_COMMIT != EXPECTED_SOURCE_COMMIT:
    raise RuntimeError("recovery verifier identity differs from control_common")


def _binding_sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validated_source_admission(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "checkedPathCount",
            "passed",
            "runtimePythonPathCount",
            "runtimePythonSourceInventorySha256",
            "sourceCommit",
            "sourceUnionInventorySha256",
        }
        or value.get("passed") is not True
        or value.get("sourceCommit") != EXPECTED_SOURCE_COMMIT
        or value.get("runtimePythonPathCount") != EXPECTED_RUNTIME_PYTHON_PATH_COUNT
        or value.get("checkedPathCount") != EXPECTED_CHECKED_SOURCE_PATH_COUNT
        or value.get("runtimePythonSourceInventorySha256")
        != EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256
        or value.get("sourceUnionInventorySha256")
        != EXPECTED_SOURCE_UNION_INVENTORY_SHA256
    ):
        raise ValueError("full source-blob admission identity drifted")
    return dict(value)


def _source_modules(run_root: Path) -> dict[str, object]:
    source = run_root / "source-checkout/gpu-training"
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError("sealed gpu-training checkout is absent")
    source_text = str(source)
    if source_text not in sys.path:
        # v5_workflow deliberately imports v5_collection_plan, v5_dataset,
        # and v5_low_disk_stage only when its dataset verifiers run.
        sys.path.insert(0, source_text)
    modules = {
        name: importlib.import_module(name)
        for name in ("torch", "v5_gpu_memory_preflight", "v5_train", "v5_workflow")
    }
    sealed = source.resolve()
    for name in ("v5_gpu_memory_preflight", "v5_train", "v5_workflow"):
        module_path = Path(str(modules[name].__file__)).resolve()
        if sealed not in module_path.parents:
            raise ImportError(f"{name} resolved outside the sealed checkout")
    return modules


def _validated_nofile(value: object, *, durable: bool = False) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("RLIMIT_NOFILE record is absent")
    expected = (
        {"afterHard", "afterSoft", "beforeHard", "beforeSoft"}
        if durable
        else {"hard", "minimum", "soft"}
    )
    if set(value) != expected or any(type(value.get(key)) is not int for key in expected):
        raise ValueError("RLIMIT_NOFILE record shape drifted")
    record = {key: int(value[key]) for key in expected}
    soft = record["afterSoft"] if durable else record["soft"]
    hard = record["afterHard"] if durable else record["hard"]
    if soft != -1 and soft < MINIMUM_NOFILE:
        raise ValueError("RLIMIT_NOFILE is below the sealed minimum")
    if hard != -1 and soft != -1 and soft > hard:
        raise ValueError("RLIMIT_NOFILE soft limit exceeds hard limit")
    if not durable and record["minimum"] != MINIMUM_NOFILE:
        raise ValueError("RLIMIT_NOFILE minimum drifted")
    return record


def _remote_python_fingerprint() -> dict[str, object]:
    if not os.path.lexists(os.fspath(REMOTE_PYTHON)):
        raise FileNotFoundError("sealed remote Python is absent")
    resolved = REMOTE_PYTHON.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise PermissionError("sealed remote Python is not executable")
    return {
        "bytes": metadata.st_size,
        "declaredIsSymlink": REMOTE_PYTHON.is_symlink(),
        "declaredPath": REMOTE_PYTHON_TEXT,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "resolvedPath": str(resolved),
        "sha256": sha256_file(resolved),
    }


def _durable_paths(run_root: Path, stage: str) -> dict[str, Path]:
    attempt = run_root / "durable-training" / stage / "attempt-001"
    return {
        "attempt": attempt,
        "intent": attempt / "intent.json",
        "launch": attempt / "launch.json",
        "process": attempt / "process.json",
        "stderr": attempt / "stage-stderr.log",
        "stdout": attempt / "stage-stdout.log",
        "terminal": attempt / "terminal.json",
        "workerStderr": attempt / "worker-stderr.log",
        "workerStdout": attempt / "worker-stdout.log",
    }


def _expected_durable_intent(
    run_root: Path, stage: str, source_admission: Mapping[str, object]
) -> dict[str, object]:
    return {
        "command": [
            REMOTE_PYTHON_TEXT,
            str(run_root / "controls/run_training_iteration_001.py"),
            stage,
            "--confirm-run-namespace",
            EXPECTED_RUN_NAMESPACE,
        ],
        "controlSha256": TRAINING_CONTROL_SHA256,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(run_root / "source-checkout/gpu-training"),
            "PYTHONUNBUFFERED": "1",
        },
        "format": "dalmuti-v5-run007-durable-stage-intent",
        "launcherSha256": DURABLE_TRAINING_LAUNCHER_SHA256,
        "minimumNoFile": MINIMUM_NOFILE,
        "python": REMOTE_PYTHON_TEXT,
        "pythonRuntime": _remote_python_fingerprint(),
        "runNamespace": EXPECTED_RUN_NAMESPACE,
        "sourceAdmission": dict(source_admission),
        "sourceCommit": EXPECTED_SOURCE_COMMIT,
        "sourceSealSha256": SOURCE_MANIFEST_SHA256,
        "stage": stage,
        "version": 1,
        "workflowSha256": WORKFLOW_SHA256,
    }


def _verify_durable_stage(
    run_root: Path,
    stage: str,
    source_admission: Mapping[str, object],
    *,
    expected_passed: bool,
    expected_intent_sha256: str | None = None,
    expected_process_sha256: str | None = None,
    expected_terminal_sha256: str | None = None,
    expected_launch_sha256: str | None = None,
) -> dict[str, object]:
    paths = _durable_paths(run_root, stage)
    intent_sha = verify_sidecar(paths["intent"])
    intent = read_canonical_json(paths["intent"], f"durable {stage} intent")
    if (
        (expected_intent_sha256 is not None and intent_sha != expected_intent_sha256)
        or intent != _expected_durable_intent(run_root, stage, source_admission)
    ):
        raise ValueError(f"durable {stage} intent drifted")
    launch_sha = verify_sidecar(paths["launch"])
    launch = read_canonical_json(paths["launch"], f"durable {stage} launch")
    if (
        (expected_launch_sha256 is not None and launch_sha != expected_launch_sha256)
        or
        set(launch) != {"format", "intentSha256", "pid", "runNamespace", "stage", "version"}
        or launch.get("format") != "dalmuti-v5-run007-durable-launch"
        or launch.get("intentSha256") != intent_sha
        or type(launch.get("pid")) is not int
        or int(launch["pid"]) < 1
        or launch.get("runNamespace") != EXPECTED_RUN_NAMESPACE
        or launch.get("stage") != stage
        or launch.get("version") != 1
    ):
        raise ValueError(f"durable {stage} launch drifted")
    process_sha = verify_sidecar(paths["process"])
    process = read_canonical_json(paths["process"], f"durable {stage} process")
    if (
        (expected_process_sha256 is not None and process_sha != expected_process_sha256)
        or set(process)
        != {
            "format",
            "intentSha256",
            "noFile",
            "pid",
            "pythonRuntime",
            "runNamespace",
            "stage",
            "version",
        }
        or process.get("format") != "dalmuti-v5-run007-durable-process"
        or process.get("intentSha256") != intent_sha
        or type(process.get("pid")) is not int
        or int(process["pid"]) < 1
        or process.get("pythonRuntime") != _remote_python_fingerprint()
        or process.get("runNamespace") != EXPECTED_RUN_NAMESPACE
        or process.get("stage") != stage
        or process.get("version") != 1
    ):
        raise ValueError(f"durable {stage} process drifted")
    _validated_nofile(process.get("noFile"), durable=True)
    for name in ("stdout", "stderr"):
        if not paths[name].is_file() or paths[name].is_symlink():
            raise FileNotFoundError(f"durable {stage} {name} log is absent or linked")
    terminal_sha = verify_sidecar(paths["terminal"])
    terminal = read_canonical_json(paths["terminal"], f"durable {stage} terminal")
    expected_terminal = {
        "exitCode": terminal.get("exitCode"),
        "format": "dalmuti-v5-run007-durable-terminal",
        "intentSha256": intent_sha,
        "passed": expected_passed,
        "processSha256": process_sha,
        "runNamespace": EXPECTED_RUN_NAMESPACE,
        "stage": stage,
        "stderrSha256": sha256_file(paths["stderr"]),
        "stdoutSha256": sha256_file(paths["stdout"]),
        "version": 1,
    }
    if (
        (expected_terminal_sha256 is not None and terminal_sha != expected_terminal_sha256)
        or type(terminal.get("exitCode")) is not int
        or (int(terminal["exitCode"]) == 0) is not expected_passed
        or terminal != expected_terminal
    ):
        raise ValueError(f"durable {stage} terminal drifted")
    return {
        "intentSha256": intent_sha,
        "launchSha256": launch_sha,
        "processSha256": process_sha,
        "stderrSha256": terminal["stderrSha256"],
        "stdoutSha256": terminal["stdoutSha256"],
        "terminalSha256": terminal_sha,
    }


def _verify_completed_training_topology(run_root: Path) -> dict[str, object]:
    durable_root = run_root / "durable-training"
    train_root = durable_root / "train"
    attempt = train_root / "attempt-001"
    if any(path.is_symlink() for path in (durable_root, train_root, attempt)):
        raise ValueError("completed-training durable topology contains a symlink")
    if (
        not attempt.is_dir()
        or {path.name for path in train_root.iterdir()} != {"attempt-001"}
        or os.path.lexists(os.fspath(durable_root / "verify-training"))
    ):
        raise ValueError("completed-training durable attempt topology drifted")
    expected_files = {
        "intent.json",
        "intent.json.sha256",
        "launch.json",
        "launch.json.sha256",
        "process.json",
        "process.json.sha256",
        "stage-stderr.log",
        "stage-stdout.log",
        "terminal.json",
        "terminal.json.sha256",
        "worker-stderr.log",
        "worker-stdout.log",
    }
    actual_files: set[str] = set()
    for path in attempt.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("completed-training durable attempt contains an unsafe entry")
        actual_files.add(path.name)
    if actual_files != expected_files:
        raise ValueError("completed-training durable attempt file set drifted")
    for name in ("stage-stdout.log", "worker-stderr.log", "worker-stdout.log"):
        path = attempt / name
        if path.stat().st_size != 0 or sha256_file(path) != EXPECTED_EMPTY_SHA256:
            raise ValueError(f"completed-training durable empty log drifted: {name}")
    stderr = attempt / "stage-stderr.log"
    if stderr.stat().st_size != 1006 or sha256_file(stderr) != EXPECTED_DURABLE_TRAIN_STDERR_SHA256:
        raise ValueError("completed-training durable failure stderr drifted")
    return {
        "attemptCount": 1,
        "attemptFileCount": len(actual_files),
        "verifyTrainingAbsent": True,
    }


def _exact_configs(modules: Mapping[str, object]) -> tuple[object, object]:
    training = modules["v5_train"].V5TrainingConfig(**EXPECTED_TRAINING_CONFIG)  # type: ignore[attr-defined]
    preflight = modules["v5_gpu_memory_preflight"].V5GPUMemoryPreflightConfig(  # type: ignore[attr-defined]
        **EXPECTED_PREFLIGHT_CONFIG
    )
    if (
        training.to_dict() != EXPECTED_TRAINING_CONFIG  # type: ignore[attr-defined]
        or preflight.__dict__ != EXPECTED_PREFLIGHT_CONFIG
        or training.use_amp is not False  # type: ignore[attr-defined]
        or preflight.use_amp is not False  # type: ignore[attr-defined]
    ):
        raise ValueError("sealed FP32 configuration drifted")
    return training, preflight


def _verify_dataset(
    run_root: Path, modules: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], Path]:
    workflow = modules["v5_workflow"].load_v5_run(run_root)  # type: ignore[attr-defined]
    initial_pair = modules["v5_train"].verify_v5_model_pair(  # type: ignore[attr-defined]
        run_root / "initialization"
    )
    dataset = run_root / "collection/index"
    if (
        workflow.get("sourceCommit") != EXPECTED_SOURCE_COMMIT
        or workflow.get("seeds", {}).get("training") != TRAINING_SEED
        or initial_pair.get("pairId") != PAIR_ID
        or initial_pair.get("pairManifestSha256") != PAIR_MANIFEST_SHA256
        or verify_sidecar(dataset / "manifest.json") != HYBRID_INDEX_SHA256
    ):
        raise ValueError("workflow/initial-pair/dataset binding drifted")
    modules["v5_workflow"]._verify_v5_training_execution_source(  # type: ignore[attr-defined]
        run_root, workflow, run_root / "source-checkout"
    )
    verified_dataset = modules["v5_workflow"]._verify_v5_training_dataset(  # type: ignore[attr-defined]
        run_root, workflow, dataset, initial_pair
    )
    sealed = (run_root / "source-checkout/gpu-training").resolve()
    for name in ("v5_collection_plan", "v5_dataset", "v5_low_disk_stage"):
        delayed = sys.modules.get(name)
        delayed_path = None if delayed is None else getattr(delayed, "__file__", None)
        if delayed_path is None or sealed not in Path(str(delayed_path)).resolve().parents:
            raise ImportError(f"delayed source module resolved outside sealed checkout: {name}")
    if verified_dataset != dataset.resolve():
        raise ValueError("dataset verifier returned a foreign path")
    return dict(workflow), dict(initial_pair), verified_dataset


def _verify_execution_attempt(
    run_root: Path,
    stage: str,
    *,
    expected_sha256: str | None = None,
    preflight_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    path = (
        run_root / "preflight/execution-attempt.json"
        if stage == "preflight"
        else run_root / "training-execution/attempt-001.json"
    )
    digest = verify_sidecar(path)
    value = read_canonical_json(path, f"{stage} execution attempt")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{stage} execution attempt exact SHA-256 drifted")
    _validated_nofile(value.get("inheritedNoFile"))
    if stage == "train":
        if value != EXPECTED_TRAINING_ATTEMPT:
            raise ValueError("training execution attempt differs from captured failure")
        return value, digest
    workflow = read_canonical_json(run_root / "workflow.json", "workflow")
    expected = {
        "corpusLineageSha256": _binding_sha(workflow["corpusLineage"]),
        "datasetIdentitySha256": HYBRID_INDEX_SHA256,
        "device": DEVICE,
        "format": "dalmuti-v5-run007-preflight-execution-attempt",
        "gpuPreflightConfig": EXPECTED_PREFLIGHT_CONFIG,
        "gpuPreflightConfigSha256": _binding_sha(EXPECTED_PREFLIGHT_CONFIG),
        "inheritedNoFile": value.get("inheritedNoFile"),
        "initialModelPairId": PAIR_ID,
        "initialModelPairManifestSha256": PAIR_MANIFEST_SHA256,
        "pid": value.get("pid"),
        "runNamespace": EXPECTED_RUN_NAMESPACE,
        "sourceCommit": EXPECTED_SOURCE_COMMIT,
        "stage": "preflight",
        "trainingConfig": EXPECTED_TRAINING_CONFIG,
        "trainingConfigSha256": _binding_sha(EXPECTED_TRAINING_CONFIG),
        "trainingSeed": TRAINING_SEED,
        "version": 1,
        "workflowSha256": WORKFLOW_SHA256,
    }
    if type(value.get("pid")) is not int or int(value["pid"]) < 1 or value != expected:
        raise ValueError("preflight execution attempt binding drifted")
    return value, digest


def _verify_preflight(
    run_root: Path,
    modules: Mapping[str, object],
    dataset: Path,
    source_admission: Mapping[str, object],
) -> dict[str, object]:
    training_config, preflight_config = _exact_configs(modules)
    path = run_root / "preflight/gpu-memory.json"
    if verify_sidecar(path) != EXPECTED_PREFLIGHT_SHA256:
        raise ValueError("captured GPU preflight report SHA-256 drifted")
    report, report_sha = modules["v5_gpu_memory_preflight"].load_v5_gpu_memory_preflight_report(path)  # type: ignore[attr-defined]
    admission = modules["v5_gpu_memory_preflight"].verify_v5_gpu_memory_admission(  # type: ignore[attr-defined]
        path,
        dataset,
        run_root / "initialization",
        config=preflight_config,
        training_config=training_config,
        device=DEVICE,
    )
    if (
        report_sha != EXPECTED_PREFLIGHT_SHA256
        or report.get("passed") is not True
        or report.get("trainingConfig") != EXPECTED_TRAINING_CONFIG
        or report.get("config") != EXPECTED_PREFLIGHT_CONFIG
        or report.get("trainingCanary", {}).get("precision") != "fp32"
        or report.get("trainingCanaryExecution", {}).get("actor", {}).get("allFinite") is not True
        or admission != EXPECTED_GPU_ADMISSION
    ):
        raise ValueError("original FP32 GPU admission drifted")
    _, attempt_sha = _verify_execution_attempt(run_root, "preflight")
    durable = _verify_durable_stage(
        run_root, "preflight", source_admission, expected_passed=True
    )
    return {
        "admissionSha256": _binding_sha(admission),
        "attemptSha256": attempt_sha,
        "durable": durable,
        "reportSha256": report_sha,
    }


def _verify_training_inventory(training_path: Path) -> tuple[dict[str, object], str, str]:
    if training_path.is_symlink():
        raise ValueError("training output root must not be a symlink")
    training = training_path.resolve(strict=True)
    if not training.is_dir():
        raise ValueError("training output root is not a directory")
    manifest_path = training / "manifest.json"
    manifest_sha = verify_sidecar(manifest_path)
    manifest = read_canonical_json(manifest_path, "training manifest")
    if manifest_sha != EXPECTED_TRAINING_MANIFEST_SHA256 or manifest != EXPECTED_MANIFEST:
        raise ValueError("training manifest differs from the captured completed output")
    inventory = manifest.get("files")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("training artifact inventory is absent")
    expected_files = {"manifest.json", "manifest.json.sha256", *inventory.keys()}
    actual_files: set[str] = set()
    for path in training.rglob("*"):
        if path.is_symlink():
            raise ValueError("training output contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(training).as_posix())
    if actual_files != expected_files:
        raise ValueError("training artifact inventory file set drifted")
    normalized: dict[str, dict[str, object]] = {}
    for relative, record in inventory.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("training artifact record is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in ("", ".", "..") for part in pure.parts)
            or set(record) != {"bytes", "sha256"}
            or type(record.get("bytes")) is not int
            or int(record["bytes"]) < 0
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(str(record["sha256"])) is None
        ):
            raise ValueError("training artifact record shape drifted")
        unresolved = training.joinpath(*pure.parts)
        if unresolved.is_symlink():
            raise ValueError("training artifact must not be a symlink")
        path = unresolved.resolve(strict=True)
        if training not in path.parents or not path.is_file():
            raise ValueError("training artifact escaped its immutable root")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError(f"training artifact hash drifted: {relative}")
        normalized[relative] = dict(record)
    return manifest, manifest_sha, _binding_sha(normalized)


def _validated_epoch(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("completed training epoch must be an object")
    epoch = dict(value)
    batching = epoch.get("batching")
    if (
        epoch.get("epoch") != 1
        or _binding_sha(epoch) != EXPECTED_EPOCH_SHA256
        or epoch.get("actorDecisionRowsSeen") != 1_602_500
        or epoch.get("criticDecisionRowsSeen") != 4_514_492
        or epoch.get("decisionRowsSeen") != 4_514_492
        or epoch.get("actorOptimizerSteps") != 50_078
        or epoch.get("criticOptimizerSteps") != 17_634
        or epoch.get("optimizerSteps") != 67_712
        or epoch.get("optimizerSteps")
        != epoch.get("actorOptimizerSteps") + epoch.get("criticOptimizerSteps")
        or epoch.get("valueOnlyOptimizerSteps") != epoch.get("criticOptimizerSteps")
        or not isinstance(batching, Mapping)
        or batching.get("shardCount") != 29
    ):
        raise ValueError("completed training epoch semantic binding drifted")
    return epoch


def _verify_training_result(
    run_root: Path,
    modules: Mapping[str, object],
    initial_pair: Mapping[str, object],
) -> dict[str, object]:
    training = run_root / "training"
    manifest, manifest_sha, inventory_sha = _verify_training_inventory(training)
    result_path = training / "result.json"
    if sha256_file(result_path) != EXPECTED_TRAINING_RESULT_SHA256:
        raise ValueError("training result exact SHA-256 drifted")
    result = read_canonical_json(result_path, "training result")
    pair_manifest = read_canonical_json(training / "model-pair.json", "model pair")
    if (
        sha256_file(training / "model-pair.json") != EXPECTED_MODEL_PAIR_SHA256
        or pair_manifest != EXPECTED_MODEL_PAIR
    ):
        raise ValueError("captured model-pair manifest drifted")
    output_pair = modules["v5_train"].verify_v5_model_pair(training)  # type: ignore[attr-defined]
    actor = modules["v5_train"].v5_actor_bundle_digests(training / "actor-bundle")  # type: ignore[attr-defined]
    epoch = _validated_epoch(result.get("epoch"))
    if (
        result.get("format") != "dalmuti-v5-mappo-training-result"
        or result.get("version") != 1
        or result.get("config") != EXPECTED_TRAINING_CONFIG
        or result.get("datasetIdentitySha256") != HYBRID_INDEX_SHA256
        or result.get("datasetDecisionCount") != 4_514_492
        or result.get("datasetShardCount") != 29
        or result.get("gpuMemoryPreflight") != EXPECTED_GPU_ADMISSION
        or result.get("hardGates") != EXPECTED_HARD_GATES
        or _binding_sha(result.get("hardGates")) != EXPECTED_HARD_GATES_SHA256
        or result.get("initialBehaviorBindings") != EXPECTED_INITIAL_BINDINGS
        or _binding_sha(result.get("initialBehaviorBindings")) != EXPECTED_INITIAL_BINDINGS_SHA256
        or result.get("outputModelPair") != output_pair
        or output_pair != EXPECTED_OUTPUT_PAIR
        or result.get("outputActor") != actor
        or result.get("outputCritic", {}).get("criticSha256") != output_pair.get("criticSha256")
        or result.get("outputCritic", {}).get("tensorStateSha256")
        != output_pair.get("criticTensorStateSha256")
        or result.get("initialActorStateSha256") != initial_pair.get("actorTensorStateSha256")
        or result.get("initialCriticStateSha256") != initial_pair.get("criticTensorStateSha256")
        or manifest.get("resultSha256") != EXPECTED_TRAINING_RESULT_SHA256
        or manifest.get("gpuMemoryPreflight") != EXPECTED_GPU_ADMISSION
        or manifest.get("initialBehaviorBindings") != EXPECTED_INITIAL_BINDINGS
        or result.get("config", {}).get("use_amp") is not False
    ):
        raise ValueError("completed training result semantic binding drifted")
    torch = modules["torch"]
    checkpoint = torch.load(  # type: ignore[attr-defined]
        training / "training-checkpoint.pt", map_location="cpu", weights_only=False
    )
    checkpoint_keys = {
        "actorStateSha256",
        "actorStateDict",
        "config",
        "criticStateSha256",
        "criticStateDict",
        "datasetIdentitySha256",
        "epoch",
        "format",
        "modelPairId",
        "numpyRngState",
        "optimizerSha256",
        "policyNumericsSha256",
        "pythonRngState",
        "torchCpuRngState",
        "torchCudaRngStates",
        "version",
    }
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != checkpoint_keys
        or checkpoint.get("format") != modules["v5_train"].V5_CHECKPOINT_FORMAT  # type: ignore[attr-defined]
        or checkpoint.get("version") != 1
        or checkpoint.get("config") != EXPECTED_TRAINING_CONFIG
        or checkpoint.get("epoch") != 1
        or checkpoint.get("modelPairId") != output_pair["pairId"]
        or checkpoint.get("datasetIdentitySha256") != HYBRID_INDEX_SHA256
        or checkpoint.get("optimizerSha256") != sha256_file(training / "optimizer.pt")
        or checkpoint.get("actorStateSha256")
        != modules["v5_train"].tensor_state_sha256(checkpoint.get("actorStateDict"))  # type: ignore[attr-defined]
        or checkpoint.get("actorStateSha256") != output_pair["actorTensorStateSha256"]
        or checkpoint.get("criticStateSha256")
        != modules["v5_train"].tensor_state_sha256(checkpoint.get("criticStateDict"))  # type: ignore[attr-defined]
        or checkpoint.get("criticStateSha256") != output_pair["criticTensorStateSha256"]
    ):
        raise ValueError("completed training checkpoint binding drifted")
    optimizer = torch.load(  # type: ignore[attr-defined]
        training / "optimizer.pt", map_location="cpu", weights_only=False
    )
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer)
        != {
            "actorOptimizer",
            "criticOptimizer",
            "format",
            "gradientScaler",
            "modelPairId",
            "policyNumericsSha256",
            "version",
        }
        or optimizer.get("format") != modules["v5_train"].V5_OPTIMIZER_FORMAT  # type: ignore[attr-defined]
        or optimizer.get("version") != 1
        or optimizer.get("modelPairId") != output_pair["pairId"]
    ):
        raise ValueError("completed optimizer checkpoint binding drifted")
    attempt, attempt_sha = _verify_execution_attempt(
        run_root,
        "train",
        expected_sha256=EXPECTED_TRAINING_ATTEMPT_SHA256,
        preflight_sha256=EXPECTED_PREFLIGHT_SHA256,
    )
    if attempt.get("gpuMemoryPreflightReportSha256") != EXPECTED_PREFLIGHT_SHA256:
        raise ValueError("training attempt/preflight binding drifted")
    return {
        "candidateActorSha256": output_pair["actorSha256"],
        "candidatePairId": output_pair["pairId"],
        "candidatePairManifestSha256": output_pair["pairManifestSha256"],
        "checkpointSha256": sha256_file(training / "training-checkpoint.pt"),
        "epochSha256": _binding_sha(epoch),
        "hardGatesSha256": _binding_sha(result["hardGates"]),
        "initialBehaviorBindingsSha256": _binding_sha(result["initialBehaviorBindings"]),
        "manifestSha256": manifest_sha,
        "optimizerSha256": sha256_file(training / "optimizer.pt"),
        "resultSha256": EXPECTED_TRAINING_RESULT_SHA256,
        "trainingArtifactFileCount": len(manifest["files"]),
        "trainingArtifactInventorySha256": inventory_sha,
        "trainingAttemptSha256": attempt_sha,
    }


def build_recovery_receipt(run_root: Path) -> dict[str, object]:
    root = validate_run_root(run_root)
    source_admission = _validated_source_admission(
        verify_source_blob_admission(root)
    )
    modules = _source_modules(root)
    workflow, initial_pair, dataset = _verify_dataset(root, modules)
    preflight = _verify_preflight(root, modules, dataset, source_admission)
    training = _verify_training_result(root, modules, initial_pair)
    durable = _verify_durable_stage(
        root,
        "train",
        source_admission,
        expected_passed=False,
        expected_intent_sha256=EXPECTED_DURABLE_TRAIN_INTENT_SHA256,
        expected_launch_sha256=EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256,
        expected_process_sha256=EXPECTED_DURABLE_TRAIN_PROCESS_SHA256,
        expected_terminal_sha256=EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256,
    )
    topology = _verify_completed_training_topology(root)
    stderr_path = _durable_paths(root, "train")["stderr"]
    stdout_path = _durable_paths(root, "train")["stdout"]
    if (
        stderr_path.read_bytes() != EXPECTED_FAILURE_STDERR
        or sha256_file(stderr_path) != EXPECTED_DURABLE_TRAIN_STDERR_SHA256
        or stdout_path.read_bytes() != b""
        or sha256_file(stdout_path) != EXPECTED_EMPTY_SHA256
        or durable["terminalSha256"] != EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256
        or read_canonical_json(
            _durable_paths(root, "train")["terminal"], "failed train terminal"
        )
        != EXPECTED_DURABLE_TRAIN_TERMINAL
    ):
        raise ValueError("exact completed-training failure traceback drifted")
    if _validated_source_admission(verify_source_blob_admission(root)) != source_admission:
        raise ValueError("source admission changed during recovery verification")
    return {
        **training,
        "datasetIdentitySha256": HYBRID_INDEX_SHA256,
        "durableFailure": durable,
        "durableTopology": topology,
        "epochOnlyVerifierProof": {
            "elapsedSeconds": 156.377,
            "patchCalls": 1,
            "probeScriptSha256": EXPECTED_EPOCH_ONLY_PROBE_SHA256,
            "proofSha256": EXPECTED_EPOCH_ONLY_PROOF_SHA256,
            "remoteMutationCount": 0,
        },
        "format": "dalmuti-v5-run007-completed-training-recovery-verification",
        "passed": True,
        "preflight": preflight,
        "recoveryVerifierSha256": sha256_file(Path(__file__)),
        "runNamespace": EXPECTED_RUN_NAMESPACE,
        "sourceAdmission": source_admission,
        "sourceCommit": EXPECTED_SOURCE_COMMIT,
        "sourceManifestSha256": SOURCE_MANIFEST_SHA256,
        "trainingControlSha256": TRAINING_CONTROL_SHA256,
        "version": 1,
        "workflowSha256": WORKFLOW_SHA256,
    }


def publish_recovery_receipt(run_root: Path) -> dict[str, object]:
    root = validate_run_root(run_root)
    receipt = root / RECOVERY_RECEIPT_RELATIVE
    sidecar = receipt.with_name(receipt.name + ".sha256")
    if os.path.lexists(os.fspath(receipt)) or os.path.lexists(os.fspath(sidecar)):
        raise FileExistsError("completed-training recovery receipt already exists")
    document = build_recovery_receipt(root)
    if os.path.lexists(os.fspath(receipt)) or os.path.lexists(os.fspath(sidecar)):
        raise FileExistsError("completed-training recovery receipt appeared during verification")
    digest = publish_json_pair_exclusive(receipt, document)
    return {"passed": True, "receipt": str(receipt), "receiptSha256": digest}


def verify_existing_receipt(run_root: Path) -> dict[str, object]:
    root = validate_run_root(run_root)
    receipt = root / RECOVERY_RECEIPT_RELATIVE
    digest = verify_sidecar(receipt)
    stored = read_canonical_json(receipt, "completed-training recovery receipt")
    current = build_recovery_receipt(root)
    if stored != current:
        raise ValueError("completed-training recovery receipt binding drifted")
    return {"passed": True, "receiptSha256": digest}


def describe() -> dict[str, object]:
    return {
        "failureStage": "train",
        "format": "dalmuti-v5-run007-training-recovery-description",
        "originalTerminalPreserved": True,
        "receiptRelative": RECOVERY_RECEIPT_RELATIVE.as_posix(),
        "recoveryReason": "sealed-verifier-expected-scalar-epoch-but-trainer-published-epoch-object",
        "retryOrTrainingMutation": False,
        "runNamespace": EXPECTED_RUN_NAMESPACE,
        "sourceCommit": EXPECTED_SOURCE_COMMIT,
        "version": 1,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--describe", action="store_true")
    commands = value.add_subparsers(dest="command")
    for name in ("publish", "verify-existing"):
        command = commands.add_parser(name)
        command.add_argument("--confirm-run-namespace", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.describe:
        if arguments.command is not None:
            raise ValueError("--describe cannot be combined with a command")
        print(canonical_json_bytes(describe()).decode("ascii"))
        return 0
    if arguments.command is None:
        raise ValueError("publish, verify-existing, or --describe is required")
    if arguments.confirm_run_namespace != EXPECTED_RUN_NAMESPACE:
        raise ValueError("execution confirmation differs from the sealed run namespace")
    run_root = script_run_root(Path(__file__))
    expected = (REMOTE_PARENT / EXPECTED_RUN_NAMESPACE).resolve()
    if os.name != "posix" or run_root.resolve() != expected:
        raise RuntimeError(f"recovery execution is sealed to {expected}")
    result = (
        publish_recovery_receipt(run_root)
        if arguments.command == "publish"
        else verify_existing_receipt(run_root)
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    replacements = {
        "__RUN_NAMESPACE__": repr(run_namespace),
        "__SOURCE_COMMIT__": repr(source_commit),
        "__TRAINING_CONTROL_SHA256__": repr(training_control_sha256),
        "__DURABLE_TRAINING_LAUNCHER_SHA256__": repr(
            durable_training_launcher_sha256
        ),
        "__SOURCE_MANIFEST_SHA256__": repr(source_manifest_sha256),
        "__WORKFLOW_SHA256__": repr(workflow_sha256),
        "__RECOVERY_RECEIPT_RELATIVE__": repr(
            RECOVERY_RECEIPT_RELATIVE.as_posix()
        ),
        "__EXPECTED_PREFLIGHT_SHA256__": repr(
            FAILURE_REFERENCE_SHA256["gpu-memory.json"]
        ),
        "__EXPECTED_TRAINING_MANIFEST_SHA256__": repr(
            FAILURE_REFERENCE_SHA256["manifest.json"]
        ),
        "__EXPECTED_TRAINING_RESULT_SHA256__": repr(
            FAILURE_REFERENCE_SHA256["result.json"]
        ),
        "__EXPECTED_MODEL_PAIR_SHA256__": repr(
            FAILURE_REFERENCE_SHA256["model-pair.json"]
        ),
        "__EXPECTED_TRAINING_ATTEMPT_SHA256__": repr(
            FAILURE_REFERENCE_SHA256["training-attempt.json"]
        ),
        "__EXPECTED_DURABLE_TRAIN_INTENT_SHA256__": repr(
            EXPECTED_DURABLE_TRAIN_INTENT_SHA256
        ),
        "__EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256__": repr(
            EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256
        ),
        "__EXPECTED_DURABLE_TRAIN_PROCESS_SHA256__": repr(
            EXPECTED_DURABLE_TRAIN_PROCESS_SHA256
        ),
        "__EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256__": repr(
            FAILURE_REFERENCE_SHA256["terminal.json"]
        ),
        "__EXPECTED_DURABLE_TRAIN_STDERR_SHA256__": repr(
            FAILURE_REFERENCE_SHA256["stage-stderr.log"]
        ),
        "__EXPECTED_EMPTY_SHA256__": repr(EMPTY_SHA256),
        "__EXPECTED_EPOCH_SHA256__": repr(str(reference["epochSha256"])),
        "__EXPECTED_HARD_GATES_SHA256__": repr(
            str(reference["hardGatesSha256"])
        ),
        "__EXPECTED_INITIAL_BINDINGS_SHA256__": repr(
            str(reference["initialBehaviorBindingsSha256"])
        ),
        "__EXPECTED_EPOCH_ONLY_PROOF_SHA256__": repr(
            str(reference["epochOnlyProofSha256"])
        ),
        "__EXPECTED_EPOCH_ONLY_PROBE_SHA256__": repr(
            EPOCH_ONLY_PROBE_SHA256
        ),
        "__EXPECTED_FAILURE_STDERR_B64__": repr(str(reference["stderrBase64"])),
        "__EXPECTED_MANIFEST_JSON__": repr(
            canonical_json_bytes(manifest).decode("ascii")
        ),
        "__EXPECTED_MODEL_PAIR_JSON__": repr(
            canonical_json_bytes(model_pair).decode("ascii")
        ),
        "__EXPECTED_TRAINING_ATTEMPT_JSON__": repr(
            canonical_json_bytes(training_attempt).decode("ascii")
        ),
        "__EXPECTED_TERMINAL_JSON__": repr(
            canonical_json_bytes(terminal).decode("ascii")
        ),
        "__EXPECTED_TRAINING_CONFIG_JSON__": repr(
            canonical_json_bytes(result["config"]).decode("ascii")
        ),
        "__EXPECTED_PREFLIGHT_CONFIG_JSON__": repr(
            canonical_json_bytes(preflight["config"]).decode("ascii")
        ),
        "__EXPECTED_GPU_ADMISSION_JSON__": repr(
            canonical_json_bytes(result["gpuMemoryPreflight"]).decode("ascii")
        ),
        "__EXPECTED_HARD_GATES_JSON__": repr(
            canonical_json_bytes(result["hardGates"]).decode("ascii")
        ),
        "__EXPECTED_INITIAL_BINDINGS_JSON__": repr(
            canonical_json_bytes(result["initialBehaviorBindings"]).decode("ascii")
        ),
        "__EXPECTED_OUTPUT_PAIR_JSON__": repr(
            canonical_json_bytes(result["outputModelPair"]).decode("ascii")
        ),
    }
    source = template
    for marker, value in replacements.items():
        if source.count(marker) != 1:
            raise ValueError(f"recovery verifier marker count drifted: {marker}")
        source = source.replace(marker, value)
    if "__" + "RUN_NAMESPACE__" in source:
        raise ValueError("recovery verifier contains an unresolved marker")
    ast.parse(source)
    if any(value in source for value in ("recover-promotion-lock", "retry-training")):
        raise ValueError("recovery verifier exposes a retry/recovery mutation surface")
    return source.encode("utf-8")


def _evaluation_control_header(
    *,
    run_namespace: str,
    source_commit: str,
    training_control_sha256: str,
    recovery_verifier_sha256: str,
    remote_parent: str,
) -> str:
    return f'''from __future__ import annotations

"""Generated fail-closed evaluation/promotion control for {run_namespace}.

This file has no training, retry, lock-recovery, network, or deployment surface.
It delegates only the immutable evaluation/promotion operations extracted from
the audited run-004 control.
"""

import argparse
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Mapping, Sequence

from control_common import (
    RUN_NAME,
    SOURCE_COMMIT,
    canonical_json_bytes,
    read_canonical_json,
    script_run_root,
    sha256_file,
    verify_sidecar,
    verify_source_blob_admission,
)


EXPECTED_RUN_NAMESPACE = {run_namespace!r}
EXPECTED_SOURCE_COMMIT = {source_commit!r}
TRAINING_CONTROL_NAME = {TRAINING_CONTROL_NAME!r}
TRAINING_CONTROL_SHA256 = {training_control_sha256!r}
RECOVERY_VERIFIER_SHA256 = {recovery_verifier_sha256!r}
EXPECTED_TRAINING_MANIFEST_SHA256 = {FAILURE_REFERENCE_SHA256["manifest.json"]!r}
EXPECTED_TRAINING_RESULT_SHA256 = {FAILURE_REFERENCE_SHA256["result.json"]!r}
EXPECTED_TRAINING_ATTEMPT_SHA256 = {FAILURE_REFERENCE_SHA256["training-attempt.json"]!r}
EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256 = {FAILURE_REFERENCE_SHA256["terminal.json"]!r}
EXPECTED_DURABLE_TRAIN_STDERR_SHA256 = {FAILURE_REFERENCE_SHA256["stage-stderr.log"]!r}
EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256 = {EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256!r}
EXPECTED_DURABLE_TRAIN_INTENT_SHA256 = {EXPECTED_DURABLE_TRAIN_INTENT_SHA256!r}
EXPECTED_DURABLE_TRAIN_PROCESS_SHA256 = {EXPECTED_DURABLE_TRAIN_PROCESS_SHA256!r}
EXPECTED_EMPTY_SHA256 = {EMPTY_SHA256!r}
EXPECTED_EPOCH_ONLY_PROOF_SHA256 = {EPOCH_ONLY_PROOF_SHA256!r}
EXPECTED_EPOCH_ONLY_PROBE_SHA256 = {EPOCH_ONLY_PROBE_SHA256!r}
DEVICE = "cuda:0"
EVALUATION_LANES = 32
TRAINING_RELATIVE = Path("training")
RECOVERY_RECEIPT_RELATIVE = Path({RECOVERY_RECEIPT_RELATIVE.as_posix()!r})
EXPECTED_RUNTIME_PYTHON_PATH_COUNT = 143
EXPECTED_CHECKED_SOURCE_PATH_COUNT = 154
EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256 = (
    "fdf1aebbdf29606997f4b4ff4965a73f3240c2b2a4bff445cfdc48698442117f"
)
EXPECTED_SOURCE_UNION_INVENTORY_SHA256 = (
    "7208de2a882c4941675dd8a5e9036f44095d6b5857bb7dd18889efcf5054d2ab"
)
REMOTE_PARENT = Path({remote_parent!r})
SHARED_REGISTRY_NAME = "v5-promotion-registry"
RTX_3080 = re.compile(r"(?:geforce\\s+)?rtx\\s*3080", re.IGNORECASE)
EVALUATION_STAGES = {EVALUATION_STAGES!r}

if RUN_NAME != EXPECTED_RUN_NAMESPACE or SOURCE_COMMIT != EXPECTED_SOURCE_COMMIT:
    raise RuntimeError("generated evaluation control identity differs from control_common")


def _validated_source_admission(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {{
            "checkedPathCount",
            "passed",
            "runtimePythonPathCount",
            "runtimePythonSourceInventorySha256",
            "sourceCommit",
            "sourceUnionInventorySha256",
        }}
        or value.get("passed") is not True
        or value.get("sourceCommit") != EXPECTED_SOURCE_COMMIT
        or value.get("runtimePythonPathCount")
        != EXPECTED_RUNTIME_PYTHON_PATH_COUNT
        or value.get("checkedPathCount") != EXPECTED_CHECKED_SOURCE_PATH_COUNT
        or value.get("runtimePythonSourceInventorySha256")
        != EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256
        or value.get("sourceUnionInventorySha256")
        != EXPECTED_SOURCE_UNION_INVENTORY_SHA256
    ):
        raise ValueError("full source-blob admission identity drifted")
    return dict(value)


def _source_modules(run_root: Path) -> dict[str, object]:
    source_admission = _validated_source_admission(
        verify_source_blob_admission(run_root)
    )
    source = run_root / "source-checkout" / "gpu-training"
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError("sealed gpu-training source checkout is absent")
    source_text = str(source)
    if source_text not in sys.path:
        # Keep the sealed checkout available for v5_workflow's deliberately
        # delayed v5_collection_plan/v5_dataset/v5_low_disk_stage imports.
        sys.path.insert(0, source_text)
    modules = {{
        name: importlib.import_module(name)
        for name in (
            "torch",
            "v5_gpu_memory_preflight",
            "v5_promotion",
            "v5_train",
            "v5_workflow",
        )
    }}
    sealed = source.resolve()
    for name in (
        "v5_gpu_memory_preflight",
        "v5_promotion",
        "v5_train",
        "v5_workflow",
    ):
        module_path = Path(str(modules[name].__file__)).resolve()
        if sealed not in module_path.parents:
            raise ImportError(f"{{name}} resolved outside the sealed checkout")
    modules["_sourceAdmission"] = dict(source_admission)
    return modules


def _paths(run_root: Path) -> dict[str, Path]:
    return {{
        "candidate": run_root / TRAINING_RELATIVE / "actor-bundle",
        "recoveryReceipt": run_root / RECOVERY_RECEIPT_RELATIVE,
        "registry": run_root.parent / SHARED_REGISTRY_NAME,
        "sourceCheckout": run_root / "source-checkout",
        "training": run_root / TRAINING_RELATIVE,
    }}


def _require_confirmation(value: str) -> None:
    if value != EXPECTED_RUN_NAMESPACE:
        raise ValueError(
            f"execution confirmation must equal {{EXPECTED_RUN_NAMESPACE}}"
        )


def _require_remote_rtx_3080(
    run_root: Path, modules: Mapping[str, object]
) -> str:
    expected = (REMOTE_PARENT / EXPECTED_RUN_NAMESPACE).resolve()
    if os.name != "posix" or run_root.resolve() != expected:
        raise RuntimeError(
            f"production execution is sealed to the remote run root {{expected}}"
        )
    torch = modules["torch"]
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        raise RuntimeError("evaluation requires an available CUDA device")
    requested = torch.device(DEVICE)  # type: ignore[attr-defined]
    name = str(torch.cuda.get_device_properties(requested).name)  # type: ignore[attr-defined]
    if RTX_3080.search(name) is None:
        raise RuntimeError(f"evaluation is calibrated for RTX 3080, got {{name}}")
    return name


def _verified_training_inventory(
    training_path: Path,
) -> tuple[dict[str, object], str, str]:
    supplied = training_path
    if supplied.is_symlink():
        raise ValueError("training output root must not be a symlink")
    training = supplied.resolve(strict=True)
    if not training.is_dir():
        raise ValueError("training output root is not a directory")
    manifest_path = training / "manifest.json"
    manifest_sha = verify_sidecar(manifest_path)
    manifest = read_canonical_json(manifest_path, "training manifest")
    inventory = manifest.get("files")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("training artifact inventory is absent")
    expected_files = {{"manifest.json", "manifest.json.sha256", *inventory.keys()}}
    if not all(isinstance(value, str) for value in expected_files):
        raise ValueError("training artifact inventory path is invalid")
    actual_files: set[str] = set()
    for path in training.rglob("*"):
        if path.is_symlink():
            raise ValueError("training output contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(training).as_posix())
    if actual_files != expected_files:
        raise ValueError("training artifact inventory file set drifted")
    normalized: dict[str, dict[str, object]] = {{}}
    for relative, record in inventory.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("training artifact record is invalid")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in ("", ".", "..") for part in relative_path.parts)
            or set(record) != {{"bytes", "sha256"}}
            or type(record.get("bytes")) is not int
            or int(record["bytes"]) < 0
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{{64}}", str(record["sha256"])) is None
        ):
            raise ValueError("training artifact record shape drifted")
        unresolved = training.joinpath(*relative_path.parts)
        if unresolved.is_symlink():
            raise ValueError("training artifact must not be a symlink")
        path = unresolved.resolve(strict=True)
        if training not in path.parents or not path.is_file():
            raise ValueError("training artifact escaped its immutable root")
        if (
            record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"training artifact hash drifted: {{relative}}")
        normalized[relative] = dict(record)
    inventory_sha = hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
    return manifest, manifest_sha, inventory_sha


def _candidate_receipt_document(
    run_root: Path, modules: Mapping[str, object]
) -> dict[str, object]:
    paths = _paths(run_root)
    manifest, manifest_sha, inventory_sha = _verified_training_inventory(
        paths["training"]
    )
    pair = modules["v5_train"].verify_v5_model_pair(paths["training"])  # type: ignore[attr-defined]
    candidate = paths["candidate"].resolve(strict=True)
    digests = modules["v5_train"].v5_actor_bundle_digests(candidate)  # type: ignore[attr-defined]
    result = read_canonical_json(paths["training"] / "result.json", "training result")
    source_admission = _validated_source_admission(
        modules.get("_sourceAdmission")
    )
    if (
        not isinstance(source_admission, Mapping)
        or source_admission.get("passed") is not True
        or source_admission.get("sourceCommit") != EXPECTED_SOURCE_COMMIT
        or result.get("outputModelPair") != pair
        or result.get("hardGates", {{}}).get("passed") is not True
        or digests.get("actorSha256") != pair.get("actorSha256")
    ):
        raise ValueError("candidate receipt inputs are incomplete or drifted")
    return {{
        "candidateActorSha256": pair["actorSha256"],
        "candidatePairId": pair["pairId"],
        "candidatePairManifestSha256": pair["pairManifestSha256"],
        "datasetIdentitySha256": result["datasetIdentitySha256"],
        "manifestSha256": manifest_sha,
        "resultSha256": sha256_file(paths["training"] / "result.json"),
        "sourceAdmission": dict(source_admission),
        "sourceManifestSha256": verify_sidecar(
            run_root / "source-seal/manifest.json"
        ),
        "trainingAttemptSha256": verify_sidecar(
            run_root / "training-execution/attempt-001.json"
        ),
        "trainingArtifactFileCount": len(manifest["files"]),
        "trainingArtifactInventorySha256": inventory_sha,
        "trainingControlSha256": TRAINING_CONTROL_SHA256,
        "workflowSha256": verify_sidecar(run_root / "workflow.json"),
    }}


def _current_completed_training_failure(run_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    durable_root = run_root / "durable-training"
    train_root = durable_root / "train"
    attempt = train_root / "attempt-001"
    if (
        any(path.is_symlink() for path in (durable_root, train_root, attempt))
        or not attempt.is_dir()
        or {{path.name for path in train_root.iterdir()}} != {{"attempt-001"}}
        or os.path.lexists(os.fspath(durable_root / "verify-training"))
    ):
        raise ValueError("completed-training durable attempt topology drifted")
    expected_files = {{
        "intent.json",
        "intent.json.sha256",
        "launch.json",
        "launch.json.sha256",
        "process.json",
        "process.json.sha256",
        "stage-stderr.log",
        "stage-stdout.log",
        "terminal.json",
        "terminal.json.sha256",
        "worker-stderr.log",
        "worker-stdout.log",
    }}
    actual_files: set[str] = set()
    for path in attempt.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("completed-training durable attempt contains an unsafe entry")
        actual_files.add(path.name)
    if actual_files != expected_files:
        raise ValueError("completed-training durable attempt file set drifted")
    exact_json = {{
        "intentSha256": ("intent.json", EXPECTED_DURABLE_TRAIN_INTENT_SHA256),
        "launchSha256": ("launch.json", EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256),
        "processSha256": ("process.json", EXPECTED_DURABLE_TRAIN_PROCESS_SHA256),
        "terminalSha256": ("terminal.json", EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256),
    }}
    durable: dict[str, object] = {{}}
    for key, (name, expected) in exact_json.items():
        digest = verify_sidecar(attempt / name)
        if digest != expected:
            raise ValueError(f"completed-training durable record drifted: {{name}}")
        durable[key] = digest
    logs = {{
        "stderrSha256": ("stage-stderr.log", EXPECTED_DURABLE_TRAIN_STDERR_SHA256, 1006),
        "stdoutSha256": ("stage-stdout.log", EXPECTED_EMPTY_SHA256, 0),
    }}
    for key, (name, expected, expected_bytes) in logs.items():
        path = attempt / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected
        ):
            raise ValueError(f"completed-training durable log drifted: {{name}}")
        durable[key] = expected
    for name in ("worker-stderr.log", "worker-stdout.log"):
        path = attempt / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != 0 or sha256_file(path) != EXPECTED_EMPTY_SHA256:
            raise ValueError(f"completed-training worker log drifted: {{name}}")
    topology = {{
        "attemptCount": 1,
        "attemptFileCount": len(actual_files),
        "verifyTrainingAbsent": True,
    }}
    return durable, topology


def _verified_candidate_receipt(
    run_root: Path, modules: Mapping[str, object]
) -> tuple[dict[str, object], str]:
    path = _paths(run_root)["recoveryReceipt"]
    digest = verify_sidecar(path)
    receipt = read_canonical_json(path, "completed-training recovery receipt")
    binding = _candidate_receipt_document(run_root, modules)
    durable = receipt.get("durableFailure")
    topology = receipt.get("durableTopology")
    current_durable, current_topology = _current_completed_training_failure(run_root)
    proof = receipt.get("epochOnlyVerifierProof")
    if (
        receipt.get("format")
        != "dalmuti-v5-run007-completed-training-recovery-verification"
        or receipt.get("version") != 1
        or receipt.get("passed") is not True
        or receipt.get("runNamespace") != EXPECTED_RUN_NAMESPACE
        or receipt.get("sourceCommit") != EXPECTED_SOURCE_COMMIT
        or receipt.get("recoveryVerifierSha256") != RECOVERY_VERIFIER_SHA256
        or any(receipt.get(name) != value for name, value in binding.items())
        or receipt.get("manifestSha256") != EXPECTED_TRAINING_MANIFEST_SHA256
        or receipt.get("resultSha256") != EXPECTED_TRAINING_RESULT_SHA256
        or receipt.get("trainingAttemptSha256") != EXPECTED_TRAINING_ATTEMPT_SHA256
        or not isinstance(durable, Mapping)
        or durable != current_durable
        or durable.get("terminalSha256")
        != EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256
        or durable.get("launchSha256") != EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256
        or durable.get("stderrSha256") != EXPECTED_DURABLE_TRAIN_STDERR_SHA256
        or topology != current_topology
        or not isinstance(proof, Mapping)
        or proof.get("proofSha256") != EXPECTED_EPOCH_ONLY_PROOF_SHA256
        or proof.get("probeScriptSha256") != EXPECTED_EPOCH_ONLY_PROBE_SHA256
        or proof.get("patchCalls") != 1
        or proof.get("remoteMutationCount") != 0
    ):
        raise ValueError("completed-training recovery receipt binding drifted")
    return receipt, digest


def _candidate(
    run_root: Path,
    modules: Mapping[str, object],
) -> Path:
    paths = _paths(run_root)
    receipt, _ = _verified_candidate_receipt(run_root, modules)
    pair = modules["v5_train"].verify_v5_model_pair(paths["training"])  # type: ignore[attr-defined]
    candidate = paths["candidate"].resolve(strict=True)
    digests = modules["v5_train"].v5_actor_bundle_digests(candidate)  # type: ignore[attr-defined]
    if (
        pair.get("pairId") != receipt.get("candidatePairId")
        or pair.get("actorSha256") != receipt.get("candidateActorSha256")
        or pair.get("pairManifestSha256")
        != receipt.get("candidatePairManifestSha256")
        or digests.get("actorSha256") != receipt.get("candidateActorSha256")
    ):
        raise ValueError("candidate changed after completed-training recovery verification")
    return candidate

'''


def _evaluation_control_footer() -> str:
    return '''

def describe() -> dict[str, object]:
    return {
        "candidateVerification": "immutable-completed-training-recovery-receipt",
        "checkedSourcePathCount": EXPECTED_CHECKED_SOURCE_PATH_COUNT,
        "device": DEVICE,
        "evaluationLanes": EVALUATION_LANES,
        "format": "dalmuti-v5-generated-evaluation-control-description",
        "fullSourceBlobAdmissionRequired": True,
        "recoveryCommandsExposed": False,
        "recoveryVerifierSha256": RECOVERY_VERIFIER_SHA256,
        "runNamespace": EXPECTED_RUN_NAMESPACE,
        "runtimePythonPathCount": EXPECTED_RUNTIME_PYTHON_PATH_COUNT,
        "runtimePythonSourceInventorySha256": EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256,
        "sourceCommit": EXPECTED_SOURCE_COMMIT,
        "sourceUnionInventorySha256": EXPECTED_SOURCE_UNION_INVENTORY_SHA256,
        "stages": list(EVALUATION_STAGES),
        "trainingControlSha256": TRAINING_CONTROL_SHA256,
        "version": 1,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    commands = parser.add_subparsers(dest="stage")

    def guarded(name: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name)
        command.add_argument("--confirm-run-namespace", required=True)
        return command

    guarded("reserve-screening")
    screening = guarded("screening")
    screening.add_argument("--reservation", required=True)
    reserve_cert = guarded("reserve-certification")
    reserve_cert.add_argument("--screening-reservation", required=True)
    reserve_cert.add_argument("--screening-report", required=True)
    for label in ("a", "b"):
        cert = guarded(f"certification-{label}")
        cert.add_argument("--reservation", required=True)
        cert.add_argument("--screening-report", required=True)
    reserve = guarded("reserve-final")
    reserve.add_argument("--certification-report", action="append", required=True)
    reserve.add_argument("--final-shard-count", type=int, required=True)
    claim = guarded("claim-final")
    claim.add_argument("--plan", required=True)
    claim.add_argument("--shard-count", type=int, required=True)
    claim.add_argument("--shard-index", type=int, required=True)
    final = guarded("final")
    final.add_argument("--plan", required=True)
    final.add_argument("--claim", required=True)
    final.add_argument("--shard-count", type=int, required=True)
    final.add_argument("--shard-index", type=int, required=True)
    merge = guarded("merge-final")
    merge.add_argument("--plan", required=True)
    merge.add_argument("--report", action="append", required=True)
    approve = guarded("approve-final")
    approve.add_argument("--plan", required=True)
    approve.add_argument("--final-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    if arguments.describe:
        if arguments.stage is not None:
            raise ValueError("--describe may not be combined with an execution stage")
        print(canonical_json_bytes(describe()).decode("ascii"))
        return 0
    if arguments.stage is None:
        raise ValueError("one explicit evaluation stage or --describe is required")
    _require_confirmation(arguments.confirm_run_namespace)
    run_root = script_run_root(Path(__file__))
    if arguments.stage == "reserve-screening":
        result = reserve_screening(run_root)
    elif arguments.stage == "screening":
        result = evaluate_screening(run_root, arguments.reservation, None)
    elif arguments.stage == "reserve-certification":
        result = reserve_certification(
            run_root, arguments.screening_reservation, arguments.screening_report
        )
    elif arguments.stage in {"certification-a", "certification-b"}:
        result = evaluate_certification(
            run_root,
            arguments.stage[-1],
            arguments.reservation,
            arguments.screening_report,
            None,
        )
    elif arguments.stage == "reserve-final":
        result = reserve_final(
            run_root, arguments.certification_report, arguments.final_shard_count
        )
    elif arguments.stage == "claim-final":
        result = claim_final(
            run_root, arguments.plan, arguments.shard_count, arguments.shard_index
        )
    elif arguments.stage == "final":
        result = evaluate_final(
            run_root,
            arguments.plan,
            arguments.claim,
            arguments.shard_count,
            arguments.shard_index,
            None,
        )
    elif arguments.stage == "merge-final":
        result = merge_final_reports(run_root, arguments.plan, arguments.report)
    elif arguments.stage == "approve-final":
        result = approve_final(run_root, arguments.plan, arguments.final_report)
    else:  # pragma: no cover
        raise AssertionError("unhandled evaluation stage")
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _render_evaluation_control(
    template: str,
    *,
    run_namespace: str,
    source_commit: str,
    training_control_sha256: str,
    recovery_verifier_sha256: str,
    remote_parent: str,
) -> bytes:
    extracted = _extract_run004_functions(template)
    source = (
        _evaluation_control_header(
            run_namespace=run_namespace,
            source_commit=source_commit,
            training_control_sha256=training_control_sha256,
            recovery_verifier_sha256=recovery_verifier_sha256,
            remote_parent=remote_parent,
        )
        + extracted
        + _evaluation_control_footer()
    )
    ast.parse(source)
    if any(
        token in source
        for token in ("recover-promotion-lock", "--recover-crashed-attempt-reason")
    ):
        raise ValueError("generated evaluation control exposes a recovery command")
    return source.encode("utf-8")


def _render_launcher(
    template: str,
    *,
    run_namespace: str,
    source_commit: str,
    evaluation_control_sha256: str,
    training_control_sha256: str,
    control_common_sha256: str,
    source_manifest_sha256: str,
    source_inventory_sha256: str,
    low_disk_sha256: str,
) -> bytes:
    namespace_marker = "__DALMUTI_GENERATED_RUN_NAMESPACE__"
    if namespace_marker in template:
        raise ValueError("launcher template contains the namespace marker")
    source = template.replace(RUN006_NAMESPACE, namespace_marker)
    source = source.replace("run-006", "run-007").replace("run006", "run007")
    source = source.replace(namespace_marker, run_namespace)
    source = _replace_once(
        source,
        f'RUN_NAMESPACE = "{run_namespace}"\n',
        f'RUN_NAMESPACE = "{run_namespace}"\nSOURCE_COMMIT = "{source_commit}"\n',
        "launcher source commit",
    )
    source = _replace_regex_once(
        source,
        r'^CONTROL_NAME = "[^"]+"$',
        f'CONTROL_NAME = "{EVALUATION_CONTROL_NAME}"',
        "launcher control name",
    )
    source = _replace_regex_once(
        source,
        r'^LAUNCHER_NAME = "[^"]+"$',
        f'LAUNCHER_NAME = "{LAUNCHER_NAME}"',
        "launcher side-by-side filename",
    )
    source = _replace_regex_once(
        source,
        r'^CONTROL_SHA256 = "[0-9a-f]{64}"$',
        f'CONTROL_SHA256 = "{evaluation_control_sha256}"',
        "launcher control SHA",
    )
    source = _replace_regex_once(
        source,
        r'^ATTEMPT_RELATIVE = Path\("[^"]+"\)$',
        f'ATTEMPT_RELATIVE = Path("{EVALUATION_ATTEMPT_RELATIVE}")',
        "launcher fresh durable attempt namespace",
    )
    source = _replace_regex_once(
        source,
        r'^CONTROL_HELPER_SHA256 = \{\n(?:    .*\n)+?\}$',
        "CONTROL_HELPER_SHA256 = {\n"
        f'    "control_common.py": "{control_common_sha256}",\n'
        f'    "{TRAINING_CONTROL_NAME}": "{training_control_sha256}",\n'
        "}",
        "launcher helper SHAs",
    )
    source = _replace_regex_once(
        source,
        r'^SOURCE_MANIFEST_SHA256 = "[0-9a-f]{64}"$',
        f'SOURCE_MANIFEST_SHA256 = "{source_manifest_sha256}"',
        "launcher source manifest SHA",
    )
    source = _replace_regex_once(
        source,
        r'^EVALUATION_SOURCE_INVENTORY_SHA256 = \(\n    "[0-9a-f]{64}"\n\)$',
        f'EVALUATION_SOURCE_INVENTORY_SHA256 = "{source_inventory_sha256}"',
        "launcher evaluation inventory SHA",
    )
    source = _replace_regex_once(
        source,
        r'^V5_LOW_DISK_STAGE_SHA256 = \(\n    "[0-9a-f]{64}"\n\)$',
        f'V5_LOW_DISK_STAGE_SHA256 = "{low_disk_sha256}"',
        "launcher low-disk SHA",
    )
    source = _replace_once(
        source,
        '    if manifest_sha != SOURCE_MANIFEST_SHA256:\n'
        '        raise ValueError("source seal manifest SHA-256 drifted")\n'
        '    evaluation = manifest.get("evaluationSource")\n',
        '    if manifest_sha != SOURCE_MANIFEST_SHA256:\n'
        '        raise ValueError("source seal manifest SHA-256 drifted")\n'
        '    if manifest.get("sourceCommit") != SOURCE_COMMIT:\n'
        '        raise ValueError("source seal commit drifted")\n'
        '    evaluation = manifest.get("evaluationSource")\n',
        "launcher manifest commit check",
    )
    source = _replace_once(
        source,
        '        not isinstance(evaluation, dict)\n'
        '        or evaluation.get("sourceInventorySha256") != EVALUATION_SOURCE_INVENTORY_SHA256\n',
        '        not isinstance(evaluation, dict)\n'
        '        or evaluation.get("sourceCommit") != SOURCE_COMMIT\n'
        '        or evaluation.get("sourceInventorySha256") != EVALUATION_SOURCE_INVENTORY_SHA256\n',
        "launcher evaluation commit check",
    )
    source = _replace_once(
        source,
        '        "sourceManifestSha256": SOURCE_MANIFEST_SHA256,\n'
        '        "version": 1,\n'
        '    }\n\n\ndef _blank_arguments',
        '        "sourceCommit": SOURCE_COMMIT,\n'
        '        "sourceManifestSha256": SOURCE_MANIFEST_SHA256,\n'
        '        "version": 1,\n'
        '    }\n\n\ndef _blank_arguments',
        "launcher description commit",
    )
    source = _replace_once(
        source,
        "\ndef _training_completion_binding(run_root: Path) -> dict[str, str]:\n",
        '''
def _completed_training_epoch(value: object) -> int:
    """Validate the producer's object-shaped epoch without replaying recovery."""

    if not isinstance(value, dict) or value.get("epoch") != 1:
        raise ValueError("completed training epoch binding drifted")
    return 1


def _training_completion_binding(run_root: Path) -> dict[str, str]:
''',
        "launcher completed-training epoch helper",
    )
    source = _replace_once(
        source,
        '        or result.get("epoch") != 1\n',
        '        or _completed_training_epoch(result.get("epoch")) != 1\n',
        "launcher object-shaped epoch admission",
    )
    source = _replace_once(
        source,
        "from typing import Mapping, Sequence\n\n\nRUN_NAMESPACE",
        "from typing import Mapping, Sequence\n\n"
        "from control_common import verify_source_blob_admission\n\n\n"
        "RUN_NAMESPACE",
        "launcher full source admission import",
    )
    source = _replace_once(
        source,
        "NOFILE_SOFT_LIMIT = 65536\n",
        "NOFILE_SOFT_LIMIT = 65536\n"
        f"MINIMUM_EVALUATION_FREE_BYTES = {MINIMUM_EVALUATION_FREE_BYTES}\n"
        f"EXPECTED_RUNTIME_PYTHON_PATH_COUNT = {EXPECTED_RUNTIME_PYTHON_PATH_COUNT}\n"
        f"EXPECTED_CHECKED_SOURCE_PATH_COUNT = {EXPECTED_CHECKED_SOURCE_PATH_COUNT}\n"
        "EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256 = "
        f'"{EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256}"\n'
        "EXPECTED_SOURCE_UNION_INVENTORY_SHA256 = "
        f'"{EXPECTED_SOURCE_UNION_INVENTORY_SHA256}"\n',
        "launcher disk minimum",
    )
    source = _replace_once(
        source,
        '    _validate_admission_shape("reserve-screening", {"trainingCompletion": completion})\n',
        '    _validate_admission_shape("reserve-screening", {"trainingCompletion": completion})\n'
        '    if _completed_training_epoch({"epoch": 1}) != 1:\n'
        '        raise AssertionError("object-shaped completed-training epoch was rejected")\n'
        '    for invalid_epoch in (1, {"epoch": 0}):\n'
        '        try:\n'
        '            _completed_training_epoch(invalid_epoch)\n'
        '        except ValueError:\n'
        '            pass\n'
        '        else:\n'
        '            raise AssertionError("invalid completed-training epoch was admitted")\n',
        "launcher completed-training epoch self-test",
    )
    source = _replace_once(
        source,
        '        "tests": 27,\n',
        '        "tests": 29,\n',
        "launcher self-test count",
    )
    disk_helpers = '''

def _validated_source_admission(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "checkedPathCount",
            "passed",
            "runtimePythonPathCount",
            "runtimePythonSourceInventorySha256",
            "sourceCommit",
            "sourceUnionInventorySha256",
        }
        or value.get("passed") is not True
        or value.get("sourceCommit") != SOURCE_COMMIT
        or value.get("runtimePythonPathCount")
        != EXPECTED_RUNTIME_PYTHON_PATH_COUNT
        or value.get("checkedPathCount") != EXPECTED_CHECKED_SOURCE_PATH_COUNT
        or value.get("runtimePythonSourceInventorySha256")
        != EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256
        or value.get("sourceUnionInventorySha256")
        != EXPECTED_SOURCE_UNION_INVENTORY_SHA256
    ):
        raise ValueError("full source-blob admission identity drifted")
    return dict(value)


def _disk_free_admission(run_root: Path) -> dict[str, object]:
    target = REMOTE_REGISTRY.parent
    if target.is_symlink():
        raise ValueError("evaluation registry parent must not be a symlink")
    resolved = target.resolve(strict=True)
    if run_root.parent.resolve(strict=True) != resolved:
        raise ValueError("evaluation registry and run parent identities drifted")
    filesystem = os.statvfs(resolved)
    fragment = int(filesystem.f_frsize or filesystem.f_bsize)
    available = int(filesystem.f_bavail) * fragment
    if available < MINIMUM_EVALUATION_FREE_BYTES:
        raise RuntimeError(
            "evaluation filesystem free space is below the sealed minimum"
        )
    return {
        "availableBytes": available,
        "filesystemDevice": int(resolved.stat().st_dev),
        "minimumFreeBytes": MINIMUM_EVALUATION_FREE_BYTES,
        "path": str(resolved),
    }


def _validate_recorded_disk_admission(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {"availableBytes", "filesystemDevice", "minimumFreeBytes", "path"}
        or type(value.get("availableBytes")) is not int
        or int(value["availableBytes"]) < MINIMUM_EVALUATION_FREE_BYTES
        or type(value.get("filesystemDevice")) is not int
        or int(value["filesystemDevice"]) < 0
        or value.get("minimumFreeBytes") != MINIMUM_EVALUATION_FREE_BYTES
        or value.get("path") != str(REMOTE_REGISTRY.parent)
    ):
        raise ValueError("recorded evaluation disk admission drifted")
    return dict(value)


def _refresh_disk_free_admission(recorded: object, run_root: Path) -> dict[str, object]:
    prior = _validate_recorded_disk_admission(recorded)
    current = _disk_free_admission(run_root)
    if any(
        current[name] != prior[name]
        for name in ("filesystemDevice", "minimumFreeBytes", "path")
    ):
        raise ValueError("evaluation filesystem identity changed after launch")
    return current
'''
    source = _replace_once(
        source,
        "\ndef _nofile_admission() -> dict[str, int]:\n",
        disk_helpers + "\n\ndef _nofile_admission() -> dict[str, int]:\n",
        "launcher fresh disk admission helpers",
    )
    source = _replace_once(
        source,
        "    admission = _pipeline_admission(run_root, launcher, control, stage, bindings)\n"
        "    admission_limit = _nofile_admission()\n",
        "    admission = _pipeline_admission(run_root, launcher, control, stage, bindings)\n"
        "    source_admission = _validated_source_admission(\n"
        "        verify_source_blob_admission(run_root)\n"
        "    )\n"
        "    disk_admission = _disk_free_admission(run_root)\n"
        "    admission_limit = _nofile_admission()\n",
        "launcher pre-attempt source and disk admission",
    )
    source = _replace_once(
        source,
        '        "controlSha256": CONTROL_SHA256,\n'
        '        "environment": selected_environment,\n',
        '        "controlSha256": CONTROL_SHA256,\n'
        '        "diskFreeAdmission": disk_admission,\n'
        '        "environment": selected_environment,\n',
        "launcher disk intent binding",
    )
    source = _replace_once(
        source,
        '            "requiredSoft": NOFILE_SOFT_LIMIT,\n'
        '        },\n'
        '        "runNamespace": RUN_NAMESPACE,\n'
        '        "sourceManifestSha256": SOURCE_MANIFEST_SHA256,\n'
        '        "stage": stage,\n',
        '            "requiredSoft": NOFILE_SOFT_LIMIT,\n'
        '        },\n'
        '        "runNamespace": RUN_NAMESPACE,\n'
        '        "sourceBlobAdmission": source_admission,\n'
        '        "sourceManifestSha256": SOURCE_MANIFEST_SHA256,\n'
        '        "stage": stage,\n',
        "launcher source intent binding",
    )
    source = _replace_once(
        source,
        '        "controlSha256",\n'
        '        "environment",\n',
        '        "controlSha256",\n'
        '        "diskFreeAdmission",\n'
        '        "environment",\n',
        "launcher intent disk key",
    )
    source = _replace_once(
        source,
        '        "runNamespace",\n'
        '        "sourceManifestSha256",\n',
        '        "runNamespace",\n'
        '        "sourceBlobAdmission",\n'
        '        "sourceManifestSha256",\n',
        "launcher intent source key",
    )
    source = _replace_once(
        source,
        "    _validate_admission_shape(stage, intent.get(\"admission\"))\n"
        "    if (\n",
        "    _validate_admission_shape(stage, intent.get(\"admission\"))\n"
        "    _validate_recorded_disk_admission(intent.get(\"diskFreeAdmission\"))\n"
        "    source_admission = _validated_source_admission(\n"
        "        intent.get(\"sourceBlobAdmission\")\n"
        "    )\n"
        "    if (\n",
        "launcher intent admission validators",
    )
    source = _replace_once(
        source,
        '        or intent.get("sourceManifestSha256") != SOURCE_MANIFEST_SHA256\n',
        '        or intent.get("sourceManifestSha256") != SOURCE_MANIFEST_SHA256\n'
        '        or source_admission != intent.get("sourceBlobAdmission")\n',
        "launcher intent source admission shape",
    )
    source = _replace_once(
        source,
        "    if _pipeline_admission(run_root, launcher, control, stage, bindings) != intent[\"admission\"]:\n"
        "        raise ValueError(\"durable evaluation admission changed before supervisor execution\")\n",
        "    if _pipeline_admission(run_root, launcher, control, stage, bindings) != intent[\"admission\"]:\n"
        "        raise ValueError(\"durable evaluation admission changed before supervisor execution\")\n"
        "    if _validated_source_admission(verify_source_blob_admission(run_root)) != intent[\"sourceBlobAdmission\"]:\n"
        "        raise ValueError(\"full source-blob admission changed before supervisor execution\")\n"
        "    _refresh_disk_free_admission(intent[\"diskFreeAdmission\"], run_root)\n",
        "launcher worker source and disk refresh",
    )
    source = _replace_once(
        source,
        "        if _pipeline_admission(run_root, launcher, control, stage, bindings) != intent[\"admission\"]:\n"
        "            raise ValueError(\"durable evaluation admission changed before stage process\")\n"
        "        process = subprocess.Popen(\n",
        "        if _pipeline_admission(run_root, launcher, control, stage, bindings) != intent[\"admission\"]:\n"
        "            raise ValueError(\"durable evaluation admission changed before stage process\")\n"
        "        if _validated_source_admission(verify_source_blob_admission(run_root)) != intent[\"sourceBlobAdmission\"]:\n"
        "            raise ValueError(\"full source-blob admission changed before stage process\")\n"
        "        _refresh_disk_free_admission(intent[\"diskFreeAdmission\"], run_root)\n"
        "        process = subprocess.Popen(\n",
        "launcher stage-process source and disk refresh",
    )
    source = _replace_once(
        source,
        '        "finalShardCount": FINAL_SHARD_COUNT,\n'
        '        "format": "dalmuti-v5-run007-durable-evaluation-launcher-description",\n',
        '        "finalShardCount": FINAL_SHARD_COUNT,\n'
        '        "format": "dalmuti-v5-run007-durable-evaluation-launcher-description",\n'
        '        "checkedSourcePathCount": EXPECTED_CHECKED_SOURCE_PATH_COUNT,\n'
        '        "fullSourceBlobAdmissionRequired": True,\n'
        '        "minimumEvaluationFreeBytes": MINIMUM_EVALUATION_FREE_BYTES,\n'
        '        "runtimePythonPathCount": EXPECTED_RUNTIME_PYTHON_PATH_COUNT,\n'
        '        "runtimePythonSourceInventorySha256": EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256,\n'
        '        "sourceUnionInventorySha256": EXPECTED_SOURCE_UNION_INVENTORY_SHA256,\n',
        "launcher source/disk description",
    )
    ast.parse(source)
    if 'result.get("epoch") != 1' in source:
        raise ValueError("launcher retained the scalar completed-training epoch bug")
    return source.encode("utf-8")


def _render_validator(
    template: str,
    *,
    run_namespace: str,
    source_commit: str,
    evaluation_control_sha256: str,
    recovery_verifier_sha256: str,
    training_control_sha256: str,
    control_common_sha256: str,
    source_manifest_sha256: str,
    source_inventory_sha256: str,
    low_disk_sha256: str,
) -> bytes:
    namespace_marker = "__DALMUTI_GENERATED_RUN_NAMESPACE__"
    if namespace_marker in template:
        raise ValueError("validator template contains the namespace marker")
    source = template.replace(RUN006_NAMESPACE, namespace_marker)
    source = source.replace("run-006", "run-007").replace("run006", "run007")
    source = source.replace(namespace_marker, run_namespace)
    source = _replace_once(
        source,
        f'RUN_NAMESPACE = "{run_namespace}"\n',
        f'RUN_NAMESPACE = "{run_namespace}"\nSOURCE_COMMIT = "{source_commit}"\n'
        f'TRAINING_CONTROL_SHA256 = "{training_control_sha256}"\n'
        f'RECOVERY_VERIFIER_SHA256 = "{recovery_verifier_sha256}"\n'
        f'CONTROL_COMMON_SHA256 = "{control_common_sha256}"\n',
        "validator identities",
    )
    for name, value in (
        ("CONTROL_SHA256", evaluation_control_sha256),
        ("SOURCE_MANIFEST_SHA256", source_manifest_sha256),
        ("SOURCE_INVENTORY_SHA256", source_inventory_sha256),
        ("LOW_DISK_SHA256", low_disk_sha256),
    ):
        source = _replace_regex_once(
            source,
            rf'^{name} = "[0-9a-f]{{64}}"$',
            f'{name} = "{value}"',
            f"validator {name}",
        )
    source = source.replace(
        'run_root / "controls/run_training_iteration_001.py"',
        f'run_root / "controls/{EVALUATION_CONTROL_NAME}"',
    )
    source = _replace_once(
        source,
        '    if sha256_file(control) != CONTROL_SHA256:\n'
        '        raise ValueError("wrapped control SHA-256 drifted")\n'
        '    manifest_path = run_root / "source-seal/manifest.json"\n',
        '    if sha256_file(control) != CONTROL_SHA256:\n'
        '        raise ValueError("wrapped control SHA-256 drifted")\n'
        f'    training_control = run_root / "controls/{TRAINING_CONTROL_NAME}"\n'
        f'    recovery_verifier = run_root / "controls/{RECOVERY_VERIFIER_NAME}"\n'
        '    control_common = run_root / "controls/control_common.py"\n'
        '    if sha256_file(training_control) != TRAINING_CONTROL_SHA256:\n'
        '        raise ValueError("training control SHA-256 drifted")\n'
        '    if sha256_file(recovery_verifier) != RECOVERY_VERIFIER_SHA256:\n'
        '        raise ValueError("completed-training recovery verifier SHA-256 drifted")\n'
        '    if sha256_file(control_common) != CONTROL_COMMON_SHA256:\n'
        '        raise ValueError("control_common SHA-256 drifted")\n'
        '    manifest_path = run_root / "source-seal/manifest.json"\n',
        "validator control dependencies",
    )
    source = _replace_once(
        source,
        '        or self_test.get("tests") != 27\n',
        '        or self_test.get("tests") != 29\n',
        "validator launcher self-test count",
    )
    source = _replace_once(
        source,
        '    if launcher.name != "launch_durable_evaluation_pipeline.py":\n',
        f'    if launcher.name != "{LAUNCHER_NAME}":\n',
        "validator side-by-side launcher filename",
    )
    source = _replace_once(
        source,
        '        "ATTEMPT_NAME = \\"attempt-001\\"",\n',
        '        "ATTEMPT_NAME = \\"attempt-001\\"",\n'
        f'        \'ATTEMPT_RELATIVE = Path("{EVALUATION_ATTEMPT_RELATIVE}")\',\n',
        "validator fresh durable attempt namespace",
    )
    source = _replace_once(
        source,
        '        description.get("runNamespace") != RUN_NAMESPACE\n',
        '        description.get("runNamespace") != RUN_NAMESPACE\n'
        f'        or description.get("attemptRoot") != "{EVALUATION_ATTEMPT_RELATIVE}"\n',
        "validator dynamic durable attempt namespace",
    )
    source = _replace_once(
        source,
        '    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))\n'
        '    evaluation = manifest.get("evaluationSource")\n',
        '    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))\n'
        '    if manifest.get("sourceCommit") != SOURCE_COMMIT:\n'
        '        raise ValueError("source seal commit drifted")\n'
        '    evaluation = manifest.get("evaluationSource")\n',
        "validator manifest commit",
    )
    source = _replace_once(
        source,
        '        not isinstance(evaluation, dict)\n'
        '        or evaluation.get("sourceInventorySha256") != SOURCE_INVENTORY_SHA256\n',
        '        not isinstance(evaluation, dict)\n'
        '        or evaluation.get("sourceCommit") != SOURCE_COMMIT\n'
        '        or evaluation.get("sourceInventorySha256") != SOURCE_INVENTORY_SHA256\n',
        "validator evaluation commit",
    )
    control_surface = '''

def _validate_control_surface(run_root: Path) -> dict[str, object]:
    control = run_root / "controls/run_evaluation_iteration_001.py"
    source = control.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {"paramiko", "requests", "socket", "urllib", "http", "ftplib"}
    imported: set[str] = set()
    forbidden_calls = {
        "os.system",
        "shutil.rmtree",
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and _call_name(node) in forbidden_calls:
            raise ValueError(f"evaluation control execution surface is forbidden: {_call_name(node)}")
    if imported & forbidden_imports:
        raise ValueError("evaluation control contains a network import")
    forbidden_fragments = (
        "recover-promotion-lock",
        "--recover-crashed-attempt-reason",
        "verify_training_output",
        "establish_receipt",
        "publish_json_pair_exclusive",
    )
    if any(value in source for value in forbidden_fragments):
        raise ValueError("evaluation control exposes retry/recovery")
    required_fragments = (
        "verify_source_blob_admission(run_root)",
        '"v5_gpu_memory_preflight"',
        "RECOVERY_RECEIPT_RELATIVE",
        "RECOVERY_VERIFIER_SHA256",
        "completed-training recovery receipt binding drifted",
        "EXPECTED_TRAINING_MANIFEST_SHA256",
        "EXPECTED_EPOCH_ONLY_PROOF_SHA256",
        "EXPECTED_RUNTIME_PYTHON_PATH_COUNT = 143",
        "EXPECTED_CHECKED_SOURCE_PATH_COUNT = 154",
        "fdf1aebbdf29606997f4b4ff4965a73f3240c2b2a4bff445cfdc48698442117f",
        "7208de2a882c4941675dd8a5e9036f44095d6b5857bb7dd18889efcf5054d2ab",
    )
    missing = [value for value in required_fragments if value not in source]
    if missing:
        raise ValueError(f"evaluation control admission fragments are missing: {missing}")
    completed = subprocess.run(
        [sys.executable, "-B", str(control), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.stderr:
        raise ValueError("evaluation control --help wrote stderr")
    stages = [
        "reserve-screening",
        "screening",
        "reserve-certification",
        "certification-a",
        "certification-b",
        "reserve-final",
        "claim-final",
        "final",
        "merge-final",
        "approve-final",
    ]
    if any(stage not in completed.stdout for stage in stages):
        raise ValueError("evaluation control help omits a required stage")
    description = _run_json(control, "--describe")
    if (
        description.get("runNamespace") != RUN_NAMESPACE
        or description.get("sourceCommit") != SOURCE_COMMIT
        or description.get("trainingControlSha256") != TRAINING_CONTROL_SHA256
        or description.get("stages") != stages
        or description.get("evaluationLanes") != 32
        or description.get("candidateVerification")
        != "immutable-completed-training-recovery-receipt"
        or description.get("recoveryVerifierSha256") != RECOVERY_VERIFIER_SHA256
        or description.get("fullSourceBlobAdmissionRequired") is not True
        or description.get("runtimePythonPathCount") != 143
        or description.get("checkedSourcePathCount") != 154
        or description.get("runtimePythonSourceInventorySha256")
        != "fdf1aebbdf29606997f4b4ff4965a73f3240c2b2a4bff445cfdc48698442117f"
        or description.get("sourceUnionInventorySha256")
        != "7208de2a882c4941675dd8a5e9036f44095d6b5857bb7dd18889efcf5054d2ab"
        or description.get("recoveryCommandsExposed") is not False
    ):
        raise ValueError("evaluation control description drifted")
    return {"commandCount": len(stages), "description": description, "passed": True}


def _validate_recovery_surface(run_root: Path) -> dict[str, object]:
    control = run_root / "controls/verify_completed_training_recovery.py"
    source = control.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {"paramiko", "requests", "socket", "urllib", "http", "ftplib"}
    imported: set[str] = set()
    forbidden_calls = {
        "os.system",
        "shutil.rmtree",
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and _call_name(node) in forbidden_calls:
            raise ValueError(f"recovery verifier execution surface is forbidden: {_call_name(node)}")
    if imported & forbidden_imports:
        raise ValueError("recovery verifier contains a network import")
    forbidden_fragments = ("retry-training", "recover-promotion-lock")
    required_fragments = (
        "publish_json_pair_exclusive",
        "EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256",
        "_verify_completed_training_topology",
        "EXPECTED_EPOCH_ONLY_PROOF_SHA256",
        'batching.get("shardCount") != 29',
        "completed training epoch must be an object",
        "worker-stdout.log",
        'durable_root / "verify-training"',
    )
    if any(value in source for value in forbidden_fragments):
        raise ValueError("recovery verifier exposes a retry surface")
    missing = [value for value in required_fragments if value not in source]
    if missing:
        raise ValueError(f"recovery verifier admission fragments are missing: {missing}")
    description = _run_json(control, "--describe")
    if (
        description.get("runNamespace") != RUN_NAMESPACE
        or description.get("sourceCommit") != SOURCE_COMMIT
        or description.get("failureStage") != "train"
        or description.get("originalTerminalPreserved") is not True
        or description.get("retryOrTrainingMutation") is not False
        or description.get("receiptRelative")
        != "training-recovery/completed-training-verification.json"
    ):
        raise ValueError("recovery verifier description drifted")
    return {"description": description, "passed": True}
'''
    source = _replace_once(
        source,
        "\ndef validate(launcher: Path, expected_launcher_sha256: str | None) -> dict[str, object]:\n",
        control_surface
        + "\n\ndef validate(launcher: Path, expected_launcher_sha256: str | None) -> dict[str, object]:\n",
        "validator control surface function",
    )
    source = _replace_once(
        source,
        "    materialized = _validate_materialized_source(run_root)\n"
        "    self_test = _run_json(launcher, \"self-test\")\n",
        "    materialized = _validate_materialized_source(run_root)\n"
        "    control_surface = _validate_control_surface(run_root)\n"
        "    recovery_surface = _validate_recovery_surface(run_root)\n"
        "    self_test = _run_json(launcher, \"self-test\")\n",
        "validator control surface call",
    )
    source = _replace_once(
        source,
        '        or description.get("sourceManifestSha256") != SOURCE_MANIFEST_SHA256\n',
        '        or description.get("sourceCommit") != SOURCE_COMMIT\n'
        '        or description.get("sourceManifestSha256") != SOURCE_MANIFEST_SHA256\n',
        "validator launcher description commit",
    )
    source = _replace_once(
        source,
        '        SOURCE_MANIFEST_SHA256,\n'
        '        REMOTE_SOURCE,\n',
        '        "verify_source_blob_admission",\n'
        '        "MINIMUM_EVALUATION_FREE_BYTES",\n'
        '        "diskFreeAdmission",\n'
        '        "EXPECTED_RUNTIME_PYTHON_PATH_COUNT = 143",\n'
        '        "EXPECTED_CHECKED_SOURCE_PATH_COUNT = 154",\n'
        f'        "{EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256}",\n'
        f'        "{EXPECTED_SOURCE_UNION_INVENTORY_SHA256}",\n'
        '        SOURCE_COMMIT,\n'
        '        SOURCE_MANIFEST_SHA256,\n'
        '        REMOTE_SOURCE,\n',
        "validator required source and disk fragments",
    )
    source = _replace_once(
        source,
        '        or description.get("finalShardCount") != 32\n'
        '        or description.get("recoveryCommandsExposed") is not False\n',
        '        or description.get("finalShardCount") != 32\n'
        '        or description.get("fullSourceBlobAdmissionRequired") is not True\n'
        '        or description.get("minimumEvaluationFreeBytes")\n'
        f'        != {MINIMUM_EVALUATION_FREE_BYTES}\n'
        '        or description.get("runtimePythonPathCount") != 143\n'
        '        or description.get("checkedSourcePathCount") != 154\n'
        '        or description.get("runtimePythonSourceInventorySha256")\n'
        f'        != "{EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256}"\n'
        '        or description.get("sourceUnionInventorySha256")\n'
        f'        != "{EXPECTED_SOURCE_UNION_INVENTORY_SHA256}"\n'
        '        or description.get("recoveryCommandsExposed") is not False\n',
        "validator launcher source and disk description",
    )
    source = _replace_once(
        source,
        '        **materialized,\n'
        '        "controlSha256": CONTROL_SHA256,\n',
        '        **materialized,\n'
        '        "controlSha256": CONTROL_SHA256,\n'
        '        "controlSurface": control_surface,\n'
        '        "recoverySurface": recovery_surface,\n',
        "validator result control surface",
    )
    ast.parse(source)
    return source.encode("utf-8")


def _run_json(script: Path, *arguments: str, pythonpath: Path) -> dict[str, object]:
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(pythonpath)
        if not old_pythonpath
        else str(pythonpath) + os.pathsep + old_pythonpath
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(script), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    if completed.stderr:
        raise ValueError(f"local generated-control check wrote stderr: {completed.stderr}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("local generated-control check did not return a JSON object")
    return value


def _semantic_fixture_check(control: Path, controls: Path) -> dict[str, object]:
    """Exercise every extracted operation with inert in-memory source modules."""

    module_name = "_dalmuti_generated_evaluation_fixture_" + sha256_file(control)[:16]
    specification = importlib.util.spec_from_file_location(module_name, control)
    if specification is None or specification.loader is None:
        raise ImportError("generated evaluation control cannot be loaded for fixture test")
    module = importlib.util.module_from_spec(specification)
    controls_text = str(controls)
    sys.path.insert(0, controls_text)
    try:
        specification.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == controls_text:
            sys.path.pop(0)

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def operation(name: str):
        def invoke(*arguments: object, **keywords: object) -> dict[str, object]:
            calls.append((name, arguments, keywords))
            return {"operation": name}

        return invoke

    report_indices = {"final-0.json": 0, "final-1.json": 1}

    def load_report(path: Path) -> dict[str, object]:
        return {
            "mode": "final",
            "shard": {"count": 2, "index": report_indices[path.name]},
        }

    workflow = type(
        "FixtureWorkflow",
        (),
        {
            "reserve_v5_screening_run": staticmethod(operation("reserve-screening")),
            "evaluate_v5_run_stage": staticmethod(operation("evaluate")),
            "reserve_v5_certification_run": staticmethod(
                operation("reserve-certification")
            ),
            "claim_v5_final_run_shard": staticmethod(operation("claim-final")),
            "_load_workflow_evaluation_report": staticmethod(load_report),
            "merge_v5_evaluation_report_files": staticmethod(
                operation("merge-final")
            ),
        },
    )()
    promotion = type(
        "FixturePromotion",
        (),
        {
            "reserve_v5_final_holdout": staticmethod(operation("reserve-final")),
            "load_v5_promotion_plan": staticmethod(
                lambda _path: {
                    "final": {"matchShardCount": 2},
                    "reservationId": "fixture-reservation",
                }
            ),
            "approve_v5_final_holdout": staticmethod(operation("approve-final")),
        },
    )()
    modules = {"v5_workflow": workflow, "v5_promotion": promotion}
    fixture_root = Path("/fixture") / str(module.EXPECTED_RUN_NAMESPACE)
    fixture_registry = Path("/fixture/v5-promotion-registry")
    fixture_source = fixture_root / "source-checkout"
    module._source_modules = lambda _root: modules
    module._require_remote_rtx_3080 = lambda _root, _modules: "fixture RTX 3080"
    candidate_calls: list[Path] = []

    def candidate(
        _root: Path,
        _modules: Mapping[str, object],
    ) -> Path:
        candidate_calls.append(_root)
        return Path("/fixture/candidate")

    module._candidate = candidate
    module._require_registry_path = lambda _root, value, _label: Path(value)
    module._paths = lambda _root: {
        "registry": fixture_registry,
        "sourceCheckout": fixture_source,
    }

    module.reserve_screening(fixture_root)
    module.evaluate_screening(fixture_root, "/fixture/screening.json", None)
    module.reserve_certification(
        fixture_root, "/fixture/screening-reservation.json", "/fixture/screening.json"
    )
    module.evaluate_certification(
        fixture_root,
        "a",
        "/fixture/certification.json",
        "/fixture/screening.json",
        None,
    )
    module.evaluate_certification(
        fixture_root,
        "b",
        "/fixture/certification.json",
        "/fixture/screening.json",
        None,
    )
    module.reserve_final(
        fixture_root, ["/fixture/cert-a.json", "/fixture/cert-b.json"], 2
    )
    module.claim_final(fixture_root, "/fixture/plan.json", 2, 0)
    module.evaluate_final(
        fixture_root,
        "/fixture/plan.json",
        "/fixture/claim.json",
        2,
        0,
        None,
    )
    module.merge_final_reports(
        fixture_root,
        "/fixture/plan.json",
        ["/fixture/final-0.json", "/fixture/final-1.json"],
    )
    module.approve_final(
        fixture_root, "/fixture/plan.json", "/fixture/merged.json"
    )

    names = [name for name, _, _ in calls]
    expected = [
        "reserve-screening",
        "evaluate",
        "reserve-certification",
        "evaluate",
        "evaluate",
        "reserve-final",
        "claim-final",
        "evaluate",
        "merge-final",
        "approve-final",
    ]
    if names != expected:
        raise ValueError(f"evaluation semantic fixture call order drifted: {names}")
    if candidate_calls != [fixture_root] * 10:
        raise ValueError("candidate recovery receipt was not consumed by every stage")
    evaluation_calls = [keywords for name, _, keywords in calls if name == "evaluate"]
    if (
        [value.get("stage") for value in evaluation_calls]
        != ["screening", "certification-a", "certification-b", "final"]
        or any(value.get("lane_count") != 32 for value in evaluation_calls)
        or any(value.get("device") != "cuda:0" for value in evaluation_calls)
        or any(value.get("recovery_reason") is not None for value in evaluation_calls)
    ):
        raise ValueError("evaluation semantic fixture arguments drifted")
    return {"operationCount": len(calls), "passed": True}


def _fixture_source_admission(source_commit: str) -> dict[str, object]:
    return {
        "checkedPathCount": EXPECTED_CHECKED_SOURCE_PATH_COUNT,
        "passed": True,
        "runtimePythonPathCount": EXPECTED_RUNTIME_PYTHON_PATH_COUNT,
        "runtimePythonSourceInventorySha256": (
            EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256
        ),
        "sourceCommit": source_commit,
        "sourceUnionInventorySha256": EXPECTED_SOURCE_UNION_INVENTORY_SHA256,
    }


def _load_fixture_control(control: Path, controls: Path, label: str) -> object:
    module_name = (
        f"_dalmuti_generated_{label}_fixture_" + sha256_file(control)[:16]
    )
    specification = importlib.util.spec_from_file_location(module_name, control)
    if specification is None or specification.loader is None:
        raise ImportError(f"generated evaluation control cannot load {label} fixture")
    module = importlib.util.module_from_spec(specification)
    controls_text = str(controls)
    sys.path.insert(0, controls_text)
    try:
        specification.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == controls_text:
            sys.path.pop(0)
    return module


def _source_module_fixture_check(control: Path, controls: Path) -> dict[str, object]:
    """Exercise the actual full-admission/import closure without importing Torch."""

    module = _load_fixture_control(control, controls, "source_modules")
    source_commit = str(module.EXPECTED_SOURCE_COMMIT)
    admission = _fixture_source_admission(source_commit)
    events: list[str] = []
    with tempfile.TemporaryDirectory(prefix="dalmuti-source-module-fixture-") as raw:
        root = Path(raw) / str(module.EXPECTED_RUN_NAMESPACE)
        sealed = root / "source-checkout/gpu-training"
        sealed.mkdir(parents=True)

        def verify(_root: Path) -> dict[str, object]:
            events.append("admission")
            return dict(admission)

        imported: dict[str, object] = {}

        def import_module(name: str) -> object:
            events.append(f"import:{name}")
            value = type(
                f"Fixture_{name}",
                (),
                {"__file__": str(sealed / f"{name}.py")},
            )()
            imported[name] = value
            return value

        module.verify_source_blob_admission = verify
        module.importlib = type(
            "FixtureImportlib", (), {"import_module": staticmethod(import_module)}
        )()
        loaded = module._source_modules(root)
        expected_names = {
            "torch",
            "v5_gpu_memory_preflight",
            "v5_promotion",
            "v5_train",
            "v5_workflow",
            "_sourceAdmission",
        }
        if (
            set(loaded) != expected_names
            or loaded["_sourceAdmission"] != admission
            or events[0:1] != ["admission"]
            or events[1:]
            != [
                "import:torch",
                "import:v5_gpu_memory_preflight",
                "import:v5_promotion",
                "import:v5_train",
                "import:v5_workflow",
            ]
        ):
            raise ValueError("generated source-module admission/import closure drifted")

        malformed = dict(admission)
        malformed["runtimePythonPathCount"] = 142
        module.verify_source_blob_admission = lambda _root: dict(malformed)
        try:
            module._source_modules(root)
        except ValueError:
            pass
        else:
            raise ValueError("malformed full source admission was accepted")
    return {"importCount": len(imported), "negativeTests": 1, "passed": True}


def _publish_fixture_json_pair(path: Path, value: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value))
    path.write_bytes(payload)
    digest = sha256_bytes(payload)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


def _candidate_receipt_fixture_check(
    control: Path, controls: Path
) -> dict[str, object]:
    """Consume only an immutable recovery receipt and reject bound tampering."""

    module = _load_fixture_control(control, controls, "candidate_receipt")
    source_text = control.read_text(encoding="utf-8")
    if hasattr(module, "_load_training_control") or "verify_training_output" in source_text:
        raise ValueError("evaluation control retained the buggy training verifier path")
    pair = {
        "actorSha256": "a" * 64,
        "criticSha256": "9" * 64,
        "pairId": "b" * 64,
        "pairManifestSha256": "c" * 64,
    }
    dataset_identity = "d" * 64
    with tempfile.TemporaryDirectory(prefix="dalmuti-candidate-receipt-fixture-") as raw:
        root = Path(raw) / str(module.EXPECTED_RUN_NAMESPACE)
        training = root / "training"
        actor = training / "actor-bundle"
        actor.mkdir(parents=True)
        (actor / "actor.bin").write_bytes(b"sealed actor")
        checkpoint = training / "training-checkpoint.pt"
        checkpoint.write_bytes(b"sealed checkpoint")
        result = {
            "datasetIdentitySha256": dataset_identity,
            "hardGates": {"passed": True},
            "outputModelPair": dict(pair),
        }
        result_path = training / "result.json"
        result_path.write_bytes(canonical_json_bytes(result))
        inventory: dict[str, dict[str, object]] = {}
        for path in sorted(
            (value for value in training.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(training).as_posix(),
        ):
            relative = path.relative_to(training).as_posix()
            inventory[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest_sha = _publish_fixture_json_pair(
            training / "manifest.json",
            {"files": inventory, "format": "fixture-training-manifest"},
        )
        source_manifest_sha = _publish_fixture_json_pair(
            root / "source-seal/manifest.json",
            {"sourceCommit": str(module.EXPECTED_SOURCE_COMMIT)},
        )
        workflow_sha = _publish_fixture_json_pair(
            root / "workflow.json",
            {"runNamespace": str(module.EXPECTED_RUN_NAMESPACE)},
        )
        attempt_sha = _publish_fixture_json_pair(
            root / "training-execution/attempt-001.json", {"attempt": 1}
        )
        result_sha = sha256_file(result_path)
        durable_attempt = root / "durable-training/train/attempt-001"
        intent_sha = _publish_fixture_json_pair(
            durable_attempt / "intent.json", {"stage": "train"}
        )
        launch_sha = _publish_fixture_json_pair(
            durable_attempt / "launch.json", {"stage": "train"}
        )
        process_sha = _publish_fixture_json_pair(
            durable_attempt / "process.json", {"stage": "train"}
        )
        terminal_sha = _publish_fixture_json_pair(
            durable_attempt / "terminal.json", {"passed": False}
        )
        (durable_attempt / "stage-stderr.log").write_bytes(b"x" * 1006)
        (durable_attempt / "stage-stdout.log").write_bytes(b"")
        (durable_attempt / "worker-stderr.log").write_bytes(b"")
        (durable_attempt / "worker-stdout.log").write_bytes(b"")
        stderr_sha = sha256_file(durable_attempt / "stage-stderr.log")
        proof_sha = "1" * 64
        probe_sha = "2" * 64
        recovery_sha = "3" * 64
        module.EXPECTED_TRAINING_MANIFEST_SHA256 = manifest_sha
        module.EXPECTED_TRAINING_RESULT_SHA256 = result_sha
        module.EXPECTED_TRAINING_ATTEMPT_SHA256 = attempt_sha
        module.EXPECTED_DURABLE_TRAIN_TERMINAL_SHA256 = terminal_sha
        module.EXPECTED_DURABLE_TRAIN_STDERR_SHA256 = stderr_sha
        module.EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256 = launch_sha
        module.EXPECTED_DURABLE_TRAIN_INTENT_SHA256 = intent_sha
        module.EXPECTED_DURABLE_TRAIN_PROCESS_SHA256 = process_sha
        module.EXPECTED_EPOCH_ONLY_PROOF_SHA256 = proof_sha
        module.EXPECTED_EPOCH_ONLY_PROBE_SHA256 = probe_sha
        module.RECOVERY_VERIFIER_SHA256 = recovery_sha

        fake_train = type(
            "FixtureV5Train",
            (),
            {
                "verify_v5_model_pair": staticmethod(lambda _path: dict(pair)),
                "v5_actor_bundle_digests": staticmethod(
                    lambda _path: {"actorSha256": pair["actorSha256"]}
                ),
            },
        )()
        admission = _fixture_source_admission(str(module.EXPECTED_SOURCE_COMMIT))
        modules = {"_sourceAdmission": admission, "v5_train": fake_train}
        normalized_inventory_sha = sha256_bytes(canonical_json_bytes(inventory))
        receipt_document = {
            "candidateActorSha256": pair["actorSha256"],
            "candidatePairId": pair["pairId"],
            "candidatePairManifestSha256": pair["pairManifestSha256"],
            "datasetIdentitySha256": dataset_identity,
            "durableFailure": {
                "intentSha256": intent_sha,
                "launchSha256": launch_sha,
                "processSha256": process_sha,
                "stderrSha256": stderr_sha,
                "stdoutSha256": str(module.EXPECTED_EMPTY_SHA256),
                "terminalSha256": terminal_sha,
            },
            "durableTopology": {
                "attemptCount": 1,
                "attemptFileCount": 12,
                "verifyTrainingAbsent": True,
            },
            "epochOnlyVerifierProof": {
                "patchCalls": 1,
                "probeScriptSha256": probe_sha,
                "proofSha256": proof_sha,
                "remoteMutationCount": 0,
            },
            "format": "dalmuti-v5-run007-completed-training-recovery-verification",
            "manifestSha256": manifest_sha,
            "passed": True,
            "recoveryVerifierSha256": recovery_sha,
            "resultSha256": result_sha,
            "runNamespace": str(module.EXPECTED_RUN_NAMESPACE),
            "sourceAdmission": admission,
            "sourceCommit": str(module.EXPECTED_SOURCE_COMMIT),
            "sourceManifestSha256": source_manifest_sha,
            "trainingArtifactFileCount": len(inventory),
            "trainingArtifactInventorySha256": normalized_inventory_sha,
            "trainingAttemptSha256": attempt_sha,
            "trainingControlSha256": str(module.TRAINING_CONTROL_SHA256),
            "version": 1,
            "workflowSha256": workflow_sha,
        }
        receipt = root / module.RECOVERY_RECEIPT_RELATIVE
        _publish_fixture_json_pair(receipt, receipt_document)
        if module._candidate(root, modules) != actor.resolve():
            raise ValueError("recovery receipt returned the wrong candidate bundle")

        tamper_tests = 0
        original_checkpoint = checkpoint.read_bytes()
        checkpoint.write_bytes(b"tampered checkpoint")
        try:
            module._candidate(root, modules)
        except ValueError:
            tamper_tests += 1
        else:
            raise ValueError("checkpoint tamper escaped recovery receipt verification")
        checkpoint.write_bytes(original_checkpoint)

        extra = training / "unexpected-artifact.bin"
        extra.write_bytes(b"unexpected")
        try:
            module._candidate(root, modules)
        except ValueError:
            tamper_tests += 1
        else:
            raise ValueError("extra training artifact escaped recovery verification")
        extra.unlink()

        source_manifest = root / "source-seal/manifest.json"
        original_source_manifest = source_manifest.read_bytes()
        source_manifest.write_bytes(canonical_json_bytes({"tampered": True}))
        try:
            module._candidate(root, modules)
        except ValueError:
            tamper_tests += 1
        else:
            raise ValueError("source-seal tamper escaped recovery verification")
        source_manifest.write_bytes(original_source_manifest)

        worker_stdout = durable_attempt / "worker-stdout.log"
        worker_stdout.write_bytes(b"tampered")
        try:
            module._candidate(root, modules)
        except ValueError:
            tamper_tests += 1
        else:
            raise ValueError("durable worker-log tamper escaped recovery verification")
        worker_stdout.write_bytes(b"")

        original_receipt = receipt.read_bytes()
        original_sidecar = receipt.with_name(receipt.name + ".sha256").read_bytes()
        tampered_receipt = dict(receipt_document)
        tampered_receipt["epochOnlyVerifierProof"] = {
            **receipt_document["epochOnlyVerifierProof"],
            "patchCalls": 2,
        }
        _publish_fixture_json_pair(receipt, tampered_receipt)
        try:
            module._candidate(root, modules)
        except ValueError:
            tamper_tests += 1
        else:
            raise ValueError("epoch proof tamper escaped recovery verification")
        receipt.write_bytes(original_receipt)
        receipt.with_name(receipt.name + ".sha256").write_bytes(original_sidecar)

        if module._candidate(root, modules) != actor.resolve():
            raise ValueError("recovery receipt failed after fixture restoration")
    return {
        "buggyVerifierCalls": 0,
        "tamperTests": tamper_tests,
        "passed": True,
    }


def _recovery_epoch_fixture_check(
    recovery: Path, controls: Path, failure_reference: Mapping[str, object]
) -> dict[str, object]:
    """Prove the captured object epoch passes and scalar/schema drifts fail."""

    module = _load_fixture_control(recovery, controls, "recovery_epoch")
    result = failure_reference.get("result.json")
    if not isinstance(result, Mapping) or not isinstance(result.get("epoch"), Mapping):
        raise ValueError("captured recovery epoch fixture is absent")
    epoch = dict(result["epoch"])
    if module._validated_epoch(epoch) != epoch:
        raise ValueError("captured object epoch did not pass recovery validation")
    negative_tests = 0
    for mutated in (
        1,
        {**epoch, "actorDecisionRowsSeen": 1_602_499},
        {**epoch, "batching": {**epoch["batching"], "shardCount": 28}},
        {**epoch, "optimizerSteps": 67_711},
    ):
        try:
            module._validated_epoch(mutated)
        except ValueError:
            negative_tests += 1
        else:
            raise ValueError("invalid completed-training epoch escaped validation")
    return {
        "capturedObjectEpochPassed": True,
        "negativeTests": negative_tests,
        "shardCountPath": "epoch.batching.shardCount",
        "passed": True,
    }


def _recovery_delayed_import_fixture_check(
    recovery: Path, controls: Path
) -> dict[str, object]:
    """Exercise a real delayed import from the sealed checkout path."""

    module = _load_fixture_control(recovery, controls, "recovery_delayed_import")
    names = (
        "torch",
        "v5_gpu_memory_preflight",
        "v5_train",
        "v5_workflow",
        "v5_collection_plan",
    )
    saved = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    source_text = ""
    try:
        with tempfile.TemporaryDirectory(prefix="dalmuti-delayed-import-fixture-") as raw:
            root = Path(raw) / str(module.EXPECTED_RUN_NAMESPACE)
            source = root / "source-checkout/gpu-training"
            source.mkdir(parents=True)
            for name in ("torch", "v5_gpu_memory_preflight", "v5_train"):
                (source / f"{name}.py").write_text(
                    f'IDENTITY = "{name}"\n', encoding="utf-8"
                )
            (source / "v5_collection_plan.py").write_text(
                'IDENTITY = "delayed-sealed-module"\n', encoding="utf-8"
            )
            (source / "v5_workflow.py").write_text(
                "def delayed_identity():\n"
                "    import v5_collection_plan\n"
                "    return v5_collection_plan.IDENTITY, v5_collection_plan.__file__\n",
                encoding="utf-8",
            )
            source_text = str(source)
            loaded = module._source_modules(root)
            identity, delayed_path = loaded["v5_workflow"].delayed_identity()
            if (
                identity != "delayed-sealed-module"
                or source.resolve() not in Path(delayed_path).resolve().parents
                or source_text not in sys.path
            ):
                raise ValueError("sealed delayed source import did not remain available")
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]  # type: ignore[assignment]
        while source_text and source_text in sys.path:
            sys.path.remove(source_text)
    return {"delayedImportPassed": True, "passed": True}


def _recovery_inventory_fixture_check(
    recovery: Path, controls: Path
) -> dict[str, object]:
    """Exercise exact file inventory and content tamper gates."""

    module = _load_fixture_control(recovery, controls, "recovery_inventory")
    with tempfile.TemporaryDirectory(prefix="dalmuti-recovery-inventory-") as raw:
        training = Path(raw) / "training"
        training.mkdir()
        checkpoint = training / "training-checkpoint.pt"
        checkpoint.write_bytes(b"sealed checkpoint")
        result_path = training / "result.json"
        result_path.write_bytes(b"{}")
        inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (checkpoint, result_path)
        }
        manifest = {"files": inventory, "format": "fixture-training-manifest"}
        manifest_sha = _publish_fixture_json_pair(training / "manifest.json", manifest)
        module.EXPECTED_MANIFEST = manifest
        module.EXPECTED_TRAINING_MANIFEST_SHA256 = manifest_sha
        module._verify_training_inventory(training)
        negative_tests = 0

        original_checkpoint = checkpoint.read_bytes()
        checkpoint.write_bytes(b"tampered checkpoint")
        try:
            module._verify_training_inventory(training)
        except ValueError:
            negative_tests += 1
        else:
            raise ValueError("recovery inventory accepted content tampering")
        checkpoint.write_bytes(original_checkpoint)

        extra = training / "extra.bin"
        extra.write_bytes(b"extra")
        try:
            module._verify_training_inventory(training)
        except ValueError:
            negative_tests += 1
        else:
            raise ValueError("recovery inventory accepted an extra artifact")
        extra.unlink()

        symlink_tested = False
        linked = training / "linked.bin"
        try:
            linked.symlink_to(checkpoint)
        except OSError:
            pass
        else:
            symlink_tested = True
            try:
                module._verify_training_inventory(training)
            except ValueError:
                negative_tests += 1
            else:
                raise ValueError("recovery inventory accepted a symlink")
            linked.unlink()
        module._verify_training_inventory(training)
    return {
        "negativeTests": negative_tests,
        "symlinkRuntimeTested": symlink_tested,
        "passed": True,
    }


def _recovery_topology_fixture_check(
    recovery: Path, controls: Path
) -> dict[str, object]:
    """Exercise exact attempt topology, empty logs, and no verify stage."""

    module = _load_fixture_control(recovery, controls, "recovery_topology")
    with tempfile.TemporaryDirectory(prefix="dalmuti-recovery-topology-") as raw:
        root = Path(raw) / str(module.EXPECTED_RUN_NAMESPACE)
        attempt = root / "durable-training/train/attempt-001"
        attempt.mkdir(parents=True)
        files = {
            "intent.json": b"{}",
            "intent.json.sha256": b"fixture",
            "launch.json": b"{}",
            "launch.json.sha256": b"fixture",
            "process.json": b"{}",
            "process.json.sha256": b"fixture",
            "stage-stderr.log": b"x" * 1006,
            "stage-stdout.log": b"",
            "terminal.json": b"{}",
            "terminal.json.sha256": b"fixture",
            "worker-stderr.log": b"",
            "worker-stdout.log": b"",
        }
        for name, payload in files.items():
            (attempt / name).write_bytes(payload)
        module.EXPECTED_DURABLE_TRAIN_STDERR_SHA256 = sha256_file(
            attempt / "stage-stderr.log"
        )
        module._verify_completed_training_topology(root)
        negative_tests = 0

        verify_stage = root / "durable-training/verify-training"
        verify_stage.mkdir()
        try:
            module._verify_completed_training_topology(root)
        except ValueError:
            negative_tests += 1
        else:
            raise ValueError("recovery topology accepted a verify-training stage")
        verify_stage.rmdir()

        second_attempt = root / "durable-training/train/attempt-002"
        second_attempt.mkdir()
        try:
            module._verify_completed_training_topology(root)
        except ValueError:
            negative_tests += 1
        else:
            raise ValueError("recovery topology accepted a second attempt")
        second_attempt.rmdir()

        worker_stdout = attempt / "worker-stdout.log"
        worker_stdout.write_bytes(b"unexpected")
        try:
            module._verify_completed_training_topology(root)
        except ValueError:
            negative_tests += 1
        else:
            raise ValueError("recovery topology accepted worker output")
        worker_stdout.write_bytes(b"")
        module._verify_completed_training_topology(root)
    return {"negativeTests": negative_tests, "passed": True}


def _recovery_exclusive_publish_fixture_check(
    recovery: Path, controls: Path
) -> dict[str, object]:
    """Prove the recovery receipt is created once and never overwritten."""

    module = _load_fixture_control(recovery, controls, "recovery_exclusive")
    with tempfile.TemporaryDirectory(prefix="dalmuti-recovery-exclusive-") as raw:
        root = Path(raw) / str(module.EXPECTED_RUN_NAMESPACE)
        root.mkdir()
        module.validate_run_root = lambda value: value
        module.build_recovery_receipt = lambda _root: {
            "format": "fixture-recovery",
            "passed": True,
        }
        first = module.publish_recovery_receipt(root)
        try:
            module.publish_recovery_receipt(root)
        except FileExistsError:
            duplicate_rejected = True
        else:
            raise ValueError("recovery receipt was overwritten")
        receipt = root / module.RECOVERY_RECEIPT_RELATIVE
        if (
            first.get("passed") is not True
            or not duplicate_rejected
            or module.verify_sidecar(receipt) != first.get("receiptSha256")
        ):
            raise ValueError("exclusive recovery receipt publication drifted")
    return {"duplicateRejected": True, "passed": True}


def _validate_rendered(
    run_root: Path,
    files: Mapping[str, bytes],
    *,
    run_namespace: str,
    source_commit: str,
    evaluation_control_sha256: str,
    recovery_verifier_sha256: str,
    source_manifest_sha256: str,
    failure_reference: Mapping[str, object],
) -> dict[str, object]:
    for name, payload in files.items():
        if name.endswith(".py"):
            ast.parse(payload.decode("utf-8"), filename=name)
    with tempfile.TemporaryDirectory(prefix="dalmuti-evaluation-controls-") as raw:
        staging = Path(raw)
        for name, payload in files.items():
            (staging / name).write_bytes(payload)
        control = staging / EVALUATION_CONTROL_NAME
        environment = dict(os.environ)
        controls = run_root / "controls"
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(controls)
            if not old_pythonpath
            else str(controls) + os.pathsep + old_pythonpath
        )
        help_result = subprocess.run(
            [sys.executable, "-B", str(control), "--help"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
        if help_result.stderr or any(
            stage not in help_result.stdout for stage in EVALUATION_STAGES
        ):
            raise ValueError("generated evaluation control help surface drifted")
        semantic_fixture = _semantic_fixture_check(control, controls)
        source_module_fixture = _source_module_fixture_check(control, controls)
        candidate_receipt_fixture = _candidate_receipt_fixture_check(
            control, controls
        )
        recovery = staging / RECOVERY_VERIFIER_NAME
        recovery_epoch_fixture = _recovery_epoch_fixture_check(
            recovery, controls, failure_reference
        )
        recovery_delayed_import_fixture = _recovery_delayed_import_fixture_check(
            recovery, controls
        )
        recovery_inventory_fixture = _recovery_inventory_fixture_check(
            recovery, controls
        )
        recovery_topology_fixture = _recovery_topology_fixture_check(
            recovery, controls
        )
        recovery_exclusive_fixture = _recovery_exclusive_publish_fixture_check(
            recovery, controls
        )
        description = _run_json(control, "--describe", pythonpath=controls)
        if (
            description.get("runNamespace") != run_namespace
            or description.get("sourceCommit") != source_commit
            or description.get("stages") != list(EVALUATION_STAGES)
            or description.get("candidateVerification")
            != "immutable-completed-training-recovery-receipt"
            or description.get("recoveryVerifierSha256")
            != recovery_verifier_sha256
            or description.get("fullSourceBlobAdmissionRequired") is not True
            or description.get("runtimePythonPathCount")
            != EXPECTED_RUNTIME_PYTHON_PATH_COUNT
            or description.get("checkedSourcePathCount")
            != EXPECTED_CHECKED_SOURCE_PATH_COUNT
            or description.get("runtimePythonSourceInventorySha256")
            != EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256
            or description.get("sourceUnionInventorySha256")
            != EXPECTED_SOURCE_UNION_INVENTORY_SHA256
        ):
            raise ValueError("generated evaluation control description drifted")
        recovery_description = _run_json(
            recovery, "--describe", pythonpath=controls
        )
        if (
            recovery_description.get("runNamespace") != run_namespace
            or recovery_description.get("sourceCommit") != source_commit
            or recovery_description.get("failureStage") != "train"
            or recovery_description.get("originalTerminalPreserved") is not True
            or recovery_description.get("retryOrTrainingMutation") is not False
            or recovery_description.get("receiptRelative")
            != RECOVERY_RECEIPT_RELATIVE.as_posix()
        ):
            raise ValueError("generated recovery verifier description drifted")
        launcher = staging / LAUNCHER_NAME
        launcher_description = _run_json(launcher, "describe", pythonpath=controls)
        if (
            launcher_description.get("runNamespace") != run_namespace
            or launcher_description.get("sourceCommit") != source_commit
            or launcher_description.get("controlSha256")
            != evaluation_control_sha256
            or launcher_description.get("sourceManifestSha256")
            != source_manifest_sha256
            or launcher_description.get("fullSourceBlobAdmissionRequired") is not True
            or launcher_description.get("minimumEvaluationFreeBytes")
            != MINIMUM_EVALUATION_FREE_BYTES
            or launcher_description.get("runtimePythonPathCount")
            != EXPECTED_RUNTIME_PYTHON_PATH_COUNT
            or launcher_description.get("checkedSourcePathCount")
            != EXPECTED_CHECKED_SOURCE_PATH_COUNT
            or launcher_description.get("runtimePythonSourceInventorySha256")
            != EXPECTED_RUNTIME_PYTHON_INVENTORY_SHA256
            or launcher_description.get("sourceUnionInventorySha256")
            != EXPECTED_SOURCE_UNION_INVENTORY_SHA256
        ):
            raise ValueError("generated launcher description drifted")
        self_test = _run_json(launcher, "self-test", pythonpath=controls)
        if self_test.get("passed") is not True or self_test.get("stageCount") != 72:
            raise ValueError("generated launcher self-test failed")
        gate = _run_json(
            launcher, "gate", "--stage", "reserve-screening", pythonpath=controls
        )
        if gate.get("stage") != "reserve-screening" or gate.get("inputBindings") != []:
            raise ValueError("generated launcher screening gate drifted")
    return {
        "controlHelp": True,
        "candidateReceiptFixture": candidate_receipt_fixture,
        "controlSemanticFixture": semantic_fixture,
        "sourceModuleFixture": source_module_fixture,
        "recoveryDelayedImportFixture": recovery_delayed_import_fixture,
        "recoveryEpochFixture": recovery_epoch_fixture,
        "recoveryExclusivePublishFixture": recovery_exclusive_fixture,
        "recoveryInventoryFixture": recovery_inventory_fixture,
        "recoveryTopologyFixture": recovery_topology_fixture,
        "controlStageCount": len(EVALUATION_STAGES),
        "launcherSelfTest": True,
        "launcherStageCount": 72,
    }


def _publish_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace non-identical output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def build_controls(
    run_root: Path,
    output_directory: Path,
    *,
    run_namespace: str,
    source_commit: str,
    remote_parent: str = "/home/pangmin/dalmuti",
    run004_control_template: Path = RUN004_CONTROL_TEMPLATE,
    run006_launcher_template: Path = RUN006_LAUNCHER_TEMPLATE,
    run006_validator_template: Path = RUN006_VALIDATOR_TEMPLATE,
) -> dict[str, object]:
    root = run_root.resolve(strict=True)
    identity = _validate_target(root, run_namespace, source_commit)
    evaluation_contract = _evaluation_contract(root)
    failure_reference = _load_failure_reference()
    run004 = _require_template(
        run004_control_template, TEMPLATE_SHA256["run004Control"], "run-004 control"
    )
    run006_launcher = _require_template(
        run006_launcher_template,
        TEMPLATE_SHA256["run006Launcher"],
        "run-006 launcher",
    )
    run006_validator = _require_template(
        run006_validator_template,
        TEMPLATE_SHA256["run006Validator"],
        "run-006 validator",
    )
    training_sha = str(identity["trainingControlSha256"])
    common_sha = str(identity["controlCommonSha256"])
    manifest_sha = str(identity["sourceManifestSha256"])
    inventory_sha = str(identity["evaluationSourceInventorySha256"])
    low_disk_sha = str(identity["lowDiskStageSha256"])
    recovery_verifier = _render_training_recovery_verifier(
        run_namespace=run_namespace,
        source_commit=source_commit,
        training_control_sha256=training_sha,
        durable_training_launcher_sha256=str(
            identity["durableTrainingLauncherSha256"]
        ),
        source_manifest_sha256=manifest_sha,
        workflow_sha256=str(identity["workflowSha256"]),
        reference=failure_reference,
    )
    recovery_sha = sha256_bytes(recovery_verifier)
    evaluation_control = _render_evaluation_control(
        run004,
        run_namespace=run_namespace,
        source_commit=source_commit,
        training_control_sha256=training_sha,
        recovery_verifier_sha256=recovery_sha,
        remote_parent=remote_parent,
    )
    evaluation_sha = sha256_bytes(evaluation_control)
    launcher = _render_launcher(
        run006_launcher,
        run_namespace=run_namespace,
        source_commit=source_commit,
        evaluation_control_sha256=evaluation_sha,
        training_control_sha256=training_sha,
        control_common_sha256=common_sha,
        source_manifest_sha256=manifest_sha,
        source_inventory_sha256=inventory_sha,
        low_disk_sha256=low_disk_sha,
    )
    launcher_sha = sha256_bytes(launcher)
    validator = _render_validator(
        run006_validator,
        run_namespace=run_namespace,
        source_commit=source_commit,
        evaluation_control_sha256=evaluation_sha,
        recovery_verifier_sha256=recovery_sha,
        training_control_sha256=training_sha,
        control_common_sha256=common_sha,
        source_manifest_sha256=manifest_sha,
        source_inventory_sha256=inventory_sha,
        low_disk_sha256=low_disk_sha,
    )
    validator_sha = sha256_bytes(validator)
    generated = {
        EVALUATION_CONTROL_NAME: evaluation_control,
        RECOVERY_VERIFIER_NAME: recovery_verifier,
        LAUNCHER_NAME: launcher,
        VALIDATOR_NAME: validator,
    }
    checks = _validate_rendered(
        root,
        generated,
        run_namespace=run_namespace,
        source_commit=source_commit,
        evaluation_control_sha256=evaluation_sha,
        recovery_verifier_sha256=recovery_sha,
        source_manifest_sha256=manifest_sha,
        failure_reference=failure_reference,
    )
    receipt: dict[str, object] = {
        "controlCommonSha256": common_sha,
        "corpusLowDiskPlanSha256": identity["corpusLowDiskPlanSha256"],
        "evaluationControlSha256": evaluation_sha,
        "evaluationAttemptRelative": EVALUATION_ATTEMPT_RELATIVE,
        "evaluationContract": evaluation_contract,
        "evaluationSourceInventorySha256": inventory_sha,
        "epochOnlyProbeSha256": EPOCH_ONLY_PROBE_SHA256,
        "epochOnlyProofSha256": EPOCH_ONLY_PROOF_SHA256,
        "format": "dalmuti-v5-evaluation-controls-build",
        "launcherName": LAUNCHER_NAME,
        "launcherSha256": launcher_sha,
        "lowDiskStageSha256": low_disk_sha,
        "recoveryVerifierSha256": recovery_sha,
        "runNamespace": run_namespace,
        "sourceCommit": source_commit,
        "sourceManifestSha256": manifest_sha,
        "templateSha256": dict(TEMPLATE_SHA256),
        "trainingControlSha256": training_sha,
        "validation": checks,
        "validatorSha256": validator_sha,
        "validatorName": VALIDATOR_NAME,
        "version": 1,
        "workflowSha256": identity["workflowSha256"],
    }
    receipt_bytes = canonical_json_bytes(receipt)
    receipt_sha = sha256_bytes(receipt_bytes)
    outputs = {
        **generated,
        RECEIPT_NAME: receipt_bytes,
        RECEIPT_NAME + ".sha256": f"{receipt_sha}  {RECEIPT_NAME}\n".encode(
            "ascii"
        ),
    }
    target = output_directory.resolve()
    for name, payload in outputs.items():
        path = target / name
        if path.exists() and (
            not path.is_file() or path.is_symlink() or path.read_bytes() != payload
        ):
            raise FileExistsError(f"refusing a partial overwrite: {path}")
    for name, payload in outputs.items():
        _publish_exact(target / name, payload)
    if target == (root / "controls").resolve():
        validation = _run_json(
            target / VALIDATOR_NAME,
            "--launcher",
            str(target / LAUNCHER_NAME),
            "--expected-launcher-sha256",
            launcher_sha,
            pythonpath=target,
        )
        if validation.get("passed") is not True:
            raise ValueError("installed validator did not pass")
        receipt["installedValidation"] = validation
    return {**receipt, "buildReceiptSha256": receipt_sha, "outputDirectory": str(target)}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-namespace", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--remote-parent", default="/home/pangmin/dalmuti")
    parser.add_argument("--run004-control-template", type=Path, default=RUN004_CONTROL_TEMPLATE)
    parser.add_argument("--run006-launcher-template", type=Path, default=RUN006_LAUNCHER_TEMPLATE)
    parser.add_argument("--run006-validator-template", type=Path, default=RUN006_VALIDATOR_TEMPLATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    result = build_controls(
        arguments.run_root,
        arguments.output_directory,
        run_namespace=arguments.run_namespace,
        source_commit=arguments.source_commit,
        remote_parent=arguments.remote_parent,
        run004_control_template=arguments.run004_control_template,
        run006_launcher_template=arguments.run006_launcher_template,
        run006_validator_template=arguments.run006_validator_template,
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
