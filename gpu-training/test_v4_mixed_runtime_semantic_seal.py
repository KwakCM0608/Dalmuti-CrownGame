from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import v4_mixed_package_runtime as runtime
import v4_mixed_workflow as workflow

from v4_mixed_package_runtime import (
    FINALIZATION_COMMAND_IDS,
    RUN_NAMESPACE,
    canonical_json_bytes,
    seal_run,
    verify_run_seal,
)
from v4_model import canonical_v4_policy_numerics_contract


def _pair(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    Path(f"{path}.sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


def _json_pair(path: Path, value: object) -> str:
    return _pair(path, canonical_json_bytes(value))


def _npz(path: Path, metadata: object | None = None) -> None:
    _pair(path, b"fixture-npz")
    _json_pair(
        Path(f"{path}.metadata.json"),
        {"fixture": path.name} if metadata is None else metadata,
    )


def _valid_pretraining_replay(
    merged_path: Path, plan_sha: str, dataset_fingerprint: str
) -> dict[str, object]:
    maximum_error = 1.0e-6
    mean_error = 5.0e-7
    rows = [
        {
            "backend": workflow.BACKEND_MAP[shard_index],
            "count": 1,
            "maximumAbsoluteLogProbabilityError": maximum_error,
            "meanAbsoluteLogProbabilityError": mean_error,
            "playerCount": player_count,
            "shardIndex": shard_index,
        }
        for player_count in range(4, 11)
        for shard_index in range(14)
    ]
    zero_counts = {str(player_count): 0 for player_count in range(4, 11)}
    nonforced_counts = {
        str(player_count): 14 for player_count in range(4, 11)
    }
    unit_masses = {str(player_count): 1.0 for player_count in range(4, 11)}
    entropies = {str(player_count): 0.5 for player_count in range(4, 11)}
    return {
        "actorSha256": workflow.BEHAVIOR_ACTOR_SHA256,
        "audit": {
            "absoluteTolerance": 2.0e-5,
            "actorAutocastEnabled": False,
            "actorForwardDtype": "torch.float32",
            "actorMode": "eval",
            "auditBatchSize": 64,
            "effectiveNonforcedPpoRowCount": 98,
            "forcedMaximumAbsoluteLogProbabilityError": 0.0,
            "forcedSingletonPpoRowCount": 0,
            "forcedSingletonRowsByPlayerCount": zero_counts,
            "maximumAbsoluteLogProbabilityError": maximum_error,
            "meanAbsoluteLogProbabilityError": mean_error,
            "nonforcedBalancedEntropy": 0.5,
            "nonforcedEntropyByPlayerCount": entropies,
            "nonforcedRowsByPlayerCount": nonforced_counts,
            "nonforcedTotalWeightMass": 7.0,
            "nonforcedWeightMassByPlayerCount": unit_masses,
            "passed": True,
            "ppoEligibleRowCount": 98,
            "storedOldActionLogProbabilityDtype": "torch.float32",
            "version": 2,
        },
        "datasetFingerprint": dataset_fingerprint,
        "datasetSha256": hashlib.sha256(merged_path.read_bytes()).hexdigest(),
        "device": "cuda",
        "fixedCollectionPlanSha256": plan_sha,
        "format": "dalmuti-v4-mixed-pretraining-replay",
        "manifestSha256": workflow.BEHAVIOR_MANIFEST_SHA256,
        "passed": True,
        "policyNumerics": canonical_v4_policy_numerics_contract(),
        "strata": {"byPlayerCountShardAndBackend": rows},
        "version": 1,
    }


def _decision_audit(include_players: bool) -> dict[str, object]:
    def row(**extra: object) -> dict[str, object]:
        return {
            "actorDecisions": 10,
            "actorRate": 1.0,
            "candidateDecisions": 10,
            "fallbackDecisions": 0,
            "fallbackRate": 0.0,
            **extra,
        }

    value: dict[str, object] = {
        "byAct": [row(act=act) for act in range(1, 6)],
        "overall": row(),
    }
    if include_players:
        value["byPlayerCount"] = [
            row(playerCount=player) for player in range(4, 11)
        ]
    return value


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            os.chmod(
                path,
                stat.S_IRUSR
                | stat.S_IWUSR
                | (stat.S_IXUSR if path.is_dir() else 0),
            )
        except OSError:
            pass
    try:
        os.chmod(root, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except OSError:
        pass


class MixedRuntimeSemanticSealTests(unittest.TestCase):
    def test_finalization_audit_is_bound_to_sealed_plan_and_current_counterparts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            source = root / "source" / "gpu-training"
            source.mkdir(parents=True)
            recipe_path = source / "v4_mixed_execution_recipe.json"
            workflow_path = source / "v4_mixed_workflow.py"
            shutil.copy2(
                Path(__file__).with_name("v4_mixed_execution_recipe.json"),
                recipe_path,
            )
            shutil.copy2(
                Path(__file__).with_name("v4_mixed_workflow.py"), workflow_path
            )
            recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
            recipe = workflow.load_recipe(recipe_path)
            phases = workflow.build_mixed_phase_plan(recipe)
            ordered = [
                (phase, command)
                for phase in phases
                for command in phase.commands
                if command.command_id in FINALIZATION_COMMAND_IDS
            ]
            plan_sha = "c" * 64
            remote_run = "/home/pangmin/dalmuti/v4-test-run"
            runtime_bindings = {
                "behaviorActorBundle": f"{remote_run}/behavior-actor",
                "format": "dalmuti-v4-mixed-remote-runtime-bindings",
                "frozenBaselineRepository": f"{remote_run}/frozen-baseline",
                "packageDirectory": f"{remote_run}/package",
                "packageManifestSha256": "a" * 64,
                "pythonExecutable": "/home/pangmin/dalmuti/.venv/bin/python",
                "recipeSha256": recipe_sha,
                "runDirectory": remote_run,
                "runNamespace": RUN_NAMESPACE,
                "sourceRoot": f"{remote_run}/source",
                "version": 1,
            }
            runtime_sha = _json_pair(
                root / "control" / "runtime-bindings.json", runtime_bindings
            )
            (root / "behavior-actor").mkdir()
            (root / "behavior-actor" / "actor.pt").write_bytes(b"actor")
            (root / "frozen-baseline").mkdir()
            (root / "package").mkdir()
            merged_metadata = root / "merged" / "production.npz.metadata.json"
            _json_pair(
                merged_metadata,
                {
                    "lossEligibility": {
                        "fixedCollectionPlans": [
                            {"canonicalSha256": plan_sha}
                        ]
                    }
                },
            )
            replacements = {
                "{remote_source_root}": f"{remote_run}/source",
                "{remote_run_directory}": remote_run,
                "{remote_behavior_actor_bundle}": f"{remote_run}/behavior-actor",
                "{remote_frozen_baseline_repository}": f"{remote_run}/frozen-baseline",
                "{remote_python}": runtime_bindings["pythonExecutable"],
                "{remote_package_directory}": f"{remote_run}/package",
                "{package_manifest_sha256}": "a" * 64,
                "{merged_collection_plan_sha256}": plan_sha,
            }

            expected_files: set[Path] = set()
            for _, command in ordered:
                for template in command.outputs:
                    if runtime._local_output_suffix(template) is not None:
                        continue
                    value = workflow.materialize_argv((template,), replacements)[0]
                    relative = runtime.PurePosixPath(value).relative_to(
                        runtime.PurePosixPath(remote_run)
                    )
                    current = root.joinpath(*relative.parts)
                    if current in (root / "source", root / "behavior-actor"):
                        continue
                    expected_files.add(current)
            for path in sorted(
                (item for item in expected_files if not item.name.endswith(".sha256")),
                key=str,
            ):
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json":
                    path.write_bytes(canonical_json_bytes({"fixture": path.name}))
                else:
                    path.write_bytes(f"fixture:{path.name}".encode("ascii"))
            for path in sorted(
                (item for item in expected_files if item.name.endswith(".sha256")),
                key=str,
            ):
                if path.exists():
                    continue
                payload = Path(str(path)[: -len(".sha256")])
                digest = hashlib.sha256(payload.read_bytes()).hexdigest()
                path.write_bytes(f"{digest}  {payload.name}\n".encode("ascii"))

            remote_records: dict[str, list[dict[str, object] | None]] = {}
            for _, command in ordered:
                records: list[dict[str, object] | None] = []
                for template in command.outputs:
                    if runtime._local_output_suffix(template) is not None:
                        records.append(None)
                        continue
                    expected = workflow.materialize_argv((template,), replacements)[0]
                    record, _ = runtime._current_remote_output_record(
                        root, runtime.PurePosixPath(remote_run), expected
                    )
                    records.append(record)
                remote_records[command.command_id] = records
            for local_id, counterparts in runtime.LOCAL_OUTPUT_COUNTERPARTS.items():
                command = next(
                    command for _, command in ordered if command.command_id == local_id
                )
                for index, (remote_id, remote_index) in enumerate(counterparts):
                    counterpart = remote_records[remote_id][remote_index]
                    self.assertIsNotNone(counterpart)
                    suffix = runtime._local_output_suffix(command.outputs[index])
                    remote_records[local_id][index] = {
                        **dict(counterpart),
                        "path": f"C:/sealed-local-run/{suffix}",
                    }

            required_commands = []
            for phase, command in ordered:
                outputs = [dict(item) for item in remote_records[command.command_id]]
                spec_sha = hashlib.sha256(
                    canonical_json_bytes(command.to_dict())
                ).hexdigest()
                if command.host == "remote":
                    argv = workflow.materialize_argv(command.argv, replacements)
                    argv_sha = hashlib.sha256(
                        canonical_json_bytes(list(argv))
                    ).hexdigest()
                else:
                    argv_sha = "d" * 64
                receipt = {
                    "commandId": command.command_id,
                    "commandSpecSha256": spec_sha,
                    "format": "dalmuti-v4-mixed-command-completion",
                    "host": command.host,
                    "materializedArgvSha256": argv_sha,
                    "outputs": outputs,
                    "packageManifestSha256": "a" * 64,
                    "passed": True,
                    "phaseId": phase.phase_id,
                    "recipeSha256": recipe_sha,
                    "runNamespace": RUN_NAMESPACE,
                    "runtimeBindingsSha256": runtime_sha,
                    "version": 1,
                }
                receipt_sha = _json_pair(
                    root
                    / "control"
                    / "completions"
                    / f"{command.command_id}.json",
                    receipt,
                )
                required_commands.append(
                    {
                        "commandId": command.command_id,
                        "commandSpecSha256": spec_sha,
                        "completionReceiptSha256": receipt_sha,
                        "outputs": outputs,
                        "phaseId": phase.phase_id,
                    }
                )
            audit_value = {
                "fixedCollectionPlanSha256": plan_sha,
                "format": "dalmuti-v4-mixed-finalization-audit",
                "packageManifestSha256": "a" * 64,
                "passed": True,
                "recipeSha256": recipe_sha,
                "requiredCommands": required_commands,
                "runNamespace": RUN_NAMESPACE,
                "runtimeBindingsSha256": runtime_sha,
                "version": 1,
            }
            audit = root / "provenance" / "finalization-audit.json"
            audit_sha = _json_pair(audit, audit_value)
            source_snapshots = {
                "gpu-training/v4_mixed_execution_recipe.json": runtime.stable_snapshot(
                    recipe_path, "recipe"
                ),
                "gpu-training/v4_mixed_workflow.py": runtime.stable_snapshot(
                    workflow_path, "workflow"
                ),
            }
            snapshots, _ = runtime._verify_finalization_audit(
                root,
                audit,
                audit_sha,
                package_manifest_sha256="a" * 64,
                recipe_sha256=recipe_sha,
                runtime_bindings_sha256=runtime_sha,
                source_snapshots=source_snapshots,
            )
            self.assertIn("finalization audit", snapshots)

            current_output = root / "calibration" / "cuda.npz"
            original_output = current_output.read_bytes()
            current_output.write_bytes(b"tampered-current-output")
            with self.assertRaisesRegex(
                ValueError, "remote finalization output bytes drifted"
            ):
                runtime._verify_finalization_audit(
                    root,
                    audit,
                    audit_sha,
                    package_manifest_sha256="a" * 64,
                    recipe_sha256=recipe_sha,
                    runtime_bindings_sha256=runtime_sha,
                    source_snapshots=source_snapshots,
                )
            current_output.write_bytes(original_output)

            local_id = "collect-calibration-cpu"
            local_index = list(FINALIZATION_COMMAND_IDS).index(local_id)
            local_receipt_path = (
                root / "control" / "completions" / f"{local_id}.json"
            )
            original_local_receipt = json.loads(
                local_receipt_path.read_text(encoding="utf-8")
            )
            tampered_local_receipt = json.loads(
                json.dumps(original_local_receipt)
            )
            tampered_local_receipt["outputs"][0]["sha256"] = "e" * 64
            tampered_local_sha = _json_pair(
                local_receipt_path, tampered_local_receipt
            )
            original_local_audit = dict(
                audit_value["requiredCommands"][local_index]
            )
            audit_value["requiredCommands"][local_index] = {
                **original_local_audit,
                "completionReceiptSha256": tampered_local_sha,
                "outputs": tampered_local_receipt["outputs"],
            }
            tampered_audit_sha = _json_pair(audit, audit_value)
            with self.assertRaisesRegex(
                ValueError, "local/remote counterpart bytes drifted"
            ):
                runtime._verify_finalization_audit(
                    root,
                    audit,
                    tampered_audit_sha,
                    package_manifest_sha256="a" * 64,
                    recipe_sha256=recipe_sha,
                    runtime_bindings_sha256=runtime_sha,
                    source_snapshots=source_snapshots,
                )
            restored_local_sha = _json_pair(
                local_receipt_path, original_local_receipt
            )
            audit_value["requiredCommands"][local_index] = {
                **original_local_audit,
                "completionReceiptSha256": restored_local_sha,
            }

            forged_id = FINALIZATION_COMMAND_IDS[0]
            forged_receipt_path = (
                root / "control" / "completions" / f"{forged_id}.json"
            )
            forged_receipt = json.loads(
                forged_receipt_path.read_text(encoding="utf-8")
            )
            forged_receipt["commandSpecSha256"] = "f" * 64
            forged_receipt_sha = _json_pair(forged_receipt_path, forged_receipt)
            audit_value["requiredCommands"][0]["commandSpecSha256"] = "f" * 64
            audit_value["requiredCommands"][0][
                "completionReceiptSha256"
            ] = forged_receipt_sha
            forged_audit_sha = _json_pair(audit, audit_value)
            with self.assertRaisesRegex(
                ValueError, "spec disagrees with sealed workflow"
            ):
                runtime._verify_finalization_audit(
                    root,
                    audit,
                    forged_audit_sha,
                    package_manifest_sha256="a" * 64,
                    recipe_sha256=recipe_sha,
                    runtime_bindings_sha256=runtime_sha,
                    source_snapshots=source_snapshots,
                )

    def test_frozen_baseline_verifier_rejects_tracked_source_tamper(self) -> None:
        bundle_source = (
            Path(__file__).parent.parent
            / "artifacts"
            / "rl"
            / "v4-frozen-baseline-git-bundle-run-001"
            / runtime.FROZEN_BASELINE_BUNDLE_NAME
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            bundle = root / "baseline-bundle" / bundle_source.name
            bundle.parent.mkdir(parents=True)
            shutil.copy2(bundle_source, bundle)
            shutil.copy2(
                Path(f"{bundle_source}.sha256"), Path(f"{bundle}.sha256")
            )
            repository = root / "frozen-baseline"
            subprocess.run(
                ["git", "clone", "--no-checkout", str(bundle), str(repository)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "core.autocrlf", "false"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "core.eol", "lf"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "checkout",
                    "--detach",
                    runtime.FROZEN_BASELINE_COMMIT,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            observation = root / "source" / "training" / "v4-public-history.ts"
            observation.parent.mkdir(parents=True)
            shutil.copy2(
                Path(__file__).parent.parent / "training" / "v4-public-history.ts",
                observation,
            )
            verified = runtime._verify_remote_frozen_baseline(root)
            self.assertIn("frozen Normal source", verified)
            for path in sorted(repository.rglob("*"), reverse=True):
                os.chmod(path, 0o555 if path.is_dir() else 0o444)
            os.chmod(repository, 0o555)
            self.assertIn(
                "frozen Normal source",
                runtime._verify_remote_frozen_baseline(root),
            )
            _make_writable(repository)
            normal = repository / "lib" / "bot-strategy.ts"
            normal.write_bytes(normal.read_bytes() + b"\n// tamper\n")
            with self.assertRaisesRegex(
                ValueError, "Normal source drifted|worktree is dirty"
            ):
                runtime._verify_remote_frozen_baseline(root)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "checkout",
                    "--",
                    "lib/bot-strategy.ts",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            (repository / "untracked.txt").write_bytes(b"untracked")
            with self.assertRaisesRegex(ValueError, "worktree is dirty"):
                runtime._verify_remote_frozen_baseline(root)

    def test_local_aggregate_requires_exact_remote_success_bound_to_remote_seal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "aggregate"
            nested_seal = (
                root
                / "remote-sealed-run"
                / "provenance"
                / "final-files.json"
            )
            nested_digest = _json_pair(
                nested_seal, {"profile": "remote-semantic"}
            )
            success = root / "remote-sealed-run" / "status" / "999-succeeded.json"
            _json_pair(
                success,
                {
                    "detail": "complete",
                    "format": "dalmuti-v4-mixed-stage-status",
                    "runSealSha256": nested_digest,
                    "stage": "complete",
                    "state": "succeeded",
                    "version": 1,
                },
            )
            with mock.patch.object(
                runtime, "verify_run_seal", return_value=nested_digest
            ):
                _, sealed_sha, status_sha = (
                    runtime._verify_local_aggregate_remote_copy(root)
                )
                self.assertEqual(sealed_sha, nested_digest)
                self.assertEqual(status_sha, hashlib.sha256(success.read_bytes()).hexdigest())

                status_value = json.loads(success.read_text(encoding="utf-8"))
                status_value["state"] = "failed"
                _json_pair(success, status_value)
                with self.assertRaisesRegex(
                    ValueError, "success status does not bind"
                ):
                    runtime._verify_local_aggregate_remote_copy(root)

    def test_pretraining_replay_semantics_reject_missing_or_drifted_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged = root / "production.npz"
            merged.write_bytes(b"sealed merged dataset")
            merged_snapshot = runtime.stable_snapshot(merged, "merged fixture")
            plan_sha = "c" * 64
            dataset_fingerprint = "f" * 64
            metadata = {
                "fingerprint": dataset_fingerprint,
                "lossEligibility": {
                    "fixedCollectionPlans": [
                        {"canonicalSha256": plan_sha}
                    ]
                },
            }
            recipe = {
                "runContract": {
                    "behaviorActor": {
                        "actorSha256": workflow.BEHAVIOR_ACTOR_SHA256,
                        "manifestSha256": workflow.BEHAVIOR_MANIFEST_SHA256,
                    },
                    "policyNumerics": canonical_v4_policy_numerics_contract(),
                }
            }
            valid = _valid_pretraining_replay(
                merged, plan_sha, dataset_fingerprint
            )
            runtime._validate_pretraining_replay(
                valid, recipe, merged_snapshot, metadata
            )

            missing_policy = json.loads(json.dumps(valid))
            del missing_policy["policyNumerics"]
            with self.assertRaisesRegex(ValueError, "report header"):
                runtime._validate_pretraining_replay(
                    missing_policy, recipe, merged_snapshot, metadata
                )

            drifted_policy = json.loads(json.dumps(valid))
            drifted_policy["policyNumerics"]["mathSdpEnabled"] = False
            with self.assertRaisesRegex(ValueError, "replay policy numerics"):
                runtime._validate_pretraining_replay(
                    drifted_policy, recipe, merged_snapshot, metadata
                )

            wrong_actor = json.loads(json.dumps(valid))
            wrong_actor["actorSha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "behavior Actor bundle"):
                runtime._validate_pretraining_replay(
                    wrong_actor, recipe, merged_snapshot, metadata
                )

            missing_cuda_stratum = json.loads(json.dumps(valid))
            missing_cuda_stratum["strata"][
                "byPlayerCountShardAndBackend"
            ].pop()
            with self.assertRaisesRegex(ValueError, "complete p4-p10 CPU/CUDA"):
                runtime._validate_pretraining_replay(
                    missing_cuda_stratum, recipe, merged_snapshot, metadata
                )

            excessive_error = json.loads(json.dumps(valid))
            excessive_error["audit"][
                "maximumAbsoluteLogProbabilityError"
            ] = 2.1e-5
            with self.assertRaisesRegex(ValueError, "exceeded.*2e-5"):
                runtime._validate_pretraining_replay(
                    excessive_error, recipe, merged_snapshot, metadata
                )

    def test_full_remote_semantic_profile_is_durable_and_receipt_bound(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "run"
        try:
            (root / "status").mkdir(parents=True)
            package_sha = _json_pair(
                root / "package" / "package-manifest.json", {"fixture": "package"}
            )
            recipe = {
                "runContract": {
                    "behaviorActor": {
                        "actorSha256": workflow.BEHAVIOR_ACTOR_SHA256,
                        "manifestSha256": workflow.BEHAVIOR_MANIFEST_SHA256,
                    },
                    "fixture": "exact",
                    "policyNumerics": canonical_v4_policy_numerics_contract(),
                }
            }
            recipe_path = (
                root / "source" / "gpu-training" / "v4_mixed_execution_recipe.json"
            )
            recipe_sha = _json_pair(recipe_path, recipe)
            # The source package does not normally have an adjacent recipe
            # sidecar. It is harmless here, and is itself sealed.
            run_contract_sha = hashlib.sha256(
                canonical_json_bytes(recipe["runContract"])
            ).hexdigest()
            runtime_bindings = {
                "format": "dalmuti-v4-mixed-remote-runtime-bindings",
                "packageManifestSha256": package_sha,
                "recipeSha256": recipe_sha,
                "runNamespace": RUN_NAMESPACE,
                "version": 1,
            }
            runtime_sha = _json_pair(
                root / "control" / "runtime-bindings.json", runtime_bindings
            )

            calibration = root / "calibration"
            _json_pair(calibration / "backend-comparison.json", {"passed": True})
            _npz(calibration / "cpu.npz")
            _npz(calibration / "cuda.npz")
            for index in range(14):
                _npz(root / "rollouts" / f"shard-{index:02d}.npz")
            plan_sha = "c" * 64
            dataset_fingerprint = "f" * 64
            merged_path = root / "merged" / "production.npz"
            _npz(
                merged_path,
                {
                    "fingerprint": dataset_fingerprint,
                    "lossEligibility": {
                        "fixedCollectionPlans": [
                            {"canonicalSha256": plan_sha}
                        ]
                    }
                },
            )
            _json_pair(
                root / "replay" / "pretraining.json",
                _valid_pretraining_replay(
                    merged_path, plan_sha, dataset_fingerprint
                ),
            )
            training = root / "training" / "train-seed-610000001-run-001"
            (training / "result.json").parent.mkdir(parents=True, exist_ok=True)
            (training / "result.json").write_bytes(
                canonical_json_bytes({"passed": True})
            )
            (training / "run-manifest.json").write_bytes(
                canonical_json_bytes({"seed": 610000001})
            )
            candidate = training / "candidate"
            actor_sha = _pair(candidate / "actor.pt", b"candidate")
            manifest_sha = _json_pair(
                candidate / "manifest.json",
                {"files": {"actor.pt": {"sha256": actor_sha}}},
            )
            _json_pair(
                root / "training" / "epoch-0001-hard-gates.json",
                {
                    "candidateActorSha256": actor_sha,
                    "candidateManifestSha256": manifest_sha,
                    "format": "dalmuti-v4-mixed-training-hard-gates",
                    "passed": True,
                    "version": 1,
                },
            )
            results = []
            for player in range(4, 11):
                results.append(
                    {
                        "actsPerMatch": 5,
                        "candidateDecisionAudit": _decision_audit(False),
                        "matchClusters": {"count": 60},
                        "matches": 60,
                        "meanChipDifference": 0.5,
                        "meanChipDifferenceInference": {
                            "clusters": 60,
                            "low": 0.2,
                            "resamples": 10000,
                            "unit": "seed-matched-match",
                        },
                        "pairwiseCandidateBeforeNormal": {"rate": 0.6},
                        "playerCount": player,
                    }
                )
            screening = root / "screening" / "epoch-0001.json"
            screening_sha = _json_pair(
                screening,
                {
                    "actsPerMatch": 5,
                    "bindings": {
                        "artifactSha256": manifest_sha,
                        "modelSha256": actor_sha,
                    },
                    "candidateDecisionAudit": _decision_audit(True),
                    "evaluationMode": "screening",
                    "format": "dalmuti-model-benchmark",
                    "matchCountsByPlayerCount": {
                        str(player): 60 for player in range(4, 11)
                    },
                    "playerCounts": list(range(4, 11)),
                    "results": results,
                },
            )
            promotion = root / "screening" / "epoch-0001-promotion-gates.json"
            promotion_sha = _json_pair(
                promotion,
                {
                    "allPlayerCountsPassed": True,
                    "format": "dalmuti-v4-mixed-promotion-gates",
                    "gates": {
                        "minimumClustered95LowerBound": 0.15,
                        "minimumMeanChipDifferencePerAct": 0.25,
                        "minimumPairwiseBeforeNormal": 0.55,
                    },
                    "passed": True,
                    "perPlayerCount": {
                        str(player): {
                            "clustered95LowerBound": 0.2,
                            "meanChipDifferencePerAct": 0.5,
                            "pairwiseBeforeNormal": 0.6,
                            "passed": True,
                        }
                        for player in range(4, 11)
                    },
                    "screeningReportSha256": screening_sha,
                    "version": 1,
                },
            )

            required_commands = []
            for index, command_id in enumerate(FINALIZATION_COMMAND_IDS):
                spec_sha = f"{index + 1:064x}"
                receipt = {
                    "commandId": command_id,
                    "commandSpecSha256": spec_sha,
                    "format": "dalmuti-v4-mixed-command-completion",
                    "host": "fixture",
                    "materializedArgvSha256": "d" * 64,
                    "outputs": [],
                    "packageManifestSha256": package_sha,
                    "passed": True,
                    "phaseId": f"phase-{index:02d}",
                    "recipeSha256": recipe_sha,
                    "runNamespace": RUN_NAMESPACE,
                    "runtimeBindingsSha256": runtime_sha,
                    "version": 1,
                }
                receipt_sha = _json_pair(
                    root / "control" / "completions" / f"{command_id}.json",
                    receipt,
                )
                required_commands.append(
                    {
                        "commandId": command_id,
                        "commandSpecSha256": spec_sha,
                        "completionReceiptSha256": receipt_sha,
                        "outputs": [],
                        "phaseId": f"phase-{index:02d}",
                    }
                )
            audit = root / "provenance" / "finalization-audit.json"
            audit_sha = _json_pair(
                audit,
                {
                    "fixedCollectionPlanSha256": plan_sha,
                    "format": "dalmuti-v4-mixed-finalization-audit",
                    "packageManifestSha256": package_sha,
                    "passed": True,
                    "recipeSha256": recipe_sha,
                    "requiredCommands": required_commands,
                    "runNamespace": RUN_NAMESPACE,
                    "runtimeBindingsSha256": runtime_sha,
                    "version": 1,
                },
            )
            seal_path = root / "provenance" / "final-files.json"
            source_verification = SimpleNamespace(
                package_snapshots={}, source_snapshots={}
            )
            with mock.patch.object(
                runtime,
                "verify_remote_package_source",
                return_value=source_verification,
            ), mock.patch.object(
                runtime, "recheck_remote_package_source"
            ), mock.patch.object(
                runtime, "_verify_remote_frozen_baseline", return_value={}
            ), mock.patch.object(
                runtime, "_verify_finalization_audit", return_value=({}, {})
            ):
                value = seal_run(
                    root,
                    seal_path,
                    root / "status",
                    screening,
                    promotion,
                    package_sha,
                    recipe_sha,
                    run_contract_sha,
                    runtime_sha,
                    audit,
                    audit_sha,
                    "remote-semantic",
                )
                self.assertEqual(value["profile"], "remote-semantic")
                self.assertEqual(
                    value["semanticBindings"]["finalizationAuditSha256"], audit_sha
                )
                self.assertEqual(
                    value["semanticBindings"]["promotionReportSha256"], promotion_sha
                )
                self.assertEqual(verify_run_seal(seal_path), value["sealSha256"])
                receipt = (
                    root
                    / "control"
                    / "completions"
                    / f"{FINALIZATION_COMMAND_IDS[0]}.json"
                )
                os.chmod(receipt, 0o600)
                receipt.write_bytes(canonical_json_bytes({"tampered": True}))
                with self.assertRaisesRegex(
                    ValueError,
                    "sealed run (file remains writable|size mismatch|digest mismatch)",
                ):
                    verify_run_seal(seal_path)
        finally:
            _make_writable(root)
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
