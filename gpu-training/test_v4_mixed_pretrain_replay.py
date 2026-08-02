from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v4_export import canonical_json_bytes as artifact_canonical_json_bytes
from v4_mixed_pretrain_replay import (
    V4_MIXED_REPLAY_AUDIT_BATCH_SIZE,
    _fixed_plan,
    _freeze_file,
    _parser,
    _rehash_and_recheck,
    publish_candidate_sidecar,
    verify_promotion_gates,
    verify_training_gates,
)
from v4_mixed_workflow import (
    BACKEND_MAP,
    BEHAVIOR_ACTOR_SHA256,
    BEHAVIOR_MANIFEST_SHA256,
    canonical_json_bytes,
)
from v4_model import canonical_v4_policy_numerics_contract
from v4_train import V4_CUDA_POLICY_AUDIT_BATCH_SIZE


def _write_canonical(path: Path, value: object, *, sidecar: bool = False) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if sidecar:
        Path(f"{path}.sha256").write_bytes(
            f"{digest}  {path.name}\n".encode("ascii")
        )
    return digest


def _write_artifact_canonical(
    path: Path, value: object, *, sidecar: bool = False
) -> str:
    payload = artifact_canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if sidecar:
        Path(f"{path}.sha256").write_bytes(
            f"{digest}  {path.name}\n".encode("ascii")
        )
    return digest


class MixedPretrainingAuditTests(unittest.TestCase):
    def test_replay_batch_is_bound_to_training_audit_contract(self) -> None:
        self.assertEqual(V4_CUDA_POLICY_AUDIT_BATCH_SIZE, 4)
        self.assertEqual(
            V4_MIXED_REPLAY_AUDIT_BATCH_SIZE,
            V4_CUDA_POLICY_AUDIT_BATCH_SIZE,
        )

    def test_fixed_plan_requires_canonical_string_key_backend_map(self) -> None:
        digest = "d" * 64
        fields = {
            "matchShardCount": 14,
            "shardBackendMap": {
                str(index): backend for index, backend in enumerate(BACKEND_MAP)
            },
            "version": 2,
        }
        plan = {
            "canonicalFields": fields,
            "canonicalSha256": digest,
            "opaqueId": (
                "fixed-complete-mixed-backend-shard-plan-v2:sha256=" + digest
            ),
        }
        self.assertEqual(_fixed_plan({"lossEligibility": {"fixedCollectionPlans": [plan]}}), plan)
        fields["shardBackendMap"] = list(BACKEND_MAP)
        with self.assertRaisesRegex(ValueError, "backend map drifted"):
            _fixed_plan({"lossEligibility": {"fixedCollectionPlans": [plan]}})

    def test_frozen_input_rejects_identical_byte_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"immutable input")
            snapshot = _freeze_file(source, root / "frozen.bin", "input")
            replacement = root / "replacement.bin"
            replacement.write_bytes(source.read_bytes())
            os.replace(replacement, source)
            with self.assertRaisesRegex(ValueError, "changed after freezing"):
                _rehash_and_recheck(snapshot, "input")
            os.chmod(snapshot.frozen, stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def _training_values() -> tuple[dict[str, object], dict[str, object]]:
        plan_sha = "e" * 64
        dataset = "f" * 64
        initial_actor = {
            "actorSha256": BEHAVIOR_ACTOR_SHA256,
            "manifestSha256": BEHAVIOR_MANIFEST_SHA256,
        }
        initial_replay = {
            "maximumAbsoluteLogProbabilityError": 1.0e-6,
            "passed": True,
        }
        initial_fingerprint = hashlib.sha256(
            artifact_canonical_json_bytes(initial_replay)
        ).hexdigest()
        training_contract = {
            "fixedPpoExecutionContract": {
                "policyNumerics": canonical_v4_policy_numerics_contract(),
                "version": 2,
            },
            "fixedCollectionPlanIds": [
                "fixed-complete-mixed-backend-shard-plan-v2:sha256=" + plan_sha
            ],
            "fixedCollectionPlanSha256": plan_sha,
            "initialPolicyReproductionAudit": initial_replay,
            "initialPolicyReproductionAuditFingerprint": initial_fingerprint,
            "ppoBehaviorActorSha256s": [BEHAVIOR_ACTOR_SHA256],
            "playerCountBalancedLoss": {
                "fixedPpoPolicyNumerics": canonical_v4_policy_numerics_contract(),
            },
            "requestedWeights": {
                "behaviorCloning": 0.05,
                "critic": 0.2,
                "ppo": 1.0,
            },
        }
        audit = {
            "actorAutocastEnabled": False,
            "actorForwardDtype": "torch.float32",
            "actorMode": "eval",
            "approxKl": 0.019,
            "clipFraction": 0.24,
            "datasetFingerprint": dataset,
            "entropyCollapseExceeds30Percent": False,
            "entropyRetentionRatio": 0.71,
            "fixedCollectionPlanSha256": plan_sha,
            "perPlayerCount": {str(player): {} for player in range(4, 11)},
        }
        actor_payload = b"candidate Actor"
        actor_sha = hashlib.sha256(actor_payload).hexdigest()
        candidate_manifest = {
            "files": {
                "actor.pt": {
                    "bytes": len(actor_payload),
                    "sha256": actor_sha,
                }
            },
            "format": "dalmuti-v4-candidate-manifest",
            "metadata": {
                "datasetFingerprint": dataset,
                "initialActor": initial_actor,
                "seed": 670000001,
            },
            "version": 2,
        }
        result = {
            "candidate": candidate_manifest,
            "completedEpochs": 1,
            "datasetFingerprint": dataset,
            "finalPostEpochPolicyDriftAudit": audit,
            "finalPostEpochPolicyDriftAuditFingerprint": hashlib.sha256(
                artifact_canonical_json_bytes(audit)
            ).hexdigest(),
            "format": "dalmuti-v4-training-result",
            "trainingContract": training_contract,
        }
        manifest = {
            "ampEnabled": True,
            "datasetFingerprint": dataset,
            "device": "cuda",
            "format": "dalmuti-v4-training-run",
            "initialActor": initial_actor,
            "privilegedCriticExported": False,
            "trainingConfig": {
                "actor_learning_rate": 2.0e-5,
                "amp": True,
                "batch_size": 2,
                "bc_weight": 0.05,
                "checkpoint_every": 1,
                "clip_ratio": 0.12,
                "critic_learning_rate": 2.0e-4,
                "critic_weight": 0.2,
                "entropy_coefficient": 0.0005,
                "epochs": 1,
                "expected_fixed_collection_plan_sha256": plan_sha,
                "gamma": 1.0,
                "gradient_accumulation": 1,
                "lambda_": 0.95,
                "max_gradient_norm": 1.0,
                "num_workers": 0,
                "ppo_weight": 1.0,
                "q_boost_coefficient": 0.0,
                "seed": 670000001,
                "weight_decay": 0.0001,
            },
            "trainingContract": training_contract,
        }
        return result, manifest

    @staticmethod
    def _write_training_candidate(root: Path, result: dict[str, object]) -> Path:
        candidate = root / "candidate"
        candidate.mkdir(parents=True)
        actor = candidate / "actor.pt"
        actor.write_bytes(b"candidate Actor")
        actor_sha = hashlib.sha256(actor.read_bytes()).hexdigest()
        manifest = result["candidate"]
        assert isinstance(manifest, dict)
        manifest_path = candidate / "manifest.json"
        manifest_sha = _write_artifact_canonical(
            manifest_path, manifest, sidecar=True
        )
        (candidate / "actor.pt.sha256").write_bytes(
            f"{actor_sha}  actor.pt\n".encode("ascii")
        )
        self_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert manifest_sha == self_manifest
        return candidate

    def test_training_gate_binds_full_recipe_and_rejects_weight_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, manifest = self._training_values()
            result_path = root / "result.json"
            manifest_path = root / "run-manifest.json"
            _write_artifact_canonical(result_path, result)
            _write_artifact_canonical(manifest_path, manifest)
            candidate = self._write_training_candidate(root, result)
            verified = verify_training_gates(
                result_path,
                manifest_path,
                candidate,
                root / "gates.json",
                maximum_approx_kl=0.020,
                maximum_clip_fraction=0.25,
                minimum_entropy_retention=0.70,
            )
            self.assertTrue(verified["passed"])
            result, manifest = self._training_values()
            result["trainingContract"]["requestedWeights"]["ppo"] = 0.9  # type: ignore[index]
            manifest["trainingContract"] = result["trainingContract"]
            drift_result = root / "drift-result.json"
            drift_manifest = root / "drift-manifest.json"
            _write_artifact_canonical(drift_result, result)
            _write_artifact_canonical(drift_manifest, manifest)
            drift_candidate = self._write_training_candidate(root / "drift", result)
            with self.assertRaisesRegex(ValueError, "weights.*drifted"):
                verify_training_gates(
                    drift_result,
                    drift_manifest,
                    drift_candidate,
                    root / "drift-gates.json",
                    maximum_approx_kl=0.020,
                    maximum_clip_fraction=0.25,
                    minimum_entropy_retention=0.70,
                )

    def test_training_gate_rejects_missing_or_tampered_policy_numerics(self) -> None:
        def missing_execution_policy(contract: dict[str, object]) -> None:
            execution = contract["fixedPpoExecutionContract"]
            assert isinstance(execution, dict)
            execution.pop("policyNumerics")

        def tampered_execution_policy(contract: dict[str, object]) -> None:
            execution = contract["fixedPpoExecutionContract"]
            assert isinstance(execution, dict)
            numerics = execution["policyNumerics"]
            assert isinstance(numerics, dict)
            numerics["mhaFastpathEnabled"] = True

        def wrong_execution_version(contract: dict[str, object]) -> None:
            execution = contract["fixedPpoExecutionContract"]
            assert isinstance(execution, dict)
            execution["version"] = 1

        def missing_balance_policy(contract: dict[str, object]) -> None:
            balance = contract["playerCountBalancedLoss"]
            assert isinstance(balance, dict)
            balance.pop("fixedPpoPolicyNumerics")

        def tampered_balance_policy(contract: dict[str, object]) -> None:
            balance = contract["playerCountBalancedLoss"]
            assert isinstance(balance, dict)
            numerics = balance["fixedPpoPolicyNumerics"]
            assert isinstance(numerics, dict)
            numerics["flashSdpEnabled"] = True

        cases = (
            (
                "missing-execution-policy",
                missing_execution_policy,
                "fixed PPO execution policy numerics",
            ),
            (
                "tampered-execution-policy",
                tampered_execution_policy,
                "fixed PPO execution policy numerics",
            ),
            (
                "wrong-execution-version",
                wrong_execution_version,
                "fixed PPO execution policy numerics",
            ),
            (
                "missing-balance-policy",
                missing_balance_policy,
                "player-count balance policy numerics",
            ),
            (
                "tampered-balance-policy",
                tampered_balance_policy,
                "player-count balance policy numerics",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    case_root = root / name
                    result, manifest = self._training_values()
                    contract = result["trainingContract"]
                    assert isinstance(contract, dict)
                    mutate(contract)
                    result_path = case_root / "result.json"
                    manifest_path = case_root / "run-manifest.json"
                    _write_artifact_canonical(result_path, result)
                    _write_artifact_canonical(manifest_path, manifest)
                    candidate = self._write_training_candidate(case_root, result)
                    with self.assertRaisesRegex(ValueError, message):
                        verify_training_gates(
                            result_path,
                            manifest_path,
                            candidate,
                            case_root / "gates.json",
                            maximum_approx_kl=0.020,
                            maximum_clip_fraction=0.25,
                            minimum_entropy_retention=0.70,
                        )

    def test_training_gate_rejects_candidate_swap_after_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, manifest = self._training_values()
            result_path = root / "result.json"
            manifest_path = root / "run-manifest.json"
            _write_artifact_canonical(result_path, result)
            _write_artifact_canonical(manifest_path, manifest)
            candidate = self._write_training_candidate(root, result)

            replacement_payload = b"different valid Actor bytes"
            actor = candidate / "actor.pt"
            actor.write_bytes(replacement_payload)
            actor_sha = hashlib.sha256(replacement_payload).hexdigest()
            replacement_manifest = dict(result["candidate"])  # type: ignore[arg-type]
            replacement_manifest["files"] = {
                "actor.pt": {
                    "bytes": len(replacement_payload),
                    "sha256": actor_sha,
                }
            }
            _write_artifact_canonical(
                candidate / "manifest.json",
                replacement_manifest,
                sidecar=True,
            )
            (candidate / "actor.pt.sha256").write_bytes(
                f"{actor_sha}  actor.pt\n".encode("ascii")
            )
            with self.assertRaisesRegex(ValueError, "exact candidate manifest"):
                verify_training_gates(
                    result_path,
                    manifest_path,
                    candidate,
                    root / "swapped-gates.json",
                    maximum_approx_kl=0.020,
                    maximum_clip_fraction=0.25,
                    minimum_entropy_retention=0.70,
                )

    @staticmethod
    def _screening_report(mean: float = 0.25) -> dict[str, object]:
        results = []
        for player_count in range(4, 11):
            results.append(
                {
                    "actsPerMatch": 5,
                    "matchClusters": {"count": 60},
                    "matches": 60,
                    "meanChipDifference": mean,
                    "meanChipDifferenceInference": {
                        "clusters": 60,
                        "low": 0.15,
                        "resamples": 10000,
                        "unit": "seed-matched-match",
                    },
                    "pairwiseCandidateBeforeNormal": {"rate": 0.55},
                    "playerCount": player_count,
                }
            )
        return {
            "candidatePolicy": {
                "actorCount": 1,
                "routing": {"mode": "pure-actor", "runtimeErrorFallback": False},
            },
            "evaluationMode": "screening",
            "format": "dalmuti-model-benchmark",
            "matchCountsByPlayerCount": {
                str(player_count): 60 for player_count in range(4, 11)
            },
            "playerCounts": list(range(4, 11)),
            "results": results,
            "seed": 450000001,
            "seedFamily": {"id": "attempt004-screening-seed450000001"},
        }

    def test_promotion_gate_requires_every_player_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "screening.json"
            _write_canonical(report, self._screening_report(), sidecar=True)
            value = verify_promotion_gates(
                report,
                root / "promotion.json",
                minimum_mean=0.25,
                minimum_lower=0.15,
                minimum_pairwise=0.55,
            )
            self.assertTrue(value["allPlayerCountsPassed"])
            failed = root / "failed-screening.json"
            bad = self._screening_report()
            bad["results"][3]["meanChipDifference"] = 0.249  # type: ignore[index]
            _write_canonical(failed, bad, sidecar=True)
            with self.assertRaisesRegex(ValueError, "p7 did not pass"):
                verify_promotion_gates(
                    failed,
                    root / "failed-promotion.json",
                    minimum_mean=0.25,
                    minimum_lower=0.15,
                    minimum_pairwise=0.55,
                )

    def test_promotion_gate_rejects_report_replacement_during_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "screening.json"
            _write_canonical(report, self._screening_report(), sidecar=True)
            real_publish = __import__(
                "v4_mixed_pretrain_replay", fromlist=["_publish"]
            )._publish

            def replace_then_publish(path: Path, value: object) -> str:
                replacement = root / "replacement.json"
                replacement.write_bytes(report.read_bytes())
                os.replace(replacement, report)
                return real_publish(path, value)

            with mock.patch(
                "v4_mixed_pretrain_replay._publish",
                side_effect=replace_then_publish,
            ), self.assertRaisesRegex(ValueError, "changed after freezing"):
                verify_promotion_gates(
                    report,
                    root / "promotion.json",
                    minimum_mean=0.25,
                    minimum_lower=0.15,
                    minimum_pairwise=0.55,
                )

    def test_candidate_sidecar_is_exact_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary)
            actor = candidate / "actor.pt"
            actor.write_bytes(b"candidate")
            digest = hashlib.sha256(actor.read_bytes()).hexdigest()
            manifest = {"files": {"actor.pt": {"sha256": digest}}}
            real_sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            with mock.patch(
                "v4_mixed_pretrain_replay.verify_v4_actor_bundle",
                return_value=manifest,
            ), mock.patch(
                "v4_mixed_pretrain_replay.sha256_file", side_effect=real_sha
            ):
                value = publish_candidate_sidecar(candidate)
                self.assertEqual(value["actorSha256"], digest)
                self.assertEqual(
                    (candidate / "actor.pt.sha256").read_bytes(),
                    f"{digest}  actor.pt\n".encode("ascii"),
                )
                with self.assertRaisesRegex(ValueError, "immutable"):
                    publish_candidate_sidecar(candidate)
            os.chmod(candidate / "actor.pt.sha256", stat.S_IRUSR | stat.S_IWUSR)

    def test_all_concrete_subcommands_parse(self) -> None:
        parser = _parser()
        self.assertEqual(
            parser.parse_args(
                [
                    "replay",
                    "--dataset",
                    "data.npz",
                    "--actor-bundle",
                    "actor",
                    "--device",
                    "cuda",
                    "--maximum-absolute-log-probability-error",
                    "2e-5",
                    "--output",
                    "audit.json",
                ]
            ).command,
            "replay",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "verify-training-gates",
                    "--training-result",
                    "result.json",
                    "--run-manifest",
                    "run-manifest.json",
                    "--candidate",
                    "candidate",
                    "--maximum-approx-kl",
                    ".020",
                    "--maximum-clip-fraction",
                    ".25",
                    "--minimum-entropy-retention",
                    ".70",
                    "--output",
                    "gates.json",
                ]
            ).command,
            "verify-training-gates",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "verify-promotion-gates",
                    "--screening-report",
                    "screen.json",
                    "--minimum-mean-chip-difference-per-act",
                    ".25",
                    "--minimum-clustered-95-lower-bound",
                    ".15",
                    "--minimum-pairwise-before-normal",
                    ".55",
                    "--output",
                    "promotion.json",
                ]
            ).command,
            "verify-promotion-gates",
        )
        self.assertEqual(
            parser.parse_args(
                ["publish-candidate-sidecar", "--candidate", "candidate"]
            ).command,
            "publish-candidate-sidecar",
        )


if __name__ == "__main__":
    unittest.main()
