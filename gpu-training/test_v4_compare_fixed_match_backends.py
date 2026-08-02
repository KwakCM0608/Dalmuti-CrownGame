from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:  # pragma: no cover
    raise unittest.SkipTest("torch and numpy are required") from error

from v4_collect_fixed_match_ppo import (
    FixedMatchPPOCollectionConfig,
    collect_v4_fixed_match_ppo,
)
import v4_compare_fixed_match_backends as backend_comparator
from v4_compare_fixed_match_backends import (
    CALIBRATION_FORMAT,
    OLD_ACTION_LOG_PROBABILITY_MAX_ABS,
    compare_fixed_match_backends,
    load_fixed_match_backend_calibration_report,
    load_verified_fixed_match_backend_calibration,
)
from v4_dataset import (
    V4TrajectoryTensors,
    fingerprint_v4_tensors,
)
from v4_env import ACTION_COUNT, V4ActorObservation
from v4_export import canonical_json_bytes, export_v4_actor_bundle
from v4_model import V4ActorConfig, V4PublicActor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    boolean_names = {
        "player_mask",
        "history_mask",
        "legal_masks",
        "dones",
        "valid_masks",
    }
    integer_names = {"actions", "expert_actions"}
    tensors: dict[str, torch.Tensor] = {}
    for field in fields(V4TrajectoryTensors):
        array = arrays[field.name]
        if field.name in boolean_names:
            value = torch.from_numpy(array.astype(np.bool_, copy=False))
        elif field.name in integer_names:
            value = torch.from_numpy(array.astype(np.int64, copy=False))
        else:
            value = torch.from_numpy(array.astype(np.float32, copy=False))
        tensors[field.name] = value
    return fingerprint_v4_tensors(V4TrajectoryTensors(**tensors))


def _entropy_summary(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=0)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


class FixedMatchBackendCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        torch.manual_seed(20260802)
        actor = V4PublicActor(
            V4ActorConfig(
                max_players=10,
                max_history=8,
                d_model=16,
                layers=1,
                heads=4,
                feedforward=32,
                action_hidden=16,
            )
        ).eval()
        cls.bundle = cls.root / "candidate"
        export_v4_actor_bundle(
            actor,
            cls.bundle,
            metadata={"purpose": "fixed-backend-calibration-test"},
            include_onnx=False,
        )

        def deterministic_logits(
            model: object,
            observations: list[V4ActorObservation],
            device: torch.device,
        ) -> list[torch.Tensor]:
            del model, device
            output: list[torch.Tensor] = []
            for observation in observations:
                logits = torch.full(
                    (ACTION_COUNT,), float("-inf"), dtype=torch.float64
                )
                legal = torch.nonzero(
                    observation.legal_mask, as_tuple=False
                ).flatten()
                logits[legal] = torch.linspace(
                    -0.25, 0.75, len(legal), dtype=torch.float64
                )
                output.append(logits)
            return output

        cls.cpu = cls.root / "cpu" / "fixed.npz"
        with mock.patch(
            "v4_collect_fixed_match_ppo._batch_candidate_logits",
            side_effect=deterministic_logits,
        ):
            collect_v4_fixed_match_ppo(
                cls.bundle,
                cls.cpu,
                FixedMatchPPOCollectionConfig(
                    run_namespace="cross-device-small",
                    seed_base=970_000_001,
                    match_counts=tuple(
                        (player_count, 1) for player_count in range(4, 11)
                    ),
                    standardize_advantages=False,
                    lane_count=1,
                    device="cpu",
                ),
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _clone(self, name: str, role: str, mutate=None) -> Path:
        self.assertIn(role, {"cpu", "cuda"})
        destination = self.root / name / "fixed.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with np.load(self.cpu, allow_pickle=False) as archive:
            arrays = {
                array_name: np.array(archive[array_name], copy=True)
                for array_name in archive.files
            }
        metadata = json.loads(str(arrays["metadata_json"].item()))
        valid = arrays["valid_masks"]
        if role == "cuda":
            execution = metadata["execution"]
            execution.update(
                {
                    "torchVersion": "2.7.1+cu118",
                    "numpyVersion": "2.2.4",
                    "device": "cuda",
                    "cudaAvailable": True,
                    "tf32Allowed": False,
                    "cublasWorkspaceConfig": ":4096:8",
                }
            )
            metadata["collection"]["batchedGpuMaskedLogitInference"] = True
            eligible = np.argwhere(
                valid & (arrays["old_action_log_probs"] < -0.05)
            )
            self.assertGreater(len(eligible), 0)
            row, step = (int(value) for value in eligible[0])
            arrays["old_action_log_probs"][row, step] -= np.float32(1.0e-6)
            arrays["selected_action_probabilities"][row, step] = math_exp = float(
                np.exp(float(arrays["old_action_log_probs"][row, step]))
            )
            self.assertGreater(math_exp, 0.0)
            arrays["policy_entropies"][row, step] += np.float32(1.0e-6)

        if mutate is not None:
            mutate(metadata, arrays)
        metadata["fingerprint"] = _tensor_fingerprint(arrays)
        metadata["policyEntropy"] = _entropy_summary(
            arrays["policy_entropies"][valid]
        )
        arrays["metadata_json"] = np.asarray(
            canonical_json_bytes(metadata).decode("utf-8")
        )
        np.savez_compressed(destination, **arrays)
        npz_sha = _sha256(destination)
        Path(f"{destination}.sha256").write_bytes(
            f"{npz_sha}  {destination.name}\n".encode("ascii")
        )
        external = dict(metadata)
        external["npzSha256"] = npz_sha
        metadata_path = Path(f"{destination}.metadata.json")
        metadata_path.write_bytes(canonical_json_bytes(external) + b"\n")
        Path(f"{metadata_path}.sha256").write_bytes(
            f"{_sha256(metadata_path)}  {metadata_path.name}\n".encode("ascii")
        )
        return destination

    def _cuda_clone(self, name: str, mutate=None) -> Path:
        return self._clone(name, "cuda", mutate)

    def _cpu_clone(self, name: str, mutate=None) -> Path:
        return self._clone(name, "cpu", mutate)

    def _expected_report_bindings(self) -> tuple[str, str, dict[str, str]]:
        metadata = json.loads(
            Path(f"{self.cpu}.metadata.json").read_text(encoding="utf-8")
        )
        return (
            metadata["modelBinding"]["actorCheckpointSha256"],
            metadata["modelBinding"]["bundleManifestSha256"],
            metadata["sourceHashes"],
        )

    def _verified_handle(self, name: str):
        cpu = self._cpu_clone(f"{name}-cpu")
        cuda = self._cuda_clone(f"{name}-cuda")
        report = self.root / name / "calibration.json"
        compare_fixed_match_backends(cpu, cuda, report)
        actor_sha, manifest_sha, source_hashes = self._expected_report_bindings()
        verification = load_verified_fixed_match_backend_calibration(
            report,
            cpu,
            cuda,
            expected_actor_checkpoint_sha256=actor_sha,
            expected_bundle_manifest_sha256=manifest_sha,
            expected_source_hashes=source_hashes,
        )
        return verification, report, cpu, cuda

    def _replace_with_identical_bytes(self, path: Path) -> None:
        replacement = path.with_name(f".{path.name}.identical-replacement")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)

    def _rewrite_report(self, source: Path, name: str, mutate) -> Path:
        report = json.loads(source.read_text(encoding="utf-8"))
        mutate(report)
        destination = self.root / "rewritten-reports" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(report) + b"\n")
        Path(f"{destination}.sha256").write_bytes(
            f"{_sha256(destination)}  {destination.name}\n".encode("ascii")
        )
        return destination

    def test_pass_emits_canonical_hash_bound_detailed_report(self) -> None:
        cuda = self._cuda_clone("pass")
        output = self.root / "reports" / "calibration.json"
        result = compare_fixed_match_backends(self.cpu, cuda, output)
        self.assertEqual(result.report_sha256, _sha256(output))
        self.assertEqual(
            Path(f"{output}.sha256").read_text(encoding="ascii"),
            f"{result.report_sha256}  {output.name}\n",
        )
        raw = output.read_bytes()
        report = json.loads(raw)
        self.assertEqual(raw, canonical_json_bytes(report) + b"\n")
        self.assertEqual(report["format"], CALIBRATION_FORMAT)
        self.assertEqual(report["result"], "pass")
        self.assertEqual(
            report["collectionBinding"]["matchCounts"],
            {str(player_count): 1 for player_count in range(4, 11)},
        )
        self.assertEqual(report["inputs"]["cpu"]["npzSha256"], _sha256(self.cpu))
        self.assertEqual(report["inputs"]["cuda"]["npzSha256"], _sha256(cuda))
        self.assertNotEqual(
            report["inputs"]["cpu"]["torchVersion"],
            report["inputs"]["cuda"]["torchVersion"],
        )
        self.assertNotEqual(
            report["inputs"]["cpu"]["numpyVersion"],
            report["inputs"]["cuda"]["numpyVersion"],
        )
        self.assertEqual(report["inputs"]["cuda"]["torchVersion"], "2.7.1+cu118")
        self.assertEqual(report["inputs"]["cuda"]["numpyVersion"], "2.2.4")
        self.assertFalse(report["inputs"]["cpu"]["cudaAvailable"])
        self.assertTrue(report["inputs"]["cuda"]["cudaAvailable"])
        self.assertIn(
            "execution.torchVersion",
            report["comparisonContract"]["allowedMetadataDifferencePaths"],
        )
        self.assertIn(
            "execution.numpyVersion",
            report["comparisonContract"]["allowedMetadataDifferencePaths"],
        )
        old = report["selectedActionOldLogProbabilityDifference"]
        self.assertEqual(old["tolerance"], OLD_ACTION_LOG_PROBABILITY_MAX_ABS)
        self.assertGreater(old["total"]["count"], 0)
        self.assertGreater(old["total"]["differingCount"], 0)
        self.assertLessEqual(
            old["total"]["maxAbsDifference"],
            OLD_ACTION_LOG_PROBABILITY_MAX_ABS,
        )
        self.assertEqual(set(old["byPlayerCount"]), {f"p{p}" for p in range(4, 11)})
        self.assertEqual(set(old["byShard"]), {"0"})
        self.assertIn("actions", report["exactArrays"]["names"])
        self.assertIn("privileged_states", report["exactArrays"]["names"])
        self.assertEqual(
            report["modelAndSourceBinding"]["actorCheckpointSha256"],
            json.loads(Path(f"{self.cpu}.metadata.json").read_text(encoding="utf-8"))[
                "modelBinding"
            ]["actorCheckpointSha256"],
        )
        actor_sha, manifest_sha, source_hashes = self._expected_report_bindings()
        self.assertEqual(
            load_fixed_match_backend_calibration_report(
                output,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            ),
            result.report_sha256,
        )

    def test_action_and_public_state_drift_fail(self) -> None:
        def action_drift(metadata, arrays) -> None:
            del metadata
            valid = arrays["valid_masks"]
            legal_counts = arrays["legal_masks"].sum(axis=-1)
            row, step = (
                int(value)
                for value in np.argwhere(valid & (legal_counts > 1))[0]
            )
            legal = np.flatnonzero(arrays["legal_masks"][row, step])
            current = int(arrays["actions"][row, step])
            arrays["actions"][row, step] = int(
                next(value for value in legal if int(value) != current)
            )

        action = self._cuda_clone("action-drift", action_drift)
        with self.assertRaisesRegex(ValueError, "actions drifted"):
            compare_fixed_match_backends(
                self.cpu, action, self.root / "action-drift.json"
            )

        def state_drift(metadata, arrays) -> None:
            del metadata
            row, step = (int(value) for value in np.argwhere(arrays["valid_masks"])[0])
            arrays["rank_features"][row, step, 0, 0] += np.float32(1.0e-4)

        state = self._cuda_clone("state-drift", state_drift)
        with self.assertRaisesRegex(ValueError, "rank_features drifted"):
            compare_fixed_match_backends(
                self.cpu, state, self.root / "state-drift.json"
            )

    def test_old_log_probability_tolerance_failure(self) -> None:
        def excessive_drift(metadata, arrays) -> None:
            del metadata
            valid = arrays["valid_masks"]
            row, step = (
                int(value)
                for value in np.argwhere(
                    valid & (arrays["old_action_log_probs"] < -0.05)
                )[0]
            )
            arrays["old_action_log_probs"][row, step] -= np.float32(4.0e-5)
            arrays["selected_action_probabilities"][row, step] = np.exp(
                float(arrays["old_action_log_probs"][row, step])
            )

        cuda = self._cuda_clone("tolerance", excessive_drift)
        with self.assertRaisesRegex(ValueError, "old_action_log_probs exceeds"):
            compare_fixed_match_backends(
                self.cpu, cuda, self.root / "tolerance.json"
            )

    def test_metadata_and_hash_failures(self) -> None:
        corrupt_sidecar = self._cuda_clone("bad-sidecar")
        Path(f"{corrupt_sidecar}.sha256").write_bytes(
            f"{'0' * 64}  {corrupt_sidecar.name}\n".encode("ascii")
        )
        with self.assertRaisesRegex(ValueError, "checksum sidecar"):
            compare_fixed_match_backends(
                self.cpu,
                corrupt_sidecar,
                self.root / "bad-sidecar.json",
            )

        def manifest_drift(metadata, arrays) -> None:
            del arrays
            metadata["modelBinding"]["bundleManifestSha256"] = "f" * 64

        metadata_drift = self._cuda_clone("metadata-drift", manifest_drift)
        with self.assertRaisesRegex(ValueError, "metadata differs outside"):
            compare_fixed_match_backends(
                self.cpu,
                metadata_drift,
                self.root / "metadata-drift.json",
            )

        def false_cpu_cuda_availability(metadata, arrays) -> None:
            del arrays
            metadata["execution"]["cudaAvailable"] = True

        false_cpu = self._cpu_clone(
            "false-cpu-cuda-availability", false_cpu_cuda_availability
        )
        valid_cuda = self._cuda_clone("valid-cuda-for-cpu-role-check")
        with self.assertRaisesRegex(ValueError, "cpu cudaAvailable must be false"):
            compare_fixed_match_backends(
                false_cpu,
                valid_cuda,
                self.root / "false-cpu-cuda-availability.json",
            )

        def empty_numpy_version(metadata, arrays) -> None:
            del arrays
            metadata["execution"]["numpyVersion"] = ""

        empty_version = self._cuda_clone("empty-numpy-version", empty_numpy_version)
        with self.assertRaisesRegex(ValueError, "library versions are invalid"):
            compare_fixed_match_backends(
                self.cpu,
                empty_version,
                self.root / "empty-numpy-version.json",
            )

        def non_cuda_torch_build(metadata, arrays) -> None:
            del arrays
            metadata["execution"]["torchVersion"] = "2.7.1+cpu"

        wrong_torch_build = self._cuda_clone(
            "non-cuda-torch-build", non_cuda_torch_build
        )
        with self.assertRaisesRegex(ValueError, "CUDA torch build"):
            compare_fixed_match_backends(
                self.cpu,
                wrong_torch_build,
                self.root / "non-cuda-torch-build.json",
            )

        def policy_numerics_drift(metadata, arrays) -> None:
            del arrays
            metadata["execution"]["policyNumerics"]["mhaFastpathEnabled"] = True

        wrong_policy_numerics = self._cuda_clone(
            "policy-numerics-drift", policy_numerics_drift
        )
        with self.assertRaisesRegex(ValueError, "policy numerics"):
            compare_fixed_match_backends(
                self.cpu,
                wrong_policy_numerics,
                self.root / "policy-numerics-drift.json",
            )

    def test_input_replacement_during_validation_fails_without_report(self) -> None:
        cpu = self._cpu_clone("snapshot-race-cpu")
        cuda = self._cuda_clone("snapshot-race-cuda")
        output = self.root / "snapshot-race" / "calibration.json"
        original_loader = backend_comparator.load_v4_dataset_npz
        replaced = False

        def replace_caller_after_snapshot(snapshot_path: Path):
            nonlocal replaced
            dataset = original_loader(snapshot_path)
            if not replaced:
                replaced = True
                cpu.write_bytes(cpu.read_bytes() + b"changed")
            return dataset

        with mock.patch.object(
            backend_comparator,
            "load_v4_dataset_npz",
            side_effect=replace_caller_after_snapshot,
        ):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                compare_fixed_match_backends(cpu, cuda, output)
        self.assertFalse(output.exists())
        self.assertFalse(Path(f"{output}.sha256").exists())

    def test_verified_handle_captures_exact_ten_file_inventory(self) -> None:
        verification, report, cpu, cuda = self._verified_handle(
            "verified-ten-file-inventory"
        )
        expected = {
            report,
            Path(f"{report}.sha256"),
            cpu,
            Path(f"{cpu}.sha256"),
            Path(f"{cpu}.metadata.json"),
            Path(f"{cpu}.metadata.json.sha256"),
            cuda,
            Path(f"{cuda}.sha256"),
            Path(f"{cuda}.metadata.json"),
            Path(f"{cuda}.metadata.json.sha256"),
        }
        self.assertEqual(set(verification.artifact_paths), expected)
        self.assertEqual(len(verification.artifact_paths), 10)
        verification.recheck_unchanged()

    def test_verified_handle_rejects_identical_report_replacement(self) -> None:
        verification, report, _, _ = self._verified_handle(
            "verified-identical-report-replacement"
        )
        self._replace_with_identical_bytes(report)
        with self.assertRaisesRegex(ValueError, "changed after immutable verification"):
            verification.recheck_unchanged()

    def test_verified_handle_rejects_report_sidecar_swap(self) -> None:
        verification, report, _, _ = self._verified_handle(
            "verified-report-sidecar-swap"
        )
        self._replace_with_identical_bytes(Path(f"{report}.sha256"))
        with self.assertRaisesRegex(ValueError, "changed after immutable verification"):
            verification.recheck_unchanged()

    def test_verified_handle_rejects_cpu_and_cuda_artifact_swaps(self) -> None:
        for role, selector in (
            ("cpu", lambda cpu, cuda: cpu),
            ("cuda", lambda cpu, cuda: Path(f"{cuda}.metadata.json.sha256")),
        ):
            with self.subTest(role=role):
                verification, _, cpu, cuda = self._verified_handle(
                    f"verified-{role}-artifact-swap"
                )
                self._replace_with_identical_bytes(selector(cpu, cuda))
                with self.assertRaisesRegex(
                    ValueError, "changed after immutable verification"
                ):
                    verification.recheck_unchanged()

    def test_identity_and_equal_nonzero_policy_suffix_tampering_fail(self) -> None:
        def identity_drift(metadata, arrays) -> None:
            del arrays
            metadata["shard"]["identitySha256"] = "f" * 64

        identity = self._cuda_clone("identity-drift", identity_drift)
        with self.assertRaisesRegex(ValueError, "identitySha256"):
            compare_fixed_match_backends(
                self.cpu, identity, self.root / "identity-drift.json"
            )

        def nonzero_suffix(metadata, arrays) -> None:
            del metadata
            invalid = ~arrays["valid_masks"]
            self.assertGreater(int(invalid.sum()), 0)
            for name in (
                "old_action_log_probs",
                "selected_action_probabilities",
                "policy_entropies",
            ):
                arrays[name][invalid] = np.asarray(0.125, dtype=arrays[name].dtype)

        cpu = self._cpu_clone("equal-nonzero-suffix-cpu", nonzero_suffix)
        cuda = self._cuda_clone("equal-nonzero-suffix-cuda", nonzero_suffix)
        with self.assertRaisesRegex(ValueError, "invalid suffix.*positive zero"):
            compare_fixed_match_backends(
                cpu, cuda, self.root / "equal-nonzero-suffix.json"
            )

    def test_report_loader_rejects_wrong_binding_and_canonical_tampering(self) -> None:
        cuda = self._cuda_clone("loader-valid")
        output = self.root / "loader-valid.json"
        compare_fixed_match_backends(self.cpu, cuda, output)
        actor_sha, manifest_sha, source_hashes = self._expected_report_bindings()
        with self.assertRaisesRegex(ValueError, "model/source binding"):
            load_fixed_match_backend_calibration_report(
                output,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256="0" * 64,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

        failed = self._rewrite_report(
            output,
            "failed.json",
            lambda report: report.__setitem__("result", "fail"),
        )
        with self.assertRaisesRegex(ValueError, "canonical pass"):
            load_fixed_match_backend_calibration_report(
                failed,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

        wrong_comparator = self._rewrite_report(
            output,
            "wrong-comparator.json",
            lambda report: report.__setitem__(
                "comparatorSourceSha256", "0" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "current comparator source"):
            load_fixed_match_backend_calibration_report(
                wrong_comparator,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

        wrong_sources = dict(source_hashes)
        wrong_sources["gpu-training/v4_env.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model/source binding"):
            load_fixed_match_backend_calibration_report(
                output,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=wrong_sources,
            )

        missing_evidence = self._rewrite_report(
            output,
            "missing-p10.json",
            lambda report: report["policyNumericDifferences"][
                "policy_entropies"
            ]["byPlayerCount"].__delitem__("p10"),
        )
        with self.assertRaisesRegex(ValueError, "evidence"):
            load_fixed_match_backend_calibration_report(
                missing_evidence,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

    def test_report_loader_recomputes_actual_inputs_and_exact_inventory(self) -> None:
        cuda = self._cuda_clone("loader-recompute-valid")
        output = self.root / "loader-recompute-valid.json"
        result = compare_fixed_match_backends(self.cpu, cuda, output)
        actor_sha, manifest_sha, source_hashes = self._expected_report_bindings()
        self.assertEqual(
            load_fixed_match_backend_calibration_report(
                output,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            ),
            result.report_sha256,
        )

        forged_binding = self._rewrite_report(
            output,
            "forged-valid-looking-input-binding.json",
            lambda report: report["inputs"]["cpu"].__setitem__(
                "npzSha256", "f" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "recomputed from the supplied"):
            load_fixed_match_backend_calibration_report(
                forged_binding,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

        def delete_exact_array(report: dict[str, object]) -> None:
            exact = report["exactArrays"]
            assert isinstance(exact, dict)
            names = exact["names"]
            hashes = exact["sha256ByName"]
            assert isinstance(names, list)
            assert isinstance(hashes, dict)
            names.remove("source_decision_indices")
            hashes.pop("source_decision_indices")
            exact["count"] = len(names)
            exact["compositeSha256"] = hashlib.sha256(
                canonical_json_bytes(dict(sorted(hashes.items())))
            ).hexdigest()

        missing_exact = self._rewrite_report(
            output,
            "missing-exact-array.json",
            delete_exact_array,
        )
        with self.assertRaisesRegex(ValueError, "exact-array binding is invalid"):
            load_fixed_match_backend_calibration_report(
                missing_exact,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

    def test_report_loader_rejects_inconsistent_difference_aggregates(self) -> None:
        cuda = self._cuda_clone("loader-aggregate-valid")
        output = self.root / "loader-aggregate-valid.json"
        compare_fixed_match_backends(self.cpu, cuda, output)
        actor_sha, manifest_sha, source_hashes = self._expected_report_bindings()

        def zero_differing_with_nonzero_magnitude(report: dict[str, object]) -> None:
            record = report["policyNumericDifferences"]["policy_entropies"]
            record["total"]["differingCount"] = 0
            record["byShard"]["0"]["differingCount"] = 0

        inconsistent_zero = self._rewrite_report(
            output,
            "inconsistent-zero-difference.json",
            zero_differing_with_nonzero_magnitude,
        )
        with self.assertRaisesRegex(ValueError, "statistics are invalid"):
            load_fixed_match_backend_calibration_report(
                inconsistent_zero,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

        def zero_mean_with_differences(report: dict[str, object]) -> None:
            record = report["policyNumericDifferences"]["policy_entropies"]
            record["total"]["meanAbsDifference"] = 0.0
            record["byShard"]["0"]["meanAbsDifference"] = 0.0

        inconsistent_mean = self._rewrite_report(
            output,
            "inconsistent-nonzero-difference-mean.json",
            zero_mean_with_differences,
        )
        with self.assertRaisesRegex(ValueError, "statistics are invalid"):
            load_fixed_match_backend_calibration_report(
                inconsistent_mean,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

        def break_player_aggregation(report: dict[str, object]) -> None:
            record = report["policyNumericDifferences"]["policy_entropies"]
            rows = record["byPlayerCount"]
            selected = next(
                row
                for row in rows.values()
                if row["differingCount"] < row["count"]
            )
            selected["differingCount"] += 1

        inconsistent_players = self._rewrite_report(
            output,
            "inconsistent-player-aggregation.json",
            break_player_aggregation,
        )
        with self.assertRaisesRegex(ValueError, "coverage is inconsistent"):
            load_fixed_match_backend_calibration_report(
                inconsistent_players,
                self.cpu,
                cuda,
                expected_actor_checkpoint_sha256=actor_sha,
                expected_bundle_manifest_sha256=manifest_sha,
                expected_source_hashes=source_hashes,
            )

    def test_unclassified_array_difference_and_no_overwrite_fail(self) -> None:
        def unclassified_drift(metadata, arrays) -> None:
            del metadata
            row, step = (
                int(value) for value in np.argwhere(arrays["valid_masks"])[0]
            )
            arrays["source_decision_indices"][row, step] += np.int64(1)

        unclassified = self._cuda_clone("unclassified-drift", unclassified_drift)
        with self.assertRaisesRegex(ValueError, "source_decision_indices drifted"):
            compare_fixed_match_backends(
                self.cpu,
                unclassified,
                self.root / "unclassified-drift.json",
            )

        cuda = self._cuda_clone("immutable")
        output = self.root / "immutable.json"
        compare_fixed_match_backends(self.cpu, cuda, output)
        snapshots = {
            path: path.read_bytes()
            for path in (output, Path(f"{output}.sha256"))
        }
        with self.assertRaises(FileExistsError):
            compare_fixed_match_backends(self.cpu, cuda, output)
        for path, payload in snapshots.items():
            self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
