from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import test_v3_ppo_pipeline as pipeline_fixture
from run_gpu_v3_ppo import EXPECTED_DETERMINISM, EXPECTED_PATH_POLICY
from v3_ppo_result_contract import (
    STRICT_PROVENANCE_MODE,
    load_source_contract,
    validate_result_directory,
)


ROOT = pipeline_fixture.ROOT


class V3PpoResultContractTests(unittest.TestCase):
    def _run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def _prepare_output(self, root: Path) -> tuple[Path, Path, Path]:
        model, rollout = pipeline_fixture.V3PpoPipelineTests().create_fixture(root)
        output = root / "fresh-v3-contract-run"
        standalone_report = root / "standalone-data-verification.json"
        verified = self._run(
            [
                str(ROOT / "verify_v3_ppo_data.py"),
                "--data",
                str(rollout),
                "--behavior-model",
                str(model),
                "--rollout-temperature",
                "1.25",
                "--output",
                str(standalone_report),
            ]
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)
        trained = self._run(
            [
                str(ROOT / "train_v3_ppo.py"),
                "--data",
                str(rollout),
                "--behavior-model",
                str(model),
                "--output",
                str(output),
                "--data-verification-output",
                str(output / "data-verification.json"),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--rollout-temperature",
                "1.25",
                "--target-kl",
                "0",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(trained.returncode, 0, trained.stderr)
        self.assertEqual(
            self._json(standalone_report),
            self._json(output / "data-verification.json"),
        )
        (output / "hardware-report.json").write_text("{}\n", encoding="utf-8")
        (output / "training.log").write_text(trained.stdout, encoding="utf-8")
        return model, rollout, output

    def _package_legacy(self, output: Path, results: Path) -> tuple[Path, Path]:
        packaged = self._run(
            [
                str(ROOT / "package_v3_ppo_results.py"),
                "--model-dir",
                str(output),
                "--results-dir",
                str(results),
                "--allow-legacy-smoke",
            ]
        )
        self.assertEqual(packaged.returncode, 0, packaged.stderr)
        archive = results / f"{output.name}-result.zip"
        return archive, archive.with_name(f"{archive.name}.sha256")

    @staticmethod
    def _json(path: Path) -> dict:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object: {path}")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _prepare_strict_contract(self, root: Path) -> tuple[Path, dict]:
        model, rollout, output = self._prepare_output(root)
        bundle = root / "strict-source-bundle"
        data_root = bundle / "data"
        data_root.mkdir(parents=True)
        behavior = bundle / "behavior-model.json"
        shutil.copyfile(model, behavior)
        parent_sha256 = hashlib.sha256(behavior.read_bytes()).hexdigest()

        data_verification_path = output / "data-verification.json"
        metadata_path = output / "v3-ppo-metadata.json"
        hardware_path = output / "hardware-report.json"
        data_verification = self._json(data_verification_path)
        metadata = self._json(metadata_path)
        base_counts = {
            "learnerSamples": data_verification["samples"],
            "forcedSamples": data_verification["forcedSamples"],
            "nonForcedSamples": data_verification["policySamples"],
            "environmentDecisions": data_verification["samples"],
        }
        source_data: list[dict] = []
        rollout_inventory: list[dict] = []
        for player_count in range(4, 11):
            copied = data_root / f"p{player_count}.ndjson"
            shutil.copyfile(rollout, copied)
            item = {
                "path": f"data/{copied.name}",
                "bytes": copied.stat().st_size,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }
            source_data.append(item)
            rollout_inventory.append(
                {
                    "filename": copied.name,
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "playerCount": player_count,
                    "acts": 1,
                    "seed": player_count,
                    "temperature": 1.25,
                    "episodes": 1,
                    **base_counts,
                }
            )

        algorithm = {
            "epochs": 12,
            "batchSize": 4096,
            "learningRate": 0.0001,
            "weightDecay": 0.00001,
            "gamma": 1,
            "gaeLambda": 1,
            "skipForcedPolicyTime": True,
            "rolloutTemperature": 1.25,
            "clipCoefficient": 0.2,
            "valueCoefficient": 0.5,
            "entropyCoefficient": 0.01,
            "maxGradientNorm": 0.5,
            "targetKl": 0.015,
            "bindingTolerance": 0.00002,
            "behaviorBindingBatchSize": 8192,
            "loaderWorkers": 7,
            "device": "cuda",
            "seed": 202608061,
        }
        run_config = {
            "format": "dalmuti-v3-ppo-gpu-run-config",
            "version": 2,
            "parentModelSha256": parent_sha256,
            "rolloutTemperature": 1.25,
            "algorithm": algorithm,
            "allowedTerminalRankAuxiliaryCoefficients": [0, 0.05],
            "determinism": dict(EXPECTED_DETERMINISM),
            "pathPolicy": dict(EXPECTED_PATH_POLICY),
            "requiredCommandArguments": [
                "--output",
                "models/<fresh-v3-run>",
                "--results-dir",
                "returned/<fresh-v3-run>",
                "--epochs",
                "12",
                "--batch-size",
                "4096",
                "--learning-rate",
                "0.0001",
                "--weight-decay",
                "0.00001",
                "--gamma",
                "1",
                "--gae-lambda",
                "1",
                "--skip-forced-policy-time",
                "--terminal-rank-auxiliary-coefficient",
                "<0-or-0.05>",
                "--rollout-temperature",
                "1.25",
                "--clip-coefficient",
                "0.2",
                "--value-coefficient",
                "0.5",
                "--entropy-coefficient",
                "0.01",
                "--max-gradient-norm",
                "0.5",
                "--target-kl",
                "0.015",
                "--binding-tolerance",
                "0.00002",
                "--behavior-binding-batch-size",
                "8192",
                "--loader-workers",
                "7",
                "--seed",
                "202608061",
                "--device",
                "cuda",
            ],
        }
        run_config_path = bundle / "gpu-run-config.json"
        self._write_json(run_config_path, run_config)

        def entry(path: Path) -> dict:
            return {
                "path": path.relative_to(bundle).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        file_entries = [
            entry(behavior),
            *(entry(path) for path in sorted(data_root.iterdir())),
            entry(run_config_path),
        ]
        totals = {
            "episodes": len(rollout_inventory),
            **{
                key: value * len(rollout_inventory)
                for key, value in base_counts.items()
            },
        }
        bundle_manifest = {
            "format": "dalmuti-v3-ppo-gpu-bundle",
            "version": 1,
            "createdAt": "2026-08-01T00:00:00+00:00",
            "parentModel": {
                "filename": model.name,
                "format": "dalmuti-action-conditioned-actor-critic",
                "bytes": behavior.stat().st_size,
                "sha256": parent_sha256,
            },
            "behaviorModelSha256": parent_sha256,
            "observationSchemaVersion": 2,
            "observationFeatures": 172,
            "actionCatalogueVersion": 1,
            "actionCount": 236,
            "rollouts": rollout_inventory,
            "dataCounts": totals,
            "files": file_entries,
            "totalBytes": sum(item["bytes"] for item in file_entries),
        }
        bundle_manifest_path = bundle / "bundle-manifest.json"
        self._write_json(bundle_manifest_path, bundle_manifest)
        source = load_source_contract(
            bundle_manifest_path,
            run_config_path,
            verify_source_files=True,
        )

        total_samples = totals["learnerSamples"]
        total_trajectories = data_verification["trajectories"] * len(source_data)
        source_paths = [str((bundle / item["path"]).resolve()) for item in source_data]
        data_verification.update(
            {
                "files": source_paths,
                "sourceFiles": [
                    {"path": path, "bytes": item["bytes"], "sha256": item["sha256"]}
                    for path, item in zip(source_paths, source_data, strict=True)
                ],
                "samples": total_samples,
                "trajectories": total_trajectories,
                "observationShape": [total_samples, 172],
                "legalMaskShape": [total_samples, 236],
                "forcedSamples": totals["forcedSamples"],
                "policySamples": totals["nonForcedSamples"],
                "terminalSamples": total_trajectories,
            }
        )
        self._write_json(data_verification_path, data_verification)

        deterministic_runtime = {
            "algorithmsEnabled": True,
            "warnOnly": False,
            "seed": algorithm["seed"],
            "pythonHashSeed": str(algorithm["seed"]),
            "cublasWorkspaceConfig": ":4096:8",
            "cudnnDeterministic": True,
            "cudnnBenchmark": False,
            "cudaMatmulAllowTf32": False,
            "cudnnAllowTf32": False,
        }
        gpu = {
            "index": 0,
            "name": "Test CUDA GPU",
            "computeCapability": "9.0",
            "totalMemoryBytes": 24 * 1024**3,
            "multiProcessorCount": 64,
            "uuid": "GPU-test-fixture",
        }
        metadata.update(
            {
                "behaviorModel": str(behavior.resolve()),
                "device": "cuda",
                "cudaAvailable": True,
                "cudaDevice": gpu["name"],
                "gpuIdentity": gpu,
                "sourceProvenance": {
                    "runId": output.name,
                    "bundleManifestSha256": source["bundleManifestSha256"],
                    "runConfigSha256": source["runConfigSha256"],
                    "parentModelSha256": parent_sha256,
                },
                "deterministicRuntime": deterministic_runtime,
                "samples": total_samples,
                "trajectories": total_trajectories,
                "forcedSamples": totals["forcedSamples"],
                "policySamples": totals["nonForcedSamples"],
                "completedEpochs": 1,
                "stoppedForTargetKl": True,
                "sourceFiles": source_paths,
                "sourceData": source_data,
                "arguments": {
                    "data": source_paths,
                    "behavior_model": str(behavior.resolve()),
                    "output": str(output.resolve()),
                    "epochs": algorithm["epochs"],
                    "batch_size": algorithm["batchSize"],
                    "learning_rate": algorithm["learningRate"],
                    "weight_decay": algorithm["weightDecay"],
                    "gamma": algorithm["gamma"],
                    "gae_lambda": algorithm["gaeLambda"],
                    "skip_forced_policy_time": algorithm["skipForcedPolicyTime"],
                    "terminal_rank_auxiliary_coefficient": 0.0,
                    "rollout_temperature": algorithm["rolloutTemperature"],
                    "clip_coefficient": algorithm["clipCoefficient"],
                    "value_coefficient": algorithm["valueCoefficient"],
                    "entropy_coefficient": algorithm["entropyCoefficient"],
                    "max_gradient_norm": algorithm["maxGradientNorm"],
                    "target_kl": algorithm["targetKl"],
                    "binding_tolerance": algorithm["bindingTolerance"],
                    "seed": algorithm["seed"],
                    "device": algorithm["device"],
                    "run_id": output.name,
                    "bundle_manifest": str(bundle_manifest_path.resolve()),
                    "run_config": str(run_config_path.resolve()),
                },
            }
        )
        self._write_json(metadata_path, metadata)
        hardware = {
            "format": "dalmuti-gpu-preflight",
            "version": 1,
            "platform": "test-platform",
            "pythonVersion": sys.version,
            "pythonExecutable": sys.executable,
            "processArchitecture": "64bit",
            "cpuCount": 1,
            "numpyVersion": "test",
            "torchVersion": metadata["torchVersion"],
            "torchCudaVersion": "12.8",
            "cudnnVersion": 9000,
            "cudaAvailable": True,
            "requestedDevice": "cuda",
            "deterministicRuntime": deterministic_runtime,
            "bundleFreeDiskBytes": 8 * 1024**3,
            "nvidiaSmi": "test",
            "gpuDevices": [gpu],
        }
        self._write_json(hardware_path, hardware)
        return output, source

    @staticmethod
    def _write_checksum(archive: Path) -> Path:
        checksum = archive.with_name(f"{archive.name}.sha256")
        checksum.write_text(
            f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
            encoding="ascii",
        )
        return checksum

    def test_legacy_smoke_requires_explicit_opt_in_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, output = self._prepare_output(root)
            denied = self._run(
                [
                    str(ROOT / "package_v3_ppo_results.py"),
                    "--model-dir",
                    str(output),
                    "--results-dir",
                    str(root / "denied"),
                ]
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("strict packaging requires", denied.stderr)

            archive, checksum = self._package_legacy(output, root / "returned")
            denied_verify = self._run(
                [
                    str(ROOT / "verify_v3_ppo_results.py"),
                    "--archive",
                    str(archive),
                    "--checksum",
                    str(checksum),
                ]
            )
            self.assertNotEqual(denied_verify.returncode, 0)
            self.assertIn("strict verification requires", denied_verify.stderr)
            accepted = self._run(
                [
                    str(ROOT / "verify_v3_ppo_results.py"),
                    "--archive",
                    str(archive),
                    "--checksum",
                    str(checksum),
                    "--allow-legacy-smoke",
                    "--extract-dir",
                    str(root / "extracted"),
                ]
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            checksum.write_text(
                f"{'0' * 64}  {archive.name}\n",
                encoding="ascii",
            )
            corrupted_checksum = self._run(
                [
                    str(ROOT / "verify_v3_ppo_results.py"),
                    "--archive",
                    str(archive),
                    "--checksum",
                    str(checksum),
                    "--allow-legacy-smoke",
                ]
            )
            self.assertNotEqual(corrupted_checksum.returncode, 0)
            self.assertIn("checksum mismatch", corrupted_checksum.stderr)

    def test_packager_rejects_a_stale_selected_final_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_model, _, output = self._prepare_output(root)
            shutil.copyfile(parent_model, output / "v3-actor-critic-weights.json")
            packaged = self._run(
                [
                    str(ROOT / "package_v3_ppo_results.py"),
                    "--model-dir",
                    str(output),
                    "--results-dir",
                    str(root / "returned"),
                    "--allow-legacy-smoke",
                ]
            )
            self.assertNotEqual(packaged.returncode, 0)
            self.assertRegex(packaged.stderr, "final|model/checkpoint")

    def test_strict_result_binds_semantics_source_hashes_and_gpu_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, source = self._prepare_strict_contract(root)
            cache = source["root"] / "__pycache__"
            cache.mkdir()
            (cache / "fixture.cpython-312.pyc").write_bytes(b"unmanifested")
            with self.assertRaisesRegex(ValueError, "bytecode cache"):
                load_source_contract(
                    source["root"] / "bundle-manifest.json",
                    source["root"] / "gpu-run-config.json",
                    verify_source_files=True,
                )
            shutil.rmtree(cache)
            validation = validate_result_directory(
                output,
                source=source,
                allow_legacy_smoke=False,
                allow_manifest=False,
                expected_run_id=output.name,
            )
            self.assertEqual(validation["provenanceMode"], STRICT_PROVENANCE_MODE)

            data_path = output / "data-verification.json"
            metadata_path = output / "v3-ppo-metadata.json"
            hardware_path = output / "hardware-report.json"
            originals = {
                data_path: self._json(data_path),
                metadata_path: self._json(metadata_path),
                hardware_path: self._json(hardware_path),
            }
            mutations = []

            bad_semantics = json.loads(json.dumps(originals[data_path]))
            bad_semantics["rolloutSemanticsContract"]["sha256"] = "0" * 64
            mutations.append((data_path, bad_semantics, "semantics"))

            bad_source = json.loads(json.dumps(originals[metadata_path]))
            bad_source["sourceData"][0]["sha256"] = "0" * 64
            mutations.append((metadata_path, bad_source, "rollout provenance"))

            bad_hardware = json.loads(json.dumps(originals[hardware_path]))
            bad_hardware["gpuDevices"][0]["name"] = "Different CUDA GPU"
            mutations.append((hardware_path, bad_hardware, "GPU identity"))

            for path, mutation, message in mutations:
                with self.subTest(message=message):
                    self._write_json(path, mutation)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_result_directory(
                            output,
                            source=source,
                            allow_legacy_smoke=False,
                            allow_manifest=False,
                            expected_run_id=output.name,
                        )
                    self._write_json(path, originals[path])

            returned = root / "strict-returned"
            packaged = self._run(
                [
                    str(ROOT / "package_v3_ppo_results.py"),
                    "--model-dir",
                    str(output),
                    "--results-dir",
                    str(returned),
                    "--expected-bundle-manifest",
                    str(source["root"] / "bundle-manifest.json"),
                    "--expected-run-config",
                    str(source["root"] / "gpu-run-config.json"),
                ]
            )
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            archive = returned / f"{output.name}-result.zip"
            checksum = archive.with_name(f"{archive.name}.sha256")
            verified = self._run(
                [
                    str(ROOT / "verify_v3_ppo_results.py"),
                    "--archive",
                    str(archive),
                    "--checksum",
                    str(checksum),
                    "--expected-bundle-manifest",
                    str(source["root"] / "bundle-manifest.json"),
                    "--expected-run-config",
                    str(source["root"] / "gpu-run-config.json"),
                    "--extract-dir",
                    str(root / "strict-extracted"),
                ]
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_verifier_rejects_unmanifested_duplicate_and_symlink_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, output = self._prepare_output(root)
            archive, _ = self._package_legacy(output, root / "returned")

            cases: list[tuple[str, callable]] = []

            def add_unmanifested(source: zipfile.ZipFile, target: zipfile.ZipFile) -> None:
                for info in source.infolist():
                    target.writestr(info, source.read(info))
                target.writestr("unexpected.bin", b"not listed")

            def add_duplicate(source: zipfile.ZipFile, target: zipfile.ZipFile) -> None:
                for info in source.infolist():
                    target.writestr(info, source.read(info))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    target.writestr(
                        "v3-actor-critic-weights.json",
                        source.read("v3-actor-critic-weights.json"),
                    )

            def replace_with_symlink(source: zipfile.ZipFile, target: zipfile.ZipFile) -> None:
                for info in source.infolist():
                    if info.filename == "training.log":
                        link = zipfile.ZipInfo("training.log")
                        link.create_system = 3
                        link.external_attr = (stat.S_IFLNK | 0o777) << 16
                        target.writestr(link, "v3-ppo-metadata.json")
                    else:
                        target.writestr(info, source.read(info))

            def add_traversal(source: zipfile.ZipFile, target: zipfile.ZipFile) -> None:
                for info in source.infolist():
                    target.writestr(info, source.read(info))
                target.writestr("../escaped.bin", b"escape")

            cases.extend(
                [
                    ("unmanifested", add_unmanifested),
                    ("duplicate", add_duplicate),
                    ("symlink", replace_with_symlink),
                    ("traversal", add_traversal),
                ]
            )
            for label, mutate in cases:
                corrupted = root / f"{label}.zip"
                with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
                    corrupted, "x", compression=zipfile.ZIP_DEFLATED
                ) as target:
                    mutate(source, target)
                checksum = self._write_checksum(corrupted)
                verified = self._run(
                    [
                        str(ROOT / "verify_v3_ppo_results.py"),
                        "--archive",
                        str(corrupted),
                        "--checksum",
                        str(checksum),
                        "--allow-legacy-smoke",
                    ]
                )
                self.assertNotEqual(verified.returncode, 0, label)


if __name__ == "__main__":
    unittest.main()
