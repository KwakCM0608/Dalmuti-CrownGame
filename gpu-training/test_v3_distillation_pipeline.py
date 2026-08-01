from __future__ import annotations

import hashlib
import json
import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from actor_critic import ActorCriticNetwork, export_actor_critic_json
from package_v3_distillation_handoff import _validate_runtime_prompt
from v3_distillation_dataset import (
    group_split_mask,
    legacy_action_index_to_v3,
    legacy_legal_action_indices_to_v3,
    load_v3_distillation_data,
    v3_action_index_to_legacy,
)
from v3_ppo_dataset import legal_actions_from_observation
from verify_v3_distillation_bundle import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_SPLIT,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_TRAINING,
)
from verify_v3_distillation_results import (
    EXPECTED_GPU_ARGUMENTS,
    _verify_expected_handoff_result,
)


ROOT = Path(__file__).resolve().parent


def observation(hand: dict[int, int]) -> list[float]:
    values = [0.0] * 172
    for rank, count in hand.items():
        values[23 + rank - 1] = count / (2 if rank == 13 else rank)
    return values


class V3DistillationPipelineTests(unittest.TestCase):
    def test_gpu_prompt_disables_bytecode_before_first_python_process(self) -> None:
        prompt_path = ROOT / "PROMPT_FOR_GPU_V3_DISTILLATION.md"
        _validate_runtime_prompt(prompt_path)
        prompt = prompt_path.read_text(encoding="utf-8")
        self.assertLess(
            prompt.index("export PYTHONDONTWRITEBYTECODE=1"),
            prompt.index('"$PY"'),
        )
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid-prompt.md"
            invalid.write_text(
                '"$PY" verify.py\nexport PYTHONDONTWRITEBYTECODE=1\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "before the first Python"):
                _validate_runtime_prompt(invalid)

    def test_bridge_is_bijective_and_canonicalizes_joker_order(self) -> None:
        for v3_index in range(236):
            self.assertEqual(
                legacy_action_index_to_v3(v3_action_index_to_legacy(v3_index)),
                v3_index,
            )
        legacy = [44, 47, 48]
        v3 = legacy_legal_action_indices_to_v3(legacy)
        self.assertEqual(v3, [5, 6, 8])
        self.assertEqual(
            sorted(v3_action_index_to_legacy(index) for index in v3), legacy
        )

    def create_fixture(self, root: Path) -> tuple[Path, Path]:
        teacher_path = root / "teacher-ppo4-like.json"
        teacher = ActorCriticNetwork(hidden_sizes=(4,))
        for index, parameter in enumerate(teacher.parameters()):
            parameter.data.fill_(0.001 * (index + 1))
        export_actor_critic_json(teacher, teacher_path)
        rollout_path = root / "legacy-rollout.ndjson"
        manifest = {
            "type": "manifest",
            "format": "dalmuti-ppo-ndjson",
            "formatVersion": 1,
            "environment": {"playerCount": 4},
            "behaviorModel": {
                "sha256": "a" * 64,
                "format": "dalmuti-actor-critic",
            },
            "observation": {"version": 2, "featureCount": 172},
            "actionSpace": {"size": 506},
        }
        records: list[dict] = [manifest]
        hands = (
            {2: 2, 13: 1},
            {1: 1, 3: 2},
            {4: 3, 13: 2},
            {5: 2, 8: 1},
            {6: 4, 13: 1},
            {2: 1, 7: 3},
        )
        for index, hand in enumerate(hands, start=1):
            encoded = observation(hand)
            v3_legal = legal_actions_from_observation(encoded, f"fixture-{index}")
            legacy_legal = sorted(
                v3_action_index_to_legacy(action) for action in v3_legal
            )
            records.append(
                {
                    "type": "sample",
                    "episodeId": f"episode-{index}",
                    "trajectoryId": f"episode-{index}:player-1",
                    "observation": encoded,
                    "legalActionIndices": legacy_legal,
                    "actionIndex": legacy_legal[0],
                    "oldLogProbability": 0.0,
                    "oldValue": 0.0,
                    "reward": 0.0,
                    "terminal": True,
                    "forced": len(legacy_legal) == 1,
                    "policyVersion": f"sha256:{'a' * 64}",
                }
            )
        rollout_path.write_text(
            "".join(
                json.dumps(record, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return teacher_path, rollout_path

    def test_prepare_train_package_and_verify_cpu_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher, rollout = self.create_fixture(root)
            dataset_dir = root / "fresh-dataset"
            prepared = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "prepare_v3_distillation_data.py"),
                    "--rollout",
                    str(rollout),
                    "--teacher-model",
                    str(teacher),
                    "--output-dir",
                    str(dataset_dir),
                    "--temperature",
                    "2.5",
                    "--max-samples-per-source",
                    "6",
                    "--batch-size",
                    "3",
                    "--device",
                    "cpu",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            data_path = dataset_dir / "v3-distillation.ndjson"
            loaded = load_v3_distillation_data(
                data_path, teacher_model_path=teacher
            )
            self.assertEqual(len(loaded), 6)
            np.testing.assert_allclose(
                loaded.teacher_probabilities.sum(axis=1), 1.0, atol=1.0e-6
            )
            split_seed = None
            for seed in range(1000):
                try:
                    group_split_mask(
                        loaded.group_keys,
                        validation_fraction=0.34,
                        seed=seed,
                    )
                except ValueError:
                    continue
                split_seed = seed
                break
            self.assertIsNotNone(split_seed)
            output = root / "fresh-training"
            trained = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train_v3_distillation.py"),
                    "--data",
                    str(data_path),
                    "--teacher-model",
                    str(teacher),
                    "--output",
                    str(output),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "3",
                    "--validation-fraction",
                    "0.34",
                    "--split-seed",
                    str(split_seed),
                    "--seed",
                    "7",
                    "--device",
                    "cpu",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_distillation_results.py"),
                    "--result-dir",
                    str(output),
                    "--teacher-model",
                    str(teacher),
                    "--data",
                    str(data_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            archive = root / "packages" / "v3-warmstart-smoke.zip"
            packaged = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "package_v3_distillation_results.py"),
                    "--result-dir",
                    str(output),
                    "--output",
                    str(archive),
                    "--teacher-model",
                    str(teacher),
                    "--data",
                    str(data_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            checksum = archive.with_suffix(".zip.sha256")
            self.assertEqual(
                hashlib.sha256(archive.read_bytes()).hexdigest(),
                checksum.read_text(encoding="ascii").split()[0],
            )
            extraction = root / "fresh-extraction"
            checked = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_distillation_results.py"),
                    "--archive",
                    str(archive),
                    "--checksum",
                    str(checksum),
                    "--extract-dir",
                    str(extraction),
                    "--teacher-model",
                    str(teacher),
                    "--data",
                    str(data_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertTrue(
                (extraction / "v3-actor-critic-weights.json").is_file()
            )
            legacy = root / "legacy-no-inventory"
            shutil.copytree(extraction, legacy)
            legacy_manifest_path = legacy / "training-manifest.json"
            legacy_manifest = json.loads(
                legacy_manifest_path.read_text(encoding="utf-8")
            )
            legacy_manifest.pop("resultInventory")
            legacy_manifest_path.write_text(
                json.dumps(legacy_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            strict_legacy = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_distillation_results.py"),
                    "--result-dir",
                    str(legacy),
                    "--teacher-model",
                    str(teacher),
                    "--data",
                    str(data_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(strict_legacy.returncode, 0)
            self.assertIn("strict result inventory is required", strict_legacy.stderr)
            explicit_legacy = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_distillation_results.py"),
                    "--result-dir",
                    str(legacy),
                    "--teacher-model",
                    str(teacher),
                    "--data",
                    str(data_path),
                    "--allow-legacy-inventory",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(explicit_legacy.returncode, 0, explicit_legacy.stderr)
            (extraction / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
            unexpected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "verify_v3_distillation_results.py"),
                    "--result-dir",
                    str(extraction),
                    "--teacher-model",
                    str(teacher),
                    "--data",
                    str(data_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(unexpected.returncode, 0)
            self.assertIn("file inventory mismatch", unexpected.stderr)
            repeated = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "prepare_v3_distillation_data.py"),
                    "--rollout",
                    str(rollout),
                    "--teacher-model",
                    str(teacher),
                    "--output-dir",
                    str(dataset_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("must be fresh", repeated.stderr)

    def test_expected_handoff_rejects_gpu_argument_and_provenance_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff = root / "handoff"
            result = root / "result"
            provenance = result / "provenance"
            handoff.mkdir()
            provenance.mkdir(parents=True)
            run_id = "fixture-gpu-run-001"
            handoff_manifest = {"runId": run_id}
            run_config = {"runId": run_id}
            for name, payload in (
                ("bundle-manifest.json", json.dumps(handoff_manifest) + "\n"),
                ("gpu-run-config.json", json.dumps(run_config) + "\n"),
                ("handoff-files.sha256", "a" * 64 + "  fixture\n"),
            ):
                (handoff / name).write_text(payload, encoding="utf-8")
                shutil.copyfile(handoff / name, provenance / name)
            gpu = {
                "index": 0,
                "name": "Fixture GPU",
                "totalMemoryBytes": 1024,
                "computeCapability": "8.6",
                "multiProcessorCount": 1,
            }
            hardware = {
                "format": "dalmuti-v3-distillation-training-hardware",
                "version": 1,
                "pythonVersion": "fixture-python",
                "numpyVersion": "fixture-numpy",
                "torchVersion": "fixture-torch",
                "torchCudaVersion": "fixture-cuda",
                "cudnnVersion": 1,
                "cudaAvailable": True,
                "device": "cuda",
                "gpu": gpu,
            }
            preflight = {
                "format": "dalmuti-gpu-preflight",
                "version": 1,
                "pythonVersion": hardware["pythonVersion"],
                "numpyVersion": hardware["numpyVersion"],
                "torchVersion": hardware["torchVersion"],
                "torchCudaVersion": hardware["torchCudaVersion"],
                "cudnnVersion": hardware["cudnnVersion"],
                "cudaAvailable": True,
                "gpuDevices": [gpu],
            }
            (provenance / "hardware-report.json").write_text(
                json.dumps(preflight) + "\n", encoding="utf-8"
            )
            (provenance / "training.log").write_text(
                "epoch 001 | val KL 1.0\n", encoding="utf-8"
            )
            manifest = {
                "teacher": {
                    "filename": "ppo4-actor-critic-weights.json",
                    "sha256": EXPECTED_TEACHER_SHA256,
                    "format": "dalmuti-actor-critic",
                    "actionCount": 506,
                    "temperature": 2.5,
                },
                "dataset": {
                    "filename": "v3-distillation.ndjson",
                    "sha256": EXPECTED_DATASET_SHA256,
                    "samples": 140000,
                },
                "arguments": dict(EXPECTED_GPU_ARGUMENTS),
                "split": {
                    "groupSplitKey": EXPECTED_SPLIT["groupSplitKey"],
                    "train": {
                        "samples": EXPECTED_SPLIT["trainSamples"],
                        "uniqueGroups": EXPECTED_SPLIT["trainGroups"],
                        "sampleIdsSha256": "a" * 64,
                    },
                    "validation": {
                        "samples": EXPECTED_SPLIT["validationSamples"],
                        "uniqueGroups": EXPECTED_SPLIT["validationGroups"],
                        "sampleIdsSha256": "b" * 64,
                    },
                    "overlappingGroups": 0,
                },
                "device": "cuda",
                "hardware": hardware,
                "reproducibility": {
                    "seed": EXPECTED_TRAINING["seed"],
                    "deterministicAlgorithms": True,
                    "cudnnDeterministic": True,
                    "cudnnBenchmark": False,
                    "cublasWorkspaceConfig": ":4096:8",
                },
            }
            with mock.patch(
                "verify_v3_distillation_results.verify_bundle",
                return_value={"runId": run_id},
            ):
                report = _verify_expected_handoff_result(
                    result,
                    manifest,
                    expected_handoff=handoff,
                    teacher_model="teacher-present",
                    dataset="dataset-present",
                )
                self.assertEqual(report["runId"], run_id)
                wrong_seed = copy.deepcopy(manifest)
                wrong_seed["arguments"]["seed"] += 1
                with self.assertRaisesRegex(ValueError, "training arguments"):
                    _verify_expected_handoff_result(
                        result,
                        wrong_seed,
                        expected_handoff=handoff,
                        teacher_model="teacher-present",
                        dataset="dataset-present",
                    )
                (provenance / "handoff-files.sha256").write_text(
                    "b" * 64 + "  fixture\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    _verify_expected_handoff_result(
                        result,
                        manifest,
                        expected_handoff=handoff,
                        teacher_model="teacher-present",
                        dataset="dataset-present",
                    )


if __name__ == "__main__":
    unittest.main()
