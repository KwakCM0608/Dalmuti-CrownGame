from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_dataset import ACTION_COUNT, OBSERVATION_FEATURES, load_ppo_rollouts
from run_gpu_ppo import create_unique_output_directory


class PpoCoreUpgradeTest(unittest.TestCase):
    def write_rollout(self, directory: str, samples: list[dict]) -> Path:
        path = Path(directory) / "rollout.ndjson"
        model_sha256 = "b" * 64
        manifest = {
            "type": "manifest",
            "format": "dalmuti-ppo-ndjson",
            "formatVersion": 1,
            "environment": {"playerCount": 4},
            "observation": {
                "version": 2,
                "featureCount": OBSERVATION_FEATURES,
            },
            "actionSpace": {"size": ACTION_COUNT},
            "behaviorModel": {"sha256": model_sha256},
            "behaviorPolicy": {
                "sampling": "softmax",
                "temperature": 1.25,
            },
        }
        for sample in samples:
            sample.update(
                {
                    "type": "sample",
                    "trajectoryId": "trajectory",
                    "observation": [0.0] * OBSERVATION_FEATURES,
                    "legalActionIndices": [0, 1],
                    "actionIndex": 0,
                    "oldLogProbability": float(np.log(0.5)),
                    "policyVersion": f"sha256:{model_sha256}",
                }
            )
        path.write_text(
            "".join(
                f"{json.dumps(record)}\n"
                for record in [manifest, *samples]
            ),
            encoding="utf-8",
        )
        return path

    def test_forced_steps_are_skipped_only_for_policy_credit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_rollout(
                temporary,
                [
                    {
                        "oldValue": 0.0,
                        "reward": 0.0,
                        "terminal": False,
                        "forced": False,
                        "finishPlace": 1,
                    },
                    {
                        "oldValue": 0.25,
                        "reward": 1.0,
                        "terminal": True,
                        "forced": True,
                        "finishPlace": 1,
                    },
                ],
            )
            loaded = load_ppo_rollouts(
                [str(path)],
                gamma=0.5,
                gae_lambda=1.0,
                skip_forced_policy_time=True,
            )

            np.testing.assert_allclose(loaded.returns, [0.5, 1.0])
            np.testing.assert_allclose(loaded.advantages, [1.0, 0.75])
            self.assertTrue(loaded.forced[1])
            self.assertEqual(loaded.behavior_temperature, 1.25)

    def test_undiscounted_returns_include_terminal_rank_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_rollout(
                temporary,
                [
                    {
                        "oldValue": 0.2,
                        "reward": 0.0,
                        "terminal": False,
                        "forced": False,
                        "finishPlace": 1,
                    },
                    {
                        "oldValue": 0.6,
                        "reward": 0.0,
                        "terminal": True,
                        "forced": True,
                        "finishPlace": 1,
                    },
                ],
            )
            loaded = load_ppo_rollouts(
                [str(path)],
                gamma=1.0,
                gae_lambda=1.0,
                skip_forced_policy_time=True,
                terminal_rank_auxiliary_coefficient=0.05,
            )

            np.testing.assert_allclose(
                loaded.rank_auxiliary_rewards,
                [0.0, 0.05],
            )
            np.testing.assert_allclose(
                loaded.returns,
                [0.05, 0.05],
                atol=1.0e-7,
            )
            np.testing.assert_allclose(
                loaded.advantages,
                [-0.15, -0.55],
                atol=1.0e-7,
            )
            self.assertEqual(
                loaded.terminal_rank_auxiliary_coefficient,
                0.05,
            )

    def test_legacy_manifest_normalizes_temperature_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_rollout(
                temporary,
                [
                    {
                        "oldValue": 0.0,
                        "reward": 1.0,
                        "terminal": True,
                        "forced": False,
                        "finishPlace": 1,
                    },
                ],
            )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            records[0].pop("behaviorPolicy")
            path.write_text(
                "".join(f"{json.dumps(record)}\n" for record in records),
                encoding="utf-8",
            )

            loaded = load_ppo_rollouts([str(path)])
            self.assertEqual(loaded.behavior_temperature, 1.0)
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parent
                        / "verify_ppo_data.py"
                    ),
                    "--data",
                    str(path),
                    "--rollout-temperature",
                    "1.25",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "rollout-temperature does not match",
                rejected.stderr,
            )

    def test_training_output_directory_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model"
            create_unique_output_directory(output)
            with self.assertRaises(FileExistsError):
                create_unique_output_directory(output)

    def test_packaging_includes_every_completed_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_directory = root / "model"
            results_directory = root / "results"
            model_directory.mkdir()
            required_files = (
                "checkpoint.pt",
                "actor-critic-weights.json",
                "ppo-metadata.json",
                "hardware-report.json",
                "data-verification.json",
                "training.log",
            )
            for filename in required_files:
                (model_directory / filename).write_text(
                    filename,
                    encoding="utf-8",
                )
            metrics = [{"epoch": 1}, {"epoch": 2}]
            (model_directory / "training-metrics.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )
            for epoch in (1, 2):
                directory = (
                    model_directory / "checkpoints" / f"epoch-{epoch:02d}"
                )
                directory.mkdir(parents=True)
                for filename in (
                    "checkpoint.pt",
                    "actor-critic-weights.json",
                    "metrics.json",
                ):
                    (directory / filename).write_text(
                        f"{epoch}:{filename}",
                        encoding="utf-8",
                    )

            subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parent
                        / "package_ppo_results.py"
                    ),
                    "--model-dir",
                    str(model_directory),
                    "--results-dir",
                    str(results_directory),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            archive = results_directory / "model-result.zip"
            with zipfile.ZipFile(archive) as packaged:
                names = set(packaged.namelist())
                self.assertIn(
                    "checkpoints/epoch-01/actor-critic-weights.json",
                    names,
                )
                self.assertIn(
                    "checkpoints/epoch-02/checkpoint.pt",
                    names,
                )
                manifest = json.loads(
                    packaged.read("result-manifest.json")
                )
            self.assertEqual(manifest["completedEpochs"], 2)
            self.assertEqual(
                manifest["epochCheckpointDirectories"],
                ["checkpoints/epoch-01", "checkpoints/epoch-02"],
            )
            archive_bytes = archive.read_bytes()
            repeated = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parent
                        / "package_ppo_results.py"
                    ),
                    "--model-dir",
                    str(model_directory),
                    "--results-dir",
                    str(results_directory),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(archive.read_bytes(), archive_bytes)


if __name__ == "__main__":
    unittest.main()
