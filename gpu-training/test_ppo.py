from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_dataset import ACTION_COUNT, OBSERVATION_FEATURES, load_ppo_rollouts


class PpoDatasetTest(unittest.TestCase):
    def test_interleaved_trajectories_receive_independent_returns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rollout.ndjson"
            model_sha256 = "a" * 64
            manifest = {
                "type": "manifest",
                "format": "dalmuti-ppo-ndjson",
                "formatVersion": 1,
                "observation": {
                    "version": 2,
                    "featureCount": OBSERVATION_FEATURES,
                },
                "actionSpace": {"size": ACTION_COUNT},
                "behaviorModel": {"sha256": model_sha256},
            }

            def sample(
                trajectory_id: str,
                reward: float,
                terminal: bool,
            ) -> dict:
                return {
                    "type": "sample",
                    "trajectoryId": trajectory_id,
                    "observation": [0.0] * OBSERVATION_FEATURES,
                    "legalActionIndices": [0, 1],
                    "actionIndex": 0,
                    "oldLogProbability": float(np.log(0.5)),
                    "oldValue": 0.0,
                    "reward": reward,
                    "terminal": terminal,
                    "forced": False,
                    "policyVersion": f"sha256:{model_sha256}",
                }

            records = [
                manifest,
                sample("a", 0.0, False),
                sample("b", 0.0, False),
                sample("a", 1.0, True),
                sample("b", -1.0, True),
            ]
            path.write_text(
                "".join(f"{json.dumps(record)}\n" for record in records),
                encoding="utf-8",
            )
            loaded = load_ppo_rollouts(
                [str(path)],
                gamma=1.0,
                gae_lambda=1.0,
            )

            np.testing.assert_allclose(
                loaded.advantages,
                np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
            )
            np.testing.assert_allclose(
                loaded.returns,
                loaded.advantages,
            )
            self.assertEqual(loaded.trajectory_count, 2)


if __name__ == "__main__":
    unittest.main()
