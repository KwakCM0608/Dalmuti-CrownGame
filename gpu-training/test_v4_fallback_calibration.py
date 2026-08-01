from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from v4_fallback_calibration import (
    FORMAT,
    VERSION,
    PublicPreparedDataset,
    _pair_report,
    _unique_sweep,
    build_calibration_report,
    canonical_json_bytes,
    legal_diagnostics_from_logits,
    load_public_prepared_normal_dataset,
    parse_threshold_grid,
    route_actor_inclusive,
    trajectory_cluster_bootstrap,
    write_calibration_report_exclusive,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FallbackCalibrationUnitTests(unittest.TestCase):
    def test_routing_is_inclusive_and_matches_evaluator_contract(self) -> None:
        from v4_evaluate import CandidatePolicyRouting

        routing = CandidatePolicyRouting(
            "confidence-fallback",
            minimum_legal_logit_margin=0.75,
            minimum_top_probability=0.8,
        )
        self.assertEqual(
            routing.report_value()["thresholdComparison"],
            "all-configured-thresholds-met-inclusive",
        )
        self.assertTrue(
            route_actor_inclusive(
                legal_action_count=2,
                legal_logit_margin=0.75,
                top_probability=0.8,
                minimum_legal_logit_margin=0.75,
                minimum_top_probability=0.8,
            )
        )
        self.assertFalse(
            route_actor_inclusive(
                legal_action_count=2,
                legal_logit_margin=0.749999,
                top_probability=0.8,
                minimum_legal_logit_margin=0.75,
                minimum_top_probability=0.8,
            )
        )
        self.assertFalse(
            route_actor_inclusive(
                legal_action_count=1,
                legal_logit_margin=None,
                top_probability=1.0,
                minimum_legal_logit_margin=0.0,
                minimum_top_probability=0.0,
            )
        )

    def test_forced_rows_always_fallback(self) -> None:
        diagnostics = {
            "actions": np.array([3, 7]),
            "normalActions": np.array([3, 7]),
            "margins": np.array([np.nan, np.nan]),
            "probabilities": np.array([1.0, 1.0]),
            "legalCounts": np.array([1, 1]),
            "trajectoryIndexes": np.array([0, 1]),
            "playerCounts": np.array([4, 5]),
            "roles": np.array([0, 1]),
            "acts": np.array([1, 1]),
        }
        report = _pair_report(
            diagnostics,
            margin=0.0,
            probability=0.0,
            binding="a" * 64,
            bootstrap_resamples=20,
        )
        self.assertEqual(report["actorDecisions"], 0)
        self.assertEqual(report["fallbackDecisions"], 2)
        self.assertEqual(report["forcedFallbackDecisions"], 2)
        self.assertIsNone(report["trajectoryClusterBootstrap95"]["low"])

    def test_unique_sweep_coverage_is_monotone(self) -> None:
        values = np.array([0.1, 0.4, 0.4, 0.9, np.nan])
        forced = np.array([False, False, False, False, True])
        agreement = np.array([True, False, True, True, True])
        rows = _unique_sweep(
            values, forced, agreement, label="minimumLegalLogitMargin"
        )
        self.assertEqual(
            [row["minimumLegalLogitMargin"] for row in rows], [0.1, 0.4, 0.9]
        )
        coverage = [row["actorCoverage"] for row in rows]
        self.assertEqual(coverage, sorted(coverage, reverse=True))
        self.assertEqual(rows[1]["actorDecisions"], 3)  # inclusive at 0.4

    def test_thresholds_and_cluster_bootstrap_are_deterministic(self) -> None:
        self.assertEqual(
            parse_threshold_grid(".5,0,0.5,1", probability=True),
            (0.0, 0.5, 1.0),
        )
        selected = np.array([True, True, True, False, True, True])
        agreement = np.array([True, False, True, True, False, True])
        trajectories = np.array([0, 0, 1, 1, 2, 2])
        first = trajectory_cluster_bootstrap(
            selected, agreement, trajectories, seed=12345, resamples=500
        )
        second = trajectory_cluster_bootstrap(
            selected, agreement, trajectories, seed=12345, resamples=500
        )
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 3)
        self.assertLessEqual(first["low"], first["mean"])
        self.assertGreaterEqual(first["high"], first["mean"])

    def test_illegal_logits_have_zero_mass_and_cannot_win(self) -> None:
        logits = np.full((2, 236), -3.0)
        legal = np.zeros((2, 236), dtype=np.bool_)
        legal[0, [2, 5]] = True
        legal[1, 17] = True
        logits[0, 99] = 1_000_000.0
        logits[0, 5] = 4.0
        logits[0, 2] = 3.0
        logits[1, 17] = -7.0
        diagnostics = legal_diagnostics_from_logits(logits, legal)
        self.assertEqual(diagnostics["actions"].tolist(), [5, 17])
        self.assertEqual(diagnostics["legalCounts"].tolist(), [2, 1])
        self.assertTrue(np.isnan(diagnostics["margins"][1]))
        self.assertTrue((diagnostics["illegalProbabilityMass"] == 0.0).all())

    def test_immutable_report_and_checksum(self) -> None:
        report = {
            "format": FORMAT,
            "version": VERSION,
            "purpose": "offline-behavioral-safety-calibration",
            "outcomeSuperiorityEvidence": False,
            "deploymentTriggered": False,
            "datasetAudit": {"privilegedArraysLoaded": []},
            "routingContract": {
                "thresholdComparison": "all-configured-thresholds-met-inclusive",
                "forcedDecisionRule": "exact-normal",
            },
        }
        with tempfile.TemporaryDirectory() as directory_value:
            output = Path(directory_value) / "report.json"
            published = write_calibration_report_exclusive(output, report)
            self.assertEqual(_sha(output), published["sha256"])
            self.assertEqual(
                Path(f"{output}.sha256").read_text("ascii"),
                f"{published['sha256']}  report.json\n",
            )
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_calibration_report_exclusive(output, report)
            self.assertEqual(output.read_bytes(), before)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch unavailable")
class FallbackCalibrationDatasetTests(unittest.TestCase):
    def _write_dataset(self, directory: Path, *, private_object: bool) -> Path:
        from v4_model import V4ActorConfig

        config = V4ActorConfig(
            max_history=8,
            d_model=24,
            layers=1,
            heads=4,
            feedforward=48,
            action_hidden=24,
        )
        trajectories, times = 3, 2
        prefix = (trajectories, times)
        valid = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.bool_)
        player_mask = np.zeros((*prefix, 10), dtype=np.bool_)
        player_mask[..., :4] = True
        history_mask = np.zeros((*prefix, 8), dtype=np.bool_)
        legal = np.zeros((*prefix, 236), dtype=np.bool_)
        legal[..., 0] = True
        legal[..., 2] = True
        legal[1, 0, 2] = False  # one forced valid sample
        actions = np.zeros(prefix, dtype=np.int64)
        global_features = np.zeros((*prefix, 12), dtype=np.float32)
        global_features[..., 2] = 1.0
        metadata = {
            "format": "dalmuti-v4-trajectory-npz",
            "version": 1,
            "preparationFormat": "dalmuti-v4-prepared-dataset-metadata",
            "preparationVersion": 1,
            "actorConfig": config.to_dict(),
            "criticConfig": {
                "privileged_features": 512,
                "d_model": 512,
                "hidden_layers": 3,
                "action_hidden": 256,
                "dropout": 0.0,
            },
            "fingerprint": "f" * 64,
            "inputs": [
                {
                    "sha256": "a" * 64,
                    "sourceHashes": {
                        "actorObservationContract": "b" * 64,
                        "privilegedCriticContract": "c" * 64,
                        "actionCatalogue": "d" * 64,
                    },
                }
            ],
            "trajectoryCount": trajectories,
            "maxTimeSteps": times,
            "trajectoryIds": ["t0", "t1", "t2"],
            "trajectoryInputSha256s": ["a" * 64] * trajectories,
            "sampleFieldsPresent": [],
            "syntheticDefaults": {
                "expertActionIndex": True,
                "oldLogProbability": True,
                "advantage": True,
            },
            "padding": "zero-valued invalid suffix strictly after the sole actor terminal",
            "auxiliaryArrays": [
                "finish_places",
                "environment_terminals",
                "source_steps",
            ],
            "privilegedCriticExportAllowed": False,
        }
        arrays = {
            "global_features": global_features,
            "rank_features": np.zeros((*prefix, 13, 6), np.float32),
            "player_features": np.zeros((*prefix, 10, 12), np.float32),
            "player_mask": player_mask,
            "memory_trace_features": np.zeros((*prefix, 4, 20), np.float32),
            "history_features": np.zeros((*prefix, 8, 20), np.float32),
            "history_mask": history_mask,
            "legal_masks": legal,
            "actions": actions,
            "expert_actions": actions.copy(),
            "valid_masks": valid,
            "trajectory_ids": np.array(["t0", "t1", "t2"]),
            "metadata_json": np.array(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            # An object array raises under allow_pickle=False if indexed.  A
            # successful load therefore proves the private payload was not read.
            "privileged_states": (
                np.array([[{"secret": 1}]], dtype=object)
                if private_object
                else np.zeros((*prefix, 512), np.float32)
            ),
        }
        output = directory / "normal.npz"
        np.savez_compressed(output, **arrays)
        digest = _sha(output)
        Path(f"{output}.sha256").write_text(digest + "\n", encoding="ascii")
        external = dict(metadata)
        external["npzSha256"] = digest
        Path(f"{output}.metadata.json").write_bytes(canonical_json_bytes(external))
        return output

    def test_privileged_object_array_is_never_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            dataset = self._write_dataset(Path(directory_value), private_object=True)
            prepared = load_public_prepared_normal_dataset(dataset)
            self.assertIsInstance(prepared, PublicPreparedDataset)
            self.assertNotIn("privileged_states", prepared.arrays)
            self.assertEqual(prepared.arrays["actions"].shape, (3, 2))

    def test_checksum_and_exact_normal_contract_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            dataset = self._write_dataset(directory, private_object=False)
            Path(f"{dataset}.sha256").write_text("0" * 64 + "\n", "ascii")
            with self.assertRaisesRegex(ValueError, "checksum does not match"):
                load_public_prepared_normal_dataset(dataset)

    def test_cuda_actor_bundle_end_to_end_when_available(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from v4_evaluate import _load_cli_actor_policy
        from v4_export import export_v4_actor_bundle
        from v4_model import V4ActorConfig, V4PublicActor

        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            dataset = self._write_dataset(directory, private_object=False)
            config = V4ActorConfig(
                max_history=8,
                d_model=24,
                layers=1,
                heads=4,
                feedforward=48,
                action_hidden=24,
            )
            torch.manual_seed(71)
            bundle = directory / "bundle"
            export_v4_actor_bundle(
                V4PublicActor(config).eval(), bundle, metadata={"seed": 71}
            )
            policy, binding, _ = _load_cli_actor_policy(
                [str(bundle)], actor_seeds=[71], device="cuda", compile_actor=False
            )
            prepared = load_public_prepared_normal_dataset(dataset)
            report = build_calibration_report(
                prepared=prepared,
                policy=policy,
                actor_binding=binding,
                margin_grid=(0.0, 0.5),
                probability_grid=(0.0, 0.75),
                target_agreement_lcb=0.0,
                minimum_coverage=0.0,
                bootstrap_resamples=20,
                batch_size=3,
                device="cuda",
            )
            self.assertEqual(report["format"], FORMAT)
            self.assertEqual(report["datasetAudit"]["validDecisions"], 5)
            self.assertEqual(report["datasetAudit"]["privilegedArraysLoaded"], [])
            self.assertTrue(report["actorAudit"]["allTop1ActionsLegal"])
            self.assertEqual(report["actorAudit"]["illegalProbabilityMassMaximum"], 0.0)


if __name__ == "__main__":
    unittest.main()
