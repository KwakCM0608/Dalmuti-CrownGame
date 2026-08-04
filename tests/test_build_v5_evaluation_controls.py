from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPOSITORY_ROOT / "scripts/build_v5_evaluation_controls.py"
RUN_NAMESPACE = "v5-mappo-normalresidual-i001-s840060001-run-007"
SOURCE_COMMIT = "86621835674c0fe3eb6785feb169d57c9a35c49e"
RUN_ROOT = REPOSITORY_ROOT / "artifacts/rl/v5-runs" / RUN_NAMESPACE


def load_builder() -> object:
    specification = importlib.util.spec_from_file_location(
        "_dalmuti_v5_evaluation_builder_test", BUILDER
    )
    if specification is None or specification.loader is None:
        raise ImportError("evaluation-controls builder cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class EvaluationControlsBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def test_sealed_evaluation_contract_is_exact(self) -> None:
        contract = self.builder._evaluation_contract(RUN_ROOT)
        self.assertEqual(
            contract["stageMatchTotals"],
            {
                "screening": 420,
                "certification-a": 420,
                "certification-b": 420,
            },
        )
        self.assertEqual(contract["playerCounts"], list(range(4, 11)))
        self.assertEqual(contract["finalTotalMatches"], 6800)
        self.assertEqual(contract["bootstrapResamples"], 10_000)
        self.assertEqual(
            contract["exactGates"],
            {
                "minMeanChipDifference": 0.25,
                "minCluster95LowerBound": 0.15,
                "minPairwiseRate": 0.55,
            },
        )

    def test_disposable_build_exercises_admissions_receipt_and_72_stages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="dalmuti-evaluation-builder-test-"
        ) as raw:
            output = Path(raw)
            result = self.builder.build_controls(
                RUN_ROOT,
                output,
                run_namespace=RUN_NAMESPACE,
                source_commit=SOURCE_COMMIT,
            )
            validation = result["validation"]
            self.assertEqual(validation["launcherStageCount"], 72)
            self.assertTrue(validation["launcherSelfTest"])
            self.assertEqual(
                validation["candidateReceiptFixture"],
                {
                    "buggyVerifierCalls": 0,
                    "tamperTests": 5,
                    "passed": True,
                },
            )
            self.assertEqual(
                validation["recoveryEpochFixture"],
                {
                    "capturedObjectEpochPassed": True,
                    "negativeTests": 4,
                    "shardCountPath": "epoch.batching.shardCount",
                    "passed": True,
                },
            )
            self.assertEqual(
                validation["recoveryTopologyFixture"],
                {"negativeTests": 3, "passed": True},
            )
            self.assertEqual(
                validation["recoveryExclusivePublishFixture"],
                {"duplicateRejected": True, "passed": True},
            )
            self.assertTrue(validation["recoveryDelayedImportFixture"]["passed"])
            self.assertGreaterEqual(
                validation["recoveryInventoryFixture"]["negativeTests"], 2
            )
            self.assertEqual(
                validation["sourceModuleFixture"],
                {"importCount": 5, "negativeTests": 1, "passed": True},
            )
            receipt = json.loads(
                (output / self.builder.RECEIPT_NAME).read_text(encoding="ascii")
            )
            self.assertEqual(
                receipt["evaluationAttemptRelative"],
                "controls/durable-evaluation-attempts-recovery-r3",
            )
            self.assertEqual(
                receipt["launcherName"],
                "launch_durable_evaluation_pipeline_recovery_r3.py",
            )
            self.assertEqual(
                receipt["validatorName"],
                "validate_durable_evaluation_pipeline_recovery_r3.py",
            )
            self.assertEqual(
                receipt["evaluationContract"]["stageMatchTotals"],
                {
                    "screening": 420,
                    "certification-a": 420,
                    "certification-b": 420,
                },
            )
            control = (
                output / self.builder.EVALUATION_CONTROL_NAME
            ).read_text(encoding="utf-8")
            recovery = (
                output / self.builder.RECOVERY_VERIFIER_NAME
            ).read_text(encoding="utf-8")
            launcher = (output / self.builder.LAUNCHER_NAME).read_text(
                encoding="utf-8"
            )
            for fragment in (
                '"v5_gpu_memory_preflight"',
                "def _verified_training_inventory(",
                "RECOVERY_RECEIPT_RELATIVE",
                "RECOVERY_VERIFIER_SHA256",
                "EXPECTED_RUNTIME_PYTHON_PATH_COUNT = 143",
                "EXPECTED_CHECKED_SOURCE_PATH_COUNT = 154",
            ):
                self.assertIn(fragment, control)
            for fragment in (
                "verify_training_output",
                "establish_receipt",
                "publish_json_pair_exclusive",
            ):
                self.assertNotIn(fragment, control)
            for fragment in (
                "def _validated_epoch(",
                'batching.get("shardCount") != 29',
                "EXPECTED_DURABLE_TRAIN_LAUNCH_SHA256",
                "def _verify_completed_training_topology(",
                'durable_root / "verify-training"',
                "worker-stdout.log",
            ):
                self.assertIn(fragment, recovery)
            for fragment in (
                "def _validated_source_admission(",
                "def _disk_free_admission(",
                "def _completed_training_epoch(",
                '_completed_training_epoch(result.get("epoch"))',
                '_completed_training_epoch({"epoch": 1})',
                'ATTEMPT_RELATIVE = Path("controls/durable-evaluation-attempts-recovery-r3")',
                'LAUNCHER_NAME = "launch_durable_evaluation_pipeline_recovery_r3.py"',
                "MINIMUM_EVALUATION_FREE_BYTES = 6442450944",
                "diskFreeAdmission",
            ):
                self.assertIn(fragment, launcher)
            self.assertNotIn('result.get("epoch") != 1', launcher)
            self.assertIn('"tests": 29', launcher)

    def test_failure_reference_and_epoch_only_proof_are_exact(self) -> None:
        reference = self.builder._load_failure_reference()
        result = reference["result.json"]
        epoch = result["epoch"]
        self.assertIsInstance(epoch, dict)
        self.assertEqual(epoch["epoch"], 1)
        self.assertEqual(epoch["batching"]["shardCount"], 29)
        self.assertEqual(epoch["actorDecisionRowsSeen"], 1_602_500)
        self.assertEqual(epoch["criticDecisionRowsSeen"], 4_514_492)
        self.assertEqual(
            reference["epochOnlyProofSha256"],
            self.builder.EPOCH_ONLY_PROOF_SHA256,
        )

    def test_disposable_build_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="dalmuti-evaluation-determinism-a-"
        ) as raw_a, tempfile.TemporaryDirectory(
            prefix="dalmuti-evaluation-determinism-b-"
        ) as raw_b:
            first = Path(raw_a)
            second = Path(raw_b)
            first_result = self.builder.build_controls(
                RUN_ROOT,
                first,
                run_namespace=RUN_NAMESPACE,
                source_commit=SOURCE_COMMIT,
            )
            second_result = self.builder.build_controls(
                RUN_ROOT,
                second,
                run_namespace=RUN_NAMESPACE,
                source_commit=SOURCE_COMMIT,
            )
            names = {
                self.builder.EVALUATION_CONTROL_NAME,
                self.builder.RECOVERY_VERIFIER_NAME,
                self.builder.LAUNCHER_NAME,
                self.builder.VALIDATOR_NAME,
                self.builder.RECEIPT_NAME,
                self.builder.RECEIPT_NAME + ".sha256",
            }
            self.assertEqual(
                {name: (first / name).read_bytes() for name in names},
                {name: (second / name).read_bytes() for name in names},
            )
            self.assertEqual(
                first_result["buildReceiptSha256"],
                second_result["buildReceiptSha256"],
            )


if __name__ == "__main__":
    unittest.main()
