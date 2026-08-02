from __future__ import annotations

from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import v5_gpu_memory_preflight as preflight
from v5_gpu_memory_preflight import V5GPUMemoryPreflightConfig


def _shard(
    history_end: list[int],
    player_counts: list[int],
    legal_counts: list[int] | None = None,
) -> object:
    rows = len(history_end)
    global_codes = np.zeros((rows, 2), dtype=np.int64)
    global_codes[:, 1] = np.asarray(player_counts, dtype=np.int64)
    counts = legal_counts or [1] * rows
    legal_mask = np.zeros((rows, 236), dtype=np.uint8)
    for row, count in enumerate(counts):
        legal_mask[row, :count] = 1
    return SimpleNamespace(
        actor=SimpleNamespace(
            arrays={
                "forced": np.zeros(rows, dtype=np.bool_),
                "global_codes": global_codes,
                "history_end": np.asarray(history_end, dtype=np.int64),
                "legal_action_bits": np.packbits(
                    legal_mask, axis=1, bitorder="little"
                ),
            }
        )
    )


class V5GPUMemoryPreflightTests(unittest.TestCase):
    def test_worst_case_selection_compares_history_lengths_across_shards(self) -> None:
        short_late = _shard([9_990, 10_000, 10_010], [4, 10, 10])
        long_early = _shard([100, 200], [10, 10])
        source = SimpleNamespace(shards=[short_late, long_early])

        selected_shard, selected_indices = preflight._worst_case_indices(
            source, 2, nonforced=True
        )

        self.assertIs(selected_shard, long_early)
        np.testing.assert_array_equal(selected_indices, np.asarray([1, 0]))

    def test_audit_candidates_cover_legal_width_and_composite_pressure(self) -> None:
        wide = _shard([1, 2], [10, 10], [236, 200])
        long = _shard([200, 400], [10, 10], [2, 2])
        source = SimpleNamespace(shards=[wide, long])

        legal_shard, _ = preflight._worst_case_indices(
            source, 2, nonforced=True, criterion="legal"
        )
        composite_shard, _ = preflight._worst_case_indices(
            source, 2, nonforced=True, criterion="history_x_legal"
        )

        self.assertIs(legal_shard, wide)
        self.assertIs(composite_shard, long)

    def test_timing_summary_uses_deterministic_nearest_rank_p95(self) -> None:
        result = preflight._timing_summary([0.7, 0.1, 0.4, 0.3, 0.2])
        self.assertEqual(result["iterations"], 5)
        self.assertAlmostEqual(float(result["medianSeconds"]), 0.3)
        self.assertAlmostEqual(float(result["p95Seconds"]), 0.7)

    def test_phase_peak_uses_direct_device_free_memory_observation(self) -> None:
        device = object()
        with mock.patch.object(
            preflight.torch.cuda, "synchronize"
        ), mock.patch.object(
            preflight.torch.cuda, "mem_get_info", return_value=(123, 456)
        ) as memory, mock.patch.object(
            preflight.torch.cuda, "max_memory_allocated", return_value=78
        ), mock.patch.object(
            preflight.torch.cuda, "max_memory_reserved", return_value=90
        ):
            peak = preflight._phase_peak(device)  # type: ignore[arg-type]
        memory.assert_called_once_with(device)
        self.assertEqual(
            peak,
            {
                "allocatedBytes": 78,
                "deviceFreeBytes": 123,
                "deviceTotalBytes": 456,
                "reservedBytes": 90,
            },
        )

    def test_optimizer_step_matches_trainer_amp_clip_and_finite_order(self) -> None:
        events: list[str] = []
        optimizer = object()
        module = SimpleNamespace(parameters=lambda: ("parameter",))

        class Scaler:
            def unscale_(self, received: object) -> None:
                self.assert_optimizer(received)
                events.append("unscale")

            def step(self, received: object) -> None:
                self.assert_optimizer(received)
                events.append("step")

            def update(self) -> None:
                events.append("update")

            @staticmethod
            def assert_optimizer(received: object) -> None:
                if received is not optimizer:
                    raise AssertionError("optimizer identity drifted")

        def clip(parameters: object, maximum: float) -> object:
            self.assertEqual(tuple(parameters), ("parameter",))
            self.assertEqual(maximum, 0.5)
            events.append("clip")
            return object()

        def finite(_: object) -> bool:
            events.append("finite")
            return True

        with mock.patch.object(
            preflight.torch.nn.utils, "clip_grad_norm_", side_effect=clip
        ), mock.patch.object(preflight.torch, "isfinite", side_effect=finite):
            preflight._step_preflight_optimizer(  # type: ignore[arg-type]
                module, optimizer, Scaler()
            )
        self.assertEqual(events, ["unscale", "clip", "finite", "step", "update"])

    def test_optimizer_step_rejects_nonfinite_gradient_before_step(self) -> None:
        optimizer = object()
        module = SimpleNamespace(parameters=lambda: ())
        scaler = mock.Mock()
        with mock.patch.object(
            preflight.torch.nn.utils, "clip_grad_norm_", return_value=object()
        ), mock.patch.object(preflight.torch, "isfinite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                preflight._step_preflight_optimizer(  # type: ignore[arg-type]
                    module, optimizer, scaler
                )
        scaler.unscale_.assert_called_once_with(optimizer)
        scaler.step.assert_not_called()
        scaler.update.assert_not_called()

    def test_config_and_non_cuda_admission_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            V5GPUMemoryPreflightConfig(timing_iterations=0)
        with self.assertRaisesRegex(ValueError, "effective 32"):
            V5GPUMemoryPreflightConfig(microbatch_size=4, gradient_accumulation=4)
        with mock.patch.object(preflight.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an available CUDA"):
                preflight.run_v5_gpu_memory_preflight(
                    "dataset", "pair", "output", device="cuda:0"
                )
        with self.assertRaisesRegex(RuntimeError, "requires an available CUDA"):
            preflight.run_v5_gpu_memory_preflight(
                "dataset", "pair", "output", device="cpu"
            )

    def test_report_loader_requires_canonical_checksum_sidecar(self) -> None:
        config = V5GPUMemoryPreflightConfig()
        report = {
            "behaviorBindings": {
                "behaviorActorManifestSha256": "1" * 64,
                "behaviorActorSha256": "2" * 64,
                "behaviorCriticSha256": "3" * 64,
                "behaviorModelPairId": "4" * 64,
                "behaviorModelPairManifestSha256": "5" * 64,
            },
            "config": asdict(config),
            "datasetIdentitySha256": "a" * 64,
            "datasetStatistics": {
                "nonforcedDecisionCount": 10,
                "totalDecisionCount": 12,
            },
            "device": {
                "capability": [8, 6],
                "name": "test",
                "requested": "cuda:0",
                "totalMemoryBytes": 10_000,
                "type": "cuda",
            },
            "failure": "synthetic failure",
            "format": preflight.V5_GPU_MEMORY_PREFLIGHT_FORMAT,
            "model": {
                "actorSha256": "6" * 64,
                "manifestSha256": "7" * 64,
                "policyNumericsSha256": "8" * 64,
                "publicContractSha256": "9" * 64,
                "tensorStateSha256": "c" * 64,
            },
            "modelPairId": "b" * 64,
            "passed": False,
            "peaks": {
                "actorBackwardAndOptimizer": {
                    "allocatedBytes": 0,
                    "deviceFreeBytes": 10_000,
                    "deviceTotalBytes": 10_000,
                    "reservedBytes": 0,
                },
                "auditForward": {
                    "allocatedBytes": 0,
                    "deviceFreeBytes": 10_000,
                    "deviceTotalBytes": 10_000,
                    "reservedBytes": 0,
                },
                "auditForwardCandidates": {
                    criterion: {
                        "allocatedBytes": 0,
                        "deviceFreeBytes": 10_000,
                        "deviceTotalBytes": 10_000,
                        "reservedBytes": 0,
                    }
                    for criterion in ("history", "legal", "history_x_legal")
                },
                "criticBackwardAndOptimizer": {
                    "allocatedBytes": 0,
                    "deviceFreeBytes": 10_000,
                    "deviceTotalBytes": 10_000,
                    "reservedBytes": 0,
                },
                "allocatorMaximumReservedFraction": 0.0,
                "minimumObservedDeviceFreeBytes": 10_000,
            },
            "policyNumericsSha256": preflight.V5_POLICY_NUMERICS_SHA256,
            "runtime": {},
            "timing": None,
            "version": preflight.V5_GPU_MEMORY_PREFLIGHT_VERSION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            digest = preflight._write_report(path, report)
            loaded, loaded_digest = preflight.load_v5_gpu_memory_preflight_report(
                path
            )
            self.assertEqual(loaded, report)
            self.assertEqual(loaded_digest, digest)
            path.with_name(path.name + ".sha256").write_text(
                "0" * 64 + "  preflight.json\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "checksum sidecar"):
                preflight.load_v5_gpu_memory_preflight_report(path)


if __name__ == "__main__":
    unittest.main()
