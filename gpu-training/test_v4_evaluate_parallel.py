from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_v4_evaluate import _BINDINGS, _FakeBatchPolicy, _FakeExactAdapter
from v4_evaluate import (
    FINAL_MATCH_COUNTS,
    EvaluationBindings,
    EvaluationSeedSchedule,
    canonical_json_bytes,
    evaluate_benchmark,
)
from v4_evaluate_parallel import (
    build_evaluation_shard,
    load_evaluation_shard,
    merge_evaluation_shards,
    partition_player_counts,
    validate_evaluation_shard,
    validate_parallel_evaluation_plan,
    write_evaluation_shard_exclusive,
)


def _final_reservation(
    schedule: EvaluationSeedSchedule,
    bindings: EvaluationBindings = _BINDINGS,
) -> dict[str, object]:
    return {
        "format": "dalmuti-v4-final-seed-reservation",
        "version": 1,
        "baseSeed": schedule.base_seed,
        "matchSeedRanges": schedule.ranges(FINAL_MATCH_COUNTS),
        "reuseForbidden": True,
        "finalFeedbackPolicy": "sealed-holdout-not-a-training-input",
        "bindings": bindings.report_value(),
    }


class V4ParallelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = EvaluationSeedSchedule(
            "screening", "parallel-byte-parity", 600_001
        )
        cls.bootstrap_resamples = 20
        cls.batch_size = 16
        cls.single = evaluate_benchmark(
            mode="screening",
            seed_schedule=cls.schedule,
            candidate_policy=_FakeBatchPolicy(),
            bindings=_BINDINGS,
            adapter=_FakeExactAdapter(),
            bootstrap_resamples=cls.bootstrap_resamples,
            batch_size=cls.batch_size,
        )
        cls.results_by_player_count = {
            int(result["playerCount"]): result for result in cls.single["results"]
        }

    @classmethod
    def _shards(cls, workers: int) -> list[dict[str, object]]:
        return [
            build_evaluation_shard(
                mode="screening",
                seed_schedule=cls.schedule,
                bindings=_BINDINGS,
                player_counts=player_counts,
                results=[
                    cls.results_by_player_count[value] for value in player_counts
                ],
                candidate_policy_metadata=cls.single["candidatePolicy"],
                candidate_batched_forward=cls.single["evaluationDesign"][
                    "candidateBatchedForward"
                ],
                bootstrap_resamples=cls.bootstrap_resamples,
                batch_size=cls.batch_size,
            )
            for player_counts in partition_player_counts(workers)
        ]

    @classmethod
    def _merge(cls, shards: list[dict[str, object]]) -> dict[str, object]:
        return merge_evaluation_shards(
            shards,
            mode="screening",
            seed_schedule=cls.schedule,
            bindings=_BINDINGS,
            candidate_policy_metadata=cls.single["candidatePolicy"],
            candidate_batched_forward=cls.single["evaluationDesign"][
                "candidateBatchedForward"
            ],
            bootstrap_resamples=cls.bootstrap_resamples,
            batch_size=cls.batch_size,
        )

    def test_worker_partitions_one_three_and_four_are_whole_and_exact(self) -> None:
        for workers in (1, 3, 4):
            partitions = partition_player_counts(workers)
            self.assertEqual(len(partitions), workers)
            flattened = [value for partition in partitions for value in partition]
            self.assertEqual(sorted(flattened), list(range(4, 11)))
            self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(partition_player_counts(1), ((4, 5, 6, 7, 8, 9, 10),))
        self.assertEqual(
            partition_player_counts(4),
            ((10,), (4, 9), (5, 8), (6, 7)),
        )
        with self.assertRaisesRegex(ValueError, "exceed"):
            partition_player_counts(5)

    def test_parallel_merge_is_canonical_byte_and_sha_identical(self) -> None:
        expected_bytes = canonical_json_bytes(self.single)
        expected_sha = hashlib.sha256(expected_bytes).hexdigest()
        for workers in (1, 3, 4):
            merged = self._merge(list(reversed(self._shards(workers))))
            actual_bytes = canonical_json_bytes(merged)
            self.assertEqual(actual_bytes, expected_bytes)
            self.assertEqual(hashlib.sha256(actual_bytes).hexdigest(), expected_sha)
            self.assertEqual(
                [result["playerCount"] for result in merged["results"]],
                list(range(4, 11)),
            )

    def test_missing_and_duplicate_player_count_shards_fail_closed(self) -> None:
        shards = self._shards(4)
        with self.assertRaisesRegex(ValueError, "missing"):
            self._merge(shards[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._merge([*shards, copy.deepcopy(shards[0])])

    def test_mismatched_seed_binding_count_and_order_fail_closed(self) -> None:
        shards = self._shards(4)

        wrong_seed = copy.deepcopy(shards[0])
        wrong_seed["seedFamily"]["id"] = "other-family"
        with self.assertRaisesRegex(ValueError, "seed"):
            self._merge([wrong_seed, *shards[1:]])

        wrong_binding = copy.deepcopy(shards[0])
        wrong_binding["bindings"]["modelSha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "bindings"):
            self._merge([wrong_binding, *shards[1:]])

        wrong_count = copy.deepcopy(shards[0])
        key = str(wrong_count["playerCounts"][0])
        wrong_count["matchCountsByPlayerCount"][key] += 1
        with self.assertRaisesRegex(ValueError, "match counts"):
            self._merge([wrong_count, *shards[1:]])

        two_count_shard = copy.deepcopy(next(
            shard for shard in shards if len(shard["playerCounts"]) == 2
        ))
        two_count_shard["results"].reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            validate_evaluation_shard(two_count_shard)

    def test_immutable_canonical_shards_reject_payload_and_encoding_tamper(self) -> None:
        shard = self._shards(1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            published = write_evaluation_shard_exclusive(path, shard)
            self.assertEqual(load_evaluation_shard(path), shard)
            self.assertEqual(
                Path(str(path) + ".sha256").read_text("ascii"),
                f"{published['sha256']}  shard.json\n",
            )
            with self.assertRaises(FileExistsError):
                write_evaluation_shard_exclusive(path, shard)

            original = path.read_bytes()
            path.write_bytes(original.replace(b'"version":1', b'"version":2', 1))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_evaluation_shard(path)

            pretty = json.dumps(shard, indent=2, sort_keys=True).encode("utf-8")
            path.write_bytes(pretty)
            digest = hashlib.sha256(pretty).hexdigest()
            Path(str(path) + ".sha256").write_text(
                f"{digest}  shard.json\n", encoding="ascii", newline=""
            )
            with self.assertRaisesRegex(ValueError, "canonical"):
                load_evaluation_shard(path)

    def test_exact_mode_presets_and_final_reservation_are_enforced(self) -> None:
        validate_parallel_evaluation_plan(
            mode="screening",
            seed_schedule=self.schedule,
            bindings=_BINDINGS,
            final_seed_reservation=None,
        )
        validate_parallel_evaluation_plan(
            mode="development",
            seed_schedule=EvaluationSeedSchedule(
                "development", "parallel-development", 50_000_001
            ),
            bindings=_BINDINGS,
            final_seed_reservation=None,
        )
        final_schedule = EvaluationSeedSchedule(
            "final", "parallel-final", 900_000_001
        )
        reservation = _final_reservation(final_schedule)
        validate_parallel_evaluation_plan(
            mode="final",
            seed_schedule=final_schedule,
            bindings=_BINDINGS,
            final_seed_reservation=reservation,
        )
        final_report = evaluate_benchmark(
            mode="final",
            seed_schedule=final_schedule,
            candidate_policy=_FakeBatchPolicy(),
            bindings=_BINDINGS,
            adapter=_FakeExactAdapter(),
            bootstrap_resamples=20,
            batch_size=128,
            final_seed_reservation=reservation,
        )
        final_results = {
            int(result["playerCount"]): result for result in final_report["results"]
        }
        final_shards = [
            build_evaluation_shard(
                mode="final",
                seed_schedule=final_schedule,
                bindings=_BINDINGS,
                player_counts=player_counts,
                results=[final_results[value] for value in player_counts],
                candidate_policy_metadata=final_report["candidatePolicy"],
                candidate_batched_forward=True,
                bootstrap_resamples=20,
                batch_size=128,
                final_seed_reservation=reservation,
            )
            for player_counts in partition_player_counts(
                4, match_counts=FINAL_MATCH_COUNTS
            )
        ]
        final_merged = merge_evaluation_shards(
            list(reversed(final_shards)),
            mode="final",
            seed_schedule=final_schedule,
            bindings=_BINDINGS,
            candidate_policy_metadata=final_report["candidatePolicy"],
            candidate_batched_forward=True,
            bootstrap_resamples=20,
            batch_size=128,
            final_seed_reservation=reservation,
        )
        self.assertEqual(
            canonical_json_bytes(final_merged), canonical_json_bytes(final_report)
        )
        tampered_final_shard = copy.deepcopy(final_shards[0])
        tampered_final_shard["finalReservationSha256"] = "7" * 64
        with self.assertRaisesRegex(ValueError, "reservation"):
            merge_evaluation_shards(
                [tampered_final_shard, *final_shards[1:]],
                mode="final",
                seed_schedule=final_schedule,
                bindings=_BINDINGS,
                candidate_policy_metadata=final_report["candidatePolicy"],
                candidate_batched_forward=True,
                bootstrap_resamples=20,
                batch_size=128,
                final_seed_reservation=reservation,
            )
        with self.assertRaisesRegex(ValueError, "reservation"):
            validate_parallel_evaluation_plan(
                mode="final",
                seed_schedule=final_schedule,
                bindings=_BINDINGS,
                final_seed_reservation=None,
            )
        mismatched = copy.deepcopy(reservation)
        mismatched["bindings"]["modelSha256"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "modelSha256"):
            validate_parallel_evaluation_plan(
                mode="final",
                seed_schedule=final_schedule,
                bindings=_BINDINGS,
                final_seed_reservation=mismatched,
            )
        with self.assertRaisesRegex(ValueError, "only final"):
            validate_parallel_evaluation_plan(
                mode="screening",
                seed_schedule=self.schedule,
                bindings=_BINDINGS,
                final_seed_reservation=reservation,
            )

        wrong_preset = copy.deepcopy(self._shards(1)[0])
        wrong_preset["evaluationMode"] = "development"
        wrong_preset["seedFamily"]["mode"] = "development"
        with self.assertRaisesRegex(ValueError, "match counts"):
            validate_evaluation_shard(wrong_preset)


if __name__ == "__main__":
    unittest.main()
