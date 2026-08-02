from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import v5_collection_plan
from v4_env import DalmutiScalarEnv
from v5_collection_plan import (
    CALIBRATION_VALUE_ATOL,
    V5_CALIBRATION_SCHEDULE_CONTRACT,
    allocate_mixed_backend_shards,
    build_collection_plan,
    calibration_schedule_id,
    completion_balanced_cpu_matches,
    compare_calibration_shards,
    expected_planned_shard_metadata,
    load_collection_plan,
    planned_shard_path,
    publish_calibration_report,
    publish_collection_plan,
    publish_verified_index,
    resume_verified_shard,
    validate_actual_nonforced_corpus,
    verify_planned_collection_corpus,
    verify_planned_shard,
)
from v5_dataset import publish_v5_shard
from v5_model import V5_POLICY_NUMERICS_SHA256
from v5_collect_mappo import (
    V5_MAPPO_COLLECTION_CONTRACT,
    V5_MAPPO_REWARD_CONTRACT,
    V5_MATCH_PROVENANCE_CONTRACT,
    derive_v5_collection_match_seed,
)
from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_public import (
    pack_v5_public_observations,
    v5_public_from_v4_actor_observation,
)


def _source_inventory() -> dict[str, str]:
    return {
        "gpu-training/a.py": "d" * 64,
        "gpu-training/b.py": "e" * 64,
    }


def _plan(**overrides: object):
    arguments: dict[str, object] = {
        "run_namespace": "v5-large-s900000001-run-001",
        "seed_base": 900_000_001,
        "behavior_actor_sha256": "a" * 64,
        "behavior_actor_manifest_sha256": "b" * 64,
        "behavior_critic_sha256": "c" * 64,
        "behavior_pair_id": "1" * 64,
        "behavior_pair_manifest_sha256": "2" * 64,
        "calibration_report_sha256": "f" * 64,
        "source_inventory": _source_inventory(),
    }
    if "preflight_strata" not in overrides and "diagnostic_unbalanced" not in overrides:
        arguments["diagnostic_unbalanced"] = True
    arguments.update(overrides)
    return build_collection_plan(**arguments)  # type: ignore[arg-type]


def _fixture_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    decisions = 7
    players = np.arange(4, 11, dtype=np.uint8)
    environments = [
        DalmutiScalarEnv(
            int(player_count),
            acts=1,
            seed=971_000_000 + int(player_count),
        )
        for player_count in players
    ]
    observations = [
        v5_public_from_v4_actor_observation(environment.public_observation())
        for environment in environments
    ]
    public_arrays, history_events, history_end = pack_v5_public_observations(
        observations
    )
    normal_actions = np.asarray(
        [environment.normal_action() for environment in environments],
        dtype=np.uint16,
    )
    actor_ids = np.asarray(
        [environment.current_player_id for environment in environments],
        dtype=np.uint8,
    )
    actor = {
        **public_arrays,
        "match_offsets": np.arange(decisions + 1, dtype=np.uint32),
        "candidate_bitsets": np.asarray(
            [1 << int(actor_id) for actor_id in actor_ids], dtype=np.uint16
        ),
        "player_counts": players,
        "decision_actor_ids": actor_ids,
        "decision_acts": np.ones(decisions, np.uint8),
        "normal_actions": normal_actions,
        "actions": normal_actions.copy(),
        "old_log_probs": np.zeros(decisions, np.float32),
        "old_values": np.linspace(-0.3, 0.3, decisions, dtype=np.float32),
        "reward_to_next": np.linspace(-1.0, 1.0, decisions, dtype=np.float32),
        "done": np.ones(decisions, np.bool_),
        "forced": np.asarray(
            [int(observation.legal_mask.sum()) == 1 for observation in observations],
            dtype=np.bool_,
        ),
        "next_decision": np.full(decisions, -1, np.int32),
        "history_events": history_events,
        "history_end": history_end,
    }
    privileged = {
        "privileged_states": np.arange(decisions * 5, dtype=np.float16).reshape(decisions, 5)
    }
    return actor, privileged


def _calibration_metadata(backend: str) -> dict[str, object]:
    return {
        "behaviorActorManifestSha256": "b" * 64,
        "behaviorActorSha256": "a" * 64,
        "behaviorCriticSha256": "c" * 64,
        "behaviorModelPairId": "1" * 64,
        "behaviorModelPairManifestSha256": "2" * 64,
        "calibrationBackend": backend,
        "calibrationScheduleContract": V5_CALIBRATION_SCHEDULE_CONTRACT,
        "calibrationScheduleId": calibration_schedule_id(
            "v5-calibration-s899000001",
            899_000_001,
            {player: 1 for player in range(4, 11)},
        ),
        "collectionContract": V5_MAPPO_COLLECTION_CONTRACT,
        "matchCounts": {str(player): 1 for player in range(4, 11)},
        "matchStart": 0,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "rewardContract": V5_MAPPO_REWARD_CONTRACT,
        "runNamespace": "v5-calibration-s899000001",
        "seedBase": 899_000_001,
        "sourceInventory": _source_inventory(),
        "sourceInventorySha256": __import__(
            "v5_collection_plan"
        ).source_inventory_sha256(_source_inventory()),
    }


def _planned_one_match_arrays(
    plan: object,
    shard: object,
    *,
    match_index: int | None = None,
    match_seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    player_count = int(shard.player_count)  # type: ignore[attr-defined]
    index = int(shard.match_start if match_index is None else match_index)  # type: ignore[attr-defined]
    seed = (
        derive_v5_collection_match_seed(
            str(plan.run_namespace),  # type: ignore[attr-defined]
            int(plan.seed_base),  # type: ignore[attr-defined]
            player_count,
            index,
        )
        if match_seed is None
        else match_seed
    )
    environment = DalmutiScalarEnv(
        player_count,
        acts=1,
        seed=972_000_000 + player_count,
    )
    observation = v5_public_from_v4_actor_observation(
        environment.public_observation()
    )
    public_arrays, history_events, history_end = pack_v5_public_observations(
        [observation]
    )
    normal_action = np.uint16(environment.normal_action())
    actor_id = np.uint8(environment.current_player_id)
    actor = {
        **public_arrays,
        "match_offsets": np.asarray([0, 1], dtype=np.uint32),
        "candidate_bitsets": np.asarray([1 << int(actor_id)], dtype=np.uint16),
        "player_counts": np.asarray([player_count], dtype=np.uint8),
        "decision_actor_ids": np.asarray([actor_id], dtype=np.uint8),
        "decision_acts": np.ones(1, dtype=np.uint8),
        "normal_actions": np.asarray([normal_action], dtype=np.uint16),
        "actions": np.asarray([normal_action], dtype=np.uint16),
        "old_log_probs": np.zeros(1, dtype=np.float32),
        "old_values": np.zeros(1, dtype=np.float32),
        "reward_to_next": np.zeros(1, dtype=np.float32),
        "done": np.ones(1, dtype=np.bool_),
        "forced": np.asarray(
            [int(observation.legal_mask.sum()) == 1], dtype=np.bool_
        ),
        "next_decision": np.full(1, -1, dtype=np.int32),
        "history_events": history_events,
        "history_end": history_end,
    }
    privileged = {
        "match_indices": np.asarray([index], dtype=np.uint32),
        "match_seeds": np.asarray([seed], dtype=np.uint32),
        "privileged_states": np.zeros((1, 5), dtype=np.float16),
    }
    return actor, privileged


def _planned_shard_metadata(plan: object, shard: object) -> dict[str, object]:
    behavior = plan.behavior  # type: ignore[attr-defined]
    return {
        **expected_planned_shard_metadata(plan, shard),  # type: ignore[arg-type]
        "behaviorActorManifestSha256": behavior["actorManifestSha256"],
        "behaviorActorSha256": behavior["actorSha256"],
        "behaviorCriticSha256": behavior["criticSha256"],
        "behaviorModelPairId": behavior["pairId"],
        "behaviorModelPairManifestSha256": behavior["pairManifestSha256"],
        "collectionContract": V5_MAPPO_COLLECTION_CONTRACT,
        "matchCounts": {str(shard.player_count): shard.match_count},  # type: ignore[attr-defined]
        "matchProvenanceContract": V5_MATCH_PROVENANCE_CONTRACT,
        "matchShardCount": 1,
        "matchShardIndex": 0,
        "matchStart": shard.match_start,  # type: ignore[attr-defined]
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "rewardContract": V5_MAPPO_REWARD_CONTRACT,
        "runNamespace": plan.run_namespace,  # type: ignore[attr-defined]
        "seedBase": plan.seed_base,  # type: ignore[attr-defined]
    }


class V5CollectionPlanTests(unittest.TestCase):
    def test_plan_directory_publish_is_noreplace_crash_tolerant_and_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            recovered = root / "recovered"
            (root / ".recovered.publish.lock").write_text("pid=dead\n", encoding="ascii")
            orphan = root / ".recovered.crash-orphan"
            orphan.mkdir()
            v5_collection_plan._exclusive_publish_directory(
                recovered, {"payload": b"recovered"}
            )
            self.assertEqual((recovered / "payload").read_bytes(), b"recovered")
            self.assertTrue(orphan.is_dir())

            empty = root / "empty-existing"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                v5_collection_plan._exclusive_publish_directory(
                    empty, {"payload": b"must-not-replace"}
                )
            self.assertEqual(list(empty.iterdir()), [])

            target = root / "concurrent"
            barrier = threading.Barrier(2)
            original_rename = v5_collection_plan._rename_directory_noreplace

            def synchronized_rename(source: Path, destination: Path) -> None:
                barrier.wait(timeout=10)
                original_rename(source, destination)

            def publish(payload: bytes) -> str:
                try:
                    v5_collection_plan._exclusive_publish_directory(
                        target, {"payload": payload}
                    )
                    return "published"
                except FileExistsError:
                    return "exists"

            with mock.patch.object(
                v5_collection_plan,
                "_rename_directory_noreplace",
                side_effect=synchronized_rename,
            ), ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(publish, (b"first", b"second")))
            self.assertCountEqual(outcomes, ["published", "exists"])
            self.assertIn((target / "payload").read_bytes(), (b"first", b"second"))
            self.assertEqual(list(root.glob(".concurrent.*")), [])

            durable = root / "durable"
            fsynced: list[Path] = []
            original_fsync_directory = v5_collection_plan._fsync_directory

            def observe_fsync(path: Path) -> None:
                fsynced.append(Path(path))
                original_fsync_directory(path)

            with mock.patch.object(
                v5_collection_plan, "_fsync_directory", side_effect=observe_fsync
            ):
                v5_collection_plan._exclusive_publish_directory(
                    durable, {"payload": b"durable"}
                )
            self.assertEqual(fsynced[-1], root)
            self.assertTrue(any(path.name.startswith(".durable.") for path in fsynced))

    def test_measured_per_stratum_allocation_balances_predicted_finish_time(self) -> None:
        totals = {player: 100 + player for player in range(4, 11)}
        cpu = {player: float(player) for player in range(4, 11)}
        cuda = {player: float(player) / 2.0 for player in range(4, 11)}
        allocation = completion_balanced_cpu_matches(totals, cpu, cuda)
        for player in range(4, 11):
            expected = int(np.floor(totals[player] / 3.0 + 0.5))
            self.assertEqual(allocation[player], expected)
            cpu_finish = allocation[player] * cpu[player]
            cuda_finish = (totals[player] - allocation[player]) * cuda[player]
            self.assertLessEqual(abs(cpu_finish - cuda_finish), cpu[player])

    def test_plan_seals_measured_allocation_and_splits_both_backends(self) -> None:
        rates = {player: (20, 2_000 + player * 20) for player in range(4, 11)}
        cpu = {player: 2.0 + player / 10.0 for player in range(4, 11)}
        cuda = {player: 1.0 + player / 20.0 for player in range(4, 11)}
        plan = _plan(
            preflight_strata=rates,
            cpu_seconds_per_match=cpu,
            cuda_seconds_per_match=cuda,
            cpu_worker_count=2,
            cuda_worker_count=4,
            cpu_torch_threads_per_worker=1,
            cuda_torch_threads_per_worker=2,
            max_matches_per_shard=100,
        )
        policy = plan.document["backendPolicy"]
        self.assertEqual(
            policy["allocationContract"],
            "measured-per-stratum-equal-finish-time-v1",
        )
        counts = {int(key): value for key, value in plan.document["matchCounts"].items()}
        expected = completion_balanced_cpu_matches(
            counts, cpu, cuda, cpu_worker_count=2, cuda_worker_count=4
        )
        self.assertEqual(policy["cpuWorkerCount"], 2)
        self.assertEqual(policy["cudaWorkerCount"], 4)
        self.assertEqual(policy["cudaTorchThreadsPerWorker"], 2)
        self.assertEqual(
            policy["cpuMatchesByPlayerCount"],
            {str(player): expected[player] for player in range(4, 11)},
        )
        for player in range(4, 11):
            cpu_shards = [
                item for item in plan.shards
                if item.player_count == player and item.backend == "cpu"
            ]
            cuda_shards = [
                item for item in plan.shards
                if item.player_count == player and item.backend == "cuda"
            ]
            self.assertEqual(sum(item.match_count for item in cpu_shards), expected[player])
            self.assertEqual(
                sum(item.match_count for item in cuda_shards),
                counts[player] - expected[player],
            )
            self.assertLessEqual(max(item.match_count for item in cpu_shards + cuda_shards), 100)

        tampered = copy.deepcopy(plan.document)
        tampered["backendPolicy"]["cpuMatchesByPlayerCount"]["4"] += 1
        with self.assertRaisesRegex(ValueError, "does not recompute"):
            __import__("v5_collection_plan").validate_collection_plan_document(tampered)

    def test_planned_shard_metadata_excludes_publisher_owned_contract_fields(self) -> None:
        plan = _plan()
        metadata = expected_planned_shard_metadata(plan, plan.shards[0])

        self.assertNotIn("matchProvenanceContract", metadata)
        self.assertEqual(metadata["collectionPlanManifestSha256"], plan.manifest_sha256)
        self.assertEqual(metadata["plannedShardIndex"], plan.shards[0].index)

    def test_production_rejects_undersampled_stratified_rate_estimate(self) -> None:
        rates = {player: (20, 2_000) for player in range(4, 11)}
        rates[10] = (19, 1_900)
        with self.assertRaisesRegex(ValueError, "at least 20"):
            _plan(preflight_strata=rates)

    def test_stratified_preflight_equalizes_decisions_and_covers_ranges(self) -> None:
        rates = {
            4: (20, 1_980),
            5: (20, 2_160),
            6: (20, 2_360),
            7: (20, 2_560),
            8: (20, 2_760),
            9: (20, 2_960),
            10: (20, 3_120),
        }
        plan = _plan(preflight_strata=rates)
        self.assertEqual(plan.purpose, "production")
        self.assertGreaterEqual(plan.document["totalMatches"], 11_000)
        self.assertLessEqual(plan.document["totalMatches"], 14_000)
        targets = plan.document["targets"]
        estimates = targets["estimatedNonforcedByPlayerCount"]
        self.assertLess(max(estimates.values()) - min(estimates.values()), 160)
        counts = {int(key): value for key, value in plan.document["matchCounts"].items()}
        for player_count in range(4, 11):
            group = [item for item in plan.shards if item.player_count == player_count]
            self.assertEqual({item.backend for item in group}, {"cpu", "cuda"})
            self.assertEqual(group[0].match_start, 0)
            self.assertEqual(group[-1].match_stop, counts[player_count])
            for left, right in zip(group, group[1:]):
                self.assertEqual(left.match_stop, right.match_start)
            self.assertLessEqual(max(item.match_count for item in group), 600)

    def test_explicit_diagnostic_plan_is_deterministic_and_canonical(self) -> None:
        first = _plan()
        second = _plan()
        self.assertEqual(first.document, second.document)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.document["totalMatches"], 12_000)
        self.assertEqual(first.purpose, "diagnostic-unbalanced")
        self.assertEqual(len(first.shards), 28)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "plan"
            digest = publish_collection_plan(target, first)
            self.assertEqual(digest, first.manifest_sha256)
            self.assertEqual(load_collection_plan(target), first)
            raw = (target / "plan.json").read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(json.loads(raw), first.document)

    def test_production_fails_fast_without_stratified_measurements(self) -> None:
        with self.assertRaisesRegex(ValueError, "stratified preflight"):
            _plan(diagnostic_unbalanced=False)
        with self.assertRaisesRegex(ValueError, "stratified preflight"):
            _plan(total_matches=140, diagnostic_unbalanced=False)

    def test_diagnostic_plan_can_never_publish_production_index(self) -> None:
        plan = _plan(total_matches=140)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "diagnostic-unbalanced"):
                publish_verified_index(plan, root / "shards", root / "index")
            self.assertFalse((root / "index").exists())

    def test_allocation_rejects_missing_stratum(self) -> None:
        with self.assertRaisesRegex(ValueError, "p4..p10"):
            allocate_mixed_backend_shards({player: 100 for player in range(4, 10)})

    def test_calibration_report_passes_then_value_drift_fails(self) -> None:
        actor, privileged = _fixture_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            publish_v5_shard(cpu, actor, privileged, metadata=_calibration_metadata("cpu"))
            publish_v5_shard(cuda, actor, privileged, metadata=_calibration_metadata("cuda"))
            report = compare_calibration_shards(cpu, cuda)
            self.assertTrue(report["passed"])
            self.assertIn("actions", report["comparisons"]["exactArrays"])
            report_path = root / "report"
            self.assertEqual(len(publish_calibration_report(report_path, cpu, cuda)), 64)

            bad_actor = copy.deepcopy(actor)
            bad_actor["old_values"] = actor["old_values"].copy()
            bad_actor["old_values"][0] += np.float32(CALIBRATION_VALUE_ATOL * 20)
            bad_cuda = root / "bad-cuda"
            publish_v5_shard(
                bad_cuda, bad_actor, privileged, metadata=_calibration_metadata("cuda")
            )
            with self.assertRaisesRegex(ValueError, "old_values"):
                compare_calibration_shards(cpu, bad_cuda)

    def test_partial_existing_shard_is_never_resumed(self) -> None:
        plan = _plan(total_matches=140)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / plan.shards[0].name
            partial.mkdir()
            (partial / "actor").mkdir()
            with self.assertRaises((ValueError, FileNotFoundError, OSError)):
                resume_verified_shard(plan, plan.shards[0], root)

    def test_global_match_provenance_exactly_covers_plan_and_index_inventory(self) -> None:
        plan = _plan(total_matches=14, max_matches_per_shard=1)
        self.assertTrue(all(shard.match_count == 1 for shard in plan.shards))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for shard in plan.shards:
                actor, privileged = _planned_one_match_arrays(plan, shard)
                publish_v5_shard(
                    planned_shard_path(root, shard),
                    actor,
                    privileged,
                    metadata=_planned_shard_metadata(plan, shard),
                )

            paths = [planned_shard_path(root, shard) for shard in plan.shards]
            result = verify_planned_collection_corpus(
                plan,
                root,
                index_shard_paths=paths,
            )
            self.assertEqual(result["totalUniqueMatches"], 14)
            self.assertEqual(
                result["actualMatchCountsByPlayerCount"],
                {str(player): 2 for player in range(4, 11)},
            )
            self.assertEqual(
                result["matchProvenanceContract"],
                V5_MATCH_PROVENANCE_CONTRACT,
            )
            self.assertEqual(len(str(result["matchCoordinatesSha256"])), 64)

            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_planned_collection_corpus(
                    plan,
                    root,
                    index_shard_paths=paths[:-1],
                )
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_planned_collection_corpus(
                    plan,
                    root,
                    index_shard_paths=[*paths, paths[0]],
                )

    def test_hybrid_index_routes_staged_paths_by_unique_planned_index(self) -> None:
        plan = _plan(total_matches=14, max_matches_per_shard=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged_paths: list[Path] = []
            for shard in plan.shards:
                actor, privileged = _planned_one_match_arrays(plan, shard)
                path = (
                    root
                    / ("persistent" if shard.index % 2 == 0 else "volatile")
                    / f"staged-{len(plan.shards) - shard.index:03d}"
                )
                publish_v5_shard(
                    path,
                    actor,
                    privileged,
                    metadata=_planned_shard_metadata(plan, shard),
                )
                staged_paths.append(path)

            result = verify_planned_collection_corpus(
                plan,
                root / "canonical-root-is-unused-for-indexed-hybrid",
                index_shard_paths=list(reversed(staged_paths)),
            )
            self.assertEqual(result["totalUniqueMatches"], 14)

            duplicate = root / "duplicate-planned-index"
            actor, privileged = _planned_one_match_arrays(plan, plan.shards[0])
            publish_v5_shard(
                duplicate,
                actor,
                privileged,
                metadata=_planned_shard_metadata(plan, plan.shards[0]),
            )
            duplicated_inventory = list(staged_paths)
            duplicated_inventory[1] = duplicate
            with self.assertRaisesRegex(ValueError, "repeats one planned shard"):
                verify_planned_collection_corpus(
                    plan,
                    root,
                    index_shard_paths=duplicated_inventory,
                )

            foreign = root / "foreign-staged-shard"
            foreign_metadata = _planned_shard_metadata(plan, plan.shards[1])
            foreign_metadata["plannedShardIndex"] = len(plan.shards)
            actor, privileged = _planned_one_match_arrays(plan, plan.shards[1])
            publish_v5_shard(
                foreign,
                actor,
                privileged,
                metadata=foreign_metadata,
            )
            foreign_inventory = list(staged_paths)
            foreign_inventory[1] = foreign
            with self.assertRaisesRegex(ValueError, "foreign planned shard"):
                verify_planned_collection_corpus(
                    plan,
                    root,
                    index_shard_paths=foreign_inventory,
                )

    def test_planned_shard_rejects_relabelled_index_and_seed(self) -> None:
        plan = _plan(total_matches=14, max_matches_per_shard=1)
        shard = plan.shards[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor, wrong_index = _planned_one_match_arrays(
                plan,
                shard,
                match_index=shard.match_start + 1,
            )
            wrong_index_path = root / "wrong-index"
            publish_v5_shard(
                wrong_index_path,
                actor,
                wrong_index,
                metadata=_planned_shard_metadata(plan, shard),
            )
            with self.assertRaisesRegex(ValueError, "exact range"):
                verify_planned_shard(plan, shard, wrong_index_path)

            actor, wrong_seed = _planned_one_match_arrays(
                plan,
                shard,
                match_seed=123,
            )
            wrong_seed_path = root / "wrong-seed"
            publish_v5_shard(
                wrong_seed_path,
                actor,
                wrong_seed,
                metadata=_planned_shard_metadata(plan, shard),
            )
            with self.assertRaisesRegex(ValueError, "do not recompute"):
                verify_planned_shard(plan, shard, wrong_seed_path)

    def test_global_match_seed_collision_fails_even_when_every_shard_matches(self) -> None:
        plan = _plan(total_matches=14, max_matches_per_shard=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for shard in plan.shards:
                actor, privileged = _planned_one_match_arrays(
                    plan,
                    shard,
                    match_seed=123,
                )
                publish_v5_shard(
                    planned_shard_path(root, shard),
                    actor,
                    privileged,
                    metadata=_planned_shard_metadata(plan, shard),
                )
            with mock.patch(
                "v5_collection_plan.derive_v5_collection_match_seed",
                return_value=123,
            ), self.assertRaisesRegex(ValueError, "duplicate, gap, or seed drift"):
                verify_planned_collection_corpus(plan, root)

    def test_actual_corpus_gate_rejects_under_over_and_imbalanced_counts(self) -> None:
        plan = _plan(preflight_strata={
            4: (20, 1_980),
            5: (20, 2_160),
            6: (20, 2_360),
            7: (20, 2_560),
            8: (20, 2_760),
            9: (20, 2_960),
            10: (20, 3_120),
        })
        target = float(plan.document["targets"]["stratumTargetNonforcedDecisions"])
        valid = {str(player): int(round(target)) for player in range(4, 11)}
        self.assertTrue(validate_actual_nonforced_corpus(plan, valid)["passed"])

        under = {str(player): 200_000 for player in range(4, 11)}
        with self.assertRaisesRegex(ValueError, "under target"):
            validate_actual_nonforced_corpus(plan, under)
        over = {str(player): 300_000 for player in range(4, 11)}
        with self.assertRaisesRegex(ValueError, "over target"):
            validate_actual_nonforced_corpus(plan, over)
        imbalanced = dict(valid)
        imbalanced["4"] = int(round(target * 1.20))
        imbalanced["5"] = int(round(target * 0.80))
        with self.assertRaisesRegex(ValueError, "equal-stratum"):
            validate_actual_nonforced_corpus(plan, imbalanced)


if __name__ == "__main__":
    unittest.main()
