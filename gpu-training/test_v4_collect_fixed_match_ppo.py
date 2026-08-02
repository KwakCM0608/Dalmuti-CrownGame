from __future__ import annotations

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
    DEFAULT_MATCH_COUNTS,
    FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR,
    FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT,
    FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT_VERSION,
    FIXED_MATCH_BEHAVIOR_TEMPERATURE,
    FIXED_MATCH_INITIAL_LOG_PROBABILITY_TOLERANCE,
    FIXED_MATCH_PPO_PREPARATION_FORMAT,
    FixedMatchPPOCollectionConfig,
    _build_complete_match_specs,
    _parser,
    assert_evaluator_candidate_seat_parity,
    balanced_learner_physical_ids,
    collect_v4_fixed_match_ppo,
    evaluation_candidate_initial_seats,
    evaluator_group_reward_components,
    greedy_masked_candidate_action,
    suffix_reward_components,
)
from v4_collect_ppo import masked_categorical_probabilities
from v4_compare_fixed_match_backends import (
    FixedMatchBackendCalibrationVerification,
    _snapshot_file,
)
from v4_dataset import (
    V4_FIXED_MIXED_BACKEND_COLLECTION_PLAN_ID,
    V4LossEligibility,
    _canonical_fixed_collection_plan_fields,
    _loss_contract_fingerprint,
    fixed_match_shard_identity_sha256,
    load_v4_dataset_npz,
)
from v4_env import ACTION_COUNT, V4ActorObservation, round_chip_award
from v4_evaluate import rotating_candidate_seats
from v4_export import export_v4_actor_bundle
from v4_model import V4ActorConfig, V4PublicActor
from v4_merge_datasets import merge_v4_datasets
from v4_train import (
    V4TrainingConfig,
    _resolve_fixed_collection_plan_sha256,
    train_v4,
)


def _mock_calibration_verification(
    report_sha256: str,
    *,
    recheck_side_effect: object | None = None,
) -> mock.Mock:
    verification = mock.Mock()
    verification.report_sha256 = report_sha256
    if recheck_side_effect is not None:
        verification.recheck_unchanged.side_effect = recheck_side_effect
    return verification


def _rewrite_npz(
    source: Path,
    destination: Path,
    mutate,
) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    mutate(metadata, arrays)
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    Path(f"{destination}.sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii"
    )


class FixedMatchPPOTests(unittest.TestCase):
    MOCK_CALIBRATION_SHA256 = "e" * 64

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        torch.manual_seed(20260802)
        actor = V4PublicActor(V4ActorConfig(
            max_players=10,
            max_history=8,
            d_model=16,
            layers=1,
            heads=4,
            feedforward=32,
            action_hidden=16,
        )).eval()
        cls.bundle = cls.root / "candidate"
        export_v4_actor_bundle(actor, cls.bundle, metadata={"purpose": "fixed-match-test"}, include_onnx=False)

        cls.seen_public_only = True

        def deterministic_logits(model: object, observations: list[V4ActorObservation], device: torch.device) -> list[torch.Tensor]:
            del model, device
            output: list[torch.Tensor] = []
            for observation in observations:
                if not isinstance(observation, V4ActorObservation) or hasattr(observation, "privileged_state"):
                    cls.seen_public_only = False
                    raise AssertionError("Actor crossed the public observation boundary")
                logits = torch.full((ACTION_COUNT,), float("-inf"), dtype=torch.float64)
                legal = torch.nonzero(observation.legal_mask, as_tuple=False).flatten()
                logits[legal] = torch.linspace(-0.25, 0.75, len(legal), dtype=torch.float64)
                output.append(logits)
            return output

        cls.deterministic_logits = staticmethod(deterministic_logits)

        cls.output = cls.root / "collection" / "fixed.npz"
        cls.config = FixedMatchPPOCollectionConfig(
            run_namespace="fixed-small",
            seed_base=950_000_001,
            match_counts=((4, 1),),
            temperature=1.0,
            epsilon_floor=0.0,
            pairwise_coefficient=0.25,
            standardize_advantages=False,
            lane_count=1,
            device="cpu",
        )
        with mock.patch("v4_collect_fixed_match_ppo._batch_candidate_logits", side_effect=deterministic_logits):
            cls.result = collect_v4_fixed_match_ppo(cls.bundle, cls.output, cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _collect_variant(
        self,
        name: str,
        seed_base: int,
        **overrides: object,
    ) -> Path:
        values: dict[str, object] = {
            "run_namespace": name,
            "seed_base": seed_base,
            "match_counts": ((4, 1),),
            "standardize_advantages": False,
            "lane_count": 1,
            "device": "cpu",
        }
        values.update(overrides)
        if (
            values.get("shard_backend_map") is not None
            and "cross_backend_calibration_report" not in values
        ):
            values["cross_backend_calibration_report"] = (
                self.root / "mock-calibration.json"
            )
        if values.get("shard_backend_map") is not None:
            values.setdefault(
                "cross_backend_calibration_cpu_npz",
                self.root / "mock-calibration-cpu.npz",
            )
            values.setdefault(
                "cross_backend_calibration_cuda_npz",
                self.root / "mock-calibration-cuda.npz",
            )
        config = FixedMatchPPOCollectionConfig(**values)
        output = self.root / "variants" / f"{name}-{config.match_shard_index}.npz"
        with mock.patch(
            "v4_collect_fixed_match_ppo._batch_candidate_logits",
            side_effect=self.deterministic_logits,
        ), mock.patch(
            "v4_collect_fixed_match_ppo.load_verified_fixed_match_backend_calibration",
            return_value=_mock_calibration_verification(
                self.MOCK_CALIBRATION_SHA256
            ),
        ):
            collect_v4_fixed_match_ppo(self.bundle, output, config)
        return output

    def test_candidate_initial_seats_match_evaluator_exactly(self) -> None:
        for player_count in range(4, 11):
            for match_index in range(37):
                self.assertEqual(
                    evaluation_candidate_initial_seats(player_count, match_index),
                    rotating_candidate_seats(player_count, match_index),
                )
        audit = assert_evaluator_candidate_seat_parity({p: tuple(range(37)) for p in range(4, 11)})
        self.assertTrue(audit["allEntriesMatched"])
        self.assertEqual(audit["checkedScheduleEntries"], 259)

    def test_group_reward_and_five_act_suffix_math(self) -> None:
        finish = (0, 1, 2, 3)
        chips = {actor: round_chip_award(place, 4) for place, actor in enumerate(finish, 1)}
        candidate_mean, normal_mean, difference, before, comparisons, rate, centered = (
            evaluator_group_reward_components(finish, chips, (0, 2))
        )
        self.assertEqual(candidate_mean, 2.5)
        self.assertEqual(normal_mean, 1.5)
        self.assertEqual(difference, 1.0)
        self.assertEqual((before, comparisons, rate, centered), (3, 4, 0.75, 0.25))
        total, suffix_chip, suffix_pair, suffix_total = suffix_reward_components(
            [1, 2, 3, 4, 5],
            [-0.5, -0.25, 0, 0.25, 0.5],
            0.25,
        )
        self.assertTrue(np.allclose(total, (np.arange(1, 6) + 0.25 * np.linspace(-0.5, 0.5, 5)) / 5))
        self.assertEqual(float(suffix_chip[0]), 15.0)
        self.assertEqual(float(suffix_pair[0]), 0.0)
        self.assertEqual(float(suffix_total[0]), 3.0)
        self.assertEqual(float(suffix_total[-1]), (5 + 0.25 * 0.5) / 5)

    def test_behavior_policy_is_canonical_raw_masked_softmax_only(self) -> None:
        defaults = FixedMatchPPOCollectionConfig("canonical", 1)
        self.assertEqual(defaults.temperature, FIXED_MATCH_BEHAVIOR_TEMPERATURE)
        self.assertEqual(defaults.epsilon_floor, FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR)
        logits = np.linspace(-2.0, 2.0, ACTION_COUNT, dtype=np.float64)
        legal = np.zeros(ACTION_COUNT, dtype=np.bool_)
        legal[[3, 9, 21, 144, 235]] = True
        probabilities = masked_categorical_probabilities(
            logits,
            legal,
            temperature=defaults.temperature,
            epsilon_floor=defaults.epsilon_floor,
        )
        centered = logits[legal] - np.max(logits[legal])
        expected = np.zeros(ACTION_COUNT, dtype=np.float64)
        expected[legal] = np.exp(centered) / np.exp(centered).sum()
        self.assertTrue(np.allclose(probabilities, expected, rtol=0.0, atol=1.0e-15))
        for value in (0.8, 1.2, True):
            with self.subTest(temperature=value):
                with self.assertRaisesRegex(ValueError, "temperature=1.0"):
                    FixedMatchPPOCollectionConfig("noncanonical-temperature", 1, temperature=value)
        for value in (1.0e-6, -1.0e-6, False):
            with self.subTest(epsilon_floor=value):
                with self.assertRaisesRegex(ValueError, "epsilon_floor=0.0"):
                    FixedMatchPPOCollectionConfig("noncanonical-floor", 1, epsilon_floor=value)

    def test_mixed_backend_map_is_complete_and_rejects_wrong_shard_device(self) -> None:
        backend_map = ("cpu", "cpu", *("cuda",) * 12)
        parsed = _parser().parse_args(
            [
                "--actor-bundle", "candidate",
                "--output", "shard.npz",
                "--run-namespace", "mixed-fourteen-cli",
                "--seed-base", "1",
                "--match-shard-count", "14",
                "--match-shard-index", "1",
                "--device", "cpu",
                "--shard-backend-map", ",".join(backend_map),
                "--cross-backend-calibration-report", "calibration.json",
                "--cross-backend-calibration-cpu-npz", "calibration-cpu.npz",
                "--cross-backend-calibration-cuda-npz", "calibration-cuda.npz",
            ]
        )
        self.assertEqual(parsed.shard_backend_map, backend_map)
        self.assertEqual(
            parsed.cross_backend_calibration_report, Path("calibration.json")
        )
        self.assertEqual(
            parsed.cross_backend_calibration_cpu_npz, Path("calibration-cpu.npz")
        )
        self.assertEqual(
            parsed.cross_backend_calibration_cuda_npz, Path("calibration-cuda.npz")
        )
        accepted_cpu = FixedMatchPPOCollectionConfig(
            "mixed-fourteen",
            1,
            match_shard_count=14,
            match_shard_index=1,
            device="cpu:0",
            shard_backend_map=backend_map,
            cross_backend_calibration_report=Path("calibration.json"),
            cross_backend_calibration_cpu_npz=Path("calibration-cpu.npz"),
            cross_backend_calibration_cuda_npz=Path("calibration-cuda.npz"),
        )
        accepted_cuda = FixedMatchPPOCollectionConfig(
            "mixed-fourteen",
            1,
            match_shard_count=14,
            match_shard_index=13,
            device="cuda:1",
            shard_backend_map=backend_map,
            cross_backend_calibration_report=Path("calibration.json"),
            cross_backend_calibration_cpu_npz=Path("calibration-cpu.npz"),
            cross_backend_calibration_cuda_npz=Path("calibration-cuda.npz"),
        )
        self.assertEqual(accepted_cpu.shard_backend_map, backend_map)
        self.assertEqual(accepted_cuda.shard_backend_map, backend_map)

        with self.assertRaisesRegex(ValueError, "precommitted to backend cuda"):
            FixedMatchPPOCollectionConfig(
                "mixed-wrong-device",
                1,
                match_shard_count=14,
                match_shard_index=2,
                device="cpu",
                shard_backend_map=backend_map,
                cross_backend_calibration_report=Path("calibration.json"),
            )
        invalid_maps = (
            backend_map[:-1],
            ("cpu",) * 14,
            ("cuda",) * 14,
            ("cpu", "cpu", *("gpu",) * 12),
            list(backend_map),
        )
        for invalid in invalid_maps:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "complete mixed cpu/cuda"):
                    FixedMatchPPOCollectionConfig(
                        "mixed-invalid-map",
                        1,
                        match_shard_count=14,
                        device="cpu",
                        shard_backend_map=invalid,  # type: ignore[arg-type]
                        cross_backend_calibration_report=Path("calibration.json"),
                    )

        incomplete_calibrations = (
            {},
            {"cross_backend_calibration_report": Path("calibration.json")},
            {
                "cross_backend_calibration_report": Path("calibration.json"),
                "cross_backend_calibration_cpu_npz": Path("calibration-cpu.npz"),
            },
            {
                "cross_backend_calibration_report": Path("calibration.json"),
                "cross_backend_calibration_cuda_npz": Path("calibration-cuda.npz"),
            },
            {"cross_backend_calibration_cpu_npz": Path("calibration-cpu.npz")},
            {"cross_backend_calibration_cuda_npz": Path("calibration-cuda.npz")},
            {
                "cross_backend_calibration_cpu_npz": Path("calibration-cpu.npz"),
                "cross_backend_calibration_cuda_npz": Path("calibration-cuda.npz"),
            },
        )
        for calibration in incomplete_calibrations:
            with self.subTest(calibration=calibration):
                with self.assertRaisesRegex(ValueError, "required together iff"):
                    FixedMatchPPOCollectionConfig(
                        "mixed-incomplete-calibration",
                        1,
                        match_shard_count=14,
                        device="cpu",
                        shard_backend_map=backend_map,
                        **calibration,
                    )
        with self.assertRaisesRegex(ValueError, "required together iff"):
            FixedMatchPPOCollectionConfig(
                "v1-report-forbidden",
                1,
                device="cpu",
                cross_backend_calibration_report=Path("calibration.json"),
            )

    def test_v2_fourteen_shard_plan_hash_binds_complete_backend_map(self) -> None:
        hashes = {
            "completeUnshardedLearnerAssignmentSha256": "a" * 64,
            "actorCheckpointSha256": "b" * 64,
            "bundleManifestSha256": "c" * 64,
            "sourceHashesSha256": "d" * 64,
            "crossBackendCalibrationReportSha256": "e" * 64,
        }
        backend_map = {
            str(index): "cpu" if index < 2 else "cuda"
            for index in range(14)
        }
        fields = {
            "version": 2,
            "runNamespace": "mixed-fourteen-plan",
            "seedBase": 530_000_001,
            "matchCounts": {str(player): 1 for player in range(4, 11)},
            "matchStart": 0,
            "matchShardCount": 14,
            **hashes,
            "rewardContract": "reward-id",
            "behaviorPolicyContract": "behavior-id",
            "shardBackendMap": backend_map,
        }
        plan_id, canonical = _canonical_fixed_collection_plan_fields(fields)
        self.assertTrue(
            plan_id.startswith(
                f"{V4_FIXED_MIXED_BACKEND_COLLECTION_PLAN_ID}:sha256="
            )
        )
        self.assertEqual(canonical["shardBackendMap"], backend_map)
        self.assertEqual(
            canonical["crossBackendCalibrationReportSha256"], "e" * 64
        )

        reordered = dict(fields)
        reordered["shardBackendMap"] = {
            key: backend_map[key] for key in reversed(list(backend_map))
        }
        self.assertEqual(
            _canonical_fixed_collection_plan_fields(reordered)[0], plan_id
        )

        changed = dict(fields)
        changed_map = dict(backend_map)
        changed_map["1"] = "cuda"
        changed["shardBackendMap"] = changed_map
        self.assertNotEqual(
            _canonical_fixed_collection_plan_fields(changed)[0], plan_id
        )

        changed_calibration = dict(fields)
        changed_calibration["crossBackendCalibrationReportSha256"] = "f" * 64
        self.assertNotEqual(
            _canonical_fixed_collection_plan_fields(changed_calibration)[0],
            plan_id,
        )

        for invalid_map in (
            {key: value for key, value in backend_map.items() if key != "13"},
            {**backend_map, "14": "cuda"},
            {**backend_map, "02": "cpu"},
            {str(index): "cpu" for index in range(14)},
        ):
            with self.subTest(invalid_map=invalid_map):
                invalid = dict(fields)
                invalid["shardBackendMap"] = invalid_map
                with self.assertRaisesRegex(ValueError, "backend"):
                    _canonical_fixed_collection_plan_fields(invalid)

    def test_nonzero_actor_dropout_is_rejected_before_collection(self) -> None:
        actor = V4PublicActor(V4ActorConfig(
            max_players=10,
            max_history=8,
            d_model=16,
            layers=1,
            heads=4,
            feedforward=32,
            action_hidden=16,
            dropout=0.1,
        )).eval()
        bundle = self.root / "dropout-candidate"
        export_v4_actor_bundle(
            actor,
            bundle,
            metadata={"purpose": "dropout-rejection-test"},
            include_onnx=False,
        )
        output = self.root / "dropout-rejected" / "fixed.npz"
        with self.assertRaisesRegex(ValueError, "dropout=0.0"):
            collect_v4_fixed_match_ppo(
                bundle,
                output,
                FixedMatchPPOCollectionConfig(
                    run_namespace="dropout-rejected",
                    seed_base=950_000_002,
                    match_counts=((4, 1),),
                    lane_count=1,
                    device="cpu",
                ),
            )
        self.assertFalse(output.exists())

    def test_mixed_collection_rejects_wrong_calibration_before_rollout(self) -> None:
        output = self.root / "wrong-calibration" / "fixed.npz"
        config = FixedMatchPPOCollectionConfig(
            run_namespace="wrong-calibration",
            seed_base=950_000_003,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
            lane_count=1,
            device="cpu",
            shard_backend_map=("cpu", "cuda"),
            cross_backend_calibration_report=self.root / "wrong-report.json",
            cross_backend_calibration_cpu_npz=self.root / "wrong-cpu.npz",
            cross_backend_calibration_cuda_npz=self.root / "wrong-cuda.npz",
        )
        with mock.patch(
            "v4_collect_fixed_match_ppo.load_verified_fixed_match_backend_calibration",
            side_effect=ValueError("calibration report model/source binding does not match"),
        ) as calibration, mock.patch(
            "v4_collect_fixed_match_ppo._batch_candidate_logits"
        ) as logits:
            with self.assertRaisesRegex(ValueError, "model/source binding"):
                collect_v4_fixed_match_ppo(self.bundle, output, config)
        self.assertEqual(calibration.call_count, 1)
        self.assertEqual(
            calibration.call_args.args,
            (
                self.root / "wrong-report.json",
                self.root / "wrong-cpu.npz",
                self.root / "wrong-cuda.npz",
            ),
        )
        logits.assert_not_called()
        self.assertFalse(output.exists())

    def test_mixed_collection_rejects_identical_byte_replacement_before_publish(
        self,
    ) -> None:
        marker = self.root / "calibration-lifetime" / "report.json"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"immutable calibration bytes")
        verification = FixedMatchBackendCalibrationVerification(
            report_sha256=self.MOCK_CALIBRATION_SHA256,
            snapshots=(_snapshot_file(marker, "calibration lifetime marker"),),
        )
        output = self.root / "calibration-lifetime" / "fixed.npz"
        config = FixedMatchPPOCollectionConfig(
            run_namespace="calibration-lifetime",
            seed_base=950_000_004,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
            lane_count=1,
            device="cpu",
            shard_backend_map=("cpu", "cuda"),
            cross_backend_calibration_report=marker,
            cross_backend_calibration_cpu_npz=self.root / "unused-cpu.npz",
            cross_backend_calibration_cuda_npz=self.root / "unused-cuda.npz",
        )
        replaced = False

        def replace_with_identical_bytes(*args, **kwargs):
            nonlocal replaced
            if not replaced:
                replaced = True
                replacement = marker.with_suffix(".replacement")
                replacement.write_bytes(marker.read_bytes())
                os.replace(replacement, marker)
            return self.deterministic_logits(*args, **kwargs)

        with mock.patch(
            "v4_collect_fixed_match_ppo.load_verified_fixed_match_backend_calibration",
            return_value=verification,
        ), mock.patch(
            "v4_collect_fixed_match_ppo._batch_candidate_logits",
            side_effect=replace_with_identical_bytes,
        ):
            with self.assertRaisesRegex(ValueError, "changed after immutable verification"):
                collect_v4_fixed_match_ppo(self.bundle, output, config)
        self.assertTrue(replaced)
        for path in (
            output,
            Path(f"{output}.sha256"),
            Path(f"{output}.metadata.json"),
            Path(f"{output}.metadata.json.sha256"),
        ):
            self.assertFalse(path.exists())

    def test_mixed_collection_rolls_back_if_post_publish_recheck_fails(self) -> None:
        output = self.root / "calibration-post-publish" / "fixed.npz"
        config = FixedMatchPPOCollectionConfig(
            run_namespace="calibration-post-publish",
            seed_base=950_000_005,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
            lane_count=1,
            device="cpu",
            shard_backend_map=("cpu", "cuda"),
            cross_backend_calibration_report=self.root / "unused-report.json",
            cross_backend_calibration_cpu_npz=self.root / "unused-cpu.npz",
            cross_backend_calibration_cuda_npz=self.root / "unused-cuda.npz",
        )
        verification = _mock_calibration_verification(
            self.MOCK_CALIBRATION_SHA256,
            recheck_side_effect=[
                None,
                ValueError("calibration changed immediately after publication"),
            ],
        )
        with mock.patch(
            "v4_collect_fixed_match_ppo.load_verified_fixed_match_backend_calibration",
            return_value=verification,
        ), mock.patch(
            "v4_collect_fixed_match_ppo._batch_candidate_logits",
            side_effect=self.deterministic_logits,
        ):
            with self.assertRaisesRegex(ValueError, "immediately after publication"):
                collect_v4_fixed_match_ppo(self.bundle, output, config)
        self.assertEqual(verification.recheck_unchanged.call_count, 2)
        for path in (
            output,
            Path(f"{output}.sha256"),
            Path(f"{output}.metadata.json"),
            Path(f"{output}.metadata.json.sha256"),
        ):
            self.assertFalse(path.exists())

    def test_frozen_candidate_teammate_is_exact_greedy_masked_argmax(self) -> None:
        logits = np.full(ACTION_COUNT, -100.0)
        legal = np.zeros(ACTION_COUNT, np.bool_)
        legal[[3, 9, 21]] = True
        logits[[3, 9, 21]] = [4.0, 7.0, 7.0]
        self.assertEqual(greedy_masked_candidate_action(logits, legal), 9)
        logits[9] = -999.0
        self.assertEqual(greedy_masked_candidate_action(logits, legal), 21)

    def test_legacy_loss_fingerprint_bytes_remain_unchanged(self) -> None:
        mask = torch.tensor([[True, False]], dtype=torch.bool)
        actor_sha = hashlib.sha256(b"legacy-actor").hexdigest()
        eligibility = V4LossEligibility(
            mask,
            mask,
            mask,
            "dalmuti-v4-ppo-league-direct-npz",
            1,
            (actor_sha,),
            ("legacy-per-act-v1",),
            False,
        )
        digest = hashlib.sha256()
        digest.update(eligibility.preparation_format.encode("utf-8"))
        digest.update(b"1")
        for name, value in eligibility.masks().items():
            tensor = value.contiguous()
            digest.update(name.encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(actor_sha.encode("ascii"))
        self.assertEqual(_loss_contract_fingerprint(eligibility), digest.hexdigest())
        self.assertFalse(eligibility.requires_qboost_coefficient_zero)
        self.assertEqual(eligibility.ppo_reward_contracts, ())
        self.assertEqual(eligibility.ppo_behavior_policy_contracts, ())
        self.assertEqual(eligibility.fixed_collection_plan_ids, ())

    def test_physical_learner_assignment_is_complete_plan_balanced(self) -> None:
        orders = {index: tuple((actor + index) % 4 for actor in range(4)) for index in range(20)}
        assignment = balanced_learner_physical_ids(4, orders)
        counts = [sum(int(actor == value) for value in assignment.values()) for actor in range(4)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        for index, actor in assignment.items():
            candidate = {orders[index][seat] for seat in evaluation_candidate_initial_seats(4, index)}
            self.assertIn(actor, candidate)

    def test_production_twelve_shards_cover_exact_counts_without_semantic_drift(self) -> None:
        base = dict(
            run_namespace="production-plan",
            seed_base=951_000_001,
            match_counts=DEFAULT_MATCH_COUNTS,
            match_shard_count=12,
            lane_count=16,
            device="cpu",
        )
        complete, _ = _build_complete_match_specs(FixedMatchPPOCollectionConfig(match_shard_index=0, **base))
        expected_total = sum(count for _, count in DEFAULT_MATCH_COUNTS)
        self.assertEqual(len(complete), expected_total)
        seen: set[tuple[int, int]] = set()
        expected_groups = {
            **{index: [27, 22, 16, 14, 11, 10, 8] for index in range(4)},
            **{index: [27, 21, 16, 13, 11, 9, 8] for index in range(4, 8)},
            **{index: [26, 21, 16, 13, 10, 9, 8] for index in range(8, 12)},
        }
        assignment_hash_rows = [
            (spec.player_count, spec.match_index, spec.seed, spec.learner_initial_seat, spec.learner_physical_id)
            for spec in complete
        ]
        for shard_index in range(12):
            config = FixedMatchPPOCollectionConfig(match_shard_index=shard_index, **base)
            shard_complete, _ = _build_complete_match_specs(config)
            self.assertEqual(
                assignment_hash_rows,
                [(s.player_count, s.match_index, s.seed, s.learner_initial_seat, s.learner_physical_id) for s in shard_complete],
            )
            selected = [s for s in shard_complete if s.match_index % 12 == shard_index]
            counts = [sum(int(s.player_count == p) for s in selected) for p in range(4, 11)]
            self.assertEqual(counts, expected_groups[shard_index])
            self.assertEqual(len(selected) * 5, sum(expected_groups[shard_index]) * 5)
            keys = {(s.player_count, s.match_index) for s in selected}
            self.assertTrue(seen.isdisjoint(keys))
            seen.update(keys)
        self.assertEqual(len(seen), expected_total)

    def test_collected_contract_is_public_fixed_complete_and_direct_loadable(self) -> None:
        self.assertTrue(self.seen_public_only)
        self.assertEqual(self.result.complete_matches, 1)
        self.assertEqual(self.result.trajectories, 5)
        dataset = load_v4_dataset_npz(self.output)
        self.assertTrue(torch.equal(dataset.loss_eligibility.ppo, dataset.tensors.valid_masks))
        self.assertEqual(dataset.loss_eligibility.ppo_source_contracts, ("fixed-physical-id-five-act-suffix-v1",))
        self.assertTrue(dataset.loss_eligibility.requires_player_count_balanced_loss)
        self.assertTrue(dataset.loss_eligibility.requires_qboost_coefficient_zero)
        self.assertEqual(len(dataset.loss_eligibility.ppo_reward_contracts), 1)
        self.assertIn("lambda=0.25:sha256=", dataset.loss_eligibility.ppo_reward_contracts[0])
        self.assertEqual(len(dataset.loss_eligibility.ppo_behavior_policy_contracts), 1)
        self.assertTrue(
            dataset.loss_eligibility.ppo_behavior_policy_contracts[0].startswith(
                "raw-masked-softmax-v1:sha256="
            )
        )
        with np.load(self.output, allow_pickle=False) as archive:
            self.assertEqual(archive["trajectory_acts"].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(len(set(archive["trajectory_actor_ids"].tolist())), 1)
            self.assertEqual(len(set(archive["trajectory_complete_match_ids"].tolist())), 1)
            self.assertEqual(len(set(archive["trajectory_candidate_ids"].tolist())), 1)
            valid = archive["valid_masks"]
            self.assertTrue(np.allclose(
                archive["old_action_log_probs"][valid],
                np.log(archive["selected_action_probabilities"][valid]),
                atol=2.0e-6,
            ))
            self.assertTrue(np.allclose(archive["raw_returns"][valid], archive["suffix_total_returns"][valid]))
        metadata = json.loads(self.result.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["preparationFormat"], FIXED_MATCH_PPO_PREPARATION_FORMAT)
        collection = metadata["collection"]
        self.assertEqual(collection["behaviorPolicyContract"], FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT)
        self.assertEqual(
            collection["behaviorPolicyContractVersion"],
            FIXED_MATCH_BEHAVIOR_POLICY_CONTRACT_VERSION,
        )
        self.assertTrue(collection["rawMaskedSoftmaxExactBinding"])
        self.assertTrue(
            collection["initialOldCurrentRatioMathematicallyOneForFrozenActor"]
        )
        self.assertEqual(
            collection["initialOldCurrentLogProbabilityAbsoluteTolerance"],
            FIXED_MATCH_INITIAL_LOG_PROBABILITY_TOLERANCE,
        )
        self.assertTrue(collection["fixedPpoActorAutocastDisabled"])
        self.assertTrue(
            collection["requiresFullDatasetInitialPolicyReproductionAudit"]
        )
        self.assertTrue(collection["dropoutDisabled"])
        self.assertEqual(collection["temperature"], FIXED_MATCH_BEHAVIOR_TEMPERATURE)
        self.assertEqual(
            collection["epsilonFloorPerLegalAction"],
            FIXED_MATCH_BEHAVIOR_EPSILON_FLOOR,
        )
        self.assertEqual(collection["requestedLaneCount"], self.config.lane_count)
        self.assertEqual(collection["rollingCpuEnvironmentLanes"], 1)
        self.assertEqual(metadata["execution"]["device"], "cpu")
        self.assertEqual(metadata["trainingRequirements"]["qBoostCoefficient"], 0.0)
        self.assertTrue(metadata["environmentBinding"]["candidateSeatParityAudit"]["allEntriesMatched"])
        self.assertGreater(metadata["actionRates"]["greedyCandidateTeammate"]["decisions"], 0)
        self.assertGreater(metadata["actionRates"]["exactNormalOpponent"]["decisions"], 0)
        self.assertEqual(metadata["playerCountDistribution"]["4"]["learnerActTrajectories"], 5)
        for relative, digest in metadata["sourceHashes"].items():
            self.assertEqual(len(digest), 64, relative)

    def test_resume_existing_is_exact_and_fail_closed(self) -> None:
        resumed_config = FixedMatchPPOCollectionConfig(**{
            **self.config.__dict__,
            "resume_existing": True,
        })
        resumed = collect_v4_fixed_match_ppo(self.bundle, self.output, resumed_config)
        self.assertEqual(resumed.npz_sha256, self.result.npz_sha256)
        mismatches = (
            {"pairwise_coefficient": 0.5},
            {"standardize_advantages": True},
            # The selected corpus has one match, so lane_count=2 has the same
            # effective rolling lane count as lane_count=1.  The requested
            # execution contract must still remain exact.
            {"lane_count": 2},
            {"device": "cpu:0"},
        )
        for change in mismatches:
            with self.subTest(change=change):
                mismatched = FixedMatchPPOCollectionConfig(**{
                    **self.config.__dict__,
                    **change,
                    "resume_existing": True,
                })
                with self.assertRaisesRegex(ValueError, "exact requested shard"):
                    collect_v4_fixed_match_ppo(self.bundle, self.output, mismatched)

    def test_raw_outcome_tampering_is_rejected_without_actor_input_leakage(self) -> None:
        tampered = self.root / "tampered.npz"
        with np.load(self.output, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            original_public = arrays["global_features"].copy()
        arrays["trajectory_act_group_chip_differences"][0] += 1.0
        np.savez_compressed(tampered, **arrays)
        with np.load(tampered, allow_pickle=False) as archive:
            self.assertTrue(np.array_equal(archive["global_features"], original_public))
        with self.assertRaisesRegex(ValueError, "raw component|raw group reward math"):
            load_v4_dataset_npz(tampered)

    def test_fixed_shard_merges_and_qboost_requirement_propagates(self) -> None:
        merged = self.root / "merged" / "fixed-merged.npz"
        result = merge_v4_datasets(self.output, merged)
        dataset = load_v4_dataset_npz(result.output_path)
        self.assertTrue(torch.equal(dataset.loss_eligibility.ppo, dataset.tensors.valid_masks))
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(metadata["lossEligibility"]["requiresQBoostCoefficientZero"])
        self.assertEqual(metadata["lossEligibility"]["ppoSourceContracts"], ["fixed-physical-id-five-act-suffix-v1"])
        self.assertTrue(metadata["lossEligibility"]["requiresPlayerCountBalancedLoss"])
        with np.load(result.output_path, allow_pickle=False) as archive:
            self.assertEqual(archive["trajectory_player_counts"].tolist(), [4] * 5)
            self.assertEqual(archive["trajectory_complete_match_ids"].tolist(), archive["trajectory_match_clusters"].tolist())
        train_output = self.root / "forbidden-qboost"
        with self.assertRaisesRegex(ValueError, "q_boost_coefficient=0"):
            train_v4(
                dataset,
                train_output,
                V4TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    bc_weight=0.0,
                    ppo_weight=1.0,
                    critic_weight=0.1,
                    q_boost_coefficient=0.1,
                    entropy_coefficient=0.0,
                    amp=False,
                ),
                device="cpu",
                initialize_actor_bundle=self.bundle,
            )
        self.assertFalse(train_output.exists())

    def test_full_player_count_plan_survives_canonical_json_reload(self) -> None:
        full_counts = tuple((player_count, 1) for player_count in range(4, 11))
        source = self._collect_variant(
            "all-player-counts",
            951_500_001,
            match_counts=full_counts,
            lane_count=7,
        )
        result = merge_v4_datasets(
            source,
            self.root / "all-player-counts" / "merged.npz",
        )

        # Canonical JSON deliberately sorts object keys lexicographically, so
        # p10 is reloaded before p4.  Loading must treat mapping order as wire
        # representation only and restore the numeric p4..p10 provenance order.
        with np.load(result.output_path, allow_pickle=False) as archive:
            raw_metadata = json.loads(str(archive["metadata_json"].item()))
            raw_plan = raw_metadata["lossEligibility"]["fixedCollectionPlans"][0]
            raw_counts = raw_plan["canonicalFields"]["matchCounts"]
            self.assertEqual(list(raw_counts), ["10", "4", "5", "6", "7", "8", "9"])
            self.assertEqual(
                sorted(set(archive["trajectory_player_counts"].tolist())),
                list(range(4, 11)),
            )
            self.assertEqual(
                int(np.count_nonzero(archive["trajectory_player_counts"] == 10)),
                5,
            )

        dataset = load_v4_dataset_npz(result.output_path)
        self.assertEqual(len(dataset.loss_eligibility.fixed_collection_plan_ids), 1)

        reserialized = self.root / "all-player-counts" / "reserialized.npz"

        def no_change(
            metadata: dict[str, object], arrays: dict[str, np.ndarray]
        ) -> None:
            del metadata, arrays

        _rewrite_npz(result.output_path, reserialized, no_change)
        reloaded = load_v4_dataset_npz(reserialized)
        self.assertEqual(
            reloaded.loss_eligibility.fixed_collection_plan_ids,
            dataset.loss_eligibility.fixed_collection_plan_ids,
        )

        fields = raw_plan["canonicalFields"]
        plan_id, canonical = _canonical_fixed_collection_plan_fields(fields)
        self.assertEqual(list(canonical["matchCounts"]), [str(p) for p in range(4, 11)])
        reversed_fields = dict(fields)
        reversed_fields["matchCounts"] = {
            key: raw_counts[key] for key in reversed(list(raw_counts))
        }
        reversed_id, reversed_canonical = _canonical_fixed_collection_plan_fields(
            reversed_fields
        )
        self.assertEqual(reversed_id, plan_id)
        self.assertEqual(reversed_canonical, canonical)

        invalid_counts = (
            {**raw_counts, "04": 1},
            {**raw_counts, 4: 1},
            {key: value for key, value in raw_counts.items() if key != "10"}
            | {"11": 1},
            {**raw_counts, "4": True},
            {**raw_counts, "4": 0},
            {**raw_counts, "4": 1.0},
        )
        for counts in invalid_counts:
            with self.subTest(counts=counts):
                tampered_fields = dict(fields)
                tampered_fields["matchCounts"] = counts
                with self.assertRaisesRegex(
                    ValueError, "match counts are non-canonical"
                ):
                    _canonical_fixed_collection_plan_fields(tampered_fields)

        for field_mutation in ("missing", "extra"):
            with self.subTest(field_mutation=field_mutation):
                tampered_fields = dict(fields)
                if field_mutation == "missing":
                    tampered_fields.pop("sourceHashesSha256")
                else:
                    tampered_fields["unexpected"] = True
                with self.assertRaisesRegex(
                    ValueError, "plan fields are non-canonical"
                ):
                    _canonical_fixed_collection_plan_fields(tampered_fields)

    def test_direct_behavior_and_dropout_contract_tampering_is_rejected(self) -> None:
        behavior_tamper = self.root / "tamper-contract" / "behavior.npz"

        def change_behavior(metadata: dict[str, object], arrays: dict[str, np.ndarray]) -> None:
            del arrays
            collection = metadata["collection"]
            assert isinstance(collection, dict)
            collection["temperature"] = 0.8

        _rewrite_npz(self.output, behavior_tamper, change_behavior)
        with self.assertRaisesRegex(ValueError, "raw masked Actor softmax|temperature=1.0"):
            load_v4_dataset_npz(behavior_tamper)

        dropout_tamper = self.root / "tamper-contract" / "dropout.npz"

        def change_dropout(metadata: dict[str, object], arrays: dict[str, np.ndarray]) -> None:
            del arrays
            actor_config = metadata["actorConfig"]
            assert isinstance(actor_config, dict)
            actor_config["dropout"] = 0.1

        _rewrite_npz(self.output, dropout_tamper, change_dropout)
        with self.assertRaisesRegex(ValueError, "metadata semantics|dropout"):
            load_v4_dataset_npz(dropout_tamper)

    def test_fixed_collection_plan_namespace_and_seed_domains_fail_closed(self) -> None:
        direct_tampers = {
            "namespace": (".hidden", "metadata semantics"),
            "seed": (2**32, "metadata semantics"),
        }
        for name, (invalid_value, pattern) in direct_tampers.items():
            with self.subTest(kind="direct", name=name):
                tampered = self.root / "plan-domain" / f"direct-{name}.npz"

                def mutate_direct(
                    metadata: dict[str, object],
                    arrays: dict[str, np.ndarray],
                    *,
                    name=name,
                    invalid_value=invalid_value,
                ) -> None:
                    del arrays
                    shard = metadata["shard"]
                    assert isinstance(shard, dict)
                    shard[
                        "runNamespace" if name == "namespace" else "seedBase"
                    ] = invalid_value

                _rewrite_npz(self.output, tampered, mutate_direct)
                with self.assertRaisesRegex(ValueError, pattern):
                    load_v4_dataset_npz(tampered)

        merged = merge_v4_datasets(
            self.output,
            self.root / "plan-domain" / "nested-source.npz",
        )
        nested_tampers = {
            "namespace": ".hidden",
            "seed": 2**32,
        }
        for name, invalid_value in nested_tampers.items():
            with self.subTest(kind="nested", name=name):
                tampered = self.root / "plan-domain" / f"nested-{name}.npz"

                def mutate_nested(
                    metadata: dict[str, object],
                    arrays: dict[str, np.ndarray],
                    *,
                    name=name,
                    invalid_value=invalid_value,
                ) -> None:
                    del arrays
                    contract = metadata["lossEligibility"]
                    assert isinstance(contract, dict)
                    plans = contract["fixedCollectionPlans"]
                    assert isinstance(plans, list) and isinstance(plans[0], dict)
                    fields = plans[0]["canonicalFields"]
                    assert isinstance(fields, dict)
                    fields[
                        "runNamespace" if name == "namespace" else "seedBase"
                    ] = invalid_value

                _rewrite_npz(merged.output_path, tampered, mutate_nested)
                with self.assertRaisesRegex(
                    ValueError, "fixed collection plan provenance"
                ):
                    load_v4_dataset_npz(tampered)

    def test_mismatched_fixed_reward_lambda_is_rejected_before_merge(self) -> None:
        alternate = self._collect_variant(
            "lambda-half",
            952_000_001,
            pairwise_coefficient=0.5,
        )
        with self.assertRaisesRegex(ValueError, "share one reward lambda/formula"):
            merge_v4_datasets(
                [self.output, alternate],
                self.root / "lambda-mismatch" / "merged.npz",
            )

    def test_incomplete_multishard_collection_plan_is_rejected(self) -> None:
        shard_zero = self._collect_variant(
            "coverage-two",
            953_000_001,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
        )
        with self.assertRaisesRegex(ValueError, "every shard index exactly once"):
            merge_v4_datasets(
                shard_zero,
                self.root / "incomplete-plan" / "merged.npz",
            )

    def test_complete_multishard_collection_plan_is_accepted(self) -> None:
        shard_zero = self._collect_variant(
            "coverage-complete",
            954_000_001,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
        )
        shard_one = self._collect_variant(
            "coverage-complete",
            954_000_001,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=1,
        )
        result = merge_v4_datasets(
            [shard_zero, shard_one],
            self.root / "complete-plan" / "merged.npz",
        )
        dataset = load_v4_dataset_npz(result.output_path)
        self.assertEqual(len(dataset.loss_eligibility.fixed_collection_plan_ids), 1)
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        plan = metadata["lossEligibility"]["fixedCollectionPlans"][0]
        self.assertEqual(plan["coveredShardIndices"], [0, 1])
        self.assertEqual(
            set(np.load(result.output_path, allow_pickle=False)["trajectory_match_indices"].tolist()),
            {0, 1},
        )

    def test_mixed_backend_plan_survives_merge_and_trainer_binding(self) -> None:
        backend_map = ("cpu", "cuda")
        cpu_shard = self._collect_variant(
            "mixed-coverage",
            954_500_001,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
            shard_backend_map=backend_map,
        )
        cuda_source = self._collect_variant(
            "mixed-coverage",
            954_500_001,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=1,
        )
        cuda_shard = self.root / "mixed-plan" / "synthetic-cuda-shard.npz"

        def bind_synthetic_cuda_execution(
            metadata: dict[str, object], arrays: dict[str, np.ndarray]
        ) -> None:
            del arrays
            shard = metadata["shard"]
            collection = metadata["collection"]
            execution = metadata["execution"]
            assert isinstance(shard, dict)
            assert isinstance(collection, dict)
            assert isinstance(execution, dict)
            shard["collectionPlanVersion"] = 2
            shard["shardBackendMap"] = {"0": "cpu", "1": "cuda"}
            shard["crossBackendCalibrationReportSha256"] = (
                self.MOCK_CALIBRATION_SHA256
            )
            shard["identitySha256"] = fixed_match_shard_identity_sha256(shard)
            execution["device"] = "cuda"
            execution["cudaAvailable"] = True
            execution["tf32Allowed"] = False
            execution["cublasWorkspaceConfig"] = ":4096:8"
            execution["fixedCollectionPlanVersion"] = 2
            execution["plannedShardBackend"] = "cuda"
            collection["batchedGpuMaskedLogitInference"] = True

        _rewrite_npz(cuda_source, cuda_shard, bind_synthetic_cuda_execution)
        load_v4_dataset_npz(cuda_shard)

        result = merge_v4_datasets(
            [cpu_shard, cuda_shard],
            self.root / "mixed-plan" / "merged.npz",
        )
        dataset = load_v4_dataset_npz(result.output_path)
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        plan = metadata["lossEligibility"]["fixedCollectionPlans"][0]
        self.assertEqual(plan["canonicalFields"]["version"], 2)
        self.assertEqual(
            plan["canonicalFields"]["shardBackendMap"],
            {"0": "cpu", "1": "cuda"},
        )
        self.assertEqual(
            plan["canonicalFields"]["crossBackendCalibrationReportSha256"],
            self.MOCK_CALIBRATION_SHA256,
        )
        self.assertEqual(plan["coveredShardIndices"], [0, 1])
        plan_id = dataset.loss_eligibility.fixed_collection_plan_ids[0]
        self.assertTrue(
            plan_id.startswith(
                f"{V4_FIXED_MIXED_BACKEND_COLLECTION_PLAN_ID}:sha256="
            )
        )
        plan_sha = plan_id.rsplit("=", 1)[1]
        self.assertEqual(
            _resolve_fixed_collection_plan_sha256(
                dataset,
                V4TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    bc_weight=0.0,
                    ppo_weight=1.0,
                    critic_weight=0.1,
                    q_boost_coefficient=0.0,
                    entropy_coefficient=0.0,
                    amp=False,
                    expected_fixed_collection_plan_sha256=plan_sha,
                ),
            ),
            plan_sha,
        )

        wrong_backend = self.root / "mixed-plan" / "wrong-backend.npz"

        def swap_precommitted_backends(
            source_metadata: dict[str, object], arrays: dict[str, np.ndarray]
        ) -> None:
            del arrays
            shard = source_metadata["shard"]
            assert isinstance(shard, dict)
            shard["shardBackendMap"] = {"0": "cuda", "1": "cpu"}
            shard["identitySha256"] = fixed_match_shard_identity_sha256(shard)

        _rewrite_npz(cpu_shard, wrong_backend, swap_precommitted_backends)
        with self.assertRaisesRegex(ValueError, "precommitted backend"):
            load_v4_dataset_npz(wrong_backend)

        forged_cpu_cuda = self.root / "mixed-plan" / "forged-cpu-cuda.npz"

        def claim_cuda_available_on_cpu(
            source_metadata: dict[str, object], arrays: dict[str, np.ndarray]
        ) -> None:
            del arrays
            execution = source_metadata["execution"]
            assert isinstance(execution, dict)
            execution["cudaAvailable"] = True

        _rewrite_npz(cpu_shard, forged_cpu_cuda, claim_cuda_available_on_cpu)
        with self.assertRaisesRegex(ValueError, "precommitted backend"):
            load_v4_dataset_npz(forged_cpu_cuda)

        downgraded = self.root / "mixed-plan" / "downgraded-v2.npz"

        def strip_all_v2_fields(
            source_metadata: dict[str, object], arrays: dict[str, np.ndarray]
        ) -> None:
            del arrays
            shard = source_metadata["shard"]
            execution = source_metadata["execution"]
            assert isinstance(shard, dict)
            assert isinstance(execution, dict)
            shard.pop("collectionPlanVersion")
            shard.pop("shardBackendMap")
            shard.pop("crossBackendCalibrationReportSha256")
            execution.pop("fixedCollectionPlanVersion")
            execution.pop("plannedShardBackend")

        _rewrite_npz(cpu_shard, downgraded, strip_all_v2_fields)
        with self.assertRaisesRegex(ValueError, "identitySha256"):
            load_v4_dataset_npz(downgraded)

    def test_mixed_resume_requires_the_same_calibration_report_sha(self) -> None:
        backend_map = ("cpu", "cuda")
        output = self._collect_variant(
            "mixed-resume",
            954_600_001,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
            shard_backend_map=backend_map,
        )
        config = FixedMatchPPOCollectionConfig(
            run_namespace="mixed-resume",
            seed_base=954_600_001,
            match_counts=((4, 2),),
            match_shard_count=2,
            match_shard_index=0,
            standardize_advantages=False,
            lane_count=1,
            device="cpu",
            resume_existing=True,
            shard_backend_map=backend_map,
            cross_backend_calibration_report=self.root / "mock-calibration.json",
            cross_backend_calibration_cpu_npz=self.root / "mock-calibration-cpu.npz",
            cross_backend_calibration_cuda_npz=self.root / "mock-calibration-cuda.npz",
        )
        with mock.patch(
            "v4_collect_fixed_match_ppo.load_verified_fixed_match_backend_calibration",
            return_value=_mock_calibration_verification(
                self.MOCK_CALIBRATION_SHA256
            ),
        ) as loader:
            resumed = collect_v4_fixed_match_ppo(self.bundle, output, config)
        self.assertEqual(resumed.npz_sha256, hashlib.sha256(output.read_bytes()).hexdigest())
        loader.return_value.recheck_unchanged.assert_called_once_with()

        with mock.patch(
            "v4_collect_fixed_match_ppo.load_verified_fixed_match_backend_calibration",
            return_value=_mock_calibration_verification("f" * 64),
        ):
            with self.assertRaisesRegex(ValueError, "exact requested shard"):
                collect_v4_fixed_match_ppo(self.bundle, output, config)

    def test_nested_contracts_preserve_and_fail_closed_on_tampering(self) -> None:
        first = merge_v4_datasets(
            self.output,
            self.root / "nested" / "first.npz",
        )
        second = merge_v4_datasets(
            first.output_path,
            self.root / "nested" / "second.npz",
        )
        first_dataset = load_v4_dataset_npz(first.output_path)
        second_dataset = load_v4_dataset_npz(second.output_path)
        self.assertEqual(
            first_dataset.loss_eligibility.ppo_reward_contracts,
            second_dataset.loss_eligibility.ppo_reward_contracts,
        )
        self.assertEqual(
            first_dataset.loss_eligibility.ppo_behavior_policy_contracts,
            second_dataset.loss_eligibility.ppo_behavior_policy_contracts,
        )
        self.assertEqual(
            first_dataset.loss_eligibility.fixed_collection_plan_ids,
            second_dataset.loss_eligibility.fixed_collection_plan_ids,
        )

        metadata_tampers = {
            "qboost": lambda contract: contract.__setitem__(
                "requiresQBoostCoefficientZero", False
            ),
            "balance": lambda contract: contract.__setitem__(
                "requiresPlayerCountBalancedLoss", False
            ),
            "source": lambda contract: contract.__setitem__(
                "ppoSourceContracts", ["legacy-per-act-v1"]
            ),
            "behavior": lambda contract: contract[
                "ppoBehaviorPolicyContractRecords"
            ][0]["canonicalFields"].__setitem__("temperature", 0.8),
        }
        for name, mutate_contract in metadata_tampers.items():
            with self.subTest(name=name):
                tampered = self.root / "nested-tamper" / f"{name}.npz"

                def mutate(
                    metadata: dict[str, object],
                    arrays: dict[str, np.ndarray],
                    mutate_contract=mutate_contract,
                ) -> None:
                    del arrays
                    contract = metadata["lossEligibility"]
                    assert isinstance(contract, dict)
                    mutate_contract(contract)

                _rewrite_npz(first.output_path, tampered, mutate)
                with self.assertRaises(ValueError):
                    load_v4_dataset_npz(tampered)

        raw_tamper = self.root / "nested-tamper" / "raw-dq.npz"

        def change_raw_dq(metadata: dict[str, object], arrays: dict[str, np.ndarray]) -> None:
            del metadata
            valid_length = int(arrays["valid_masks"][0].sum())
            last = valid_length - 1
            arrays["trajectory_act_group_chip_differences"][0] += 0.25
            arrays["raw_act_group_chip_differences"][0, last] += 0.25

        _rewrite_npz(first.output_path, raw_tamper, change_raw_dq)
        with self.assertRaisesRegex(ValueError, "raw finish/chips D/Q|suffix"):
            load_v4_dataset_npz(raw_tamper)
        with self.assertRaises(ValueError):
            merge_v4_datasets(
                raw_tamper,
                self.root / "nested-tamper" / "recursive-output.npz",
            )

        selected_tamper = self.root / "nested-tamper" / "selected-logprob.npz"

        def change_selected(metadata: dict[str, object], arrays: dict[str, np.ndarray]) -> None:
            del metadata
            arrays["selected_action_probabilities"][0, 0] *= 0.9

        _rewrite_npz(first.output_path, selected_tamper, change_selected)
        with self.assertRaisesRegex(ValueError, "old log probabilities"):
            load_v4_dataset_npz(selected_tamper)

        gamma_tamper = self.root / "nested-tamper" / "gamma.npz"

        def change_gamma(metadata: dict[str, object], arrays: dict[str, np.ndarray]) -> None:
            del metadata
            arrays["trajectory_monte_carlo_gammas"][0] = 0.5

        _rewrite_npz(first.output_path, gamma_tamper, change_gamma)
        with self.assertRaisesRegex(ValueError, "gamma|Monte Carlo"):
            load_v4_dataset_npz(gamma_tamper)


if __name__ == "__main__":
    unittest.main()
